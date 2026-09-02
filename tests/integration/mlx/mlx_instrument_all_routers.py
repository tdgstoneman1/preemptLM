"""Scale-up smoke test: instrument *every* MoE router block in a Qwen3.6 MLX model.

Where `mlx_single_router_layer.py` proves one capture wrapper round-trips a single
Parquet row, this script exercises the pieces that only matter at full scale:

- `mlx_instrument_model()` replacing all resolved blocks in one `update_modules()`
  call (rather than the single hardcoded attribute assignment the smoke test uses),
- the recorder's multi-token explosion path, which is dead code for a one-token
  forward but carries every prefill row,
- the sink's batching, since a full prefill overflows the batch threshold,
- and, most importantly, **EXACT INFERENCE**: greedy token ids produced by the
  uninstrumented model must equal those produced after instrumentation. The capture
  wrapper is a verbatim fork of the upstream forward, and this is the assertion that
  catches it drifting.

The generation loop itself lives in `preempt/` now (`GenerationPipeline` over
`MlxModelRunner`); this script is a thin composition root around it. It stays
separate from `main.py` because it needs a sequencing `main.build_pipeline` does not
expose: run the reference generation *before* instrumenting the same loaded model,
so both passes share one set of weights.

This model is larger than the host's RAM, so a single forward pass costs roughly a
minute and every pass counts. Per-token coverage therefore comes from the *prefill*
-- one forward yields `len(prompt)` token positions across all 40 layers -- rather
than from decode steps. Lengthen `--prompt` for more coverage; raise `--max-tokens`
only to exercise the decode path. The exactness check runs generation a second time
(same weights, no second load), so it is opt-in via `--verify-exactness`.

Usage (from the repo root, on the macOS host)::

    python tests/integration/mlx_all_router_layers.py \
        --config tests/integration/qwen3_6-35b-mlx-pipeline.toml \
        --output out/router-events.parquet \
        --verify-exactness
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Iterable, Sequence

import argparse
import asyncio
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import mlx.core as mx
import mlx.nn as nn

from preempt.backends.mlx_metal.instrumentation.instrument import (
    mlx_instrument_model,
    resolve_mlx_target_layers,
)
from preempt.backends.mlx_metal.instrumentation.qwen3_x_moe import (
    Qwen3_xMoEWrapper,
)
from preempt.backends.mlx_metal.recorder import MoERecorder
from preempt.backends.mlx_metal.pipeline.runner import MlxModelRunner
from preempt.backends.mlx_metal.utils import load_mlx_model

from preempt.config.pipeline import PipelineConfig

from preempt.datamodel.tracing.context import TraceRunContext
from preempt.datamodel.tracing.expert_routing import (
    EXPERT_ROUTING_EVENT_TYPE,
    EXPERT_ROUTING_SCHEMA_VERSION,
    ExpertRoutingEvent,
)
from preempt.engine.sinks import ParquetEventSink
from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
)
from preempt.engine.metrics import StepMetrics
from preempt.engine.pipeline import GenerationResult, GenerationPipeline

from preempt.utils.io_utils import read_and_validate_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "tests" / "integration" / "qwen3_6-35b-mlx-pipeline.toml"

# Router weights are renormalized in bf16/quantized arithmetic, so the per-token
# top-k weights sum to 1.0 only to a couple of decimal places.
_WEIGHT_SUM_TOLERANCE = 0.02


class TraceVerificationError(RuntimeError):
    """Raised when the captured Parquet trace violates its expected contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instrument every MoE router block in an MLX Qwen3.6 model "
        "and verify the captured Parquet trace."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"TOML pipeline config with a [trace_settings] table. Default: {_DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id in --config. Any MLX model ID or local "
        "directory accepted by mlx_lm.load().",
    )
    parser.add_argument(
        "--prompt",
        # Per-token coverage comes from the prefill, which processes the whole
        # prompt in a single forward pass. Lengthening the prompt is therefore
        # nearly free, while each extra decode token costs a full pass.
        default="The capital of France is Paris, and the capital of Germany is",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override [generation].max_tokens from --config. Greedy tokens to "
        "generate, one forward pass each. This model is larger than the host's "
        "RAM, so a pass costs roughly a minute; prefer a longer --prompt over a "
        "larger value here.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Parquet output file. Default is a temporary file deleted on exit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Sink batch size; smaller values exercise more row groups.",
    )
    parser.add_argument(
        "--no-gate-logits",
        action="store_true",
        help="Skip capturing the full gate-logit distribution.",
    )
    parser.add_argument(
        "--verify-exactness",
        action="store_true",
        help="Also run an uninstrumented reference pass and require identical "
        "greedy tokens. This is the strongest check in the script, but it "
        "doubles the number of forward passes, so it is opt-in.",
    )
    return parser.parse_args()


def sort_and_check_layers(
    resolved: dict[str, tuple[LayerCandidate, ...]],
) -> tuple[LayerCandidate, ...]:
    """Flatten resolved target layers into one ordered tuple of MoE blocks.

    Parameters
    ----------
    resolved : dict[str, tuple[LayerCandidate, ...]]
        Per-target matches as returned by `resolve_mlx_target_layers`. The
        config's per-target `count` is what asserts the expected number of MoE
        blocks, so a model that does not match fails at resolve time rather
        than silently tracing fewer layers.

    Returns
    -------
    tuple[LayerCandidate, ...]
        Matched layers, ordered by transformer block index.

    Raises
    ------
    RuntimeError
        If fewer than two layers resolve (this is a *scale-up* test), or if any
        match has no derivable block index.
    """
    layers = [candidate for matches in resolved.values() for candidate in matches]

    if len(layers) < 2:
        raise RuntimeError(
            f"Expected the config to resolve multiple router layers; got {len(layers)}. "
            "Use `mlx_single_router_layer.py` for the single-layer case."
        )

    missing_idx = [c.layer_path for c in layers if c.block_idx is None]
    if missing_idx:
        raise RuntimeError(
            "Could not derive a transformer block index for: " f"{missing_idx!r}"
        )

    # `block_idx` is not None for any entry here, but the checker cannot see that.
    return tuple(sorted(layers, key=lambda c: (c.block_idx or 0, c.layer_path)))


def get_modules_by_path(
    model: nn.Module,
    paths: Iterable[str],
) -> dict[str, Any]:
    """Look up each dotted module path in `model`, raising if any is missing."""

    # Materialize first: `paths` is routinely a generator, and this function
    # walks it twice.
    paths = tuple(paths)

    modules = dict(model.named_modules())
    missing = [path for path in paths if path not in modules]

    if missing:
        raise RuntimeError(f"Module paths not present in model: {missing!r}")

    return {path: modules[path] for path in paths}


def moe_spec_from_layers(
    model: nn.Module,
    layers: Sequence[LayerCandidate],
) -> tuple[int, int, bool]:
    """Read `top_k`, `num_routed_experts`, and `norm_topk_prob` off the resolved blocks,
    which must be the same for all of `layers`.

    Returns
    -------
    tuple[int, int, bool]
        `(top_k, num_routed_experts, norm_topk_prob)`.

    Raises
    ------
    RuntimeError
        If the resolved blocks disagree on any of the three values.
    """
    modules = get_modules_by_path(model, (c.layer_path for c in layers))

    topologies = {
        (module.top_k, module.num_routed_experts, bool(module.norm_topk_prob))
        for module in modules.values()
    }

    if len(topologies) != 1:
        raise RuntimeError(
            "Resolved router layers disagree on MoE spec. "
            f"(top_k, num_routed_experts, norm_topk_prob): {sorted(topologies)!r}"
        )

    return topologies.pop()


def instrument_router_layers(
    *,
    model: nn.Module,
    layers: Sequence[LayerCandidate],
    recorder: MoERecorder,
    capture_gate_logits: bool,
) -> None:
    """Swap every resolved block for a capture wrapper, then verify all landed."""

    mlx_instrument_model(
        model,
        layers,
        Qwen3_xMoEWrapper.make_factory(
            recorder,
            capture_gate_logits=capture_gate_logits,
        ),
    )

    installed = get_modules_by_path(model, (c.layer_path for c in layers))
    not_wrapped = [
        path
        for path, module in installed.items()
        if not isinstance(module, Qwen3_xMoEWrapper)
    ]

    if not_wrapped:
        raise RuntimeError(
            f"Wrapper installation failed for {len(not_wrapped)} layer(s): "
            f"{not_wrapped!r}"
        )


def make_step_printer(label: str) -> Callable[[StepMetrics], None]:
    """Return an `on_step` callback printing one progress line per forward pass.

    This model is larger than the host's physical memory, so a pass can take
    tens of seconds. Without a line per step there is no way to tell a slow run
    apart from a hung one, or a constant per-step cost (paging) apart from a
    growing one (accumulating graph/memory).
    """

    def print_step(step: StepMetrics) -> None:
        print(
            f"  {label} step {step.step_idx}: {step.n_tokens} token(s) in "
            f"{step.duration_s:6.1f}s | peak {mx.get_peak_memory() / 1e9:5.2f} GB",
            flush=True,
        )

    return print_step


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceVerificationError(message)


def verify_trace(
    *,
    output_path: Path,
    expected_rows: int,
    layers: Sequence[LayerCandidate],
    expected_token_idxs: set[int],
    top_k: int,
    num_routed_experts: int,
    norm_topk_prob: bool,
    expect_gate_logits: bool,
) -> None:
    """Re-read the Parquet trace and assert the full-model capture contract.

    The point of the scale-up is coverage per *token position*: every
    instrumented layer must contribute exactly one row for every token that
    passed through the model, with no duplicates and no gaps.

    Raises
    ------
    TraceVerificationError
        On the first violated invariant.
    """
    parquet_file = pq.ParquetFile(output_path)
    table = parquet_file.read()

    metadata = table.schema.metadata or {}
    _require(
        metadata.get(b"preempt.event_type") == EXPERT_ROUTING_EVENT_TYPE.encode(),
        f"Unexpected `preempt.event_type` in file metadata: "
        f"{metadata.get(b'preempt.event_type')!r}",
    )
    _require(
        metadata.get(b"preempt.schema_version")
        == str(EXPERT_ROUTING_SCHEMA_VERSION).encode(),
        f"Unexpected `preempt.schema_version` in file metadata: "
        f"{metadata.get(b'preempt.schema_version')!r}",
    )
    _require(
        table.num_rows == expected_rows,
        f"Parquet row count mismatch: {table.num_rows} != {expected_rows}.",
    )

    expected_paths = {candidate.layer_path for candidate in layers}

    # `LayerCandidate.block_idx` is `int | None`. `sort_and_check_layers` already
    # rejects candidates without a block index, but re-check here so this function
    # is safe to call on its own and so the set narrows to `set[int]` -- otherwise
    # the `sorted()` calls below are comparing against a possible `None`.
    expected_idxs: set[int] = set()
    for candidate in layers:
        block_idx = candidate.block_idx
        if block_idx is None:
            raise TraceVerificationError(
                f"Layer {candidate.layer_path!r} has no transformer block index."
            )
        expected_idxs.add(block_idx)

    rows = table.to_pylist()

    seen_paths: set[str] = set()
    seen_idxs: set[int] = set()
    event_idxs: list[int] = []
    coverage: defaultdict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    for row_idx, row in enumerate(rows):
        seen_paths.add(row["layer_path"])
        seen_idxs.add(row["block_idx"])
        event_idxs.append(row["event_idx"])
        coverage[(row["sequence_id"], row["token_idx"])][row["block_idx"]] += 1

        expert_ids = row["expert_ids"]
        expert_weights = row["expert_weights"]

        _require(
            len(expert_ids) == top_k,
            f"Row {row_idx}: expected {top_k} expert ids, got {len(expert_ids)}.",
        )
        _require(
            len(set(expert_ids)) == top_k,
            f"Row {row_idx}: expert ids contain duplicates: {expert_ids!r}.",
        )
        _require(
            all(0 <= expert_id < num_routed_experts for expert_id in expert_ids),
            f"Row {row_idx}: expert id out of range [0, {num_routed_experts}): "
            f"{expert_ids!r}.",
        )
        _require(
            len(expert_weights) == top_k,
            f"Row {row_idx}: expected {top_k} expert weights, "
            f"got {len(expert_weights)}.",
        )

        if norm_topk_prob:
            weight_sum = sum(expert_weights)
            _require(
                abs(weight_sum - 1.0) <= _WEIGHT_SUM_TOLERANCE,
                f"Row {row_idx}: normalized expert weights sum to {weight_sum:.6f}, "
                f"expected 1.0 +/- {_WEIGHT_SUM_TOLERANCE}.",
            )

        gate_logits = row["gate_logits"]

        if expect_gate_logits:
            _require(
                gate_logits is not None and len(gate_logits) == num_routed_experts,
                f"Row {row_idx}: expected {num_routed_experts} gate logits, got "
                f"{None if gate_logits is None else len(gate_logits)}.",
            )
        else:
            _require(
                gate_logits is None,
                f"Row {row_idx}: gate logits captured despite --no-gate-logits.",
            )

    _require(
        seen_paths == expected_paths,
        "Traced layer paths do not match the resolved target layers. "
        f"Missing: {sorted(expected_paths - seen_paths)!r}, "
        f"unexpected: {sorted(seen_paths - expected_paths)!r}.",
    )
    _require(
        seen_idxs == expected_idxs,
        "Traced layer indices do not match the resolved target layers. "
        f"Missing: {sorted(expected_idxs - seen_idxs)!r}, "
        f"unexpected: {sorted(seen_idxs - expected_idxs)!r}.",
    )
    _require(
        len(set(event_idxs)) == len(event_idxs),
        f"`event_idx` is not unique across the trace "
        f"({len(event_idxs) - len(set(event_idxs))} duplicate(s)).",
    )

    traced_token_idxs = {token_idx for _, token_idx in coverage}
    _require(
        traced_token_idxs == expected_token_idxs,
        "Traced token positions do not match the tokens forwarded. "
        f"Missing: {sorted(expected_token_idxs - traced_token_idxs)!r}, "
        f"unexpected: {sorted(traced_token_idxs - expected_token_idxs)!r}.",
    )

    for (sequence_id, token_idx), layer_counts in sorted(coverage.items()):
        duplicated = {idx: n for idx, n in layer_counts.items() if n != 1}
        _require(
            not duplicated,
            f"sequence {sequence_id}, token {token_idx}: layers captured more "
            f"than once: {duplicated!r}.",
        )
        _require(
            set(layer_counts) == expected_idxs,
            f"sequence {sequence_id}, token {token_idx}: expected all "
            f"{len(expected_idxs)} layers, missing "
            f"{sorted(expected_idxs - set(layer_counts))!r}.",
        )

    print(
        f"Verified Parquet file: {table.num_rows} row(s), "
        f"{parquet_file.metadata.num_row_groups} row group(s), "
        f"{len(expected_idxs)} layer(s) x {len(expected_token_idxs)} token(s)."
    )


async def run(args: argparse.Namespace, output_path: Path) -> None:
    config = read_and_validate_toml(args.config, PipelineConfig)

    if config.trace_settings is None:
        raise ValueError(
            f"{args.config} has no [trace_settings] table; this script needs one."
        )

    model_id = args.model if args.model is not None else config.llm.model_id
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else config.generation_settings.max_tokens
    )

    print(f"Loading model: {model_id}")
    loaded = load_mlx_model(model_id)

    resolved = resolve_mlx_target_layers(
        loaded.model, config.trace_settings.to_target_layer_config()
    )
    ensure_no_target_layer_overlap(resolved)
    layers = sort_and_check_layers(resolved)
    top_k, num_routed_experts, norm_topk_prob = moe_spec_from_layers(
        loaded.model, layers
    )

    print(
        f"Resolved {len(layers)} router layer(s): "
        f"block indices {layers[0].block_idx}..{layers[-1].block_idx}, "
        f"class={layers[0].layer_class!r}"
    )
    print(
        f"model moe spec: top_k={top_k}, num_routed_experts={num_routed_experts}, "
        f"norm_topk_prob={norm_topk_prob}"
    )
    print(f"Prompt: {args.prompt!r}")

    runner = MlxModelRunner(loaded.model)

    reference: GenerationResult | None = None

    if args.verify_exactness:
        print(f"Reference pass (uninstrumented), {max_tokens} token(s)...")
        reference_pipeline = GenerationPipeline(
            runner=runner,
            tokenizer=loaded.tokenizer,
            max_tokens=max_tokens,
            on_step=make_step_printer("reference"),
        )
        reference = await reference_pipeline.generate(args.prompt)
        print(f"Reference output: {reference.text!r}")

    capture_gate_logits = not args.no_gate_logits

    run_context = TraceRunContext.with_generated_run_id(
        run_id_prefix="mlx-all-router-layers",
        model_id=model_id,
        model_architecture=config.llm.architecture,
        model_revision=config.llm.revision,
    )
    recorder = MoERecorder(run_context=run_context)

    instrument_router_layers(
        model=loaded.model,
        layers=layers,
        recorder=recorder,
        capture_gate_logits=capture_gate_logits,
    )
    print(f"Instrumented all {len(layers)} router layer(s).")

    sink = ParquetEventSink(
        path=output_path,
        schema=ExpertRoutingEvent.arrow_schema(),
        batch_size=args.batch_size,
        overwrite=True,
    )
    traced_pipeline = GenerationPipeline(
        runner=runner,
        tokenizer=loaded.tokenizer,
        max_tokens=max_tokens,
        recorder=recorder,
        sink=sink,
        on_step=make_step_printer("traced"),
    )

    print(f"Traced pass (instrumented), {max_tokens} token(s)...")
    traced = await traced_pipeline.generate(args.prompt)
    print(f"Traced output: {traced.text!r}")

    # Exactness first: it is the headline signal, and a row-count mismatch is a
    # far less interesting failure to surface ahead of it.
    if reference is not None:
        if traced.token_ids != reference.token_ids:
            raise TraceVerificationError(
                "EXACT INFERENCE violated: instrumented output differs from the "
                f"uninstrumented reference.\n  reference: {reference.token_ids!r}\n"
                f"  traced:    {traced.token_ids!r}"
            )
        print(
            f"Exactness verified: {len(traced.token_ids)} greedy token(s) identical "
            "with and without instrumentation."
        )
    else:
        print("Exactness not verified (pass --verify-exactness to enable).")

    tokens_forwarded = traced.metrics.tokens_forwarded
    records_written = traced.metrics.records_written
    expected_rows = len(layers) * tokens_forwarded

    if records_written != expected_rows:
        raise TraceVerificationError(
            f"Recorder wrote {records_written} record(s); expected "
            f"{len(layers)} layer(s) x {tokens_forwarded} token(s) = {expected_rows}."
        )

    # `sink.records_written` only counts rows already handed to the Parquet
    # writer, so this catches a batch stranded in the sink buffer. The traced
    # pipeline closed the sink when `generate` returned, so the final partial
    # batch is included.
    if sink.records_written != expected_rows:
        raise TraceVerificationError(
            f"Sink persisted {sink.records_written} record(s); "
            f"expected {expected_rows}."
        )

    print(f"Captured and wrote {records_written} router record(s).")
    print(f"Parquet output: {output_path}")

    verify_trace(
        output_path=output_path,
        expected_rows=expected_rows,
        layers=layers,
        expected_token_idxs=set(range(tokens_forwarded)),
        top_k=top_k,
        num_routed_experts=num_routed_experts,
        norm_topk_prob=norm_topk_prob,
        expect_gate_logits=capture_gate_logits,
    )


def main() -> None:
    args = parse_args()

    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.output is not None:
        asyncio.run(run(args, args.output))
        return

    with tempfile.TemporaryDirectory(prefix="preempt-mlx-all-routers-") as temp_dir:
        output_path = Path(temp_dir) / "router-events.parquet"
        asyncio.run(run(args, output_path))


if __name__ == "__main__":
    main()
