"""***GENERATED WITH CLAUDE CODE***

Phase-3 exactness gate (Task 11): streaming never changes a token.

This is the proof of the EXACT INFERENCE invariant for the streaming path. It
runs the *same* streaming expert-major forward twice over one wide prompt and
asserts the greedy token ids are bit-identical:

* **resident** — a cache budget large enough to hold the store's whole working
  set, so nothing ever evicts (the oracle);
* **starved** — a budget small enough that eviction bites on nearly every layer.

Eviction changes only *when* an expert's weights load, never the arithmetic, so
the two are identical by construction; this gate proves the plumbing honours
that. The oracle here is deliberately fully-resident expert-major, **not** stock
`mlx_lm`/`gather_qmm` — that separate, accepted precision difference is
quantified once at the bottom of this script and recorded in
`.claude/docs/expert-major-precision-decision.md`, but it is not what the gate
measures.

Three checks run in sequence, each an independent model load (~60-90 s):

1. **Gate.** resident tokens `R` vs starved tokens `S`; assert `R == S`, assert
   the starved cache was genuinely exercised (hit rate < 1.0 and bytes read >
   budget), and report peak memory for both to substantiate the streaming claim.
2. **Deliberate failure.** Re-run the starved gate with one evicted expert's
   *reinstall* corrupted, and confirm the tokens now diverge from `R`. A gate
   never seen failing is not yet known to be a gate.
3. **Precision measurement.** A plain, non-streaming (`gather_qmm`) run `K` over
   the same prompt, reported against `R`: how many token ids differ and where.
   This is a measurement, not a pass/fail.

Usage (from the repo root, on the macOS host)::

    python tests/integration/mlx/mlx_streaming_exactness.py \
        --store expert-bank/qwen3.6-35b-8bit --max-tokens 16
"""

from __future__ import annotations

import argparse
import asyncio
import gc
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

import preempt.backends.mlx_metal.expert_cache as residency_module
from preempt.config.pipeline import (
    GenerationSettings,
    LlmConfig,
    PipelineConfig,
    StreamSettings,
)
from preempt.datamodel.identity import ExpertKey
from preempt.datamodel.experts import ExpertPayload

from preempt.engine.metrics import GenerationMetrics

from preempt.expert_bank.manifest import ExpertBankManifest

from preempt.backends.mlx_metal.pipeline.build import mlx_build_generation_pipeline

# TODO CLEAN UP CLAUDE SLOP, BOTH CODE AND LOGS ARE UTTERLY UNINTERPRETABLE.

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORE = _REPO_ROOT / "expert-bank" / "qwen3.6-35b-8bit"

# Starved: ~32 MB holds ~18 experts of ~1.77 MB, so eviction bites well within a
# single 8-of-256 layer over a wide prompt while staying above one expert (a
# budget below one expert would be a configuration error, not a stress test).
_STARVED_BUDGET_BYTES = 32 * 1024**2

# A wide, coherent prompt so the router touches many distinct experts per layer
# and eviction pressure is real. Tokenised length is asserted >= 200 at runtime.
_PROMPT = (
    "Mixture-of-Experts models activate only a small fraction of their "
    "parameters for each token, which is exactly what makes streaming their "
    "weights from disk practical. A sparse router selects a handful of experts "
    "per layer, the dense backbone stays resident in memory, and the routed "
    "expert weights are staged across the memory hierarchy precisely when the "
    "router proves they are needed. The central promise of this design is that "
    "residency and caching decide only when a weight arrives, never which "
    "tokens the model produces. To justify that promise we compare a run whose "
    "cache is large enough to hold everything against a run whose cache is "
    "starved so aggressively that experts are evicted and re-read on nearly "
    "every layer. If the two produce different tokens, the streaming machinery "
    "is wrong, because eviction must change timing alone. This paragraph is "
    "deliberately long and repetitive so that the prefill pass spans many "
    "token positions and forces a wide union of distinct experts through the "
    "cache, exercising admission, eviction, and reinstallation under genuine "
    "memory pressure. The router has structure, workloads repeat, and compute "
    "overlaps input and output, but none of that may ever be allowed to move a "
    "single produced token away from the fully-resident reference result."
)


def _clear_mlx_cache() -> None:
    """Return MLX's buffer cache to the OS between independent model loads."""
    clear = getattr(mx, "clear_cache", None)
    if clear is None:
        metal = getattr(mx, "metal", None)
        clear = getattr(metal, "clear_cache", None) if metal is not None else None
    if clear is not None:
        clear()


def _streaming_config(
    store: Path, model_id: str, budget_bytes: int, max_tokens: int
) -> PipelineConfig:
    return PipelineConfig(
        llm=LlmConfig(
            model_id=model_id, backend="mlx_metal", architecture="qwen3-next"
        ),
        generation_settings=GenerationSettings(max_tokens=max_tokens),
        stream_settings=StreamSettings(
            expert_bank_path=store.resolve(),
            memory_budget_gb=budget_bytes / 1024**3,
            bypass_page_cache=True,
        ),
    )


def _plain_config(model_id: str, max_tokens: int) -> PipelineConfig:
    # No [streaming] and no [tracing]: expert application goes through
    # inner.switch_mlp / gather_qmm, bit-identical to stock mlx_lm.
    return PipelineConfig(
        llm=LlmConfig(
            model_id=model_id, backend="mlx_metal", architecture="qwen3-next"
        ),
        generation_settings=GenerationSettings(max_tokens=max_tokens),
    )


async def _run(
    config: PipelineConfig,
    loop: asyncio.AbstractEventLoop,
    prompt: str,
    max_tokens: int,
) -> tuple[list[int], GenerationMetrics, int, int]:
    """Build a fresh pipeline, generate greedily, and free it."""
    mx.reset_peak_memory()
    metrics = GenerationMetrics()
    pipeline = mlx_build_generation_pipeline(
        config,
        config_dir=None,
        event_loop=loop,
        metrics=metrics,
        stream_experts=config.stream_settings is not None,
        save_traces=False,
    )
    prompt_len = len(pipeline.tokenizer.encode(prompt))
    result = await pipeline.generate(prompt, max_tokens=max_tokens)
    peak = mx.get_peak_memory()

    # Merge step-level execution telemetry into the provider's streaming metrics
    metrics.steps = result.metrics.steps
    metrics.records_written = result.metrics.records_written

    del pipeline
    gc.collect()
    _clear_mlx_cache()
    return result.token_ids, metrics, peak, prompt_len


def _make_corrupting_decoder(
    original,
) -> tuple[object, dict[ExpertKey, int], list[int]]:
    """Wrap `decode_serialized_weights` to corrupt an expert's *reinstall*.

    The first decode of any key is left clean; the second and later decodes —
    which only happen after the cache evicted the key and demanded it again —
    scale one projection's `scales` tensor, so a reinstalled expert dequantises
    to different weights. This targets exactly the eviction/reinstall path the
    gate exercises, and is discarded after the failure check.
    """
    install_counts: dict[ExpertKey, int] = {}
    fires: list[int] = [0]

    def corrupt(payload: ExpertPayload, encoding):  # type: ignore[no-untyped-def]
        tensors = original(payload, encoding)
        key = payload.key
        install_counts[key] = install_counts.get(key, 0) + 1
        if install_counts[key] < 2:
            return tensors
        for name in tensors:
            if name.endswith(".scales"):
                tensors[name] = tensors[name] * mx.array(2.0, dtype=tensors[name].dtype)
                fires[0] += 1
                break
        return tensors

    return corrupt, install_counts, fires


def _hit_rate(metrics: GenerationMetrics) -> float:
    demands = metrics.cache_hits + metrics.cache_misses
    return metrics.cache_hits / demands if demands else 0.0


def _n_differ(a: Sequence[int], b: Sequence[int]) -> int:
    differ = sum(1 for x, y in zip(a, b) if x != y)
    return differ + abs(len(a) - len(b))


def _first_divergence(a: Sequence[int], b: Sequence[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove streaming exactness: resident vs starved expert-major "
        "tokens must be identical; a corrupted reinstall must diverge; and the "
        "resident-vs-stock precision difference is measured once."
    )
    parser.add_argument("--store", type=Path, default=_DEFAULT_STORE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--starved-budget", type=int, default=_STARVED_BUDGET_BYTES)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    manifest = ExpertBankManifest.load(args.store)
    model_id = args.model if args.model is not None else manifest.model_id
    expert_num_bytes = manifest.expert_num_bytes()

    # A resident budget above the whole store's byte size guarantees the cache
    # never evicts, so the resident run is the true no-eviction oracle.
    resident_budget = expert_num_bytes * len(manifest.blobs) + expert_num_bytes

    loop = asyncio.get_running_loop()

    print(f"Store: {args.store}")
    print(f"Model: {model_id}")
    print(
        f"Expert blob size: {expert_num_bytes / 1e6:.2f} MB, {len(manifest.blobs)} blobs."
    )
    print(
        f"Resident budget: {resident_budget / 1e9:.1f} GB (no eviction). "
        f"Starved budget: {args.starved_budget / 1e6:.0f} MB."
    )
    print(f"Max tokens: {args.max_tokens}.\n")

    # --- Check 1: the EXACT INFERENCE gate ---------------------------------
    print("=== Resident run (oracle) ===", flush=True)
    resident_ids, resident_metrics, resident_peak, prompt_len = await _run(
        _streaming_config(args.store, model_id, resident_budget, args.max_tokens),
        loop,
        _PROMPT,
        args.max_tokens,
    )
    print(f"Prompt tokens: {prompt_len}.")
    if prompt_len < 200:
        raise RuntimeError(
            f"Prompt is only {prompt_len} tokens; the gate needs >= 200 so the "
            "pass is wide and eviction pressure is real."
        )
    print(f"Resident tokens R: {resident_ids!r}")
    print(
        f"Resident cache: {resident_metrics.cache_hits} hit(s), "
        f"{resident_metrics.cache_misses} miss(es); peak "
        f"{resident_peak / 1e9:.2f} GB.\n"
    )

    print("=== Starved run ===", flush=True)
    starved_ids, starved_metrics, starved_peak, _ = await _run(
        _streaming_config(args.store, model_id, args.starved_budget, args.max_tokens),
        loop,
        _PROMPT,
        args.max_tokens,
    )
    starved_bytes_read = starved_metrics.cache_misses * expert_num_bytes
    starved_hit_rate = _hit_rate(starved_metrics)
    print(f"Starved tokens S: {starved_ids!r}")
    print(
        f"Starved cache: {starved_metrics.cache_hits} hit(s), "
        f"{starved_metrics.cache_misses} miss(es) (hit rate "
        f"{starved_hit_rate:.1%}); bytes read {starved_bytes_read / 1e9:.2f} GB "
        f"vs budget {args.starved_budget / 1e6:.0f} MB; peak "
        f"{starved_peak / 1e9:.2f} GB.\n"
    )

    if starved_hit_rate >= 1.0 or starved_bytes_read <= args.starved_budget:
        raise RuntimeError(
            "Starved cache was not genuinely exercised: need hit rate < 1.0 and "
            f"bytes read > budget, got hit rate {starved_hit_rate:.1%} and "
            f"{starved_bytes_read} bytes vs {args.starved_budget} budget. A "
            "passing exactness check over an unexercised cache proves nothing."
        )

    gate_ok = resident_ids == starved_ids
    print(f"GATE R == S: {gate_ok}")
    print(
        f"Peak memory: starved {starved_peak / 1e9:.2f} GB vs resident "
        f"{resident_peak / 1e9:.2f} GB.\n"
    )
    if not gate_ok:
        first = _first_divergence(resident_ids, starved_ids)
        raise AssertionError(
            "EXACT INFERENCE violated: starved streaming produced different "
            f"tokens than the resident oracle, first differing at index {first}. "
            "Eviction changed the arithmetic, not just the timing."
        )
    print("Streaming exactness verified: starved tokens are bit-identical to R.\n")

    # --- Check 2: the gate has teeth ---------------------------------------
    print("=== Deliberate-failure run (corrupted reinstall) ===", flush=True)
    original_decode = residency_module.decode_serialized_weights
    corrupt_decode, _, fires = _make_corrupting_decoder(original_decode)
    residency_module.decode_serialized_weights = corrupt_decode  # type: ignore[assignment]
    try:
        corrupt_ids, corrupt_metrics, _, _ = await _run(
            _streaming_config(
                args.store, model_id, args.starved_budget, args.max_tokens
            ),
            loop,
            _PROMPT,
            args.max_tokens,
        )
    finally:
        residency_module.decode_serialized_weights = original_decode  # type: ignore[assignment]

    print(f"Corrupted {fires[0]} expert reinstall(s) (scales x2 on second decode).")
    print(f"Corrupted tokens: {corrupt_ids!r}")
    corrupt_diverged = corrupt_ids != resident_ids
    print(f"FAILURE-CHECK corrupted != R: {corrupt_diverged}")
    if fires[0] == 0:
        raise RuntimeError(
            "No reinstall was corrupted, so the failure check proved nothing: "
            "the starved budget did not force any expert to be evicted and "
            "re-read within this run."
        )
    if not corrupt_diverged:
        raise RuntimeError(
            "Corrupting an evicted expert's reinstall did not change any token; "
            "the gate cannot be trusted to catch a real residency bug."
        )
    print(
        "Gate has teeth: corrupting a reinstall diverged from R, and the "
        "corruption was discarded.\n"
    )

    # --- Check 3: one-time precision measurement (stock mlx_lm vs R) --------
    print("=== Plain run (stock mlx_lm / gather_qmm) ===", flush=True)
    plain_ids, _, plain_peak, _ = await _run(
        _plain_config(model_id, args.max_tokens), loop, _PROMPT, args.max_tokens
    )
    print(f"Plain tokens K: {plain_ids!r}; peak {plain_peak / 1e9:.2f} GB.")
    n_differ = _n_differ(resident_ids, plain_ids)
    first_div = _first_divergence(resident_ids, plain_ids)
    print(
        f"PRECISION resident-vs-stock: {n_differ} of {len(resident_ids)} token "
        f"id(s) differ; first divergence at index "
        f"{first_div if first_div >= 0 else 'none'}.\n"
    )

    print("All checks complete.")


def main() -> None:
    args = parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
