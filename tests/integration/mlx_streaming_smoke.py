"""***GENERATED WITH CLAUDE CODE***

End-to-end streaming wiring smoke test (Task 10).

The first run of the whole streaming path through the composition root: build a
pipeline from a `[streaming]` config, generate a couple of tokens, and prove the
experts actually came off disk (the cache recorded at least one miss). This is a
wiring proof, **not** the exactness gate — that is Task 11's
`mlx_streaming_exactness.py`, which compares token ids across cache budgets.

Because it is the first integration of the streaming forward, this is where the
residency tensor-key contract gets exercised: the forward reads
`gate_proj.weight`/`.scales`/`.biases` etc. off the residency, and those keys
must be exactly the names the converter stamped into the store. A `KeyError` on a
projection tensor here points straight at that convention.

Usage (from the repo root, on the macOS host)::

    python tests/integration/mlx_streaming_smoke.py \
        --store expert-bank/expert-bank/qwen3.5-35b --prompt "Fire and fury like the world has never"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import mlx.core as mx

from preempt.config.pipeline import (
    GenerationSettings,
    LlmConfig,
    PipelineConfig,
    StreamSettings,
)
from preempt.engine.metrics import GenerationMetrics
from preempt.storage.manifest import ExpertBankManifest

from main import build_pipeline

# TODO CLEAN UP CLAUDE SLOP.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORE = _REPO_ROOT / "expert-bank" / "store"

# Starved-but-workable: the streaming wrapper acquires one expert at a time, so a
# budget below a whole layer's expert union is safe (MLX refcounting keeps a
# just-evicted expert's buffers alive for the graph already built against them).
# ~32 MB holds ~18 experts of ~1.77 MB each, so eviction bites well within a
# single 8-of-256 layer over a multi-token prompt while staying comfortably above
# one expert.
_DEFAULT_BUDGET_BYTES = 32_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove the streaming path is wired: build a pipeline from a "
        "[streaming] config, generate a couple of tokens, and assert the expert "
        "cache took at least one miss (experts streamed off disk)."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=_DEFAULT_STORE,
        help=f"Packed store directory. Default: {_DEFAULT_STORE}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to the store manifest's `model_id`.",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--budget-bytes", type=int, default=_DEFAULT_BUDGET_BYTES)
    return parser.parse_args()


def _build_config(args: argparse.Namespace, model_id: str) -> PipelineConfig:
    return PipelineConfig(
        llm=LlmConfig(
            model_id=model_id, backend="mlx_metal", architecture="qwen3-next"
        ),
        generation_settings=GenerationSettings(max_tokens=args.max_tokens),
        stream_settings=StreamSettings(
            expert_bank_path=args.store.resolve(),
            memory_bytes_budget=args.budget_bytes,
            bypass_page_cache=True,
        ),
    )


async def _run(args: argparse.Namespace, config: PipelineConfig) -> GenerationMetrics:
    loop = asyncio.get_running_loop()
    metrics = GenerationMetrics()

    # Absolute store path, so config_dir is irrelevant.
    pipeline = build_pipeline(config, config_dir=None, loop=loop, metrics=metrics)
    result = await pipeline.generate(args.prompt, max_tokens=args.max_tokens)

    print(f"Generated {len(result.token_ids)} token(s): {result.token_ids!r}")
    print(f"Output text: {result.text!r}")
    return metrics


def main() -> None:
    args = parse_args()

    manifest = ExpertBankManifest.load(args.store)
    model_id = args.model if args.model is not None else manifest.model_id
    print(f"Store: {args.store}")
    print(f"Model: {model_id}")
    print(f"Budget: {args.budget_bytes / 1e6:.0f} MB")

    config = _build_config(args, model_id)
    metrics = asyncio.run(_run(args, config))

    demands = metrics.cache_hits + metrics.cache_misses
    hit_rate = metrics.cache_hits / demands if demands else 0.0
    peak_gb = mx.get_peak_memory() / 1e9
    print(
        f"Expert cache: {metrics.cache_hits} hit(s), {metrics.cache_misses} "
        f"miss(es) over {demands} demand(s) (hit rate {hit_rate:.1%})."
    )
    print(
        f"Demand stall: {metrics.demand_stall_s:.1f}s. Peak memory: {peak_gb:.2f} GB."
    )

    if metrics.cache_misses < 1:
        raise RuntimeError(
            "The expert cache recorded no misses: experts never streamed off "
            "disk, so the streaming path was not actually exercised."
        )

    print(
        f"Streaming wiring verified: {metrics.cache_misses} demand read(s) "
        "streamed experts off disk end-to-end."
    )


if __name__ == "__main__":
    main()
