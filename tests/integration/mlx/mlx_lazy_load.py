"""Never-materialize load smoke test (Task 8).

Proves the streaming loader can bring up a MoE model whose experts do not fit in
RAM without ever reading those experts off disk. The whole streaming design
hinges on this: `mlx_lm.load` eagerly evaluates every parameter, so the eventual
larger target cannot be loaded even once the stock way. On the 35B checkpoint the
experts are ~18 GB of an ~18 GB 4-bit model, so a correct strip leaves only the
small dense backbone resident and peak memory drops sharply — which is the whole
assertion here.

The sequence under test, in the one order that works::

    load(lazy=True)  ->  instrument MoE blocks  ->  strip switch_mlp  ->  mx.eval

`lazy=True` keeps every weight an unevaluated mmap-backed array; instrumentation
installs the capture wrappers; `mlx_strip_instrumented_expert_weights` removes each
wrapper's `inner.switch_mlp` expert projections *before* any eval; and the final
`mx.eval(model.parameters())` then materializes the dense backbone alone. Getting
the order wrong (evaluating before stripping) silently pulls the full ~18 GB, so
the assertion is peak resident memory, not the mechanism's name.

**No forward pass is run.** After stripping, the non-streaming expert path is
intentionally broken (its weights are gone) and the streaming path needs the
composition wiring a later task provides. This script only proves the experts
never materialized.

Usage (from the repo root, on the macOS host)::

    python tests/integration/mlx/mlx_lazy_load.py \
        --config tests/integration/mlx/configs/qwen3_6-35b-mlx-pipeline.toml
"""

# TODO DESLOP

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from preempt.backends.mlx_metal.instrumentation.instrument import (
    mlx_instrument_model,
    mlx_strip_instrumented_expert_weights,
)
from preempt.backends.mlx_metal.instrumentation.qwen3_x_moe import (
    InstrumentedQwen3_xMoE,
)
from preempt.backends.mlx_metal.layer_discovery import resolve_mlx_target_layers
from preempt.backends.mlx_metal.utils import load_mlx_model

from preempt.config.pipeline import PipelineConfig

from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
)
from preempt.utils.io_utils import read_and_validate_toml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "qwen3_6-35b-mlx-pipeline.toml"
)
DEFAULT_MAX_MEMORY_GB = 8.0


def sort_and_check_layers(
    resolved: dict[str, tuple[LayerCandidate, ...]],
) -> tuple[LayerCandidate, ...]:
    """Flatten resolved target layers into one block-index-ordered tuple of MoE blocks.

    Mirrors the scale-up gate script's helper: the config's per-target `count`
    is what asserts the expected block count, so a model that does not match
    fails at resolve time rather than silently stripping fewer layers.
    """
    layers = [candidate for matches in resolved.values() for candidate in matches]

    if len(layers) < 2:
        raise RuntimeError(
            f"Expected the config to resolve multiple MoE blocks; got {len(layers)}."
        )

    missing_idx = [c.layer_path for c in layers if c.block_idx is None]
    if missing_idx:
        raise RuntimeError(
            f"Could not derive a transformer block index for: {missing_idx!r}"
        )

    return tuple(sorted(layers, key=lambda c: (c.block_idx or 0, c.layer_path)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove the never-materialize load never reads the experts: "
        "lazy load, instrument, strip switch_mlp, eval the dense backbone, and "
        "assert peak memory stays well below the full-model footprint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"TOML pipeline config with a [tracing] table. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id in --config.",
    )
    parser.add_argument(
        "--peak-ceiling-gb",
        type=float,
        default=DEFAULT_MAX_MEMORY_GB,
        help="Fail if peak resident memory (GB) exceeds this after the dense "
        f"backbone eval. Default: {DEFAULT_MAX_MEMORY_GB}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_and_validate_toml(args.config, PipelineConfig)
    if config.trace_settings is None:
        raise ValueError(
            f"{args.config} has no [tracing] table; this script needs one."
        )

    model_id = args.model if args.model is not None else config.llm.model_id

    print(f"Loading model lazily: {model_id}")
    loaded = load_mlx_model(model_id, lazy=True)

    resolved = resolve_mlx_target_layers(
        loaded.model, config.trace_settings.to_target_layer_config()
    )
    ensure_no_target_layer_overlap(resolved)
    layers = sort_and_check_layers(resolved)
    print(
        f"Resolved {len(layers)} MoE block(s): "
        f"indices {layers[0].block_idx}..{layers[-1].block_idx}."
    )

    # No recorder, no provider: instrumentation only needs to install the
    # wrappers so the strip can reach each `inner.switch_mlp`.
    mlx_instrument_model(
        loaded.model,
        layers,
        InstrumentedQwen3_xMoE.make_factory(recorder=None),
    )
    print(f"Instrumented {len(layers)} MoE block(s).")

    mlx_strip_instrumented_expert_weights(loaded.model, layers)
    print("Stripped switch_mlp expert weights from every instrumented block.")

    # Only now force the dense backbone into memory. If the strip worked, the
    # experts are not in the parameter tree and are never read.
    mx.eval(loaded.model.parameters())

    peak_gb = mx.get_peak_memory() / 1e9
    print(f"Peak resident memory after dense-backbone eval: {peak_gb:.2f} GB")
    print(f"Ceiling: {args.peak_ceiling_gb:.2f} GB")

    if peak_gb > args.peak_ceiling_gb:
        raise RuntimeError(
            f"Peak memory {peak_gb:.2f} GB exceeded the ceiling "
            f"{args.peak_ceiling_gb:.2f} GB: the strip did not prevent the "
            "experts from materializing."
        )

    print(
        f"Never-materialize load verified: peak {peak_gb:.2f} GB is below the "
        f"{args.peak_ceiling_gb:.2f} GB ceiling; experts never materialized."
    )


if __name__ == "__main__":
    main()
