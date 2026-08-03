"""Composition root: the one place concrete backends, sinks, and the engine meet.

Usage (macOS host)::

    python main.py --config configs/pipeline-qwen3_6-35b-mlx.toml \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from preempt.config.pipeline import PipelineConfig
from preempt.engine.metrics import StepMetrics
from preempt.engine.pipeline import InferencePipeline
from preempt.utils.io_utils import read_and_validate_toml


def _print_step(step: StepMetrics) -> None:
    print(
        f"  step {step.step_idx}: {step.n_tokens} token(s) in {step.duration_s:6.1f}s",
        flush=True,
    )


def build_pipeline(
    config: PipelineConfig, *, config_dir: Path | None = None
) -> InferencePipeline:
    """Wire concrete backend components per `config` into an `InferencePipeline`.

    A relative `[tracing].output_path` is resolved against `config_dir`, so a
    config file names its outputs relative to itself rather than to whatever
    directory the process happened to start in. Absolute paths pass through
    untouched.

    Parameters
    ----------
    config : PipelineConfig
        Parsed pipeline configuration.
    config_dir : Path | None
        Directory the config file was read from, used as the base for relative
        paths. `None` keeps the current working directory as the base, which is
        what callers constructing a `PipelineConfig` in memory want.

    Returns
    -------
    InferencePipeline
        A pipeline wired to the backend `config` selects.

    Raises
    ------
    NotImplementedError
        If `config` requests expert streaming.
    ValueError
        If `config` names an unknown backend.
    FileExistsError
        If tracing would overwrite an existing trace file without
        `[tracing].overwrite_output`. Checked before the model load so the
        failure costs seconds rather than minutes.
    """

    if config.streaming is not None:
        raise NotImplementedError("Streaming lands in phase 3 of the v1 plan.")

    if config.model.backend != "mlx_metal":
        raise ValueError(f"Unknown backend: {config.model.backend!r}")

    # Backend imports stay inside the branch: only the composition root may
    # import concrete backends, and only for the backend actually selected.
    from preempt.backends.mlx_metal.instrument import mlx_instrument_model
    from preempt.backends.mlx_metal.instrumented.qwen3_next_moe import (
        make_qwen3next_moe_wrapper_factory,
    )
    from preempt.backends.mlx_metal.layer_discovery import resolve_mlx_target_layers
    from preempt.backends.mlx_metal.loader import load_mlx_model
    from preempt.backends.mlx_metal.recorder import MlxExpertRoutingRecorder
    from preempt.backends.mlx_metal.runner import MlxModelRunner
    from preempt.core.sinks import ParquetEventSink
    from preempt.datamodel.tracing.context import TraceRunContext
    from preempt.datamodel.tracing.expert_routing import ExpertRoutingEvent
    from preempt.engine.layer_resolution import ensure_no_target_layer_overlap

    output_path: Path | None = None

    if config.tracing is not None:
        base_dir = config_dir if config_dir is not None else Path.cwd()
        output_path = base_dir / config.tracing.output_path

        # The sink checks this too, but only lazily on first write -- by which
        # point a multi-minute model load has already been paid for.
        if output_path.exists() and not config.tracing.overwrite_output:
            raise FileExistsError(
                f"Trace output already exists: {output_path}. Set "
                "`overwrite_output = true` under [tracing], or choose another "
                "`output_path`."
            )

    print(f"Loading model: {config.model.id}")
    loaded = load_mlx_model(config.model.id)

    recorder = None
    sink = None

    if config.tracing is not None and output_path is not None:
        resolved = resolve_mlx_target_layers(
            loaded.model, config.tracing.to_target_layer_config()
        )
        ensure_no_target_layer_overlap(resolved)
        layers = [candidate for matches in resolved.values() for candidate in matches]

        run_context = TraceRunContext.with_generated_run_id(
            run_id_prefix=config.tracing.run_id_prefix,
            model_id=config.model.id,
            model_architecture=config.model.architecture,
            model_revision=config.model.revision,
        )
        recorder = MlxExpertRoutingRecorder(run_context=run_context)

        mlx_instrument_model(
            loaded.model,
            layers,
            make_qwen3next_moe_wrapper_factory(
                recorder, capture_gate_logits=config.tracing.capture_gate_logits
            ),
        )
        print(f"Instrumented {len(layers)} router layer(s).")

        sink = ParquetEventSink(
            path=output_path,
            schema=ExpertRoutingEvent.arrow_schema(),
            batch_size=config.tracing.batch_size,
            overwrite=config.tracing.overwrite_output,
        )

    return InferencePipeline(
        runner=MlxModelRunner(loaded.model),
        tokenizer=loaded.tokenizer,
        max_tokens=config.generation.max_tokens,
        recorder=recorder,
        sink=sink,
        on_step=_print_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured inference pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    config = read_and_validate_toml(args.config, PipelineConfig)
    pipeline = build_pipeline(config, config_dir=args.config.resolve().parent)

    result = asyncio.run(pipeline.generate(args.prompt, max_tokens=args.max_tokens))

    print(f"Output: {result.text!r}")
    print(
        f"{result.metrics.tokens_forwarded} token position(s) forwarded, "
        f"{result.metrics.records_written} trace record(s), "
        f"{result.metrics.total_duration_s:.1f}s total."
    )


if __name__ == "__main__":
    main()
