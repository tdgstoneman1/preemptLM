"""***GENERATED WITH CLAUDE CODE***

Composition root: the one place concrete backends, sinks, and the engine meet.

Usage (macOS host)::

    python main.py --config configs/pipeline-qwen3_6-35b-mlx.toml \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from preempt.config.pipeline import PipelineConfig
from preempt.config.target_layers import (
    TargetLayerConfig,
    TargetLayerSearchParams,
    TargetLayerSpec,
)
from preempt.engine.metrics import GenerationMetrics, StepMetrics
from preempt.engine.pipeline import GenerationPipeline
from preempt.utils.io_utils import read_and_validate_toml

# TODO clean up claude slop + refactor
# TODO add cli args to optionally disable expert reads from bank and trace writes
# currently, this requires commenting out fields in config file.

# MoE block class per model architecture, so a streaming run with no [trace_settings]
# table still knows which modules to instrument. v1 is deliberately one
# architecture; the map is the extension point rather than a hardcoded literal
# buried in the wiring.
_MOE_BLOCK_CLASS_BY_ARCHITECTURE: dict[str, str] = {
    "qwen3-next": "Qwen3NextSparseMoeBlock",
}


def _print_step(step: StepMetrics) -> None:
    # One line per forward pass, not per token: a prefill chunk forwards many
    # tokens at once, so `n_tokens` is the pass width, not a token count.
    print(
        f"  forward {step.step_idx}: {step.n_tokens} token position(s) in "
        f"{step.duration_s:6.1f}s",
        flush=True,
        end="\r",
    )


def _streaming_target_config(
    architecture: str, moe_layer_count: int
) -> TargetLayerConfig:
    """Build the instrumentation target for a streaming-only run.

    A streaming run with no `[trace_settings]` table has no target list of its own, but
    every MoE block must still be instrumented so its experts stream rather than
    stay resident. This matches the architecture's MoE block class and asserts
    the expected count, so a model that resolves a different number of blocks
    than the expert bank describes fails loudly at resolve time.

    Parameters
    ----------
    architecture : str
        Model architecture from `[model].architecture`.
    moe_layer_count : int
        Number of MoE layers the expert bank holds; used as the `count` assertion.

    Returns
    -------
    TargetLayerConfig
        A single target matching every MoE block of the architecture.

    Raises
    ------
    ValueError
        If no MoE block class is known for `architecture`.
    """
    block_class = _MOE_BLOCK_CLASS_BY_ARCHITECTURE.get(architecture)
    if block_class is None:
        raise ValueError(
            f"No MoE block class known for architecture {architecture!r}; "
            f"streaming supports {sorted(_MOE_BLOCK_CLASS_BY_ARCHITECTURE)}."
        )

    return TargetLayerConfig(
        target_layers=(
            TargetLayerSpec(
                name="streaming-moe",
                search_params=TargetLayerSearchParams(
                    layer_class=block_class, count=moe_layer_count
                ),
            ),
        )
    )


# TODO rewrite docsring, Claude-generated slop
def build_pipeline(
    config: PipelineConfig,
    *,
    config_dir: Path | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    metrics: GenerationMetrics | None = None,
) -> GenerationPipeline:
    """Wire concrete backend components per `config` into an `GenerationPipeline`.

    A relative `[trace_settings].output_path` or `[stream_settings].expert_bank_path` is resolved
    against `config_dir`, so a config file names its outputs relative to itself
    rather than to whatever directory the process happened to start in. Absolute
    paths pass through untouched.

    Tracing and streaming compose. When both are configured the same MoE blocks
    are instrumented once with a factory carrying both the recorder and the
    provider; when only one is configured the other's wiring is absent.

    Parameters
    ----------
    config : PipelineConfig
        Parsed pipeline configuration.
    config_dir : Path | None
        Directory the config file was read from, used as the base for relative
        paths. `None` keeps the current working directory as the base, which is
        what callers constructing a `PipelineConfig` in memory want.
    loop : asyncio.AbstractEventLoop | None
        The engine's running event loop, required only for streaming. The
        streaming provider's `load` runs on the runner thread and bridges to
        the expert bank's async `read` via `run_coroutine_threadsafe(..., loop)`, so
        the loop it schedules onto must be handed in here — the runner thread
        has no running loop of its own to discover.
    metrics : GenerationMetrics | None
        Streaming counters the provider accumulates into (cache hits/misses,
        demand stall time). Owned by the caller so it can report them after
        generation; ignored unless `config` requests streaming.

    Returns
    -------
    GenerationPipeline
        A pipeline wired to the backend `config` selects.

    Raises
    ------
    ValueError
        If `config` names an unknown backend, or requests streaming without a
        running event loop.
    FileExistsError
        If tracing would overwrite an existing trace file without
        `[trace_settings].overwrite_output`. Checked before the model load so the
        failure costs seconds rather than minutes.
    ExpertBankCompatibilityError
        If an expert bank does not match the loaded model.
    """

    if config.llm.backend != "mlx_metal":
        raise ValueError(f"Unknown backend: {config.llm.backend!r}")

    # Backend imports stay inside the branch: only the composition root may
    # import concrete backends, and only for the backend actually selected.
    import mlx.core as mx

    from preempt.backends.mlx_metal.instrument import (
        mlx_instrument_model,
        mlx_strip_instrumented_expert_weights,
    )
    from preempt.backends.mlx_metal.instrumented.qwen3_next_moe import (
        make_qwen3next_moe_wrapper_factory,
    )
    from preempt.backends.mlx_metal.layer_discovery import resolve_mlx_target_layers
    from preempt.backends.mlx_metal.loader import load_mlx_model
    from preempt.backends.mlx_metal.recorder import MlxExpertRoutingRecorder
    from preempt.backends.mlx_metal.residency import MlxExpertResidency
    from preempt.backends.mlx_metal.runner import MlxModelRunner
    from preempt.core.encoding import parse_payload_encoding_tag
    from preempt.core.sinks import ParquetEventSink
    from preempt.datamodel.tracing.context import TraceRunContext
    from preempt.datamodel.tracing.expert_routing import ExpertRoutingEvent
    from preempt.engine.expert_cache import ExpertCache
    from preempt.engine.layer_resolution import ensure_no_target_layer_overlap
    from preempt.engine.expert_loaders import DiskBackedExpertLoader
    from preempt.storage.expert_io import ExpertBank

    base_dir = config_dir if config_dir is not None else Path.cwd()

    output_path: Path | None = None
    if config.trace_settings is not None:
        output_path = base_dir / config.trace_settings.output_path

        # The sink checks this too, but only lazily on first write -- by which
        # point a multi-minute model load has already been paid for.
        if output_path.exists() and not config.trace_settings.overwrite_output:
            raise FileExistsError(
                f"Trace output already exists: {output_path}. Set "
                "`overwrite_output = true` under [trace_settings], or choose another "
                "`output_path`."
            )

    # --- Streaming components (expert bank/residency/cache/provider). Pure of
    # the model, so building them ahead of the load is harmless; the provider
    # must exist before the wrapper factory that references it. ---
    expert_bank: ExpertBank | None = None
    residency: MlxExpertResidency | None = None
    provider: DiskBackedExpertLoader | None = None
    if config.stream_settings is not None:
        if loop is None:
            raise ValueError(
                "Streaming requires the engine's running event loop: the "
                "provider bridges its sync `load` to the expert bank's async "
                "`read` via `run_coroutine_threadsafe`. Call `build_pipeline` "
                "from within the loop (see `main`)."
            )

        expert_bank_path = base_dir / config.stream_settings.expert_bank_path
        expert_bank = ExpertBank(
            expert_bank_path,
            bypass_page_cache=config.stream_settings.bypass_page_cache,
        )
        residency = MlxExpertResidency(
            encoding=parse_payload_encoding_tag(expert_bank.manifest.payload_encoding)  # type: ignore
        )
        cache = ExpertCache(budget_bytes=config.stream_settings.memory_bytes_budget)
        provider = DiskBackedExpertLoader(
            expert_bank=expert_bank,
            residency=residency,
            cache=cache,
            loop=loop,
            metrics=metrics,
        )

    # Streaming needs the never-materialize load: lazy leaves every weight an
    # unevaluated mmap-backed array so the strip can drop the experts before any
    # eval. Non-streaming keeps stock eager behaviour.
    lazy = config.stream_settings is not None
    print(f"Loading model: {config.llm.model_id}")
    loaded = load_mlx_model(config.llm.model_id, lazy=lazy)

    # Which blocks to instrument: tracing targets when tracing, else the
    # streaming default target over the architecture's MoE blocks.
    target_config: TargetLayerConfig | None = None
    if config.trace_settings is not None:
        target_config = config.trace_settings.to_target_layer_config()
    elif expert_bank is not None:
        target_config = _streaming_target_config(
            config.llm.architecture,
            len(expert_bank.manifest.model_moe_spec.moe_block_idxs),
        )

    layers = []
    if target_config is not None:
        resolved = resolve_mlx_target_layers(loaded.model, target_config)
        ensure_no_target_layer_overlap(resolved)
        layers = [candidate for matches in resolved.values() for candidate in matches]

    recorder: MlxExpertRoutingRecorder | None = None
    sink: ParquetEventSink | None = None
    if config.trace_settings is not None and output_path is not None:
        run_context = TraceRunContext.with_generated_run_id(
            run_id_prefix=config.trace_settings.run_id_prefix,
            model_id=config.llm.model_id,
            model_architecture=config.llm.architecture,
            model_revision=config.llm.revision,
        )
        recorder = MlxExpertRoutingRecorder(run_context=run_context)

    if layers:
        mlx_instrument_model(
            loaded.model,
            layers,
            make_qwen3next_moe_wrapper_factory(
                recorder,
                capture_gate_logits=(
                    config.trace_settings.capture_gate_logits
                    if config.trace_settings is not None
                    else False
                ),
                provider=provider,
                model_fingerprint=(
                    expert_bank.model_fingerprint if expert_bank is not None else None
                ),
                residency=residency,
            ),
        )
        print(f"Instrumented {len(layers)} router layer(s).")

    if config.trace_settings is not None and output_path is not None:
        sink = ParquetEventSink(
            path=output_path,
            schema=ExpertRoutingEvent.arrow_schema(),
            batch_size=config.trace_settings.batch_size,
            overwrite=config.trace_settings.overwrite_output,
        )

    if expert_bank is not None:
        # Read the model's own MoE config off a resolved block *before* the strip
        # removes `switch_mlp`, and gate the expert bank against it. Reading
        # `model_moe_spec`` from the expert bank would compare the expert bank to itself.
        block = dict(loaded.model.named_modules())[layers[0].layer_path].inner
        expert_bank.check_model_compatibility(
            model_id=config.llm.model_id,
            num_routed_experts=int(block.switch_mlp.gate_proj.num_experts),
            top_k=int(block.top_k),
            moe_block_idxs=tuple(sorted(candidate.block_idx for candidate in layers)),  # type: ignore
        )

        mlx_strip_instrumented_expert_weights(loaded.model, layers)
        # Materialize the dense backbone alone: the experts are out of the
        # parameter tree, so this never pulls the ~18 GB off disk.
        mx.eval(loaded.model.parameters())

    return GenerationPipeline(
        runner=MlxModelRunner(loaded.model),
        tokenizer=loaded.tokenizer,
        max_tokens=config.generation_settings.max_tokens,
        prefill_chunk_size=config.generation_settings.prefill_chunk_size,
        recorder=recorder,
        sink=sink,
        on_step=_print_step,
    )


async def _run(
    config: PipelineConfig, args: argparse.Namespace, config_dir: Path
) -> None:
    """Build and run a pipeline inside one running event loop.

    Build and generate share the loop so a streaming provider can schedule
    expert bank reads onto it. The model load blocks the loop, but nothing else contends for
    it during startup, so that is fine.
    """
    loop = asyncio.get_running_loop()
    metrics = GenerationMetrics()

    pipeline = build_pipeline(config, config_dir=config_dir, loop=loop, metrics=metrics)
    result = await pipeline.generate(args.prompt, max_tokens=args.max_tokens)

    print(f"Output:\n{result.text!r}")
    print(
        f"\n{result.metrics.tokens_forwarded} token position(s) forwarded, "
        f"\n{result.metrics.records_written} trace record(s), "
        f"\n{result.metrics.total_duration_s:.1f}s total."
    )

    if config.stream_settings is not None:
        demands = metrics.cache_hits + metrics.cache_misses
        hit_rate = metrics.cache_hits / demands if demands else 0.0
        print(
            f"\nExpert cache: {metrics.cache_hits} hit(s), "
            f"\n{metrics.cache_misses} miss(es) over {demands} demand(s) "
            f"(hit rate {hit_rate:.1%}), {metrics.demand_stall_s:.1f}s demand stall."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured inference pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    config = read_and_validate_toml(args.config, PipelineConfig)
    asyncio.run(_run(config, args, args.config.resolve().parent))


if __name__ == "__main__":
    main()
