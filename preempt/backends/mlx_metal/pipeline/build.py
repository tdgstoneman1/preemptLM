from __future__ import annotations

from pathlib import Path

from rich.console import Console

import asyncio

import mlx.core as mx

from preempt.core.enums import Backends
from preempt.core.protocols import IExpertLoader

from preempt.config.pipeline import PipelineConfig
from preempt.config.target_layers import (
    TargetLayers,
)
from preempt.engine.metrics import GenerationMetrics
from preempt.engine.pipeline import GenerationPipeline
from preempt.engine.expert_cache import ExpertCacheManager
from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
)
from preempt.engine.expert_loaders import DiskBackedExpertLoader

from preempt.expert_bank.encoding import parse_encoding_tag
from preempt.expert_bank.banks import BaseExpertBank, PreadExpertBank, MmapExpertBank

from preempt.datamodel.tracing.context import TraceRunContext

from preempt.utils.pipeline_utils import (
    validate_output_path,
    get_parquet_sink,
    target_layers_for_model,
)
from ..instrumentation.instrument import (
    mlx_instrument_model,
    mlx_strip_instrumented_expert_weights,
)
from ..registry import DefaultArchClassRegistry
from ..layer_discovery import resolve_mlx_target_layers
from ..recorder import MoERecorder
from ..expert_cache import MlxExpertCache
from ..types import MlxLoadedModel
from ..utils import load_mlx_model

from .runner import MlxModelRunner


def _get_streaming_deps(
    *,
    base_dir: Path,
    config: PipelineConfig,
    metrics: GenerationMetrics | None,
    event_loop: asyncio.AbstractEventLoop | None,
) -> tuple[BaseExpertBank, MlxExpertCache, DiskBackedExpertLoader]:
    if config.stream_settings is None:
        raise ValueError()

    if event_loop is None:
        raise ValueError()

    expert_bank_path = base_dir / config.stream_settings.expert_bank_path
    expert_bank = PreadExpertBank(
        expert_bank_path,
        bypass_page_cache=config.stream_settings.bypass_page_cache,
    )
    cache = MlxExpertCache(
        encoding=parse_encoding_tag(expert_bank.manifest.encoding)  # type: ignore
    )
    cache_manager = ExpertCacheManager(
        budget_bytes=config.stream_settings.memory_bytes_budget
    )
    loader = DiskBackedExpertLoader(
        expert_bank=expert_bank,
        cache=cache,  # type: ignore
        cache_manager=cache_manager,
        loop=event_loop,
        metrics=metrics,
    )
    return expert_bank, cache, loader


def _get_recorder(
    config: PipelineConfig, output_path: Path | None
) -> MoERecorder | None:
    if config.trace_settings is None or output_path is None:
        return

    ctx = TraceRunContext.with_generated_run_id(
        run_id_prefix=config.trace_settings.run_id_prefix,
        model_id=config.llm.model_id,
        model_architecture=config.llm.architecture,
        model_revision=config.llm.revision,
    )
    return MoERecorder(run_context=ctx)


def _moe_blocks_for_model(
    loaded_model: MlxLoadedModel, target_layer_config: TargetLayers | None
) -> list[LayerCandidate]:
    blocks = []
    if target_layer_config is not None:
        resolved = resolve_mlx_target_layers(loaded_model.model, target_layer_config)
        ensure_no_target_layer_overlap(resolved)
        blocks.extend(
            [candidate for matches in resolved.values() for candidate in matches]
        )
    return blocks


def _instrument_model(
    loaded_model: MlxLoadedModel,
    *,
    moe_blocks: list[LayerCandidate],
    config: PipelineConfig,
    expert_bank: BaseExpertBank | None,
    expert_loader: IExpertLoader | None,
    expert_cache: MlxExpertCache | None,
    recorder: MoERecorder | None,
) -> int:
    capture_gate_logits = (
        config.trace_settings.capture_gate_logits
        if config.trace_settings is not None
        else False
    )
    model_fingerprint = (
        expert_bank.model_fingerprint if expert_bank is not None else None
    )
    wrapper_cls = DefaultArchClassRegistry.get_moe_wrapper(loaded_model.model)
    wrapper_factory = wrapper_cls.make_factory(
        recorder=recorder,
        capture_gate_logits=capture_gate_logits,
        expert_loader=expert_loader,
        model_fingerprint=model_fingerprint,
        expert_cache=expert_cache,
    )
    mlx_instrument_model(
        loaded_model.model,
        candidates=moe_blocks,
        wrapper_factory=wrapper_factory,
    )
    return len(moe_blocks)


def _evaluate_model(
    loaded_model: MlxLoadedModel,
    *,
    moe_blocks: list[LayerCandidate],
    config: PipelineConfig,
    expert_bank: BaseExpertBank | None,
) -> None:
    if expert_bank is not None:
        block = dict(loaded_model.model.named_modules())[moe_blocks[0].layer_path].inner
        expert_bank.check_model_compatibility(
            model_id=config.llm.model_id,
            num_routed_experts=int(block.switch_mlp.gate_proj.num_experts),
            top_k=int(block.top_k),
            moe_block_idxs=tuple(sorted(candidate.block_idx for candidate in moe_blocks)),  # type: ignore
        )
        # Only evaluate dense backbone
        mlx_strip_instrumented_expert_weights(loaded_model.model, moe_blocks)
        mx.eval(loaded_model.model.parameters())


def mlx_build_generation_pipeline(
    config: PipelineConfig,
    *,
    config_dir: Path | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    metrics: GenerationMetrics | None = None,
    stream_experts: bool,
    save_traces: bool,
    console: Console | None = None,
) -> GenerationPipeline:

    def maybe_print_to_console(msg: str) -> None:
        nonlocal console
        if console is not None:
            console.print(msg)
        else:
            print(msg, flush=True)

    # * Device settings
    device = mx.default_device()
    maybe_print_to_console(f"Default device: {mx.device_info(device)!r}")

    memory_budget = (
        f"{config.stream_settings.memory_budget_gb} GB"
        if config.stream_settings
        else "N/A"
    )
    maybe_print_to_console(f"Expert cache memory budget: {memory_budget}")

    # * Validate config
    if config.llm.backend != Backends.MLX:
        raise ValueError(
            f"Invalid LLM backend for MLX pipeline: {config.llm.backend!r}. For MLX, "
            f"set to {Backends.MLX!r}."
        )

    # * Validate trace output path
    base_dir, output_path = validate_output_path(config, config_dir)

    # * Resolve path to model the model
    try:
        path = Path(config.llm.model_id).absolute().resolve(strict=True).as_posix()
    except FileNotFoundError:
        path = config.llm.model_id

    maybe_print_to_console(f"Loading model: {path!r}")
    loaded = load_mlx_model(path, lazy=stream_experts)

    # * Configure optional streaming
    if stream_experts:
        expert_bank, cache, loader = _get_streaming_deps(
            base_dir=base_dir, config=config, metrics=metrics, event_loop=event_loop
        )
    else:
        expert_bank, cache, loader = None, None, None

    # * Instrument the model
    target_layer_config = target_layers_for_model(config, expert_bank)
    moe_blocks = _moe_blocks_for_model(loaded, target_layer_config)

    if save_traces:
        recorder = _get_recorder(config, output_path)
        sink = get_parquet_sink(config, output_path)
    else:
        recorder, sink = None, None

    num_instrumented = _instrument_model(
        loaded,
        moe_blocks=moe_blocks,
        config=config,
        expert_bank=expert_bank,
        expert_loader=loader,
        expert_cache=cache,
        recorder=recorder,
    )
    maybe_print_to_console(f"Instrumented {num_instrumented} MoE block(s)")

    _evaluate_model(
        loaded,
        moe_blocks=moe_blocks,
        config=config,
        expert_bank=expert_bank,
    )
    runner = MlxModelRunner(
        loaded.model,
        # max_tokens=config.generation_settings.max_tokens,
        # prefill_chunk_size=config.generation_settings.prefill_chunk_size,
    )  # TODO pass max_kv_size from config

    return GenerationPipeline(
        runner=runner,
        tokenizer=loaded.tokenizer,
        max_tokens=config.generation_settings.max_tokens,
        prefill_chunk_size=config.generation_settings.prefill_chunk_size,
        recorder=recorder,
        sink=sink,
        on_step=None,
    )
