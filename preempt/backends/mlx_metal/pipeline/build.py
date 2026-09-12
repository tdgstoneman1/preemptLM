from __future__ import annotations

from pathlib import Path
import asyncio

from rich.console import Console

import mlx.core as mx

from preempt.core.enums import Backends
from preempt.core.config.pipeline import PipelineConfig
from preempt.core.config.target_layers import (
    TargetLayers,
)
from preempt.engine.expert_io.cache_manager import ExpertCacheManager
from preempt.engine.expert_io.loader import DiskBackedExpertLoader
from preempt.engine.layer_resolution import (
    LayerCandidate,
    ensure_no_target_layer_overlap,
    target_layers_for_model,
)
from preempt.engine.pipeline import GenerationPipeline
from preempt.engine.metrics import GenerationMetrics

from preempt.datamodel.expert_bank.encoding import parse_encoding_tag
from preempt.datamodel.expert_bank.banks import (
    BaseExpertBank,
    PreadExpertBank,
)
from preempt.utils.pipeline_utils import (
    get_recorder,
    get_parquet_sink,
)
from ..expert_cache import MlxExpertCache
from ..module_wrappers.instrument import (
    mlx_instrument_model,
    mlx_strip_instrumented_expert_weights,
    resolve_mlx_target_layers,
)
from ..registry import DefaultArchClassRegistry
from ..recorder import MlxTraceRecorder
from ..types import MlxLoadedModel
from ..utils import load_mlx_model

from .runner import MlxModelRunner

# TODO pass 'streamed_expert_matmul' mode for MoE wrapper module (from config)
# TODO pass max_kv_size from config


def _get_expert_io_deps(
    *,
    config: PipelineConfig,
    metrics: GenerationMetrics | None,
    event_loop: asyncio.AbstractEventLoop,
) -> tuple[BaseExpertBank, MlxExpertCache, DiskBackedExpertLoader]:
    if config.stream_settings is None:
        raise ValueError()

    expert_bank = PreadExpertBank(
        config.stream_settings.expert_bank_path,
        bypass_page_cache=config.stream_settings.bypass_page_cache,
    )
    encoding = parse_encoding_tag(expert_bank.manifest.encoding)
    cache_manager = ExpertCacheManager(
        budget_bytes=config.stream_settings.memory_budget_bytes
    )
    cache = MlxExpertCache(
        encoding=encoding,
        manager=cache_manager,
    )
    loader = DiskBackedExpertLoader(
        expert_bank=expert_bank,
        cache=cache,
        cache_manager=cache_manager,
        event_loop=event_loop,
        metrics=metrics,
    )
    return expert_bank, cache, loader


def _moe_blocks_for_model(
    loaded: MlxLoadedModel, target_layer_config: TargetLayers | None
) -> list[LayerCandidate]:
    blocks = []
    if target_layer_config is not None:
        resolved = resolve_mlx_target_layers(loaded.model, target_layer_config)
        ensure_no_target_layer_overlap(resolved)
        blocks.extend(
            [candidate for matches in resolved.values() for candidate in matches]
        )
    return blocks


def _instrument_model(
    loaded: MlxLoadedModel,
    *,
    moe_blocks: list[LayerCandidate],
    config: PipelineConfig,
    expert_bank: BaseExpertBank | None,
    expert_loader: DiskBackedExpertLoader | None,
    expert_cache: MlxExpertCache | None,
    recorder: MlxTraceRecorder | None,
) -> int:
    capture_gate_logits = (
        config.trace_settings.capture_gate_logits
        if config.trace_settings is not None
        else False
    )
    model_fingerprint = (
        expert_bank.model_fingerprint if expert_bank is not None else None
    )
    wrapper_cls = DefaultArchClassRegistry.get_moe_wrapper(loaded.model)
    wrapper_factory = wrapper_cls.make_wrapper_factory(
        recorder=recorder,
        capture_gate_logits=capture_gate_logits,
        expert_loader=expert_loader,
        model_fingerprint=model_fingerprint,
        expert_cache=expert_cache,
    )
    mlx_instrument_model(
        loaded.model,
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
    event_loop: asyncio.AbstractEventLoop | None = None,
    metrics: GenerationMetrics | None = None,
    stream_experts: bool,
    profile: bool,
    console: Console | None = None,
) -> GenerationPipeline:

    print_ = lambda x: (
        console.print(x) if console is not None else lambda x: print(x, flush=True)
    )
    # * Device settings
    device = mx.default_device()
    print_(f"Default device: {mx.device_info(device)!r}")

    memory_budget = (
        f"{config.stream_settings.memory_budget_gb} GB"
        if config.stream_settings
        else "N/A"
    )
    print_(f"Expert cache memory budget: {memory_budget}")

    # * Validate config
    if config.llm.backend != Backends.MLX:
        raise ValueError(
            f"Invalid LLM backend for MLX pipeline: {config.llm.backend!r}. For MLX, "
            f"set to {Backends.MLX!r}."
        )
    # * Resolve path to model the model
    try:
        path = Path(config.llm.model_id).absolute().resolve(strict=True).as_posix()
    except FileNotFoundError:
        path = config.llm.model_id

    print_(f"Loading model: {path!r}")
    loaded = load_mlx_model(path, lazy=stream_experts)

    # * Configure optional streaming
    if stream_experts:
        event_loop = event_loop or asyncio.get_event_loop()
        expert_bank, cache, loader = _get_expert_io_deps(
            config=config,
            metrics=metrics,
            event_loop=event_loop,
        )
    else:
        expert_bank, cache, loader = None, None, None

    # * Instrument the model
    moe_cls = DefaultArchClassRegistry.get_moe_module_cls(loaded.model).__name__
    target_layer_config = target_layers_for_model(
        config,
        expert_bank=expert_bank,
        target_layer_class=moe_cls,
    )
    moe_blocks = _moe_blocks_for_model(loaded, target_layer_config)

    if profile:
        recorder = get_recorder(config, MlxTraceRecorder)
        sink = get_parquet_sink(config)
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
    print_(f"Instrumented {num_instrumented} MoE block(s)")

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
    )
    return GenerationPipeline(
        runner=runner,
        tokenizer=loaded.tokenizer,
        max_tokens=config.generation_settings.max_tokens,
        prefill_chunk_size=config.generation_settings.prefill_chunk_size,
        recorder=recorder,
        sink=sink,
        on_step=None,
    )
