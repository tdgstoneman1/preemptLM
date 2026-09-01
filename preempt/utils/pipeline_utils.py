from pathlib import Path

import textwrap

from preempt.config.pipeline import PipelineConfig
from preempt.config.target_layers import (
    TargetLayerConfig,
    TargetLayerSpec,
    TargetLayerSearchParams,
)

from preempt.core.sinks import ParquetEventSink

from preempt.datamodel.tracing.expert_routing import ExpertRoutingEvent

from preempt.expert_bank.banks import PreadExpertBank

from preempt.engine.metrics import GenerationMetrics


def validate_output_path(
    config: PipelineConfig, config_dir: Path | None
) -> tuple[Path, Path | None]:
    base_dir = config_dir if config_dir is not None else Path.cwd()

    if config.trace_settings is not None:
        output_path = base_dir / config.trace_settings.output_path
        if output_path.exists() and not config.trace_settings.overwrite_output:
            raise FileExistsError(
                f"File already exists at {output_path.as_posix()!r}. Configure pipeline "
                "trace settings with a different path, or set `overwrite_output = true`)."
            )
        return base_dir, output_path

    return base_dir, None


def get_parquet_sink(
    config: PipelineConfig, output_path: Path | None
) -> ParquetEventSink | None:
    if config.trace_settings is not None and output_path is not None:
        return ParquetEventSink(
            path=output_path,
            schema=ExpertRoutingEvent.arrow_schema(),
            batch_size=config.trace_settings.batch_size,
            overwrite=config.trace_settings.overwrite_output,
        )


# TODO use a registry for this, this is a temporary placeholder
_MOE_BLOCK_CLASS_BY_ARCHITECTURE: dict[str, str] = {
    "qwen3-next": "Qwen3NextSparseMoeBlock",
}


# TODO pass registry as an arg
def target_layers_for_architecture(
    architecture: str, target_layer_count: int | None
) -> TargetLayerConfig:
    block_class = _MOE_BLOCK_CLASS_BY_ARCHITECTURE.get(architecture)
    if block_class is None:
        raise ValueError(
            f"No MoE block class known for architecture {architecture!r}; "
            f"streaming supports {sorted(_MOE_BLOCK_CLASS_BY_ARCHITECTURE)}."
        )
    return TargetLayerConfig(
        target_layers=(
            TargetLayerSpec(
                name="stream-moe",  # TODO change this to something meaningful
                search_params=TargetLayerSearchParams(
                    layer_class=block_class, count=target_layer_count
                ),
            ),
        )
    )


def target_layers_for_model(
    config: PipelineConfig, expert_bank: PreadExpertBank | None
) -> TargetLayerConfig | None:

    if config.trace_settings is not None:
        return config.trace_settings.to_target_layer_config()

    elif expert_bank is not None:
        return target_layers_for_architecture(
            config.llm.architecture,
            len(expert_bank.manifest.model_moe_spec.moe_block_idxs),
        )


# TODO move to dedicated logging module
# TODO add prefill_s field to GenerationMetrics
def generation_metrics_log_msg(metrics: GenerationMetrics, prefill_s: float | int):
    return textwrap.dedent(f"""
    Generation stats
    ----------------
    Total time: {metrics.total_duration_s:.2f} seconds
    Prefill time: {prefill_s:.2f} seconds
    Trace records written: {metrics.records_written:,}
    """)


# TODO move to dedicated logging module
def cache_metrics_log_msg(metrics: GenerationMetrics) -> str:
    demands = metrics.cache_hits + metrics.cache_misses
    hit_rate = metrics.cache_hits / demands if demands else 0.0
    miss_rate = metrics.cache_misses / demands if demands else 0.0

    prefetch_mb = metrics.prefetched_bytes / 1024**2
    wasted_mb = metrics.wasted_prefetch_bytes / 1024**2

    return textwrap.dedent(f"""
    Expert bank stats
    -----------------
    Total experts routed: {demands:,}
    Total stall time: {metrics.demand_stall_s:.2f} seconds

    Cache hits: {metrics.cache_hits} ({hit_rate:.1%})
    Cache misses: {metrics.cache_misses} ({miss_rate:.1%})
    
    Total read from disk: {prefetch_mb:,.2f} MB 
    Total wasted disk prefetches: {wasted_mb:,.2f} MB 
    """)
