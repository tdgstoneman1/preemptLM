from pathlib import Path

import textwrap

from preempt.config.pipeline import PipelineConfig

from preempt.engine.sinks import ParquetTraceSink
from preempt.engine.metrics import GenerationMetrics

from preempt.datamodel.tracing.expert_selection import ExpertSelectionTrace


def validate_output_path(
    config: PipelineConfig, config_dir: Path | None
) -> tuple[Path, Path | None]:
    base_dir = config_dir if config_dir is not None else Path.cwd()

    if config.trace_settings is not None:
        output_path = base_dir / config.trace_settings.output_path
        if output_path.exists() and not config.trace_settings.overwrite_output:
            raise FileExistsError(
                f"File already exists at {output_path.as_posix()!r}. Configure "
                "trace settings with a different path or `overwrite_output=True`."
            )
        return base_dir, output_path

    return base_dir, None


def get_parquet_sink(
    config: PipelineConfig, output_path: Path | None
) -> ParquetTraceSink | None:
    if config.trace_settings is not None and output_path is not None:
        return ParquetTraceSink(
            path=output_path,
            schema=ExpertSelectionTrace.arrow_schema(),
            batch_size=config.trace_settings.batch_size,
            overwrite=config.trace_settings.overwrite_output,
        )


# TODO move to dedicated logging module
# TODO add prefill_s field to GenerationMetrics
def generation_metrics_log_msg(
    metrics: GenerationMetrics, prefill_s: float | int
) -> str:
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
