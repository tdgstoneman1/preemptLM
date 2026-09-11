from typing import TypeVar

from pathlib import Path

import textwrap

from preempt.config.pipeline import PipelineConfig

from preempt.engine.recorder import BaseTraceRecorder
from preempt.engine.sinks import ParquetTraceSink
from preempt.engine.metrics import GenerationMetrics

from preempt.datamodel.tracing.context import TraceRunContext
from preempt.datamodel.tracing.expert_selection import ExpertSelectionTrace

RecorderT = TypeVar("RecorderT", bound=BaseTraceRecorder)


def get_recorder(
    config: PipelineConfig, recorder_cls: type[RecorderT]
) -> RecorderT | None:
    if config.trace_settings is None:
        return

    ctx = TraceRunContext.with_generated_run_id(
        run_id_prefix=config.trace_settings.run_id_prefix,
        model_id=config.llm.model_id,
        model_architecture=config.llm.architecture,
        model_revision=config.llm.revision,
    )
    return recorder_cls(run_context=ctx)


def get_parquet_sink(
    config: PipelineConfig,
) -> ParquetTraceSink | None:
    if config.trace_settings is not None:
        return ParquetTraceSink(
            path=config.trace_settings.output_path,
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
