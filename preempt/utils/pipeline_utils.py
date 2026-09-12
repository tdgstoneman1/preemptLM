from typing import TypeVar

import textwrap

from preempt.core.config.pipeline import PipelineConfig

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
# TODO add prefill_s and token throughput to GenerationMetrics
def generation_metrics_log_msg(
    metrics: GenerationMetrics,
    prefill_s: float | int,
) -> str:
    mean_toks_per_s = (metrics.steps[-1].step_idx + 1) / (
        metrics.total_duration_s - prefill_s
    )
    return textwrap.dedent(f"""
    Generation stats
    ----------------
    {metrics.steps[-1].step_idx + 1} tokens generated
    {metrics.records_written:,} traces written to parquet

    Total duration: {metrics.total_duration_s:.2f} seconds
    Prefill time: {prefill_s:.2f} seconds
    Avg throughput: {mean_toks_per_s:.2f} tok/s
    """)


# TODO move to dedicated logging module
def cache_metrics_log_msg(metrics: GenerationMetrics) -> str:
    num_requests = metrics.cache_hits + metrics.cache_misses
    hit_rate = metrics.cache_hits / num_requests if num_requests else 0.0
    miss_rate = metrics.cache_misses / num_requests if num_requests else 0.0

    prefetch_gb = metrics.prefetched_bytes / 1024**3
    wasted_gb = metrics.wasted_prefetch_bytes / 1024**3

    return textwrap.dedent(f"""
    Expert bank stats
    -----------------
    Total experts routed: {num_requests:,}
    Total I/O stall time: {metrics.demand_stall_s:.2f} seconds

    Cache hits: {metrics.cache_hits} ({hit_rate:.1%})
    Cache misses: {metrics.cache_misses} ({miss_rate:.1%})
    
    Total read from disk: {prefetch_gb:,.2f} GB
    Total wasted disk prefetches: {wasted_gb:,.2f} MB 
    """)
