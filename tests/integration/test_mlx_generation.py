from __future__ import annotations

import numpy as np

from rich import print

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.core.config.pipeline import PipelineConfig

from preempt.engine.metrics import GenerationMetrics, StepMetrics

from preempt.backends.mlx_metal.pipeline.build import mlx_build_generation_pipeline

from preempt.utils.pipeline_utils import (
    generation_metrics_log_msg,
    cache_metrics_log_msg,
)


async def run(
    prompt: str,
    config: PipelineConfig,
    metrics: GenerationMetrics,
    stream_experts: bool,
) -> None:
    loop = asyncio.get_event_loop()
    pipeline = mlx_build_generation_pipeline(
        config,
        event_loop=loop,
        metrics=metrics,
        stream_experts=stream_experts,
        profile=False,
    )
    prompt = prompt
    print(f"\n[bold]Prompt[/bold]: {prompt}", end="\n")
    print("Prefill...", end="\n")

    num_output_toks = 0
    decoded_text_parts = []
    step_times: list[float] = []
    prefill_s: float = 0

    def stream_printer(step: StepMetrics) -> None:
        nonlocal num_output_toks, decoded_text_parts, step_times, metrics, prefill_s

        output = step.generated_token_id
        decoded_text_parts.append(pipeline.tokenizer.decode([output]))
        num_output_toks += 1

        print(pipeline.tokenizer.decode([output]), flush=True, end="")

        if len(step_times) == 0:
            prefill_s = step.duration_s
            step_times.clear()  # prefill time doesn't count

        step_times.append(step.duration_s)

    result = await pipeline.generate(
        prompt, max_tokens=config.generation_settings.max_tokens, on_step=stream_printer
    )
    print(f"\n\n{num_output_toks} tokens")
    print(generation_metrics_log_msg(result.metrics, prefill_s))
    print(cache_metrics_log_msg(metrics))


def main() -> None:
    prompt = "Explain the differences between Newtonian physics and Einstein's relativistic physics."

    config = PipelineConfig.from_toml(
        Path(__file__).parents[1] / "configs" / "quantized.toml",
        resolve_relative_paths=False,
    )
    metrics = GenerationMetrics()
    stream_experts = True

    try:
        asyncio.run(run(prompt, config, metrics, stream_experts))

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
