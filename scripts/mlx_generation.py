"""
Example usage (macOS)::

    uv run scripts/mlx_generation.py\
        --config configs/mlx/unsloth-qwen3.6-35b-4bit.toml\
        --stream-experts\
        --max-tokens 1024\
        --prompt "Fire and fury like "
        --profile\
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
from pathlib import Path
from time import perf_counter
import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from rich.console import Group
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.traceback import install

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.core.config.pipeline import PipelineConfig

from preempt.engine.metrics import GenerationMetrics, StepMetrics

from preempt.backends.mlx_metal.pipeline.build import mlx_build_generation_pipeline

from preempt.utils.pipeline_utils import (
    generation_metrics_log_msg,
    cache_metrics_log_msg,
)

# TODO pass 'streamed_expert_matmul' param from config
# TODO separate panel for final generation stats


async def run(
    config: PipelineConfig, args: argparse.Namespace, console: Console
) -> None:
    metrics = GenerationMetrics()  # TODO configure this in TOML config
    if config.stream_settings and (maxsize := args.expert_cache_max_gb):
        config.stream_settings.memory_budget_gb = maxsize

    pipeline = mlx_build_generation_pipeline(
        config,
        event_loop=asyncio.get_running_loop(),
        metrics=metrics,
        stream_experts=args.stream_experts,
        profile=args.profile,
        console=console,
    )
    prompt = (
        args.prompt
        if args.prompt is not None
        else Prompt.ask("[bold]Enter prompt[/bold]", console=console)
    )

    with Live(
        console=console,
        screen=False,
        auto_refresh=True,
        refresh_per_second=5,
        vertical_overflow="visible",
    ) as live:
        prompt_panel = Panel(
            prompt,
            title="Input",
            title_align="left",
            border_style="blue",
            width=console.width,
        )
        initial_view = Align.center(
            Group(
                prompt_panel,
                Panel(
                    "",
                    title="Output",
                    title_align="left",
                    border_style="green",
                    width=console.width,
                ),
                Text("Prefill...", justify="left"),
            ),
            vertical="top",
        )
        live.update(initial_view)

        num_output_toks = 0
        decoded_text_parts = []
        step_times: list[float] = []
        prefill_s: float = 0
        start_t = perf_counter()

        def stream_printer(step: StepMetrics) -> None:
            nonlocal num_output_toks, decoded_text_parts, step_times, metrics, prefill_s, args, start_t

            output = step.generated_token_id
            decoded_text_parts.append(pipeline.tokenizer.decode([output]))
            num_output_toks += 1

            chat_panel = Panel(
                "".join(decoded_text_parts),
                title="Output",
                title_align="left",
                border_style="green",
                width=console.width,
            )
            elapsed = int(perf_counter() - start_t)
            clock_and_toks = Text(
                f"{str(datetime.timedelta(seconds=elapsed))} | {num_output_toks} tokens",
                justify="left",
            )
            if len(step_times) == 0:
                prefill_s = step.duration_s
                step_times.clear()  # prefill time doesn't count

            step_times.append(step.duration_s)

            # Calculate token throughput
            window_size = 4
            window = step_times[-min(len(step_times), window_size) :]
            toks_per_s = 1 / float(np.mean(window))

            num_routed = metrics.cache_hits + metrics.cache_misses
            hit_rate = metrics.cache_hits / num_routed if metrics.cache_hits > 0 else 0
            miss_rate = (
                metrics.cache_misses / num_routed if metrics.cache_hits > 0 else 0
            )

            stats = Text(
                f"step time: {step.duration_s:.2f}s | "
                f"tok/s: {toks_per_s:.2f} | "
                f"cache hits: {metrics.cache_hits:,} ({hit_rate:.2%}) | "
                f"cache misses: {metrics.cache_misses:,} ({miss_rate:.2%})",
                justify="right",
            )
            view = Align.center(
                Group(
                    prompt_panel,
                    chat_panel,
                    Columns([clock_and_toks, stats], expand=True),
                ),
                vertical="bottom",
            )
            live.update(view)

        result = await pipeline.generate(
            prompt, max_tokens=args.max_tokens, on_step=stream_printer
        )

    console.print(generation_metrics_log_msg(result.metrics, prefill_s))
    console.print(cache_metrics_log_msg(metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference pipeline.")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stream-experts",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expert-cache-max-gb",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Whether to save traces to parquet file.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    if args.debug:
        install(show_locals=True)

    config = PipelineConfig.from_toml(args.config)
    console = Console()

    try:
        asyncio.run(run(config, args, console))

    except KeyboardInterrupt:
        console.print("Exiting...")


if __name__ == "__main__":
    main()
