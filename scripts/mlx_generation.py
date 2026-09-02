"""
Example usage (macOS host)::

    uv run scripts/mlx_generation.py\
        --config configs/mlx/unsloth-qwen3.6-35b-4bit.toml\
        --stream-experts\
        --save-traces\
        --max-tokens 1024\
        --prompt "Fire and fury like "
"""

from __future__ import annotations

import asyncio

import argparse

from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from rich.console import Group
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

import numpy as np

from pathlib import Path

import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.config.pipeline import PipelineConfig

from preempt.engine.metrics import GenerationMetrics, StepMetrics

from preempt.backends.mlx_metal.pipeline.build import mlx_build_generation_pipeline

from preempt.utils.io_utils import read_and_validate_toml
from preempt.utils.pipeline_utils import (
    generation_metrics_log_msg,
    cache_metrics_log_msg,
)

# TODO pass 'expert_matmul' param from config
# TODO separate panel for final generation stats


async def run(
    config: PipelineConfig, args: argparse.Namespace, config_dir: Path, console: Console
) -> None:
    metrics = GenerationMetrics()  # TODO configure this in TOML config
    pipeline = mlx_build_generation_pipeline(
        config,
        config_dir=config_dir,
        event_loop=asyncio.get_running_loop(),
        metrics=metrics,
        stream_experts=args.stream_experts,
        save_traces=args.save_traces,
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
        refresh_per_second=10,
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

        generated_tokens: list[int] = []
        decoded_text = ""
        step_times: list[float] = []
        prefill_s: float = 0

        def stream_printer(step: StepMetrics) -> None:
            nonlocal generated_tokens, decoded_text, step_times, metrics, prefill_s, args

            generated_tokens.append(step.generated_token_id)
            full_text = pipeline.tokenizer.decode(generated_tokens)
            decoded_text = full_text

            chat_panel = Panel(
                decoded_text,
                title="Output",
                title_align="left",
                border_style="green",
                width=console.width,
            )

            tok_count = Text(f"{len(generated_tokens)} tokens", justify="left")
            if len(step_times) == 0:
                prefill_s = step.duration_s

            if len(step_times) == 1:
                step_times = [step.duration_s]  # prefill time doesn't count
            else:
                step_times.append(step.duration_s)

            toks_per_s = 1 / float(np.mean(step_times))

            num_routed = metrics.cache_hits + metrics.cache_misses
            hit_rate = metrics.cache_hits / num_routed if args.stream_experts else 0
            miss_rate = metrics.cache_misses / num_routed if args.stream_experts else 0

            stats = Text(
                f"step time: {step.duration_s:.2f}s | "
                f"tok/s: {toks_per_s:.2f} | "
                f"cache hits: {metrics.cache_hits} ({hit_rate:.2%}) | "
                f"cache misses: {metrics.cache_misses} ({miss_rate:.2%})",
                justify="right",
            )
            view = Align.center(
                Group(
                    prompt_panel, chat_panel, Columns([tok_count, stats], expand=True)
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
    parser = argparse.ArgumentParser(description="Run a generation pipeline.")

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--stream-experts", default=False, action="store_true")
    parser.add_argument("--save-traces", default=False, action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)

    args = parser.parse_args()
    config = read_and_validate_toml(args.config, PipelineConfig)
    console = Console()
    try:
        asyncio.run(run(config, args, args.config.resolve().parent, console))

    except KeyboardInterrupt:
        console.print("Exiting...")


if __name__ == "__main__":
    main()
