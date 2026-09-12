import pytest

from collections.abc import Callable

import mlx.core as mx

import asyncio
import gc
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preempt.config.pipeline import PipelineConfig

from preempt.backends.mlx_metal.pipeline.build import mlx_build_generation_pipeline

# ! CURRENTLY NOT WORKING DUE TO SEGMENTATION FAULTS WITH MLX AND PYTEST


async def run(
    prompt: str,
    config: PipelineConfig,
    stream_experts: bool,
) -> str:
    """Helper to build and run pipeline and ensure all C++/Metal/thread resources
    are cleaned up.
    """
    pipeline = mlx_build_generation_pipeline(
        config,
        event_loop=asyncio.get_running_loop(),
        stream_experts=stream_experts,
        profile=False,
    )
    try:
        result = await pipeline.generate(prompt, max_tokens=16)
        return result.text

    finally:
        for _, module in pipeline.runner.model.named_modules():
            if hasattr(module, "expert_loader") and module.expert_loader is not None:
                module.expert_loader.close()

        if hasattr(pipeline.runner, "kv_cache"):
            pipeline.runner.kv_cache.clear()

        del pipeline
        gc.collect()
        mx.metal.clear_cache()


@pytest.mark.slow
async def test_outputs_identical_with_quantized_expert_bank(
    prompt: str,
    make_pipe_config: Callable[[bool], PipelineConfig],
) -> None:
    """
    Tests that generating with a streaming expert bank produces identical
    output to generating with the full model in memory.
    """
    pipe_config = make_pipe_config(True)

    text_with_expert_bank = await run(prompt, pipe_config, True)
    text_without_expert_bank = await run(prompt, pipe_config, False)

    assert text_with_expert_bank and text_without_expert_bank
    assert text_with_expert_bank == text_without_expert_bank


@pytest.mark.slow
@pytest.mark.skip(reason="Not enough VRAM for full precision.")
async def test_outputs_identical_with_unquantized_expert_bank(
    prompt: str,
    make_pipe_config: Callable[[bool], PipelineConfig],
) -> None:
    pipe_config = make_pipe_config(False)
    text_with_expert_bank = await run(prompt, pipe_config, True)
    text_without_expert_bank = await run(prompt, pipe_config, False)

    assert text_with_expert_bank == text_without_expert_bank
