from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from mlx_lm import load

import mlx.core as mx

from preempt.backends.mlx_metal.instrumentation.instrument import iter_layer_candidates
from preempt.backends.mlx_metal.instrumentation.qwen3_x_moe import (
    Qwen3_xMoEWrapper,
)
from preempt.backends.mlx_metal.recorder import MoERecorder
from preempt.config.target_layers import TargetLayers
from preempt.datamodel.tracing.context import TraceRunContext, TraceStepContext
from preempt.datamodel.tracing.expert_routing import ExpertRoutingEvent
from preempt.engine.layer_resolution import match_target_layers
from preempt.engine.sinks import ParquetEventSink
from preempt.utils.io_utils import read_and_validate_toml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one MLX Qwen MoE router capture wrapper."
    )
    parser.add_argument(
        "--model",
        help="MLX model ID or local directory accepted by mlx_lm.load().",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML config that resolves exactly one router layer.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with one short word:",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Parquet output file; default is a temporary file.",
    )
    return parser.parse_args()


def get_single_target(config: TargetLayers):
    if len(config.target_layers) != 1:
        raise ValueError("This smoke test expects exactly one [[target_layers]] entry.")

    target = config.target_layers[0]

    if target.search_params.count != 1:
        raise ValueError(
            "This smoke test requires `count = 1` in the target search params."
        )

    return target


def get_module_by_path(model, path: str):
    modules = dict(model.named_modules())

    try:
        return modules[path]
    except KeyError as exc:
        raise RuntimeError(
            f"Resolved module path no longer exists in model: {path!r}"
        ) from exc


def replace_single_router_layer(
    *,
    model,
    path: str,
    block_idx: int,
    recorder: MoERecorder,
) -> None:
    inner = get_module_by_path(model, path)

    wrapper = Qwen3_xMoEWrapper(
        inner=inner,
        recorder=recorder,
        capture_gate_logits=True,
        layer_path=path,
        block_idx=block_idx,
        expert_cache=None,
        expert_loader=None,
        expert_matmul="sequential",
        model_fingerprint=None,
    )

    # MLX module trees expose this common Qwen location as:
    # model.layers.<block_idx>.mlp
    #
    # Keep this explicit for the first smoke test rather than introducing
    # a general path-to-update-tree utility before it is needed.
    model.layers[block_idx].mlp = wrapper

    installed = get_module_by_path(model, path)

    if not isinstance(installed, Qwen3_xMoEWrapper):
        raise RuntimeError(
            f"Wrapper installation failed at {path!r}; "
            f"got {type(installed).__name__}."
        )


async def run(args: argparse.Namespace, output_path: Path) -> None:
    config = read_and_validate_toml(args.config, TargetLayers)
    target = get_single_target(config)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)  # type: ignore

    candidates = tuple(iter_layer_candidates(model))
    matches = match_target_layers(
        candidates=candidates,
        search_params=target.search_params,
    )

    if len(matches) != 1:
        raise RuntimeError(f"Expected one resolved layer; got {len(matches)}.")

    candidate = matches[0]

    if candidate.block_idx is None:
        raise RuntimeError(
            f"Could not derive block_idx from resolved path: {candidate.layer_path!r}"
        )

    print(
        "Resolved one router layer: "
        f"path={candidate.layer_path!r}, "
        f"layer_class={candidate.layer_class!r}, "
        f"block_idx={candidate.block_idx}"
    )

    run_context = TraceRunContext.with_generated_run_id(
        run_id_prefix="mlx-single-router-smoke-test",
        model_id=args.model,
        model_architecture="qwen3-next",
    )

    async with ParquetEventSink(
        path=output_path,
        schema=ExpertRoutingEvent.arrow_schema(),
        batch_size=16,
        overwrite=True,
    ) as sink:
        recorder = MoERecorder(
            run_context=run_context,
        )

        replace_single_router_layer(
            model=model,
            path=candidate.layer_path,
            block_idx=candidate.block_idx,
            recorder=recorder,
        )

        token_id = 1
        input_ids = mx.array([[token_id]], dtype=mx.int32)

        recorder.start_step(
            TraceStepContext(
                sequence_id=0,
                token_idx=0,
                token_id=token_id,
            )
        )

        print("Running one token through the instrumented model...")

        logits = model(input_ids)
        mx.eval(logits)

        records_written = await recorder.flush(sink)
        await sink.flush()

        if records_written != 1:
            raise RuntimeError(
                "Expected one router record from one input token and one "
                f"wrapped layer; got {records_written}."
            )

    # print(f"Model output: {response!r}")
    print(f"Captured and wrote {records_written} router record(s).")
    print(f"Parquet output: {output_path}")

    if records_written < 1:
        raise RuntimeError("Expected at least one router record.")

    parquet_file = pq.ParquetFile(output_path)
    table = parquet_file.read()

    if table.num_rows != records_written:
        raise RuntimeError(
            "Parquet row count does not match flush result: "
            f"{table.num_rows} != {records_written}."
        )

    print(
        f"Verified Parquet file: "
        f"{table.num_rows} row(s), "
        f"{parquet_file.metadata.num_row_groups} row group(s)."
    )


def main() -> None:
    args = parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1.")

    if args.output is not None:
        asyncio.run(run(args, args.output))
        return

    with tempfile.TemporaryDirectory(prefix="preempt-mlx-router-smoke-") as temp_dir:
        output_path = Path(temp_dir) / "router-events.parquet"
        asyncio.run(run(args, output_path))


if __name__ == "__main__":
    main()
