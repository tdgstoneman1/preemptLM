"""Host-only gate: is a per-expert `mx.quantized_matmul` loop bitwise identical
to MLX's fused `mx.gather_qmm`?

Phase 3 of the expert-streaming plan replaces the fused kernel inside the
instrumented MoE forward with an **expert-major** loop -- collect the distinct
experts a batch routes to, compute one at a time -- because that is the only
shape in which an expert's weights can be read from disk, used, and dropped.
Every later task assumes the substitution is numerically free. This script
measures whether it is.

Two paths are built over the same inputs and the same real quantized weights:

- **reference** -- upstream `SwitchGLU.__call__`'s framing around
  `mx.gather_qmm`, reproduced as it actually ships: the
  `mx.expand_dims(x, (-2, -3))` reshape, the `_gather_sort` taken when
  `indices.size >= 64`, the matching `sorted_indices=` argument, and the
  `_scatter_unsort` afterwards. A reference that skipped the sort would be
  testing something we do not run.
- **candidate** -- rows grouped by the expert they route to, one
  `mx.quantized_matmul` per distinct expert over that expert's rows,
  reassembled by inverting the concatenated row permutation. No scatter ops:
  the reassembly is exactly the `_gather_sort`/`_scatter_unsort` idea, done in
  the grouping order the streaming forward will use.

Several candidate decompositions are measured so that a mismatch can be
attributed rather than merely observed:

- `rows_2d` -- each expert's rows as `(n_rows, D)`. This is the layout the
  streaming forward will ship, and it alone decides the gate.
- `rows_3d` -- each expert's rows as `(n_rows, 1, D)`, matching the rank the
  reference feeds `gather_qmm`. MLX selects kernels partly on input rank.
- `rows_1` (`--single-row`) -- one call per row, the most conservative
  decomposition there is. If `rows_1` mismatches too, the candidate's blocking
  is not the variable and the reference is what changed.

A fourth variant, `mx.dequantize` + `mx.matmul` per expert, is timed at every
prefill shape for Task 9's benefit. `waste/src/model.c:2571` finds
dequantize-once-then-GEMM wins on CPU; that is a dense-FMA-versus-LUT-gather
result and does not automatically transfer to Metal.

`--prefill-tokens` takes several values, because MLX dispatches quantized
matmuls on problem size: equality at one shape says nothing about another, and
sweeping is the only way the dispatch threshold becomes visible.

**This script is a gate.** Bitwise identity is required to proceed with the
plan as written. A near-miss is not a pass: a kernel-level precision change
needs the user's explicit sign-off, and the fallback design is theirs to
choose.

Routing indices are synthetic under a fixed seed rather than taken from a real
router pass. The equivalence property must hold for arbitrary indices, and
synthetic ones guarantee the prefill shapes actually span many distinct
experts; the achieved distinct-expert count and rows-per-expert spread are
printed at each shape.

Usage (from the repo root, on the macOS host)::

    python tests/integration/mlx_expert_matmul_spike.py \
        --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
        --prefill-tokens 64 128 256 \
        --single-row \
        --output out/
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Sequence

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

from preempt.backends.mlx_metal.loader import load_mlx_model

# `named_modules()` yields dotted paths like `model.layers.7.mlp.switch_mlp`;
# the transformer block index is the only part of the path that is stable
# across `mlx_lm` refactors.
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class GateFailure(RuntimeError):
    """Raised when the candidate path is not bitwise identical to the reference."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a per-expert `mx.quantized_matmul` loop against "
        "MLX's fused `mx.gather_qmm` on real quantized expert weights."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="MLX model id or local checkpoint directory accepted by `mlx_lm.load`.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Transformer block index whose `switch_mlp` is measured.",
    )
    parser.add_argument(
        "--projection",
        choices=_PROJECTIONS,
        default="gate_proj",
        # One projection settles the equivalence question: all three are the
        # same `QuantizedSwitchLinear` kernel over different weights. Testing
        # all three would triple the runtime and answer nothing new.
        help="Which `SwitchGLU` projection to measure.",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="Token counts for the prefill shapes, measured in order. Each must "
        "be large enough that `tokens * top_k >= 64`, which is what triggers "
        "upstream's sort. Several values sweep the shape axis, which is how the "
        "kernel-selection threshold becomes visible.",
    )
    parser.add_argument(
        "--single-row",
        action="store_true",
        help="Also measure a candidate that issues one `mx.quantized_matmul` per "
        "row. Diagnostic only: it isolates whether a mismatch comes from MLX "
        "picking a different kernel for wider row blocks. It is slow at large "
        "shapes, so it is opt-in.",
    )
    parser.add_argument(
        "--rows-per-call",
        type=int,
        nargs="+",
        default=[],
        help="Also measure candidates that split each expert's rows into blocks "
        "of at most this many rows, one `mx.quantized_matmul` per block. "
        "Diagnostic: `--single-row` established that the per-call row count is a "
        "variable in its own right, separate from the total flattened row count, "
        "so sweeping it is what locates the per-call dispatch threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Router top-k. Defaults to the block's own `top_k`.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repetitions per path, after one warm-up. The best time is "
        "reported, since the host is not otherwise quiesced.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for both the synthetic routing indices and the activations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write `expert-matmul-spike.json` into. Anything not "
        "written here vanishes when the run's temporary directory is removed.",
    )
    return parser.parse_args()


def find_switch_glu(model: nn.Module, block_idx: int) -> tuple[str, SwitchGLU]:
    """Locate the `SwitchGLU` belonging to transformer block `block_idx`.

    Parameters
    ----------
    model : nn.Module
        A loaded `mlx_lm` model.
    block_idx : int
        Transformer block index to select.

    Returns
    -------
    tuple[str, SwitchGLU]
        The module's dotted path and the module itself.

    Raises
    ------
    RuntimeError
        If the model has no `SwitchGLU`, if a match's path carries no block
        index, or if `block_idx` names a block that has none.
    """
    found: dict[int, tuple[str, SwitchGLU]] = {}

    for path, module in model.named_modules():
        if not isinstance(module, SwitchGLU):
            continue
        match = _LAYER_IDX_RE.search(path)
        if match is None:
            raise RuntimeError(
                f"Could not derive a transformer block index from {path!r}."
            )
        found[int(match.group(1))] = (path, module)

    if not found:
        raise RuntimeError("Model contains no `SwitchGLU` module; is this an MoE?")
    if block_idx not in found:
        raise RuntimeError(
            f"Block {block_idx} has no `SwitchGLU`. MoE blocks: {sorted(found)!r}"
        )

    return found[block_idx]


def describe_projection(projection: QuantizedSwitchLinear) -> dict[str, Any]:
    """Read the quantization parameters and tensor shapes off the module itself.

    Nothing here is hardcoded: a store converted from a different checkpoint
    would carry different `bits`/`group_size`, and a spike that assumed the
    current ones would silently measure the wrong kernel configuration.
    """
    biases = projection.get("biases")

    return {
        "num_routed_experts": int(projection.num_routed_experts),  # type: ignore
        "input_dims": int(projection.input_dims),
        "output_dims": int(projection.output_dims),
        "group_size": int(projection.group_size),
        "bits": int(projection.bits),
        "mode": str(projection.mode),
        "weight_shape": list(projection["weight"].shape),
        "weight_dtype": str(projection["weight"].dtype),
        "scales_shape": list(projection["scales"].shape),
        "scales_dtype": str(projection["scales"].dtype),
        "biases_shape": None if biases is None else list(biases.shape),
        "biases_dtype": None if biases is None else str(biases.dtype),
    }


def synthetic_indices(
    *,
    rng: np.random.Generator,
    n_tokens: int,
    top_k: int,
    num_routed_experts: int,
) -> mx.array:
    """Build router-shaped indices: `top_k` distinct experts per token.

    Distinctness within a token mirrors what `mx.argpartition` produces in the
    real router, so the candidate's grouping sees the row multiplicities it
    will see in production.
    """
    rows = np.stack(
        [
            rng.choice(num_routed_experts, size=top_k, replace=False)
            for _ in range(n_tokens)
        ]
    )
    return mx.array(rows.astype(np.uint32)).reshape(1, n_tokens, top_k)


def group_rows_by_expert(
    row_experts: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group flat row indices by the expert each row routes to.

    Ascending expert order, ascending row order within each expert. This is a
    spike-local stand-in for `preempt.engine.expert_batching.group_rows_by_expert`,
    which Task 2 introduces; duplicating ten lines here keeps the gate
    independent of code the gate is supposed to authorize.
    """
    groups: dict[int, list[int]] = {}
    for row_idx, expert_idx in enumerate(row_experts):
        groups.setdefault(int(expert_idx), []).append(row_idx)

    return tuple(
        (expert_idx, tuple(groups[expert_idx])) for expert_idx in sorted(groups)
    )


def reference_gather_qmm(
    projection: QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
) -> mx.array:
    """The shipping path: `mx.gather_qmm` in upstream's own framing.

    Reproduces `SwitchGLU.__call__` (mlx_lm/models/switch_layers.py:176-199)
    and `QuantizedSwitchLinear.__call__` (:75-90) verbatim, reduced to the one
    projection under test:
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py

    `_gather_sort` and `_scatter_unsort` are imported from upstream rather than
    retyped, and the `indices.size >= 64` sort threshold is upstream's own, so
    this measures the kernel path the engine actually takes rather than an
    idealized one.

    Returns
    -------
    mx.array
        Shape `indices.shape + (output_dims,)`.
    """
    xs = mx.expand_dims(x, (-2, -3))

    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        xs, idx, inv_order = _gather_sort(xs, indices)

    y = mx.gather_qmm(
        xs,
        projection["weight"],
        projection["scales"],
        projection.get("biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode=projection.mode,
        sorted_indices=do_sort,
    )

    if do_sort:
        y = _scatter_unsort(y, inv_order, indices.shape)

    return y.squeeze(-2)


def candidate_per_expert(
    projection: QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
    *,
    keep_row_axis: bool,
    max_rows_per_call: int | None = None,
) -> mx.array:
    """The proposed path: one `mx.quantized_matmul` per distinct expert.

    Parameters
    ----------
    projection : QuantizedSwitchLinear
        The projection whose per-expert weight slices are used.
    x : mx.array
        Hidden states, shape `(batch, tokens, input_dims)`.
    indices : mx.array
        Router selection, shape `(batch, tokens, top_k)`.
    keep_row_axis : bool
        When `True`, each expert's rows are fed as `(n_rows, 1, input_dims)`,
        matching the rank the reference hands `gather_qmm`. When `False`, they
        are fed as `(n_rows, input_dims)` -- the layout the streaming forward
        will use. Both are measured because MLX selects kernels on input rank.
    max_rows_per_call : int | None
        Split each expert's rows into blocks of at most this many rows, one
        `mx.quantized_matmul` per block. `None` means one call per expert.
        Diagnostic: MLX selects among quantized matmul kernels partly on the
        row count, so capping it isolates kernel choice from decomposition.

    Returns
    -------
    mx.array
        Shape `indices.shape + (output_dims,)`, in the same row order as the
        reference.
    """
    weight = projection["weight"]
    scales = projection["scales"]
    biases = projection.get("biases")

    top_k = indices.shape[-1]
    x_tokens = x.reshape(-1, x.shape[-1])
    row_experts = np.asarray(indices).reshape(-1)

    outputs: list[mx.array] = []
    permutation: list[np.ndarray] = []

    for expert_idx, rows in group_rows_by_expert(row_experts.tolist()):
        expert_rows = np.asarray(rows, dtype=np.int32)
        block = len(expert_rows) if max_rows_per_call is None else max_rows_per_call

        for start in range(0, len(expert_rows), block):
            row_array = expert_rows[start : start + block]

            # Row r of the flattened (token, slot) grid reads token r // top_k.
            x_rows = x_tokens[mx.array(row_array // top_k)]
            if keep_row_axis:
                x_rows = mx.expand_dims(x_rows, -2)

            y_rows = mx.quantized_matmul(
                x_rows,
                weight[expert_idx],
                scales[expert_idx],
                None if biases is None else biases[expert_idx],
                transpose=True,
                group_size=projection.group_size,
                bits=projection.bits,
                mode=projection.mode,
            )
            if keep_row_axis:
                y_rows = y_rows.squeeze(-2)

            outputs.append(y_rows)
            permutation.append(row_array)

    # Concatenated in group order, then un-permuted by inverting the row order
    # the grouping imposed -- the same idea as upstream's
    # `_gather_sort`/`_scatter_unsort` pair, and a gather rather than a scatter.
    grouped = mx.concatenate(outputs, axis=0)
    inverse = np.argsort(np.concatenate(permutation)).astype(np.int32)

    return grouped[mx.array(inverse)].reshape(*indices.shape, -1)


def dequantized_per_expert(
    projection: QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
) -> mx.array:
    """Third variant: `mx.dequantize` then a dense `mx.matmul` per expert.

    Timed only, for Task 9. Reassembly is identical to `candidate_per_expert`
    so the timing difference is the kernel and nothing else.
    """
    weight = projection["weight"]
    scales = projection["scales"]
    biases = projection.get("biases")

    top_k = indices.shape[-1]
    x_tokens = x.reshape(-1, x.shape[-1])
    row_experts = np.asarray(indices).reshape(-1)

    outputs: list[mx.array] = []
    permutation: list[np.ndarray] = []

    for expert_idx, rows in group_rows_by_expert(row_experts.tolist()):
        row_array = np.asarray(rows, dtype=np.int32)
        x_rows = x_tokens[mx.array(row_array // top_k)]

        dense = mx.dequantize(
            weight[expert_idx],
            scales[expert_idx],
            None if biases is None else biases[expert_idx],
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
        )

        outputs.append(mx.matmul(x_rows, dense.swapaxes(-1, -2)))
        permutation.append(row_array)

    grouped = mx.concatenate(outputs, axis=0)
    inverse = np.argsort(np.concatenate(permutation)).astype(np.int32)

    return grouped[mx.array(inverse)].reshape(*indices.shape, -1)


def time_path(
    build: Callable[[], mx.array],
    *,
    repeats: int,
) -> tuple[mx.array, float]:
    """Build and evaluate `build()` once as warm-up, then time `repeats` runs.

    The best time is reported rather than the mean: the host is not quiesced,
    and this is a relative comparison between two kernels, not a benchmark.

    Returns
    -------
    tuple[mx.array, float]
        The evaluated result of the last run and the best wall-clock seconds.
    """
    result = build()
    mx.eval(result)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = build()
        mx.eval(result)
        best = min(best, time.perf_counter() - start)

    return result, best


def compare(reference: mx.array, candidate: mx.array) -> dict[str, Any]:
    """Compare two outputs bitwise, and quantify the difference either way.

    A bare `True` from `mx.array_equal` is weak evidence; printing the max
    absolute and max relative difference alongside it shows the comparison ran
    on the tensors it claims to have run on.

    Raises
    ------
    RuntimeError
        If the two arrays disagree on shape or dtype, which would make the
        element-wise comparison meaningless.
    """
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Shape mismatch: reference {reference.shape} vs "
            f"candidate {candidate.shape}."
        )
    if reference.dtype != candidate.dtype:
        raise RuntimeError(
            f"Dtype mismatch: reference {reference.dtype} vs "
            f"candidate {candidate.dtype}."
        )

    equal = bool(mx.array_equal(reference, candidate))

    reference32 = reference.astype(mx.float32)
    candidate32 = candidate.astype(mx.float32)
    absolute = mx.abs(reference32 - candidate32)
    denominator = mx.maximum(
        mx.abs(reference32), mx.array(float(np.finfo(np.float32).tiny))
    )

    max_abs = float(mx.max(absolute))
    max_rel = float(mx.max(absolute / denominator))
    mismatches = int(mx.sum(absolute != 0))

    return {
        "bitwise_equal": equal,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatched_elements": mismatches,
        "total_elements": int(reference.size),
    }


def measure_shape(
    *,
    label: str,
    projection: QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
    repeats: int,
    include_dequantized: bool,
    include_single_row: bool,
    rows_per_call: Sequence[int] = (),
) -> dict[str, Any]:
    """Run every path at one input shape and report equality plus timings."""
    row_experts = np.asarray(indices).reshape(-1)
    _, counts = np.unique(row_experts, return_counts=True)
    distinct = int(counts.size)
    do_sort = bool(indices.size >= 64)

    print(f"\n--- {label} shape ---", flush=True)
    print(
        f"  x={tuple(x.shape)} {x.dtype}, indices={tuple(indices.shape)} "
        f"{indices.dtype}, rows={int(indices.size)}, distinct experts={distinct}, "
        f"rows/expert mean={counts.mean():.2f} max={int(counts.max())}, "
        f"upstream sort={do_sort}",
        flush=True,
    )

    reference, reference_s = time_path(
        lambda: reference_gather_qmm(projection, x, indices), repeats=repeats
    )
    print(
        f"  reference  gather_qmm           : {reference_s * 1e3:8.3f} ms", flush=True
    )

    results: dict[str, Any] = {
        "label": label,
        "x_shape": list(x.shape),
        "indices_shape": list(indices.shape),
        "rows": int(indices.size),
        "distinct_experts": distinct,
        "rows_per_expert_mean": float(counts.mean()),
        "rows_per_expert_max": int(counts.max()),
        "upstream_sort": do_sort,
        "output_shape": list(reference.shape),
        "reference_seconds": reference_s,
        "candidates": {},
    }

    variants: list[tuple[str, bool, int | None]] = [
        ("rows_2d", False, None),
        ("rows_3d", True, None),
    ]
    if include_single_row:
        variants.append(("rows_1", False, 1))
    variants.extend((f"rows_cap{cap}", False, cap) for cap in rows_per_call)

    for name, keep_row_axis, max_rows_per_call in variants:
        candidate, candidate_s = time_path(
            lambda: candidate_per_expert(
                projection,
                x,
                indices,
                keep_row_axis=keep_row_axis,
                max_rows_per_call=max_rows_per_call,
            ),
            repeats=repeats,
        )
        comparison = compare(reference, candidate)
        comparison["seconds"] = candidate_s
        comparison["speedup_vs_reference"] = reference_s / candidate_s
        results["candidates"][name] = comparison

        print(
            f"  candidate  quantized_matmul {name}: {candidate_s * 1e3:9.3f} ms  "
            f"array_equal={comparison['bitwise_equal']}  "
            f"max_abs={comparison['max_abs_diff']:.3e}  "
            f"max_rel={comparison['max_rel_diff']:.3e}  "
            f"mismatched={comparison['mismatched_elements']}"
            f"/{comparison['total_elements']}",
            flush=True,
        )

    if include_dequantized:
        dequantized, dequantized_s = time_path(
            lambda: dequantized_per_expert(projection, x, indices), repeats=repeats
        )
        comparison = compare(reference, dequantized)
        comparison["seconds"] = dequantized_s
        comparison["speedup_vs_reference"] = reference_s / dequantized_s
        results["dequantized"] = comparison

        print(
            f"  variant    dequantize+matmul    : {dequantized_s * 1e3:8.3f} ms  "
            f"array_equal={comparison['bitwise_equal']}  "
            f"max_abs={comparison['max_abs_diff']:.3e}  "
            f"max_rel={comparison['max_rel_diff']:.3e}",
            flush=True,
        )

    return results


def report_gate(shapes: Sequence[dict[str, Any]]) -> bool:
    """Print the gate verdict and return whether the shipping layout passed.

    The shipping layout is `rows_2d`: `(n_rows, input_dims)` per expert. The
    `rows_3d` result is diagnostic -- if the two disagree, the difference is
    MLX's kernel selection by input rank, not the expert-major decomposition.
    """
    print("\n=== GATE ===", flush=True)

    passed = True
    for shape in shapes:
        for name, comparison in shape["candidates"].items():
            status = "PASS" if comparison["bitwise_equal"] else "FAIL"
            print(
                f"  {status}  {shape['label']:>12} ({shape['rows']:>5} rows) / {name}: "
                f"array_equal={comparison['bitwise_equal']}, "
                f"max_abs={comparison['max_abs_diff']:.6e}, "
                f"max_rel={comparison['max_rel_diff']:.6e}",
                flush=True,
            )
            if name == "rows_2d" and not comparison["bitwise_equal"]:
                passed = False

    return passed


def run(args: argparse.Namespace) -> dict[str, Any]:
    print(f"Loading model: {args.model}", flush=True)
    loaded = load_mlx_model(args.model)

    path, switch_glu = find_switch_glu(loaded.model, args.block_idx)
    projection = getattr(switch_glu, args.projection)

    if not isinstance(projection, QuantizedSwitchLinear):
        raise RuntimeError(
            f"{path}.{args.projection} is a {type(projection).__name__}, not a "
            "QuantizedSwitchLinear; this spike measures the quantized kernel."
        )
    if "bias" in projection:
        raise RuntimeError(
            f"{path}.{args.projection} carries a per-expert `bias`, which neither "
            "path here applies. Add the bias term to both before trusting the gate."
        )

    description = describe_projection(projection)
    print(f"Projection: {path}.{args.projection}", flush=True)
    for key, value in description.items():
        print(f"  {key}: {value}", flush=True)

    top_k = args.top_k
    if top_k is None:
        moe_block = dict(loaded.model.named_modules())[path.rsplit(".", 1)[0]]
        top_k = int(moe_block.top_k)
    print(f"  top_k: {top_k}", flush=True)

    num_routed_experts = description["num_routed_experts"]
    input_dims = description["input_dims"]

    if top_k > num_routed_experts:
        raise ValueError(
            f"top_k {top_k} exceeds num_routed_experts {num_routed_experts}."
        )
    for n_tokens in args.prefill_tokens:
        if n_tokens * top_k < 64:
            raise ValueError(
                f"--prefill-tokens {n_tokens} x top_k {top_k} is "
                f"{n_tokens * top_k} rows, below upstream's 64-row sort "
                "threshold; that prefill shape would not exercise the sorted path."
            )

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    activation_dtype = projection["scales"].dtype
    shapes: list[dict[str, Any]] = []

    plan: list[tuple[str, int]] = [("decode", 1)]
    plan.extend((f"prefill-{n}", n) for n in args.prefill_tokens)

    for label, n_tokens in plan:
        x = mx.random.normal(shape=(1, n_tokens, input_dims)).astype(activation_dtype)
        indices = synthetic_indices(
            rng=rng,
            n_tokens=n_tokens,
            top_k=top_k,
            num_routed_experts=num_routed_experts,
        )
        mx.eval(x, indices)

        shapes.append(
            measure_shape(
                label=label,
                projection=projection,
                x=x,
                indices=indices,
                repeats=args.repeats,
                include_dequantized=n_tokens > 1,
                include_single_row=args.single_row,
                rows_per_call=args.rows_per_call,
            )
        )

    passed = report_gate(shapes)
    print(f"\nPeak memory: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)

    return {
        "model": args.model,
        "layer_path": path,
        "projection": args.projection,
        "top_k": top_k,
        "seed": args.seed,
        "repeats": args.repeats,
        "projection_description": description,
        "shapes": shapes,
        "gate_passed": passed,
    }


def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if any(n < 1 for n in args.prefill_tokens):
        raise ValueError("--prefill-tokens values must be at least 1.")

    summary = run(args)

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        summary_path = args.output / "expert-matmul-spike.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {summary_path}", flush=True)

    if not summary["gate_passed"]:
        raise GateFailure(
            "GATE FAILED: per-expert `mx.quantized_matmul` is not bitwise "
            "identical to `mx.gather_qmm`. Do not proceed with the expert-major "
            "forward, and do not loosen this comparison to a tolerance -- this "
            "is a kernel-level precision change and needs the user's sign-off."
        )

    print(
        "\nGATE PASSED: per-expert `mx.quantized_matmul` reproduces "
        "`mx.gather_qmm` bitwise at both shapes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
