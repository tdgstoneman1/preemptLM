from __future__ import annotations

from typing import Optional, Literal
from collections.abc import Callable, Mapping

import mlx.core as mx

import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from functools import partial

from ..expert_kernel import (
    ExpertProjections,
    QuantizedProjection,
    UnquantizedProjection,
    SwitchQuantParams,
    sequential_run_selected_experts,
    swiglu_forward_fn,
    fused_run_selected_experts,
    describe_switch_quantization,
)
from ..recorder import MoERecorder
from ..cache import MlxExpertCache
from ..types import MlxWrapperFactory
from ..constants import SWIGLU_PROJECTION_NAMES

from preempt.core.identity import ExpertKey
from preempt.core.protocols import IExpertLoader

from preempt.engine.layer_resolution import LayerCandidate

# TODO make module wrapper hold experts in memory, external cache manager handles eviction decisions


@mx.compile
def _compute_topk_routing(
    logits: mx.array,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)

    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)

    return inds, scores


@mx.compile
def _combine_and_apply_experts(
    y: mx.array,
    scores: mx.array,
    shared_y: mx.array,
    shared_gate: mx.array,
) -> mx.array:
    sum_routed_experts = (y * scores[..., None]).sum(axis=-2)
    return sum_routed_experts + (mx.sigmoid(shared_gate) * shared_y)


class InstrumentedQwen3_xMoE(nn.Module):
    """Module wrapper for instrumenting Qwen3.x and Qwen3-Next MoE blocks.

    Forward pass currently computes one expert at a time when reading from disk.

    `__call__` forked from `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock` (see
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py#L308).

    :Note: The `mlx_lm` implementations of Qwen3.5 (for Qwen3.x) and Qwen3-Next both
    use the same `Qwen3NextSparseMoeBlock` for MoE blocks, and this wrapper can
    thus be applied to either of the two architectures.
    """

    inner: Qwen3NextSparseMoeBlock
    recorder: MoERecorder | None
    capture_gate_logits: bool
    layer_path: str
    block_idx: int
    expert_loader: IExpertLoader | None
    expert_cache: MlxExpertCache | None
    model_fingerprint: str | None

    quantization: SwitchQuantParams | None
    _is_quantized: bool
    _apply_experts_fn: Callable[[mx.array, mx.array], mx.array]

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        recorder: Optional[MoERecorder],
        capture_gate_logits: bool,
        layer_path: str,
        block_idx: int,
        expert_loader: Optional[IExpertLoader] = None,
        expert_cache: Optional[MlxExpertCache] = None,
        model_fingerprint: Optional[str] = None,
        expert_kernel: Literal["sequential", "fused"] = "sequential",
    ) -> None:
        super().__init__()

        if expert_loader is not None and (
            model_fingerprint is None or expert_cache is None
        ):
            raise ValueError(
                f"{type(expert_loader).__name__=}, "
                f"{type(model_fingerprint).__name__=}, "
                f"{type(expert_cache).__name__=}"
            )
        if expert_loader is None and expert_cache is not None:
            raise ValueError()

        self.inner = inner

        self.recorder = recorder
        self.capture_gate_logits = capture_gate_logits

        self.layer_path = layer_path
        self.block_idx = block_idx

        self.expert_loader = expert_loader
        self.expert_cache = expert_cache
        self.model_fingerprint = model_fingerprint

        self.quantization = describe_switch_quantization(
            inner.switch_mlp, SWIGLU_PROJECTION_NAMES
        )
        self._is_quantized = self.quantization is not None

        if expert_kernel == "sequential":
            self._apply_experts_fn = partial(
                sequential_run_selected_experts,
                expert_forward_fn=swiglu_forward_fn,
                load_expert_fn=self._read_expert_from_disk,
            )
        else:
            self._apply_experts_fn = partial(
                fused_run_selected_experts,
                load_expert_fn=self._read_expert_from_disk,
                is_quantized=self._is_quantized,
            )

    def _read_expert_from_disk(self, expert_idx: int) -> ExpertProjections:
        assert self.expert_loader is not None
        assert self.expert_cache is not None
        assert self.model_fingerprint is not None

        key = ExpertKey(
            model_fingerprint=self.model_fingerprint,
            block_idx=self.block_idx,
            expert_idx=expert_idx,
        )
        self.expert_loader.load((key,))
        tensors = self.expert_cache.tensors(key)
        projections = {
            name: self._projection_from_tensors(tensors, name)
            for name in SWIGLU_PROJECTION_NAMES
        }
        return ExpertProjections(projections=projections)

    def _projection_from_tensors(
        self,
        tensors: Mapping[str, mx.array],
        name: str,
    ) -> QuantizedProjection | UnquantizedProjection:
        if not self._is_quantized or self.quantization is None:
            return UnquantizedProjection(
                weight=tensors[f"{name}.weight"],
            )
        quant = self.quantization.params[name]
        return QuantizedProjection(
            weight=tensors[f"{name}.weight"],
            scales=tensors[f"{name}.scales"],
            biases=tensors.get(f"{name}.biases"),
            group_size=quant.group_size,
            bits=quant.bits,
            mode=quant.mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.inner.sharding_group is not None:
            x = sum_gradients(self.inner.sharding_group)(x)  # type: ignore

        logits = self.inner.gate(x)
        inds, scores = _compute_topk_routing(
            logits, self.inner.top_k, self.inner.norm_topk_prob
        )
        # * Lazy record state
        if self.recorder is not None:
            self.recorder.capture(
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                block_idx=self.block_idx,
                expert_ids=inds,
                expert_weights=scores,
                gate_logits=logits if self.capture_gate_logits else None,
            )

        if self.expert_loader is None:
            y = self.inner.switch_mlp(x, inds)
        else:
            y = self._apply_experts_fn(x, inds)

        # * Evaluate shared expert paths
        shared_y = self.inner.shared_expert(x)
        shared_gate = self.inner.shared_expert_gate(x)

        y = _combine_and_apply_experts(y, scores, shared_y, shared_gate)

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)
        return y


# TODO rename to meta_factory?
def make_qwen3_x_moe_wrapper_factory(
    recorder: MoERecorder | None,
    *,
    capture_gate_logits: bool = False,
    expert_loader: IExpertLoader | None = None,  # TODO rename to 'loader'
    expert_cache: MlxExpertCache | None = None,
    model_fingerprint: str | None = None,
) -> MlxWrapperFactory:

    def factory(
        module: Qwen3NextSparseMoeBlock,
        candidate: LayerCandidate,
    ) -> nn.Module:
        if candidate.block_idx is None:
            raise ValueError(
                f"Cannot instrument layer '{candidate.layer_path!r}' because it has no "
                "block index."
            )

        return InstrumentedQwen3_xMoE(
            inner=module,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            layer_path=candidate.layer_path,
            block_idx=candidate.block_idx,
            expert_loader=expert_loader,
            expert_cache=expert_cache,
            model_fingerprint=model_fingerprint,
        )

    return factory
