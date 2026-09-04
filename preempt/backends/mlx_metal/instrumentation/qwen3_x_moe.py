from __future__ import annotations

from typing import Optional, Literal
from collections.abc import Callable, Mapping

from attrs import asdict

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from preempt.datamodel.identity import ExpertKey
from preempt.core.protocols import IExpertLoader
from preempt.engine.layer_resolution import LayerCandidate

from ..types import (
    ModuleWrapperFactory,
    ExpertLayerWeights,
    WeightsTensor,
    QuantizedWeightsTensor,
)
from ..ops import (
    sequential_expert_matmul,
    swiglu_forward_fn,
    fused_expert_matmul,
)
from ..expert_cache import MlxExpertCache
from ..recorder import MlxTraceRecorder
from ..constants import SWITCHGLU_LINEAR_PROJ_NAMES
from ..utils import get_expert_quants

from .base_moe_wrapper import BaseMoEWrapper

# TODO load and hold experts in module wrapper rather than external cache,
# keep external cache manager responsible for eviction decisions


@mx.compile
def _compute_topk_routing(
    logits: mx.array,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Copied from upstream __call__"""
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
    """Copied from upstream __call__"""
    sum_routed_experts = (y * scores[..., None]).sum(axis=-2)
    return sum_routed_experts + (mx.sigmoid(shared_gate) * shared_y)


class Qwen3_xMoEWrapper(BaseMoEWrapper):
    """Instrumentation wrapper module for Qwen3.x and Qwen3-Next MoE blocks.

    :Note: `__call__` forked from `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock`, see
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py#L308
    for more details.
    """

    inner: Qwen3NextSparseMoeBlock
    _apply_experts_fn: Callable[[mx.array, mx.array], mx.array]

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        recorder: Optional[MlxTraceRecorder],
        capture_gate_logits: bool,
        layer_path: str,
        block_idx: int,
        expert_loader: Optional[IExpertLoader],
        expert_cache: Optional[MlxExpertCache],
        model_fingerprint: Optional[str],
        expert_matmul: Literal["sequential", "fused"],
    ) -> None:
        super().__init__(
            inner=inner,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            layer_path=layer_path,
            block_idx=block_idx,
            expert_loader=expert_loader,
            expert_cache=expert_cache,
            model_fingerprint=model_fingerprint,
        )
        self._quants = get_expert_quants(inner.switch_mlp, SWITCHGLU_LINEAR_PROJ_NAMES)

        if expert_matmul == "sequential":
            self._apply_experts_fn = partial(
                sequential_expert_matmul,
                expert_forward_fn=swiglu_forward_fn,
                load_expert_fn=self._get_expert_weights,
            )
        else:
            self._apply_experts_fn = partial(
                fused_expert_matmul,
                load_expert_fn=self._get_expert_weights,
                is_quantized=self.is_quantized,
            )

    def _get_expert_weights(self, expert_idx: int) -> ExpertLayerWeights:
        assert self.expert_loader is not None
        assert self.expert_cache is not None
        assert self.model_fingerprint is not None

        key = ExpertKey(
            model_fingerprint=self.model_fingerprint,
            block_idx=self.block_idx,
            expert_idx=expert_idx,
        )
        self.expert_loader.load((key,))
        tensors = self.expert_cache.get(key)

        return {
            name: self._projection_from_tensors(tensors, name)
            for name in SWITCHGLU_LINEAR_PROJ_NAMES
        }

    def _projection_from_tensors(
        self,
        tensors: Mapping[str, mx.array],
        name: str,
    ) -> WeightsTensor | QuantizedWeightsTensor:
        if self.is_quantized:
            return QuantizedWeightsTensor(
                weight=tensors[f"{name}.weight"],
                scales=tensors[f"{name}.scales"],
                biases=tensors.get(f"{name}.biases"),
                **asdict(self._quants[name]),  # type: ignore
            )
        return WeightsTensor(
            weight=tensors[f"{name}.weight"],
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.inner.sharding_group is not None:
            x = nn.layers.distributed.sum_gradients(self.inner.sharding_group)(x)  # type: ignore

        logits = self.inner.gate(x)
        inds, scores = _compute_topk_routing(
            logits, self.inner.top_k, self.inner.norm_topk_prob
        )
        # * Lazy record state
        if self.is_traced:
            self.recorder.capture(  # type: ignore
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                block_idx=self.block_idx,
                expert_ids=inds,
                softmax_weights=scores,
                gate_logits=logits if self.capture_gate_logits else None,
            )
        # * Apply selected experts
        y = (
            self._apply_experts_fn(x, inds)
            if self.expert_cache is not None
            else self.inner.switch_mlp(x, inds)
        )
        # * Apply shared expert and combine y
        shared_y = self.inner.shared_expert(x)
        shared_gate = self.inner.shared_expert_gate(x)
        y = _combine_and_apply_experts(y, scores, shared_y, shared_gate)

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)

        return y

    @staticmethod
    def make_wrapper_factory(
        recorder: Optional[MlxTraceRecorder],
        capture_gate_logits: bool = False,
        expert_loader: Optional[IExpertLoader] = None,
        expert_cache: Optional[MlxExpertCache] = None,
        model_fingerprint: Optional[str] = None,
        expert_matmul: Literal["sequential", "fused"] = "sequential",
    ) -> ModuleWrapperFactory:

        def factory(
            module: Qwen3NextSparseMoeBlock,
            candidate: LayerCandidate,
        ) -> nn.Module:
            if candidate.block_idx is None:
                raise ValueError(
                    f"Cannot instrument layer {candidate.layer_path!r} because "
                    "the index of its parent transformer block could not be "
                    "determined."
                )

            return Qwen3_xMoEWrapper(
                inner=module,
                recorder=recorder,
                capture_gate_logits=capture_gate_logits,
                layer_path=candidate.layer_path,
                block_idx=candidate.block_idx,
                expert_loader=expert_loader,
                expert_cache=expert_cache,
                model_fingerprint=model_fingerprint,
                expert_matmul=expert_matmul,
            )

        return factory
