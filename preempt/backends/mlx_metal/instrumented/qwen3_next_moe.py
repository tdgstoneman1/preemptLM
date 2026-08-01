"""`mlx_lm.models.Qwen3NextSparseMoeBlock` wrapper that records internal expert routing during
forward pass. `Qwen3NextSparseMoeBlock` is used in MLX implementations of Qwen3-next and Qwen3.6
"""

from typing import TypeVar
from collections.abc import Callable

import mlx.core as mx

import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from ..recorder import MlxExpertRoutingRecorder
from ..types import MlxWrapperFactory

from preempt.engine.layer_resolution import LayerCandidate


class InstrumentedQwen3NextMoE(nn.Module):
    inner: Qwen3NextSparseMoeBlock
    recorder: MlxExpertRoutingRecorder
    capture_gate_logits: bool
    layer_path: str
    layer_idx: int

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        recorder: MlxExpertRoutingRecorder,
        capture_gate_logits: bool,
        layer_path: str,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.inner = inner
        self.recorder = recorder
        self.capture_gate_logits = capture_gate_logits
        self.layer_path = layer_path
        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        """See https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py#L308
        for original `Qwen3NextSparseMoeBlock.__call__(...)`.
        """
        if self.inner.sharding_group is not None:
            x = sum_gradients(self.inner.sharding_group)(x)

        # Record logits before softmax
        logits = self.inner.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)

        k = self.inner.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.inner.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Record state
        self.recorder.capture(
            layer_path=self.layer_path,
            layer_class=self.inner.__class__.__name__,
            layer_idx=self.layer_idx,
            expert_ids=inds,
            expert_weights=scores,
            gate_logits=logits if self.capture_gate_logits else None,
        )

        y = self.inner.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.inner.shared_expert(x)
        shared_y = mx.sigmoid(self.inner.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)

        return y


# TODO Get rid of this, just slop from Perplexity session
def make_qwen3next_moe_wrapper_factory(
    recorder: MlxExpertRoutingRecorder,
    *,
    capture_gate_logits: bool = False,
) -> MlxWrapperFactory:

    def factory(
        module: Qwen3NextSparseMoeBlock,
        candidate: LayerCandidate,
    ) -> nn.Module:
        if candidate.layer_idx is None:
            raise ValueError()  # TODO descriptive error msg

        return InstrumentedQwen3NextMoE(
            inner=module,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            layer_path=candidate.layer_path,
            layer_idx=candidate.layer_idx,
        )

    return factory
