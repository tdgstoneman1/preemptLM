"""Instrumented wrapper for `mlx_lm.models.Qwen3NextSparseMoeBlock` (used in `mlx_lm`
implementations of Qwen3-next and Qwen3.6)
"""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx

import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from ..expert_kernel import (
    ExpertProjections,
    QuantizedProjection,
    SwitchQuantParams,
    sequential_run_selected_experts,
    describe_switch_quantization,
    project_rows,
)
from ..recorder import MlxExpertRoutingRecorder
from ..cache import MlxExpertCache
from ..types import MlxWrapperFactory
from ..constants import SWIGLU_PROJECTION_NAMES

from preempt.core.identity import ExpertKey
from preempt.core.protocols.loader import IExpertLoader

from preempt.engine.layer_resolution import LayerCandidate


class InstrumentedQwen3NextMoE(nn.Module):
    """Module wrapper for instrumenting Qwen3-Next MoE block.

    `__call__` forked from `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock` (see
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py#L308)

    Forward pass currently computes one expert at a time when reading from disk.
    """

    inner: Qwen3NextSparseMoeBlock
    recorder: MlxExpertRoutingRecorder | None
    capture_gate_logits: bool
    layer_path: str
    block_idx: int
    provider: IExpertLoader | None
    model_fingerprint: str | None
    cache: MlxExpertCache | None
    quantization: SwitchQuantParams
    activation: nn.Module

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        recorder: MlxExpertRoutingRecorder | None,
        capture_gate_logits: bool,
        layer_path: str,
        block_idx: int,
        provider: IExpertLoader | None = None,  # TODO rename
        model_fingerprint: str | None = None,
        cache: MlxExpertCache | None = None,  # TODO rename
    ) -> None:
        super().__init__()

        if provider is not None and (model_fingerprint is None or cache is None):
            raise ValueError(
                f"{type(provider).__name__=}, "
                f"{type(model_fingerprint).__name__=}, "
                f"{type(cache).__name__=}"
            )
        if provider is None and cache is not None:
            raise ValueError()

        self.inner = inner
        self.recorder = recorder
        self.capture_gate_logits = capture_gate_logits
        self.layer_path = layer_path
        self.block_idx = block_idx
        self.provider = provider
        self.model_fingerprint = model_fingerprint
        self.cache = cache

        self.quantization = describe_switch_quantization(
            inner.switch_mlp, SWIGLU_PROJECTION_NAMES
        )
        self.activation = inner.switch_mlp.activation

    # TODO rewrite docstring slop
    def _read_expert_from_disk(self, expert_idx: int) -> ExpertProjections:

        assert self.provider is not None
        assert self.cache is not None
        assert self.model_fingerprint is not None

        key = ExpertKey(
            model_fingerprint=self.model_fingerprint,
            block_idx=self.block_idx,
            expert_idx=expert_idx,
        )
        self.provider.load((key,))
        tensors = self.cache.tensors(key)

        projections = {
            name: self._projection_from_tensors(tensors, name)
            for name in SWIGLU_PROJECTION_NAMES
        }
        return ExpertProjections(projections=projections)

    def _projection_from_tensors(
        self,
        tensors: Mapping[str, mx.array],
        name: str,
    ) -> QuantizedProjection:
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
            x = sum_gradients(self.inner.sharding_group)(x)

        # TODO move do separate method, copied from original __call__
        logits = self.inner.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)

        k = self.inner.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.inner.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Lazy record state
        if self.recorder is not None:
            self.recorder.capture(
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                block_idx=self.block_idx,
                expert_ids=inds,
                expert_weights=scores,
                gate_logits=logits if self.capture_gate_logits else None,
            )

        if self.provider is None:
            y = self.inner.switch_mlp(x, inds)
        else:
            activation = self.activation

            def _swiglu_forward(
                x_rows: mx.array,
                projections: Mapping[str, QuantizedProjection],
            ) -> mx.array:
                x_up = project_rows(x_rows, projections["up_proj"])
                x_gate = project_rows(x_rows, projections["gate_proj"])

                return project_rows(activation(x_up, x_gate), projections["down_proj"])

            y = sequential_run_selected_experts(
                x,
                [int(expert) for expert in inds.flatten().tolist()],
                top_k=k,
                expert_forward_fn=_swiglu_forward,
                expert_load_weights_fn=self._read_expert_from_disk,
            )

        # TODO move do separate method, copied from original __call__
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.inner.shared_expert(x)
        shared_y = mx.sigmoid(self.inner.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)

        return y


# TODO rename as meta_factory?
def make_qwen3next_moe_wrapper_factory(
    recorder: MlxExpertRoutingRecorder | None,
    *,
    capture_gate_logits: bool = False,
    provider: IExpertLoader | None = None,  # TODO rename to 'loader'
    model_fingerprint: str | None = None,
    cache: MlxExpertCache | None = None,
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

        return InstrumentedQwen3NextMoE(
            inner=module,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            layer_path=candidate.layer_path,
            block_idx=candidate.block_idx,
            provider=provider,
            model_fingerprint=model_fingerprint,
            cache=cache,
        )

    return factory
