from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

import numpy as np

from preempt.core.enums import CacheSlotState

from preempt.engine.expert_io.loader import DiskBackedExpertLoader
from preempt.engine.layer_resolution import LayerCandidate

from ..constants import GLU_PROJECTION_NAMES
from ..expert_cache import decode_serialized_expert, MlxExpertCache, SlottedGLU
from ..recorder import MlxTraceRecorder
from ..ops.experts import get_expert_quants, swiglu_activation
from ..types import ModuleWrapperFactory
from ..utils import mlx_to_numpy

from .base_moe_wrapper import BaseMoEWrapper


@mx.compile
def _get_routings(
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
def _linear_proj_matmul(
    x: mx.array,
    idxs: mx.array,
    proj_weights: mx.array,
) -> mx.array:
    """Expects 2d `x` and `idxs`"""

    x = mx.expand_dims(x, (-2, -3))
    idx = idxs
    inv_order = None

    if do_sort := idx.size >= 64:
        x, idx, inv_order = _gather_sort(x, idx)

    # Mirrors mlx-lm `SwitchLinear.__call__`
    y = mx.gather_mm(  # type: ignore
        x,
        proj_weights.swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    if do_sort:
        y = _scatter_unsort(y, inv_order, idxs.shape)

    return y.squeeze(-2)  # shape = (B * S, top-k, d_out)


@mx.compile
def _combine_shared(
    y: mx.array,
    scores: mx.array,
    shared_y: mx.array,
    shared_gate: mx.array,
) -> mx.array:
    """Copied from upstream __call__"""
    sum_routed_experts = (y * scores[..., None]).sum(axis=-2)
    return sum_routed_experts + (mx.sigmoid(shared_gate) * shared_y)


# TODO support for quantization
# TODO set MLX stream device
class Qwen3_xMoEWrapper(BaseMoEWrapper):
    """Instrumentation wrapper module for Qwen3.x and Qwen3-Next MoE blocks.

    :Note: `__call__` forked from `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock`, see
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py#L308
    for more details.
    """

    inner: Qwen3NextSparseMoeBlock
    _weight_slots: SlottedGLU

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        model_fingerprint: str | None,
        layer_path: str,
        block_idx: int,
        recorder: MlxTraceRecorder | None,
        capture_gate_logits: bool,
        expert_loader: DiskBackedExpertLoader | None,
        expert_cache: MlxExpertCache | None,
        num_slots: int,
        slot_size: int,
        # stream: mx.Stream | mx.Device,
    ) -> None:
        super().__init__(
            inner=inner,
            model_fingerprint=model_fingerprint,
            layer_path=layer_path,
            block_idx=block_idx,
            num_experts=inner.num_experts,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            expert_loader=expert_loader,
            expert_cache=expert_cache,
            num_slots=num_slots,
            slot_size=slot_size,
        )
        self._quants = get_expert_quants(inner.switch_mlp, GLU_PROJECTION_NAMES)
        self._weight_slots = SlottedGLU(
            self.num_slots,
            self.inner.shared_expert.up_proj.weight.shape,
            self.inner.shared_expert.down_proj.weight.shape,
            self.inner.shared_expert.down_proj.weight.dtype,
        )
        mx.eval(  # TODO pass thread local stream
            self._weight_slots.gate_proj,
            self._weight_slots.down_proj,
            self._weight_slots.up_proj,
        )

    def _maybe_admit(self, routed_eids: mx.array) -> tuple[np.ndarray, mx.array]:
        assert self.expert_cache is not None and self.expert_loader is not None
        assert (
            self._weight_slots
            and self._eid_lookup_table.size > 0
            and self._sid_lookup_table.size > 0
        )

        # Force eager evaluation to materialize selected experts
        routed_eids_np = mlx_to_numpy(routed_eids, copy=True).reshape(-1)
        unique, inverse_map = np.unique(routed_eids_np, return_inverse=True)

        if unique.size > self.num_slots:
            raise RuntimeError(
                f"MoE block {self.block_idx}: total slots available ({self.num_slots}) "
                f" < unique experts routed across the batch ({unique.size}). Either "
                f"decrease batch size or initialize {self.__class__.__name__!r} with "
                "more slots."
            )

        missed_eids = unique[self._eid_lookup_table[unique] < 0]
        if missed_eids.size == 0:
            return unique, mx.asarray(inverse_map)

        glob_eids = self.block_idx * self.inner.num_experts + missed_eids

        # TODO support for global manager
        # evicted_eids = np.asarray(
        #     self.expert_cache.manager.admit(
        #         glob_eids.tolist(),
        #         self.slot_size,
        #         ReadPriority.DEMAND,
        #     ),
        #     dtype=np.int32,
        # )
        # evicted_eids = evicted_eids[
        #     evicted_eids // self.inner.num_experts == self.block_idx
        # ]
        # acquired_slots = np.take(self._eid_lookup_table, missed_eids, axis=0)
        acquired_slots = self._choose_slots(missed_eids.size, protected=unique)

        old_values = self._sid_lookup_table[acquired_slots]
        mask = old_values >= 0
        evicted_eids = old_values[mask]

        self._eid_lookup_table[evicted_eids] = CacheSlotState.EMPTY
        self._sid_lookup_table[acquired_slots] = CacheSlotState.EMPTY

        blobs = self.expert_loader.load_sync(glob_eids.tolist())

        for eid, sid, payload in zip(missed_eids, acquired_slots, blobs):
            weights = decode_serialized_expert(payload)

            # TODO use proj name variables instead of strings
            self._weight_slots.up_proj[sid] = weights["up_proj.weight"]
            self._weight_slots.down_proj[sid] = weights["down_proj.weight"]
            self._weight_slots.gate_proj[sid] = weights["gate_proj.weight"]

            self._eid_lookup_table[eid] = sid
            self._sid_lookup_table[sid] = eid

            # Pin weights to prevent unintended gc
            # self._slot_refs[sid] = payload.data # TODO check if actually needed

        mx.eval(
            self._weight_slots.gate_proj,
            self._weight_slots.up_proj,
            self._weight_slots.down_proj,
        )
        # for sid in acquired_slots:
        #     del self._slot_refs[sid]  # TODO check if actually needed

        return unique, mx.asarray(inverse_map, dtype=mx.int32, copy=True)

    def _gather_forward(
        self,
        x,
        gate_proj: mx.array,
        up_proj: mx.array,
        down_proj: mx.array,
        slot_idxs: mx.array,
        inverse_map: mx.array,
    ) -> mx.array:
        """Mirrors mlx-lm `SwitchGLU.__call__`."""

        # slot_idxs acts as a proxy for expert indices
        B, S, d_model = x.shape

        # Slice cached weights arrays
        up_proj = mx.take(up_proj, slot_idxs, axis=0)
        gate_proj = mx.take(gate_proj, slot_idxs, axis=0)
        down_proj = mx.take(down_proj, slot_idxs, axis=0)

        # Flatten x and sort indices to 2d
        inverse_map = inverse_map.reshape(-1, self.inner.top_k)

        x_up = _linear_proj_matmul(
            x,
            inverse_map,
            up_proj,
        )
        x_gate = _linear_proj_matmul(
            x,
            inverse_map.reshape(B * S, self.inner.top_k),
            gate_proj,
        )
        x_swiglu = swiglu_activation(x_up, x_gate)
        x_down = _linear_proj_matmul(
            x_swiglu.reshape(B * S * self.inner.top_k, -1),
            inverse_map.reshape(-1, 1),
            down_proj,
        )
        return x_down.reshape(B, S, self.inner.top_k, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        if self.inner.sharding_group is not None:
            x = nn.layers.distributed.sum_gradients(self.inner.sharding_group)(x)  # type: ignore

        logits = self.inner.gate(x)
        inds, scores = _get_routings(
            logits, self.inner.top_k, self.inner.norm_topk_prob
        )
        shared_y = self.inner.shared_expert(x)
        shared_gate = self.inner.shared_expert_gate(x)

        if self.should_capture_traces:
            self.recorder.capture(  # type: ignore
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                block_idx=self.block_idx,
                expert_ids=inds,
                softmax_weights=scores,
                gate_logits=logits if self._capture_gate_logits else None,
            )

        # TODO add `self.should_stream` flag to make this cleaner
        if self.expert_cache is None:
            y = self.inner.switch_mlp(x, inds)

        else:
            unique_eids, inverse_map = self._maybe_admit(inds)
            slot_ids = self._eid_lookup_table[unique_eids]
            if (failed := np.argwhere(slot_ids < 0) < 0).any():
                raise RuntimeError(
                    f"MoE block {self.block_idx}: Failed to load experts {failed!r}"
                )
            y = self._gather_forward(
                x,
                self._weight_slots.gate_proj,
                self._weight_slots.up_proj,
                self._weight_slots.down_proj,
                mx.asarray(slot_ids, copy=True),
                inverse_map,
            )
            self._touch(slot_ids)  # lol

        y = _combine_shared(y, scores, shared_y, shared_gate)

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)

        return y

    @staticmethod
    def make_wrapper_factory(
        *,
        model_fingerprint: str | None = None,
        recorder: MlxTraceRecorder | None = None,
        capture_gate_logits: bool = False,
        expert_loader: DiskBackedExpertLoader | None = None,
        expert_cache: MlxExpertCache | None = None,
        num_slots: int = 0,
        slot_size: int = 0,
        # stream: mx.Stream | mx.Device,
    ) -> ModuleWrapperFactory:

        def factory(
            module: Qwen3NextSparseMoeBlock,  # TODO rename
            candidate: LayerCandidate,  # TODO rename
        ) -> nn.Module:
            if candidate.block_idx is None:
                raise ValueError(
                    f"Cannot instrument layer {candidate.layer_path!r} because "
                    "the index of its parent transformer block could not be "
                    "determined."
                )

            return Qwen3_xMoEWrapper(
                inner=module,
                model_fingerprint=model_fingerprint,
                layer_path=candidate.layer_path,
                block_idx=candidate.block_idx,
                recorder=recorder,
                capture_gate_logits=capture_gate_logits,
                expert_loader=expert_loader,
                expert_cache=expert_cache,
                num_slots=num_slots,
                slot_size=slot_size,
                # stream=stream,
            )

        return factory
