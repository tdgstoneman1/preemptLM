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

from preempt.core.identity import ExpertKey
from preempt.core.protocols.provider import ExpertProvider

from preempt.engine.layer_resolution import LayerCandidate


class InstrumentedQwen3NextMoE(nn.Module):
    inner: Qwen3NextSparseMoeBlock
    recorder: MlxExpertRoutingRecorder | None
    capture_gate_logits: bool
    layer_path: str
    layer_idx: int
    provider: ExpertProvider | None
    model_fingerprint: str | None

    def __init__(
        self,
        inner: Qwen3NextSparseMoeBlock,
        recorder: MlxExpertRoutingRecorder | None,
        capture_gate_logits: bool,
        layer_path: str,
        layer_idx: int,
        provider: ExpertProvider | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        super().__init__()

        if provider is not None and model_fingerprint is None:
            raise ValueError(
                "`model_fingerprint` is required when a `provider` is given."
            )

        self.inner = inner
        self.recorder = recorder
        self.capture_gate_logits = capture_gate_logits
        self.layer_path = layer_path
        self.layer_idx = layer_idx
        self.provider = provider
        self.model_fingerprint = model_fingerprint

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

        # Record state (lazily -- no eval in the capture path)
        if self.recorder is not None:
            self.recorder.capture(
                layer_path=self.layer_path,
                layer_class=self.inner.__class__.__name__,
                layer_idx=self.layer_idx,
                expert_ids=inds,
                expert_weights=scores,
                gate_logits=logits if self.capture_gate_logits else None,
            )

        # Demand sync point: only a streaming run wires a provider, and only
        # then do we pay the eval that materializing the routed ids forces.
        if self.provider is not None:
            assert self.model_fingerprint is not None
            unique_ids = sorted({int(e) for e in inds.flatten().tolist()})
            self.provider.acquire(
                tuple(
                    ExpertKey(
                        model_fingerprint=self.model_fingerprint,
                        layer_idx=self.layer_idx,
                        expert_idx=expert_idx,
                    )
                    for expert_idx in unique_ids
                )
            )

        y = self.inner.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.inner.shared_expert(x)
        shared_y = mx.sigmoid(self.inner.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.inner.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.inner.sharding_group)

        return y


def make_qwen3next_moe_wrapper_factory(
    recorder: MlxExpertRoutingRecorder | None,
    *,
    capture_gate_logits: bool = False,
    provider: ExpertProvider | None = None,
    model_fingerprint: str | None = None,
) -> MlxWrapperFactory:
    """Returns a factory that swaps `Qwen3NextSparseMoeBlock` for an instrumented
    wrapper layer.

    Parameters
    ----------
    recorder : MlxExpertRoutingRecorder | None
        Trace recorder; `None` disables routing capture.
    capture_gate_logits : bool
        Whether to buffer the full `num_experts`-wide gate distribution.
    provider : ExpertProvider | None
        Residency hook called after top-k selection; `None` disables it.
    model_fingerprint : str | None
        Required when `provider` is given — it qualifies each `ExpertKey`.

    Returns
    -------
    MlxWrapperFactory
        Callable accepting the upstream module and its `LayerCandidate`.
    """

    def factory(
        module: Qwen3NextSparseMoeBlock,
        candidate: LayerCandidate,
    ) -> nn.Module:
        if candidate.layer_idx is None:
            raise ValueError(
                f"Cannot instrument {candidate.layer_path!r}: no transformer block index."
            )

        return InstrumentedQwen3NextMoE(
            inner=module,
            recorder=recorder,
            capture_gate_logits=capture_gate_logits,
            layer_path=candidate.layer_path,
            layer_idx=candidate.layer_idx,
            provider=provider,
            model_fingerprint=model_fingerprint,
        )

    return factory
