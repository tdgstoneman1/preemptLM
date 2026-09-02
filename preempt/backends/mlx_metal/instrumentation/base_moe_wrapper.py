from typing import Optional

from abc import ABC, abstractmethod

import mlx.nn as nn

from ..types import ExpertLayerQuants, ModuleWrapperFactory
from ..expert_cache import MlxExpertCache
from ..recorder import MoERecorder

from preempt.core.protocols import IExpertLoader


class BaseMoEWrapper(ABC, nn.Module):
    inner: nn.Module

    recorder: MoERecorder | None
    capture_gate_logits: bool

    layer_path: str
    block_idx: int
    model_fingerprint: str | None

    expert_loader: IExpertLoader | None
    expert_cache: MlxExpertCache | None

    _quants: ExpertLayerQuants | None

    def __init__(
        self,
        *,
        inner: nn.Module,
        recorder: Optional[MoERecorder],
        capture_gate_logits: bool,
        layer_path: str,
        block_idx: int,
        expert_loader: Optional[IExpertLoader],
        expert_cache: Optional[MlxExpertCache],
        model_fingerprint: Optional[str],
    ):
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

    @property
    def is_quantized(self) -> bool:
        return self._quants is not None

    @property
    def is_traced(self) -> bool:
        return self.recorder is not None

    @property
    def should_stream_experts(self) -> bool:
        return self.expert_loader is not None

    @staticmethod
    @abstractmethod
    def make_factory(**kwargs) -> ModuleWrapperFactory: ...
