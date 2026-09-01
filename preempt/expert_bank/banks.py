from __future__ import annotations

from typing import Self, TypedDict
from types import TracebackType

from abc import ABC, abstractmethod

from pathlib import Path

import asyncio

import fcntl
import os
import sys
import mmap

from preempt.core.protocols import ExpertPayload
from preempt.core.enums import ReadPriority
from preempt.core.exceptions import ExpertBankCompatibilityError
from preempt.core.constants import EXPERTS_FILENAME, F_NOCACHE

from preempt.datamodel.identity import ExpertKey, TensorSpec

from .manifest import (
    ExpertBlobRecord,
    ExpertBankManifest,
)


class _CompatibilitySpec(TypedDict):
    model_id: str
    num_routed_experts: int
    top_k: int
    moe_block_idxs: tuple[int, ...]


class BaseExpertBank(ABC):
    """*Abstract; do not instantiate.*"""

    _manifest: ExpertBankManifest
    _index: dict[ExpertKey, ExpertBlobRecord]
    _tensor_specs: tuple[TensorSpec, ...]
    _fd: int | None

    def __init__(self, expert_bank_path: Path):
        self._manifest = ExpertBankManifest.load(expert_bank_path)
        self._index = self._manifest.blob_index()
        self._tensor_specs = self._manifest.tensor_specs

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def manifest(self) -> ExpertBankManifest:
        return self._manifest

    @property
    def model_fingerprint(self) -> str:
        return self._manifest.model_fingerprint

    def key_for(
        self, block_idx: int, expert_idx: int, variant: str = "all"
    ) -> ExpertKey:
        return ExpertKey(
            model_fingerprint=self.model_fingerprint,
            block_idx=block_idx,
            expert_idx=expert_idx,
            variant=variant,
        )

    def check_model_compatibility(
        self,
        model_id: str,
        num_routed_experts: int,
        top_k: int,
        moe_block_idxs: tuple[int, ...],
    ) -> None:
        observed = _CompatibilitySpec(
            model_id=model_id,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            moe_block_idxs=moe_block_idxs,
        )
        expected = _CompatibilitySpec(
            model_id=self._manifest.model_id,
            num_routed_experts=self._manifest.model_moe_spec.num_routed_experts,
            top_k=self._manifest.model_moe_spec.top_k,
            moe_block_idxs=self._manifest.model_moe_spec.moe_block_idxs,
        )
        if mismatches := {
            name: (expected[name], observed[name])
            for name in expected
            if expected[name] != observed[name]
        }:
            raise ExpertBankCompatibilityError(repr(mismatches))

    @abstractmethod
    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload: ...

    @abstractmethod
    def close(self) -> None: ...


class PreadExpertBank(BaseExpertBank):
    """Uses positioned reads (via `os.pread`) to load expert weights from
    `experts.bin`.

    On MacOS, this can be substantially faster than memory mapping.
    """

    def __init__(self, expert_bank_path: Path, bypass_page_cache: bool = True) -> None:
        super().__init__(expert_bank_path)

        fd = os.open(expert_bank_path / EXPERTS_FILENAME, os.O_RDONLY)
        if bypass_page_cache and sys.platform == "darwin":
            fcntl.fcntl(fd, F_NOCACHE, 1)

        self._fd = fd

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        if self._fd is None:
            raise RuntimeError("Cannot read closed expert bank.")

        blob = self._index[key]  # intentional KeyError
        data = await asyncio.to_thread(os.pread, self._fd, blob.length, blob.offset)

        if len(data) != blob.length:
            raise IOError()  # TODO error msg

        return ExpertPayload(
            key=key,
            data=data,
            encoding=self._manifest.payload_encoding,
            tensor_specs=self._tensor_specs,
        )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class MmapExpertBank(BaseExpertBank):
    """Expert bank using memory mapping to read expert weights from
    `experts.bin`.
    """

    _mmap: mmap.mmap | None

    def __init__(self, expert_bank_path: Path, **kwargs) -> None:
        super().__init__(expert_bank_path)

        self._mmap = None
        self._fd = None

        success = False
        fd = os.open(expert_bank_path / EXPERTS_FILENAME, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            if file_size > 0:
                self._mmap = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)

            self._fd = fd
            success = True

        finally:
            if not success:
                os.close(fd)

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        if self._mmap is None:
            raise RuntimeError()  # TODO add message

        blob = self._index[key]  # intentional KeyError
        slice_ = self._mmap[blob.offset : blob.offset + blob.length]

        if len(slice_) != blob.length:
            raise IOError()  # TODO add message

        return ExpertPayload(
            key=key,
            data=slice_,
            encoding=self._manifest.payload_encoding,
            tensor_specs=self._tensor_specs,
        )

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
