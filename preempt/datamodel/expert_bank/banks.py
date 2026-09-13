from __future__ import annotations

from typing import Self, TypedDict
from types import TracebackType

from abc import ABC, abstractmethod
import asyncio
import fcntl
import mmap
import os
from pathlib import Path
import sys

import attrs

from preempt.core.constants import EXPERTS_FILENAME, F_NOCACHE
from preempt.core.exceptions import ExpertBankCompatibilityError
from preempt.core.enums import ReadPriority

from preempt.datamodel.identity import ExpertKey, TensorSpec

from .blob import SerializedExpert
from .manifest import ExpertBankManifest, ExpertBlobDescriptor


class _CompatibilitySpec(TypedDict):
    model_id: str
    num_routed_experts: int
    top_k: int
    moe_block_idxs: tuple[int, ...]


class BaseExpertBank(ABC):
    """*Abstract; do not instantiate.*"""

    _manifest: ExpertBankManifest
    _tensor_specs: tuple[TensorSpec, ...]
    _fd: int | None

    def __init__(self, expert_bank_path: Path) -> None:
        self._manifest = ExpertBankManifest.load(expert_bank_path)
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

    def __del__(self) -> None:
        self.close()

    @property
    def manifest(self) -> ExpertBankManifest:
        return self._manifest

    @property
    def blob_index(self) -> tuple[ExpertBlobDescriptor, ...]:
        return self.manifest.blob_index

    @property
    def model_fingerprint(self) -> str:
        return self._manifest.model_fingerprint

    @abstractmethod
    async def read(
        self,
        key: int,
        priority: ReadPriority,
    ) -> SerializedExpert: ...

    @abstractmethod
    def read_sync(
        self,
        key: int,
        priority: ReadPriority = ReadPriority.DEMAND,
    ) -> SerializedExpert: ...

    @abstractmethod
    def close(self) -> None: ...

    # TODO remove
    def key_for(
        self,
        block_idx: int,
        expert_idx: int,
        variant: str = "all",
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
            k: (expected[k], observed[k])
            for k in expected
            if expected[k] != observed[k]
        }:
            raise ExpertBankCompatibilityError(repr(mismatches))


@attrs.define(slots=True)
class PreadExpertBank(BaseExpertBank):
    """Uses positioned reads (via `os.pread`) to load expert weights from
    `experts.bin`. On MacOS, this can be substantially faster than memory mapping.
    """

    def __init__(
        self,
        expert_bank_path: Path,
        bypass_page_cache: bool = True,
    ) -> None:
        super().__init__(expert_bank_path)

        fd = os.open(expert_bank_path / EXPERTS_FILENAME, os.O_RDONLY)

        if bypass_page_cache and sys.platform == "darwin":
            fcntl.fcntl(fd, F_NOCACHE, 1)

        self._fd = fd

    def _validate_fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("Cannot read closed expert bank.")

        return self._fd

    def _validate_data_size(
        self,
        data: bytes,
        blob: ExpertBlobDescriptor,
    ) -> bytes:
        if len(data) != blob.length:
            raise IOError()  # TODO error msg

        return data

    def read_sync(
        self,
        expert_idx: int,
        priority: ReadPriority = ReadPriority.DEMAND,
    ) -> SerializedExpert:

        fd = self._validate_fd()
        blob = self.blob_index[expert_idx]
        data = os.pread(fd, blob.length, blob.offset)
        data = self._validate_data_size(data, blob)

        return SerializedExpert(
            idx=expert_idx,
            data=data,
            encoding=self._manifest.encoding,
            tensor_specs=self._tensor_specs,
        )

    async def read(
        self,
        expert_idx: int,
        priority: ReadPriority,
    ) -> SerializedExpert:

        fd = self._validate_fd()
        blob = self.blob_index[expert_idx]  # intentional KeyError

        data = await asyncio.to_thread(os.pread, fd, blob.length, blob.offset)
        data = self._validate_data_size(data, blob)

        return SerializedExpert(
            idx=expert_idx,
            data=data,
            encoding=self._manifest.encoding,
            tensor_specs=self._tensor_specs,
        )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@attrs.define(slots=True)
class MmapExpertBank(BaseExpertBank):
    """Expert bank using memory mapping to read expert weights from
    `experts.bin`.
    """

    _mmap: mmap.mmap | None

    def __init__(
        self,
        expert_bank_path: Path,
        **kwargs,
    ) -> None:
        super().__init__(expert_bank_path)

        self._mmap = None
        self._fd = None

        fd = os.open(expert_bank_path / EXPERTS_FILENAME, os.O_RDONLY)
        success = False
        try:
            file_size = os.fstat(fd).st_size
            if file_size > 0:
                self._mmap = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)

            self._fd = fd
            success = True

        finally:
            if not success:
                os.close(fd)

    def read_sync(
        self,
        expert_idx: int,
        priority: ReadPriority,
    ) -> SerializedExpert:
        if self._mmap is None:
            raise RuntimeError()  # TODO add message

        blob = self.blob_index[expert_idx]  # intentional KeyError
        slice_ = self._mmap[blob.offset : blob.offset + blob.length]

        if len(slice_) != blob.length:
            raise IOError()  # TODO add message

        return SerializedExpert(
            idx=expert_idx,
            data=slice_,
            encoding=self._manifest.encoding,
            tensor_specs=self._tensor_specs,
        )

    async def read(
        self,
        key: int,
        priority: ReadPriority,
    ) -> SerializedExpert:
        return self.read_sync(key, priority)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
