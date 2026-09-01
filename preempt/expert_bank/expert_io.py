from __future__ import annotations

from typing import Self, BinaryIO
from types import TracebackType

from pathlib import Path

import asyncio

import fcntl
import os
import sys
import mmap

from preempt.core.protocols import ExpertPayload, ReadPriority
from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.exceptions import ExpertBankCompatibilityError
from preempt.core.constants import EXPERTS_FILENAME, MANIFEST_FILENAME, F_NOCACHE

from .manifest import (
    ExpertBlobRecord,
    ExpertBankManifest,
    ModelMoESpec,
)

# TODO rewrite this comment slop
# macOS `<sys/fcntl.h>` value of `F_NOCACHE`; absent from Python's `fcntl`
# module, so it is spelled out here. Turns off page caching for reads/writes
# on the fd, so blobs come off the SSD rather than the kernel's file cache.


class ExpertBank:  # TODO rename to PreadExpertBank
    """Implements `IExpertBank` interface."""

    _manifest: ExpertBankManifest
    _index: dict[ExpertKey, ExpertBlobRecord]
    _tensor_specs: tuple[TensorSpec, ...]
    _fd: int | None

    def __init__(self, store_dir: Path, *, bypass_page_cache: bool = True) -> None:
        self._manifest = ExpertBankManifest.load(store_dir)
        self._index = self._manifest.blob_index()
        self._tensor_specs = self._manifest.tensor_specs

        fd = os.open(store_dir / EXPERTS_FILENAME, os.O_RDONLY)
        if bypass_page_cache and sys.platform == "darwin":
            fcntl.fcntl(fd, F_NOCACHE, 1)

        self._fd = fd

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
        *,
        model_id: str,
        num_routed_experts: int,
        top_k: int,
        moe_block_idxs: tuple[int, ...],
    ) -> None:
        observed = {
            "model_id": model_id,
            "num_routed_experts": num_routed_experts,
            "top_k": top_k,
            "moe_block_idxs": moe_block_idxs,
        }  # TODO use TypedDict
        expected = {
            "model_id": self._manifest.model_id,
            "num_routed_experts": self._manifest.model_moe_spec.num_routed_experts,
            "top_k": self._manifest.model_moe_spec.top_k,
            "moe_block_idxs": self._manifest.model_moe_spec.moe_block_idxs,
        }  # TODO use TypedDict

        mismatches = {
            name: (expected[name], observed[name])
            for name in expected
            if expected[name] != observed[name]
        }
        if mismatches:
            raise ExpertBankCompatibilityError(repr(mismatches))

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        """Read blob for one expert."""
        if self._fd is None:
            raise RuntimeError("Cannot read closed expert bank.")

        blob = self._index[key]  # intentional KeyError
        data = await asyncio.to_thread(os.pread, self._fd, blob.length, blob.offset)

        if len(data) != blob.length:
            raise IOError()

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

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class MmapExpertBank(ExpertBank):
    """Implements the `IExpertBank` interface using a memory-mapped file."""

    _manifest: ExpertBankManifest
    _index: dict[ExpertKey, ExpertBlobRecord]
    _tensor_specs: tuple[TensorSpec, ...]
    _fd: int | None
    _mmap: mmap.mmap | None

    def __init__(self, store_dir: Path, **kwargs) -> None:
        self._manifest = ExpertBankManifest.load(store_dir)
        self._index = self._manifest.blob_index()
        self._tensor_specs = self._manifest.tensor_specs

        self._mmap = None
        self._fd = None

        success = False
        fd = os.open(store_dir / EXPERTS_FILENAME, os.O_RDONLY)
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


class ExpertBankWriter:
    _store_dir: Path

    _model_id: str
    _model_fingerprint: str

    _payload_encoding: str
    _tensor_specs: tuple[TensorSpec, ...]
    _model_moe_spec: ModelMoESpec

    _file: BinaryIO
    _alignment: int
    _expected_num_bytes: int
    _offset: int
    _blobs: list[ExpertBlobRecord]
    _identities: set[tuple[int, int, str]]
    _is_closed: bool

    def __init__(
        self,
        store_dir: Path,
        *,
        model_id: str,
        model_fingerprint: str,
        payload_encoding: str,
        tensor_specs: tuple[TensorSpec, ...],
        model_moe_spec: ModelMoESpec,
        alignment: int = 4096,
        overwrite: bool = False,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._model_id = model_id
        self._model_fingerprint = model_fingerprint
        self._payload_encoding = payload_encoding
        self._tensor_specs = tensor_specs
        self._model_moe_spec = model_moe_spec
        self._alignment = alignment
        self._expected_num_bytes = sum(spec.num_bytes for spec in tensor_specs)

        self._offset = 0
        self._blobs = []
        self._identities = set()
        self._is_closed = False

        # TODO dedicated method for i/o stuff
        bin_path = self._store_dir / EXPERTS_FILENAME
        if bin_path.exists() and not overwrite:
            raise FileExistsError(f"expert bank already exists at `{self._store_dir}`.")

        self._store_dir.mkdir(parents=True, exist_ok=True)
        (self._store_dir / MANIFEST_FILENAME).unlink(missing_ok=True)

        self._file = bin_path.open("wb")

    def add_expert(
        self, *, block_idx: int, expert_idx: int, data: bytes, variant: str = "all"
    ) -> None:
        if len(data) != self._expected_num_bytes:
            raise ValueError(
                f"Expected blob to be {self._expected_num_bytes} bytes, but got {len(data)}."
            )

        identity = (block_idx, expert_idx, variant)
        if identity in self._identities:
            raise ValueError(f"Found duplicate expert blobs: {identity!r}")

        self._identities.add(identity)

        padding = -self._offset % self._alignment
        if padding:
            self._file.write(b"\x00" * padding)
            self._offset += padding

        self._file.write(data)
        self._blobs.append(
            ExpertBlobRecord(
                block_idx=block_idx,
                expert_idx=expert_idx,
                variant=variant,
                offset=self._offset,
                length=len(data),
            )
        )
        self._offset += len(data)

    def finalize(self) -> ExpertBankManifest:
        if self._is_closed:
            raise RuntimeError()

        self._file.close()
        self._is_closed = True

        manifest = ExpertBankManifest(
            model_id=self._model_id,
            model_fingerprint=self._model_fingerprint,
            payload_encoding=self._payload_encoding,
            alignment=self._alignment,
            tensor_specs=self._tensor_specs,
            model_moe_spec=self._model_moe_spec,
            blobs=tuple(self._blobs),
        )
        manifest.save(self._store_dir)

        return manifest

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:  # abandon w/o writing manifest
            self._file.close()
            return

        if not self._is_closed:
            self.finalize()
