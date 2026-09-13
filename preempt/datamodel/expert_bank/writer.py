from __future__ import annotations

from typing import Self, BinaryIO
from types import TracebackType
from pathlib import Path

import attrs

from rich import print

import numpy as np

from preempt.core.constants import EXPERTS_FILENAME, MANIFEST_FILENAME

from preempt.datamodel.identity import TensorSpec

from .manifest import (
    ExpertBlobDescriptor,
    ExpertBankManifest,
    ModelMoESpec,
)

# TODO support heterogenous weight sizes between MoE blocks
# TODO double check resource cleanup and context management


@attrs.define(slots=True)
class ExpertBankWriter:
    path: Path
    model_id: str
    model_fingerprint: str

    encoding: str
    tensor_specs: tuple[TensorSpec, ...]
    model_moe_spec: ModelMoESpec

    verbose: bool

    _blob_size: int
    _aligned_blob_size: int

    _file: BinaryIO
    _alignment: int
    _blob_descriptors: np.ndarray
    _num_writes: int
    _is_closed: bool

    def __init__(
        self,
        expert_bank_path: Path,
        *,
        model_id: str,
        model_fingerprint: str,
        encoding: str,
        tensor_specs: tuple[TensorSpec, ...],
        model_moe_spec: ModelMoESpec,
        alignment: int = 4096,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        self.path = Path(expert_bank_path)
        self.model_id = model_id
        self.model_fingerprint = model_fingerprint
        self.encoding = encoding
        self.tensor_specs = tensor_specs
        self.model_moe_spec = model_moe_spec
        self.verbose = verbose

        self._alignment = alignment
        self._blob_size, self._aligned_blob_size = self._get_blob_sizes(alignment)

        self._blob_descriptors = np.empty(
            shape=(model_moe_spec.total_routed_experts),
            dtype=object,
        )

        self._num_writes = 0
        self._is_closed = False

        bin_path = self.path / EXPERTS_FILENAME
        if bin_path.exists() and not overwrite:
            raise FileExistsError(
                f"Expert bank already exists at {self.path.as_posix()!r}."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / MANIFEST_FILENAME).unlink(missing_ok=True)
        self._file = bin_path.open("wb")

    def _get_blob_sizes(self, alignment: int) -> tuple[int, int]:
        """Returns smallest multiple of `alignment` greater than or equal to expert blob size"""
        blob_size = sum(spec.num_bytes for spec in self.tensor_specs)
        aligned_blob_size = (blob_size + alignment - 1) // alignment * alignment

        return blob_size, aligned_blob_size

    def _get_expert_glob_idx(self, block_idx: int, expert_idx: int) -> int:
        return (block_idx * self.model_moe_spec.num_routed_experts) + expert_idx

    def add_expert(
        self,
        block_idx: int,
        expert_idx: int,
        data: bytes,
    ) -> None:
        # Offset based on expert's position in the model, not write order
        glob_idx = self._get_expert_glob_idx(block_idx, expert_idx)
        target_offset = glob_idx * self._aligned_blob_size

        self._file.seek(target_offset)
        self._file.write(data)

        padding_needed = self._aligned_blob_size - len(data)
        if padding_needed > 0:
            self._file.write(b"\x00" * padding_needed)

        blob_desc = ExpertBlobDescriptor(
            offset=target_offset,
            length=len(data),
            block_idx=block_idx,
            expert_idx=expert_idx,
        )
        self._blob_descriptors[glob_idx] = blob_desc
        self._num_writes += 1
        if self.verbose:
            print(
                f"Expert {self._num_writes}/{self.model_moe_spec.total_routed_experts}",
                end="\r",
                flush=True,
            )

    def finalize(self) -> ExpertBankManifest:
        if self._is_closed:
            raise RuntimeError()  # TODO error msg

        if self._num_writes != self.model_moe_spec.total_routed_experts:
            raise RuntimeError()  # TODO error msg

        self._file.close()
        self._is_closed = True

        manifest = ExpertBankManifest(
            model_id=self.model_id,
            model_fingerprint=self.model_fingerprint,
            encoding=self.encoding,
            alignment=self._alignment,
            tensor_specs=self.tensor_specs,
            model_moe_spec=self.model_moe_spec,
            blob_index=tuple(self._blob_descriptors.tolist()),
        )
        manifest.save(self.path)

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
