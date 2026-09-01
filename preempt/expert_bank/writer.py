from __future__ import annotations

from typing import Self, BinaryIO
from types import TracebackType

from pathlib import Path

from preempt.datamodel.identity import TensorSpec
from preempt.core.constants import EXPERTS_FILENAME, MANIFEST_FILENAME

from .manifest import (
    ExpertBlobRecord,
    ExpertBankManifest,
    ModelMoESpec,
)


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
        expert_bank_path: Path,
        *,
        model_id: str,
        model_fingerprint: str,
        payload_encoding: str,
        tensor_specs: tuple[TensorSpec, ...],
        model_moe_spec: ModelMoESpec,
        alignment: int = 4096,
        overwrite: bool = False,
    ) -> None:
        self._store_dir = Path(expert_bank_path)
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
