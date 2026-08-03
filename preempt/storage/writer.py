from __future__ import annotations

from typing import BinaryIO, Self

from pathlib import Path
from types import TracebackType

from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    MANIFEST_FILENAME,
    ExpertBlobRecord,
    ExpertStoreManifest,
    ExpertTopology,
    TensorSpec,
)


class PackedStoreWriter:
    """Sole authority on the packed store container layout.

    Appends page-aligned expert blobs to `experts.bin` and writes the
    manifest on `finalize()`. Converters (per source format, living in their
    backend) extract tensors and feed `add_expert`; nothing else in the
    project writes this layout.

    Padding exists only in the gaps between blobs: a recorded `length` is
    always the exact per-expert payload size implied by `tensor_specs`, which
    is what `ExpertStoreManifest` validates.

    A manifest exists only for a store that finished writing: construction
    removes any manifest already present in `store_dir`, so an interrupted
    session leaves an unreadable store rather than a stale index pointing at
    replaced bytes.

    Parameters
    ----------
    store_dir : Path
        Directory to hold `experts.bin` and `manifest.json`; created if absent.
    model_id : str
        Human-readable source model identifier, e.g. a Hugging Face repo id.
    model_fingerprint : str
        Fingerprint of the weights being packed.
    payload_encoding : str
        Opaque tag naming how blob bytes are encoded.
    tensor_specs : tuple[TensorSpec, ...]
        Per-expert tensor layout, in the order tensors appear in a blob.
    topology : ExpertTopology
        Routed-expert topology of the source model.
    alignment : int
        Byte alignment every blob offset satisfies, by default 4096.
    overwrite : bool
        Whether to replace an existing store in `store_dir`, by default False.

    Raises
    ------
    FileExistsError
        If `store_dir` already holds a payload file and `overwrite` is False.
    """

    def __init__(
        self,
        store_dir: Path,
        *,
        model_id: str,
        model_fingerprint: str,
        payload_encoding: str,
        tensor_specs: tuple[TensorSpec, ...],
        topology: ExpertTopology,
        alignment: int = 4096,
        overwrite: bool = False,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._model_id = model_id
        self._model_fingerprint = model_fingerprint
        self._payload_encoding = payload_encoding
        self._tensor_specs = tensor_specs
        self._topology = topology
        self._alignment = alignment
        self._expected_length = sum(spec.nbytes for spec in tensor_specs)

        bin_path = self._store_dir / EXPERTS_FILENAME
        if bin_path.exists() and not overwrite:
            raise FileExistsError(f"Store already exists at `{self._store_dir}`.")

        self._store_dir.mkdir(parents=True, exist_ok=True)
        # Truncating the payload invalidates any manifest already in the
        # directory, so drop it before the first byte is written: a store is
        # only ever readable after a successful `finalize()`, and a session
        # that dies mid-write (exception or hard kill) must not leave an old
        # manifest describing bytes that no longer exist.
        (self._store_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
        self._file: BinaryIO = bin_path.open("wb")
        self._offset = 0
        self._blobs: list[ExpertBlobRecord] = []
        self._identities: set[tuple[int, int, str]] = set()
        self._finalized = False

    def add_expert(
        self, *, layer_idx: int, expert_idx: int, data: bytes, variant: str = "all"
    ) -> None:
        """Append one expert's fused blob at the next aligned offset.

        Parameters
        ----------
        layer_idx : int
            Block index of the MoE layer owning this expert.
        expert_idx : int
            Router-local expert index within that layer.
        data : bytes
            Fused blob bytes; must total the size implied by `tensor_specs`.
        variant : str
            Weight variant the blob holds, by default `all`.

        Raises
        ------
        ValueError
            If `data` is not exactly the per-expert size, or this
            (layer, expert, variant) identity was already written.
        """
        if len(data) != self._expected_length:
            raise ValueError(
                f"Expert blob must be exactly {self._expected_length} bytes "
                f"(sum of tensor specs); got {len(data)}."
            )

        identity = (layer_idx, expert_idx, variant)
        if identity in self._identities:
            raise ValueError(f"Duplicate expert blob: {identity!r}")
        self._identities.add(identity)

        padding = -self._offset % self._alignment
        if padding:
            self._file.write(b"\x00" * padding)
            self._offset += padding

        self._file.write(data)
        self._blobs.append(
            ExpertBlobRecord(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                variant=variant,
                offset=self._offset,
                length=len(data),
            )
        )
        self._offset += len(data)

    def finalize(self) -> ExpertStoreManifest:
        """Close the payload file and write the manifest describing it.

        Returns
        -------
        ExpertStoreManifest
            The manifest written into the store directory.

        Raises
        ------
        RuntimeError
            If the writer was already finalized.
        """
        if self._finalized:
            raise RuntimeError("Writer already finalized.")

        self._file.close()
        self._finalized = True

        manifest = ExpertStoreManifest(
            model_id=self._model_id,
            model_fingerprint=self._model_fingerprint,
            payload_encoding=self._payload_encoding,
            alignment=self._alignment,
            tensor_specs=self._tensor_specs,
            topology=self._topology,
            blobs=tuple(self._blobs),
        )
        manifest.save(self._store_dir)
        return manifest

    def __enter__(self) -> Self:
        """Enter the writing context.

        Returns
        -------
        Self
            This writer.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize on clean exit; abandon the partial store on error.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Class of the exception leaving the context, if any.
        exc : BaseException | None
            Exception instance leaving the context, if any.
        traceback : TracebackType | None
            Traceback of that exception, if any.
        """
        if exc_type is not None:
            self._file.close()  # abandon a partial store; no manifest written
            return
        if not self._finalized:
            self.finalize()
