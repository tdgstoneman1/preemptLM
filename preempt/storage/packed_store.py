from __future__ import annotations

from typing import Self

import asyncio
import os
from pathlib import Path
from types import TracebackType

from preempt.core.identity import ExpertKey
from preempt.core.protocols.store import ExpertPayload, ReadPriority
from preempt.storage.manifest import (
    EXPERTS_FILENAME,
    ExpertBlobRecord,
    ExpertStoreManifest,
    StoreCompatibilityError,
)


class PackedExpertStore:
    """`ExpertStore` over a packed store directory: one `pread` per expert.

    Priority is accepted per the protocol but ignored here — read *ordering*
    is the scheduler's job; the store just reads what it is told.

    Parameters
    ----------
    store_dir : Path
        Directory holding `manifest.json` and `experts.bin`.

    Raises
    ------
    FileNotFoundError
        If `store_dir` holds no manifest or no payload file.
    pydantic.ValidationError
        If the manifest is malformed or internally inconsistent.
    """

    def __init__(self, store_dir: Path) -> None:
        self._manifest = ExpertStoreManifest.load(store_dir)
        self._index: dict[ExpertKey, ExpertBlobRecord] = self._manifest.blob_index()
        self._fd: int | None = os.open(store_dir / EXPERTS_FILENAME, os.O_RDONLY)

    @property
    def manifest(self) -> ExpertStoreManifest:
        """Return the manifest describing this store.

        Returns
        -------
        ExpertStoreManifest
            The validated manifest loaded from the store directory.
        """
        return self._manifest

    @property
    def fingerprint(self) -> str:
        """Return the model fingerprint every key in this store carries.

        Returns
        -------
        str
            The manifest's `model_fingerprint`.
        """
        return self._manifest.model_fingerprint

    def key_for(
        self, layer_idx: int, expert_idx: int, variant: str = "all"
    ) -> ExpertKey:
        """Build an `ExpertKey` for this store from a router-local identity.

        Callers know a layer and an expert index; only the store knows which
        fingerprint its blobs were packed under. Minting the key here is what
        keeps an invented fingerprint from turning a valid expert into a
        `KeyError` at read time.

        Parameters
        ----------
        layer_idx : int
            Block index of the MoE layer owning the expert.
        expert_idx : int
            Router-local expert index within that layer.
        variant : str
            Weight variant to address; `all` is the full fused blob.

        Returns
        -------
        ExpertKey
            A key carrying this store's own fingerprint. Whether the store
            actually holds a blob for it is decided by `read`.
        """
        return ExpertKey(
            model_fingerprint=self.fingerprint,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            variant=variant,
        )

    def ensure_compatible(
        self,
        *,
        model_id: str,
        num_experts: int,
        top_k: int,
        moe_layer_idxs: tuple[int, ...],
    ) -> None:
        """Refuse to serve a store that does not describe the loaded model.

        Parameters
        ----------
        model_id : str
            Identifier of the loaded model.
        num_experts : int
            Routed experts per MoE layer in the loaded model.
        top_k : int
            Experts the loaded model's router selects per token.
        moe_layer_idxs : tuple[int, ...]
            Block indices of the loaded model's MoE layers.

        Raises
        ------
        StoreCompatibilityError
            If any of the above disagrees with the store's manifest.
        """
        observed = {
            "model_id": model_id,
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_idxs": moe_layer_idxs,
        }
        expected = {
            "model_id": self._manifest.model_id,
            "num_experts": self._manifest.topology.num_experts,
            "top_k": self._manifest.topology.top_k,
            "moe_layer_idxs": self._manifest.topology.moe_layer_idxs,
        }
        mismatches = {
            name: (expected[name], observed[name])
            for name in expected
            if expected[name] != observed[name]
        }
        if mismatches:
            raise StoreCompatibilityError(
                f"Packed store does not match the loaded model: {mismatches!r}"
            )

    async def read(self, key: ExpertKey, priority: ReadPriority) -> ExpertPayload:
        """Read one expert's blob out of the payload file.

        Parameters
        ----------
        key : ExpertKey
            Fully qualified identity of the expert to read.
        priority : ReadPriority
            Accepted for protocol conformance; ordering is the scheduler's job.

        Returns
        -------
        ExpertPayload
            The blob bytes tagged with the store's payload encoding.

        Raises
        ------
        RuntimeError
            If the store has been closed.
        KeyError
            If the store holds no blob for `key`.
        IOError
            If the payload file returns fewer bytes than the manifest records.
        """
        if self._fd is None:
            raise RuntimeError("Store is closed.")

        blob = self._index[key]  # KeyError for unknown keys, by design
        data = await asyncio.to_thread(os.pread, self._fd, blob.length, blob.offset)

        if len(data) != blob.length:
            raise IOError(
                f"Short read for {key!r}: got {len(data)} of {blob.length} bytes."
            )

        return ExpertPayload(
            key=key, data=data, encoding=self._manifest.payload_encoding
        )

    def close(self) -> None:
        """Close the payload file descriptor; idempotent."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Self:
        """Enter the reading context.

        Returns
        -------
        Self
            This store.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the payload file descriptor on exit.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Class of the exception leaving the context, if any.
        exc : BaseException | None
            Exception instance leaving the context, if any.
        traceback : TracebackType | None
            Traceback of that exception, if any.
        """
        self.close()
