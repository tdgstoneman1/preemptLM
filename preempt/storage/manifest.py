from __future__ import annotations

from typing import Final, Self

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.core.identity import ExpertKey

MANIFEST_FILENAME: Final[str] = "manifest.json"
EXPERTS_FILENAME: Final[str] = "experts.bin"
STORE_SCHEMA_VERSION: Final[int] = 1


class StoreCompatibilityError(RuntimeError):
    """Raised when a packed store does not match the loaded model."""


class TensorSpec(BaseModel):
    """Shape/dtype of one per-expert tensor inside a blob, in blob order.

    Attributes
    ----------
    name : str
        Tensor name relative to one expert, e.g. `gate_proj.weight`.
    dtype : str
        Numpy dtype name the raw bytes decode as, e.g. `uint32`.
    shape : tuple[int, ...]
        Shape of this tensor for a single expert.
    nbytes : int
        Size of this tensor's contribution to one expert blob.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    nbytes: int = Field(gt=0)


class ExpertTopology(BaseModel):
    """Routed-expert topology of the model a store was packed from.

    Attributes
    ----------
    moe_layer_idxs : tuple[int, ...]
        Block indices of the layers that route to experts.
    num_experts : int
        Number of routed experts per MoE layer.
    top_k : int
        Number of experts the router selects per token.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    moe_layer_idxs: tuple[int, ...] = Field(min_length=1)
    num_experts: int = Field(ge=1)
    top_k: int = Field(ge=1)


class ExpertBlobRecord(BaseModel):
    """Location of one expert's fused weight blob within `experts.bin`.

    Attributes
    ----------
    layer_idx : int
        Block index of the MoE layer owning this expert.
    expert_idx : int
        Router-local expert index within that layer.
    variant : str
        Weight variant the blob holds; `all` is the full fused blob.
    offset : int
        Byte offset of the blob from the start of the payload file.
    length : int
        Blob length in bytes; equal for every expert in a store.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    variant: str = Field(default="all", min_length=1)
    offset: int = Field(ge=0)
    length: int = Field(gt=0)


class ExpertStoreManifest(BaseModel):
    """Index of a packed expert store, written alongside `experts.bin`.

    The manifest crosses a file boundary and is the compatibility contract
    between a packed store and a loaded model: the reader checks
    `model_fingerprint`, `payload_encoding`, and `topology` before serving a
    single byte.

    Attributes
    ----------
    schema_version : int
        Layout version of this manifest; bumped when the format changes.
    model_id : str
        Human-readable source model identifier, e.g. a Hugging Face repo id.
    model_fingerprint : str
        Fingerprint of the weights the store was packed from.
    payload_encoding : str
        Opaque tag naming how blob bytes are encoded, e.g. `mlx-affine-q4-g64`.
    alignment : int
        Byte alignment every blob offset satisfies in the payload file.
    tensor_specs : tuple[TensorSpec, ...]
        Per-expert tensor layout, in the order the tensors appear in a blob.
    topology : ExpertTopology
        Routed-expert topology of the source model.
    blobs : tuple[ExpertBlobRecord, ...]
        One record per packed expert.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=STORE_SCHEMA_VERSION, ge=1)
    model_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)
    payload_encoding: str = Field(min_length=1)
    alignment: int = Field(default=4096, ge=1)
    tensor_specs: tuple[TensorSpec, ...] = Field(min_length=1)
    topology: ExpertTopology
    blobs: tuple[ExpertBlobRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blobs(self) -> Self:
        """Reject blobs that disagree with the tensor specs or repeat an identity.

        Returns
        -------
        Self
            The validated manifest.

        Raises
        ------
        ValueError
            If a blob's length differs from the per-expert size implied by
            `tensor_specs`, or two blobs share a (layer, expert, variant)
            identity.
        """
        expected = self.expert_nbytes()
        identities = set()

        for blob in self.blobs:
            if blob.length != expected:
                raise ValueError(
                    f"Blob (layer {blob.layer_idx}, expert {blob.expert_idx}) has "
                    f"length {blob.length}; tensor specs total {expected}."
                )
            identity = (blob.layer_idx, blob.expert_idx, blob.variant)
            if identity in identities:
                raise ValueError(f"Duplicate blob identity: {identity!r}")
            identities.add(identity)

        return self

    def expert_nbytes(self) -> int:
        """Return the byte size of one expert blob.

        Returns
        -------
        int
            Sum of `nbytes` over all tensor specs.
        """
        return sum(spec.nbytes for spec in self.tensor_specs)

    def blob_index(self) -> dict[ExpertKey, ExpertBlobRecord]:
        """Return the blob records keyed by fully qualified expert identity.

        Returns
        -------
        dict[ExpertKey, ExpertBlobRecord]
            Lookup table the reader uses to resolve a routed expert to bytes.
        """
        return {
            ExpertKey(
                model_fingerprint=self.model_fingerprint,
                layer_idx=blob.layer_idx,
                expert_idx=blob.expert_idx,
                variant=blob.variant,
            ): blob
            for blob in self.blobs
        }

    def save(self, store_dir: Path) -> Path:
        """Write the manifest into `store_dir`, creating it if needed.

        Parameters
        ----------
        store_dir : Path
            Directory holding the store's payload file.

        Returns
        -------
        Path
            Path of the written manifest file.
        """
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, store_dir: Path) -> Self:
        """Read and validate the manifest inside `store_dir`.

        Parameters
        ----------
        store_dir : Path
            Directory holding the store's manifest and payload file.

        Returns
        -------
        Self
            The validated manifest.

        Raises
        ------
        FileNotFoundError
            If `store_dir` contains no manifest file.
        pydantic.ValidationError
            If the manifest is malformed or internally inconsistent.
        """
        raw = (store_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)
