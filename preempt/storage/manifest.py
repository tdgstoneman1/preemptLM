"""Pydantic models for an expert bank's `manifest.json`"""

from __future__ import annotations

from typing import Self

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.core.identity import ExpertKey, TensorSpec
from preempt.core.constants import (
    MANIFEST_FILENAME,
    EXPERT_BANK_SCHEMA,
)


class ModelMoESpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    moe_block_idxs: tuple[int, ...] = Field(min_length=1)
    num_routed_experts: int = Field(ge=1)
    top_k: int = Field(ge=1)


class ExpertBlobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    variant: str = Field(default="all", min_length=1)
    offset: int = Field(ge=0)
    length: int = Field(gt=0)


class ExpertBankManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=EXPERT_BANK_SCHEMA, ge=1)

    model_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)

    payload_encoding: str = Field(min_length=1)  # TODO rename
    tensor_specs: tuple[TensorSpec, ...] = Field(min_length=1)
    model_moe_spec: ModelMoESpec

    alignment: int = Field(default=4096, ge=1)
    blobs: tuple[ExpertBlobRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blobs(self) -> Self:
        expected = self.expert_num_bytes()
        ids = set()

        for blob in self.blobs:
            if blob.length != expected:
                raise ValueError()

            id = (blob.block_idx, blob.expert_idx, blob.variant)
            if id in ids:
                raise ValueError("Manifest contains duplicate blobs.")

            ids.add(id)

        return self

    def expert_num_bytes(self) -> int:  # TODO rename
        return sum(spec.num_bytes for spec in self.tensor_specs)

    def blob_index(self) -> dict[ExpertKey, ExpertBlobRecord]:
        return {
            ExpertKey(
                model_fingerprint=self.model_fingerprint,
                block_idx=blob.block_idx,
                expert_idx=blob.expert_idx,
                variant=blob.variant,
            ): blob
            for blob in self.blobs
        }

    def save(self, expert_bank_dir: Path) -> Path:
        expert_bank_dir.mkdir(parents=True, exist_ok=True)
        path = expert_bank_dir / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

        return path

    @classmethod
    def load(cls, expert_bank_dir: Path) -> ExpertBankManifest:  # TODO rename
        raw = (expert_bank_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)
