from __future__ import annotations

from typing import Self
from pathlib import Path

import attrs
from attrs import field

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preempt.datamodel.identity import TensorSpec
from preempt.core.constants import (
    EXPERT_BANK_SCHEMA_VERSION,
    MANIFEST_FILENAME,
)


class ModelMoESpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    moe_block_idxs: tuple[int, ...] = Field(min_length=1)
    num_routed_experts: int = Field(ge=1)
    top_k: int = Field(ge=1)

    @property
    def total_routed_experts(self) -> int:
        return len(self.moe_block_idxs) * self.num_routed_experts


class ExpertBlobDescriptor(BaseModel):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    block_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)


# TODO figure out how to use attrs dataclasses with pydantic
# @attrs.define(slots=True)
# class ExpertBlobDescriptor:
#     offset: int = field(validator=attrs.validators.ge(0))
#     length: int = field(validator=attrs.validators.ge(0))
#     block_idx: int = field(validator=attrs.validators.ge(0))
#     expert_idx: int = field(validator=attrs.validators.ge(0))


class ExpertBankManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(default=EXPERT_BANK_SCHEMA_VERSION, ge=1)

    model_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)

    encoding: str = Field(min_length=1)
    tensor_specs: tuple[TensorSpec, ...] = Field(min_length=1)
    model_moe_spec: ModelMoESpec

    alignment: int = Field(default=4096, ge=1)
    blob_index: tuple[ExpertBlobDescriptor, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_blobs(self) -> Self:
        if (
            len(self.blob_index)
            != len(self.model_moe_spec.moe_block_idxs)
            * self.model_moe_spec.num_routed_experts
        ):
            raise ValueError()  # TODO error msg

        expected_length = self.expert_num_bytes()
        offsets = set()
        expert_idxs = set()

        for blob in self.blob_index:
            if blob.length != expected_length:
                raise ValueError()  # TODO error msg

            if (idx := (blob.block_idx, blob.expert_idx)) in expert_idxs:
                raise ValueError("Manifest contains duplicate blobs")

            if blob.offset in offsets:
                raise ValueError()  # TODO error msg

            expert_idxs.add(idx)
            offsets.add(blob.offset)

        return self

    def expert_num_bytes(self) -> int:
        return sum(spec.num_bytes for spec in self.tensor_specs)

    def save(self, expert_bank_path: Path) -> Path:
        expert_bank_path.mkdir(parents=True, exist_ok=True)

        path = expert_bank_path / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

        return path

    @classmethod
    def load(cls, expert_bank_path: Path) -> ExpertBankManifest:
        raw = (expert_bank_path / MANIFEST_FILENAME).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)
