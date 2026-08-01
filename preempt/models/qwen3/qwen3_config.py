from typing import Optional, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

import torch


class Qwen3Config(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace repo ID used to load pretrained tokenizer.",
    )
    tokenizer_fp: Optional[str] = Field(
        default=None, description="File path to saved tokenizer"
    )
    vocab_size: int = Field(default=151_936)
    context_length: int = Field(
        default=40_960, description="Context length model was trained with"
    )

    # Model size and dimensionality
    num_transformer_blocks: int = Field(
        description="Number of transformer blocks in the model"
    )
    d_model: int = Field(description="Model's embedding dimension")
    d_hidden: int = Field(description="Intermediate dimension in feed-forward nets")

    # Grouped query attention
    num_attn_heads: int = Field(description="Number of GQA heads")
    d_head: int | None = Field(default=None, description="Size of the attention heads")
    num_attn_kv_groups: int = Field(description="Key-Value groups for GQA")
    apply_qk_norm: bool = Field(
        default=True, description="Whether to normalize queries and keys in GQA"
    )

    rope_theta_base: float = Field(
        default=10_000.0, description="The base in RoPE's 'theta'"
    )
    rms_norm_eps: float = Field(default=1e-6)

    dtype: torch.dtype = Field(default=torch.bfloat16)
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.model_id is None and self.tokenizer_fp is None:
            raise ValueError(
                "Either `model_id` or `tokenizer_fp` must be provided to load tokenizer."
            )
        return self


# ================================================================================================ #
#                Preset factories with official settings for different size variants               #
# ================================================================================================ #


def Qwen3Config_0_6b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-0.6B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=1024,
        num_attn_heads=16,
        num_transformer_blocks=28,
        d_hidden=3072,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )


def Qwen3Config_1_3b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-1.3B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=2048,
        num_attn_heads=16,
        num_transformer_blocks=28,
        d_hidden=6144,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )


def Qwen3Config_4b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-4B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=2560,
        num_attn_heads=32,
        num_transformer_blocks=36,
        d_hidden=9728,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )


def Qwen3Config_8b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-8B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=4096,
        num_attn_heads=32,
        num_transformer_blocks=36,
        d_hidden=12288,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )


def Qwen3Config_14b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-14B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=5120,
        num_attn_heads=40,
        num_transformer_blocks=40,
        d_hidden=17408,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )


def Qwen3Config_32b() -> Qwen3Config:
    return Qwen3Config(
        model_id="Qwen/Qwen3-32B-Base",
        vocab_size=151_936,
        context_length=40_960,
        d_model=5120,
        num_attn_heads=64,
        num_transformer_blocks=64,
        d_hidden=25600,
        d_head=128,
        apply_qk_norm=True,
        num_attn_kv_groups=8,
        rope_theta_base=1_000_000.0,
        dtype=torch.bfloat16,
    )
