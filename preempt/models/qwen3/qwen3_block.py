"""Layers used in transformer blocks"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .qwen3_config import Qwen3Config

from ..ops.rope import apply_rope

# TODO define weight transfer logic for concatenating
# pretrained weights for `gate_up` and `gate_down`
# for use with fused SwiGLU's `gate_up_proj` layer


class SwiGLU_MLP(nn.Module):
    """Fused SwiGLU MLP"""

    gate_proj: nn.Linear
    down_proj: nn.Linear
    up_proj: nn.Linear
    # gate_up_proj: nn.Linear # ~ fused SwiGLU

    def __init__(
        self, d_model: int, d_hidden: int, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, dtype=dtype, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, dtype=dtype, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, dtype=dtype, bias=False)
        # self.gate_up_proj = nn.Linear(
        #     config.d_model, 2 * config.d_hidden, dtype=config.dtype, bias=False
        # ) # ~ use for more efficient fused SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up = self.gate_up_proj(x) # ~ Fused SwiGLU
        # gate, up = gate_up.chunk(2, dim=-1) # ~ Fused SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = nn.functional.silu(gate) * up

        return self.down_proj(x)


class RMSNorm(nn.Module):
    eps: float
    scale: nn.Parameter
    shift: nn.Parameter | None

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class GroupedQueryAttn(nn.Module):
    d_head: int
    d_out: int
    num_attn_heads: int
    num_attn_kv_groups: int
    group_size: int

    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear

    q_norm: RMSNorm | None
    k_norm: RMSNorm | None

    def __init__(
        self,
        d_in: int,
        num_attn_heads: int,
        num_attn_kv_groups: int,
        d_head: Optional[int] = None,
        apply_qk_norm: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if num_attn_heads % num_attn_kv_groups != 0:
            raise ValueError(
                "Number of attention heads must be evenly divisible by "
                f"number of kv heads; got {num_attn_heads=} and "
                f"{num_attn_kv_groups=})"
            )

        self.num_attn_heads = num_attn_heads
        self.num_attn_kv_groups = num_attn_kv_groups
        self.group_size = num_attn_heads // num_attn_kv_groups

        if d_head is None:
            if d_in % num_attn_heads != 0:
                "Input dimension must be evenly divisible by number of"
                "attention heads if head dimension is not provided; got "
                f"{num_attn_heads=} and {d_head=})"

            d_head = d_in // num_attn_heads

        self.d_head = d_head
        self.d_out = num_attn_heads * d_head

        self.q_proj = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(
            d_in, num_attn_kv_groups * d_head, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            d_in, num_attn_kv_groups * d_head, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if apply_qk_norm:
            self.q_norm = RMSNorm(d_head, eps=1e-6)
            self.k_norm = RMSNorm(d_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[
        Any, tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]
    ]:
        B, S, _ = x.shape

        # Apply projections
        q = self.q_proj(x)  # (B, S, num_attn_heads * d_head)
        k = self.k_proj(x)  # (B, S, num_attn_kv_groups * d_head)
        v = self.v_proj(x)  # (B, S, num_attn_kv_groups * d_head)

        # Reshape
        q = q.view(B, S, self.num_attn_heads, self.d_head).transpose(1, 2)
        if self.q_norm:
            q = self.q_norm(q)

        next_k = k.view(B, S, self.num_attn_kv_groups, self.d_head).transpose(1, 2)
        if self.k_norm:
            next_k = self.k_norm(next_k)

        next_v = v.view(B, S, self.num_attn_kv_groups, self.d_head).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos, sin, offset=start_pos)
        next_k = apply_rope(next_k, cos, sin, offset=start_pos)

        if cache is None:
            start_pos = 0  # reset RoPE
            k, v = next_k, next_v
            next_cache = (k, v)
        else:
            prev_k, prev_v = cache
            k = torch.cat([prev_k, next_k], dim=2)
            v = torch.cat([prev_v, next_v], dim=2)
            next_cache = (k, v)

        # Expand K and V to match number of heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Apply attention
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_head**0.5, dim=-1)

        context = (attn_weights @ v).transpose(1, 2).reshape(B, S, self.d_out)

        return self.out_proj(context), next_cache


class GQATransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttn(
            d_in=config.d_model,
            num_attn_heads=config.num_attn_heads,
            d_head=config.d_head,
            num_attn_kv_groups=config.num_attn_kv_groups,
            apply_qk_norm=config.apply_qk_norm,
            dtype=config.dtype,
        )

        self.mlp = SwiGLU_MLP(
            d_model=config.d_model, d_hidden=config.d_hidden, dtype=config.dtype
        )
        self.rms_norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.rms_norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any]:
        skip = x

        x = self.rms_norm1(x)
        x, next_cache = self.self_attn(
            x=x, cos=cos, sin=sin, mask=mask, start_pos=start_pos, cache=cache
        )

        x += skip
        skip = x

        x = self.rms_norm2(x)
        x = self.mlp(x)

        x += skip

        return x, next_cache
