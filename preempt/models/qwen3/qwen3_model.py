from typing import Optional
from collections.abc import Generator

import torch
import torch.nn as nn

from .qwen3_config import Qwen3Config
from .qwen3_block import GQATransformerBlock, RMSNorm

from ..kv_cache import KVCache
from ..ops.rope import compute_rope_params


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, dtype=config.dtype
        )
        self.transformer_blocks = nn.ModuleList(
            [GQATransformerBlock(config) for _ in range(config.num_transformer_blocks)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.out_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False, dtype=config.dtype
        )

        if config.d_head is None:
            d_head = config.d_model // config.num_attn_heads
        else:
            d_head = config.d_head

        cos, sin = compute_rope_params(
            d_head=d_head,
            theta_base=config.rope_theta_base,
            context_length=config.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0  # Track current position in KV cache

        self.dtype = config.dtype
        self.num_blocks = config.num_transformer_blocks

    def forward(
        self, in_idx: torch.Tensor, cache: Optional[KVCache] = None
    ) -> torch.Tensor:
        x = self.token_embedding(in_idx)
        S = x.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + S

            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
            self.current_pos = pos_end
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        mask = mask[
            None, None, :, :
        ]  # (1, 1, S, S) to broadcast across batch and heads

        for i, block in enumerate(self.transformer_blocks):
            block_cache = cache.get(i) if cache else None
            x, new_block_cache = block(
                x, mask, self.cos, self.sin, start_pos=pos_start, cache=block_cache
            )
            if cache is not None:
                cache.update(i, new_block_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))

        return logits

    def reset_kv_cache(self) -> None:
        self.current_pos = 0


def qwen3_generate_text(
    model: Qwen3Model,
    tokens: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Generator[torch.Tensor]:
    model.eval()

    cache = KVCache(num_blocks=model.num_blocks)
    model.reset_kv_cache()

    with torch.no_grad():
        logits = model(tokens, cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            # Only pass the new token to model, cache handles history
            logits = model(next_token, cache=cache)


def qwen3_generate_text_naive(
    model: Qwen3Model,
    tokens: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Generator[torch.Tensor]:
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens, cache=None)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            tokens = torch.cat([tokens, next_token], dim=1)
