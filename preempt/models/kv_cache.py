from typing import Literal

import torch


class KVCache:
    cache: list[torch.Tensor | None]
    device: Literal["cpu", "cuda", "mps"]

    def __init__(
        self, num_blocks: int, device: Literal["cpu", "cuda", "mps"] = "mps"
    ) -> None:
        self.cache = [None] * num_blocks
        self.device = device

    def get(self, block_idx: int) -> torch.Tensor | None:
        value = self.cache[block_idx]
        if isinstance(value, torch.Tensor):
            value = value.to(self.device)
        return value

    def update(self, block_idx: int, value: torch.Tensor) -> None:
        self.cache[block_idx] = value

    def get_all(self) -> list[torch.Tensor | None]:
        return self.cache

    def reset(self) -> None:
        for i in range(len(self.cache)):
            self.cache[i] = None
