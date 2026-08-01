from typing import Literal

import torch
import torch.nn as nn

import math


def variance_scaling_(
    tensor: torch.Tensor,
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
    distribution: Literal["truncated_normal", "normal", "uniform"] = "normal",
    generator: torch.Generator | None = None,
) -> None:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denom = fan_in

    elif mode == "fan_out":
        denom = fan_out

    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    else:
        raise ValueError(
            f"`mode` must be one of 'fan_in', 'fan_out', or 'fan_avg'; got '{mode}'"
        )

    variance = 1.0 / denom

    if distribution == "truncated_normal":
        nn.init.trunc_normal_(
            tensor, std=math.sqrt(variance) / 0.87962566103423978, generator=generator
        )
    elif distribution == "normal":
        nn.init.normal_(tensor, std=math.sqrt(variance), generator=generator)

    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        nn.init.uniform_(tensor, -bound, bound, generator=generator)

    else:
        raise ValueError(
            f"`distribution` must be one of 'truncated_normal', 'normal', or 'uniform'; "
            f"got '{distribution}'"
        )


def lecun_normal_(
    tensor: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    variance_scaling_(
        tensor, mode="fan_in", distribution="truncated_normal", generator=generator
    )
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return tensor.copy_(other)
    return tensor
