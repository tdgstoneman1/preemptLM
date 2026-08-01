import torch
import torch.nn as nn


def get_unique_module_names(state_dict: dict[str, torch.Tensor]) -> set[str]:
    keys: list[str] = []
    for k in list(state_dict.keys()):
        keys.extend(
            list(
                filter(
                    lambda k: not k.isdigit() and k not in ("weight", "bias"),
                    k.split("."),
                )
            )
        )
    return set(keys)


@torch.no_grad
def assign_weights(
    source: torch.Tensor,
    target: torch.Tensor | nn.Parameter,
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: {source.shape=}, {target.shape=}")

    if isinstance(target, torch.Tensor):
        source.copy_(target)
    else:
        source.copy_(torch.as_tensor(target, dtype=source.dtype, device=source.device))

    return source
