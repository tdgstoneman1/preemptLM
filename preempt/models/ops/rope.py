import torch

__all__ = ["compute_rope_params", "apply_rope"]


def compute_rope_params(
    d_head: int,
    theta_base: int | float = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if d_head % 2 != 0:
        raise ValueError(f"Attention head dimension must be even, but got `{d_head=}`")

    # Get inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, d_head, 2, dtype=dtype)[: (d_head // 2)].float() / d_head)
    )
    # Get position indices
    positions = torch.arange(context_length, dtype=dtype)
    # Compute angles, -> (context_length, d_head // 2)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    # Expand angles to match the d_head -> (context_length, d_head)
    angles = torch.cat([angles, angles], dim=1)
    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    # x: (batch_size, num_heads, seq_len, d_head)
    _, _, seq_len, d_head = x.shape
    assert d_head % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : d_head // 2]  # First half
    x2 = x[..., d_head // 2 :]  # Second half

    # Adjust sin and cos shapes -> (1, 1, seq_len, d_head)
    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply rotary transform
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    x_rotated = x_rotated.to(dtype=x.dtype)

    return x_rotated
