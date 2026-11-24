import torch


def randn_like_with_gen(
    ref: torch.Tensor,
    gen: torch.Generator,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    """Deterministic randn_like supporting explicit torch.Generator and memory_format."""
    return torch.randn(
        ref.shape, device=ref.device, dtype=ref.dtype, generator=gen
    ).contiguous(memory_format=memory_format)
