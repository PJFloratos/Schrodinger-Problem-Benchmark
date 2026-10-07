import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding for diffusion/flow models."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device

        # Scale continuous time [0,1] to [0, 1000] so frequencies resolve properly
        t = t * 1000.0

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
