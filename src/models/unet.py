from src.models.components import SinusoidalTimeEmbedding
from src.models.components import ResBlock, EnergyResBlock

import torch
import torch.nn as nn

import math
from tqdm import tqdm
from typing import Optional


class SimpleUNet(nn.Module):
    def __init__(self, channels=1, image_size=28, model_type="sde"):
        super().__init__()
        self.model_type = model_type
        self.channels = channels
        self.image_size = image_size
        # time_dim = 64
        time_dim = 256

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.conv0 = nn.Conv2d(channels, 64, 3, padding=1)

        # Down blocks
        self.down1 = ResBlock(64, 128, time_dim)
        self.down2 = ResBlock(128, 256, time_dim)

        # Up blocks
        self.up1 = ResBlock(256, 128, time_dim, up=True)
        self.up2 = ResBlock(256, 64, time_dim, up=True)

        self.out_norm = nn.GroupNorm(8, 128)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(128, channels, 3, padding=1)

        # Initialize output layer to zero
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # EBM Pooling Head
        if self.model_type == "ebm":
            self.ebm_pool = nn.AdaptiveAvgPool2d(1)
            self.ebm_head = nn.Linear(128, 1)

    def forward(self, x, t=None):
        # EBM is static; feed a dummy zero-tensor to satisfy the ResBlocks' time dependencies
        if t is None:
            t = torch.zeros(x.size(0), device=x.device)

        t = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)

        # Early exit for Energy-Based Model
        if self.model_type == "ebm":
            pooled = self.ebm_pool(x2).view(x2.size(0), -1)
            return self.ebm_head(pooled)

        x = self.up1(x2, t)
        x = self.up2(torch.cat([x, x1], dim=1), t)

        # Apply the new output projection
        x = self.out(self.out_act(self.out_norm(torch.cat([x, x0], dim=1))))

        return x


class EnergyUNet(nn.Module):
    """Scalar potential R^{C x H x W} -> R built as a U-Net.

    The U-Net produces a per-pixel energy map which is reduced to one scalar
    per image, so grad_x E has the same shape as x.
    """

    def __init__(
        self, channels=1, image_size=28, base=32, reduce="sum", model_type="ebm"
    ):
        super().__init__()
        assert image_size % 4 == 0, "image_size must be divisible by 4"
        assert reduce in ("sum", "mean")
        self.model_type = model_type
        self.channels = channels
        self.image_size = image_size
        self.reduce = reduce

        c1, c2 = base * 2, base * 4
        self.conv0 = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = EnergyResBlock(base, c1, "down")
        self.down2 = EnergyResBlock(c1, c2, "down")
        self.mid = EnergyResBlock(c2, c2)
        self.up1 = EnergyResBlock(c2, c1, "up")
        self.up2 = EnergyResBlock(c1 * 2, base, "up")

        self.out_norm = nn.GroupNorm(8, base * 2)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(base * 2, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t=None):
        # `t` is accepted only for API compatibility with the solvers and ignored.
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.mid(self.down2(x1))

        h = self.up1(x2)
        h = self.up2(torch.cat([h, x1], dim=1))
        e = self.out(self.out_act(self.out_norm(torch.cat([h, x0], dim=1))))

        # Reduce in fp32 so AMP doesn't overflow when summing H*W values.
        e = e.float().flatten(1)
        return (
            e.sum(1, keepdim=True) if self.reduce == "sum" else e.mean(1, keepdim=True)
        )
