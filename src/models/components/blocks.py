import torch
import torch.nn as nn

import math


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t):
        # First layer
        h = self.conv1(self.act(self.norm1(x)))

        # Add time embedding
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb

        # Second layer
        h = self.conv2(self.act(self.norm2(h)))

        # Residual connection + spatial transform
        return self.transform(h + self.shortcut(x))


class EnergyResBlock(nn.Module):
    """Time-independent residual block (ResBlock without the time embedding)."""

    def __init__(self, in_ch, out_ch, resample=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        if resample == "down":
            self.resample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        elif resample == "up":
            self.resample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return self.resample(h + self.shortcut(x))


class FourierFeatures(nn.Module):
    """
    Projects low-dimensional coordinates into a high-dimensional
    frequency space to overcome neural network spectral bias.
    """

    def __init__(self, in_features=2, mapping_size=64, scale=1.0):
        super().__init__()
        # A static random projection matrix (Gaussian)
        B = torch.randn((in_features, mapping_size)) * scale
        # Register as buffer so it moves to device with the model but isn't updated by the optimizer
        self.register_buffer("B", B)

    def forward(self, x):
        # Project and multiply by 2*pi
        projected = x @ self.B * 2.0 * math.pi
        # Return concatenated sines and cosines
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
