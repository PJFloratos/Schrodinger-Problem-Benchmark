import torch
import torch.nn as nn
import math

from tqdm import tqdm
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
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


class SimpleUNet(nn.Module):
    def __init__(self, channels=1, image_size=28, model_type="sde"):
        super().__init__()
        self.model_type = model_type
        self.channels = channels
        self.image_size = image_size
        time_dim = 64

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.conv0 = nn.Conv2d(channels, 32, 3, padding=1)

        # Down blocks
        self.down1 = ResBlock(32, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)

        # Up blocks
        self.up1 = ResBlock(128, 64, time_dim, up=True)
        self.up2 = ResBlock(128, 32, time_dim, up=True)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(64, channels, 3, padding=1)

        # Initialize output layer to zero
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)

        x = self.up1(x2, t)
        x = self.up2(torch.cat([x, x1], dim=1), t)

        # Apply the new output projection
        x = self.out(self.out_act(self.out_norm(torch.cat([x, x0], dim=1))))

        return x
