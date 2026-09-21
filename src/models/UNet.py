import torch
import torch.nn as nn
import math

from tqdm import tqdm


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # We define conv1 normally without the hardcoded '2 * in_ch'
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions for broadcasting across feature maps
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    def __init__(self, model_type="sde"):
        super().__init__()
        self.model_type = model_type
        time_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)

        # Down blocks
        self.down1 = Block(32, 64, time_dim)
        self.down2 = Block(64, 128, time_dim)

        # Up blocks
        # up1 only takes x2 (128 channels) from the bottleneck
        self.up1 = Block(128, 64, time_dim, up=True)

        # up2 takes the concatenation of up1 output (64) + x1 skip connection (64) = 128
        self.up2 = Block(128, 32, time_dim, up=True)

        # out takes the concatenation of up2 output (32) + x0 skip connection (32) = 64
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)

        x = self.up1(x2, t)
        x = self.up2(torch.cat([x, x1], dim=1), t)
        x = self.out(torch.cat([x, x0], dim=1))

        return x

    @torch.no_grad()
    def generate(
        self,
        n_samples=64,
        steps=50,
        device="cpu",
        return_path=False,
        t_end=1.0,
        batch_size=512,
        amp=True,  # bf16/fp16 autocast for the network on CUDA; state stays fp32
    ):
        device = torch.device(device)
        on_cuda = device.type == "cuda"

        if on_cuda:
            torch.backends.cudnn.benchmark = True
            self.to(memory_format=torch.channels_last)

        h = 1.0 / steps
        integration_steps = int(steps * t_end)

        n_chunks = math.ceil(n_samples / batch_size)
        pbar = tqdm(
            total=n_chunks * integration_steps, ascii=True, desc="    Generating"
        )

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = torch.randn(n, 1, 28, 28, device=device)
            if on_cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            t = torch.empty(n, 1, device=device)  # reused every step

            for i in range(integration_steps):
                t.fill_(i * h)
                drift = self.forward(x, t)

                if self.model_type == "sde":
                    noise = (
                        torch.randn_like(x)
                        if i < integration_steps - 1
                        else torch.zeros_like(x)
                    )
                    x = x + h * drift + (h**0.5) * noise
                else:
                    x = x + h * drift
                pbar.update(1)

            out.append(x)

        pbar.close()
        return torch.cat(out).contiguous()
