import torch
import math
from tqdm import tqdm
from typing import Optional, Tuple


class EulerSampler:
    """
    A generalized Euler/Euler-Maruyama integrator for generative models.
    """

    def __init__(
        self,
        model_type: str = "flow_m",
        steps: int = 50,
        t_end: float = 1.0,
        use_amp: bool = True,
    ):
        self.model_type = model_type
        self.steps = steps
        self.t_end = t_end
        self.use_amp = use_amp

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        n_samples: int = 64,
        device: str = "cpu",
        batch_size: int = 512,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        device = torch.device(device)
        on_cuda = device.type == "cuda"

        # Safely determine AMP variables
        self.use_amp = self.use_amp and on_cuda
        amp_dtype = torch.float16 if on_cuda else torch.bfloat16

        # Local generators: one for initial noise, one for SDE step noise
        if seed is None:
            g_init = g_step = None
        else:
            g_init = torch.Generator(device=device).manual_seed(2 * seed)
            g_step = torch.Generator(device=device).manual_seed(2 * seed + 1)

        # if on_cuda:
        #     torch.backends.cudnn.benchmark = True

        h = 1.0 / self.steps
        integration_steps = int(self.steps * self.t_end)

        n_chunks = math.ceil(n_samples / batch_size)
        pbar = tqdm(
            total=n_chunks * integration_steps,
            ascii=True,
            desc="    Generating Samples",
        )

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)

            # The shape tuple allows it to work for both (C, H, W) and (D,)
            x = torch.randn(n, *shape, device=device, generator=g_init)

            t = torch.empty(n, 1, device=device)

            for i in range(integration_steps):
                t.fill_(i * h)

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=self.use_amp
                ):
                    drift = model(x, t)

                if self.model_type in ["minibatch", "flow_m"]:
                    x = x + h * drift
                elif self.model_type == "sde":
                    noise = (
                        torch.randn(
                            x.shape, device=device, dtype=x.dtype, generator=g_step
                        )
                        if i < integration_steps - 1
                        else torch.zeros_like(x)
                    )
                    x = x + h * drift + (h**0.5) * noise
                pbar.update(1)

            out.append(x)

        pbar.close()
        return torch.cat(out).contiguous()
