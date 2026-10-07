import torch
import math
from tqdm import tqdm
from typing import Optional, Tuple, Union


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

    @staticmethod
    def euler_maruyama_step(
        x: torch.Tensor,
        drift: torch.Tensor,
        h: float,
        noise: Optional[torch.Tensor] = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Computes a single forward Euler-Maruyama SDE or ODE step"""

        if noise is not None:
            return x + h * drift + sigma * math.sqrt(h) * noise

        return x + h * drift

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...] = None,
        x_init: Optional[torch.Tensor] = None,
        n_samples: int = 64,
        device: str = "cpu",
        batch_size: int = 512,
        drop_last_noise: bool = True,
        verbose: bool = True,
        init_seed: Optional[Union[int, torch.Generator]] = None,
        step_seed: Optional[Union[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """
        Integrates the vector field.
        - If `x_init` is provided, simulates the forward process (Data -> Prior).
        - If `shape` is provided, simulates the backward process (Prior -> Data).
        """
        # --- Setup based on direction ---
        is_forward = x_init is not None
        if is_forward:
            device = x_init.device
            n_samples = x_init.shape[0]
            is_channels_last = x_init.is_contiguous(memory_format=torch.channels_last)
            desc = "    Forward Sim"
        else:
            if shape is None:
                raise ValueError("Must provide either `x_init` or `shape`.")
            device = torch.device(device)
            is_channels_last = False
            desc = "    Generating Samples"

        if drop_last_noise is None:
            drop_last_noise = not is_forward

        on_cuda = device.type == "cuda"
        self.use_amp = self.use_amp and on_cuda
        amp_dtype = torch.float16 if on_cuda else torch.bfloat16

        if isinstance(init_seed, torch.Generator):
            g_init = init_seed
        else:
            g_init = (
                torch.Generator(device=device).manual_seed(init_seed)
                if init_seed is not None
                else None
            )

        if isinstance(step_seed, torch.Generator):
            g_step = step_seed
        else:
            g_step = (
                torch.Generator(device=device).manual_seed(step_seed)
                if step_seed is not None
                else None
            )

        # if on_cuda:
        #     torch.backends.cudnn.benchmark = True

        h = 1.0 / self.steps
        integration_steps = int(self.steps * self.t_end)

        n_chunks = math.ceil(n_samples / batch_size)
        pbar = None
        if verbose:
            pbar = tqdm(
                total=n_chunks * integration_steps, ascii=True, desc=desc, leave=False
            )

        sigma = getattr(model, "sigma", 1.0)

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)

            # Initial State
            if is_forward:
                x = x_init[start : start + n].clone()
            else:
                # The shape tuple allows it to work for both (C, H, W) and (D,)
                x = torch.randn(n, *shape, device=device, generator=g_init)

            t = torch.empty(n, 1, device=device)

            for i in range(integration_steps):
                # The wrapper inherently handles time flipping internally if reverse simulation is requested.
                t.fill_(i * h)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=self.use_amp
                ):
                    drift = model(x, t)

                # Integrate in the state's precision, whatever autocast produced.
                drift = drift.to(x.dtype)

                if self.model_type in ["minibatch", "flow_m"]:
                    x = self.euler_maruyama_step(x, drift, h)

                elif self.model_type in ["sde", "sf2m", "imf", "ipf"]:
                    # Backward chain drops SDE noise on the final step; Forward chain does not.
                    drop_noise = (not is_forward) and (i == integration_steps - 1)

                    if drop_noise:
                        noise = torch.zeros_like(x)
                    else:
                        noise = torch.randn(
                            x.shape, device=device, dtype=x.dtype, generator=g_step
                        )

                        if is_channels_last:
                            noise = noise.contiguous(memory_format=torch.channels_last)

                    x = self.euler_maruyama_step(x, drift, h, noise, sigma)

                if verbose:
                    pbar.update(1)

            out.append(x)

        if verbose:
            pbar.close()

        res = torch.cat(out)
        if is_channels_last:
            return res.contiguous(memory_format=torch.channels_last)

        return res.contiguous()
