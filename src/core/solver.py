import torch

import math
from tqdm import tqdm
from typing import Optional, Tuple, Union, List


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

    # =========================================================================
    # UTILITIES
    # =========================================================================
    @staticmethod
    def _resolve_generator(
        seed_input: Optional[Union[int, torch.Generator]],
        device: Union[str, torch.device],
    ) -> Optional[torch.Generator]:
        if isinstance(seed_input, torch.Generator):
            return seed_input
        if seed_input is not None:
            return torch.Generator(device=device).manual_seed(seed_input)
        return None

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

    # =========================================================================
    # Generate
    # =========================================================================

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
        **solver_kwargs,
    ) -> torch.Tensor:
        """Integrates the vector field using the appropriate solver strategy."""

        # 1. Resolve Direction & Device
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

        # 2. Hardware Precision Settings
        on_cuda = device.type == "cuda"
        self.use_amp = self.use_amp and on_cuda
        amp_dtype = torch.float16 if on_cuda else torch.bfloat16

        # 3. Secure RNG Streams
        g_init = self._resolve_generator(init_seed, device)
        g_step = self._resolve_generator(step_seed, device)

        # 4. Route to the appropriate solver strategy
        if self.model_type == "ebm":
            out = self._generate_ebm(
                model=model,
                shape=shape,
                x_init=x_init,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                drop_last_noise=drop_last_noise,
                verbose=verbose,
                desc=desc,
                g_init=g_init,
                g_step=g_step,
                **solver_kwargs,
            )
        elif self.model_type == "lightsbm":
            out = self._generate_lightsbm(
                model=model,
                shape=shape,
                x_init=x_init,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                verbose=verbose,
                desc=desc,
                g_init=g_init,
            )
        else:
            out = self._generate_standard(
                model=model,
                shape=shape,
                x_init=x_init,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                drop_last_noise=drop_last_noise,
                is_forward=is_forward,
                is_channels_last=is_channels_last,
                amp_dtype=amp_dtype,
                verbose=verbose,
                desc=desc,
                g_init=g_init,
                g_step=g_step,
            )

        # 5. Output Formatting
        res = torch.cat(out)
        if is_channels_last:
            return res.contiguous(memory_format=torch.channels_last)
        return res.contiguous()

    # =========================================================================
    # INTERNAL SOLVER STRATEGIES
    # =========================================================================

    def _generate_standard(
        self,
        model,
        shape,
        x_init,
        n_samples,
        device,
        batch_size,
        drop_last_noise,
        is_forward,
        is_channels_last,
        amp_dtype,
        verbose,
        desc,
        g_init,
        g_step,
    ) -> List[torch.Tensor]:
        """Strategy: Standard SDE/ODE Euler-Maruyama Integration"""
        h = 1.0 / self.steps
        integration_steps = int(self.steps * self.t_end)
        sigma = getattr(model, "sigma", 1.0)

        pbar = (
            tqdm(
                total=math.ceil(n_samples / batch_size) * integration_steps,
                ascii=True,
                desc=desc,
                leave=False,
            )
            if verbose
            else None
        )

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = (
                x_init[start : start + n].clone()
                if is_forward
                else torch.randn(n, *shape, device=device, generator=g_init)
            )
            t = torch.empty(n, 1, device=device)

            for i in range(integration_steps):
                current_t = i * h

                # Align the time tensor with the model's training convention
                if self.model_type == "imf":
                    # IMF: t=0 is Data, t=1 is Prior
                    t_net = current_t if is_forward else 1.0 - current_t
                else:
                    # Standard: t=0 is Prior, t=1 is Data
                    t_net = 1.0 - current_t if is_forward else current_t

                # t.fill_(i * h)
                t.fill_(t_net)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=self.use_amp
                ):
                    drift = model(x, t)

                drift = drift.to(x.dtype)

                if self.model_type in ["minibatch", "flow_m"]:
                    x = self.euler_maruyama_step(x, drift, h)
                else:
                    drop_noise = (not is_forward) and (i == integration_steps - 1)
                    noise = (
                        torch.zeros_like(x)
                        if drop_noise
                        else torch.randn(
                            x.shape, device=device, dtype=x.dtype, generator=g_step
                        )
                    )

                    if not drop_noise and is_channels_last:
                        noise = noise.contiguous(memory_format=torch.channels_last)

                    x = self.euler_maruyama_step(x, drift, h, noise, sigma)

                if pbar:
                    pbar.update(1)

            out.append(x)

        if pbar:
            pbar.close()
        return out

    def _generate_lightsbm(
        self, model, shape, x_init, n_samples, device, batch_size, verbose, desc, g_init
    ) -> List[torch.Tensor]:
        """Strategy: Single-step data-driven generation"""
        integration_steps = int(self.steps * self.t_end)
        pbar = (
            tqdm(
                total=math.ceil(n_samples / batch_size) * integration_steps,
                ascii=True,
                desc=desc,
                leave=False,
            )
            if verbose
            else None
        )

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = (
                x_init[start : start + n].clone()
                if x_init is not None
                else torch.randn(n, *shape, device=device, generator=g_init)
            )

            x_1 = model.sample_x1(x)
            out.append(x_1)

            if pbar:
                pbar.update(integration_steps)

        if pbar:
            pbar.close()
        return out

    def _generate_ebm(
        self,
        model,
        shape,
        x_init,
        n_samples,
        device,
        batch_size,
        drop_last_noise,
        verbose,
        desc,
        g_init,
        g_step,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Strategy: Unadjusted Langevin Algorithm (ULA)"""
        epsilon = kwargs.get("epsilon", None)
        eta_max = kwargs.get("eta_max_infer", None)
        eta_min = kwargs.get("eta_min_infer", None)
        spatial_clip = kwargs.get("spatial_clip", None)
        ebm_steps = kwargs.get("ebm_steps", 1000)

        missing = [
            k
            for k, v in zip(
                ["epsilon", "eta_max_infer", "eta_min_infer", "spatial_clip"],
                [epsilon, eta_max, eta_min, spatial_clip],
            )
            if v is None
        ]
        if missing:
            raise AttributeError(
                f"EBM missing hyperparams: {missing}. Ensure they are passed via solver_kwargs."
            )

        target_horizon = 5.0 * epsilon
        if eta_max > eta_min:
            required_steps = int(
                target_horizon * math.log(eta_max / eta_min) / (eta_max - eta_min)
            )
        else:
            required_steps = int(target_horizon / eta_max)

        ebm_steps = max(ebm_steps, required_steps)
        pbar = (
            tqdm(
                total=math.ceil(n_samples / batch_size) * ebm_steps,
                ascii=True,
                desc=desc,
                leave=False,
            )
            if verbose
            else None
        )

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = (
                x_init[start : start + n].clone()
                if x_init is not None
                else torch.randn(n, *shape, device=device, generator=g_init)
            )
            y = x + math.sqrt(epsilon) * torch.randn(
                x.shape, device=device, dtype=x.dtype, generator=g_init
            )

            for i in range(ebm_steps):
                progress = i / max(1, ebm_steps - 1)
                current_eta = eta_max * (eta_min / eta_max) ** progress
                drop_noise = drop_last_noise and (i == ebm_steps - 1)

                y.requires_grad_(True)
                with torch.enable_grad():
                    psi = model(y)
                    grad_psi = torch.autograd.grad(psi.sum(), y)[0]

                if spatial_clip > 0:
                    grad_psi = grad_psi.clamp(-spatial_clip, spatial_clip)

                drift = grad_psi + (x - y) / epsilon
                y = y.detach() + 0.5 * current_eta * drift

                if not drop_noise:
                    y += math.sqrt(current_eta) * torch.randn(
                        y.shape, device=device, dtype=y.dtype, generator=g_step
                    )

                if pbar:
                    pbar.update(1)

            out.append(y)

        if pbar:
            pbar.close()
        return out
