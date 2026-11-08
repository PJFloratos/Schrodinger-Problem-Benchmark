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
            dev = torch.device(device)
            try:
                return torch.Generator(device=dev).manual_seed(seed_input)
            except RuntimeError as e:
                # Graceful fallback for local machines with broken/missing CUDA drivers
                if dev.type == "cuda":
                    return torch.Generator(device="cpu").manual_seed(seed_input)
                raise e

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
        elif self.model_type == "scone":
            out = self._generate_scone(
                model=model,
                shape=shape,
                x_init=x_init,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                verbose=verbose,
                amp_dtype=amp_dtype,
                reference=model.reference,
                drop_last_noise=drop_last_noise,
                desc=desc,
                g_init=g_init,
                g_step=g_step,
                **solver_kwargs,
            )

        elif self.model_type == "enot":
            out = self._generate_enot(
                model=model,
                shape=shape,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                verbose=verbose,
                desc=desc,
                g_init=g_init,
                g_step=g_step,
            )

        elif self.model_type == "fbsde":
            out = self._generate_fbsde(
                model=model,
                shape=shape,
                n_samples=n_samples,
                device=device,
                batch_size=batch_size,
                desc=desc,
                g_init=g_init,
                g_step=g_step,
                **solver_kwargs,
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
            if not is_forward and self.model_type in ["sde", "ipf", "imf", "sf2m"]:
                x = x * sigma

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

        # Safely extract sigma (fallback to 1.0)
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
                if x_init is not None
                else torch.randn(n, *shape, device=device, generator=g_init) * sigma
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
        sigma = kwargs.get("sigma", 1.0)
        gamma = kwargs.get("gamma", 1.0)
        eta_max = kwargs.get("eta_max_infer", None)
        eta_min = kwargs.get("eta_min_infer", None)
        spatial_clip = kwargs.get("spatial_clip", None)
        ebm_steps = kwargs.get("ebm_steps", 1000)

        # [OU Addition]: Calculate the analytical moments
        alpha = math.exp(-gamma)
        beta_sq = (sigma**2 / (2 * gamma)) * (1 - math.exp(-2 * gamma))
        beta = math.sqrt(beta_sq)

        # ebm_steps = max(ebm_steps, required_steps)
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
                else torch.randn(n, *shape, device=device, generator=g_init) * sigma
            )
            y = x + sigma * torch.randn(
                x.shape, device=device, dtype=x.dtype, generator=g_init
            )
            # # [OU Addition]: Initialize from the OU conditional prior instead of SBM
            # y = alpha * x + beta * torch.randn(
            #     x.shape, device=device, dtype=x.dtype, generator=g_init
            # )

            for i in range(ebm_steps):
                progress = i / max(1, ebm_steps - 1)
                current_eta = eta_max * (eta_min / eta_max) ** progress
                drop_noise = drop_last_noise and (i == ebm_steps - 1)

                y.requires_grad_(True)
                with torch.enable_grad():
                    psi = model(y)
                    grad_psi = torch.autograd.grad(psi.sum(), y)[0]

                drift = grad_psi + (x - y) / (sigma**2)
                # # [OU Addition]: Drift anchors to (alpha * x) and scales by 1/beta^2
                # drift = grad_psi + (alpha * x - y) / beta_sq
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

    def _generate_scone(
        self,
        model,
        shape,
        x_init,
        n_samples,
        device,
        batch_size,
        drop_last_noise,
        verbose,
        reference,
        amp_dtype,
        desc,
        g_init,
        g_step,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Strategy: Joint Unadjusted Langevin Algorithm (SCONE Evaluation)"""
        from src.core.dynamics import ConditionalVectorField

        eta_max = kwargs.get("eta_max_infer", 5e-3)
        eta_min = kwargs.get("eta_min_infer", 1e-4)
        spatial_clip = kwargs.get("spatial_clip", 10.0)
        ebm_steps = kwargs.get("ebm_steps", 1000)

        # In SCONE, sigma is explicitly bundled inside the model wrapper
        sigma = getattr(model, "sigma", 1.0)

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

            # X is the fixed latent prior. It MUST NOT move during generation.
            if x_init is not None:
                x = x_init[start : start + n].clone().to(device)
            else:
                x = torch.randn(n, *shape, device=device, generator=g_init) * sigma

            # Initialize Y from the Reference Measure R_01
            y = x + sigma * torch.randn(
                x.shape, device=device, dtype=x.dtype, generator=g_init
            )

            # noise_x = torch.empty_like(x)
            noise_y = torch.empty_like(y)

            for i in range(ebm_steps):
                progress = i / max(1, ebm_steps - 1)
                current_eta = eta_max * (eta_min / eta_max) ** progress
                drop_noise = drop_last_noise and (i == ebm_steps - 1)

                y = y.detach().requires_grad_(True)
                with torch.enable_grad():
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=self.use_amp
                    ):
                        psi = model.psi(y)
                    grad_psi = torch.autograd.grad(psi.sum(), y, create_graph=False)[0]

                if spatial_clip > 0:
                    grad_psi = grad_psi.clamp(-spatial_clip, spatial_clip)

                if reference == "brownian":
                    _, ref_y = ConditionalVectorField.get_reference_scores(
                        x, y, sigma, reference
                    )
                else:
                    raise NotImplementedError(
                        f"Evaluation for {reference} requires the score network."
                    )

                drift_y = grad_psi + ref_y
                y = y.detach() + 0.5 * current_eta * drift_y

                if not drop_noise:
                    noise_y.normal_(generator=g_step)
                    y += math.sqrt(current_eta) * noise_y

                if pbar:
                    pbar.update(1)

            # Forward direction translates Prior to Data. So we return Y.
            out.append(y.detach())

        if pbar:
            pbar.close()

        return out

    def _generate_fbsde(
        self,
        model,
        shape,
        n_samples,
        device,
        batch_size,
        desc,
        g_init,
        g_step,
        r_snr=0.05,
        n_corrector=0,
        t0=1e-4,
    ) -> List[torch.Tensor]:
        """Strategy: Predictor-Corrector sampling for SB-FBSDE."""
        sigma = getattr(model, "sigma", 1.0)

        u = torch.linspace(0, 1, self.steps, device=device)
        ts = torch.flip(t0 + (1.0 - t0) * u**2, dims=[0])  # descending
        steps_down = ts[:-1] - ts[1:]

        out = []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            x = torch.randn(n, *shape, device=device, generator=g_init) * sigma

            for idx, t_val in enumerate(ts):  # all `steps` points, incl. t0
                t_tensor = torch.full((n, 1), t_val.item(), device=device)
                z_b = model.b(x, t_tensor)
                if idx == len(ts) - 1:
                    x = x + sigma * z_b * t0  # t0 -> 0, noise-free
                    break
                hk = steps_down[idx].item()
                noise = torch.randn(
                    x.shape, device=device, dtype=x.dtype, generator=g_step
                )
                x = x + sigma * z_b * hk + sigma * math.sqrt(hk) * noise

                # CORRECTOR STEP (only on intermediate steps)
                next_t = torch.full((n, 1), ts[idx + 1].item(), device=device)
                for _ in range(n_corrector):
                    z = model.f(x, next_t) + model.b(x, next_t)
                    noise_corr = torch.randn(
                        x.shape, device=device, dtype=x.dtype, generator=g_step
                    )
                    z_norm = z.reshape(n, -1).norm(dim=1, keepdim=True)
                    noise_norm = noise_corr.reshape(n, -1).norm(dim=1, keepdim=True)
                    eps = 2.0 * ((r_snr * noise_norm) / (z_norm + 1e-8)) ** 2
                    eps = eps.view(n, *([1] * (x.ndim - 1)))

                    x = x + sigma * eps * z + sigma * torch.sqrt(2.0 * eps) * noise_corr

            out.append(x)

        return out

    # =========================================================================
    # ENOT Specific Method
    # =========================================================================
    def rollout_enot_training(
        self, model, x0, epsilon, n_last_steps_without_noise=0, gen=None
    ):
        B, N = x0.shape[0], self.steps
        h = 1.0 / N
        dev = x0.device
        ts = torch.arange(N, device=dev, dtype=x0.dtype) * h

        # one RNG call, uses the dedicated generator (for big images, draw per-step instead)
        noise = torch.randn((N, *x0.shape), device=dev, dtype=x0.dtype, generator=gen)
        if n_last_steps_without_noise:
            noise[N - n_last_steps_without_noise :].zero_()
        noise.mul_(math.sqrt(epsilon * h))

        on_cuda = dev.type == "cuda"
        amp_dtype = torch.float16 if on_cuda else torch.bfloat16

        x, energy = x0, x0.new_zeros(B)
        for i in range(N):
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=self.use_amp):
                drift = model(x, ts[i].expand(B, 1))
            drift = drift.float()
            energy = energy + drift.flatten(1).pow(2).sum(1)
            x = x + h * drift + noise[i]
        return x, energy * h

    def _generate_enot(
        self,
        model,
        shape,
        n_samples,
        device,
        batch_size,
        verbose,
        desc,
        g_init,
        g_step,
    ) -> List[torch.Tensor]:
        """Strategy: Strict ENOT forward physics rollout matching training."""
        sigma = getattr(model, "sigma", 1.0)
        epsilon = sigma**2

        pbar = (
            tqdm(
                total=math.ceil(n_samples / batch_size) * self.steps,
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
            x0 = torch.randn(n, *shape, device=device, generator=g_init) * sigma

            # Use the exact training physics with no noise at the end
            x_gen, _ = self.rollout_enot_training(
                model=model,
                x0=x0,
                epsilon=epsilon,
                n_last_steps_without_noise=1,
                gen=g_step,
            )
            out.append(x_gen)

            # Update pbar by the number of integration steps executed per batch
            if pbar:
                pbar.update(self.steps)

        if pbar:
            pbar.close()

        return out
