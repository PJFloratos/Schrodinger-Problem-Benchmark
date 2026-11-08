from src.training.ipf_trainer import IPFTrainer
from src.core.dynamics import ConditionalVectorField
from src.utils.seed import derive_seed
from src.utils.log import text_logger

import torch
from torch import nn
import math
from tqdm import tqdm
from typing import Any, Callable, Union, Optional

import time


class FBSDETrainer(IPFTrainer):
    """
    Likelihood Training of SB-FBSDEs.
    Inherits orchestration from IPFTrainer, but overrides
    simulation and training loops to implement the divergence-based objective.

    Start at N(0, I)
    """

    logger = text_logger(__name__)

    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        dataset: torch.utils.data.Dataset,
        forward_opt: torch.optim.Optimizer,
        backward_opt: torch.optim.Optimizer,
        device: torch.device,
        metric_logger: Any,
        seed: int,
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = 10,
        refresh_every: int = 500,
        sigma: float = 1.0,
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,
        lr_final_ratio: float = 0.05,
        use_amp: bool = True,
        use_ema: bool = True,
        use_compile: bool = False,  # Default to False for Hutchinson double-autograd safety
    ):
        # Skips IPFTrainer.__init__ and calls BaseTrainer.__init__ directly
        super(IPFTrainer, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
            metric_logger=metric_logger,
            sigma=sigma,
            use_amp=use_amp,
        )

        self.model_type = "fbsde"

        self.f_opt = forward_opt
        self.b_opt = backward_opt
        self.sde_steps = sde_steps
        self.num_cache_batches = num_cache_batches
        self.refresh_every = refresh_every
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.use_ema = use_ema

        # DataLoader shuffling (Isolated SB-FBSDE context)
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(derive_seed(self.seed, "sbfbsde_shuffle"))

        # Cache-simulation noise
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(derive_seed(self.seed, "sbfbsde_cache_noise"))

        # Minibatch order inside each cache
        self.perm_gen = torch.Generator(device=self.device)
        self.perm_gen.manual_seed(derive_seed(self.seed, "sbfbsde_cache_perm"))

        self._init_cache_dataloader("sbfbsde_shuffle")

        # Cache-staleness probe
        self.probe_gen = torch.Generator(device=self.device)

        # Amp scalers for the two models
        self.f_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.b_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Model setup - explicitly passing use_compile down to prevent graph breaks
        self.f_model_base, self.f_model, self.f_ema, self.f_sampler = self._setup_model(
            model=forward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            create_sampler=True,
            use_compile=use_compile,
        )
        self.b_model_base, self.b_model, self.b_ema, self.b_sampler = self._setup_model(
            model=backward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            create_sampler=True,
            use_compile=use_compile,
        )

        # --- Time grid ----------------------------------------------------------------
        # Physical time runs from t0 (just after the data) to 1.0 (the prior).
        # t0 > 0 avoids evaluating the score exactly on the data, where it is singular
        # (the score of a Brownian-blurred point behaves like -1/sqrt(t)).
        self.t0 = 1e-4

        # Quadratic spacing: t_k = t0 + (1 - t0) * u_k**2 with u_k uniform in [0, 1].
        u = torch.linspace(0.0, 1.0, self.sde_steps, device=self.device)
        self.ts = self.t0 + (1.0 - self.t0) * u**2  # ascending, dense near 0

        # Bin widths. hs[k] = ts[k+1] - ts[k] is used in two ways:
        #   * as the Euler step size when moving forward from ts[k] to ts[k+1];
        #     moving backward from ts[k] to ts[k-1] uses hs[k-1];
        #   * as the quadrature weight of grid point k in the loss, so the sum over k
        #     approximates the continuous integral over t in the SB objective.
        # The last bin has no right neighbour, so it reuses the final gap. It is only used
        # as a loss weight at t = 1; no step is taken past t = 1.
        gaps = self.ts[1:] - self.ts[:-1]
        self.hs = torch.cat([gaps, gaps[-1:]])  # per-bin width (quadrature weight)

        # Mean bin width. It is not part of the dynamics, only a stand-in for the
        # per-sample weights in _probe_loss.
        self.h = self.hs.mean().item()

    @torch.no_grad()
    def _simulate_and_cache(
        self,
        source_model: nn.Module,
        src_ema,
        sampler: nn.Module,
        direction: str,
        ipf_iteration: int,
    ):
        use_reference = ipf_iteration == 1 and direction == "f"

        if not use_reference:
            if src_ema is not None:
                src_ema.copy_to(sampler)
            else:
                getattr(sampler, "_orig_mod", sampler).load_state_dict(
                    source_model.state_dict()
                )
            sampler.eval()

        total_samples = self.num_cache_batches * self.batch_size * self.sde_steps
        X_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        T_cache = torch.empty((total_samples, 1), device=self.device)
        Z_impt_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        H_cache = torch.empty((total_samples, 1), device=self.device)

        idx = 0
        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Simulating {direction.upper()} Cache",
        ):

            if direction == "f":
                x = self._next_data()
                # propagate x0 by a tiny step t0 first
                # so x is never raw, zero-noise data when computing divergence
                dw0 = self._randn_like(x, self.noise_gen)
                if use_reference:
                    z0 = torch.zeros_like(x)
                else:
                    t0_tensor = torch.full((x.shape[0], 1), self.t0, device=self.device)
                    z0 = sampler(x, t0_tensor)
                x = (
                    x
                    + (self.sigma * z0) * self.t0
                    + self.sigma * math.sqrt(self.t0) * dw0
                )
            else:
                x = (
                    torch.randn(
                        self.batch_size,
                        *self.data_shape,
                        device=self.device,
                        generator=self.noise_gen,
                    ).contiguous(memory_format=self.memory_format)
                    * self.sigma
                )

            b_size = x.shape[0]

            N = self.sde_steps
            order = range(N) if direction == "f" else range(N - 1, -1, -1)
            for k in order:
                t_val = self.ts[k].item()
                t_tensor = torch.full((b_size, 1), t_val, device=self.device)

                if use_reference:
                    z_impt = torch.zeros_like(x)
                else:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        z_impt = sampler(x, t_tensor)

                # Cache BEFORE stepping at the exact physical time t_tensor!
                X_cache[idx : idx + b_size] = x
                T_cache[idx : idx + b_size] = t_tensor
                Z_impt_cache[idx : idx + b_size] = z_impt
                H_cache[idx : idx + b_size] = self.hs[k]

                # SDE Step: x <- x + (sigma * z) * h + sigma * sqrt(h) * dw
                h_step = (
                    self.hs[k].item()
                    if direction == "f"
                    else (self.hs[k - 1].item() if k > 0 else 0.0)
                )
                dw = self._randn_like(x, self.noise_gen)
                x = (
                    x
                    + (self.sigma * z_impt) * h_step
                    + self.sigma * math.sqrt(h_step) * dw
                )

                idx += b_size

        self._probe_tensors = (
            X_cache[:idx],
            T_cache[:idx],
            Z_impt_cache[:idx],
            H_cache[:idx],
        )

        return self._cache_iterator(X_cache, T_cache, Z_impt_cache, H_cache)

    def _train_cache(
        self,
        target_model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper,
        make_cache: Callable,
        direction: str,
        num_iter: int,
        ipf_iter: int,
    ) -> float:
        """
        Overrides IPF training. SB-FBSDE requires requires_grad=True on the input
        state to compute the Hutchinson trace estimator for the divergence loss.
        """
        target_model.train()
        scaler = self.b_scaler if direction == "b" else self.f_scaler
        base_lrs = [g["lr"] for g in opt.param_groups]

        cache_iter = make_cache()
        phase = "forward" if direction == "f" else "backward"
        desc = f"Training {phase.capitalize()} Model"

        total_loss = torch.tensor(0.0, device=self.device)

        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                pre_loss = self._probe_loss(target_model, "fbsde", phase)
                cache_iter = make_cache()
                post_loss = self._probe_loss(target_model, "fbsde", phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            x_batch, t_batch, z_impt_batch, h_batch = next(cache_iter)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Explicitly disable autocast and force float32
            # to prevent FP16 subnormal zeroing during double autograd.
            with torch.autocast(device_type=self.device.type, enabled=False):
                # Detach and cast to float32
                x_fp32 = x_batch.detach().float().requires_grad_(True)
                t_fp32 = t_batch.float()
                z_impt_fp32 = z_impt_batch.float()
                h_fp32 = h_batch.float().view(-1)

                target_model.eval()
                z_opt = target_model(x_fp32, t_fp32)

                div_gz = ConditionalVectorField.compute_div_gz(
                    z=z_opt, x=x_fp32, g_t=self.sigma, gen=self.noise_gen
                )

                target_model.train()
                loss = ConditionalVectorField.compute_sbfbsde_loss(
                    z_opt=z_opt,
                    z_impt=z_impt_fp32,
                    div_gz=div_gz,
                    h=h_fp32,
                )

            self._optimize_step(
                target_model,
                opt,
                scaler,
                loss,
                ema_helper,
                log=True,
                phase=phase,
                epoch=ipf_iter,
            )

            total_loss += loss.detach().float()

        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0

        return (total_loss / num_iter).item()
