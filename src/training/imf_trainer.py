from src.training import BaseTrainer
from src.utils import EMAHelper, save_model, text_logger

import torch
from torch import nn
from torch.utils.data import DataLoader

import math
import time
from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple, Union


class IMFTrainer(BaseTrainer):
    """
    What differs from IPF: the cache stores ENDPOINT PAIRS (x0, x1) only, not whole trajectories.
    Every gradient step draws a fresh t ~ U[eps, 1-eps] and fresh noise, and regresses onto
    the Brownian-bridge targets
        x_t   = (1 - t) x0 + t x1 + sigma sqrt(t (1 - t)) z
        f-tgt = (x1 - x_t) / (1 - t)          (= x1 - x0 - sigma sqrt(t / (1 - t)) z)
        b-tgt = (x0 - x_t) / t                (= -(x1 - x0) - sigma sqrt((1 - t) / t) z)

    One IMF iteration n:
        phase B: train b on couplings   (x0 = data, x1 = forward-chain sample)  [n = 0: first coupling]
        phase F: train f on couplings   (x1 = prior, x0 = backward-chain sample)
    first_coupling (used at n = 0 only):
        "ind": (data, N(0, I)) independent. Used for BOTH b and f at n = 0.
        "ref": (data, data + sigma * eps), the Brownian reference. Used for b only (like IPF's
               first step); f is then trained on b's simulation.
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
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = 10,  # cache holds num_cache_batches * batch_size PAIRS
        refresh_every: int = 500,  # regenerate the pairs every N gradient steps
        sigma: float = 1.0,  # Brownian reference volatility (IPF is hard-wired to 1)
        eps: float = 1e-3,  # t ~ U[eps, 1 - eps]; targets blow up like 1/t at the ends
        first_coupling: str = "ind",  # in ["ind", "ref"]
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
    ):
        super().__init__(device=device, metric_logger=metric_logger)

        if first_coupling not in ("ind", "ref"):
            raise ValueError(
                f"first_coupling must be 'ind' or 'ref', got {first_coupling!r}"
            )

        self.f_model = forward_model.to(device)
        self.b_model = backward_model.to(device)
        self.dataset = dataset
        self.f_opt = forward_opt
        self.b_opt = backward_opt
        self.device = device
        self.batch_size = batch_size
        self.sde_steps = sde_steps
        self.h = 1.0 / sde_steps
        self.num_cache_batches = num_cache_batches
        self.refresh_every = refresh_every
        self.sigma = sigma
        self.eps = eps
        self.first_coupling = first_coupling
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.data_shape = tuple(self.dataset[0].shape)

        # generate() reads model.sigma so that evaluation uses the same noise level as training
        self.f_model.sigma = sigma
        self.b_model.sigma = sigma

        self.dl = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self._data_iter = self._repeater(self.dl)

        # Probe batch for the BaseTrainer outer-loop diagnostics
        probe_batch = next(iter(self.dl))
        if isinstance(probe_batch, (list, tuple)):
            probe_batch = probe_batch[0]
        self.fixed_probe_batch = probe_batch.to(self.device)

        self.ema_f = EMAHelper(mu=ema_mu, device=self.device)
        self.ema_f.register(self.f_model)
        self.ema_b = EMAHelper(mu=ema_mu, device=self.device)
        self.ema_b.register(self.b_model)

    ############ UTILS

    @staticmethod
    def _repeater(dataloader):
        """Infinite generator to continuously yield batches."""
        while True:
            for batch in dataloader:
                yield batch

    def _next_data(self) -> torch.Tensor:
        batch = next(self._data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.to(self.device)

    ############ COUPLING GENERATION

    @torch.no_grad()
    def _simulate(self, sampler: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Euler-Maruyama simulation of `sampler`'s own chain (own time t_k = k h), starting at x.
        Noise is injected at EVERY step (as in the official DSBM code when producing couplings);
        dropping the last noise is only correct when producing final samples (see generate()).
        """
        for i in range(self.sde_steps):
            t = torch.full((x.shape[0], 1), i * self.h, device=self.device)
            x = (
                x
                + self.h * sampler(x, t)
                + self.sigma * math.sqrt(self.h) * torch.randn_like(x)
            )
        self.total_nfes += x.shape[0] * self.sde_steps
        return x

    @torch.no_grad()
    def _make_pairs(
        self, phase: str, imf_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Endpoint couplings (X0, X1) used to train the network of `phase` ("b" or "f").
          phase "b": x0 = data,  x1 = forward chain (EMA of f) simulated from x0
          phase "f": x1 = prior, x0 = backward chain (EMA of b) simulated from x1
        """
        use_first = imf_iter == 0 and (phase == "b" or self.first_coupling == "ind")

        if not use_first:
            src_model, src_ema = (
                (self.f_model, self.ema_f)
                if phase == "b"
                else (self.b_model, self.ema_b)
            )
            sampler = src_ema.ema_copy(src_model)
            sampler.eval()

        X0, X1 = [], []
        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Building {phase.upper()} couplings",
        ):
            if use_first:
                x0 = self._next_data()
                if self.first_coupling == "ref":
                    x1 = x0 + self.sigma * torch.randn_like(x0)
                else:
                    x1 = torch.randn_like(x0)
            elif phase == "b":
                x0 = self._next_data()
                x1 = self._simulate(sampler, x0)
            else:
                x1 = torch.randn(self.batch_size, *self.data_shape, device=self.device)
                x0 = self._simulate(sampler, x1)
            X0.append(x0)
            X1.append(x1)

        return torch.cat(X0), torch.cat(X1)

    ############ REGRESSION TARGETS

    def _bridge_batch(
        self, x0: torch.Tensor, x1: torch.Tensor, phase: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Brownian-bridge sample. Returns (x_t, time fed to the network, regression target)."""
        B = x0.shape[0]
        t = torch.rand(B, 1, device=self.device) * (1.0 - 2.0 * self.eps) + self.eps
        te = t.view(B, *([1] * (x0.ndim - 1)))  # broadcastable over (C, H, W) as well
        z = torch.randn_like(x0)

        x_t = (1.0 - te) * x0 + te * x1 + self.sigma * torch.sqrt(te * (1.0 - te)) * z

        if phase == "f":
            target = (x1 - x0) - self.sigma * torch.sqrt(te / (1.0 - te)) * z
            t_in = t
            # Forward scaling factor
            weight = 1.0 / (1.0 + (self.sigma**2 * te) / (1.0 - te))
        else:
            target = -(x1 - x0) - self.sigma * torch.sqrt((1.0 - te) / te) * z
            t_in = 1.0 - t  # b consumes its own chain time s = 1 - t
            # Backward scaling factor
            weight = 1.0 / (1.0 + (self.sigma**2 * (1.0 - te)) / te)
        return x_t, t_in, target, weight

    @torch.no_grad()
    def _probe_loss(
        self, model: nn.Module, X0: torch.Tensor, X1: torch.Tensor, phase: str
    ) -> float:
        """Low-noise loss estimate on a larger batch (used for the cache-staleness diagnostic)."""
        was_training = model.training
        model.eval()
        n = min(4 * self.batch_size, X0.shape[0])
        idx = torch.randint(0, X0.shape[0], (n,), device=self.device)

        x_t, t_in, target, weight = self._bridge_batch(X0[idx], X1[idx], phase)

        # Use functional MSE with no reduction to apply the per-sample weighting
        pred = model(x_t, t_in)
        raw_loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        loss = (raw_loss * weight).mean().item()

        model.train(was_training)

        return loss

    ############ TRAINING

    def _train_phase(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        phase: str,
        num_iter: int,
        imf_iter: int,
    ) -> float:
        model.train()
        base_lrs = [g["lr"] for g in opt.param_groups]
        X0, X1 = self._make_pairs(phase, imf_iter)

        if phase == "f":
            log_phase, desc = "forward", "Training Forward Model"
        else:
            log_phase, desc = "backward", "Training Backward Model"

        total_loss = 0.0
        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                # Same model, old vs. new couplings: how much did the data distribution move?
                pre_loss = self._probe_loss(model, X0, X1, phase)
                X0, X1 = self._make_pairs(phase, imf_iter)
                post_loss = self._probe_loss(model, X0, X1, phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            idx = torch.randint(0, X0.shape[0], (self.batch_size,), device=self.device)
            x_t, t_in, target, weight = self._bridge_batch(X0[idx], X1[idx], phase)

            opt.zero_grad(set_to_none=True)

            # Use functional MSE with no reduction to apply the per-sample weighting
            pred = model(x_t, t_in)
            raw_loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
            loss = (raw_loss * weight).mean()

            loss.backward()

            # --- BaseTrainer Metric Hook (pre-clip grad norm) ---
            self.log_inner_step(model, loss, opt, phase=log_phase, ipf_iter=imf_iter)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()
            ema_helper.update(model)
            total_loss += loss.item()

        for g, lr0 in zip(opt.param_groups, base_lrs):  # restore for the next phase
            g["lr"] = lr0

        return total_loss / num_iter

    def fit(
        self,
        imf_iterations: int,
        inner_iterations: int = 5000,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_callback: Optional[Callable] = None,
    ):
        for n in range(imf_iterations):
            IMFTrainer.logger.debug(f"\n--- IMF Iteration {n+1}/{imf_iterations} ---")

            # ==========================================
            # Phase 1: Train Backward Model (prior -> data)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            b_loss = self._train_phase(
                self.b_model, self.b_opt, self.ema_b, "b", inner_iterations, n
            )
            b_metrics = {
                "loss": b_loss,
                "phase_time_sec": time.time() - phase_start,
                "train_nfes": self.total_nfes
                - nfes_before,  # every cache refresh counted
            }
            if eval_callback:
                b_metrics.update(eval_callback(self.b_model, direction="b"))
            self.log_phase_end("backward", n, b_metrics)

            # ==========================================
            # Phase 2: Train Forward Model (data -> prior)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            f_loss = self._train_phase(
                self.f_model, self.f_opt, self.ema_f, "f", inner_iterations, n
            )
            f_metrics = {
                "loss": f_loss,
                "phase_time_sec": time.time() - phase_start,
                "train_nfes": self.total_nfes - nfes_before,
            }
            if eval_callback:
                f_metrics.update(eval_callback(self.f_model, direction="f"))
            self.log_phase_end("forward", n, f_metrics)

            self.logger.info(
                f"Iteration {n+1}/{imf_iterations} | "
                f"B-Loss: {b_loss:.4f} | F-Loss: {f_loss:.4f} | "
                f"MMD: {b_metrics.get('eval_MMD', float('nan')):.6f} | "
                f"NFEs: {self.total_nfes}"
            )

            # --- Outer-Loop Diagnostics (same hooks as IPF) ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving (EMA weights) ---
            if save_per and save_path and ((n + 1) % save_per == 0):
                temp_b_model = self.ema_b.ema_copy(self.b_model)
                save_model(
                    temp_b_model,
                    f"{save_path}/{self.b_model.__class__.__name__}_backward_checkpoint_{n+1}.pth",
                )

        # Load smoothed weights into the models used for evaluation
        self.ema_f.ema(self.f_model)
        self.ema_b.ema(self.b_model)

        self.log_compute_cost([self.f_model, self.b_model])

        if save_path:
            save_model(
                self.b_model,
                f"{save_path}/{self.b_model.__class__.__name__}_backward_final.pth",
            )
            save_model(
                self.f_model,
                f"{save_path}/{self.b_model.__class__.__name__}_forward_final.pth",
            )
