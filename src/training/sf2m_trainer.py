from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.models.components import SF2MInferenceWrapper
from src.utils.seed import SeedOffsets
from src.utils import save_model, text_logger

import torch
from torch import nn
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment

import copy
import math
import time
from contextlib import contextmanager
from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple, Union, Iterator


class SF2MTrainer(BaseTrainer):
    """
    Simulation-Free Score and Flow Matching (SF2M) trainer.
    """

    logger = text_logger(__name__)

    def __init__(
        self,
        u_model: nn.Module,
        s_model: nn.Module,
        dataset: torch.utils.data.Dataset,
        u_opt: torch.optim.Optimizer,
        s_opt: torch.optim.Optimizer,
        device: torch.device,
        metric_logger: Any,
        seed: int,
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = None,
        ot_method: str = "minibatch",  # 'minibatch' | 'sinkhorn' | 'greedy'
        sigma: float = 1.0,
        eps: float = 1e-4,
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,
        lr_final_ratio: float = 0.05,
        use_amp: bool = True,
        use_ema: bool = True,
        sinkhorn_iters: int = 200,
    ):
        assert ot_method in ("minibatch", "sinkhorn", "greedy"), ot_method

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
            metric_logger=metric_logger,
            use_amp=use_amp,
        )

        self.u_opt = u_opt
        self.s_opt = s_opt
        self.sde_steps = sde_steps
        self.h = 1.0 / sde_steps
        self.ot_method = ot_method
        self.sigma = sigma
        self.eps = eps
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.use_ema = use_ema
        self.sinkhorn_iters = sinkhorn_iters

        if num_cache_batches is None:
            num_cache_batches = max(1, len(dataset) // batch_size)
        self.num_cache_batches = num_cache_batches

        # DataLoader shuffling
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(self.seed + SeedOffsets.SF2M_SHUFFLE)

        # Cache-simulation noise for forward/backward trajectory endpoints
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.seed + SeedOffsets.SF2M_CACHE_NOISE)

        # Minibatch order inside each cache
        self.perm_gen = torch.Generator(device=self.device)
        self.perm_gen.manual_seed(self.seed + SeedOffsets.SF2M_CACHE_PERM)

        # Fresh (t, z) at every gradient step for Brownian bridge targets
        self.bridge_gen = torch.Generator(device=self.device)
        self.bridge_gen.manual_seed(self.seed + SeedOffsets.SF2M_BRIDGE_NOISE)

        self._init_cache_dataloader(SeedOffsets.SF2M_SHUFFLE)

        # Precision & Compilation Settings
        self.u_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.s_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Models, EMA, and Compilation
        self.u_model_base, self.u_model, self.ema_u, self.u_sampler = self._setup_model(
            model=u_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=sigma,
            create_sampler=True,
        )
        self.s_model_base, self.s_model, self.ema_s, self.s_sampler = self._setup_model(
            model=s_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=sigma,
            create_sampler=True,
        )

    # ------------------------------------------------------------------
    # COUPLINGS
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _couple(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-pair a data batch x0 and a prior batch x1 with a minibatch (entropic) OT plan under the
        squared Euclidean cost. Returns (x0, x1) with row i of each forming one pair.
        """
        B = x0.shape[0]
        C = torch.cdist(x0.flatten(1), x1.flatten(1)).square()  # (B, B)

        if self.ot_method == "minibatch":
            # exact OT (Hungarian)
            _, col = linear_sum_assignment(C.cpu().numpy())
            col = torch.as_tensor(col, device=x0.device, dtype=torch.long)
            return x0, x1[col]

        if self.ot_method == "greedy":
            row, col = ConditionalVectorField.greedy_assignment_gpu(C)
            return x0[row], x1[col]

        # Entropic OT with eps = 2 sigma^2, in the LOG domain (exp(-C/reg) underflows to 0 for
        # high-dimensional data and produces NaNs).
        reg = 2.0 * self.sigma**2
        log_b = -math.log(B)
        f = torch.zeros(B, device=C.device, dtype=C.dtype)
        g = torch.zeros(B, device=C.device, dtype=C.dtype)
        for _ in range(self.sinkhorn_iters):
            f = -reg * torch.logsumexp((g[None, :] - C) / reg + log_b, dim=1)
            g = -reg * torch.logsumexp((f[:, None] - C) / reg + log_b, dim=0)
        # For every x0 draw its partner from the plan row P[i, :] / sum_j P[i, j]:
        # every x0 is used exactly once and the x0-marginal is preserved.
        probs = torch.softmax((g[None, :] - C) / reg, dim=1)
        col = torch.multinomial(probs, 1, generator=self.perm_gen).squeeze(1)
        return x0, x1[col]

    def _ot_iterator(self):
        """Infinite generator for online Optimal Transport (Iteration 1)."""
        while True:
            x1 = self._next_data()
            x0 = self._randn_like(x1, self.noise_gen)
            yield self._couple(x0, x1)

    # ------------------------------------------------------------------
    # SIMULATION AND CACHE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_cache(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Alg. 3, loops >= 2. Half of every cache batch is built from real data
        (x1 = data, x0_hat = backward-in-time SDE endpoint); the other half from fresh prior samples
        (x0 = prior, x1_hat = SDE endpoint at the data side).
        """
        B = self.batch_size
        n_total = self.num_cache_batches * B
        X0 = torch.empty(
            (n_total, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        X1 = torch.empty_like(X0)

        # Load the weights used for simulation (EMA if available)
        u_s = getattr(self.u_sampler, "_orig_mod", self.u_sampler)
        s_s = getattr(self.s_sampler, "_orig_mod", self.s_sampler)
        if self.use_ema:
            self.ema_u.copy_to(u_s)
            self.ema_s.copy_to(s_s)
        else:
            u_s.load_state_dict(self.u_model_base.state_dict())
            s_s.load_state_dict(self.s_model_base.state_dict())
        self.u_sampler.eval()
        self.s_sampler.eval()

        self.sampler_engine = EulerSampler(
            model_type="sf2m", steps=self.sde_steps, use_amp=self.use_amp
        )
        self.sampler_b = SF2MInferenceWrapper(
            self.u_sampler, self.s_sampler, self.sigma, direction="b", eps=self.eps
        )
        self.sampler_f = SF2MInferenceWrapper(
            self.u_sampler, self.s_sampler, self.sigma, direction="f", eps=self.eps
        )

        idx = 0
        for k in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc="Building SF2M Cache",
        ):
            x1_real = self._next_data()

            # Dynamically calculate sizes based on actual batch received
            b_size = x1_real.shape[0]
            half = b_size // 2

            # data -> prior
            x1_b = x1_real[:half]
            x0_b = self.sampler_engine.generate(
                model=self.sampler_f,
                x_init=x1_b,
                batch_size=x1_b.shape[0],
                step_seed=self.noise_gen,
                drop_last_noise=False,
                verbose=False,
            )

            # prior -> data
            x0_f = self._randn_like(x1_real[: b_size - half], self.noise_gen)
            x1_f = self.sampler_engine.generate(
                model=self.sampler_b,
                x_init=x0_f,
                batch_size=x0_f.shape[0],
                step_seed=self.noise_gen,
                drop_last_noise=False,
                verbose=False,
            )

            X0[idx : idx + b_size] = torch.cat([x0_b, x0_f], dim=0)
            X1[idx : idx + b_size] = torch.cat([x1_b, x1_f], dim=0)
            idx += b_size

        if not (torch.isfinite(X0).all() and torch.isfinite(X1).all()):
            raise RuntimeError(
                "Non-finite values in the SF2M cache: the simulated SDE diverged."
            )

        # Save to the BaseTrainer state for the probe (sliced to actual size)
        self._probe_tensors = (X0[:idx], X1[:idx])

        # BaseTrainer's generic parallel-tensor iterator (uses self.perm_gen)
        return self._cache_iterator(X0, X1)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    def _train_inner_loop(
        self,
        cache_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        inner_iters: int,
        outer_idx: int,
    ) -> Tuple[float, float, float]:
        self.u_model.train()
        self.s_model.train()
        tot_u = torch.zeros((), device=self.device)
        tot_s = torch.zeros((), device=self.device)

        u_base_lrs = [g["lr"] for g in self.u_opt.param_groups]
        s_base_lrs = [g["lr"] for g in self.s_opt.param_groups]

        for it in tqdm(range(inner_iters), ascii=True, desc="Training SF2M Inner Loop"):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / inner_iters))
                for g, lr0 in zip(self.u_opt.param_groups, u_base_lrs):
                    g["lr"] = lr0 * scale
                for g, lr0 in zip(self.s_opt.param_groups, s_base_lrs):
                    g["lr"] = lr0 * scale

            x_noise, x_data = next(cache_iter)
            B = x_noise.shape[0]

            t = (
                torch.rand(B, 1, device=self.device, generator=self.bridge_gen)
                * (1.0 - 2.0 * self.eps)
                + self.eps
            )
            x_t, targets, t_net = ConditionalVectorField.get_interpolant_and_target(
                model_type="sf2m",
                z_batch=x_data,
                t=t,
                gen=self.bridge_gen,
                x0_batch=x_noise,
                sigma=self.sigma,
                memory_format=self.memory_format,
            )
            v_target, eps_target = targets

            self.u_opt.zero_grad(set_to_none=True)
            self.s_opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                v_pred = self.u_model(x_t, t)
                eps_pred = self.s_model(x_t, t)

            loss_v, loss_eps = ConditionalVectorField.compute_sf2m_loss(
                v_pred, eps_pred, v_target, eps_target
            )
            loss = loss_v + loss_eps

            if self.use_amp:
                self.u_scaler.scale(loss_v).backward()
                self.s_scaler.scale(loss_eps).backward()
                self.u_scaler.unscale_(self.u_opt)
                self.s_scaler.unscale_(self.s_opt)
            else:
                loss.backward()

            self.log_inner_step(
                model=self.u_model_base,
                loss=loss.detach(),
                optimizer=self.u_opt,
                phase="sf2m",
                ipf_iter=outer_idx,
            )

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.u_model.parameters(), self.grad_clip
                )
                torch.nn.utils.clip_grad_norm_(
                    self.s_model.parameters(), self.grad_clip
                )

            if self.use_amp:
                self.u_scaler.step(self.u_opt)
                self.s_scaler.step(self.s_opt)
                self.u_scaler.update()
                self.s_scaler.update()
            else:
                self.u_opt.step()
                self.s_opt.step()

            if self.use_ema:
                self.ema_u.update(self.u_model_base)
                self.ema_s.update(self.s_model_base)

            tot_u += loss_v.detach()
            tot_s += loss_eps.detach()

        if self.lr_decay:
            for g, lr0 in zip(self.u_opt.param_groups, u_base_lrs):
                g["lr"] = lr0
            for g, lr0 in zip(self.s_opt.param_groups, s_base_lrs):
                g["lr"] = lr0

        # single GPU -> CPU sync
        l_u, l_s = (torch.stack([tot_u, tot_s]) / inner_iters).tolist()
        return l_u + l_s, l_u, l_s

    def fit(
        self,
        outer_iterations: int,
        inner_iterations: int = 5000,
        save_per: Optional[int] = None,
        save_path: Optional[str] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ):
        u_eval = self.ema_u.model if self.use_ema else self.u_model_base
        s_eval = self.ema_s.model if self.use_ema else self.s_model_base

        for n in range(1, outer_iterations + 1):
            self.logger.debug(f"\n--- SF2M Outer Iteration {n}/{outer_iterations} ---")
            nfes_before, phase_start = self.total_nfes, time.time()

            cache_iter = self._ot_iterator() if n == 1 else self._build_cache()

            loss, loss_u, loss_s = self._train_inner_loop(
                cache_iter, inner_iterations, n
            )
            del cache_iter

            metrics = {
                "loss": loss,
                "loss_u": loss_u,
                "loss_s": loss_s,
                "phase_time_sec": time.time() - phase_start,
            }

            # Run evaluation condition
            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (n % eval_per == 0)
            )
            if _run_eval:
                s_eval.eval()
                u_eval.eval()
                with self._isolated_global_rng(self.seed + SeedOffsets.SF2M_EVAL):
                    wrapper = SF2MInferenceWrapper(
                        u_eval, s_eval, self.sigma, direction="b", eps=self.eps
                    )
                    metrics.update(eval_callback(wrapper, direction="b"))

            self.log_phase_end("sf2m", n, metrics)
            self.track_parameter_drift(self.u_model_base, ipf_iter=n)

            qual_str = self._format_eval_str(metrics, _run_eval)
            self.logger.info(f"Outer Loop {n} | Loss: {loss:.4f} | {qual_str}")

            _save_model = (
                save_per is not None and save_path is not None and (n % save_per == 0)
            )
            if _save_model:
                save_model(u_eval, f"{save_path}/SF2M_u_checkpoint_{n}.pth")
                save_model(s_eval, f"{save_path}/SF2M_s_checkpoint_{n}.pth")

        self.logger.debug(("-" * 100))

        if self.use_ema:
            self.ema_u.copy_to(self.u_model_base)
            self.ema_s.copy_to(self.s_model_base)

        self.log_compute_cost([self.u_model_base, self.s_model_base])

        if save_path:
            save_model(
                self.u_model_base,
                f"{save_path}/{self.u_model_base.__class__.__name__}_backward_final.pth",
            )
            save_model(
                self.s_model_base,
                f"{save_path}/{self.s_model_base.__class__.__name__}_forward_final.pth",
            )

        final_wrapper = SF2MInferenceWrapper(
            self.u_model_base,
            self.s_model_base,
            self.sigma,
            direction="b",
            eps=self.eps,
        )

        return final_wrapper, {"metrics": metrics}
