from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.models.ema import EMAHelper
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


class SF2MInferenceWrapper(nn.Module):
    """
    Adapter exposing the SDE drift of the learned (u, s) pair to the pipeline.
    t=0 (Prior) -> t=1 (Data).
    """

    def __init__(
        self,
        u_model: nn.Module,
        s_model: nn.Module,
        sigma: float,
        direction: str = "b",
        eps: float = 1e-4,
    ):
        super().__init__()
        self.u = u_model
        self.s = s_model
        self.sigma = sigma
        self.direction = direction
        self.eps = eps
        self.model_type = "sf2m"

    def forward(self, x, t):
        # direction="b" (Prior -> Data), t passed is 0 -> 1.
        # direction="f" (Data -> Prior), t passed is 0 -> 1 by the sampler,
        # but we must treat it internally as 1 -> 0 for the networks.
        t_real = 1.0 - t if self.direction == "f" else t
        t_safe = torch.as_tensor(t_real, device=x.device, dtype=x.dtype).reshape(-1)
        if t_safe.numel() == 1:
            t_safe = t_safe.expand(x.shape[0])
        t_safe = t_safe.reshape(-1, 1).clamp(self.eps, 1.0 - self.eps)

        v_hat = self.u(x, t_safe)
        eps_hat = self.s(x, t_safe)

        if self.direction == "b":
            # Forward SDE drift: u_t^o + 0.5 * sigma^2 * s_t
            drift = v_hat - self.sigma * torch.sqrt(t_safe / (1.0 - t_safe)) * eps_hat
        else:
            # Reverse SDE drift (Anderson): -u_t^o + 0.5 * sigma^2 * s_t
            drift = -v_hat - self.sigma * torch.sqrt((1.0 - t_safe) / t_safe) * eps_hat

        return drift


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
        **kwargs,
    ):
        super().__init__(device=device, metric_logger=metric_logger)

        assert ot_method in ("minibatch", "sinkhorn", "greedy"), ot_method

        self.dataset = dataset
        self.u_opt = u_opt
        self.s_opt = s_opt
        self.device = device
        self.batch_size = batch_size
        self.sde_steps = sde_steps
        self.h = 1.0 / sde_steps
        self.ot_method = ot_method
        self.sigma = sigma
        self.eps = eps
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.use_ema = use_ema
        self.seed = seed
        self.sinkhorn_iters = sinkhorn_iters

        self.total_nfes = getattr(self, "total_nfes", 0)  # used by _simulate_trajectory

        if num_cache_batches is None:
            num_cache_batches = max(1, len(dataset) // batch_size)
        self.num_cache_batches = num_cache_batches

        # -------------------------------------------------------------
        # 0. Private RNG streams
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 1. Global channels_last & Hardware Settings
        # -------------------------------------------------------------
        self.data_shape = tuple(self.dataset[0].shape)
        is_image = len(self.data_shape) == 3
        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and is_image)
            else torch.contiguous_format
        )

        # if self.device.type == "cuda" and is_image:
        #     torch.backends.cudnn.benchmark = True

        # -------------------------------------------------------------
        # 2. Precision & Compilation Settings
        # ------------------------------------------------------------
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.u_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.s_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # -------------------------------------------------------------
        # 3. Data Pipeline & Probe Batch
        # -------------------------------------------------------------
        self.dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=self.shuffle_gen,
        )
        self._data_iter = self._repeater(self.dl)
        self.fixed_probe_batch = self._make_probe_batch()

        # -------------------------------------------------------------
        # 4. Models, EMA, and Compilation
        # -------------------------------------------------------------
        self.u_model_base = u_model.to(device, memory_format=self.memory_format)
        self.s_model_base = s_model.to(device, memory_format=self.memory_format)
        self.u_model_base.sigma = sigma
        self.s_model_base.sigma = sigma

        if self.use_ema:
            self.ema_u = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_u.register(self.u_model_base)
            self.ema_s = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_s.register(self.s_model_base)
        else:
            self.ema_u, self.ema_s = None, None

        self.u_sampler = copy.deepcopy(self.u_model_base)
        self.s_sampler = copy.deepcopy(self.s_model_base)

        if self.device.type == "cuda":
            self.u_model = torch.compile(self.u_model_base, mode="reduce-overhead")
            self.s_model = torch.compile(self.s_model_base, mode="reduce-overhead")
            self.u_sampler = torch.compile(self.u_sampler, mode="reduce-overhead")
            self.s_sampler = torch.compile(self.s_sampler, mode="reduce-overhead")
        else:
            self.u_model, self.s_model = self.u_model_base, self.s_model_base

    ############ RNG helpers

    @contextmanager
    def _isolated_global_rng(self, seed: int):
        """
        Seed the global CPU/CUDA RNGs for the duration of the block and restore the
        previous state afterwards. Used around eval_callback, whose Evaluator and
        model.generate() draw from the global generators: it now sees the same noise
        every time it runs, and can never shift anybody else's random stream.
        """
        devices = []
        if self.device.type == "cuda":
            idx = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            devices = [idx]

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def _randn_like(self, ref: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        # torch.randn_like has no `generator` argument, so draw explicitly and put the
        # result in the memory format randn_like would have preserved.
        return torch.randn(
            ref.shape, device=ref.device, dtype=ref.dtype, generator=gen
        ).contiguous(memory_format=self.memory_format)

    def _make_probe_batch(self) -> torch.Tensor:
        """
        Fixed batch for parameter-drift tracking. Taken straight from the dataset
        (first `batch_size` items) instead of iterating a DataLoader: creating a
        DataLoader iterator draws a base seed from the loader's generator (or from
        the global one), which would shift the shuffle stream.
        """
        n = min(self.batch_size, len(self.dataset))
        items = [self.dataset[i] for i in range(n)]
        items = [it[0] if isinstance(it, (list, tuple)) else it for it in items]
        return torch.stack(items).to(self.device, memory_format=self.memory_format)

    ############ UTILS

    @staticmethod
    def _repeater(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def _next_data(self) -> torch.Tensor:
        batch = next(self._data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

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

    ############ Simulation and Cache

    @torch.no_grad()
    def _simulate_trajectory(
        self, x_start: torch.Tensor, direction: str
    ) -> torch.Tensor:
        x = x_start.clone()
        h = 1.0 / self.sde_steps
        for i in range(self.sde_steps):
            t_val = (i * h) if direction == "b" else (1.0 - i * h)
            t_val = min(max(t_val, self.eps), 1.0 - self.eps)
            t = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                v_val = self.u_sampler(x, t).float()
                eps_val = self.s_sampler(x, t).float()

            if direction == "b":
                drift = v_val - self.sigma * math.sqrt(t_val / (1.0 - t_val)) * eps_val
            else:
                drift = -v_val - self.sigma * math.sqrt((1.0 - t_val) / t_val) * eps_val

            x = x + h * drift
            if self.sigma > 0:
                noise = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=self.noise_gen
                )
                x = x + self.sigma * math.sqrt(h) * noise

        self.total_nfes += x.shape[0] * self.sde_steps
        return x

    @torch.no_grad()
    def _build_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alg. 3, loops >= 2. Half of every cache batch is forward pairs (x0 real data,
        x1_hat = forward SDE endpoint); the other half is backward pairs (x0_hat = backward SDE
        endpoint from a fresh prior sample, x1 prior).
        """
        B = self.batch_size
        n_total = self.num_cache_batches * B
        X0 = torch.empty(
            (n_total, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        X1 = torch.empty_like(X0)

        # load the weights used for simulation (EMA if available)
        if self.use_ema:
            self.ema_u.copy_to(getattr(self.u_sampler, "_orig_mod", self.u_sampler))
            self.ema_s.copy_to(getattr(self.s_sampler, "_orig_mod", self.s_sampler))
        else:
            getattr(self.u_sampler, "_orig_mod", self.u_sampler).load_state_dict(
                self.u_model_base.state_dict()
            )
            getattr(self.s_sampler, "_orig_mod", self.s_sampler).load_state_dict(
                self.s_model_base.state_dict()
            )
        self.u_sampler.eval()
        self.s_sampler.eval()

        half = B // 2
        for k in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc="Building SF2M Cache",
        ):
            x1_real = self._next_data()
            x1_b = x1_real[:half]
            x0_b = self._simulate_trajectory(x1_b, direction="f")  # Reverse to Noise

            x0_f = self._randn_like(x1_real[: B - half], self.noise_gen)
            x1_f = self._simulate_trajectory(x0_f, direction="b")  # Forward to Data

            X0[k * B : (k + 1) * B] = torch.cat([x0_b, x0_f], dim=0)
            X1[k * B : (k + 1) * B] = torch.cat([x1_b, x1_f], dim=0)

        if not (torch.isfinite(X0).all() and torch.isfinite(X1).all()):
            raise RuntimeError(
                "Non-finite values in the SF2M cache: the simulated SDE diverged."
            )

        return self._cache_iterator(X0, X1)

    def _cache_iterator(self, X0: torch.Tensor, X1: torch.Tensor):
        """Fast infinite minibatch iterator over GPU tensors (matches IPF)."""
        n = X0.shape[0]
        while True:
            perm = torch.randperm(n, device=X0.device, generator=self.perm_gen)
            for j in range(0, n - self.batch_size + 1, self.batch_size):
                idx = perm[j : j + self.batch_size]
                yield X0[idx], X1[idx]

    def _ot_iterator(self):
        """Infinite generator for online Optimal Transport (Iteration 1)."""
        while True:
            x1 = self._next_data()
            x0 = self._randn_like(x1, self.noise_gen)
            yield self._couple(x0, x1)

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
            t_e = t.view(B, *([1] * (x_noise.ndim - 1)))

            mu_t = t_e * x_data + (1.0 - t_e) * x_noise
            sigma_t = self.sigma * torch.sqrt(t_e * (1.0 - t_e))
            noise = self._randn_like(x_noise, self.bridge_gen)
            x_t = (mu_t + sigma_t * noise).contiguous(memory_format=self.memory_format)

            self.u_opt.zero_grad(set_to_none=True)
            self.s_opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                v_pred = self.u_model(x_t, t)
                eps_pred = self.s_model(x_t, t)

            # losses in fp32
            v_pred, eps_pred = v_pred.float(), eps_pred.float()

            # Unscaled Targets
            v_target = x_data - x_noise
            eps_target = noise

            loss_v = torch.mean((v_pred - v_target) ** 2)
            loss_eps = torch.mean((eps_pred - eps_target) ** 2)
            loss = loss_v + loss_eps

            if self.use_amp:
                self.u_scaler.scale(loss_v).backward()
                self.s_scaler.scale(loss_eps).backward()
                self.u_scaler.unscale_(self.u_opt)
                self.s_scaler.unscale_(self.s_opt)
            else:
                loss.backward()

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

            if it % 50 == 0:
                self.log_inner_step(
                    self.u_model,
                    loss.detach(),
                    self.u_opt,
                    phase="sf2m",
                    ipf_iter=outer_idx,
                )

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

        for l in range(1, outer_iterations + 1):
            self.logger.debug(f"\n--- SF2M Outer Iteration {l}/{outer_iterations} ---")
            phase_start = time.time()

            if l == 1:
                cache_iter = self._ot_iterator()
            else:
                cache_iter = self._build_cache()

            loss, loss_u, loss_s = self._train_inner_loop(
                cache_iter, inner_iterations, l
            )
            del cache_iter

            metrics = {
                "loss": loss,
                "loss_u": loss_u,
                "loss_s": loss_s,
                "phase_time_sec": time.time() - phase_start,
            }

            # Run evaluation condition
            run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (l % eval_per == 0)
            )

            if run_eval:
                u_eval.eval()
                s_eval.eval()
                with self._isolated_global_rng(self.seed + SeedOffsets.SF2M_EVAL):
                    wrapper = SF2MInferenceWrapper(
                        u_eval, s_eval, self.sigma, direction="b", eps=self.eps
                    )
                    metrics.update(eval_callback(wrapper, direction="b"))

            self.log_phase_end("sf2m", l, metrics)
            self.track_parameter_drift(self.u_model_base, ipf_iter=l)

            if not run_eval:
                qual_str = ""
            elif "FID" in metrics:
                qual_str = (
                    f"B-FID: {metrics['FID']:.3f} | "
                    f"B-Prec: {metrics['Precision']:.3f} | "
                    f"B-Rec: {metrics['Recall']:.3f}"
                )
            else:
                qual_str = f"B-MMD: {metrics.get('eval_MMD', float('nan')):.4f}"

            self.logger.info(f"Outer Loop {l} | Loss: {loss:.4f} | {qual_str}")

            if save_per and save_path and (l % save_per == 0):
                save_model(u_eval, f"{save_path}/SF2M_u_checkpoint_{l}.pth")
                save_model(s_eval, f"{save_path}/SF2M_s_checkpoint_{l}.pth")

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
