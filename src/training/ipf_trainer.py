from src.utils import EMAHelper, save_model

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import math
from tqdm import tqdm
from typing import Union


# def reference_drift(x: torch.Tensor) -> torch.Tensor:
#     """Ornstein-Uhlenbeck reference process used for the very first forward pass."""
#     return -x


def reference_drift(x: torch.Tensor) -> torch.Tensor:
    """Standard Brownian Motion reference process (zero drift)."""
    return torch.zeros_like(x)


class IPFTrainer:
    """
    Iterative Proportional Fitting (Diffusion Schrodinger Bridge).

    Conventions (kept identical for both networks):
      * Both networks take *forward* time t in [0, 1] as input.
      * Both networks output a *drift* (per unit time).
      * Forward chain : x_{k+1} = x_k + h * f(x_k, t_k) + sqrt(h) * z
      * Backward chain: y_{i+1} = y_i + h * b(y_i, 1 - t_i) + sqrt(h) * z
    """

    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        dataset: torch.utils.data.Dataset,
        forward_opt: torch.optim.Optimizer,
        backward_opt: torch.optim.Optimizer,
        device: torch.device,
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = 10,  # Number of dataset batches to cache per iteration
        refresh_every: int = 500,  # regenerate the cache every N gradient steps
        mean_match: bool = True,  # DSB mean-matching target (see _simulate_and_cache)
        grad_clip: float = 1.0,  # set to None / 0 to disable
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
    ):
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
        self.mean_match = mean_match
        self.grad_clip = None
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.criterion = nn.MSELoss()
        self.data_shape = tuple(self.dataset[0].shape)

        self.dl = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self._data_iter = self._repeater(self.dl)

        # Initialize and register EMA trackers for both networks
        self.ema_f = EMAHelper(mu=0.999, device=self.device)
        self.ema_f.register(self.f_model)
        self.ema_b = EMAHelper(mu=0.999, device=self.device)
        self.ema_b.register(self.b_model)

    ############ UTILS

    @staticmethod
    def _repeater(dataloader):
        """Infinite generator to continuously yield batches (mimics DSB repeater)."""
        while True:
            for batch in dataloader:
                yield batch

    def _cache_iterator(self, X, T, U):
        """Fast infinite minibatch iterator over GPU tensors (no per-item DataLoader overhead)."""
        n = X.shape[0]
        while True:
            perm = torch.randperm(n, device=X.device)
            for j in range(0, n - self.batch_size + 1, self.batch_size):
                idx = perm[j : j + self.batch_size]
                yield X[idx], T[idx], U[idx]

    ############ CACHE GENERATRION

    @torch.no_grad()
    def _simulate_and_cache(
        self,
        source_model: nn.Module,
        ema_helper: EMAHelper,
        direction: str,
        ipf_iteration: int,
    ):
        """
        Simulate the chain of `source_model` in `direction` and record regression
        data (x_{k+1}, t_{k+1}, target) for the network of the OPPOSITE direction.

        Noise is injected at EVERY step (as in the official repo with sample=False).
        Dropping the last noise is only correct when producing final samples.
        """
        use_reference = ipf_iteration == 0 and direction == "f"
        if not use_reference:
            sampler = ema_helper.ema_copy(source_model)
            sampler.eval()

        def drift_fn(x, t):
            return reference_drift(x) if use_reference else sampler(x, t)

        # Pre-allocate contiguous memory blocks (massive speedup)
        total_samples = self.num_cache_batches * self.batch_size * self.sde_steps
        X_cache = torch.empty((total_samples, *self.data_shape), device=self.device)
        T_cache = torch.empty((total_samples, 1), device=self.device)
        U_cache = torch.empty((total_samples, *self.data_shape), device=self.device)

        idx = 0

        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Simulating {direction.upper()} Cache",
        ):
            batch = next(self._data_iter)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            b_size = batch.shape[0]

            # forward chain starts at data, backward chain starts at the prior N(0, I)
            x = batch if direction == "f" else torch.randn_like(batch)

            for i in range(self.sde_steps):
                # sampler runs its own chain: time i*h
                t_now = torch.full((b_size, 1), i * self.h, device=self.device)
                # trained net (opposite chain) sees this state at its own time 1-(i+1)*h
                t_next = torch.full(
                    (b_size, 1), 1.0 - (i + 1) * self.h, device=self.device
                )

                drift = drift_fn(x, t_now)
                x_next = x + self.h * drift + math.sqrt(self.h) * torch.randn_like(x)

                if self.mean_match:
                    # DSB mean matching: F(x_k) - F(x_{k+1}) with F(x) = x + h * drift(x, t_k).
                    # Both evaluations use the same time index t_k (as in the official code).
                    drift_next = drift_fn(x_next, t_now)
                    target = (
                        (x + self.h * drift) - (x_next + self.h * drift_next)
                    ) / self.h
                else:
                    # Plain reverse-increment regression
                    target = (x - x_next) / self.h

                # The trained network is queried at x_{k+1}, at the forward time of x_{k+1}.
                X_cache[idx : idx + b_size] = x_next
                T_cache[idx : idx + b_size] = t_next
                U_cache[idx : idx + b_size] = target
                x = x_next
                idx += b_size

        return self._cache_iterator(X_cache, T_cache, U_cache)

    ############ TRAINING

    def _train_cache(
        self,
        target_model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        make_cache,
        num_iter: int,
        direction: str,
    ) -> float:
        target_model.train()
        cache_iter = make_cache()
        total_loss = 0.0
        base_lrs = [g["lr"] for g in opt.param_groups]

        desc = (
            "Training Forward Model" if direction == "f" else "Training Backward Model"
        )
        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                cache_iter = make_cache()  # fresh trajectories, old cache is freed

            x_batch, t_batch, u_batch = next(cache_iter)

            opt.zero_grad(set_to_none=True)
            pred_u = target_model(x_batch, t_batch)
            loss = self.criterion(pred_u, u_batch)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    target_model.parameters(), self.grad_clip
                )
            opt.step()

            # Update EMA shadow weights immediately after every gradient step
            ema_helper.update(target_model)
            total_loss += loss.item()

        for g, lr0 in zip(opt.param_groups, base_lrs):  # restore for the next phase
            g["lr"] = lr0

        return total_loss / num_iter

    def fit(
        self,
        ipf_iterations: int,
        inner_iterations: int = 5000,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
    ):
        for n in range(ipf_iterations):
            print(f"\n--- IPF Iteration {n+1}/{ipf_iterations} ---")

            # Phase 1: simulate forward (f, or OU at n == 0), train backward net b
            b_loss = self._train_cache(
                self.b_model,
                self.b_opt,
                self.ema_b,
                lambda: self._simulate_and_cache(self.f_model, self.ema_f, "f", n),
                inner_iterations,
                direction="b",
            )

            # Phase 2: simulate backward (b) from the prior, train forward net f
            f_loss = self._train_cache(
                self.f_model,
                self.f_opt,
                self.ema_f,
                lambda: self._simulate_and_cache(self.b_model, self.ema_b, "b", n),
                inner_iterations,
                direction="f",
            )

            print(
                f"Iteration {n+1} | Forward Loss: {f_loss:.6f} | Backward Loss: {b_loss:.6f}"
            )

            # --- Intermediate Checkpoint Saving ---
            if save_per and save_path and ((n + 1) % save_per == 0):
                # Generate a temporary copy with EMA weights applied for an accurate checkpoint
                temp_b_model = self.ema_b.ema_copy(self.b_model)
                save_model(
                    temp_b_model,
                    f"{save_path}/{self.b_model.__class__.__name__}_backward_checkpoint_{n+1}.pth",
                )

        # Load smoothed weights into the models used for evaluation
        self.ema_f.ema(self.f_model)
        self.ema_b.ema(self.b_model)

        # --- Final Model Saving ---
        if save_path:
            save_model(
                self.b_model,
                f"{save_path}/{self.b_model.__class__.__name__}_backward_final.pth",
            )
