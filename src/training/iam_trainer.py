from src.training.imf_trainer import IMFTrainer
from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.models.components import EMAHelper
from src.utils.seed import derive_seed
from src.utils import text_logger

import torch
from torch import nn

import math
from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple


class IAMTrainer(IMFTrainer):
    """
    Iterative Action Matching (IAM) solver for Schrödinger Bridges.
    Inherits outer-loop alternation, pair caching, EMA, and logging from IMFTrainer.
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
        sde_steps: int = 200,
        num_cache_batches: int = 10,
        refresh_every: int = 500,
        sigma: float = 1.0,
        gamma: float = 1.0,
        reference: str = "brownian",
        eps: float = 1e-4,
        first_coupling: str = "ind",
        exact_laplacian: Optional[bool] = None,
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,
        lr_final_ratio: float = 0.05,
        use_amp: bool = False,
        use_ema: bool = True,
        use_compile: bool = False,
    ):
        if first_coupling not in ("ind", "ref", "ot"):
            raise ValueError(
                f"first_coupling must be 'ind', 'ref', or 'ot', got {first_coupling!r}"
            )
        if reference not in ("brownian", "ou"):
            raise ValueError(f"reference must be 'brownian' or 'ou', got {reference!r}")

        # Temporarily pass 'ind' to IMFTrainer.__init__ if 'ot' is requested, then restore
        super_coupling = "ind" if first_coupling == "ot" else first_coupling

        super().__init__(
            forward_model=forward_model,
            backward_model=backward_model,
            dataset=dataset,
            forward_opt=forward_opt,
            backward_opt=backward_opt,
            device=device,
            metric_logger=metric_logger,
            seed=seed,
            batch_size=batch_size,
            sde_steps=sde_steps,
            num_cache_batches=num_cache_batches,
            refresh_every=refresh_every,
            sigma=sigma,
            eps=eps,
            first_coupling=super_coupling,
            grad_clip=grad_clip,
            ema_mu=ema_mu,
            lr_decay=lr_decay,
            lr_final_ratio=lr_final_ratio,
            use_amp=False,  # Second-order autograd (Laplacian) requires fp32
            use_ema=use_ema,
            use_compile=False,  # Protect double-autograd graph from torch.compile
        )

        self.model_type = "iam"
        self.first_coupling = first_coupling
        self.gamma = gamma
        self.reference = reference

        flat_dim = math.prod(self.data_shape)
        self.exact_laplacian = (
            (flat_dim <= 4) if exact_laplacian is None else exact_laplacian
        )

        # Attach reference metadata so EulerSampler resolves physics automatically
        for m in [
            self.f_model_base,
            self.b_model_base,
            self.f_sampler,
            self.b_sampler,
            self.f_ema.model if self.f_ema is not None else None,
            self.b_ema.model if self.b_ema is not None else None,
        ]:
            if m is not None:
                m.sigma = self.sigma
                m.gamma = self.gamma
                m.reference = self.reference

    # ------------------------------------------------------------------
    # INITIAL & SIMULATED COUPLINGS
    # ------------------------------------------------------------------
    # def _sample_prior_batch(self, ref_tensor: torch.Tensor) -> torch.Tensor:
    #     z = self._randn_like(ref_tensor, self.noise_gen)
    #     return z * self.sigma if self.reference == "brownian" else z

    def _sample_prior_batch(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            *self.data_shape,
            device=self.device,
            generator=self.noise_gen,
        ).contiguous(memory_format=self.memory_format)

    def _sample_reference_endpoint(self, x0: torch.Tensor) -> torch.Tensor:
        z = self._randn_like(x0, self.noise_gen)
        if self.reference == "brownian":
            return x0 + self.sigma * z
        alpha_01 = math.exp(-self.gamma)
        std_01 = math.sqrt(-math.expm1(-2.0 * self.gamma))
        return alpha_01 * x0 + std_01 * z

    @torch.no_grad()
    def _make_pairs(
        self, phase: str, imf_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds empirical couplings:
          - phase == 'b' (pi^{B, (k)}): (x0 ~ p0, x1 simulated forward via s^F)
          - phase == 'f' (pi^{F, (k)}): (x0 simulated backward via s^B, x1 ~ p1)
        """
        use_first = imf_iter == 1 and (
            phase == "b" or self.first_coupling in ("ind", "ot")
        )

        if not use_first:
            src_base, src_ema, sampler = (
                (self.f_model_base, self.f_ema, self.f_sampler)
                if phase == "b"
                else (self.b_model_base, self.b_ema, self.b_sampler)
            )
            if src_ema is not None:
                src_ema.copy_to(sampler)
            else:
                getattr(sampler, "_orig_mod", sampler).load_state_dict(
                    src_base.state_dict()
                )
            sampler.eval()

        sampler_engine = EulerSampler(
            model_type="iam", steps=self.sde_steps, use_amp=False
        )

        total_samples = self.num_cache_batches * self.batch_size
        X0_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        X1_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )

        idx = 0
        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Building IAM {phase.upper()} couplings",
        ):
            if use_first:
                x0 = self._next_data()
                B = x0.shape[0]
                if self.first_coupling == "ref":
                    x1 = self._sample_reference_endpoint(x0)
                elif self.first_coupling == "ot":
                    # x1 = self._sample_prior_batch(x0)
                    x1 = self._sample_prior_batch(B)
                    B = x0.shape[0]
                    cost = torch.cdist(x0.view(B, -1), x1.view(B, -1), p=2).pow(2)
                    r_ind, c_ind = ConditionalVectorField.exact_ot_assignment_cpu(cost)
                    x0, x1 = x0[r_ind], x1[c_ind]
                else:  # "ind"
                    # x1 = self._sample_prior_batch(x0)
                    x1 = self._sample_prior_batch(B)

            elif phase == "b":
                # Step 4: draw x0 ~ p0, simulate forward SDE from t=0 to t=1
                x0 = self._next_data()
                x1 = sampler_engine.generate(
                    model=sampler,
                    x_init=x0,
                    is_forward=True,
                    batch_size=self.batch_size,
                    step_seed=self.noise_gen,
                    drop_last_noise=False,
                    verbose=False,
                    sigma=self.sigma,
                    gamma=self.gamma,
                    reference=self.reference,
                )
            else:
                # Step 2: draw x1 ~ p1, simulate backward SDE from t=1 down to t=0
                # x1 = torch.randn(
                #     self.batch_size,
                #     *self.data_shape,
                #     device=self.device,
                #     generator=self.noise_gen,
                # ).contiguous(memory_format=self.memory_format)
                # if self.reference == "brownian":
                #     x1 = x1 * self.sigma
                x1 = self._sample_prior_batch(self.batch_size)
                x0 = sampler_engine.generate(
                    model=sampler,
                    x_init=x1,
                    is_forward=False,
                    batch_size=self.batch_size,
                    step_seed=self.noise_gen,
                    drop_last_noise=False,
                    verbose=False,
                    sigma=self.sigma,
                    gamma=self.gamma,
                    reference=self.reference,
                )

            b_size = x0.shape[0]
            X0_cache[idx : idx + b_size] = x0
            X1_cache[idx : idx + b_size] = x1
            idx += b_size

        self._probe_tensors = (X0_cache[:idx], X1_cache[:idx])
        return self._cache_iterator(X0_cache, X1_cache)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    def _train_phase(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        make_cache: Callable,
        phase: str,
        num_iter: int,
        imf_iter: int,
    ) -> float:
        model.train()
        scaler = self.f_scaler if phase == "f" else self.b_scaler
        base_lrs = [g["lr"] for g in opt.param_groups]
        cache_iter = make_cache()

        log_phase = "forward" if phase == "f" else "backward"
        desc = "Training Forward Action" if phase == "f" else "Training Backward Action"

        total_loss = torch.tensor(0.0, device=self.device)
        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                pre_loss = self._probe_loss(model, "iam", phase)
                cache_iter = make_cache()
                post_loss = self._probe_loss(model, "iam", phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            x0, x1 = next(cache_iter)

            loss = ConditionalVectorField.compute_iam_loss(
                model=model,
                x0=x0,
                x1=x1,
                direction=phase,
                sigma=self.sigma,
                gamma=self.gamma,
                reference=self.reference,
                exact_laplacian=self.exact_laplacian,
                gen=self.bridge_gen,
                create_graph=True,
            )

            self._optimize_step(
                model,
                opt,
                scaler,
                loss,
                ema_helper,
                log=True,
                phase=log_phase,
                epoch=imf_iter,
            )

            total_loss += loss.detach().float()

        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0

        return (total_loss / num_iter).item()
