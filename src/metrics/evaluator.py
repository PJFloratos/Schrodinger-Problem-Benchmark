from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.metrics.distances import (
    get_mmd,
    get_path_consistency_mse,
    get_drift_mse,
    get_generative_quality_metrics,
)
from src.metrics.plot import (
    generate_and_plot,
    visualize_energy_landscape,
    visualize_enot_landscape,
)
from src.utils.seed import derive_seed
from src.utils import text_logger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

import os
import math
import time
from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union, Tuple, Callable, Any


class Evaluator:
    logger = text_logger(__name__)

    def __init__(
        self,
        test_ds: Dataset,
        device: torch.device = torch.device("cpu"),
        sde_steps: int = 100,
        n_gen_samples: Optional[int] = None,  # None -> as many as the test set
        ground_truth_v: Optional[Callable] = None,
        seed: int = 0,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.sde_steps = sde_steps
        self.n_gen_samples = n_gen_samples
        self.ground_truth_v = ground_truth_v
        self.seed = seed
        self._x_test = None
        self.solver_kwargs = solver_kwargs if solver_kwargs else {}

        self.dl = DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False,
            generator=torch.Generator().manual_seed(derive_seed(seed, "eval_loader")),
        )

        # Private noise stream for the validation loss, re-seeded on every call.
        self._noise_gen = torch.Generator(device=self.device)
        self._val_seed = derive_seed(seed, "eval_val_noise")

        self._forward_sim_gen = torch.Generator(device=self.device)
        self._forward_sim_seed = derive_seed(seed, "eval_forward_sim")

        self._prior_gen = torch.Generator(device=self.device)
        self._prior_seed = derive_seed(seed, "eval_prior_noise")

        # Global RNGs that evaluate() snapshots and restores (see _rng_shield).
        if self.device.type == "cuda":
            idx = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            self._rng_devices = [idx]
        else:
            self._rng_devices = []

        # Sniff dataset shape to decide which metrics to run. Looking at one item
        item = test_ds[0]
        self.is_image_data = item.ndim == 3  # (C, H, W) per item -> 4D batches

        # Capture the raw shape of a single data item for the solver
        self.data_shape = tuple(item.shape)

        # The quality metrics compare N generated
        # samples against the first N entries of this ordering, so (a) real and generated
        # counts are always equal, (b) every evaluation uses the same real subset, and
        # (c) the subset for a smaller N is contained in the one for a larger N.
        self._ref_perm = torch.randperm(
            len(test_ds),
            generator=torch.Generator().manual_seed(derive_seed(seed, "eval_ref_perm")),
        )

        # Same rule as Trainer: channels_last for images on CUDA. The model handed to
        # evaluate() (the EMA copy) inherits the Trainer's channels_last weights, so
        # feed it inputs in that layout instead of relying on implicit conversions.
        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and self.is_image_data)
            else torch.contiguous_format
        )

    def evaluate(
        self,
        model: nn.Module,
        backward_model: nn.Module = None,
        num_samples: int = None,  # To overwrite the global one
        log: bool = False,
        use_amp: bool = True,
        direction: str = "b",
        epoch: Optional[int] = None,
        cfg: Optional[Any] = None,
        trainer: Optional[Any] = None,
    ) -> Dict[str, float]:

        with torch.random.fork_rng(devices=self._rng_devices):
            return self._evaluate(
                model,
                backward_model,
                num_samples,
                log,
                use_amp,
                direction,
                epoch,
                cfg,
                trainer,
            )

    def _evaluate(
        self,
        model: nn.Module,
        backward_model: Optional[nn.Module],
        num_samples: Optional[int],
        log: bool,
        use_amp: bool,
        direction: str,
        epoch: Optional[int] = None,
        cfg: Optional[Any] = None,
        trainer: Optional[Any] = None,
    ) -> Dict[str, float]:
        model.eval()
        if backward_model:
            backward_model.eval()

        # 1. Fetch & prep data setup
        # Delegate dataset fetching to _get_test_data unconditionally to ensure
        # it is cached in self._x_test and reused efficiently across evaluations.
        x_true = self._get_test_data()
        n_eval = self._determine_eval_samples(len(x_true), num_samples)

        # 2. Validation Loss (if applicable)
        loss = None
        if direction == "b":
            loss = self._compute_validation_loss(model, direction, use_amp)

        # 3. Generate Samples
        sampler = EulerSampler(
            model_type=model.model_type, steps=self.sde_steps, use_amp=use_amp
        )
        x_gen, x_ref, gen_time = self._generate_samples(
            model, sampler, x_true, n_eval, direction
        )

        # 4. Compute Metrics
        results = {
            "eval_samples": n_eval,
            "eval_NFEs": self.sde_steps,
            "eval_generation_time": gen_time,
        }
        if loss is not None:
            results["eval_loss"] = float(loss)

        metrics = self._compute_metrics(model, x_gen, x_ref, direction)
        results.update(metrics)

        # 5. Logging and Visualization
        if log:
            self._log_results(results)

        if epoch and cfg:
            self._visualize(model, results, x_true, epoch, cfg, trainer)

        return results

    # -------------------------------------------------------------------------
    # Modular Helper Methods
    # -------------------------------------------------------------------------

    def _get_test_data(self) -> torch.Tensor:
        if self._x_test is None:
            self._x_test = torch.cat([z.cpu() for z in self.dl], dim=0)
        return self._x_test

    def _determine_eval_samples(
        self, total_avail: int, requested: Optional[int]
    ) -> int:
        """Determines the number of samples to evaluate based on availability."""
        n_wanted = requested if requested else total_avail
        n_eval = min(n_wanted, total_avail)
        if n_eval < n_wanted:
            Evaluator.logger.warning(
                f"Requested {n_wanted} samples but the test set only has "
                f"{total_avail}; evaluating with {n_eval}."
            )
        return n_eval

    def _compute_validation_loss(
        self, model: nn.Module, direction: str, use_amp: bool
    ) -> float:
        """Calculates regression health against straight-line/SDE paths and extracts targets."""

        # EBM loss requires MCMC and spatial autograd. The EBMTrainer handles this.
        # We bypass it here to prevent inference_mode crashes.
        if model.model_type in ["ebm", "scone", "enot_drift"]:
            return None

        gen = self._noise_gen
        # Re-seed on every call: each evaluation sees the exact same (t, noise) draws,
        # so eval_loss only changes when the weights change.
        gen.manual_seed(self._val_seed)

        # Sniff model attributes safely
        model_type = getattr(model, "model_type", "flow_m")
        sigma = getattr(model, "sigma", 1.0)
        is_sf2m = model_type == "sf2m"

        # --- IPF Proxy Resolution ---
        # The true IPF loss requires running a full SDE simulation of the opposite network
        # to generate `drift_next`. To avoid massive computational overhead in the Evaluator,
        # we proxy IPF's validation health against the standard independent SDE targets.
        eval_model_type = "sde" if model_type == "ipf" else model_type

        # Define amp_dtype locally based on the device
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        total_loss = 0.0
        with torch.inference_mode():
            for z_batch in tqdm(
                self.dl, ascii=True, desc="    Calculating Loss", leave=False
            ):
                z_batch = z_batch.to(
                    self.device, non_blocking=True, memory_format=self.memory_format
                )
                B = z_batch.shape[0]

                # Sample time uniformly
                t = torch.rand(B, 1, device=self.device, generator=gen) * 0.999

                # Generate x0 (noise) for methods that require an explicit start point
                x0_batch = None
                if eval_model_type in ["imf", "sf2m"]:
                    x0_batch = torch.randn(
                        z_batch.shape,
                        device=self.device,
                        dtype=z_batch.dtype,
                        generator=gen,
                    ).contiguous(memory_format=self.memory_format)

                # IMF relies on x0=Data and x1=Prior for both forward and backward training.
                # Evaluator naturally draws z_batch=Data and generates x0_batch=Prior.
                # We swap them here so the Brownian bridge regression target matches training.
                if eval_model_type == "imf":
                    z_batch, x0_batch = x0_batch, z_batch

                x_t, target_u, t_net = (
                    ConditionalVectorField.get_interpolant_and_target(
                        model_type=eval_model_type,
                        z_batch=z_batch,
                        t=t,
                        gen=gen,
                        memory_format=self.memory_format,
                        sigma=sigma,
                        direction=direction,
                        x0_batch=x0_batch,
                    )
                )

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    if is_sf2m:
                        # SF2M wrapper holds both networks; we query them directly
                        v_pred = model.u(x_t, t_net)
                        eps_pred = model.s(x_t, t_net)
                        v_target, eps_target = target_u

                        loss_v, loss_eps = ConditionalVectorField.compute_sf2m_loss(
                            v_pred, eps_pred, v_target, eps_target
                        )
                        batch_loss = loss_v + loss_eps
                    else:
                        # Standard single-network models (SDE, Flow Matching, Minibatch, IMF)
                        pred_u = model(x_t, t_net)
                        batch_loss = ConditionalVectorField.compute_loss(
                            model_type=eval_model_type,
                            pred_u=pred_u,
                            target_u=target_u,
                            t_net=t_net,
                            sigma=sigma,
                        )

                total_loss += batch_loss.item()

        return total_loss / len(self.dl)

    def _generate_samples(
        self,
        model: nn.Module,
        sampler: EulerSampler,
        x_true: torch.Tensor,
        n_eval: int,
        direction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Handles the forward or backward generation chains."""
        gen_start = time.time()

        if direction == "b":
            # Backward Chain: Prior -> Data
            x_gen = sampler.generate(
                model=model,
                shape=self.data_shape,
                n_samples=n_eval,
                device=self.device,
                init_seed=derive_seed(self.seed, "eval_solver_init"),
                step_seed=derive_seed(self.seed, "eval_solver_step"),
                **self.solver_kwargs,
            )
            x_ref = x_true[self._ref_perm[:n_eval]]
        else:
            # Forward Chain: Data -> Prior
            x_init = x_true[self._ref_perm[:n_eval]].to(
                self.device, memory_format=self.memory_format
            )
            x_gen = sampler.generate(
                model=model,
                x_init=x_init,
                batch_size=256,
                step_seed=self._forward_sim_seed,
                **self.solver_kwargs,
            )
            # Re-seed prior target generator for comparison
            self._prior_gen.manual_seed(self._prior_seed)
            x_ref = torch.randn(
                x_gen.shape,
                device=self.device,
                dtype=x_gen.dtype,
                generator=self._prior_gen,
            ).contiguous(memory_format=self.memory_format)

        gen_time = time.time() - gen_start
        return x_gen, x_ref, gen_time

    def _compute_metrics(
        self, model: nn.Module, x_gen: torch.Tensor, x_ref: torch.Tensor, direction: str
    ) -> Dict[str, float]:
        """Maps data back to real space and computes distances/qualities."""
        results = {}
        ds = self.dl.dataset
        if hasattr(ds, "inverse_transform"):
            x_gen = ds.inverse_transform(x_gen)
            x_ref = ds.inverse_transform(x_ref)

        if self.is_image_data and direction == "b":
            results.update(get_generative_quality_metrics(x_ref, x_gen))
        else:
            results["eval_MMD"] = float(get_mmd(x_ref, x_gen, device=self.device))

            if self.ground_truth_v and direction == "b" and model.model_type != "ebm":
                results["drift_MSE"] = get_drift_mse(model, self.ground_truth_v, x_gen)

        return results

    def _log_results(self, results: Dict[str, float]) -> None:
        """Helper to pretty-print evaluation metrics."""
        self.logger.info("Evaluation Metrics:")
        for k, v in results.items():
            if isinstance(v, float):
                self.logger.info(f"{k:>20}: {v:.6f}")
            else:
                self.logger.info(f"{k:>20}: {v}")

    def _visualize(
        self,
        model: nn.Module,
        results: Dict[str, float],
        x_true: torch.Tensor,
        epoch: int,
        cfg: Any,
        trainer: Any,
    ) -> None:
        """Handles metric plotting and landscape visualization for EBM/ENOT."""
        generate_and_plot(model, results, cfg, epoch)

        if cfg.model.type not in ["ebm", "enot", "scone"] or not trainer:
            return

        num_points = min(2000, len(x_true))
        real_idx = torch.randperm(len(x_true))[:num_points]
        real_data = x_true[real_idx].to(self.device)

        # -------------------------------------------------------------
        # EBM Visualization
        # -------------------------------------------------------------
        if cfg.model.type == "ebm":
            buffer_y = getattr(trainer, "replay_buffer_y", None)
            if buffer_y is not None:
                # Ensure we don't request more points than the buffer holds
                fake_idx = torch.randperm(buffer_y.size(0))[:num_points]
                buffer_data = buffer_y[fake_idx]

                save_path = os.path.join(
                    cfg.plots_path, f"energy_landscape_epoch_{epoch}.png"
                )
                visualize_energy_landscape(
                    model, real_data, buffer_data, self.device, save_path
                )

        # -------------------------------------------------------------
        # SCONE Visualization
        # -------------------------------------------------------------
        elif cfg.model.type == "scone":
            buffer_y = getattr(trainer, "replay_buffer_y", None)

            if buffer_y is not None:
                fake_idx = torch.randperm(buffer_y.size(0))[:num_points]
                buffer_y_data = buffer_y[fake_idx]

                psi_model = (
                    trainer.psi_ema.model
                    if getattr(trainer, "psi_ema", None)
                    else trainer.psi_base
                )

                save_path_psi = os.path.join(
                    cfg.plots_path, f"scone_psi_landscape_epoch_{epoch}.png"
                )
                visualize_energy_landscape(
                    psi_model, real_data, buffer_y_data, self.device, save_path_psi
                )

        # -------------------------------------------------------------
        # ENOT Visualization
        # -------------------------------------------------------------
        elif cfg.model.type == "enot":
            with torch.no_grad():
                if getattr(trainer, "potential_ema", None):
                    potential_model = trainer.potential_ema.model
                else:
                    potential_model = trainer.potential_base

                x0 = torch.randn_like(real_data) * cfg.sigma
                xN, _ = trainer.sampler.rollout_enot_training(
                    model=model,
                    x0=x0,
                    epsilon=trainer.target_epsilon,
                )

                save_path = os.path.join(
                    cfg.plots_path, f"enot_landscape_epoch_{epoch}.png"
                )
                visualize_enot_landscape(
                    potential_model,
                    model,
                    real_data,
                    xN,
                    self.device,
                    save_path,
                )
