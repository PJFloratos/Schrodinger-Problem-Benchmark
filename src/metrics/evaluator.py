from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.metrics.distances import (
    get_mmd,
    get_path_consistency_mse,
    get_drift_mse,
    get_generative_quality_metrics,
)
from src.utils.seed import SeedOffsets
from src.utils import text_logger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

import math
import time
from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union, Tuple, Callable


def _derive_seed(seed: int, stream: int) -> int:
    """
    Independent 63-bit seed for (seed, stream), via numpy's SeedSequence.

    Unlike `seed + stream`, streams belonging to consecutive seeds never coincide
    (with `+`, seed s / stream 2 equals seed s+1 / stream 1). Requires seed >= 0.
    """
    state = np.random.SeedSequence([int(seed), int(stream)]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0]) >> 1  # python int: numpy 1.x cannot shift uint64 by an int


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
    ) -> None:
        self.device = device
        self.sde_steps = sde_steps
        self.n_gen_samples = n_gen_samples
        self.ground_truth_v = ground_truth_v
        self.seed = seed
        self._x_test = None

        self.dl = DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False,
            generator=torch.Generator().manual_seed(
                _derive_seed(seed, SeedOffsets.EVAL_LOADER)
            ),
        )

        # Private noise stream for the validation loss, re-seeded on every call.
        self._noise_gen = torch.Generator(device=self.device)
        self._val_seed = _derive_seed(seed, SeedOffsets.EVAL_VAL_NOISE)

        self._forward_sim_gen = torch.Generator(device=self.device)
        self._forward_sim_seed = _derive_seed(seed, SeedOffsets.EVAL_FORWARD_SIM)

        self._prior_gen = torch.Generator(device=self.device)
        self._prior_seed = _derive_seed(seed, SeedOffsets.EVAL_PRIOR_NOISE)

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
            generator=torch.Generator().manual_seed(
                _derive_seed(seed, SeedOffsets.EVAL_REF_PERM)
            ),
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
    ) -> Dict[str, float]:

        with torch.random.fork_rng(devices=self._rng_devices):
            return self._evaluate(
                model, backward_model, num_samples, log, use_amp, direction
            )

    def _evaluate(
        self,
        model: nn.Module,
        backward_model: Optional[nn.Module],
        num_samples: Optional[int],
        log: bool,
        use_amp: bool,
        direction: str,
    ) -> Dict[str, float]:
        model.eval()
        if backward_model:
            backward_model.eval()

        # Delegate dataset fetching to _get_test_data unconditionally to ensure
        # it is cached in self._x_test and reused efficiently across evaluations.
        x_true = self._get_test_data()

        # FID / precision / recall / MMD depend on the sample counts, so both sides use
        # exactly N samples. N can't exceed the number of real samples available.
        n_wanted = num_samples if num_samples else len(x_true)
        n_eval = min(n_wanted, len(x_true))
        if n_eval < n_wanted:
            Evaluator.logger.warning(
                f"Requested {n_wanted} samples but the test set only has "
                f"{len(x_true)}; evaluating with {n_eval}."
            )

        loss = None
        sampler = EulerSampler(
            model_type=model.model_type,
            steps=self.sde_steps,
            use_amp=use_amp,
        )

        if direction == "b":
            # Validation & Simulation
            loss = self._compute_validation_loss(model, direction, use_amp)

            # Backward Chain: Prior -> Data
            gen_start = time.time()
            x_gen = sampler.generate(
                model=model,
                shape=self.data_shape,
                n_samples=n_eval,
                device=self.device,
                init_seed=_derive_seed(self.seed, SeedOffsets.EVAL_SOLVER_INIT),
                step_seed=_derive_seed(self.seed, SeedOffsets.EVAL_SOLVER_STEP),
            )
            gen_time = time.time() - gen_start

            # Real reference set of the same size
            x_ref = x_true[self._ref_perm[:n_eval]]
        else:
            # Forward Chain: Data -> Prior
            # 1. Forward Chain: Data -> Prior
            x_init = x_true[self._ref_perm[:n_eval]].to(
                self.device, memory_format=self.memory_format
            )

            gen_start = time.time()
            x_gen = sampler.generate(
                model=model,
                x_init=x_init,
                batch_size=256,
                step_seed=self._forward_sim_seed,
            )
            gen_time = time.time() - gen_start

            # Re-seed prior target generator
            self._prior_gen.manual_seed(self._prior_seed)
            x_ref = torch.randn(
                x_gen.shape,
                device=self.device,
                dtype=x_gen.dtype,
                generator=self._prior_gen,
            ).contiguous(memory_format=self.memory_format)

        # Eval Metrics
        results = {
            "eval_samples": n_eval,
            "eval_NFEs": self.sde_steps,
            "eval_generation_time": gen_time,
        }
        if loss is not None:
            results["eval_loss"] = float(loss)

        if self.is_image_data and direction == "b":
            results.update(get_generative_quality_metrics(x_ref, x_gen))
        else:
            results["eval_MMD"] = float(get_mmd(x_ref, x_gen, device=self.device))

            # Ground Truth
            if self.ground_truth_v and direction == "b":
                results["drift_MSE"] = get_drift_mse(model, self.ground_truth_v, x_gen)

        if log:
            self.logger.info("Evaluation Metrics:")
            for k, v in results.items():
                if isinstance(v, float):
                    self.logger.info(f"{k:>20}: {v:.6f}")
                else:
                    self.logger.info(f"{k:>20}: {v}")

        return results

    def _get_test_data(self) -> torch.Tensor:
        if self._x_test is None:
            self._x_test = torch.cat([z.cpu() for z in self.dl], dim=0)
        return self._x_test

    def _compute_validation_loss(
        self, model: nn.Module, direction: str, use_amp: bool
    ) -> float:
        """Calculates regression health against straight-line/SDE paths and extracts targets."""

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
                        # v_pred = model.u(x_t, t_net)
                        # eps_pred = model.s(x_t, t_net)
                        v_pred = model.u(x_t, t)
                        eps_pred = model.s(x_t, t)
                        v_target, eps_target = target_u

                        loss_v, loss_eps = ConditionalVectorField.compute_sf2m_loss(
                            v_pred, eps_pred, v_target, eps_target
                        )
                        batch_loss = loss_v + loss_eps
                    else:
                        # Standard single-network models (SDE, Flow Matching, Minibatch, IMF)
                        # pred_u = model(x_t, t_net)
                        pred_u = model(x_t, t)
                        batch_loss = ConditionalVectorField.compute_loss(
                            model_type=eval_model_type,
                            pred_u=pred_u,
                            target_u=target_u,
                            t_net=t_net,
                            sigma=sigma,
                        )

                total_loss += batch_loss.item()

        return total_loss / len(self.dl)
