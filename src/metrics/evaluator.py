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
        if isinstance(item, (list, tuple)):
            item = item[0]
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
    ) -> Dict[str, float]:

        with torch.random.fork_rng(devices=self._rng_devices):
            return self._evaluate(model, backward_model, num_samples, log, use_amp)

    def _evaluate(
        self,
        model: nn.Module,
        backward_model: Optional[nn.Module],
        num_samples: Optional[int],
        log: bool,
        use_amp: bool,
    ) -> Dict[str, float]:
        Evaluator.logger.debug("Starting Evaluation Process...")

        model.eval()
        if backward_model:
            backward_model.eval()

        # Validation & Simulation
        loss, x_true = self._compute_validation_loss(model)

        # FID / precision / recall / MMD depend on the sample counts, so both sides use
        # exactly N samples. N can't exceed the number of real samples available.
        n_wanted = num_samples if num_samples else len(x_true)
        n_eval = min(n_wanted, len(x_true))
        if n_eval < n_wanted:
            Evaluator.logger.warning(
                f"Requested {n_wanted} samples but the test set only has "
                f"{len(x_true)}; evaluating with {n_eval}."
            )

        x_gen, gen_time = self._simulate_paths(
            model,
            num_samples=n_eval if num_samples else len(x_true),
            use_amp=use_amp,
        )

        # Real reference set of the same size: a fixed subset, identical on every call.
        # (eval_loss above still uses the full test set.)
        x_ref = x_true[self._ref_perm[:n_eval]]

        # Eval Metrics
        Evaluator.logger.debug("Calculating Evaluation Metrics...")
        results = {
            "eval_loss": float(loss),
            "eval_samples": n_eval,
            "eval_NFEs": self.sde_steps,
            "eval_generation_time": gen_time,
        }

        if self.is_image_data:
            results.update(get_generative_quality_metrics(x_ref, x_gen))
        else:
            results["eval_MMD"] = float(get_mmd(x_ref, x_gen, device=self.device))

            # Ground Truth
            if self.ground_truth_v:
                results["drift_MSE"] = get_drift_mse(model, self.ground_truth_v, x_gen)

        Evaluator.logger.debug("Evaluation Process Completed Successfully.")

        if log:
            self.logger.info("Evaluation Metrics:")
            for k, v in results.items():
                if isinstance(v, float):
                    self.logger.info(f"{k:>20}: {v:.6f}")
                else:
                    self.logger.info(f"{k:>20}: {v}")

        return results

    def _compute_validation_loss(self, model: nn.Module) -> Tuple[float, torch.Tensor]:
        """Calculates regression health against straight-line/SDE paths and extracts targets."""

        gen = self._noise_gen
        # Re-seed on every call: each evaluation sees the exact same (t, noise) draws,
        # so eval_loss only changes when the weights change.
        gen.manual_seed(self._val_seed)

        total_loss = 0.0
        x_true_list = []

        with torch.inference_mode():
            for z_batch in tqdm(self.dl, ascii=True, desc="    Calculating Loss"):
                z_batch = z_batch.to(
                    self.device, non_blocking=True, memory_format=self.memory_format
                )
                B = z_batch.shape[0]

                # Sample time uniformly
                t = torch.rand(B, 1, device=self.device, generator=gen) * 0.999

                x_t, target_u, t_expand = (
                    ConditionalVectorField.get_interpolant_and_target(
                        model_type=model.model_type,
                        z_batch=z_batch,
                        t=t,
                        gen=gen,
                        memory_format=self.memory_format,
                    )
                )

                # Model predicts the vector field
                pred_u = model(x_t, t)

                # Calculate Loss based on model dynamics
                batch_loss = ConditionalVectorField.compute_loss(
                    model_type=model.model_type,
                    pred_u=pred_u,
                    target_u=target_u,
                    t_expand=t_expand,
                )

                total_loss += batch_loss.item()

                x_true_list.append(z_batch.cpu())

        return total_loss / len(self.dl), torch.cat(x_true_list, dim=0)

    def _simulate_paths(
        self, model: nn.Module, num_samples: int, use_amp: bool
    ) -> Tuple[torch.Tensor, float]:
        """Handles trajectory simulation and generation timing."""
        gen_start = time.time()

        # Overwrite the global n_gen_samples
        if num_samples:
            n_samples = num_samples
        else:
            n_samples = self.n_gen_samples

        # Instantiate the decoupled solver
        sampler = EulerSampler(
            model_type=model.model_type,
            steps=self.sde_steps,
            use_amp=use_amp,
        )

        # Seeded => same x_T (and SDE step noise) at every evaluation and for every
        # solver type, so metrics differ only because the weights differ.
        x_gen_tensor = sampler.generate(
            model=model,
            shape=self.data_shape,
            n_samples=n_samples,
            device=self.device,
            seed=self.seed,
        )

        return x_gen_tensor, time.time() - gen_start
