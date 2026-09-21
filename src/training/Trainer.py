from src.utils.save import save_model
from src.utils.log import configure_logger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from scipy.optimize import linear_sum_assignment

import os
from tqdm import tqdm
from timeit import default_timer as timer
from typing import Callable, Tuple, Dict, Union


class Trainer:
    logger = configure_logger(__name__)

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int,
        opt: torch.optim.Optimizer,
        train_prop: float = 0.8,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device, non_blocking=True)
        self.dataset = dataset
        self.batch_size = batch_size
        self.opt = opt
        self.train_prop = train_prop
        self.device = device

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds, valid_ds = random_split(
            self.dataset, [self.train_prop, 1 - self.train_prop]
        )

        train_dl = DataLoader(
            train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        valid_dl = DataLoader(valid_ds, self.batch_size, num_workers=0, pin_memory=True)

        return train_dl, valid_dl

    def _process_data_loaders(self, dl: DataLoader) -> Tuple[float, float]:
        # Initialize batch loss and accuracy
        batch_loss, batch_eval = 0.0, 0.0
        phase = "Training Step" if self.model.training else "Validation Step"

        for z_batch in tqdm(dl, ascii=True, desc=f"             {phase}"):
            z_batch = z_batch.to(self.device, non_blocking=True)
            # B, d = z_batch.shape
            B = z_batch.shape[0]

            # Sample time uniformly t ~ U[0, 1]. Cap at 0.999 to avoid div by zero.
            t = torch.rand(B, 1, device=self.device) * 0.999

            # For 4D images (B, C, H, W), t_expand becomes (B, 1, 1, 1)
            t_expand = t.view(B, *([1] * (z_batch.ndim - 1)))

            # Check if training using Minibatch OT (straight lines) or not
            if self.model.model_type == "minibatch":
                # Sample from the standard Gaussian prior[cite: 1, 2]
                x_0 = torch.randn_like(z_batch)

                # Calculate the squared Euclidean cost matrix between prior and data
                # CPU Bottleneck Warning: This takes O(N^3) time.
                # cost_matrix = torch.cdist(x_0, z_batch, p=2).pow(2)
                cost_matrix = torch.cdist(
                    x_0.view(B, -1), z_batch.view(B, -1), p=2
                ).pow(2)

                # Solve the assignment problem on CPU (highly optimized in SciPy)
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

                # Permute the batches to align them according to the optimal transport plan
                x_0 = x_0[row_ind]
                z_batch = z_batch[col_ind]

                # Construct deterministic conditional path
                x_t = t_expand * z_batch + (1.0 - t_expand) * x_0

                # Construct exact closed-form target vector field
                target_u = (z_batch - x_t) / (1.0 - t_expand)

            elif self.model.model_type == "sde":
                # Independent endpoints relaxation: Prior is standard Gaussian
                eps = torch.randn_like(z_batch)

                # Construct conditional path: mu_t(x | z) = N(t*z, (1-t)*I)
                x_t = t_expand * z_batch + torch.sqrt(1.0 - t_expand) * eps

                # Construct exact closed-form target vector field
                target_u = (z_batch - x_t) / (1.0 - t_expand)

            elif self.model.model_type == "flow_m":
                # Prior is standard Gaussian
                x_0 = torch.randn_like(z_batch)

                # Minibatch Optimal Transport (Optional for standard FM, but highly recommended)
                # cost_matrix = torch.cdist(x_0, z_batch, p=2).pow(2)
                # cost_matrix = torch.cdist(
                #     x_0.view(B, -1), z_batch.view(B, -1), p=2
                # ).pow(2)
                # row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                # x_0 = x_0[row_ind]
                # z_batch = z_batch[col_ind]

                # Construct Flow Matching deterministic path
                x_t = t_expand * z_batch + (1.0 - t_expand) * x_0

                # 5. Standard Flow Matching Target (Constant Velocity)
                target_u = z_batch - x_0

            # Forward pass (predict velocity)
            pred_u = self.model(x_t, t)

            # Calculate Time-Weighted CGM Loss
            loss = torch.mean((1.0 - t_expand) * (pred_u - target_u) ** 2)

            batch_loss += loss.item()

            # Dummy eval since accuracy doesn't apply to vector fields
            batch_eval += 0.0

            if self.model.training:
                self.opt.zero_grad()
                loss.backward()
                # Activate gradient clipping to stabilize the t -> 1 singularity
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

        return batch_loss / len(dl), batch_eval / len(dl)

    def _training_step(self, train_dl: DataLoader) -> Tuple[float, float]:
        """
        Performs a single training step over the training DataLoader.
        """
        self.model.train()
        train_loss, train_eval = self._process_data_loaders(train_dl)
        self.model.eval()

        return train_loss, train_eval

    def _validation_step(self, valid_dl: DataLoader) -> Tuple[float, float]:
        """
        Performs a single validation step over the validation DataLoader.
        """
        self.model.eval()
        with torch.inference_mode():
            valid_loss, valid_eval = self._process_data_loaders(valid_dl)

        return valid_loss, valid_eval

    def fit(
        self,
        epochs: int,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
    ) -> Dict:
        start_time = timer()
        train_losses, train_evals = [], []
        valid_losses, valid_evals = [], []

        Trainer.logger.info("Start Training Process...")

        train_dl, valid_dl = self._get_loaders()

        for epoch in range(1, epochs + 1):
            Trainer.logger.info(f"-> Epoch: {epoch}/{epochs}")

            # Training and Evaluating the Model
            train_loss, train_eval = self._training_step(train_dl)
            valid_loss, valid_eval = self._validation_step(valid_dl)

            Trainer.logger.info(
                f"    Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}"
            )

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # Saving the model
            if save_per and save_path and (epoch % save_per == 0):
                save_model(
                    self.model,
                    f"{save_path}/{self.model.__class__.__name__}_checkpoint_{epoch}.pth",
                )

            Trainer.logger.info(("-" * 100))

        Trainer.logger.info("Training Process Completed Successfully.")

        return {
            "train_loss": train_losses,
            "valid_loss": valid_losses,
            "total_time": timer() - start_time,
        }
