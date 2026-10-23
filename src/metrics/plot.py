from src.core.solver import EulerSampler
from src.utils.seed import derive_seed
from src.utils import text_logger

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
from omegaconf import DictConfig
from typing import Dict, Tuple


logger = text_logger(__name__)


def plot_samples(
    x_gen: torch.Tensor, metrics: Dict[str, float], path: str, title: str, nrow: int = 8
) -> None:
    """Save a large grid of generated images with evaluation metrics as a caption."""
    imgs = ((x_gen.detach().cpu() + 1) / 2).clamp(0, 1)
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=1.0)[0]

    fig, ax = plt.subplots(figsize=(10, 10.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)
    ax.imshow(grid.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=14)

    metrics_text = "   |   ".join(
        f"{name.replace('_', ' ')}: {value:.4g}" for name, value in metrics.items()
    )
    fig.text(
        0.5,
        0.01,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sample_shape(cfg: DictConfig) -> Tuple[int, ...]:
    """Per-sample shape for the solver based on the active dataset configuration."""
    if "input_dim" in cfg.dataset:
        return (cfg.dataset.input_dim,)
    if "input_channels" in cfg.dataset:
        return (
            cfg.dataset.input_channels,
            cfg.dataset.image_size,
            cfg.dataset.image_size,
        )
    raise ValueError(
        "Dataset configuration must specify either 'input_dim' or 'input_channels'."
    )


def _build_experiment_signature(cfg: DictConfig) -> str:
    """Constructs a unified, dynamic filename based on the active experiment parameters."""
    parts = [f"{cfg.sigma}sig", f"{cfg.epochs}ep"]

    m_type = cfg.model.type
    if m_type in ["sde", "flow_m", "ODE"]:
        parts.append(f"{cfg.model.get('sim_steps', 0)}ss")
    elif m_type in ["ipf", "imf", "sf2m"]:
        parts.append(f"{cfg.model.get('num_iter', 0)}it")
        parts.append(f"{cfg.model.get('sim_steps', 0)}ss")
    elif m_type == "sf2m":
        parts.append(f"{cfg.model.get('ot_method', 'minibatch')}")
        parts.append(f"{cfg.model.get('sim_steps', 0)}ss")
    elif m_type == "lightsbm":
        parts.append(f"{cfg.model.get('n_potentials', 0)}p")
        parts.append(f"{cfg.model.get('sim_steps', 0)}ss")
    elif m_type in ["ebm", "scone"]:
        parts.append(f"{cfg.model.get('ula_steps', 0)}ula")
        parts.append(f"{cfg.model.get('ula_step_size', 0)}ss")

    return "_".join(parts)


@torch.no_grad()
def _generate(generative_model, cfg, n_samples: int) -> torch.Tensor:
    """Sample with the same EulerSampler setup (and seed) the Evaluator uses."""
    sampler = EulerSampler(
        model_type=generative_model.model_type,
        steps=cfg.eval_sim_steps,
        use_amp=cfg.use_amp and cfg.device.type == "cuda",
    )

    solver_kwargs = {}
    if cfg.model.type in ["ebm", "scone"]:
        solver_kwargs = {
            "sigma": cfg.sigma,
            "eta_max_infer": cfg.model.get("ula_step_size", 1e-3),
            "eta_min_infer": cfg.model.get("eta_min_infer", 1e-4),
            "spatial_clip": cfg.model.get("spatial_clip", 10.0) / (cfg.sigma**2),
            "ebm_steps": cfg.model.get("ebm_steps", 1000),
        }

    # Derive the device dynamically from the model
    device = next(generative_model.parameters()).device

    return sampler.generate(
        model=generative_model,
        shape=_sample_shape(cfg),
        device=cfg.device,
        n_samples=n_samples,
        init_seed=derive_seed(cfg.eval_seed, "plot_solver_init"),
        step_seed=derive_seed(cfg.eval_seed, "plot_solver_step"),
        **solver_kwargs,
    )


def generate_and_plot(generative_model, eval_res, cfg):
    """Handles dataset-specific generation and routing to the correct plotter."""
    logger.debug("Simulating trajectories for final plot...")
    generative_model.eval()

    # Build the experiment-aware save path
    signature = _build_experiment_signature(cfg)
    save_path = os.path.join(cfg.plots_path, f"{signature}.png")

    # Duck-typing: Does the dataset define image channels?
    is_image = "input_channels" in cfg.dataset

    if not is_image:
        x_gen = _generate(generative_model, cfg, n_samples=cfg.eval_gen_samples)
        x_np = x_gen.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.5, color="blue")
        plt.title(f"{cfg.dataset.name} Generated Distribution (t=1)")

        metrics_text = f"Loss: {eval_res.get('eval_loss', 0):.4f}\nMMD: {eval_res.get('eval_MMD', 0):.6f}\n"
        plt.text(
            0.95,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Save and close the figure
        plt.savefig(save_path)
        plt.close()
    else:
        # Standard 8x8 grid for image domains
        x_gen = _generate(generative_model, cfg, n_samples=64)
        plot_samples(
            x_gen,
            eval_res,
            path=save_path,
            title=f"{cfg.dataset.name} Generated Samples (t=1)",
        )

    logger.debug(f"Plots successfully saved to {save_path}.")
