from src.core.solver import EulerSampler
from src.configs import Toy2dConfig, MNISTConfig
from src.utils import text_logger

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
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


def _sample_shape(cfg) -> Tuple[int, ...]:
    """Per-sample shape for the solver. Must match `Evaluator.data_shape`."""
    if isinstance(cfg, Toy2dConfig):
        return (cfg.input_dim,)
    if isinstance(cfg, MNISTConfig):
        return (cfg.input_channels, cfg.image_size, cfg.image_size)
    raise ValueError(f"Unknown configuration type: {type(cfg)}")


@torch.no_grad()
def _generate(generative_model, cfg, n_samples: int) -> torch.Tensor:
    """Sample with the same EulerSampler setup (and seed) the Evaluator uses."""
    sampler = EulerSampler(
        model_type=generative_model.model_type,
        steps=cfg.eval_sim_steps,
        use_amp=cfg.use_amp and cfg.device.type == "cuda",
    )
    return sampler.generate(
        model=generative_model,
        shape=_sample_shape(cfg),
        n_samples=n_samples,
        device=cfg.device,
        seed=cfg.eval_seed,
    )


def generate_and_plot(generative_model, eval_res, cfg):
    """Handles dataset-specific generation and routing to the correct plotter."""
    logger.debug("Simulating trajectories for final plot...")
    generative_model.eval()

    if isinstance(cfg, Toy2dConfig):
        x_gen = _generate(generative_model, cfg, n_samples=cfg.eval_gen_samples)
        x_np = x_gen.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.5, color="blue")
        plt.title("Generated Distribution (t=1)")

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

        if cfg.model_type in ["ipf", "imf"]:
            save_path = os.path.join(
                cfg.plots_path,
                f"{cfg.epochs}ep_{cfg.eval_sim_steps}ss_{cfg.num_iter}it.png",
            )
        else:
            save_path = os.path.join(
                cfg.plots_path, f"{cfg.epochs}ep_{cfg.eval_sim_steps}ss.png"
            )
        plt.savefig(save_path)
        plt.close()

    elif isinstance(cfg, MNISTConfig):
        x_gen = _generate(generative_model, cfg, n_samples=64)

        if cfg.model_type in ["ipf", "imf"]:
            save_path = os.path.join(
                cfg.plots_path,
                f"{cfg.epochs}ep_{cfg.eval_sim_steps}ss_{cfg.num_iter}_grid.png",
            )
        else:
            save_path = os.path.join(
                cfg.plots_path, f"{cfg.epochs}ep_{cfg.eval_sim_steps}ss_grid.png"
            )
        plot_samples(
            x_gen, eval_res, path=save_path, title="MNIST Generated Samples (t=1)"
        )

    else:
        raise ValueError(f"Unknown configuration type: {type(cfg)}")

    logger.debug(f"Plots successfully saved to {save_path}.")
