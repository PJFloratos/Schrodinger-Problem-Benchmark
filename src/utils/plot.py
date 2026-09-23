from src.utils import text_logger
from src.configs import Toy2dConfig, MNISTConfig

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
from typing import Dict


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


def generate_and_plot(generative_model, eval_res, cfg):
    """Handles dataset-specific generation and routing to the correct plotter."""
    logger.debug("Simulating trajectories for final plot...")
    generative_model.eval()

    if isinstance(cfg, Toy2dConfig):
        x_gen = generative_model.generate(
            n_samples=cfg.eval_gen_samples, steps=cfg.eval_sim_steps, device=cfg.device
        )
        x_np = x_gen.cpu().numpy()

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

        if cfg.model_type == "ipf":
            save_path = os.path.join(
                cfg.plots_path, f"{cfg.epochs}ep_{cfg.sim_steps}ss_{cfg.num_iter}it.png"
            )
        else:
            save_path = os.path.join(
                cfg.plots_path, f"{cfg.epochs}ep_{cfg.sim_steps}ss.png"
            )
        plt.savefig(save_path)
        plt.close()

    elif isinstance(cfg, MNISTConfig):
        x_gen = generative_model.generate(
            n_samples=64, steps=cfg.eval_sim_steps, device=cfg.device
        )
        if cfg.model_type == "ipf":
            save_path = os.path.join(
                cfg.plots_path,
                f"{cfg.epochs}ep_{cfg.sim_steps}ss_{cfg.num_iter}_grid.png",
            )
        else:
            save_path = os.path.join(
                cfg.plots_path, f"{cfg.epochs}ep_{cfg.sim_steps}ss_grid.png"
            )
        plot_samples(
            x_gen, eval_res, path=save_path, title="MNIST Generated Samples (t=1)"
        )

    logger.debug(f"Plots successfully saved to {save_path}.")
