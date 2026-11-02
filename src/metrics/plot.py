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
    if m_type == "sde":
        parts.append(f"{cfg.model.get('sim_steps', 0)}ss")
    if m_type in ["flow_m", "minibatch"]:
        parts.remove(f"{cfg.sigma}sig")
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
    elif m_type == "enot":
        parts.append(f"{cfg.model.get("inner_iters", 0)}is")
        parts.append(f"{cfg.model.get("integration_steps", 0)}ss")

    return "_".join(parts)


@torch.no_grad()
def _generate(generative_model, cfg, n_samples: int) -> torch.Tensor:
    """Sample with the same EulerSampler setup (and seed) the Evaluator uses."""
    sampler = EulerSampler(
        model_type=generative_model.model_type,
        steps=cfg.eval_sim_steps,
        use_amp=cfg.use_amp and cfg.device == "cuda",
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


def generate_and_plot(generative_model, eval_res, cfg, epoch=None, n_samples=None):
    """Handles dataset-specific generation and routing to the correct plotter."""
    logger.debug("Simulating trajectories for plot...")
    generative_model.eval()

    # Build the experiment-aware save path
    signature = _build_experiment_signature(cfg)
    if epoch:
        save_path = os.path.join(cfg.plots_path, f"{epoch}__{signature}.png")
    else:
        save_path = os.path.join(cfg.plots_path, f"{signature}.png")

    # Duck-typing: Does the dataset define image channels?
    is_image = "input_channels" in cfg.dataset

    if not is_image:
        if not n_samples:
            n_samples = cfg.eval_gen_samples
        x_gen = _generate(generative_model, cfg, n_samples=n_samples)
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

    logger.debug(f"Plot successfully saved to {save_path}.")


@torch.no_grad()
def visualize_energy_landscape(model, real_data, buffer_data, device, save_path):
    # Create a dense 2D grid covering the Swiss Roll domain
    grid_size = 100
    x = torch.linspace(-2.5, 2.5, grid_size)
    y = torch.linspace(-2.5, 2.5, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)

    # Enable gradients strictly for the grid to extract the vector field
    grid.requires_grad_(True)
    with torch.enable_grad():
        psi = model(grid)
        grad_psi = torch.autograd.grad(outputs=psi.sum(), inputs=grid)[0]

    psi = psi.detach().cpu().reshape(grid_size, grid_size)
    grad_psi = grad_psi.detach().cpu().reshape(grid_size, grid_size, 2)

    plt.figure(figsize=(10, 8))

    # 1. Plot the scalar potential landscape
    contour = plt.contourf(
        xx.numpy(), yy.numpy(), psi.numpy(), levels=50, cmap="viridis", alpha=0.6
    )
    plt.colorbar(contour, label="Learned Potential $\psi_\\theta$")

    # 2. Overlay the vector field
    skip = 4
    plt.quiver(
        xx.numpy()[::skip, ::skip],
        yy.numpy()[::skip, ::skip],
        grad_psi[:, :, 0].numpy()[::skip, ::skip],
        grad_psi[:, :, 1].numpy()[::skip, ::skip],
        color="white",
        alpha=0.5,
    )

    # 3. Overlay the exact locations of the real data and the trapped MCMC particles
    plt.scatter(
        real_data[:, 0].cpu(),
        real_data[:, 1].cpu(),
        c="red",
        s=1,
        alpha=0.5,
        label="Target (Real)",
    )
    plt.scatter(
        buffer_data[:, 0].cpu(),
        buffer_data[:, 1].cpu(),
        c="blue",
        s=1,
        alpha=0.5,
        label="MCMC (Fake)",
    )

    plt.legend()
    plt.title("Energy Landscape & MCMC Particle Distribution")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save and close
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.debug(f"Plot successfully saved to {save_path}.")


@torch.no_grad()
def visualize_enot_landscape(
    potential_model,
    drift_model,
    real_data,
    generated_data,
    device,
    save_path,
    t_val=0.99,
):
    # Create a dense 2D grid covering the Swiss Roll domain
    grid_size = 100
    x = torch.linspace(-2.5, 2.5, grid_size)
    y = torch.linspace(-2.5, 2.5, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)

    # 1. Evaluate the Discriminator Potential (beta) over the space
    beta_grid = potential_model(grid).squeeze(-1)
    beta_grid = beta_grid.cpu().reshape(grid_size, grid_size)

    # 2. Evaluate the Generator Drift Field at terminal time (t ~ 1.0)
    # We use 0.99 to avoid any potential SDE endpoint singularities
    t_tensor = torch.full((grid.shape[0], 1), t_val, device=device)
    drift_grid = drift_model(grid, t_tensor)
    drift_grid = drift_grid.cpu().reshape(grid_size, grid_size, 2)

    plt.figure(figsize=(10, 8))

    # Plot the Discriminator's scalar potential landscape
    contour = plt.contourf(
        xx.numpy(), yy.numpy(), beta_grid.numpy(), levels=50, cmap="viridis", alpha=0.6
    )
    plt.colorbar(contour, label="Discriminator Potential $\\beta_\\phi(y)$")

    # Overlay the Generator's vector field (Drift at t_val)
    skip = 4
    plt.quiver(
        xx.numpy()[::skip, ::skip],
        yy.numpy()[::skip, ::skip],
        drift_grid[:, :, 0].numpy()[::skip, ::skip],
        drift_grid[:, :, 1].numpy()[::skip, ::skip],
        color="white",
        alpha=0.5,
        label=f"Drift $f_\\theta$ at t={t_val}",
    )

    # Overlay the exact locations of the real data and the terminal MCMC particles
    plt.scatter(
        real_data[:, 0].cpu(),
        real_data[:, 1].cpu(),
        c="red",
        s=1,
        alpha=0.5,
        label="Target (Real $Y$)",
    )
    plt.scatter(
        generated_data[:, 0].cpu(),
        generated_data[:, 1].cpu(),
        c="blue",
        s=1,
        alpha=0.5,
        label="Generated ($X_N$)",
    )

    plt.legend()
    plt.title("ENOT Discriminator Landscape & Terminal Drift")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save and close
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.debug(f"Plot successfully saved to {save_path}.")
