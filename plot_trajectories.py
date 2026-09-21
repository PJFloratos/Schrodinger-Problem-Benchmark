from src.dataset.Toy2DDataset import Toy2DDataset
from src.models.VelocityMLP import VelocityMLP
from src.utils import load_model
import torch
import matplotlib.pyplot as plt
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_paths():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = dict(d=2, hidden=128, time_dim=64)

    # Load your trained models
    sde_model = load_model(
        model_class=VelocityMLP,
        model_path="./models/SDE_VelocityMLP_checkpoint_2000.pth",
        device=DEVICE,
        model_type="sde",
        **model_kwargs
    )
    sde_model.eval()

    ode_model = load_model(
        model_class=VelocityMLP,
        model_path="./models/ODE_VelocityMLP_checkpoint_200.pth",
        device=DEVICE,
        model_type="minibach",
        **model_kwargs
    )
    ode_model.eval()

    # Get some ground truth data for the background
    ds = Toy2DDataset(n_samples=2000, dataset_type="swiss_roll")
    bg_data = ds.data.numpy()

    # Generate 50 trajectories
    n_tracks = 50
    steps = 100

    ode_paths = ode_model.generate(
        n_samples=n_tracks, steps=steps, device=DEVICE, return_path=True
    )
    sde_paths = sde_model.generate(
        n_samples=n_tracks,
        steps=steps,
        device=DEVICE,
        return_path=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Plot ODE Paths ---
    axes[0].scatter(bg_data[:, 0], bg_data[:, 1], s=1, color="lightgray", alpha=0.5)
    for i in range(n_tracks):
        axes[0].plot(
            ode_paths[i, :, 0], ode_paths[i, :, 1], color="blue", alpha=0.6, linewidth=1
        )
        axes[0].scatter(
            ode_paths[i, 0, 0], ode_paths[i, 0, 1], color="red", s=10
        )  # Start point
    axes[0].set_title("ODE Generation (Deterministic)")
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-2.5, 2.5)

    # --- Plot SDE Paths ---
    axes[1].scatter(bg_data[:, 0], bg_data[:, 1], s=1, color="lightgray", alpha=0.5)
    for i in range(n_tracks):
        axes[1].plot(
            sde_paths[i, :, 0],
            sde_paths[i, :, 1],
            color="orange",
            alpha=0.6,
            linewidth=1,
        )
        axes[1].scatter(
            sde_paths[i, 0, 0], sde_paths[i, 0, 1], color="red", s=10
        )  # Start point
    axes[1].set_title("SDE Generation (Stochastic)")
    axes[1].set_xlim(-2.5, 2.5)
    axes[1].set_ylim(-2.5, 2.5)

    os.makedirs("./plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("./plots/trajectory_comparison.png")
    print("Saved trajectory comparison to ./plots/trajectory_comparison.png")


if __name__ == "__main__":
    plot_paths()
