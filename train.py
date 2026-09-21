# https://github.com/JTT94/diffusion_schrodinger_bridge/

from src.datasets.Toy2DDataset import Toy2DDataset
from src.datasets.MNISTDataset import MNISTDataset
from src.models.VelocityMLP import VelocityMLP
from src.models.UNet import SimpleUNet
from src.training.Trainer import Trainer
from src.training.IPFTrainer import IPFTrainer
from src.metrics.Evaluator import Evaluator
from src.utils import configure_logger

import torch
from torch import nn, optim
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
from typing import Dict


EPOCHS = 500
SIM_STEPS = 30
MODEL_TYPE = "sde"
NUM_ITER = 5000  # Inner gradient steps per cache (matches official default)
EVAL_SIM_STEPS = 30  # Smooth evaluation paths


logger = configure_logger(__name__)

# The configuration dictionary
config = dict(
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    DATASETS_PATH=None,
    PLOTS_PATH="./plots/SwissRoll",
    MODELS_PATH="./models/SwissRoll",
    LOGS_PATH="./logs/SwissRoll",
    BATCH_SIZE=512,
    EPOCHS=EPOCHS,
    LEARNING_RATE=1e-3,
    WEIGHT_DECAY=0.0001,
)


def evaluate(
    model: nn.Module, dataset: torch.utils.data.Dataset, loss_fn: nn.Module, file: str
) -> Dict[str, float]:
    evaluator = Evaluator(
        model=model,
        test_ds=dataset,
        criterion=loss_fn,
        device=config["DEVICE"],
        sde_steps=SIM_STEPS,
    )
    logger.info("The evaluator is created.")

    eval_res = evaluator.evaluate()

    os.makedirs(config["LOGS_PATH"], exist_ok=True)
    with open(os.path.join(config["LOGS_PATH"], file), "w") as f:
        f.write(str(eval_res))

    return eval_res


def plot_samples(
    x_gen: torch.Tensor, metrics: Dict[str, float], path: str, nrow: int = 8
) -> None:
    """Save a large grid of generated images with the evaluation metrics as a caption.

    x_gen: (N, 1, H, W) in [-1, 1]. `metrics` is any {name: value} dict, so metrics
    added to the Evaluator later show up automatically.
    """
    # [-1, 1] -> [0, 1]; clamp because generated values can overshoot the range
    imgs = ((x_gen.detach().cpu() + 1) / 2).clamp(0, 1)
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=1.0)[0]  # 1 channel

    fig, ax = plt.subplots(figsize=(10, 10.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)
    ax.imshow(grid.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(
        f"{EPOCHS} epochs, {EVAL_SIM_STEPS} steps",
        fontsize=18,
        pad=14,
    )

    # Metrics go below the grid (not on top of it) so they never hide a digit
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


def main() -> None:
    os.makedirs(config["MODELS_PATH"], exist_ok=True)
    os.makedirs(config["PLOTS_PATH"], exist_ok=True)

    train_dataset = Toy2DDataset(n_samples=10000, dataset_type="swiss_roll")
    test_dataset = Toy2DDataset(n_samples=2000, dataset_type="swiss_roll")
    # train_dataset = MNISTDataset(train=True)
    # test_dataset = MNISTDataset(train=False)
    loss_fn = nn.MSELoss()

    if MODEL_TYPE == "ipf":
        # IPF requires two separate networks (forward and backward)
        f_model = VelocityMLP(d=2, hidden=128, model_type="sde").to(config["DEVICE"])
        b_model = VelocityMLP(d=2, hidden=128, model_type="sde").to(config["DEVICE"])
        # f_model = SimpleUNet(model_type="sde").to(config["DEVICE"])
        # b_model = SimpleUNet(model_type="sde").to(config["DEVICE"])
        logger.info(f'IPF Models deployed on: {config["DEVICE"]}')

        f_opt = optim.AdamW(
            f_model.parameters(),
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
        )
        b_opt = optim.AdamW(
            b_model.parameters(),
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
        )

        trainer = IPFTrainer(
            forward_model=f_model,
            backward_model=b_model,
            dataset=train_dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=config["DEVICE"],
            batch_size=config["BATCH_SIZE"],
            sde_steps=SIM_STEPS,
            num_cache_batches=10,  # Caches ~2500 trajectories per iteration
        )

        logger.info("IPF Trainer is created successfully. Starting alternating loop...")
        trainer.fit(ipf_iterations=config["EPOCHS"], inner_iterations=NUM_ITER)

        # In IPF, the backward model handles generation from Prior -> Data
        generative_model = b_model

    else:
        # Instantiate Model
        model = VelocityMLP(d=2, hidden=128, model_type=MODEL_TYPE).to(config["DEVICE"])
        # model = SimpleUNet(model_type=MODEL_TYPE).to(config["DEVICE"])
        logger.info(f'Model deployed on: {config["DEVICE"]}')

        opt = optim.AdamW(
            model.parameters(),
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
            fused=True,
        )

        trainer = Trainer(
            model=model,
            dataset=train_dataset,
            batch_size=config["BATCH_SIZE"],
            opt=opt,
            device=config["DEVICE"],
        )

        logger.info("Trainer is created succesfully.")

        # Train the model
        train_res = trainer.fit(
            epochs=config["EPOCHS"], save_per=500, save_path=config["MODELS_PATH"]
        )

        generative_model = model

    # --- Verification Step: Integrated Evaluation ---
    logger.info("Running post-training evaluation...")

    # We pass the trained model directly without reloading weights from disk
    eval_results = evaluate(
        generative_model,
        test_dataset,
        loss_fn,
        file=f"{EPOCHS}ep_{SIM_STEPS}ss{NUM_ITER}it.txt",
    )
    logger.info(f"Final Evaluation Results: {eval_results}")

    # --- Verification Step: SDE/ODE Sampling ---
    logger.info("Simulating trajectories for final plot...")
    generative_model.eval()

    # Generate directly from the model
    x_gen = generative_model.generate(
        n_samples=4000, steps=EVAL_SIM_STEPS, device=config["DEVICE"]
    )
    x_np = x_gen.cpu().numpy()

    # Plot results with metrics embedded
    plt.figure(figsize=(8, 8))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.5, color="blue")
    plt.title("Generated Distribution (t=1)")

    # Format the evaluation metrics into a text box
    metrics_text = (
        f"CGM Loss: {eval_results['CGM_Loss']:.4f}\n"
        f"MMD: {eval_results['MMD']:.6f}\n"
        # f"W-Dist (X): {eval_results['Wasserstein_X']:.4f}\n"
        # f"W-Dist (Y): {eval_results['Wasserstein_Y']:.4f}"
    )

    # Add the text box to the plot (placed in the upper right corner)
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

    plt.savefig(os.path.join(config["PLOTS_PATH"], f"{EPOCHS}ep_{SIM_STEPS}ss.png"))
    #
    # logger.info("Simulating trajectories for final plot...")
    # generative_model.eval()
    #
    # # Generate 64 images for an 8x8 grid
    # x_gen = generative_model.generate(
    #     n_samples=64, steps=EVAL_SIM_STEPS, device=config["DEVICE"]
    # )
    #
    # plot_samples(
    #     x_gen,
    #     eval_results,
    #     os.path.join(
    #         config["PLOTS_PATH"], f"{EPOCHS}ep_{SIM_STEPS}ss_{NUM_ITER}it.png"
    #     ),
    #     nrow=8,
    # )


if __name__ == "__main__":
    main()
