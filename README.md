# Schrodinger-Problem-Benchmark

## Codebase Architecture & Pipeline

This repository is built with a strong emphasis on **modularity, separation of concerns, and scalability**.

### Directory Structure

```text
├── train.py                     # High-level pipeline orchestrator
└── src/
    ├── configs/                 # Object-oriented, strongly-typed experiment configurations
    │   ├── base_config.py       
    │   ├── mnist_config.py      
    │   └── toy_2d_config.py     
    ├── datasets/                # Dataset definitions and pre-processing
    │   ├── mnist_dataset.py     
    │   └── toy_2d_dataset.py    
    ├── metrics/                 # Evaluation logic and mathematical distance calculations
    │   ├── distances.py         
    │   └── evaluator.py         
    ├── models/                  # Neural network architectures
    │   ├── unet.py              
    │   └── velocity_mlp.py      
    ├── training/                # Training loops and methodology routers
    │   ├── ipf_trainer.py       
    │   ├── trainer.py           
    │   └── training_orchestrator.py
    └── utils/                   # Helpers: Logging, EMA, plotting, saving, and data routing
        ├── data.py              
        ├── ema.py               
        ├── log.py               
        ├── plot.py              
        └── save.py              

```

* **`train.py` (The Entry Point):** A purely declarative script. It contains no raw mathematical operations or model instantiation logic. It simply requests a configuration, asks the Orchestrator for a trained model, and passes it to the Evaluator.

* **`src/configs/` (Configuration Management):** Managed via Python `dataclasses`. `BaseConfig` holds all shared hyper-parameters (hardware, standard batch sizes, epochs), while dataset-specific subclasses (e.g., `MNISTConfig`, `Toy2dConfig`) override necessary values (like learning rates or input channels). This guarantees type safety and dynamic path resolution for saving artifacts.

* **`src/training/` (The Strategy Hub):**
* **`TrainingOrchestrator`**: Acts as a Factory. It reads the active configuration and dynamically instantiates the correct model architecture (`VelocityMLP` vs `SimpleUNet`) and the correct optimizer.

* **Methodology Routing**: Depending on the `model_type` flag, the Orchestrator routes execution to either the alternating loop of `IPFTrainer` or the standard continuous-time regression of `Trainer`. Future methodologies (like Independent Marginal Fitting - IMF) can be added here without altering `train.py`.

* **`Trainer.py`**: A unified standard trainer that seamlessly handles standard SDEs, Flow Matching, and Minibatch Optimal Transport ODEs by conditionally adjusting the target vector field and loss weighting dynamically during the forward pass.

* **`src/models/` (Architectures):** Contains a `VelocityMLP` for 2D coordinate regression and a `SimpleUNet` equipped with Sinusoidal Time Embeddings for image generation. Both models share a unified `generate()` signature that handles Euler-Maruyama SDE integration or standard ODE integration depending on the internal `model_type` state.

* **`src/metrics/` (Evaluation):**
  * **`Evaluator`**: Dedicated entirely to pipeline logic—iterating over test dataloaders, calculating the Conditional Generator Matching (CGM) loss, and simulating generative trajectories.

  * **`distances.py`**: A pure mathematics module isolated from PyTorch data loaders. It handles heavy computations like the Maximum Mean Discrepancy (MMD) with median-heuristic RBF kernels and Wasserstein distances.

* **`src/utils/` (Shared Utilities):** Includes centralized Exponential Moving Average (`EMAHelper`) tracking for model weights, robust artifact saving (`save.py`), and dynamic grid/scatter visualization routing (`plot.py`).


---

## How to Run the Experiments

Running an experiment is entirely configuration-driven. You do not need to modify the training loops or orchestration logic to launch different methodologies.

**Step 1: Select the Target Dataset**
Open `train.py` and set the `ACTIVE_DATASET` flag to your desired dataset at the top of the file:

```python
# train.py
class DatasetType(str, Enum):
    TOY2D = "toy2d"
    MNIST = "mnist"

# Pick the dataset to train on
ACTIVE_DATASET = DatasetType.TOY2D

```

**Step 2: Configure the Methodology and Hyperparameters**
Global training parameters (epochs, steps, learning rate) and the generative methodology are controlled in `src/configs/base_config.py`. Change the `model_type` to switch between algorithms:

```python
# src/configs/base_config.py
    # ...
    # --- Training ---
    model_type: str = "sde"  # Options: "sde", "minibatch", "flow_m", "ipf"
    model_name: str = "SDE"
    epochs: int = 5
    batch_size: int = 512
    ...

```

For dataset-specific parameters (like the 2D shape type or image channels), open the corresponding child configuration file (e.g., `src/configs/toy_2d_config.py`):

```python
# src/configs/toy_2d_config.py
    # ...
    dataset_name: str = "Checkerboard"
    dataset_type: str = "checkerboard"  # Options: "swiss_roll", "moons", "checkerboard"
    ...
```

**Step 3: Execute the Pipeline**
Once your configurations are set, simply run the entry point from your terminal. The Orchestrator will automatically route your configuration to the correct dataset, model architecture, and training strategy.

```bash
python train.py

```

Artifacts (saved models, training logs, and generated distribution plots) will be dynamically routed and saved in `./models`, `./logs`, and `./plots` under subdirectories named after your chosen dataset and model type.
