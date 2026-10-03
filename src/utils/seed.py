import random
import numpy as np
import torch
import os


class SeedOffsets:
    """Central registry for RNG stream offsets to prevent collisions."""

    # --- Pipeline Init (0-99) ---
    GEN_INIT = 0
    AUX_INIT = 1

    # --- Evaluator Streams (100-199) ---
    # Shifted to the 100 block to guarantee no overlap with Pipeline
    EVAL_VAL_NOISE = 101
    EVAL_LOADER = 102
    EVAL_REF_PERM = 103

    # --- Trainer Streams (1000+) ---
    TRAIN_SPLIT = 1000
    TRAIN_SHUFFLE = 1001
    TRAIN_VALID_LOADER = 1002
    TRAIN_NOISE = 1003
    TRAIN_VALID_NOISE = 1004
    TRAIN_EVAL = 1005


def set_all_seeds(seed: int):
    """Locks down all RNGs and forces deterministic hardware execution."""
    # 1. Python built-in pseudo-random generator
    random.seed(seed)

    # 2. NumPy pseudo-random generator
    np.random.seed(seed)

    # 3. PyTorch CPU and GPU generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # # 4. Force cuDNN to behave deterministically (Warning: reduces speed slightly)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
