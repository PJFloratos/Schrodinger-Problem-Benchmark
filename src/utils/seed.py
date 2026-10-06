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
    EVAL_FORWARD_SIM = 104
    EVAL_PRIOR_NOISE = 105
    EVAL_SOLVER_INIT = 106
    EVAL_SOLVER_STEP = 107

    # --- Trainer Streams (1000+) ---
    TRAIN_SPLIT = 1000
    TRAIN_SHUFFLE = 1001
    TRAIN_VALID_LOADER = 1002
    TRAIN_NOISE = 1003
    TRAIN_VALID_NOISE = 1004
    TRAIN_EVAL = 1005

    # --- IPF Trainer Streams (1100+) ---
    IPF_SHUFFLE = 1100
    IPF_CACHE_NOISE = 1101
    IPF_CACHE_PERM = 1102
    IPF_EVAL = 1103

    # --- IMF Trainer Streams (1200+) ---
    IMF_SHUFFLE = 1200
    IMF_PAIR_NOISE = 1201
    IMF_PAIR_PERM = 1202
    IMF_BRIDGE_NOISE = 1203
    IMF_PROBE_NOISE = 1204
    IMF_EVAL = 1205

    # --- SF2M Trainer Streams (1300+) ---
    SF2M_SHUFFLE = 1300
    SF2M_CACHE_NOISE = 1301
    SF2M_CACHE_PERM = 1302
    SF2M_BRIDGE_NOISE = 1303
    SF2M_EVAL = 1305


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
