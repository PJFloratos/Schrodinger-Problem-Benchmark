from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple, Union


def format_ref_params(cfg: DictConfig) -> str:
    """Formats <ref_params> directory name (e.g., 'brownian_s1', 'ou_g1')."""
    ref_type = str(cfg.get("reference", "brownian")).lower()

    if ref_type == "brownian":
        sigma = float(cfg.get("sigma", 1.0))
        tag = f"brownian_s{sigma:g}"
    elif ref_type == "ou":
        gamma = float(cfg.get("gamma", 1.0))
        tag = f"ou_g{gamma:g}"
    else:
        raise ValueError(f"Unknown reference process: {ref_type}")

    return tag


def format_dataset_slug(cfg: DictConfig) -> Tuple[str, str, str]:
    """Resolves (<mode>, <dataset>, <dim>) path segments from Hydra config."""
    d = int(cfg.dataset.input_dim)
    dim_slug = f"d{d}"

    if cfg.dataset.type == "sb_benchmark":
        sb_cfg = cfg.dataset.sb
        mode = str(sb_cfg.get("mode", "synthetic")).lower()
        k = int(sb_cfg.n_components)
        cov = str(sb_cfg.get("cov_type", "isotropic")).lower()

        if mode == "empirical":
            base_type = str(sb_cfg.base_dataset.type).lower()
            dataset_slug = f"{base_type}-K{k}-{cov}"
            return mode, dim_slug, dataset_slug

        elif mode == "synthetic":
            geom = str(sb_cfg.get("geometry", "sphere")).lower()
            dataset_slug = f"{geom}-K{k}-{cov}"
            return mode, dataset_slug, dim_slug

        else:
            raise ValueError(f"Unknown SB benchmark mode: {mode}")

    dataset_slug = str(cfg.dataset.type).lower()

    return "empirical", dim_slug, dataset_slug


def _resolve_solver_and_variant(cfg: DictConfig) -> Tuple[str, str]:
    """Extracts (<solver>, <variant>) from cfg.solver or cfg.model."""
    solver_node = cfg.get("solver", cfg.get("model", None))
    if isinstance(solver_node, DictConfig):
        solver = str(solver_node.get("name", "default"))
        variant = str(cfg.get("variant", solver_node.get("variant", "base")))
    elif solver_node is not None:
        solver = str(solver_node)
        variant = str(cfg.get("variant", "base"))
    else:
        solver = "default"
        variant = str(cfg.get("variant", "base"))

    return solver, variant


def get_ref_dir(cfg: DictConfig, root: Optional[Union[str, Path]] = None) -> Path:
    """
    Creates and returns:
      - Synthetic: experiments/synthetic/<dataset>/<dim>/<ref_params>/
      - Empirical: experiments/empirical/<dim>/<dataset>/<ref_params>/
    """
    base = Path(root if root is not None else cfg.get("experiments_dir", "experiments"))
    seg1, seg2, seg3 = format_dataset_slug(cfg)
    ref_params = format_ref_params(cfg)

    ref_dir = base / seg1 / seg2 / seg3 / ref_params
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


def setup_experiment_dirs(
    cfg: DictConfig, root: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    Creates and returns the full solver experiment tree under <ref_params>/:
      experiments/<mode>/<dataset>/<dim>/<ref_params>/<solver>/<variant>/
    """
    ref_dir = get_ref_dir(cfg, root=root)
    solver, variant = _resolve_solver_and_variant(cfg)

    run_dir = ref_dir / solver / variant
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"

    for directory in (logs_dir, ckpt_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "ref_dir": ref_dir,
        "run_dir": run_dir,
        "configs_file": run_dir / "configs.log",
        "metrics_file": run_dir / "metrics.jsonl",
        "logs_dir": logs_dir,
        "train_log": logs_dir / "train.log",
        "eval_log": logs_dir / "eval.log",
        "checkpoints_dir": ckpt_dir,
        "best_ckpt": ckpt_dir / "best.pth",
        "last_ckpt": ckpt_dir / "last.pth",
        "plots_dir": plots_dir,
    }
