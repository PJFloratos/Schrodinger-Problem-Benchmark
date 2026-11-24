from src.datasets.data_utils import get_dataset
from src.datasets.sb_benchmark import SBBenchmarkDataset
from src.dynamics.references import BrownianReference, OUReference, ReferenceProcess
from src.utils.paths import get_ref_dir
from src.utils.log import text_logger
from src.utils.save import save_config_snapshot

import torch
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import math
from typing import Optional, Tuple


def build_reference(cfg: DictConfig) -> ReferenceProcess:
    """Instantiates the linear SDE reference process from top-level main.yaml keys."""
    ref_type = str(cfg.get("reference", "brownian")).lower()

    if ref_type == "ou":
        return OUReference(gamma=float(cfg.get("gamma", 1.0)))
    if ref_type == "brownian":
        return BrownianReference(sigma=float(cfg.get("sigma", 1.0)))

    raise ValueError(f"Unknown reference process: {ref_type}")


def _fit_exact_pca(
    pts: torch.Tensor, n_components: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PCA via covariance eigendecomposition (returns evr_pct, mean, components)."""
    N, d = pts.shape
    mean = pts.mean(dim=0, keepdim=True)
    centered = pts - mean

    if N >= d:
        cov = (centered.T @ centered) / max(N - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.flip(0).clamp_min(0.0)
        components = eigvecs.flip(1)[:, :n_components].T
    else:
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        eigvals = (S**2) / max(N - 1, 1)
        components = Vh[:n_components]

    evr_pct = (eigvals[:n_components] / eigvals.sum().clamp_min(1e-12)) * 100.0
    return evr_pct, mean, components


def check_sb_benchmark(
    cfg: DictConfig,
    train: SBBenchmarkDataset,
    test: SBBenchmarkDataset,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs statistical, structural, expressivity, and closed-form oracle sanity checks."""
    log = logger.info if logger is not None else print
    bridge = train.bridge
    d = bridge.d
    x_tr, y_tr = train.x, train.y
    x_te, y_te = test.x, test.y

    log(
        f"=== Inspecting {cfg.dataset.name} (ref={bridge.ref.name}, K={bridge.K}, d={d}, cov={bridge.cov_type}) ==="
    )

    def report(label: str, pts: torch.Tensor) -> None:
        m2 = (pts**2).sum(dim=1).mean().item() / d
        mean_str = np.array2string(pts.mean(dim=0)[:4].cpu().numpy(), precision=3)
        log(
            f"  {label:10s} | n={len(pts):6d} | E[|x|^2]/d={m2:7.4f} | "
            f"min={pts.min().item():6.2f} | max={pts.max().item():6.2f} | mean[:4]={mean_str}"
        )

    report("train.x", x_tr)
    report("train.y", y_tr)
    report("test.x", x_te)
    report("test.y", y_te)

    # 1. Split integrity & independence
    assert (
        train.bridge is test.bridge
    ), "Train and test must share the exact same bridge!"
    assert len(train) == cfg.dataset.train_samples
    assert len(test) == cfg.dataset.test_samples
    n_chk = min(len(y_tr), len(y_te), 100)
    assert not torch.equal(
        y_tr[:n_chk], y_te[:n_chk]
    ), "Train and test target samples are identical!"

    # 2. Prior variance check: E[||X_0||^2] / d == prior_std^2 (or r0_std^2)
    expected_prior_var = (
        bridge.prior_std**2 if bridge.prior_std is not None else bridge.ref.r0_std**2
    )
    emp_prior_var = (x_tr**2).sum(dim=1).mean().item() / d
    tol = 5.0 * expected_prior_var / math.sqrt(len(x_tr) * d) + 1e-2
    assert (
        abs(emp_prior_var - expected_prior_var) < tol
    ), f"Prior variance mismatch: empirical={emp_prior_var:.4f} vs analytic={expected_prior_var:.4f}"

    # 3. Optimal drift & bridge trajectory numerical stability near t=0 and t=1
    probe_x = x_te[:32]
    probe_y = y_te[:32]
    for t_val in [1e-5, 0.5, 1.0 - 1e-5]:
        v_xt = bridge.optimal_drift(probe_x, t=t_val)
        x_mid = bridge.sample_bridge_trajectory(probe_x, probe_y, t=t_val)
        assert (
            v_xt.shape == (32, d) and torch.isfinite(v_xt).all()
        ), f"Non-finite drift at t={t_val}"
        assert (
            x_mid.shape == (32, d) and torch.isfinite(x_mid).all()
        ), f"Non-finite bridge at t={t_val}"

    # 4. Expressivity & Conditioning Diagnostics (PCA Spectrum & Effective Mode Count K_eff)
    n_pcs = min(10, d)
    pca_fit_n = min(20_000, len(y_tr))
    evr_pct_t, pca_mean, pca_components = _fit_exact_pca(y_tr[:pca_fit_n], n_pcs)
    evr_pct = evr_pct_t.cpu().numpy()
    evr_str = np.array2string(evr_pct, precision=1, separator=", ")
    top2_var = float(evr_pct[: min(2, n_pcs)].sum())
    top8_var = float(evr_pct[: min(8, n_pcs)].sum())
    pca_summary = f"top-2={top2_var:.1f}%"
    if d >= 8:
        pca_summary += f", top-8={top8_var:.1f}%"
    log(f"  PCA Spectrum ({pca_summary}): {evr_str}%")

    k_eff = bridge.effective_num_modes(x_te, chunk_size=512)

    assert torch.isfinite(k_eff).all(), "Non-finite posterior perplexity K_eff(x)!"
    assert (k_eff >= 1.0 - 1e-3).all() and (
        k_eff <= bridge.K + 1e-3
    ).all(), f"K_eff(x) out of bounds [1, {bridge.K}]"

    keff_mean = k_eff.mean().item()
    keff_med = k_eff.median().item()
    keff_p10 = torch.quantile(k_eff, 0.10).item()
    keff_p90 = torch.quantile(k_eff, 0.90).item()

    if bridge.K == 1:
        regime_tag = "unimodal Gaussian (linear drift)"
    elif keff_mean < 1.5:
        regime_tag = "near-deterministic OT limit"
    elif keff_mean > 0.7 * bridge.K:
        regime_tag = "weak conditioning (near-independent coupling)"
    else:
        regime_tag = "nonlinear multi-mode stochastic branching"

    log(
        f"  Posterior K_eff(x): mean={keff_mean:.2f} | median={keff_med:.2f} | "
        f"[p10, p90]=[{keff_p10:.2f}, {keff_p90:.2f}] (K={bridge.K} -> {regime_tag})"
    )

    # 5. Self-consistency check: Ground-truth oracle against itself must yield ~0% cBW_2^2-UVP
    gen = torch.Generator(device=bridge.device).manual_seed(cfg.eval_seed)

    def oracle_sampler(x_batch: torch.Tensor, n_cond: int) -> torch.Tensor:
        M, dim = x_batch.shape
        x_rep = x_batch.unsqueeze(1).expand(M, n_cond, dim).reshape(M * n_cond, dim)
        return bridge.sample_conditional(x_rep, gen=gen).view(M, n_cond, dim)

    # n_cond = 2000 if d >= 100 else 1000
    n_cond = 4000 if d >= 1000 else (2000 if d >= 100 else 1000)
    self_uvp = bridge.compute_cbw2_uvp(
        solver_sampler=oracle_sampler,
        n_test_x=200,
        n_cond_samples=n_cond,
        n_target_var_samples=10_000,
        gen=gen,
    )
    max_expected_uvp = 1.5 + 45.0 * (d / n_cond)
    log(
        f"  Oracle Self-Check cBW_2^2-UVP: {self_uvp:.4f}% (Wishart ceiling: <{max_expected_uvp:.2f}%)"
    )
    assert (
        self_uvp < max_expected_uvp
    ), f"Oracle self-UVP unexpectedly high: {self_uvp:.4f}%"
    log("  All assertions passed.")

    return pca_mean.cpu(), pca_components[: min(2, n_pcs)].cpu()


def plot_2d_diagnostics(
    cfg: DictConfig, train: SBBenchmarkDataset, test: SBBenchmarkDataset, save_path: str
) -> None:
    """6-panel 2D phase-space, coupling, trajectory, drift field, and posterior visualization."""
    bridge = train.bridge
    gen = torch.Generator(device=bridge.device).manual_seed(cfg.eval_seed)

    x0 = test.x.cpu()
    x1 = test.y.cpu()
    means = bridge.means.cpu()

    all_pts = torch.cat([x0, x1, means], dim=0)
    lim = float(all_pts.abs().max()) * 1.12

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Prior mu
    ax = axes[0, 0]
    ax.scatter(x0[:, 0], x0[:, 1], s=3, alpha=0.45, color="steelblue")
    ax.set_title(r"1. Prior $\mu$ ($X_0$)")

    # Panel 2: Target Marginal nu + GMM Means
    ax = axes[0, 1]
    ax.scatter(
        x1[:, 0], x1[:, 1], s=3, alpha=0.45, color="darkorange", label=r"$Y \sim \nu$"
    )
    if bridge.K <= 250:
        ax.scatter(
            means[:, 0],
            means[:, 1],
            s=18,
            c="black",
            marker="x",
            linewidths=1.0,
            label=r"Centers $\mu_k$",
        )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(r"2. Target Marginal $\nu$ ($X_1$)")

    # Panel 3: Optimal Coupling pi*(x, y)
    ax = axes[0, 2]
    n_pairs = min(120, len(x0))
    for i in range(n_pairs):
        ax.plot(
            [x0[i, 0], x1[i, 0]],
            [x0[i, 1], x1[i, 1]],
            color="purple",
            alpha=0.22,
            linewidth=0.8,
        )
    ax.scatter(
        x0[:n_pairs, 0], x0[:n_pairs, 1], s=10, color="steelblue", label=r"$X_0$"
    )
    ax.scatter(
        x1[:n_pairs, 0], x1[:n_pairs, 1], s=10, color="darkorange", label=r"$X_1$"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(r"3. Ground-Truth Coupling $(X_0, X_1) \sim \pi^*$")

    # Panel 4: Exact Stochastic Bridge Paths R_{t|0,1}
    ax = axes[1, 0]
    n_traj = 25
    ts = torch.linspace(0.0, 1.0, 25)
    x_curr = x0[:n_traj].to(bridge.device)
    x_end = x1[:n_traj].to(bridge.device)
    traj = [x0[:n_traj]]
    for idx in range(len(ts) - 2):
        t_s = float(ts[idx])
        t_next = float(ts[idx + 1])
        v_st = float(bridge.ref.V(t_s, t_next))
        v_t1 = float(bridge.ref.V(t_next, 1.0))
        a_t1 = float(bridge.ref.A(t_next, 1.0))
        v_s1 = (a_t1**2) * v_st + v_t1
        c1 = (a_t1 * v_st) / v_s1
        v_step = (v_st * v_t1) / v_s1

        mu_st = bridge.ref.mu(x_curr, t_s, t_next)
        mu_s1 = bridge.ref.mu(x_curr, t_s, 1.0)
        z = torch.randn(
            x_curr.shape, device=bridge.device, dtype=bridge.dtype, generator=gen
        )
        x_curr = mu_st + c1 * (x_end - mu_s1) + math.sqrt(v_step) * z
        traj.append(x_curr.cpu())
    traj.append(x1[:n_traj])
    traj_arr = torch.stack(traj, dim=0).numpy()
    for i in range(n_traj):
        ax.plot(traj_arr[:, i, 0], traj_arr[:, i, 1], alpha=0.55, linewidth=1.0)
    ax.scatter(x0[:n_traj, 0], x0[:n_traj, 1], s=18, color="steelblue", zorder=3)
    ax.scatter(x1[:n_traj, 0], x1[:n_traj, 1], s=18, color="darkorange", zorder=3)
    ax.set_title(r"4. Exact Bridge Paths $X_t \sim P^*_{t|0,1}$")

    # Panel 5: Closed-Form Optimal Drift Field v*(x, t=0.5)
    ax = axes[1, 1]
    grid_1d = torch.linspace(-lim, lim, 22)
    gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
    grid_pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(bridge.device)
    drift = bridge.optimal_drift(grid_pts, t=0.5).cpu()
    ax.scatter(x1[:, 0], x1[:, 1], s=2, alpha=0.15, color="darkorange")
    ax.quiver(
        grid_pts[:, 0].cpu().numpy(),
        grid_pts[:, 1].cpu().numpy(),
        drift[:, 0].numpy(),
        drift[:, 1].numpy(),
        color="teal",
        alpha=0.85,
    )
    ax.set_title(r"5. Optimal Drift $v^*(x, t=0.5)$")

    # Panel 6: Conditional Plans pi*(. | x*) & Analytical Moments (mu*(x*), Sigma*(x*))
    ax = axes[1, 2]
    ax.scatter(x1[:, 0], x1[:, 1], s=2, alpha=0.12, color="gray")
    prior_scale = (
        bridge.prior_std if bridge.prior_std is not None else bridge.ref.r0_std
    )
    anchors = (
        torch.tensor([[-0.8, 0.8], [0.0, 0.0], [0.8, -0.8]], device=bridge.device)
        * prior_scale
    )
    mu_cond, cov_cond = bridge.conditional_moments(anchors)
    colors = ["crimson", "royalblue", "forestgreen"]

    for idx, col in enumerate(colors):
        anchor = anchors[idx : idx + 1]
        y_cond = bridge.sample_conditional(anchor.repeat(400, 1), gen=gen).cpu().numpy()
        ax.scatter(y_cond[:, 0], y_cond[:, 1], s=4, alpha=0.35, color=col)
        ax.scatter(
            anchor[0, 0].cpu(),
            anchor[0, 1].cpu(),
            s=90,
            marker="*",
            color=col,
            edgecolor="black",
            zorder=5,
        )
        eigvals, eigvecs = torch.linalg.eigh(cov_cond[idx].cpu())
        angle = math.degrees(math.atan2(eigvecs[1, 1].item(), eigvecs[0, 1].item()))
        h, w = 4.0 * torch.sqrt(eigvals.clamp_min(1e-8)).numpy()
        ell = Ellipse(
            xy=mu_cond[idx].cpu().numpy(),
            width=w,
            height=h,
            angle=angle,
            edgecolor=col,
            fc="none",
            lw=2.0,
            linestyle="--",
        )
        ax.add_patch(ell)
    ax.set_title(r"6. Conditional $\pi^*(\cdot|x^*)$ & $2\sigma$ Ellipses")

    for ax in axes.flat:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{cfg.dataset.name} | ref={bridge.ref.name} | K={bridge.K} | cov={bridge.cov_type}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_highd_diagnostics(
    cfg: DictConfig,
    train: SBBenchmarkDataset,
    test: SBBenchmarkDataset,
    save_path: str,
    pca_mean: torch.Tensor,
    pca_components: torch.Tensor,
) -> None:
    """PCA 2D subspace projection & norm diagnostics for d > 2 (plus image grid if d == 784)."""
    bridge = train.bridge
    d = bridge.d
    x0 = test.x.cpu()
    x1 = test.y.cpu()
    means = bridge.means.cpu()

    mode = str(cfg.dataset.sb.get("mode", "synthetic")).lower()
    base_type = (
        str(cfg.dataset.sb.base_dataset.type).lower() if mode == "empirical" else ""
    )

    img_shape = None
    if d == 784 and base_type == "mnist":
        img_shape = (1, 28, 28)
    elif d == 3072 and base_type == "cifar10":
        img_shape = (3, 32, 32)

    if img_shape is not None:
        is_neg_one_to_one = float(means.min()) < -0.1

        def to_img(flat: torch.Tensor) -> torch.Tensor:
            imgs = flat.view(-1, *img_shape)
            if is_neg_one_to_one:
                imgs = (imgs + 1.0) / 2.0
            return imgs.clamp(0.0, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (r"Prior $X_0$", to_img(x0[:64])),
            (r"Target Marginal $X_1 \sim \nu$", to_img(x1[:64])),
            (f"GMM Centroids (first {min(64, bridge.K)})", to_img(means[:64])),
        ]
        for ax, (title, imgs) in zip(axes, panels):
            grid = make_grid(imgs, nrow=8, pad_value=0.5).permute(1, 2, 0).numpy()
            ax.imshow(grid)
            ax.axis("off")
            ax.set_title(title)

        fig.suptitle(f"{cfg.dataset.name} | ref={bridge.ref.name} | K={bridge.K}")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    proj = lambda pts: ((pts - pca_mean) @ pca_components.T).numpy()
    z0 = proj(x0)
    z1 = proj(x1)
    z_means = proj(means)

    centered_tr = train.y[: min(10_000, len(train))].cpu() - pca_mean
    tot_var = (centered_tr**2).sum(dim=1).mean().clamp_min(1e-12)
    pc2_var = ((centered_tr @ pca_components.T) ** 2).sum(dim=1).mean()
    var_ratio = float((pc2_var / tot_var).item()) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(z0[:, 0], z0[:, 1], s=3, alpha=0.45, color="steelblue")
    axes[0].set_title(r"1. Prior $X_0$ (Top-2 Target PCs)")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        z1[:, 0], z1[:, 1], s=3, alpha=0.45, color="darkorange", label=r"$Y \sim \nu$"
    )
    if bridge.K <= 250:
        axes[1].scatter(
            z_means[:, 0],
            z_means[:, 1],
            s=20,
            c="black",
            marker="x",
            label=r"Means $\mu_k$",
        )
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title(f"2. Target $\\nu$ ({var_ratio:.1f}% Var in 2 PCs)")
    axes[1].grid(alpha=0.25)

    norms_0 = x0.norm(dim=-1).numpy()
    norms_1 = x1.norm(dim=-1).numpy()
    axes[2].hist(
        norms_0,
        bins=40,
        density=True,
        alpha=0.55,
        color="steelblue",
        label=r"$\|X_0\|_2$",
    )
    axes[2].hist(
        norms_1,
        bins=40,
        density=True,
        alpha=0.55,
        color="darkorange",
        label=r"$\|X_1\|_2$",
    )
    axes[2].set_title(r"3. Radial Norm Concentration $\|X\|_2$")
    axes[2].set_xlabel(r"$\|x\|_2$")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"{cfg.dataset.name} | ref={bridge.ref.name} | d={d} | K={bridge.K} | cov={bridge.cov_type}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    if cfg.dataset.type != "sb_benchmark":
        raise ValueError(
            f"Expected dataset=sb_benchmark, got dataset.type={cfg.dataset.type}"
        )

    ref_dir = get_ref_dir(cfg)
    log_path = ref_dir / "inspect.log"
    plot_path = ref_dir / "inspect.png"
    cfg_snapshot_path = ref_dir / "inspect_config.log"

    # Save read-only snapshot of dataset + reference SDE parameters
    sb_cfg = cfg.dataset.sb
    mode = str(sb_cfg.get("mode", "synthetic")).lower()
    dataset_type = str(sb_cfg.base_dataset.type) if mode == "empirical" else "synthetic"

    dataset_snapshot = OmegaConf.create(
        {
            "reference": ref_dir.name,
            "mode": mode,
            "type": dataset_type,
            "geometry": sb_cfg.get("geometry", None) if mode == "synthetic" else None,
            "n_components": sb_cfg.get("n_components", None),
            "cov_type": sb_cfg.get("cov_type", None),
            "var": sb_cfg.get("var", None),
            "radius": sb_cfg.get("radius", None) if mode == "synthetic" else None,
            "prior_std": sb_cfg.get("prior_std", None),
            "bridge_seed": sb_cfg.get("bridge_seed", None),
        }
    )
    save_config_snapshot(dataset_snapshot, cfg_snapshot_path, fmt="log")

    logger = text_logger(name="inspect", log_file=str(log_path), mode="w")
    logger.info(f"Benchmark Directory: {ref_dir}")
    logger.info(f"SB Config: {OmegaConf.to_container(cfg.dataset.sb, resolve=True)}")

    ref = build_reference(cfg)
    train, test = get_dataset(cfg, ref=ref)

    pca_mean, pca_components = check_sb_benchmark(cfg, train, test, logger=logger)

    if train.bridge.d == 2:
        plot_2d_diagnostics(cfg, train, test, str(plot_path))
    else:
        plot_highd_diagnostics(
            cfg, train, test, str(plot_path), pca_mean, pca_components
        )

    logger.info(f"Saved config snapshot -> {cfg_snapshot_path}")
    logger.info(f"Saved inspection log  -> {log_path}")
    logger.info(f"Saved diagnostic plot -> {plot_path}")


if __name__ == "__main__":
    main()
