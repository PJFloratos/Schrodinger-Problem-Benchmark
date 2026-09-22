# Benchmark Tracking & Diagnostics Plan

Those will be implemented on IPF first and on the other solvers later on.

## 1. Per-Step Training Diagnostics

*Log these continuously via TensorBoard, Weights & Biases, or a CSV keyed by `(ipf_iter, direction, inner_step)`. Focus on inner-phase regression health.*

| Diagnostic | Signal / Purpose | Implementation Notes |
| --- | --- | --- |
| **Inner-Phase Loss Curve** | Verifies if `inner_iterations` allows the regression to actually converge, or if phases are truncating early.

 | Log continuously; do not rely solely on phase averages. |
| **Gradient Norm** (Pre-clip) | Indicates if gradient clipping is necessary/active and flags numerical instability at high $\epsilon$ or in high dimensions.

 | Track before `clip_grad_norm_` is applied. |
| **Effective Learning Rate** | Sanity check to ensure your cosine decay schedule is functioning as intended within each phase.

 |  |
| **Cache Staleness** | Detects model drift. If regression loss visibly jumps at cache boundaries, `refresh_every` is too large.

 | Log the loss immediately before and after cache regeneration. |

## 2. Outer-Loop (IPF) Convergence

*Evaluates whether the procedure is converging to a fixed point, independent of sample quality.*

| Metric | Signal / Purpose | Implementation Notes |
| --- | --- | --- |
| **Marginal-Matching Distance** | Should decrease and plateau over $n$. Oscillations or non-monotonic curves indicate bad hyperparameters (LR, cache size, mean-matching). | Use MMD for 2D/low-D, FID for images. Evaluate after *every* IPF iteration.

 |
| **Path Self-Consistency** | Probes the alignment of the forward and backward chains. | Evaluate forward/backward consistency at 3–5 points across $t \in [0,1]$. |
| **Parameter Drift** | Direct proxy for "has the fixed point been reached." Dictates the actual number of IPF iterations required for a given solver. | Track $\Vert{}f\_model_n - f\_model_{n-1}\Vert{}$ on the outputs of a fixed probe batch. |

## 3. Correctness Against Ground Truth

Applicable only where analytic solutions exist (e.g., 2D toy sets, closed-form Gaussian SBs, LSE-constructed benchmark pairs).

* **Distance Metrics**: Track $cBW_2^2$-UVP and $BW_2^2$-UVP, alongside their conditional-plan MMD for the joint coupling.
* **Drift MSE**: Compare against the analytic $v^*(x,t)$ at multiple $t$-steps, integrated as forward/reverse KL via Girsanov. This provides a true divergence between path measures rather than just an endpoint statistic.
* **Independent-Plan Baseline**: Always report raw metrics against a null/independent-plan baseline. Raw UVP percentages are meaningless without this reference point.

## 4. Generative Quality

For standard datasets lacking ground truth (e.g., MNIST, CIFAR, CelebA, real data).

* **FID / KID**: Establishes base generative quality against the real target distribution.
* **Conditional FID**: Probes the conditional plan rather than the output marginal (essential for image-to-image translation tasks).
* **Precision / Recall**: Evaluates mode coverage. Essential for catching mode collapse that base FID might hide, particularly in barycentric-projection-style solvers.

## 5. Compute Cost

*Highlights the trade-offs between training cost and accuracy across different solver methods.*

| Metric | Description |
| --- | --- |
| **Wall-Clock Time** | Total time required to reach a specific quality threshold. |
| **NFEs** | Number of function evaluations at sampling time. Quantifies the practical step-count advantage of IPF/SB methods over vanilla diffusion. |
| **Convergence Steps** | Total gradient steps and outer IPF iterations required to reach convergence. |
| **Hardware Footprint** | Total parameter count and peak memory usage. |
