# Generative Models Documentation

This document outlines the theoretical foundations and practical implementations of the generative models utilized in our codebase. It details the formulations of the Stochastic Differential Equation (SDE) solver via Generator Matching, its deterministic Minibatch (ODE) variant, standard Flow Matching, and the Iterative Proportional Fitting (IPF) approach for the Diffusion Schrödinger Bridge.

## 1. SDE Solver (Generator Matching)

The SDE solver leverages the Generator Matching (GM) framework to approximate solutions to the Schrödinger Problem (SP). Because the exact optimal coupling $\hat{\pi}$ of the true Schrödinger Bridge is a complex, action-minimizing joint distribution, deriving the exact conditional paths is analytically intractable.

To achieve a scalable solution, this solver introduces a deliberate mathematical relaxation: it assumes an independent coupling between the prior and the data distribution, such that $\hat{\pi}(\mathrm{d}x_0 \mid z) \approx \mu_0(\mathrm{d}x_0)$. Assuming the reference measure is a reversible Brownian motion and the prior is a standard Gaussian $\mu_0 = \mathcal{N}(0, \mathbf{I})$, the conditional density becomes strictly Gaussian: $\mu_t(x \mid z) = \mathcal{N}(x; tz, (1-t)\mathbf{I})$.

This specific structural simplicity allows the Fokker-Planck and Hamilton-Jacobi-Bellman (HJB) equations to collapse, yielding a unique, closed-form conditional vector field:


$$u_t(x \mid z) = \frac{z - x}{1-t}$$

The neural network $u_t^\theta(x)$ is trained to approximate this true marginal vector field by minimizing the Conditional Generator Matching (CGM) objective under the squared $L^2$ norm:


$$\mathcal{L}_\text{CGM}(\theta) = \mathbb{E}_{t \sim U[0,1]}\mathbb{E}_{z\sim p_\text{data}, x \sim p_t(\cdot \mid z)} \left[ \Vert{} u_t(x \mid z) - u_t^\theta(x) \Vert{}^2 \right]$$

By accepting the entropic penalty of independent endpoints, this framework transforms an intractable system of coupled partial differential equations into a highly scalable regression task.

## 2. Minibatch (ODE) Solver

The Minibatch solver utilizes the exact same theoretical foundation and closed-form target vector field $u_t(x \mid z) = \frac{z - x}{1-t}$ as the SDE solver. However, it applies two key modifications to streamline the generation process:

1. **Deterministic Dynamics (ODE):** Instead of injecting Brownian noise during the forward generation phase, the solver integrates the learned vector field deterministically (an Ordinary Differential Equation).
2. **Minibatch Optimal Transport:** During training, instead of randomly pairing prior noise samples $x_0$ with data samples $z$, the solver computes a discrete optimal transport plan within each training batch. By aligning $x_0$ and $z$ to minimize their initial Euclidean distance, the resulting regression targets produce straighter, less entangled trajectories, which significantly improves the stability and efficiency of the ODE integration.

## 3. Standard Flow Matching

Flow Matching provides a simulation-free framework for training continuous normalizing flows. Unlike the SDE solver which targets the drift of a diffusion process, Standard Flow Matching defines a deterministic, probability density path directly between the prior $\mathcal{N}(0, \mathbf{I})$ and the target data distribution.

The standard approach defines constant-velocity trajectories connecting the noise $x_0$ to the data $z$:


$$x_t = (1 - t)x_0 + tz$$


The corresponding target vector field is simply the constant velocity required to move from $x_0$ to $z$:


$$u_t(x \mid z) = z - x_0$$


The network is trained using a standard Mean Squared Error loss to match this vector field. Like the Minibatch solver, Standard Flow Matching is highly compatible with minibatch optimal transport to straighten the global flow and reduce trajectory intersection.

## 4. Diffusion Schrödinger Bridge (IPF)

The Diffusion Schrödinger Bridge (DSB) algorithm seeks the true, entropy-minimizing optimal path measure $\hat{P}$ of the Schrödinger Problem. As established in the Generator Matching framework, analytical conditional vector fields can only be derived when the coupling is independent and the conditional density remains Gaussian.

Once a neural network learns an initial joint distribution, the intermediate conditional densities lose their Gaussian structure, making the PDEs intractable and analytical regression targets impossible to derive. To overcome this hard theoretical limit, the DSB abandons closed-form targets entirely and relies on Iterative Proportional Fitting (IPF).

The IPF algorithm operates by alternating between two Stochastic Differential Equations (SDEs):

* **Forward Phase:** Simulates trajectories from the current forward model and trains a backward neural network to regress on these simulated Monte Carlo paths using score matching.
* **Backward Phase:** Simulates trajectories from the newly updated backward model and trains the forward neural network to match them.

While this alternating simulation demands significantly higher computational resources than single-shot Generator Matching, it allows the system to continuously refine the complex, non-linear joint coupling $\hat{\pi}$, successfully recovering the true optimal Schrödinger Bridge without relying on suboptimal analytical relaxations.

---

### Academic References

* **Generator Matching:** *Generator Matching for Generative Modeling* (Note: Ensure the exact citation format matches your institution's specific referencing style for the official GM paper).
* **Schrödinger Problem Theory:** Léonard, C. (2014). *A survey of the Schrödinger problem and some of its connections with optimal transport*. Discrete and Continuous Dynamical Systems - A.
* **Flow Matching:** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, Matt. (2023). *Flow Matching for Generative Modeling*. International Conference on Learning Representations (ICLR).
* **Diffusion Schrödinger Bridge:** De Bortoli, V., Thornton, J., Heng, J., & Doucet, A. (2021). *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*. Advances in Neural Information Processing Systems (NeurIPS).
