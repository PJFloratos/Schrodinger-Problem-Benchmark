# Future Generative Models and Extensions

This document outlines the theoretical frameworks and planned implementations for advanced generative models to be evaluated against the baseline Iterative Proportional Fitting (IPF) Diffusion Schrödinger Bridge. These approaches aim to bypass the heavy computational burden of iterative Monte Carlo trajectory simulation while preserving or approximating the optimal transport properties of the Schrödinger Problem.

## 1. Bridge Matching Alternatives to IPF
**Key Literature:**
* [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
] (https://arxiv.org/pdf/2106.01357)
* [DIffusion Schrödinger Bridge Matching](https://proceedings.neurips.cc/paper_files/paper/2023/file/c428adf74782c2092d254329b6b02482-Paper-Conference.pdf)
* [Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrodinger Bridge Matching](https://arxiv.org/pdf/2602.15396v1)
* [Reflected Schrödinger Bridge Matching](https://arxiv.org/pdf/2607.03626)

While the standard Diffusion Schrödinger Bridge relies on IPF to iteratively update forward and backward SDEs, Bridge Matching frameworks seek to regress directly on the bridge dynamics. By leveraging known properties of the reference process and marginal constraints, these methods attempt to construct target vector fields in a single training phase, eliminating the need for alternating Markov chain simulations and significantly reducing training overhead.

## 2. Dual-Based Entropic Optimal Transport (EOT) Solvers
**Key Literature:**
* [Entropic Neural Optimal Transport via Diffusion Processes](https://arxiv.org/pdf/2211.01156)
* [Energy-Guided Entropic Neural Optimal Transport](https://arxiv.org/pdf/2304.06094)
* [Variational Entropic Optimal Transport](https://arxiv.org/pdf/2602.02241v2)
* [Learning normalizing flows from Entropy-Kantorovich potentials](https://arxiv.org/pdf/2006.06033)

Instead of solving the dynamic Schrödinger Problem via forward/backward SDEs, this approach tackles the static Entropic Optimal Transport problem using its dual formulation. By parameterizing the Kantorovich dual potentials with neural networks and incorporating energy guidance, this method recovers the optimal joint coupling directly. Once the static coupling is identified, the dynamic paths can be reconstructed, completely bypassing the instability of iterative SDE simulation.

## 3. Light Gaussian Mixture Solvers
**Key Literature:**
* [Light and Optimal Schrodinger Bridge Matching](https://openreview.net/pdf?id=EWJn6hfZ4J)
* [Light Schrodinger Bridge](https://arxiv.org/pdf/2310.01174)

Standard neural Schrödinger Bridges scale poorly due to the continuous integration required across dense temporal grids. The Light Schrödinger Bridge framework mitigates this by applying structural assumptions—such as modeling the intermediate distributions as Gaussian Mixture Models (GMMs). By constraining the state space to mixtures of tractable distributions, the transition dynamics can be computed analytically or with highly lightweight parameterizations, drastically accelerating both training and inference.

## 4. Simulation-Free Solvers via Minibatch EOT Couplings
**Key Literature:**
* [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/pdf/2307.03672)
* [Learning Generative Models with Sinkhorn Divergences](https://arxiv.org/pdf/1706.00292)

This methodology combines the scalability of standard Flow Matching with the theoretical rigor of Entropic Optimal Transport. Instead of assuming independent endpoints (which increases trajectory intersection) or running IPF (which requires heavy simulation), this approach computes a discrete Entropic OT coupling over the training minibatch. This minibatch coupling serves as a highly accurate proxy for the true global coupling, providing deterministic regression targets for score and flow matching without ever simulating the SDE during training.

## 5. Adversarial Solvers
**Key Literature:**
* [Unpaired Image-to-Image Translation via Neural Schrodinger Bridhe](https://arxiv.org/pdf/2305.15086)

For high-dimensional modalities like images, pixel-wise regression on SDE paths often leads to blurriness and slow convergence. Adversarial solvers reformulate the Schrödinger Bridge as a minimax game. By employing discriminator networks to enforce the marginal constraints at $t=0$ and $t=1$, the generator network learns to map between unpaired datasets directly. This approach sacrifices strict path-wise optimal transport for superior perceptual quality in complex visual domains.

## 6. Action Matching
**Key Literature:**
* [Action Matching: Learning Stochastic Dynamics from Samples](https://arxiv.org/pdf/2210.06662)

Action Matching steps away from score matching and path regression, instead targeting the Benamou-Brenier fluid dynamics formulation of Optimal Transport. The neural network is trained to directly minimize the kinetic energy (action) of the probability flow subject to the continuity equation. By minimizing the action functional directly across the domain, the network learns the optimally efficient transport plan natively, offering a theoretically elegant alternative to explicit path construction.


## 7. Stochastic Solvers
**Key Literature:**
* [Generalized Schrodinger Bridge Matching](https://arxiv.org/pdf/2310.02233)
* [Stochastic Optimal Control Matching](https://arxiv.org/pdf/2312.02027)
* [Likelihood Training of Schrodinger Bridge Using Forward-Backward SDEs Theory](https://arxiv.org/pdf/2110.11291)

This framework generalizes standard bridge matching to accommodate a wider class of stochastic processes. While standard implementations assume a simple Brownian motion reference, Generalized Schrödinger Bridge Matching allows for complex, domain-specific reference SDEs. This enables the model to handle structured noise, manifold constraints, or non-Gaussian priors natively, providing a highly flexible, simulation-free framework for stochastic optimal transport.
