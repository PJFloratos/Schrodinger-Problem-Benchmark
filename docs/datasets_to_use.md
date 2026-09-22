# Datasets to Add

Start with the benchmarks from: [Building the Bridge of Schrödinger: A Continuous Entropic Optimal Transport Benchmark](https://arxiv.org/pdf/2306.10161).

| Evaluation Axis | Datasets & Configurations | Primary Evaluation Purpose |
| --- | --- | --- |
| **Ground-Truth-Anchored** | • LSE-potential pairs: Configurable dimension (D) and ϵ.<br>• Closed-form Gaussian SBs: Unequal covariance matrices.<br>• Multimodal 2D toys: Checkerboard, two moons, pinwheel, concentric rings. | Establish absolute correctness. Tests rotation/scaling capabilities and provides cheap, fast visual checks for mode collapse. |
| **Image Generation** (Noise → Data) | • CIFAR-10 (32×32 RGB).<br>• CelebA (64×64 and 128×128).<br>• CelebA + Glow-constructed target pairs. | Tests solver handling of harder texture statistics and resolution scaling. Glow pairs allow for real conditional FID scoring against a known plan. |
| **Domain Translation** (Data → Data) | • MNIST → EMNIST or MNIST ↔ USPS.<br>• Colored MNIST 2→3.<br>• Unpaired super-resolution or style-transfer setups. | Evaluates domain translation capabilities and provides direct 1:1 comparison points with published EgNOT/ENOT metrics. |
| **Non-Image Data** | • Single-cell / scRNA-seq perturbations: Low-D (D≈5−50), non-Gaussian.<br>• Motion capture data: Smooth trajectories. | Verifies that solver performance is not reliant on image-specific network artifacts (e.g., UNet biases). Tests temporal correlation handling. |
