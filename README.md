# OxiCUDA

[![Crates.io](https://img.shields.io/crates/v/oxicuda.svg)](https://crates.io/crates/oxicuda)
[![Documentation](https://docs.rs/oxicuda/badge.svg)](https://docs.rs/oxicuda)
[![CI](https://github.com/cool-japan/oxicuda/workflows/CI/badge.svg)](https://github.com/cool-japan/oxicuda/actions)
[![License](https://img.shields.io/crates/l/oxicuda.svg)](LICENSE)

**Pure Rust CUDA replacement -- cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND and beyond in ~1.06M SLoC of safe Rust across 73 crates.**

OxiCUDA replaces the entire NVIDIA CUDA Toolkit software stack with type-safe,
memory-safe Rust code. The only runtime dependency is the NVIDIA driver
(`libcuda.so` / `nvcuda.dll`); no CUDA SDK, no `nvcc`, no C/C++ toolchain is
needed at build time. Optimized PTX assembly is generated directly from Rust
data structures, and a built-in autotuner benchmarks kernel variants per GPU
architecture to achieve near-peak throughput from Turing through Blackwell.

## Architecture

```
+---------------------------------------------------------------+
|   SciRS2  |  OxiONNX  |  TrustformeRS  |  ToRSh              |
|   (Scientific Computing / ML / Inference Ecosystem)           |
+-------------------------------+-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                         OxiCUDA                               |
|                     (Pure Rust GPU)                            |
|                                                               |
|  Vol.1 Foundation (4 crates)                                  |
|  +----------+ +--------+ +---------+ +---------+             |
|  | Driver   | | Memory | | Launch  | | Runtime |             |
|  +----------+ +--------+ +---------+ +---------+             |
|                                                               |
|  Vol.2 Codegen (2 crates)                                     |
|  +-----------+ +------------+                                 |
|  | PTX Gen   | | Autotune   |                                 |
|  +-----------+ +------------+                                 |
|                                                               |
|  Vol.3 Linear Algebra    Vol.4 Deep Learning                  |
|  +-------------+         +-------------+                      |
|  | BLAS        |         | DNN         |                      |
|  +-------------+         +-------------+                      |
|                                                               |
|  Vol.5 Scientific Computing (4 crates)                        |
|  +------+ +--------+ +--------+ +------+                     |
|  | FFT  | | Sparse | | Solver | | Rand |                     |
|  +------+ +--------+ +--------+ +------+                     |
|                                                               |
|  Vol.6 Signal    Vol.7 Comp.Graph  Vol.8 Training (2)         |
|  +---------+     +----------+      +-------+ +-------+        |
|  | Signal  |     | Graph    |      | Train | | Quant |        |
|  +---------+     +----------+      +-------+ +-------+        |
|                                                               |
|  Vol.9 Inference (3 crates)        Vol.10 RL                  |
|  +-------+ +------------+ +----+   +------+                   |
|  | Infer | | Dist-Infer | | LM |   |  RL  |                   |
|  +-------+ +------------+ +----+   +------+                   |
|                                                               |
|  Backends (7 crates)                                          |
|  +----------+ +--------+ +-------+ +--------+                 |
|  | backend  | | prims  | | Metal | | Vulkan |                 |
|  +----------+ +--------+ +-------+ +--------+                 |
|  +--------+ +-------+ +-----------+                           |
|  | WebGPU | | ROCm  | | LevelZero |                           |
|  +--------+ +-------+ +-----------+                           |
+-------------------------------+-------------------------------+
                                |
+-------------------------------v-------------------------------+
|              libcuda.so  (NVIDIA Driver, runtime only)        |
|              No SDK  /  No nvcc  /  No C Toolchain            |
+---------------------------------------------------------------+
```

## Feature Highlights

**Vol.1 -- Foundation** (4 crates, 26,438 SLoC)
- Dynamic driver loading via `libloading` -- zero build-time SDK dependency
- `DeviceBuffer<T>` with Rust ownership semantics -- `Send + Sync`, RAII
- Type-safe `launch!` macro with compile-time grid/block validation
- CUDA Runtime API layer for high-level device management

**Vol.2 -- PTX Codegen & Autotuner** (2 crates, 47,429 SLoC)
- Rust DSL that generates PTX IR covering SM 7.5 through SM 10.0
- Tensor Core support: WMMA, MMA, WGMMA instruction generation
- Built-in autotuner with 3-tier dispatch (cached / tuned / default)
- Disk-based PTX cache keyed by kernel hash + GPU architecture

**Vol.3 -- BLAS** (1 crate, 28,379 SLoC)
- Full BLAS Level 1/2/3 (axpy, gemv, gemm, trsm, syrk, ...)
- GEMM dispatch: SIMT, Tensor Core, Split-K paths
- Batched GEMM: standard, strided, grouped
- Precision coverage: F16, BF16, TF32, F32, F64, FP8
- Elementwise ops (relu, gelu, sigmoid, silu) and reductions (softmax, variance)

**Vol.4 -- DNN** (1 crate, 39,297 SLoC)
- Convolution: implicit GEMM, im2col, Winograd 3x3, direct, fused Conv+BN+Act
- FlashAttention forward/backward, PagedAttention, decode attention
- MoE: top-k routing, token permutation, fused MoE kernel
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm
- Pooling: max, average, adaptive, global
- Resize: nearest, bilinear, bicubic
- Quantization: FP8, INT8, block-scaled FP4

**Vol.5 -- Scientific Computing** (4 crates, 62,511 SLoC)
- FFT: Stockham, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, 2D/3D
- Sparse: CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM, SDDMM, ILU(0)/IC(0)
- Solver: LU, QR, SVD, Cholesky, eigendecomp, CG, BiCGSTAB, GMRES
- Rand: Philox, MRG32k3a, XORWOW, Sobol, uniform/normal/Poisson

**Vol.6 -- Signal Processing** (1 crate, 12,276 SLoC)
- Audio: MFCC, STFT, Mel filterbank, spectral features
- Image: Gaussian blur, Sobel edge detection, morphological ops
- DCT: Types I-IV with fast algorithms
- DWT: Haar, Daubechies wavelets
- Filtering: IIR/FIR filters, Butterworth, Chebyshev
- Correlation: cross-correlation, autocorrelation

**Vol.7 -- Computation Graph** (1 crate, 6,563 SLoC)
- CUDA Graph capture API (StreamCapture, GraphCapture)
- Execution plan with dependency-sorted node scheduling
- Event-based inter-node synchronization
- Sequential + parallel graph executors

**Vol.8 -- GPU Training** (2 crates, 13,832 SLoC)
- Mixed precision training (AMP): FP16/BF16 + loss scaling
- Gradient accumulation and clipping; EMA (exponential moving average)
- LR schedulers: cosine, warmup, cyclic, polynomial
- GPU-fused optimizers: Adam, AdamW, SGD, RMSProp, LAMB
- Checkpointing (model save/load)
- Quantization: INT8/INT4/FP8 weight quantization, block-scaled

**Vol.9 -- Inference Engine** (3 crates, 17,909 SLoC)
- KV-cache with paged attention (PagedKvCache) and prefix caching
- Speculative decoding
- Distributed inference pipeline (tensor/pipeline parallelism)
- LM inference: BPE tokenizer, vocabulary management, sampling strategies

**Vol.10 -- Reinforcement Learning** (1 crate, 11,280 SLoC)
- Replay buffers: Uniform, Prioritized (PER), N-step
- Policy distributions: Categorical, Gaussian (SAC reparameterization), Deterministic
- Advantage estimators: GAE, TD(λ), V-trace, Retrace(λ)
- Loss functions: PPO, DQN, Double-DQN, SAC, TD3
- Observation/reward normalization with Welford running stats
- Environment abstractions: Env, VecEnv (auto-reset)

**Backends** (7 crates, 28,400 SLoC)
- Backend trait abstraction for multi-GPU-runtime portability
- CUB-equivalent GPU primitives (scan, reduce, sort, histogram)
- Metal (macOS), Vulkan Compute, WebGPU, AMD ROCm, Intel oneAPI (LevelZero)

## Pure Rust, Minimal Dependencies

OxiCUDA is built on a strict **Pure Rust** policy with minimal external
dependencies. The entire codebase compiles with `cargo build` alone -- no
C compiler, no Fortran runtime, no CUDA SDK, no `nvcc`, no `pkg-config`.

| Dependency | Purpose | Type |
|------------|---------|------|
| `libloading` | Dynamic `.so`/`.dll` loading at runtime | Pure Rust |
| `thiserror` | Ergonomic error type derivation | Pure Rust |
| `num-complex` | Complex number types (FFT) | Pure Rust |
| `half` | FP16/BF16 types (optional) | Pure Rust |
| `serde` / `serde_json` | Autotune result DB (optional) | Pure Rust |

The only runtime requirement is the NVIDIA GPU driver (`libcuda.so` on Linux,
`nvcuda.dll` on Windows). On macOS the crate compiles but returns
`UnsupportedPlatform` at runtime.

## Quick Start

```rust
use oxicuda::prelude::*;

fn main() -> Result<(), oxicuda::Error> {
    // Initialize driver and select GPU device
    let device = Device::get(0)?;
    let ctx = Context::new(device)?;
    let stream = Stream::new(&ctx)?;

    // Allocate device memory
    let mut d_a = DeviceBuffer::<f32>::zeroed(1024)?;
    let mut d_b = DeviceBuffer::<f32>::zeroed(1024)?;
    let mut d_c = DeviceBuffer::<f32>::zeroed(1024)?;

    // Copy host data to device
    d_a.copy_from_host(&host_a)?;
    d_b.copy_from_host(&host_b)?;

    // Launch a GEMM: C = alpha * A @ B + beta * C
    let handle = BlasHandle::new(&stream)?;
    handle.gemm(
        Transpose::None, Transpose::None,
        m, n, k,
        1.0f32,            // alpha
        &d_a, lda,
        &d_b, ldb,
        0.0f32,            // beta
        &mut d_c, ldc,
    )?;

    stream.synchronize()?;

    // Copy result back to host
    let mut result = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut result)?;
    Ok(())
}
```

## Crate Overview

| Crate | CUDA Equivalent | Description | SLoC | Tests |
|-------|-----------------|-------------|------|-------|
| **Vol.1 -- Foundation** | | | | |
| `oxicuda-driver` | Driver API | FFI, device/context/stream/event/module | 13,508 | 383 |
| `oxicuda-memory` | cuMemAlloc | DeviceBuffer, PinnedBuffer, unified, pool | 5,297 | 211 |
| `oxicuda-launch` | cuLaunchKernel | Dim3, LaunchParams, `launch!` macro | 5,112 | 214 |
| `oxicuda-runtime` | CUDA Runtime | High-level cudaRT API layer | 2,521 | 46 |
| **Vol.2 -- PTX Codegen & Autotuner** | | | | |
| `oxicuda-ptx` | nvcc / CUTLASS | PTX IR, codegen DSL, Tensor Core gen | 31,764 | 934 |
| `oxicuda-autotune` | -- | Search space, benchmark, tuning DB | 15,665 | 449 |
| **Vol.3 -- Linear Algebra** | | | | |
| `oxicuda-blas` | cuBLAS | BLAS L1/L2/L3, GEMM, batched, elementwise | 28,379 | 776 |
| **Vol.4 -- Deep Learning** | | | | |
| `oxicuda-dnn` | cuDNN | Conv, attention, MoE, norm, pool, quantize | 39,297 | 1,075 |
| **Vol.5 -- Scientific Computing** | | | | |
| `oxicuda-fft` | cuFFT | Stockham, radix-2/4/8, Bluestein, 1D/2D/3D | 14,741 | 418 |
| `oxicuda-sparse` | cuSPARSE | CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM | 15,955 | 406 |
| `oxicuda-solver` | cuSOLVER | LU, QR, SVD, Cholesky, eig, CG, GMRES | 19,297 | 447 |
| `oxicuda-rand` | cuRAND | Philox, MRG32k3a, Sobol, distributions | 12,518 | 368 |
| **Vol.6 -- Signal Processing** | | | | |
| `oxicuda-signal` | -- | Audio/image DSP, DCT, DWT, IIR/FIR filters | 12,276 | 414 |
| **Vol.7 -- Computation Graph** | | | | |
| `oxicuda-graph` | CUDA Graphs | Graph capture, dep-sorted exec, events | 6,563 | 241 |
| **Vol.8 -- GPU Training** | | | | |
| `oxicuda-train` | -- | AMP, grad accum/clip, LR schedulers, optimizers | 8,106 | 250 |
| `oxicuda-quant` | -- | INT8/INT4/FP8 quantization, block-scaled | 5,726 | 198 |
| **Vol.9 -- Inference Engine** | | | | |
| `oxicuda-infer` | -- | KV-cache, paged attention, speculative decode | 7,754 | 297 |
| `oxicuda-dist-infer` | -- | Tensor/pipeline parallelism, distributed infer | 4,374 | 133 |
| `oxicuda-lm` | -- | BPE tokenizer, vocab, sampling strategies | 5,781 | 226 |
| **Vol.10 -- Reinforcement Learning** | | | | |
| `oxicuda-rl` | -- | Replay buffers, policy dists, PPO/DQN/SAC/TD3 | 11,280 | 425 |
| **Backends** | | | | |
| `oxicuda-backend` | -- | Backend trait abstraction | 484 | 10 |
| `oxicuda-primitives` | CUB | GPU scan, reduce, sort, histogram | 4,502 | 142 |
| `oxicuda-metal` | -- | Metal compute backend (macOS) | 4,395 | 152 |
| `oxicuda-vulkan` | -- | Vulkan Compute backend | 5,116 | 86 |
| `oxicuda-webgpu` | -- | WebGPU backend | 3,948 | 129 |
| `oxicuda-rocm` | -- | AMD ROCm backend | 3,739 | 104 |
| `oxicuda-levelzero` | -- | Intel oneAPI / LevelZero backend | 6,216 | 103 |
| **Vol.17 -- Generative AI** | | | | |
| `oxicuda-gen` | -- | Diffusion (DDPM/DDIM/DPM-Solver++/Flow Matching), CFG, VAE, LoRA | 13,605 | 520 |
| **Vol.18 -- Graph Neural Networks** | | | | |
| `oxicuda-gnn` | -- | CSR/COO/Hetero graphs, GCN/GAT/GraphSAGE/GIN, pooling | 16,973 | 613 |
| **Vol.19 -- State Space Models** | | | | |
| `oxicuda-mamba` | -- | HiPPO-NPLR, S4D/S5 selective scan, Mamba SSM, RWKV | 14,798 | 627 |
| **Vol.20 -- Vision Transformers** | | | | |
| `oxicuda-vision` | -- | ViT, patch embedding, CLIP towers | 19,287 | 768 |
| **Vol.21 -- Audio/Speech ML** | | | | |
| `oxicuda-audio` | -- | Conformer, Wav2Vec2, CTC/RNN-T, WaveNet, SpecAugment, x-vector | 16,665 | 669 |
| **Vol.22 -- Time-Series Forecasting** | | | | |
| `oxicuda-timeseries` | -- | TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN | 20,036 | 615 |
| **Vol.23 -- Bayesian Deep Learning** | | | | |
| `oxicuda-bayes` | -- | Variational inference, MC Dropout, Deep Ensembles, SWAG, Laplace | 18,258 | 606 |
| **Vol.24 -- Federated Learning** | | | | |
| `oxicuda-federated` | -- | FedAvg/FedProx/SCAFFOLD/FedAdam, DP, secure aggregation | 10,530 | 448 |
| **Vol.25 -- Neural Architecture Search** | | | | |
| `oxicuda-nas` | -- | DARTS, supernet, NSGA-II, hardware-aware FLOPs predictor | 8,543 | 302 |
| **Vol.26 -- Self-Supervised Learning** | | | | |
| `oxicuda-ssl` | -- | SimCLR/MoCo/BYOL/Barlow Twins/MAE/DINO | 13,881 | 437 |
| **Vol.27 -- Adversarial Robustness** | | | | |
| `oxicuda-adversarial` | -- | FGSM/PGD/CW/TRADES/MART | 12,038 | 505 |
| **Vol.28 -- Multi-Modal Learning** | | | | |
| `oxicuda-multimodal` | -- | Cross-modal attention, CLIP/ImageBind | 12,356 | 392 |
| **Vol.29 -- Continual Learning** | | | | |
| `oxicuda-continual` | -- | EWC/SI/PackNet/GEM/DER++ | 14,929 | 501 |
| **Vol.30 -- 3D Geometry & Point Clouds** | | | | |
| `oxicuda-geometry3d` | -- | FPS/kNN/PointNet/DGCNN/ICP | 14,490 | 461 |
| **Vol.31 -- Physics-Informed Neural Networks** | | | | |
| `oxicuda-pinn` | -- | PINN/NeuralODE/FNO/DeepONet | 18,135 | 636 |
| **Vol.32 -- RLHF & Alignment** | | | | |
| `oxicuda-rlhf` | -- | DPO/IPO/KTO/ORPO/PPO-RLHF/reward-model | 9,707 | 361 |
| **Vol.33 -- Meta-Learning** | | | | |
| `oxicuda-meta` | -- | MAML/FOMAML/ANIL/Reptile/ProtoNet | 13,338 | 363 |
| **Vol.34 -- Neural Radiance Fields** | | | | |
| `oxicuda-nerf` | -- | NeRF/Instant-NGP/Mip-NeRF/TensoRF | 12,510 | 358 |
| **Vol.35 -- Mixture of Experts** | | | | |
| `oxicuda-moe` | -- | Switch/Top-K/Expert-Choice/Soft-MoE | 9,382 | 303 |
| **Vol.36 -- Tabular Deep Learning** | | | | |
| `oxicuda-tabular` | -- | TabNet/SAINT/FT-Transformer/NODE | 15,823 | 466 |
| **Vol.37 -- Anomaly Detection** | | | | |
| `oxicuda-anomaly` | -- | DeepSVDD/LOF/COPOD/Mahalanobis/IsoForest | 23,650 | 582 |
| **Vol.38 -- Quantum Simulation** | | | | |
| `oxicuda-quantum` | -- | State-vector/VQE/QAOA/QML-kernels | 13,280 | 419 |
| **Vol.39 -- Approximate Nearest Neighbor** | | | | |
| `oxicuda-ann` | -- | HNSW/IVF/PQ/IVFPQ/LSH | 13,332 | 353 |
| **Vol.40 -- Recommender Systems** | | | | |
| `oxicuda-recsys` | -- | ALS/BPR/NCF/DeepFM/SASRec/LightGCN | 15,092 | 417 |
| **Vol.41 -- Causal Inference** | | | | |
| `oxicuda-causal` | -- | NOTEARS/IPW/S-T-X-learners/DML/CausalForest | 26,133 | 735 |
| **Vol.42 -- Parameter-Efficient Fine-Tuning** | | | | |
| `oxicuda-peft` | -- | LoRA/QLoRA/AdaLoRA/Prefix-Tuning | 19,975 | 643 |
| **Vol.43 -- Knowledge Distillation** | | | | |
| `oxicuda-distill` | -- | Hinton/FitNets/AT/CRD/DML/ZSKD | 11,889 | 447 |
| **Vol.44 -- Optimal Transport** | | | | |
| `oxicuda-ot` | -- | Sinkhorn/EMD/Gromov-Wasserstein/Wasserstein-kmeans | 24,969 | 628 |
| **Vol.45 -- Spiking Neural Networks** | | | | |
| `oxicuda-snn` | -- | LIF/IF/BPTT/STBP/SLAYER/STDP/ANN→SNN | 17,606 | 577 |
| **Vol.46 -- Differential Privacy** | | | | |
| `oxicuda-privacy` | -- | DP-FTRL/DP-Adam/RDP/zCDP/PRV/OUE/RAPPOR | 16,590 | 696 |
| **Vol.47 -- Hyperdimensional Computing** | | | | |
| `oxicuda-hdc` | -- | Binary/integer/complex HVs, AM/classifier | 10,405 | 407 |
| **Vol.48 -- Evolutionary Algorithms** | | | | |
| `oxicuda-evol` | -- | CMA-ES/NSGA-II/MOEA-D/NEAT/DE/PSO/ACO | 17,544 | 489 |
| **Vol.49 -- Topological Data Analysis** | | | | |
| `oxicuda-tda` | -- | Vietoris-Rips/persistent-homology/Mapper | 12,009 | 379 |
| **Vol.50 -- Tensor Networks** | | | | |
| `oxicuda-tn` | -- | MPS/MPO/DMRG/TEBD/PEPS/TT-cross/CP-ALS/einsum | 25,583 | 501 |
| **Vol.51 -- Sequence Models** | | | | |
| `oxicuda-seq` | -- | HMM/CRF/Kalman/EKF/Viterbi/Baum-Welch | 20,887 | 617 |
| **Vol.52 -- Numerical PDE Solvers** | | | | |
| `oxicuda-pde` | -- | FDM/FEM/spectral/multigrid/CG | 23,803 | 680 |
| **Vol.53 -- Manifold Learning** | | | | |
| `oxicuda-manifold` | -- | t-SNE/UMAP/LLE/Isomap/Diffusion-Maps/SMACOF | 26,639 | 520 |
| **Vol.54 -- Statistical Inference** | | | | |
| `oxicuda-stats` | -- | t-test/ANOVA/KS/bootstrap/regression/power | 33,056 | 938 |
| **Vol.55 -- Streaming Sketches** | | | | |
| `oxicuda-sketch` | -- | HyperLogLog/Count-Min/Bloom/t-Digest/MinHash | 11,692 | 456 |
| **Vol.56 -- Survival Analysis** | | | | |
| `oxicuda-survival` | -- | Kaplan-Meier/Cox-PH/AFT/Fine-Gray/Brier | 31,530 | 785 |
| **Vol.57 -- Convex Optimization** | | | | |
| `oxicuda-cvx` | -- | LP/QP/SOCP/SDP/ADMM/FISTA/proximal-gradient | 20,511 | 616 |
| **Vol.58 -- Compressed Sensing** | | | | |
| `oxicuda-cs` | -- | OMP/CoSaMP/IHT/AMP/K-SVD/LASSO/nuclear-norm | 10,537 | 253 |
| **Vol.59 -- Graph Algorithms** | | | | |
| `oxicuda-graphalg` | -- | BFS/DFS/Dijkstra/MST/flow/matching/SCC/TSP | 11,913 | 327 |
| **Vol.60 -- Numerical Analysis** | | | | |
| `oxicuda-numeric` | -- | Root-finding/quadrature/special-functions/ODE/interpolation | 13,644 | 466 |
| **Vol.61 -- 2D Computational Geometry** | | | | |
| `oxicuda-geom2d` | -- | Delaunay/Voronoi/convex-hull/sweep-line | 10,028 | 282 |
| **Umbrella** | | | | |
| `oxicuda` | -- | Umbrella re-export crate | 21,994 | 521 |
| | | **Total** | **~1,063,889** | **32,320** |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `driver` | on | CUDA driver API layer |
| `memory` | on | Device/pinned/unified memory |
| `launch` | on | Kernel launch primitives |
| `ptx` | off | PTX IR codegen DSL |
| `autotune` | off | Runtime autotuner with disk cache |
| `blas` | off | BLAS L1/L2/L3 and GEMM |
| `dnn` | off | Deep learning ops (conv, attention, MoE, norm) |
| `fft` | off | FFT transforms |
| `sparse` | off | Sparse matrix operations |
| `solver` | off | Linear solvers (LU, QR, SVD, Cholesky, CG) |
| `rand` | off | GPU random number generation |
| `primitives` | off | CUB-equivalent GPU primitives |
| `pool` | off | Async memory pool (CUDA 11.2+) |
| `vulkan` | off | Vulkan Compute backend |
| `metal` | off | Metal backend (macOS) |
| `webgpu` | off | WebGPU backend |
| `rocm` | off | AMD ROCm backend |
| `level-zero` | off | Intel oneAPI / LevelZero backend |
| `wasm-backend` | off | WebAssembly + WebGPU browser target |
| `gpu-tests` | off | Enable GPU hardware tests |
| `full` | off | Enable all features |

## Performance Targets

| Operation | Target vs CUDA | Notes |
|-----------|----------------|-------|
| SGEMM (FP32) | >= 95% cuBLAS | Autotuned tile sizes |
| HGEMM (FP16) | >= 95% cuBLAS | Tensor Core WMMA/MMA |
| Batch GEMM | >= 95% cuBLAS | Stream-K scheduling |
| Convolution (FP16) | >= 90% cuDNN | Implicit GEMM + Winograd |
| FlashAttention | >= 90% FA2 | Tiled, causal mask |
| FFT (power-of-2) | >= 90% cuFFT | Stockham radix-2/4/8 |
| SpMV (CSR) | >= 85% cuSPARSE | Architecture-tuned |
| LU / QR / SVD | >= 85% cuSOLVER | Blocked panel factorization |

## Supported GPU Architectures

| Architecture | SM | Codename | Key Features |
|--------------|----|----------|--------------|
| Turing | 7.5 | TU10x | INT8 Tensor Cores, RT Cores |
| Ampere | 8.0 | GA100 | TF32, FP64 Tensor Cores, Async Copy |
| Ampere | 8.6 | GA10x | Third-gen Tensor Cores |
| Ada Lovelace | 8.9 | AD10x | FP8 Tensor Cores |
| Hopper | 9.0 | GH100 | WGMMA, TMA, FP8, DPX |
| Blackwell | 10.0 | GB10x | FP4, Fifth-gen Tensor Cores |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Full support | Primary development target |
| Windows x86_64 | Full support | nvcuda.dll loaded at runtime |
| macOS (ARM/x86) | Compile-only | Returns `UnsupportedPlatform` at runtime |

## Building

```bash
# Default build (no GPU features)
cargo build

# With all GPU features
cargo build --features "ptx,autotune,blas,dnn,fft,sparse,solver,rand"

# Full build (all features including backends)
cargo build --features full

# Check without GPU
cargo check --all-targets
```

## Testing

```bash
# Unit tests (no GPU required)
cargo test

# Full test suite with GPU hardware
cargo test --features gpu-tests

# Run with nextest
cargo nextest run --all-features
```

## Roadmap

**Released (v0.2.0) -- 2026-06-16** *(32,320 tests passing, ~1.06M SLoC, 73 crates)*
- Vol.1: Driver, Memory, Launch, Runtime -- foundation layer (4 crates)
- Vol.2: PTX codegen DSL, autotuner engine (2 crates)
- Vol.3: Full BLAS L1/L2/L3 with Tensor Core GEMM, SYR2K two-operand cross-product variant
- Vol.4: Convolution, FlashAttention, MoE, normalization, pooling, quantization
- Vol.5: FFT, sparse, solver, RNG (4 crates)
- Vol.6: Signal processing -- audio/image DSP, DCT, DWT, IIR/FIR filters
- Vol.7: Computation graph -- capture API, dep-sorted scheduling, parallel executor
- Vol.8: GPU training -- AMP, optimizers, LR schedulers, checkpointing, quantization (2 crates)
- Vol.9: Inference engine -- KV-cache, speculative decode, distributed infer, LM (3 crates)
- Vol.10: Reinforcement learning -- replay buffers, policy dists, PPO/DQN/SAC/TD3
- Backends: Metal, Vulkan, WebGPU, ROCm, LevelZero (7 crates)
- Vol.17: Generative AI -- diffusion schedulers, CFG, VAE, LoRA
- Vol.18: Graph Neural Networks -- GCN/GAT/GraphSAGE/GIN, pooling
- Vol.19: State Space Models -- HiPPO-NPLR, S4D/S5, Mamba SSM, RWKV
- Vol.20: Vision Transformers & CLIP -- ViT, patch embedding, dual-tower CLIP
- Vol.21: Audio/Speech ML -- Conformer, Wav2Vec2, CTC/RNN-T, WaveNet, SpecAugment
- Vol.22: Time-Series Forecasting -- TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN
- Vol.23: Bayesian Deep Learning -- variational inference, MC Dropout, Ensembles, Laplace
- Vol.24: Federated Learning -- FedAvg/FedProx/SCAFFOLD/FedAdam, DP, secure aggregation
- Vol.25: Neural Architecture Search -- DARTS, supernet, NSGA-II, hardware-aware predictor
- Vol.26--61: SSL, Adversarial, Multimodal, Continual, 3D Geometry, PINN, RLHF, Meta-Learning, NeRF, MoE, Tabular, Anomaly, Quantum, ANN, RecSys, Causal, PEFT, Distillation, OT, SNN, DP, HDC, Evolutionary, TDA, Tensor Networks, Sequence Models, PDE, Manifold, Statistics, Sketches, Survival, CVX, Compressed Sensing, Graph Algorithms, Numerical Analysis, 2D Geometry

**Released (v0.3.0) -- 2026-06-25** *(36,984 tests passing, ~1.23M SLoC, 73 crates)*
- Workspace-wide CPU algorithm implementation depth across all volumes
- `oxicuda-rlhf` now fully gradient-capable: 20+ loss gradients verified against finite differences
- Cross-crate integration: `oxicuda-gnn` -> `oxicuda-sparse` SpMM, `oxicuda-timeseries` -> `oxicuda-fft`
- Research cores: PointFlow CNF (`oxicuda-geometry3d`), Bark RVQ neural codec (`oxicuda-audio`)
- `oxicuda-ptx`: CpAsyncGenerator (cp.async codegen) and cost-gated FusionCostModel
- Latent-bug fixes, notably the `oxicuda-ot` network-simplex EMD solver that was silently broken for all n>=4

**Next (v0.4.0) -- Unreleased**
- Published documentation on docs.rs
- GPU hardware benchmark validation (CI regression tracking)
- v1.0 completion criteria verification (see TODO.md)

## Quick Links

- [API Documentation](https://docs.rs/oxicuda)
- [GitHub Repository](https://github.com/cool-japan/oxicuda)
- [COOLJAPAN Ecosystem](https://github.com/cool-japan)
- [Changelog](CHANGELOG.md)

## Related COOLJAPAN Projects

| Project | Description |
|---------|-------------|
| [SciRS2](https://github.com/cool-japan/scirs2) | Scientific computing (NumPy/SciPy equivalent) |
| [ToRSh](https://github.com/cool-japan/torsh) | Tensor operations (PyTorch equivalent) |
| [TrustformeRS](https://github.com/cool-japan/trustformers) | Transformer models |
| [OxiONNX](https://github.com/cool-japan/oxionnx) | ONNX neural network inference |
| [OxiBLAS](https://github.com/cool-japan/oxiblas) | Pure Rust BLAS |
| [OxiFFT](https://github.com/cool-japan/oxifft) | Pure Rust FFT |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Copyright

(C) 2026 COOLJAPAN OU (Team KitaSan)
