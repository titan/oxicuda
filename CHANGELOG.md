# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8] - 2026-05-21

### Changed

- Maintenance release: numerical-stability refinements in HMC variational sampler, stream-ordered allocator tuning, and TriMap reduction polish (`crates/oxicuda-bayes/src/variational/hmc.rs`, `crates/oxicuda-driver/src/stream_ordered_alloc.rs`, `crates/oxicuda-manifold/src/reduction/trimap.rs`)

## [0.1.7] - 2026-05-16

### Added

- `oxicuda-blas`: SYR2K Tensor Core kernel with two-operand cross-product variant — efficient symmetric rank-2k update using Tensor Core hardware units with fused A×Bᵀ + B×Aᵀ accumulation (`crates/oxicuda-blas/src/level3/syr2k.rs`)
- CUDA kernel enhancements across multiple subsystems (driver, memory, launch, blas, and backend layers)
- MOS (Multi-Operation Scheduling) improvements for GPU task orchestration

## [0.1.6] - 2026-05-08

### Added

- `oxicuda-blas`: Tensor Core fast path for SYRK — triangle-masked GEMM kernel that eliminates redundant symmetric writes while hitting Tensor Core hardware units (`crates/oxicuda-blas/src/level3/syrk.rs`, `syr2k.rs`)
- Vol.26 `oxicuda-adversarial` (Adversarial robustness: attack generation, adversarial training primitives)
- Vol.27 `oxicuda-ssl` (Self-Supervised Learning: contrastive, masked-autoencoder, and distillation scaffolding)
- Vol.28 `oxicuda-continual` (Continual Learning: PackNet architecture, task-incremental training, forgetting mitigation)
- Vol.29 `oxicuda-multimodal` (Multimodal Learning: cross-modal fusion, shared-encoder scaffolding)
- Vol.30 `oxicuda-geometry3d` (3-D Geometry: point-cloud ops, mesh primitives, spatial indexing)
- Vol.31 `oxicuda-pinn` (Physics-Informed Neural Networks: PDE loss terms, residual sampling)
- Vol.32 `oxicuda-ann` (Approximate Nearest Neighbour: flat / IVF / IVFPQ / HNSW / LSH / PQ / KNN-graph, Hamming / L2 / inner-product distances, SQ4/SQ8 quantizers, k-NN heap select)
- Vol.33 `oxicuda-anomaly` (Anomaly Detection: Mahalanobis / COPOD density estimators, kNN score, LOF)
- Vol.34 `oxicuda-causal` (Causal Inference: do-calculus primitives, causal graph scaffolding)
- Vol.35 `oxicuda-meta` (Meta-Learning: MAML / Prototypical-Network scaffolding)
- Vol.36 `oxicuda-moe` (Mixture-of-Experts: top-k routing, expert dispatch, load-balancing loss)
- Vol.37 `oxicuda-nerf` (Neural Radiance Fields: ray-marching primitives, positional encoding, volume rendering)
- Vol.38 `oxicuda-quantum` (Quantum-Classical Hybrid: qubit-state simulation primitives, variational circuit scaffolding)
- Vol.39 `oxicuda-recsys` (Recommender Systems: collaborative filtering, embedding lookup, ranking loss)
- Vol.40 `oxicuda-rlhf` (RLHF: reward-model scaffolding, PPO/DPO wrappers, KL-penalty helpers)
- Vol.41 `oxicuda-tabular` (Tabular ML: feature encoding, gradient-boosted tree scaffolding, TabNet blocks)

## [0.1.5] - 2026-05-03

### Added

- macOS stub integration test suite (`crates/oxicuda-driver/tests/macos_stub.rs`) — 9 tests asserting every `gpu-tests`-gated entrypoint returns `Err(UnsupportedPlatform)` or `Err(NotInitialized)` on macOS
- `[package.metadata.docs.rs]` configuration added to all 34 subcrate `Cargo.toml` files; `cargo doc --all-features` now builds cleanly workspace-wide
- Vol.17 `oxicuda-gen` (Generative AI: DDPM/DDIM/DPM-Solver++/Flow Matching schedulers, classifier-free guidance, VAE codec, LoRA adapters, score-network blocks)
- Vol.18 `oxicuda-gnn` (Graph Neural Networks: CSR/COO/Heterogeneous graphs, scatter / gather / aggregate primitives, GCN / GAT / GAT-v2 / GraphSAGE / GIN layers, global / Top-K / DiffPool pooling, Set2Set readout)
- Vol.19 `oxicuda-mamba` (State Space Models: HiPPO-NPLR initialization, S4D / S5 selective scan, Mamba SSM block, RWKV channel-mixing, gated SSM)
- Vol.20 `oxicuda-vision` (Vision Transformers & CLIP: patch embedding, ViT encoder blocks, learnable positional embeddings, CLS token, CLIP-style image / text tower scaffolding)
- Vol.21 `oxicuda-audio` (Audio / Speech ML: Conformer encoder, Wav2Vec2 feature extractor, CTC / RNN-T loss, WaveNet causal stack, SpecAugment, x-vector speaker embedding)
- Vol.22 `oxicuda-timeseries` (Time-Series Forecasting: TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN reversible normalization)
- Vol.23 `oxicuda-bayes` (Bayesian deep learning: variational inference, Bayesian linear / conv layers, Flipout, ELBO / IWAE, normalizing flows, MC Dropout, Deep Ensembles, SWAG, Laplace approximation, calibration / ECE)
- Vol.24 `oxicuda-federated` (Federated learning: FedAvg / FedProx / SCAFFOLD / FedAdam, PowerSGD / QSGD / Top-K / Random-K compression, Gaussian / Laplacian / Moments / RDP / PATE differential privacy, Shamir-based secure aggregation, random / stratified client selection)
- Vol.25 `oxicuda-nas` (Neural Architecture Search: DARTS bilevel optimizer with derived discrete cells, one-shot weight-shared Supernet with path sampling and Slimmable widths, evolutionary NSGA-II with non-dominated sort and crowding distance, hardware-aware FLOPs predictor)
- All three new leaf crates carry `[dependencies] thiserror.workspace = true` only — no internal `oxicuda-*` dependencies, fully standalone, 100% Pure Rust
- 8 missing per-crate `README.md` files created (`oxicuda-bayes`, `oxicuda-federated`, `oxicuda-gen`, `oxicuda-gnn`, `oxicuda-mamba`, `oxicuda-nas`, `oxicuda-timeseries`, `oxicuda-vision`) so `cargo publish` no longer errors on `readme = "README.md"`

### Changed

- Preemptive `splitrs` of 5 near-cap source files: `batched.rs` (1950→1288 LoC), `tensor_backend/ops.rs` (1986→1673 LoC), `fp4_fp6_ops.rs` (1955→1587 LoC), `ir/instruction.rs` (1973→1244 LoC), `tui_explorer.rs` (1931→1438 LoC); test blocks extracted to sibling `*/tests.rs` files
- `device_attrs.rs` integration test tightened to assert error variant (not just `is_err()`)
- `launch-overhead-driver-crate` TODO entry collapsed to canonical cross-reference
- All internal dependency versions bumped to 0.1.5
- Workspace test count: **9,568 passing**, 2 skipped (GPU-gated on macOS) — up from prior ~9,000-something
- Repaired 22 clippy warnings without introducing any `#[allow]` attributes — `needless_range_loop` ×15, `useless_vec` ×3, `manual_repeat_n` ×3, `ptr_arg` ×1, `nonminimal_bool` ×1
- Fixed 6 pre-existing compile errors — `unused-named-args` in `format!` PTX templates ×4, deprecated `std::f32::LN_2` reference, two `explicit-deref-pattern` lints
- Statistical test `compression::randomk::tests::random_sparsify_unbiased` retuned from `n_trials=500` (1.1σ) to `n_trials=5_000` (3.6σ) — eliminates the historical flake

## [0.1.4] - 2026-04-18

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.4

## [0.1.3] - 2026-04-17

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.3

## [0.1.2] - 2026-04-14

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.2

## [0.1.1] - 2026-04-14

### Added

- `oxicuda-blas`: New elementwise operations — `Ceil`, `Floor`, `HardSigmoid`, `HardSwish`, `Softplus`, and `LeakyRelu`

### Changed

- General enhancements across crates: improved robustness, performance, and internal code quality

## [0.1.0] - 2026-04-13

### Added

**Foundation (Vol.1 — 4 crates, 22,972 SLoC)**
- `oxicuda-driver` (11,548 SLoC, 333 tests): CUDA Driver API wrapper with dynamic loading via libloading, device/context/stream/event/module management, multi-GPU context pool, occupancy queries
- `oxicuda-memory` (4,178 SLoC, 204 tests): Type-safe GPU memory management — DeviceBuffer<T>, PinnedBuffer<T>, unified memory, async pool, virtual memory, 2D/3D copies, peer transfer
- `oxicuda-launch` (4,728 SLoC, 207 tests): Type-safe kernel launch — Dim3, LaunchParams, launch! macro, cooperative launch, cluster launch (Hopper+), graph-based launch
- `oxicuda-runtime` (2,518 SLoC, 46 tests): High-level CUDA runtime wrapper — streams, events, texture objects, surface objects

**PTX Codegen & Autotuner (Vol.2 — 2 crates, 43,122 SLoC)**
- `oxicuda-ptx` (29,206 SLoC, 873 tests): Full PTX IR type system, Rust DSL for SM 7.5–SM 10.0, Tensor Core support (WMMA/MMA/WGMMA), kernel templates (GEMM, elementwise, reduction, softmax, scan, transpose, attention, BN, MoE, convolution), register pressure analysis, dead code elimination, constant folding, strength reduction
- `oxicuda-autotune` (13,916 SLoC, 408 tests): Search space definition, GPU benchmarking with statistical analysis, Bayesian optimization, simulated annealing, genetic algorithm, result DB (JSON), problem size interpolation, early stopping

**Linear Algebra (Vol.3 — 1 crate, 21,845 SLoC)**
- `oxicuda-blas` (21,845 SLoC, 604 tests): Full cuBLAS equivalent — BLAS Level 1/2/3, GEMM (SIMT/Tensor Core/Split-K), batched GEMM (standard/strided/grouped), precision coverage (F16/BF16/TF32/F32/F64/FP8), elementwise ops, reductions, epilogue fusion

**Deep Learning (Vol.4 — 1 crate, 34,711 SLoC)**
- `oxicuda-dnn` (34,711 SLoC, 960 tests): Full cuDNN equivalent — convolution (implicit GEMM/im2col/Winograd/direct/fused), FlashAttention v2 (forward/backward), PagedAttention, MoE (top-k routing, permutation, fusion), normalization (BN/LN/RMSNorm/GroupNorm), pooling, resize, speculative decoding, linear layers

**Scientific Computing (Vol.5 — 4 crates, 47,946 SLoC)**
- `oxicuda-fft` (9,749 SLoC, 295 tests): Stockham FFT, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, pruned FFT, 1D/2D/3D
- `oxicuda-sparse` (12,278 SLoC, 320 tests): CSR/CSC/COO/BSR/ELL/HYB/CSR5 formats, SpMV/SpMM/SpGEMM/SDDMM, ILU(0)/IC(0), Krylov solvers, auto-dispatch
- `oxicuda-solver` (15,804 SLoC, 373 tests): Dense LU/QR/SVD/Cholesky/eigendecomp, CG/BiCGSTAB/GMRES, tensor decomposition, matrix functions (exp/log/sqrt)
- `oxicuda-rand` (10,115 SLoC, 264 tests): Philox/MRG32k3a/XORWOW/Sobol PRNGs, uniform/normal/Poisson/exponential/gamma distributions, NIST statistical tests

**Signal Processing (Vol.6 — 1 crate, 6,037 SLoC)**
- `oxicuda-signal` (6,037 SLoC, 231 tests): Audio (MFCC, STFT, Mel filterbank), image processing (Gaussian blur, Sobel, morphology), DCT (types I–IV), DWT (Haar, Daubechies), IIR/FIR filtering, correlation

**Computation Graph (Vol.7 — 1 crate, 4,784 SLoC)**
- `oxicuda-graph` (4,784 SLoC, 175 tests): CUDA Graph capture, execution plan with dependency sorting, event synchronization, sequential/parallel executors

**GPU Training (Vol.8 — 2 crates, 10,244 SLoC)**
- `oxicuda-train` (5,927 SLoC, 165 tests): Mixed precision AMP (FP16/BF16 + loss scaling), gradient accumulation/clipping, EMA, LR schedulers (cosine/warmup/cyclic/polynomial), GPU-fused optimizers (Adam/AdamW/SGD/RMSProp/LAMB), checkpointing
- `oxicuda-quant` (4,317 SLoC, 150 tests): INT8/INT4/FP8 weight quantization, block-scaled FP4, GPTQ-style post-training quantization

**Inference Engine (Vol.9 — 3 crates, 11,929 SLoC)**
- `oxicuda-infer` (4,256 SLoC, 137 tests): PagedKvCache, prefix caching, speculative decoding, continuous batching
- `oxicuda-dist-infer` (3,279 SLoC, 80 tests): Distributed inference with tensor/pipeline parallelism, all-reduce primitives
- `oxicuda-lm` (4,394 SLoC, 182 tests): BPE tokenizer, vocabulary management, sampling strategies (greedy/top-k/top-p/beam)

**Reinforcement Learning (Vol.10 — 1 crate, 4,234 SLoC)**
- `oxicuda-rl` (4,234 SLoC, 164 tests): Replay buffers (Uniform/PER/N-step), policy distributions (Categorical/Gaussian/Deterministic), advantage estimators (GAE/TD-λ/V-trace/Retrace-λ), loss functions (PPO/DQN/SAC/TD3), observation/reward normalization, Env/VecEnv abstractions

**Backends & Primitives (7 crates, 11,234 SLoC)**
- `oxicuda-backend` (271 SLoC, 7 tests): ComputeBackend trait definition
- `oxicuda-primitives` (4,372 SLoC, 142 tests): CUB-equivalent parallel primitives (block reduce/scan/sort, warp ops)
- `oxicuda-metal` (1,186 SLoC, 52 tests): Apple Metal GPU backend (macOS/iOS)
- `oxicuda-vulkan` (1,445 SLoC, 38 tests): Vulkan Compute backend (cross-platform)
- `oxicuda-webgpu` (1,108 SLoC, 42 tests): WebGPU backend (browser/WASM)
- `oxicuda-rocm` (1,087 SLoC, 36 tests): AMD ROCm/HIP backend
- `oxicuda-levelzero` (1,765 SLoC, 44 tests): Intel oneAPI Level Zero backend

**Umbrella (1 crate)**
- `oxicuda` (19,614 SLoC, 494 tests): Re-exports all sub-crates, ComputeBackend trait with CudaBackend, OxiONNX GPU inference backend, ToRSh tensor backend, TrustformeRS transformer backend, global init/device pool
