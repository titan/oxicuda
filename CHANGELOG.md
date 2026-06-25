# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-25

This release adds no new crates (still 73). It is a depth pass: implementing genuine, CPU-verifiable algorithms across existing crates, reviving orphaned-but-real modules, wiring cross-crate paths, and fixing latent bugs surfaced along the way. Every algorithm was added with correctness tests (finite-difference-verified gradients, analytic-front residuals, bit-exact cross-path checks).

### Added

- Cross-crate integration: `oxicuda-gnn` `GcnLayer::forward_sparse` routes message passing through `oxicuda-sparse` HostCsr SpMM (sparse path bit-exact vs the dense path); `oxicuda-timeseries` `detect_period_fft` computes Wiener–Khinchin autocorrelation via `oxicuda-fft` rfft/irfft (matches the direct O(T²) result to 3.3e-12). Both dependencies are declared `{ workspace = true }` with no dependency cycle.
- `oxicuda-geometry3d`: PointFlow continuous-normalizing-flow core (reverse-time invertibility 1.1e-16, exact-trace logdet vs finite-difference 6.2e-11). Trained generation parts deferred.
- `oxicuda-audio`: residual-vector-quantization neural-codec core (Bark RVQ — monotone reconstruction error, exact index recovery, k-means fit non-increase). Trained generation parts deferred.
- `oxicuda-rlhf` is now fully gradient-capable: 20+ analytic, central-finite-difference-verified gradients across 17 loss modules — the closed-form preference family (DPO/IPO/KTO/SimPO/ORPO/BCO/DPOP/SLiC/Step-DPO/sDPO/RRHF/length-DPO/online-DPO), reward models (Bradley-Terry, soft-BT RLAIF, PRM), and RL estimators (PPO, GRPO clip + k3-KL, REBEL, RLOO, SAC-RLHF). Previously forward-value-only.
- `oxicuda-evol`: WFG1-9, ZDT4/6, and DTLZ3-7 multi-objective test problems (analytic-front residuals at machine epsilon).
- `oxicuda-seq`: Gaussian-HMM Baum-Welch EM (monotone log-likelihood); Kalman tracking and CRF chunker examples.
- `oxicuda-audio`: rational-quadratic spline flow (Durkan 2019) as a VITS stochastic-duration dequantizer.
- `oxicuda-hdc`: measured Hopfield-capacity and bundle-SNR scaling-law curves.
- `oxicuda-snn`: NARMA-10 reservoir benchmark and STDP sign/shape verification; sparse spike encoding and event-driven LIF.
- `oxicuda-ptx`: `CpAsyncGenerator` emitting `cp.async.cg/ca.global` PTX with multi-stage commit_group/wait_group pipelining and a pre-sm_80 fallback; `FusionCostModel` register-pressure + shared-memory + ILP heuristic wired into `kernel_fusion::plan_fusion`.
- `oxicuda-tabular`: analytic backward passes (FT-Transformer/TabNet/SAINT/NODE) with softmax/sparsemax/entmax Jacobians.
- `oxicuda-backend`: mixed-precision GEMM (binary16/bfloat16 round-to-nearest-even, FP32 accumulate) and conv2d backward.
- `oxicuda-quant`: GGUF v3 container read/write.
- `oxicuda-graph`: reduction-pattern fusion pass.
- `oxicuda-gnn`: edge-feature support in GAT.
- `oxicuda-runtime`: device-pointer cast / typed-slice helpers and stream-capture bookkeeping.
- `oxicuda-privacy`: Philox and ChaCha20 counter-based RNGs, a DP-Adam convergence harness, and PATE-GAN/DP-GAN.
- `oxicuda-nas`: Bayesian-optimization GP predictor and Once-for-All.
- `oxicuda-meta`: MAML inner-loop integration.
- `oxicuda-pinn`: PI-DeepONet forward-mode AD, a tree-GP symbolic regressor, and batched ODE solvers.
- `oxicuda-gen`: full U-Net assembly and LoRA checkpoint round-trip.
- `oxicuda-geometry3d`: straight-through FPS gradients.
- `oxicuda-pde`: convergence-verified Poisson, Crank-Nicolson, and multigrid solvers.
- `oxicuda-solver`: MINRES/QMR/LSQR Krylov solvers and Gilbert-Peierls sparse LU.
- `oxicuda-blas`: 2:4 structured-sparse SpGEMM with Ampere `mma.sp` codegen.
- `oxicuda-autotune`: persistent LRU tune-cache.
- `oxicuda-cvx`: fluent LP/QP/SOCP/SDP solver builder.
- `oxicuda-tda`: persistence and Mapper examples.
- `oxicuda-sketch`: `CuckooFilter32`.
- `oxicuda-causal`: discrete conditional-independence tests (chi-square / G-test) and the PC algorithm.
- `oxicuda-peft`: AdaLoRA, TIES, and DARE.
- `oxicuda-dist-infer`: autonomous `RebalanceMonitor` and `ElasticScaler`.
- Test suite expanded to 36,984 passing tests (workspace-wide, `--all-features`; 36,546 with default features), up from 32,320 at 0.2.0.

### Changed

- Wired 11 orphaned-but-real modules across 6 crates (180 previously-dead tests revived) — `oxicuda-evol` CMA-ME, `oxicuda-manifold` Isomap / parametric-tSNE / geodesic-regression, `oxicuda-rand` cuRAND-style host API, `oxicuda-rlhf` dpo/ppo loss + reward-norm, `oxicuda-stats` GMM + ARIMA, `oxicuda-timeseries` DTW; plus 4 more orphaned modules in `oxicuda-nas` (Once-for-All, NAS-Bench) and `oxicuda-geometry3d`. These were real, tested algorithm files never declared in `mod.rs`.
- `oxicuda-ptx`: kernel fusion is now cost-gated — `FusionCostModel` replaces the former unconditional acceptance of every structurally-legal fusion candidate with a register-spill / shared-memory / benefit-threshold fuse-or-refuse decision.

### Fixed

- `oxicuda-ot`: `network_simplex` `find_cycle` had an inverted closing-parity condition that failed 100% of n≥4 dense EMD instances — the exact optimal-transport solver was silently broken for all non-trivial problem sizes. Rewrote the alternating-axis cycle DFS; now a 100% solve rate for n=5..64, agreeing with Sinkhorn to relative gap < 8e-3 as ε→0.
- `oxicuda-peft`: corrected an NF4 codebook typo — `NF4_TABLE[3]` and `nf4_dequant_ptx` held `-0.3949468731880188`; the canonical QLoRA/bitsandbytes value (and the crate's own `nf4_quant.rs`) is `-0.39491748809814453`.
- `oxicuda-rand`: fixed an MRG32k3a `[0,1)` contract violation and scrambled-Sobol Inf/NaN (an errant `÷2^31` should have been `÷2^32`).
- `oxicuda-stats`: fixed a GMM kmeans++ degenerate fallback that could panic with a reversed range or produce wrong-length centers.
- `oxicuda-solver`: fixed a sparse-LU pivoting bug.
- `oxicuda-nas`: fixed 2 latent compile bugs (missing `PartialEq` derives) in the previously-never-compiled Once-for-All module.

## [0.2.0] - 2026-06-16

### Added

- Wave AAA+64 feature expansion: Extended Persistence and Discrete Morse theory (`oxicuda-tda`), Parametric UMAP (`oxicuda-manifold`), Fisher Information estimation (`oxicuda-bayes`), and adaptive RK45 integration with Richardson extrapolation for ODE/PDE solvers.
- Expanded CUDA kernel coverage across the driver, memory, launch, and backend layers.
- Test suite grew to 32,320 passing tests (up from 23,535 at 0.1.8).

### Changed

- Workspace-wide reliability pass: eliminated every `.unwrap()` from all `crates/*/src/` (production code and test modules now use descriptive `.expect(...)`), maintaining zero clippy warnings under `-D warnings`.

### Fixed

- `oxicuda-geometry3d`: corrected a sign error in the symmetric-3×3 Jacobi eigensolver (`crates/oxicuda-geometry3d/src/mesh/obb.rs`) whose rotation angle used `app - aqq` instead of `aqq - app`. The defect doubled the off-diagonal each sweep instead of annihilating it, so `Obb::fit_pca` returned eigenvectors tilted from the true principal axes and produced a non-tight oriented bounding box.

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
