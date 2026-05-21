# oxicuda-cs TODO

GPU-accelerated Compressed Sensing, Sparse Recovery, and Low-Rank Matrix Completion,
serving as a pure Rust replacement for SPGL1 / TFOCS / SPAMS-style toolkits.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.58).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 6,127 (64 files, tokei measurement)
- **Total lines (incl. comments+blanks):** 6,757
- **Tests:** 108 passing
- **Vol.58 scope:** Compressed sensing & sparse recovery primitives that complement
  oxicuda-blas / oxicuda-solver by providing L1/L0/nuclear-norm minimisation paths
  not covered by classical dense BLAS/LAPACK semantics.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `CsError` enum (14 variants: `NotConverged`, `ShapeMismatch`,
  `InvalidParameter`, `NumericalInstability`, `UnsupportedSmVersion`, `SingularMatrix`,
  `IndexOutOfBounds`, `DimensionMismatch`, `EmptyInput`, `SupportTooLarge`,
  `InvalidSparsity`, `InvalidRank`, `InvalidConfiguration`, `RecoveryFailed`) +
  `CsResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 boolean, Box-Muller normal),
  `CsHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `correlate`, `hard_threshold`,
  `soft_threshold`, `iht_step`, `amp_onsager`, `svt_threshold`, `tv_grad`
  (string-concatenation PTX generation, no nvcc)

#### Greedy Sparse Recovery
- [x] `greedy/omp.rs` -- Orthogonal Matching Pursuit, atom selection via max |a_j^T r|
  followed by least-squares on the active support
- [x] `greedy/stomp.rs` -- Stagewise OMP with `t * sigma_hat` threshold per stage
- [x] `greedy/romp.rs` -- Regularised OMP
- [x] `greedy/cosamp.rs` -- Needell-Tropp CoSaMP (identify -> merge -> prune)
- [x] `greedy/sp.rs` -- Subspace Pursuit

#### Iterative Hard / Soft Thresholding
- [x] `thresholding/iht.rs` -- Blumensath-Davies IHT `x <- H_K(x + mu * Phi^T (y - Phi x))`
- [x] `thresholding/niht.rs` -- Normalised IHT with adaptive `mu = ||g_S||^2 / ||Phi_S g_S||^2`
- [x] `thresholding/htp.rs` -- Hard Thresholding Pursuit with exact LS on identified support
- [x] `thresholding/aiht.rs` -- Accelerated IHT with Nesterov momentum

#### Approximate Message Passing
- [x] `amp/amp.rs` -- Donoho-Maleki-Montanari AMP with Onsager correction `b * z_{t-1} / M`
- [x] `amp/vamp.rs` -- Rangan-Schniter-Fletcher Vector AMP
- [x] `amp/eb_amp.rs` -- Empirical-Bayes AMP

#### Basis Pursuit Family
- [x] `basis_pursuit/basis_pursuit.rs` -- BP `min ||x||_1 s.t. Phi x = y` via ADMM,
  with BPDN noisy variant
- [x] `basis_pursuit/dantzig_selector.rs` -- Candes-Tao Dantzig Selector via LP

#### LASSO Family
- [x] `lasso/coord_descent.rs` -- Friedman cyclic coordinate descent with warm-start path
- [x] `lasso/lars.rs` -- Efron-Hastie LARS piecewise-linear regularisation path
- [x] `lasso/fista_lasso.rs` -- FISTA on the L1-LS problem
- [x] `lasso/group_lasso.rs` -- Yuan-Lin group LASSO with per-block soft-thresholding
- [x] `lasso/fused_lasso.rs` -- Fused LASSO `lambda_2 * sum |x_j - x_{j-1}|`
- [x] `lasso/elastic_net.rs` -- Elastic Net L1 + L2 hybrid penalty

#### Total Variation
- [x] `tv/tv_1d_chambolle.rs` -- 1D Chambolle 2004 primal-dual on the dual variable
- [x] `tv/tv_2d_chambolle.rs` -- 2D Chambolle (anisotropic + isotropic TV)
- [x] `tv/total_variation_denoise.rs` -- High-level denoising entry point

#### Matrix Completion
- [x] `matrix_completion/svt.rs` -- Cai-Candes-Shen Singular Value Thresholding
  `Y_{k+1} = Y_k + delta * P_Omega(M - X_k)` with `X_{k+1} = D_tau(Y)`
- [x] `matrix_completion/nuclear_norm.rs` -- `min ||X||_* s.t. P_Omega(X) = P_Omega(M)`
- [x] `matrix_completion/admm_completion.rs` -- ADMM-based completion

#### Robust PCA
- [x] `robust_pca/robust_pca_pcp.rs` -- Candes-Li-Ma-Wright Principal Component Pursuit
  `min ||L||_* + lambda * ||S||_1 s.t. L + S = M`
- [x] `robust_pca/godec.rs` -- Zhou-Tao GoDec via bilateral random projections

#### Sparse PCA & Bayesian Methods
- [x] `sparse_pca/sparse_pca_witten.rs` -- Witten-Tibshirani-Hastie 2009 penalised SVD
  with L1 constraint on loadings
- [x] `sbl/sparse_bayesian.rs` -- Tipping 2001 Relevance Vector Machine-style with
  per-coefficient hyperparameter gamma_i
- [x] `sbl/fast_marginal_likelihood.rs` -- Tipping-Faul 2003 fast marginal-likelihood update

#### Dictionary Learning
- [x] `dictionary/k_svd.rs` -- Aharon-Elad-Bruckstein K-SVD (sparse-code via OMP +
  per-atom SVD update)
- [x] `dictionary/mod_dl.rs` -- Engan Method of Optimal Directions
- [x] `dictionary/online_dl.rs` -- Mairal online dictionary learning

#### Measurement Matrices
- [x] `measurement/gaussian_matrix.rs` -- `Phi_ij ~ N(0, 1/m)`
- [x] `measurement/bernoulli_matrix.rs` -- `Phi_ij ~ +- 1 / sqrt(m)`
- [x] `measurement/partial_fourier.rs` -- Random DFT row selection
- [x] `measurement/rip_estimator.rs` -- RIP constant estimation via random K-subset SVD

#### Private Linear Algebra Helpers
- [x] `linalg/jacobi_svd.rs` -- Cyclic Jacobi SVD for moderate dimensions
- [x] `linalg/qr_householder.rs` -- Householder QR factorisation
- [x] `linalg/cholesky.rs` -- Cholesky decomposition for SPD systems
- [x] `linalg/lsqr.rs` -- Paige-Saunders LSQR iterative solver
- [x] `linalg/normal_equations.rs` -- Normal equations solver for small overdetermined systems

#### Diagnostics & Tests
- [x] `metrics/metrics.rs` -- sparsity, recovery_error, support_recovery_rate, MSE,
  NMSE, PSNR, SNR
- [x] `e2e_tests.rs` -- 27 cross-module integration tests (OMP K=3 recovery from
  m=20 n=50 Gaussian; CoSaMP support recovery >=80%; IHT convergence; AMP vs LASSO
  on iid Gaussian; BP exact when K<m/2; LASSO coord-descent ~ LARS; TV denoising
  improves MSE; SVT low-rank recovery; PCP L+S separation; soft-threshold and
  hard-threshold sanity; PTX x 6 SM)
- [x] `benches/cs_ops.rs` -- Criterion: 7 PTX kernels x all SM versions plus
  OMP / FISTA-LASSO / SVT / TV / Robust-PCA algorithm benches

### Future Enhancements

#### P0 -- Verification Gaps
- [ ] GPU hardware verification on Linux + NVIDIA driver 525+ for all 7 PTX kernels
  across SM 75 / 80 / 86 / 89 / 90 / 100
- [ ] Numerical-accuracy regression vs reference implementations (SPGL1, SPAMS, TFOCS)
  for OMP / FISTA-LASSO / SVT / PCP

#### P1 -- Performance Tuning
- [ ] Per-SM tuned tile / thread-block configurations for `correlate`, `iht_step`,
  `amp_onsager`, `svt_threshold` (currently fixed at portable defaults)
- [ ] Streamed batched LASSO path solves (warm-start across regularisation grid on device)
- [ ] FP16 / BF16 storage with FP32 accumulation for `correlate` and `iht_step`
  on memory-bound large-N problems

#### P2 -- Algorithmic Extensions
- [ ] Multi-GPU SVT / PCP for very large matrix completion (slab decomposition of P_Omega)
- [ ] Persistent kernel path for OMP inner least-squares on repeated small supports
- [ ] Block-OMP / simultaneous-OMP for multiple-measurement-vector (MMV) recovery

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading runtime FFI) | Yes |
| oxicuda-memory | Device / host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

No external linear-algebra dependencies; all required SVD / QR / Cholesky / LSQR
routines are implemented privately under `linalg/`.

## Quality Status

- Warnings: 0 (clippy clean, `#![forbid(unsafe_code)]`)
- Tests: 108 passing (unit + 27 e2e cross-module)
- `unwrap()` / `expect()` calls in production code: 0
- Refactoring policy: all files under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`; macOS returns
  `UnsupportedPlatform` at runtime, compiles cleanly

## Performance Targets

Reference workloads representative of compressed-sensing benchmarks (Donoho-Tanner
phase-transition regime). All targets are CPU-side dispatch + PTX generation; GPU
throughput targets are aspirational pending Linux + NVIDIA verification.

| Algorithm | Problem Size | Target |
|-----------|--------------|--------|
| OMP | m=200, n=1000, K=10 (Gaussian Phi) | recovery in < 50 ms on sm_80 |
| FISTA-LASSO | m=512, n=2048, 100 outer iterations | < 100 ms on sm_80 |
| SVT | 1024 x 1024, rank 50, 30% sampling | < 200 ms / iteration on sm_80 |
| 2D TV denoising | 1024 x 1024 image | < 30 ms / iteration on sm_80 |
| PCP (Robust PCA) | 512 x 512 with 10% outliers | < 500 ms / outer iter on sm_80 |
| Soft / hard threshold | N = 2^20 vector | bandwidth-bound (>= 90% peak HBM) |

## Notes

- Vol.58 prioritises pure-Rust portability over peak GPU throughput; the PTX layer
  exists primarily for elementwise / thresholding kernels and Onsager corrections,
  while the algorithmic outer loops (OMP, LARS, ADMM, FISTA) currently execute on the host.
- Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick)
  for deterministic, dependency-free reproducibility across all measurement-matrix
  constructions and Robust-PCA random projections.
- `linalg/` is intentionally private and minimal: it provides only the SVD / QR /
  Cholesky / LSQR routines that this crate needs internally. Public dense linear
  algebra lives in oxicuda-blas / oxicuda-solver.

## Architecture-Specific Deepening

Tile configurations and pipeline stages for the 7 PTX kernels by SM version. All
current builds use a single portable default; per-SM tuning is tracked under
Future Enhancements P1.

| SM Version | Default Tile (`correlate`) | Pipeline | Notes |
|------------|----------------------------|----------|-------|
| sm_75 (Turing) | 128 x 1 | 1 stage | baseline |
| sm_80 / sm_86 (Ampere) | 256 x 1 | 2 stages | `cp.async` ready |
| sm_89 (Ada) | 256 x 1 | 2 stages | -- |
| sm_90 (Hopper) | 512 x 1 | 3 stages | TMA candidate for `svt_threshold` |
| sm_100 (Blackwell) | 512 x 1 | 3 stages | -- |

### Deepening Opportunities
- [ ] Hopper: TMA (`cp.async.bulk`) loads for `correlate` on very tall Phi matrices
- [ ] Ampere: 3-stage `cp.async` pipeline for `iht_step` on M >= 4096
- [ ] Ada / Hopper: FP8 (e4m3) variant of `correlate` for memory-bound large-n problems
- [ ] All SMs: warp-shuffle reduction in `svt_threshold` for ranks <= 32

## Estimation vs Actual

| Metric | Estimated (estimation.md Vol.58) | Actual |
|--------|----------------------------------|--------|
| SLoC | 60K-110K (median ~85K) | 6,127 |
| Files | ~25-40 algorithm modules | 64 |
| Tests | algorithm-grade coverage | 108 |

The gap to the median estimate reflects the estimation targeting full
SPAMS-/TFOCS-grade production parity including streaming solvers, exhaustive path
algorithms, and large-scale benchmark harnesses. The current implementation
delivers a clean API surface with verified algorithmic correctness on CPU and
PTX generation for all 7 device kernels.
