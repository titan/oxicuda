# oxicuda-cs

Compressed sensing, sparse recovery, and low-rank matrix completion -- a pure Rust GPU library.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-cs` provides a comprehensive toolbox for ill-posed linear inverse
problems of the form `y = Phi x + noise`, where the unknown `x` is sparse,
group-sparse, low-rank, or otherwise structured. It covers the four main
families of recovery algorithms -- greedy pursuit, iterative thresholding,
approximate message passing, and convex relaxation -- together with dictionary
learning, robust PCA, and the measurement-matrix machinery needed to drive
them.

GPU kernels are generated and launched entirely from Rust via the OxiCUDA
driver stack. There is no C/CUDA toolchain at build time and no external BLAS
or LAPACK dependency: the internal `linalg` module supplies Jacobi SVD,
Householder QR, Cholesky, and LSQR primitives in pure Rust.

Algorithm coverage spans Orthogonal Matching Pursuit and its variants
(StOMP, ROMP, CoSaMP, Subspace Pursuit), Iterative Hard Thresholding and its
accelerated/normalised forms, Approximate Message Passing (AMP, VAMP, EB-AMP),
the LASSO family (coordinate descent, LARS, FISTA-LASSO, group/fused, elastic
net), basis pursuit / BPDN / Dantzig selector via ADMM, Singular Value
Thresholding and nuclear-norm minimisation for matrix completion, robust PCA
(PCP, GoDec), sparse PCA, Sparse Bayesian Learning, and K-SVD / MOD / online
dictionary learning.

## Modules

| Module | Description |
|--------|-------------|
| `greedy` | OMP, StOMP, ROMP, CoSaMP, Subspace Pursuit |
| `thresholding` | IHT, NIHT, HTP, Accelerated IHT, soft/hard threshold ops |
| `amp` | AMP, VAMP, Empirical-Bayes AMP |
| `basis_pursuit` | Basis Pursuit (ADMM), BPDN, Dantzig Selector |
| `lasso` | Coordinate descent, LARS, FISTA-LASSO, group/fused LASSO, Elastic Net |
| `tv` | 1D/2D Chambolle Total Variation denoising |
| `matrix_completion` | SVT, nuclear-norm minimisation, ADMM matrix completion |
| `robust_pca` | Principal Component Pursuit (PCP), GoDec |
| `sparse_pca` | Witten-Tibshirani-Hastie penalised matrix decomposition |
| `sbl` | Sparse Bayesian Learning, Fast Marginal Likelihood |
| `dictionary` | K-SVD, MOD, online dictionary learning |
| `measurement` | Gaussian, Bernoulli, partial Fourier matrices, RIP estimator |
| `linalg` | Jacobi SVD, Householder QR, Cholesky, LSQR, normal equations |
| `metrics` | Sparsity, recovery error, support recovery rate, MSE, PSNR, SNR |
| `handle` | `CsHandle`, `SmVersion`, `LcgRng` (MMIX LCG) |
| `error` | `CsError` / `CsResult` |
| `ptx_kernels` | GPU PTX kernel templates per SM target |

## Quick Start

```rust,no_run
use oxicuda_cs::greedy::omp::omp;
use oxicuda_cs::{CsHandle, SmVersion};

fn main() -> oxicuda_cs::CsResult<()> {
    // Compute handle bound to an SM target and seeded RNG.
    let _handle = CsHandle::new(SmVersion::SM_90, 0xC0FFEE);

    // Sensing matrix (m x n) in row-major order, measurements y, sparsity k.
    let m = 32usize;
    let n = 128usize;
    let k = 5usize;
    let phi: Vec<f64> = unimplemented!();
    let y: Vec<f64> = unimplemented!();

    // Orthogonal Matching Pursuit: returns coefficients, support, residual norm.
    let result = omp(&phi, m, n, &y, k, 1e-8)?;
    let _x_hat = result.x;       // length n recovered signal
    let _support = result.support;
    Ok(())
}
```

## Status

**Alpha** -- 10,537 SLoC, 291 passing tests. API may evolve before v1.0.

## License

Apache-2.0
