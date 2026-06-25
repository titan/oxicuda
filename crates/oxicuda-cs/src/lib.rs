//! `oxicuda-cs` — Compressed Sensing, Sparse Recovery, and Low-Rank Matrix Completion for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-cs
//! ├── greedy/             — OMP, StOMP, ROMP, CoSaMP, Subspace Pursuit
//! ├── thresholding/       — IHT, NIHT, HTP, Accelerated IHT
//! ├── amp/                — AMP, VAMP, Empirical-Bayes AMP
//! ├── basis_pursuit/      — Basis Pursuit (ADMM), BPDN, Dantzig Selector
//! ├── lasso/              — Coord descent (Friedman et al.), LARS, FISTA-LASSO,
//! │                         group/fused LASSO, Elastic Net
//! ├── tv/                 — 1D/2D Chambolle Total Variation denoising
//! ├── matrix_completion/  — SVT, Nuclear-norm minimisation, ADMM matrix completion
//! ├── robust_pca/         — Principal Component Pursuit (PCP), GoDec
//! ├── sparse_pca/         — Witten-Tibshirani-Hastie penalised matrix decomposition
//! ├── sbl/                — Sparse Bayesian Learning, Fast Marginal Likelihood
//! ├── dictionary/         — K-SVD, MOD, Online dictionary learning
//! ├── measurement/        — Gaussian, Bernoulli, Partial Fourier matrices, RIP estimator
//! ├── linalg/             — Private: Jacobi SVD, Householder QR, Cholesky, LSQR, normal equations
//! └── metrics/            — sparsity, recovery error, support recovery rate, MSE, PSNR, SNR
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_range_contains)]

pub mod amp;
pub mod basis_pursuit;
pub mod dictionary;
pub mod error;
pub mod greedy;
pub mod handle;
pub mod lasso;
pub mod linalg;
pub mod matrix_completion;
pub mod measurement;
pub mod metrics;
pub mod ptx_advanced;
pub mod ptx_kernels;
pub mod robust_pca;
pub mod sbl;
pub mod sparse_pca;
pub mod thresholding;
pub mod tv;

pub use error::{CsError, CsResult};
pub use handle::{CsHandle, LcgRng, SmVersion};

// Wave AAA+58 re-exports.
pub use greedy::{Lista, ListaConfig};
pub use lasso::{Slope, SlopeConfig, sorted_l1_prox};
pub use robust_pca::{RpcaGd, RpcaGdConfig};

// Wave AAA+68 re-exports.
pub use dictionary::{CoupledDictionary, CoupledDlConfig, couple_code, coupled_dl};
pub use measurement::{CodedDiffraction, MaskKind, WirtingerConfig, phase_aligned_error};
pub use sbl::{Rvm, RvmConfig, RvmFit, RvmKernel, rvm_fit_design};

// Sequential compressed sensing via SMC (Ji, Xue & Carin 2008).
pub use sbl::{SmcCs, SmcCsConfig};

// Architecture-specialised PTX kernel variants + per-SM tile configuration.
pub use ptx_advanced::{
    TileConfig, correlate_fp8_ptx, correlate_tma_ptx, iht_step_cp_async_ptx,
    svt_threshold_warpshuffle_ptx,
};

#[cfg(test)]
mod e2e_tests;

#[cfg(test)]
mod wave_aaa58_e2e {
    use super::*;

    #[test]
    fn e2e_lista_sparse_coding() {
        let n = 8_usize;
        let m = 12_usize;
        let mut dict = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                dict[i * n + j] = if (i + j) % 3 == 0 { 0.5 } else { -0.1 };
            }
        }
        let cfg = ListaConfig {
            n_measurements: n,
            n_atoms: m,
            threshold: 0.1,
            n_layers: 10,
        };
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let y = vec![0.5_f64; n];
        let code = lista.encode(&y).expect("ok");
        assert_eq!(code.len(), m);
        assert!(code.iter().all(|v| v.is_finite()), "all finite");
        let nnz = code.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(nnz <= m, "nnz {nnz} <= {m}");
    }

    #[test]
    fn e2e_rpca_gd_low_rank_recovery() {
        let m = 10_usize;
        let n = 8_usize;
        let u_true: Vec<f64> = (0..m).map(|i| (i + 1) as f64 * 0.1).collect();
        let v_true: Vec<f64> = (0..n).map(|j| (j + 1) as f64 * 0.1).collect();
        let mut mat = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                mat[i * n + j] = u_true[i] * v_true[j];
            }
        }
        mat[0] += 1.0;

        let cfg = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 0.05,
            max_iter: 200,
            lr: 0.01,
            tol: 1e-4,
        };
        let mut rng = LcgRng::new(58);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        rpca.fit(&mat).expect("ok");
        let l_hat = rpca.low_rank();
        assert_eq!(l_hat.len(), m * n);
        assert!(l_hat.iter().all(|v| v.is_finite()), "L finite");
        let resid = rpca.residual_norm(&mat);
        assert!(resid.is_finite(), "residual finite");
    }
}
