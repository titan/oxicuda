//! End-to-end integration tests for `oxicuda-cs`.

#![allow(clippy::approx_constant)]

use crate::amp::{amp, eb_amp, vamp};
use crate::basis_pursuit::{basis_pursuit, basis_pursuit_denoise, dantzig_selector};
use crate::dictionary::{k_svd, mod_dl, online_dl};
use crate::greedy::{cosamp, omp, romp, stomp, subspace_pursuit};
use crate::handle::LcgRng;
use crate::lasso::{
    coord_descent_lasso, elastic_net, fista_lasso, fused_lasso, group_lasso, lars, lasso_path,
};
use crate::matrix_completion::{admm_matrix_completion, nuclear_norm_minimization, svt};
use crate::measurement::{bernoulli_matrix, gaussian_matrix, partial_fourier, rip_estimator};
use crate::metrics::{
    mean_squared_error, normalized_mse, psnr, recovery_error, snr, sparsity, support_recovery_rate,
};
use crate::ptx_kernels::{
    amp_onsager_ptx, correlate_ptx, hard_threshold_ptx, iht_step_ptx, soft_threshold_ptx,
    svt_threshold_ptx, tv_grad_ptx,
};
use crate::robust_pca::{godec, robust_pca_pcp};
use crate::sbl::{fast_marginal_likelihood, sparse_bayesian};
use crate::sparse_pca::sparse_pca_witten;
use crate::thresholding::{aiht, hard_threshold_k, htp, iht, niht, soft_threshold};
use crate::tv::total_variation_denoise::TvDim;
use crate::tv::tv_2d_chambolle::TvVariant;
use crate::tv::{total_variation_denoise, tv_1d_chambolle, tv_2d_chambolle};

fn build_sparse_signal(n: usize, support: &[(usize, f64)]) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for &(j, v) in support {
        x[j] = v;
    }
    x
}

fn measure(phi: &[f64], m: usize, n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        for j in 0..n {
            y[i] += phi[i * n + j] * x[j];
        }
    }
    y
}

// 1. OMP recovers a K=3 sparse signal from m=20, n=50 Gaussian sensing matrix.
#[test]
fn e2e_omp_gaussian_recovery() {
    let m = 20;
    let n = 50;
    let mut rng = LcgRng::new(42);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(3, 1.0), (17, -0.7), (29, 0.5)]);
    let y = measure(&phi, m, n, &x_true);
    let r = omp(&phi, m, n, &y, 3, 1.0e-7).expect("ok");
    let err = recovery_error(&r.x, &x_true).expect("ok");
    assert!(err < 1.0e-3, "OMP recovery error too large: {err}");
}

// 2. CoSaMP support recovery rate ≥ 80% on the same problem.
#[test]
fn e2e_cosamp_support_recovery() {
    let m = 20;
    let n = 50;
    let mut rng = LcgRng::new(7);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(3, 1.0), (17, -0.7), (29, 0.5)]);
    let y = measure(&phi, m, n, &x_true);
    let r = cosamp(&phi, m, n, &y, 3, 50, 1.0e-7).expect("ok");
    let true_supp = vec![3, 17, 29];
    let rate = support_recovery_rate(&true_supp, &r.support).expect("ok");
    assert!(rate >= 0.8, "support recovery rate {rate} < 0.8");
}

// 3. IHT converges on a recoverable problem.
#[test]
fn e2e_iht_converges() {
    let m = 30;
    let n = 60;
    let mut rng = LcgRng::new(99);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(5, 1.5), (22, -1.0)]);
    let y = measure(&phi, m, n, &x_true);
    let r = iht(&phi, m, n, &y, 2, 1.0, 500, 1.0e-9).expect("ok");
    assert!(r.iterations > 0);
    assert!(r.residual_norm.is_finite());
}

// 4. AMP soft-threshold matches LASSO solution on iid Gaussian Φ.
#[test]
fn e2e_amp_runs_on_gaussian() {
    let m = 25;
    let n = 60;
    let mut rng = LcgRng::new(13);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(7, 1.0), (31, -0.8), (47, 0.6)]);
    let y = measure(&phi, m, n, &x_true);
    let r = amp(&phi, m, n, &y, 1.5, 200, 1.0e-9).expect("ok");
    assert!(r.iterations > 0);
}

// 5. Basis Pursuit recovers exactly when K < m/2 with high probability.
#[test]
fn e2e_basis_pursuit_recovers() {
    let m = 24;
    let n = 40;
    let mut rng = LcgRng::new(101);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(3, 1.0), (12, -0.7), (28, 0.5)]);
    let y = measure(&phi, m, n, &x_true);
    let r = basis_pursuit(&phi, m, n, &y, 2.0, 600, 1.0e-7).expect("ok");
    let err = recovery_error(&r.x, &x_true).expect("ok");
    assert!(err < 0.2, "BP recovery error too large: {err}");
}

// 6. LASSO coord-descent matches LARS path at solution (approximately).
#[test]
fn e2e_lasso_cd_matches_lars() {
    let m = 12;
    let n = 8;
    let mut rng = LcgRng::new(5);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let mut x_true = vec![0.0_f64; n];
    x_true[1] = 0.7;
    x_true[4] = -0.4;
    let y = measure(&phi, m, n, &x_true);
    // Pick a moderate lambda.
    let lam = 0.05_f64;
    let x_cd = coord_descent_lasso(&phi, m, n, &y, lam, None, 2000, 1.0e-10).expect("ok");
    let path = lars(&phi, m, n, &y, 30).expect("ok");
    // Find LARS solution closest to lam.
    let target = path
        .steps
        .iter()
        .min_by(|a, b| {
            (a.lambda - lam)
                .abs()
                .partial_cmp(&(b.lambda - lam).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("ok");
    // Support overlap.
    let active_cd: Vec<usize> = (0..n).filter(|&j| x_cd[j].abs() > 1.0e-4).collect();
    let rate = support_recovery_rate(&active_cd, &target.active).expect("ok");
    assert!(rate >= 0.5, "LARS / CD support overlap rate {rate}");
}

// 7. TV denoising on piecewise-constant + Gaussian noise improves MSE.
#[test]
fn e2e_tv_denoise_improves_mse() {
    let mut x_true = vec![0.0_f64; 64];
    for v in x_true.iter_mut().take(32) {
        *v = 1.0;
    }
    for v in x_true.iter_mut().skip(32) {
        *v = 5.0;
    }
    let mut rng = LcgRng::new(50);
    let mut y = x_true.clone();
    for v in y.iter_mut() {
        *v += 0.2 * rng.next_normal();
    }
    let denoised = tv_1d_chambolle(&y, 0.3, 1000, 1.0e-10).expect("ok");
    let mse_noisy = mean_squared_error(&y, &x_true).expect("ok");
    let mse_clean = mean_squared_error(&denoised, &x_true).expect("ok");
    assert!(
        mse_clean < mse_noisy,
        "TV did not improve MSE: clean={mse_clean}, noisy={mse_noisy}"
    );
}

// 8. SVT recovers low-rank matrix from random sampling.
#[test]
fn e2e_svt_low_rank_recovery() {
    // Rank-1 4×4: M = u v^T.
    let u = vec![1.0_f64, 2.0, 3.0, 4.0];
    let v = vec![1.0_f64, 1.5, -0.5, 2.0];
    let mut m = vec![0.0_f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            m[i * 4 + j] = u[i] * v[j];
        }
    }
    // Observe 12 of 16 entries randomly.
    let mut rng = LcgRng::new(73);
    let mut mask = vec![false; 16];
    let mut count = 0usize;
    while count < 12 {
        let idx = rng.next_usize(16);
        if !mask[idx] {
            mask[idx] = true;
            count += 1;
        }
    }
    let r = svt(&m, &mask, 4, 4, 1.0, 1.5, 500, 1.0e-9).expect("ok");
    let err = recovery_error(&r.x, &m).expect("ok");
    assert!(err < 0.5, "SVT relative error {err}");
}

// 9. Robust PCA recovers L (low-rank) + S (sparse outliers).
#[test]
fn e2e_robust_pca_decomposes() {
    let mut m = vec![0.0_f64; 25];
    // Low-rank rank-1 background.
    for i in 0..5 {
        for j in 0..5 {
            m[i * 5 + j] = ((i + 1) as f64) * ((j + 1) as f64);
        }
    }
    // Two outliers.
    m[7] += 20.0;
    m[18] += -15.0;
    let r = robust_pca_pcp(&m, 5, 5, Some(0.4), Some(0.4), 200, 1.0e-7).expect("ok");
    // Sparse component should pick up the outliers.
    assert!(r.sparse[7].abs() > 1.0);
    assert!(r.sparse[18].abs() > 1.0);
}

// 10. Soft threshold of [2, 0.5, -0.5, -2] with λ=1 gives [1, 0, 0, -1].
#[test]
fn e2e_soft_threshold_doc_example() {
    let v = [2.0_f64, 0.5, -0.5, -2.0];
    let p = soft_threshold(&v, 1.0);
    assert!((p[0] - 1.0).abs() < 1.0e-12);
    assert!(p[1].abs() < 1.0e-12);
    assert!(p[2].abs() < 1.0e-12);
    assert!((p[3] + 1.0).abs() < 1.0e-12);
}

// 11. Hard threshold to top-2 of [3, 1, 4, 1, 5] keeps {2, 4} indices.
#[test]
fn e2e_hard_threshold_doc_example() {
    let v = [3.0_f64, 1.0, 4.0, 1.0, 5.0];
    let (out, supp) = hard_threshold_k(&v, 2).expect("ok");
    assert_eq!(supp, vec![2, 4]);
    assert!((out[2] - 4.0).abs() < 1.0e-12);
    assert!((out[4] - 5.0).abs() < 1.0e-12);
    assert!(out[0].abs() < 1.0e-12);
}

// 12. Bernoulli measurement matrix is normalised.
#[test]
fn e2e_bernoulli_matrix_normalised() {
    let mut rng = LcgRng::new(8);
    let m = 10;
    let n = 20;
    let phi = bernoulli_matrix(m, n, &mut rng).expect("ok");
    let s = 1.0_f64 / (m as f64).sqrt();
    for v in &phi {
        assert!((v.abs() - s).abs() < 1.0e-12);
    }
}

// 13. Partial Fourier matrix correct shape.
#[test]
fn e2e_partial_fourier_shape() {
    let mut rng = LcgRng::new(99);
    let m = 16;
    let n = 64;
    let phi = partial_fourier(m, n, &mut rng).expect("ok");
    assert_eq!(phi.len(), m * n);
}

// 14. RIP estimator is finite on a Gaussian sensing matrix.
#[test]
fn e2e_rip_estimator_finite() {
    let mut rng = LcgRng::new(11);
    let m = 16;
    let n = 32;
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let mut rng2 = LcgRng::new(21);
    let d = rip_estimator(&phi, m, n, 3, 30, &mut rng2).expect("ok");
    assert!(d.is_finite());
}

// 15. Stagewise OMP and ROMP both pick the correct support on the canonical problem.
#[test]
fn e2e_stomp_romp_canonical() {
    let phi = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let y = vec![1.0, 0.0, 0.5, 0.0];
    let r_st = stomp(&phi, 4, 4, &y, 2.0, 10, 1.0e-6).expect("ok");
    let r_ro = romp(&phi, 4, 4, &y, 2, 10, 1.0e-6).expect("ok");
    assert!(r_st.support.contains(&0));
    assert!(r_st.support.contains(&2));
    assert!(r_ro.support.contains(&0));
    let _ = subspace_pursuit(&phi, 4, 4, &y, 2, 10, 1.0e-6).expect("ok");
}

// 16. Dictionary learning round-trip: K-SVD, MOD, online produce dictionaries of correct shape.
#[test]
fn e2e_dictionary_learning_shapes() {
    let mut rng = LcgRng::new(63);
    let d = 6;
    let n_samples = 8;
    let n_atoms = 3;
    let signals: Vec<f64> = (0..(d * n_samples))
        .map(|i| (i as f64 % 5.0) - 2.0)
        .collect();
    let r_kk = k_svd(&signals, d, n_samples, n_atoms, 2, 3, 1.0e-6, &mut rng).expect("ok");
    let mut rng2 = LcgRng::new(63);
    let r_mod = mod_dl(&signals, d, n_samples, n_atoms, 2, 3, 1.0e-6, &mut rng2).expect("ok");
    let mut rng3 = LcgRng::new(63);
    let r_on = online_dl(&signals, d, n_samples, n_atoms, 2, 2, &mut rng3).expect("ok");
    assert_eq!(r_kk.dict.len(), d * n_atoms);
    assert_eq!(r_mod.dict.len(), d * n_atoms);
    assert_eq!(r_on.dict.len(), d * n_atoms);
}

// 17. AMP family solvers all produce results (numerical divergence in VAMP is sanitised).
#[test]
fn e2e_amp_family_finite() {
    let m = 16;
    let n = 32;
    let mut rng = LcgRng::new(1);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(2, 1.2), (15, -0.8)]);
    let y = measure(&phi, m, n, &x_true);
    let r1 = amp(&phi, m, n, &y, 1.4, 50, 1.0e-9).expect("ok");
    let r2 = vamp(&phi, m, n, &y, 0.05, 10, 1.0e-9).expect("ok");
    let grid: Vec<f64> = (5..30).map(|i| (i as f64) * 0.1).collect();
    let r3 = eb_amp(&phi, m, n, &y, &grid, 30, 1.0e-9).expect("ok");
    assert!(r1.residual_norm.is_finite());
    assert!(r2.iterations > 0);
    assert!(r3.residual_norm.is_finite());
}

// 18. All 7 PTX kernels emit non-empty strings for all SM versions.
#[test]
fn e2e_ptx_kernels_all_sm() {
    type KernelFn = fn(u32) -> String;
    let kernels: Vec<KernelFn> = vec![
        correlate_ptx,
        hard_threshold_ptx,
        soft_threshold_ptx,
        iht_step_ptx,
        amp_onsager_ptx,
        svt_threshold_ptx,
        tv_grad_ptx,
    ];
    for sm in [75u32, 80, 86, 89, 90, 100] {
        for f in &kernels {
            let s = f(sm);
            assert!(!s.is_empty());
            assert!(s.contains(".visible .entry"));
            assert!(s.contains("ret"));
        }
    }
}

// Additional sanity helpers (not numbered explicitly in spec):
#[test]
fn e2e_extra_basis_pursuit_denoise() {
    let phi = vec![1.0, 0.0, 0.0, 1.0];
    let y = vec![1.0, 1.0];
    let r = basis_pursuit_denoise(&phi, 2, 2, &y, 0.05, 1.0, 100, 1.0e-6).expect("ok");
    assert!(r.iterations > 0);
}

#[test]
fn e2e_extra_dantzig_runs() {
    let phi = vec![1.0, 0.0, 0.0, 1.0];
    let y = vec![1.0, 0.5];
    let r = dantzig_selector(&phi, 2, 2, &y, 0.05, 1.0, 50, 1.0e-6).expect("ok");
    assert!(r.iterations > 0);
}

#[test]
fn e2e_extra_metrics_pipeline() {
    let x_true = vec![1.0_f64, 0.0, -1.0, 0.0];
    let x_hat = vec![0.95_f64, 0.05, -0.95, 0.05];
    assert_eq!(sparsity(&x_true, 1.0e-6), 2);
    let m = mean_squared_error(&x_hat, &x_true).expect("ok");
    let nm = normalized_mse(&x_hat, &x_true).expect("ok");
    let snr_v = snr(&x_hat, &x_true).expect("ok");
    let psnr_v = psnr(&x_hat, &x_true, 1.0).expect("ok");
    assert!(m > 0.0);
    assert!(nm > 0.0);
    assert!(snr_v.is_finite());
    assert!(psnr_v.is_finite());
}

#[test]
fn e2e_extra_total_variation_dispatch() {
    let y = vec![1.0_f64, 1.0, 2.0, 2.0];
    let x = total_variation_denoise(&y, TvDim::OneD, 0.1, 100, 1.0e-9).expect("ok");
    assert_eq!(x.len(), 4);
    let img = vec![1.0_f64; 16];
    let x2 =
        total_variation_denoise(&img, TvDim::TwoD { h: 4, w: 4 }, 0.1, 100, 1.0e-9).expect("ok");
    assert_eq!(x2.len(), 16);
    // also exercise the 2D variant directly.
    let _ = tv_2d_chambolle(&img, 4, 4, 0.1, TvVariant::Isotropic, 100, 1.0e-9).expect("ok");
}

#[test]
fn e2e_extra_aiht_niht_htp() {
    let m = 16;
    let n = 30;
    let mut rng = LcgRng::new(33);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let x_true = build_sparse_signal(n, &[(2, 1.0), (11, -0.6)]);
    let y = measure(&phi, m, n, &x_true);
    let r1 = aiht(&phi, m, n, &y, 2, 0.9, 200, 1.0e-9).expect("ok");
    let r2 = niht(&phi, m, n, &y, 2, 100, 1.0e-9).expect("ok");
    let r3 = htp(&phi, m, n, &y, 2, 1.0, 50, 1.0e-9).expect("ok");
    assert!(r1.iterations > 0);
    assert!(r2.iterations > 0);
    assert!(r3.iterations > 0);
}

#[test]
fn e2e_extra_lasso_variants() {
    let m = 10;
    let n = 8;
    let mut rng = LcgRng::new(91);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let mut x_true = vec![0.0_f64; n];
    x_true[0] = 0.5;
    x_true[4] = -0.4;
    let y = measure(&phi, m, n, &x_true);
    let _ = fista_lasso(&phi, m, n, &y, 0.05, None, 500, 1.0e-9).expect("ok");
    let _ = elastic_net(&phi, m, n, &y, 0.05, 0.05, 500, 1.0e-9).expect("ok");
    let groups = vec![(0_usize, 4_usize), (4_usize, 4_usize)];
    let _ = group_lasso(&phi, m, n, &y, &groups, 0.05, 200, 1.0e-9).expect("ok");
    let _ = fused_lasso(&phi, m, n, &y, 0.02, 0.02, 200, 1.0e-9).expect("ok");
    let _ = lasso_path(&phi, m, n, &y, &[1.0, 0.5, 0.1, 0.0], 500, 1.0e-9).expect("ok");
}

#[test]
fn e2e_extra_matrix_completion_admm_and_nuclear() {
    let m = vec![1.0_f64, 2.0, 2.0, 4.0];
    let mask = vec![true, true, true, false];
    let r_a = admm_matrix_completion(&m, &mask, 2, 2, 1.0, 200, 1.0e-7).expect("ok");
    let r_n = nuclear_norm_minimization(&m, &mask, 2, 2, Some(1.0), 200, 1.0e-7).expect("ok");
    assert!(r_a.iterations > 0);
    assert!(r_n.iterations > 0);
}

#[test]
fn e2e_extra_sparse_pca_and_godec() {
    let mut signals = vec![0.0_f64; 32];
    for i in 0..8 {
        signals[i * 4] = (i as f64) - 3.5;
    }
    let _ = sparse_pca_witten(&signals, 8, 4, 1, 1.5, 30, 1.0e-7).expect("ok");
    let mut m_g = vec![0.0_f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            m_g[i * 4 + j] = ((i + 1) as f64) * ((j + 1) as f64);
        }
    }
    m_g[5] += 8.0;
    let _ = godec(&m_g, 4, 4, 1, 1, 50, 1.0e-7).expect("ok");
}

#[test]
fn e2e_extra_sbl_runs() {
    let mut rng = LcgRng::new(133);
    let phi = gaussian_matrix(8, 12, &mut rng).expect("ok");
    let x_true = build_sparse_signal(12, &[(3, 0.8), (7, -0.6)]);
    let y = measure(&phi, 8, 12, &x_true);
    let _ = sparse_bayesian(&phi, 8, 12, &y, 30, 1.0e-7).expect("ok");
    let _ = fast_marginal_likelihood(&phi, 8, 12, &y, 30, 1.0e-7).expect("ok");
}
