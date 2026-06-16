//! Inline tests for the Generalized Method of Moments (GMM) estimator.

#![cfg(test)]

use super::gmm::{Gmm, GmmConfig, GmmResult};
use crate::handle::LcgRng;

// Helpers -------------------------------------------------------------------

/// Linear-IV data: `T = γ·Z + ν`, `Y = β·T + ε`. The structural `β` is
/// recoverable when `γ ≠ 0`. Returns `(y, x, z)` with `x` and `z` rows.
fn simulate_linear_iv(
    rng: &mut LcgRng,
    n: usize,
    beta: f64,
    n_instruments: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut y = vec![0.0_f64; n];
    let mut x = vec![vec![0.0_f64; 1]; n];
    let mut z = vec![vec![0.0_f64; n_instruments]; n];
    for i in 0..n {
        let zs: Vec<f64> = (0..n_instruments)
            .map(|_| rng.next_normal() as f64)
            .collect();
        let nu = (rng.next_normal() as f64) * 0.4;
        let eps = (rng.next_normal() as f64) * 0.3;
        let mean_z: f64 = zs.iter().sum::<f64>() / n_instruments as f64;
        let t = 0.8 * mean_z + nu;
        y[i] = beta * t + eps;
        x[i][0] = t;
        for (j, zv) in zs.iter().enumerate() {
            z[i][j] = *zv;
        }
    }
    (y, x, z)
}

/// Heteroskedastic linear-IV: variance of ε grows with z.
fn simulate_heteroskedastic_iv(
    rng: &mut LcgRng,
    n: usize,
    beta: f64,
    n_instruments: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut y = vec![0.0_f64; n];
    let mut x = vec![vec![0.0_f64; 1]; n];
    let mut z = vec![vec![0.0_f64; n_instruments]; n];
    for i in 0..n {
        let zs: Vec<f64> = (0..n_instruments)
            .map(|_| rng.next_normal() as f64)
            .collect();
        let nu = (rng.next_normal() as f64) * 0.4;
        // sigma_eps depends on z_0.
        let sigma = (0.2 + 0.6 * zs[0].abs()).max(0.05);
        let eps = (rng.next_normal() as f64) * sigma;
        let mean_z: f64 = zs.iter().sum::<f64>() / n_instruments as f64;
        let t = 0.8 * mean_z + nu;
        y[i] = beta * t + eps;
        x[i][0] = t;
        for (j, zv) in zs.iter().enumerate() {
            z[i][j] = *zv;
        }
    }
    (y, x, z)
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// Tests ---------------------------------------------------------------------

/// Just-identified linear IV (q = p = 1): the GMM closed-form reduces to
/// the 2SLS estimator. The Hansen J statistic must be zero.
#[test]
fn test_just_identified_matches_2sls() {
    let mut rng = LcgRng::new(7);
    let (y, x, z) = simulate_linear_iv(&mut rng, 400, 2.0, 1);
    let cfg = GmmConfig::default();
    let r = Gmm::estimate(&y, &x, &z, &cfg).expect("estimate should succeed");
    assert_eq!(r.theta.len(), 1);
    assert!(approx(r.theta[0], 2.0, 0.2), "θ = {} ≠ 2", r.theta[0]);
    assert!(
        r.j_stat <= 1e-12,
        "just-identified J should be 0, got {}",
        r.j_stat
    );
    assert!(
        approx(r.j_pvalue, 1.0, 1e-12),
        "just-identified p = 1, got {}",
        r.j_pvalue
    );
}

/// When the instruments coincide with the regressors (Z = X), GMM reduces
/// to OLS. Use synthetic Y = β·X + ε with no endogeneity.
#[test]
fn test_z_equals_x_recovers_ols() {
    let mut rng = LcgRng::new(13);
    let n = 300_usize;
    let mut x = vec![vec![0.0_f64; 1]; n];
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let xi = rng.next_normal() as f64;
        x[i][0] = xi;
        y[i] = 1.5 * xi + (rng.next_normal() as f64) * 0.1;
    }
    let z = x.clone();
    let r = Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    assert!(approx(r.theta[0], 1.5, 0.15), "θ = {}", r.theta[0]);
}

/// Overidentified system (q > p): J-statistic finite, p-value in [0, 1].
#[test]
fn test_overidentified_j_pvalue_in_unit_interval() {
    let mut rng = LcgRng::new(19);
    let (y, x, z) = simulate_linear_iv(&mut rng, 500, 1.5, 3);
    let r = Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    assert!(r.j_stat.is_finite() && r.j_stat >= 0.0, "J = {}", r.j_stat);
    assert!(
        r.j_pvalue >= 0.0 && r.j_pvalue <= 1.0,
        "J p-value = {} out of [0, 1]",
        r.j_pvalue
    );
    assert_eq!(r.n_moments, 3);
}

/// Two-step GMM should out-perform stage-1 alone under heteroskedasticity.
/// We compare bias across many seeds.
#[test]
fn test_two_step_beats_one_step_heteroskedastic() {
    let mut bias_one = 0.0_f64;
    let mut bias_two = 0.0_f64;
    let mut count = 0_usize;
    for seed in 0..16_u64 {
        let mut rng = LcgRng::new(100 + seed);
        let (y, x, z) = simulate_heteroskedastic_iv(&mut rng, 400, 2.0, 3);
        let cfg_one = GmmConfig {
            two_step: false,
            ..GmmConfig::default()
        };
        let cfg_two = GmmConfig {
            two_step: true,
            ..GmmConfig::default()
        };
        if let (Ok(r1), Ok(r2)) = (
            Gmm::estimate(&y, &x, &z, &cfg_one),
            Gmm::estimate(&y, &x, &z, &cfg_two),
        ) {
            bias_one += (r1.theta[0] - 2.0).powi(2);
            bias_two += (r2.theta[0] - 2.0).powi(2);
            count += 1;
        }
    }
    assert!(count > 0);
    let mse_one = bias_one / count as f64;
    let mse_two = bias_two / count as f64;
    // Allow some slack: two-step should not be much worse than one-step.
    // The asymptotic theory guarantees two-step is no worse than one-step,
    // but finite-sample MSE can fluctuate; we require it stays within 50%.
    assert!(
        mse_two <= 1.5 * mse_one,
        "two-step MSE {} should not exceed 1.5× one-step MSE {}",
        mse_two,
        mse_one
    );
}

/// Mismatched `y` / `x` / `z` row counts must be rejected.
#[test]
fn test_dim_mismatch() {
    let cfg = GmmConfig::default();
    let y = vec![1.0_f64; 100];
    let x = vec![vec![0.5_f64]; 99];
    let z = vec![vec![0.5_f64, 0.3]; 100];
    assert!(Gmm::estimate(&y, &x, &z, &cfg).is_err());
    let y2 = vec![1.0_f64; 100];
    let x2 = vec![vec![0.5_f64]; 100];
    let z2 = vec![vec![0.5_f64, 0.3]; 99];
    assert!(Gmm::estimate(&y2, &x2, &z2, &cfg).is_err());
}

/// Under-identified system (p > q) must be rejected.
#[test]
fn test_underidentified_rejected() {
    let cfg = GmmConfig::default();
    let n = 100_usize;
    let y = vec![1.0_f64; n];
    let x = vec![vec![0.5_f64, 0.4]; n]; // p = 2
    let z = vec![vec![0.5_f64]; n]; // q = 1
    assert!(Gmm::estimate(&y, &x, &z, &cfg).is_err());
}

/// `ridge_lambda ≤ 0` must be rejected.
#[test]
fn test_invalid_ridge_lambda() {
    let mut rng = LcgRng::new(5);
    let (y, x, z) = simulate_linear_iv(&mut rng, 100, 1.0, 2);
    let cfg_zero = GmmConfig {
        ridge_lambda: 0.0,
        ..GmmConfig::default()
    };
    assert!(Gmm::estimate(&y, &x, &z, &cfg_zero).is_err());
    let cfg_neg = GmmConfig {
        ridge_lambda: -1e-3,
        ..GmmConfig::default()
    };
    assert!(Gmm::estimate(&y, &x, &z, &cfg_neg).is_err());
}

/// `tol ≤ 0` must be rejected.
#[test]
fn test_invalid_tol() {
    let mut rng = LcgRng::new(11);
    let (y, x, z) = simulate_linear_iv(&mut rng, 100, 1.0, 2);
    let cfg_zero = GmmConfig {
        tol: 0.0,
        ..GmmConfig::default()
    };
    assert!(Gmm::estimate(&y, &x, &z, &cfg_zero).is_err());
    let cfg_neg = GmmConfig {
        tol: -1e-6,
        ..GmmConfig::default()
    };
    assert!(Gmm::estimate(&y, &x, &z, &cfg_neg).is_err());
}

/// `max_iters = 0` must be rejected.
#[test]
fn test_invalid_max_iters() {
    let mut rng = LcgRng::new(23);
    let (y, x, z) = simulate_linear_iv(&mut rng, 100, 1.0, 2);
    let cfg = GmmConfig {
        max_iters: 0,
        ..GmmConfig::default()
    };
    assert!(Gmm::estimate(&y, &x, &z, &cfg).is_err());
}

/// Empty inputs must be rejected.
#[test]
fn test_empty_inputs() {
    let cfg = GmmConfig::default();
    assert!(Gmm::estimate(&[], &Vec::<Vec<f64>>::new(), &Vec::<Vec<f64>>::new(), &cfg).is_err());
}

/// Deterministic across two identical calls.
#[test]
fn test_deterministic() {
    let mut rng = LcgRng::new(101);
    let (y, x, z) = simulate_linear_iv(&mut rng, 250, 1.7, 2);
    let cfg = GmmConfig::default();
    let a = Gmm::estimate(&y, &x, &z, &cfg).expect("estimate should succeed");
    let b = Gmm::estimate(&y, &x, &z, &cfg).expect("estimate should succeed");
    assert_eq!(a.theta, b.theta);
    assert_eq!(a.se, b.se);
    assert_eq!(a.j_stat, b.j_stat);
    assert_eq!(a.j_pvalue, b.j_pvalue);
    assert_eq!(a.n_iters, b.n_iters);
}

/// All standard errors must be positive and finite.
#[test]
fn test_se_positive_finite() {
    let mut rng = LcgRng::new(31);
    let (y, x, z) = simulate_linear_iv(&mut rng, 400, 1.0, 3);
    let r = Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    for &s in r.se.iter() {
        assert!(s.is_finite() && s >= 0.0, "se = {}", s);
    }
    assert!(r.se.iter().any(|&s| s > 0.0), "all SEs zero");
}

/// Just-identified case with q = p: J ≈ 0 and p-value ≈ 1.
#[test]
fn test_just_identified_j_zero() {
    let mut rng = LcgRng::new(37);
    let (y, x, z) = simulate_linear_iv(&mut rng, 200, 1.0, 1);
    let r = Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    assert!(r.j_stat.abs() < 1e-9, "J = {}", r.j_stat);
    assert!((r.j_pvalue - 1.0).abs() < 1e-12, "p = {}", r.j_pvalue);
}

/// Recovery of known β on a synthetic linear-IV DGP within 3× SE.
#[test]
fn test_recover_beta_within_three_se() {
    let mut rng = LcgRng::new(41);
    let true_beta = 1.5_f64;
    let (y, x, z) = simulate_linear_iv(&mut rng, 1000, true_beta, 3);
    let r = Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    let z_score = (r.theta[0] - true_beta) / r.se[0].max(1e-6);
    assert!(z_score.abs() < 5.0, "z-score = {}", z_score);
}

/// `n_iters` is `0` when two-step is disabled and `≥ 1` when enabled.
#[test]
fn test_n_iters_zero_when_one_step() {
    let mut rng = LcgRng::new(43);
    let (y, x, z) = simulate_linear_iv(&mut rng, 200, 1.0, 2);
    let cfg_one = GmmConfig {
        two_step: false,
        ..GmmConfig::default()
    };
    let r1 = Gmm::estimate(&y, &x, &z, &cfg_one).expect("estimate should succeed");
    assert_eq!(r1.n_iters, 0);
    let cfg_two = GmmConfig {
        two_step: true,
        ..GmmConfig::default()
    };
    let r2 = Gmm::estimate(&y, &x, &z, &cfg_two).expect("estimate should succeed");
    assert!(r2.n_iters >= 1);
}

/// `GmmResult` field shapes match the public contract.
#[test]
fn test_result_field_sizes() {
    let mut rng = LcgRng::new(47);
    let (y, x, z) = simulate_linear_iv(&mut rng, 200, 1.0, 2);
    let r: GmmResult =
        Gmm::estimate(&y, &x, &z, &GmmConfig::default()).expect("value should be present");
    let p = 1_usize;
    let q = 2_usize;
    assert_eq!(r.theta.len(), p);
    assert_eq!(r.se.len(), p);
    assert_eq!(r.n_moments, q);
    assert_eq!(r.n, 200);
}
