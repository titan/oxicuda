//! Inline tests for the Anderson-Rubin weak-IV-robust confidence sets.

#![cfg(test)]

use super::anderson_rubin::{
    AndersonRubin, AndersonRubinConfig, f_cdf_pub as f_cdf, f_inverse_cdf_pub as f_inverse_cdf,
};
use crate::handle::LcgRng;

/// Generate (y, d, z) for a strong instrument: z ~ N(0, 1), d = π z + ν,
/// y = β d + ε with mild noise. The structural β is recoverable.
fn simulate_strong_iv(
    rng: &mut LcgRng,
    n: usize,
    beta: f64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut y = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    let mut z = vec![vec![0.0_f64; 1]; n];
    for i in 0..n {
        let zi = rng.next_normal() as f64;
        let nu = (rng.next_normal() as f64) * 0.3;
        let eps = (rng.next_normal() as f64) * 0.3;
        z[i][0] = zi;
        d[i] = 1.5 * zi + nu;
        y[i] = beta * d[i] + eps;
    }
    (y, d, z)
}

/// Strong instrument: the AR confidence set should cover the true β = 2.
#[test]
fn test_strong_iv_recovers_beta_in_ci() {
    let mut rng = LcgRng::new(7);
    let (y, d, z) = simulate_strong_iv(&mut rng, 400, 2.0);
    let cfg = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: -5.0,
        grid_max: 5.0,
        grid_size: 401,
    };
    let res = AndersonRubin::test(&y, &d, &z, 2.0, &cfg).unwrap();
    assert!(
        res.p_value > 0.01,
        "p-value at true β too small: {}",
        res.p_value
    );
    let mut covered = false;
    for &(lo, hi) in &res.conf_set {
        if lo <= 2.0 && 2.0 <= hi {
            covered = true;
        }
    }
    assert!(covered, "true β = 2.0 not in CI {:?}", res.conf_set);
}

/// Weak instrument: AR confidence set should be wide (multiple grid
/// points include the null).
#[test]
fn test_weak_iv_yields_wide_ci() {
    let n = 200_usize;
    let mut rng = LcgRng::new(17);
    let mut y = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    let mut z = vec![vec![0.0_f64; 1]; n];
    for i in 0..n {
        let zi = rng.next_normal() as f64;
        // Very weak first stage: coefficient 0.01.
        let nu = (rng.next_normal() as f64) * 1.0;
        let eps = (rng.next_normal() as f64) * 1.0;
        z[i][0] = zi;
        d[i] = 0.01 * zi + nu;
        y[i] = 1.5 * d[i] + eps;
    }
    let cfg = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: -10.0,
        grid_max: 10.0,
        grid_size: 401,
    };
    let res = AndersonRubin::confidence_set(&y, &d, &z, &cfg).unwrap();
    let total_width: f64 = res.conf_set.iter().map(|(lo, hi)| hi - lo).sum();
    assert!(
        total_width > 5.0,
        "expected wide CI, got width {total_width}"
    );
}

/// P-value should monotonically decrease as the null moves further from
/// the true value (away from the strong-IV-implied β).
#[test]
fn test_p_value_monotone_in_null_distance() {
    let mut rng = LcgRng::new(23);
    let (y, d, z) = simulate_strong_iv(&mut rng, 400, 2.0);
    let cfg = AndersonRubinConfig::default();
    let r_at = AndersonRubin::test(&y, &d, &z, 2.0, &cfg).unwrap();
    let r_far = AndersonRubin::test(&y, &d, &z, 5.0, &cfg).unwrap();
    assert!(r_at.p_value > r_far.p_value, "expected p(2) > p(5)");
}

/// Single-instrument (q = 1) case must run without errors.
#[test]
fn test_q1_single_instrument() {
    let mut rng = LcgRng::new(101);
    let (y, d, z) = simulate_strong_iv(&mut rng, 300, 1.0);
    let cfg = AndersonRubinConfig::default();
    let res = AndersonRubin::test(&y, &d, &z, 1.0, &cfg).unwrap();
    assert!(res.ar_statistic.is_finite());
    assert!((0.0..=1.0).contains(&res.p_value));
}

/// Three-instrument (q = 3) case: build z with three independent
/// instruments, all loading on d.
#[test]
fn test_q3_multi_instrument() {
    let n = 400_usize;
    let mut rng = LcgRng::new(53);
    let mut y = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    let mut z = vec![vec![0.0_f64; 3]; n];
    for i in 0..n {
        let z0 = rng.next_normal() as f64;
        let z1 = rng.next_normal() as f64;
        let z2 = rng.next_normal() as f64;
        let nu = (rng.next_normal() as f64) * 0.2;
        let eps = (rng.next_normal() as f64) * 0.2;
        z[i][0] = z0;
        z[i][1] = z1;
        z[i][2] = z2;
        d[i] = 0.8 * z0 + 0.5 * z1 + 0.4 * z2 + nu;
        y[i] = 1.0 * d[i] + eps;
    }
    let cfg = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: -3.0,
        grid_max: 3.0,
        grid_size: 301,
    };
    let res = AndersonRubin::test(&y, &d, &z, 1.0, &cfg).unwrap();
    assert!(res.p_value > 0.001);
    // Critical value for q = 3 is positive.
    assert!(res.critical_value > 0.0);
}

/// Dimension mismatch between y and d must error.
#[test]
fn test_dim_mismatch_y_d() {
    let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let d = vec![0.0_f64, 1.0, 2.0];
    let z: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0]).collect();
    let res = AndersonRubin::test(&y, &d, &z, 0.0, &AndersonRubinConfig::default());
    assert!(res.is_err());
}

/// Dimension mismatch between rows of z and length of y must error.
#[test]
fn test_dim_mismatch_z_rows() {
    let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let d = vec![0.5_f64, 1.0, 1.5, 2.0, 2.5];
    let z: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0]).collect();
    let res = AndersonRubin::test(&y, &d, &z, 0.0, &AndersonRubinConfig::default());
    assert!(res.is_err());
}

/// A single observation cannot support AR inference.
#[test]
fn test_single_observation_errors() {
    let y = vec![1.0_f64];
    let d = vec![0.5_f64];
    let z: Vec<Vec<f64>> = vec![vec![0.0]];
    let res = AndersonRubin::test(&y, &d, &z, 0.0, &AndersonRubinConfig::default());
    assert!(res.is_err());
}

/// When the grid spans the true β, the confidence set must contain it.
#[test]
fn test_grid_spans_true_beta() {
    let mut rng = LcgRng::new(99);
    let (y, d, z) = simulate_strong_iv(&mut rng, 500, 1.7);
    let cfg = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: -3.0,
        grid_max: 3.0,
        grid_size: 601,
    };
    let res = AndersonRubin::confidence_set(&y, &d, &z, &cfg).unwrap();
    let covered = res.conf_set.iter().any(|&(lo, hi)| lo <= 1.7 && 1.7 <= hi);
    assert!(covered, "true β = 1.7 not in {:?}", res.conf_set);
}

/// Non-finite values in y/d/z must be rejected.
#[test]
fn test_nan_guard() {
    let y = vec![1.0_f64, f64::NAN, 3.0, 4.0, 5.0];
    let d = vec![0.5_f64, 1.0, 1.5, 2.0, 2.5];
    let z: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0]).collect();
    let res = AndersonRubin::test(&y, &d, &z, 0.0, &AndersonRubinConfig::default());
    assert!(res.is_err());
    let y_ok = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let z_bad: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            if i == 2 {
                vec![f64::INFINITY]
            } else {
                vec![0.0]
            }
        })
        .collect();
    let res = AndersonRubin::test(&y_ok, &d, &z_bad, 0.0, &AndersonRubinConfig::default());
    assert!(res.is_err());
}

/// Repeated calls on identical inputs must return identical results.
#[test]
fn test_idempotent() {
    let mut rng = LcgRng::new(2025);
    let (y, d, z) = simulate_strong_iv(&mut rng, 200, 1.0);
    let cfg = AndersonRubinConfig::default();
    let a = AndersonRubin::test(&y, &d, &z, 1.0, &cfg).unwrap();
    let b = AndersonRubin::test(&y, &d, &z, 1.0, &cfg).unwrap();
    assert_eq!(a.ar_statistic, b.ar_statistic);
    assert_eq!(a.p_value, b.p_value);
    assert_eq!(a.critical_value, b.critical_value);
    assert_eq!(a.conf_set.len(), b.conf_set.len());
}

/// Invalid α (≤0 or ≥1) must be rejected.
#[test]
fn test_invalid_alpha_rejected() {
    let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let d = vec![0.5_f64, 1.0, 1.5, 2.0, 2.5];
    let z: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0]).collect();
    let bad = AndersonRubinConfig {
        alpha: 0.0,
        grid_min: -1.0,
        grid_max: 1.0,
        grid_size: 3,
    };
    assert!(AndersonRubin::test(&y, &d, &z, 0.0, &bad).is_err());
    let bad2 = AndersonRubinConfig {
        alpha: 1.5,
        grid_min: -1.0,
        grid_max: 1.0,
        grid_size: 3,
    };
    assert!(AndersonRubin::test(&y, &d, &z, 0.0, &bad2).is_err());
}

/// Invalid grid (max ≤ min, size < 2) must be rejected.
#[test]
fn test_invalid_grid_rejected() {
    let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let d = vec![0.5_f64, 1.0, 1.5, 2.0, 2.5];
    let z: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0]).collect();
    let bad = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: 1.0,
        grid_max: 1.0,
        grid_size: 3,
    };
    assert!(AndersonRubin::test(&y, &d, &z, 0.0, &bad).is_err());
    let bad2 = AndersonRubinConfig {
        alpha: 0.05,
        grid_min: 0.0,
        grid_max: 1.0,
        grid_size: 1,
    };
    assert!(AndersonRubin::test(&y, &d, &z, 0.0, &bad2).is_err());
}

/// F-inverse against F-CDF sanity: critical value should satisfy
/// `f_cdf(crit, q, n-q) ≈ 1 - α`.
#[test]
fn test_critical_value_round_trip() {
    let cv = f_inverse_cdf(0.95, 2.0, 30.0);
    let p = f_cdf(cv, 2.0, 30.0);
    assert!((p - 0.95).abs() < 1e-4, "round-trip CDF = {p}");
}

/// The F-CDF at small values must be in [0, 1] and monotone.
#[test]
fn test_f_cdf_monotone() {
    let mut prev = 0.0_f64;
    let mut all_in_range = true;
    let mut monotone = true;
    for k in 0..50 {
        let f = 0.1 * k as f64 + 0.1;
        let c = f_cdf(f, 2.0, 10.0);
        if !(0.0..=1.0).contains(&c) {
            all_in_range = false;
        }
        if c + 1e-9 < prev {
            monotone = false;
        }
        prev = c;
    }
    assert!(all_in_range);
    assert!(monotone);
}
