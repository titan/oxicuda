//! Inline tests for the Imai-Keele-Tingley causal-mediation estimator.

#![cfg(test)]

use super::mediation::{
    Mediation, MediationConfig, compute_acme_ade_for_tests, empirical_ci_for_tests,
};
use crate::error::CausalError;
use crate::handle::LcgRng;

fn rng_uniform(rng: &mut LcgRng) -> f64 {
    (rng.next_f32() as f64) * 2.0 - 1.0
}

/// Synthetic mediation DGP under sequential ignorability.
///
/// Mediator:   M = α_m + β_t · T + Σ_k 0.3 · X_k + ε_m   (ε_m ~ N(0, 0.05²))
/// Outcome:    Y = α_y + γ_t · T + γ_m · M + Σ_k 0.5 · X_k + ε_y
/// Treatment:  T ~ Bernoulli(0.5) (RCT-style, independent of X)
fn make_mediation(
    n: usize,
    d: usize,
    beta_t: f64,
    gamma_t: f64,
    gamma_m: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    // `d` is consumed below via the `x` shape.
    let mut rng = LcgRng::new(seed);
    let mut x = vec![vec![0.0_f64; d]; n];
    for row in x.iter_mut() {
        for v in row.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
    }
    let mut t = vec![0.0_f64; n];
    for ti in t.iter_mut() {
        *ti = if rng.next_f32() > 0.5 { 1.0 } else { 0.0 };
    }
    let mut m = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.5_f64; // α_m
        for xij in &x[i] {
            s += 0.3 * xij;
        }
        m[i] = s + beta_t * t[i] + 0.05 * rng_uniform(&mut rng);
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.2_f64; // α_y
        for xij in &x[i] {
            s += 0.5 * xij;
        }
        y[i] = s + gamma_t * t[i] + gamma_m * m[i] + 0.05 * rng_uniform(&mut rng);
    }
    (y, t, m, x)
}

// --------- shape / validation tests -------------------------------------

#[test]
fn empty_inputs_error() {
    let cfg = MediationConfig::default();
    let r = Mediation::estimate(&[], &[], &[], &[], &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn dim_mismatch_t_errors() {
    let cfg = MediationConfig::default();
    let y = vec![0.0_f64; 100];
    let t = vec![0.0_f64; 99]; // wrong length
    let m = vec![0.0_f64; 100];
    let x = vec![vec![0.0_f64; 2]; 100];
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn dim_mismatch_m_errors() {
    let cfg = MediationConfig::default();
    let y = vec![0.0_f64; 100];
    let t = vec![0.0_f64; 100];
    let m = vec![0.0_f64; 50]; // wrong length
    let x = vec![vec![0.0_f64; 2]; 100];
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn dim_mismatch_x_rows_errors() {
    let cfg = MediationConfig::default();
    let y = vec![0.0_f64; 100];
    let t = vec![0.0_f64; 100];
    let m = vec![0.0_f64; 100];
    let x = vec![vec![0.0_f64; 2]; 99]; // wrong length
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn dim_mismatch_x_cols_errors() {
    let cfg = MediationConfig::default();
    let n = 50;
    let y = vec![0.0_f64; n];
    let t = vec![0.0_f64; n];
    let m = vec![0.0_f64; n];
    let mut x: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0_f64, 1.0]).collect();
    x[5] = vec![0.0_f64, 1.0, 2.0]; // wrong width
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn n_too_small_errors() {
    // n = 4 < 5
    let cfg = MediationConfig::default();
    let n = 4;
    let y = vec![0.0_f64; n];
    let t = vec![0.0_f64; n];
    let m = vec![0.0_f64; n];
    let x = vec![vec![0.0_f64; 2]; n];
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn ridge_lambda_zero_errors() {
    let cfg = MediationConfig {
        ridge_lambda: 0.0,
        n_simulations: 100,
        seed: 0,
    };
    let (y, t, m, x) = make_mediation(20, 2, 1.0, 1.0, 1.0, 1);
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn ridge_lambda_negative_errors() {
    let cfg = MediationConfig {
        ridge_lambda: -0.001,
        n_simulations: 100,
        seed: 0,
    };
    let (y, t, m, x) = make_mediation(20, 2, 1.0, 1.0, 1.0, 2);
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn n_simulations_too_small_errors() {
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 50,
        seed: 0,
    };
    let (y, t, m, x) = make_mediation(20, 2, 1.0, 1.0, 1.0, 3);
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn non_binary_treatment_errors() {
    let cfg = MediationConfig::default();
    let (y, mut t, m, x) = make_mediation(20, 2, 1.0, 1.0, 1.0, 4);
    t[3] = 0.5; // not binary
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn non_finite_mediator_errors() {
    let cfg = MediationConfig::default();
    let (y, t, mut m, x) = make_mediation(20, 2, 1.0, 1.0, 1.0, 5);
    m[7] = f64::NAN;
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

// --------- correctness tests --------------------------------------------

#[test]
fn no_mediation_dgp_acme_near_zero() {
    // β_t = 0 ⇒ M is not affected by T ⇒ ACME ≈ 0.
    let n = 600;
    let d = 2;
    let (y, t, m, x) = make_mediation(
        n, d, /*β_t=*/ 0.0, /*γ_t=*/ 1.0, /*γ_m=*/ 0.8, 101,
    );
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 500,
        seed: 7,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(r.acme.abs() < 0.10, "ACME = {} (expected ≈ 0)", r.acme);
}

#[test]
fn all_mediation_dgp_ade_near_zero() {
    // γ_t = 0 ⇒ no direct effect ⇒ ADE ≈ 0.
    let n = 600;
    let d = 2;
    let (y, t, m, x) = make_mediation(
        n, d, /*β_t=*/ 1.0, /*γ_t=*/ 0.0, /*γ_m=*/ 0.8, 202,
    );
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 500,
        seed: 8,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(r.ade.abs() < 0.15, "ADE = {} (expected ≈ 0)", r.ade);
}

#[test]
fn total_equals_acme_plus_ade() {
    let n = 300;
    let d = 2;
    let (y, t, m, x) = make_mediation(n, d, 1.0, 0.8, 0.5, 303);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 200,
        seed: 9,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(
        (r.total_effect - (r.acme + r.ade)).abs() < 1e-6,
        "total={} acme+ade={}",
        r.total_effect,
        r.acme + r.ade
    );
}

#[test]
fn deterministic_under_same_seed() {
    let n = 80;
    let d = 2;
    let (y, t, m, x) = make_mediation(n, d, 1.0, 0.5, 0.6, 11);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 200,
        seed: 42,
    };
    let r1 = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    let r2 = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert_eq!(r1.acme, r2.acme);
    assert_eq!(r1.ade, r2.ade);
    assert_eq!(r1.acme_ci, r2.acme_ci);
    assert_eq!(r1.ade_ci, r2.ade_ci);
}

#[test]
fn cis_contain_truth_on_synthetic_dgp() {
    // β_t = 1.0, γ_m = 0.8 ⇒ ACME_true = β_t · γ_m = 0.8.
    // γ_t = 0.5 ⇒ ADE_true = 0.5.
    let beta_t = 1.0;
    let gamma_t = 0.5;
    let gamma_m = 0.8;
    let acme_truth = beta_t * gamma_m;
    let ade_truth = gamma_t;
    let mut covered_acme = 0_usize;
    let mut covered_ade = 0_usize;
    let total: usize = 4;
    for seed in 0..total {
        let (y, t, m, x) = make_mediation(400, 2, beta_t, gamma_t, gamma_m, 1_000 + seed as u64);
        let cfg = MediationConfig {
            ridge_lambda: 1e-3,
            n_simulations: 500,
            seed: seed as u64,
        };
        let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
        if r.acme_ci.0 <= acme_truth && acme_truth <= r.acme_ci.1 {
            covered_acme += 1;
        }
        if r.ade_ci.0 <= ade_truth && ade_truth <= r.ade_ci.1 {
            covered_ade += 1;
        }
    }
    assert!(
        covered_acme >= total - 1,
        "ACME CI covered only {covered_acme}/{total} datasets"
    );
    assert!(
        covered_ade >= total - 1,
        "ADE CI covered only {covered_ade}/{total} datasets"
    );
}

#[test]
fn ci_lower_below_upper() {
    let (y, t, m, x) = make_mediation(200, 2, 1.0, 0.5, 0.6, 12);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 200,
        seed: 13,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(r.acme_ci.0 <= r.acme_ci.1);
    assert!(r.ade_ci.0 <= r.ade_ci.1);
}

#[test]
fn prop_mediated_finite_when_total_nonzero() {
    let (y, t, m, x) = make_mediation(200, 2, 0.7, 0.4, 0.6, 14);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 200,
        seed: 15,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(r.total_effect.abs() > 0.05);
    assert!(r.prop_mediated.is_finite());
}

#[test]
fn result_n_matches_input() {
    let n = 75;
    let (y, t, m, x) = make_mediation(n, 2, 1.0, 0.5, 0.6, 16);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 100,
        seed: 17,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert_eq!(r.n, n);
}

#[test]
fn acme_ade_are_finite() {
    let (y, t, m, x) = make_mediation(120, 3, 0.6, 0.3, 0.4, 18);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 100,
        seed: 19,
    };
    let r = Mediation::estimate(&y, &t, &m, &x, &cfg).unwrap();
    assert!(r.acme.is_finite());
    assert!(r.ade.is_finite());
    assert!(r.total_effect.is_finite());
}

#[test]
fn larger_gamma_m_increases_acme() {
    // Hold β_t fixed and vary γ_m; ACME should track γ_m at fixed β_t.
    let (y0, t0, m0, x0) = make_mediation(400, 2, 1.0, 0.5, 0.2, 21);
    let (y1, t1, m1, x1) = make_mediation(400, 2, 1.0, 0.5, 1.0, 21);
    let cfg = MediationConfig {
        ridge_lambda: 1e-3,
        n_simulations: 200,
        seed: 22,
    };
    let r0 = Mediation::estimate(&y0, &t0, &m0, &x0, &cfg).unwrap();
    let r1 = Mediation::estimate(&y1, &t1, &m1, &x1, &cfg).unwrap();
    assert!(
        r1.acme > r0.acme + 0.3,
        "acme small={} large={}",
        r0.acme,
        r1.acme
    );
}

// --------- helper-level tests ------------------------------------------

#[test]
fn compute_acme_ade_constant_zero_when_no_effect() {
    // β_t = 0, γ_t = 0, γ_m = 0 ⇒ both ACME = ADE = 0.
    let d = 2;
    // beta_m = [α_m, β_t, β_x1, β_x2] with β_t = 0
    let beta_m = vec![1.0, 0.0, 0.3, -0.1];
    // beta_y = [α_y, γ_t, γ_m, γ_tm, γ_x1, γ_x2] with γ_t=0, γ_m=0, γ_tm=0
    let beta_y = vec![0.5, 0.0, 0.0, 0.0, 0.2, 0.4];
    let x = vec![
        vec![0.1, -0.2],
        vec![0.3, 0.5],
        vec![-0.4, 0.0],
        vec![0.7, 0.2],
    ];
    let (acme, ade) = compute_acme_ade_for_tests(&beta_m, &beta_y, &x, d);
    assert!(acme.abs() < 1e-12);
    assert!(ade.abs() < 1e-12);
}

#[test]
fn compute_acme_ade_pure_indirect_path() {
    // β_t = 1.0, γ_m = 2.0, γ_t = 0 ⇒ ACME = β_t · γ_m = 2.0, ADE = 0.
    let d = 1;
    let beta_m = vec![0.0, 1.0, 0.5]; // [α_m, β_t, β_x]
    let beta_y = vec![0.0, 0.0, 2.0, 0.0, 0.3]; // [α_y, γ_t, γ_m, γ_tm, γ_x]
    let x = vec![vec![0.1], vec![-0.4], vec![0.7]];
    let (acme, ade) = compute_acme_ade_for_tests(&beta_m, &beta_y, &x, d);
    assert!((acme - 2.0).abs() < 1e-12, "acme = {}", acme);
    assert!(ade.abs() < 1e-12, "ade = {}", ade);
}

#[test]
fn empirical_ci_quantiles_are_ordered() {
    let mut draws: Vec<f64> = (0..200).map(|i| i as f64 / 100.0).collect();
    let (lo, hi) = empirical_ci_for_tests(&mut draws, 0.05, 0.95);
    assert!(lo < hi);
    // Roughly 5th and 95th percentile of evenly-spaced [0, 2).
    assert!(lo > 0.05 && lo < 0.15, "5th pct = {}", lo);
    assert!(hi > 1.85 && hi < 1.95, "95th pct = {}", hi);
}
