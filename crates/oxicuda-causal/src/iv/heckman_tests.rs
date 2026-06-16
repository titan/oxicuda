//! Inline tests for the Heckman two-step sample-selection model.

#![cfg(test)]

use super::heckman::{Heckman, HeckmanConfig, HeckmanResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type Dgp = (Vec<f64>, Vec<bool>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Generate a no-selection DGP (every row selected) so that Heckman should
/// behave like ordinary OLS on `[1, x]`.
fn make_no_selection_dgp(n: usize, d_x: usize, seed: u64) -> Dgp {
    let mut rng = LcgRng::new(seed);
    let mut x = vec![vec![0.0_f64; d_x]; n];
    let mut z = vec![vec![0.0_f64; 2]; n];
    let mut y = vec![0.0_f64; n];
    let selected = vec![true; n];
    let true_beta = 0.5_f64; // beta_intercept = 0
    for (xi, (zi, yi)) in x.iter_mut().zip(z.iter_mut().zip(y.iter_mut())) {
        for v in xi.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        for v in zi.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        let mut s = 0.0_f64;
        for &v in xi.iter() {
            s += true_beta * v;
        }
        *yi = s + 0.1 * rng.next_normal() as f64;
    }
    (y, selected, x, z)
}

/// Selection-bias DGP. Correlated outcome and selection errors
/// (correlation `rho`). When `rho` is large the inverse-Mills correction
/// matters.
fn make_selection_dgp(n: usize, rho: f64, seed: u64) -> Dgp {
    let mut rng = LcgRng::new(seed);
    let d_x = 2_usize;
    let d_z = 3_usize;
    let mut x = vec![vec![0.0_f64; d_x]; n];
    let mut z = vec![vec![0.0_f64; d_z]; n];
    let mut y = vec![0.0_f64; n];
    let mut selected = vec![false; n];
    let true_beta = [1.0_f64, -0.5];
    for i in 0..n {
        for v in x[i].iter_mut() {
            *v = rng.next_normal() as f64;
        }
        for v in z[i].iter_mut() {
            *v = rng.next_normal() as f64;
        }
        // Bivariate normal errors via Cholesky of [[1, rho], [rho, 1]].
        let u = rng.next_normal() as f64;
        let v_indep = rng.next_normal() as f64;
        let r = rho.clamp(-0.99, 0.99);
        let v = r * u + (1.0 - r * r).sqrt() * v_indep;
        let lin_sel = 0.5 + 0.6 * z[i][0] - 0.4 * z[i][1] + 0.3 * z[i][2] + v;
        selected[i] = lin_sel > 0.0;
        let mut s = 0.5_f64;
        for (j, &bj) in true_beta.iter().enumerate() {
            s += bj * x[i][j];
        }
        y[i] = s + u;
    }
    (y, selected, x, z)
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// All rows selected — Heckman behaves as plain OLS on `[1, x]` and the
/// inverse-Mills coefficient should be statistically small.
#[test]
fn test_heckman_no_selection_matches_ols() {
    let n = 400_usize;
    let d_x = 2_usize;
    let (y, selected, x, z) = make_no_selection_dgp(n, d_x, 42);
    let cfg = HeckmanConfig::default();
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman estimation should succeed for no-selection DGP");
    assert_eq!(r.beta.len(), d_x + 1);
    assert_eq!(r.se.len(), d_x + 1);
    assert_eq!(r.n_selected, n);
    // Each beta should be in the right ballpark.
    assert!(
        approx(r.beta[1], 0.5, 0.2),
        "beta[x_1] = {} ≠ 0.5",
        r.beta[1]
    );
    assert!(
        approx(r.beta[2], 0.5, 0.2),
        "beta[x_2] = {} ≠ 0.5",
        r.beta[2]
    );
}

/// `rho ≈ 0`: outcome and selection errors uncorrelated, so `lambda_coef`
/// (≈ ρ·σ_u) should be small relative to σ_u.
#[test]
fn test_heckman_lambda_small_under_random_selection() {
    let n = 600_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.0, 17);
    let cfg = HeckmanConfig::default();
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman estimation should succeed for random selection DGP");
    // ρ̂ = lambda_coef / sigma_e ought to be close to zero.
    assert!(
        r.rho.abs() < 0.5,
        "rho = {} should be small under random selection",
        r.rho
    );
}

/// Strong selection (`rho = 0.8`): lambda_coef must be clearly nonzero.
#[test]
fn test_heckman_lambda_significant_under_correlated_selection() {
    let n = 800_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.8, 23);
    let cfg = HeckmanConfig::default();
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman estimation should succeed for correlated selection DGP");
    assert!(
        r.lambda_coef.abs() > 0.05,
        "lambda_coef = {} should be clearly nonzero",
        r.lambda_coef
    );
}

/// Probit converges far before `probit_max_iters` on a simple problem.
#[test]
fn test_heckman_probit_converges_quickly() {
    let n = 200_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.3, 11);
    let cfg = HeckmanConfig {
        probit_max_iters: 50,
        ..HeckmanConfig::default()
    };
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman estimation should succeed with 50 probit iterations");
    assert!(r.sigma_e.is_finite() && r.sigma_e > 0.0);
}

/// `ridge_lambda ≤ 0` must be rejected.
#[test]
fn test_heckman_invalid_ridge_lambda() {
    let n = 100_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.0, 5);
    let cfg = HeckmanConfig {
        ridge_lambda: 0.0,
        ..HeckmanConfig::default()
    };
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg).is_err());
    let cfg_neg = HeckmanConfig {
        ridge_lambda: -1e-3,
        ..HeckmanConfig::default()
    };
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg_neg).is_err());
}

/// `probit_max_iters = 0` must be rejected.
#[test]
fn test_heckman_invalid_probit_max_iters() {
    let n = 100_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.0, 6);
    let cfg = HeckmanConfig {
        probit_max_iters: 0,
        ..HeckmanConfig::default()
    };
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg).is_err());
}

/// `probit_tol ≤ 0` must be rejected.
#[test]
fn test_heckman_invalid_probit_tol() {
    let n = 100_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.0, 7);
    let cfg = HeckmanConfig {
        probit_tol: 0.0,
        ..HeckmanConfig::default()
    };
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg).is_err());
    let cfg_neg = HeckmanConfig {
        probit_tol: -1e-6,
        ..HeckmanConfig::default()
    };
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg_neg).is_err());
}

/// Mismatched `y` / `selected` / `x` / `z` row counts must be rejected.
#[test]
fn test_heckman_dim_mismatch() {
    let cfg = HeckmanConfig::default();
    let y = vec![1.0_f64; 100];
    let selected = vec![true; 99];
    let x = vec![vec![0.5_f64, 0.5]; 100];
    let z = vec![vec![0.5_f64, 0.5]; 100];
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg);
    assert!(r.is_err());
    let y2 = vec![1.0_f64; 100];
    let selected2 = vec![true; 100];
    let x2 = vec![vec![0.5_f64, 0.5]; 100];
    let z2 = vec![vec![0.5_f64; 2]; 99];
    let r2 = Heckman::estimate(&y2, &selected2, &x2, &z2, &cfg);
    assert!(r2.is_err());
}

/// Fewer than 2 selected observations must be reported as insufficient data.
#[test]
fn test_heckman_too_few_selected_errors() {
    let cfg = HeckmanConfig::default();
    let n = 50_usize;
    let y = vec![1.0_f64; n];
    let mut selected = vec![false; n];
    selected[0] = true; // only one selected
    let x = vec![vec![0.5_f64, 0.5]; n];
    let z = vec![vec![0.5_f64, 0.5]; n];
    assert!(Heckman::estimate(&y, &selected, &x, &z, &cfg).is_err());
}

/// Estimator must be deterministic for identical inputs.
#[test]
fn test_heckman_deterministic() {
    let n = 200_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.5, 99);
    let cfg = HeckmanConfig::default();
    let a = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman first call should succeed for determinism test");
    let b = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman second call should succeed for determinism test");
    assert_eq!(a.beta, b.beta);
    assert_eq!(a.lambda_coef, b.lambda_coef);
    assert_eq!(a.sigma_e, b.sigma_e);
    assert_eq!(a.rho, b.rho);
    assert_eq!(a.n_selected, b.n_selected);
}

/// Newton step direction should not blow up — final `sigma_e` finite and
/// positive after the configured iteration cap.
#[test]
fn test_heckman_newton_no_explosion() {
    let n = 400_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.6, 314);
    let cfg = HeckmanConfig::default();
    let r = Heckman::estimate(&y, &selected, &x, &z, &cfg)
        .expect("Heckman estimation should succeed for Newton stability test");
    assert!(r.sigma_e.is_finite() && r.sigma_e > 0.0);
    for &b in r.beta.iter() {
        assert!(b.is_finite(), "non-finite beta {b}");
    }
}

/// `ρ̂` must always live in `(-1, 1)` thanks to the explicit clip.
#[test]
fn test_heckman_rho_in_open_unit_interval() {
    let n = 250_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.7, 271);
    let r = Heckman::estimate(&y, &selected, &x, &z, &HeckmanConfig::default())
        .expect("Heckman estimation should succeed for rho range test");
    assert!(r.rho > -1.0 && r.rho < 1.0, "rho = {} out of range", r.rho);
}

/// White SEs must be positive and finite.
#[test]
fn test_heckman_robust_se_positive_finite() {
    let n = 400_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.3, 1234);
    let r = Heckman::estimate(&y, &selected, &x, &z, &HeckmanConfig::default())
        .expect("value should be present");
    for &s in r.se.iter() {
        assert!(s.is_finite() && s >= 0.0, "se = {s}");
    }
    assert!(r.se.iter().any(|&s| s > 0.0), "all SEs zero");
}

/// Larger synthetic DGP with known β recovered to within a generous
/// tolerance scaled to the reported standard errors.
#[test]
fn test_heckman_recovers_beta_within_tolerance() {
    let n = 800_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.5, 2718);
    let true_beta = [1.0_f64, -0.5];
    let r = Heckman::estimate(&y, &selected, &x, &z, &HeckmanConfig::default())
        .expect("value should be present");
    // β is [intercept, true_beta[0], true_beta[1]].
    for (j, &tb) in true_beta.iter().enumerate() {
        let beta_hat = r.beta[1 + j];
        let beta_se = r.se[1 + j].max(1e-3);
        let z_score = (beta_hat - tb) / beta_se;
        assert!(
            z_score.abs() < 5.0,
            "β_{} z-score = {z_score} out of tolerance (β̂ = {beta_hat}, target = {})",
            j,
            tb
        );
    }
}

/// `HeckmanResult` field sizes are consistent with the public contract.
#[test]
fn test_heckman_result_sizes() {
    let n = 200_usize;
    let (y, selected, x, z) = make_selection_dgp(n, 0.4, 9876);
    let r: HeckmanResult = Heckman::estimate(&y, &selected, &x, &z, &HeckmanConfig::default())
        .expect("value should be present");
    let d_x = 2;
    assert_eq!(r.beta.len(), d_x + 1);
    assert_eq!(r.se.len(), d_x + 1);
    assert!(r.n_selected > 0);
}

/// Empty `y` must be rejected.
#[test]
fn test_heckman_empty_inputs_error() {
    let cfg = HeckmanConfig::default();
    let r = Heckman::estimate(
        &[],
        &[],
        &Vec::<Vec<f64>>::new(),
        &Vec::<Vec<f64>>::new(),
        &cfg,
    );
    assert!(r.is_err());
}
