//! Inline tests for the TMLE estimator.

#![cfg(test)]

use super::tmle::{Tmle, TmleConfig, sigmoid};
use crate::error::CausalError;
use crate::handle::LcgRng;

fn rng_uniform(rng: &mut LcgRng) -> f64 {
    (rng.next_f32() as f64) * 2.0 - 1.0
}

/// Randomised-controlled-trial-style data: T independent of X.
fn make_rct(n: usize, d: usize, tau: f64, seed: u64) -> (Vec<f64>, Vec<u32>, Vec<Vec<f64>>) {
    let mut rng = LcgRng::new(seed);
    let mut x = vec![vec![0.0_f64; d]; n];
    for row in x.iter_mut() {
        for v in row.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
    }
    let mut t = vec![0_u32; n];
    for ti in t.iter_mut() {
        *ti = if rng.next_f32() > 0.5 { 1 } else { 0 };
    }
    let alpha: Vec<f64> = (0..d).map(|i| 0.5 + 0.1 * i as f64).collect();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..d {
            s += alpha[j] * x[i][j];
        }
        y[i] = s + tau * t[i] as f64 + 0.1 * rng_uniform(&mut rng);
    }
    (y, t, x)
}

/// Confounded synthetic data: X influences both T and Y.
fn make_confounded(n: usize, d: usize, tau: f64, seed: u64) -> (Vec<f64>, Vec<u32>, Vec<Vec<f64>>) {
    let mut rng = LcgRng::new(seed);
    let mut x = vec![vec![0.0_f64; d]; n];
    for row in x.iter_mut() {
        for v in row.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
    }
    let mut t = vec![0_u32; n];
    for i in 0..n {
        // Strong dependence on x[0]: more likely to be treated if x[0] > 0.
        let lin = 1.5 * x[i][0];
        let prob = sigmoid(lin);
        t[i] = if (rng.next_f32() as f64) < prob { 1 } else { 0 };
    }
    let mut y = vec![0.0_f64; n];
    let alpha: Vec<f64> = (0..d).map(|i| 0.7 + 0.2 * i as f64).collect();
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..d {
            s += alpha[j] * x[i][j];
        }
        y[i] = s + tau * t[i] as f64 + 0.1 * rng_uniform(&mut rng);
    }
    (y, t, x)
}

#[test]
fn recovers_rct_ate() {
    let tau = 1.0_f64;
    let (y, t, x) = make_rct(2000, 3, tau, 31_415);
    let cfg = TmleConfig::default();
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("rct fit");
    assert!(
        (res.ate - tau).abs() < 0.15,
        "rct ATE = {} (expected ~{tau})",
        res.ate
    );
    assert!(res.se > 0.0 && res.se.is_finite());
    assert!(res.ic_var > 0.0);
    assert_eq!(res.n, 2000);
}

#[test]
fn confounded_beats_naive() {
    let tau = 1.0_f64;
    let (y, t, x) = make_confounded(1500, 3, tau, 271_828);
    let cfg = TmleConfig::default();
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("confounded fit");

    // Naive estimator: mean(Y | T=1) − mean(Y | T=0).
    let mut sum1 = 0.0_f64;
    let mut sum0 = 0.0_f64;
    let mut n1 = 0_usize;
    let mut n0 = 0_usize;
    for i in 0..y.len() {
        if t[i] == 1 {
            sum1 += y[i];
            n1 += 1;
        } else {
            sum0 += y[i];
            n0 += 1;
        }
    }
    let naive = sum1 / n1 as f64 - sum0 / n0 as f64;
    assert!((res.ate - tau).abs() <= (naive - tau).abs() + 0.05);
}

#[test]
fn propensity_clip_respected() {
    // Construct a tiny dataset where one side is rare → uncorrected
    // propensity could approach 0 or 1.
    let n = 200;
    let d = 2;
    let mut rng = LcgRng::new(7);
    let mut x = vec![vec![0.0_f64; d]; n];
    for row in x.iter_mut() {
        for v in row.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
    }
    // Almost-always-treat pattern.
    let mut t = vec![1_u32; n];
    t[0] = 0;
    t[1] = 0;
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        y[i] = x[i][0] + 0.5 * t[i] as f64;
    }
    let cfg = TmleConfig {
        clip_eps: 0.1,
        ..TmleConfig::default()
    };
    // The estimator must succeed (clipping prevents H from blowing up).
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("clip fit");
    assert!(res.ate.is_finite());
    assert!(res.se.is_finite());
}

#[test]
fn invalid_n_folds_too_small() {
    let cfg = TmleConfig {
        n_folds: 1,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 1);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::InvalidNumFolds { .. })));
}

#[test]
fn invalid_clip_eps_out_of_range() {
    let cfg = TmleConfig {
        clip_eps: 0.6,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 2);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_clip_eps_zero() {
    let cfg = TmleConfig {
        clip_eps: 0.0,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 3);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_negative_ridge() {
    let cfg = TmleConfig {
        ridge_lambda: -0.1,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 4);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_zero_tol() {
    let cfg = TmleConfig {
        tol: 0.0,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 5);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_zero_iters() {
    let cfg = TmleConfig {
        max_outer_iters: 0,
        ..TmleConfig::default()
    };
    let (y, t, x) = make_rct(200, 2, 1.0, 6);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_length_mismatch() {
    let cfg = TmleConfig::default();
    let (y, mut t, x) = make_rct(200, 2, 1.0, 8);
    t.pop();
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn invalid_x_row_width() {
    let cfg = TmleConfig::default();
    let (y, t, mut x) = make_rct(200, 2, 1.0, 9);
    x[3].push(0.0);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn invalid_t_value() {
    let cfg = TmleConfig::default();
    let (y, mut t, x) = make_rct(200, 2, 1.0, 10);
    t[7] = 5;
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_empty_data() {
    let cfg = TmleConfig::default();
    let r = Tmle::estimate(&[], &[], &[], &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn deterministic() {
    let (y, t, x) = make_rct(300, 2, 0.8, 11);
    let cfg = TmleConfig::default();
    let r1 = Tmle::estimate(&y, &t, &x, &cfg).expect("first");
    let r2 = Tmle::estimate(&y, &t, &x, &cfg).expect("second");
    assert_eq!(r1.ate, r2.ate);
    assert_eq!(r1.se, r2.se);
    assert_eq!(r1.ic_var, r2.ic_var);
    assert_eq!(r1.n, r2.n);
}

#[test]
fn single_feature_x() {
    // d = 1 still works.
    let n = 600;
    let mut rng = LcgRng::new(12_345);
    let mut x = vec![vec![0.0_f64; 1]; n];
    for row in x.iter_mut() {
        row[0] = rng_uniform(&mut rng);
    }
    let mut t = vec![0_u32; n];
    for ti in t.iter_mut() {
        *ti = if rng.next_f32() > 0.5 { 1 } else { 0 };
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        y[i] = 0.5 * x[i][0] + 1.0 * t[i] as f64 + 0.05 * rng_uniform(&mut rng);
    }
    let cfg = TmleConfig::default();
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("single-feat fit");
    assert!((res.ate - 1.0).abs() < 0.2);
}

#[test]
fn multi_feature_x() {
    let (y, t, x) = make_rct(800, 5, 1.5, 22);
    let cfg = TmleConfig::default();
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("multi-feat fit");
    assert!((res.ate - 1.5).abs() < 0.2);
}

#[test]
fn targeting_converges() {
    // After targeting, the IC mean should be ≈ 0 — that is the EIF score
    // equation that TMLE solves exactly.
    let (y, t, x) = make_confounded(1200, 3, 0.8, 13);
    let cfg = TmleConfig::default();
    let res = Tmle::estimate(&y, &t, &x, &cfg).expect("converge fit");
    // SE should be in a sensible range; if targeting diverged we would see
    // either a non-finite SE or one orders of magnitude larger.
    assert!(res.se > 0.0);
    assert!(res.se < 1.0);
}

#[test]
fn idempotent_repeated_calls() {
    let (y, t, x) = make_confounded(600, 3, 0.5, 999);
    let cfg = TmleConfig::default();
    let mut last_ate = None;
    for _ in 0..3 {
        let res = Tmle::estimate(&y, &t, &x, &cfg).expect("idempotent");
        if let Some(prev) = last_ate {
            assert_eq!(prev, res.ate);
        }
        last_ate = Some(res.ate);
    }
}

#[test]
fn n_too_small_for_folds() {
    let cfg = TmleConfig::default();
    // n must be ≥ n_folds * (2d + 3) = 5 * 7 = 35 for d = 2.
    let (y, t, x) = make_rct(10, 2, 1.0, 0);
    let r = Tmle::estimate(&y, &t, &x, &cfg);
    assert!(matches!(r, Err(CausalError::InvalidNumFolds { .. })));
}

#[test]
fn config_default_is_sane() {
    let cfg = TmleConfig::default();
    assert!(cfg.n_folds >= 2);
    assert!(cfg.ridge_lambda >= 0.0);
    assert!(cfg.clip_eps > 0.0 && cfg.clip_eps < 0.5);
    assert!(cfg.tol > 0.0);
    assert!(cfg.max_outer_iters >= 1);
}
