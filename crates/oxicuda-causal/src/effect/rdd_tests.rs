//! Inline tests for the sharp-RDD estimator.

#![cfg(test)]

use super::rdd::{Rdd, RddConfig, RddKernel};
use crate::error::CausalError;
use crate::handle::LcgRng;

fn rng_uniform(rng: &mut LcgRng) -> f64 {
    (rng.next_f32() as f64) * 2.0 - 1.0
}

/// Generate `(y, r)` with a known jump of size `tau` at `r = 0`.
fn make_step(n: usize, tau: f64, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut r = vec![0.0_f64; n];
    for v in r.iter_mut() {
        *v = rng_uniform(&mut rng);
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let base = 0.3 * r[i];
        let jump = if r[i] >= 0.0 { tau } else { 0.0 };
        y[i] = base + jump + noise * rng_uniform(&mut rng);
    }
    (y, r)
}

/// Smooth `y = f(r)` with no discontinuity.
fn make_smooth(n: usize, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut r = vec![0.0_f64; n];
    for v in r.iter_mut() {
        *v = rng_uniform(&mut rng);
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        y[i] = 0.5 * r[i] + 0.2 * r[i] * r[i] + noise * rng_uniform(&mut rng);
    }
    (y, r)
}

#[test]
fn recovers_step_jump() {
    let tau = 1.0_f64;
    let (y, r) = make_step(2000, tau, 0.1, 31_415);
    let cfg = RddConfig::default();
    let res = Rdd::estimate(&y, &r, &cfg).expect("step fit");
    assert!(
        (res.tau - tau).abs() < 0.15,
        "step τ̂ = {} (expected ~{tau})",
        res.tau
    );
    assert!(res.se > 0.0 && res.se.is_finite());
    assert!(res.n_left > 0);
    assert!(res.n_right > 0);
}

#[test]
fn null_smooth_returns_zero() {
    let (y, r) = make_smooth(2000, 0.05, 271_828);
    let cfg = RddConfig::default();
    let res = Rdd::estimate(&y, &r, &cfg).expect("smooth fit");
    assert!(res.tau.abs() < 0.20, "smooth τ̂ = {} (expected ~0)", res.tau);
}

#[test]
fn three_kernels_agree_on_large_n() {
    let tau = 1.0_f64;
    let (y, r) = make_step(3000, tau, 0.05, 4242);
    let mut estimates = Vec::new();
    for kernel in [
        RddKernel::Triangular,
        RddKernel::Uniform,
        RddKernel::Epanechnikov,
    ] {
        let cfg = RddConfig {
            kernel,
            ..RddConfig::default()
        };
        let res = Rdd::estimate(&y, &r, &cfg).expect("kernel fit");
        estimates.push(res.tau);
    }
    let max = estimates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = estimates.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max - min < 0.25,
        "kernel disagreement {:?}",
        estimates.as_slice()
    );
    for v in estimates {
        assert!((v - tau).abs() < 0.20, "kernel τ̂ = {v}");
    }
}

#[test]
fn ik_bandwidth_positive() {
    let (y, r) = make_step(1000, 1.0, 0.1, 7);
    let h = Rdd::optimal_bandwidth_ik(&y, &r, 0.0).expect("ik");
    assert!(h > 0.0, "h = {h}");
    assert!(h.is_finite());
    assert!(h < 10.0); // Sanity bound on synthetic [-1, 1] data.
}

#[test]
fn explicit_bandwidth_matches_value() {
    let (y, r) = make_step(2000, 1.0, 0.1, 999);
    let cfg = RddConfig {
        bandwidth: Some(0.4),
        ..RddConfig::default()
    };
    let res = Rdd::estimate(&y, &r, &cfg).expect("explicit fit");
    assert!((res.bandwidth_used - 0.4).abs() < 1e-12);
}

#[test]
fn auto_vs_explicit_consistent_when_matched() {
    let (y, r) = make_step(2000, 1.0, 0.05, 1010);
    let cfg_auto = RddConfig::default();
    let res_auto = Rdd::estimate(&y, &r, &cfg_auto).expect("auto");
    let h_used = res_auto.bandwidth_used;
    let cfg_exp = RddConfig {
        bandwidth: Some(h_used),
        ..RddConfig::default()
    };
    let res_exp = Rdd::estimate(&y, &r, &cfg_exp).expect("explicit-match");
    assert!((res_auto.tau - res_exp.tau).abs() < 1e-9);
    assert!((res_auto.se - res_exp.se).abs() < 1e-9);
}

#[test]
fn dimension_mismatch_errors() {
    let cfg = RddConfig::default();
    let r = Rdd::estimate(&[1.0, 2.0], &[1.0], &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn empty_data_errors() {
    let cfg = RddConfig::default();
    let r = Rdd::estimate(&[], &[], &cfg);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

#[test]
fn cutoff_outside_range_errors() {
    let cfg = RddConfig {
        cutoff: 100.0,
        ..RddConfig::default()
    };
    let (y, r) = make_step(200, 1.0, 0.1, 13);
    let res = Rdd::estimate(&y, &r, &cfg);
    assert!(matches!(res, Err(CausalError::IncompatibleData)));
}

#[test]
fn invalid_bandwidth_errors() {
    let cfg = RddConfig {
        bandwidth: Some(0.0),
        ..RddConfig::default()
    };
    let (y, r) = make_step(200, 1.0, 0.1, 14);
    let res = Rdd::estimate(&y, &r, &cfg);
    assert!(matches!(res, Err(CausalError::IncompatibleData)));

    let cfg_neg = RddConfig {
        bandwidth: Some(-0.5),
        ..RddConfig::default()
    };
    let res_neg = Rdd::estimate(&y, &r, &cfg_neg);
    assert!(matches!(res_neg, Err(CausalError::IncompatibleData)));
}

#[test]
fn no_points_on_left_errors() {
    // All r ≥ cutoff means no left-side points.
    let n = 200;
    let mut rng = LcgRng::new(15);
    let mut r = vec![0.0_f64; n];
    for v in r.iter_mut() {
        *v = (rng.next_f32() as f64) * 0.8 + 0.1; // r ∈ [0.1, 0.9]
    }
    let y = vec![1.0_f64; n];
    let cfg = RddConfig {
        cutoff: 0.05,
        bandwidth: Some(0.02),
        ..RddConfig::default()
    };
    let res = Rdd::estimate(&y, &r, &cfg);
    assert!(matches!(res, Err(CausalError::IncompatibleData)));
}

#[test]
fn deterministic() {
    let (y, r) = make_step(500, 1.0, 0.1, 88);
    let cfg = RddConfig::default();
    let r1 = Rdd::estimate(&y, &r, &cfg).expect("first");
    let r2 = Rdd::estimate(&y, &r, &cfg).expect("second");
    assert_eq!(r1.tau, r2.tau);
    assert_eq!(r1.se, r2.se);
    assert_eq!(r1.bandwidth_used, r2.bandwidth_used);
    assert_eq!(r1.n_left, r2.n_left);
    assert_eq!(r1.n_right, r2.n_right);
}

#[test]
fn kernel_weight_sanity() {
    assert!((RddKernel::Triangular.weight(0.0) - 1.0).abs() < 1e-12);
    assert!((RddKernel::Triangular.weight(1.0) - 0.0).abs() < 1e-12);
    assert!((RddKernel::Triangular.weight(0.5) - 0.5).abs() < 1e-12);
    assert!((RddKernel::Triangular.weight(2.0) - 0.0).abs() < 1e-12);

    assert!((RddKernel::Uniform.weight(0.0) - 1.0).abs() < 1e-12);
    assert!((RddKernel::Uniform.weight(0.999) - 1.0).abs() < 1e-12);
    assert!((RddKernel::Uniform.weight(1.5) - 0.0).abs() < 1e-12);

    assert!((RddKernel::Epanechnikov.weight(0.0) - 0.75).abs() < 1e-12);
    assert!((RddKernel::Epanechnikov.weight(1.0) - 0.0).abs() < 1e-12);
    assert!(RddKernel::Epanechnikov.weight(0.5) > 0.0);
}

#[test]
fn n_counts_match_filter() {
    let (y, r) = make_step(500, 1.0, 0.05, 23);
    let cfg = RddConfig {
        bandwidth: Some(0.3),
        ..RddConfig::default()
    };
    let res = Rdd::estimate(&y, &r, &cfg).expect("count fit");
    let n_l = r.iter().filter(|&&v| v < 0.0 && v.abs() <= 0.3).count();
    let n_r = r.iter().filter(|&&v| (0.0..=0.3).contains(&v)).count();
    assert_eq!(res.n_left, n_l);
    assert_eq!(res.n_right, n_r);
}

#[test]
fn negative_tau_recovered() {
    // y drops by 1 at r=0.
    let tau = -1.0_f64;
    let (y, r) = make_step(2000, tau, 0.08, 555);
    let cfg = RddConfig::default();
    let res = Rdd::estimate(&y, &r, &cfg).expect("negative jump");
    assert!(
        (res.tau - tau).abs() < 0.2,
        "negative τ̂ = {} (expected ~{tau})",
        res.tau
    );
}

#[test]
fn default_config_is_sane() {
    let cfg = RddConfig::default();
    assert_eq!(cfg.cutoff, 0.0);
    assert!(cfg.bandwidth.is_none());
    assert_eq!(cfg.kernel, RddKernel::Triangular);
}

#[test]
fn ik_bandwidth_dim_mismatch_errors() {
    let r = Rdd::optimal_bandwidth_ik(&[1.0, 2.0], &[1.0], 0.0);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}
