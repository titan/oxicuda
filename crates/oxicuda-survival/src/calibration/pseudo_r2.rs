//! Royston-Sauerbrei pseudo-R² and the D measure of prognostic separation
//! (Royston & Sauerbrei 2004, *Stat. Med.* 23:723-748).
//!
//! For survival models there is no single agreed coefficient of determination.
//! Royston and Sauerbrei propose `R²_D`, derived from a measure `D` of
//! prognostic separation that is robust to censoring and to outliers in the
//! prognostic index (PI) `η_i = β'x_i`.
//!
//! # Construction
//!
//! 1. Rank the `n` prognostic indices and replace each by its expected
//!    standard-normal order statistic (the *rankits* / Blom scores)
//!    `z_i = Φ⁻¹((r_i − 3/8)/(n + 1/4))`, where `r_i` is the rank of `η_i`.
//! 2. Scale the rankits by `κ = √(8/π) ≈ 1.5958`, giving `z_i / κ`.
//! 3. Fit a Cox model (here: a one-covariate partial-likelihood Newton step)
//!    of the outcome on the single covariate `z_i / κ`; the resulting
//!    coefficient **is** `D`. `D` is interpretable as the log-hazard-ratio
//!    between the upper and lower halves of a standard-normal prognostic index.
//! 4. The explained-variation summary is
//!
//! ```text
//! R²_D = (D² σ²) / (D² σ² + π²/6) ,   σ² = 8/π² ,
//! ```
//!
//! so `R²_D = D² (8/π²) / (D² (8/π²) + π²/6)`. Larger `D` ⇒ larger `R²_D`,
//! bounded in `[0, 1)`, and `R²_D = 0` when the model has no discriminative
//! power (`D = 0`).
//!
//! This module is self-contained: it computes the normal inverse-CDF and a
//! one-covariate Cox partial-likelihood fit internally, so it does not depend on
//! the full Cox machinery.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Scaling constant `κ = √(8/π)` applied to the rankits before the Cox fit.
const KAPPA: f64 = 1.595_769_121_605_731;

/// Result of the Royston-Sauerbrei pseudo-R² computation.
#[derive(Debug, Clone)]
pub struct PseudoR2Result {
    /// The `D` statistic (a log-hazard-ratio on the scaled-rankit covariate).
    pub d_statistic: f64,
    /// `R²_D`, Royston's coefficient of explained variation, in `[0, 1)`.
    pub r2_d: f64,
    /// Standard error of `D` from the one-covariate Cox information.
    pub d_se: f64,
    /// Number of subjects used.
    pub n: usize,
}

impl PseudoR2Result {
    /// Approximate `(1 − α)` Wald confidence interval for `D` using a normal
    /// quantile. Returns `(lower, upper)`.
    pub fn d_confidence_interval(&self, z: f64) -> (f64, f64) {
        (
            self.d_statistic - z * self.d_se,
            self.d_statistic + z * self.d_se,
        )
    }
}

/// Acklam's rational approximation to the inverse standard-normal CDF.
fn norm_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.024_25;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Convert a prognostic index to Blom-score rankits: rank, then map each rank to
/// its expected standard-normal order statistic. Ties receive the average rank.
fn rankits(eta: &[f64]) -> Vec<f64> {
    let n = eta.len();
    // Index sorted by value.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eta[a]
            .partial_cmp(&eta[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Average-rank assignment (1-based) handling ties.
    let mut avg_rank = vec![0.0_f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && (eta[idx[j + 1]] - eta[idx[i]]).abs() < 1.0e-12 {
            j += 1;
        }
        // Ranks i+1 .. j+1 (1-based) → average.
        let lo = (i + 1) as f64;
        let hi = (j + 1) as f64;
        let mean_rank = 0.5 * (lo + hi);
        for k in i..=j {
            avg_rank[idx[k]] = mean_rank;
        }
        i = j + 1;
    }

    // Blom transform with offset 3/8.
    let denom = n as f64 + 0.25;
    avg_rank
        .iter()
        .map(|&r| norm_inv((r - 0.375) / denom))
        .collect()
}

/// One-covariate Cox partial-likelihood fit (Breslow ties) returning the MLE
/// coefficient and its standard error. The covariate is `z` (the scaled
/// rankits). Uses Newton-Raphson with a small step cap for stability.
fn cox_one_cov(data: &Dataset, z: &[f64], max_iter: usize, tol: f64) -> SurvivalResult<(f64, f64)> {
    let n = data.len();
    let order = data.order_by_time();

    let mut beta = 0.0_f64;
    let mut last_info = 1.0_f64;
    for _ in 0..max_iter {
        // Accumulate the partial-likelihood score U(β) and information I(β)
        // with a backward risk-set sweep (largest time first).
        let mut score = 0.0_f64;
        let mut info = 0.0_f64;
        // Risk-set running sums: Σ exp(βz), Σ z exp(βz), Σ z² exp(βz).
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;

        let mut k = n; // sweep from the end of the ascending-time order
        while k > 0 {
            // Process all subjects sharing the current (largest remaining) time.
            let t_idx = order[k - 1];
            let t = data.observations[t_idx].time;
            let mut events_z_sum = 0.0_f64;
            let mut n_events = 0.0_f64;
            // Add every subject with this time to the risk set, and tally events.
            while k > 0 {
                let oi = order[k - 1];
                if (data.observations[oi].time - t).abs() > 1.0e-12 {
                    break;
                }
                let zi = z[oi];
                let w = (beta * zi).exp();
                s0 += w;
                s1 += zi * w;
                s2 += zi * zi * w;
                if data.observations[oi].event {
                    events_z_sum += zi;
                    n_events += 1.0;
                }
                k -= 1;
            }
            if n_events > 0.0 && s0 > 0.0 {
                let mean = s1 / s0;
                let var = s2 / s0 - mean * mean;
                score += events_z_sum - n_events * mean;
                info += n_events * var.max(0.0);
            }
        }

        if info <= 1.0e-12 {
            // Information has collapsed. If we never moved off β = 0 there is
            // genuinely no prognostic variation (D = 0). Otherwise we have
            // reached a (near-)separable boundary; keep the current β rather
            // than discarding it, and stop.
            if beta == 0.0 {
                return Ok((0.0, f64::INFINITY));
            }
            break;
        }
        last_info = info;
        let mut step = score / info;
        // Cap the Newton step to avoid overshoot on near-separable data.
        let cap = 5.0;
        if step > cap {
            step = cap;
        } else if step < -cap {
            step = -cap;
        }
        beta += step;
        if step.abs() < tol {
            break;
        }
    }
    let se = (1.0 / last_info.max(1.0e-12)).sqrt();
    Ok((beta, se))
}

/// Compute the Royston-Sauerbrei `D` statistic and `R²_D` from a fitted model's
/// prognostic index.
///
/// `eta[i] = β'x_i` is the linear predictor for subject `i`. The dataset
/// supplies the event/time outcome. Requires at least 2 subjects and at least
/// one event.
pub fn royston_pseudo_r2(data: &Dataset, eta: &[f64]) -> SurvivalResult<PseudoR2Result> {
    let n = data.len();
    if n < 2 {
        return Err(SurvivalError::InvalidParameter(
            "need at least 2 subjects for D / R²_D".to_string(),
        ));
    }
    if eta.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![eta.len()],
        });
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    for &e in eta {
        if !e.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "non-finite prognostic index".to_string(),
            ));
        }
    }

    // Scaled rankits z_i = rankit_i / κ.
    let z: Vec<f64> = rankits(eta).iter().map(|&r| r / KAPPA).collect();

    // D is the one-covariate Cox coefficient on z.
    let (d_statistic, d_se) = cox_one_cov(data, &z, 100, 1.0e-8)?;

    // R²_D = D²σ² / (D²σ² + π²/6) with σ² = 8/π².
    let sigma2 = 8.0 / (std::f64::consts::PI * std::f64::consts::PI);
    let pi2_over_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    let num = d_statistic * d_statistic * sigma2;
    let r2_d = num / (num + pi2_over_6);

    Ok(PseudoR2Result {
        d_statistic,
        r2_d,
        d_se,
        n,
    })
}

/// Convert a `D` statistic directly to `R²_D` (closed-form), without refitting.
#[must_use]
pub fn r2_d_from_d(d: f64) -> f64 {
    let sigma2 = 8.0 / (std::f64::consts::PI * std::f64::consts::PI);
    let pi2_over_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    let num = d * d * sigma2;
    num / (num + pi2_over_6)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn norm_inv_median_is_zero() {
        assert!(approx(norm_inv(0.5), 0.0, 1e-9));
    }

    #[test]
    fn norm_inv_symmetric() {
        assert!(approx(norm_inv(0.975) + norm_inv(0.025), 0.0, 1e-6));
        assert!(approx(norm_inv(0.975), 1.959_964, 1e-4));
    }

    #[test]
    fn rankits_are_centered_and_ordered() {
        let eta = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let r = rankits(&eta);
        // Sum of rankits ≈ 0 (symmetric Blom scores).
        let s: f64 = r.iter().sum();
        assert!(s.abs() < 1e-6, "rankit sum {s}");
        // Order preserved: smallest eta → smallest rankit.
        assert!(r[0] < r[3] && r[3] < r[1] && r[1] < r[4] && r[4] < r[2]);
    }

    #[test]
    fn rankits_handle_ties_with_average_rank() {
        let eta = vec![1.0, 1.0, 2.0, 2.0];
        let r = rankits(&eta);
        // Tied pairs share the same rankit.
        assert!(approx(r[0], r[1], 1e-12));
        assert!(approx(r[2], r[3], 1e-12));
        assert!(r[0] < r[2]);
    }

    #[test]
    fn r2_d_from_d_zero_is_zero() {
        assert!(approx(r2_d_from_d(0.0), 0.0, 1e-12));
    }

    #[test]
    fn r2_d_monotone_in_d() {
        assert!(r2_d_from_d(2.0) > r2_d_from_d(1.0));
        assert!(r2_d_from_d(1.0) > r2_d_from_d(0.5));
        assert!(r2_d_from_d(5.0) < 1.0);
    }

    #[test]
    fn no_discrimination_gives_small_d() {
        // Random-looking PI uncorrelated with outcome ordering.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true; 6];
        // PI constant → no separation → D = 0, R²_D = 0.
        let eta = vec![0.0; 6];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let res = royston_pseudo_r2(&d, &eta).expect("ok");
        assert!(res.d_statistic.abs() < 1e-6, "D = {}", res.d_statistic);
        assert!(res.r2_d < 1e-6);
    }

    #[test]
    fn strong_discrimination_gives_positive_d_and_r2() {
        // PI is informative (subjects failing earlier tend to have higher PI)
        // but with some noise so the partial-likelihood MLE stays finite. D
        // should be clearly positive and R²_D substantial but below 1.
        let times = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let events = vec![true; 12];
        // Mostly decreasing in time, with two local swaps to avoid separability.
        let eta = vec![
            6.0, 4.0, 5.0, 3.0, 2.5, 1.0, 1.5, 0.0, -1.0, -0.5, -2.0, -3.0,
        ];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let res = royston_pseudo_r2(&d, &eta).expect("ok");
        assert!(res.d_statistic > 0.5, "D = {}", res.d_statistic);
        assert!(res.r2_d > 0.1 && res.r2_d < 1.0, "R²_D = {}", res.r2_d);
        assert!(res.d_se.is_finite() && res.d_se > 0.0, "se = {}", res.d_se);
    }

    #[test]
    fn r2_d_matches_closed_form() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, false, true, true];
        let eta = vec![2.0, 1.0, 0.5, -0.5, -1.0];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let res = royston_pseudo_r2(&d, &eta).expect("ok");
        let recomputed = r2_d_from_d(res.d_statistic);
        assert!(approx(res.r2_d, recomputed, 1e-12));
    }

    #[test]
    fn confidence_interval_brackets_d() {
        // Non-separable informative PI → finite se → a proper CI.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true; 8];
        let eta = vec![4.0, 3.0, 3.5, 1.0, 1.5, 0.0, -1.0, -2.0];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let res = royston_pseudo_r2(&d, &eta).expect("ok");
        assert!(res.d_se.is_finite());
        let (lo, hi) = res.d_confidence_interval(1.96);
        assert!(lo < res.d_statistic && res.d_statistic < hi);
    }

    #[test]
    fn sign_of_d_follows_pi_direction() {
        // Flipping the PI sign flips the sign of D. Non-separable data keeps the
        // MLE finite so the sign comparison is meaningful.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true; 8];
        let eta_pos = vec![4.0, 3.0, 3.5, 1.0, 1.5, 0.0, -1.0, -2.0];
        let eta_neg: Vec<f64> = eta_pos.iter().map(|&v| -v).collect();
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let rp = royston_pseudo_r2(&d, &eta_pos).expect("ok");
        let rn = royston_pseudo_r2(&d, &eta_neg).expect("ok");
        assert!(rp.d_statistic > 0.0 && rn.d_statistic < 0.0);
        // R²_D is symmetric in sign of D.
        assert!(approx(rp.r2_d, rn.r2_d, 1e-9));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true, true, true]).expect("ok");
        let res = royston_pseudo_r2(&d, &[1.0, 2.0]);
        assert!(matches!(res, Err(SurvivalError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_no_events() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[false, false]).expect("ok");
        let res = royston_pseudo_r2(&d, &[1.0, 2.0]);
        assert!(matches!(res, Err(SurvivalError::NoEvents)));
    }

    #[test]
    fn rejects_single_subject() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        let res = royston_pseudo_r2(&d, &[1.0]);
        assert!(matches!(res, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn rejects_non_finite_eta() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let res = royston_pseudo_r2(&d, &[1.0, f64::NAN]);
        assert!(matches!(res, Err(SurvivalError::NumericalInstability(_))));
    }
}
