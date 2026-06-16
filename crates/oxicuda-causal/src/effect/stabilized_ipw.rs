//! Stabilized inverse-probability weighting (IPW) with overlap trimming.
//!
//! The plain Horvitz–Thompson IPW estimator in [`crate::effect::ipw`] can have
//! very high variance when some propensity scores approach `0` or `1`. Two
//! standard remedies — both implemented here — are:
//!
//! * **Hájek (self-normalised / stabilised) weighting**: divide each arm's
//!   weighted outcome sum by the *sum of weights* in that arm rather than by
//!   `n`. This is the ratio estimator
//!
//!   ```text
//!   μ̂₁ = (Σ_i T_i Y_i / ê_i) / (Σ_i T_i / ê_i)
//!   μ̂₀ = (Σ_i (1−T_i) Y_i / (1−ê_i)) / (Σ_i (1−T_i) / (1−ê_i))
//!   ATE = μ̂₁ − μ̂₀
//!   ```
//!
//!   which is exactly unbiased for constant outcomes and is invariant to the
//!   weight scale, dramatically reducing variance versus Horvitz–Thompson.
//!
//! * **Stabilised weights** (Robins, Hernán & Brumback 2000): multiply the
//!   inverse-propensity weight by the *marginal* treatment probability,
//!   `sw_i = P(T=t_i) / P(T=t_i | X_i)`, which keeps the weights tightly
//!   distributed around 1.
//!
//! * **Overlap trimming** (Crump et al. 2009): discard units whose propensity
//!   falls outside `[trim, 1−trim]`, restoring positivity / common support
//!   before estimation. The estimand becomes the ATE on the trimmed
//!   (overlap) population.

use crate::error::{CausalError, CausalResult};

/// Configuration for stabilised IPW.
#[derive(Debug, Clone)]
pub struct StabilizedIpwConfig {
    /// Symmetric trimming threshold: units with `ê ∉ [trim, 1−trim]` are
    /// dropped. Use `0.0` to disable trimming. Must satisfy `0 ≤ trim < 0.5`.
    pub trim: f32,
    /// Use stabilised weights `P(T=t)/P(T=t|X)` (multiply IP weights by the
    /// marginal treatment probability). When `false`, plain `1/ê` weights are
    /// used inside the Hájek ratio.
    pub stabilize: bool,
    /// Hard clip applied to propensity scores before weighting, `[clip, 1−clip]`.
    /// Guards against division blow-ups even within the trimmed region.
    pub clip: f32,
}

impl Default for StabilizedIpwConfig {
    fn default() -> Self {
        Self {
            trim: 0.05,
            stabilize: true,
            clip: 0.01,
        }
    }
}

/// Result of stabilised IPW estimation.
#[derive(Debug, Clone)]
pub struct StabilizedIpwResult {
    /// Estimated ATE on the (possibly trimmed) population: `μ̂₁ − μ̂₀`.
    pub ate: f32,
    /// Hájek estimate of `E[Y(1)]` over the retained sample.
    pub mu1: f32,
    /// Hájek estimate of `E[Y(0)]` over the retained sample.
    pub mu0: f32,
    /// Number of units retained after trimming.
    pub n_retained: usize,
    /// Number of units trimmed (outside the overlap region).
    pub n_trimmed: usize,
    /// Effective sample size of the treated arm `(Σw)² / Σw²` (Kish ESS).
    pub ess_treated: f32,
    /// Effective sample size of the control arm.
    pub ess_control: f32,
}

/// Stabilised, self-normalised (Hájek) IPW with optional overlap trimming.
///
/// # Arguments
/// * `y` — outcomes, length `n`.
/// * `t` — binary treatment (`1.0` / `0.0`), length `n`.
/// * `propensity` — estimated `ê(x_i) ∈ (0,1)`, length `n`.
/// * `config` — estimation configuration.
///
/// # Errors
/// Returns [`CausalError::EmptyInput`] for empty inputs,
/// [`CausalError::DimensionMismatch`] on length mismatch,
/// [`CausalError::InvalidParameter`] for an out-of-range `trim`, and
/// [`CausalError::NotFitted`] if after trimming either arm has no units or a
/// zero total weight.
pub fn stabilized_ipw(
    y: &[f32],
    t: &[f32],
    propensity: &[f32],
    config: &StabilizedIpwConfig,
) -> CausalResult<StabilizedIpwResult> {
    let n = y.len();
    if n == 0 {
        return Err(CausalError::EmptyInput);
    }
    if t.len() != n || propensity.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len().min(propensity.len()),
        });
    }
    if !config.trim.is_finite() || config.trim < 0.0 || config.trim >= 0.5 {
        return Err(CausalError::InvalidParameter {
            reason: format!("trim must be in [0, 0.5), got {}", config.trim),
        });
    }

    let lo = config.trim;
    let hi = 1.0 - config.trim;
    let clip_lo = config.clip.clamp(1e-6, 0.49);
    let clip_hi = 1.0 - clip_lo;

    // Marginal treatment probability P(T=1) over the *retained* sample is
    // computed in a first pass (needed for stabilised weights).
    let retained: Vec<usize> = (0..n)
        .filter(|&i| {
            let p = propensity[i];
            p >= lo && p <= hi
        })
        .collect();
    if retained.is_empty() {
        return Err(CausalError::NotFitted);
    }
    let n_treat_retained = retained.iter().filter(|&&i| t[i] > 0.5).count();
    let n_ctrl_retained = retained.len() - n_treat_retained;
    if n_treat_retained == 0 || n_ctrl_retained == 0 {
        return Err(CausalError::NotFitted);
    }
    let p_treat = n_treat_retained as f32 / retained.len() as f32;
    let p_ctrl = 1.0 - p_treat;

    // Accumulate Hájek numerators / denominators per arm.
    let mut num1 = 0.0_f32; // Σ w_i Y_i (treated)
    let mut den1 = 0.0_f32; // Σ w_i      (treated)
    let mut sw1 = 0.0_f32; // Σ w_i²     (treated, for ESS)
    let mut num0 = 0.0_f32;
    let mut den0 = 0.0_f32;
    let mut sw0 = 0.0_f32;

    for &i in &retained {
        let p = propensity[i].clamp(clip_lo, clip_hi);
        if t[i] > 0.5 {
            let mut w = 1.0 / p;
            if config.stabilize {
                w *= p_treat;
            }
            num1 += w * y[i];
            den1 += w;
            sw1 += w * w;
        } else {
            let mut w = 1.0 / (1.0 - p);
            if config.stabilize {
                w *= p_ctrl;
            }
            num0 += w * y[i];
            den0 += w;
            sw0 += w * w;
        }
    }

    if den1 <= 0.0 || den0 <= 0.0 {
        return Err(CausalError::NotFitted);
    }

    let mu1 = num1 / den1;
    let mu0 = num0 / den0;
    let ate = mu1 - mu0;

    let ess_treated = if sw1 > 0.0 { den1 * den1 / sw1 } else { 0.0 };
    let ess_control = if sw0 > 0.0 { den0 * den0 / sw0 } else { 0.0 };

    Ok(StabilizedIpwResult {
        ate,
        mu1,
        mu0,
        n_retained: retained.len(),
        n_trimmed: n - retained.len(),
        ess_treated,
        ess_control,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Randomised-treatment data: propensity ≈ 0.5, outcome `y = base + tau·T`.
    /// With true randomisation IPW is unbiased for `tau`.
    fn randomized_data(n: usize, tau: f32, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut p = vec![0.0_f32; n];
        for i in 0..n {
            let treated = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
            t[i] = treated;
            p[i] = 0.5;
            y[i] = 1.0 + 0.1 * rng.next_normal() + tau * treated;
        }
        (y, t, p)
    }

    #[test]
    fn sipw_recovers_randomized_effect() {
        let mut rng = LcgRng::new(1);
        let (y, t, p) = randomized_data(500, 2.0, &mut rng);
        let r = stabilized_ipw(&y, &t, &p, &StabilizedIpwConfig::default()).expect("ok");
        assert!(
            (r.ate - 2.0).abs() < 0.3,
            "ATE {} should be near 2.0",
            r.ate
        );
    }

    #[test]
    fn sipw_ate_finite() {
        let mut rng = LcgRng::new(2);
        let (y, t, p) = randomized_data(100, 1.0, &mut rng);
        let r = stabilized_ipw(&y, &t, &p, &StabilizedIpwConfig::default()).expect("ok");
        assert!(r.ate.is_finite() && r.mu0.is_finite() && r.mu1.is_finite());
    }

    #[test]
    fn sipw_constant_outcome_zero_ate() {
        // Hájek estimator is exact for constant outcomes regardless of weights.
        let n = 50;
        let y = vec![3.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut p = vec![0.0_f32; n];
        for i in 0..n {
            t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            p[i] = 0.3 + 0.4 * (i as f32 / n as f32);
        }
        let cfg = StabilizedIpwConfig {
            trim: 0.0,
            ..StabilizedIpwConfig::default()
        };
        let r = stabilized_ipw(&y, &t, &p, &cfg).expect("ok");
        assert!(
            r.ate.abs() < 1e-4,
            "constant outcome ATE {} should be 0",
            r.ate
        );
        assert!((r.mu1 - 3.0).abs() < 1e-4);
        assert!((r.mu0 - 3.0).abs() < 1e-4);
    }

    #[test]
    fn sipw_trimming_drops_extremes() {
        let n = 100;
        let mut y = vec![0.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut p = vec![0.0_f32; n];
        for i in 0..n {
            t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            // First 10 units have extreme propensities.
            p[i] = if i < 5 {
                0.01
            } else if i < 10 {
                0.99
            } else {
                0.5
            };
            y[i] = 1.0 + t[i];
        }
        let cfg = StabilizedIpwConfig {
            trim: 0.05,
            ..StabilizedIpwConfig::default()
        };
        let r = stabilized_ipw(&y, &t, &p, &cfg).expect("ok");
        assert_eq!(r.n_trimmed, 10, "should trim the 10 extreme units");
        assert_eq!(r.n_retained, n - 10);
    }

    #[test]
    fn sipw_no_trim_retains_all() {
        let mut rng = LcgRng::new(3);
        let (y, t, p) = randomized_data(80, 1.0, &mut rng);
        let cfg = StabilizedIpwConfig {
            trim: 0.0,
            ..StabilizedIpwConfig::default()
        };
        let r = stabilized_ipw(&y, &t, &p, &cfg).expect("ok");
        assert_eq!(r.n_retained, 80);
        assert_eq!(r.n_trimmed, 0);
    }

    #[test]
    fn sipw_ess_not_exceeding_count() {
        let mut rng = LcgRng::new(4);
        let (y, t, p) = randomized_data(200, 1.5, &mut rng);
        let r = stabilized_ipw(&y, &t, &p, &StabilizedIpwConfig::default()).expect("ok");
        let n_treat = t.iter().filter(|&&v| v > 0.5).count() as f32;
        let n_ctrl = t.iter().filter(|&&v| v <= 0.5).count() as f32;
        assert!(
            r.ess_treated <= n_treat + 1e-3,
            "ESS treated {} > {}",
            r.ess_treated,
            n_treat
        );
        assert!(
            r.ess_control <= n_ctrl + 1e-3,
            "ESS control {} > {}",
            r.ess_control,
            n_ctrl
        );
        assert!(r.ess_treated > 0.0 && r.ess_control > 0.0);
    }

    #[test]
    fn sipw_empty_errors() {
        let cfg = StabilizedIpwConfig::default();
        assert!(matches!(
            stabilized_ipw(&[], &[], &[], &cfg),
            Err(CausalError::EmptyInput)
        ));
    }

    #[test]
    fn sipw_length_mismatch_errors() {
        let cfg = StabilizedIpwConfig::default();
        let r = stabilized_ipw(&[1.0, 2.0], &[1.0, 0.0], &[0.5], &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn sipw_invalid_trim_errors() {
        let y = vec![1.0, 2.0];
        let t = vec![1.0, 0.0];
        let p = vec![0.5, 0.5];
        let cfg = StabilizedIpwConfig {
            trim: 0.6,
            ..StabilizedIpwConfig::default()
        };
        assert!(matches!(
            stabilized_ipw(&y, &t, &p, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn sipw_all_trimmed_errors() {
        // All units have extreme propensity → trimming leaves nothing.
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let t = vec![1.0, 0.0, 1.0, 0.0];
        let p = vec![0.01, 0.99, 0.005, 0.995];
        let cfg = StabilizedIpwConfig {
            trim: 0.1,
            ..StabilizedIpwConfig::default()
        };
        assert!(matches!(
            stabilized_ipw(&y, &t, &p, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn sipw_one_arm_missing_errors() {
        // Only treated units retained → no control arm.
        let y = vec![1.0, 2.0, 3.0];
        let t = vec![1.0, 1.0, 1.0];
        let p = vec![0.5, 0.5, 0.5];
        let cfg = StabilizedIpwConfig {
            trim: 0.0,
            ..StabilizedIpwConfig::default()
        };
        assert!(matches!(
            stabilized_ipw(&y, &t, &p, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn sipw_stabilize_vs_unstabilized_both_run() {
        let mut rng = LcgRng::new(5);
        let (y, t, p) = randomized_data(150, 2.0, &mut rng);
        let cfg_s = StabilizedIpwConfig {
            stabilize: true,
            ..StabilizedIpwConfig::default()
        };
        let cfg_u = StabilizedIpwConfig {
            stabilize: false,
            ..StabilizedIpwConfig::default()
        };
        let rs = stabilized_ipw(&y, &t, &p, &cfg_s).expect("ok");
        let ru = stabilized_ipw(&y, &t, &p, &cfg_u).expect("ok");
        // The Hájek ratio is scale-invariant, so both give the same ATE.
        assert!((rs.ate - ru.ate).abs() < 1e-3, "{} vs {}", rs.ate, ru.ate);
    }

    #[test]
    fn sipw_retained_plus_trimmed_equals_n() {
        let n = 120;
        let mut y = vec![0.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut p = vec![0.0_f32; n];
        let mut rng = LcgRng::new(6);
        for i in 0..n {
            t[i] = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
            p[i] = 0.02 + 0.96 * rng.next_f32();
            y[i] = 1.0 + t[i];
        }
        let r = stabilized_ipw(&y, &t, &p, &StabilizedIpwConfig::default()).expect("ok");
        assert_eq!(r.n_retained + r.n_trimmed, n);
    }

    #[test]
    fn sipw_default_config() {
        let cfg = StabilizedIpwConfig::default();
        assert_eq!(cfg.trim, 0.05);
        assert!(cfg.stabilize);
        assert_eq!(cfg.clip, 0.01);
    }

    #[test]
    fn sipw_reduces_bias_under_confounding() {
        // Confounded selection: treated tend to have higher baseline. Stabilised
        // IPW with the correct propensity recovers tau better than naive means.
        let mut rng = LcgRng::new(7);
        let n = 600;
        let mut y = vec![0.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut p = vec![0.0_f32; n];
        let tau = 2.0;
        for i in 0..n {
            let u = rng.next_f32(); // confounder in [0,1]
            let prop = (0.2 + 0.6 * u).clamp(0.05, 0.95);
            let treated = if rng.next_f32() < prop { 1.0 } else { 0.0 };
            p[i] = prop;
            t[i] = treated;
            y[i] = 3.0 * u + tau * treated + 0.1 * rng.next_normal();
        }
        let r = stabilized_ipw(&y, &t, &p, &StabilizedIpwConfig::default()).expect("ok");
        let naive = {
            let ty: f32 = (0..n).filter(|&i| t[i] > 0.5).map(|i| y[i]).sum();
            let tn = (0..n).filter(|&i| t[i] > 0.5).count() as f32;
            let cy: f32 = (0..n).filter(|&i| t[i] <= 0.5).map(|i| y[i]).sum();
            let cn = (0..n).filter(|&i| t[i] <= 0.5).count() as f32;
            ty / tn - cy / cn
        };
        assert!(
            (r.ate - tau).abs() < (naive - tau).abs(),
            "stabilised IPW ATE {} not closer to {tau} than naive {naive}",
            r.ate
        );
    }
}
