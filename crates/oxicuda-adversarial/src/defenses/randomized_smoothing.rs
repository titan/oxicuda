//! Randomized smoothing — certified L2 robustness via Gaussian noise.
//!
//! Reference: Cohen, Rosenfeld & Kolter (2019),
//! *"Certified Adversarial Robustness via Randomized Smoothing"*, ICML.
//!
//! Given a base classifier `f : R^d → {1, …, K}` and a smoothing standard
//! deviation `σ`, the *smoothed classifier* is
//!
//! ```text
//! g(x) = argmax_c  Pr_{η ∼ N(0, σ² I_d)} [ f(x + η) = c ].
//! ```
//!
//! Cohen et al. (Theorem 1) prove that if class `c_A` is returned by `g(x)`
//! with probability at least `p_A`, then `g(x + δ) = c_A` for every
//! perturbation `δ` with `‖δ‖_2 < σ · Φ⁻¹(p_A)`.
//!
//! In practice we estimate `p_A` from `n` Monte-Carlo samples and use a
//! Clopper-Pearson **lower** bound `p_A^{lower}` to maintain a `(1 − α)`
//! one-sided confidence on the certificate. This module exports two
//! primitives:
//!
//! * [`smoothed_predict`] — Monte-Carlo voting over noisy inputs, returning
//!   `(top_class, n_top, n_runner_up)`.
//! * [`certified_radius`] — full Cohen-style certificate
//!   `r = σ · Φ⁻¹(p_A^{lower})`, returning `AttackFailedAll` when `p_A` is
//!   not provably above `1/2`.
//!
//! The Gaussian quantile `Φ⁻¹` is implemented via the Beasley–Springer–Moro
//! rational approximation (max abs error `≈ 1.15 e-9` over `(0, 1)`).

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for randomized smoothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RsConfig {
    /// Standard deviation of the smoothing noise. Must be finite and `>= 0`.
    /// Typical default: `0.25`.
    pub sigma: f32,
    /// Number of Monte-Carlo samples for prediction / certification.
    /// Must be `>= 1`. Typical default: `100_000`.
    pub n_samples: usize,
    /// One-sided failure probability for the Clopper-Pearson lower bound.
    /// Must satisfy `0 < α < 1`. Typical default: `0.001`.
    pub alpha: f32,
}

impl RsConfig {
    /// Build a new `RsConfig`.
    ///
    /// # Errors
    /// * [`AdvError::InvalidNoiseSigma`] if `sigma` is negative or non-finite.
    /// * [`AdvError::InsufficientCertSamples`] if `n_samples < 1`.
    /// * [`AdvError::InvalidConfidence`] if `alpha` is not strictly in `(0, 1)`.
    pub fn new(sigma: f32, n_samples: usize, alpha: f32) -> AdvResult<Self> {
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(AdvError::InvalidNoiseSigma { sigma });
        }
        if n_samples == 0 {
            return Err(AdvError::InsufficientCertSamples {
                min: 1,
                got: n_samples,
            });
        }
        if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return Err(AdvError::InvalidConfidence { alpha });
        }
        Ok(Self {
            sigma,
            n_samples,
            alpha,
        })
    }
}

impl Default for RsConfig {
    fn default() -> Self {
        Self {
            sigma: 0.25,
            n_samples: 100_000,
            alpha: 0.001,
        }
    }
}

// ─── Φ⁻¹ — inverse standard-normal CDF (Beasley–Springer–Moro) ──────────────

/// Beasley–Springer–Moro approximation of the inverse standard-normal CDF.
///
/// Returns `Φ⁻¹(p)` for `p ∈ (0, 1)`. Saturates at `±∞` for `p → 0` and
/// `p → 1`.
///
/// Maximum absolute error `≈ 1.15 e-9` over `(0, 1)` (Moro 1995).
pub(crate) fn inverse_normal_cdf(p: f64) -> f64 {
    // Coefficients for the central rational approximation (|p − 0.5| ≤ 0.425).
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    // Coefficients for the tail approximation (|p − 0.5| > 0.425).
    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let y = p - 0.5;
    if y.abs() < 0.425 {
        let r = y * y;
        let num = ((A[3] * r + A[2]) * r + A[1]) * r + A[0];
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        y * num / den
    } else {
        let r = if y < 0.0 { p } else { 1.0 - p };
        // log(-log(r)) substitution.
        let s = (-r.ln()).ln();
        let mut x = C[0];
        let mut t = 1.0;
        for &c in &C[1..] {
            t *= s;
            x += c * t;
        }
        if y < 0.0 { -x } else { x }
    }
}

// ─── Smoothed prediction ─────────────────────────────────────────────────────

/// Run `cfg.n_samples` noisy classifier evaluations and return the top class
/// vote count along with the runner-up count.
///
/// # Parameters
/// * `x`             — clean input, length `d`.
/// * `cfg`           — sigma / n_samples / alpha (alpha unused here).
/// * `rng`           — mutable RNG used to draw `N(0, σ² I_d)` noise.
/// * `base_classify` — closure returning the predicted class index for a
///   given noisy input.
///
/// # Returns
/// `(predicted_class, count_top, count_runner_up)`.
///
/// # Errors
/// * [`AdvError::EmptyInput`]     — empty input.
/// * [`AdvError::NanEncountered`] — non-finite input.
/// * Any error returned by `base_classify`.
pub fn smoothed_predict<F>(
    x: &[f32],
    cfg: &RsConfig,
    rng: &mut LcgRng,
    base_classify: F,
) -> AdvResult<(usize, usize, usize)>
where
    F: Fn(&[f32]) -> AdvResult<usize>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "smoothed_predict:x",
        });
    }

    // Sparse class-vote table (we don't know K in advance).
    let mut counts: Vec<(usize, usize)> = Vec::new();
    let mut noise = vec![0.0_f32; x.len()];
    let mut noisy = vec![0.0_f32; x.len()];

    for _ in 0..cfg.n_samples {
        if cfg.sigma > 0.0 {
            rng.fill_normal(&mut noise);
            for i in 0..x.len() {
                noisy[i] = x[i] + cfg.sigma * noise[i];
            }
        } else {
            noisy.copy_from_slice(x);
        }

        let cls = base_classify(&noisy)?;
        if let Some(slot) = counts.iter_mut().find(|(c, _)| *c == cls) {
            slot.1 += 1;
        } else {
            counts.push((cls, 1));
        }
    }

    // Sort descending by count for top-1 / top-2 retrieval.
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    let (top_class, top_count) = counts[0];
    let runner_up = counts.get(1).map(|(_, n)| *n).unwrap_or(0);
    Ok((top_class, top_count, runner_up))
}

// ─── Clopper-Pearson normal approximation ───────────────────────────────────

/// Normal-approximation lower bound on a binomial success probability:
///
/// ```text
/// p_lower = p_hat − z_alpha · sqrt(p_hat (1 − p_hat) / n)
/// ```
///
/// where `z_alpha = Φ⁻¹(1 − α)`. Returned value is clamped to
/// `[0, 1 − 1/(2n)]` — the upper cap mimics the proper Clopper-Pearson
/// lower bound at `k = n` and avoids the singularity of `Φ⁻¹(1)` in the
/// downstream radius formula.
pub(crate) fn clopper_pearson_lower(top_count: usize, n: usize, alpha: f32) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let p_hat = top_count as f64 / n_f;
    let z = inverse_normal_cdf(1.0 - alpha as f64);
    let se = (p_hat * (1.0 - p_hat) / n_f).sqrt();
    let lower = p_hat - z * se;
    // Cap strictly below 1 to keep Φ⁻¹(p_lower) finite: 1 − 1/(2n) matches
    // the median-unbiased Clopper-Pearson bound at k = n.
    let upper_cap = 1.0 - 0.5 / n_f;
    lower.clamp(0.0, upper_cap) as f32
}

// ─── Certified radius ───────────────────────────────────────────────────────

/// Cohen 2019 Theorem 1 certified L2 radius.
///
/// Draws `cfg.n_samples` noisy classifications, computes the Clopper-Pearson
/// lower bound on the top-class proportion, and returns
/// `r = σ · Φ⁻¹(p_lower)`. If the lower bound is below `0.5` (i.e. the
/// smoothed top-1 is not statistically significant) we return
/// [`AdvError::AttackFailedAll`] — the certificate is meaningful only when
/// `p_lower > 0.5`.
///
/// # Errors
/// All errors of [`smoothed_predict`], plus:
/// * [`AdvError::AttackFailedAll`] when the lower bound on the top class
///   proportion is `≤ 0.5`.
pub fn certified_radius<F>(
    x: &[f32],
    cfg: &RsConfig,
    rng: &mut LcgRng,
    base_classify: F,
) -> AdvResult<(usize, f32)>
where
    F: Fn(&[f32]) -> AdvResult<usize>,
{
    let (top_class, top_count, _runner_up) = smoothed_predict(x, cfg, rng, base_classify)?;
    let p_lower = clopper_pearson_lower(top_count, cfg.n_samples, cfg.alpha);
    if p_lower <= 0.5 {
        return Err(AdvError::AttackFailedAll);
    }
    let q = inverse_normal_cdf(p_lower as f64) as f32;
    let radius = cfg.sigma * q;
    if !radius.is_finite() {
        return Err(AdvError::NanEncountered {
            location: "certified_radius:radius",
        });
    }
    Ok((top_class, radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validates_parameters() {
        assert!(RsConfig::new(-0.1, 100, 0.001).is_err());
        assert!(RsConfig::new(f32::NAN, 100, 0.001).is_err());
        assert!(RsConfig::new(0.25, 0, 0.001).is_err());
        assert!(RsConfig::new(0.25, 100, 0.0).is_err());
        assert!(RsConfig::new(0.25, 100, 1.0).is_err());
        assert!(RsConfig::new(0.25, 100, -0.1).is_err());
        assert!(RsConfig::new(0.25, 100, 0.001).is_ok());
        let d = RsConfig::default();
        assert!((d.sigma - 0.25).abs() < 1e-6);
        assert_eq!(d.n_samples, 100_000);
    }

    #[test]
    fn inverse_normal_cdf_matches_known_quantiles() {
        // Φ⁻¹(0.5) = 0; Φ⁻¹(0.975) ≈ 1.95996; Φ⁻¹(0.999) ≈ 3.09023
        assert!(inverse_normal_cdf(0.5).abs() < 1e-7);
        let q975 = inverse_normal_cdf(0.975);
        assert!((q975 - 1.959_963_984_540_054).abs() < 1e-6);
        let q999 = inverse_normal_cdf(0.999);
        assert!((q999 - 3.090_232_306_167_813).abs() < 1e-6);
        // Symmetry: Φ⁻¹(1 − p) = −Φ⁻¹(p).
        let q_lo = inverse_normal_cdf(0.025);
        assert!((q_lo + 1.959_963_984_540_054).abs() < 1e-6);
    }

    #[test]
    fn inverse_normal_cdf_handles_extremes() {
        assert!(inverse_normal_cdf(0.0).is_infinite());
        assert!(inverse_normal_cdf(1.0).is_infinite());
        assert!(inverse_normal_cdf(0.0) < 0.0);
        assert!(inverse_normal_cdf(1.0) > 0.0);
    }

    #[test]
    fn deterministic_classifier_zero_sigma_full_vote() {
        // Constant classifier always returns class 7. With σ=0 there is no
        // noise; every sample votes 7.
        let x = vec![0.1_f32, 0.2, 0.3];
        let cfg = RsConfig::new(0.0, 200, 0.001).expect("cfg");
        let mut rng = LcgRng::new(42);
        let (cls, top, ru) =
            smoothed_predict(&x, &cfg, &mut rng, |_y| Ok(7_usize)).expect("predict");
        assert_eq!(cls, 7);
        assert_eq!(top, 200);
        assert_eq!(ru, 0);
    }

    #[test]
    fn constant_classifier_certifies_positive_radius() {
        // A constant classifier gives p_hat = 1.0 → p_lower close to 1 →
        // Φ⁻¹(p_lower) is finite and large but bounded by the binomial
        // approximation. We just need radius > 0 for sigma > 0.
        let x = vec![0.0_f32; 4];
        let cfg = RsConfig::new(0.5, 1000, 0.001).expect("cfg");
        let mut rng = LcgRng::new(0);
        let (cls, r) = certified_radius(&x, &cfg, &mut rng, |_y| Ok(3_usize)).expect("radius");
        assert_eq!(cls, 3);
        assert!(r.is_finite());
        assert!(r > 0.0, "radius must be positive when p_hat = 1");
    }

    #[test]
    fn ambiguous_classifier_fails_certification() {
        // A classifier that flips between two classes 50/50 leaves p_lower
        // well below 0.5 → AttackFailedAll.
        use std::cell::Cell;
        let x = vec![0.0_f32; 4];
        let cfg = RsConfig::new(0.5, 2000, 0.001).expect("cfg");
        let mut rng = LcgRng::new(0);
        let flip = Cell::new(0_usize);
        let res = certified_radius(&x, &cfg, &mut rng, |_y| {
            let v = flip.get();
            flip.set(v + 1);
            Ok(if v % 2 == 0 { 0_usize } else { 1_usize })
        });
        assert!(matches!(res, Err(AdvError::AttackFailedAll)));
    }

    #[test]
    fn classifier_error_propagates() {
        let x = vec![0.0_f32; 4];
        let cfg = RsConfig::new(0.25, 50, 0.001).expect("cfg");
        let mut rng = LcgRng::new(0);
        let res: AdvResult<_> = smoothed_predict(&x, &cfg, &mut rng, |_y| {
            Err(AdvError::Internal("classifier failed".into()))
        });
        assert!(matches!(res, Err(AdvError::Internal(_))));
    }

    #[test]
    fn empty_and_nan_input_rejected() {
        let cfg = RsConfig::new(0.25, 50, 0.001).expect("cfg");
        let mut rng = LcgRng::new(0);
        let empty: Vec<f32> = vec![];
        assert_eq!(
            smoothed_predict(&empty, &cfg, &mut rng, |_y| Ok(0_usize)).unwrap_err(),
            AdvError::EmptyInput
        );
        let nan = vec![f32::NAN, 0.0];
        assert!(matches!(
            smoothed_predict(&nan, &cfg, &mut rng, |_y| Ok(0_usize)).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn clopper_pearson_lower_bounds_p_hat() {
        // For p_hat = 1.0 the lower bound is capped at 1 − 1/(2n) < 1.0.
        let lb_full = clopper_pearson_lower(1000, 1000, 0.001);
        assert!(lb_full < 1.0 && lb_full > 0.99);
        let cap = 1.0 - 0.5 / 1000.0;
        assert!((lb_full - cap as f32).abs() < 1e-5);
        // For p_hat = 0.5 with α=0.001 the lower bound is well below 0.5.
        let lb_half = clopper_pearson_lower(500, 1000, 0.001);
        assert!(lb_half < 0.5);
        // For n=0 we return 0.
        assert_eq!(clopper_pearson_lower(0, 0, 0.001), 0.0);
    }

    #[test]
    fn certified_radius_scales_with_sigma() {
        // For the same constant classifier, a larger sigma yields a strictly
        // larger certified radius (p_lower is identical → r ∝ sigma).
        let x = vec![0.0_f32; 3];
        let cfg_a = RsConfig::new(0.25, 2000, 0.001).expect("a");
        let cfg_b = RsConfig::new(1.00, 2000, 0.001).expect("b");
        let mut rng_a = LcgRng::new(7);
        let mut rng_b = LcgRng::new(7);
        let (_, r_a) = certified_radius(&x, &cfg_a, &mut rng_a, |_y| Ok(0_usize)).expect("ra");
        let (_, r_b) = certified_radius(&x, &cfg_b, &mut rng_b, |_y| Ok(0_usize)).expect("rb");
        assert!(r_b > r_a + 1e-6);
        // Ratio should be approximately 4× (sigma ratio).
        let ratio = r_b / r_a;
        assert!((ratio - 4.0).abs() < 0.05, "ratio={ratio}");
    }
}
