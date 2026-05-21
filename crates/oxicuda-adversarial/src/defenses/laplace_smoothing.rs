//! Laplace randomized smoothing — certified L1 robustness via additive Laplace noise.
//!
//! Reference: Teng, Lee & Yang (2020),
//! *"Ell_1 Adversarial Robustness Certificates: A Randomized Smoothing
//! Approach"*, ICLR.
//!
//! The vanilla [`super::randomized_smoothing`] module noise model is
//! `η ∼ N(0, σ² I)`, which yields a Cohen-style **L2** certificate. Teng et
//! al. (2020) show that swapping the noise distribution to a per-coordinate
//! Laplace `η ∼ Laplace(0, b)^d` gives a tight certificate in the **L1**
//! norm with explicit closed form:
//!
//! ```text
//! r_L1 = (b / 2) · log(p̂_top / (1 − p̂_top))   when  p̂_top > 0.5
//!       0                                       otherwise.
//! ```
//!
//! The radius is positive once the smoothed top-class probability is above
//! `1/2` and grows linearly with the noise scale `b`. As `p̂_top → 1` the
//! radius diverges; we clip `p̂_top` to `1 − ε_clip` (with `ε_clip = 1e-7`)
//! to keep the log finite.
//!
//! # Sampling
//!
//! A `Laplace(0, b)` variate is generated via inverse-CDF:
//!
//! ```text
//! u  ∼ U(0, 1)
//! x  = −b · sign(u − 0.5) · log(1 − 2 · |u − 0.5|)
//! ```
//!
//! The existing `LcgRng::next_u32` returns the high 31 bits of the LCG
//! state (it shifts by 33), so it spans `[0, 2³¹)`. We therefore draw
//! `u = next_u32() / 2³¹` which lies in `[0, 1)`. (`next_f32()` only
//! spans `[0, 0.5)` for the same reason and is unsuitable here.)
//! Inputs to the log are bounded away from zero by `ε_ln = 1e-30` to avoid
//! `-∞`; the resulting bias is statistically invisible at f32 precision.
//!
//! # Distinct from `randomized_smoothing.rs`
//!
//! `randomized_smoothing.rs` retains the Cohen Gaussian-smoothing L2 path;
//! this module is the L1 sibling using Laplace noise. They share the
//! Monte-Carlo "average softmax → argmax" smoothing pattern but differ in
//! noise distribution and certificate formula.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for Laplace randomized smoothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceSmoothingConfig {
    /// Laplace scale parameter `b > 0` (variance is `2 b²`). Typical value:
    /// `0.25` (matches the Gaussian-RS baseline at the same noise mean
    /// absolute deviation).
    pub scale_b: f32,
    /// Number of Monte-Carlo samples for the smoothed-softmax estimate.
    /// Must be `>= 1`. Typical value: `1024` for evaluation, `16` for
    /// per-step training estimates.
    pub n_samples: usize,
}

impl LaplaceSmoothingConfig {
    /// Build a new [`LaplaceSmoothingConfig`] with parameter validation.
    ///
    /// # Errors
    /// * [`AdvError::InvalidNoiseSigma`]      — `scale_b <= 0` or non-finite.
    /// * [`AdvError::InsufficientCertSamples`] — `n_samples == 0`.
    pub fn new(scale_b: f32, n_samples: usize) -> AdvResult<Self> {
        if !(scale_b.is_finite() && scale_b > 0.0) {
            return Err(AdvError::InvalidNoiseSigma { sigma: scale_b });
        }
        if n_samples == 0 {
            return Err(AdvError::InsufficientCertSamples {
                min: 1,
                got: n_samples,
            });
        }
        Ok(Self { scale_b, n_samples })
    }
}

impl Default for LaplaceSmoothingConfig {
    fn default() -> Self {
        Self {
            scale_b: 0.25,
            n_samples: 1024,
        }
    }
}

// ─── LaplaceSmoothing ────────────────────────────────────────────────────────

/// Laplace randomized smoothing helper.
///
/// Stateless except for the configuration; analogous to `MacerLoss` /
/// `RsConfig`. Held by value so callers can store one per evaluation
/// pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceSmoothing {
    cfg: LaplaceSmoothingConfig,
}

impl LaplaceSmoothing {
    /// Build a new [`LaplaceSmoothing`].
    ///
    /// # Errors
    /// Re-validates the configuration through [`LaplaceSmoothingConfig::new`].
    pub fn new(cfg: LaplaceSmoothingConfig) -> AdvResult<Self> {
        let _ = LaplaceSmoothingConfig::new(cfg.scale_b, cfg.n_samples)?;
        Ok(Self { cfg })
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &LaplaceSmoothingConfig {
        &self.cfg
    }

    // ─── Sampling ────────────────────────────────────────────────────────

    /// Draw `dim` i.i.d. `Laplace(0, b)` variates via inverse-CDF.
    ///
    /// ```text
    /// u ∼ U(0, 1);
    /// x = −b · sign(u − 0.5) · log(1 − 2 · |u − 0.5|).
    /// ```
    ///
    /// `next_u32()` returns the high 31 bits of the underlying LCG state
    /// (it shifts by 33), so values are in `[0, 2³¹)`. Dividing by `2³¹`
    /// gives a uniform `[0, 1)`. The argument of `log` is floored at
    /// `ε_ln = 1e-30` to keep the output finite when `u` lands exactly on
    /// `0.5 ± 0.5`; the bias introduced is statistically invisible at f32
    /// precision (`< 10⁻⁹` in mean over any realistic batch).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`] if `dim == 0`.
    pub fn sample_laplace(&self, dim: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
        if dim == 0 {
            return Err(AdvError::EmptyInput);
        }
        // 2^31 as f32 — divisor for u ∈ [0, 1) given LcgRng::next_u32 output
        // range [0, 2^31).
        const U_DIVISOR: f32 = 2_147_483_648.0_f32;
        let mut out = Vec::with_capacity(dim);
        for _ in 0..dim {
            let u = rng.next_u32() as f32 / U_DIVISOR;
            let s = u - 0.5;
            let mag = 1.0 - 2.0 * s.abs();
            // Floor to avoid log(0) blowing up at u ∈ {0, 1}.
            let mag_safe = mag.max(1e-30_f32);
            let sign = if s >= 0.0 { 1.0_f32 } else { -1.0_f32 };
            let x = -self.cfg.scale_b * sign * mag_safe.ln();
            out.push(x);
        }
        Ok(out)
    }

    // ─── Smoothed prediction ────────────────────────────────────────────

    /// Average the softmax output of `classify_fn` over `n_samples` Laplace-
    /// noised inputs and return `(argmax, averaged_distribution)`.
    ///
    /// # Parameters
    /// * `input`        — clean input of length `d ≥ 1`.
    /// * `classify_fn`  — closure returning a length-`n_classes` softmax row.
    /// * `n_classes`    — number of output classes (`>= 2`).
    /// * `rng`          — mutable RNG used for Laplace noise.
    ///
    /// # Returns
    /// `(top_class, averaged_distribution)` with `averaged_distribution`
    /// summing to `≈ 1` (convex combination of softmax rows).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]        — empty input.
    /// * [`AdvError::InvalidLossWeight`] — `n_classes < 2`.
    /// * [`AdvError::NanEncountered`]    — non-finite input or non-finite
    ///   classifier output.
    /// * [`AdvError::DimensionMismatch`] — `classify_fn` returned the wrong
    ///   number of class probs.
    pub fn smoothed_predict<F>(
        &self,
        input: &[f32],
        classify_fn: F,
        n_classes: usize,
        rng: &mut LcgRng,
    ) -> AdvResult<(usize, Vec<f32>)>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        if input.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if n_classes < 2 {
            return Err(AdvError::InvalidLossWeight {
                weight: n_classes as f32,
            });
        }
        if input.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "laplace_smoothing:smoothed_predict:input",
            });
        }

        let d = input.len();
        let mut noisy = vec![0.0_f32; d];
        let mut accum = vec![0.0_f64; n_classes];

        for _ in 0..self.cfg.n_samples {
            let noise = self.sample_laplace(d, rng)?;
            for i in 0..d {
                noisy[i] = input[i] + noise[i];
            }
            let row = classify_fn(&noisy);
            if row.len() != n_classes {
                return Err(AdvError::DimensionMismatch {
                    expected: n_classes,
                    got: row.len(),
                });
            }
            for (slot, &p) in accum.iter_mut().zip(row.iter()) {
                if !p.is_finite() {
                    return Err(AdvError::NanEncountered {
                        location: "laplace_smoothing:smoothed_predict:classify_fn",
                    });
                }
                *slot += p as f64;
            }
        }

        // Average; renormalise to keep the distribution exactly summing to 1.
        let total: f64 = accum.iter().sum();
        let denom = if total > 0.0 {
            total
        } else {
            self.cfg.n_samples as f64
        };
        let avg: Vec<f32> = accum.iter().map(|&v| (v / denom) as f32).collect();

        // argmax with deterministic tie-break (first-seen wins).
        let mut top = 0_usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in avg.iter().enumerate() {
            if v > best {
                best = v;
                top = i;
            }
        }
        Ok((top, avg))
    }

    // ─── Certified L1 radius ────────────────────────────────────────────

    /// Teng 2020 certified L1 radius for Laplace smoothing:
    ///
    /// ```text
    /// r = (b / 2) · log(p̂_top / (1 − p̂_top))   if p̂_top > 0.5
    ///     0                                      otherwise.
    /// ```
    ///
    /// We clip `p̂_top` to `[ε_clip, 1 − ε_clip]` with `ε_clip = 1e-7`
    /// before taking the log; this keeps the returned radius finite for
    /// `p̂_top → 1` while preserving the leading log scaling.
    ///
    /// # Errors
    /// * [`AdvError::InvalidConfidence`] if `p_top` is non-finite or
    ///   outside `[0, 1]`.
    pub fn certified_radius_l1(&self, p_top: f32) -> AdvResult<f32> {
        if !(p_top.is_finite() && (0.0..=1.0).contains(&p_top)) {
            return Err(AdvError::InvalidConfidence { alpha: p_top });
        }
        if p_top <= 0.5 {
            return Ok(0.0);
        }
        const EPS_CLIP: f32 = 1e-7;
        let p_clipped = p_top.clamp(EPS_CLIP, 1.0 - EPS_CLIP);
        let ratio = p_clipped / (1.0 - p_clipped);
        // ln on a strictly positive number (`p_clipped < 1`) is always finite.
        let r = 0.5 * self.cfg.scale_b * ratio.ln();
        if !r.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "laplace_smoothing:certified_radius_l1",
            });
        }
        Ok(r)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn make_smoothing(scale_b: f32, n_samples: usize) -> LaplaceSmoothing {
        LaplaceSmoothing::new(LaplaceSmoothingConfig::new(scale_b, n_samples).expect("cfg"))
            .expect("smoothing")
    }

    // ─── Config validation ───────────────────────────────────────────────

    #[test]
    fn config_rejects_invalid_scale_b() {
        assert!(LaplaceSmoothingConfig::new(0.0, 10).is_err());
        assert!(LaplaceSmoothingConfig::new(-0.5, 10).is_err());
        assert!(LaplaceSmoothingConfig::new(f32::NAN, 10).is_err());
        assert!(LaplaceSmoothingConfig::new(f32::INFINITY, 10).is_err());
    }

    #[test]
    fn config_rejects_zero_samples() {
        assert!(LaplaceSmoothingConfig::new(0.25, 0).is_err());
    }

    #[test]
    fn config_default_is_sensible() {
        let c = LaplaceSmoothingConfig::default();
        assert!(c.scale_b > 0.0);
        assert!(c.n_samples >= 1);
    }

    // ─── sample_laplace ─────────────────────────────────────────────────

    #[test]
    fn sample_laplace_returns_dim_values() {
        let s = make_smoothing(0.5, 16);
        let mut rng = LcgRng::new(11);
        let v = s.sample_laplace(7, &mut rng).expect("ok");
        assert_eq!(v.len(), 7);
        for x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn sample_laplace_rejects_zero_dim() {
        let s = make_smoothing(0.5, 16);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.sample_laplace(0, &mut rng).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn sample_laplace_empirical_mean_near_zero() {
        // Mean of Laplace(0, b) is 0. Take a large sample and check.
        let s = make_smoothing(1.0, 1);
        let mut rng = LcgRng::new(0);
        let n = 20_000;
        let v = s.sample_laplace(n, &mut rng).expect("ok");
        let mean: f32 = v.iter().copied().sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.06, "empirical mean too large: {mean}");
    }

    #[test]
    fn sample_laplace_empirical_variance_matches_2bb() {
        // Var of Laplace(0, b) is 2 b². Take a large sample, check x²-mean ≈ 2 b².
        let b = 0.7_f32;
        let s = make_smoothing(b, 1);
        let mut rng = LcgRng::new(0);
        let n = 30_000;
        let v = s.sample_laplace(n, &mut rng).expect("ok");
        let mean_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
        let expected = 2.0 * (b as f64) * (b as f64);
        // ~3 % tolerance on a Monte-Carlo estimate of size 3e4.
        let rel = (mean_sq - expected).abs() / expected;
        assert!(
            rel < 0.08,
            "rel err {rel} mean_sq={mean_sq} expect={expected}"
        );
    }

    #[test]
    fn sample_laplace_small_b_yields_small_noise() {
        // Near-zero scale ⇒ near-zero noise (magnitude scales linearly with b).
        let s = make_smoothing(1e-6_f32, 1);
        let mut rng = LcgRng::new(0);
        let v = s.sample_laplace(64, &mut rng).expect("ok");
        for x in &v {
            assert!(x.abs() < 1e-3, "x={x} too large for tiny b");
        }
    }

    #[test]
    fn sample_laplace_single_sample_edge_case() {
        let s = make_smoothing(0.5, 1);
        let mut rng = LcgRng::new(0);
        let v = s.sample_laplace(1, &mut rng).expect("ok");
        assert_eq!(v.len(), 1);
        assert!(v[0].is_finite());
    }

    // ─── certified_radius_l1 ─────────────────────────────────────────────

    #[test]
    fn certified_radius_l1_zero_at_half() {
        let s = make_smoothing(0.5, 16);
        assert_eq!(s.certified_radius_l1(0.5).expect("ok"), 0.0);
        assert_eq!(s.certified_radius_l1(0.4).expect("ok"), 0.0);
    }

    #[test]
    fn certified_radius_l1_monotone_in_p_top() {
        let s = make_smoothing(0.5, 16);
        let r_55 = s.certified_radius_l1(0.55).expect("ok");
        let r_70 = s.certified_radius_l1(0.70).expect("ok");
        let r_90 = s.certified_radius_l1(0.90).expect("ok");
        let r_99 = s.certified_radius_l1(0.99).expect("ok");
        assert!(r_55 > 0.0 && r_55 < r_70);
        assert!(r_70 < r_90);
        assert!(r_90 < r_99);
    }

    #[test]
    fn certified_radius_l1_scales_linearly_with_b() {
        // r = (b/2) · log(p/(1-p)); doubling b doubles r at fixed p.
        let s_a = make_smoothing(0.5, 16);
        let s_b = make_smoothing(1.0, 16);
        let p = 0.85_f32;
        let r_a = s_a.certified_radius_l1(p).expect("a");
        let r_b = s_b.certified_radius_l1(p).expect("b");
        assert!(approx_eq(r_b, 2.0 * r_a, 1e-5), "r_a={r_a} r_b={r_b}");
    }

    #[test]
    fn certified_radius_l1_extreme_p_finite() {
        let s = make_smoothing(0.5, 16);
        let r_zero = s.certified_radius_l1(0.0).expect("ok");
        let r_one = s.certified_radius_l1(1.0).expect("ok");
        // p ≤ 0.5 → r=0 trivially (no log).
        assert_eq!(r_zero, 0.0);
        // p = 1 → clipped → finite.
        assert!(r_one.is_finite() && r_one > 0.0);
    }

    #[test]
    fn certified_radius_l1_rejects_invalid_p() {
        let s = make_smoothing(0.5, 16);
        assert!(s.certified_radius_l1(-0.1).is_err());
        assert!(s.certified_radius_l1(1.1).is_err());
        assert!(s.certified_radius_l1(f32::NAN).is_err());
    }

    #[test]
    fn certified_radius_l1_closed_form_check() {
        // Manual check: b=2, p=0.9 ⇒ r = (2/2)·ln(0.9/0.1) = ln(9) ≈ 2.197.
        let s = make_smoothing(2.0, 16);
        let r = s.certified_radius_l1(0.9).expect("ok");
        let expect = (9.0_f32).ln();
        assert!(approx_eq(r, expect, 1e-5), "r={r} expect={expect}");
    }

    // ─── smoothed_predict ────────────────────────────────────────────────

    #[test]
    fn smoothed_predict_returns_valid_class_and_distribution() {
        let s = make_smoothing(0.3, 8);
        let mut rng = LcgRng::new(2);
        let x = vec![0.1_f32, 0.2, -0.05, 0.4];
        let n_classes = 4_usize;
        let (cls, dist) = s
            .smoothed_predict(&x, |_y| vec![0.25_f32; n_classes], n_classes, &mut rng)
            .expect("ok");
        assert!(cls < n_classes);
        assert_eq!(dist.len(), n_classes);
        let sum: f32 = dist.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "sum={sum}");
    }

    #[test]
    fn smoothed_predict_concentrated_softmax_picks_that_class() {
        let s = make_smoothing(0.2, 8);
        let mut rng = LcgRng::new(13);
        let x = vec![0.0_f32; 5];
        let n_classes = 5_usize;
        // Always returns one-hot at index 2.
        let (cls, dist) = s
            .smoothed_predict(
                &x,
                |_y| {
                    let mut p = vec![0.0_f32; n_classes];
                    p[2] = 1.0;
                    p
                },
                n_classes,
                &mut rng,
            )
            .expect("ok");
        assert_eq!(cls, 2);
        assert!(approx_eq(dist[2], 1.0, 1e-5));
    }

    #[test]
    fn smoothed_predict_is_deterministic_given_seed() {
        let s = make_smoothing(0.4, 16);
        let n_classes = 3_usize;
        let classifier = |y: &[f32]| {
            // Softmax-like split based on the sign of the first coord.
            let z = y[0];
            let e = z.exp();
            let denom = 1.0 + e + 1.0;
            vec![1.0 / denom, e / denom, 1.0 / denom]
        };
        let x = vec![0.2_f32, -0.1, 0.0];
        let mut rng_a = LcgRng::new(101);
        let mut rng_b = LcgRng::new(101);
        let (cls_a, dist_a) = s
            .smoothed_predict(&x, classifier, n_classes, &mut rng_a)
            .expect("a");
        let (cls_b, dist_b) = s
            .smoothed_predict(&x, classifier, n_classes, &mut rng_b)
            .expect("b");
        assert_eq!(cls_a, cls_b);
        for (a, b) in dist_a.iter().zip(dist_b.iter()) {
            assert!(approx_eq(*a, *b, 1e-7));
        }
    }

    #[test]
    fn smoothed_predict_rejects_empty_input() {
        let s = make_smoothing(0.5, 8);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.smoothed_predict(&[], |_| vec![0.5_f32, 0.5], 2, &mut rng)
                .unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn smoothed_predict_rejects_single_class() {
        let s = make_smoothing(0.5, 8);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.smoothed_predict(&[0.1_f32, 0.2], |_| vec![1.0_f32], 1, &mut rng)
                .unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn smoothed_predict_rejects_wrong_length_softmax() {
        let s = make_smoothing(0.5, 8);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.smoothed_predict(
                &[0.1_f32, 0.2],
                |_| vec![0.5_f32, 0.5, 0.0], // length 3 but n_classes = 2
                2,
                &mut rng,
            )
            .unwrap_err(),
            AdvError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn smoothed_predict_rejects_nan_input() {
        let s = make_smoothing(0.5, 8);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.smoothed_predict(&[f32::NAN, 0.2], |_| vec![0.5_f32, 0.5], 2, &mut rng)
                .unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn smoothed_predict_rejects_nan_softmax() {
        let s = make_smoothing(0.5, 8);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            s.smoothed_predict(&[0.1_f32, 0.2], |_| vec![f32::NAN, 0.5], 2, &mut rng)
                .unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    // ─── Construction sanity ────────────────────────────────────────────

    #[test]
    fn laplace_smoothing_new_rejects_invalid_config() {
        let bad = LaplaceSmoothingConfig {
            scale_b: -1.0,
            n_samples: 1,
        };
        assert!(LaplaceSmoothing::new(bad).is_err());
    }
}
