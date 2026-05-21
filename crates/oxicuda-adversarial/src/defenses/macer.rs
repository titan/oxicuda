//! MACER / SmoothAdv — Maximise Certified Radius of a Gaussian-smoothed classifier.
//!
//! References:
//! * Salman, Li, Razenshteyn, Zhang, Zhang, Bubeck & Yang (2019),
//!   *"Provably Robust Deep Learning via Adversarially Trained Smoothed
//!   Classifiers"* (SmoothAdv), NeurIPS.
//! * Zhai, Dan, He, Zhang, Gong, Wang & Liu (2020),
//!   *"MACER: Attack-free and Scalable Robust Training via Maximizing
//!   Certified Radius"*, ICLR.
//!
//! # Smoothed classifier
//!
//! For a base classifier `f : R^d → Δ_K` returning class probabilities, the
//! Cohen (2019) smoothed classifier under additive Gaussian noise is
//!
//! ```text
//! g(x) = argmax_c  E_{η ∼ N(0, σ² I_d)} [ f_c(x + η) ].
//! ```
//!
//! Its certified L2 radius around `x` is `r = σ · Φ⁻¹(p̂_top)`, where
//! `p̂_top` is the smoothed-softmax probability of the top class (Cohen 2019
//! Theorem 1; `Φ⁻¹` is the standard normal inverse-CDF / probit).
//!
//! # MACER training objective
//!
//! MACER turns this radius into a *training* signal by adding a hinge term
//! on the radius to the usual classification loss:
//!
//! ```text
//! L = L_cls + λ · max(0,  γ − r)        with r = σ · Φ⁻¹(p̂_top).
//! ```
//!
//! When the (estimated) certified radius `r` exceeds the target `γ`, the
//! hinge vanishes — only the classification loss remains. Otherwise the
//! optimiser is rewarded for increasing `p̂_top` so that `Φ⁻¹(p̂_top)`
//! climbs above `γ / σ`.
//!
//! # Probit implementation
//!
//! `probit(p) = Φ⁻¹(p)` is implemented via the deterministic Acklam (2003)
//! rational approximation. It is accurate to ≈ 1.15·10⁻⁹ in the body
//! (`p ∈ [0.02425, 0.97575]`) and to ≈ 1.15·10⁻⁹ in the tails. The
//! existing `randomized_smoothing` module uses Beasley–Springer–Moro for
//! the same purpose; we use Acklam here following the MACER reference code.
//!
//! # Conventions
//!
//! * `classify_fn` receives a noisy input and returns a length-`n_classes`
//!   softmax distribution (rows of the smoothed expectation are averaged
//!   inside `smoothed_predict`).
//! * Noise is drawn coordinate-wise from `N(0, σ²)` using
//!   `LcgRng::next_normal_pair()`.
//! * `loss(cls_loss, p_top)` is the *outer* training-time loss; the caller
//!   supplies their own classification loss (e.g. cross-entropy).

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for the MACER training-time loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MacerConfig {
    /// Smoothing-noise standard deviation `σ` (must be `> 0` and finite).
    /// Typical value: `0.25` (matches the Cohen 2019 RS baseline).
    pub sigma: f32,
    /// Weight `λ` on the radius-hinge term (must be `>= 0` and finite).
    /// Typical value: `12.0` (Zhai et al. 2020).
    pub lambda_robust: f32,
    /// Target radius `γ` for the hinge (must be `>= 0` and finite).
    /// Typical value: `8.0 · σ` (covers ≈ Φ⁻¹(1 − 1e-15)).
    pub gamma_hinge: f32,
    /// Number of Gaussian samples used to estimate the smoothed softmax.
    /// Must be `>= 1`. Typical value: `16` (Zhai et al. 2020 uses 16–64).
    pub n_samples_smooth: usize,
}

impl MacerConfig {
    /// Build a new [`MacerConfig`] with parameter validation.
    ///
    /// # Errors
    /// * [`AdvError::InvalidNoiseSigma`]      — `sigma <= 0` or non-finite.
    /// * [`AdvError::InvalidLossWeight`]      — `lambda_robust` or `gamma_hinge`
    ///   non-finite / negative.
    /// * [`AdvError::InsufficientCertSamples`] — `n_samples_smooth == 0`.
    pub fn new(
        sigma: f32,
        lambda_robust: f32,
        gamma_hinge: f32,
        n_samples_smooth: usize,
    ) -> AdvResult<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(AdvError::InvalidNoiseSigma { sigma });
        }
        if !(lambda_robust.is_finite() && lambda_robust >= 0.0) {
            return Err(AdvError::InvalidLossWeight {
                weight: lambda_robust,
            });
        }
        if !(gamma_hinge.is_finite() && gamma_hinge >= 0.0) {
            return Err(AdvError::InvalidLossWeight {
                weight: gamma_hinge,
            });
        }
        if n_samples_smooth == 0 {
            return Err(AdvError::InsufficientCertSamples {
                min: 1,
                got: n_samples_smooth,
            });
        }
        Ok(Self {
            sigma,
            lambda_robust,
            gamma_hinge,
            n_samples_smooth,
        })
    }
}

impl Default for MacerConfig {
    fn default() -> Self {
        Self {
            sigma: 0.25,
            lambda_robust: 12.0,
            gamma_hinge: 2.0, // 8 · σ for σ = 0.25
            n_samples_smooth: 16,
        }
    }
}

// ─── MacerLoss ────────────────────────────────────────────────────────────────

/// MACER training-time loss helper.
///
/// Stateless except for the configuration; all heavy lifting goes through
/// the methods. Held by value so callers can store one per training step
/// (e.g. inside a Trainer struct).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MacerLoss {
    cfg: MacerConfig,
}

impl MacerLoss {
    /// Build a new [`MacerLoss`].
    ///
    /// # Errors
    /// Any error from [`MacerConfig::new`] re-validated through the stored
    /// configuration.
    pub fn new(cfg: MacerConfig) -> AdvResult<Self> {
        // Re-validate to guard against direct field construction by callers.
        let _ = MacerConfig::new(
            cfg.sigma,
            cfg.lambda_robust,
            cfg.gamma_hinge,
            cfg.n_samples_smooth,
        )?;
        Ok(Self { cfg })
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &MacerConfig {
        &self.cfg
    }

    // ─── Acklam probit (Φ⁻¹) ────────────────────────────────────────────────

    /// Standard-normal inverse CDF via Acklam (2003) rational approximation.
    ///
    /// Returns `Φ⁻¹(p)` for `p ∈ [0, 1]`. Inputs near the boundaries are
    /// clipped to avoid `±∞`:
    ///
    /// * `p ≤ ε`        → returns the value at `ε` (large negative).
    /// * `p ≥ 1 − ε`    → returns the value at `1 − ε` (large positive).
    ///
    /// where `ε = 1e-7` (consistent with f32 precision).
    ///
    /// Accuracy in the body (`p ∈ [0.02425, 0.97575]`) is ≈ 1.15·10⁻⁹
    /// (Acklam 2003). The tail branches use the log-of-log substitution
    /// `r = √(−2 ln(min(p, 1 − p)))` and are accurate to ≈ 1·10⁻⁸ in f32.
    ///
    /// # Errors
    /// * [`AdvError::InvalidConfidence`] if `p` is non-finite or outside
    ///   `[0, 1]`.
    pub fn probit(&self, p: f32) -> AdvResult<f32> {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            return Err(AdvError::InvalidConfidence { alpha: p });
        }
        Ok(acklam_probit(p as f64) as f32)
    }

    // ─── Smoothed prediction ───────────────────────────────────────────────

    /// Estimate the smoothed-softmax distribution at `input`.
    ///
    /// Draws `cfg.n_samples_smooth` Gaussian-perturbed inputs
    /// `input + σ · z`, `z ∼ N(0, I)`, averages the resulting softmax
    /// distributions, then returns `(argmax, averaged_distribution)`.
    ///
    /// The averaged distribution is itself a valid probability vector
    /// (convex combination of softmax outputs).
    ///
    /// # Parameters
    /// * `input`        — clean input, length `d ≥ 1`.
    /// * `classify_fn`  — closure returning a length-`n_classes` softmax row.
    /// * `n_classes`    — number of output classes (`≥ 2`).
    /// * `rng`          — mutable RNG for noise generation.
    ///
    /// # Returns
    /// `(top_class, averaged_distribution)` where `averaged_distribution`
    /// has length `n_classes` and sums to `≈ 1` (within rounding).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — empty input.
    /// * [`AdvError::NanEncountered`]     — non-finite input or non-finite
    ///   classifier output.
    /// * [`AdvError::InvalidLossWeight`]  — `n_classes < 2`.
    /// * [`AdvError::DimensionMismatch`]  — `classify_fn` returned the wrong
    ///   number of class probabilities.
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
                location: "macer:smoothed_predict:input",
            });
        }

        let d = input.len();
        let mut noisy = vec![0.0_f32; d];
        let mut accum = vec![0.0_f64; n_classes];

        for _ in 0..self.cfg.n_samples_smooth {
            // Fill noisy = input + σ · z, drawing pairs of standard normals.
            let mut i = 0;
            while i + 1 < d {
                let (z_a, z_b) = rng.next_normal_pair();
                noisy[i] = input[i] + self.cfg.sigma * z_a;
                noisy[i + 1] = input[i + 1] + self.cfg.sigma * z_b;
                i += 2;
            }
            if i < d {
                let (z_a, _z_b) = rng.next_normal_pair();
                noisy[i] = input[i] + self.cfg.sigma * z_a;
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
                        location: "macer:smoothed_predict:classify_fn",
                    });
                }
                *slot += p as f64;
            }
        }

        let n_f = self.cfg.n_samples_smooth as f64;
        // Average and renormalise to compensate for tiny rounding drift.
        let total: f64 = accum.iter().sum();
        let denom = if total > 0.0 { total } else { n_f };
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

    // ─── Certified radius ──────────────────────────────────────────────────

    /// Cohen 2019 / MACER certified L2 radius:
    ///
    /// ```text
    /// r = σ · Φ⁻¹(p̂_top)   if p̂_top > 0.5
    ///     0                  otherwise.
    /// ```
    ///
    /// `p_top` near `1` is clipped to `1 − 1e-7` to keep `Φ⁻¹` finite
    /// (matches the `clip_p` constant inside `probit`).
    ///
    /// # Errors
    /// * [`AdvError::InvalidConfidence`] if `p_top` is non-finite or
    ///   outside `[0, 1]`.
    pub fn certified_radius(&self, p_top: f32) -> AdvResult<f32> {
        if !(p_top.is_finite() && (0.0..=1.0).contains(&p_top)) {
            return Err(AdvError::InvalidConfidence { alpha: p_top });
        }
        if p_top <= 0.5 {
            return Ok(0.0);
        }
        let q = self.probit(p_top)?;
        let r = self.cfg.sigma * q;
        if !r.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "macer:certified_radius",
            });
        }
        Ok(r)
    }

    // ─── MACER training loss ───────────────────────────────────────────────

    /// MACER outer training loss:
    ///
    /// ```text
    /// L = cls_loss + λ · max(0, γ − r),    r = σ · Φ⁻¹(p̂_top).
    /// ```
    ///
    /// When `p̂_top ≤ 0.5` we set `r = 0` and the hinge is at its maximum
    /// `λ · γ`. When `r ≥ γ` the hinge vanishes and the returned loss
    /// equals `cls_loss`.
    ///
    /// # Errors
    /// * [`AdvError::NanEncountered`]     — `cls_loss` non-finite.
    /// * [`AdvError::InvalidConfidence`]  — `p_top` invalid.
    pub fn loss(&self, cls_loss: f32, p_top: f32) -> AdvResult<f32> {
        if !cls_loss.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "macer:loss:cls_loss",
            });
        }
        let r = self.certified_radius(p_top)?;
        let hinge = (self.cfg.gamma_hinge - r).max(0.0);
        let total = cls_loss + self.cfg.lambda_robust * hinge;
        if !total.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "macer:loss:total",
            });
        }
        Ok(total)
    }
}

// ─── Acklam probit (free function in f64) ────────────────────────────────────

/// Internal Acklam (2003) rational approximation of `Φ⁻¹(p)` in `f64`.
///
/// Clamps `p` to `[1e-12, 1 − 1e-12]` to keep tail evaluation finite.
fn acklam_probit(p_in: f64) -> f64 {
    // Coefficients (Acklam 2003).
    const A: [f64; 6] = [
        -3.969_683_028_665_376_2e1,
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
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    const EPS: f64 = 1e-12;

    let p = p_in.clamp(EPS, 1.0 - EPS);

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -num / den
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn make_loss(sigma: f32, lambda: f32, gamma: f32, n: usize) -> MacerLoss {
        MacerLoss::new(MacerConfig::new(sigma, lambda, gamma, n).expect("cfg")).expect("macer loss")
    }

    // ─── Config validation ────────────────────────────────────────────────

    #[test]
    fn config_rejects_invalid_sigma() {
        assert!(MacerConfig::new(0.0, 1.0, 1.0, 4).is_err());
        assert!(MacerConfig::new(-0.1, 1.0, 1.0, 4).is_err());
        assert!(MacerConfig::new(f32::NAN, 1.0, 1.0, 4).is_err());
        assert!(MacerConfig::new(f32::INFINITY, 1.0, 1.0, 4).is_err());
    }

    #[test]
    fn config_rejects_invalid_lambda_and_gamma() {
        assert!(MacerConfig::new(0.25, -1.0, 1.0, 4).is_err());
        assert!(MacerConfig::new(0.25, 1.0, -0.1, 4).is_err());
        assert!(MacerConfig::new(0.25, f32::NAN, 1.0, 4).is_err());
        assert!(MacerConfig::new(0.25, 1.0, f32::INFINITY, 4).is_err());
    }

    #[test]
    fn config_rejects_zero_samples() {
        assert!(MacerConfig::new(0.25, 1.0, 1.0, 0).is_err());
    }

    #[test]
    fn config_default_is_sensible() {
        let c = MacerConfig::default();
        assert!(c.sigma > 0.0);
        assert!(c.lambda_robust >= 0.0);
        assert!(c.gamma_hinge >= 0.0);
        assert!(c.n_samples_smooth >= 1);
    }

    // ─── Probit tests ─────────────────────────────────────────────────────

    #[test]
    fn probit_midpoint_is_zero() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let q = m.probit(0.5).expect("ok");
        assert!(approx_eq(q, 0.0, 1e-6));
    }

    #[test]
    fn probit_97_5_pct_quantile() {
        // Φ⁻¹(0.975) ≈ 1.959963984540054
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let q = m.probit(0.975).expect("ok");
        assert!(approx_eq(q, 1.959_964, 5e-3));
    }

    #[test]
    fn probit_is_monotone() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let xs = [0.05_f32, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95];
        let mut prev = m.probit(0.01).expect("ok");
        for &x in &xs {
            let q = m.probit(x).expect("ok");
            assert!(q >= prev - 1e-5, "not monotone at p={x}: {q} < {prev}");
            prev = q;
        }
    }

    #[test]
    fn probit_extremes_are_finite() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let lo = m.probit(0.0).expect("ok");
        let hi = m.probit(1.0).expect("ok");
        assert!(lo.is_finite() && lo < 0.0);
        assert!(hi.is_finite() && hi > 0.0);
    }

    #[test]
    fn probit_rejects_out_of_range() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        assert!(m.probit(-0.01).is_err());
        assert!(m.probit(1.01).is_err());
        assert!(m.probit(f32::NAN).is_err());
    }

    // ─── certified_radius tests ───────────────────────────────────────────

    #[test]
    fn certified_radius_zero_at_half() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        assert_eq!(m.certified_radius(0.5).expect("ok"), 0.0);
        assert_eq!(m.certified_radius(0.3).expect("ok"), 0.0);
    }

    #[test]
    fn certified_radius_monotone_increasing() {
        let m = make_loss(0.5, 1.0, 1.0, 4);
        let r_55 = m.certified_radius(0.55).expect("ok");
        let r_75 = m.certified_radius(0.75).expect("ok");
        let r_99 = m.certified_radius(0.99).expect("ok");
        assert!(r_55 < r_75);
        assert!(r_75 < r_99);
        assert!(r_55 > 0.0);
    }

    #[test]
    fn certified_radius_scales_linearly_with_sigma() {
        let m1 = make_loss(0.25, 1.0, 1.0, 4);
        let m2 = make_loss(1.00, 1.0, 1.0, 4);
        let p = 0.9_f32;
        let r1 = m1.certified_radius(p).expect("ok");
        let r2 = m2.certified_radius(p).expect("ok");
        let ratio = r2 / r1;
        assert!(approx_eq(ratio, 4.0, 1e-3), "ratio={ratio}");
    }

    #[test]
    fn certified_radius_extreme_p_finite() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let r0 = m.certified_radius(0.0).expect("ok");
        let r1 = m.certified_radius(1.0).expect("ok");
        // p <= 0.5 ⇒ r == 0 (zero clipping case).
        assert_eq!(r0, 0.0);
        // p == 1 ⇒ r is large but finite due to internal clipping.
        assert!(r1.is_finite() && r1 > 0.0);
    }

    #[test]
    fn certified_radius_rejects_invalid_p() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        assert!(m.certified_radius(-0.01).is_err());
        assert!(m.certified_radius(1.01).is_err());
        assert!(m.certified_radius(f32::NAN).is_err());
    }

    // ─── loss tests ───────────────────────────────────────────────────────

    #[test]
    fn loss_no_hinge_when_radius_exceeds_gamma() {
        // σ=1, γ=0.1, p_top=0.99 ⇒ r ≈ 2.33 ≥ γ ⇒ hinge=0 ⇒ loss == cls_loss.
        let m = make_loss(1.0, 5.0, 0.1, 4);
        let cls = 0.7_f32;
        let l = m.loss(cls, 0.99).expect("ok");
        assert!(approx_eq(l, cls, 1e-5), "l={l} cls={cls}");
    }

    #[test]
    fn loss_includes_hinge_when_radius_below_gamma() {
        // σ=0.25, γ=2.0, p_top=0.6 ⇒ r ≈ 0.25·0.2533 ≈ 0.063 ≪ γ.
        let m = make_loss(0.25, 5.0, 2.0, 4);
        let cls = 0.3_f32;
        let l = m.loss(cls, 0.6).expect("ok");
        assert!(l > cls + 1e-5, "loss must include hinge; l={l} cls={cls}");
    }

    #[test]
    fn loss_at_p_half_equals_cls_plus_lambda_gamma() {
        // p_top=0.5 ⇒ r=0 ⇒ hinge=γ ⇒ loss = cls_loss + λ·γ.
        let lambda = 4.0_f32;
        let gamma = 1.5_f32;
        let cls = 0.2_f32;
        let m = make_loss(0.25, lambda, gamma, 4);
        let l = m.loss(cls, 0.5).expect("ok");
        let expect = cls + lambda * gamma;
        assert!(approx_eq(l, expect, 1e-5), "l={l} expect={expect}");
    }

    #[test]
    fn loss_rejects_nan_cls_loss() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        assert!(m.loss(f32::NAN, 0.9).is_err());
        assert!(m.loss(f32::INFINITY, 0.9).is_err());
    }

    #[test]
    fn loss_rejects_invalid_p_top() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        assert!(m.loss(0.5, -0.1).is_err());
        assert!(m.loss(0.5, 1.1).is_err());
    }

    // ─── smoothed_predict tests ──────────────────────────────────────────

    #[test]
    fn smoothed_predict_returns_valid_class() {
        // Constant classifier always returns one-hot at class 3 (n_classes=5).
        let m = make_loss(0.25, 1.0, 1.0, 8);
        let mut rng = LcgRng::new(123);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let n_classes = 5_usize;
        let (cls, dist) = m
            .smoothed_predict(
                &x,
                |_y| {
                    let mut p = vec![0.0_f32; n_classes];
                    p[3] = 1.0;
                    p
                },
                n_classes,
                &mut rng,
            )
            .expect("ok");
        assert_eq!(cls, 3);
        assert_eq!(dist.len(), n_classes);
        // dist should be one-hot up to rounding.
        assert!(approx_eq(dist[3], 1.0, 1e-5));
    }

    #[test]
    fn smoothed_predict_distribution_sums_to_one() {
        // Use a uniform softmax classifier — averaged dist must still sum to 1.
        let m = make_loss(0.3, 1.0, 1.0, 32);
        let mut rng = LcgRng::new(7);
        let x = vec![0.5_f32; 6];
        let n_classes = 4_usize;
        let (_cls, dist) = m
            .smoothed_predict(&x, |_y| vec![0.25_f32; n_classes], n_classes, &mut rng)
            .expect("ok");
        let sum: f32 = dist.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "sum={sum}");
        for &v in &dist {
            assert!(approx_eq(v, 0.25, 1e-5));
        }
    }

    #[test]
    fn smoothed_predict_is_deterministic_given_seed() {
        // Deterministic classifier + fixed seed ⇒ identical outputs.
        let m = make_loss(0.25, 1.0, 1.0, 16);
        let n_classes = 3_usize;
        let classifier = |y: &[f32]| {
            // Class 0 wins when the first coordinate is positive; else class 1.
            // Use a smooth softmax via a sigmoid-like rule on the first coord.
            let s = y[0];
            let p0 = 1.0 / (1.0 + (-s).exp());
            vec![p0, 1.0 - p0, 0.0]
        };
        let x = vec![0.2_f32, 0.4, -0.1];
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let (cls_a, dist_a) = m
            .smoothed_predict(&x, classifier, n_classes, &mut rng_a)
            .expect("a");
        let (cls_b, dist_b) = m
            .smoothed_predict(&x, classifier, n_classes, &mut rng_b)
            .expect("b");
        assert_eq!(cls_a, cls_b);
        for (a, b) in dist_a.iter().zip(dist_b.iter()) {
            assert!(approx_eq(*a, *b, 1e-7));
        }
    }

    #[test]
    fn smoothed_predict_rejects_empty_input() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let mut rng = LcgRng::new(0);
        let err = m
            .smoothed_predict(&[], |_| vec![0.5_f32, 0.5], 2, &mut rng)
            .unwrap_err();
        assert!(matches!(err, AdvError::EmptyInput));
    }

    #[test]
    fn smoothed_predict_rejects_single_class() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let mut rng = LcgRng::new(0);
        let err = m
            .smoothed_predict(&[0.1_f32, 0.2], |_| vec![1.0_f32], 1, &mut rng)
            .unwrap_err();
        assert!(matches!(err, AdvError::InvalidLossWeight { .. }));
    }

    #[test]
    fn smoothed_predict_rejects_wrong_length_softmax() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let mut rng = LcgRng::new(0);
        let err = m
            .smoothed_predict(
                &[0.1_f32, 0.2],
                |_| vec![0.5_f32, 0.5, 0.0], // length 3 but n_classes=2
                2,
                &mut rng,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            AdvError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn smoothed_predict_rejects_nan_input() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let mut rng = LcgRng::new(0);
        let err = m
            .smoothed_predict(&[f32::NAN, 0.2], |_| vec![0.5_f32, 0.5], 2, &mut rng)
            .unwrap_err();
        assert!(matches!(err, AdvError::NanEncountered { .. }));
    }

    #[test]
    fn smoothed_predict_rejects_nan_softmax() {
        let m = make_loss(0.25, 1.0, 1.0, 4);
        let mut rng = LcgRng::new(0);
        let err = m
            .smoothed_predict(&[0.1_f32, 0.2], |_| vec![f32::NAN, 0.5], 2, &mut rng)
            .unwrap_err();
        assert!(matches!(err, AdvError::NanEncountered { .. }));
    }

    // ─── Construction sanity ──────────────────────────────────────────────

    #[test]
    fn macer_loss_new_rejects_invalid_config() {
        // Direct field construction of invalid config caught by new().
        let bad = MacerConfig {
            sigma: -0.1,
            lambda_robust: 1.0,
            gamma_hinge: 1.0,
            n_samples_smooth: 1,
        };
        assert!(MacerLoss::new(bad).is_err());
    }
}
