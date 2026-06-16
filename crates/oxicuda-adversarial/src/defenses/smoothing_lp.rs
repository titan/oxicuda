//! Lp randomized-smoothing certificates — certified robustness under general
//! `Lp` adversaries via matched noise distributions.
//!
//! Reference: Yang, Duan, Hu, Salman, Razenshteyn & Li (2020),
//! *"Randomized Smoothing of All Shapes and Sizes"*, ICML.
//!
//! Cohen et al. (2019) certify the **L2** radius of a Gaussian-smoothed
//! classifier as `r = σ · Φ⁻¹(p_A)`. Yang et al. (2020) generalise this to
//! arbitrary `Lp` adversaries by matching the smoothing distribution to the
//! norm: the *generalised Gaussian* (a.k.a. exponential-power / generalised
//! normal) family
//!
//! ```text
//! q_p(η) ∝ exp( − ‖η / α‖_p^p )      (i.i.d. per coordinate)
//! ```
//!
//! has level sets that are `Lp` balls and yields a certificate in the `Lp`
//! norm. The two best-known special cases are
//!
//! * `p = 2` ← Gaussian noise          → `r = σ · Φ⁻¹(p_A)`            (Cohen).
//! * `p = 1` ← Laplace noise           → `r = b · ln( 1 / (2(1−p_A)) )` (b = √2·σ).
//!
//! and the general `p` certificate (Cohen/Yang "only-`p_A`" Neyman–Pearson
//! bound for a symmetric, unimodal i.i.d. measure) is
//!
//! ```text
//! r(p_A) = σ · √2 · [ P⁻¹( 1/p ,  2 p_A − 1 ) ]^{1/p}
//! ```
//!
//! where `P⁻¹(a, ·)` is the inverse of the regularised lower incomplete
//! Gamma function `P(a, x) = γ(a, x) / Γ(a)`. The scale convention pins
//! `σ` to be exactly the Gaussian standard deviation at `p = 2`, so the
//! formula reduces *exactly* to `σ · Φ⁻¹(p_A)` there:
//!
//! ```text
//! P⁻¹(1/2, 2 p_A − 1) = ( erf⁻¹(2 p_A − 1) )²
//! ⇒ r = σ √2 · erf⁻¹(2 p_A − 1) = σ · Φ⁻¹(p_A).
//! ```
//!
//! # API
//!
//! * [`LpSmoothingCertifier::certified_radius`] — dispatching certificate that
//!   uses the exact closed forms for `p ∈ {1, 2}` and the generalised-Gamma
//!   path otherwise.
//! * [`LpSmoothingCertifier::certified_radius_generalized`] — always uses the
//!   generalised-Gamma path (reduces to the Gaussian value at `p = 2`).
//! * [`LpSmoothingCertifier::sample_noise`] — draws `d` i.i.d. samples from the
//!   matched smoothing distribution.
//! * [`LpSmoothingCertifier::certify`] — turns Monte-Carlo class-vote counts
//!   into `(predicted_class, radius)` via a Clopper–Pearson lower bound.
//! * [`LpSmoothingCertifier::predict_and_certify`] — full Monte-Carlo pipeline
//!   over a base classifier closure.
//!
//! # RNG note
//!
//! `LcgRng::next_u32` returns the high 31 bits of the LCG state (it shifts by
//! 33), so it spans `[0, 2³¹)`. We therefore draw uniforms as
//! `next_u32() / 2³¹ ∈ [0, 1)`. The crate's `next_f32` only spans `[0, 0.5)`
//! and its `next_normal_pair` is consequently biased, so this module
//! implements its **own** Box–Muller transform on a correct uniform.

use crate::defenses::randomized_smoothing::{clopper_pearson_lower, inverse_normal_cdf};
use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;
use std::f64::consts::{PI, SQRT_2, TAU};

/// `2³²` as `f64` — divisor mapping `LcgRng::next_u32() ∈ [0, 2³¹)` to `[0, 1)`.
const U32_DIVISOR: f64 = 4_294_967_296.0;

/// Probability clip kept away from `1` so the inverse certificates stay finite.
const P_CLIP: f64 = 1e-7;

// ─── Special functions (f64) ────────────────────────────────────────────────

/// Natural logarithm of the Gamma function via the Lanczos approximation
/// (`g = 7`, 9 coefficients). Accurate to `≈ 1e-13` relative for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: Γ(x)·Γ(1−x) = π / sin(πx).
        PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * TAU.ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Series expansion of the regularised lower incomplete Gamma `P(a, x)`,
/// valid (and convergent) for `x < a + 1`.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..400 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Lentz continued-fraction for the regularised *upper* incomplete Gamma
/// `Q(a, x)`, valid for `x ≥ a + 1`.
fn gamma_q_cont_frac(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let gln = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..400 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularised lower incomplete Gamma `P(a, x) = γ(a, x) / Γ(a)`, monotone
/// increasing from `0` (at `x = 0`) to `1` (as `x → ∞`).
fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cont_frac(a, x)
    }
}

/// Inverse of `gamma_p` in its second argument: returns `x ≥ 0` such that
/// `P(a, x) = target` for `target ∈ (0, 1)`.
///
/// Robust bracketed bisection (`gamma_p` is strictly increasing in `x`).
/// Used only for scalar certificate evaluation, never in a hot loop.
fn gamma_p_inverse(a: f64, target: f64) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    if target >= 1.0 {
        return f64::INFINITY;
    }
    // Expand the upper bracket until P(a, hi) ≥ target.
    let mut hi = 1.0_f64;
    let mut guard = 0;
    while gamma_p(a, hi) < target {
        hi *= 2.0;
        guard += 1;
        if guard > 200 {
            break;
        }
    }
    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-13 * (hi + 1.0) {
            break;
        }
    }
    0.5 * (lo + hi)
}

// ─── RNG helpers (correct uniform + Box–Muller) ─────────────────────────────

/// Uniform `[0, 1)` from the high-31-bit LCG output.
#[inline]
fn uniform01(rng: &mut LcgRng) -> f64 {
    rng.next_u32() as f64 / U32_DIVISOR
}

/// A pair of independent standard-normal `N(0, 1)` samples via Box–Muller on a
/// correct full-range uniform (the crate's `next_normal_pair` is biased here).
fn standard_normal_pair(rng: &mut LcgRng) -> (f64, f64) {
    let u1 = uniform01(rng).max(1e-12);
    let u2 = uniform01(rng);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// One `Gamma(shape = k, scale = 1)` sample via Marsaglia–Tsang (`k ≥ 1`) with
/// the Ahrens–Dieter boost (`k < 1`).
fn sample_gamma(k: f64, rng: &mut LcgRng) -> f64 {
    if k < 1.0 {
        let u = uniform01(rng).max(1e-300);
        return sample_gamma(k + 1.0, rng) * u.powf(1.0 / k);
    }
    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x = standard_normal_pair(rng).0;
        let mut v = 1.0 + c * x;
        while v <= 0.0 {
            x = standard_normal_pair(rng).0;
            v = 1.0 + c * x;
        }
        v = v * v * v;
        let u = uniform01(rng).max(1e-300);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ─── Certifier ──────────────────────────────────────────────────────────────

/// Certified `Lp` radius helper for randomized smoothing with a matched
/// generalised-Gaussian noise distribution.
///
/// `p` selects both the adversary norm and the smoothing distribution shape;
/// `sigma` is the noise scale, normalised so it equals the Gaussian standard
/// deviation at `p = 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpSmoothingCertifier {
    /// `Lp` norm / generalised-Gaussian shape parameter (`> 0`, finite).
    /// `1` → L1 (Laplace), `2` → L2 (Gaussian).
    pub p: f32,
    /// Noise scale (`> 0`, finite); the Gaussian std at `p = 2`.
    pub sigma: f32,
}

impl LpSmoothingCertifier {
    /// Build a new certifier.
    ///
    /// # Errors
    /// * [`AdvError::InvalidLpNorm`]     — `p` is non-finite or `≤ 0`.
    /// * [`AdvError::InvalidNoiseSigma`] — `sigma` is non-finite or `≤ 0`.
    pub fn new(p: f32, sigma: f32) -> AdvResult<Self> {
        if !(p.is_finite() && p > 0.0) {
            return Err(AdvError::InvalidLpNorm);
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(AdvError::InvalidNoiseSigma { sigma });
        }
        Ok(Self { p, sigma })
    }

    /// Theoretical standard deviation of one coordinate of the matched noise
    /// distribution: `α · sqrt( Γ(3/p) / Γ(1/p) )` with `α = √2 · σ`.
    ///
    /// Equals `σ` at `p = 2` and `2σ` at `p = 1` (Laplace).
    #[must_use]
    pub fn noise_std(&self) -> f32 {
        let beta = self.p as f64;
        let alpha = self.sigma as f64 * SQRT_2;
        let ln_ratio = ln_gamma(3.0 / beta) - ln_gamma(1.0 / beta);
        (alpha * (0.5 * ln_ratio).exp()) as f32
    }

    /// Validate a lower-confidence top-class probability and map it to the
    /// `(p_A ≤ 0.5 → 0)` regime, returning the clipped value when certifiable.
    fn prepare_pa(p_a_lower: f32) -> AdvResult<Option<f64>> {
        if !(p_a_lower.is_finite() && (0.0..=1.0).contains(&p_a_lower)) {
            return Err(AdvError::InvalidConfidence { alpha: p_a_lower });
        }
        if p_a_lower <= 0.5 {
            return Ok(None);
        }
        Ok(Some((p_a_lower as f64).min(1.0 - P_CLIP)))
    }

    /// Generalised-Gamma certificate `σ · √2 · [P⁻¹(1/p, 2 p_A − 1)]^{1/p}`
    /// for an already-validated `p_A ∈ (0.5, 1)`.
    fn radius_generalized_inner(&self, pa: f64) -> f64 {
        let p = self.p as f64;
        let a = 1.0 / p;
        let target = 2.0 * pa - 1.0;
        let g = gamma_p_inverse(a, target);
        let q = g.powf(1.0 / p);
        self.sigma as f64 * SQRT_2 * q
    }

    /// Certified `Lp` radius given a lower-confidence bound `p_a_lower` on the
    /// top-class probability under noise.
    ///
    /// Uses the exact closed forms for `p ∈ {1, 2}` and the generalised-Gamma
    /// path otherwise. Returns `0.0` when `p_a_lower ≤ 0.5` (the certificate is
    /// vacuous — the smoothed top-1 is not provably above one-half).
    ///
    /// # Errors
    /// * [`AdvError::InvalidConfidence`] — `p_a_lower` non-finite or outside
    ///   `[0, 1]`.
    /// * [`AdvError::NanEncountered`]    — the computed radius is non-finite.
    pub fn certified_radius(&self, p_a_lower: f32) -> AdvResult<f32> {
        let Some(pa) = Self::prepare_pa(p_a_lower)? else {
            return Ok(0.0);
        };
        let sigma = self.sigma as f64;
        let r = if (self.p - 2.0).abs() < 1e-4 {
            // L2 / Gaussian — exact Cohen certificate.
            sigma * inverse_normal_cdf(pa)
        } else if (self.p - 1.0).abs() < 1e-4 {
            // L1 / Laplace — exact closed form with scale b = √2·σ.
            sigma * SQRT_2 * (1.0 / (2.0 * (1.0 - pa))).ln()
        } else {
            self.radius_generalized_inner(pa)
        };
        finite_nonneg(r, "smoothing_lp:certified_radius")
    }

    /// Certified `Lp` radius via the generalised-Gamma path for **all** `p`
    /// (including `p ∈ {1, 2}`). At `p = 2` this matches
    /// [`Self::certified_radius`] to numerical precision, demonstrating that the
    /// generalised certificate reduces to the Gaussian one.
    ///
    /// # Errors
    /// As [`Self::certified_radius`].
    pub fn certified_radius_generalized(&self, p_a_lower: f32) -> AdvResult<f32> {
        let Some(pa) = Self::prepare_pa(p_a_lower)? else {
            return Ok(0.0);
        };
        finite_nonneg(
            self.radius_generalized_inner(pa),
            "smoothing_lp:certified_radius_generalized",
        )
    }

    /// Draw `dim` i.i.d. samples from the matched smoothing distribution.
    ///
    /// * `p = 2` → `N(0, σ²)` via Box–Muller.
    /// * `p = 1` → `Laplace(0, √2·σ)` via inverse-CDF.
    /// * else    → generalised normal `(shape p, scale √2·σ)` via the Gamma
    ///   representation `X = α · sign · G^{1/p}`, `G ∼ Gamma(1/p, 1)`.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`] — `dim == 0`.
    pub fn sample_noise(&self, dim: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
        if dim == 0 {
            return Err(AdvError::EmptyInput);
        }
        let sigma = self.sigma as f64;
        let mut out = Vec::with_capacity(dim);
        if (self.p - 2.0).abs() < 1e-4 {
            while out.len() + 1 < dim {
                let (a, b) = standard_normal_pair(rng);
                out.push((sigma * a) as f32);
                out.push((sigma * b) as f32);
            }
            if out.len() < dim {
                let (a, _) = standard_normal_pair(rng);
                out.push((sigma * a) as f32);
            }
        } else if (self.p - 1.0).abs() < 1e-4 {
            let b = sigma * SQRT_2;
            for _ in 0..dim {
                let s = uniform01(rng) - 0.5;
                let mag = (1.0 - 2.0 * s.abs()).max(1e-300);
                let sign = if s >= 0.0 { 1.0 } else { -1.0 };
                out.push((-b * sign * mag.ln()) as f32);
            }
        } else {
            let alpha = sigma * SQRT_2;
            let p = self.p as f64;
            let k = 1.0 / p;
            for _ in 0..dim {
                let g = sample_gamma(k, rng);
                let sign = if uniform01(rng) < 0.5 { -1.0 } else { 1.0 };
                out.push((alpha * sign * g.powf(1.0 / p)) as f32);
            }
        }
        Ok(out)
    }

    /// Turn Monte-Carlo class-vote counts into `(predicted_class, radius)`.
    ///
    /// `class_counts[c]` is the number of noisy samples that voted for class
    /// `c`; `n` is the total Monte-Carlo budget; `alpha` is the one-sided
    /// Clopper–Pearson failure probability. The top class is the argmax of
    /// `class_counts`; its lower-confidence probability is
    /// `clopper_pearson_lower(top_count, n, alpha)` and the radius is
    /// [`Self::certified_radius`] of that bound (`0` when not provably above `½`).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]          — `class_counts` empty.
    /// * [`AdvError::InsufficientCertSamples`] — `n == 0`.
    /// * [`AdvError::InvalidConfidence`]   — `alpha` outside `(0, 1)`.
    pub fn certify(&self, class_counts: &[usize], n: usize, alpha: f32) -> AdvResult<(usize, f32)> {
        if class_counts.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if n == 0 {
            return Err(AdvError::InsufficientCertSamples { min: 1, got: n });
        }
        if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return Err(AdvError::InvalidConfidence { alpha });
        }
        let (top_class, top_count) =
            class_counts
                .iter()
                .enumerate()
                .fold(
                    (0_usize, 0_usize),
                    |(bi, bc), (i, &c)| if c > bc { (i, c) } else { (bi, bc) },
                );
        let p_lower = clopper_pearson_lower(top_count, n, alpha);
        let radius = self.certified_radius(p_lower)?;
        Ok((top_class, radius))
    }

    /// Full Monte-Carlo pipeline: draw `n` noisy copies of `x`, classify each,
    /// tally the votes, and certify. Mirrors Cohen's `CERTIFY` procedure but
    /// under the matched `Lp` noise.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]          — `x` empty.
    /// * [`AdvError::NanEncountered`]      — non-finite input.
    /// * [`AdvError::InsufficientCertSamples`] — `n == 0`.
    /// * [`AdvError::InvalidConfidence`]   — `alpha` outside `(0, 1)`.
    /// * Any error returned by `base_classify`.
    pub fn predict_and_certify<F>(
        &self,
        x: &[f32],
        n: usize,
        alpha: f32,
        rng: &mut LcgRng,
        base_classify: F,
    ) -> AdvResult<(usize, f32)>
    where
        F: Fn(&[f32]) -> AdvResult<usize>,
    {
        if x.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "smoothing_lp:predict_and_certify:x",
            });
        }
        if n == 0 {
            return Err(AdvError::InsufficientCertSamples { min: 1, got: n });
        }
        if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return Err(AdvError::InvalidConfidence { alpha });
        }
        // Sparse (class, count) table — K is unknown a-priori.
        let mut counts: Vec<(usize, usize)> = Vec::new();
        let mut noisy = vec![0.0_f32; x.len()];
        for _ in 0..n {
            let noise = self.sample_noise(x.len(), rng)?;
            for ((dst, &xi), &ni) in noisy.iter_mut().zip(x.iter()).zip(noise.iter()) {
                *dst = xi + ni;
            }
            let cls = base_classify(&noisy)?;
            if let Some(slot) = counts.iter_mut().find(|(c, _)| *c == cls) {
                slot.1 += 1;
            } else {
                counts.push((cls, 1));
            }
        }
        let (top_class, top_count) =
            counts
                .iter()
                .fold((0_usize, 0_usize), |(bi, bc), &(c, n_c)| {
                    if n_c > bc { (c, n_c) } else { (bi, bc) }
                });
        let p_lower = clopper_pearson_lower(top_count, n, alpha);
        let radius = self.certified_radius(p_lower)?;
        Ok((top_class, radius))
    }
}

/// Guard a computed radius: must be finite and is clamped to `≥ 0`.
fn finite_nonneg(r: f64, location: &'static str) -> AdvResult<f32> {
    if !r.is_finite() {
        return Err(AdvError::NanEncountered { location });
    }
    Ok((r.max(0.0)) as f32)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_validates_parameters() {
        assert!(LpSmoothingCertifier::new(2.0, 0.25).is_ok());
        assert!(matches!(
            LpSmoothingCertifier::new(0.0, 0.25).unwrap_err(),
            AdvError::InvalidLpNorm
        ));
        assert!(matches!(
            LpSmoothingCertifier::new(-1.0, 0.25).unwrap_err(),
            AdvError::InvalidLpNorm
        ));
        assert!(matches!(
            LpSmoothingCertifier::new(f32::NAN, 0.25).unwrap_err(),
            AdvError::InvalidLpNorm
        ));
        assert!(matches!(
            LpSmoothingCertifier::new(2.0, 0.0).unwrap_err(),
            AdvError::InvalidNoiseSigma { .. }
        ));
        assert!(matches!(
            LpSmoothingCertifier::new(2.0, f32::INFINITY).unwrap_err(),
            AdvError::InvalidNoiseSigma { .. }
        ));
    }

    // ── L2 = σ·Φ⁻¹(p_A) ──────────────────────────────────────────────────────

    #[test]
    fn l2_radius_equals_sigma_times_phi_inverse() {
        let sigma = 0.5_f32;
        let c = LpSmoothingCertifier::new(2.0, sigma).expect("new should succeed");
        // Φ⁻¹(0.8413447) ≈ 1 ⇒ R ≈ σ.
        let r1 = c
            .certified_radius(0.841_344_7)
            .expect("certified_radius should succeed");
        assert!(approx(r1, sigma, 1e-2), "r={r1} sigma={sigma}");
        // Φ⁻¹(0.97725) ≈ 2 ⇒ R ≈ 2σ.
        let r2 = c
            .certified_radius(0.977_25)
            .expect("certified_radius should succeed");
        assert!(approx(r2, 2.0 * sigma, 1e-2), "r={r2}");
        // Exact comparison against the reused inverse-normal CDF.
        let expect = sigma * inverse_normal_cdf(0.9) as f32;
        let got = c
            .certified_radius(0.9)
            .expect("certified_radius should succeed");
        assert!(approx(got, expect, 1e-5), "got={got} expect={expect}");
    }

    // ── Zero / clamp below ½ ────────────────────────────────────────────────

    #[test]
    fn radius_zero_at_or_below_half() {
        for &p in &[1.0_f32, 2.0, 3.0] {
            let c = LpSmoothingCertifier::new(p, 1.0).expect("new should succeed");
            assert_eq!(
                c.certified_radius(0.5)
                    .expect("certified_radius should succeed"),
                0.0
            );
            assert_eq!(
                c.certified_radius(0.4)
                    .expect("certified_radius should succeed"),
                0.0
            );
            assert_eq!(
                c.certified_radius(0.0)
                    .expect("certified_radius should succeed"),
                0.0
            );
            assert_eq!(
                c.certified_radius_generalized(0.5)
                    .expect("certified_radius_generalized should succeed"),
                0.0
            );
        }
    }

    // ── Monotone in p_A ──────────────────────────────────────────────────────

    #[test]
    fn radius_monotone_increasing_in_pa() {
        for &p in &[1.0_f32, 2.0, 4.0] {
            let c = LpSmoothingCertifier::new(p, 0.7).expect("new should succeed");
            let r6 = c
                .certified_radius(0.6)
                .expect("certified_radius should succeed");
            let r7 = c
                .certified_radius(0.7)
                .expect("certified_radius should succeed");
            let r8 = c
                .certified_radius(0.8)
                .expect("certified_radius should succeed");
            let r9 = c
                .certified_radius(0.9)
                .expect("certified_radius should succeed");
            assert!(r6 > 0.0 && r6 < r7, "p={p} r6={r6} r7={r7}");
            assert!(r7 < r8, "p={p} r7={r7} r8={r8}");
            assert!(r8 < r9, "p={p} r8={r8} r9={r9}");
        }
    }

    // ── Monotone / linear in sigma ───────────────────────────────────────────

    #[test]
    fn radius_scales_linearly_with_sigma() {
        for &p in &[1.0_f32, 2.0, 3.5] {
            let c1 = LpSmoothingCertifier::new(p, 1.0).expect("new should succeed");
            let c2 = LpSmoothingCertifier::new(p, 2.0).expect("new should succeed");
            let r1 = c1
                .certified_radius(0.85)
                .expect("certified_radius should succeed");
            let r2 = c2
                .certified_radius(0.85)
                .expect("certified_radius should succeed");
            assert!(r2 > r1, "p={p}");
            assert!(approx(r2, 2.0 * r1, 1e-4), "p={p} r1={r1} r2={r2}");
        }
    }

    // ── Generalized reduces to Gaussian at p = 2 ────────────────────────────

    #[test]
    fn generalized_reduces_to_gaussian_at_p2() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        for &pa in &[0.6_f32, 0.7, 0.8, 0.9, 0.99] {
            let gen_r = c
                .certified_radius_generalized(pa)
                .expect("certified_radius_generalized should succeed");
            let gauss = 0.5_f32 * inverse_normal_cdf(pa as f64) as f32;
            assert!(
                approx(gen_r, gauss, 1e-3),
                "pa={pa} gen={gen_r} gauss={gauss}"
            );
        }
    }

    // ── L1 closed form ───────────────────────────────────────────────────────

    #[test]
    fn l1_matches_closed_form() {
        let sigma = 0.5_f32;
        let c = LpSmoothingCertifier::new(1.0, sigma).expect("new should succeed");
        for &pa in &[0.6_f32, 0.75, 0.9] {
            let expect = sigma * SQRT_2 as f32 * (1.0 / (2.0 * (1.0 - pa))).ln();
            let got = c
                .certified_radius(pa)
                .expect("certified_radius should succeed");
            assert!(
                approx(got, expect, 1e-4),
                "pa={pa} got={got} expect={expect}"
            );
        }
        // The dedicated L1 path and the generalised path must agree at p = 1.
        let gen_r = c
            .certified_radius_generalized(0.8)
            .expect("certified_radius_generalized should succeed");
        let exact = c
            .certified_radius(0.8)
            .expect("certified_radius should succeed");
        assert!(approx(gen_r, exact, 2e-3), "gen={gen_r} exact={exact}");
    }

    // ── Invalid p_A ──────────────────────────────────────────────────────────

    #[test]
    fn certified_radius_rejects_invalid_pa() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        assert!(matches!(
            c.certified_radius(-0.1).unwrap_err(),
            AdvError::InvalidConfidence { .. }
        ));
        assert!(matches!(
            c.certified_radius(1.1).unwrap_err(),
            AdvError::InvalidConfidence { .. }
        ));
        assert!(matches!(
            c.certified_radius(f32::NAN).unwrap_err(),
            AdvError::InvalidConfidence { .. }
        ));
    }

    // ── Finiteness near p_A → 1 ─────────────────────────────────────────────

    #[test]
    fn radius_finite_near_one() {
        for &p in &[1.0_f32, 2.0, 3.0] {
            let c = LpSmoothingCertifier::new(p, 1.0).expect("new should succeed");
            let r = c
                .certified_radius(1.0)
                .expect("certified_radius should succeed");
            assert!(r.is_finite() && r > 0.0, "p={p} r={r}");
        }
    }

    // ── noise_std analytic values ───────────────────────────────────────────

    #[test]
    fn noise_std_analytic_special_cases() {
        let c2 = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        assert!(approx(c2.noise_std(), 0.5, 1e-4));
        let c1 = LpSmoothingCertifier::new(1.0, 0.5).expect("new should succeed");
        // Laplace(0, √2·σ) std = 2σ.
        assert!(approx(c1.noise_std(), 1.0, 1e-4));
    }

    // ── Empirical std of the sampler ────────────────────────────────────────

    fn empirical_std(samples: &[f32]) -> f32 {
        let n = samples.len() as f64;
        let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = samples
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        var.sqrt() as f32
    }

    #[test]
    fn sampler_empirical_std_gaussian() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        let mut rng = LcgRng::new(7);
        let s = c
            .sample_noise(40_000, &mut rng)
            .expect("sample_noise should succeed");
        assert!(s.iter().all(|v| v.is_finite()));
        let sd = empirical_std(&s);
        assert!(approx(sd, 0.5, 0.03), "sd={sd}");
        // Symmetric ⇒ mean ≈ 0.
        let mean: f32 = s.iter().sum::<f32>() / s.len() as f32;
        assert!(mean.abs() < 0.02, "mean={mean}");
    }

    #[test]
    fn sampler_empirical_std_laplace() {
        let c = LpSmoothingCertifier::new(1.0, 0.5).expect("new should succeed");
        let mut rng = LcgRng::new(11);
        let s = c
            .sample_noise(40_000, &mut rng)
            .expect("sample_noise should succeed");
        let sd = empirical_std(&s);
        // Expected std = 2σ = 1.0.
        assert!(
            approx(sd, c.noise_std(), 0.06),
            "sd={sd} expect={}",
            c.noise_std()
        );
    }

    #[test]
    fn sampler_empirical_std_generalized() {
        let c = LpSmoothingCertifier::new(4.0, 0.6).expect("new should succeed");
        let mut rng = LcgRng::new(101);
        let s = c
            .sample_noise(40_000, &mut rng)
            .expect("sample_noise should succeed");
        assert!(s.iter().all(|v| v.is_finite()));
        let sd = empirical_std(&s);
        let expect = c.noise_std();
        let rel = (sd - expect).abs() / expect;
        assert!(rel < 0.08, "sd={sd} expect={expect} rel={rel}");
    }

    #[test]
    fn sample_noise_rejects_zero_dim() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            c.sample_noise(0, &mut rng).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn sample_noise_deterministic_given_seed() {
        let c = LpSmoothingCertifier::new(3.0, 0.4).expect("new should succeed");
        let mut a = LcgRng::new(123);
        let mut b = LcgRng::new(123);
        let sa = c
            .sample_noise(64, &mut a)
            .expect("sample_noise should succeed");
        let sb = c
            .sample_noise(64, &mut b)
            .expect("sample_noise should succeed");
        assert_eq!(sa, sb);
    }

    // ── certify (counts → radius) ───────────────────────────────────────────

    #[test]
    fn certify_picks_top_class_and_positive_radius() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        // Class 2 dominates overwhelmingly.
        let counts = vec![5_usize, 3, 980, 12];
        let (cls, r) = c
            .certify(&counts, 1000, 0.001)
            .expect("certify should succeed");
        assert_eq!(cls, 2);
        assert!(r.is_finite() && r > 0.0, "r={r}");
    }

    #[test]
    fn certify_ambiguous_yields_zero_radius() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        // 50/50 split ⇒ lower bound below ½ ⇒ radius 0.
        let counts = vec![500_usize, 500];
        let (_, r) = c
            .certify(&counts, 1000, 0.001)
            .expect("certify should succeed");
        assert_eq!(r, 0.0);
    }

    #[test]
    fn certify_validates_inputs() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        assert!(matches!(
            c.certify(&[], 10, 0.01).unwrap_err(),
            AdvError::EmptyInput
        ));
        assert!(matches!(
            c.certify(&[1_usize], 0, 0.01).unwrap_err(),
            AdvError::InsufficientCertSamples { .. }
        ));
        assert!(matches!(
            c.certify(&[1_usize], 10, 0.0).unwrap_err(),
            AdvError::InvalidConfidence { .. }
        ));
        assert!(matches!(
            c.certify(&[1_usize], 10, 1.0).unwrap_err(),
            AdvError::InvalidConfidence { .. }
        ));
    }

    // ── predict_and_certify ─────────────────────────────────────────────────

    #[test]
    fn predict_and_certify_constant_classifier() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let x = vec![0.0_f32; 6];
        let (cls, r) = c
            .predict_and_certify(&x, 800, 0.001, &mut rng, |_| Ok(3_usize))
            .expect("value should be present");
        assert_eq!(cls, 3);
        assert!(r > 0.0, "r={r}");
    }

    #[test]
    fn predict_and_certify_l1_path_runs() {
        let c = LpSmoothingCertifier::new(1.0, 0.4).expect("new should succeed");
        let mut rng = LcgRng::new(5);
        let x = vec![0.1_f32, -0.2, 0.3, 0.0];
        let (cls, r) = c
            .predict_and_certify(&x, 600, 0.01, &mut rng, |_| Ok(1_usize))
            .expect("value should be present");
        assert_eq!(cls, 1);
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn predict_and_certify_rejects_bad_input() {
        let c = LpSmoothingCertifier::new(2.0, 0.5).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            c.predict_and_certify(&[], 10, 0.01, &mut rng, |_| Ok(0_usize))
                .unwrap_err(),
            AdvError::EmptyInput
        ));
        assert!(matches!(
            c.predict_and_certify(&[f32::NAN], 10, 0.01, &mut rng, |_| Ok(0_usize))
                .unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
        let err = c
            .predict_and_certify(&[0.1_f32], 10, 0.01, &mut rng, |_| {
                Err(AdvError::Internal("boom".into()))
            })
            .unwrap_err();
        assert!(matches!(err, AdvError::Internal(_)));
    }

    // ── Special-function sanity ─────────────────────────────────────────────

    #[test]
    fn gamma_p_inverse_round_trips() {
        for &a in &[0.25_f64, 0.5, 1.0, 2.0, 3.0] {
            for &x in &[0.1_f64, 0.5, 1.0, 2.5, 5.0] {
                let target = gamma_p(a, x);
                let back = gamma_p_inverse(a, target);
                assert!((back - x).abs() < 1e-4, "a={a} x={x} back={back}");
            }
        }
    }

    #[test]
    fn gamma_p_half_is_erf() {
        // P(1/2, x) = erf(√x). Check P(1/2, 1) = erf(1) ≈ 0.842700793.
        let v = gamma_p(0.5, 1.0);
        assert!((v - 0.842_700_793).abs() < 1e-6, "v={v}");
    }

    #[test]
    fn ln_gamma_known_values() {
        // Γ(1)=1, Γ(2)=1, Γ(5)=24, Γ(1/2)=√π.
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-9);
    }
}
