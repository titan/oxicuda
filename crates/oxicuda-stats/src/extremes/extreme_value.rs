//! Extreme-value distributions: the Generalised Extreme Value (GEV) distribution
//! for block maxima and the Generalised Pareto Distribution (GPD) for
//! peaks-over-threshold exceedances.
//!
//! # Generalised Extreme Value — GEV(μ, σ, ξ)
//!
//! With location μ, scale σ > 0 and shape ξ, for `1 + ξ(x − μ)/σ > 0`,
//!
//! ```text
//! F(x) = exp{ −[1 + ξ (x − μ)/σ]^(−1/ξ) },   ξ ≠ 0
//! F(x) = exp{ −exp(−(x − μ)/σ) },             ξ = 0   (Gumbel limit)
//! ```
//!
//! # Generalised Pareto — GPD(σ, ξ)
//!
//! For exceedances `y = x − u ≥ 0` over a threshold u, with `1 + ξ y/σ > 0`,
//!
//! ```text
//! F(y) = 1 − [1 + ξ y/σ]^(−1/ξ),   ξ ≠ 0
//! F(y) = 1 − exp(−y/σ),            ξ = 0   (exponential limit)
//! ```
//!
//! Parameters are estimated by **Probability-Weighted Moments** (Hosking &
//! Wallis 1987), which are robust for small samples. Return levels — the value
//! expected to be exceeded once every `T` blocks — are obtained as the
//! `1 − 1/T` quantile.
//!
//! # References
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
//!   Values*, Springer.
//! - Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter and Quantile Estimation
//!   for the Generalized Pareto Distribution." *Technometrics* 29(3):339-349.
//! - Hosking, Wallis & Wood (1985). "Estimation of the GEV Distribution by the
//!   Method of Probability-Weighted Moments." *Technometrics* 27(3):251-261.

use crate::error::{StatsError, StatsResult};

/// Threshold below which the shape parameter ξ is treated as the limiting case
/// (Gumbel for GEV, exponential for GPD).
const XI_EPS: f64 = 1e-8;

// ─────────────────────────────────────────────────────────────────────────────
// Generalised Extreme Value distribution
// ─────────────────────────────────────────────────────────────────────────────

/// Generalised Extreme Value distribution GEV(μ, σ, ξ).
#[derive(Debug, Clone, Copy)]
pub struct Gev {
    /// Location parameter μ.
    pub loc: f64,
    /// Scale parameter σ > 0.
    pub scale: f64,
    /// Shape parameter ξ.
    pub shape: f64,
}

impl Gev {
    /// Construct a GEV distribution, validating σ > 0.
    pub fn new(loc: f64, scale: f64, shape: f64) -> StatsResult<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "GEV: scale must be > 0; got {scale}"
            )));
        }
        if !loc.is_finite() || !shape.is_finite() {
            return Err(StatsError::InvalidDistributionParameter(
                "GEV: loc and shape must be finite".to_string(),
            ));
        }
        Ok(Self { loc, scale, shape })
    }

    /// Reduced variable `t(x) = [1 + ξ(x − μ)/σ]^(−1/ξ)` (or its ξ→0 limit).
    ///
    /// Returns `None` when `x` is outside the support.
    fn reduced(&self, x: f64) -> Option<f64> {
        let z = (x - self.loc) / self.scale;
        if self.shape.abs() < XI_EPS {
            Some((-z).exp())
        } else {
            let arg = 1.0 + self.shape * z;
            if arg <= 0.0 {
                None
            } else {
                Some(arg.powf(-1.0 / self.shape))
            }
        }
    }

    /// Cumulative distribution function `F(x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        match self.reduced(x) {
            Some(t) => (-t).exp(),
            None => {
                // Outside support: saturate to 0 or 1 depending on the tail.
                let z = (x - self.loc) / self.scale;
                if self.shape > 0.0 {
                    // Lower bound at x = μ − σ/ξ; below it F = 0.
                    if 1.0 + self.shape * z <= 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    // ξ < 0: upper bound at x = μ − σ/ξ; above it F = 1.
                    1.0
                }
            }
        }
    }

    /// Probability density function `f(x)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if self.shape.abs() < XI_EPS {
            let t = (-z).exp();
            (1.0 / self.scale) * t * (-t).exp()
        } else {
            let arg = 1.0 + self.shape * z;
            if arg <= 0.0 {
                return 0.0;
            }
            let t = arg.powf(-1.0 / self.shape);
            (1.0 / self.scale) * arg.powf(-1.0 / self.shape - 1.0) * (-t).exp()
        }
    }

    /// Quantile (inverse CDF) for probability `p ∈ (0, 1)`.
    pub fn quantile(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        if p == 0.0 {
            return Ok(if self.shape > 0.0 {
                self.loc - self.scale / self.shape
            } else {
                f64::NEG_INFINITY
            });
        }
        if p == 1.0 {
            return Ok(if self.shape < 0.0 {
                self.loc - self.scale / self.shape
            } else {
                f64::INFINITY
            });
        }
        let ln_p = -(p.ln()); // y = −ln F = t
        let q = if self.shape.abs() < XI_EPS {
            self.loc - self.scale * ln_p.ln()
        } else {
            self.loc + (self.scale / self.shape) * (ln_p.powf(-self.shape) - 1.0)
        };
        Ok(q)
    }

    /// The `T`-block return level, i.e. the `1 − 1/T` quantile.
    pub fn return_level(&self, return_period: f64) -> StatsResult<f64> {
        if !(return_period > 1.0 && return_period.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "return_period".to_string(),
                reason: format!("must be > 1 and finite; got {return_period}"),
            });
        }
        self.quantile(1.0 - 1.0 / return_period)
    }

    /// Fit GEV parameters by Probability-Weighted Moments (Hosking–Wallis–Wood).
    pub fn fit_pwm(data: &[f64]) -> StatsResult<Self> {
        let n = data.len();
        if n < 3 {
            return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
        }
        let mut sorted = data.to_vec();
        if sorted.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::NonFiniteValue(0));
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Unbiased PWMs b₀, b₁, b₂ (ascending-order plotting positions).
        let nf = n as f64;
        let b0 = sorted.iter().sum::<f64>() / nf;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for (j, &x) in sorted.iter().enumerate() {
            let jf = (j + 1) as f64; // 1-based rank
            // b1 weight: (j−1)/(n−1); b2 weight: (j−1)(j−2)/[(n−1)(n−2)]
            b1 += (jf - 1.0) / (nf - 1.0) * x;
            b2 += (jf - 1.0) * (jf - 2.0) / ((nf - 1.0) * (nf - 2.0)) * x;
        }
        b1 /= nf;
        b2 /= nf;

        // Shape via the ratio of PWMs (Hosking, Wallis & Wood 1985):
        //   (2 b₁ − b₀) / (3 b₂ − b₀) = (2^k − 1) / (3^k − 1),
        // where `k` is the shape in the *Hosking/Jenkinson* sign convention.
        let c = (2.0 * b1 - b0) / (3.0 * b2 - b0) - std::f64::consts::LN_2 / 3.0_f64.ln();
        // HWW (1985) rational approximation, accurate for −0.5 < k < 0.5.
        let k_hosking = 7.859 * c + 2.9554 * c * c;
        // The CDF here uses the von Mises / Coles convention, in which the shape
        // is the negation of Hosking's `k` (ξ_Coles = −k).
        let shape = -k_hosking;

        // The relevant GEV PWM identities (Hosking, Wallis & Wood 1985) are
        //   b₀          = μ − (σ/ξ)[1 − Γ(1 − ξ)],
        //   2 b₁ − b₀   = (σ/ξ) Γ(1 − ξ) (2^ξ − 1),
        // which invert to σ and μ below.
        let (scale, loc);
        if shape.abs() < XI_EPS {
            // Gumbel limit: 2b₁ − b₀ = σ ln 2; b₀ = μ + σγ.
            scale = (2.0 * b1 - b0) / std::f64::consts::LN_2;
            loc = b0 - scale * 0.5772156649015329; // Euler–Mascheroni γ
        } else {
            let gamma_1m = gamma(1.0 - shape);
            scale = (2.0 * b1 - b0) * shape / (gamma_1m * (2.0_f64.powf(shape) - 1.0));
            loc = b0 + scale * (1.0 - gamma_1m) / shape;
        }

        Self::new(loc, scale, shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generalised Pareto distribution
// ─────────────────────────────────────────────────────────────────────────────

/// Generalised Pareto Distribution GPD(σ, ξ) of exceedances over a threshold.
#[derive(Debug, Clone, Copy)]
pub struct Gpd {
    /// Scale parameter σ > 0.
    pub scale: f64,
    /// Shape parameter ξ.
    pub shape: f64,
}

impl Gpd {
    /// Construct a GPD, validating σ > 0.
    pub fn new(scale: f64, shape: f64) -> StatsResult<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(StatsError::InvalidDistributionParameter(format!(
                "GPD: scale must be > 0; got {scale}"
            )));
        }
        if !shape.is_finite() {
            return Err(StatsError::InvalidDistributionParameter(
                "GPD: shape must be finite".to_string(),
            ));
        }
        Ok(Self { scale, shape })
    }

    /// CDF `F(y)` of an exceedance `y = x − u ≥ 0`.
    #[must_use]
    pub fn cdf(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        if self.shape.abs() < XI_EPS {
            1.0 - (-y / self.scale).exp()
        } else {
            let arg = 1.0 + self.shape * y / self.scale;
            if arg <= 0.0 {
                // ξ < 0: above the upper endpoint y = −σ/ξ, F saturates to 1.
                1.0
            } else {
                1.0 - arg.powf(-1.0 / self.shape)
            }
        }
    }

    /// PDF `f(y)` of an exceedance `y ≥ 0`.
    #[must_use]
    pub fn pdf(&self, y: f64) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        if self.shape.abs() < XI_EPS {
            (1.0 / self.scale) * (-y / self.scale).exp()
        } else {
            let arg = 1.0 + self.shape * y / self.scale;
            if arg <= 0.0 {
                return 0.0;
            }
            (1.0 / self.scale) * arg.powf(-1.0 / self.shape - 1.0)
        }
    }

    /// Quantile (inverse CDF) for probability `p ∈ [0, 1)`.
    pub fn quantile(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(if self.shape < 0.0 {
                -self.scale / self.shape
            } else {
                f64::INFINITY
            });
        }
        let q = if self.shape.abs() < XI_EPS {
            -self.scale * (1.0 - p).ln()
        } else {
            (self.scale / self.shape) * ((1.0 - p).powf(-self.shape) - 1.0)
        };
        Ok(q)
    }

    /// Mean of the exceedance distribution, `σ/(1 − ξ)` for ξ < 1 (else infinite).
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.shape < 1.0 {
            self.scale / (1.0 - self.shape)
        } else {
            f64::INFINITY
        }
    }

    /// Return level on the *exceedance* scale for `return_period` exceedances,
    /// i.e. the `1 − 1/T` quantile of the GPD.
    pub fn return_level(&self, return_period: f64) -> StatsResult<f64> {
        if !(return_period > 1.0 && return_period.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "return_period".to_string(),
                reason: format!("must be > 1 and finite; got {return_period}"),
            });
        }
        self.quantile(1.0 - 1.0 / return_period)
    }

    /// Fit GPD parameters by Probability-Weighted Moments (Hosking & Wallis 1987).
    ///
    /// `exceedances` are the values `y = x − u ≥ 0`.
    pub fn fit_pwm(exceedances: &[f64]) -> StatsResult<Self> {
        let n = exceedances.len();
        if n < 2 {
            return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
        }
        if exceedances.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(StatsError::InvalidParameter {
                name: "exceedances".to_string(),
                reason: "exceedances must be finite and ≥ 0".to_string(),
            });
        }
        let mut sorted = exceedances.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let nf = n as f64;

        // PWMs of the GPD: a_r = E[X (1 − F)^r] = σ / [(r + 1)(r + 1 − ξ)] under
        // the Coles parameterisation F(y) = 1 − (1 + ξ y/σ)^(−1/ξ). Hence
        //   a₀ = σ/(1 − ξ),   a₁ = σ/[2(2 − ξ)].
        // The plotting-position estimator uses p_j = (j − 0.35)/n.
        let a0 = sorted.iter().sum::<f64>() / nf;
        let mut a1 = 0.0;
        for (j, &x) in sorted.iter().enumerate() {
            let pj = ((j + 1) as f64 - 0.35) / nf;
            a1 += (1.0 - pj) * x;
        }
        a1 /= nf;

        // Inverting the two moment equations:
        //   ξ = (a₀ − 4 a₁) / (a₀ − 2 a₁),   σ = 2 a₀ a₁ / (a₀ − 2 a₁).
        let denom = a0 - 2.0 * a1;
        if denom.abs() < 1e-300 {
            return Err(StatsError::NumericalInstability(
                "degenerate PWM denominator in GPD fit".to_string(),
            ));
        }
        let shape = (a0 - 4.0 * a1) / denom;
        let scale = 2.0 * a0 * a1 / denom;
        if scale <= 0.0 || !scale.is_finite() {
            return Err(StatsError::NumericalInstability(
                "non-positive scale estimate in GPD fit".to_string(),
            ));
        }
        Self::new(scale, shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma function (Lanczos approximation) — needed for the GEV PWM estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Gamma function Γ(x) via the Lanczos approximation (g = 7, n = 9).
fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula Γ(x)Γ(1−x) = π / sin(πx).
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        let t = x + G + 0.5;
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn gev_gumbel_limit_matches_closed_form() {
        // ξ = 0 must equal exp(−exp(−(x−μ)/σ)).
        let g = Gev::new(1.0, 2.0, 0.0).expect("ok");
        for &x in &[-3.0_f64, 0.0, 1.0, 4.0, 8.0] {
            let z = (x - 1.0) / 2.0;
            let expected = (-(-z).exp()).exp();
            assert!((g.cdf(x) - expected).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn gpd_exponential_limit_matches_closed_form() {
        // ξ = 0 must equal 1 − exp(−y/σ).
        let g = Gpd::new(1.5, 0.0).expect("ok");
        for &y in &[0.0_f64, 0.5, 1.0, 3.0, 7.0] {
            let expected = 1.0 - (-y / 1.5).exp();
            assert!((g.cdf(y) - expected).abs() < 1e-12, "y={y}");
        }
    }

    #[test]
    fn gev_cdf_monotone_and_in_unit_interval() {
        let g = Gev::new(0.0, 1.0, 0.2).expect("ok");
        let mut prev = -1.0;
        let mut x = -2.0;
        while x < 20.0 {
            let f = g.cdf(x);
            assert!((0.0..=1.0).contains(&f), "F({x}) = {f}");
            assert!(f >= prev - 1e-12, "not monotone at {x}");
            prev = f;
            x += 0.1;
        }
    }

    #[test]
    fn gev_quantile_inverts_cdf() {
        let g = Gev::new(2.0, 1.5, 0.15).expect("ok");
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95, 0.99] {
            let x = g.quantile(p).expect("q ok");
            assert!((g.cdf(x) - p).abs() < 1e-8, "p={p} x={x} cdf={}", g.cdf(x));
        }
    }

    #[test]
    fn gpd_quantile_inverts_cdf() {
        let g = Gpd::new(2.0, 0.3).expect("ok");
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let y = g.quantile(p).expect("q ok");
            assert!((g.cdf(y) - p).abs() < 1e-8, "p={p} y={y}");
        }
    }

    #[test]
    fn gev_negative_shape_saturates_outside_support() {
        // ξ < 0 ⇒ finite upper endpoint at x = μ − σ/ξ.
        let g = Gev::new(0.0, 1.0, -0.25).expect("ok");
        let upper = 0.0 - 1.0 / -0.25; // = 4.0
        assert!(g.cdf(upper + 5.0) >= 1.0 - 1e-12);
        // ξ > 0 ⇒ finite lower endpoint; below it F = 0.
        let g2 = Gev::new(0.0, 1.0, 0.25).expect("ok");
        let lower = 0.0 - 1.0 / 0.25; // = −4.0
        assert!(g2.cdf(lower - 5.0) <= 1e-12);
    }

    #[test]
    fn gpd_negative_shape_saturates() {
        let g = Gpd::new(1.0, -0.5).expect("ok");
        let upper = -1.0 / -0.5; // = 2.0
        assert!((g.cdf(upper + 3.0) - 1.0).abs() < 1e-12);
        assert!(g.pdf(upper + 3.0) <= 1e-12);
    }

    #[test]
    fn return_level_increases_with_period() {
        let g = Gev::new(10.0, 3.0, 0.1).expect("ok");
        let r10 = g.return_level(10.0).expect("ok");
        let r50 = g.return_level(50.0).expect("ok");
        let r100 = g.return_level(100.0).expect("ok");
        assert!(r10 < r50 && r50 < r100, "{r10} {r50} {r100}");

        let gp = Gpd::new(2.0, 0.2).expect("ok");
        let p10 = gp.return_level(10.0).expect("ok");
        let p100 = gp.return_level(100.0).expect("ok");
        assert!(p10 < p100, "{p10} {p100}");
    }

    #[test]
    fn gpd_mean_matches_formula_and_samples() {
        let scale = 2.0;
        let shape = 0.3;
        let g = Gpd::new(scale, shape).expect("ok");
        let analytic = g.mean();
        assert!((analytic - scale / (1.0 - shape)).abs() < 1e-12);

        // Inverse-CDF sampling and compare the empirical mean.
        let mut rng = LcgRng::new(123);
        let n = 200_000;
        let mut acc = 0.0;
        for _ in 0..n {
            let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
            acc += g.quantile(u).expect("q ok");
        }
        let empirical = acc / n as f64;
        assert!(
            (empirical - analytic).abs() < 0.05,
            "{empirical} vs {analytic}"
        );
    }

    #[test]
    fn gpd_pwm_recovers_parameters() {
        // Generate GPD data by inverse-CDF sampling and recover (σ, ξ) by PWM.
        let scale = 1.5;
        let shape = 0.2;
        let g = Gpd::new(scale, shape).expect("ok");
        let mut rng = LcgRng::new(2024);
        let n = 20_000;
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
                g.quantile(u).expect("q ok")
            })
            .collect();
        let fit = Gpd::fit_pwm(&data).expect("fit ok");
        assert!(
            (fit.scale - scale).abs() < 0.15,
            "scale {} vs {scale}",
            fit.scale
        );
        assert!(
            (fit.shape - shape).abs() < 0.1,
            "shape {} vs {shape}",
            fit.shape
        );
    }

    #[test]
    fn gev_pwm_recovers_parameters() {
        // Generate GEV data by inverse-CDF sampling and recover (μ, σ, ξ) by PWM.
        let loc = 5.0;
        let scale = 2.0;
        let shape = 0.1;
        let g = Gev::new(loc, scale, shape).expect("ok");
        let mut rng = LcgRng::new(7);
        let n = 20_000;
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
                g.quantile(u).expect("q ok")
            })
            .collect();
        let fit = Gev::fit_pwm(&data).expect("fit ok");
        assert!((fit.loc - loc).abs() < 0.2, "loc {} vs {loc}", fit.loc);
        assert!(
            (fit.scale - scale).abs() < 0.2,
            "scale {} vs {scale}",
            fit.scale
        );
        assert!(
            (fit.shape - shape).abs() < 0.12,
            "shape {} vs {shape}",
            fit.shape
        );
    }

    #[test]
    fn gamma_known_values() {
        // Γ(1) = 1, Γ(5) = 24, Γ(0.5) = √π.
        assert!((gamma(1.0) - 1.0).abs() < 1e-9);
        assert!((gamma(5.0) - 24.0).abs() < 1e-6);
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn rejects_invalid_scale() {
        assert!(Gev::new(0.0, -1.0, 0.0).is_err());
        assert!(Gpd::new(0.0, 0.0).is_err());
    }
}
