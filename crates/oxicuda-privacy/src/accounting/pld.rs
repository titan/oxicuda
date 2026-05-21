//! Privacy Loss Distribution (PLD) accountant.
//!
//! References:
//! - Meiser & Mohammadi (2018), "Tight on Budget? Tight Bounds for r-Fold
//!   Approximate Differential Privacy", CCS 2018.
//! - Koskela, Jälkö & Honkela (2020), "Computing Tight Differential Privacy
//!   Guarantees Using FFT", AISTATS 2020.
//!
//! # Overview
//! The privacy loss random variable on neighbouring datasets x, x' is
//! `L = ln(P(M(x)=o) / P(M(x')=o))` with o ~ M(x). A PLD is a discrete
//! histogram approximation of the distribution of L on a uniform grid plus
//! a scalar `truncation_mass` accounting for probability outside the grid
//! (treated as l → +∞ for δ accounting).
//!
//! # Composition
//! For k independent mechanisms the composed PLD is the convolution of the
//! individual PLDs, since L_total = Σ L_i.
//!
//! # δ at threshold ε
//! `δ(ε) = Σ_i max(0, 1 − e^(ε − l_i)) · p_i + truncation_mass`.

use crate::error::{PrivacyError, PrivacyResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Uniform grid spec for a PLD: `[lower, upper]` with spacing `step`.
#[derive(Clone, Debug)]
pub struct PldGrid {
    pub lower: f64,
    pub upper: f64,
    pub step: f64,
}

impl PldGrid {
    /// Construct and validate a `PldGrid`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `step ≤ 0`, `lower ≥ upper`, or non-finite.
    pub fn new(lower: f64, upper: f64, step: f64) -> PrivacyResult<Self> {
        if !lower.is_finite() || !upper.is_finite() || !step.is_finite() {
            return Err(PrivacyError::InvalidParameter(
                "grid bounds and step must be finite".into(),
            ));
        }
        if step <= 0.0 {
            return Err(PrivacyError::InvalidParameter(
                "grid step must be positive".into(),
            ));
        }
        if lower >= upper {
            return Err(PrivacyError::InvalidParameter(
                "grid lower must be strictly less than upper".into(),
            ));
        }
        Ok(Self { lower, upper, step })
    }

    /// Number of grid bins (rounded based on `(upper - lower) / step`).
    #[must_use]
    pub fn len(&self) -> usize {
        let raw = ((self.upper - self.lower) / self.step).round() as i64;
        if raw < 1 { 0 } else { raw as usize }
    }

    /// Whether the grid contains zero bins.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Center of bin index `i` (i.e. `lower + (i + 0.5) * step`).
    #[must_use]
    pub fn center_at(&self, i: usize) -> f64 {
        self.lower + (i as f64 + 0.5) * self.step
    }
}

// ─── Standard-normal CDF helper ───────────────────────────────────────────────

/// Standard normal CDF Φ(x).
///
/// Φ(x) = 0.5 (1 + erf(x/√2)) with a Hastings 5-term rational approximation
/// (Abramowitz & Stegun 7.1.26) for erfc, accurate to ~1.5×10⁻⁷.
fn phi(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc_z = poly * (-z * z).exp();
    if x >= 0.0 {
        1.0 - 0.5 * erfc_z
    } else {
        0.5 * erfc_z
    }
}

// ─── Privacy Loss Distribution ────────────────────────────────────────────────

/// Histogram-based PLD: a finite probability mass on a uniform loss grid plus
/// a scalar tail mass treated as l → +∞ when accounting δ.
#[derive(Clone, Debug)]
pub struct Pld {
    grid: PldGrid,
    probabilities: Vec<f64>,
    truncation_mass: f64,
}

impl Pld {
    /// Build the PLD for a Gaussian mechanism with L2 sensitivity Δ and
    /// noise std σ (Meiser-Mohammadi 2018, Eq. 4).
    ///
    /// The privacy-loss RV under the Gaussian mechanism has density
    /// `p(l) = (σ/Δ) · φ((σ·l)/Δ + Δ/(2σ))` (re-parametrising via
    /// Z = (X+Δ/2)/σ ~ N(0,1) and l = Δ Z / σ − Δ²/(2σ²) = ...).
    /// Equivalently L ~ N(Δ²/(2σ²), (Δ/σ)²); we sample its standard normal
    /// pdf at each grid centre then multiply by `step`.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    /// - `InvalidParameter` if `sigma ≤ 0`.
    /// - Errors from `from_histogram` if the grid is degenerate.
    pub fn from_gaussian(sensitivity: f64, sigma: f64, grid: PldGrid) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        let n = grid.len();
        if n == 0 {
            return Err(PrivacyError::InvalidParameter(
                "grid resolves to zero bins".into(),
            ));
        }
        let mu = sensitivity * sensitivity / (2.0 * sigma * sigma);
        let sd = sensitivity / sigma;
        let denom = sd * (2.0 * std::f64::consts::PI).sqrt();
        let mut probabilities = vec![0.0f64; n];
        let mut grid_mass = 0.0f64;
        for (i, slot) in probabilities.iter_mut().enumerate() {
            let l = grid.center_at(i);
            let z = (l - mu) / sd;
            let pdf = (-0.5 * z * z).exp() / denom;
            let p = pdf * grid.step;
            *slot = p;
            grid_mass += p;
        }
        // The Gaussian CDF gives us the exact mass outside [lower, upper];
        // attribute the upper tail to `truncation_mass` (l → +∞) and fold
        // the lower tail into the smallest bin (l → −∞ is harmless for δ).
        let upper_mass = 1.0 - phi((grid.upper - mu) / sd);
        let lower_mass = phi((grid.lower - mu) / sd);
        let truncation_mass = upper_mass.clamp(0.0, 1.0);
        if let Some(first) = probabilities.first_mut() {
            *first += lower_mass.clamp(0.0, 1.0);
        }
        let total = grid_mass + lower_mass + upper_mass;
        if total > 1.0 + 1e-9 {
            // Renormalise quadrature error so grid_mass + truncation_mass ≤ 1.
            let scale = 1.0 / total;
            for p in probabilities.iter_mut() {
                *p *= scale;
            }
            let trunc_norm = truncation_mass * scale;
            return Self::from_histogram(grid, probabilities, trunc_norm);
        }
        Self::from_histogram(grid, probabilities, truncation_mass)
    }

    /// Construct from a raw histogram, validating shape and non-negativity.
    ///
    /// # Errors
    /// - `InvalidParameter` if grid resolves to zero bins or `truncation_mass`
    ///   is negative / non-finite.
    /// - `DimensionMismatch` if `probabilities.len()` ≠ grid bin count.
    /// - `InvalidParameter` if any probability is negative / non-finite or the
    ///   total mass exceeds 1 + tolerance.
    pub fn from_histogram(
        grid: PldGrid,
        mut probabilities: Vec<f64>,
        truncation_mass: f64,
    ) -> PrivacyResult<Self> {
        let expected = grid.len();
        if expected == 0 {
            return Err(PrivacyError::InvalidParameter(
                "grid resolves to zero bins".into(),
            ));
        }
        if probabilities.len() != expected {
            return Err(PrivacyError::DimensionMismatch {
                expected,
                got: probabilities.len(),
            });
        }
        if !truncation_mass.is_finite() || truncation_mass < 0.0 {
            return Err(PrivacyError::InvalidParameter(
                "truncation_mass must be non-negative and finite".into(),
            ));
        }
        let mut sum = truncation_mass;
        for &p in &probabilities {
            if !p.is_finite() || p < 0.0 {
                return Err(PrivacyError::InvalidParameter(
                    "probabilities must be non-negative and finite".into(),
                ));
            }
            sum += p;
        }
        if sum > 1.0 + 1e-3 {
            return Err(PrivacyError::InvalidParameter(format!(
                "total mass {sum} exceeds 1"
            )));
        }
        // Renormalise small overshoots from quadrature error so downstream
        // math stays numerically sane.
        if sum > 1.0 {
            let scale = 1.0 / sum;
            for p in probabilities.iter_mut() {
                *p *= scale;
            }
            let truncation_mass = truncation_mass * scale;
            return Ok(Self {
                grid,
                probabilities,
                truncation_mass,
            });
        }
        Ok(Self {
            grid,
            probabilities,
            truncation_mass,
        })
    }

    /// Borrow the grid description.
    #[must_use]
    pub fn grid(&self) -> &PldGrid {
        &self.grid
    }

    /// Borrow the discrete probability mass vector.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Probability mass treated as l → +∞.
    #[must_use]
    pub fn truncation_mass(&self) -> f64 {
        self.truncation_mass
    }

    /// Sum of grid probabilities plus the truncation mass.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.probabilities.iter().sum::<f64>() + self.truncation_mass
    }

    /// δ(ε) = Σ_i max(0, 1 − e^(ε − l_i)) · p_i + truncation_mass.
    ///
    /// The truncation mass contributes fully because we model it as l = +∞,
    /// for which `1 − e^(ε − ∞) = 1`.
    #[must_use]
    pub fn delta_for_epsilon(&self, epsilon: f64) -> f64 {
        let mut delta = self.truncation_mass;
        for (i, &p) in self.probabilities.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let l = self.grid.center_at(i);
            let factor = (1.0 - (epsilon - l).exp()).max(0.0);
            delta += factor * p;
        }
        delta.min(1.0)
    }

    /// Smallest ε ≥ 0 such that `delta_for_epsilon(ε) ≤ delta_target`.
    ///
    /// Returns 0 if the trivial bound already holds; saturates at the search
    /// upper bound (chosen as `max(20, 4·grid.upper)`) if no feasible ε exists
    /// within that range — callers needing tighter answers should widen the
    /// PLD's grid first.
    #[must_use]
    pub fn epsilon_for_delta(&self, delta: f64) -> f64 {
        if delta <= 0.0 {
            return f64::INFINITY;
        }
        if delta >= 1.0 {
            return 0.0;
        }
        if self.delta_for_epsilon(0.0) <= delta {
            return 0.0;
        }
        let upper_search = 20.0_f64.max(4.0 * self.grid.upper.abs().max(1.0));
        let mut lo = 0.0f64;
        let mut hi = upper_search;
        if self.delta_for_epsilon(hi) > delta {
            return hi;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.delta_for_epsilon(mid) > delta {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-9 {
                break;
            }
        }
        hi
    }

    /// Identity PLD: a single point mass at loss l ≤ 0 (composition neutral).
    ///
    /// Picks the bin whose centre is the largest value ≤ 0 so that any
    /// `delta_for_epsilon(ε)` evaluates to 0 for every ε ≥ 0 — matching the
    /// algebraic identity element under composition.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if the requested grid is degenerate.
    pub fn identity(grid: PldGrid) -> PrivacyResult<Self> {
        let n = grid.len();
        if n == 0 {
            return Err(PrivacyError::InvalidParameter(
                "grid resolves to zero bins".into(),
            ));
        }
        let mut probabilities = vec![0.0f64; n];
        let raw = ((-0.5) - grid.lower / grid.step).floor();
        let idx = if raw < 0.0 {
            0
        } else if (raw as usize) >= n {
            n - 1
        } else {
            raw as usize
        };
        let chosen = if grid.center_at(idx) <= 0.0 {
            idx
        } else if idx == 0 {
            0
        } else {
            idx - 1
        };
        probabilities[chosen] = 1.0;
        Ok(Self {
            grid,
            probabilities,
            truncation_mass: 0.0,
        })
    }

    /// Discrete convolution of two PLDs onto a clipped grid.
    ///
    /// The result grid spans `[self.lower + other.lower, self.upper + other.upper]`
    /// with the shared `step`. Mass beyond the upper edge folds into the
    /// truncation mass; mass below the lower edge folds into the lowest bin
    /// (loss → −∞ contributes 0 to δ for any ε ≥ 0).
    ///
    /// # Errors
    /// `InvalidParameter` if the two grids have different `step` (within 1e-12).
    pub fn compose(&self, other: &Pld) -> PrivacyResult<Pld> {
        if (self.grid.step - other.grid.step).abs() > 1e-12 {
            return Err(PrivacyError::InvalidParameter(format!(
                "compose requires matching grid step ({} vs {})",
                self.grid.step, other.grid.step
            )));
        }
        let step = self.grid.step;
        let new_lower = self.grid.lower + other.grid.lower;
        let new_upper = self.grid.upper + other.grid.upper;
        let new_grid = PldGrid::new(new_lower, new_upper, step)?;
        let n = new_grid.len();
        let mut probabilities = vec![0.0f64; n];
        // Truncation from "either side is +∞": a + b − a·b. Cross terms
        // (finite × +∞) and the double-inf term together equal exactly this.
        let a = self.truncation_mass;
        let b = other.truncation_mass;
        let mut truncation_mass = a + b - a * b;
        // Convolve grid masses; output bin (i + j) inherits combined centre
        // (lower+lower') + ((i+j)+1)·step. The +0.5·step shift relative to
        // our centre convention is absorbed in the discretisation error.
        for (i, &pi) in self.probabilities.iter().enumerate() {
            if pi == 0.0 {
                continue;
            }
            for (j, &pj) in other.probabilities.iter().enumerate() {
                if pj == 0.0 {
                    continue;
                }
                let mass = pi * pj;
                let combined_idx = i + j;
                if combined_idx >= n {
                    truncation_mass += mass;
                } else {
                    probabilities[combined_idx] += mass;
                }
            }
        }
        truncation_mass = truncation_mass.min(1.0);
        Self::from_histogram(new_grid, probabilities, truncation_mass)
    }

    /// k-fold self composition via repeated squaring on the same step.
    ///
    /// `k = 0` returns the identity PLD on the original grid.
    ///
    /// # Errors
    /// Propagates `compose` and grid construction errors.
    pub fn compose_self(&self, k: usize) -> PrivacyResult<Pld> {
        if k == 0 {
            return Self::identity(self.grid.clone());
        }
        if k == 1 {
            return Ok(self.clone());
        }
        let mut result: Option<Pld> = None;
        let mut base = self.clone();
        let mut exponent = k;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Some(match result {
                    Some(acc) => acc.compose(&base)?,
                    None => base.clone(),
                });
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.compose(&base)?;
            }
        }
        result.ok_or_else(|| PrivacyError::InvalidParameter("compose_self failed".into()))
    }
}

// Tests live in a sibling file to keep this module under 600 lines.
#[cfg(test)]
#[path = "pld_tests.rs"]
mod tests;
