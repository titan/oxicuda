//! Lp-stable random projection sketch for streaming Lp norm estimation (Indyk 2006).
//!
//! Handles STREAMING TURNSTILE updates `(index, delta)` where `x[index] += delta`.
//! Supports arbitrary `p ∈ (0, 2]` via the Chambers-Mallows-Stuck stable distribution.
//!
//! For `p = 1`: Cauchy via inverse-CDF `tan(π(u − 0.5))`, median constant = 1.0.
//! For `p = 2`: Gaussian `N(0,1)`, median constant ≈ 0.6745.
//! For general `p ∈ (0, 2)`: Chambers-Mallows-Stuck (1976) algorithm.
//!
//! Estimate: `‖x‖_p ≈ median(|S[j]|) / c_p` where `c_p = median(|Stable(p)|)`.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Number of Monte-Carlo samples used to estimate the median normalisation constant
/// for general-p stable distributions (not needed for p=1 or p=2 which are exact).
const MEDIAN_SAMPLES: usize = 10_000;

/// Tolerance for comparing p to boundary values.
const P_EPS: f64 = 1e-9;

/// Draw one sample from the standard `p`-stable distribution using the
/// Chambers-Mallows-Stuck (1976) algorithm.
///
/// Valid for `p ∈ (0, 2)`.  For the boundary cases p→1 and p→2 the formula
/// degenerates, so callers should use the closed-form Cauchy / Gaussian
/// generators for those exact values.
fn sample_stable_p(p: f64, rng: &mut LcgRng) -> f64 {
    let pi_2 = std::f64::consts::FRAC_PI_2;
    // U ~ Uniform(-π/2, π/2)
    let u = rng.next_f64() * std::f64::consts::PI - pi_2;
    // E ~ Exp(1): draw from -ln(Uniform(0,1))
    let e = -(rng.next_f64().max(1e-300).ln());
    let sin_pu = (p * u).sin();
    let cos_u = u.cos();
    // Guard against cos(U) ≈ 0 at u ≈ ±π/2.
    let cos_u_safe = if cos_u.abs() < 1e-300 {
        1e-300_f64.copysign(cos_u)
    } else {
        cos_u
    };
    let cos_u_minus_pu = ((1.0 - p) * u).cos();
    // X = sin(pU) / cos(U)^(1/p) * (cos((1-p)U) / E)^((1-p)/p)
    sin_pu / cos_u_safe.powf(1.0 / p) * (cos_u_minus_pu / e).powf((1.0 - p) / p)
}

/// Estimate `median(|Stable(p)|)` via Monte Carlo with `n_samples` draws.
fn estimate_median_const(p: f64, n_samples: usize, rng: &mut LcgRng) -> f64 {
    let mut samples: Vec<f64> = (0..n_samples)
        .map(|_| sample_stable_p(p, rng).abs())
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Return the lower-median.
    samples[n_samples / 2]
}

/// Lp-stable random projection sketch for streaming turnstile Lp norm estimation.
///
/// Maintains `width` projections, each accumulating `Σ delta * g_{ij}`.
/// The estimate is `median(|state[j]|) / median_const`.
#[derive(Debug, Clone)]
pub struct LpStableSketch {
    /// The Lp norm order: p ∈ (0, 2].
    pub p: f64,
    /// Number of independent projections.
    pub width: usize,
    /// Dimension of the input vector.
    pub dim: usize,
    /// Accumulated projection values, length `width`.
    pub state: Vec<f64>,
    /// Projection matrix, row-major: `coeffs[j * dim + i] = g_{ij}`.
    pub coeffs: Vec<f64>,
    /// Normalisation constant: `median(|Stable(p)|)`.
    median_const: f64,
}

impl LpStableSketch {
    /// Construct a new Lp-stable sketch.
    ///
    /// # Parameters
    /// - `p`: Norm order in `(0, 2]`.
    /// - `dim`: Dimension of the input vector.
    /// - `width`: Number of independent projections (more = more accurate).
    /// - `rng`: Seeded random number generator for coefficient generation.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidParameter`] if `p ∉ (0, 2]`, `dim = 0`, or `width = 0`.
    pub fn new(p: f64, dim: usize, width: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if p <= 0.0 || p > 2.0 + P_EPS {
            return Err(SketchError::InvalidParameter {
                name: "p".to_string(),
                reason: "must be in (0, 2]".to_string(),
            });
        }
        if dim == 0 {
            return Err(SketchError::InvalidParameter {
                name: "dim".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if width == 0 {
            return Err(SketchError::InvalidParameter {
                name: "width".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        // Clamp p to [0, 2] to avoid floating-point overflow from the +P_EPS allowance.
        let p = p.min(2.0);

        // Generate the projection matrix coefficients using the appropriate p-stable
        // distribution. Row-major: coeffs[j * dim + i] = g_{ij}.
        let total = width * dim;
        let mut coeffs = vec![0.0f64; total];

        let is_l1 = (p - 1.0).abs() < P_EPS;
        let is_l2 = (p - 2.0).abs() < P_EPS;

        if is_l1 {
            // Cauchy via inverse CDF: tan(π(u - 0.5))
            for v in coeffs.iter_mut() {
                let u = rng.next_f64();
                *v = (std::f64::consts::PI * (u - 0.5)).tan();
            }
        } else if is_l2 {
            // Standard Gaussian N(0,1) via Box-Muller (available from LcgRng).
            for v in coeffs.iter_mut() {
                *v = rng.next_normal();
            }
        } else {
            // General p ∈ (0, 2) via Chambers-Mallows-Stuck.
            for v in coeffs.iter_mut() {
                *v = sample_stable_p(p, rng);
            }
        }

        // Compute the normalisation constant c_p = median(|Stable(p)|).
        let median_const = if is_l1 {
            // median(|Cauchy|) = tan(π/4) = 1.0
            1.0_f64
        } else if is_l2 {
            // median(|N(0,1)|) ≈ 0.6744897501960817
            0.6745_f64
        } else {
            // Monte-Carlo estimate for general p.
            estimate_median_const(p, MEDIAN_SAMPLES, rng)
        };

        Ok(Self {
            p,
            width,
            dim,
            state: vec![0.0f64; width],
            coeffs,
            median_const,
        })
    }

    /// Streaming turnstile update: `x[idx] += delta`.
    ///
    /// Silently ignores updates where `idx >= dim` (out-of-bounds indices do not
    /// affect other projections).
    pub fn update(&mut self, idx: usize, delta: f64) {
        if idx >= self.dim {
            return;
        }
        for j in 0..self.width {
            self.state[j] += delta * self.coeffs[j * self.dim + idx];
        }
    }

    /// Estimate the Lp norm of the implicit vector `x`.
    ///
    /// Returns `median(|state[j]|) / median_const`.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let mut abs_vals: Vec<f64> = self.state.iter().map(|s| s.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = abs_vals[abs_vals.len() / 2];
        if self.median_const.abs() < 1e-300 {
            return 0.0;
        }
        median / self.median_const
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validation ──────────────────────────────────────────────────────────

    #[test]
    fn lp_stable_invalid_p_zero() {
        let mut rng = LcgRng::new(1);
        assert!(LpStableSketch::new(0.0, 10, 20, &mut rng).is_err());
    }

    #[test]
    fn lp_stable_invalid_p_too_large() {
        let mut rng = LcgRng::new(2);
        assert!(LpStableSketch::new(2.5, 10, 20, &mut rng).is_err());
    }

    #[test]
    fn lp_stable_invalid_dim() {
        let mut rng = LcgRng::new(3);
        assert!(LpStableSketch::new(1.0, 0, 20, &mut rng).is_err());
    }

    #[test]
    fn lp_stable_invalid_width() {
        let mut rng = LcgRng::new(4);
        assert!(LpStableSketch::new(1.0, 10, 0, &mut rng).is_err());
    }

    // ── L1 estimate ─────────────────────────────────────────────────────────

    #[test]
    fn lp_stable_l1_estimate() {
        // 100 items each with value 1.0 → true L1 norm = 100.0.
        // Use a generous 30% tolerance.
        let mut rng = LcgRng::new(42);
        let n_items = 100usize;
        let mut sketch = LpStableSketch::new(1.0, n_items, 301, &mut rng).expect("ok");
        for i in 0..n_items {
            sketch.update(i, 1.0);
        }
        let est = sketch.estimate();
        let true_norm = 100.0_f64;
        assert!(
            (est - true_norm).abs() / true_norm < 0.30,
            "L1 estimate {est} not within 30% of {true_norm}"
        );
    }

    // ── L2 estimate ─────────────────────────────────────────────────────────

    #[test]
    fn lp_stable_l2_estimate() {
        // Insert values 1..=10, true L2 = sqrt(1^2 + 2^2 + ... + 10^2) = sqrt(385) ≈ 19.62.
        let mut rng = LcgRng::new(99);
        let dim = 10usize;
        let mut sketch = LpStableSketch::new(2.0, dim, 401, &mut rng).expect("ok");
        for i in 0..dim {
            sketch.update(i, (i + 1) as f64);
        }
        let true_l2 = (385.0_f64).sqrt(); // ≈ 19.62
        let est = sketch.estimate();
        assert!(
            (est - true_l2).abs() / true_l2 < 0.30,
            "L2 estimate {est} not within 30% of {true_l2}"
        );
    }

    // ── negative deltas ─────────────────────────────────────────────────────

    #[test]
    fn lp_stable_update_negative_delta() {
        // Insert 50 with delta=2.0 then 50 with delta=-2.0 → zero vector.
        let mut rng = LcgRng::new(7);
        let dim = 50usize;
        let mut sketch = LpStableSketch::new(1.0, dim, 201, &mut rng).expect("ok");
        for i in 0..dim {
            sketch.update(i, 2.0);
        }
        for i in 0..dim {
            sketch.update(i, -2.0);
        }
        // After cancellation the norm should be very small.
        let est = sketch.estimate();
        assert!(
            est < 5.0,
            "expected near-zero estimate after cancellation, got {est}"
        );
    }

    // ── zero vector ─────────────────────────────────────────────────────────

    #[test]
    fn lp_stable_zero_vector_estimate() {
        let mut rng = LcgRng::new(13);
        let sketch = LpStableSketch::new(1.0, 20, 51, &mut rng).expect("ok");
        // No updates → all-zero state.
        let est = sketch.estimate();
        assert!(est.abs() < 1e-10, "expected 0 for zero vector, got {est}");
    }

    // ── general p ───────────────────────────────────────────────────────────

    #[test]
    fn lp_stable_general_p_1_5() {
        // p=1.5: draw from 1.5-stable distribution. Estimate should be finite and positive.
        let mut rng = LcgRng::new(55);
        let dim = 40usize;
        let mut sketch = LpStableSketch::new(1.5, dim, 201, &mut rng).expect("ok");
        for i in 0..dim {
            sketch.update(i, 1.0);
        }
        let est = sketch.estimate();
        assert!(
            est.is_finite(),
            "p=1.5 estimate should be finite, got {est}"
        );
        assert!(est > 0.0, "p=1.5 estimate should be positive, got {est}");
    }

    // ── width affects accuracy ───────────────────────────────────────────────

    #[test]
    fn lp_stable_width_affects_accuracy() {
        // Wider sketch should achieve smaller relative error on average.
        // Run multiple seeds and check that wider beats narrow.
        let dim = 20usize;
        let true_norm = dim as f64; // all ones, L1 = 20.0

        let mut wide_err_sum = 0.0f64;
        let mut narrow_err_sum = 0.0f64;
        let n_trials = 10usize;

        for trial in 0..n_trials {
            let seed = trial as u64 * 1_234_567 + 98765;

            let mut rng_wide = LcgRng::new(seed);
            let mut wide = LpStableSketch::new(1.0, dim, 501, &mut rng_wide).expect("ok");
            for i in 0..dim {
                wide.update(i, 1.0);
            }
            wide_err_sum += (wide.estimate() - true_norm).abs() / true_norm;

            let mut rng_narrow = LcgRng::new(seed);
            let mut narrow = LpStableSketch::new(1.0, dim, 21, &mut rng_narrow).expect("ok");
            for i in 0..dim {
                narrow.update(i, 1.0);
            }
            narrow_err_sum += (narrow.estimate() - true_norm).abs() / true_norm;
        }

        let wide_avg = wide_err_sum / n_trials as f64;
        let narrow_avg = narrow_err_sum / n_trials as f64;
        assert!(
            wide_avg < narrow_avg,
            "wider sketch (avg err {wide_avg:.3}) should outperform narrow (avg err {narrow_avg:.3})"
        );
    }

    // ── p exactly at boundary 2.0 ────────────────────────────────────────────

    #[test]
    fn lp_stable_p_exactly_two() {
        let mut rng = LcgRng::new(22);
        // p = 2.0 + P_EPS should be clamped to 2.0 and succeed.
        let result = LpStableSketch::new(2.0, 10, 51, &mut rng);
        assert!(result.is_ok(), "p=2.0 must succeed");
    }
}
