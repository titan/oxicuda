//! Wrapped Normal distribution on the Poincaré ball (Nagano et al., NeurIPS 2019).
//!
//! Enables generative models in hyperbolic space by pushing a Euclidean Gaussian
//! through the exponential map at a base point μ on the Poincaré ball.
//!
//! # Mathematical Background
//!
//! The Poincaré ball `B^d = {x ∈ ℝ^d : ‖x‖ < 1}` has conformal factor
//! `λ(x) = 2 / (1 - ‖x‖²)` at each point.
//!
//! ## Exponential map
//! For base point μ and tangent vector v ∈ T_μ:
//! ```text
//! exp_μ(v) = μ ⊕ tanh(λ(μ) ‖v‖ / 2) * v / ‖v‖
//! ```
//!
//! ## Logarithmic map
//! For base point μ and point x on the ball:
//! ```text
//! log_μ(x) = (2 / λ(μ)) * arctanh(‖(-μ) ⊕ x‖) * ((-μ) ⊕ x) / ‖(-μ) ⊕ x‖
//! ```
//!
//! ## Wrapped Normal distribution
//! Sample z ~ N(0, σ²I) in ℝ^d, then push through exp_μ(z).
//!
//! ## Log-probability (change-of-variables)
//! ```text
//! log p(x) = log p_E(log_μ(x)) - log |det J_exp_μ(log_μ(x))|
//! ```
//! where the Jacobian determinant is:
//! ```text
//! det J = (sinh(‖v‖) / ‖v‖)^(d-1) * (λ_μ / 2)^d
//! ```
//!
//! # Reference
//! Nagano, Y., Yamaguchi, S., Fujita, Y., Koyama, M. (NeurIPS 2019).
//! *A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning.*

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::riemannian::hyperbolic_poincare::{mobius_add, poincare_project};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Poincaré ball wrapped normal distribution.
#[derive(Debug, Clone)]
pub struct WrappedNormalConfig {
    /// Dimension of the Poincaré ball.
    pub dim: usize,
    /// Base point μ in the Poincaré ball (length dim, ‖μ‖ < 1).
    pub mu: Vec<f64>,
    /// Isotropic standard deviation σ > 0 in the tangent space.
    pub sigma: f64,
    /// Small epsilon to keep points strictly inside the ball.
    pub ball_epsilon: f64,
}

/// Sampled result from a Wrapped Normal distribution.
#[derive(Debug, Clone)]
pub struct WrappedNormalSample {
    /// Sampled point on the Poincaré ball.
    pub point: Vec<f64>,
    /// Euclidean sample v in tangent space T_μ(Poincaré ball) ≅ ℝ^d.
    pub tangent: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core map functions
// ─────────────────────────────────────────────────────────────────────────────

/// Conformal factor λ(x) = 2 / (1 - ‖x‖²) at a point on the Poincaré ball.
#[inline]
fn conformal_factor(x: &[f64]) -> f64 {
    let n2: f64 = x.iter().map(|v| v * v).sum();
    let denom = (1.0 - n2).max(1e-30);
    2.0 / denom
}

/// Euclidean norm of a vector.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Exponential map at base point `mu` in tangent direction `v`.
///
/// `exp_μ(v) = μ ⊕ tanh(λ(μ) ‖v‖ / 2) * v / ‖v‖`
///
/// When ‖v‖ < 1e-15, returns μ (identity limit).
pub fn poincare_exp(mu: &[f64], v: &[f64]) -> ManifoldResult<Vec<f64>> {
    if mu.len() != v.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: mu.len(),
            b: v.len(),
        });
    }
    let v_norm = vec_norm(v);

    // When ‖v‖ is negligibly small, exp_μ(v) = μ (first-order expansion)
    if v_norm < 1.0e-15 {
        return Ok(mu.to_vec());
    }

    let lambda_mu = conformal_factor(mu);
    // tanh(λ_μ ‖v‖ / 2) — clamp argument to avoid overflow
    let arg = (lambda_mu * v_norm / 2.0).min(88.0); // tanh(88) ≈ 1 - 2e-77
    let tanh_arg = arg.tanh();

    // Direction vector: tanh_arg * v / ‖v‖
    let direction: Vec<f64> = v.iter().map(|vi| tanh_arg * vi / v_norm).collect();

    // Möbius addition μ ⊕ direction
    mobius_add(mu, &direction)
}

/// Logarithmic map at base point `mu` for point `x` on the Poincaré ball.
///
/// `log_μ(x) = (2 / λ(μ)) * arctanh(‖(-μ) ⊕ x‖) * ((-μ) ⊕ x) / ‖(-μ) ⊕ x‖`
///
/// When ‖(-μ) ⊕ x‖ < 1e-15, returns the zero vector (antipodal limit).
pub fn poincare_log(mu: &[f64], x: &[f64]) -> ManifoldResult<Vec<f64>> {
    if mu.len() != x.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: mu.len(),
            b: x.len(),
        });
    }
    let d = mu.len();

    // neg_mu = -μ
    let neg_mu: Vec<f64> = mu.iter().map(|v| -v).collect();

    // diff = (-μ) ⊕ x
    let diff = mobius_add(&neg_mu, x)?;
    let diff_norm = vec_norm(&diff);

    // When diff is negligibly small, log_μ(x) = 0 (x ≈ μ)
    if diff_norm < 1.0e-15 {
        return Ok(vec![0.0; d]);
    }

    let lambda_mu = conformal_factor(mu);
    // arctanh is defined on (-1, 1); clamp diff_norm strictly inside
    let clamped_norm = diff_norm.min(1.0 - 1.0e-15);
    let atanh_norm = atanh(clamped_norm);

    // Scale factor: (2 / λ_μ) * arctanh(‖diff‖) / ‖diff‖
    let scale = (2.0 / lambda_mu) * atanh_norm / diff_norm;

    let result: Vec<f64> = diff.iter().map(|di| scale * di).collect();
    Ok(result)
}

/// arctanh(x) = 0.5 * ln((1+x)/(1-x)), computed in a numerically stable way.
#[inline]
fn atanh(x: f64) -> f64 {
    let cx = x.clamp(-1.0 + 1.0e-15, 1.0 - 1.0e-15);
    0.5 * ((1.0 + cx) / (1.0 - cx)).ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validate a [`WrappedNormalConfig`].
///
/// Checks:
/// - `dim` > 0
/// - `mu.len()` == `dim`
/// - `‖μ‖` < 1 (μ is strictly inside the ball)
/// - `sigma` > 0
/// - `ball_epsilon` in (0, 1)
pub fn validate_wrapped_normal_config(config: &WrappedNormalConfig) -> ManifoldResult<()> {
    if config.dim == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "dim".into(),
            reason: "must be > 0".into(),
        });
    }
    if config.mu.len() != config.dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![config.dim],
            got: vec![config.mu.len()],
        });
    }
    let mu_norm2: f64 = config.mu.iter().map(|v| v * v).sum();
    if mu_norm2 >= 1.0 {
        return Err(ManifoldError::ManifoldConstraint(
            "wrapped_normal: mu must be strictly inside the unit ball (‖μ‖ < 1)".into(),
        ));
    }
    if config.sigma <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "sigma".into(),
            reason: "must be > 0".into(),
        });
    }
    if config.ball_epsilon <= 0.0 || config.ball_epsilon >= 1.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "ball_epsilon".into(),
            reason: "must be in (0, 1)".into(),
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a single point from the Wrapped Normal distribution on the Poincaré ball.
///
/// Draws v ~ N(0, σ²I) in ℝ^d, then pushes through `exp_μ(v)`, and finally
/// projects into the ball with the configured `ball_epsilon` margin.
pub fn wrapped_normal_sample(
    config: &WrappedNormalConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<WrappedNormalSample> {
    validate_wrapped_normal_config(config)?;

    let d = config.dim;
    let mut tangent = vec![0.0f64; d];
    for t in tangent.iter_mut() {
        *t = config.sigma * rng.next_normal();
    }

    let point_raw = poincare_exp(&config.mu, &tangent)?;
    // Project strictly inside the ball (handles numerical boundary cases)
    let point = poincare_project(&point_raw, config.ball_epsilon);

    Ok(WrappedNormalSample { point, tangent })
}

/// Sample `n` points from the Wrapped Normal distribution on the Poincaré ball.
///
/// Returns a `Vec` of `n` [`WrappedNormalSample`] structs.
pub fn wrapped_normal_sample_n(
    config: &WrappedNormalConfig,
    n: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<WrappedNormalSample>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    validate_wrapped_normal_config(config)?;

    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        // Reuse sampling without re-validating each iteration
        let d = config.dim;
        let mut tangent = vec![0.0f64; d];
        for t in tangent.iter_mut() {
            *t = config.sigma * rng.next_normal();
        }
        let point_raw = poincare_exp(&config.mu, &tangent)?;
        let point = poincare_project(&point_raw, config.ball_epsilon);
        samples.push(WrappedNormalSample { point, tangent });
    }
    Ok(samples)
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-probability
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the log-likelihood `log p(x | μ, σ)` for a point `x` on the Poincaré ball.
///
/// Uses the change-of-variables formula:
/// ```text
/// log p(x) = log p_E(log_μ(x)) - log |det J_exp_μ(log_μ(x))|
/// ```
///
/// The Euclidean Gaussian log-density is:
/// ```text
/// log p_E(v) = -d/2 * log(2π σ²) - ‖v‖² / (2σ²)
/// ```
///
/// The Jacobian determinant of `exp_μ` at `v`:
/// ```text
/// det J = (sinh(‖v‖) / ‖v‖)^(d-1) * (λ_μ / 2)^d
/// ```
///
/// When ‖v‖ < 1e-15 (x ≈ μ), the Jacobian factor `sinh(‖v‖)/‖v‖ → 1`.
pub fn wrapped_normal_log_prob(config: &WrappedNormalConfig, x: &[f64]) -> ManifoldResult<f64> {
    validate_wrapped_normal_config(config)?;

    if x.len() != config.dim {
        return Err(ManifoldError::DimensionMismatch {
            a: config.dim,
            b: x.len(),
        });
    }

    let d = config.dim as f64;
    let sigma = config.sigma;
    let sigma2 = sigma * sigma;

    // Compute v = log_μ(x)
    let v = poincare_log(&config.mu, x)?;
    let v_norm = vec_norm(&v);
    let v_norm2: f64 = v.iter().map(|z| z * z).sum();

    // Euclidean Gaussian log-density: -d/2 * ln(2πσ²) - ‖v‖²/(2σ²)
    let log_p_e = -0.5 * d * (std::f64::consts::TAU * sigma2).ln() - v_norm2 / (2.0 * sigma2);

    // Jacobian log-determinant: (d-1) * log(sinh(‖v‖)/‖v‖) + d * log(λ_μ/2)
    let log_jac = if v_norm < 1.0e-15 {
        // sinh(‖v‖)/‖v‖ → 1 as ‖v‖ → 0, so log(1) = 0
        let lambda_mu = conformal_factor(&config.mu);
        d * (lambda_mu / 2.0).ln()
    } else {
        let sinh_ratio = v_norm.sinh() / v_norm;
        // Guard against sinh being zero or negative (shouldn't happen for v_norm > 0)
        let log_sinh_ratio = if sinh_ratio > 0.0 {
            sinh_ratio.ln()
        } else {
            return Err(ManifoldError::NumericalInstability(
                "wrapped_normal_log_prob: sinh(‖v‖)/‖v‖ <= 0".into(),
            ));
        };
        let lambda_mu = conformal_factor(&config.mu);
        (d - 1.0) * log_sinh_ratio + d * (lambda_mu / 2.0).ln()
    };

    Ok(log_p_e - log_jac)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config_2d() -> WrappedNormalConfig {
        WrappedNormalConfig {
            dim: 2,
            mu: vec![0.1, 0.2],
            sigma: 0.5,
            ball_epsilon: 1.0e-5,
        }
    }

    fn default_config_origin() -> WrappedNormalConfig {
        WrappedNormalConfig {
            dim: 3,
            mu: vec![0.0, 0.0, 0.0],
            sigma: 0.3,
            ball_epsilon: 1.0e-5,
        }
    }

    // 1. exp_μ(0) = μ
    #[test]
    fn poincare_exp_zero_tangent_is_basepoint() {
        let mu = vec![0.3, -0.1];
        let v = vec![0.0, 0.0];
        let result = poincare_exp(&mu, &v).expect("exp should succeed");
        for (a, b) in mu.iter().zip(&result) {
            assert!((a - b).abs() < 1.0e-12, "exp_μ(0) should equal μ");
        }
    }

    // 2. log_μ(μ) = 0
    #[test]
    fn poincare_log_at_basepoint_is_zero() {
        let mu = vec![0.2, -0.3];
        let result = poincare_log(&mu, &mu).expect("log should succeed");
        for v in &result {
            assert!(v.abs() < 1.0e-12, "log_μ(μ) should be zero");
        }
    }

    // 3. log_μ(exp_μ(v)) ≈ v for small v
    #[test]
    fn poincare_exp_log_roundtrip() {
        let mu = vec![0.1, 0.2];
        let v = vec![0.05, -0.07];
        let x = poincare_exp(&mu, &v).expect("exp ok");
        let v_back = poincare_log(&mu, &x).expect("log ok");
        for (a, b) in v.iter().zip(&v_back) {
            assert!(
                (a - b).abs() < 1.0e-9,
                "roundtrip log∘exp failed: {a} vs {b}"
            );
        }
    }

    // 4. exp_μ(log_μ(x)) ≈ x
    #[test]
    fn poincare_log_exp_roundtrip() {
        let mu = vec![0.1, 0.2];
        let x = vec![-0.3, 0.15];
        let v = poincare_log(&mu, &x).expect("log ok");
        let x_back = poincare_exp(&mu, &v).expect("exp ok");
        for (a, b) in x.iter().zip(&x_back) {
            assert!(
                (a - b).abs() < 1.0e-9,
                "roundtrip exp∘log failed: {a} vs {b}"
            );
        }
    }

    // 5. All sampled points must satisfy ‖x‖ < 1
    #[test]
    fn wrapped_normal_sample_inside_ball() {
        let config = default_config_2d();
        let mut rng = LcgRng::new(42);
        for _ in 0..200 {
            let s = wrapped_normal_sample(&config, &mut rng).expect("sample ok");
            let norm2: f64 = s.point.iter().map(|v| v * v).sum();
            assert!(norm2 < 1.0, "sampled point outside ball: norm² = {norm2}");
        }
    }

    // 6. wrapped_normal_sample_n returns exactly n samples
    #[test]
    fn wrapped_normal_sample_n_correct_count() {
        let config = default_config_2d();
        let mut rng = LcgRng::new(99);
        let samples = wrapped_normal_sample_n(&config, 50, &mut rng).expect("ok");
        assert_eq!(samples.len(), 50);
    }

    // 7. sigma = 0 should fail validation
    #[test]
    fn wrapped_normal_sigma_zero_invalid() {
        let config = WrappedNormalConfig {
            dim: 2,
            mu: vec![0.0, 0.0],
            sigma: 0.0,
            ball_epsilon: 1.0e-5,
        };
        let result = validate_wrapped_normal_config(&config);
        assert!(result.is_err(), "sigma=0 should return Err");
    }

    // 8. ‖μ‖ >= 1 should fail validation
    #[test]
    fn wrapped_normal_mu_outside_ball_invalid() {
        let config = WrappedNormalConfig {
            dim: 2,
            mu: vec![0.8, 0.8], // norm ≈ 1.131
            sigma: 0.5,
            ball_epsilon: 1.0e-5,
        };
        let result = validate_wrapped_normal_config(&config);
        assert!(result.is_err(), "mu outside ball should return Err");
    }

    // 9. log_prob returns a finite value for valid input
    #[test]
    fn wrapped_normal_log_prob_finite() {
        let config = default_config_2d();
        let x = vec![0.05, 0.1];
        let lp = wrapped_normal_log_prob(&config, &x).expect("log_prob ok");
        assert!(lp.is_finite(), "log_prob should be finite");
    }

    // 10. With μ=0, samples should be roughly centred near origin
    #[test]
    fn wrapped_normal_origin_symmetry() {
        let config = default_config_origin();
        let mut rng = LcgRng::new(2025);
        let n = 500;
        let samples = wrapped_normal_sample_n(&config, n, &mut rng).expect("ok");
        // Compute mean of sampled points
        let mut mean = vec![0.0f64; config.dim];
        for s in &samples {
            for (m, p) in mean.iter_mut().zip(&s.point) {
                *m += p;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        // With mu=0 and moderate sigma, empirical mean should be near 0
        let mean_norm: f64 = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            mean_norm < 0.15,
            "With mu=0, empirical mean should be near 0, got norm={mean_norm}"
        );
    }
}
