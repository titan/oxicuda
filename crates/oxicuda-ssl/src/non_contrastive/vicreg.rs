//! VICReg — Bardes, Ponce, LeCun 2022 — Variance-Invariance-Covariance.
//!
//! Three loss terms:
//! - **Invariance** `s(z_a, z_b) = MSE(z_a, z_b)` — pulls views together.
//! - **Variance** `v(z) = (1/d)·Σ_j max(0, γ − std(z_{·,j}))` — per-feature
//!   variance hinge to prevent collapse.
//! - **Covariance** `c(z) = (1/d)·Σ_{i≠j} C_ij²` where `C` is the centred
//!   feature covariance — decorrelates feature dimensions.
//!
//! Total loss `L = λ·s + μ·v(z_a)/2 + μ·v(z_b)/2 + ν·c(z_a)/2 + ν·c(z_b)/2`.

use crate::error::{SslError, SslResult};

/// VICReg configuration. Default coefficients follow the paper (λ=25, μ=25, ν=1).
#[derive(Debug, Clone)]
pub struct VicRegConfig {
    /// Invariance weight.
    pub lambda: f32,
    /// Variance weight.
    pub mu: f32,
    /// Covariance weight.
    pub nu: f32,
    /// Variance hinge threshold γ (default 1.0).
    pub gamma: f32,
}

impl Default for VicRegConfig {
    fn default() -> Self {
        Self {
            lambda: 25.0,
            mu: 25.0,
            nu: 1.0,
            gamma: 1.0,
        }
    }
}

impl VicRegConfig {
    /// New validated config.
    ///
    /// # Errors
    /// [`SslError::InvalidLossWeight`] for non-finite or negative weights.
    pub fn new(lambda: f32, mu: f32, nu: f32, gamma: f32) -> SslResult<Self> {
        for w in [lambda, mu, nu, gamma] {
            if !(w.is_finite() && w >= 0.0) {
                return Err(SslError::InvalidLossWeight { weight: w });
            }
        }
        Ok(Self {
            lambda,
            mu,
            nu,
            gamma,
        })
    }
}

/// Mean-squared invariance loss: `(1/(N·D))·Σ ‖z_a − z_b‖²`.
fn mse(z_a: &[f32], z_b: &[f32]) -> f32 {
    let n = z_a.len() as f32;
    let mut s = 0.0_f64;
    for (a, b) in z_a.iter().zip(z_b.iter()) {
        let d = (a - b) as f64;
        s += d * d;
    }
    (s / n as f64) as f32
}

/// Variance hinge `(1/d)·Σ_j max(0, γ − std_j)` after centring.
fn variance_hinge(z: &[f32], n: usize, d: usize, gamma: f32) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let inv_n = 1.0_f32 / n as f32;
    let mut total = 0.0_f64;
    for j in 0..d {
        let mut mean = 0.0_f32;
        for i in 0..n {
            mean += z[i * d + j];
        }
        mean *= inv_n;
        let mut var = 0.0_f32;
        for i in 0..n {
            let v = z[i * d + j] - mean;
            var += v * v;
        }
        let std = (var * inv_n + 1e-4).sqrt();
        total += (gamma - std).max(0.0) as f64;
    }
    (total / d as f64) as f32
}

/// Off-diagonal squared covariance penalty `(1/d)·Σ_{i≠j} C_ij²`.
fn covariance_penalty(z: &[f32], n: usize, d: usize) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let inv_n = 1.0_f32 / (n as f32 - 1.0);
    // Column means
    let mut mean = vec![0.0_f32; d];
    for j in 0..d {
        let mut s = 0.0_f32;
        for i in 0..n {
            s += z[i * d + j];
        }
        mean[j] = s / n as f32;
    }
    // Centred matrix → feature covariance C = Z̄ᵀ Z̄ / (n-1)
    let mut total = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            if i == j {
                continue;
            }
            let mut c = 0.0_f32;
            for k in 0..n {
                c += (z[k * d + i] - mean[i]) * (z[k * d + j] - mean[j]);
            }
            c *= inv_n;
            total += (c * c) as f64;
        }
    }
    (total / d as f64) as f32
}

/// Compute the VICReg loss.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] when shapes disagree.
pub fn vicreg_loss(
    z_a: &[f32],
    z_b: &[f32],
    n: usize,
    d: usize,
    cfg: &VicRegConfig,
) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z_a.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_a.len(),
        });
    }
    if z_b.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_b.len(),
        });
    }
    let s = mse(z_a, z_b);
    let v_a = variance_hinge(z_a, n, d, cfg.gamma);
    let v_b = variance_hinge(z_b, n, d, cfg.gamma);
    let c_a = covariance_penalty(z_a, n, d);
    let c_b = covariance_penalty(z_b, n, d);
    Ok(cfg.lambda * s + cfg.mu * 0.5 * (v_a + v_b) + cfg.nu * 0.5 * (c_a + c_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vicreg_default_paper_weights() {
        let cfg = VicRegConfig::default();
        assert!((cfg.lambda - 25.0).abs() < 1e-6);
        assert!((cfg.mu - 25.0).abs() < 1e-6);
        assert!((cfg.nu - 1.0).abs() < 1e-6);
        assert!((cfg.gamma - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vicreg_rejects_negative_weight() {
        assert!(VicRegConfig::new(-1.0, 1.0, 1.0, 1.0).is_err());
        assert!(VicRegConfig::new(1.0, -1.0, 1.0, 1.0).is_err());
        assert!(VicRegConfig::new(1.0, 1.0, -1.0, 1.0).is_err());
        assert!(VicRegConfig::new(1.0, 1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn vicreg_invariance_only_for_identical() {
        let n = 8;
        let d = 4;
        let z: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.1).collect();
        let cfg = VicRegConfig::new(1.0, 0.0, 0.0, 1.0).expect("new should succeed");
        // Identical inputs → MSE = 0 → loss = 0.
        let l = vicreg_loss(&z, &z, n, d, &cfg).expect("vicreg_loss should succeed");
        assert!(l.abs() < 1e-5);
    }

    #[test]
    fn vicreg_loss_finite_on_random_data() {
        let n = 32;
        let d = 8;
        let z_a: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let z_b: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.027).cos()).collect();
        let cfg = VicRegConfig::default();
        let l = vicreg_loss(&z_a, &z_b, n, d, &cfg).expect("vicreg_loss should succeed");
        assert!(l.is_finite() && l >= 0.0);
    }

    #[test]
    fn vicreg_zero_variance_triggers_hinge() {
        // Constant column → std = 0 → hinge = γ
        let n = 16;
        let d = 4;
        let z = vec![1.0_f32; n * d];
        let cfg = VicRegConfig::new(0.0, 1.0, 0.0, 1.0).expect("new should succeed");
        let l = vicreg_loss(&z, &z, n, d, &cfg).expect("vicreg_loss should succeed");
        // Every column has zero variance → variance hinge per column ≈ γ = 1.
        // Both halves contribute, mean ≈ 1.
        assert!(l > 0.5, "l = {l}");
    }

    #[test]
    fn vicreg_rejects_dim_mismatch() {
        let a = vec![1.0_f32; 8];
        let b = vec![1.0_f32; 6];
        let cfg = VicRegConfig::default();
        assert!(vicreg_loss(&a, &b, 2, 4, &cfg).is_err());
    }

    #[test]
    fn vicreg_rejects_empty() {
        let r = vicreg_loss(&[], &[], 0, 0, &VicRegConfig::default());
        assert!(r.is_err());
    }
}
