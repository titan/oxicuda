//! Barlow Twins — Zbontar et al. 2021 — redundancy reduction via cross-correlation.
//!
//! Given two views `Z_A`, `Z_B ∈ ℝ^{N×D}`, batch-normalise the columns to zero
//! mean / unit variance, then compute the cross-correlation matrix
//! `C = (Z_A^⊤ Z_B) / N ∈ ℝ^{D×D}`. The loss is
//! ```text
//!     L = Σ_i (1 − C_ii)² + λ · Σ_{i≠j} C_ij²
//! ```
//! pulling diagonal entries toward 1 (invariance) and pushing off-diagonal
//! entries toward 0 (decorrelation).

use crate::error::{SslError, SslResult};

/// Configuration for Barlow Twins.
#[derive(Debug, Clone)]
pub struct BarlowTwinsConfig {
    /// Off-diagonal weighting coefficient λ. Default 0.005 (paper).
    pub lambda: f32,
}

impl Default for BarlowTwinsConfig {
    fn default() -> Self {
        Self { lambda: 0.005 }
    }
}

impl BarlowTwinsConfig {
    /// Validated config.
    ///
    /// # Errors
    /// [`SslError::InvalidLossWeight`] if `lambda` is non-finite or negative.
    pub fn new(lambda: f32) -> SslResult<Self> {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(SslError::InvalidLossWeight { weight: lambda });
        }
        Ok(Self { lambda })
    }
}

/// Standardise columns of a `[N × D]` matrix to zero mean and unit variance,
/// returning a fresh allocation.
fn standardise(z: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = z.to_vec();
    if n < 2 {
        return out;
    }
    let inv_n = 1.0_f32 / n as f32;
    for j in 0..d {
        // Column mean
        let mut mean = 0.0_f32;
        for i in 0..n {
            mean += out[i * d + j];
        }
        mean *= inv_n;
        // Column variance
        let mut var = 0.0_f32;
        for i in 0..n {
            let v = out[i * d + j] - mean;
            var += v * v;
        }
        let std = (var * inv_n + 1e-5).sqrt();
        let inv_std = 1.0 / std;
        for i in 0..n {
            out[i * d + j] = (out[i * d + j] - mean) * inv_std;
        }
    }
    out
}

/// Compute the Barlow Twins loss on two `[N × D]` projection matrices.
///
/// Returns the scalar loss `L`.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] when shapes disagree.
/// - [`SslError::BatchTooSmall`] when `n < 2`.
pub fn barlow_twins_loss(
    z_a: &[f32],
    z_b: &[f32],
    n: usize,
    d: usize,
    cfg: &BarlowTwinsConfig,
) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if n < 2 {
        return Err(SslError::BatchTooSmall);
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

    let za = standardise(z_a, n, d);
    let zb = standardise(z_b, n, d);
    let inv_n = 1.0_f32 / n as f32;
    let mut diag_sum = 0.0_f64;
    let mut off_sum = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            let mut c = 0.0_f32;
            for k in 0..n {
                c += za[k * d + i] * zb[k * d + j];
            }
            c *= inv_n;
            if i == j {
                let r = 1.0 - c;
                diag_sum += (r * r) as f64;
            } else {
                off_sum += (c * c) as f64;
            }
        }
    }
    Ok(diag_sum as f32 + cfg.lambda * off_sum as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barlow_default_lambda() {
        let cfg = BarlowTwinsConfig::default();
        assert!((cfg.lambda - 0.005).abs() < 1e-7);
    }

    #[test]
    fn barlow_identical_inputs_low_diag_loss() {
        // Two identical views → diagonal of cross-correlation ≈ 1, off-diagonal small.
        let n = 16;
        let d = 4;
        let mut z = vec![0.0_f32; n * d];
        for (i, v) in z.iter_mut().enumerate() {
            *v = ((i as f32) * 0.31415).sin();
        }
        let cfg = BarlowTwinsConfig::default();
        let l = barlow_twins_loss(&z, &z, n, d, &cfg).unwrap();
        assert!(l < 1.0, "l = {l}");
    }

    #[test]
    fn barlow_uncorrelated_inputs_high_diag_loss() {
        // Z_A vs Z_B with random independent values → diag of C is near zero,
        // so (1 - C_ii)² is near 1 per dim.
        let n = 256;
        let d = 4;
        let mut rng_a = 13u64;
        let mut rng_b = 17u64;
        let mut z_a = vec![0.0_f32; n * d];
        let mut z_b = vec![0.0_f32; n * d];
        for v in z_a.iter_mut() {
            rng_a = rng_a
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((rng_a >> 33) as f32 / (u32::MAX as f32 + 1.0)) - 0.5;
        }
        for v in z_b.iter_mut() {
            rng_b = rng_b
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((rng_b >> 33) as f32 / (u32::MAX as f32 + 1.0)) - 0.5;
        }
        let cfg = BarlowTwinsConfig::default();
        let l = barlow_twins_loss(&z_a, &z_b, n, d, &cfg).unwrap();
        // Each diagonal contributes ~1; total >= ~3 for d=4.
        assert!(l > 2.0, "l = {l}");
    }

    #[test]
    fn barlow_rejects_n_lt_2() {
        let z = vec![1.0_f32, 2.0];
        let cfg = BarlowTwinsConfig::default();
        assert!(barlow_twins_loss(&z, &z, 1, 2, &cfg).is_err());
    }

    #[test]
    fn barlow_rejects_dim_mismatch() {
        let a = vec![1.0_f32; 8];
        let b = vec![1.0_f32; 6];
        let cfg = BarlowTwinsConfig::default();
        assert!(barlow_twins_loss(&a, &b, 2, 4, &cfg).is_err());
    }

    #[test]
    fn barlow_rejects_negative_lambda() {
        assert!(BarlowTwinsConfig::new(-0.1).is_err());
    }

    #[test]
    fn barlow_lambda_zero_is_diagonal_only() {
        let n = 8;
        let d = 3;
        let mut z = vec![0.0_f32; n * d];
        for (i, v) in z.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let cfg_zero = BarlowTwinsConfig::new(0.0).unwrap();
        let cfg_default = BarlowTwinsConfig::default();
        let l_zero = barlow_twins_loss(&z, &z, n, d, &cfg_zero).unwrap();
        let l_default = barlow_twins_loss(&z, &z, n, d, &cfg_default).unwrap();
        // λ=0 should lose only the diagonal contribution — λ=default adds the
        // off-diagonal squared sum on top.
        assert!(l_default >= l_zero - 1e-4);
    }
}
