//! Mip-NeRF Integrated Positional Encoding (IPE).
//!
//! Encodes a Gaussian-approximated cone frustum rather than a single point.
//! For a Gaussian with mean μ and diagonal variance σ²:
//!
//! `E[sin(ω·x)] = sin(ω·μ) · exp(-ω²·σ²/2)`
//! `E[cos(ω·x)] = cos(ω·μ) · exp(-ω²·σ²/2)`
//!
//! where `ω = 2^k · π` for frequency level k.

use crate::error::{NerfError, NerfResult};

/// Configuration for integrated positional encoding.
#[derive(Debug, Clone, Copy)]
pub struct IpeConfig {
    /// Number of frequency levels L.
    pub n_freq: usize,
    /// Input dimensionality.
    pub input_dim: usize,
}

impl IpeConfig {
    /// Output dimensionality: `input_dim * 2 * n_freq`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.input_dim * 2 * self.n_freq
    }
}

/// Compute IPE for a Gaussian with mean `mu` and diagonal variance `diag_sigma2`.
///
/// Output layout: for each frequency level k, for each dimension d:
///   `[sin(ω·μ_d)·exp(-ω²·σ²_d/2), cos(ω·μ_d)·exp(-ω²·σ²_d/2)]`
///
/// # Errors
///
/// Returns `DimensionMismatch` if `mu.len() != diag_sigma2.len()` or they
/// don't match `cfg.input_dim`, or `InvalidFreqLevels` if `n_freq == 0`.
pub fn integrated_pe(mu: &[f32], diag_sigma2: &[f32], cfg: &IpeConfig) -> NerfResult<Vec<f32>> {
    if cfg.n_freq == 0 {
        return Err(NerfError::InvalidFreqLevels { levels: 0 });
    }
    if mu.len() != cfg.input_dim {
        return Err(NerfError::DimensionMismatch {
            expected: cfg.input_dim,
            got: mu.len(),
        });
    }
    if diag_sigma2.len() != cfg.input_dim {
        return Err(NerfError::DimensionMismatch {
            expected: cfg.input_dim,
            got: diag_sigma2.len(),
        });
    }

    let out_dim = cfg.output_dim();
    let mut out = vec![0.0_f32; out_dim];
    let mut write_pos = 0;

    for k in 0..cfg.n_freq {
        let omega = (2_u32.pow(k as u32)) as f32 * std::f32::consts::PI;
        let omega_sq = omega * omega;

        for d in 0..cfg.input_dim {
            let mu_d = mu[d];
            let sigma2_d = diag_sigma2[d].max(0.0);
            let attenuation = (-0.5 * omega_sq * sigma2_d).exp();

            out[write_pos] = (omega * mu_d).sin() * attenuation;
            write_pos += 1;
            out[write_pos] = (omega * mu_d).cos() * attenuation;
            write_pos += 1;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipe_output_dim() {
        let cfg = IpeConfig {
            n_freq: 4,
            input_dim: 3,
        };
        assert_eq!(cfg.output_dim(), 24);
    }

    #[test]
    fn ipe_zero_variance_matches_pe() {
        // With zero variance, IPE should match regular PE
        let cfg = IpeConfig {
            n_freq: 2,
            input_dim: 1,
        };
        let mu = [0.5_f32];
        let sigma2 = [0.0_f32];
        let ipe_out = integrated_pe(&mu, &sigma2, &cfg).unwrap();

        // Manually compute PE
        let expected: Vec<f32> = (0..2)
            .flat_map(|k| {
                let omega = (2_u32.pow(k)) as f32 * std::f32::consts::PI;
                [(omega * 0.5).sin(), (omega * 0.5).cos()]
            })
            .collect();

        for (a, b) in ipe_out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "IPE mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn ipe_high_variance_attenuates() {
        let cfg = IpeConfig {
            n_freq: 4,
            input_dim: 1,
        };
        let mu = [0.5_f32];
        let low_var = [0.0001_f32];
        let high_var = [100.0_f32];
        let out_low = integrated_pe(&mu, &low_var, &cfg).unwrap();
        let out_high = integrated_pe(&mu, &high_var, &cfg).unwrap();

        // High variance should attenuate higher frequencies more
        let mag_low: f32 = out_low.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mag_high: f32 = out_high.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            mag_high < mag_low,
            "high variance should attenuate encoding"
        );
    }
}
