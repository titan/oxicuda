//! Reversible Instance Normalisation (RevIN).
//!
//! RevIN normalises each variate independently over the time axis to remove
//! distribution shift, then reverses the normalisation in the decoder so the
//! model sees a stationary input without losing the original scale.
//!
//! Forward (normalise): `y = (x - μ) / (σ + ε) * γ + β`
//! Inverse (denormalise): `x = (y - β) / γ * (σ + ε) + μ`
//!
//! Where `μ` and `σ` are computed per `(batch, variate)` over `T`, and
//! `γ`, `β` are learnable per-variate affine parameters.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

/// Reversible Instance Normalisation layer.
///
/// Operates on `[T, C]` tensors (time-major layout).
#[derive(Debug, Clone)]
pub struct RevIn {
    /// Learnable affine scale `[C]`.
    pub gamma: Vec<f32>,
    /// Learnable affine bias `[C]`.
    pub beta: Vec<f32>,
    /// Number of variates (channels).
    pub c: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
}

impl RevIn {
    /// Construct a `RevIn` layer with `gamma=1`, `beta=0`, `eps=1e-5`.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidNumVariates`] when `c == 0`.
    pub fn new(c: usize) -> TsResult<Self> {
        if c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        Ok(Self {
            gamma: vec![1.0_f32; c],
            beta: vec![0.0_f32; c],
            c,
            eps: 1e-5,
        })
    }

    /// Perturb gamma/beta with small N(0, 0.02) noise for non-trivial tests.
    pub fn randomise(&mut self, rng: &mut LcgRng) {
        let mut noise = vec![0.0_f32; self.c];
        rng.fill_normal(&mut noise);
        for (g, n) in self.gamma.iter_mut().zip(noise.iter()) {
            *g = 1.0 + 0.02 * n;
        }
        let mut noise2 = vec![0.0_f32; self.c];
        rng.fill_normal(&mut noise2);
        for (b, n) in self.beta.iter_mut().zip(noise2.iter()) {
            *b = 0.02 * n;
        }
    }

    /// Compute per-variate mean and std over time for a `[T, C]` tensor.
    ///
    /// Returns `(mean, std)` each of length `C`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::DimensionMismatch`] when `features.len() != t * self.c`.
    pub fn compute_stats(&self, features: &[f32], t: usize) -> TsResult<(Vec<f32>, Vec<f32>)> {
        if t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let expected = t * self.c;
        if features.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        let mut mean = vec![0.0_f32; self.c];
        let mut var = vec![0.0_f32; self.c];

        for ti in 0..t {
            for ci in 0..self.c {
                mean[ci] += features[ti * self.c + ci];
            }
        }
        let inv_t = 1.0 / t as f32;
        for m in mean.iter_mut() {
            *m *= inv_t;
        }

        for ti in 0..t {
            for ci in 0..self.c {
                let d = features[ti * self.c + ci] - mean[ci];
                var[ci] += d * d;
            }
        }
        // Bessel correction when t > 1
        let denom = if t > 1 { (t - 1) as f32 } else { 1.0 };
        let std: Vec<f32> = var.iter().map(|&v| (v / denom).sqrt()).collect();

        Ok((mean, std))
    }

    /// Apply normalisation to a `[T, C]` tensor and return `(normalised, stats)`.
    ///
    /// The returned stats `(mean, std)` are needed to reverse the normalisation.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_stats`].
    pub fn forward(&self, features: &[f32], t: usize) -> TsResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (mean, std) = self.compute_stats(features, t)?;
        let c = self.c;

        let mut out = vec![0.0_f32; t * c];
        for ti in 0..t {
            for ci in 0..c {
                let x = features[ti * c + ci];
                let norm = (x - mean[ci]) / (std[ci] + self.eps);
                out[ti * c + ci] = norm * self.gamma[ci] + self.beta[ci];
            }
        }
        Ok((out, mean, std))
    }

    /// Reverse the normalisation applied by [`Self::forward`].
    ///
    /// # Arguments
    ///
    /// * `y` — `[T, C]` normalised output from decoder.
    /// * `mean` — per-variate mean returned by `forward`.
    /// * `std`  — per-variate std returned by `forward`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::DimensionMismatch`] when sizes don't match.
    pub fn inverse(&self, y: &[f32], t: usize, mean: &[f32], std: &[f32]) -> TsResult<Vec<f32>> {
        if t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let c = self.c;
        let expected = t * c;
        if y.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: y.len(),
            });
        }
        if mean.len() != c || std.len() != c {
            return Err(TsError::ShapeMismatch {
                msg: format!("mean/std length {} != C={}", mean.len(), c),
            });
        }

        let mut out = vec![0.0_f32; t * c];
        for ti in 0..t {
            for ci in 0..c {
                let yi = y[ti * c + ci];
                // reverse affine
                let z = (yi - self.beta[ci]) / (self.gamma[ci] + self.eps);
                // reverse normalisation
                out[ti * c + ci] = z * (std[ci] + self.eps) + mean[ci];
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn revin_new_ok() {
        let rv = RevIn::new(8).expect("ok");
        assert_eq!(rv.c, 8);
        assert!(rv.gamma.iter().all(|&g| (g - 1.0).abs() < 1e-7));
        assert!(rv.beta.iter().all(|&b| b.abs() < 1e-7));
    }

    #[test]
    fn revin_zero_c_error() {
        assert!(matches!(
            RevIn::new(0).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn revin_forward_shape() {
        let rv = RevIn::new(4).expect("ok");
        let features = vec![1.0_f32; 10 * 4];
        let (out, mean, std) = rv.forward(&features, 10).expect("forward ok");
        assert_eq!(out.len(), 10 * 4);
        assert_eq!(mean.len(), 4);
        assert_eq!(std.len(), 4);
    }

    #[test]
    fn revin_forward_zero_mean() {
        // After RevIN, each variate should have ~zero mean over time.
        let mut rng = make_rng();
        let rv = RevIn::new(8).expect("ok");
        let mut features = vec![0.0_f32; 32 * 8];
        rng.fill_normal(&mut features);
        let (out, _, _) = rv.forward(&features, 32).expect("ok");
        for ci in 0..8 {
            let s: f32 = (0..32).map(|ti| out[ti * 8 + ci]).sum::<f32>() / 32.0;
            assert!(s.abs() < 1e-5, "variate {ci} mean={s} not near 0");
        }
    }

    #[test]
    fn revin_inverse_recovers_input() {
        let rv = RevIn::new(4).expect("ok");
        let features: Vec<f32> = (0..20 * 4).map(|i| i as f32 * 0.1).collect();
        let (normed, mean, std) = rv.forward(&features, 20).expect("ok");
        let recovered = rv.inverse(&normed, 20, &mean, &std).expect("ok");
        for (i, (&orig, &rec)) in features.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "idx={i}: orig={orig} recovered={rec}"
            );
        }
    }

    #[test]
    fn revin_forward_zero_t_error() {
        let rv = RevIn::new(4).expect("ok");
        assert!(matches!(
            rv.forward(&[], 0).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn revin_forward_dim_mismatch() {
        let rv = RevIn::new(4).expect("ok");
        let features = vec![0.0_f32; 7]; // wrong size
        assert!(matches!(
            rv.forward(&features, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn revin_stats_finite() {
        let mut rng = make_rng();
        let rv = RevIn::new(16).expect("ok");
        let mut features = vec![0.0_f32; 50 * 16];
        rng.fill_normal(&mut features);
        let (mean, std) = rv.compute_stats(&features, 50).expect("ok");
        assert!(mean.iter().all(|v| v.is_finite()));
        assert!(std.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn revin_inverse_zero_t_error() {
        let rv = RevIn::new(4).expect("ok");
        let mean = vec![0.0_f32; 4];
        let std = vec![1.0_f32; 4];
        assert!(matches!(
            rv.inverse(&[], 0, &mean, &std).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn revin_inverse_shape_mismatch() {
        let rv = RevIn::new(4).expect("ok");
        let y = vec![0.0_f32; 10 * 4];
        let mean = vec![0.0_f32; 3]; // wrong
        let std = vec![1.0_f32; 4];
        assert!(matches!(
            rv.inverse(&y, 10, &mean, &std).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }
}
