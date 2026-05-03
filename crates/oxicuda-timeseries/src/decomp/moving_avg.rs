//! Moving average for trend extraction.
//!
//! Applies a symmetric (centred) moving average of odd kernel size `K`
//! over the time axis of a `[T, C]` tensor using replicate boundary padding.
//! The CPU reference matches `moving_average_ptx`.

use crate::error::{TsError, TsResult};

/// Centred moving average operator.
///
/// Computes `y[t, c] = (1/K) * Σ_{k=-h}^{h} x[clamp(t+k, 0, T-1), c]`
/// where `h = (K - 1) / 2`.
#[derive(Debug, Clone)]
pub struct MovingAvg {
    /// Kernel size (must be odd and ≥ 1).
    pub kernel_size: usize,
}

impl MovingAvg {
    /// Construct a `MovingAvg` with the given kernel size.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidKernelSize`] when `kernel_size == 0`.
    pub fn new(kernel_size: usize) -> TsResult<Self> {
        if kernel_size == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }
        Ok(Self { kernel_size })
    }

    /// Apply moving average to a `[T, C]` tensor.
    ///
    /// Returns a new `[T, C]` tensor.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::DimensionMismatch`] when `features.len() != t * c`.
    pub fn forward(&self, features: &[f32], t: usize, c: usize) -> TsResult<Vec<f32>> {
        if t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let expected = t * c;
        if features.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        let k = self.kernel_size;
        let half = k / 2;
        let inv_k = 1.0 / k as f32;
        let mut out = vec![0.0_f32; t * c];

        for ti in 0..t {
            for ci in 0..c {
                let mut sum = 0.0_f32;
                for ki in 0..k {
                    let src_t = if ti + ki >= half {
                        (ti + ki - half).min(t - 1)
                    } else {
                        0
                    };
                    sum += features[src_t * c + ci];
                }
                out[ti * c + ci] = sum * inv_k;
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_avg_flat_input() {
        let ma = MovingAvg::new(5).expect("ok");
        let features = vec![3.0_f32; 20 * 4];
        let out = ma.forward(&features, 20, 4).expect("ok");
        assert_eq!(out.len(), 20 * 4);
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-5, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn moving_avg_ramp_smoothed() {
        // A ramp 0..T at C=1 — moving average should stay close to each value
        // for interior elements.
        let t = 20;
        let ma = MovingAvg::new(3).expect("ok");
        let features: Vec<f32> = (0..t).map(|i| i as f32).collect();
        let out = ma.forward(&features, t, 1).expect("ok");
        assert_eq!(out.len(), t);
        // Interior: out[t] ≈ t (mean of t-1, t, t+1)
        for ti in 1..t - 1 {
            let expected = (features[ti - 1] + features[ti] + features[ti + 1]) / 3.0;
            assert!(
                (out[ti] - expected).abs() < 1e-5,
                "t={ti}: out={} exp={expected}",
                out[ti]
            );
        }
    }

    #[test]
    fn moving_avg_zero_k_error() {
        assert!(matches!(
            MovingAvg::new(0).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    #[test]
    fn moving_avg_zero_t_error() {
        let ma = MovingAvg::new(3).expect("ok");
        assert!(matches!(
            ma.forward(&[], 0, 4).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn moving_avg_dim_mismatch() {
        let ma = MovingAvg::new(3).expect("ok");
        let f = vec![0.0_f32; 7];
        assert!(matches!(
            ma.forward(&f, 2, 4).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn moving_avg_k1_identity() {
        let ma = MovingAvg::new(1).expect("ok");
        let features: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = ma.forward(&features, 4, 3).expect("ok");
        for (a, b) in features.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn moving_avg_output_finite() {
        let ma = MovingAvg::new(7).expect("ok");
        let features: Vec<f32> = (0..50 * 8).map(|i| (i as f32) * 0.01 - 2.0).collect();
        let out = ma.forward(&features, 50, 8).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
