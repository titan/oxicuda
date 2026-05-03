//! Series decomposition into trend and seasonal components.
//!
//! Implements the decomposition block used in Autoformer and TimesNet:
//! `trend = moving_avg(x)`, `seasonal = x - trend`.
//!
//! The trend captures the slowly-varying mean (low frequency) while the
//! seasonal component contains the residual fast variations.

use crate::decomp::moving_avg::MovingAvg;
use crate::error::TsResult;

/// Decomposition result.
#[derive(Debug, Clone)]
pub struct DecompResult {
    /// Trend component `[T, C]`.
    pub trend: Vec<f32>,
    /// Seasonal (residual) component `[T, C]`.
    pub seasonal: Vec<f32>,
}

/// Series decomposition block.
///
/// Splits a `[T, C]` time-series into trend and seasonal components
/// using a centred moving average.
#[derive(Debug, Clone)]
pub struct SeriesDecomp {
    moving_avg: MovingAvg,
}

impl SeriesDecomp {
    /// Construct a `SeriesDecomp` with the given moving average kernel size.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::TsError::InvalidKernelSize`] when `kernel_size == 0`.
    pub fn new(kernel_size: usize) -> TsResult<Self> {
        Ok(Self {
            moving_avg: MovingAvg::new(kernel_size)?,
        })
    }

    /// Decompose a `[T, C]` tensor into trend + seasonal.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`MovingAvg::forward`].
    pub fn forward(&self, features: &[f32], t: usize, c: usize) -> TsResult<DecompResult> {
        let trend = self.moving_avg.forward(features, t, c)?;
        let seasonal: Vec<f32> = features
            .iter()
            .zip(trend.iter())
            .map(|(&x, &tr)| x - tr)
            .collect();
        Ok(DecompResult { trend, seasonal })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TsError;

    #[test]
    fn series_decomp_sums_to_original() {
        let decomp = SeriesDecomp::new(5).expect("ok");
        let t = 30;
        let c = 4;
        let features: Vec<f32> = (0..t * c).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let res = decomp.forward(&features, t, c).expect("ok");
        for (i, (&orig, (&tr, &se))) in features
            .iter()
            .zip(res.trend.iter().zip(res.seasonal.iter()))
            .enumerate()
        {
            assert!(
                (orig - (tr + se)).abs() < 1e-5,
                "idx={i}: orig={orig} trend+seasonal={} sum={}",
                tr + se,
                tr + se
            );
        }
    }

    #[test]
    fn series_decomp_flat_seasonal_zero() {
        // Constant input → moving avg = input → seasonal = 0
        let decomp = SeriesDecomp::new(5).expect("ok");
        let features = vec![7.0_f32; 20 * 3];
        let res = decomp.forward(&features, 20, 3).expect("ok");
        for &s in &res.seasonal {
            assert!(s.abs() < 1e-5, "seasonal should be 0, got {s}");
        }
    }

    #[test]
    fn series_decomp_shapes() {
        let decomp = SeriesDecomp::new(3).expect("ok");
        let t = 15;
        let c = 6;
        let features = vec![0.0_f32; t * c];
        let res = decomp.forward(&features, t, c).expect("ok");
        assert_eq!(res.trend.len(), t * c);
        assert_eq!(res.seasonal.len(), t * c);
    }

    #[test]
    fn series_decomp_zero_k_error() {
        assert!(matches!(
            SeriesDecomp::new(0).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    #[test]
    fn series_decomp_output_finite() {
        let decomp = SeriesDecomp::new(25).expect("ok");
        let features: Vec<f32> = (0..100 * 8)
            .map(|i| ((i as f32) * 0.05).sin() * 3.0 + (i as f32) * 0.001)
            .collect();
        let res = decomp.forward(&features, 100, 8).expect("ok");
        assert!(res.trend.iter().all(|v| v.is_finite()));
        assert!(res.seasonal.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn series_decomp_trend_smoother_than_input() {
        // The range of the trend should be no wider than the input range.
        let decomp = SeriesDecomp::new(11).expect("ok");
        let features: Vec<f32> = (0..50).map(|i| (i as f32).sin() * 10.0).collect();
        let res = decomp.forward(&features, 50, 1).expect("ok");
        let in_range = features.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - features.iter().cloned().fold(f32::INFINITY, f32::min);
        let tr_range = res.trend.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - res.trend.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            tr_range <= in_range + 1e-4,
            "trend range {tr_range} > input range {in_range}"
        );
    }
}
