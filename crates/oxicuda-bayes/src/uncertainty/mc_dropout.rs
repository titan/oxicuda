//! Monte-Carlo Dropout (Gal & Ghahramani 2016).
//!
//! Treats dropout — applied at test time — as a variational approximation to a
//! deep Gaussian process posterior. Forward `T` stochastic passes through the
//! network and use the Monte-Carlo mean as the predictive probability and the
//! sample variance as a per-example uncertainty estimate.
//!
//! This module provides:
//! - [`mc_dropout_predict`] — pure-functional version that takes a closure
//!   producing one stochastic prediction per call.
//! - [`McDropoutPredictor`] — owning wrapper that bundles the closure with
//!   the chosen sample count.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

/// Predictive statistics of an MC-Dropout ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct McDropoutStats {
    /// Per-output mean over `T` stochastic forward passes (length = output dim).
    pub mean: Vec<f32>,
    /// Per-output variance (sample variance, Bessel-corrected when T ≥ 2).
    pub variance: Vec<f32>,
    /// Total per-output predictive standard deviation `sqrt(variance)`.
    pub std_dev: Vec<f32>,
    /// Number of forward passes used.
    pub n_samples: usize,
}

impl McDropoutStats {
    /// Largest component of `std_dev`.
    #[must_use]
    pub fn max_std(&self) -> f32 {
        self.std_dev.iter().copied().fold(0.0_f32, f32::max)
    }

    /// Mean of `std_dev`.
    #[must_use]
    pub fn avg_std(&self) -> f32 {
        if self.std_dev.is_empty() {
            return 0.0;
        }
        let s: f32 = self.std_dev.iter().sum();
        s / self.std_dev.len() as f32
    }
}

/// Run `n_samples` stochastic forward passes via `forward_fn` and return the
/// per-output mean and (Bessel-corrected) variance.
///
/// Each call to `forward_fn(&mut rng)` should perform one inference pass with
/// dropout enabled; the returned `Vec<f32>` length must be the same on every
/// call (validated at runtime).
///
/// # Errors
/// - [`BayesError::InsufficientSamples`] when `n_samples == 0`.
/// - [`BayesError::EmptyInputs`] when the first prediction is empty.
/// - [`BayesError::DimensionMismatch`] when prediction shapes vary across passes.
pub fn mc_dropout_predict<F>(
    n_samples: usize,
    rng: &mut LcgRng,
    mut forward_fn: F,
) -> BayesResult<McDropoutStats>
where
    F: FnMut(&mut LcgRng) -> BayesResult<Vec<f32>>,
{
    if n_samples == 0 {
        return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
    }
    let first = forward_fn(rng)?;
    if first.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    let dim = first.len();

    // Welford online stats using f64 accumulators for stability.
    let mut mean = vec![0.0_f64; dim];
    let mut m2 = vec![0.0_f64; dim];

    let update = |sample: &[f32], mean: &mut [f64], m2: &mut [f64], n: usize| {
        let inv_n = 1.0_f64 / n as f64;
        for ((mu, m2_i), &x) in mean.iter_mut().zip(m2.iter_mut()).zip(sample.iter()) {
            let delta = x as f64 - *mu;
            *mu += delta * inv_n;
            let delta2 = x as f64 - *mu;
            *m2_i += delta * delta2;
        }
    };

    update(&first, &mut mean, &mut m2, 1);

    for n in 2..=n_samples {
        let sample = forward_fn(rng)?;
        if sample.len() != dim {
            return Err(BayesError::DimensionMismatch {
                expected: dim,
                got: sample.len(),
            });
        }
        update(&sample, &mut mean, &mut m2, n);
    }

    let denom = if n_samples >= 2 {
        (n_samples - 1) as f64
    } else {
        1.0
    };
    let mean_f32: Vec<f32> = mean.iter().map(|v| *v as f32).collect();
    let variance: Vec<f32> = m2.iter().map(|v| (*v / denom) as f32).collect();
    let std_dev: Vec<f32> = variance.iter().map(|v| v.max(0.0).sqrt()).collect();

    Ok(McDropoutStats {
        mean: mean_f32,
        variance,
        std_dev,
        n_samples,
    })
}

/// Owned wrapper that bundles a forward-pass closure with a sample count.
pub struct McDropoutPredictor<F>
where
    F: FnMut(&mut LcgRng) -> BayesResult<Vec<f32>>,
{
    /// Number of stochastic forward passes to draw.
    pub n_samples: usize,
    /// Stochastic forward function with dropout enabled.
    pub forward_fn: F,
}

impl<F> McDropoutPredictor<F>
where
    F: FnMut(&mut LcgRng) -> BayesResult<Vec<f32>>,
{
    /// Construct a predictor.
    pub fn new(n_samples: usize, forward_fn: F) -> Self {
        Self {
            n_samples,
            forward_fn,
        }
    }

    /// Run inference and collect statistics.
    ///
    /// # Errors
    /// Propagates errors from [`mc_dropout_predict`].
    pub fn predict(&mut self, rng: &mut LcgRng) -> BayesResult<McDropoutStats> {
        let n = self.n_samples;
        let f = &mut self.forward_fn;
        mc_dropout_predict(n, rng, |r| f(r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mc_dropout_constant_forward_zero_variance() {
        let mut rng = LcgRng::new(0);
        let stats = mc_dropout_predict(8, &mut rng, |_r| Ok(vec![0.7_f32, 0.2, 0.1]))
            .expect("value should be present");
        assert_eq!(stats.mean.len(), 3);
        assert!((stats.mean[0] - 0.7).abs() < 1e-6);
        for v in &stats.variance {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn mc_dropout_normal_forward_positive_variance() {
        let mut rng = LcgRng::new(1234);
        let stats = mc_dropout_predict(256, &mut rng, |r| {
            let (a, b) = r.next_normal_pair();
            let (c, _) = r.next_normal_pair();
            Ok(vec![a, b, c])
        })
        .expect("value should be present");
        assert_eq!(stats.n_samples, 256);
        // For N(0, 1) the sample variance should be near 1.
        let avg_var = stats.variance.iter().copied().sum::<f32>() / 3.0;
        assert!(avg_var > 0.5 && avg_var < 1.5, "avg_var = {avg_var}");
    }

    #[test]
    fn mc_dropout_rejects_zero_samples() {
        let mut rng = LcgRng::new(0);
        let r = mc_dropout_predict(0, &mut rng, |_r| Ok(vec![0.0_f32]));
        assert!(r.is_err());
    }

    #[test]
    fn mc_dropout_rejects_dim_mismatch() {
        let mut rng = LcgRng::new(0);
        let mut count = 0_usize;
        let r = mc_dropout_predict(4, &mut rng, |_r| {
            count += 1;
            if count == 1 {
                Ok(vec![0.0_f32, 1.0_f32])
            } else {
                Ok(vec![0.0_f32])
            }
        });
        assert!(r.is_err());
    }

    #[test]
    fn mc_dropout_rejects_empty_first_prediction() {
        let mut rng = LcgRng::new(0);
        let r = mc_dropout_predict(2, &mut rng, |_r| Ok(vec![]));
        assert!(r.is_err());
    }

    #[test]
    fn mc_dropout_predictor_wraps_closure() {
        let mut rng = LcgRng::new(7);
        let mut p = McDropoutPredictor::new(4, |r| Ok(vec![r.next_f32(), r.next_f32()]));
        let stats = p.predict(&mut rng).expect("predict should succeed");
        assert_eq!(stats.n_samples, 4);
        assert_eq!(stats.mean.len(), 2);
    }

    #[test]
    fn mc_dropout_max_and_avg_std() {
        let stats = McDropoutStats {
            mean: vec![0.0_f32, 0.0, 0.0],
            variance: vec![0.04_f32, 0.16, 0.01],
            std_dev: vec![0.2_f32, 0.4, 0.1],
            n_samples: 16,
        };
        assert!((stats.max_std() - 0.4).abs() < 1e-6);
        let avg = (0.2 + 0.4 + 0.1) / 3.0;
        assert!((stats.avg_std() - avg).abs() < 1e-6);
    }

    #[test]
    fn mc_dropout_stats_empty_avg() {
        let stats = McDropoutStats {
            mean: vec![],
            variance: vec![],
            std_dev: vec![],
            n_samples: 0,
        };
        assert_eq!(stats.avg_std(), 0.0);
        assert_eq!(stats.max_std(), 0.0);
    }
}
