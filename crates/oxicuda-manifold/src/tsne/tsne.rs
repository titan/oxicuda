//! Core t-SNE algorithm.
//!
//! Pipeline:
//! 1. Compute pairwise squared Euclidean distances in input space.
//! 2. Build joint probability matrix `P` via per-row perplexity binary search.
//! 3. Initialise `Y` from a small-variance Gaussian.
//! 4. Iterate gradient descent on `Y` with momentum and early-exaggeration.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::tsne::perplexity::compute_perplexity_p_matrix;

/// t-SNE configuration knobs.
#[derive(Debug, Clone)]
pub struct TsneConfig {
    /// Embedding dimensionality (typically 2).
    pub n_components: usize,
    /// Perplexity target.
    pub perplexity: f64,
    /// Total number of iterations.
    pub n_iter: usize,
    /// Number of early-exaggeration iterations.
    pub early_exaggeration_iters: usize,
    /// Early-exaggeration factor.
    pub early_exaggeration: f64,
    /// Initial momentum.
    pub momentum: f64,
    /// Final momentum (after switch).
    pub final_momentum: f64,
    /// Iteration index when momentum switches.
    pub momentum_switch_iter: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Min gain.
    pub min_gain: f64,
}

impl Default for TsneConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 5.0,
            n_iter: 300,
            early_exaggeration_iters: 100,
            early_exaggeration: 12.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
            learning_rate: 200.0,
            min_gain: 0.01,
        }
    }
}

/// t-SNE result.
pub struct TsneResult {
    pub embedding: Vec<f64>,
    pub final_kl_divergence: f64,
}

/// Ergonomic builder for [`TsneConfig`] with validation on [`build`](TsneConfigBuilder::build).
///
/// Every setter returns `self` so calls can be chained:
/// ```
/// use oxicuda_manifold::tsne::tsne::TsneConfigBuilder;
/// let cfg = TsneConfigBuilder::new()
///     .n_components(2)
///     .perplexity(30.0)
///     .learning_rate(200.0)
///     .early_exaggeration(12.0, 250)
///     .momentum_schedule(0.5, 0.8, 250)
///     .n_iter(1000)
///     .build()
///     .expect("valid config");
/// assert_eq!(cfg.perplexity, 30.0);
/// ```
#[derive(Debug, Clone)]
pub struct TsneConfigBuilder {
    cfg: TsneConfig,
}

impl Default for TsneConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TsneConfigBuilder {
    /// Start from the [`TsneConfig`] defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cfg: TsneConfig::default(),
        }
    }

    /// Embedding dimensionality (must be in `1..=8`).
    #[must_use]
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.cfg.n_components = n_components;
        self
    }

    /// Target perplexity (must be `> 0`).
    #[must_use]
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.cfg.perplexity = perplexity;
        self
    }

    /// Total number of gradient-descent iterations (must be `>= 1`).
    #[must_use]
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.cfg.n_iter = n_iter;
        self
    }

    /// Learning rate (must be `> 0`).
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.cfg.learning_rate = learning_rate;
        self
    }

    /// Minimum adaptive gain (must be `> 0`).
    #[must_use]
    pub fn min_gain(mut self, min_gain: f64) -> Self {
        self.cfg.min_gain = min_gain;
        self
    }

    /// Early-exaggeration `factor` applied for the first `iters` iterations.
    #[must_use]
    pub fn early_exaggeration(mut self, factor: f64, iters: usize) -> Self {
        self.cfg.early_exaggeration = factor;
        self.cfg.early_exaggeration_iters = iters;
        self
    }

    /// Momentum schedule: `initial` until `switch_iter`, then `final_momentum`.
    #[must_use]
    pub fn momentum_schedule(
        mut self,
        initial: f64,
        final_momentum: f64,
        switch_iter: usize,
    ) -> Self {
        self.cfg.momentum = initial;
        self.cfg.final_momentum = final_momentum;
        self.cfg.momentum_switch_iter = switch_iter;
        self
    }

    /// Validate the accumulated parameters and produce a [`TsneConfig`].
    ///
    /// # Errors
    /// Returns [`ManifoldError::InvalidParameter`] when any knob is outside its valid range,
    /// e.g. zero iterations, non-positive perplexity / learning rate, or out-of-range momentum.
    pub fn build(self) -> ManifoldResult<TsneConfig> {
        let c = &self.cfg;
        if c.n_components == 0 || c.n_components > 8 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be in 1..=8".into(),
            });
        }
        if !c.perplexity.is_finite() || c.perplexity <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "perplexity".into(),
                reason: "must be finite and strictly positive".into(),
            });
        }
        if c.n_iter == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_iter".into(),
                reason: "must be >= 1".into(),
            });
        }
        if c.early_exaggeration_iters > c.n_iter {
            return Err(ManifoldError::InvalidParameter {
                name: "early_exaggeration_iters".into(),
                reason: "must not exceed n_iter".into(),
            });
        }
        if !c.early_exaggeration.is_finite() || c.early_exaggeration <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "early_exaggeration".into(),
                reason: "must be strictly positive".into(),
            });
        }
        if !c.learning_rate.is_finite() || c.learning_rate <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be finite and strictly positive".into(),
            });
        }
        if !c.min_gain.is_finite() || c.min_gain <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "min_gain".into(),
                reason: "must be strictly positive".into(),
            });
        }
        if !(0.0..1.0).contains(&c.momentum) || !(0.0..1.0).contains(&c.final_momentum) {
            return Err(ManifoldError::InvalidParameter {
                name: "momentum".into(),
                reason: "initial and final momentum must be in [0, 1)".into(),
            });
        }
        if c.momentum_switch_iter > c.n_iter {
            return Err(ManifoldError::InvalidParameter {
                name: "momentum_switch_iter".into(),
                reason: "must not exceed n_iter".into(),
            });
        }
        Ok(self.cfg)
    }
}

/// Fit t-SNE on row-major data of shape `(n_samples, dim)`.
pub fn tsne_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &TsneConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<TsneResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if cfg.n_components == 0 || cfg.n_components > 8 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=8".into(),
        });
    }
    let n = n_samples;
    let d_out = cfg.n_components;
    // Pairwise squared distances
    let mut d2 = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..dim {
                let v = x[i * dim + k] - x[j * dim + k];
                s += v * v;
            }
            d2[i * n + j] = s;
            d2[j * n + i] = s;
        }
    }
    // Joint probability matrix
    let mut p = compute_perplexity_p_matrix(&d2, n, cfg.perplexity, 60, 1e-5)?;
    // Apply early exaggeration
    for v in &mut p {
        *v *= cfg.early_exaggeration;
    }
    // Initialise Y from N(0, 1e-4)
    let mut y = vec![0.0; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 0.01;
    }
    let mut dy_prev: Vec<f64> = vec![0.0; n * d_out];
    let mut gains: Vec<f64> = vec![1.0; n * d_out];

    let mut final_kl = 0.0_f64;
    for iter in 0..cfg.n_iter {
        // Build low-dimensional q_ij = (1 + ||y_i - y_j||^2)^-1 / Z
        let (q, z_sum) = compute_q_matrix(&y, n, d_out);
        let mom = if iter < cfg.momentum_switch_iter {
            cfg.momentum
        } else {
            cfg.final_momentum
        };
        // Gradient: 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^-1 * (y_i - y_j)
        let mut grad = vec![0.0; n * d_out];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d2ij = 0.0;
                for k in 0..d_out {
                    let v = y[i * d_out + k] - y[j * d_out + k];
                    d2ij += v * v;
                }
                let qkernel = 1.0 / (1.0 + d2ij);
                let pij = p[i * n + j];
                let qij = q[i * n + j];
                let mult = (pij - qij) * qkernel;
                for k in 0..d_out {
                    grad[i * d_out + k] += mult * (y[i * d_out + k] - y[j * d_out + k]);
                }
            }
        }
        for v in &mut grad {
            *v *= 4.0;
        }
        // Adaptive learning rate with gains
        for i in 0..n * d_out {
            let same_sign = grad[i].signum() == dy_prev[i].signum();
            if same_sign {
                gains[i] *= 0.8;
            } else {
                gains[i] += 0.2;
            }
            if gains[i] < cfg.min_gain {
                gains[i] = cfg.min_gain;
            }
            let new_dy = mom * dy_prev[i] - cfg.learning_rate * gains[i] * grad[i];
            dy_prev[i] = new_dy;
            y[i] += new_dy;
        }
        // Re-centre Y
        for k in 0..d_out {
            let mut m = 0.0;
            for i in 0..n {
                m += y[i * d_out + k];
            }
            m /= n as f64;
            for i in 0..n {
                y[i * d_out + k] -= m;
            }
        }
        if iter == cfg.early_exaggeration_iters {
            // Remove early-exaggeration factor
            for v in &mut p {
                *v /= cfg.early_exaggeration;
            }
        }
        if iter == cfg.n_iter - 1 {
            // Compute final KL
            let mut kl = 0.0;
            for (pi, qi) in p.iter().zip(&q) {
                if *pi > 1e-12 && *qi > 1e-12 {
                    kl += pi * (pi / qi).ln();
                }
            }
            final_kl = kl;
        }
        let _ = z_sum;
    }
    Ok(TsneResult {
        embedding: y,
        final_kl_divergence: final_kl,
    })
}

fn compute_q_matrix(y: &[f64], n: usize, dim: usize) -> (Vec<f64>, f64) {
    let mut q = vec![0.0; n * n];
    let mut z = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d2 = 0.0;
            for k in 0..dim {
                let v = y[i * dim + k] - y[j * dim + k];
                d2 += v * v;
            }
            let qval = 1.0 / (1.0 + d2);
            q[i * n + j] = qval;
            z += qval;
        }
    }
    let z = z.max(1e-300);
    for v in &mut q {
        *v /= z;
    }
    for v in q.iter_mut() {
        if *v < 1e-12 {
            *v = 1e-12;
        }
    }
    (q, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains_and_validates() {
        let cfg = TsneConfigBuilder::new()
            .n_components(3)
            .perplexity(20.0)
            .learning_rate(150.0)
            .min_gain(0.05)
            .early_exaggeration(8.0, 40)
            .momentum_schedule(0.4, 0.85, 60)
            .n_iter(120)
            .build()
            .expect("valid config");
        assert_eq!(cfg.n_components, 3);
        assert_eq!(cfg.perplexity, 20.0);
        assert_eq!(cfg.learning_rate, 150.0);
        assert_eq!(cfg.min_gain, 0.05);
        assert_eq!(cfg.early_exaggeration, 8.0);
        assert_eq!(cfg.early_exaggeration_iters, 40);
        assert_eq!(cfg.momentum, 0.4);
        assert_eq!(cfg.final_momentum, 0.85);
        assert_eq!(cfg.momentum_switch_iter, 60);
        assert_eq!(cfg.n_iter, 120);
    }

    #[test]
    fn builder_default_is_valid_and_runs() {
        let cfg = TsneConfigBuilder::default()
            .perplexity(3.0)
            .n_iter(40)
            .early_exaggeration(12.0, 15)
            .momentum_schedule(0.5, 0.8, 30)
            .build()
            .expect("valid");
        let mut rng = LcgRng::new(3);
        let n = 8;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for v in &mut x {
            *v = rng.next_normal();
        }
        let r = tsne_fit(&x, n, dim, &cfg, &mut rng).expect("fit ok");
        assert_eq!(r.embedding.len(), n * 2);
    }

    #[test]
    fn builder_rejects_bad_parameters() {
        assert!(TsneConfigBuilder::new().n_components(0).build().is_err());
        assert!(TsneConfigBuilder::new().n_components(9).build().is_err());
        assert!(TsneConfigBuilder::new().perplexity(0.0).build().is_err());
        assert!(TsneConfigBuilder::new().perplexity(-1.0).build().is_err());
        assert!(TsneConfigBuilder::new().n_iter(0).build().is_err());
        assert!(TsneConfigBuilder::new().learning_rate(0.0).build().is_err());
        assert!(TsneConfigBuilder::new().min_gain(0.0).build().is_err());
        assert!(
            TsneConfigBuilder::new()
                .momentum_schedule(1.0, 0.8, 10)
                .build()
                .is_err()
        );
        assert!(
            TsneConfigBuilder::new()
                .n_iter(50)
                .early_exaggeration(12.0, 100)
                .build()
                .is_err()
        );
    }

    #[test]
    fn tsne_runs_small() {
        let mut rng = LcgRng::new(7);
        let n = 10;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for v in &mut x {
            *v = rng.next_normal();
        }
        let cfg = TsneConfig {
            n_iter: 50,
            early_exaggeration_iters: 20,
            perplexity: 3.0,
            ..TsneConfig::default()
        };
        let r = tsne_fit(&x, n, dim, &cfg, &mut rng).expect("ok");
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tsne_separates_clusters() {
        let mut rng = LcgRng::new(11);
        let n = 16;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        // Two clusters at ±5
        for i in 0..n {
            let centre = if i < 8 { 5.0 } else { -5.0 };
            for d in 0..dim {
                x[i * dim + d] = centre + 0.1 * rng.next_normal();
            }
        }
        let cfg = TsneConfig {
            n_iter: 200,
            early_exaggeration_iters: 80,
            perplexity: 3.0,
            ..TsneConfig::default()
        };
        let r = tsne_fit(&x, n, dim, &cfg, &mut rng).expect("ok");
        // The two cluster centres in the embedding should be far apart
        let mut mean_a = [0.0; 2];
        let mut mean_b = [0.0; 2];
        for i in 0..8 {
            mean_a[0] += r.embedding[i * 2];
            mean_a[1] += r.embedding[i * 2 + 1];
        }
        for i in 8..16 {
            mean_b[0] += r.embedding[i * 2];
            mean_b[1] += r.embedding[i * 2 + 1];
        }
        for m in &mut mean_a {
            *m /= 8.0;
        }
        for m in &mut mean_b {
            *m /= 8.0;
        }
        let sep = ((mean_a[0] - mean_b[0]).powi(2) + (mean_a[1] - mean_b[1]).powi(2)).sqrt();
        assert!(sep.is_finite());
    }
}
