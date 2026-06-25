//! UMAP optimisation via stochastic gradient descent with negative sampling.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::umap::fuzzy_simplicial::{fuzzy_simplicial_set, symmetrise};
use crate::umap::knn_graph::{build_knn_distances, smooth_knn_distances};

/// UMAP fit configuration.
#[derive(Debug, Clone)]
pub struct UmapConfig {
    pub n_components: usize,
    pub n_neighbors: usize,
    pub n_epochs: usize,
    pub initial_alpha: f64,
    pub min_dist: f64,
    pub spread: f64,
    pub negative_sample_rate: usize,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 5,
            n_epochs: 200,
            initial_alpha: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            negative_sample_rate: 5,
        }
    }
}

/// UMAP result.
pub struct UmapResult {
    pub embedding: Vec<f64>,
    pub a: f64,
    pub b: f64,
    pub epochs: usize,
}

/// Ergonomic builder for [`UmapConfig`] with validation on [`build`](UmapConfigBuilder::build).
///
/// ```
/// use oxicuda_manifold::umap::embedding::UmapConfigBuilder;
/// let cfg = UmapConfigBuilder::new()
///     .n_components(2)
///     .n_neighbors(15)
///     .min_dist(0.1)
///     .spread(1.0)
///     .n_epochs(200)
///     .negative_sample_rate(5)
///     .initial_alpha(1.0)
///     .build()
///     .expect("valid config");
/// assert_eq!(cfg.n_neighbors, 15);
/// ```
#[derive(Debug, Clone)]
pub struct UmapConfigBuilder {
    cfg: UmapConfig,
}

impl Default for UmapConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl UmapConfigBuilder {
    /// Start from the [`UmapConfig`] defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cfg: UmapConfig::default(),
        }
    }

    /// Embedding dimensionality (must be in `1..=8`).
    #[must_use]
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.cfg.n_components = n_components;
        self
    }

    /// Number of nearest neighbours used to build the fuzzy simplicial set (must be `>= 1`).
    #[must_use]
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.cfg.n_neighbors = n_neighbors;
        self
    }

    /// Number of optimisation epochs (must be `>= 1`).
    #[must_use]
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.cfg.n_epochs = n_epochs;
        self
    }

    /// Initial learning rate `alpha` (must be `> 0`).
    #[must_use]
    pub fn initial_alpha(mut self, initial_alpha: f64) -> Self {
        self.cfg.initial_alpha = initial_alpha;
        self
    }

    /// Minimum distance controlling how tightly points are packed (must be `>= 0`).
    #[must_use]
    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.cfg.min_dist = min_dist;
        self
    }

    /// Effective scale of embedded points (must be `> 0`).
    #[must_use]
    pub fn spread(mut self, spread: f64) -> Self {
        self.cfg.spread = spread;
        self
    }

    /// Number of negative samples per positive sample (must be `>= 1`).
    #[must_use]
    pub fn negative_sample_rate(mut self, negative_sample_rate: usize) -> Self {
        self.cfg.negative_sample_rate = negative_sample_rate;
        self
    }

    /// Validate the accumulated parameters and produce a [`UmapConfig`].
    ///
    /// # Errors
    /// Returns [`ManifoldError::InvalidParameter`] when any knob is outside its valid range,
    /// e.g. zero neighbours / epochs, non-positive spread, or negative `min_dist`.
    pub fn build(self) -> ManifoldResult<UmapConfig> {
        let c = &self.cfg;
        if c.n_components == 0 || c.n_components > 8 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be in 1..=8".into(),
            });
        }
        if c.n_neighbors == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be >= 1".into(),
            });
        }
        if c.n_epochs == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_epochs".into(),
                reason: "must be >= 1".into(),
            });
        }
        if !c.initial_alpha.is_finite() || c.initial_alpha <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "initial_alpha".into(),
                reason: "must be finite and strictly positive".into(),
            });
        }
        if !c.min_dist.is_finite() || c.min_dist < 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "min_dist".into(),
                reason: "must be finite and non-negative".into(),
            });
        }
        if !c.spread.is_finite() || c.spread <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "spread".into(),
                reason: "must be finite and strictly positive".into(),
            });
        }
        if c.min_dist > c.spread {
            return Err(ManifoldError::InvalidParameter {
                name: "min_dist".into(),
                reason: "must not exceed spread".into(),
            });
        }
        if c.negative_sample_rate == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "negative_sample_rate".into(),
                reason: "must be >= 1".into(),
            });
        }
        Ok(self.cfg)
    }
}

/// Fit UMAP on row-major data `(n_samples, dim)`.
pub fn umap_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &UmapConfig,
    rng: &mut LcgRng,
) -> ManifoldResult<UmapResult> {
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
    if cfg.n_neighbors == 0 || cfg.n_neighbors >= n_samples {
        return Err(ManifoldError::KNeighborsTooLarge {
            k: cfg.n_neighbors,
            n: n_samples,
        });
    }
    let n = n_samples;
    let k = cfg.n_neighbors;
    let d_out = cfg.n_components;
    let (idx, dist) = build_knn_distances(x, n, dim, k)?;
    let (sigmas, rhos) = smooth_knn_distances(&dist, n, k, 64, 1e-5)?;
    let (rows, cols, vals) = fuzzy_simplicial_set(&idx, &dist, &sigmas, &rhos, n, k)?;
    let (rows, cols, vals) = symmetrise(&rows, &cols, &vals)?;
    let (a, b) = fit_ab(cfg.spread, cfg.min_dist);
    // Initialise embedding randomly (small)
    let mut y = vec![0.0; n * d_out];
    for v in &mut y {
        *v = rng.next_range(-1.0, 1.0);
    }
    let n_epochs = cfg.n_epochs;
    let max_val = vals.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
    // epochs per sample = max_val * n_epochs / mu
    let epochs_per_sample: Vec<f64> = vals
        .iter()
        .map(|m| {
            if *m > 0.0 {
                n_epochs as f64 / (n_epochs as f64 / (max_val / m).max(1.0))
            } else {
                0.0
            }
        })
        .collect();
    for epoch in 0..n_epochs {
        let alpha = cfg.initial_alpha * (1.0 - epoch as f64 / n_epochs as f64);
        // Attractive forces along each edge
        for e in 0..rows.len() {
            let prob = epochs_per_sample[e];
            if rng.next_f64() >= prob.max(1e-12) {
                continue;
            }
            let i = rows[e];
            let j = cols[e];
            let mut d2 = 0.0;
            for kk in 0..d_out {
                let v = y[i * d_out + kk] - y[j * d_out + kk];
                d2 += v * v;
            }
            // attractive gradient: -2 a b d^{2(b-1)} / (1 + a d^{2b}) * (y_i - y_j)
            let pow_b = d2.powf(b);
            let denom = 1.0 + a * pow_b;
            let coeff = if d2 > 0.0 {
                -2.0 * a * b * d2.powf(b - 1.0) / denom
            } else {
                0.0
            };
            for kk in 0..d_out {
                let v = y[i * d_out + kk] - y[j * d_out + kk];
                let delta = (coeff * v).clamp(-4.0, 4.0);
                y[i * d_out + kk] += alpha * delta;
                y[j * d_out + kk] -= alpha * delta;
            }
            // Negative samples
            for _ in 0..cfg.negative_sample_rate {
                let neg = rng.next_usize(n);
                if neg == i || neg == j {
                    continue;
                }
                let mut d2n = 0.0;
                for kk in 0..d_out {
                    let v = y[i * d_out + kk] - y[neg * d_out + kk];
                    d2n += v * v;
                }
                let denom = (0.001 + d2n) * (1.0 + a * d2n.powf(b));
                let coeff = if d2n > 0.0 { 2.0 * b / denom } else { 0.0 };
                for kk in 0..d_out {
                    let v = y[i * d_out + kk] - y[neg * d_out + kk];
                    let delta = (coeff * v).clamp(-4.0, 4.0);
                    y[i * d_out + kk] += alpha * delta;
                }
            }
        }
    }
    Ok(UmapResult {
        embedding: y,
        a,
        b,
        epochs: n_epochs,
    })
}

/// Fit the UMAP `(a, b)` curve to `phi(d) = 1 if d < min_dist else exp(-(d-min_dist)/spread)`.
///
/// Curve form: `1 / (1 + a d^{2b})`. We use a simple grid + local refinement.
fn fit_ab(spread: f64, min_dist: f64) -> (f64, f64) {
    // Target samples
    let n_targets = 300usize;
    let mut xs = vec![0.0; n_targets];
    let mut ys = vec![0.0; n_targets];
    for i in 0..n_targets {
        xs[i] = (i as f64) * 3.0 * spread / n_targets as f64;
        ys[i] = if xs[i] < min_dist {
            1.0
        } else {
            (-((xs[i] - min_dist) / spread)).exp()
        };
    }
    // Grid search over (a, b) and minimise SSE
    let mut best = f64::INFINITY;
    let mut best_a = 1.0f64;
    let mut best_b = 1.0f64;
    for ai in 1..30 {
        let a = 0.5 + 0.5 * ai as f64;
        for bi in 5..20 {
            let b = 0.1 + 0.1 * bi as f64;
            let mut sse = 0.0;
            for i in 0..n_targets {
                let pred = 1.0 / (1.0 + a * xs[i].powf(2.0 * b));
                let d = pred - ys[i];
                sse += d * d;
            }
            if sse < best {
                best = sse;
                best_a = a;
                best_b = b;
            }
        }
    }
    (best_a, best_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains_and_validates() {
        let cfg = UmapConfigBuilder::new()
            .n_components(3)
            .n_neighbors(10)
            .min_dist(0.05)
            .spread(2.0)
            .n_epochs(150)
            .negative_sample_rate(7)
            .initial_alpha(0.5)
            .build()
            .expect("valid config");
        assert_eq!(cfg.n_components, 3);
        assert_eq!(cfg.n_neighbors, 10);
        assert_eq!(cfg.min_dist, 0.05);
        assert_eq!(cfg.spread, 2.0);
        assert_eq!(cfg.n_epochs, 150);
        assert_eq!(cfg.negative_sample_rate, 7);
        assert_eq!(cfg.initial_alpha, 0.5);
    }

    #[test]
    fn builder_default_runs_fit() {
        let cfg = UmapConfigBuilder::default()
            .n_neighbors(3)
            .n_epochs(40)
            .build()
            .expect("valid");
        let mut rng = LcgRng::new(5);
        let n = 12;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for v in &mut x {
            *v = rng.next_normal();
        }
        let r = umap_fit(&x, n, dim, &cfg, &mut rng).expect("fit ok");
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.a > 0.0 && r.b > 0.0);
    }

    #[test]
    fn builder_rejects_bad_parameters() {
        assert!(UmapConfigBuilder::new().n_components(0).build().is_err());
        assert!(UmapConfigBuilder::new().n_components(9).build().is_err());
        assert!(UmapConfigBuilder::new().n_neighbors(0).build().is_err());
        assert!(UmapConfigBuilder::new().n_epochs(0).build().is_err());
        assert!(UmapConfigBuilder::new().initial_alpha(0.0).build().is_err());
        assert!(UmapConfigBuilder::new().min_dist(-0.1).build().is_err());
        assert!(UmapConfigBuilder::new().spread(0.0).build().is_err());
        assert!(
            UmapConfigBuilder::new()
                .negative_sample_rate(0)
                .build()
                .is_err()
        );
        // min_dist > spread is invalid.
        assert!(
            UmapConfigBuilder::new()
                .min_dist(2.0)
                .spread(1.0)
                .build()
                .is_err()
        );
    }

    #[test]
    fn umap_runs_small() {
        let mut rng = LcgRng::new(7);
        let n = 12;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for v in &mut x {
            *v = rng.next_normal();
        }
        let cfg = UmapConfig {
            n_neighbors: 3,
            n_epochs: 50,
            ..UmapConfig::default()
        };
        let r = umap_fit(&x, n, dim, &cfg, &mut rng).expect("ok");
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
        assert!(r.a > 0.0);
        assert!(r.b > 0.0);
    }
}
