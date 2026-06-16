//! Mahalanobis distance with a learned positive-semidefinite metric.
//!
//! The Mahalanobis distance between two vectors is
//! `d_M(x, y) = sqrt((x − y)ᵀ M (x − y))` where `M` is positive semidefinite
//! (PSD). To guarantee PSD-ness by construction the metric is parameterized via
//! a lower-triangular Cholesky-style factor `L` so that `M = L Lᵀ`. For any `L`
//! the product `L Lᵀ` is PSD, hence no projection step is ever required during
//! learning.
//!
//! The factor `L` is learned from labeled *similar* / *dissimilar* pairs using a
//! contrastive margin loss (in the spirit of ITML / LMNN):
//!
//! ```text
//! loss = Σ_{(i,j) similar}    d²(i, j)
//!      + Σ_{(i,j) dissimilar} max(0, margin − d(i, j))²
//! ```
//!
//! Gradient descent is performed directly on the entries of `L`; after every
//! step the upper triangle is forced back to zero so `L` stays lower-triangular.

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;

/// A small positive value used to guard divisions by the (square-rooted)
/// distance when differentiating the hinge term of the contrastive loss.
const DIST_EPS: f32 = 1e-12;

/// Configuration for constructing and training a [`MahalanobisMetric`].
#[derive(Debug, Clone)]
pub struct MahalanobisConfig {
    /// Vector dimensionality. Must be `>= 1`.
    pub dim: usize,
    /// Learning rate for the gradient-descent updates. Must be `> 0`.
    pub lr: f32,
    /// Margin used by the dissimilar-pair hinge term. May be any finite value;
    /// a positive margin pushes dissimilar pairs at least `margin` apart.
    pub margin: f32,
    /// Number of gradient-descent iterations performed by [`MahalanobisMetric::fit`].
    /// Must be `>= 1`.
    pub n_iter: usize,
}

/// A learned Mahalanobis metric parameterized as `M = L Lᵀ`.
///
/// `l` stores the `dim × dim` lower-triangular factor in row-major order; the
/// upper triangle (strictly above the diagonal) is always zero.
#[derive(Debug, Clone)]
pub struct MahalanobisMetric {
    /// Lower-triangular Cholesky-style factor `L`, `dim × dim`, row-major.
    l: Vec<f32>,
    /// Vector dimensionality.
    dim: usize,
}

impl MahalanobisMetric {
    /// Build the identity metric (`L = I`, hence `M = I`), which reproduces the
    /// ordinary Euclidean distance.
    ///
    /// # Errors
    /// Returns [`AnnError::InvalidVectorDim`] if `dim == 0`.
    pub fn identity(dim: usize) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim });
        }
        let mut l = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            l[i * dim + i] = 1.0;
        }
        Ok(Self { l, dim })
    }

    /// Build a metric whose factor is the identity plus a small random
    /// lower-triangular perturbation, suitable as a starting point for
    /// [`MahalanobisMetric::fit`].
    ///
    /// The perturbation entries are drawn from `next_normal_pair` scaled by
    /// `0.01`. The diagonal therefore stays close to `1`, keeping `L` close to
    /// the identity (and `M` close to Euclidean) at initialization.
    ///
    /// # Errors
    /// Returns [`AnnError::InvalidVectorDim`] if `cfg.dim == 0`.
    pub fn new(cfg: &MahalanobisConfig, rng: &mut LcgRng) -> AnnResult<Self> {
        if cfg.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: cfg.dim });
        }
        let dim = cfg.dim;
        let mut metric = Self::identity(dim)?;
        // Perturb only the lower triangle (including the diagonal).
        for i in 0..dim {
            let mut j = 0;
            while j <= i {
                let (a, b) = rng.next_normal_pair();
                metric.l[i * dim + j] += a * 0.01;
                if j < i {
                    metric.l[i * dim + (j + 1)] += b * 0.01;
                }
                j += 2;
            }
        }
        Ok(metric)
    }

    /// Vector dimensionality this metric operates on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Read-only view of the lower-triangular factor `L` (`dim × dim`, row-major).
    #[must_use]
    pub fn factor(&self) -> &[f32] {
        &self.l
    }

    /// Compute `t = Lᵀ · diff` into `t` (length `dim`). `diff` must have length
    /// `dim`. Exploits the lower-triangular structure of `L`.
    #[inline]
    fn apply_lt(&self, diff: &[f32], t: &mut [f32]) {
        let dim = self.dim;
        for tk in t.iter_mut() {
            *tk = 0.0;
        }
        // L is lower-triangular: L[i][k] != 0 only for i >= k. Therefore
        // (Lᵀ diff)[k] = Σ_{i >= k} L[i][k] * diff[i].
        for (i, &di) in diff.iter().enumerate() {
            let row = i * dim;
            // Only columns k <= i contribute (lower triangle).
            for (k, tk) in t.iter_mut().enumerate().take(i + 1) {
                *tk += self.l[row + k] * di;
            }
        }
    }

    /// Validate that `x` and `y` both match the configured dimension.
    fn check_pair(&self, x: &[f32], y: &[f32]) -> AnnResult<()> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        if y.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: y.len(),
            });
        }
        Ok(())
    }

    /// Squared Mahalanobis distance `(x − y)ᵀ M (x − y) = ‖Lᵀ (x − y)‖²`.
    ///
    /// # Errors
    /// Returns [`AnnError::DimensionMismatch`] if either input length differs
    /// from `dim`.
    pub fn distance_sq(&self, x: &[f32], y: &[f32]) -> AnnResult<f32> {
        self.check_pair(x, y)?;
        let diff: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        let mut t = vec![0.0_f32; self.dim];
        self.apply_lt(&diff, &mut t);
        Ok(t.iter().map(|v| v * v).sum())
    }

    /// Mahalanobis distance `sqrt((x − y)ᵀ M (x − y))`.
    ///
    /// The squared distance is clamped to be non-negative before taking the
    /// square root to defend against tiny negative values from rounding.
    ///
    /// # Errors
    /// Returns [`AnnError::DimensionMismatch`] if either input length differs
    /// from `dim`.
    pub fn distance(&self, x: &[f32], y: &[f32]) -> AnnResult<f32> {
        let sq = self.distance_sq(x, y)?;
        Ok(sq.max(0.0).sqrt())
    }

    /// Materialize the metric matrix `M = L Lᵀ` as a `dim × dim` row-major
    /// vector. The result is symmetric by construction.
    #[must_use]
    pub fn metric_matrix(&self) -> Vec<f32> {
        let dim = self.dim;
        let mut m = vec![0.0_f32; dim * dim];
        // M[a][b] = Σ_k L[a][k] * L[b][k].
        for a in 0..dim {
            for b in 0..dim {
                let mut acc = 0.0_f32;
                // L is lower-triangular: L[a][k] != 0 only for k <= a,
                // L[b][k] != 0 only for k <= b. So k <= min(a, b).
                let kmax = a.min(b);
                for k in 0..=kmax {
                    acc += self.l[a * dim + k] * self.l[b * dim + k];
                }
                m[a * dim + b] = acc;
            }
        }
        m
    }

    /// Learn the factor `L` from labeled pairs via contrastive-margin gradient
    /// descent, returning the final loss value.
    ///
    /// `data` is a flat `n × dim` row-major matrix. Each entry of `pairs` is
    /// `(i, j, similar)` where `i` and `j` index rows of `data` and `similar`
    /// is `true` for a should-be-close pair and `false` for a should-be-far
    /// pair. The loss is
    ///
    /// ```text
    /// loss = Σ_{similar} d²(i, j) + Σ_{dissimilar} max(0, margin − d(i, j))²
    /// ```
    ///
    /// and the gradient of a similar pair's `d²` w.r.t. `L` is the outer product
    /// `2 · diff · (Lᵀ diff)ᵀ`; the dissimilar hinge contributes only while
    /// `d(i, j) < margin`, scaled by `−(margin − d) / d` through the chain rule
    /// of `d = sqrt(d²)` (guarded by a small epsilon on `d`).
    ///
    /// # Errors
    /// - [`AnnError::InvalidVectorDim`] if `cfg.dim == 0`.
    /// - [`AnnError::Internal`] if `cfg.lr <= 0` or `cfg.n_iter == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - [`AnnError::IdOutOfRange`] if any pair index is `>= n`.
    pub fn fit(
        &mut self,
        data: &[f32],
        n: usize,
        pairs: &[(usize, usize, bool)],
        cfg: &MahalanobisConfig,
    ) -> AnnResult<f32> {
        if cfg.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: cfg.dim });
        }
        if cfg.lr <= 0.0 || !cfg.lr.is_finite() {
            return Err(AnnError::Internal {
                msg: format!("learning rate must be a finite value > 0, got {}", cfg.lr),
            });
        }
        if cfg.n_iter == 0 {
            return Err(AnnError::Internal {
                msg: "n_iter must be >= 1".to_string(),
            });
        }
        let dim = self.dim;
        if cfg.dim != dim {
            return Err(AnnError::DimensionMismatch {
                expected: dim,
                got: cfg.dim,
            });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        for &(i, j, _) in pairs {
            if i >= n {
                return Err(AnnError::IdOutOfRange { id: i, n });
            }
            if j >= n {
                return Err(AnnError::IdOutOfRange { id: j, n });
            }
        }

        let row = |idx: usize| -> &[f32] { &data[idx * dim..idx * dim + dim] };

        // Reusable scratch buffers.
        let mut diff = vec![0.0_f32; dim];
        let mut t = vec![0.0_f32; dim];
        let mut grad = vec![0.0_f32; dim * dim];

        let mut last_loss = 0.0_f32;
        for _ in 0..cfg.n_iter {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
            let mut loss = 0.0_f32;

            for &(i, j, similar) in pairs {
                let xi = row(i);
                let xj = row(j);
                for (d, (a, b)) in diff.iter_mut().zip(xi.iter().zip(xj.iter())) {
                    *d = a - b;
                }
                self.apply_lt(&diff, &mut t);
                let d_sq: f32 = t.iter().map(|v| v * v).sum();

                // weight applied to the outer-product gradient grad_dsq = 2·diff·tᵀ
                // of d² for this pair.
                let weight = if similar {
                    loss += d_sq;
                    1.0
                } else {
                    let d = d_sq.max(0.0).sqrt();
                    let slack = cfg.margin - d;
                    if slack > 0.0 {
                        loss += slack * slack;
                        // d/dL max(0, margin - d)² = 2(margin - d)(-1) dd/dL
                        //   = -(margin - d) / d * grad_dsq   (since dd/dL = grad_dsq/(2d))
                        if d > DIST_EPS { -slack / d } else { 0.0 }
                    } else {
                        0.0
                    }
                };

                if weight != 0.0 {
                    // grad += weight * (2 · diff ⊗ t). Only the lower triangle of
                    // L is trainable, so accumulate only there (column k <= row a).
                    let scale = 2.0 * weight;
                    for (a, &da) in diff.iter().enumerate() {
                        let base = a * dim;
                        let coeff = scale * da;
                        for (k, &tk) in t.iter().enumerate().take(a + 1) {
                            grad[base + k] += coeff * tk;
                        }
                    }
                }
            }

            // Gradient-descent step on the lower triangle; the upper triangle is
            // never touched and therefore stays exactly zero.
            for a in 0..dim {
                let base = a * dim;
                for k in 0..=a {
                    self.l[base + k] -= cfg.lr * grad[base + k];
                }
            }
            last_loss = loss;
        }

        Ok(last_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::l2_sq;

    /// `(data, n, dim, pairs)` returned by the labeled-dataset fixture.
    type LabeledSet = (Vec<f32>, usize, usize, Vec<(usize, usize, bool)>);

    fn rng() -> LcgRng {
        LcgRng::new(20240520)
    }

    fn rand_vec(n: usize, r: &mut LcgRng) -> Vec<f32> {
        (0..n)
            .map(|_| r.next_u32() as f32 / (u32::MAX as f32 + 1.0) - 0.5)
            .collect()
    }

    #[test]
    fn identity_distance_equals_euclidean() {
        let dim = 7;
        let metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let mut r = rng();
        for _ in 0..20 {
            let x = rand_vec(dim, &mut r);
            let y = rand_vec(dim, &mut r);
            let m = metric
                .distance(&x, &y)
                .expect("test invariant: should succeed");
            let e = crate::distance::l2::l2(&x, &y).expect("test invariant: should succeed");
            assert!((m - e).abs() < 1e-5, "mahalanobis={m} euclidean={e}");
        }
    }

    #[test]
    fn identity_distance_sq_equals_euclidean_sq() {
        let dim = 5;
        let metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let mut r = rng();
        let x = rand_vec(dim, &mut r);
        let y = rand_vec(dim, &mut r);
        let m = metric
            .distance_sq(&x, &y)
            .expect("test invariant: should succeed");
        let e = l2_sq(&x, &y).expect("test invariant: should succeed");
        assert!((m - e).abs() < 1e-5, "m_sq={m} e_sq={e}");
    }

    #[test]
    fn distance_self_is_zero() {
        let dim = 6;
        let mut r = rng();
        let metric = MahalanobisMetric::new(
            &MahalanobisConfig {
                dim,
                lr: 0.01,
                margin: 1.0,
                n_iter: 1,
            },
            &mut r,
        )
        .expect("test invariant: should succeed");
        let x = rand_vec(dim, &mut r);
        let d = metric
            .distance(&x, &x)
            .expect("test invariant: should succeed");
        assert!(d.abs() < 1e-5, "d(x,x)={d}");
    }

    #[test]
    fn distance_is_symmetric() {
        let dim = 5;
        let mut r = rng();
        let metric = MahalanobisMetric::new(
            &MahalanobisConfig {
                dim,
                lr: 0.01,
                margin: 1.0,
                n_iter: 1,
            },
            &mut r,
        )
        .expect("test invariant: should succeed");
        let x = rand_vec(dim, &mut r);
        let y = rand_vec(dim, &mut r);
        let dxy = metric
            .distance(&x, &y)
            .expect("test invariant: should succeed");
        let dyx = metric
            .distance(&y, &x)
            .expect("test invariant: should succeed");
        assert!((dxy - dyx).abs() < 1e-6, "dxy={dxy} dyx={dyx}");
    }

    #[test]
    fn distance_sq_is_square_of_distance() {
        let dim = 4;
        let mut r = rng();
        let metric = MahalanobisMetric::new(
            &MahalanobisConfig {
                dim,
                lr: 0.01,
                margin: 1.0,
                n_iter: 1,
            },
            &mut r,
        )
        .expect("test invariant: should succeed");
        let x = rand_vec(dim, &mut r);
        let y = rand_vec(dim, &mut r);
        let d = metric
            .distance(&x, &y)
            .expect("test invariant: should succeed");
        let dsq = metric
            .distance_sq(&x, &y)
            .expect("test invariant: should succeed");
        assert!((d * d - dsq).abs() < 1e-5, "d^2={} dsq={dsq}", d * d);
    }

    #[test]
    fn metric_matrix_is_symmetric() {
        let dim = 6;
        let mut r = rng();
        let metric = MahalanobisMetric::new(
            &MahalanobisConfig {
                dim,
                lr: 0.01,
                margin: 1.0,
                n_iter: 1,
            },
            &mut r,
        )
        .expect("test invariant: should succeed");
        let m = metric.metric_matrix();
        for a in 0..dim {
            for b in 0..dim {
                let ab = m[a * dim + b];
                let ba = m[b * dim + a];
                assert!((ab - ba).abs() < 1e-6, "M[{a}][{b}]={ab} M[{b}][{a}]={ba}");
            }
        }
    }

    #[test]
    fn identity_metric_matrix_is_identity() {
        let dim = 5;
        let metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let m = metric.metric_matrix();
        for a in 0..dim {
            for b in 0..dim {
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((m[a * dim + b] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn distance_is_non_negative() {
        let dim = 5;
        let mut r = rng();
        let metric = MahalanobisMetric::new(
            &MahalanobisConfig {
                dim,
                lr: 0.01,
                margin: 1.0,
                n_iter: 1,
            },
            &mut r,
        )
        .expect("test invariant: should succeed");
        for _ in 0..30 {
            let x = rand_vec(dim, &mut r);
            let y = rand_vec(dim, &mut r);
            let d = metric
                .distance(&x, &y)
                .expect("test invariant: should succeed");
            assert!(d >= 0.0, "d={d}");
        }
    }

    /// Build a tiny labeled dataset: two well-separated clusters along axis 0.
    /// Similar pairs lie within a cluster; dissimilar pairs cross clusters.
    fn make_labeled() -> LabeledSet {
        let dim = 3;
        // 4 points: 0,1 near (-2,..); 2,3 near (+2,..)
        let data = vec![
            -2.0, 0.1, 0.0, // 0
            -1.9, -0.1, 0.2, // 1
            2.0, 0.0, -0.1, // 2
            2.1, 0.2, 0.1, // 3
        ];
        let pairs = vec![(0, 1, true), (2, 3, true), (0, 2, false), (1, 3, false)];
        (data, 4, dim, pairs)
    }

    #[test]
    fn fit_reduces_loss() {
        let (data, n, dim, pairs) = make_labeled();
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.01,
            margin: 2.0,
            n_iter: 1,
        };
        let mut r = rng();
        let mut metric =
            MahalanobisMetric::new(&cfg, &mut r).expect("test invariant: should succeed");

        // Initial loss = one fit iteration's reported loss.
        let initial = metric
            .clone()
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");
        // Run many more iterations.
        let many = MahalanobisConfig {
            n_iter: 200,
            ..cfg.clone()
        };
        let final_loss = metric
            .fit(&data, n, &pairs, &many)
            .expect("test invariant: should succeed");
        assert!(final_loss < initial, "final={final_loss} initial={initial}");
    }

    #[test]
    fn fit_decreases_similar_pair_distance() {
        let (data, n, dim, pairs) = make_labeled();
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.02,
            margin: 2.0,
            n_iter: 300,
        };
        let mut r = rng();
        let mut metric =
            MahalanobisMetric::new(&cfg, &mut r).expect("test invariant: should succeed");

        let x0 = &data[0..dim];
        let x1 = &data[dim..2 * dim];
        let before = metric
            .distance(x0, x1)
            .expect("test invariant: should succeed");
        metric
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");
        let after = metric
            .distance(x0, x1)
            .expect("test invariant: should succeed");
        assert!(after < before, "before={before} after={after}");
    }

    #[test]
    fn metric_stays_psd_after_fit() {
        let (data, n, dim, pairs) = make_labeled();
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.02,
            margin: 2.0,
            n_iter: 250,
        };
        let mut r = rng();
        let mut metric =
            MahalanobisMetric::new(&cfg, &mut r).expect("test invariant: should succeed");
        metric
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");

        let m = metric.metric_matrix();
        // zᵀ M z >= 0 for several random z.
        let mut zr = LcgRng::new(99);
        for _ in 0..40 {
            let z = rand_vec(dim, &mut zr);
            let mut quad = 0.0_f32;
            for a in 0..dim {
                for b in 0..dim {
                    quad += z[a] * m[a * dim + b] * z[b];
                }
            }
            assert!(quad >= -1e-6, "zᵀMz={quad}");
        }
    }

    #[test]
    fn fit_is_deterministic_given_seed() {
        let (data, n, dim, pairs) = make_labeled();
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.02,
            margin: 2.0,
            n_iter: 50,
        };
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        let mut m1 = MahalanobisMetric::new(&cfg, &mut r1).expect("test invariant: should succeed");
        let mut m2 = MahalanobisMetric::new(&cfg, &mut r2).expect("test invariant: should succeed");
        let l1 = m1
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");
        let l2 = m2
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");
        assert_eq!(l1, l2);
        assert_eq!(m1.factor(), m2.factor());
    }

    #[test]
    fn learned_factor_differs_from_identity() {
        let (data, n, dim, pairs) = make_labeled();
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.05,
            margin: 2.0,
            n_iter: 100,
        };
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        metric
            .fit(&data, n, &pairs, &cfg)
            .expect("test invariant: should succeed");
        let ident = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let mut diff = 0.0_f32;
        for (a, b) in metric.factor().iter().zip(ident.factor().iter()) {
            diff += (a - b).abs();
        }
        assert!(diff > 1e-4, "factor barely changed: {diff}");
    }

    #[test]
    fn distance_sq_matches_manual_quadratic_form() {
        // Hand example: dim=2, L = [[2,0],[1,3]].
        // M = L Lᵀ = [[4, 2],[2, 10]].
        // x-y = [1, 1] => (x-y)ᵀ M (x-y) = 4 + 2 + 2 + 10 = 18.
        let dim = 2;
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        metric.l = vec![2.0, 0.0, 1.0, 3.0];
        let x = vec![1.0_f32, 1.0];
        let y = vec![0.0_f32, 0.0];
        let dsq = metric
            .distance_sq(&x, &y)
            .expect("test invariant: should succeed");
        assert!((dsq - 18.0).abs() < 1e-5, "dsq={dsq}");
        // Cross-check against M built by metric_matrix.
        let m = metric.metric_matrix();
        assert!((m[0] - 4.0).abs() < 1e-5);
        assert!((m[1] - 2.0).abs() < 1e-5);
        assert!((m[2] - 2.0).abs() < 1e-5);
        assert!((m[3] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn err_distance_x_dim_mismatch() {
        let metric = MahalanobisMetric::identity(3).expect("test invariant: should succeed");
        let x = vec![1.0_f32, 2.0];
        let y = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            metric.distance(&x, &y),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_distance_y_dim_mismatch() {
        let metric = MahalanobisMetric::identity(3).expect("test invariant: should succeed");
        let x = vec![1.0_f32, 2.0, 3.0];
        let y = vec![1.0_f32, 2.0];
        assert!(matches!(
            metric.distance_sq(&x, &y),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_pair_index_out_of_range() {
        let dim = 2;
        let data = vec![0.0_f32, 0.0, 1.0, 1.0];
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.01,
            margin: 1.0,
            n_iter: 1,
        };
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let pairs = vec![(0, 5, true)];
        assert!(matches!(
            metric.fit(&data, 2, &pairs, &cfg),
            Err(AnnError::IdOutOfRange { .. })
        ));
    }

    #[test]
    fn err_data_length_mismatch() {
        let dim = 2;
        let data = vec![0.0_f32, 0.0, 1.0]; // should be 4 for n=2
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.01,
            margin: 1.0,
            n_iter: 1,
        };
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let pairs = vec![(0, 1, true)];
        assert!(matches!(
            metric.fit(&data, 2, &pairs, &cfg),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_lr_non_positive() {
        let dim = 2;
        let data = vec![0.0_f32, 0.0, 1.0, 1.0];
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.0,
            margin: 1.0,
            n_iter: 1,
        };
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let pairs = vec![(0, 1, true)];
        assert!(matches!(
            metric.fit(&data, 2, &pairs, &cfg),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_n_iter_zero() {
        let dim = 2;
        let data = vec![0.0_f32, 0.0, 1.0, 1.0];
        let cfg = MahalanobisConfig {
            dim,
            lr: 0.01,
            margin: 1.0,
            n_iter: 0,
        };
        let mut metric = MahalanobisMetric::identity(dim).expect("test invariant: should succeed");
        let pairs = vec![(0, 1, true)];
        assert!(matches!(
            metric.fit(&data, 2, &pairs, &cfg),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_dim_zero_identity() {
        assert!(matches!(
            MahalanobisMetric::identity(0),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn err_dim_zero_new() {
        let cfg = MahalanobisConfig {
            dim: 0,
            lr: 0.01,
            margin: 1.0,
            n_iter: 1,
        };
        let mut r = rng();
        assert!(matches!(
            MahalanobisMetric::new(&cfg, &mut r),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn new_is_deterministic_given_seed() {
        let cfg = MahalanobisConfig {
            dim: 4,
            lr: 0.01,
            margin: 1.0,
            n_iter: 1,
        };
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let m1 = MahalanobisMetric::new(&cfg, &mut r1).expect("test invariant: should succeed");
        let m2 = MahalanobisMetric::new(&cfg, &mut r2).expect("test invariant: should succeed");
        assert_eq!(m1.factor(), m2.factor());
    }
}
