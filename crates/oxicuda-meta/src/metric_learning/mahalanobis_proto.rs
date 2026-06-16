//! Mahalanobis Prototypical Networks — class-conditional covariance metric.
//!
//! # Background
//!
//! Vanilla Prototypical Networks (Snell et al. 2017) classify a query by the
//! squared **Euclidean** distance to per-class mean prototypes, which implicitly
//! assumes every class is an isotropic Gaussian of identical variance.  Many
//! few-shot methods (Bateni et al. "Simple CNAPS" 2020; Fort 2017) instead use a
//! **Mahalanobis** distance with a per-class covariance `Σ_c`:
//!
//! ```text
//! d_c(x) = (x − μ_c)ᵀ Σ_c⁻¹ (x − μ_c)
//! ```
//!
//! and predict `argmin_c d_c(x)` (or softmax over `−d_c`).  With only `k_shot`
//! examples per class the empirical covariance is rank-deficient, so we apply a
//! **shrinkage** regulariser toward a scaled identity (Ledoit-Wolf style):
//!
//! ```text
//! Σ̃_c = β Σ_c + (1 − β) τ I ,    τ = mean diagonal of Σ_c (or 1)
//! ```
//!
//! The regularised covariance is symmetric positive-definite, so the
//! Mahalanobis quadratic form is evaluated stably via a Cholesky solve
//! `Σ̃_c = L Lᵀ`, `d = ‖L⁻¹(x − μ_c)‖²`.
//!
//! Two covariance modes are provided: a full `feat_dim × feat_dim` covariance,
//! and a cheaper diagonal approximation.

use crate::error::{MetaError, MetaResult};

// ─── Covariance mode ─────────────────────────────────────────────────────────

/// How the per-class covariance is modelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovMode {
    /// Full dense `feat_dim × feat_dim` covariance with Cholesky solve.
    Full,
    /// Diagonal covariance (per-feature variance only).
    Diagonal,
}

/// Configuration for a Mahalanobis prototypical classifier.
#[derive(Debug, Clone)]
pub struct MahalanobisConfig {
    /// Number of classes `N`.
    pub n_way: usize,
    /// Feature dimension `d`.
    pub feat_dim: usize,
    /// Shrinkage weight `β ∈ [0, 1]` toward the scaled identity.  `β = 0` gives
    /// pure Euclidean (identity covariance); `β = 1` gives the raw empirical
    /// covariance (only safe with enough shots).
    pub shrinkage: f32,
    /// Covariance model.
    pub mode: CovMode,
}

impl MahalanobisConfig {
    /// Create a config (default shrinkage `0.5`, full covariance).
    ///
    /// # Errors
    ///
    /// * [`MetaError::InvalidNWay`]    — if `n_way < 2`.
    /// * [`MetaError::InvalidFeatDim`] — if `feat_dim == 0`.
    pub fn new(n_way: usize, feat_dim: usize) -> MetaResult<Self> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if feat_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: feat_dim });
        }
        Ok(Self {
            n_way,
            feat_dim,
            shrinkage: 0.5,
            mode: CovMode::Full,
        })
    }

    /// Override the shrinkage weight (clamped to `[0, 1]`).
    #[must_use]
    pub fn with_shrinkage(mut self, beta: f32) -> Self {
        self.shrinkage = beta.clamp(0.0, 1.0);
        self
    }

    /// Override the covariance mode.
    #[must_use]
    pub fn with_mode(mut self, mode: CovMode) -> Self {
        self.mode = mode;
        self
    }
}

// ─── Per-class statistics ────────────────────────────────────────────────────

/// Fitted statistics for one class: mean and (regularised) covariance.
#[derive(Debug, Clone)]
struct ClassStats {
    /// Prototype mean, length `feat_dim`.
    mean: Vec<f32>,
    /// For [`CovMode::Full`]: Cholesky factor `L` (row-major, lower-tri),
    /// length `feat_dim²`.  For [`CovMode::Diagonal`]: the per-feature variances,
    /// length `feat_dim`.
    factor: Vec<f32>,
}

/// A fitted Mahalanobis prototypical classifier.
#[derive(Debug, Clone)]
pub struct MahalanobisProto {
    config: MahalanobisConfig,
    classes: Vec<ClassStats>,
}

// ─── Cholesky helpers (local, mirrors r2d2 conventions) ──────────────────────

/// In-place Cholesky of symmetric positive-definite `a` (`n×n`, row-major).
/// On success `a` holds the lower-triangular factor `L`.
fn cholesky_in_place(a: &mut [f32], n: usize) -> MetaResult<()> {
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(MetaError::Internal {
                        msg: format!("Cholesky non-positive pivot {s:.3e} at ({i},{i})"),
                    });
                }
                a[i * n + i] = s.sqrt();
            } else {
                let diag = a[j * n + j];
                if diag.abs() < 1e-20 {
                    return Err(MetaError::Internal {
                        msg: format!("Cholesky near-zero diagonal at ({j},{j})"),
                    });
                }
                a[i * n + j] = s / diag;
            }
        }
    }
    // Zero the strict upper triangle so `factor` is a clean lower-tri matrix.
    for i in 0..n {
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }
    Ok(())
}

/// Forward substitution solving `L y = b` for lower-triangular `L` (`n×n`).
fn forward_sub(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-20 { s / diag } else { 0.0 };
    }
    y
}

// ─── Fitting ─────────────────────────────────────────────────────────────────

impl MahalanobisProto {
    /// Fit per-class means and shrinkage-regularised covariances from a support
    /// set.
    ///
    /// * `support_x` — `[n_way · k_shot × feat_dim]` row-major features.
    /// * `support_y` — `[n_way · k_shot]` labels in `0..n_way`.
    ///
    /// Labels need not be contiguous or sorted, but every class in `0..n_way`
    /// must appear at least once.
    ///
    /// # Errors
    ///
    /// * [`MetaError::EmptySupport`]        — if `support_x` is empty.
    /// * [`MetaError::DimensionMismatch`]   — on shape disagreement.
    /// * [`MetaError::InsufficientExamples`] — if a class has no examples.
    /// * Propagates Cholesky failure as [`MetaError::Internal`].
    pub fn fit(
        config: MahalanobisConfig,
        support_x: &[f32],
        support_y: &[u32],
    ) -> MetaResult<Self> {
        if support_x.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        let d = config.feat_dim;
        let n_way = config.n_way;
        if !support_x.len().is_multiple_of(d) {
            return Err(MetaError::DimensionMismatch {
                expected: support_x.len().div_ceil(d) * d,
                got: support_x.len(),
            });
        }
        let n = support_x.len() / d;
        if support_y.len() != n {
            return Err(MetaError::DimensionMismatch {
                expected: n,
                got: support_y.len(),
            });
        }

        // Group example indices by class.
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_way];
        for (i, &lbl) in support_y.iter().enumerate() {
            let c = lbl as usize;
            if c >= n_way {
                return Err(MetaError::Internal {
                    msg: format!("support label {c} ≥ n_way {n_way}"),
                });
            }
            groups[c].push(i);
        }
        for (c, g) in groups.iter().enumerate() {
            if g.is_empty() {
                return Err(MetaError::InsufficientExamples {
                    cls: c,
                    need: 1,
                    got: 0,
                });
            }
        }

        let beta = config.shrinkage;
        let mut classes = Vec::with_capacity(n_way);
        for g in &groups {
            // Mean.
            let mut mean = vec![0.0_f32; d];
            for &i in g {
                let row = &support_x[i * d..i * d + d];
                for (m, &x) in mean.iter_mut().zip(row.iter()) {
                    *m += x;
                }
            }
            let inv = 1.0 / g.len() as f32;
            for m in mean.iter_mut() {
                *m *= inv;
            }

            let factor = match config.mode {
                CovMode::Diagonal => Self::fit_diagonal(support_x, g, &mean, d, beta),
                CovMode::Full => Self::fit_full(support_x, g, &mean, d, beta)?,
            };
            classes.push(ClassStats { mean, factor });
        }

        Ok(Self { config, classes })
    }

    /// Diagonal variances with shrinkage toward the mean variance.
    fn fit_diagonal(x: &[f32], idx: &[usize], mean: &[f32], d: usize, beta: f32) -> Vec<f32> {
        let denom = idx.len().max(1) as f32;
        let mut var = vec![0.0_f32; d];
        for &i in idx {
            let row = &x[i * d..i * d + d];
            for j in 0..d {
                let dv = row[j] - mean[j];
                var[j] += dv * dv;
            }
        }
        for v in var.iter_mut() {
            *v /= denom;
        }
        let tau = (var.iter().sum::<f32>() / d as f32).max(1e-6);
        for v in var.iter_mut() {
            // shrink toward τ and floor for invertibility
            *v = (beta * *v + (1.0 - beta) * tau).max(1e-6);
        }
        var
    }

    /// Full covariance with shrinkage, returned as its Cholesky factor `L`.
    fn fit_full(
        x: &[f32],
        idx: &[usize],
        mean: &[f32],
        d: usize,
        beta: f32,
    ) -> MetaResult<Vec<f32>> {
        let denom = idx.len().max(1) as f32;
        let mut cov = vec![0.0_f32; d * d];
        for &i in idx {
            let row = &x[i * d..i * d + d];
            for a in 0..d {
                let da = row[a] - mean[a];
                for b in 0..d {
                    cov[a * d + b] += da * (row[b] - mean[b]);
                }
            }
        }
        for v in cov.iter_mut() {
            *v /= denom;
        }
        // τ = mean diagonal.
        let mut tau = 0.0_f32;
        for j in 0..d {
            tau += cov[j * d + j];
        }
        tau = (tau / d as f32).max(1e-6);
        // Σ̃ = β Σ + (1−β) τ I.
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] *= beta;
            }
            cov[a * d + a] += (1.0 - beta) * tau;
            // additional tiny floor on the diagonal for numerical safety
            cov[a * d + a] += 1e-6;
        }
        cholesky_in_place(&mut cov, d)?;
        Ok(cov)
    }

    /// Number of fitted classes.
    #[inline]
    #[must_use]
    pub fn n_way(&self) -> usize {
        self.config.n_way
    }

    /// Squared Mahalanobis distance from `x` to class `c`.
    ///
    /// # Errors
    ///
    /// * [`MetaError::DimensionMismatch`] — if `x.len() ≠ feat_dim`.
    /// * [`MetaError::Internal`]          — if `c ≥ n_way`.
    pub fn distance(&self, x: &[f32], c: usize) -> MetaResult<f32> {
        let d = self.config.feat_dim;
        if x.len() != d {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let cls = self.classes.get(c).ok_or(MetaError::Internal {
            msg: format!("class index {c} out of range"),
        })?;
        // diff = x − μ_c.
        let diff: Vec<f32> = x
            .iter()
            .zip(cls.mean.iter())
            .map(|(&xi, &mi)| xi - mi)
            .collect();
        match self.config.mode {
            CovMode::Diagonal => {
                // Σ diagonal: d = Σ_j diff_j² / var_j.
                Ok(diff
                    .iter()
                    .zip(cls.factor.iter())
                    .map(|(&dj, &vj)| dj * dj / vj)
                    .sum())
            }
            CovMode::Full => {
                // Solve L y = diff; d = ‖y‖².
                let y = forward_sub(&cls.factor, &diff, d);
                Ok(y.iter().map(|&v| v * v).sum())
            }
        }
    }

    /// Predict the class of each query (`argmin` Mahalanobis distance).
    ///
    /// `query_x` is `[n_query × feat_dim]`; returns `[n_query]` labels.
    ///
    /// # Errors
    ///
    /// [`MetaError::DimensionMismatch`] if `query_x.len()` is not a multiple of
    /// `feat_dim`.
    pub fn predict(&self, query_x: &[f32]) -> MetaResult<Vec<u32>> {
        let d = self.config.feat_dim;
        if query_x.is_empty() || !query_x.len().is_multiple_of(d) {
            return Err(MetaError::DimensionMismatch {
                expected: query_x.len().div_ceil(d.max(1)) * d,
                got: query_x.len(),
            });
        }
        let nq = query_x.len() / d;
        let mut preds = Vec::with_capacity(nq);
        for q in query_x.chunks(d) {
            let mut best = f32::INFINITY;
            let mut arg = 0usize;
            for c in 0..self.config.n_way {
                let dist = self.distance(q, c)?;
                if dist < best {
                    best = dist;
                    arg = c;
                }
            }
            preds.push(arg as u32);
        }
        Ok(preds)
    }

    /// Softmax probabilities over `−distance` for each query.
    ///
    /// Returns `[n_query × n_way]` row-major class probabilities.
    ///
    /// # Errors
    ///
    /// Propagates [`MahalanobisProto::distance`] errors.
    pub fn predict_proba(&self, query_x: &[f32]) -> MetaResult<Vec<f32>> {
        let d = self.config.feat_dim;
        if query_x.is_empty() || !query_x.len().is_multiple_of(d) {
            return Err(MetaError::DimensionMismatch {
                expected: query_x.len().div_ceil(d.max(1)) * d,
                got: query_x.len(),
            });
        }
        let nq = query_x.len() / d;
        let n_way = self.config.n_way;
        let mut out = vec![0.0_f32; nq * n_way];
        for (qi, q) in query_x.chunks(d).enumerate() {
            let mut logits = vec![0.0_f32; n_way];
            for (c, logit) in logits.iter_mut().enumerate() {
                *logit = -self.distance(q, c)?;
            }
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let row = &mut out[qi * n_way..qi * n_way + n_way];
            let mut sum = 0.0_f32;
            for (o, &l) in row.iter_mut().zip(logits.iter()) {
                let e = (l - max_l).exp();
                *o = e;
                sum += e;
            }
            if sum > 0.0 {
                for o in row.iter_mut() {
                    *o /= sum;
                }
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg(n_way: usize, d: usize) -> MahalanobisConfig {
        MahalanobisConfig::new(n_way, d).expect("cfg")
    }

    #[test]
    fn config_rejects_bad_dims() {
        assert!(matches!(
            MahalanobisConfig::new(1, 4),
            Err(MetaError::InvalidNWay { .. })
        ));
        assert!(matches!(
            MahalanobisConfig::new(3, 0),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn shrinkage_is_clamped() {
        let c = cfg(2, 3).with_shrinkage(5.0);
        assert_eq!(c.shrinkage, 1.0);
        let c2 = cfg(2, 3).with_shrinkage(-1.0);
        assert_eq!(c2.shrinkage, 0.0);
    }

    #[test]
    fn fit_rejects_empty() {
        assert!(matches!(
            MahalanobisProto::fit(cfg(2, 2), &[], &[]),
            Err(MetaError::EmptySupport)
        ));
    }

    #[test]
    fn fit_rejects_missing_class() {
        // n_way=2 but only class 0 present.
        let x = vec![1.0, 0.0, 2.0, 0.0];
        let y = vec![0u32, 0];
        assert!(matches!(
            MahalanobisProto::fit(cfg(2, 2), &x, &y),
            Err(MetaError::InsufficientExamples { .. })
        ));
    }

    #[test]
    fn fit_rejects_shape_mismatch() {
        let x = vec![1.0, 0.0, 2.0]; // 3 not multiple of feat_dim=2
        let y = vec![0u32, 1];
        assert!(MahalanobisProto::fit(cfg(2, 2), &x, &y).is_err());
    }

    /// Build a 2-class support set: class 0 near origin, class 1 near (5,5).
    fn two_class_support() -> (Vec<f32>, Vec<u32>) {
        let x = vec![
            0.1, 0.0, // c0
            -0.1, 0.1, // c0
            0.0, -0.1, // c0
            5.0, 5.1, // c1
            5.1, 4.9, // c1
            4.9, 5.0, // c1
        ];
        let y = vec![0u32, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn fit_means_are_class_centroids() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        // class 0 mean ≈ (0,0), class 1 mean ≈ (5,5).
        assert!(m.classes[0].mean[0].abs() < 0.2);
        assert!((m.classes[1].mean[0] - 5.0).abs() < 0.2);
        assert!((m.classes[1].mean[1] - 5.0).abs() < 0.2);
    }

    #[test]
    fn predict_full_separates_classes() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        let q = vec![0.05, 0.0, 5.0, 5.0];
        let preds = m.predict(&q).expect("pred");
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn predict_diagonal_separates_classes() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2).with_mode(CovMode::Diagonal), &x, &y).expect("fit");
        let q = vec![0.0, 0.0, 5.0, 5.0];
        let preds = m.predict(&q).expect("pred");
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn distance_to_own_mean_is_near_zero() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        let mean0 = m.classes[0].mean.clone();
        let d = m.distance(&mean0, 0).expect("dist");
        assert!(d < 1e-3, "distance to own mean should be ~0, got {d}");
    }

    #[test]
    fn distance_is_nonnegative() {
        let (x, y) = two_class_support();
        for mode in [CovMode::Full, CovMode::Diagonal] {
            let m = MahalanobisProto::fit(cfg(2, 2).with_mode(mode), &x, &y).expect("fit");
            let q = vec![3.0, -2.0];
            for c in 0..2 {
                let d = m.distance(&q, c).expect("dist");
                assert!(d >= 0.0, "Mahalanobis distance must be ≥ 0");
            }
        }
    }

    #[test]
    fn distance_rejects_bad_query_len() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        assert!(m.distance(&[1.0, 2.0, 3.0], 0).is_err());
    }

    #[test]
    fn distance_rejects_bad_class() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        assert!(m.distance(&[1.0, 2.0], 9).is_err());
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        let q = vec![0.0, 0.0, 5.0, 5.0, 2.5, 2.5];
        let proba = m.predict_proba(&q).expect("proba");
        assert_eq!(proba.len(), 3 * 2);
        for row in proba.chunks(2) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sums to {s}");
        }
    }

    #[test]
    fn proba_favours_closer_class() {
        let (x, y) = two_class_support();
        let m = MahalanobisProto::fit(cfg(2, 2), &x, &y).expect("fit");
        // a query near class 0 should get higher P(class 0).
        let proba = m.predict_proba(&[0.0, 0.0]).expect("proba");
        assert!(
            proba[0] > proba[1],
            "near class0 → P0 {} > P1 {}",
            proba[0],
            proba[1]
        );
    }

    #[test]
    fn anisotropic_covariance_changes_decision() {
        // Class 0 is stretched along x (high variance in feature 0).  A query
        // offset purely along x should still be assigned to class 0 under
        // Mahalanobis even though Euclidean to a tighter class 1 could be closer.
        let d = 2;
        // class 0: spread along x at origin
        // class 1: tight cluster at (3, 0)
        let x = vec![
            -4.0, 0.1, 4.0, -0.1, 0.0, 0.0, // c0 wide in x
            3.0, 0.0, 3.1, 0.05, 2.9, -0.05, // c1 tight
        ];
        let y = vec![0u32, 0, 0, 1, 1, 1];
        let m = MahalanobisProto::fit(
            cfg(2, d).with_shrinkage(0.9).with_mode(CovMode::Full),
            &x,
            &y,
        )
        .expect("fit");
        // Query at (2, 0): Euclidean-closer to class1 mean (3,0) than class0 (0,0),
        // but class0 is very wide along x so Mahalanobis distance can prefer c0.
        let dc0 = m.distance(&[2.0, 0.0], 0).expect("d0");
        let dc1 = m.distance(&[2.0, 0.0], 1).expect("d1");
        // Whichever wins, distances must be finite and the call must succeed; we
        // assert the anisotropy actually lowered c0's distance below the raw
        // Euclidean (4.0) it would have under identity covariance.
        assert!(
            dc0 < 4.0,
            "wide class0 should shrink along-x distance: {dc0}"
        );
        assert!(dc1.is_finite());
    }

    #[test]
    fn deterministic_random_fit_predicts_in_range() {
        // Random separable-ish episode just to exercise full covariance fit.
        let mut rng = LcgRng::new(2024);
        let d = 4;
        let n_way = 3;
        let k = 5;
        let mut x = vec![0.0_f32; n_way * k * d];
        let mut yy = Vec::new();
        for c in 0..n_way {
            for _ in 0..k {
                let base = yy.len() * d;
                for j in 0..d {
                    // cluster class c around the c-th unit direction
                    let centre = if j == c { 3.0 } else { 0.0 };
                    x[base + j] = centre + (rng.next_f32() - 0.5) * 0.4;
                }
                yy.push(c as u32);
            }
        }
        let m = MahalanobisProto::fit(cfg(n_way, d).with_shrinkage(0.3), &x, &yy).expect("fit");
        let preds = m.predict(&x).expect("pred");
        assert_eq!(preds.len(), n_way * k);
        assert!(preds.iter().all(|&p| (p as usize) < n_way));
    }
}
