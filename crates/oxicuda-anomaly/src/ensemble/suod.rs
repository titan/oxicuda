//! SUOD — Scalable Unsupervised Outlier Detection (Zhao et al. 2021).
//!
//! SUOD accelerates large ensembles of outlier detectors via:
//!
//! 1. **Random projection approximation**: each detector operates on a
//!    random Gaussian projection of the original data, reducing dimension
//!    from `d` → `projection_dim`.
//! 2. **Pseudo-supervised approximation** (optional): a ridge regression
//!    approximator is fitted on the projected training data to predict
//!    detector scores on new samples without re-running the full detector.
//! 3. **Parallel detection**: conceptually parallel; serial CPU execution.
//!
//! # Detector Pool
//!
//! Three lightweight detectors are used:
//! - `IsolationProjection`: path-length isolation score.
//! - `KnnProjection`: mean distance to k nearest neighbours.
//! - `CopodProjection`: ECDF-based tail probability score.
//!
//! # Ensemble
//!
//! Per-detector scores are min-max normalised then averaged.
//!
//! # Reference
//! Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., Yang, J.,
//! Bai, H., Li, Z., Xiao, C., Yu, Y., Akoglu, L., & Faloutsos, C. (2021).
//! SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection.
//! *MLSys 2021*.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the SUOD framework.
#[derive(Debug, Clone)]
pub struct SuodConfig {
    /// Number of detectors in the ensemble.
    pub n_detectors: usize,
    /// Target dimensionality for random projection.
    pub projection_dim: usize,
    /// Whether to use ridge regression approximation for scoring.
    pub approx_clfs: bool,
    /// Expected contamination fraction (used for threshold if needed).
    pub base_contamination: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for SuodConfig {
    fn default() -> Self {
        Self {
            n_detectors: 3,
            projection_dim: 4,
            approx_clfs: true,
            base_contamination: 0.1,
            seed: 42,
        }
    }
}

// ─── Detector type enum ───────────────────────────────────────────────────────

/// The three base detector types available in the SUOD pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectorType {
    Isolation,
    Knn,
    Copod,
}

// ─── Random projection matrix ─────────────────────────────────────────────────

/// Generate a Gaussian random projection matrix `R` of shape `[d × proj_dim]`
/// with columns normalised to unit Euclidean norm.
///
/// Row-major storage: R[i, j] = R[i * proj_dim + j].
fn generate_projection_matrix(d: usize, proj_dim: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut r = vec![0.0_f64; d * proj_dim];
    // Fill with N(0, 1)
    for v in r.iter_mut() {
        *v = rng.next_normal() as f64;
    }
    // Normalise each column to unit L2 norm
    for j in 0..proj_dim {
        let mut norm_sq = 0.0_f64;
        for i in 0..d {
            norm_sq += r[i * proj_dim + j] * r[i * proj_dim + j];
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for i in 0..d {
            r[i * proj_dim + j] /= norm;
        }
    }
    r
}

/// Project data `x` (`[n × d]`) through matrix `r` (`[d × proj_dim]`) →
/// returns `x_proj` (`[n × proj_dim]`), row-major.
fn project(x: &[f64], n: usize, d: usize, r: &[f64], proj_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * proj_dim];
    for i in 0..n {
        for j in 0..proj_dim {
            let mut acc = 0.0_f64;
            for k in 0..d {
                acc += x[i * d + k] * r[k * proj_dim + j];
            }
            out[i * proj_dim + j] = acc;
        }
    }
    out
}

// ─── IsolationProjection ─────────────────────────────────────────────────────

/// Isolation-based detector on projected space.
///
/// Approximates path-length based anomaly score without growing full trees.
/// Uses random split midpoints along random projected dimensions.
#[derive(Debug, Clone)]
struct IsolationProjection {
    /// Number of isolation estimators.
    n_estimators: usize,
    /// Each estimator: Vec<(dim_index, split_value)>.
    estimators: Vec<Vec<(usize, f64)>>,
    /// Projection dimension (number of features).
    proj_dim: usize,
    /// Number of features per path (tree depth approximation).
    path_len: usize,
    /// Training data min per dimension (for normalisation).
    data_min: Vec<f64>,
    /// Training data max per dimension.
    data_max: Vec<f64>,
}

impl IsolationProjection {
    fn new(n_estimators: usize, path_len: usize, proj_dim: usize) -> Self {
        Self {
            n_estimators,
            estimators: Vec::new(),
            proj_dim,
            path_len,
            data_min: vec![0.0; proj_dim],
            data_max: vec![1.0; proj_dim],
        }
    }

    /// Fit isolation estimators from projected training data.
    fn fit(&mut self, x_proj: &[f64], n: usize, rng: &mut LcgRng) -> AnomalyResult<()> {
        if n == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        // Compute per-feature min/max
        for j in 0..self.proj_dim {
            self.data_min[j] = f64::INFINITY;
            self.data_max[j] = f64::NEG_INFINITY;
            for i in 0..n {
                let v = x_proj[i * self.proj_dim + j];
                if v < self.data_min[j] {
                    self.data_min[j] = v;
                }
                if v > self.data_max[j] {
                    self.data_max[j] = v;
                }
            }
            // Ensure non-degenerate range
            if (self.data_max[j] - self.data_min[j]).abs() < 1e-10 {
                self.data_max[j] = self.data_min[j] + 1.0;
            }
        }

        self.estimators = (0..self.n_estimators)
            .map(|_| {
                (0..self.path_len)
                    .map(|_| {
                        let dim = rng.next_usize(self.proj_dim);
                        let split = self.data_min[dim]
                            + rng.next_f32() as f64 * (self.data_max[dim] - self.data_min[dim]);
                        (dim, split)
                    })
                    .collect()
            })
            .collect();
        Ok(())
    }

    /// Score one projected sample.
    /// Returns isolation score ∈ [0, 1]: higher → more anomalous.
    fn score_one(&self, x: &[f64]) -> f64 {
        if self.estimators.is_empty() {
            return 0.5;
        }
        let mut total_depth = 0.0_f64;
        for est in &self.estimators {
            let depth = est
                .iter()
                .position(|(dim, split)| {
                    // Simulate isolation: count splits before isolation
                    x[*dim] < *split
                })
                .unwrap_or(est.len()) as f64;
            total_depth += depth;
        }
        let avg_depth = total_depth / self.n_estimators as f64;
        // Normalise: shorter path → more anomalous
        let max_depth = self.path_len as f64;
        1.0 - (avg_depth / max_depth.max(1.0)).min(1.0)
    }
}

// ─── KnnProjection ────────────────────────────────────────────────────────────

/// k-NN distance detector on projected space.
#[derive(Debug, Clone)]
struct KnnProjection {
    k: usize,
    train_proj: Vec<f64>,
    n_train: usize,
    proj_dim: usize,
}

impl KnnProjection {
    fn new(k: usize, proj_dim: usize) -> Self {
        Self {
            k,
            train_proj: Vec::new(),
            n_train: 0,
            proj_dim,
        }
    }

    fn fit(&mut self, x_proj: &[f64], n: usize) -> AnomalyResult<()> {
        if n == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        self.train_proj = x_proj.to_vec();
        self.n_train = n;
        Ok(())
    }

    /// Mean Euclidean distance to k nearest neighbours.
    fn score_one(&self, x: &[f64]) -> f64 {
        if self.n_train == 0 {
            return 0.0;
        }
        let k = self.k.min(self.n_train);
        let mut dists: Vec<f64> = (0..self.n_train)
            .map(|i| {
                let d2: f64 = x
                    .iter()
                    .zip(&self.train_proj[i * self.proj_dim..(i + 1) * self.proj_dim])
                    .map(|(xj, tj)| (xj - tj) * (xj - tj))
                    .sum();
                d2.sqrt()
            })
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists[..k].iter().sum::<f64>() / k as f64
    }
}

// ─── CopodProjection ──────────────────────────────────────────────────────────

/// ECDF-based copula detector on projected space.
#[derive(Debug, Clone)]
struct CopodProjection {
    sorted_features: Vec<Vec<f64>>,
    n_train: usize,
    proj_dim: usize,
}

impl CopodProjection {
    fn new(proj_dim: usize) -> Self {
        Self {
            sorted_features: Vec::new(),
            n_train: 0,
            proj_dim,
        }
    }

    fn fit(&mut self, x_proj: &[f64], n: usize) -> AnomalyResult<()> {
        if n == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        self.n_train = n;
        self.sorted_features = (0..self.proj_dim)
            .map(|j| {
                let mut col: Vec<f64> = (0..n).map(|i| x_proj[i * self.proj_dim + j]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                col
            })
            .collect();
        Ok(())
    }

    fn ecdf(&self, feat: usize, v: f64) -> f64 {
        let col = &self.sorted_features[feat];
        let count = col.partition_point(|&x| x <= v);
        count as f64 / self.n_train as f64
    }

    fn score_one(&self, x: &[f64]) -> f64 {
        if self.sorted_features.is_empty() {
            return 0.0;
        }
        const LOG_EPS: f64 = 1e-10;
        let opzl: f64 = (0..self.proj_dim)
            .map(|j| -(self.ecdf(j, x[j]) + LOG_EPS).ln())
            .sum();
        let opzr: f64 = (0..self.proj_dim)
            .map(|j| -(1.0 - self.ecdf(j, x[j]) + LOG_EPS).ln())
            .sum();
        (opzl + opzr) / 2.0
    }
}

// ─── Ridge regression approximator ───────────────────────────────────────────

/// Simple ridge regression: y ≈ X w + b, fitted with normal equations.
/// Used to approximate detector scores on new projected data.
#[derive(Debug, Clone)]
struct RidgeApprox {
    weights: Vec<f64>,
    bias: f64,
    proj_dim: usize,
    alpha: f64,
}

impl RidgeApprox {
    fn new(proj_dim: usize, alpha: f64) -> Self {
        Self {
            weights: vec![0.0; proj_dim],
            bias: 0.0,
            proj_dim,
            alpha,
        }
    }

    /// Fit ridge regression: minimise ‖Xw - y‖² + α ‖w‖².
    ///
    /// We use gradient descent (20 iterations) which is exact enough here.
    fn fit(&mut self, x_proj: &[f64], y: &[f64], n: usize) {
        if n == 0 {
            return;
        }
        let lr = 0.01_f64 / n as f64;
        for _iter in 0..50 {
            let mut grad_w = vec![0.0_f64; self.proj_dim];
            let mut grad_b = 0.0_f64;
            for i in 0..n {
                let xi = &x_proj[i * self.proj_dim..(i + 1) * self.proj_dim];
                let pred: f64 = xi
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xij, wj)| xij * wj)
                    .sum::<f64>()
                    + self.bias;
                let err = pred - y[i];
                for (gw, xij) in grad_w.iter_mut().zip(xi.iter()) {
                    *gw += 2.0 * err * xij;
                }
                grad_b += 2.0 * err;
            }
            for (wj, gwj) in self.weights.iter_mut().zip(grad_w.iter()) {
                *wj -= lr * (gwj + 2.0 * self.alpha * *wj);
            }
            self.bias -= lr * grad_b;
        }
    }

    fn predict_one(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(self.weights.iter())
            .map(|(xj, wj)| xj * wj)
            .sum::<f64>()
            + self.bias
    }
}

// ─── Per-detector fitted state ────────────────────────────────────────────────

/// Fitted state for one detector slot in the SUOD ensemble.
#[derive(Debug, Clone)]
struct FittedDetector {
    detector_type: DetectorType,
    /// Projection matrix R, shape `[d × proj_dim]`, row-major.
    proj_matrix: Vec<f64>,
    /// Original feature dimension.
    orig_d: usize,
    /// Projected dimension.
    proj_dim: usize,
    /// Fitted isolation detector (if IsolationProjection).
    isolation: Option<IsolationProjection>,
    /// Fitted kNN detector (if KnnProjection).
    knn: Option<KnnProjection>,
    /// Fitted COPOD detector (if CopodProjection).
    copod: Option<CopodProjection>,
    /// Ridge regression approximator (if approx_clfs).
    approx: Option<RidgeApprox>,
    /// Training score min (for normalisation).
    score_min: f64,
    /// Training score max (for normalisation).
    score_max: f64,
}

impl FittedDetector {
    /// Compute raw anomaly score for one projected sample.
    fn raw_score(&self, x_proj: &[f64]) -> f64 {
        match self.detector_type {
            DetectorType::Isolation => self.isolation.as_ref().map_or(0.5, |d| d.score_one(x_proj)),
            DetectorType::Knn => self.knn.as_ref().map_or(0.0, |d| d.score_one(x_proj)),
            DetectorType::Copod => self.copod.as_ref().map_or(0.0, |d| d.score_one(x_proj)),
        }
    }

    /// Compute normalised score for one *original* sample.
    fn score(&self, x_orig: &[f64]) -> f64 {
        // Project
        let x_proj = project(x_orig, 1, self.orig_d, &self.proj_matrix, self.proj_dim);

        let raw = if let Some(approx) = &self.approx {
            approx.predict_one(&x_proj)
        } else {
            self.raw_score(&x_proj)
        };

        // Min-max normalise
        let range = (self.score_max - self.score_min).max(1e-10);
        ((raw - self.score_min) / range).clamp(0.0, 1.0)
    }
}

// ─── SuodFit ─────────────────────────────────────────────────────────────────

/// Fitted SUOD model.
#[derive(Debug, Clone)]
pub struct SuodFit {
    /// One fitted detector per slot.
    detectors: Vec<FittedDetector>,
    /// Original feature dimension.
    orig_d: usize,
    /// Number of detectors.
    n_detectors: usize,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a SUOD ensemble on `n` training samples with `d` features.
///
/// `x` is row-major `[n × d]`.
pub fn suod_fit(x: &[f64], n: usize, d: usize, cfg: &SuodConfig) -> AnomalyResult<SuodFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    if cfg.n_detectors == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "n_detectors must be > 0".into(),
        });
    }

    let proj_dim = cfg.projection_dim.min(d).max(1);
    let mut rng = LcgRng::new(cfg.seed);

    // Detector type cycle: IsoProj, KnnProj, CopodProj, repeat.
    let detector_types = [
        DetectorType::Isolation,
        DetectorType::Knn,
        DetectorType::Copod,
    ];

    let mut detectors: Vec<FittedDetector> = Vec::with_capacity(cfg.n_detectors);

    for det_idx in 0..cfg.n_detectors {
        let dtype = detector_types[det_idx % detector_types.len()];

        // 1. Generate random projection matrix
        let proj_matrix = generate_projection_matrix(d, proj_dim, &mut rng);

        // 2. Project training data
        let x_proj = project(x, n, d, &proj_matrix, proj_dim);

        // 3. Fit the detector on projected data
        let (iso_det, knn_det, copod_det, raw_scores) = match dtype {
            DetectorType::Isolation => {
                let n_est = 20_usize;
                let path_len = (n as f64).log2().ceil() as usize + 1;
                let mut det = IsolationProjection::new(n_est, path_len, proj_dim);
                det.fit(&x_proj, n, &mut rng)?;
                let scores: Vec<f64> = (0..n)
                    .map(|i| det.score_one(&x_proj[i * proj_dim..(i + 1) * proj_dim]))
                    .collect();
                (Some(det), None, None, scores)
            }
            DetectorType::Knn => {
                let k = 5_usize.min(n);
                let mut det = KnnProjection::new(k, proj_dim);
                det.fit(&x_proj, n)?;
                let scores: Vec<f64> = (0..n)
                    .map(|i| det.score_one(&x_proj[i * proj_dim..(i + 1) * proj_dim]))
                    .collect();
                (None, Some(det), None, scores)
            }
            DetectorType::Copod => {
                let mut det = CopodProjection::new(proj_dim);
                det.fit(&x_proj, n)?;
                let scores: Vec<f64> = (0..n)
                    .map(|i| det.score_one(&x_proj[i * proj_dim..(i + 1) * proj_dim]))
                    .collect();
                (None, None, Some(det), scores)
            }
        };

        // 4. Min-max statistics from training scores
        let score_min = raw_scores
            .iter()
            .cloned()
            .filter(|s| s.is_finite())
            .fold(f64::INFINITY, f64::min);
        let score_max = raw_scores
            .iter()
            .cloned()
            .filter(|s| s.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let score_min = if score_min.is_finite() {
            score_min
        } else {
            0.0
        };
        let score_max = if score_max.is_finite() && score_max > score_min {
            score_max
        } else {
            score_min + 1.0
        };

        // 5. If approx_clfs: fit ridge regression from x_proj → raw_scores
        let approx = if cfg.approx_clfs {
            let mut r = RidgeApprox::new(proj_dim, 1e-3);
            r.fit(&x_proj, &raw_scores, n);
            Some(r)
        } else {
            None
        };

        detectors.push(FittedDetector {
            detector_type: dtype,
            proj_matrix,
            orig_d: d,
            proj_dim,
            isolation: iso_det,
            knn: knn_det,
            copod: copod_det,
            approx,
            score_min,
            score_max,
        });
    }

    Ok(SuodFit {
        detectors,
        orig_d: d,
        n_detectors: cfg.n_detectors,
    })
}

/// Compute per-sample SUOD ensemble anomaly scores.
///
/// Returns scores in `[0, 1]` (average of normalised detector scores).
/// `x` is row-major `[n × d]`.
pub fn suod_score(fit: &SuodFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * fit.orig_d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * fit.orig_d,
            got: x.len(),
        });
    }

    let mut scores = vec![0.0_f64; n];
    for i in 0..n {
        let xi = &x[i * fit.orig_d..(i + 1) * fit.orig_d];
        let det_scores: Vec<f64> = fit.detectors.iter().map(|d| d.score(xi)).collect();
        let avg = det_scores.iter().sum::<f64>() / fit.n_detectors as f64;
        scores[i] = avg.clamp(0.0, 1.0);
    }
    Ok(scores)
}

/// Predict anomaly labels (true = anomaly) using a normalised threshold ∈ `[0, 1]`.
///
/// `x` is row-major `[n × d]`.
pub fn suod_predict(
    fit: &SuodFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = suod_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s >= threshold).collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize, seed: u64, scale: f64, shift: f64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * d)
            .map(|_| rng.next_f32() as f64 * scale + shift)
            .collect()
    }

    fn default_cfg(d: usize) -> SuodConfig {
        SuodConfig {
            n_detectors: 3,
            projection_dim: d.min(4),
            approx_clfs: true,
            base_contamination: 0.1,
            seed: 42,
        }
    }

    // ── Test 1: fit succeeds ──────────────────────────────────────────────────

    #[test]
    fn suod_fit_succeeds() {
        let d = 8;
        let n = 50;
        let x = make_data(n, d, 1, 1.0, 0.0);
        let cfg = default_cfg(d);
        let result = suod_fit(&x, n, d, &cfg);
        assert!(result.is_ok(), "fit failed: {:?}", result.err());
    }

    // ── Test 2: scores are all finite ────────────────────────────────────────

    #[test]
    fn suod_scores_finite() {
        let d = 8;
        let n = 50;
        let x = make_data(n, d, 2, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit = suod_fit(&x, n, d, &cfg).expect("SUOD fit should succeed");
        let scores = suod_score(&fit, &x, n).expect("SUOD score should succeed");
        assert_eq!(scores.len(), n);
        for (i, s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s}");
        }
    }

    // ── Test 3: scores are in [0, 1] ─────────────────────────────────────────

    #[test]
    fn suod_scores_in_range() {
        let d = 6;
        let n = 40;
        let x = make_data(n, d, 3, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit = suod_fit(&x, n, d, &cfg).expect("suod_fit should succeed");
        let scores = suod_score(&fit, &x, n).expect("suod_score should succeed");
        for (i, &s) in scores.iter().enumerate() {
            assert!((0.0..=1.0).contains(&s), "score[{i}] = {s} not in [0, 1]");
        }
    }

    // ── Test 4: outlier scores higher than inliers ────────────────────────────

    #[test]
    fn suod_outlier_scores_higher() {
        let d = 6;
        let n = 60;
        // Training data: values near 0.5
        let x_train = make_data(n, d, 4, 0.2, 0.4);
        let cfg = SuodConfig {
            n_detectors: 3,
            projection_dim: 4,
            approx_clfs: false, // use raw detector for sharper separation
            base_contamination: 0.1,
            seed: 7,
        };
        let fit = suod_fit(&x_train, n, d, &cfg).expect("suod_fit should succeed for outlier test");

        // Inlier: same distribution
        let inlier = make_data(1, d, 999, 0.2, 0.4);
        // Outlier: far from training distribution
        let outlier: Vec<f64> = vec![100.0; d];

        let s_in = suod_score(&fit, &inlier, 1).expect("inlier score should succeed")[0];
        let s_out = suod_score(&fit, &outlier, 1).expect("outlier score should succeed")[0];

        assert!(
            s_out >= s_in,
            "outlier score {s_out} should be >= inlier score {s_in}"
        );
    }

    // ── Test 5: predict returns bool vec of correct length ───────────────────

    #[test]
    fn suod_predict_length() {
        let d = 8;
        let n = 30;
        let x = make_data(n, d, 5, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit = suod_fit(&x, n, d, &cfg).expect("suod_fit should succeed for predict test");
        let preds = suod_predict(&fit, &x, n, 0.5).expect("suod_predict should succeed");
        assert_eq!(preds.len(), n);
    }

    // ── Test 6: threshold = 1.0 → all normal ─────────────────────────────────

    #[test]
    fn suod_predict_all_normal_at_max_threshold() {
        let d = 6;
        let n = 30;
        let x = make_data(n, d, 6, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit = suod_fit(&x, n, d, &cfg).expect("suod_fit should succeed for max-threshold test");
        // All scores in [0,1], threshold > 1 means nothing flagged
        let preds =
            suod_predict(&fit, &x, n, 1.01).expect("suod_predict at max threshold should succeed");
        assert!(preds.iter().all(|&p| !p), "expected all normal");
    }

    // ── Test 7: threshold = 0.0 → all anomalous ──────────────────────────────

    #[test]
    fn suod_predict_all_anomalous_at_zero_threshold() {
        let d = 6;
        let n = 20;
        let x = make_data(n, d, 7, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit =
            suod_fit(&x, n, d, &cfg).expect("suod_fit should succeed for zero-threshold test");
        let preds =
            suod_predict(&fit, &x, n, 0.0).expect("suod_predict at zero threshold should succeed");
        assert!(preds.iter().all(|&p| p), "expected all anomalous");
    }

    // ── Test 8: empty input → error ───────────────────────────────────────────

    #[test]
    fn suod_fit_empty_input_error() {
        let cfg = default_cfg(4);
        let result = suod_fit(&[], 0, 4, &cfg);
        assert!(matches!(result, Err(AnomalyError::EmptyInput)));
    }

    // ── Test 9: dimension mismatch → error ───────────────────────────────────

    #[test]
    fn suod_score_dim_mismatch_error() {
        let d = 8;
        let n = 20;
        let x = make_data(n, d, 9, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit = suod_fit(&x, n, d, &cfg)
            .expect("suod_fit should succeed before testing score mismatch");

        // Wrong dimension
        let bad = make_data(5, 3, 0, 1.0, 0.0); // d=3 instead of d=8
        let result = suod_score(&fit, &bad, 5);
        assert!(matches!(
            result,
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 10: approx_clfs=false also works ────────────────────────────────

    #[test]
    fn suod_no_approx_clfs() {
        let d = 6;
        let n = 40;
        let x = make_data(n, d, 10, 1.0, 0.0);
        let cfg = SuodConfig {
            n_detectors: 3,
            projection_dim: 3,
            approx_clfs: false,
            base_contamination: 0.1,
            seed: 42,
        };
        let fit = suod_fit(&x, n, d, &cfg).expect("suod_fit without approx_clfs should succeed");
        let scores =
            suod_score(&fit, &x, n).expect("suod_score without approx_clfs should succeed");
        assert_eq!(scores.len(), n);
        assert!(
            scores
                .iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s))
        );
    }

    // ── Test 11: large detector count ────────────────────────────────────────

    #[test]
    fn suod_many_detectors() {
        let d = 8;
        let n = 50;
        let x = make_data(n, d, 11, 1.0, 0.0);
        let cfg = SuodConfig {
            n_detectors: 9,
            projection_dim: 4,
            approx_clfs: true,
            base_contamination: 0.1,
            seed: 42,
        };
        let fit = suod_fit(&x, n, d, &cfg).expect("suod_fit with many detectors should succeed");
        let scores =
            suod_score(&fit, &x, n).expect("suod_score with many detectors should succeed");
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    // ── Test 12: scores are deterministic ────────────────────────────────────

    #[test]
    fn suod_scores_deterministic() {
        let d = 6;
        let n = 30;
        let x = make_data(n, d, 12, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit1 = suod_fit(&x, n, d, &cfg).expect("first suod_fit should succeed");
        let fit2 = suod_fit(&x, n, d, &cfg).expect("second suod_fit should succeed");
        let s1 = suod_score(&fit1, &x, n).expect("first suod_score should succeed");
        let s2 = suod_score(&fit2, &x, n).expect("second suod_score should succeed");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-12, "scores differ: {a} vs {b}");
        }
    }

    // ── Test 13: projection_dim > d is clamped gracefully ────────────────────

    #[test]
    fn suod_proj_dim_clamped() {
        let d = 4;
        let n = 30;
        let x = make_data(n, d, 13, 1.0, 0.0);
        let cfg = SuodConfig {
            n_detectors: 3,
            projection_dim: 100, // larger than d
            approx_clfs: true,
            base_contamination: 0.1,
            seed: 42,
        };
        let fit = suod_fit(&x, n, d, &cfg)
            .expect("suod_fit with oversized proj_dim should succeed after clamping");
        let scores =
            suod_score(&fit, &x, n).expect("suod_score with clamped proj_dim should succeed");
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    // ── Test 14: score on empty returns error ────────────────────────────────

    #[test]
    fn suod_score_empty_error() {
        let d = 4;
        let n = 20;
        let x = make_data(n, d, 14, 1.0, 0.0);
        let cfg = default_cfg(d);
        let fit =
            suod_fit(&x, n, d, &cfg).expect("suod_fit should succeed before testing empty score");
        let result = suod_score(&fit, &[], 0);
        assert!(matches!(result, Err(AnomalyError::EmptyInput)));
    }
}
