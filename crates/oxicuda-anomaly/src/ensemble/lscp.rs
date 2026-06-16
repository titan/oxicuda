//! LSCP — Locally Selective Combination in Parallel outlier ensembles
//! (Zhao et al. 2019).
//!
//! Given the per-sample scores of several base detectors, LSCP combines them by
//! **selecting locally competent detectors** instead of averaging globally:
//!
//! 1. For a test point (represented by its base-detector score vector) find its
//!    `local_region_size` (`k`) nearest neighbours among the training set in the
//!    base-detector **score space**. These neighbours form the *local region*.
//! 2. Generate a **pseudo ground truth** over the region — the maximum (or
//!    average) base-detector score per training point in the region.
//! 3. Measure each base detector's **Pearson correlation** with the pseudo
//!    ground truth over the region. The detector(s) most correlated locally are
//!    the *competent* ones.
//! 4. Score the test point with the competent detector(s): either the single
//!    best (`SelectBest`) or the average of all detectors within a correlation
//!    margin of the best (`AverageCompetent`).
//!
//! With a single base detector LSCP reduces to that detector. Locality is taken
//! in score space so the API only needs detector scores (matching the rest of
//! the `ensemble` module).
//!
//! # Reference
//! Zhao, Y., Nasrullah, Z., Hryniewicki, M. K., & Li, Z. (2019). *LSCP:
//! Locally Selective Combination in Parallel Outlier Ensembles*. SDM 2019.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Strategy for generating the local pseudo ground truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LscpTarget {
    /// Per-region-point maximum across base detectors (the LSCP default).
    Maximum,
    /// Per-region-point average across base detectors.
    Average,
}

/// Strategy for combining the locally competent detectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LscpStrategy {
    /// Use the single most-correlated detector's score.
    SelectBest,
    /// Average the scores of all detectors within `competent_margin` of the
    /// best local correlation.
    AverageCompetent,
}

/// Hyper-parameters for [`LscpEnsemble`].
#[derive(Debug, Clone)]
pub struct LscpConfig {
    /// Local region size `k` (number of nearest training points; default 10).
    pub local_region_size: usize,
    /// Pseudo-ground-truth generation strategy (default [`LscpTarget::Maximum`]).
    pub target: LscpTarget,
    /// Detector-combination strategy (default [`LscpStrategy::SelectBest`]).
    pub strategy: LscpStrategy,
    /// Correlation margin for [`LscpStrategy::AverageCompetent`] (default 0.1).
    pub competent_margin: f32,
}

impl Default for LscpConfig {
    fn default() -> Self {
        Self {
            local_region_size: 10,
            target: LscpTarget::Maximum,
            strategy: LscpStrategy::SelectBest,
            competent_margin: 0.1,
        }
    }
}

// ─── Pearson correlation ──────────────────────────────────────────────────────

/// Pearson correlation between two equal-length slices.
///
/// Returns `NaN` when either slice has (near-)zero variance, signalling an
/// undefined / non-informative correlation.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 1.0 {
        return f32::NAN;
    }
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 { f32::NAN } else { cov / denom }
}

/// Index of the largest finite value, or `0` when none are finite.
fn argmax_finite(values: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut found = false;
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_v = v;
            best = i;
            found = true;
        }
    }
    if found { best } else { 0 }
}

// ─── LscpEnsemble ─────────────────────────────────────────────────────────────

/// Locally Selective Combination in Parallel outlier ensemble.
///
/// # Usage
///
/// ```rust,ignore
/// let mut lscp = LscpEnsemble::new(LscpConfig::default());
/// lscp.fit(&train_scores, n_train, n_detectors)?;
/// let s = lscp.score(&test_scores)?;
/// ```
#[derive(Debug, Clone)]
pub struct LscpEnsemble {
    config: LscpConfig,
    /// Training base-detector scores, row-major `[n_train * n_detectors]`.
    train_scores: Vec<f32>,
    n_train: usize,
    n_detectors: usize,
    fitted: bool,
}

impl LscpEnsemble {
    /// Create an unfitted ensemble from the supplied configuration.
    #[must_use]
    pub fn new(config: LscpConfig) -> Self {
        Self {
            config,
            train_scores: Vec::new(),
            n_train: 0,
            n_detectors: 0,
            fitted: false,
        }
    }

    /// Fit on training base-detector scores.
    ///
    /// `train_scores` is row-major `[n_train × n_detectors]` (row `i` = the
    /// scores of every base detector for training sample `i`).
    ///
    /// # Errors
    /// * [`AnomalyError::EmptyInput`] if `n_train == 0`.
    /// * [`AnomalyError::InvalidFeatureCount`] if `n_detectors == 0`.
    /// * [`AnomalyError::InvalidK`] if `local_region_size == 0`.
    /// * [`AnomalyError::DimensionMismatch`] if
    ///   `train_scores.len() != n_train * n_detectors`.
    pub fn fit(
        &mut self,
        train_scores: &[f32],
        n_train: usize,
        n_detectors: usize,
    ) -> AnomalyResult<()> {
        if n_train == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_detectors == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if self.config.local_region_size == 0 {
            return Err(AnomalyError::InvalidK { k: 0 });
        }
        if train_scores.len() != n_train * n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_train * n_detectors,
                got: train_scores.len(),
            });
        }
        self.train_scores = train_scores.to_vec();
        self.n_train = n_train;
        self.n_detectors = n_detectors;
        self.fitted = true;
        Ok(())
    }

    /// Indices of the `k` nearest training rows to `s` in score space.
    fn local_region(&self, s: &[f32]) -> Vec<usize> {
        let k = self.config.local_region_size.min(self.n_train);
        let b = self.n_detectors;
        let mut dists: Vec<(usize, f32)> = (0..self.n_train)
            .map(|i| {
                let row = &self.train_scores[i * b..(i + 1) * b];
                let d: f32 = s
                    .iter()
                    .zip(row.iter())
                    .map(|(a, c)| {
                        let e = a - c;
                        e * e
                    })
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        dists[..k].iter().map(|&(i, _)| i).collect()
    }

    /// Pseudo ground truth (one value per region point).
    fn pseudo_target(&self, region: &[usize]) -> Vec<f32> {
        let b = self.n_detectors;
        region
            .iter()
            .map(|&i| {
                let row = &self.train_scores[i * b..(i + 1) * b];
                match self.config.target {
                    LscpTarget::Average => row.iter().sum::<f32>() / b as f32,
                    LscpTarget::Maximum => row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                }
            })
            .collect()
    }

    /// Per-detector Pearson correlation with the local pseudo ground truth.
    fn detector_correlations(&self, region: &[usize]) -> Vec<f32> {
        let b = self.n_detectors;
        let target = self.pseudo_target(region);
        (0..b)
            .map(|d| {
                let column: Vec<f32> = region
                    .iter()
                    .map(|&i| self.train_scores[i * b + d])
                    .collect();
                pearson(&column, &target)
            })
            .collect()
    }

    /// Index of the locally most competent base detector for `s`.
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] / [`AnomalyError::DimensionMismatch`].
    pub fn select_local_detector(&self, s: &[f32]) -> AnomalyResult<usize> {
        self.check_query(s)?;
        let region = self.local_region(s);
        let corrs = self.detector_correlations(&region);
        Ok(argmax_finite(&corrs))
    }

    /// Combined LSCP anomaly score for a single test point `s` (its base
    /// detectors' scores, length `n_detectors`).
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] / [`AnomalyError::DimensionMismatch`].
    pub fn score(&self, s: &[f32]) -> AnomalyResult<f32> {
        self.check_query(s)?;
        let region = self.local_region(s);
        let corrs = self.detector_correlations(&region);

        let mean_all = || s.iter().sum::<f32>() / self.n_detectors as f32;

        match self.config.strategy {
            LscpStrategy::SelectBest => {
                if corrs.iter().all(|c| !c.is_finite()) {
                    Ok(mean_all())
                } else {
                    Ok(s[argmax_finite(&corrs)])
                }
            }
            LscpStrategy::AverageCompetent => {
                let max_corr = corrs
                    .iter()
                    .copied()
                    .filter(|c| c.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if !max_corr.is_finite() {
                    return Ok(mean_all());
                }
                let threshold = max_corr - self.config.competent_margin;
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for (d, &c) in corrs.iter().enumerate() {
                    if c.is_finite() && c >= threshold {
                        sum += s[d];
                        count += 1;
                    }
                }
                if count == 0 {
                    Ok(mean_all())
                } else {
                    Ok(sum / count as f32)
                }
            }
        }
    }

    /// Batch scoring; `scores` is `[n × n_detectors]` row-major, returns `[n]`.
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] / [`AnomalyError::DimensionMismatch`].
    pub fn score_batch(&self, scores: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if scores.len() != n * self.n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_detectors,
                got: scores.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &scores[i * self.n_detectors..(i + 1) * self.n_detectors];
            out.push(self.score(row)?);
        }
        Ok(out)
    }

    /// Number of base detectors (0 if unfitted).
    #[inline]
    #[must_use]
    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    /// Validate a query score vector.
    fn check_query(&self, s: &[f32]) -> AnomalyResult<()> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if s.len() != self.n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.n_detectors,
                got: s.len(),
            });
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pearson_pub(a: &[f32], b: &[f32]) -> f32 {
        pearson(a, b)
    }

    // ── Test (a): output correlates with the globally best base detector ──────
    #[test]
    fn output_correlates_with_best_detector() {
        // Detector 0 carries the latent anomaly signal; 1 & 2 are low-variance
        // noise. The (average) pseudo target is dominated by detector 0, so
        // LSCP should keep selecting it.
        let n = 40_usize;
        let b = 3_usize;
        let mut train = Vec::with_capacity(n * b);
        let mut det0_train = Vec::with_capacity(n);
        for i in 0..n {
            let u = i as f32 * 0.25; // 0 … ~10
            let noise1 = 5.0 + 0.01 * (((i * 7) % 5) as f32 - 2.0);
            let noise2 = 5.0 + 0.01 * (((i * 3) % 5) as f32 - 2.0);
            train.push(u);
            train.push(noise1);
            train.push(noise2);
            det0_train.push(u);
        }
        let cfg = LscpConfig {
            local_region_size: 7,
            target: LscpTarget::Average,
            strategy: LscpStrategy::SelectBest,
            competent_margin: 0.1,
        };
        let mut lscp = LscpEnsemble::new(cfg);
        lscp.fit(&train, n, b).expect("fit");

        // Test points across the signal range.
        let mut outputs = Vec::new();
        let mut det0_test = Vec::new();
        let mut det1_test = Vec::new();
        for j in 0..15_usize {
            let u = j as f32 * 0.6;
            let s = [u, 5.0, 5.0];
            outputs.push(lscp.score(&s).expect("score"));
            det0_test.push(u);
            det1_test.push(5.0_f32);
        }
        let corr0 = pearson_pub(&outputs, &det0_test);
        assert!(
            corr0 > 0.95,
            "output should track detector 0 (corr={corr0})"
        );
        // Detector 1 is constant → undefined correlation, definitely not better.
        let corr1 = pearson_pub(&outputs, &det1_test);
        assert!(
            corr1.partial_cmp(&corr0) != Some(std::cmp::Ordering::Greater),
            "output should align with detector 0, not 1 (corr0={corr0}, corr1={corr1})"
        );
    }

    // ── Test (b): combined score is finite and bounded ────────────────────────
    #[test]
    fn combined_score_finite_and_bounded() {
        // Base scores in [0, 1] → any selection/average stays in [0, 1].
        let n = 30_usize;
        let b = 4_usize;
        let mut train = Vec::with_capacity(n * b);
        for i in 0..n * b {
            // deterministic pseudo-random values in [0, 1)
            let v = ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 16_777_216.0;
            train.push(v.fract());
        }
        let mut lscp = LscpEnsemble::new(LscpConfig::default());
        lscp.fit(&train, n, b).expect("fit");

        let queries = [
            [0.1_f32, 0.9, 0.3, 0.7],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 1.0],
            [0.25, 0.75, 0.6, 0.4],
        ];
        for q in &queries {
            let s = lscp.score(q).expect("score");
            assert!(s.is_finite(), "score must be finite, got {s}");
            assert!((0.0..=1.0).contains(&s), "score {s} must be in [0, 1]");
        }
    }

    // ── Test (c): locally best detector is the one selected ───────────────────
    #[test]
    fn locally_best_detector_is_selected() {
        // Region 1 (rows 0..6): detector 0 varies, 1 & 2 constant → detector 0
        // tracks the target. Region 2 (rows 6..12): detector 1 varies → it wins.
        let b = 3_usize;
        let mut train: Vec<f32> = Vec::new();
        for i in 0..6 {
            train.extend_from_slice(&[i as f32, 0.5, 0.5]);
        }
        for i in 0..6 {
            train.extend_from_slice(&[50.0, i as f32, 50.0]);
        }
        let n = 12_usize;
        let cfg = LscpConfig {
            local_region_size: 6,
            target: LscpTarget::Average,
            strategy: LscpStrategy::SelectBest,
            competent_margin: 0.1,
        };
        let mut lscp = LscpEnsemble::new(cfg);
        lscp.fit(&train, n, b).expect("fit");

        // Query inside region 1 → detector 0 competent.
        let in_region1 = [2.5_f32, 0.5, 0.5];
        assert_eq!(
            lscp.select_local_detector(&in_region1).expect("select"),
            0,
            "detector 0 should win in region 1"
        );
        // Query inside region 2 → detector 1 competent.
        let in_region2 = [50.0_f32, 2.5, 50.0];
        assert_eq!(
            lscp.select_local_detector(&in_region2).expect("select"),
            1,
            "detector 1 should win in region 2"
        );
    }

    // ── Test (d): a single base detector reduces to that detector ─────────────
    #[test]
    fn single_detector_reduces_to_itself() {
        let n = 12_usize;
        let b = 1_usize;
        let train: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let mut lscp = LscpEnsemble::new(LscpConfig::default());
        lscp.fit(&train, n, b).expect("fit");

        for &v in &[0.0_f32, 1.1, 2.7, 9.9] {
            let s = lscp.score(&[v]).expect("score");
            assert!(
                (s - v).abs() < 1e-6,
                "single-detector output {s} should equal {v}"
            );
        }
    }

    // ── Test (e): shape mismatch → error ──────────────────────────────────────
    #[test]
    fn shape_mismatch_errors() {
        let n = 10_usize;
        let b = 3_usize;
        let train = vec![0.5_f32; n * b];
        let mut lscp = LscpEnsemble::new(LscpConfig::default());

        // train_scores length inconsistent with n * b
        assert!(matches!(
            lscp.fit(&train[..n * b - 1], n, b),
            Err(AnomalyError::DimensionMismatch { .. })
        ));

        lscp.fit(&train, n, b).expect("fit");
        // query with wrong number of detectors
        assert!(matches!(
            lscp.score(&[0.1_f32, 0.2]),
            Err(AnomalyError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        // empty training set
        let mut lscp2 = LscpEnsemble::new(LscpConfig::default());
        assert!(matches!(
            lscp2.fit(&[], 0, b),
            Err(AnomalyError::EmptyInput)
        ));
        // score before fit
        let lscp3 = LscpEnsemble::new(LscpConfig::default());
        assert!(matches!(
            lscp3.score(&[0.1_f32]),
            Err(AnomalyError::NotFitted)
        ));
    }

    // ── Extra: AverageCompetent strategy stays bounded ────────────────────────
    #[test]
    fn average_competent_strategy_runs() {
        let n = 20_usize;
        let b = 3_usize;
        let mut train = Vec::with_capacity(n * b);
        for i in 0..n {
            let u = i as f32 * 0.1;
            train.extend_from_slice(&[u, u + 0.01, 0.5]);
        }
        let cfg = LscpConfig {
            local_region_size: 5,
            target: LscpTarget::Average,
            strategy: LscpStrategy::AverageCompetent,
            competent_margin: 0.2,
        };
        let mut lscp = LscpEnsemble::new(cfg);
        lscp.fit(&train, n, b).expect("fit");
        let out = lscp.score_batch(&train, n).expect("batch");
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|s| s.is_finite()));
    }
}
