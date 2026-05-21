//! LODA — Lightweight On-line Detector of Anomalies (Pevný 2016).
//!
//! **Fit phase:**
//!
//! For each of `T` projectors `t = 1..T`:
//! 1. Draw a sparse Rademacher projection vector `w_t ∈ Rᵈ` with
//!    `n_nonzero = max(1, round(√d))` non-zero entries, each ±1/√n_nonzero at
//!    randomly chosen positions (partial Fisher-Yates without replacement).
//! 2. Project all training samples: `z_{ti} = w_t · x_i`.
//! 3. Build an equi-width histogram over `[min(z_t), max(z_t)]` with `B` bins.
//! 4. Normalise to a density: `density[b] = count[b] / (n_samples · bin_width)`.
//!
//! **Score phase:**
//!
//! ```text
//! LODA(x) = (1/T) Σ_t  –log( density_t( w_t · x ) + ε ),   ε = 1e-10
//! ```
//!
//! Higher score ↔ lower projected density ↔ stronger anomaly.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the LODA detector.
#[derive(Debug, Clone)]
pub struct LodaConfig {
    /// Number of sparse projectors `T` (default 100).
    pub n_projectors: usize,
    /// Number of histogram bins per projector (default 10).
    pub n_bins: usize,
    /// RNG seed for reproducibility (default 42).
    pub seed: u64,
}

impl Default for LodaConfig {
    fn default() -> Self {
        Self {
            n_projectors: 100,
            n_bins: 10,
            seed: 42,
        }
    }
}

// ─── Internal histogram projector ─────────────────────────────────────────────

/// A single sparse projection vector together with its fitted density histogram.
struct ProjectorHistogram {
    /// Full-length weight vector (`n_features` entries; most are 0.0).
    weights: Vec<f32>,
    /// Left edge of the first bin.
    bin_min: f32,
    /// Width of each bin.
    bin_width: f32,
    /// Normalised density per bin (`n_bins` entries).
    densities: Vec<f32>,
    /// Number of bins (cached for fast lookup).
    n_bins: usize,
}

impl ProjectorHistogram {
    /// Project `x` and return the estimated density at `w · x`.
    #[inline]
    fn density(&self, x: &[f32]) -> f32 {
        let z: f32 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        // Return 0.0 for projections that fall outside the training histogram range:
        // out-of-range ↔ zero observed density ↔ maximum anomaly contribution.
        if self.bin_width >= 1e-12
            && (z < self.bin_min || z > self.bin_min + self.bin_width * self.n_bins as f32)
        {
            return 0.0;
        }
        let idx = bin_index(z, self.bin_min, self.bin_width, self.n_bins);
        self.densities[idx]
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────────

/// Generate a sparse Rademacher projection vector of length `n_features` with
/// exactly `n_nonzero` non-zero entries, each ±1/√n_nonzero.
///
/// Non-zero positions are chosen by partial Fisher-Yates (without replacement).
fn random_sparse_projection(n_features: usize, n_nonzero: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut weights = vec![0.0_f32; n_features];
    let mut indices: Vec<usize> = (0..n_features).collect();
    let scale = 1.0_f32 / (n_nonzero as f32).sqrt();
    for k in 0..n_nonzero {
        let j = k + rng.next_usize(n_features - k);
        indices.swap(k, j);
        let sign = if rng.next_u32() & 1 == 0 {
            1.0_f32
        } else {
            -1.0_f32
        };
        weights[indices[k]] = sign * scale;
    }
    weights
}

/// Map a projected value to its bin index, clamped to `[0, n_bins - 1]`.
#[inline]
fn bin_index(val: f32, bin_min: f32, bin_width: f32, n_bins: usize) -> usize {
    if bin_width < 1e-12 {
        return 0;
    }
    let idx = ((val - bin_min) / bin_width) as isize;
    idx.max(0).min(n_bins as isize - 1) as usize
}

// ─── Loda ─────────────────────────────────────────────────────────────────────

/// LODA anomaly detector.
///
/// # Usage
///
/// ```rust,ignore
/// let mut detector = Loda::new(LodaConfig::default());
/// detector.fit(&train_data, n_samples, n_features)?;
/// let score = detector.score(&query)?;
/// ```
pub struct Loda {
    config: LodaConfig,
    projectors: Vec<ProjectorHistogram>,
    n_features: usize,
    n_samples: usize,
}

impl Loda {
    /// Create an unfitted LODA detector with the supplied configuration.
    #[must_use]
    pub fn new(config: LodaConfig) -> Self {
        Self {
            config,
            projectors: Vec::new(),
            n_features: 0,
            n_samples: 0,
        }
    }

    /// Fit LODA to `data` (`n_samples × n_features`, row-major).
    ///
    /// Builds `n_projectors` sparse projection histograms from the training set.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        // ── Validation ──────────────────────────────────────────────────────────
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }
        if self.config.n_projectors == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_projectors must be > 0".into(),
            });
        }
        if self.config.n_bins == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_bins must be > 0".into(),
            });
        }

        // ── Initialise RNG ──────────────────────────────────────────────────────
        let mut rng = LcgRng::new(self.config.seed);
        let n_nonzero = ((n_features as f32).sqrt().round() as usize).max(1);
        let n_bins = self.config.n_bins;
        let mut projectors = Vec::with_capacity(self.config.n_projectors);

        // ── Build each projector ────────────────────────────────────────────────
        for _ in 0..self.config.n_projectors {
            let weights = random_sparse_projection(n_features, n_nonzero, &mut rng);

            // Project every training point
            let mut projections: Vec<f32> = (0..n_samples)
                .map(|i| {
                    let row = &data[i * n_features..(i + 1) * n_features];
                    weights.iter().zip(row.iter()).map(|(w, x)| w * x).sum()
                })
                .collect();

            // Determine histogram range
            let z_min = projections.iter().copied().fold(f32::INFINITY, f32::min);
            let z_max = projections
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            let bin_width = if (z_max - z_min).abs() < 1e-12 {
                1.0_f32 // degenerate: all projections identical
            } else {
                (z_max - z_min) / n_bins as f32
            };

            // Accumulate counts
            let mut counts = vec![0_u64; n_bins];
            for &z in &projections {
                let idx = bin_index(z, z_min, bin_width, n_bins);
                counts[idx] += 1;
            }

            // Normalise to density: count / (n_samples * bin_width)
            let denom = n_samples as f32 * bin_width;
            let densities: Vec<f32> = counts.iter().map(|&c| c as f32 / denom).collect();

            // Drop projections buffer eagerly
            projections.clear();
            projections.shrink_to_fit();

            projectors.push(ProjectorHistogram {
                weights,
                bin_min: z_min,
                bin_width,
                densities,
                n_bins,
            });
        }

        self.projectors = projectors;
        self.n_features = n_features;
        self.n_samples = n_samples;
        Ok(())
    }

    /// Compute the LODA anomaly score for a single query `x`.
    ///
    /// Returns `(1/T) Σ_t –log(density_t(w_t · x) + ε)`.
    /// Higher values indicate stronger anomaly.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_samples == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        const EPSILON: f32 = 1e-10;
        let t = self.projectors.len() as f32;
        let sum: f32 = self
            .projectors
            .iter()
            .map(|ph| -(ph.density(x) + EPSILON).ln())
            .sum();
        Ok(sum / t)
    }

    /// Batch LODA scoring; `x` is `[n × n_features]` row-major; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if self.n_samples == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }

    /// Number of features the model was fitted on (0 if not fitted).
    #[inline]
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of training samples (0 if not fitted).
    #[inline]
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 2-D grid: 10×10 inliers + one distant outlier
    fn make_2d_data() -> (Vec<f32>, usize, usize) {
        let mut data: Vec<f32> = Vec::new();
        for i in 0..10_i32 {
            for j in 0..10_i32 {
                data.push(i as f32 * 0.1);
                data.push(j as f32 * 0.1);
            }
        }
        let n = 100;
        (data, n, 2)
    }

    // 1-D data: inliers in [0, 1], outlier at 10.0
    fn make_1d_data() -> (Vec<f32>, usize, usize) {
        let mut data: Vec<f32> = Vec::new();
        for i in 0..50 {
            data.push(i as f32 * 0.02); // 0.0 … 0.98
        }
        (data, 50, 1)
    }

    // ── Test 1: fit + score 2D data returns a finite value ────────────────────
    #[test]
    fn test_fit_score_basic_2d() {
        let (data, n, d) = make_2d_data();
        let mut det = Loda::new(LodaConfig::default());
        det.fit(&data, n, d).expect("fit should succeed");
        let s = det.score(&[0.5_f32, 0.5]).expect("score should succeed");
        assert!(s.is_finite(), "score should be finite, got {s}");
    }

    // ── Test 2: score before fit returns NotFitted ─────────────────────────────
    #[test]
    fn test_unfitted_returns_not_fitted() {
        let det = Loda::new(LodaConfig::default());
        match det.score(&[0.0_f32]) {
            Err(AnomalyError::NotFitted) => {}
            other => panic!("expected NotFitted, got {other:?}"),
        }
    }

    // ── Test 3: outlier score > inlier score on 1-D data ─────────────────────
    #[test]
    fn test_outlier_score_exceeds_inlier() {
        let (data, n, d) = make_1d_data();
        let cfg = LodaConfig {
            n_projectors: 200,
            n_bins: 20,
            seed: 17,
        };
        let mut det = Loda::new(cfg);
        det.fit(&data, n, d).expect("fit");
        let inlier_score = det.score(&[0.5_f32]).expect("inlier score");
        let outlier_score = det.score(&[10.0_f32]).expect("outlier score");
        assert!(
            outlier_score > inlier_score,
            "outlier ({outlier_score}) should score higher than inlier ({inlier_score})"
        );
    }

    // ── Test 4: score_batch returns correct length ─────────────────────────────
    #[test]
    fn test_score_batch_length() {
        let (data, n, d) = make_2d_data();
        let mut det = Loda::new(LodaConfig::default());
        det.fit(&data, n, d).expect("fit");
        let queries: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let scores = det.score_batch(&queries, 3).expect("batch score");
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|s| s.is_finite()), "all scores finite");
    }

    // ── Test 5: empty input returns EmptyInput ────────────────────────────────
    #[test]
    fn test_empty_input_error() {
        let mut det = Loda::new(LodaConfig::default());
        match det.fit(&[], 0, 2) {
            Err(AnomalyError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    // ── Test 6: feature count mismatch at score time ───────────────────────────
    #[test]
    fn test_feature_count_mismatch() {
        let (data, n, d) = make_2d_data();
        let mut det = Loda::new(LodaConfig::default());
        det.fit(&data, n, d).expect("fit");
        // Query with wrong dimensionality (3 instead of 2)
        match det.score(&[0.1_f32, 0.2, 0.3]) {
            Err(AnomalyError::FeatureCountMismatch {
                expected: 2,
                got: 3,
            }) => {}
            other => panic!("expected FeatureCountMismatch, got {other:?}"),
        }
    }

    // ── Test 7: minimal config (n_projectors=1, n_bins=5) works ───────────────
    #[test]
    fn test_minimal_config() {
        let (data, n, d) = make_2d_data();
        let cfg = LodaConfig {
            n_projectors: 1,
            n_bins: 5,
            seed: 99,
        };
        let mut det = Loda::new(cfg);
        det.fit(&data, n, d).expect("fit with minimal config");
        let s = det.score(&[0.3_f32, 0.7]).expect("score");
        assert!(s.is_finite(), "score={s}");
    }

    // ── Test 8: determinism — same seed gives identical scores ────────────────
    #[test]
    fn test_deterministic_same_seed() {
        let (data, n, d) = make_2d_data();
        let cfg = LodaConfig {
            n_projectors: 50,
            n_bins: 8,
            seed: 7,
        };
        let mut det_a = Loda::new(cfg.clone());
        let mut det_b = Loda::new(cfg);
        det_a.fit(&data, n, d).expect("fit a");
        det_b.fit(&data, n, d).expect("fit b");
        let s_a = det_a.score(&[0.4_f32, 0.6]).expect("score a");
        let s_b = det_b.score(&[0.4_f32, 0.6]).expect("score b");
        assert!((s_a - s_b).abs() < 1e-6, "scores differ: {s_a} vs {s_b}");
    }

    // ── Test 9: different seeds give different projectors (verifiable via 5-D data) ──
    #[test]
    fn test_different_seeds_give_different_scores() {
        // Use 5-D data so seeds meaningfully diverge in sparse index selection.
        let n = 80_usize;
        let d = 5_usize;
        let mut rng_gen = LcgRng::new(777);
        let data: Vec<f32> = (0..n * d).map(|_| rng_gen.next_f32()).collect();

        let cfg_a = LodaConfig {
            seed: 1,
            n_projectors: 60,
            n_bins: 10,
        };
        let cfg_b = LodaConfig {
            seed: 9999,
            n_projectors: 60,
            n_bins: 10,
        };
        let mut det_a = Loda::new(cfg_a);
        let mut det_b = Loda::new(cfg_b);
        det_a.fit(&data, n, d).expect("fit a");
        det_b.fit(&data, n, d).expect("fit b");

        // Accumulate scores over multiple query points; at least one pair must differ.
        let queries: Vec<[f32; 5]> = vec![
            [0.1, 0.9, 0.3, 0.7, 0.5],
            [0.8, 0.2, 0.6, 0.1, 0.4],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ];
        let all_same = queries.iter().all(|q| {
            let sa = det_a.score(q.as_ref()).unwrap();
            let sb = det_b.score(q.as_ref()).unwrap();
            (sa - sb).abs() < 1e-9
        });
        assert!(
            !all_same,
            "expected at least one score to differ between seed=1 and seed=9999 on 5-D data"
        );
    }

    // ── Extra: zero n_features returns InvalidFeatureCount ────────────────────
    #[test]
    fn test_zero_n_features_error() {
        let mut det = Loda::new(LodaConfig::default());
        match det.fit(&[], 0, 0) {
            // EmptyInput fires before InvalidFeatureCount — either is correct
            Err(AnomalyError::EmptyInput) | Err(AnomalyError::InvalidFeatureCount { n: 0 }) => {}
            other => panic!("expected EmptyInput or InvalidFeatureCount, got {other:?}"),
        }
        // Specifically trigger InvalidFeatureCount (n_samples > 0, n_features == 0)
        match det.fit(&[1.0_f32], 1, 0) {
            Err(AnomalyError::InvalidFeatureCount { n: 0 }) => {}
            other => panic!("expected InvalidFeatureCount{{n:0}}, got {other:?}"),
        }
    }

    // ── Extra: dimension mismatch in data slice ────────────────────────────────
    #[test]
    fn test_data_dimension_mismatch() {
        let mut det = Loda::new(LodaConfig::default());
        // data has 5 elements but n_samples*n_features == 6
        match det.fit(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], 2, 3) {
            Err(AnomalyError::DimensionMismatch {
                expected: 6,
                got: 5,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }
}
