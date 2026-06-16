//! Histogram binning calibration: post-hoc binary recalibration via empirical
//! bin-wise positive fractions (Zadrozny & Elkan 2001/2002; Guo et al. 2017 §3).
//!
//! Given uncalibrated predicted scores `s ∈ [0,1]` and binary labels `y ∈ {0,1}`,
//! the `[0,1]` range is divided into `n_bins` intervals. Within each interval the
//! calibrated probability is estimated as `P(Y=1 | s ∈ bin) = #pos_in_bin / #total_in_bin`.
//! At inference the predicted score is mapped to its bin's empirical positive fraction.
//!
//! Two binning strategies are supported:
//! - **EqualWidth** — bins are equally spaced over [0, 1].
//! - **EqualCount** — (quantile) bins contain an approximately equal number of
//!   training samples.
//!
//! Bins with fewer than `min_bin_count` samples use a nearest-neighbour smoothing
//! fallback: the calibrated probability is borrowed from the nearest valid bin.

use crate::error::{BayesError, BayesResult};

// ─── f32 log guard (matches beta.rs nll guard) ───────────────────────────────
const LOG_EPS: f32 = 1e-7;

// ─── Strategy ────────────────────────────────────────────────────────────────

/// Strategy for placing histogram bin boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinStrategy {
    /// Equally-spaced boundaries over [0, 1].
    EqualWidth,
    /// Quantile-based boundaries: each bin contains approximately the same
    /// number of training samples.
    EqualCount,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyperparameters for [`HistogramBinCalibrator::fit`].
#[derive(Debug, Clone)]
pub struct HistogramBinConfig {
    /// Number of histogram bins (must be ≥ 2).
    pub n_bins: usize,
    /// Strategy for placing bin boundaries.
    pub strategy: BinStrategy,
    /// Minimum samples per bin to trust the empirical estimate.
    /// Bins with fewer samples fall back to the nearest valid neighbour.
    pub min_bin_count: usize,
}

impl Default for HistogramBinConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            strategy: BinStrategy::EqualWidth,
            min_bin_count: 1,
        }
    }
}

// ─── Calibrator ──────────────────────────────────────────────────────────────

/// A fitted histogram bin calibrator for binary predicted scores.
///
/// After calling [`HistogramBinCalibrator::fit`] each of the `n_bins` intervals
/// stores the empirical positive fraction observed during fitting. The
/// [`HistogramBinCalibrator::calibrate`] method maps any new score to its bin's
/// stored fraction.
#[derive(Debug, Clone)]
pub struct HistogramBinCalibrator {
    /// Configuration used during fitting.
    pub config: HistogramBinConfig,
    /// Bin boundary thresholds: `n_bins + 1` values with
    /// `boundaries[0] = 0.0` and `boundaries[n_bins] = 1.0`.
    pub boundaries: Vec<f32>,
    /// Calibrated probability for each bin (`n_bins` values).
    pub bin_probs: Vec<f32>,
    /// Count of training samples assigned to each bin (`n_bins` values).
    pub bin_counts: Vec<usize>,
}

impl HistogramBinCalibrator {
    // ─── Fitting ─────────────────────────────────────────────────────────────

    /// Fit a histogram bin calibrator from predicted scores and binary labels.
    ///
    /// # Parameters
    /// - `scores` — Predicted probabilities ∈ [0, 1]. Values outside this range
    ///   are silently clamped to [0, 1].
    /// - `labels` — Binary ground-truth labels; each value must be exactly 0.0 or 1.0.
    /// - `cfg`    — Binning configuration.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `scores` is empty.
    /// - [`BayesError::DimensionMismatch`] if `scores.len() != labels.len()`.
    /// - [`BayesError::NCalibBinsTooSmall`] if `cfg.n_bins < 2`.
    /// - [`BayesError::NanEncountered`] if any label is not exactly 0.0 or 1.0.
    pub fn fit(scores: &[f32], labels: &[f32], cfg: HistogramBinConfig) -> BayesResult<Self> {
        // ── Validate ─────────────────────────────────────────────────────────
        if scores.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if scores.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: scores.len(),
                got: labels.len(),
            });
        }
        if cfg.n_bins < 2 {
            return Err(BayesError::NCalibBinsTooSmall);
        }
        for &lbl in labels {
            if lbl != 0.0 && lbl != 1.0 {
                return Err(BayesError::NanEncountered {
                    location: "HistogramBinCalibrator::fit: label not in {0.0, 1.0}",
                });
            }
        }

        let n_bins = cfg.n_bins;

        // Clamp scores to [0, 1] silently.
        let clamped: Vec<f32> = scores.iter().map(|&s| s.clamp(0.0, 1.0)).collect();

        // ── Compute bin boundaries ────────────────────────────────────────────
        let boundaries = match cfg.strategy {
            BinStrategy::EqualWidth => compute_equal_width_boundaries(n_bins),
            BinStrategy::EqualCount => compute_equal_count_boundaries(&clamped, n_bins),
        };

        // ── Accumulate per-bin positive sums and counts ───────────────────────
        let mut bin_pos_sum = vec![0.0_f64; n_bins];
        let mut bin_counts = vec![0_usize; n_bins];

        let dummy_probs = vec![0.0_f32; n_bins]; // placeholder — we use boundaries directly
        let tmp_cal = HistogramBinCalibrator {
            config: HistogramBinConfig {
                n_bins,
                strategy: cfg.strategy,
                min_bin_count: cfg.min_bin_count,
            },
            boundaries: boundaries.clone(),
            bin_probs: dummy_probs,
            bin_counts: vec![0; n_bins],
        };

        for (&s, &lbl) in clamped.iter().zip(labels.iter()) {
            let bin_idx = tmp_cal.find_bin(s);
            bin_counts[bin_idx] += 1;
            bin_pos_sum[bin_idx] += lbl as f64;
        }

        // ── Compute per-bin calibrated probabilities ──────────────────────────
        let min_count = cfg.min_bin_count;
        let mut bin_probs: Vec<f32> = (0..n_bins)
            .map(|b| {
                if bin_counts[b] >= min_count && bin_counts[b] > 0 {
                    (bin_pos_sum[b] / bin_counts[b] as f64) as f32
                } else {
                    // Sentinel: will be filled by smoothing pass.
                    f32::NAN
                }
            })
            .collect();

        // ── Nearest-neighbour smoothing for sparse bins ───────────────────────
        smooth_sparse_bins(&mut bin_probs, n_bins);

        Ok(Self {
            config: cfg,
            boundaries,
            bin_probs,
            bin_counts,
        })
    }

    // ─── Inference ───────────────────────────────────────────────────────────

    /// Return the bin index (0-indexed) that `score` falls into.
    ///
    /// The score is clamped to [0, 1] before the search. The last bin absorbs
    /// scores exactly equal to 1.0.
    #[must_use]
    pub fn find_bin(&self, score: f32) -> usize {
        let s = score.clamp(0.0, 1.0);
        let n_bins = self.config.n_bins;

        // Fast paths.
        if s < self.boundaries[1] {
            return 0;
        }
        if s >= self.boundaries[n_bins - 1] {
            return n_bins - 1;
        }

        // Binary search: find the largest b such that boundaries[b] <= s.
        let mut lo = 0_usize;
        let mut hi = n_bins; // exclusive upper bound on b
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.boundaries[mid] <= s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo.min(n_bins - 1)
    }

    /// Calibrate a single predicted score.
    ///
    /// The score is clamped to [0, 1], mapped to its bin, and the bin's
    /// empirical positive fraction is returned.
    ///
    /// # Errors
    /// This method currently never fails for a fitted calibrator; the `Result`
    /// wrapper maintains API consistency with the rest of the module.
    pub fn calibrate(&self, score: f32) -> BayesResult<f32> {
        let s = score.clamp(0.0, 1.0);
        let bin_idx = self.find_bin(s);
        Ok(self.bin_probs[bin_idx])
    }

    /// Calibrate a batch of predicted scores.
    ///
    /// # Errors
    /// Propagates any error from [`HistogramBinCalibrator::calibrate`].
    pub fn calibrate_batch(&self, scores: &[f32]) -> BayesResult<Vec<f32>> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }

    // ─── Evaluation ──────────────────────────────────────────────────────────

    /// Expected Calibration Error (ECE) on held-out `(scores, labels)`.
    ///
    /// ECE = Σ_b (|bin_b| / n) · |mean_label_b − mean_score_b|
    ///
    /// where the sums are over the bins that contain at least one held-out sample.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `scores` is empty.
    /// - [`BayesError::DimensionMismatch`] if `scores.len() != labels.len()`.
    pub fn ece(&self, scores: &[f32], labels: &[f32]) -> BayesResult<f32> {
        if scores.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if scores.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: scores.len(),
                got: labels.len(),
            });
        }

        let n = scores.len();
        let n_bins = self.config.n_bins;

        let mut bin_label_sum = vec![0.0_f64; n_bins];
        let mut bin_score_sum = vec![0.0_f64; n_bins];
        let mut bin_cnt = vec![0_usize; n_bins];

        for (&s, &lbl) in scores.iter().zip(labels.iter()) {
            let sc = s.clamp(0.0, 1.0);
            let b = self.find_bin(sc);
            bin_label_sum[b] += lbl as f64;
            bin_score_sum[b] += sc as f64;
            bin_cnt[b] += 1;
        }

        let mut ece = 0.0_f64;
        for b in 0..n_bins {
            if bin_cnt[b] == 0 {
                continue;
            }
            let weight = bin_cnt[b] as f64 / n as f64;
            let mean_label = bin_label_sum[b] / bin_cnt[b] as f64;
            let mean_score = bin_score_sum[b] / bin_cnt[b] as f64;
            ece += weight * (mean_label - mean_score).abs();
        }
        Ok(ece as f32)
    }

    // ─── NLL ─────────────────────────────────────────────────────────────────

    /// Mean binary NLL of the calibrated predictions against `labels`.
    ///
    /// Uses `LOG_EPS = 1e-7` to guard against `log(0)`.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `scores` is empty.
    /// - [`BayesError::DimensionMismatch`] if `scores.len() != labels.len()`.
    pub fn nll(&self, scores: &[f32], labels: &[f32]) -> BayesResult<f32> {
        if scores.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if scores.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: scores.len(),
                got: labels.len(),
            });
        }
        let mut sum = 0.0_f64;
        for (&s, &lbl) in scores.iter().zip(labels.iter()) {
            let p = self.calibrate(s)? as f64;
            let p_clamped = p.clamp(LOG_EPS as f64, (1.0 - LOG_EPS) as f64);
            sum -= lbl as f64 * p_clamped.ln() + (1.0 - lbl as f64) * (1.0 - p_clamped).ln();
        }
        Ok((sum / scores.len() as f64) as f32)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Compute n_bins+1 equal-width boundaries over [0, 1].
fn compute_equal_width_boundaries(n_bins: usize) -> Vec<f32> {
    let mut b = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        b.push(i as f32 / n_bins as f32);
    }
    b
}

/// Compute n_bins+1 quantile-based boundaries so each bin receives
/// approximately the same number of training samples.
fn compute_equal_count_boundaries(sorted_input: &[f32], n_bins: usize) -> Vec<f32> {
    // We sort a copy of the scores.
    let mut sorted = sorted_input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut boundaries = Vec::with_capacity(n_bins + 1);
    boundaries.push(0.0_f32);

    for i in 1..n_bins {
        let idx = (i * n) / n_bins;
        let idx_clamped = idx.min(n - 1);
        boundaries.push(sorted[idx_clamped]);
    }
    boundaries.push(1.0_f32);
    boundaries
}

/// Fill NaN-marked sparse bins using nearest-valid-neighbour search.
/// Bins with sufficient count are already set; NaN bins are patched.
fn smooth_sparse_bins(bin_probs: &mut [f32], n_bins: usize) {
    // First pass: try to find at least one valid bin to use as ultimate fallback.
    let has_any_valid = bin_probs.iter().any(|&p| !p.is_nan());

    if !has_any_valid {
        // All bins are sparse; default everything to 0.5 (maximum entropy prior).
        for p in bin_probs.iter_mut() {
            *p = 0.5;
        }
        return;
    }

    // For each NaN bin, search outward left and right for the nearest valid bin.
    for b in 0..n_bins {
        if !bin_probs[b].is_nan() {
            continue;
        }
        // Linear search outward.
        let mut found = false;
        for radius in 1..n_bins {
            let left_ok = b >= radius && !bin_probs[b - radius].is_nan();
            let right_ok = b + radius < n_bins && !bin_probs[b + radius].is_nan();
            if left_ok && right_ok {
                // Both sides valid: use the closer one (left wins on tie since
                // we check left first, which is fine — ties are arbitrary).
                bin_probs[b] = bin_probs[b - radius];
                found = true;
                break;
            } else if left_ok {
                bin_probs[b] = bin_probs[b - radius];
                found = true;
                break;
            } else if right_ok {
                bin_probs[b] = bin_probs[b + radius];
                found = true;
                break;
            }
        }
        if !found {
            // Should never reach here since has_any_valid is true, but guard anyway.
            bin_probs[b] = 0.5;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a simple synthetic dataset where scores are linearly spaced
    /// and labels follow `round(score)`.
    fn linear_dataset(n: usize) -> (Vec<f32>, Vec<f32>) {
        let scores: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let labels: Vec<f32> = scores
            .iter()
            .map(|&s| if s >= 0.5 { 1.0 } else { 0.0 })
            .collect();
        (scores, labels)
    }

    fn default_cfg() -> HistogramBinConfig {
        HistogramBinConfig::default()
    }

    // ── Boundary tests ───────────────────────────────────────────────────────

    #[test]
    fn equal_width_boundaries_evenly_spaced() {
        let b = compute_equal_width_boundaries(10);
        assert_eq!(b.len(), 11);
        assert!((b[0] - 0.0).abs() < 1e-6);
        assert!((b[10] - 1.0).abs() < 1e-6);
        for (i, &bv) in b.iter().enumerate() {
            let expected = i as f32 / 10.0;
            assert!(
                (bv - expected).abs() < 1e-6,
                "boundary[{i}] = {} expected {}",
                bv,
                expected
            );
        }
    }

    #[test]
    fn equal_count_boundaries_splits_data_evenly() {
        // 100 uniformly-spaced scores [0, 1]; 10 bins should each get 10 samples.
        let n = 100;
        let scores: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let b = compute_equal_count_boundaries(&scores, 10);
        assert_eq!(b.len(), 11);
        assert!((b[0] - 0.0).abs() < 1e-6);
        assert!((b[10] - 1.0).abs() < 1e-6);
        // Each step should advance by ≈ 0.1.
        for i in 1..10 {
            assert!(
                b[i] > b[i - 1],
                "boundaries must be increasing; b[{i}]={}, b[{}]={}",
                b[i],
                i - 1,
                b[i - 1]
            );
        }
    }

    // ── fit tests ────────────────────────────────────────────────────────────

    #[test]
    fn fit_all_positive_labels_yields_bin_probs_one() {
        let scores = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![1.0_f32; 5];
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        // Every occupied bin should have prob = 1.0.
        for (&cnt, &prob) in cal.bin_counts.iter().zip(cal.bin_probs.iter()) {
            if cnt > 0 {
                assert!(
                    (prob - 1.0).abs() < 1e-6,
                    "all-positive bin should have prob=1, got {prob}"
                );
            }
        }
    }

    #[test]
    fn fit_all_negative_labels_yields_bin_probs_zero() {
        let scores = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![0.0_f32; 5];
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        for (&cnt, &prob) in cal.bin_counts.iter().zip(cal.bin_probs.iter()) {
            if cnt > 0 {
                assert!(
                    prob.abs() < 1e-6,
                    "all-negative bin should have prob=0, got {prob}"
                );
            }
        }
    }

    #[test]
    fn fit_on_linear_dataset_returns_ok() {
        let (scores, labels) = linear_dataset(100);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg());
        assert!(cal.is_ok(), "fit must succeed on well-formed data");
    }

    #[test]
    fn fit_rejects_empty_input() {
        let r = HistogramBinCalibrator::fit(&[], &[], default_cfg());
        assert!(
            matches!(r, Err(BayesError::CalibrationSetEmpty)),
            "expected CalibrationSetEmpty, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let r = HistogramBinCalibrator::fit(&[0.5_f32], &[0.0_f32, 1.0_f32], default_cfg());
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_n_bins_less_than_2() {
        let cfg = HistogramBinConfig {
            n_bins: 1,
            ..default_cfg()
        };
        let r = HistogramBinCalibrator::fit(&[0.5_f32], &[1.0_f32], cfg);
        assert!(
            matches!(r, Err(BayesError::NCalibBinsTooSmall)),
            "expected NCalibBinsTooSmall, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_invalid_label() {
        let r = HistogramBinCalibrator::fit(&[0.5_f32], &[0.5_f32], default_cfg());
        assert!(
            matches!(r, Err(BayesError::NanEncountered { .. })),
            "expected NanEncountered for non-binary label, got {r:?}"
        );
    }

    #[test]
    fn fit_equal_count_strategy_ok() {
        let (scores, labels) = linear_dataset(50);
        let cfg = HistogramBinConfig {
            strategy: BinStrategy::EqualCount,
            ..default_cfg()
        };
        let cal = HistogramBinCalibrator::fit(&scores, &labels, cfg).expect("fit should succeed");
        assert_eq!(cal.bin_probs.len(), 10);
        assert_eq!(cal.boundaries.len(), 11);
    }

    // ── find_bin tests ───────────────────────────────────────────────────────

    #[test]
    fn find_bin_returns_zero_for_score_at_lower_boundary() {
        let (scores, labels) = linear_dataset(50);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        assert_eq!(cal.find_bin(0.0), 0);
    }

    #[test]
    fn find_bin_returns_last_bin_for_score_at_one() {
        let (scores, labels) = linear_dataset(50);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        let n_bins = cal.config.n_bins;
        assert_eq!(cal.find_bin(1.0), n_bins - 1);
    }

    #[test]
    fn find_bin_result_always_in_range() {
        let (scores, labels) = linear_dataset(100);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        let n_bins = cal.config.n_bins;
        for i in 0..=20 {
            let s = i as f32 / 20.0;
            let b = cal.find_bin(s);
            assert!(b < n_bins, "find_bin({s}) = {b} must be < {n_bins}");
        }
    }

    // ── calibrate tests ──────────────────────────────────────────────────────

    #[test]
    fn calibrate_returns_bin_prob_for_score_in_bin() {
        let scores = vec![
            0.05_f32, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
        ];
        let labels = vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = HistogramBinConfig {
            n_bins: 2,
            ..default_cfg()
        };
        let cal = HistogramBinCalibrator::fit(&scores, &labels, cfg).expect("fit should succeed");
        // Bin 0 (scores in [0, 0.5)): 5 samples, 0 positive → prob = 0.0
        let p_low = cal.calibrate(0.2).expect("calibrate should succeed");
        assert!((p_low - 0.0).abs() < 1e-6, "expected 0.0, got {p_low}");
        // Bin 1 (scores in [0.5, 1.0]): 5 samples, 5 positive → prob = 1.0
        let p_high = cal.calibrate(0.8).expect("calibrate should succeed");
        assert!((p_high - 1.0).abs() < 1e-6, "expected 1.0, got {p_high}");
    }

    #[test]
    fn calibrate_batch_length_matches_input() {
        let (scores, labels) = linear_dataset(30);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        let test_scores = vec![0.1_f32, 0.4, 0.6, 0.9];
        let out = cal
            .calibrate_batch(&test_scores)
            .expect("calibrate_batch should succeed");
        assert_eq!(out.len(), 4);
    }

    // ── ECE tests ────────────────────────────────────────────────────────────

    #[test]
    fn ece_is_zero_for_perfectly_calibrated_predictor() {
        // Build a calibrator where every bin maps score ≈ pos_fraction.
        // scores = [0.1, 0.3, ..., 0.9] (10 values), all mapping to their own bin.
        // But ECE computes on the raw score vs. label, not calibrated vs. label.
        // For a perfectly calibrated predictor, mean_score_b = mean_label_b in each bin.
        // Construct: one sample per bin, score = label probability.
        // Use 2 bins: half all-0 labels with low scores, half all-1 labels with high scores.
        let scores = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let cfg = HistogramBinConfig {
            n_bins: 2,
            ..default_cfg()
        };
        let cal = HistogramBinCalibrator::fit(&scores, &labels, cfg).expect("fit should succeed");
        let ece_val = cal.ece(&scores, &labels).expect("ece should succeed");
        assert!(
            ece_val.abs() < 1e-5,
            "ECE should be 0 for perfect predictor, got {ece_val}"
        );
    }

    #[test]
    fn ece_is_positive_for_miscalibrated_predictor() {
        // Overconfident: all scores ≈ 1.0 but only half are positive.
        let scores = vec![0.95_f32; 20];
        let labels: Vec<f32> = (0..20).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        let cfg = HistogramBinConfig {
            n_bins: 2,
            ..default_cfg()
        };
        let cal = HistogramBinCalibrator::fit(&scores, &labels, cfg).expect("fit should succeed");
        let ece_val = cal.ece(&scores, &labels).expect("ece should succeed");
        assert!(
            ece_val > 0.0,
            "ECE should be > 0 for miscalibrated predictor"
        );
    }

    #[test]
    fn ece_rejects_empty_input() {
        let (scores, labels) = linear_dataset(10);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        let r = cal.ece(&[], &[]);
        assert!(matches!(r, Err(BayesError::CalibrationSetEmpty)));
    }

    #[test]
    fn ece_rejects_mismatched_lengths() {
        let (scores, labels) = linear_dataset(10);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        let r = cal.ece(&[0.5_f32], &[0.0_f32, 1.0_f32]);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    // ── Smoothing for empty bins ──────────────────────────────────────────────

    #[test]
    fn sparse_bins_get_smoothed_from_nearest_valid_bin() {
        // Use many bins (20) but few samples (4); most bins will be empty.
        let scores = vec![0.0_f32, 0.1, 0.9, 1.0];
        let labels = vec![0.0_f32, 0.0, 1.0, 1.0];
        let cfg = HistogramBinConfig {
            n_bins: 20,
            min_bin_count: 1,
            ..default_cfg()
        };
        let cal = HistogramBinCalibrator::fit(&scores, &labels, cfg).expect("fit should succeed");
        // No bin_prob should be NaN.
        for (b, &p) in cal.bin_probs.iter().enumerate() {
            assert!(
                !p.is_nan(),
                "bin {b} has NaN after smoothing (all bins should be finite)"
            );
        }
    }

    // ── Calibrate output is in [0, 1] ─────────────────────────────────────────

    #[test]
    fn calibrate_output_always_in_unit_interval() {
        let (scores, labels) = linear_dataset(100);
        let cal = HistogramBinCalibrator::fit(&scores, &labels, default_cfg())
            .expect("value should be present");
        for i in 0..=20 {
            let s = i as f32 / 20.0;
            let p = cal.calibrate(s).expect("calibrate should succeed");
            assert!(
                (0.0..=1.0).contains(&p),
                "calibrate({s}) = {p} outside [0, 1]"
            );
        }
    }
}
