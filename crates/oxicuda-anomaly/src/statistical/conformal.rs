//! Conformal Anomaly Detection (Vovk et al. 2005).
//!
//! Wraps any base anomaly scorer to produce **distribution-free p-values**
//! with valid frequentist coverage guarantees.
//!
//! # Algorithm Overview
//!
//! **Calibration phase:**
//! Score all calibration examples with a base scorer; collect calibration
//! scores `{α_1, …, α_m}` (higher ⇒ more anomalous).
//!
//! **Prediction phase:**
//! For a new test point `x` with score `α_x`:
//! ```text
//! p(x) = #{i : α_i ≥ α_x} / (m + 1)
//! ```
//! A point is anomalous at significance level `ε` if `p(x) < ε`.
//!
//! **Mondrian conformal:**
//! Class-conditional p-values — compute calibration split by class label,
//! then evaluate per-class p-values independently.
//!
//! **Online (sliding-window) conformal:**
//! Maintain a fixed-capacity `VecDeque` of recent scores.  On each new
//! point, compute p-value against the window, then slide the window forward.

use std::collections::VecDeque;

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for conformal anomaly detection.
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Significance level ε ∈ (0, 1).  A test point is declared anomalous
    /// when its p-value falls strictly below this threshold.  Default `0.05`.
    pub significance: f64,
    /// If `true`, add a `Uniform(0, 1)` random jitter to the p-value
    /// formula to break ties (required for exact type-I error control).
    /// Default `false`.
    pub smoothing: bool,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            significance: 0.05,
            smoothing: false,
        }
    }
}

// ─── Fitted detector ──────────────────────────────────────────────────────────

/// Fitted conformal anomaly detector backed by a fixed calibration set.
#[derive(Debug, Clone)]
pub struct ConformalDetector {
    /// Anomaly scores of the calibration set (higher = more anomalous).
    pub calibration_scores: Vec<f64>,
    /// Runtime configuration.
    pub config: ConformalConfig,
}

// ─── Online detector ──────────────────────────────────────────────────────────

/// Sliding-window conformal detector for online / streaming settings.
///
/// Each call to [`online_conformal_update`] returns the p-value and label for
/// the incoming point, then inserts its score into the window (evicting the
/// oldest entry when the window is full).
#[derive(Debug, Clone)]
pub struct OnlineConformalDetector {
    /// Recent calibration scores (ring buffer).
    pub window: VecDeque<f64>,
    /// Maximum number of scores kept in the window.
    pub window_size: usize,
    /// Runtime configuration.
    pub config: ConformalConfig,
}

// ─── Output type ─────────────────────────────────────────────────────────────

/// Batch prediction result from a conformal detector.
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// Conformal p-value for each test point.
    pub p_values: Vec<f64>,
    /// Binary label: `+1` = normal, `-1` = anomaly.
    pub labels: Vec<i32>,
    /// Total number of points labelled as anomalies.
    pub n_anomalies: usize,
}

// ─── Core p-value computation ─────────────────────────────────────────────────

/// Compute the conformal p-value of a single test score against a calibration
/// set.
///
/// ```text
/// p(x) = #{i : α_i ≥ α_x} / (m + 1)
/// ```
///
/// The denominator `m + 1` accounts for the test point itself, ensuring that
/// p-values are uniformly distributed under the null hypothesis (i.i.d. data).
///
/// Returns `0.0` when the calibration set is empty.
#[must_use]
pub fn conformal_p_value(calibration_scores: &[f64], test_score: f64) -> f64 {
    let m = calibration_scores.len();
    if m == 0 {
        return 0.0;
    }
    let count_ge = calibration_scores
        .iter()
        .filter(|&&s| s >= test_score)
        .count();
    count_ge as f64 / (m + 1) as f64
}

/// Smoothed conformal p-value (with tie-breaking uniform jitter).
///
/// ```text
/// p_smooth(x) = (#{i : α_i > α_x} + U · #{i : α_i = α_x} + U) / (m + 1)
/// ```
/// where `U ~ Uniform(0, 1)` is drawn from `rng`.
///
/// Smoothing guarantees that, under the null, p-values are *exactly* Uniform(0,1)
/// rather than stochastically larger.
#[must_use]
pub fn conformal_p_value_smoothed(
    calibration_scores: &[f64],
    test_score: f64,
    rng: &mut LcgRng,
) -> f64 {
    let m = calibration_scores.len();
    if m == 0 {
        return rng.next_f32() as f64 / (1.0 + 1.0); // 0 calibration points: U/(1+1)
    }
    let count_gt = calibration_scores
        .iter()
        .filter(|&&s| s > test_score)
        .count();
    let count_eq = calibration_scores
        .iter()
        .filter(|&&s| s == test_score)
        .count();
    let u = rng.next_f32() as f64;
    (count_gt as f64 + u * count_eq as f64 + u) / (m + 1) as f64
}

// ─── Calibration ──────────────────────────────────────────────────────────────

/// Build a [`ConformalDetector`] from a pre-computed set of calibration scores.
///
/// The calibration scores must be from the **same scorer** that will be used
/// to score test points (they are stored as-is; no sorting or sorting is done).
pub fn conformal_calibrate(scores: Vec<f64>, cfg: ConformalConfig) -> ConformalDetector {
    ConformalDetector {
        calibration_scores: scores,
        config: cfg,
    }
}

// ─── Batch prediction ─────────────────────────────────────────────────────────

/// Predict p-values and anomaly labels for a batch of test scores.
///
/// Each test score is compared against the detector's fixed calibration set.
/// Optionally applies smoothed p-values if `config.smoothing == true`.
///
/// # Errors
///
/// Returns [`AnomalyError::EmptyInput`] if `test_scores` is empty or the
/// calibration set is empty.
pub fn conformal_predict(
    detector: &ConformalDetector,
    test_scores: &[f64],
) -> AnomalyResult<ConformalResult> {
    if test_scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }
    if detector.calibration_scores.is_empty() {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }

    let eps = detector.config.significance;
    let mut p_values = Vec::with_capacity(test_scores.len());
    let mut labels = Vec::with_capacity(test_scores.len());

    if detector.config.smoothing {
        // Smoothed variant: requires a local RNG derived from a fixed seed so
        // outputs remain deterministic when called multiple times with the same
        // detector state.
        let mut rng = LcgRng::new(0xDEAD_BEEF_CAFE_BABE);
        for &ts in test_scores {
            let pv = conformal_p_value_smoothed(&detector.calibration_scores, ts, &mut rng);
            let label = if pv < eps { -1 } else { 1 };
            p_values.push(pv);
            labels.push(label);
        }
    } else {
        for &ts in test_scores {
            let pv = conformal_p_value(&detector.calibration_scores, ts);
            let label = if pv < eps { -1 } else { 1 };
            p_values.push(pv);
            labels.push(label);
        }
    }

    let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
    Ok(ConformalResult {
        p_values,
        labels,
        n_anomalies,
    })
}

// ─── Mondrian conformal ───────────────────────────────────────────────────────

/// Mondrian (class-conditional) conformal anomaly detection.
///
/// Calibration scores are grouped by their class label.  For each test point
/// its p-value is computed only against the calibration scores that share the
/// same class label.  This produces valid conditional coverage:
///
/// ```text
/// P(p(x, y) < ε | y = c) ≤ ε   for all classes c.
/// ```
///
/// # Arguments
///
/// * `calibration_scores` – anomaly scores of calibration points.
/// * `calibration_labels` – class membership of each calibration point (0-indexed).
/// * `test_scores`  – anomaly scores of test points.
/// * `test_labels`  – class membership of each test point.
/// * `significance` – significance level ε.
///
/// # Errors
///
/// * [`AnomalyError::EmptyInput`] – either set is empty.
/// * [`AnomalyError::DimensionMismatch`] – lengths do not match within each pair.
pub fn mondrian_conformal_predict(
    calibration_scores: &[f64],
    calibration_labels: &[usize],
    test_scores: &[f64],
    test_labels: &[usize],
    significance: f64,
) -> AnomalyResult<ConformalResult> {
    if calibration_scores.is_empty() || test_scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }
    if calibration_scores.len() != calibration_labels.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: calibration_scores.len(),
            got: calibration_labels.len(),
        });
    }
    if test_scores.len() != test_labels.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: test_scores.len(),
            got: test_labels.len(),
        });
    }

    // Build per-class calibration score lookup.
    let n_classes = calibration_labels.iter().copied().max().unwrap_or(0) + 1;
    let mut class_scores: Vec<Vec<f64>> = vec![Vec::new(); n_classes];
    for (i, &label) in calibration_labels.iter().enumerate() {
        class_scores[label].push(calibration_scores[i]);
    }

    let mut p_values = Vec::with_capacity(test_scores.len());
    let mut labels = Vec::with_capacity(test_scores.len());

    for (i, &ts) in test_scores.iter().enumerate() {
        let class = test_labels[i];
        // If the class is unseen in calibration, use an empty slice → p-value 0.
        let cal_slice: &[f64] = if class < class_scores.len() {
            &class_scores[class]
        } else {
            &[]
        };
        let pv = conformal_p_value(cal_slice, ts);
        let label = if pv < significance { -1 } else { 1 };
        p_values.push(pv);
        labels.push(label);
    }

    let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
    Ok(ConformalResult {
        p_values,
        labels,
        n_anomalies,
    })
}

// ─── Online conformal ─────────────────────────────────────────────────────────

/// Create a new online (sliding-window) conformal detector.
///
/// The detector starts with an empty window.  Scores arrive one at a time via
/// [`online_conformal_update`].
#[must_use]
pub fn online_conformal_detector_new(
    window_size: usize,
    cfg: ConformalConfig,
) -> OnlineConformalDetector {
    OnlineConformalDetector {
        window: VecDeque::with_capacity(window_size),
        window_size,
        config: cfg,
    }
}

/// Process a single incoming score in the online conformal detector.
///
/// Steps:
/// 1. Compute the conformal p-value of `score` against the current window.
/// 2. Determine the binary label based on `config.significance`.
/// 3. Insert `score` into the back of the window; if the window exceeds
///    `window_size`, pop the oldest score from the front.
///
/// Returns `(p_value, label)` where `label` is `+1` (normal) or `-1` (anomaly).
///
/// When the window is empty the p-value is defined as `0.0`.
pub fn online_conformal_update(detector: &mut OnlineConformalDetector, score: f64) -> (f64, i32) {
    // Step 1: compute p-value against current window.
    let window_slice: Vec<f64> = detector.window.iter().copied().collect();
    let pv = if window_slice.is_empty() {
        // No calibration data yet: treat as maximally suspicious.
        0.0
    } else {
        conformal_p_value(&window_slice, score)
    };

    // Step 2: label.
    let label = if pv < detector.config.significance {
        -1_i32
    } else {
        1_i32
    };

    // Step 3: slide window forward.
    if detector.window.len() >= detector.window_size && detector.window_size > 0 {
        detector.window.pop_front();
    }
    if detector.window_size > 0 {
        detector.window.push_back(score);
    }

    (pv, label)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AnomalyResult;

    // ── Test 1: score equal to max calibration → p = 1/(m+1) ────────────────

    #[test]
    fn conformal_p_value_basic() {
        // Calibration: [1, 2, 3, 4, 5].  Test score = 5 (equal to max).
        // count_ge(5) = 1 (only the 5 itself), so p = 1 / (5 + 1) = 1/6 ≈ 0.1667.
        let cal = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let p = conformal_p_value(&cal, 5.0);
        let expected = 1.0 / 6.0;
        assert!((p - expected).abs() < 1e-10, "p={p} expected={expected}");
    }

    // ── Test 2: score much higher than all calibration → p ≈ 0 ─────────────

    #[test]
    fn conformal_p_value_low() {
        let cal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // score = 1e6, no calibration value ≥ 1e6 → count_ge = 0.
        let p = conformal_p_value(&cal, 1_000_000.0);
        assert_eq!(p, 0.0, "p should be exactly 0, got {p}");
    }

    // ── Test 3: p-value is always in [0, 1] ─────────────────────────────────

    #[test]
    fn conformal_p_value_range() {
        let mut rng = LcgRng::new(42);
        let cal: Vec<f64> = (0..50).map(|_| rng.next_f32() as f64 * 10.0).collect();
        for _ in 0..200 {
            let ts = rng.next_f32() as f64 * 20.0; // may exceed calibration range
            let p = conformal_p_value(&cal, ts);
            assert!((0.0..=1.0).contains(&p), "p={p} outside [0,1] for ts={ts}");
        }
    }

    // ── Test 4: conformal_predict — output shape matches n_test ─────────────

    #[test]
    fn conformal_predict_shape() -> AnomalyResult<()> {
        let cal: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let cfg = ConformalConfig::default();
        let detector = conformal_calibrate(cal, cfg);
        let test_scores = vec![5.0_f64, 15.0, 100.0, -1.0, 0.0];
        let result = conformal_predict(&detector, &test_scores)?;
        assert_eq!(result.labels.len(), test_scores.len());
        assert_eq!(result.p_values.len(), test_scores.len());
        Ok(())
    }

    // ── Test 5: FPR on clean data ≈ significance level ───────────────────────
    //
    // Under the null (test data drawn from the same distribution as
    // calibration), the expected FPR is bounded by `significance`.

    #[test]
    fn conformal_false_positive_rate() -> AnomalyResult<()> {
        let significance = 0.05;
        let m = 500_usize; // calibration size
        let n_test = 500_usize;
        let mut rng = LcgRng::new(0xCAFE_DEAD);

        // Calibration scores: Uniform(0, 1).
        let cal: Vec<f64> = (0..m).map(|_| rng.next_f32() as f64).collect();
        // Test scores: also Uniform(0, 1) — same distribution.
        let test: Vec<f64> = (0..n_test).map(|_| rng.next_f32() as f64).collect();

        let cfg = ConformalConfig {
            significance,
            smoothing: false,
        };
        let det = conformal_calibrate(cal, cfg);
        let result = conformal_predict(&det, &test)?;

        let fpr = result.n_anomalies as f64 / n_test as f64;
        // The FPR must be ≤ significance + some slack for finite samples.
        assert!(
            fpr <= significance + 0.04,
            "FPR={fpr:.4} too high (significance={significance})"
        );
        Ok(())
    }

    // ── Test 6: known outlier (very high score) is flagged ───────────────────

    #[test]
    fn conformal_anomaly_detected() -> AnomalyResult<()> {
        // Calibration in [0, 1]; outlier score = 100.
        let cal: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let cfg = ConformalConfig {
            significance: 0.05,
            smoothing: false,
        };
        let det = conformal_calibrate(cal, cfg);
        // p-value for score=100 is 0.0 (no cal score ≥ 100) → labelled -1.
        let result = conformal_predict(&det, &[100.0_f64])?;
        assert_eq!(result.labels[0], -1, "outlier must be labelled -1");
        assert_eq!(result.p_values[0], 0.0);
        Ok(())
    }

    // ── Test 7: Mondrian — p-values computed per class ───────────────────────

    #[test]
    fn mondrian_per_class() -> AnomalyResult<()> {
        // Class 0: calibration scores in [0, 1].
        // Class 1: calibration scores in [10, 11].
        let mut cal_scores = Vec::new();
        let mut cal_labels = Vec::new();
        for i in 0..20_usize {
            cal_scores.push(i as f64 / 20.0);
            cal_labels.push(0_usize);
        }
        for i in 0..20_usize {
            cal_scores.push(10.0 + i as f64 / 20.0);
            cal_labels.push(1_usize);
        }

        // Test: class-0 point with score 0.5 (normal for class 0),
        //       class-1 point with score 100 (anomalous for class 1).
        let test_scores = vec![0.5_f64, 100.0];
        let test_labels = vec![0_usize, 1_usize];

        let result =
            mondrian_conformal_predict(&cal_scores, &cal_labels, &test_scores, &test_labels, 0.05)?;

        // Class-0 score 0.5 → p-value should be > 0 (it's within the calibration range).
        assert!(
            result.p_values[0] > 0.0,
            "class-0 normal point p={}",
            result.p_values[0]
        );
        // Class-1 score 100 → no calibration score ≥ 100 → p = 0 → anomaly.
        assert_eq!(result.labels[1], -1, "class-1 outlier must be -1");
        assert_eq!(result.p_values[1], 0.0);
        Ok(())
    }

    // ── Test 8: online detector window fills correctly ───────────────────────

    #[test]
    fn online_detector_window_fills() {
        let cfg = ConformalConfig::default();
        let mut det = online_conformal_detector_new(10, cfg);
        assert_eq!(det.window.len(), 0);
        for i in 0..10_usize {
            online_conformal_update(&mut det, i as f64);
        }
        assert_eq!(
            det.window.len(),
            10,
            "window should be full after 10 updates"
        );
    }

    // ── Test 9: old scores dropped when window is full ───────────────────────

    #[test]
    fn online_detector_rolling() {
        let cfg = ConformalConfig::default();
        let window_size = 5_usize;
        let mut det = online_conformal_detector_new(window_size, cfg);

        // Insert 0..5 first (fills the window).
        for i in 0..window_size {
            online_conformal_update(&mut det, i as f64);
        }
        // Now insert 100.0 — should evict 0.0 from the front.
        online_conformal_update(&mut det, 100.0);
        assert_eq!(
            det.window.len(),
            window_size,
            "window size must stay constant"
        );
        // The front element should now be 1.0 (0.0 evicted).
        let front = *det
            .window
            .front()
            .expect("window is non-empty after eviction");
        assert!(
            (front - 1.0).abs() < 1e-10,
            "oldest entry should be 1.0 after eviction, got {front}"
        );
    }

    // ── Test 10: extreme score triggers -1 label ─────────────────────────────

    #[test]
    fn online_detector_alarm() {
        let cfg = ConformalConfig {
            significance: 0.05,
            smoothing: false,
        };
        let mut det = online_conformal_detector_new(50, cfg);

        // Fill window with scores drawn from a tight normal cluster.
        let mut rng = LcgRng::new(0xBEEF_CAFE);
        for _ in 0..50 {
            // scores ≈ N(0, 0.1), all small
            let v = rng.next_normal() as f64 * 0.1;
            online_conformal_update(&mut det, v);
        }

        // Feed an extreme outlier; no window score should be ≥ 1000.
        let (p, label) = online_conformal_update(&mut det, 1000.0);
        assert_eq!(label, -1, "extreme score must trigger alarm, p={p}");
        assert_eq!(p, 0.0, "p-value must be 0 for extreme outlier");
    }

    // ── Test 11: smoothed p-value is still in [0, 1] ─────────────────────────

    #[test]
    fn conformal_smoothed_in_range() {
        let mut rng_data = LcgRng::new(777);
        let cal: Vec<f64> = (0..100).map(|_| rng_data.next_f32() as f64 * 5.0).collect();
        let mut rng_smooth = LcgRng::new(999);
        for _ in 0..200 {
            let ts = rng_data.next_f32() as f64 * 10.0;
            let p = conformal_p_value_smoothed(&cal, ts, &mut rng_smooth);
            assert!((0.0..=1.0).contains(&p), "smoothed p={p} outside [0,1]");
        }
    }

    // ── Test 12: mondrian empty-class falls back to p=0 ─────────────────────

    #[test]
    fn mondrian_unseen_class() -> AnomalyResult<()> {
        let cal_scores = vec![0.1_f64, 0.2, 0.3];
        let cal_labels = vec![0_usize, 0, 0];
        // Test point claims to be class 99, which is absent from calibration.
        let result =
            mondrian_conformal_predict(&cal_scores, &cal_labels, &[0.5_f64], &[99_usize], 0.05)?;
        assert_eq!(result.p_values[0], 0.0, "unseen class → p=0");
        assert_eq!(result.labels[0], -1);
        Ok(())
    }

    // ── Test 13: empty test_scores returns error ─────────────────────────────

    #[test]
    fn conformal_predict_empty_test_error() {
        let cal = vec![1.0_f64, 2.0, 3.0];
        let cfg = ConformalConfig::default();
        let det = conformal_calibrate(cal, cfg);
        let err = conformal_predict(&det, &[]).unwrap_err();
        assert!(
            matches!(err, crate::error::AnomalyError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    // ── Test 14: n_anomalies is consistent with labels ───────────────────────

    #[test]
    fn conformal_n_anomalies_consistent() -> AnomalyResult<()> {
        let cal: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let cfg = ConformalConfig::default();
        let det = conformal_calibrate(cal, cfg);
        // Mix of in-range and out-of-range scores.
        let test = vec![10.0_f64, 1000.0, 25.0, 999.0, 5.0];
        let result = conformal_predict(&det, &test)?;
        let counted = result.labels.iter().filter(|&&l| l == -1).count();
        assert_eq!(result.n_anomalies, counted);
        Ok(())
    }
}
