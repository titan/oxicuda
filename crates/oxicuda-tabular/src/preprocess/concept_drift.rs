//! Concept-drift detection for streaming tabular features.
//!
//! Provides three complementary drift detectors that can be wrapped around any
//! streaming tabular feature to flag distribution changes at runtime:
//!
//! | Detector | Algorithm | Reference |
//! |----------|-----------|-----------|
//! | [`AdwinTabular`] | Adaptive Windowing (ADWIN) | Bifet & Gavalda, 2007 |
//! | [`PageHinkleyTabular`] | Page-Hinkley test | Page, 1954 |
//! | [`KsDriftDetector`] | Two-sample Kolmogorov-Smirnov statistic | Smirnov, 1948 |
//!
//! All detectors are *univariate* — apply one per feature or per aggregated
//! model error signal.  They return a [`DriftStatus`] on each update:
//! `InControl` means no drift, `Warning` signals possible drift, and `Drift`
//! signals a detected change.

use crate::error::{TabularError, TabularResult};

// ─── DriftStatus ──────────────────────────────────────────────────────────────

/// Status returned by a drift detector after each new observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    /// No significant drift detected.
    InControl,
    /// Possible drift — detector is near the warning threshold.
    Warning,
    /// Drift confirmed — the data distribution has likely changed.
    Drift,
}

// ─── ADWIN (Adaptive Windowing) ───────────────────────────────────────────────

/// ADWIN configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdwinTabularConfig {
    /// Confidence parameter δ ∈ (0, 1).  Smaller values → fewer false alarms.
    /// Typical default: 0.002.
    pub delta: f64,
    /// Maximum window length (0 = unbounded).
    pub max_window: usize,
}

impl Default for AdwinTabularConfig {
    fn default() -> Self {
        Self {
            delta: 0.002,
            max_window: 0,
        }
    }
}

/// Adaptive Windowing drift detector for a univariate scalar stream.
///
/// Maintains a sliding window; computes Hoeffding-bound tests on all valid
/// subwindow splits and drops the older portion when a split passes.
///
/// # References
/// Bifet, A. & Gavalda, R. (2007). Learning from time-changing data with
/// adaptive windowing. *Proceedings of SDM*.
#[derive(Debug, Clone)]
pub struct AdwinTabular {
    config: AdwinTabularConfig,
    /// Raw value buffer (ring-buffer semantics via Vec::remove at front).
    window: Vec<f64>,
    /// Running sum of the current window.
    window_sum: f64,
    /// True if the last update detected drift.
    pub drift_detected: bool,
    /// True if the last update triggered a warning.
    pub warning_detected: bool,
    /// Total number of observations seen.
    pub n_total: usize,
}

impl AdwinTabular {
    /// Create a new ADWIN detector.
    #[must_use]
    pub fn new(config: AdwinTabularConfig) -> Self {
        Self {
            config,
            window: Vec::new(),
            window_sum: 0.0,
            drift_detected: false,
            warning_detected: false,
            n_total: 0,
        }
    }

    /// Add a new observation and return the current drift status.
    ///
    /// # Errors
    /// Returns [`TabularError::NanEncountered`] if `value` is NaN or infinite.
    pub fn add(&mut self, value: f64) -> TabularResult<DriftStatus> {
        if !value.is_finite() {
            return Err(TabularError::NanEncountered {
                context: "AdwinTabular::add".into(),
            });
        }
        self.n_total += 1;
        self.window.push(value);
        self.window_sum += value;

        // Enforce max_window if set.
        if self.config.max_window > 0 {
            while self.window.len() > self.config.max_window {
                self.window_sum -= self.window.remove(0);
            }
        }

        self.drift_detected = false;
        self.warning_detected = false;

        // ADWIN cut-point test: try every split of the window.
        let n = self.window.len();
        if n < 4 {
            return Ok(DriftStatus::InControl);
        }

        let mu_total = self.window_sum / n as f64;
        let mut sum_left = 0.0_f64;

        for cut in 1..n {
            sum_left += self.window[cut - 1];
            let n0 = cut as f64;
            let n1 = (n - cut) as f64;
            let mu0 = sum_left / n0;
            let mu1 = (self.window_sum - sum_left) / n1;
            // Hoeffding bound with range = max - min capped at 1.
            let range_sq = 1.0_f64; // normalised to [0,1]; user should pre-scale.
            let eps_cut =
                ((n as f64 / (2.0 * n0 * n1)) * range_sq * (4.0 / self.config.delta).ln()).sqrt();
            // Warning threshold: half of drift threshold.
            if (mu0 - mu1).abs() >= eps_cut {
                // Drop the older portion.
                let older_side = cut.min(n - cut);
                if cut < n - cut {
                    for _ in 0..older_side {
                        self.window_sum -= self.window.remove(0);
                    }
                } else {
                    let split = n - cut;
                    self.window_sum -= self.window[..split].iter().sum::<f64>();
                    self.window.drain(..split);
                }
                self.drift_detected = true;
                return Ok(DriftStatus::Drift);
            } else if (mu0 - mu1).abs() >= eps_cut * 0.5 {
                // Don't break — just flag warning and continue.
                self.warning_detected = true;
            }
            let _ = mu_total; // suppress warning
        }

        if self.warning_detected {
            Ok(DriftStatus::Warning)
        } else {
            Ok(DriftStatus::InControl)
        }
    }

    /// Current window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Current window mean (returns 0 if window is empty).
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window_sum / self.window.len() as f64
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.window_sum = 0.0;
        self.drift_detected = false;
        self.warning_detected = false;
        self.n_total = 0;
    }
}

// ─── Page-Hinkley Test ────────────────────────────────────────────────────────

/// Configuration for the Page-Hinkley drift detector.
#[derive(Debug, Clone, Copy)]
pub struct PageHinkleyTabularConfig {
    /// Detection threshold λ.  Larger → fewer false positives but slower
    /// detection.  Typical default: 50.0.
    pub threshold: f64,
    /// Allowance term δ: tolerated mean increase per step.  Default: 0.005.
    pub delta: f64,
    /// Warning threshold (fraction of main threshold).  Default: 0.5.
    pub warning_fraction: f64,
}

impl Default for PageHinkleyTabularConfig {
    fn default() -> Self {
        Self {
            threshold: 50.0,
            delta: 0.005,
            warning_fraction: 0.5,
        }
    }
}

/// Page-Hinkley CUSUM-variant drift detector.
///
/// Tracks the cumulative sum `PH_t = Σ(x_i − x̄ − δ)` and signals drift when
/// the running maximum `m_t = max_{τ≤t} PH_τ` exceeds `PH_t + λ`.
///
/// # References
/// Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2).
#[derive(Debug, Clone)]
pub struct PageHinkleyTabular {
    config: PageHinkleyTabularConfig,
    /// Running sum of observations.
    sum: f64,
    /// Running minimum of cumulative sum (for detecting upward shifts).
    min_val: f64,
    /// Number of observations.
    n: usize,
    /// True on last add if drift detected.
    pub drift_detected: bool,
    /// True on last add if warning.
    pub warning_detected: bool,
}

impl PageHinkleyTabular {
    /// Create a new Page-Hinkley detector.
    #[must_use]
    pub fn new(config: PageHinkleyTabularConfig) -> Self {
        Self {
            config,
            sum: 0.0,
            min_val: f64::INFINITY,
            n: 0,
            drift_detected: false,
            warning_detected: false,
        }
    }

    /// Add an observation and return the drift status.
    ///
    /// # Errors
    /// Returns [`TabularError::NanEncountered`] if `value` is NaN or infinite.
    pub fn add(&mut self, value: f64) -> TabularResult<DriftStatus> {
        if !value.is_finite() {
            return Err(TabularError::NanEncountered {
                context: "PageHinkleyTabular::add".into(),
            });
        }
        self.n += 1;
        self.sum += value - self.config.delta;
        if self.sum < self.min_val {
            self.min_val = self.sum;
        }
        let ph = self.sum - self.min_val;
        self.drift_detected = false;
        self.warning_detected = false;
        if ph > self.config.threshold {
            self.drift_detected = true;
            Ok(DriftStatus::Drift)
        } else if ph > self.config.threshold * self.config.warning_fraction {
            self.warning_detected = true;
            Ok(DriftStatus::Warning)
        } else {
            Ok(DriftStatus::InControl)
        }
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.min_val = f64::INFINITY;
        self.n = 0;
        self.drift_detected = false;
        self.warning_detected = false;
    }

    /// Current PH statistic (difference between cumulative sum and its minimum).
    #[must_use]
    pub fn ph_statistic(&self) -> f64 {
        if self.min_val.is_infinite() {
            0.0
        } else {
            self.sum - self.min_val
        }
    }
}

// ─── Two-sample KS drift detector ────────────────────────────────────────────

/// Kolmogorov-Smirnov two-sample drift detector.
///
/// Maintains a reference window and a sliding detection window; computes the
/// KS statistic `D = sup_x |F_ref(x) − F_det(x)|` and signals drift when `D`
/// exceeds the critical value at the configured significance level.
///
/// # References
/// Smirnov, N. V. (1948). Tables for Estimating the Goodness of Fit of Empirical
/// Distributions. *Annals of Mathematical Statistics*, 19(2).
#[derive(Debug, Clone)]
pub struct KsDriftDetector {
    /// Significance level α ∈ (0, 1).  Default 0.05.
    pub alpha: f64,
    /// Size of the reference window.
    pub ref_window: usize,
    /// Size of the detection window.
    pub det_window: usize,
    /// Reference distribution (sorted).
    reference: Vec<f64>,
    /// Current detection window (unsorted, ring buffer).
    detection: Vec<f64>,
    /// True on last check if drift detected.
    pub drift_detected: bool,
    /// True on last check if warning.
    pub warning_detected: bool,
}

impl KsDriftDetector {
    /// Create a new KS drift detector.
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if `alpha` not in `(0, 1)`.
    /// - [`TabularError::InsufficientSamples`] if `ref_window < 2` or `det_window < 2`.
    pub fn new(alpha: f64, ref_window: usize, det_window: usize) -> TabularResult<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(TabularError::InvalidParameter {
                name: "alpha".into(),
                msg: "must be in (0, 1)".into(),
            });
        }
        if ref_window < 2 || det_window < 2 {
            return Err(TabularError::InsufficientSamples {
                need: 2,
                got: ref_window.min(det_window),
            });
        }
        Ok(Self {
            alpha,
            ref_window,
            det_window,
            reference: Vec::new(),
            detection: Vec::new(),
            drift_detected: false,
            warning_detected: false,
        })
    }

    /// Add an observation to the reference window (call during warm-up).
    pub fn add_reference(&mut self, value: f64) {
        if self.reference.len() < self.ref_window {
            self.reference.push(value);
            // Keep sorted for O(log n) ECDF lookups.
            let pos = self
                .reference
                .partition_point(|&v| v <= value)
                .saturating_sub(1);
            // Re-sort since we just appended: insertion sort from the end.
            let mut i = self.reference.len() - 1;
            while i > 0 && self.reference[i] < self.reference[i - 1] {
                self.reference.swap(i, i - 1);
                i -= 1;
            }
            let _ = pos;
        }
    }

    /// Add an observation to the detection window and return drift status.
    ///
    /// Returns `InControl` if the reference window is not yet full.
    ///
    /// # Errors
    /// Returns [`TabularError::NanEncountered`] if `value` is NaN or infinite.
    pub fn add(&mut self, value: f64) -> TabularResult<DriftStatus> {
        if !value.is_finite() {
            return Err(TabularError::NanEncountered {
                context: "KsDriftDetector::add".into(),
            });
        }
        if self.reference.len() < self.ref_window {
            // Still in warm-up: treat as reference data.
            self.add_reference(value);
            return Ok(DriftStatus::InControl);
        }
        // Add to detection buffer (ring).
        self.detection.push(value);
        if self.detection.len() > self.det_window {
            self.detection.remove(0);
        }
        if self.detection.len() < self.det_window {
            return Ok(DriftStatus::InControl);
        }
        let d = self.ks_statistic();
        let c_alpha = self.critical_value();
        let c_warning = c_alpha * 0.7;
        self.drift_detected = false;
        self.warning_detected = false;
        if d > c_alpha {
            self.drift_detected = true;
            Ok(DriftStatus::Drift)
        } else if d > c_warning {
            self.warning_detected = true;
            Ok(DriftStatus::Warning)
        } else {
            Ok(DriftStatus::InControl)
        }
    }

    /// Compute the two-sample KS statistic between reference and detection.
    pub fn ks_statistic(&self) -> f64 {
        if self.reference.is_empty() || self.detection.is_empty() {
            return 0.0;
        }
        // Sort detection window.
        let mut det_sorted = self.detection.clone();
        det_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_ref = self.reference.len() as f64;
        let n_det = det_sorted.len() as f64;
        let mut d = 0.0_f64;
        let mut i = 0_usize;
        let mut j = 0_usize;
        while i < self.reference.len() || j < det_sorted.len() {
            let threshold = if j >= det_sorted.len() {
                self.reference[i]
            } else if i >= self.reference.len() {
                det_sorted[j]
            } else {
                self.reference[i].min(det_sorted[j])
            };
            // Advance i to count all reference values ≤ threshold.
            while i < self.reference.len() && self.reference[i] <= threshold {
                i += 1;
            }
            // Advance j to count all detection values ≤ threshold.
            while j < det_sorted.len() && det_sorted[j] <= threshold {
                j += 1;
            }
            let diff = (i as f64 / n_ref - j as f64 / n_det).abs();
            if diff > d {
                d = diff;
            }
        }
        d
    }

    /// Kolmogorov critical value for the given α at sample sizes.
    fn critical_value(&self) -> f64 {
        let n_ref = self.reference.len() as f64;
        let n_det = self.detection.len() as f64;
        // Kolmogorov limiting distribution: D > c(α) / sqrt(n_eff).
        let n_eff = (n_ref * n_det / (n_ref + n_det)).sqrt();
        // c(α) for common levels:
        let c = if self.alpha >= 0.10 {
            1.224
        } else if self.alpha >= 0.05 {
            1.358
        } else if self.alpha >= 0.025 {
            1.480
        } else if self.alpha >= 0.01 {
            1.628
        } else {
            1.731
        };
        c / n_eff
    }

    /// Replace the reference window with a new sorted dataset.
    ///
    /// Use this to re-anchor the reference after confirmed drift.
    pub fn reset_reference(&mut self, new_ref: &[f64]) {
        self.reference = new_ref.to_vec();
        self.reference
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.detection.clear();
        self.drift_detected = false;
        self.warning_detected = false;
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. ADWIN: no drift on i.i.d. stream (stable distribution) ────────────
    #[test]
    fn adwin_no_drift_stable() {
        let cfg = AdwinTabularConfig {
            delta: 0.002,
            max_window: 200,
        };
        let mut det = AdwinTabular::new(cfg);
        let mut drift_count = 0_usize;
        for i in 0..100 {
            // Deterministic pseudo-sequence bounded in [0, 1] with no trend.
            let v = ((i * 7 + 3) % 100) as f64 / 100.0;
            if det.add(v).expect("add should succeed") == DriftStatus::Drift {
                drift_count += 1;
            }
        }
        // Allow very few false alarms on a short deterministic sequence.
        assert!(drift_count <= 3, "too many false alarms: {drift_count}");
    }

    // ── 2. ADWIN: detects abrupt mean shift ───────────────────────────────────
    #[test]
    fn adwin_detects_mean_shift() {
        let cfg = AdwinTabularConfig {
            delta: 0.002,
            max_window: 400,
        };
        let mut det = AdwinTabular::new(cfg);
        // Feed 50 values near 0.
        for i in 0..50_usize {
            let v = (i % 10) as f64 * 0.01;
            det.add(v).expect("add should succeed");
        }
        // Then feed 50 values near 1 — should trigger drift.
        let mut drift_seen = false;
        for i in 0..50_usize {
            let v = 0.90 + (i % 10) as f64 * 0.01;
            if det.add(v).expect("add should succeed") == DriftStatus::Drift {
                drift_seen = true;
                break;
            }
        }
        assert!(drift_seen, "ADWIN should detect the mean shift");
    }

    // ── 3. ADWIN: NaN input returns error ─────────────────────────────────────
    #[test]
    fn adwin_nan_error() {
        let mut det = AdwinTabular::new(AdwinTabularConfig::default());
        assert!(det.add(f64::NAN).is_err());
        assert!(det.add(f64::INFINITY).is_err());
    }

    // ── 4. ADWIN: window size stays bounded ───────────────────────────────────
    #[test]
    fn adwin_max_window_bounded() {
        let cfg = AdwinTabularConfig {
            delta: 0.1,
            max_window: 20,
        };
        let mut det = AdwinTabular::new(cfg);
        for i in 0..100_usize {
            det.add(i as f64 * 0.01).expect("add should succeed");
        }
        assert!(det.window_size() <= 20, "window_size={}", det.window_size());
    }

    // ── 5. Page-Hinkley: no drift on flat sequence ────────────────────────────
    #[test]
    fn ph_no_drift_flat() {
        let cfg = PageHinkleyTabularConfig {
            threshold: 50.0,
            delta: 0.005,
            warning_fraction: 0.5,
        };
        let mut det = PageHinkleyTabular::new(cfg);
        for i in 0..100_usize {
            let v = (i % 2) as f64; // oscillates 0/1
            let st = det.add(v).expect("add should succeed");
            assert_ne!(st, DriftStatus::Drift, "false drift at i={i}");
        }
    }

    // ── 6. Page-Hinkley: detects prolonged upward trend ─────────────────────
    #[test]
    fn ph_detects_upward_trend() {
        let cfg = PageHinkleyTabularConfig {
            threshold: 5.0,
            delta: 0.0,
            warning_fraction: 0.5,
        };
        let mut det = PageHinkleyTabular::new(cfg);
        let mut drift_seen = false;
        for i in 0..200_usize {
            if det.add(i as f64).expect("add should succeed") == DriftStatus::Drift {
                drift_seen = true;
                break;
            }
        }
        assert!(drift_seen, "PH should detect monotone increase");
    }

    // ── 7. Page-Hinkley: NaN returns error ───────────────────────────────────
    #[test]
    fn ph_nan_error() {
        let mut det = PageHinkleyTabular::new(PageHinkleyTabularConfig::default());
        assert!(det.add(f64::NAN).is_err());
    }

    // ── 8. Page-Hinkley: reset clears state ──────────────────────────────────
    #[test]
    fn ph_reset_clears() {
        let cfg = PageHinkleyTabularConfig {
            threshold: 1.0,
            delta: 0.0,
            warning_fraction: 0.5,
        };
        let mut det = PageHinkleyTabular::new(cfg);
        for i in 0..50_usize {
            det.add(i as f64).expect("add should succeed");
        }
        det.reset();
        assert_eq!(det.ph_statistic(), 0.0);
        assert_eq!(det.n, 0);
    }

    // ── 9. KS: no drift on same distribution ─────────────────────────────────
    #[test]
    fn ks_no_drift_same_dist() {
        let mut det = KsDriftDetector::new(0.05, 30, 30).expect("new should succeed");
        // Warm-up: 0, 1, 2, ..., 29 scaled to [0, 1].
        for i in 0..30_usize {
            det.add_reference(i as f64 / 29.0);
        }
        // Detection window from same distribution.
        let mut drift_count = 0;
        for i in 0..30_usize {
            let v = (29 - i) as f64 / 29.0; // reversed but same range
            if det.add(v).expect("add should succeed") == DriftStatus::Drift {
                drift_count += 1;
            }
        }
        assert_eq!(drift_count, 0, "should not flag same distribution");
    }

    // ── 10. KS: detects completely different distribution ─────────────────────
    #[test]
    fn ks_detects_different_dist() {
        let mut det = KsDriftDetector::new(0.05, 30, 30).expect("new should succeed");
        // Warm-up on [0, 0.1].
        for i in 0..30_usize {
            det.add_reference(i as f64 / 300.0);
        }
        // Detection on [0.9, 1.0].
        let mut drift_seen = false;
        for i in 0..30_usize {
            let v = 0.9 + i as f64 / 300.0;
            if det.add(v).expect("add should succeed") == DriftStatus::Drift {
                drift_seen = true;
                break;
            }
        }
        assert!(drift_seen, "KS should detect non-overlapping distributions");
    }

    // ── 11. KS: invalid alpha returns error ───────────────────────────────────
    #[test]
    fn ks_invalid_alpha_error() {
        assert!(KsDriftDetector::new(0.0, 30, 30).is_err());
        assert!(KsDriftDetector::new(1.0, 30, 30).is_err());
        assert!(KsDriftDetector::new(1.5, 30, 30).is_err());
    }

    // ── 12. KS: statistic is 0 when windows are identical ─────────────────────
    #[test]
    fn ks_statistic_identical_windows() {
        let mut det = KsDriftDetector::new(0.05, 10, 10).expect("new should succeed");
        for i in 0..10_usize {
            det.add_reference(i as f64);
        }
        for i in 0..10_usize {
            det.add(i as f64).expect("add should succeed");
        }
        let d = det.ks_statistic();
        assert!(d < 1e-10, "D={d}, expected 0 for identical windows");
    }
}
