//! Online streaming anomaly detectors.
//!
//! Provides three complementary algorithms for processing data streams where
//! the full batch is not available upfront:
//!
//! * [`OnlineZScore`] — Welford one-pass mean/variance → z-score.
//! * [`ExponentialZ`] — EWMA-based z-score with configurable forgetting factor.
//! * [`SlidingMad`] — Sliding-window Median Absolute Deviation.
//! * [`StreamingThresholdDetector`] — Unified facade over all three methods.

use std::collections::VecDeque;

// ─── OnlineZScore ─────────────────────────────────────────────────────────────

/// Welford one-pass online mean / variance tracker with z-score anomaly detection.
///
/// Uses Welford's numerically stable algorithm:
/// ```text
/// delta  = x − mean
/// mean  += delta / n
/// delta2 = x − mean  (updated mean!)
/// M2    += delta * delta2
/// var    = M2 / (n − 1)   (sample variance, for n ≥ 2)
/// ```
#[derive(Debug, Clone)]
pub struct OnlineZScore {
    /// Running mean (Welford).
    pub mean: f64,
    /// Running sum of squared deviations (Welford M2).
    pub m2: f64,
    /// Number of observations seen so far.
    pub n: u64,
    /// Anomaly threshold: `|z| > threshold` → anomaly.
    pub threshold: f64,
}

impl OnlineZScore {
    /// Create a fresh detector with the given threshold.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            n: 0,
            threshold,
        }
    }

    /// Compute the z-score of `x` against the **current** (pre-update) distribution.
    ///
    /// Returns `0.0` if fewer than 2 observations have been seen.
    #[must_use]
    pub fn score(&self, x: f64) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let variance = self.m2 / (self.n - 1) as f64;
        let std = (variance + 1e-12).sqrt();
        (x - self.mean).abs() / std
    }

    /// Incorporate a new observation `x` into the running statistics.
    ///
    /// The anomaly decision is made using the **pre-update** distribution so that
    /// the outlier does not inflate its own mean/variance before being scored.
    ///
    /// Returns `Some(true)` if `x` is anomalous (|z| > threshold), `Some(false)`
    /// if normal, or `None` if there are not yet enough samples (n < 2 before update).
    pub fn update(&mut self, x: f64) -> Option<bool> {
        // Score against current (pre-update) distribution first
        let decision = if self.n >= 2 {
            let z = self.score(x);
            Some(z > self.threshold)
        } else {
            None
        };

        // Update Welford statistics
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        decision
    }

    /// Update state and return the z-score of `x` (using pre-update stats for scoring,
    /// then updates).  Returns `0.0` for the first two observations.
    pub fn window_z_score_update(&mut self, x: f64) -> f64 {
        // Score using current stats (before update) so result is comparable
        let z = self.score(x);
        let _ = self.update(x);
        z
    }

    /// Reset all statistics to initial state.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.n = 0;
    }
}

impl Default for OnlineZScore {
    fn default() -> Self {
        Self::new(3.0)
    }
}

// ─── ExponentialZ ─────────────────────────────────────────────────────────────

/// EWMA-based z-score detector with configurable forgetting factor `α`.
///
/// State update per new observation `x`:
/// ```text
/// ema_mean = (1 − α) * ema_mean + α * x
/// ema_var  = (1 − α) * ema_var  + α * (x − ema_mean)²
/// z        = |x − ema_mean| / sqrt(ema_var + ε)
/// ```
///
/// Smaller `α` → longer memory (slower adaptation).
/// Default `α = 0.05`.
#[derive(Debug, Clone)]
pub struct ExponentialZ {
    /// Exponential moving average of the mean.
    pub ema_mean: f64,
    /// Exponential moving average of the variance.
    pub ema_var: f64,
    /// Forgetting factor `α ∈ (0, 1)`.
    pub alpha: f64,
    /// Anomaly threshold on the z-score.
    pub threshold: f64,
    /// Number of observations processed.
    pub n: u64,
}

impl ExponentialZ {
    /// Create a new EWMA z-score detector.
    ///
    /// # Arguments
    /// * `alpha` — forgetting factor (0.05 is a good default).
    /// * `threshold` — z-score threshold above which a point is anomalous.
    #[must_use]
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            ema_mean: 0.0,
            ema_var: 1.0, // start with non-zero variance to avoid div-by-zero early on
            alpha,
            threshold,
            n: 0,
        }
    }

    /// Compute the EWMA z-score for `x` **before** updating the state.
    #[must_use]
    pub fn score(&self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let std = (self.ema_var + 1e-12).sqrt();
        (x - self.ema_mean).abs() / std
    }

    /// Update EWMA state with `x` and return anomaly decision.
    ///
    /// Returns `None` on the first observation (no variance estimate yet),
    /// `Some(true)` if anomalous, `Some(false)` otherwise.
    pub fn update(&mut self, x: f64) -> Option<bool> {
        if self.n == 0 {
            // Seed the mean with the first observation
            self.ema_mean = x;
            self.n = 1;
            return None;
        }
        let z = self.score(x);
        // Update EMA mean first, then EMA variance
        self.ema_mean = (1.0 - self.alpha) * self.ema_mean + self.alpha * x;
        self.ema_var = (1.0 - self.alpha) * self.ema_var + self.alpha * (x - self.ema_mean).powi(2);
        self.n += 1;
        Some(z > self.threshold)
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.ema_mean = 0.0;
        self.ema_var = 1.0;
        self.n = 0;
    }
}

impl Default for ExponentialZ {
    fn default() -> Self {
        Self::new(0.05, 3.0)
    }
}

// ─── SlidingMad ──────────────────────────────────────────────────────────────

/// Sliding-window Median Absolute Deviation (MAD) detector.
///
/// Maintains a circular buffer of the last `window_size` observations.
/// Each `update` call recomputes the median and MAD over the current window.
///
/// Score = `|x − median(window)| / (MAD(window) × mad_scale + ε)`.
///
/// `mad_scale = 1.4826` makes MAD consistent with the standard deviation
/// for Gaussian data.
#[derive(Debug, Clone)]
pub struct SlidingMad {
    /// Circular buffer of the `window_size` most recent observations.
    window: VecDeque<f64>,
    /// Consistency scale factor (1.4826 for Gaussian consistency).
    pub mad_scale: f64,
    /// Anomaly threshold on the normalised MAD score.
    pub threshold: f64,
    /// Maximum window size.
    pub window_size: usize,
}

impl SlidingMad {
    /// Create a new sliding MAD detector.
    ///
    /// # Arguments
    /// * `window_size` — number of recent samples to retain.
    /// * `mad_scale`   — consistency factor (1.4826 for Gaussian).
    /// * `threshold`   — normalised-MAD anomaly threshold.
    #[must_use]
    pub fn new(window_size: usize, mad_scale: f64, threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size + 1),
            mad_scale,
            threshold,
            window_size: window_size.max(1),
        }
    }

    /// Compute the median of a sorted slice (helper, avoids extra sort).
    fn sorted_median(sorted: &[f64]) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
        } else {
            sorted[n / 2]
        }
    }

    /// Compute the MAD score for `x` given the **current** window (without modifying it).
    #[must_use]
    pub fn score(&self, x: f64) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = Self::sorted_median(&sorted);
        let mut abs_devs: Vec<f64> = sorted.iter().map(|v| (v - med).abs()).collect();
        abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = Self::sorted_median(&abs_devs);
        let denom = mad * self.mad_scale + 1e-12;
        (x - med).abs() / denom
    }

    /// Incorporate `x` into the sliding window and return anomaly decision.
    ///
    /// Returns `None` if the window has fewer than 3 observations (not enough for MAD),
    /// otherwise `Some(true)` if anomalous, `Some(false)` if normal.
    pub fn update(&mut self, x: f64) -> Option<bool> {
        // Evict oldest if window is full
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(x);

        if self.window.len() < 3 {
            return None;
        }
        let z = self.score(x);
        Some(z > self.threshold)
    }

    /// Reset the window.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    /// Current window length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// True if the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

// ─── StreamMethod / StreamingResult ──────────────────────────────────────────

/// Method selection for [`StreamingThresholdDetector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMethod {
    /// Welford online z-score.
    Zscore,
    /// EWMA z-score.
    EwmaZ,
    /// Sliding-window MAD.
    SlidingMad,
}

/// Result returned by each `update` call on [`StreamingThresholdDetector`].
#[derive(Debug, Clone, Copy)]
pub struct StreamingResult {
    /// Anomaly score for the most recent observation.
    pub score: f64,
    /// Whether the observation is classified as anomalous.
    pub is_anomaly: bool,
    /// Total number of observations processed so far.
    pub n_processed: u64,
}

// ─── Internal state union ─────────────────────────────────────────────────────

enum StreamState {
    Zscore(OnlineZScore),
    EwmaZ(ExponentialZ),
    Mad(SlidingMad),
}

// ─── StreamingThresholdDetector ───────────────────────────────────────────────

/// Unified streaming anomaly detector that dispatches to one of three backends.
///
/// # Example
/// ```
/// use oxicuda_anomaly::statistical::online_stats::{
///     StreamingThresholdDetector, StreamMethod,
/// };
/// let mut det = StreamingThresholdDetector::with_zscore(3.0);
/// for x in &[1.0_f64, 1.1, 0.9, 1.05, 50.0] {
///     if let Some(r) = det.update(*x) {
///         println!("x={x:.2} score={:.3} anomaly={}", r.score, r.is_anomaly);
///     }
/// }
/// ```
pub struct StreamingThresholdDetector {
    /// Selected streaming method.
    pub method: StreamMethod,
    state: StreamState,
    /// Total observations processed.
    n_processed: u64,
}

impl StreamingThresholdDetector {
    /// Create with Welford z-score backend.
    #[must_use]
    pub fn with_zscore(threshold: f64) -> Self {
        Self {
            method: StreamMethod::Zscore,
            state: StreamState::Zscore(OnlineZScore::new(threshold)),
            n_processed: 0,
        }
    }

    /// Create with EWMA z-score backend.
    #[must_use]
    pub fn with_ewma_z(alpha: f64, threshold: f64) -> Self {
        Self {
            method: StreamMethod::EwmaZ,
            state: StreamState::EwmaZ(ExponentialZ::new(alpha, threshold)),
            n_processed: 0,
        }
    }

    /// Create with sliding MAD backend.
    #[must_use]
    pub fn with_sliding_mad(window_size: usize, mad_scale: f64, threshold: f64) -> Self {
        Self {
            method: StreamMethod::SlidingMad,
            state: StreamState::Mad(SlidingMad::new(window_size, mad_scale, threshold)),
            n_processed: 0,
        }
    }

    /// Push a new observation and get a streaming result.
    ///
    /// Returns `None` during the warm-up phase (not enough data to score).
    pub fn update(&mut self, x: f64) -> Option<StreamingResult> {
        self.n_processed += 1;
        let n_processed = self.n_processed;

        match &mut self.state {
            StreamState::Zscore(oz) => {
                let score = oz.score(x);
                let decision = oz.update(x)?;
                Some(StreamingResult {
                    score,
                    is_anomaly: decision,
                    n_processed,
                })
            }
            StreamState::EwmaZ(ez) => {
                let score = ez.score(x);
                let decision = ez.update(x)?;
                Some(StreamingResult {
                    score,
                    is_anomaly: decision,
                    n_processed,
                })
            }
            StreamState::Mad(sm) => {
                let score = sm.score(x);
                let decision = sm.update(x)?;
                Some(StreamingResult {
                    score,
                    is_anomaly: decision,
                    n_processed,
                })
            }
        }
    }

    /// Reset the detector to its initial (unfitted) state.
    pub fn reset(&mut self) {
        self.n_processed = 0;
        match &mut self.state {
            StreamState::Zscore(oz) => oz.reset(),
            StreamState::EwmaZ(ez) => ez.reset(),
            StreamState::Mad(sm) => sm.reset(),
        }
    }

    /// Total number of observations processed so far.
    #[must_use]
    pub fn n_processed(&self) -> u64 {
        self.n_processed
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OnlineZScore ──────────────────────────────────────────────────────────

    #[test]
    fn online_zscore_warmup_returns_none() {
        let mut oz = OnlineZScore::new(3.0);
        assert!(oz.update(1.0).is_none());
        assert!(oz.update(2.0).is_none());
        // Third observation should return Some(...)
        assert!(oz.update(1.5).is_some());
    }

    #[test]
    fn online_zscore_extreme_outlier_detected() {
        let mut oz = OnlineZScore::new(3.0);
        for v in &[1.0_f64, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98] {
            oz.update(*v);
        }
        // 1000.0 should be far outside the distribution
        let r = oz.update(1000.0);
        assert_eq!(r, Some(true), "1000.0 should be an anomaly");
    }

    #[test]
    fn online_zscore_inlier_not_anomaly() {
        let mut oz = OnlineZScore::new(5.0);
        for v in &[10.0_f64, 10.1, 9.9, 10.05, 9.95, 10.02, 9.98] {
            oz.update(*v);
        }
        // A value within the cluster should not be anomalous
        let r = oz.update(10.0);
        assert_eq!(r, Some(false), "inlier should not be anomaly");
    }

    #[test]
    fn online_zscore_score_before_update() {
        let mut oz = OnlineZScore::new(3.0);
        oz.update(5.0);
        oz.update(10.0);
        // Score computed with current mean/var
        let z = oz.score(5.0);
        assert!(z.is_finite(), "z={z}");
    }

    #[test]
    fn online_zscore_window_z_score_update() {
        let mut oz = OnlineZScore::new(3.0);
        let z = oz.window_z_score_update(1.0);
        assert_eq!(z, 0.0, "first call should return 0.0 (n<2)");
        oz.window_z_score_update(2.0);
        let z2 = oz.window_z_score_update(1.5);
        assert!(z2.is_finite(), "z2={z2}");
    }

    #[test]
    fn online_zscore_reset() {
        let mut oz = OnlineZScore::new(3.0);
        for v in 0..10 {
            oz.update(v as f64);
        }
        oz.reset();
        assert_eq!(oz.n, 0);
        assert_eq!(oz.mean, 0.0);
        assert_eq!(oz.m2, 0.0);
    }

    // ── ExponentialZ ─────────────────────────────────────────────────────────

    #[test]
    fn ewma_z_first_observation_returns_none() {
        let mut ez = ExponentialZ::new(0.05, 3.0);
        assert!(ez.update(1.0).is_none());
    }

    #[test]
    fn ewma_z_outlier_detected() {
        let mut ez = ExponentialZ::new(0.1, 3.0);
        for v in &[1.0_f64, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01] {
            ez.update(*v);
        }
        let r = ez.update(500.0);
        assert_eq!(r, Some(true), "500.0 should be anomaly for EWMA Z");
    }

    #[test]
    fn ewma_z_score_finite() {
        let mut ez = ExponentialZ::new(0.05, 3.0);
        ez.update(1.0);
        ez.update(2.0);
        let s = ez.score(1.5);
        assert!(s.is_finite(), "score={s}");
    }

    #[test]
    fn ewma_z_reset_clears_state() {
        let mut ez = ExponentialZ::new(0.1, 3.0);
        for v in 0..20 {
            ez.update(v as f64);
        }
        ez.reset();
        assert_eq!(ez.n, 0);
        assert!(ez.update(1.0).is_none());
    }

    // ── SlidingMad ────────────────────────────────────────────────────────────

    #[test]
    fn sliding_mad_warmup_returns_none() {
        let mut sm = SlidingMad::new(10, 1.4826, 3.0);
        assert!(sm.update(1.0).is_none());
        assert!(sm.update(2.0).is_none());
        assert!(sm.update(1.5).is_some());
    }

    #[test]
    fn sliding_mad_outlier_detected() {
        let mut sm = SlidingMad::new(20, 1.4826, 3.0);
        for v in 0..15 {
            sm.update(v as f64 * 0.1);
        }
        let r = sm.update(1000.0);
        assert_eq!(r, Some(true), "1000.0 should be anomaly");
    }

    #[test]
    fn sliding_mad_window_evicts_old_values() {
        let mut sm = SlidingMad::new(5, 1.4826, 3.0);
        for v in 0..10 {
            sm.update(v as f64);
        }
        assert_eq!(sm.len(), 5, "window must have exactly window_size elements");
    }

    #[test]
    fn sliding_mad_score_finite() {
        let mut sm = SlidingMad::new(10, 1.4826, 3.0);
        for v in 0..5 {
            sm.update(v as f64);
        }
        let s = sm.score(2.5);
        assert!(s.is_finite(), "score={s}");
    }

    #[test]
    fn sliding_mad_is_empty_after_reset() {
        let mut sm = SlidingMad::new(10, 1.4826, 3.0);
        sm.update(1.0);
        sm.reset();
        assert!(sm.is_empty());
    }

    // ── StreamingThresholdDetector ────────────────────────────────────────────

    #[test]
    fn streaming_zscore_method() {
        let mut det = StreamingThresholdDetector::with_zscore(3.0);
        let values = [1.0_f64, 1.1, 0.9, 1.05, 0.95, 500.0];
        let mut results = Vec::new();
        for &v in &values {
            if let Some(r) = det.update(v) {
                results.push(r);
            }
        }
        // At least the last result should indicate an anomaly
        assert!(
            results.last().is_some_and(|r| r.is_anomaly),
            "500.0 should be anomaly"
        );
        assert_eq!(det.n_processed(), values.len() as u64);
    }

    #[test]
    fn streaming_ewma_method() {
        let mut det = StreamingThresholdDetector::with_ewma_z(0.1, 3.0);
        assert!(det.update(1.0).is_none());
        for v in 1..10 {
            det.update(v as f64 * 0.1);
        }
        let r = det.update(1000.0);
        assert!(
            r.is_some_and(|r| r.is_anomaly),
            "1000.0 should be anomaly for EWMA"
        );
    }

    #[test]
    fn streaming_sliding_mad_method() {
        let mut det = StreamingThresholdDetector::with_sliding_mad(15, 1.4826, 3.0);
        for v in 0..10 {
            det.update(v as f64 * 0.1);
        }
        let r = det.update(9999.0);
        assert!(r.is_some_and(|r| r.is_anomaly), "9999.0 should be anomaly");
    }

    #[test]
    fn streaming_detector_reset() {
        let mut det = StreamingThresholdDetector::with_zscore(3.0);
        for v in 0..20 {
            det.update(v as f64);
        }
        det.reset();
        assert_eq!(det.n_processed(), 0);
        // After reset, warm-up should return None again
        assert!(det.update(1.0).is_none());
    }

    #[test]
    fn streaming_result_n_processed_increments() {
        let mut det = StreamingThresholdDetector::with_zscore(3.0);
        for i in 1_u64..=10 {
            det.update(i as f64);
            assert_eq!(det.n_processed(), i);
        }
    }
}
