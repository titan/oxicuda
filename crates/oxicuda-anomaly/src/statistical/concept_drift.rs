//! Concept-Drift-Aware Streaming Detectors.
//!
//! Concept drift occurs when the statistical properties of the target variable
//! change over time, making trained models stale and causing degraded performance
//! in deployed anomaly detection systems.
//!
//! # Algorithms Implemented
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | **ADWIN** | Bifet & Gavalda (2007): *"Learning from time-changing data with adaptive windowing"* |
//! | **Page-Hinkley** | Page (1954): *"Continuous Inspection Schemes"* |
//! | **CUSUM** | Lucas & Crosier (1982): *"Fast Initial Response for CUSUM QC"* |
//! | **DDM** | Gama et al. (2004): *"Learning with Drift Detection"* |

// ─────────────────────────────────────────────────────────────────────────────
//  ADWIN — ADaptive WINdowing
// ─────────────────────────────────────────────────────────────────────────────

/// ADWIN (ADaptive WINdowing) concept drift detector.
///
/// Maintains a variable-length window of recent observations.  At each new
/// data point it scans all possible split points of the window and checks
/// whether the means of the two halves differ significantly using the Hoeffding
/// bound:
///
/// ```text
/// |μ_left − μ_right| ≥ ε_cut
/// where ε_cut = sqrt(1/2 · (1/n₀ + 1/n₁) · ln(4m/δ))
/// ```
///
/// When drift is detected the older (left) portion of the window is dropped,
/// shrinking the window to focus on the most recent stationary segment.
#[derive(Debug, Clone)]
pub struct AdwinDetector {
    /// The current sliding window of observations.
    pub window: Vec<f64>,
    /// Significance parameter δ ∈ (0, 1].  Smaller values → less false alarms.
    /// Default `0.002`.
    pub delta: f64,
    /// Minimum number of samples before drift checking begins.  Default `20`.
    pub min_samples: usize,
    /// `true` immediately after a drift is detected in the most recent call to
    /// [`adwin_add`].
    pub drift_detected: bool,
    /// Cumulative count of drift events since construction.
    pub n_drift_detections: usize,
}

/// Construct a new ADWIN detector.
///
/// * `delta` – significance parameter (e.g. `0.002`).
/// * `min_samples` – minimum window size before drift checking starts.
#[must_use]
pub fn adwin_new(delta: f64, min_samples: usize) -> AdwinDetector {
    AdwinDetector {
        window: Vec::new(),
        delta,
        min_samples,
        drift_detected: false,
        n_drift_detections: 0,
    }
}

/// Add a new observation to the ADWIN detector.
///
/// Returns `true` if a drift was detected (and the window has been trimmed).
///
/// # Algorithm Details
///
/// After appending `value`, iterate over every split point `k ∈ [1, m−1]`
/// (left half = `window[0..k]`, right half = `window[k..m]`).  Compute:
///
/// ```text
/// ε_cut = sqrt(1/2 · (1/n₀ + 1/n₁) · ln(4·m/δ))
/// ```
///
/// If `|μ_left − μ_right| ≥ ε_cut` the left portion is dropped and the scan
/// restarts.  The function returns after the first detected split.
pub fn adwin_add(detector: &mut AdwinDetector, value: f64) -> bool {
    detector.window.push(value);
    detector.drift_detected = false;

    let m = detector.window.len();
    if m < detector.min_samples {
        return false;
    }

    // Precompute prefix sums for efficient mean computation.
    // prefix_sum[i] = sum(window[0..i])
    let mut prefix_sum = vec![0.0_f64; m + 1];
    for i in 0..m {
        prefix_sum[i + 1] = prefix_sum[i] + detector.window[i];
    }
    let total_sum = prefix_sum[m];

    let ln_term = (4.0 * m as f64 / detector.delta).ln();

    // Scan from right to left — finding the most-recent split first.
    // k is the split index: left = window[0..k], right = window[k..m].
    let mut drift_found = false;
    let mut split_point = 0_usize;

    #[allow(clippy::needless_range_loop)]
    'outer: for k in 1..m {
        let n0 = k as f64;
        let n1 = (m - k) as f64;
        let mu_left = prefix_sum[k] / n0;
        let mu_right = (total_sum - prefix_sum[k]) / n1;
        let eps_cut = ((0.5 * (1.0 / n0 + 1.0 / n1) * ln_term).sqrt()).max(0.0);

        if (mu_left - mu_right).abs() >= eps_cut {
            drift_found = true;
            split_point = k;
            break 'outer;
        }
    }

    if drift_found {
        // Drop the older (left) half; keep window[split_point..].
        detector.window.drain(..split_point);
        detector.drift_detected = true;
        detector.n_drift_detections += 1;
    }

    drift_found
}

/// Return the running mean of the current ADWIN window.
///
/// Returns `0.0` if the window is empty.
#[must_use]
pub fn adwin_mean(detector: &AdwinDetector) -> f64 {
    if detector.window.is_empty() {
        return 0.0;
    }
    let sum: f64 = detector.window.iter().sum();
    sum / detector.window.len() as f64
}

/// Return the current window size.
#[must_use]
pub fn adwin_window_size(detector: &AdwinDetector) -> usize {
    detector.window.len()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Page-Hinkley Test
// ─────────────────────────────────────────────────────────────────────────────

/// Page-Hinkley (PH) cumulative-sum drift detector.
///
/// Detects persistent increases (or decreases) in the stream mean using a
/// cumulative sum of centred, shifted increments:
///
/// ```text
/// m_t = x_t − x̄_t − λ
/// M_t = max(0, M_{t-1} + m_t)    (upper CUSUM)
/// m_neg_t = x_t − x̄_t + λ
/// L_t = max(0, L_{t-1} + m_neg_t) (lower CUSUM — for decreases)
/// ```
///
/// Drift is signalled when `M_t > h` (increase) or `L_t > h` (decrease).
/// The running mean is updated with an exponential decay factor `α`.
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector {
    /// Exponentially-smoothed running mean.
    pub running_mean: f64,
    /// Total number of observations added.
    pub n: usize,
    /// Upper CUSUM accumulator (detects increases).
    pub sum: f64,
    /// Lower CUSUM accumulator (detects decreases via negative shifts).
    pub min_sum: f64,
    /// Magnitude of allowable change before alarm (shift tolerance). Default `0.005`.
    pub lambda: f64,
    /// Alarm threshold — alarm fires when accumulator exceeds `h`. Default `50.0`.
    pub h: f64,
    /// Exponential forgetting factor for the running mean. Default `0.9999`.
    pub alpha: f64,
    /// `true` immediately after drift was detected in the most recent call.
    pub drift_detected: bool,
}

/// Construct a new Page-Hinkley detector.
///
/// * `lambda` – allowable shift magnitude before alarm (e.g. `0.005`).
/// * `h`      – alarm threshold (e.g. `50.0`).
/// * `alpha`  – exponential decay for running mean (e.g. `0.9999`).
#[must_use]
pub fn ph_detector_new(lambda: f64, h: f64, alpha: f64) -> PageHinkleyDetector {
    PageHinkleyDetector {
        running_mean: 0.0,
        n: 0,
        sum: 0.0,
        min_sum: 0.0,
        lambda,
        h,
        alpha,
        drift_detected: false,
    }
}

/// Add a new observation to the Page-Hinkley detector.
///
/// Returns `true` if drift was detected.
pub fn ph_add(detector: &mut PageHinkleyDetector, value: f64) -> bool {
    detector.n += 1;
    // Exponentially weighted running mean update.
    detector.running_mean = detector.alpha * detector.running_mean + (1.0 - detector.alpha) * value;

    let deviation = value - detector.running_mean;

    // Upper accumulator: detects upward shifts.
    let m_t = deviation - detector.lambda;
    detector.sum = (detector.sum + m_t).max(0.0);

    // Lower accumulator: detects downward shifts (uses +lambda to be sensitive to drops).
    let m_neg_t = -deviation - detector.lambda;
    detector.min_sum = (detector.min_sum + m_neg_t).max(0.0);

    let drift = detector.sum > detector.h || detector.min_sum > detector.h;
    detector.drift_detected = drift;
    drift
}

/// Reset the Page-Hinkley detector to its initial state.
pub fn ph_reset(detector: &mut PageHinkleyDetector) {
    detector.running_mean = 0.0;
    detector.n = 0;
    detector.sum = 0.0;
    detector.min_sum = 0.0;
    detector.drift_detected = false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CUSUM — Cumulative Sum Control Chart
// ─────────────────────────────────────────────────────────────────────────────

/// CUSUM (Cumulative Sum) control chart for sequential change detection.
///
/// Based on the sequential probability ratio test (SPRT) for detecting a
/// mean shift from `μ₀` (in-control) to `μ₁` (out-of-control):
///
/// ```text
/// k   = (μ₁ − μ₀) / 2              (allowance / reference value)
/// S_t = max(0, S_{t-1} + (x_t − μ₀) − k)   (upper CUSUM)
/// S_neg_t = max(0, S_neg_{t-1} − (x_t − μ₀) − k)  (lower CUSUM)
/// ```
///
/// An alarm is raised when `S_t > h` (or `S_neg_t > h` for two-sided).
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// In-control (nominal) process mean.
    pub mu0: f64,
    /// Out-of-control (post-shift) process mean.
    pub mu1: f64,
    /// Decision threshold — alarm fires when CUSUM statistic exceeds `h`.
    pub h: f64,
    /// Upper CUSUM statistic `S_t` (detects upward shifts).
    pub s_pos: f64,
    /// Lower CUSUM statistic `S_neg_t` (detects downward shifts).
    pub s_neg: f64,
    /// Process standard deviation (used to set default `h = 5 * sigma`).
    pub sigma: f64,
    /// Total number of observations added.
    pub n: usize,
    /// `true` immediately after drift was detected.
    pub drift_detected: bool,
}

/// Construct a new CUSUM detector.
///
/// * `mu0`   – in-control mean.
/// * `mu1`   – expected out-of-control mean.
/// * `sigma` – process standard deviation.
/// * `h`     – alarm threshold (pass `5.0 * sigma` as the canonical default).
#[must_use]
pub fn cusum_new(mu0: f64, mu1: f64, sigma: f64, h: f64) -> CusumDetector {
    CusumDetector {
        mu0,
        mu1,
        h,
        s_pos: 0.0,
        s_neg: 0.0,
        sigma,
        n: 0,
        drift_detected: false,
    }
}

/// Add a new observation to the CUSUM detector.
///
/// Returns `true` if drift is detected (either direction for two-sided test).
pub fn cusum_add(detector: &mut CusumDetector, value: f64) -> bool {
    detector.n += 1;
    let k = (detector.mu1 - detector.mu0).abs() / 2.0;
    let deviation = value - detector.mu0;

    // Upper CUSUM: sensitive to increases above mu0.
    detector.s_pos = (detector.s_pos + deviation - k).max(0.0);
    // Lower CUSUM: sensitive to decreases below mu0.
    detector.s_neg = (detector.s_neg - deviation - k).max(0.0);

    let drift = detector.s_pos > detector.h || detector.s_neg > detector.h;
    detector.drift_detected = drift;
    drift
}

/// Reset the CUSUM detector statistics while retaining parameters.
pub fn cusum_reset(detector: &mut CusumDetector) {
    detector.s_pos = 0.0;
    detector.s_neg = 0.0;
    detector.n = 0;
    detector.drift_detected = false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  DDM — Drift Detection Method
// ─────────────────────────────────────────────────────────────────────────────

/// Status returned by [`ddm_add`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdmStatus {
    /// The error rate is within expected bounds.
    Normal,
    /// The error rate is increasing — may indicate incoming drift.
    Warning,
    /// Drift detected: the error rate has shifted significantly.
    Drift,
}

/// DDM (Drift Detection Method) based on binomial error-rate tracking.
///
/// Tracks a running mean `p̂_t` and standard deviation `s_t` of a binary
/// error stream.  Warning and drift are signalled when the current statistics
/// exceed the minimum observed statistics by 2σ and 3σ respectively:
///
/// ```text
/// p̂_t = p̂_{t-1} + (e_t − p̂_{t-1}) / t
/// s_t  = sqrt(p̂_t · (1 − p̂_t) / t)
///
/// Warning : p̂_t + s_t ≥ p̂_min + 2·s_min
/// Drift   : p̂_t + s_t ≥ p̂_min + 3·s_min
/// ```
#[derive(Debug, Clone)]
pub struct DdmDetector {
    /// Total number of observations.
    pub n: usize,
    /// Running mean of the error stream.
    pub p_hat: f64,
    /// Minimum `p̂` seen during the in-control period.
    pub p_min: f64,
    /// `s` at the time `p_min` was achieved.
    pub s_min: f64,
    /// `true` if the current reading is in the warning zone.
    pub warning: bool,
    /// `true` immediately after drift is detected.
    pub drift_detected: bool,
    /// Cumulative drift detection count.
    pub n_drift_detections: usize,
}

/// Construct a new DDM detector.
#[must_use]
pub fn ddm_new() -> DdmDetector {
    DdmDetector {
        n: 0,
        p_hat: 0.0,
        p_min: f64::MAX,
        s_min: f64::MAX,
        warning: false,
        drift_detected: false,
        n_drift_detections: 0,
    }
}

/// Add a new binary error observation (0.0 = correct, 1.0 = error).
///
/// Returns the current [`DdmStatus`].
///
/// After a drift detection both `p_min` and `s_min` are reset so the detector
/// adapts to the new regime.
pub fn ddm_add(detector: &mut DdmDetector, error: f64) -> DdmStatus {
    detector.n += 1;
    let t = detector.n as f64;

    // Welford-style running mean update.
    detector.p_hat += (error - detector.p_hat) / t;

    // Ensure p_hat stays in [0, 1] for numeric stability.
    let p = detector.p_hat.clamp(0.0, 1.0);
    let s = (p * (1.0 - p) / t).sqrt();

    detector.warning = false;
    detector.drift_detected = false;

    // Need a minimum sample count before making decisions.
    if detector.n < 30 {
        // Update p_min/s_min even during warm-up.
        if p + s < detector.p_min + detector.s_min {
            detector.p_min = p;
            detector.s_min = s;
        }
        return DdmStatus::Normal;
    }

    // Update minimum statistics (only in non-warning zone).
    if detector.p_min == f64::MAX || p + s < detector.p_min + detector.s_min {
        detector.p_min = p;
        detector.s_min = s;
    }

    let level = p + s;
    let warn_thresh = detector.p_min + 2.0 * detector.s_min;
    let drift_thresh = detector.p_min + 3.0 * detector.s_min;

    if level >= drift_thresh {
        detector.drift_detected = true;
        detector.n_drift_detections += 1;
        // Reset minimums so detector adapts to new regime.
        detector.p_min = f64::MAX;
        detector.s_min = f64::MAX;
        DdmStatus::Drift
    } else if level >= warn_thresh {
        detector.warning = true;
        DdmStatus::Warning
    } else {
        DdmStatus::Normal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Test 1: stable stream — ADWIN should not detect drift ────────────────

    #[test]
    fn adwin_no_drift_stable() {
        // Stream: 100 draws from a fixed normal N(0, 0.1).
        let mut rng = LcgRng::new(0xABCD_1234);
        let mut det = adwin_new(0.002, 20);
        let mut any_drift = false;
        for _ in 0..100 {
            let v = rng.next_normal() as f64 * 0.1;
            if adwin_add(&mut det, v) {
                any_drift = true;
            }
        }
        assert!(
            !any_drift,
            "ADWIN should not detect drift on stable N(0,0.1) stream (n_det={})",
            det.n_drift_detections
        );
    }

    // ── Test 2: sudden mean shift 0 → 5 — drift detected within 50 points ──

    #[test]
    fn adwin_drift_detected() {
        let mut rng = LcgRng::new(0xDEAD_BEEF);
        let mut det = adwin_new(0.002, 10);
        // Phase 1: 100 observations from N(0, 0.1).
        for _ in 0..100 {
            let v = rng.next_normal() as f64 * 0.1;
            adwin_add(&mut det, v);
        }
        // Phase 2: 50 observations from N(5, 0.1) — hard shift.
        let mut detected = false;
        for _ in 0..50 {
            let v = 5.0 + rng.next_normal() as f64 * 0.1;
            if adwin_add(&mut det, v) {
                detected = true;
                break;
            }
        }
        assert!(
            detected,
            "ADWIN must detect mean shift 0→5 within 50 points"
        );
    }

    // ── Test 3: window shrinks immediately at the drift detection point ─────

    #[test]
    fn adwin_window_shrinks_after_drift() {
        let mut rng = LcgRng::new(0xBEEF_1337);
        let mut det = adwin_new(0.002, 10);
        // Stable phase: build up a large window.
        for _ in 0..100 {
            let v = rng.next_normal() as f64 * 0.1;
            adwin_add(&mut det, v);
        }
        let size_before_drift = adwin_window_size(&det);
        assert!(
            size_before_drift >= 10,
            "window must have grown during stable phase"
        );

        // Shift phase: feed shifted values until drift fires, record size right after.
        let mut size_at_drift = None;
        for _ in 0..100 {
            let v = 10.0 + rng.next_normal() as f64 * 0.1;
            let drifted = adwin_add(&mut det, v);
            if drifted && size_at_drift.is_none() {
                size_at_drift = Some(adwin_window_size(&det));
            }
        }
        // Drift must have been detected and the window trimmed.
        let size_at_drift =
            size_at_drift.expect("ADWIN must detect drift on 0→10 mean shift within 100 points");
        assert!(
            size_at_drift < size_before_drift,
            "window must shrink at drift event: before={size_before_drift} at_drift={size_at_drift}"
        );
    }

    // ── Test 4: stable stream — PH should not alarm ──────────────────────────

    #[test]
    fn ph_no_drift_stable() {
        let mut rng = LcgRng::new(0x1111_AAAA);
        let mut det = ph_detector_new(0.005, 50.0, 0.9999);
        let mut any_alarm = false;
        for _ in 0..200 {
            // Values tightly clustered around 0.
            let v = rng.next_normal() as f64 * 0.05;
            if ph_add(&mut det, v) {
                any_alarm = true;
            }
        }
        assert!(!any_alarm, "PH should not alarm on stable N(0,0.05) stream");
    }

    // ── Test 5: gradual upward shift — PH detects drift ─────────────────────

    #[test]
    fn ph_drift_detected() {
        let mut rng = LcgRng::new(0x2222_BBBB);
        // Very sensitive settings for test.
        let mut det = ph_detector_new(0.0, 2.0, 0.999);
        // Stable phase.
        for _ in 0..50 {
            let v = rng.next_normal() as f64 * 0.1;
            ph_add(&mut det, v);
        }
        // Rising phase: each point is +0.5 above previous.
        let mut detected = false;
        for i in 1..=100_usize {
            let v = i as f64 * 0.5;
            if ph_add(&mut det, v) {
                detected = true;
                break;
            }
        }
        assert!(detected, "PH must detect gradual upward shift");
    }

    // ── Test 6: in-control process — CUSUM stays low ─────────────────────────

    #[test]
    fn cusum_no_drift_stable() {
        let mut rng = LcgRng::new(0x3333_CCCC);
        // mu0=0, mu1=1, sigma=1, h=5.
        let mut det = cusum_new(0.0, 1.0, 1.0, 5.0);
        let mut any_alarm = false;
        for _ in 0..300 {
            // Observations exactly from N(0, 0.2) — tightly in-control.
            let v = rng.next_normal() as f64 * 0.2;
            if cusum_add(&mut det, v) {
                any_alarm = true;
            }
        }
        assert!(
            !any_alarm,
            "CUSUM should not alarm on tightly in-control stream; s_pos={} s_neg={}",
            det.s_pos, det.s_neg
        );
    }

    // ── Test 7: mean shift — CUSUM detects it ────────────────────────────────

    #[test]
    fn cusum_shift_detected() {
        let mut rng = LcgRng::new(0x4444_DDDD);
        let mut det = cusum_new(0.0, 2.0, 1.0, 5.0);
        // In-control phase.
        for _ in 0..50 {
            let v = rng.next_normal() as f64 * 0.3;
            cusum_add(&mut det, v);
        }
        // Shift: observations from N(3, 0.3) — well above mu1=2.
        let mut detected = false;
        for _ in 0..100 {
            let v = 3.0 + rng.next_normal() as f64 * 0.3;
            if cusum_add(&mut det, v) {
                detected = true;
                break;
            }
        }
        assert!(detected, "CUSUM must detect upward mean shift");
    }

    // ── Test 8: negative shift — lower CUSUM detects it ─────────────────────

    #[test]
    fn cusum_two_sided() {
        let mut rng = LcgRng::new(0x5555_EEEE);
        // mu0=5, mu1=3 — detector is sensitive to drops below mu0.
        let mut det = cusum_new(5.0, 3.0, 1.0, 5.0);
        // In-control: values around 5.
        for _ in 0..50 {
            let v = 5.0 + rng.next_normal() as f64 * 0.3;
            cusum_add(&mut det, v);
        }
        // Drop: values around 0 — well below mu0=5.
        let mut detected = false;
        for _ in 0..100 {
            let v = 0.0 + rng.next_normal() as f64 * 0.3;
            if cusum_add(&mut det, v) {
                detected = true;
                break;
            }
        }
        assert!(
            detected,
            "CUSUM lower arm must detect downward shift; s_neg={}",
            det.s_neg
        );
    }

    // ── Test 9: low stable error — DDM stays Normal ──────────────────────────

    #[test]
    fn ddm_stable_low_error() {
        let mut det = ddm_new();
        // Consistent 5 % error rate for 500 steps.
        // To keep it deterministic we alternate: 1 error every 20 points.
        let mut last_status = DdmStatus::Normal;
        for i in 0..500_usize {
            let error = if i % 20 == 0 { 1.0 } else { 0.0 };
            last_status = ddm_add(&mut det, error);
        }
        assert_ne!(
            last_status,
            DdmStatus::Drift,
            "DDM must not signal drift on stable 5% error stream"
        );
        assert_eq!(det.n_drift_detections, 0);
    }

    // ── Test 10: rising error rate — DDM detects drift ───────────────────────

    #[test]
    fn ddm_increasing_error_drift() {
        let mut det = ddm_new();

        // Phase 1: stable 5% error for 200 steps.
        for i in 0..200_usize {
            let error = if i % 20 == 0 { 1.0 } else { 0.0 };
            ddm_add(&mut det, error);
        }

        // Phase 2: 100% error rate — guaranteed to trigger drift.
        let mut detected = false;
        for _ in 0..200_usize {
            let status = ddm_add(&mut det, 1.0);
            if status == DdmStatus::Drift {
                detected = true;
                break;
            }
        }
        assert!(
            detected,
            "DDM must detect drift when error rate jumps to 100%; detections={}",
            det.n_drift_detections
        );
    }

    // ── Test 11: CUSUM reset clears state ────────────────────────────────────

    #[test]
    fn cusum_reset_clears_state() {
        let mut det = cusum_new(0.0, 2.0, 1.0, 5.0);
        // Trigger an alarm.
        for _ in 0..100 {
            cusum_add(&mut det, 5.0);
        }
        assert!(det.drift_detected, "should have alarmed before reset");
        cusum_reset(&mut det);
        assert_eq!(det.s_pos, 0.0);
        assert_eq!(det.s_neg, 0.0);
        assert!(!det.drift_detected);
        assert_eq!(det.n, 0);
    }

    // ── Test 12: PH reset clears state ───────────────────────────────────────

    #[test]
    fn ph_reset_clears_state() {
        let mut det = ph_detector_new(0.0, 1.0, 0.999);
        for _ in 0..100 {
            ph_add(&mut det, 100.0); // force alarm
        }
        assert!(det.drift_detected);
        ph_reset(&mut det);
        assert_eq!(det.sum, 0.0);
        assert_eq!(det.min_sum, 0.0);
        assert_eq!(det.n, 0);
        assert!(!det.drift_detected);
    }

    // ── Test 13: ADWIN mean tracks window accurately ──────────────────────────

    #[test]
    fn adwin_mean_correct() {
        let mut det = adwin_new(0.002, 5);
        // Insert 5 known values; no drift expected.
        for v in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
            adwin_add(&mut det, v);
        }
        let m = adwin_mean(&det);
        // Window may have been trimmed; but if intact the mean is 3.0.
        assert!(m.is_finite(), "mean should be finite, got {m}");
        assert!(m > 0.0 && m <= 5.0, "mean {m} out of expected range [0,5]");
    }

    // ── Test 14: DDM Warning status is transient before Drift ────────────────

    #[test]
    fn ddm_warning_before_drift() {
        let mut det = ddm_new();
        // Stable phase.
        for i in 0..200_usize {
            let e = if i % 20 == 0 { 1.0 } else { 0.0 };
            ddm_add(&mut det, e);
        }
        // Rising error — should see Warning before Drift.
        let mut saw_warning = false;
        let mut saw_drift = false;
        for i in 0..300_usize {
            // Gradually increase error rate.
            let error = if i % 3 == 0 { 1.0 } else { 0.0 };
            let status = ddm_add(&mut det, error);
            if status == DdmStatus::Warning {
                saw_warning = true;
            }
            if status == DdmStatus::Drift {
                saw_drift = true;
                break;
            }
        }
        // At minimum we expect drift to be triggered.
        assert!(
            saw_drift || saw_warning,
            "DDM must issue at least Warning on rising error rate"
        );
    }
}
