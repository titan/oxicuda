//! Calibration metrics and post-hoc temperature scaling.
//!
//! Implements reliability binning, Expected / Maximum Calibration Error
//! (ECE / MCE), the multi-class Brier score, and Guo et al. temperature
//! scaling for confidence recalibration.
//!
//! References:
//! - Naeini, Cooper & Hauskrecht (2015), "Obtaining Well Calibrated
//!   Probabilities Using Bayesian Binning", *AAAI* (ECE / reliability bins).
//! - Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural
//!   Networks", *ICML* (temperature scaling, ECE / MCE definitions).
//! - Brier (1950), "Verification of Forecasts Expressed in Terms of
//!   Probability" (Brier score).

use crate::error::{TabularError, TabularResult};

/// Strategy for partitioning the confidence range `[0, 1]` into bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinningScheme {
    /// Uniform-width bins covering `[0, 1]`.
    #[default]
    EqualWidth,
    /// Quantile (equal-mass) bins, each containing ≈ `N / n_bins` samples.
    EqualMass,
}

/// Configuration for reliability binning and the ECE / MCE metrics.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationConfig {
    /// Number of confidence bins.
    pub n_bins: usize,
    /// How bin edges are chosen.
    pub binning: BinningScheme,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            n_bins: 15,
            binning: BinningScheme::EqualWidth,
        }
    }
}

/// A single reliability-diagram bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBin {
    /// Lower edge of the confidence interval (inclusive).
    pub lo: f32,
    /// Upper edge of the confidence interval (inclusive for the final bin).
    pub hi: f32,
    /// Number of samples falling into this bin.
    pub count: usize,
    /// Mean predicted confidence of samples in the bin.
    pub avg_confidence: f32,
    /// Empirical accuracy of samples in the bin.
    pub accuracy: f32,
}

/// Validate the `(confidences, predictions, labels)` triple and `cfg`.
fn check_inputs(
    confidences: &[f32],
    predictions: &[u32],
    labels: &[u32],
    cfg: &CalibrationConfig,
) -> TabularResult<()> {
    if cfg.n_bins == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_bins".into(),
            msg: "must be >= 1".into(),
        });
    }
    if confidences.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if confidences.len() != labels.len() {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len(),
            got: confidences.len(),
        });
    }
    if predictions.len() != labels.len() {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len(),
            got: predictions.len(),
        });
    }
    Ok(())
}

/// Build reliability bins for the given confidences / predictions / labels.
///
/// `confidences[i]` is the model's confidence (typically the max softmax
/// probability) for sample `i`, `predictions[i]` the predicted class, and
/// `labels[i]` the true class. A sample is *correct* when
/// `predictions[i] == labels[i]`.
///
/// For [`BinningScheme::EqualWidth`] the edges are `k / n_bins`. For
/// [`BinningScheme::EqualMass`] the edges are the empirical quantiles of the
/// confidences so each bin holds ≈ `N / n_bins` samples. Returned bins
/// partition `[0, 1]` and their counts sum to `N`.
///
/// # Errors
/// Errors on `n_bins == 0`, empty input, or length mismatch.
pub fn reliability_bins(
    confidences: &[f32],
    predictions: &[u32],
    labels: &[u32],
    cfg: CalibrationConfig,
) -> TabularResult<Vec<ReliabilityBin>> {
    check_inputs(confidences, predictions, labels, &cfg)?;
    let n = confidences.len();
    let edges = bin_edges(confidences, &cfg)?;

    let mut counts = vec![0usize; cfg.n_bins];
    let mut conf_sum = vec![0.0_f32; cfg.n_bins];
    let mut correct = vec![0usize; cfg.n_bins];

    for i in 0..n {
        let c = confidences.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let bin = locate_bin(c, &edges);
        if let (Some(cnt), Some(cs)) = (counts.get_mut(bin), conf_sum.get_mut(bin)) {
            *cnt += 1;
            *cs += c;
            let pred = predictions.get(i).copied().unwrap_or(u32::MAX);
            let label = labels.get(i).copied().unwrap_or(u32::MAX);
            if pred == label
                && let Some(corr) = correct.get_mut(bin)
            {
                *corr += 1;
            }
        }
    }

    let mut bins = Vec::with_capacity(cfg.n_bins);
    for b in 0..cfg.n_bins {
        let lo = edges.get(b).copied().unwrap_or(0.0);
        let hi = edges.get(b + 1).copied().unwrap_or(1.0);
        let count = counts.get(b).copied().unwrap_or(0);
        let (avg_confidence, accuracy) = if count == 0 {
            (0.0, 0.0)
        } else {
            let cs = conf_sum.get(b).copied().unwrap_or(0.0);
            let corr = correct.get(b).copied().unwrap_or(0);
            (cs / count as f32, corr as f32 / count as f32)
        };
        bins.push(ReliabilityBin {
            lo,
            hi,
            count,
            avg_confidence,
            accuracy,
        });
    }
    Ok(bins)
}

/// Compute `n_bins + 1` ascending bin edges spanning `[0, 1]`.
fn bin_edges(confidences: &[f32], cfg: &CalibrationConfig) -> TabularResult<Vec<f32>> {
    let n_bins = cfg.n_bins;
    match cfg.binning {
        BinningScheme::EqualWidth => {
            let mut edges = Vec::with_capacity(n_bins + 1);
            for k in 0..=n_bins {
                edges.push(k as f32 / n_bins as f32);
            }
            Ok(edges)
        }
        BinningScheme::EqualMass => {
            let mut sorted: Vec<f32> = confidences.iter().map(|&c| c.clamp(0.0, 1.0)).collect();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let mut edges = Vec::with_capacity(n_bins + 1);
            edges.push(0.0_f32);
            for k in 1..n_bins {
                // Quantile position for the k-th internal edge.
                let pos = (k as f64 / n_bins as f64) * n as f64;
                let idx = (pos.floor() as usize).min(n.saturating_sub(1));
                let edge = sorted.get(idx).copied().unwrap_or(1.0);
                edges.push(edge);
            }
            edges.push(1.0_f32);
            // Enforce non-decreasing edges (ties in confidences can collapse
            // adjacent edges; clamp keeps them monotone for bin location).
            for k in 1..edges.len() {
                let prev = edges.get(k - 1).copied().unwrap_or(0.0);
                if let Some(e) = edges.get_mut(k)
                    && *e < prev
                {
                    *e = prev;
                }
            }
            Ok(edges)
        }
    }
}

/// Locate the bin index for confidence `c` given ascending `edges`.
///
/// Bins are `[edge_b, edge_{b+1})` except the final bin which is closed on the
/// right so `c == 1.0` is included.
fn locate_bin(c: f32, edges: &[f32]) -> usize {
    let n_bins = edges.len().saturating_sub(1);
    if n_bins == 0 {
        return 0;
    }
    for b in 0..n_bins {
        let hi = edges.get(b + 1).copied().unwrap_or(1.0);
        let is_last = b + 1 == n_bins;
        if c < hi || (is_last && c <= hi) {
            return b;
        }
    }
    n_bins - 1
}

/// Expected Calibration Error: `Σ_b (n_b / N)·|acc_b − conf_b|`.
///
/// # Errors
/// Errors on `n_bins == 0`, empty input, or length mismatch.
pub fn expected_calibration_error(
    confidences: &[f32],
    predictions: &[u32],
    labels: &[u32],
    cfg: CalibrationConfig,
) -> TabularResult<f32> {
    let bins = reliability_bins(confidences, predictions, labels, cfg)?;
    let n = confidences.len() as f32;
    let mut ece = 0.0_f32;
    for bin in &bins {
        if bin.count == 0 {
            continue;
        }
        ece += (bin.count as f32 / n) * (bin.accuracy - bin.avg_confidence).abs();
    }
    Ok(ece)
}

/// Maximum Calibration Error: `max_b |acc_b − conf_b|` over non-empty bins.
///
/// # Errors
/// Errors on `n_bins == 0`, empty input, or length mismatch.
pub fn maximum_calibration_error(
    confidences: &[f32],
    predictions: &[u32],
    labels: &[u32],
    cfg: CalibrationConfig,
) -> TabularResult<f32> {
    let bins = reliability_bins(confidences, predictions, labels, cfg)?;
    let mut mce = 0.0_f32;
    for bin in &bins {
        if bin.count == 0 {
            continue;
        }
        let gap = (bin.accuracy - bin.avg_confidence).abs();
        if gap > mce {
            mce = gap;
        }
    }
    Ok(mce)
}

/// Multi-class Brier score: mean over samples of `Σ_k (p_k − 1{y = k})²`.
///
/// `probs` is `n_samples × n_classes` row-major.
///
/// # Errors
/// Errors on empty input, `n_classes < 1`, shape mismatch, or a label outside
/// `[0, n_classes)`.
pub fn brier_score(probs: &[f32], labels: &[u32], n_classes: usize) -> TabularResult<f32> {
    if n_classes == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_classes".into(),
            msg: "must be >= 1".into(),
        });
    }
    if labels.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if probs.len() != labels.len() * n_classes {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: probs.len(),
        });
    }
    let mut total = 0.0_f32;
    for (i, &y) in labels.iter().enumerate() {
        let y = y as usize;
        if y >= n_classes {
            return Err(TabularError::LabelOutOfRange {
                label: y,
                n_classes,
            });
        }
        let row = probs
            .get(i * n_classes..(i + 1) * n_classes)
            .ok_or_else(|| TabularError::Internal {
                msg: "probability row out of bounds".into(),
            })?;
        let mut sample = 0.0_f32;
        for (k, &p) in row.iter().enumerate() {
            let target = if k == y { 1.0 } else { 0.0 };
            let diff = p - target;
            sample += diff * diff;
        }
        total += sample;
    }
    Ok(total / labels.len() as f32)
}

// ─── Temperature scaling ────────────────────────────────────────────────────────

/// Numerically-stable softmax of a logit row (max-subtraction).
fn softmax_row(logits: &[f32], out: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = (l - max).exp();
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    }
}

/// Post-hoc temperature scaling (Guo et al. 2017).
///
/// Learns a single positive scalar `T` that divides the logits before softmax,
/// minimising the multi-class negative log-likelihood (NLL). Along `T` the NLL
/// is smooth and unimodal, so a 1-D Newton iteration on `dNLL/dT` (with a
/// bisection safeguard and clamping to a sane positive range) converges
/// reliably. `apply` returns `softmax(logits / T)`.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureScaler {
    temperature: f32,
}

/// Minimum / maximum temperatures considered during fitting.
const T_MIN: f32 = 1e-2;
const T_MAX: f32 = 1e2;

impl TemperatureScaler {
    /// Construct directly from a known temperature `T > 0`.
    ///
    /// # Errors
    /// Errors when `temperature` is not strictly positive / finite.
    pub fn new(temperature: f32) -> TabularResult<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "temperature".into(),
                msg: format!("must be > 0, got {temperature}"),
            });
        }
        Ok(Self { temperature })
    }

    /// The learned (or supplied) temperature.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Fit `T` on validation logits (`n_samples × n_classes`, row-major) and
    /// integer labels by minimising the multi-class NLL.
    ///
    /// # Errors
    /// Errors on empty input, `n_classes < 2`, shape mismatch, or a label
    /// outside `[0, n_classes)`.
    pub fn fit(logits: &[f32], labels: &[u32], n_classes: usize) -> TabularResult<Self> {
        if n_classes < 2 {
            return Err(TabularError::InvalidParameter {
                name: "n_classes".into(),
                msg: "must be >= 2 for temperature scaling".into(),
            });
        }
        if labels.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if logits.len() != labels.len() * n_classes {
            return Err(TabularError::DimensionMismatch {
                expected: labels.len() * n_classes,
                got: logits.len(),
            });
        }
        for &y in labels {
            if y as usize >= n_classes {
                return Err(TabularError::LabelOutOfRange {
                    label: y as usize,
                    n_classes,
                });
            }
        }

        // Newton's method on g(T) = dNLL/dT with a bisection bracket safeguard.
        // We bracket a sign change of g over [T_MIN, T_MAX] (g is increasing in
        // T for over-confident models and decreasing for under-confident ones),
        // then refine with damped Newton steps, falling back to bisection.
        let g = |t: f32| nll_derivative(logits, labels, n_classes, t);

        let mut lo = T_MIN;
        let mut hi = T_MAX;
        let mut g_lo = g(lo);
        let mut g_hi = g(hi);

        // If no sign change, the optimum sits at a boundary (monotone NLL).
        if g_lo.signum() == g_hi.signum() {
            // dNLL/dT > 0 everywhere ⇒ NLL increasing ⇒ smallest T best.
            // dNLL/dT < 0 everywhere ⇒ NLL decreasing ⇒ largest T best.
            let t = if g_lo > 0.0 { T_MIN } else { T_MAX };
            return Self::new(t);
        }

        let mut t = 1.0_f32.clamp(T_MIN, T_MAX);
        for _ in 0..100 {
            let g_t = g(t);
            if g_t.abs() < 1e-7 {
                break;
            }
            // Maintain the bracket.
            if g_t.signum() == g_lo.signum() {
                lo = t;
                g_lo = g_t;
            } else {
                hi = t;
                g_hi = g_t;
            }
            // Newton step using a finite-difference second derivative.
            let h = (1e-3_f32).max(t * 1e-3);
            let g_plus = g(t + h);
            let dg = (g_plus - g_t) / h;
            let newton = if dg.abs() > 1e-12 { t - g_t / dg } else { t };
            // Accept Newton only if it stays inside the bracket; else bisect.
            t = if newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if (hi - lo).abs() < 1e-6 {
                break;
            }
        }
        let _ = g_hi; // bracket bookkeeping; final value unused directly
        Self::new(t.clamp(T_MIN, T_MAX))
    }

    /// Apply temperature scaling: `softmax(logits / T)` for one row.
    ///
    /// # Errors
    /// Errors when the stored temperature is not strictly positive (e.g. a
    /// scaler constructed via an unchecked path) or `logits` is empty.
    pub fn apply(&self, logits: &[f32]) -> TabularResult<Vec<f32>> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "temperature".into(),
                msg: format!("must be > 0, got {}", self.temperature),
            });
        }
        if logits.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let mut out = vec![0.0_f32; scaled.len()];
        softmax_row(&scaled, &mut out);
        Ok(out)
    }
}

/// Multi-class NLL of temperature-scaled logits at temperature `t`.
///
/// `NLL(T) = −(1/N) Σ_i log softmax(z_i / T)_{y_i}`. Exposed for tests.
pub fn temperature_nll(logits: &[f32], labels: &[u32], n_classes: usize, t: f32) -> f32 {
    if t <= 0.0 || n_classes == 0 || labels.is_empty() {
        return f32::INFINITY;
    }
    let mut total = 0.0_f64;
    for (i, &y) in labels.iter().enumerate() {
        let y = y as usize;
        let row = match logits.get(i * n_classes..(i + 1) * n_classes) {
            Some(r) => r,
            None => return f32::INFINITY,
        };
        let scaled: Vec<f32> = row.iter().map(|&l| l / t).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        for &s in &scaled {
            sum += ((s - max) as f64).exp();
        }
        let log_sum = max as f64 + sum.ln();
        let target = scaled.get(y).copied().unwrap_or(f32::NEG_INFINITY) as f64;
        total += log_sum - target; // −log p_{y}
    }
    (total / labels.len() as f64) as f32
}

/// Analytic derivative `dNLL/dT` for the softmax NLL under temperature scaling.
///
/// For one sample with logits `z`, scaled `s = z / T`,
/// `−log p_y = log Σ_k exp(s_k) − s_y`, and
/// `d(−log p_y)/dT = (s_y − Σ_k p_k s_k) / T` where `p = softmax(s)`. Averaged
/// over the batch this gives `dNLL/dT`.
fn nll_derivative(logits: &[f32], labels: &[u32], n_classes: usize, t: f32) -> f32 {
    if t <= 0.0 {
        return f32::INFINITY;
    }
    let mut total = 0.0_f64;
    for (i, &y) in labels.iter().enumerate() {
        let y = y as usize;
        let row = match logits.get(i * n_classes..(i + 1) * n_classes) {
            Some(r) => r,
            None => return f32::NAN,
        };
        let scaled: Vec<f32> = row.iter().map(|&l| l / t).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        for &s in &scaled {
            sum += ((s - max) as f64).exp();
        }
        // Expected scaled-logit under the softmax distribution.
        let mut expected = 0.0_f64;
        for &s in &scaled {
            let p = ((s - max) as f64).exp() / sum;
            expected += p * s as f64;
        }
        let s_y = scaled.get(y).copied().unwrap_or(0.0) as f64;
        total += (s_y - expected) / t as f64;
    }
    (total / labels.len() as f64) as f32
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn equal_width(n_bins: usize) -> CalibrationConfig {
        CalibrationConfig {
            n_bins,
            binning: BinningScheme::EqualWidth,
        }
    }

    // ── Perfectly calibrated ⇒ ECE ≈ 0 ──────────────────────────────────────
    #[test]
    fn perfectly_calibrated_zero_ece() {
        // In each bin, fraction-correct equals the confidence. Build a dataset
        // where confidence c implies an accuracy of exactly c per bin.
        // Bin centers 0.05, 0.15, ..., 0.95 with 100 samples each; make the
        // number correct ≈ confidence * count.
        let n_bins = 10;
        let mut confidences = Vec::new();
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for b in 0..n_bins {
            let conf = (b as f32 + 0.5) / n_bins as f32;
            let count = 100usize;
            let n_correct = (conf * count as f32).round() as usize;
            for j in 0..count {
                confidences.push(conf);
                // Correct sample: prediction == label.
                if j < n_correct {
                    predictions.push(1);
                    labels.push(1);
                } else {
                    predictions.push(1);
                    labels.push(0);
                }
            }
        }
        let ece =
            expected_calibration_error(&confidences, &predictions, &labels, equal_width(n_bins))
                .expect("value should be present");
        assert!(ece < 0.02, "ece={ece}");
    }

    #[test]
    fn overconfident_positive_ece() {
        // Confidence 0.95 everywhere but only 50% correct ⇒ large ECE.
        let n = 200usize;
        let confidences = vec![0.95_f32; n];
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for j in 0..n {
            predictions.push(1u32);
            labels.push(if j % 2 == 0 { 1u32 } else { 0u32 });
        }
        let ece = expected_calibration_error(&confidences, &predictions, &labels, equal_width(10))
            .expect("value should be present");
        assert!(ece > 0.3, "ece={ece}");
    }

    #[test]
    fn mce_at_least_ece() {
        let mut rng = LcgRng::new(5);
        let n = 500usize;
        let mut confidences = Vec::new();
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n {
            let c = 0.5 + 0.5 * rng.next_f32();
            confidences.push(c);
            predictions.push(1u32);
            // Correct with some probability below the confidence (over-conf).
            labels.push(if rng.next_f32() < c * 0.7 { 1u32 } else { 0u32 });
        }
        let cfg = equal_width(12);
        let ece = expected_calibration_error(&confidences, &predictions, &labels, cfg)
            .expect("expected_calibration_error should succeed");
        let mce = maximum_calibration_error(&confidences, &predictions, &labels, cfg)
            .expect("maximum_calibration_error should succeed");
        assert!(mce >= ece - 1e-6, "mce={mce} < ece={ece}");
    }

    #[test]
    fn equal_mass_bins_roughly_equal_count() {
        let mut rng = LcgRng::new(17);
        let n = 1000usize;
        let n_bins = 10usize;
        let mut confidences = Vec::new();
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n {
            confidences.push(rng.next_f32());
            predictions.push(1u32);
            labels.push(1u32);
        }
        let cfg = CalibrationConfig {
            n_bins,
            binning: BinningScheme::EqualMass,
        };
        let bins = reliability_bins(&confidences, &predictions, &labels, cfg)
            .expect("reliability_bins should succeed");
        let target = (n / n_bins) as i64;
        for bin in &bins {
            // Allow generous slack but each bin near N/n_bins.
            let diff = (bin.count as i64 - target).abs();
            assert!(
                diff < target,
                "bin count {} far from target {target}",
                bin.count
            );
        }
        let total: usize = bins.iter().map(|b| b.count).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn bins_partition_unit_interval() {
        let confidences = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
        let predictions = vec![1u32; 5];
        let labels = vec![1u32; 5];
        let cfg = equal_width(5);
        let bins = reliability_bins(&confidences, &predictions, &labels, cfg)
            .expect("reliability_bins should succeed");
        assert!((bins.first().map(|b| b.lo).unwrap_or(1.0)).abs() < 1e-6);
        assert!((bins.last().map(|b| b.hi).unwrap_or(0.0) - 1.0).abs() < 1e-6);
        // Edges contiguous: each bin.hi == next bin.lo.
        for w in bins.windows(2) {
            if let [a, b] = w {
                assert!((a.hi - b.lo).abs() < 1e-6, "gap between {a:?} and {b:?}");
            }
        }
    }

    #[test]
    fn bin_counts_sum_to_n() {
        let mut rng = LcgRng::new(123);
        let n = 333usize;
        let mut confidences = Vec::new();
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n {
            confidences.push(rng.next_f32());
            predictions.push(rng.next_usize(3) as u32);
            labels.push(rng.next_usize(3) as u32);
        }
        let bins = reliability_bins(&confidences, &predictions, &labels, equal_width(8))
            .expect("value should be present");
        let total: usize = bins.iter().map(|b| b.count).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn brier_one_hot_correct_is_zero() {
        let k = 3;
        // Three samples, each one-hot on the true class.
        let probs = vec![
            1.0_f32, 0.0, 0.0, // y = 0
            0.0, 1.0, 0.0, // y = 1
            0.0, 0.0, 1.0, // y = 2
        ];
        let labels = vec![0u32, 1, 2];
        let bs = brier_score(&probs, &labels, k).expect("brier_score should succeed");
        assert!(bs.abs() < 1e-6, "bs={bs}");
    }

    #[test]
    fn brier_uniform_closed_form() {
        // Uniform over K: each sample contributes (K-1)·(1/K)² + (1 - 1/K)²
        //              = (K-1)/K² + ((K-1)/K)² = (K-1)/K.
        let k = 4usize;
        let p = 1.0_f32 / k as f32;
        let n = 5usize;
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            for _ in 0..k {
                probs.push(p);
            }
            labels.push((i % k) as u32);
        }
        let bs = brier_score(&probs, &labels, k).expect("brier_score should succeed");
        let expected = (k as f32 - 1.0) / k as f32;
        assert!((bs - expected).abs() < 1e-5, "bs={bs}, expected={expected}");
    }

    #[test]
    fn temperature_softens_overconfident() {
        // Sharp logits ⇒ T > 1 lowers the max probability.
        let logits = vec![6.0_f32, 1.0, 0.5];
        let cold = TemperatureScaler::new(1.0).expect("new should succeed");
        let warm = TemperatureScaler::new(3.0).expect("new should succeed");
        let p_cold = cold.apply(&logits).expect("apply should succeed");
        let p_warm = warm.apply(&logits).expect("apply should succeed");
        let max_cold = p_cold.iter().cloned().fold(0.0_f32, f32::max);
        let max_warm = p_warm.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_warm < max_cold, "warm {max_warm} >= cold {max_cold}");
    }

    #[test]
    fn apply_t_one_equals_plain_softmax() {
        let logits = vec![0.5_f32, -1.0, 2.0, 0.0];
        let scaler = TemperatureScaler::new(1.0).expect("new should succeed");
        let scaled = scaler.apply(&logits).expect("apply should succeed");
        let mut plain = vec![0.0_f32; logits.len()];
        softmax_row(&logits, &mut plain);
        for (a, b) in scaled.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        // And it sums to 1.
        let s: f32 = scaled.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fitted_temperature_reduces_nll() {
        // Over-confident logits relative to noisy labels.
        let mut rng = LcgRng::new(2024);
        let k = 3usize;
        let n = 600usize;
        let mut logits = Vec::with_capacity(n * k);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let y = rng.next_usize(k);
            // Sharp logits favouring a *possibly wrong* class to induce
            // miscalibration.
            let mut row = vec![0.0_f32; k];
            for r in row.iter_mut() {
                let (a, _) = rng.next_normal_pair();
                *r = a;
            }
            // Add a large spike to the true class only 60% of the time.
            let spike_class = if rng.next_f32() < 0.6 {
                y
            } else {
                rng.next_usize(k)
            };
            if let Some(slot) = row.get_mut(spike_class) {
                *slot += 5.0;
            }
            logits.extend_from_slice(&row);
            labels.push(y as u32);
        }
        let scaler = TemperatureScaler::fit(&logits, &labels, k).expect("fit should succeed");
        let nll_fit = temperature_nll(&logits, &labels, k, scaler.temperature());
        let nll_one = temperature_nll(&logits, &labels, k, 1.0);
        assert!(
            nll_fit <= nll_one + 1e-4,
            "fitted NLL {nll_fit} > NLL@1 {nll_one}"
        );
        assert!(scaler.temperature() > 0.0);
    }

    #[test]
    fn ece_after_temperature_scaling_not_worse() {
        // Build an over-confident multi-class dataset, fit T, and verify ECE on
        // the (max-softmax) confidences does not increase.
        let mut rng = LcgRng::new(909);
        let k = 4usize;
        let n = 1200usize;
        let mut logits = Vec::with_capacity(n * k);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let y = rng.next_usize(k);
            let mut row = vec![0.0_f32; k];
            for r in row.iter_mut() {
                let (a, _) = rng.next_normal_pair();
                *r = a;
            }
            // Strong, over-confident spike that is correct only ~55% of time.
            let spike = if rng.next_f32() < 0.55 {
                y
            } else {
                rng.next_usize(k)
            };
            if let Some(slot) = row.get_mut(spike) {
                *slot += 8.0;
            }
            logits.extend_from_slice(&row);
            labels.push(y as u32);
        }

        let to_conf_pred = |t: f32| -> (Vec<f32>, Vec<u32>) {
            let scaler = TemperatureScaler::new(t).expect("new should succeed");
            let mut confs = Vec::with_capacity(n);
            let mut preds = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(row) = logits.get(i * k..(i + 1) * k) {
                    let p = scaler.apply(row).expect("apply should succeed");
                    let mut best = 0usize;
                    let mut best_p = f32::NEG_INFINITY;
                    for (j, &pj) in p.iter().enumerate() {
                        if pj > best_p {
                            best_p = pj;
                            best = j;
                        }
                    }
                    confs.push(best_p);
                    preds.push(best as u32);
                }
            }
            (confs, preds)
        };

        let scaler = TemperatureScaler::fit(&logits, &labels, k).expect("fit should succeed");
        let cfg = equal_width(15);

        let (c0, p0) = to_conf_pred(1.0);
        let ece_before = expected_calibration_error(&c0, &p0, &labels, cfg)
            .expect("expected_calibration_error should succeed");
        let (c1, p1) = to_conf_pred(scaler.temperature());
        let ece_after = expected_calibration_error(&c1, &p1, &labels, cfg)
            .expect("expected_calibration_error should succeed");

        assert!(
            ece_after <= ece_before + 0.02,
            "ece_after={ece_after} > ece_before={ece_before} (T={})",
            scaler.temperature()
        );
    }

    #[test]
    fn calibration_errs_on_bad_input() {
        let confidences = vec![0.5_f32, 0.6];
        let predictions = vec![1u32, 0];
        let labels = vec![1u32, 0];
        // n_bins == 0.
        assert!(reliability_bins(&confidences, &predictions, &labels, equal_width(0)).is_err());
        // empty input.
        assert!(reliability_bins(&[], &[], &[], equal_width(5)).is_err());
        // length mismatch.
        assert!(reliability_bins(&confidences, &[1u32], &labels, equal_width(5)).is_err());
    }

    #[test]
    fn brier_errs_on_bad_input() {
        // probs/labels length mismatch.
        assert!(brier_score(&[0.5, 0.5], &[0u32, 1], 2).is_err());
        // empty input.
        assert!(brier_score(&[], &[], 3).is_err());
        // label out of range.
        assert!(brier_score(&[0.5, 0.5], &[5u32], 2).is_err());
        // n_bins/classes == 0.
        assert!(brier_score(&[], &[0u32], 0).is_err());
    }

    #[test]
    fn temperature_apply_errs_on_nonpositive() {
        // Construct a scaler with an invalid temperature by bypassing `new`.
        let bad = TemperatureScaler { temperature: 0.0 };
        assert!(bad.apply(&[1.0, 2.0]).is_err());
        let neg = TemperatureScaler { temperature: -1.0 };
        assert!(neg.apply(&[1.0, 2.0]).is_err());
        // new() rejects T <= 0.
        assert!(TemperatureScaler::new(0.0).is_err());
        assert!(TemperatureScaler::new(-2.0).is_err());
    }

    #[test]
    fn temperature_fit_errs_on_bad_input() {
        // n_classes < 2.
        assert!(TemperatureScaler::fit(&[1.0, 2.0], &[0u32], 1).is_err());
        // empty.
        assert!(TemperatureScaler::fit(&[], &[], 3).is_err());
        // shape mismatch.
        assert!(TemperatureScaler::fit(&[1.0, 2.0, 3.0], &[0u32, 1], 2).is_err());
        // label out of range.
        assert!(TemperatureScaler::fit(&[1.0, 2.0, 3.0, 4.0], &[0u32, 5], 2).is_err());
    }

    #[test]
    fn empty_bins_have_zero_stats() {
        // All confidences in the top bin; lower bins must be empty/zeroed.
        let confidences = vec![0.99_f32, 0.98, 0.97];
        let predictions = vec![1u32, 1, 1];
        let labels = vec![1u32, 1, 1];
        let bins = reliability_bins(&confidences, &predictions, &labels, equal_width(10))
            .expect("value should be present");
        let empties: Vec<&ReliabilityBin> = bins.iter().filter(|b| b.count == 0).collect();
        assert!(!empties.is_empty());
        for b in empties {
            assert_eq!(b.avg_confidence, 0.0);
            assert_eq!(b.accuracy, 0.0);
        }
    }
}
