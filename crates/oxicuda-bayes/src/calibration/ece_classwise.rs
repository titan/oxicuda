//! Class-wise Expected Calibration Error and per-class reliability diagrams
//! (Kull et al. 2019, "Beyond temperature scaling: Obtaining well-calibrated
//! multi-class probabilities with Dirichlet calibration", NeurIPS).
//!
//! The top-label ECE in [`super::metrics`] only inspects the confidence of the
//! *predicted* (argmax) class and is blind to the calibration of the remaining
//! `K − 1` probabilities. Class-wise ECE repairs this by treating each class as
//! its own one-vs-rest binary calibration problem:
//!
//! For class `c`, the predicted probability `p_{n,c}` is the "confidence" and the
//! indicator `1[y_n = c]` is the "event". Binning these `(confidence, event)`
//! pairs and accumulating the weighted gap between mean confidence and empirical
//! frequency gives a per-class ECE. The **class-wise ECE** is the unweighted mean
//! of the `K` per-class ECEs:
//!
//! ```text
//! ECE_c        = Σ_b (n_{b,c} / N) · |conf(b,c) − freq(b,c)|
//! ECE_classwise = (1/K) · Σ_c ECE_c
//! ```
//!
//! Two binning schemes are offered, mirroring [`super::metrics`]:
//! - [`BinningScheme::Static`]  — equal-width bins over `[0, 1]`.
//! - [`BinningScheme::Adaptive`] — equal-mass (equal-count) quantile bins, fitted
//!   independently per class on that class's probability column.
//!
//! This module also provides the **multi-class Brier score** together with its
//! classic three-term decomposition (Murphy 1973)
//!
//! ```text
//! BS = reliability − resolution + uncertainty
//! ```
//!
//! computed on the full predicted-probability vectors, and re-exposes the
//! top-label **ECE** / **MCE** (which delegate to [`super::metrics`]) so the whole
//! metric family lives behind one configuration object.
//!
//! All accumulation is performed in `f64` for numerical robustness; inputs are
//! the crate-standard `f32` row-major probability matrix plus `usize` labels.

use crate::calibration::metrics::top1_confidences;
use crate::error::{BayesError, BayesResult};

// ─── Constants ─────────────────────────────────────────────────────────────────

/// Tolerance used when validating that a probability lies within `[0, 1]`.
/// A small slack absorbs `f32` round-off from upstream softmax kernels.
const PROB_RANGE_SLACK: f64 = 1e-4;

// ─── Binning scheme ────────────────────────────────────────────────────────────

/// Strategy for placing the calibration bins of a single class column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinningScheme {
    /// Equal-width bins: `[0, 1]` is split into `num_bins` intervals of width
    /// `1/num_bins`. A probability of exactly `1.0` falls in the last bin.
    Static,
    /// Equal-mass (equal-count) bins: the class column is sorted and split into
    /// `num_bins` groups of (approximately) equal cardinality. The last group
    /// absorbs the remainder when `N` is not divisible by `num_bins`.
    Adaptive,
}

impl BinningScheme {
    /// Human-readable name, handy for diagnostics / serialisation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            BinningScheme::Static => "static-equal-width",
            BinningScheme::Adaptive => "adaptive-equal-mass",
        }
    }
}

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the class-wise calibration metric family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClasswiseEceConfig {
    /// Number of calibration bins per class (must be ≥ 1).
    pub num_bins: usize,
    /// Bin-placement strategy.
    pub binning: BinningScheme,
}

impl Default for ClasswiseEceConfig {
    fn default() -> Self {
        Self {
            num_bins: 15,
            binning: BinningScheme::Static,
        }
    }
}

impl ClasswiseEceConfig {
    /// Convenience constructor with equal-width (static) bins.
    #[must_use]
    pub fn static_bins(num_bins: usize) -> Self {
        Self {
            num_bins,
            binning: BinningScheme::Static,
        }
    }

    /// Convenience constructor with equal-mass (adaptive) bins.
    #[must_use]
    pub fn adaptive_bins(num_bins: usize) -> Self {
        Self {
            num_bins,
            binning: BinningScheme::Adaptive,
        }
    }
}

// ─── Reliability data structures ───────────────────────────────────────────────

/// A single calibration bin of one class's one-vs-rest reliability curve.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityPoint {
    /// Inclusive lower probability bound of the bin.
    pub lo: f64,
    /// Exclusive upper probability bound of the bin (inclusive for the last bin).
    pub hi: f64,
    /// Mean predicted probability of the class within the bin.
    pub mean_confidence: f64,
    /// Empirical frequency `mean(1[y == c])` within the bin (the "accuracy").
    pub empirical_accuracy: f64,
    /// Number of samples that fell in the bin.
    pub count: usize,
}

impl ReliabilityPoint {
    /// Signed gap `mean_confidence − empirical_accuracy`. Positive ⇒ the class is
    /// over-confident in this bin; negative ⇒ under-confident.
    #[must_use]
    pub fn gap(&self) -> f64 {
        self.mean_confidence - self.empirical_accuracy
    }
}

/// Per-class reliability diagram (one-vs-rest) plus that class's scalar ECE.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassReliability {
    /// Index of the class this diagram describes (`0..k`).
    pub class: usize,
    /// Bins in ascending probability order. Empty bins are retained for
    /// [`BinningScheme::Static`] so a curve can be drawn on a fixed grid; for
    /// [`BinningScheme::Adaptive`] every returned bin is non-empty.
    pub bins: Vec<ReliabilityPoint>,
    /// One-vs-rest ECE for this class:
    /// `Σ_b (count_b / N) · |mean_confidence_b − empirical_accuracy_b|`.
    pub ece: f64,
}

impl ClassReliability {
    /// Total number of samples represented across all bins (should equal `N`).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.bins.iter().map(|b| b.count).sum()
    }

    /// Maximum per-bin gap over non-empty bins (per-class MCE, one-vs-rest).
    #[must_use]
    pub fn mce(&self) -> f64 {
        self.bins
            .iter()
            .filter(|b| b.count > 0)
            .map(|b| b.gap().abs())
            .fold(0.0_f64, f64::max)
    }
}

// ─── Brier decomposition ───────────────────────────────────────────────────────

/// Murphy's (1973) three-term decomposition of the multi-class Brier score,
/// computed via class-wise binning of the predicted probabilities.
///
/// The identity `brier = reliability − resolution + uncertainty` holds exactly
/// for the binned estimator (the same bins are used for every term); see
/// [`brier_decomposition`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrierDecomposition {
    /// Multi-class Brier score `(1/N) Σ_n Σ_c (p_{n,c} − 1[y_n = c])²`.
    pub brier: f64,
    /// Reliability (lower is better): mean squared gap between bin confidence and
    /// bin frequency, summed over classes.
    pub reliability: f64,
    /// Resolution (higher is better): how far bin frequencies depart from the
    /// class base rate, summed over classes.
    pub resolution: f64,
    /// Uncertainty: the Brier score of the constant base-rate forecast,
    /// `Σ_c base_c · (1 − base_c)`. Depends only on the label distribution.
    pub uncertainty: f64,
}

impl BrierDecomposition {
    /// Residual of the decomposition identity
    /// `brier − (reliability − resolution + uncertainty)`. Should be ≈ 0.
    #[must_use]
    pub fn identity_residual(&self) -> f64 {
        self.brier - (self.reliability - self.resolution + self.uncertainty)
    }
}

// ─── Top-label summary ─────────────────────────────────────────────────────────

/// Top-label calibration summary (argmax confidence vs. correctness), provided
/// for completeness alongside the class-wise metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopLabelCalibration {
    /// Expected Calibration Error of the top-1 confidence.
    pub ece: f64,
    /// Maximum Calibration Error of the top-1 confidence.
    pub mce: f64,
}

// ─── Input validation ──────────────────────────────────────────────────────────

/// Validate a row-major `probs` matrix (`n × k`), the `labels`, and `k`, and
/// return `n` on success.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] if there are no samples or `k == 0`.
/// - [`BayesError::DimensionMismatch`] if `probs.len() != labels.len() * k`.
/// - [`BayesError::InvalidConfig`] if a label is `≥ k`.
/// - [`BayesError::NanEncountered`] if a probability is non-finite or far
///   outside `[0, 1]`.
fn validate_inputs(probs: &[f32], labels: &[usize], k: usize) -> BayesResult<usize> {
    if k == 0 {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if labels.is_empty() || probs.is_empty() {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if probs.len() != labels.len() * k {
        return Err(BayesError::DimensionMismatch {
            expected: labels.len() * k,
            got: probs.len(),
        });
    }
    for &y in labels {
        if y >= k {
            return Err(BayesError::InvalidConfig(format!(
                "label {y} out of range for k = {k} classes"
            )));
        }
    }
    for &p in probs {
        if !p.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "ece_classwise: non-finite probability",
            });
        }
        let pf = p as f64;
        if !(-PROB_RANGE_SLACK..=1.0 + PROB_RANGE_SLACK).contains(&pf) {
            return Err(BayesError::NanEncountered {
                location: "ece_classwise: probability outside [0, 1]",
            });
        }
    }
    Ok(labels.len())
}

/// Validate the configuration's bin count.
fn validate_config(cfg: &ClasswiseEceConfig) -> BayesResult<()> {
    if cfg.num_bins == 0 {
        return Err(BayesError::NCalibBinsTooSmall);
    }
    Ok(())
}

// ─── Binning core ──────────────────────────────────────────────────────────────

/// Per-bin accumulators for a single class column.
struct BinAccumulator {
    /// Lower bound of each bin.
    lo: Vec<f64>,
    /// Upper bound of each bin.
    hi: Vec<f64>,
    /// Σ confidence within each bin.
    conf_sum: Vec<f64>,
    /// Σ event indicator (`1[y == c]`) within each bin.
    event_sum: Vec<f64>,
    /// Number of samples within each bin.
    count: Vec<usize>,
}

impl BinAccumulator {
    fn new(num_bins: usize) -> Self {
        Self {
            lo: vec![0.0; num_bins],
            hi: vec![0.0; num_bins],
            conf_sum: vec![0.0; num_bins],
            event_sum: vec![0.0; num_bins],
            count: vec![0; num_bins],
        }
    }

    /// Finalise into reliability points and the class ECE.
    ///
    /// When `keep_empty` is false, empty bins are dropped from the returned
    /// vector (used by adaptive binning, where empty quantile bins are not
    /// meaningful). `n` is the total sample count used for the ECE weighting.
    fn finalize(&self, n: usize, keep_empty: bool) -> (Vec<ReliabilityPoint>, f64) {
        let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        let mut points = Vec::with_capacity(self.lo.len());
        let mut ece = 0.0_f64;
        for b in 0..self.lo.len() {
            let count = self.count[b];
            if count == 0 {
                if keep_empty {
                    points.push(ReliabilityPoint {
                        lo: self.lo[b],
                        hi: self.hi[b],
                        mean_confidence: 0.0,
                        empirical_accuracy: 0.0,
                        count: 0,
                    });
                }
                continue;
            }
            let inv_c = 1.0 / count as f64;
            let mean_confidence = self.conf_sum[b] * inv_c;
            let empirical_accuracy = self.event_sum[b] * inv_c;
            ece += (count as f64 * inv_n) * (mean_confidence - empirical_accuracy).abs();
            points.push(ReliabilityPoint {
                lo: self.lo[b],
                hi: self.hi[b],
                mean_confidence,
                empirical_accuracy,
                count,
            });
        }
        (points, ece)
    }
}

/// Locate the equal-width bin index for `p ∈ [0, 1]` with `num_bins` bins.
/// A value of exactly `1.0` (or above) maps to the last bin.
#[inline]
fn static_bin_index(p: f64, num_bins: usize) -> usize {
    if p <= 0.0 {
        return 0;
    }
    let raw = (p * num_bins as f64).floor() as isize;
    if raw < 0 {
        0
    } else {
        (raw as usize).min(num_bins - 1)
    }
}

/// Build the per-class accumulator for class `c` under static (equal-width)
/// binning.
fn accumulate_static(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    c: usize,
    num_bins: usize,
) -> BinAccumulator {
    let mut acc = BinAccumulator::new(num_bins);
    let width = 1.0 / num_bins as f64;
    for b in 0..num_bins {
        acc.lo[b] = b as f64 * width;
        acc.hi[b] = (b + 1) as f64 * width;
    }
    for (n, &y) in labels.iter().enumerate() {
        let p = probs[n * k + c] as f64;
        let idx = static_bin_index(p, num_bins);
        acc.conf_sum[idx] += p;
        acc.event_sum[idx] += if y == c { 1.0 } else { 0.0 };
        acc.count[idx] += 1;
    }
    acc
}

/// Build the per-class accumulator for class `c` under adaptive (equal-mass)
/// binning. The class column is sorted; samples are split into `num_bins`
/// equal-count groups (the leading `remainder` groups receive one extra sample).
fn accumulate_adaptive(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    c: usize,
    num_bins: usize,
) -> BinAccumulator {
    let n = labels.len();
    // Sort sample indices by the class-c probability.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let pa = probs[a * k + c];
        let pb = probs[b * k + c];
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let effective_bins = num_bins.min(n).max(1);
    let mut acc = BinAccumulator::new(effective_bins);
    let chunk = n / effective_bins;
    let remainder = n % effective_bins;

    let mut start = 0usize;
    for b in 0..effective_bins {
        let extra = usize::from(b < remainder);
        let len = chunk + extra;
        let end = start + len;
        let mut conf_sum = 0.0_f64;
        let mut event_sum = 0.0_f64;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in &order[start..end] {
            let p = probs[i * k + c] as f64;
            conf_sum += p;
            event_sum += if labels[i] == c { 1.0 } else { 0.0 };
            if p < lo {
                lo = p;
            }
            if p > hi {
                hi = p;
            }
        }
        if len == 0 {
            lo = 0.0;
            hi = 0.0;
        }
        acc.lo[b] = lo;
        acc.hi[b] = hi;
        acc.conf_sum[b] = conf_sum;
        acc.event_sum[b] = event_sum;
        acc.count[b] = len;
        start = end;
    }
    acc
}

/// Compute the per-class reliability + ECE for class `c` under the chosen scheme.
fn class_reliability_one(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    c: usize,
    cfg: &ClasswiseEceConfig,
) -> ClassReliability {
    let n = labels.len();
    let (bins, ece) = match cfg.binning {
        BinningScheme::Static => {
            let acc = accumulate_static(probs, labels, k, c, cfg.num_bins);
            acc.finalize(n, true)
        }
        BinningScheme::Adaptive => {
            let acc = accumulate_adaptive(probs, labels, k, c, cfg.num_bins);
            acc.finalize(n, false)
        }
    };
    ClassReliability {
        class: c,
        bins,
        ece,
    }
}

// ─── Public API ────────────────────────────────────────────────────────────────

/// Per-class one-vs-rest reliability diagrams (one entry per class `0..k`).
///
/// Each [`ClassReliability`] carries the binned `(mean_confidence,
/// empirical_accuracy, count)` triples needed to plot a reliability curve, plus
/// that class's scalar one-vs-rest ECE.
///
/// # Errors
/// Propagates input validation from `validate_inputs` and config validation
/// from `validate_config`.
pub fn per_class_reliability(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    cfg: &ClasswiseEceConfig,
) -> BayesResult<Vec<ClassReliability>> {
    validate_config(cfg)?;
    validate_inputs(probs, labels, k)?;
    let mut out = Vec::with_capacity(k);
    for c in 0..k {
        out.push(class_reliability_one(probs, labels, k, c, cfg));
    }
    Ok(out)
}

/// Per-class one-vs-rest ECE values (length `k`).
///
/// `class_wise_eces()[c] == per_class_reliability(...)[c].ece`.
///
/// # Errors
/// See [`per_class_reliability`].
pub fn class_wise_eces(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    cfg: &ClasswiseEceConfig,
) -> BayesResult<Vec<f64>> {
    Ok(per_class_reliability(probs, labels, k, cfg)?
        .into_iter()
        .map(|r| r.ece)
        .collect())
}

/// Class-wise ECE: the unweighted mean of the `k` one-vs-rest per-class ECEs
/// (Kull et al. 2019). Always lies in `[0, 1]`.
///
/// # Errors
/// See [`per_class_reliability`].
pub fn classwise_ece(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    cfg: &ClasswiseEceConfig,
) -> BayesResult<f64> {
    let eces = class_wise_eces(probs, labels, k, cfg)?;
    // `k ≥ 1` is guaranteed by validation, so the divisor is non-zero.
    let sum: f64 = eces.iter().sum();
    Ok(sum / k as f64)
}

/// Top-label ECE and MCE over the argmax confidence, delegating to
/// [`super::metrics`] (equal-width binning). Provided for completeness.
///
/// # Errors
/// - Propagates `validate_config` / `validate_inputs`.
/// - Propagates errors from [`top1_confidences`] and the underlying metric.
pub fn top_label_calibration(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    cfg: &ClasswiseEceConfig,
) -> BayesResult<TopLabelCalibration> {
    validate_config(cfg)?;
    validate_inputs(probs, labels, k)?;
    let (confidences, correct) = top1_confidences(probs, labels, k)?;
    let diagram =
        crate::calibration::metrics::reliability_diagram(&confidences, &correct, cfg.num_bins)?;
    Ok(TopLabelCalibration {
        ece: diagram.ece() as f64,
        mce: diagram.mce() as f64,
    })
}

/// Multi-class Brier score `(1/N) Σ_n Σ_c (p_{n,c} − 1[y_n = c])²`, accumulated
/// in `f64`.
///
/// # Errors
/// See `validate_inputs`.
pub fn multiclass_brier_score(probs: &[f32], labels: &[usize], k: usize) -> BayesResult<f64> {
    let n = validate_inputs(probs, labels, k)?;
    let mut sum = 0.0_f64;
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs[i * k..(i + 1) * k];
        for (c, &p) in row.iter().enumerate() {
            let target = if c == y { 1.0 } else { 0.0 };
            let d = p as f64 - target;
            sum += d * d;
        }
    }
    Ok(sum / n as f64)
}

/// Multi-class Brier score with Murphy's reliability/resolution/uncertainty
/// decomposition, computed by binning each class column under `cfg`.
///
/// The binned estimator satisfies the exact identity
/// `brier = reliability − resolution + uncertainty`, where each term is summed
/// over classes:
///
/// ```text
/// reliability = (1/N) Σ_c Σ_b n_{b,c} · (conf_{b,c} − freq_{b,c})²
/// resolution  = (1/N) Σ_c Σ_b n_{b,c} · (freq_{b,c} − base_c)²
/// uncertainty = Σ_c base_c · (1 − base_c)
/// ```
///
/// with `base_c` the empirical base rate of class `c`. Here the binned forecast
/// is the in-bin mean confidence; the cross-term that would otherwise break the
/// identity is handled by the `2 Σ n_{b,c} (conf − freq)(freq − base)` correction
/// folded into the reported `brier` (which is the *binned* Brier of the
/// per-bin-mean forecast, not the raw one — see notes below).
///
/// # Notes
/// To keep the identity exact, the returned `brier` is the Brier score of the
/// **binned** forecast (each prediction replaced by its bin's mean confidence).
/// For the raw, un-binned Brier score use [`multiclass_brier_score`]; the two
/// coincide as `num_bins → N` under adaptive binning.
///
/// # Errors
/// See [`per_class_reliability`].
pub fn brier_decomposition(
    probs: &[f32],
    labels: &[usize],
    k: usize,
    cfg: &ClasswiseEceConfig,
) -> BayesResult<BrierDecomposition> {
    validate_config(cfg)?;
    let n = validate_inputs(probs, labels, k)?;
    let inv_n = 1.0 / n as f64;

    // Class base rates.
    let mut base = vec![0.0_f64; k];
    for &y in labels {
        base[y] += 1.0;
    }
    for b in base.iter_mut() {
        *b *= inv_n;
    }

    let mut reliability = 0.0_f64;
    let mut resolution = 0.0_f64;
    let mut binned_brier = 0.0_f64;

    for (c, &base_c) in base.iter().enumerate() {
        let acc = match cfg.binning {
            BinningScheme::Static => accumulate_static(probs, labels, k, c, cfg.num_bins),
            BinningScheme::Adaptive => accumulate_adaptive(probs, labels, k, c, cfg.num_bins),
        };
        for b in 0..acc.count.len() {
            let count = acc.count[b];
            if count == 0 {
                continue;
            }
            let cf = count as f64;
            let conf = acc.conf_sum[b] / cf;
            let freq = acc.event_sum[b] / cf;
            // reliability: squared gap between forecast and realised frequency.
            reliability += cf * (conf - freq) * (conf - freq);
            // resolution: how far the bin frequency departs from the base rate.
            resolution += cf * (freq - base_c) * (freq - base_c);
            // binned Brier (one-vs-rest, summed over classes): each of the
            // `count` samples carries event ∈ {0,1}; with constant in-bin
            // forecast `conf` the mean squared error over the bin is
            //   freq·(1 − conf)² + (1 − freq)·conf²  (per sample),
            // because Σ (conf − event)² = count·[conf² − 2·conf·freq + freq].
            binned_brier += cf * (conf * conf - 2.0 * conf * freq + freq);
        }
    }

    reliability *= inv_n;
    resolution *= inv_n;
    binned_brier *= inv_n;

    let uncertainty: f64 = base.iter().map(|&p| p * (1.0 - p)).sum();

    Ok(BrierDecomposition {
        brier: binned_brier,
        reliability,
        resolution,
        uncertainty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a perfectly calibrated `k`-class set: for each of `groups` target
    /// frequencies, emit `per_group` samples whose class-`c` probability equals
    /// the frequency at which `y == c`. Concretely we use a two-class structure
    /// embedded in `k` classes so the per-class one-vs-rest frequency exactly
    /// matches the stated confidence.
    fn perfectly_calibrated_binary_in_k(k: usize) -> (Vec<f32>, Vec<usize>) {
        // Bin masses at confidence p with empirical positive rate p.
        // For class 0 vs rest: p in {0.2, 0.8}. Remaining mass goes to class 1.
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        let levels = [0.2_f32, 0.8_f32];
        let per = 50usize;
        for &p in &levels {
            let positives = (p * per as f32).round() as usize;
            for i in 0..per {
                // Row of length k: class 0 gets p, class 1 gets (1 - p), rest 0.
                let mut row = vec![0.0_f32; k];
                row[0] = p;
                if k > 1 {
                    row[1] = 1.0 - p;
                }
                probs.extend_from_slice(&row);
                // First `positives` samples are class 0, rest class 1.
                labels.push(if i < positives { 0 } else { 1.min(k - 1) });
            }
        }
        (probs, labels)
    }

    /// Build a deliberately miscalibrated, badly over-confident set: class 0 is
    /// always predicted with probability 0.95 but is correct only half the time.
    fn overconfident(k: usize, n: usize) -> (Vec<f32>, Vec<usize>) {
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let mut row = vec![0.0_f32; k];
            row[0] = 0.95;
            if k > 1 {
                let rem = 0.05 / (k - 1) as f32;
                for slot in row.iter_mut().skip(1) {
                    *slot = rem;
                }
            }
            probs.extend_from_slice(&row);
            labels.push(if i % 2 == 0 { 0 } else { 1.min(k - 1) });
        }
        (probs, labels)
    }

    #[test]
    fn classwise_ece_near_zero_for_perfectly_calibrated() {
        let (probs, labels) = perfectly_calibrated_binary_in_k(3);
        let cfg = ClasswiseEceConfig::static_bins(10);
        let ece = classwise_ece(&probs, &labels, 3, &cfg).expect("classwise_ece should succeed");
        assert!(ece < 1e-3, "expected ~0 ECE for calibrated set, got {ece}");
    }

    #[test]
    fn classwise_ece_larger_for_miscalibrated() {
        let k = 3;
        let (cp, cl) = perfectly_calibrated_binary_in_k(k);
        let (mp, ml) = overconfident(k, 200);
        let cfg = ClasswiseEceConfig::static_bins(10);
        let calibrated =
            classwise_ece(&cp, &cl, k, &cfg).expect("classwise_ece should succeed (calibrated)");
        let miscalibrated =
            classwise_ece(&mp, &ml, k, &cfg).expect("classwise_ece should succeed (miscalibrated)");
        assert!(
            miscalibrated > calibrated + 0.1,
            "miscalibrated ECE ({miscalibrated}) should exceed calibrated ({calibrated})"
        );
    }

    #[test]
    fn classwise_ece_within_unit_interval() {
        let (mp, ml) = overconfident(4, 120);
        for cfg in [
            ClasswiseEceConfig::static_bins(8),
            ClasswiseEceConfig::adaptive_bins(8),
        ] {
            let ece = classwise_ece(&mp, &ml, 4, &cfg).expect("classwise_ece should succeed");
            assert!(
                (0.0..=1.0).contains(&ece),
                "classwise ECE must lie in [0, 1], got {ece} ({:?})",
                cfg.binning
            );
        }
    }

    #[test]
    fn per_class_reliability_counts_sum_to_n_static() {
        let k = 3;
        let (probs, labels) = perfectly_calibrated_binary_in_k(k);
        let n = labels.len();
        let cfg = ClasswiseEceConfig::static_bins(12);
        let rel = per_class_reliability(&probs, &labels, k, &cfg)
            .expect("per_class_reliability should succeed");
        assert_eq!(rel.len(), k);
        for cr in &rel {
            assert_eq!(
                cr.total_count(),
                n,
                "class {} bin counts must sum to N = {n}",
                cr.class
            );
        }
    }

    #[test]
    fn per_class_reliability_counts_sum_to_n_adaptive() {
        let k = 3;
        let (probs, labels) = perfectly_calibrated_binary_in_k(k);
        let n = labels.len();
        let cfg = ClasswiseEceConfig::adaptive_bins(7);
        let rel = per_class_reliability(&probs, &labels, k, &cfg)
            .expect("per_class_reliability should succeed");
        for cr in &rel {
            assert_eq!(
                cr.total_count(),
                n,
                "adaptive class {} bin counts must sum to N = {n}",
                cr.class
            );
            // Adaptive bins are all non-empty.
            assert!(cr.bins.iter().all(|b| b.count > 0));
        }
    }

    #[test]
    fn adaptive_and_static_both_valid_and_finite() {
        let (mp, ml) = overconfident(3, 150);
        let s = classwise_ece(&mp, &ml, 3, &ClasswiseEceConfig::static_bins(10))
            .expect("static classwise_ece should succeed");
        let a = classwise_ece(&mp, &ml, 3, &ClasswiseEceConfig::adaptive_bins(10))
            .expect("adaptive classwise_ece should succeed");
        assert!(s.is_finite() && a.is_finite());
        assert!(
            s > 0.0 && a > 0.0,
            "both schemes should detect miscalibration"
        );
    }

    #[test]
    fn brier_decomposition_identity_holds_static() {
        let (probs, labels) = perfectly_calibrated_binary_in_k(3);
        let cfg = ClasswiseEceConfig::static_bins(10);
        let d =
            brier_decomposition(&probs, &labels, 3, &cfg).expect("brier_decomposition should work");
        let residual = d.identity_residual();
        assert!(
            residual.abs() < 1e-9,
            "reliability - resolution + uncertainty must equal brier; residual = {residual}"
        );
    }

    #[test]
    fn brier_decomposition_identity_holds_adaptive() {
        let (mp, ml) = overconfident(4, 160);
        let cfg = ClasswiseEceConfig::adaptive_bins(8);
        let d = brier_decomposition(&mp, &ml, 4, &cfg).expect("brier_decomposition should work");
        assert!(
            d.identity_residual().abs() < 1e-9,
            "adaptive decomposition identity broken: residual = {}",
            d.identity_residual()
        );
        assert!(d.reliability >= -1e-12 && d.resolution >= -1e-12 && d.uncertainty >= -1e-12);
    }

    #[test]
    fn binned_brier_matches_raw_at_full_resolution_adaptive() {
        // With as many adaptive bins as samples, each bin holds one point, so the
        // binned forecast equals the raw forecast and the binned Brier equals the
        // raw multi-class Brier.
        let (mp, ml) = overconfident(3, 60);
        let n = ml.len();
        let cfg = ClasswiseEceConfig::adaptive_bins(n);
        let d = brier_decomposition(&mp, &ml, 3, &cfg).expect("decomposition should work");
        let raw = multiclass_brier_score(&mp, &ml, 3).expect("raw brier should work");
        assert!(
            (d.brier - raw).abs() < 1e-6,
            "binned brier {} should match raw brier {} at full resolution",
            d.brier,
            raw
        );
    }

    #[test]
    fn top_label_calibration_detects_overconfidence() {
        let (mp, ml) = overconfident(3, 200);
        let cfg = ClasswiseEceConfig::static_bins(10);
        let tl =
            top_label_calibration(&mp, &ml, 3, &cfg).expect("top_label_calibration should work");
        // Predicting argmax class 0 at conf 0.95 with 50% accuracy → big gap.
        assert!(
            tl.ece > 0.3,
            "top-label ECE should be large, got {}",
            tl.ece
        );
        assert!(tl.mce >= tl.ece - 1e-9, "MCE must be >= ECE");
    }

    #[test]
    fn single_class_is_trivially_calibrated() {
        // k = 1: every probability is 1.0 and every label is 0.
        let probs = vec![1.0_f32; 5];
        let labels = vec![0usize; 5];
        let cfg = ClasswiseEceConfig::static_bins(10);
        let ece = classwise_ece(&probs, &labels, 1, &cfg).expect("k=1 should be valid");
        assert!(ece < 1e-9, "single-class set is perfectly calibrated");
        let d = brier_decomposition(&probs, &labels, 1, &cfg).expect("decomposition should work");
        assert!(d.identity_residual().abs() < 1e-12);
        assert!(d.brier < 1e-12 && d.uncertainty < 1e-12);
    }

    #[test]
    fn rejects_empty_inputs() {
        let cfg = ClasswiseEceConfig::default();
        let r = classwise_ece(&[], &[], 3, &cfg);
        assert!(
            matches!(r, Err(BayesError::CalibrationSetEmpty)),
            "got {r:?}"
        );
    }

    #[test]
    fn rejects_label_out_of_range() {
        let probs = vec![0.5_f32, 0.5, 0.5, 0.5];
        let labels = vec![0usize, 5];
        let cfg = ClasswiseEceConfig::default();
        let r = classwise_ece(&probs, &labels, 2, &cfg);
        assert!(matches!(r, Err(BayesError::InvalidConfig(_))), "got {r:?}");
    }

    #[test]
    fn rejects_probs_length_mismatch() {
        let probs = vec![0.5_f32, 0.5, 0.5]; // 3 entries, not 2*2
        let labels = vec![0usize, 1];
        let cfg = ClasswiseEceConfig::default();
        let r = classwise_ece(&probs, &labels, 2, &cfg);
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn rejects_zero_bins() {
        let probs = vec![0.5_f32, 0.5];
        let labels = vec![0usize];
        let cfg = ClasswiseEceConfig {
            num_bins: 0,
            binning: BinningScheme::Static,
        };
        let r = classwise_ece(&probs, &labels, 2, &cfg);
        assert!(
            matches!(r, Err(BayesError::NCalibBinsTooSmall)),
            "got {r:?}"
        );
    }

    #[test]
    fn rejects_non_finite_probability() {
        let probs = vec![f32::NAN, 0.5];
        let labels = vec![0usize];
        let cfg = ClasswiseEceConfig::default();
        let r = classwise_ece(&probs, &labels, 2, &cfg);
        assert!(
            matches!(r, Err(BayesError::NanEncountered { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn class_wise_eces_match_reliability_eces() {
        let (mp, ml) = overconfident(3, 90);
        let cfg = ClasswiseEceConfig::static_bins(10);
        let eces = class_wise_eces(&mp, &ml, 3, &cfg).expect("class_wise_eces should work");
        let rel = per_class_reliability(&mp, &ml, 3, &cfg).expect("reliability should work");
        assert_eq!(eces.len(), rel.len());
        for (e, r) in eces.iter().zip(rel.iter()) {
            assert!((e - r.ece).abs() < 1e-12);
        }
    }
}
