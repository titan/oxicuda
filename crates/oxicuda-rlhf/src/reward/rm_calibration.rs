//! Reward-model calibration (Touvron et al. 2023, *Llama 2*).
//!
//! Reference: Touvron, H., Martin, L., Stone, K., et al. (2023). *Llama 2: Open
//! Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288. §3.2.2 discusses
//! reward-model **calibration**: a well-calibrated reward model should assign a
//! preference probability that matches the empirical rate at which the
//! higher-scoring response is actually preferred by humans.
//!
//! A Bradley–Terry reward model turns a pair of scalar rewards `(r_chosen,
//! r_rejected)` into a preference probability with a logistic link:
//!
//! ```text
//!   p = σ( (r_chosen − r_rejected) / T )
//! ```
//!
//! where `T` is a **temperature**. Raw reward models are frequently
//! *over-confident* — the margins `d = r_chosen − r_rejected` are larger in
//! magnitude than the actual reliability warrants — so the predicted `σ(d)`
//! over-states accuracy. This module provides two post-hoc calibration methods
//! that do **not** retrain the model:
//!
//! 1. **Temperature scaling** (Guo et al. 2017): fit a single scalar `T` that
//!    minimises the held-out negative log-likelihood
//!    `NLL(T) = −Σ log σ(d_i / T)`. Substituting `u = 1/T` makes `NLL(u)` a
//!    sum of `softplus(−d_i · u)` terms, which is **convex** in `u`; its
//!    derivative is monotone, so the unique optimum is found by bisecting the
//!    derivative. Because `σ(·/T)` is strictly monotone for every `T > 0`,
//!    temperature scaling **never** changes the ranking of preferences — it only
//!    rescales the confidence.
//!
//! 2. **Isotonic regression** (Pool-Adjacent-Violators Algorithm, PAVA): fit a
//!    monotone non-decreasing step map from raw scores to calibrated
//!    probabilities by minimising `Σ (g(x_i) − y_i)²` subject to monotonicity.
//!    This is non-parametric and can correct shapes that a single temperature
//!    cannot.
//!
//! Calibration quality is summarised by the **Expected Calibration Error (ECE)**
//! computed from a reliability diagram: bin the predicted *confidence*
//! `σ(|d| / T)` and compare, per bin, the mean confidence against the empirical
//! accuracy (the fraction of pairs whose chosen response really does out-score
//! the rejected one). For a perfectly calibrated model the two agree in every
//! bin and `ECE = 0`.

use crate::error::{RlhfError, RlhfResult};

/// Lowest temperature considered by [`fit_temperature_pairs`].
const T_MIN: f32 = 1.0e-2;
/// Highest temperature considered by [`fit_temperature_pairs`].
const T_MAX: f32 = 1.0e2;
/// Number of bisection steps used to locate the temperature optimum.
const BISECT_STEPS: usize = 80;

/// Numerically stable logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Post-hoc calibrator for a Bradley–Terry reward model.
///
/// Holds a fitted temperature (default `1.0`, i.e. the identity calibration) and
/// an optional isotonic map produced by [`RewardModelCalibrator::fit_isotonic`].
#[derive(Debug, Clone)]
pub struct RewardModelCalibrator {
    temperature: f32,
    /// Representative score of each isotonic block (non-decreasing).
    iso_x: Vec<f32>,
    /// Fitted calibrated probability of each isotonic block (non-decreasing).
    iso_y: Vec<f32>,
}

impl Default for RewardModelCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardModelCalibrator {
    /// Construct an identity calibrator (`T = 1`, no isotonic map).
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            iso_x: Vec::new(),
            iso_y: Vec::new(),
        }
    }

    /// The currently fitted temperature.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Whether an isotonic map has been fitted.
    #[must_use]
    pub fn has_isotonic(&self) -> bool {
        !self.iso_x.is_empty()
    }

    /// Fit a single temperature `T` minimising the held-out NLL of the
    /// preference probabilities `σ((chosen − rejected) / T)`, store it, and
    /// return it.
    ///
    /// # Errors
    /// - [`RlhfError::EmptyInput`] if either slice is empty.
    /// - [`RlhfError::MismatchedPairLength`] if the slices differ in length.
    /// - [`RlhfError::NanEncountered`] if any margin is non-finite.
    pub fn fit_temperature(&mut self, chosen: &[f32], rejected: &[f32]) -> RlhfResult<f32> {
        let t = fit_temperature_pairs(chosen, rejected)?;
        self.temperature = t;
        Ok(t)
    }

    /// Fit an isotonic (monotone) calibration map from raw `scores` to binary
    /// `labels` via PAVA and store it for [`RewardModelCalibrator::calibrate`].
    ///
    /// # Errors
    /// Propagates errors from [`isotonic_regression`].
    pub fn fit_isotonic(&mut self, scores: &[f32], labels: &[f32]) -> RlhfResult<()> {
        let (xs, ys) = isotonic_regression(scores, labels)?;
        self.iso_x = xs;
        self.iso_y = ys;
        Ok(())
    }

    /// Map a raw score to a calibrated probability.
    ///
    /// If an isotonic map has been fitted, the (clamped, linearly interpolated)
    /// monotone map is applied; otherwise the temperature-scaled sigmoid
    /// `σ(score / T)` is returned. The result is always non-decreasing in
    /// `score`.
    #[must_use]
    pub fn calibrate(&self, score: f32) -> f32 {
        if self.iso_x.is_empty() {
            return sigmoid(score / self.temperature);
        }
        let xs = &self.iso_x;
        let ys = &self.iso_y;
        let last = xs.len() - 1;
        if score <= xs[0] {
            return ys[0];
        }
        if score >= xs[last] {
            return ys[last];
        }
        // Binary search for the bracketing segment [lo, hi].
        let mut lo = 0_usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if xs[mid] <= score {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x0 = xs[lo];
        let x1 = xs[hi];
        let denom = x1 - x0;
        if denom <= 0.0 {
            return ys[hi];
        }
        let frac = (score - x0) / denom;
        ys[lo] + frac * (ys[hi] - ys[lo])
    }

    /// Expected Calibration Error of the held-out pairs at the *currently
    /// fitted* temperature.
    ///
    /// # Errors
    /// Propagates errors from [`expected_calibration_error`].
    pub fn ece(&self, chosen: &[f32], rejected: &[f32], n_bins: usize) -> RlhfResult<f32> {
        expected_calibration_error(chosen, rejected, self.temperature, n_bins)
    }
}

/// Fit the temperature that minimises `NLL(T) = −Σ log σ((chosen − rejected) / T)`.
///
/// The objective is convex in `u = 1/T`; its derivative
/// `g(u) = −Σ d_i · σ(−d_i · u)` is monotone increasing, so the minimiser is
/// located by bisecting `g` over `u ∈ [1/T_MAX, 1/T_MIN]`. When the data are
/// perfectly separable (every margin positive) the optimum lies at the boundary
/// and the clamped `T_MIN` is returned; when the model is worse than random
/// (every margin negative) `T_MAX` is returned.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if either slice is empty.
/// - [`RlhfError::MismatchedPairLength`] if the slices differ in length.
/// - [`RlhfError::NanEncountered`] if any margin is non-finite.
pub fn fit_temperature_pairs(chosen: &[f32], rejected: &[f32]) -> RlhfResult<f32> {
    if chosen.is_empty() || rejected.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen.len() != rejected.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen.len(),
            rejected: rejected.len(),
        });
    }
    let mut margins = Vec::with_capacity(chosen.len());
    for (&c, &r) in chosen.iter().zip(rejected.iter()) {
        let d = c - r;
        if !d.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        margins.push(d);
    }

    // Derivative of the NLL w.r.t. the inverse temperature u = 1/T.
    let grad = |u: f32| -> f32 { margins.iter().map(|&d| -d * sigmoid(-d * u)).sum::<f32>() };

    let u_lo = 1.0 / T_MAX;
    let u_hi = 1.0 / T_MIN;
    let g_lo = grad(u_lo);
    let g_hi = grad(u_hi);

    let u_star = if g_lo >= 0.0 {
        // NLL increasing on the whole range → minimum at the smallest u (T_MAX).
        u_lo
    } else if g_hi <= 0.0 {
        // NLL decreasing on the whole range → minimum at the largest u (T_MIN).
        u_hi
    } else {
        let mut a = u_lo;
        let mut b = u_hi;
        for _ in 0..BISECT_STEPS {
            let m = 0.5 * (a + b);
            if grad(m) > 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        0.5 * (a + b)
    };

    Ok(1.0 / u_star)
}

/// Expected Calibration Error from a reliability diagram of the preference pairs.
///
/// For each pair the predicted **confidence** is `σ(|chosen − rejected| / T)`
/// and the outcome is **correct** (`1`) iff the chosen reward exceeds the
/// rejected reward. Confidences are bucketed into `n_bins` equal-width bins over
/// `[0, 1]`; the ECE is the sample-weighted mean absolute gap between the mean
/// confidence and the empirical accuracy of each non-empty bin. `ECE ∈ [0, 1]`.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if either slice is empty.
/// - [`RlhfError::MismatchedPairLength`] if the slices differ in length.
/// - [`RlhfError::InvalidTemp`] if `temperature` is non-finite or non-positive.
/// - [`RlhfError::Internal`] if `n_bins == 0`.
/// - [`RlhfError::NanEncountered`] if any margin is non-finite.
pub fn expected_calibration_error(
    chosen: &[f32],
    rejected: &[f32],
    temperature: f32,
    n_bins: usize,
) -> RlhfResult<f32> {
    if chosen.is_empty() || rejected.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen.len() != rejected.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen.len(),
            rejected: rejected.len(),
        });
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(RlhfError::InvalidTemp { temp: temperature });
    }
    if n_bins == 0 {
        return Err(RlhfError::Internal {
            msg: "n_bins must be >= 1".to_string(),
        });
    }

    let mut bin_conf = vec![0.0_f32; n_bins];
    let mut bin_acc = vec![0.0_f32; n_bins];
    let mut bin_cnt = vec![0_usize; n_bins];

    for (&c, &r) in chosen.iter().zip(rejected.iter()) {
        let d = c - r;
        if !d.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        let conf = sigmoid(d.abs() / temperature);
        let correct = if d > 0.0 { 1.0 } else { 0.0 };
        let mut idx = (conf * n_bins as f32) as usize;
        if idx >= n_bins {
            idx = n_bins - 1;
        }
        bin_conf[idx] += conf;
        bin_acc[idx] += correct;
        bin_cnt[idx] += 1;
    }

    let n = chosen.len() as f32;
    let mut ece = 0.0_f32;
    for ((&conf_sum, &acc_sum), &cnt) in bin_conf.iter().zip(bin_acc.iter()).zip(bin_cnt.iter()) {
        if cnt == 0 {
            continue;
        }
        let cnt_f = cnt as f32;
        let avg_conf = conf_sum / cnt_f;
        let avg_acc = acc_sum / cnt_f;
        ece += (cnt_f / n) * (avg_conf - avg_acc).abs();
    }
    Ok(ece)
}

/// Isotonic regression via the Pool-Adjacent-Violators Algorithm (PAVA).
///
/// Returns `(xs, ys)` where `xs` are the per-block representative scores (the
/// mean score of each pooled block, non-decreasing) and `ys` are the fitted
/// non-decreasing calibrated values. The fit minimises `Σ (g(x_i) − y_i)²`
/// over all non-decreasing step functions `g`.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if either slice is empty.
/// - [`RlhfError::DimensionMismatch`] if the slices differ in length.
/// - [`RlhfError::NanEncountered`] if any value is non-finite.
pub fn isotonic_regression(scores: &[f32], labels: &[f32]) -> RlhfResult<(Vec<f32>, Vec<f32>)> {
    if scores.is_empty() || labels.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if scores.len() != labels.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: scores.len(),
            got: labels.len(),
        });
    }
    for (&s, &y) in scores.iter().zip(labels.iter()) {
        if !s.is_finite() || !y.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
    }

    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Each block stores: pooled value, total weight, sum of scores, member count.
    let mut b_val: Vec<f32> = Vec::with_capacity(n);
    let mut b_wgt: Vec<f32> = Vec::with_capacity(n);
    let mut b_xsum: Vec<f32> = Vec::with_capacity(n);
    let mut b_cnt: Vec<usize> = Vec::with_capacity(n);

    for &i in &order {
        b_val.push(labels[i]);
        b_wgt.push(1.0);
        b_xsum.push(scores[i]);
        b_cnt.push(1);
        // Pool while the previous block violates monotonicity.
        while b_val.len() >= 2 && b_val[b_val.len() - 2] > b_val[b_val.len() - 1] {
            let last = b_val.len() - 1;
            let prev = last - 1;
            let merged_w = b_wgt[prev] + b_wgt[last];
            let merged_v = (b_val[prev] * b_wgt[prev] + b_val[last] * b_wgt[last]) / merged_w;
            b_val[prev] = merged_v;
            b_wgt[prev] = merged_w;
            b_xsum[prev] += b_xsum[last];
            b_cnt[prev] += b_cnt[last];
            b_val.pop();
            b_wgt.pop();
            b_xsum.pop();
            b_cnt.pop();
        }
    }

    let mut xs = Vec::with_capacity(b_val.len());
    let mut ys = Vec::with_capacity(b_val.len());
    for ((&v, &xsum), &cnt) in b_val.iter().zip(b_xsum.iter()).zip(b_cnt.iter()) {
        xs.push(xsum / cnt as f32);
        ys.push(v);
    }
    Ok((xs, ys))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `(chosen, rejected)` pairs from signed margins: `+m` is a correctly
    /// ordered pair (chosen wins), `−m` is a mistake (human chose the lower
    /// reward).
    fn pairs_from_margins(margins: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut chosen = Vec::with_capacity(margins.len());
        let mut rejected = Vec::with_capacity(margins.len());
        for &m in margins {
            if m >= 0.0 {
                chosen.push(m);
                rejected.push(0.0);
            } else {
                chosen.push(0.0);
                rejected.push(-m);
            }
        }
        (chosen, rejected)
    }

    /// Calibrated-by-construction set: margins ±ln(9) with a 9:1 correct:wrong
    /// ratio, so the empirical accuracy σ(ln 9) = 0.9 exactly matches the
    /// predicted confidence at T = 1.
    fn calibrated_pairs() -> (Vec<f32>, Vec<f32>) {
        let m = 9.0_f32.ln();
        let mut margins = vec![m; 9];
        margins.push(-m);
        pairs_from_margins(&margins)
    }

    /// Over-confident set: margins ±4 with a 22:3 ratio → accuracy 0.88, far
    /// below the predicted σ(4) ≈ 0.982, so the fitted temperature exceeds 1.
    fn overconfident_pairs() -> (Vec<f32>, Vec<f32>) {
        let mut margins = vec![4.0_f32; 22];
        for _ in 0..3 {
            margins.push(-4.0);
        }
        pairs_from_margins(&margins)
    }

    #[test]
    fn default_temperature_is_one() {
        let cal = RewardModelCalibrator::new();
        assert!((cal.temperature() - 1.0).abs() < 1e-6);
        assert!(!cal.has_isotonic());
    }

    #[test]
    fn temperature_scaling_preserves_ranking() {
        let (chosen, rejected) = overconfident_pairs();
        let t = fit_temperature_pairs(&chosen, &rejected).expect("fit");
        // σ(d / T) must be strictly monotone in the margin d for any T > 0.
        let margins = [-3.0_f32, -1.0, 0.0, 0.5, 2.0, 5.0];
        let mut prev = f32::NEG_INFINITY;
        for &d in &margins {
            let p = sigmoid(d / t);
            assert!(
                p > prev,
                "preference probability must preserve margin ranking: d={d}, p={p}, prev={prev}"
            );
            prev = p;
        }
    }

    #[test]
    fn overconfident_fits_temperature_above_one_and_lowers_ece() {
        let (chosen, rejected) = overconfident_pairs();
        let t = fit_temperature_pairs(&chosen, &rejected).expect("fit");
        assert!(t > 1.05, "over-confident model should need T > 1, got {t}");

        let ece_before =
            expected_calibration_error(&chosen, &rejected, 1.0, 10).expect("ece before");
        let ece_after = expected_calibration_error(&chosen, &rejected, t, 10).expect("ece after");
        assert!(
            ece_after < ece_before - 0.05,
            "calibration should reduce ECE: before={ece_before}, after={ece_after}"
        );
    }

    #[test]
    fn perfectly_calibrated_gives_unit_temperature_and_zero_ece() {
        let (chosen, rejected) = calibrated_pairs();
        let t = fit_temperature_pairs(&chosen, &rejected).expect("fit");
        assert!((t - 1.0).abs() < 0.1, "calibrated data → T ≈ 1, got {t}");

        let ece = expected_calibration_error(&chosen, &rejected, 1.0, 10).expect("ece");
        assert!(ece < 1e-3, "calibrated data → ECE ≈ 0, got {ece}");
    }

    #[test]
    fn ece_is_bounded() {
        let (chosen, rejected) = overconfident_pairs();
        let ece = expected_calibration_error(&chosen, &rejected, 1.0, 8).expect("ece");
        assert!((0.0..=1.0).contains(&ece), "ECE out of [0,1]: {ece}");
    }

    #[test]
    fn pava_output_is_non_decreasing() {
        // Noisy binary labels along an increasing score axis.
        let scores = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let labels = [0.0_f32, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (xs, ys) = isotonic_regression(&scores, &labels).expect("pava");
        for w in ys.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "PAVA values must be non-decreasing: {:?}",
                ys
            );
        }
        for w in xs.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "block scores must be non-decreasing");
        }
        // Fitted values must stay within the label range [0, 1].
        for &y in &ys {
            assert!((0.0..=1.0).contains(&y), "fitted value out of range: {y}");
        }
    }

    #[test]
    fn calibrate_is_monotone_after_isotonic_fit() {
        let scores = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let labels = [0.0_f32, 0.0, 1.0, 0.0, 1.0, 1.0];
        let mut cal = RewardModelCalibrator::new();
        cal.fit_isotonic(&scores, &labels).expect("fit isotonic");
        assert!(cal.has_isotonic());
        let mut prev = f32::NEG_INFINITY;
        let mut s = -2.0_f32;
        while s <= 7.0 {
            let p = cal.calibrate(s);
            assert!(
                p >= prev - 1e-6,
                "calibrate must be monotone non-decreasing: s={s}, p={p}, prev={prev}"
            );
            assert!(
                (0.0..=1.0).contains(&p),
                "calibrated prob out of [0,1]: {p}"
            );
            prev = p;
            s += 0.5;
        }
    }

    #[test]
    fn calibrate_falls_back_to_temperature_sigmoid() {
        let mut cal = RewardModelCalibrator::new();
        let (chosen, rejected) = overconfident_pairs();
        let t = cal.fit_temperature(&chosen, &rejected).expect("fit");
        // No isotonic map → calibrate == σ(score / T).
        let expected = sigmoid(2.0 / t);
        assert!((cal.calibrate(2.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn isotonic_recovers_monotone_signal() {
        // Perfectly separable: scores below 3 are negative, above are positive.
        let scores = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let labels = [0.0_f32, 0.0, 0.0, 1.0, 1.0];
        let mut cal = RewardModelCalibrator::new();
        cal.fit_isotonic(&scores, &labels).expect("fit");
        assert!(cal.calibrate(1.5) < 0.5, "low score → low probability");
        assert!(cal.calibrate(4.5) > 0.5, "high score → high probability");
    }

    #[test]
    fn fit_temperature_empty_errors() {
        assert!(matches!(
            fit_temperature_pairs(&[], &[]),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn fit_temperature_mismatch_errors() {
        assert!(matches!(
            fit_temperature_pairs(&[1.0, 2.0], &[1.0]),
            Err(RlhfError::MismatchedPairLength {
                chosen: 2,
                rejected: 1
            })
        ));
    }

    #[test]
    fn fit_temperature_nan_errors() {
        assert!(matches!(
            fit_temperature_pairs(&[f32::NAN], &[0.0]),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn isotonic_mismatch_errors() {
        assert!(matches!(
            isotonic_regression(&[1.0, 2.0, 3.0], &[0.0, 1.0]),
            Err(RlhfError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn ece_invalid_temperature_errors() {
        let (chosen, rejected) = calibrated_pairs();
        assert!(matches!(
            expected_calibration_error(&chosen, &rejected, 0.0, 10),
            Err(RlhfError::InvalidTemp { .. })
        ));
        assert!(matches!(
            expected_calibration_error(&chosen, &rejected, 1.0, 0),
            Err(RlhfError::Internal { .. })
        ));
    }

    #[test]
    fn all_correct_margins_yield_confident_low_temperature() {
        // Every margin positive → optimum at the confident boundary T_MIN.
        let (chosen, rejected) = pairs_from_margins(&[1.0, 2.0, 3.0, 4.0]);
        let t = fit_temperature_pairs(&chosen, &rejected).expect("fit");
        assert!(
            t < 1.0,
            "perfectly separable data → very confident T, got {t}"
        );
    }
}
