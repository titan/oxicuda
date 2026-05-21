//! Split conformal prediction for distribution-free uncertainty.
//!
//! Implements split (inductive) conformal prediction with finite-sample
//! marginal coverage guarantees of `1 − alpha` under exchangeability.
//!
//! References:
//! - Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
//! - Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), "Distribution-Free
//!   Predictive Inference for Regression", *JASA*.
//! - Romano, Patterson & Candès (2019), "Conformalized Quantile Regression",
//!   *NeurIPS* (CQR).
//! - Romano, Sesia & Candès (2020), "Classification with Valid and Adaptive
//!   Coverage", *NeurIPS* (APS).
//!
//! All routines are deterministic. The optional *randomized* APS variant
//! (which removes a small fraction of the boundary class to achieve exact
//! coverage) is documented but not enabled; the deterministic version returns
//! conservative (slightly larger) sets that retain the `1 − alpha` guarantee.

use crate::error::{TabularError, TabularResult};

/// Configuration for split conformal prediction.
#[derive(Debug, Clone, Copy)]
pub struct ConformalConfig {
    /// Target miscoverage rate `alpha ∈ (0, 1)`; coverage is `1 − alpha`.
    pub alpha: f32,
}

impl ConformalConfig {
    /// Create a configuration, validating that `alpha ∈ (0, 1)`.
    pub fn new(alpha: f32) -> TabularResult<Self> {
        validate_alpha(alpha)?;
        Ok(Self { alpha })
    }
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

/// Validate that `alpha` lies strictly in `(0, 1)` and is finite.
fn validate_alpha(alpha: f32) -> TabularResult<()> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(TabularError::InvalidParameter {
            name: "alpha".into(),
            msg: format!("must lie strictly in (0, 1), got {alpha}"),
        });
    }
    Ok(())
}

/// Finite-sample empirical quantile with the conformal `(n + 1)` correction.
///
/// Sorts `scores` ascending and returns the value at
/// `rank = ceil((n + 1) · level)` (one-based), clamped to `[1, n]`.
///
/// This `(n + 1)` correction (rather than the plain empirical quantile) is what
/// gives split conformal prediction its finite-sample marginal coverage
/// guarantee of `level` for exchangeable data.
///
/// # Errors
/// Returns [`TabularError::EmptyInput`] when `scores` is empty, and
/// [`TabularError::InvalidParameter`] when `level` is not in `(0, 1]` or any
/// score is non-finite.
pub fn empirical_quantile(scores: &[f32], level: f32) -> TabularResult<f32> {
    if scores.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if !level.is_finite() || level <= 0.0 || level > 1.0 {
        return Err(TabularError::InvalidParameter {
            name: "level".into(),
            msg: format!("must lie in (0, 1], got {level}"),
        });
    }
    let mut sorted = Vec::with_capacity(scores.len());
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            return Err(TabularError::InvalidParameter {
                name: "scores".into(),
                msg: format!("non-finite score at index {i}: {s}"),
            });
        }
        sorted.push(s);
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    // rank = ceil((n + 1) * level), one-based, clamped to [1, n].
    let raw = ((n as f64 + 1.0) * level as f64).ceil();
    let rank = (raw as usize).clamp(1, n);
    // sorted[rank - 1] is safe: rank ∈ [1, n] ⇒ rank - 1 ∈ [0, n).
    sorted
        .get(rank - 1)
        .copied()
        .ok_or_else(|| TabularError::Internal {
            msg: "quantile rank out of bounds".into(),
        })
}

/// Validate that prediction / target vectors are non-empty and equal length.
fn check_paired(pred: &[f32], target: &[f32]) -> TabularResult<()> {
    if pred.len() != target.len() {
        return Err(TabularError::DimensionMismatch {
            expected: target.len(),
            got: pred.len(),
        });
    }
    if pred.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    Ok(())
}

// ─── Regression ────────────────────────────────────────────────────────────────

/// Split conformal regressor producing symmetric prediction intervals.
///
/// Calibration uses absolute-residual nonconformity scores
/// `s_i = |y_i − ŷ_i|`; the calibrated radius is
/// `q̂ = empirical_quantile(s, 1 − alpha)`. The interval for a new point
/// prediction `ŷ` is `(ŷ − q̂, ŷ + q̂)`.
#[derive(Debug, Clone)]
pub struct SplitConformalRegressor {
    alpha: f32,
    q_hat: f32,
}

impl SplitConformalRegressor {
    /// Calibrate on held-out predictions and targets.
    ///
    /// # Errors
    /// Errors on length mismatch, empty input, or invalid `alpha`.
    pub fn calibrate(
        cfg: ConformalConfig,
        cal_pred: &[f32],
        cal_target: &[f32],
    ) -> TabularResult<Self> {
        validate_alpha(cfg.alpha)?;
        check_paired(cal_pred, cal_target)?;
        let scores: Vec<f32> = cal_pred
            .iter()
            .zip(cal_target.iter())
            .map(|(&p, &y)| (y - p).abs())
            .collect();
        let q_hat = empirical_quantile(&scores, 1.0 - cfg.alpha)?;
        Ok(Self {
            alpha: cfg.alpha,
            q_hat,
        })
    }

    /// The calibrated radius `q̂`.
    #[must_use]
    pub fn q_hat(&self) -> f32 {
        self.q_hat
    }

    /// The configured miscoverage rate `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Symmetric prediction interval `(ŷ − q̂, ŷ + q̂)` for a point prediction.
    #[must_use]
    pub fn predict_interval(&self, point_pred: f32) -> (f32, f32) {
        (point_pred - self.q_hat, point_pred + self.q_hat)
    }
}

// ─── Conformalized Quantile Regression (CQR) ─────────────────────────────────────

/// Conformalized Quantile Regression (Romano et al. 2019).
///
/// Given calibration lower/upper quantile predictions and the target, the
/// conformity score is `E_i = max(q_lo_i − y_i, y_i − q_hi_i)` and the
/// calibrated correction is `q̂ = empirical_quantile(E, 1 − alpha)`. New
/// intervals are `(q_lo − q̂, q_hi + q̂)`.
#[derive(Debug, Clone)]
pub struct ConformalizedQuantileRegressor {
    alpha: f32,
    q_hat: f32,
}

impl ConformalizedQuantileRegressor {
    /// Calibrate on held-out quantile predictions and targets.
    ///
    /// # Errors
    /// Errors on length mismatch, empty input, or invalid `alpha`.
    pub fn calibrate_cqr(
        cfg: ConformalConfig,
        q_lo_cal: &[f32],
        q_hi_cal: &[f32],
        target: &[f32],
    ) -> TabularResult<Self> {
        validate_alpha(cfg.alpha)?;
        if q_lo_cal.len() != q_hi_cal.len() {
            return Err(TabularError::DimensionMismatch {
                expected: q_lo_cal.len(),
                got: q_hi_cal.len(),
            });
        }
        check_paired(q_lo_cal, target)?;
        let scores: Vec<f32> = (0..target.len())
            .map(|i| {
                // Indices in range: all three slices share length, checked above.
                let lo = q_lo_cal.get(i).copied().unwrap_or(0.0);
                let hi = q_hi_cal.get(i).copied().unwrap_or(0.0);
                let y = target.get(i).copied().unwrap_or(0.0);
                (lo - y).max(y - hi)
            })
            .collect();
        let q_hat = empirical_quantile(&scores, 1.0 - cfg.alpha)?;
        Ok(Self {
            alpha: cfg.alpha,
            q_hat,
        })
    }

    /// The calibrated correction `q̂` (may be negative when base quantiles
    /// already over-cover).
    #[must_use]
    pub fn q_hat(&self) -> f32 {
        self.q_hat
    }

    /// The configured miscoverage rate `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// CQR prediction interval `(q_lo − q̂, q_hi + q̂)`.
    #[must_use]
    pub fn predict_interval_cqr(&self, q_lo: f32, q_hi: f32) -> (f32, f32) {
        (q_lo - self.q_hat, q_hi + self.q_hat)
    }
}

// ─── Classification ──────────────────────────────────────────────────────────────

/// Nonconformity score used by the split conformal classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierScore {
    /// Adaptive Prediction Sets (Romano et al. 2020): cumulative sorted-
    /// descending probability mass down to and including the true class.
    Aps,
    /// Least-Ambiguous set-valued / threshold score (Sadinle et al. 2019):
    /// `1 − p_true`. Also known as LAC or THR.
    Lac,
}

/// Split conformal classifier producing distribution-free prediction *sets*.
///
/// Supports two nonconformity scores selectable at calibration time:
/// - [`ClassifierScore::Aps`] — Adaptive Prediction Sets. The score is the
///   cumulative probability mass obtained by adding classes in descending
///   probability order until the true class is included. The set prediction
///   adds classes in descending order until the cumulative mass reaches `q̂`
///   (always non-empty).
/// - [`ClassifierScore::Lac`] — the LAC/THR score `1 − p_true`; a class is in
///   the set iff its probability is `≥ 1 − q̂`.
#[derive(Debug, Clone)]
pub struct SplitConformalClassifier {
    alpha: f32,
    n_classes: usize,
    score: ClassifierScore,
    q_hat: f32,
}

impl SplitConformalClassifier {
    /// Calibrate on held-out class probabilities (`n_samples × n_classes`,
    /// row-major) and integer labels.
    ///
    /// # Errors
    /// Errors on empty input, label/probability shape mismatch, invalid
    /// `alpha`, `n_classes < 1`, or a label outside `[0, n_classes)`.
    pub fn calibrate(
        cfg: ConformalConfig,
        cal_probs: &[f32],
        labels: &[usize],
        n_classes: usize,
        score: ClassifierScore,
    ) -> TabularResult<Self> {
        validate_alpha(cfg.alpha)?;
        if n_classes == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_classes".into(),
                msg: "must be >= 1".into(),
            });
        }
        if labels.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if cal_probs.len() != labels.len() * n_classes {
            return Err(TabularError::DimensionMismatch {
                expected: labels.len() * n_classes,
                got: cal_probs.len(),
            });
        }

        let mut scores = Vec::with_capacity(labels.len());
        for (i, &label) in labels.iter().enumerate() {
            if label >= n_classes {
                return Err(TabularError::LabelOutOfRange { label, n_classes });
            }
            let row = cal_probs
                .get(i * n_classes..(i + 1) * n_classes)
                .ok_or_else(|| TabularError::Internal {
                    msg: "probability row out of bounds".into(),
                })?;
            let s = match score {
                ClassifierScore::Aps => aps_calibration_score(row, label)?,
                ClassifierScore::Lac => {
                    let p_true = row
                        .get(label)
                        .copied()
                        .ok_or_else(|| TabularError::Internal {
                            msg: "true-class probability out of bounds".into(),
                        })?;
                    1.0 - p_true
                }
            };
            scores.push(s);
        }
        let q_hat = empirical_quantile(&scores, 1.0 - cfg.alpha)?;
        Ok(Self {
            alpha: cfg.alpha,
            n_classes,
            score,
            q_hat,
        })
    }

    /// The calibrated threshold `q̂`.
    #[must_use]
    pub fn q_hat(&self) -> f32 {
        self.q_hat
    }

    /// The configured miscoverage rate `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Number of classes the classifier was calibrated for.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// The nonconformity score variant in use.
    #[must_use]
    pub fn score(&self) -> ClassifierScore {
        self.score
    }

    /// Prediction set (sorted ascending by class index) for a probability row.
    ///
    /// For [`ClassifierScore::Aps`] the set is built by adding classes in
    /// descending probability order until the cumulative mass is `≥ q̂`;
    /// it is always non-empty (the top class is always included). For
    /// [`ClassifierScore::Lac`] a class is included iff `p ≥ 1 − q̂`; the
    /// argmax class is included as a fallback to keep the set non-empty.
    ///
    /// # Errors
    /// Errors when `probs.len() != n_classes`.
    pub fn predict_set(&self, probs: &[f32]) -> TabularResult<Vec<usize>> {
        if probs.len() != self.n_classes {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_classes,
                got: probs.len(),
            });
        }
        let order = descending_order(probs);
        let mut set = match self.score {
            ClassifierScore::Aps => {
                let mut cumulative = 0.0_f32;
                let mut chosen = Vec::new();
                for &c in &order {
                    let p = probs.get(c).copied().unwrap_or(0.0);
                    cumulative += p;
                    chosen.push(c);
                    if cumulative >= self.q_hat {
                        break;
                    }
                }
                chosen
            }
            ClassifierScore::Lac => {
                let threshold = 1.0 - self.q_hat;
                let mut chosen: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&c| probs.get(c).copied().unwrap_or(0.0) >= threshold)
                    .collect();
                // Guarantee non-emptiness: include the argmax class.
                if chosen.is_empty()
                    && let Some(&top) = order.first()
                {
                    chosen.push(top);
                }
                chosen
            }
        };
        set.sort_unstable();
        Ok(set)
    }
}

/// APS calibration score: cumulative descending probability mass down to and
/// including the true class.
fn aps_calibration_score(row: &[f32], true_label: usize) -> TabularResult<f32> {
    let order = descending_order(row);
    let mut cumulative = 0.0_f32;
    for &c in &order {
        cumulative += row.get(c).copied().unwrap_or(0.0);
        if c == true_label {
            return Ok(cumulative);
        }
    }
    Err(TabularError::Internal {
        msg: "true label not found while scoring APS row".into(),
    })
}

/// Return class indices sorted by descending probability (stable on ties by
/// preferring the smaller index, so behaviour is fully deterministic).
fn descending_order(probs: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        let pa = probs.get(a).copied().unwrap_or(0.0);
        let pb = probs.get(b).copied().unwrap_or(0.0);
        pb.partial_cmp(&pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── empirical_quantile finite-sample rank ────────────────────────────────
    #[test]
    fn empirical_quantile_finite_sample_rank() {
        // n = 9, level = 1 - 0.1 = 0.9 ⇒ rank = ceil(10 * 0.9) = 9 ⇒ the max.
        let scores: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let q = empirical_quantile(&scores, 0.9).unwrap();
        assert!((q - 9.0).abs() < 1e-6, "q={q}");
    }

    #[test]
    fn empirical_quantile_rank_clamped_to_n() {
        // level = 1.0 ⇒ raw rank = n + 1, clamped to n ⇒ the max score.
        let scores = vec![3.0_f32, 1.0, 2.0, 5.0, 4.0];
        let q = empirical_quantile(&scores, 1.0).unwrap();
        assert!((q - 5.0).abs() < 1e-6, "q={q}");
    }

    #[test]
    fn empirical_quantile_monotone_in_level() {
        let scores = vec![5.0_f32, 1.0, 3.0, 2.0, 4.0, 6.0, 8.0, 7.0];
        let levels = [0.1_f32, 0.25, 0.5, 0.75, 0.9, 1.0];
        let mut prev = f32::NEG_INFINITY;
        for &lvl in &levels {
            let q = empirical_quantile(&scores, lvl).unwrap();
            assert!(q >= prev - 1e-6, "quantile not monotone: {q} < {prev}");
            prev = q;
        }
    }

    #[test]
    fn empirical_quantile_empty_errs() {
        assert!(empirical_quantile(&[], 0.9).is_err());
    }

    #[test]
    fn empirical_quantile_bad_level_errs() {
        let scores = vec![1.0_f32, 2.0, 3.0];
        assert!(empirical_quantile(&scores, 0.0).is_err());
        assert!(empirical_quantile(&scores, 1.5).is_err());
    }

    // ── Regression coverage (LOAD-BEARING) ───────────────────────────────────
    #[test]
    fn regression_marginal_coverage() {
        // Exchangeable synthetic: residuals are i.i.d. N(0, sigma). The model
        // prediction is the true mean, so |y - yhat| ~ |N(0, sigma)|.
        let mut rng = LcgRng::new(2024);
        let n_cal = 2000usize;
        let n_test = 2000usize;
        let sigma = 1.3_f32;
        let alpha = 0.1_f32;

        let mut cal_target = vec![0.0_f32; n_cal];
        rng.fill_normal_scaled(&mut cal_target, sigma);
        let cal_pred = vec![0.0_f32; n_cal]; // model predicts the mean (0)

        let cfg = ConformalConfig::new(alpha).unwrap();
        let reg = SplitConformalRegressor::calibrate(cfg, &cal_pred, &cal_target).unwrap();

        let mut covered = 0usize;
        for _ in 0..n_test {
            let (a, _) = rng.next_normal_pair();
            let y = a * sigma;
            let (lo, hi) = reg.predict_interval(0.0);
            if y >= lo && y <= hi {
                covered += 1;
            }
        }
        let coverage = covered as f32 / n_test as f32;
        // Target 0.9; allow a tolerance band for finite-sample noise.
        assert!(
            (coverage - (1.0 - alpha)).abs() < 0.04,
            "coverage={coverage}, expected ~{}",
            1.0 - alpha
        );
    }

    #[test]
    fn regression_interval_symmetric() {
        let cfg = ConformalConfig::new(0.1).unwrap();
        let pred = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let target = vec![1.5_f32, 1.5, 3.5, 3.5, 5.5];
        let reg = SplitConformalRegressor::calibrate(cfg, &pred, &target).unwrap();
        let center = 7.0_f32;
        let (lo, hi) = reg.predict_interval(center);
        assert!(
            ((center - lo) - (hi - center)).abs() < 1e-6,
            "interval not symmetric: lo={lo}, hi={hi}"
        );
        assert!((0.5 * (lo + hi) - center).abs() < 1e-6);
    }

    #[test]
    fn regression_width_grows_as_alpha_decreases() {
        let mut rng = LcgRng::new(7);
        let n = 1000;
        let mut target = vec![0.0_f32; n];
        rng.fill_normal_scaled(&mut target, 2.0);
        let pred = vec![0.0_f32; n];

        let mut prev_width = -1.0_f32;
        // Smaller alpha ⇒ higher coverage ⇒ wider interval.
        for &alpha in &[0.5_f32, 0.2, 0.1, 0.05] {
            let cfg = ConformalConfig::new(alpha).unwrap();
            let reg = SplitConformalRegressor::calibrate(cfg, &pred, &target).unwrap();
            let width = 2.0 * reg.q_hat();
            assert!(
                width >= prev_width - 1e-6,
                "width should grow as alpha shrinks: {width} < {prev_width}"
            );
            prev_width = width;
        }
    }

    #[test]
    fn regression_near_perfect_predictor_tiny_interval() {
        let pred = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // Targets essentially equal to predictions.
        let target: Vec<f32> = pred.iter().map(|&p| p + 1e-4).collect();
        let cfg = ConformalConfig::new(0.1).unwrap();
        let reg = SplitConformalRegressor::calibrate(cfg, &pred, &target).unwrap();
        assert!(reg.q_hat() < 1e-3, "q_hat={}", reg.q_hat());
    }

    #[test]
    fn regression_errs_on_bad_input() {
        let cfg = ConformalConfig::new(0.1).unwrap();
        assert!(SplitConformalRegressor::calibrate(cfg, &[1.0], &[1.0, 2.0]).is_err());
        assert!(SplitConformalRegressor::calibrate(cfg, &[], &[]).is_err());
        assert!(ConformalConfig::new(0.0).is_err());
        assert!(ConformalConfig::new(1.0).is_err());
        assert!(ConformalConfig::new(-0.2).is_err());
    }

    // ── CQR ──────────────────────────────────────────────────────────────────
    #[test]
    fn cqr_interval_formula() {
        let q_lo = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let q_hi = vec![2.0_f32, 3.0, 4.0, 5.0, 6.0];
        // Targets inside bands ⇒ E_i negative (over-coverage).
        let target = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let cfg = ConformalConfig::new(0.2).unwrap();
        let cqr =
            ConformalizedQuantileRegressor::calibrate_cqr(cfg, &q_lo, &q_hi, &target).unwrap();
        let q = cqr.q_hat();
        let (lo, hi) = cqr.predict_interval_cqr(10.0, 12.0);
        assert!((lo - (10.0 - q)).abs() < 1e-6, "lo={lo}, q={q}");
        assert!((hi - (12.0 + q)).abs() < 1e-6, "hi={hi}, q={q}");
    }

    #[test]
    fn cqr_marginal_coverage() {
        // Base quantile predictor is intentionally slightly too narrow; CQR
        // should widen it to reach ~1 - alpha coverage.
        let mut rng = LcgRng::new(321);
        let n_cal = 2000usize;
        let n_test = 2000usize;
        let sigma = 1.0_f32;
        let alpha = 0.1_f32;
        // Narrow band: +/- 1.0 sigma (nominal ~68% before conformalization).
        let band = 1.0_f32;

        let mut cal_target = vec![0.0_f32; n_cal];
        rng.fill_normal_scaled(&mut cal_target, sigma);
        let q_lo_cal = vec![-band; n_cal];
        let q_hi_cal = vec![band; n_cal];

        let cfg = ConformalConfig::new(alpha).unwrap();
        let cqr =
            ConformalizedQuantileRegressor::calibrate_cqr(cfg, &q_lo_cal, &q_hi_cal, &cal_target)
                .unwrap();

        let mut covered = 0usize;
        for _ in 0..n_test {
            let (a, _) = rng.next_normal_pair();
            let y = a * sigma;
            let (lo, hi) = cqr.predict_interval_cqr(-band, band);
            if y >= lo && y <= hi {
                covered += 1;
            }
        }
        let coverage = covered as f32 / n_test as f32;
        assert!(
            (coverage - (1.0 - alpha)).abs() < 0.04,
            "cqr coverage={coverage}"
        );
    }

    #[test]
    fn cqr_errs_on_bad_input() {
        let cfg = ConformalConfig::new(0.1).unwrap();
        assert!(
            ConformalizedQuantileRegressor::calibrate_cqr(cfg, &[0.0, 1.0], &[1.0], &[0.5, 0.5])
                .is_err()
        );
        assert!(ConformalizedQuantileRegressor::calibrate_cqr(cfg, &[], &[], &[]).is_err());
    }

    // ── APS classifier ─────────────────────────────────────────────────────────
    fn synthetic_probs(rng: &mut LcgRng, k: usize, true_label: usize, sharpness: f32) -> Vec<f32> {
        // Build logits, boost the true class, softmax.
        let mut logits = vec![0.0_f32; k];
        for l in logits.iter_mut() {
            let (a, _) = rng.next_normal_pair();
            *l = a;
        }
        if let Some(slot) = logits.get_mut(true_label) {
            *slot += sharpness;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in exps.iter_mut() {
            *e /= sum;
        }
        exps
    }

    #[test]
    fn aps_set_always_non_empty() {
        let mut rng = LcgRng::new(11);
        let k = 5;
        let n_cal = 1500usize;
        let mut probs = Vec::with_capacity(n_cal * k);
        let mut labels = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 2.0);
            probs.extend_from_slice(&row);
            labels.push(y);
        }
        let cfg = ConformalConfig::new(0.1).unwrap();
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Aps)
                .unwrap();

        for _ in 0..200 {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 2.0);
            let set = clf.predict_set(&row).unwrap();
            assert!(!set.is_empty(), "APS set must be non-empty");
        }
    }

    #[test]
    fn aps_contains_argmax_when_qhat_above_top() {
        // Construct a classifier with q_hat >= top prob; argmax must be in set.
        let cfg = ConformalConfig::new(0.1).unwrap();
        // All calibration rows have the true class as a low-probability class so
        // that scores (and hence q_hat) are large.
        let k = 3;
        // probs row: argmax is class 0 but the true label is the low one.
        let probs = vec![
            0.8_f32, 0.15, 0.05, // true label 2 (lowest)
            0.7, 0.2, 0.1, // true label 2
            0.6, 0.3, 0.1, // true label 2
            0.9, 0.05, 0.05, // true label 1
        ];
        let labels = vec![2usize, 2, 2, 1];
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Aps)
                .unwrap();
        let row = vec![0.7_f32, 0.2, 0.1];
        let argmax = 0usize;
        let set = clf.predict_set(&row).unwrap();
        assert!(set.contains(&argmax), "set {set:?} should contain argmax");
    }

    #[test]
    fn aps_higher_coverage_larger_sets() {
        let mut rng = LcgRng::new(99);
        let k = 6;
        let n_cal = 1500usize;
        let mut probs = Vec::with_capacity(n_cal * k);
        let mut labels = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 1.0);
            probs.extend_from_slice(&row);
            labels.push(y);
        }

        // Fixed evaluation rows.
        let mut eval_rng = LcgRng::new(424242);
        let mut eval_rows = Vec::new();
        for _ in 0..300 {
            let y = eval_rng.next_usize(k);
            eval_rows.push(synthetic_probs(&mut eval_rng, k, y, 1.0));
        }

        let avg_size = |alpha: f32| -> f32 {
            let cfg = ConformalConfig::new(alpha).unwrap();
            let clf =
                SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Aps)
                    .unwrap();
            let total: usize = eval_rows
                .iter()
                .map(|r| clf.predict_set(r).map(|s| s.len()).unwrap_or(0))
                .sum();
            total as f32 / eval_rows.len() as f32
        };

        let size_low_cov = avg_size(0.3); // coverage 0.7
        let size_high_cov = avg_size(0.05); // coverage 0.95
        assert!(
            size_high_cov >= size_low_cov - 1e-6,
            "higher coverage should give larger sets: {size_high_cov} vs {size_low_cov}"
        );
    }

    #[test]
    fn aps_marginal_coverage() {
        let mut rng = LcgRng::new(2026);
        let k = 4;
        let n_cal = 2000usize;
        let n_test = 2000usize;
        let alpha = 0.1_f32;

        let mut probs = Vec::with_capacity(n_cal * k);
        let mut labels = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 1.5);
            probs.extend_from_slice(&row);
            labels.push(y);
        }
        let cfg = ConformalConfig::new(alpha).unwrap();
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Aps)
                .unwrap();

        let mut covered = 0usize;
        for _ in 0..n_test {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 1.5);
            let set = clf.predict_set(&row).unwrap();
            if set.contains(&y) {
                covered += 1;
            }
        }
        let coverage = covered as f32 / n_test as f32;
        // APS is conservative; coverage should be at least ~1 - alpha (with band).
        assert!(
            coverage >= (1.0 - alpha) - 0.04,
            "aps coverage={coverage}, expected >= {}",
            1.0 - alpha
        );
    }

    #[test]
    fn lac_score_is_one_minus_p_true() {
        // With LAC, calibration score = 1 - p_true; verify via reconstructing
        // q_hat from a tiny hand-checkable example.
        // n = 9 scores, alpha = 0.1 ⇒ rank 9 ⇒ max score = 1 - min p_true.
        let cfg = ConformalConfig::new(0.1).unwrap();
        let k = 2;
        // p_true values: 0.9, 0.8, ..., 0.1 for 9 samples (true label = class 0).
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..9 {
            let p_true = 0.9 - 0.1 * i as f32;
            probs.push(p_true);
            probs.push(1.0 - p_true);
            labels.push(0usize);
        }
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Lac)
                .unwrap();
        // Max (1 - p_true) over scores = 1 - 0.1 (approx) = 0.9.
        assert!(
            (clf.q_hat() - 0.9).abs() < 1e-5,
            "lac q_hat={}",
            clf.q_hat()
        );
    }

    #[test]
    fn lac_predict_set_threshold() {
        let cfg = ConformalConfig::new(0.1).unwrap();
        let k = 3;
        // Sharp calibration: true class always ~0.95 ⇒ small q_hat.
        let n = 20usize;
        let mut probs = Vec::new();
        for _ in 0..n {
            probs.extend_from_slice(&[0.95_f32, 0.03, 0.02]);
        }
        let labels = vec![0usize; n];
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Lac)
                .unwrap();
        // q_hat ~ 0.05, threshold ~ 0.95; only very confident classes survive.
        let confident = clf.predict_set(&[0.96_f32, 0.02, 0.02]).unwrap();
        assert_eq!(confident, vec![0usize]);
        // Non-empty fallback even when nothing clears the threshold.
        let ambiguous = clf.predict_set(&[0.4_f32, 0.35, 0.25]).unwrap();
        assert!(!ambiguous.is_empty());
    }

    #[test]
    fn classifier_errs_on_bad_input() {
        let cfg = ConformalConfig::new(0.1).unwrap();
        // n_classes mismatch in calibration shape.
        assert!(
            SplitConformalClassifier::calibrate(
                cfg,
                &[0.5, 0.5, 0.5],
                &[0, 1],
                2,
                ClassifierScore::Aps
            )
            .is_err()
        );
        // empty calibration.
        assert!(
            SplitConformalClassifier::calibrate(cfg, &[], &[], 3, ClassifierScore::Aps).is_err()
        );
        // label out of range.
        assert!(
            SplitConformalClassifier::calibrate(cfg, &[0.5, 0.5], &[5], 2, ClassifierScore::Aps)
                .is_err()
        );
        // n_classes == 0.
        assert!(
            SplitConformalClassifier::calibrate(cfg, &[], &[0], 0, ClassifierScore::Aps).is_err()
        );
        // predict_set with wrong width.
        let clf = SplitConformalClassifier::calibrate(
            cfg,
            &[0.6, 0.4, 0.7, 0.3],
            &[0, 0],
            2,
            ClassifierScore::Aps,
        )
        .unwrap();
        assert!(clf.predict_set(&[0.5, 0.3, 0.2]).is_err());
    }

    #[test]
    fn aps_set_contains_argmax_when_qhat_small() {
        // Sharp predictor ⇒ small q_hat ⇒ APS returns just the argmax.
        let mut rng = LcgRng::new(8);
        let k = 4;
        let n_cal = 1000usize;
        let mut probs = Vec::with_capacity(n_cal * k);
        let mut labels = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            let y = rng.next_usize(k);
            let row = synthetic_probs(&mut rng, k, y, 6.0); // very sharp
            probs.extend_from_slice(&row);
            labels.push(y);
        }
        let cfg = ConformalConfig::new(0.1).unwrap();
        let clf =
            SplitConformalClassifier::calibrate(cfg, &probs, &labels, k, ClassifierScore::Aps)
                .unwrap();
        let row = vec![0.97_f32, 0.01, 0.01, 0.01];
        let set = clf.predict_set(&row).unwrap();
        assert!(set.contains(&0usize));
    }
}
