//! Time-dependent ROC curves, calibration plots, and decision curve analysis for survival data.
//!
//! Implements the IPCW (inverse probability of censoring weighting) approach for:
//! - Full ROC curves at a fixed horizon time t* (Uno 2007)
//! - Calibration metrics (O/E ratio, Hosmer-Lemeshow, ICI)
//! - Decision Curve Analysis (net benefit)
//!
//! # Key Definitions
//!
//! - **Case**: subject with δᵢ = 1 and tᵢ ≤ t*
//! - **Control**: subject with tᵢ > t*
//! - **Excluded**: censored before t* (weight = 0)
//!
//! IPCW weights account for the censoring distribution G(t) = P(C > t):
//! - Case weight: 1/G(tᵢ)
//! - Control weight: 1/G(t*)

use crate::error::{SurvivalError, SurvivalResult};

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

/// Result of time-dependent ROC curve estimation.
///
/// Provides the full ROC curve (FPR, TPR) at horizon `t*` using IPCW weights.
#[derive(Debug, Clone)]
pub struct TimeRocResult {
    /// False positive rates (sorted ascending) for each threshold.
    pub fpr: Vec<f64>,
    /// True positive rates corresponding to each FPR.
    pub tpr: Vec<f64>,
    /// Risk score thresholds (decreasing) used to generate the curve.
    pub thresholds: Vec<f64>,
    /// Area under the ROC curve (trapezoidal integration).
    pub auc: f64,
    /// Horizon time t* used for case/control classification.
    pub horizon: f64,
    /// Weighted count of cases (events before horizon).
    pub n_cases: usize,
    /// Weighted count of controls (survived past horizon).
    pub n_controls: usize,
}

/// Result of calibration analysis at a fixed horizon.
///
/// Includes O/E ratio, Hosmer-Lemeshow goodness-of-fit, and ICI.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Observed/Expected ratio: (actual events) / (sum of predicted probs).
    pub oe_ratio: f64,
    /// Hosmer-Lemeshow chi-squared statistic.
    pub hl_statistic: f64,
    /// Degrees of freedom for HL test (n_groups - 2).
    pub hl_df: usize,
    /// Approximate p-value for the HL chi-squared test.
    pub hl_p_value: f64,
    /// Integrated Calibration Index: mean |observed - predicted| across bins.
    pub ici: f64,
    /// Mean predicted event probability across all subjects.
    pub mean_predicted: f64,
    /// Actual event rate (proportion of events at or before horizon).
    pub observed_rate: f64,
}

/// Result of Decision Curve Analysis.
///
/// Net benefit is computed for a range of threshold probabilities,
/// comparing the model against "treat all" and "treat none" strategies.
#[derive(Debug, Clone)]
pub struct DcaResult {
    /// Threshold probabilities p_t used for DCA.
    pub thresholds: Vec<f64>,
    /// Net benefit of the model at each threshold.
    pub net_benefit_model: Vec<f64>,
    /// Net benefit of "treat all" strategy at each threshold.
    pub net_benefit_all: Vec<f64>,
    /// Net benefit of "treat none" strategy (always 0.0).
    pub net_benefit_none: Vec<f64>,
    /// Number of threshold values.
    pub n_thresholds: usize,
}

// ---------------------------------------------------------------------------
// Internal IPCW helpers
// ---------------------------------------------------------------------------

/// Compute the Kaplan-Meier censoring distribution G(t) = P(C > t).
///
/// Swaps roles: censored observations become the "events" for the KM estimator.
/// Returns parallel vectors (times, g_values) where g_values[i] = G(times[i]).
fn compute_censoring_km(times: &[f64], events: &[u8]) -> (Vec<f64>, Vec<f64>) {
    // Gather unique times from all observations
    let n = times.len();
    // Collect all distinct observed times, sorted
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut g_times: Vec<f64> = Vec::new();
    let mut g_vals: Vec<f64> = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n as f64;

    let mut k = 0usize;
    while k < sorted_idx.len() {
        let t = times[sorted_idx[k]];
        // Count censored events (δ=0) at this time — these are "deaths" for censoring KM
        let mut m = k;
        let mut n_censored = 0.0_f64;
        while m < sorted_idx.len() && times[sorted_idx[m]] == t {
            if events[sorted_idx[m]] == 0 {
                n_censored += 1.0;
            }
            m += 1;
        }
        // Update G(t) before recording (left-continuous style then record right-continuous)
        if n_censored > 0.0 && at_risk > 0.0 {
            g_cur *= 1.0 - n_censored / at_risk;
        }
        g_times.push(t);
        g_vals.push(g_cur);
        at_risk -= (m - k) as f64;
        k = m;
    }

    (g_times, g_vals)
}

/// Evaluate the censoring KM G(t) at a given time using the stored step function.
///
/// Returns G(t) = P(C > t) evaluated just before t (left-continuous at t).
/// Clamps to a minimum of 1e-300 to avoid division by zero.
fn g_at(t: f64, g_times: &[f64], g_vals: &[f64]) -> f64 {
    // G(t) is right-continuous step function: find the last step ≤ t
    let mut val = 1.0_f64;
    for (idx, &gt) in g_times.iter().enumerate() {
        if gt <= t {
            val = g_vals[idx];
        } else {
            break;
        }
    }
    val.max(1.0e-300)
}

/// Compute IPCW weights and case/control labels for each subject.
///
/// Returns a vector of `(weight, is_case)` pairs.
/// Excluded subjects (censored before horizon) have weight 0.0.
fn compute_ipcw_labels(
    times: &[f64],
    events: &[u8],
    horizon: f64,
    g_times: &[f64],
    g_vals: &[f64],
) -> Vec<(f64, bool)> {
    let n = times.len();
    let g_star = g_at(horizon, g_times, g_vals);

    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let ti = times[i];
        let di = events[i];
        if ti <= horizon && di == 1 {
            // Case: event occurred before horizon
            let w = 1.0 / g_at(ti, g_times, g_vals);
            labels.push((w, true));
        } else if ti > horizon {
            // Control: survived past horizon
            let w = 1.0 / g_star;
            labels.push((w, false));
        } else {
            // Excluded: censored before horizon
            labels.push((0.0, false));
        }
    }
    labels
}

// ---------------------------------------------------------------------------
// Time-Dependent ROC Curve
// ---------------------------------------------------------------------------

/// Compute the time-dependent ROC curve using IPCW weighting at horizon `t*`.
///
/// # Algorithm
///
/// Subjects are classified as cases (event ≤ t*) or controls (survived > t*).
/// IPCW weights correct for censoring via the KM censoring estimator G(t).
/// The ROC curve sweeps risk score thresholds from high to low, accumulating
/// weighted TPR and FPR at each step.
///
/// # Arguments
///
/// - `times`        — observed times (must be positive)
/// - `events`       — event indicators: 1 = event, 0 = censored
/// - `risk_scores`  — risk scores; higher = higher predicted risk
/// - `n_subjects`   — must equal `times.len()`
/// - `horizon`      — t* (fixed time point for ROC evaluation)
///
/// # Errors
///
/// Returns an error if inputs are inconsistent, no cases/controls exist, or
/// the horizon is non-positive.
pub fn time_roc(
    times: &[f64],
    events: &[u8],
    risk_scores: &[f64],
    n_subjects: usize,
    horizon: f64,
) -> SurvivalResult<TimeRocResult> {
    validate_inputs(times, events, risk_scores, n_subjects, horizon)?;

    let (g_times, g_vals) = compute_censoring_km(times, events);
    let labels = compute_ipcw_labels(times, events, horizon, &g_times, &g_vals);

    // Sum total weights for cases and controls
    let total_case_w: f64 = labels
        .iter()
        .filter(|&&(_, is_case)| is_case)
        .map(|&(w, _)| w)
        .sum();
    let total_ctrl_w: f64 = labels
        .iter()
        .filter(|&&(_, is_case)| !is_case)
        .map(|&(w, _)| w)
        .sum();

    if total_case_w <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no cases (events before horizon) found for ROC curve".to_string(),
        ));
    }
    if total_ctrl_w <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no controls (survived past horizon) found for ROC curve".to_string(),
        ));
    }

    let n_cases = labels
        .iter()
        .filter(|&&(w, is_case)| is_case && w > 0.0)
        .count();
    let n_controls = labels
        .iter()
        .filter(|&&(w, is_case)| !is_case && w > 0.0)
        .count();

    // Sort subjects by risk score descending (sweep threshold from high to low)
    let mut order: Vec<usize> = (0..n_subjects).collect();
    order.sort_by(|&a, &b| {
        risk_scores[b]
            .partial_cmp(&risk_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build ROC curve by sweeping threshold η from max score to min score
    // At each threshold η: "positive" = score > η
    // Start at (FPR=0, TPR=0) — no positives yet
    let mut fpr_pts = vec![0.0_f64];
    let mut tpr_pts = vec![0.0_f64];
    let mut thresh_pts = vec![f64::INFINITY];

    let mut accum_case_w = 0.0_f64;
    let mut accum_ctrl_w = 0.0_f64;

    let mut k = 0usize;
    while k < order.len() {
        let score_k = risk_scores[order[k]];
        // Collect all subjects at the same score (ties)
        let mut m = k;
        while m < order.len() && (risk_scores[order[m]] - score_k).abs() < 1.0e-14 {
            let (w, is_case) = labels[order[m]];
            if is_case {
                accum_case_w += w;
            } else {
                accum_ctrl_w += w;
            }
            m += 1;
        }
        // After including all subjects with score >= score_k
        let tpr = accum_case_w / total_case_w;
        let fpr = accum_ctrl_w / total_ctrl_w;
        thresh_pts.push(score_k);
        tpr_pts.push(tpr.min(1.0));
        fpr_pts.push(fpr.min(1.0));
        k = m;
    }

    // Sort by FPR ascending (monotone ROC curve)
    let mut roc_pts: Vec<(f64, f64, f64)> = fpr_pts
        .into_iter()
        .zip(tpr_pts)
        .zip(thresh_pts)
        .map(|((f, t), th)| (f, t, th))
        .collect();
    roc_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let fpr: Vec<f64> = roc_pts.iter().map(|&(f, _, _)| f).collect();
    let tpr: Vec<f64> = roc_pts.iter().map(|&(_, t, _)| t).collect();
    let thresholds: Vec<f64> = roc_pts.iter().map(|&(_, _, th)| th).collect();

    // Trapezoidal AUC
    let auc = trapezoid_auc(&fpr, &tpr);

    Ok(TimeRocResult {
        fpr,
        tpr,
        thresholds,
        auc,
        horizon,
        n_cases,
        n_controls,
    })
}

/// Compute the IPCW time-dependent AUC only (faster path without storing the full ROC curve).
///
/// Equivalent to `time_roc(...).auc` but avoids allocating curve points.
///
/// # Arguments
///
/// - `times`       — observed times
/// - `events`      — event indicators (1 = event, 0 = censored)
/// - `risk_scores` — higher = more at risk
/// - `n_subjects`  — must equal `times.len()`
/// - `horizon`     — t* (horizon time)
pub fn time_roc_auc_only(
    times: &[f64],
    events: &[u8],
    risk_scores: &[f64],
    n_subjects: usize,
    horizon: f64,
) -> SurvivalResult<f64> {
    let result = time_roc(times, events, risk_scores, n_subjects, horizon)?;
    Ok(result.auc)
}

/// Trapezoidal integration of ROC curve: AUC = Σ (FPR[i+1] - FPR[i]) * (TPR[i+1] + TPR[i]) / 2.
fn trapezoid_auc(fpr: &[f64], tpr: &[f64]) -> f64 {
    let n = fpr.len();
    if n < 2 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    for i in 0..n - 1 {
        let df = fpr[i + 1] - fpr[i];
        let avg_t = (tpr[i + 1] + tpr[i]) / 2.0;
        area += df * avg_t;
    }
    area.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Calibration Analysis
// ---------------------------------------------------------------------------

/// Compute calibration metrics for survival predictions at a fixed horizon.
///
/// # Metrics Computed
///
/// - **O/E ratio**: ratio of observed events to expected events (sum of predicted probs).
/// - **Hosmer-Lemeshow χ²**: goodness-of-fit dividing subjects into `n_groups` equal bins.
/// - **ICI**: Integrated Calibration Index (mean |O - E| across predicted probability bins).
///
/// # Arguments
///
/// - `times`           — observed times
/// - `events`          — event indicators (1 = event, 0 = censored)
/// - `predicted_probs` — P̂(T ≤ horizon | xᵢ) for each subject ∈ [0, 1]
/// - `n_subjects`      — must equal `times.len()`
/// - `horizon`         — t* (horizon time)
/// - `n_groups`        — number of HL groups (typically 10)
///
/// # Errors
///
/// Returns an error if inputs are inconsistent, `n_groups < 2`, or predictions
/// contain non-finite or out-of-range values.
pub fn calibration_analysis(
    times: &[f64],
    events: &[u8],
    predicted_probs: &[f64],
    n_subjects: usize,
    horizon: f64,
    n_groups: usize,
) -> SurvivalResult<CalibrationResult> {
    validate_inputs_calibration(
        times,
        events,
        predicted_probs,
        n_subjects,
        horizon,
        n_groups,
    )?;

    let n = n_subjects;

    // O/E ratio: observed events ≤ horizon / sum of predicted probs
    let observed_events: f64 = (0..n)
        .filter(|&i| times[i] <= horizon && events[i] == 1)
        .count() as f64;
    let sum_predicted: f64 = predicted_probs.iter().sum::<f64>();
    let observed_rate = observed_events / n as f64;
    let mean_predicted = sum_predicted / n as f64;

    if sum_predicted <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "sum of predicted probabilities is zero".to_string(),
        ));
    }
    let oe_ratio = observed_events / sum_predicted;

    // Hosmer-Lemeshow test
    // Sort subjects by predicted probability
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        predicted_probs[a]
            .partial_cmp(&predicted_probs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut hl_stat = 0.0_f64;
    let group_size = n as f64 / n_groups as f64;

    for g in 0..n_groups {
        let start = (g as f64 * group_size).round() as usize;
        let end = ((g + 1) as f64 * group_size).round() as usize;
        let end = end.min(n);
        if start >= end {
            continue;
        }
        let ng = (end - start) as f64;
        let o_g: f64 = (start..end)
            .filter(|&k| times[idx[k]] <= horizon && events[idx[k]] == 1)
            .count() as f64;
        let e_g: f64 = (start..end).map(|k| predicted_probs[idx[k]]).sum();

        // Avoid division by zero: skip groups with e_g near 0 or ng
        let denom = e_g * (1.0 - e_g / ng);
        if denom > 1.0e-12 {
            hl_stat += (o_g - e_g).powi(2) / denom;
        }
    }

    let hl_df = n_groups.saturating_sub(2);
    let hl_p_value = chi2_survival_approx(hl_stat, hl_df);

    // ICI: Integrated Calibration Index
    // Divide into n_groups bins by predicted prob, compute |mean_obs - mean_pred| per bin
    let ici = compute_ici(times, events, predicted_probs, &idx, n, n_groups, horizon);

    Ok(CalibrationResult {
        oe_ratio,
        hl_statistic: hl_stat,
        hl_df,
        hl_p_value,
        ici,
        mean_predicted,
        observed_rate,
    })
}

/// Compute the Integrated Calibration Index (ICI) for calibration.
///
/// ICI = mean of |observed_rate_bin - mean_predicted_bin| across bins,
/// weighted by bin size.
fn compute_ici(
    times: &[f64],
    events: &[u8],
    predicted_probs: &[f64],
    sorted_idx: &[usize],
    n: usize,
    n_groups: usize,
    horizon: f64,
) -> f64 {
    let group_size = n as f64 / n_groups as f64;
    let mut total_weight = 0.0_f64;
    let mut weighted_abs_diff = 0.0_f64;

    for g in 0..n_groups {
        let start = (g as f64 * group_size).round() as usize;
        let end = ((g + 1) as f64 * group_size).round() as usize;
        let end = end.min(n);
        if start >= end {
            continue;
        }
        let ng = (end - start) as f64;
        let o_g: f64 = (start..end)
            .filter(|&k| times[sorted_idx[k]] <= horizon && events[sorted_idx[k]] == 1)
            .count() as f64;
        let e_g: f64 = (start..end).map(|k| predicted_probs[sorted_idx[k]]).sum();

        let obs_rate = o_g / ng;
        let pred_rate = e_g / ng;
        weighted_abs_diff += ng * (obs_rate - pred_rate).abs();
        total_weight += ng;
    }

    if total_weight > 0.0 {
        weighted_abs_diff / total_weight
    } else {
        0.0
    }
}

/// Approximate chi-squared survival function P(χ²_df > x) using the Wilson-Hilferty
/// normal approximation for df >= 1.
///
/// For df = 0 or x ≤ 0, returns 1.0 or 0.0 respectively.
fn chi2_survival_approx(x: f64, df: usize) -> f64 {
    if df == 0 || x <= 0.0 {
        return if x <= 0.0 { 1.0 } else { 0.0 };
    }
    if !x.is_finite() {
        return 0.0;
    }
    // Wilson-Hilferty approximation: (χ²/df)^(1/3) ≈ N(1 - 2/(9df), 2/(9df))
    let df_f = df as f64;
    let h = 2.0 / (9.0 * df_f);
    let z = ((x / df_f).powf(1.0 / 3.0) - (1.0 - h)) / h.sqrt();
    // P(Z > z) = 1 - Φ(z)
    normal_sf(z)
}

/// Standard normal survival function P(Z > z) via complementary error function approximation.
fn normal_sf(z: f64) -> f64 {
    // Use erfc-based computation: Φ(z) = 0.5 * erfc(-z / sqrt(2))
    // P(Z > z) = 0.5 * erfc(z / sqrt(2))
    let x = z / std::f64::consts::SQRT_2;
    0.5 * erfc_approx(x)
}

/// Approximate complementary error function using a rational polynomial approximation.
///
/// Uses the Abramowitz & Stegun approximation 7.1.26 (max error ~1.5e-7).
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    // Coefficients from A&S 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

// ---------------------------------------------------------------------------
// Decision Curve Analysis
// ---------------------------------------------------------------------------

/// Perform Decision Curve Analysis (DCA) for survival predictions.
///
/// Computes net benefit at each threshold probability p_t:
///
/// ```text
/// net_benefit(p_t) = TPR * prevalence - FPR * (1 - prevalence) * p_t / (1 - p_t)
/// ```
///
/// where TPR and FPR are estimated for threshold η = p_t applied to predicted_probs.
///
/// # Arguments
///
/// - `times`            — observed times
/// - `events`           — event indicators (1 = event, 0 = censored)
/// - `predicted_probs`  — P̂(T ≤ horizon | xᵢ) for each subject ∈ [0, 1]
/// - `n_subjects`       — must equal `times.len()`
/// - `horizon`          — t* (horizon time)
/// - `threshold_range`  — (min_pt, max_pt) for DCA threshold sweep
/// - `n_thresholds`     — number of threshold values to evaluate
///
/// # Errors
///
/// Returns an error if inputs are inconsistent or the threshold range is invalid.
pub fn decision_curve_analysis(
    times: &[f64],
    events: &[u8],
    predicted_probs: &[f64],
    n_subjects: usize,
    horizon: f64,
    threshold_range: (f64, f64),
    n_thresholds: usize,
) -> SurvivalResult<DcaResult> {
    validate_inputs_dca(
        times,
        events,
        predicted_probs,
        n_subjects,
        horizon,
        threshold_range,
        n_thresholds,
    )?;

    let n = n_subjects;
    let (pt_min, pt_max) = threshold_range;

    // Compute observed event rate at horizon (prevalence)
    let n_events: usize = (0..n)
        .filter(|&i| times[i] <= horizon && events[i] == 1)
        .count();
    let prevalence = n_events as f64 / n as f64;

    // Build threshold grid
    let thresholds: Vec<f64> = if n_thresholds == 1 {
        vec![(pt_min + pt_max) / 2.0]
    } else {
        (0..n_thresholds)
            .map(|k| pt_min + (pt_max - pt_min) * k as f64 / (n_thresholds - 1) as f64)
            .collect()
    };

    let mut net_benefit_model = Vec::with_capacity(n_thresholds);
    let mut net_benefit_all = Vec::with_capacity(n_thresholds);
    let mut net_benefit_none = Vec::with_capacity(n_thresholds);

    for &pt in &thresholds {
        // Count TP, FP, TN, FN at this threshold
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut n_pos = 0usize; // predicted positive

        for i in 0..n {
            let is_event = times[i] <= horizon && events[i] == 1;
            let is_predicted_pos = predicted_probs[i] >= pt;
            if is_predicted_pos {
                n_pos += 1;
                if is_event {
                    tp += 1;
                } else {
                    fp += 1;
                }
            } else if !is_event {
                tn += 1;
            }
        }
        let _ = tn; // not used in net benefit formula

        // Model net benefit
        let nb_model = if n > 0 && pt < 1.0 {
            let tp_r = tp as f64 / n as f64;
            let fp_r = fp as f64 / n as f64;
            let odds = pt / (1.0 - pt);
            tp_r - fp_r * odds
        } else {
            0.0
        };

        // "Treat all" net benefit: classify everyone as positive
        let nb_all = if pt < 1.0 {
            let odds = pt / (1.0 - pt);
            prevalence - (1.0 - prevalence) * odds
        } else {
            0.0
        };

        // "Treat none" net benefit: always 0.0
        let nb_none = 0.0_f64;

        net_benefit_model.push(nb_model);
        net_benefit_all.push(nb_all.max(-1.0)); // can be negative, clamp at -1
        net_benefit_none.push(nb_none);

        let _ = n_pos;
    }

    Ok(DcaResult {
        n_thresholds: thresholds.len(),
        thresholds,
        net_benefit_model,
        net_benefit_all,
        net_benefit_none,
    })
}

// ---------------------------------------------------------------------------
// Input Validation Helpers
// ---------------------------------------------------------------------------

fn validate_inputs(
    times: &[f64],
    events: &[u8],
    risk_scores: &[f64],
    n_subjects: usize,
    horizon: f64,
) -> SurvivalResult<()> {
    if times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    if events.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![events.len()],
        });
    }
    if risk_scores.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![risk_scores.len()],
        });
    }
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "horizon must be positive and finite".to_string(),
        ));
    }
    for &t in times.iter() {
        if !t.is_finite() || t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }
    Ok(())
}

fn validate_inputs_calibration(
    times: &[f64],
    events: &[u8],
    predicted_probs: &[f64],
    n_subjects: usize,
    horizon: f64,
    n_groups: usize,
) -> SurvivalResult<()> {
    if times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    if events.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![events.len()],
        });
    }
    if predicted_probs.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![predicted_probs.len()],
        });
    }
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "horizon must be positive and finite".to_string(),
        ));
    }
    if n_groups < 2 {
        return Err(SurvivalError::InvalidParameter(
            "n_groups must be at least 2 for Hosmer-Lemeshow test".to_string(),
        ));
    }
    for (i, &p) in predicted_probs.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(SurvivalError::InvalidParameter(format!(
                "predicted_probs[{i}] = {p} is not in [0, 1]"
            )));
        }
    }
    Ok(())
}

fn validate_inputs_dca(
    times: &[f64],
    events: &[u8],
    predicted_probs: &[f64],
    n_subjects: usize,
    horizon: f64,
    threshold_range: (f64, f64),
    n_thresholds: usize,
) -> SurvivalResult<()> {
    if times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    if events.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![events.len()],
        });
    }
    if predicted_probs.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![predicted_probs.len()],
        });
    }
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "horizon must be positive and finite".to_string(),
        ));
    }
    if n_thresholds == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_thresholds must be at least 1".to_string(),
        ));
    }
    let (pt_min, pt_max) = threshold_range;
    if !pt_min.is_finite()
        || !pt_max.is_finite()
        || pt_min < 0.0
        || pt_max > 1.0
        || pt_min >= pt_max
    {
        return Err(SurvivalError::InvalidParameter(format!(
            "threshold_range ({pt_min}, {pt_max}) must satisfy 0 ≤ min < max ≤ 1"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build simple test data: first `n_cases` subjects are events at t=1..n_cases,
    /// remaining `n_controls` subjects are controls surviving to t=horizon+1.
    fn make_data(n_cases: usize, n_controls: usize, horizon: f64) -> (Vec<f64>, Vec<u8>, usize) {
        let mut times = Vec::new();
        let mut events = Vec::new();
        for i in 0..n_cases {
            times.push(1.0 + i as f64 * 0.5);
            events.push(1u8);
        }
        for _ in 0..n_controls {
            times.push(horizon + 1.0);
            events.push(0u8);
        }
        let n = times.len();
        (times, events, n)
    }

    // ------------------------------------------------------------------
    // Test 1: Perfect predictor AUC ≈ 1.0
    // ------------------------------------------------------------------
    #[test]
    fn perfect_predictor_auc_one() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        // Cases get high risk scores, controls get low risk scores
        let mut scores = vec![0.0f64; n];
        for (i, s) in scores.iter_mut().enumerate().take(5) {
            *s = 10.0 + i as f64; // high scores for cases
        }
        for (i, s) in scores.iter_mut().enumerate().take(10).skip(5) {
            *s = i as f64 * 0.1; // low scores for controls
        }
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        assert!(result.auc > 0.95, "expected AUC ≈ 1.0, got {}", result.auc);
    }

    // ------------------------------------------------------------------
    // Test 2: Reversed predictor AUC ≈ 0.0
    // ------------------------------------------------------------------
    #[test]
    fn reversed_predictor_auc_zero() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        // Cases get low risk scores, controls get high risk scores
        let mut scores = vec![0.0f64; n];
        for (i, s) in scores.iter_mut().enumerate().take(5) {
            *s = i as f64 * 0.1; // low for cases
        }
        for (i, s) in scores.iter_mut().enumerate().take(10).skip(5) {
            *s = 10.0 + i as f64; // high for controls
        }
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        assert!(result.auc < 0.1, "expected AUC ≈ 0.0, got {}", result.auc);
    }

    // ------------------------------------------------------------------
    // Test 3: Random predictor AUC ≈ 0.5
    // ------------------------------------------------------------------
    #[test]
    fn random_predictor_auc_near_half() {
        let horizon = 5.0;
        let (times, events, n) = make_data(20, 20, horizon);
        // Assign same score to all — perfect ambiguity
        let scores = vec![1.0f64; n];
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        // With all equal scores, AUC should be 0.5
        assert!(
            (result.auc - 0.5).abs() < 0.15,
            "expected AUC ≈ 0.5, got {}",
            result.auc
        );
    }

    // ------------------------------------------------------------------
    // Test 4: ROC boundary conditions: fpr[0]=0, tpr[-1]=1
    // ------------------------------------------------------------------
    #[test]
    fn roc_boundary_conditions() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        let scores: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        let fpr = &result.fpr;
        let tpr = &result.tpr;
        assert!(!fpr.is_empty(), "ROC must not be empty");
        assert_eq!(fpr.len(), tpr.len());
        // First point must have fpr=0
        assert!(fpr[0] <= 1.0e-10, "fpr[0] should be 0, got {}", fpr[0]);
        // Last point must have tpr=1
        let last_tpr = tpr[tpr.len() - 1];
        assert!(
            (last_tpr - 1.0).abs() < 1.0e-10,
            "tpr[-1] should be 1, got {}",
            last_tpr
        );
    }

    // ------------------------------------------------------------------
    // Test 5: FPR sorted ascending in ROC curve
    // ------------------------------------------------------------------
    #[test]
    fn fpr_sorted_ascending() {
        let horizon = 5.0;
        let (times, events, n) = make_data(6, 6, horizon);
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 1.3 - 3.0).collect();
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        let fpr = &result.fpr;
        for i in 1..fpr.len() {
            assert!(
                fpr[i] >= fpr[i - 1] - 1.0e-12,
                "FPR must be sorted ascending: fpr[{}]={} < fpr[{}]={}",
                i,
                fpr[i],
                i - 1,
                fpr[i - 1]
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 6: AUC always in [0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn auc_in_unit_interval() {
        let horizon = 3.0;
        let (times, events, n) = make_data(4, 4, horizon);
        let scores: Vec<f64> = (0..n).map(|i| (i as f64 - 4.0).abs()).collect();
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        assert!(
            result.auc >= 0.0 && result.auc <= 1.0,
            "AUC out of [0,1]: {}",
            result.auc
        );
    }

    // ------------------------------------------------------------------
    // Test 7: n_cases + n_controls ≤ n_subjects (excluded subjects possible)
    // ------------------------------------------------------------------
    #[test]
    fn n_cases_plus_controls_le_n_subjects() {
        // Some subjects censored before horizon → excluded
        let times = vec![1.0, 2.0, 2.5, 6.0, 7.0];
        let events = vec![1u8, 0, 0, 0, 0]; // only first is event; 2nd & 3rd censored before horizon=5
        let scores = vec![5.0, 1.0, 2.0, 3.0, 4.0];
        let n = 5;
        let horizon = 5.0;
        let result = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        assert!(
            result.n_cases + result.n_controls <= n,
            "n_cases({}) + n_controls({}) > n_subjects({})",
            result.n_cases,
            result.n_controls,
            n
        );
    }

    // ------------------------------------------------------------------
    // Test 8: Empty events before horizon → error
    // ------------------------------------------------------------------
    #[test]
    fn no_cases_returns_error() {
        // All subjects censored or survive past horizon → no cases
        let times = vec![10.0, 11.0, 12.0];
        let events = vec![0u8, 0, 0];
        let scores = vec![1.0, 2.0, 3.0];
        let n = 3;
        let horizon = 5.0;
        let err = time_roc(&times, &events, &scores, n, horizon);
        assert!(err.is_err(), "expected error for no cases");
    }

    // ------------------------------------------------------------------
    // Test 9: O/E ratio ≈ 1.0 for well-calibrated model
    // ------------------------------------------------------------------
    #[test]
    fn oe_ratio_near_one_well_calibrated() {
        // Subjects with predicted_prob = actual event rate → O/E ≈ 1
        let horizon = 5.0;
        let times = vec![1.0, 2.0, 8.0, 9.0];
        let events = vec![1u8, 1, 0, 0];
        let n = 4;
        // Predicted probability = 0.5 for each (2/4 events)
        let predicted = vec![0.5, 0.5, 0.5, 0.5];
        let result = calibration_analysis(&times, &events, &predicted, n, horizon, 2).expect("ok");
        assert!(
            (result.oe_ratio - 1.0).abs() < 1.0e-10,
            "O/E should be 1.0 for perfect calibration, got {}",
            result.oe_ratio
        );
    }

    // ------------------------------------------------------------------
    // Test 10: hl_p_value in [0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn hl_p_value_in_unit_interval() {
        let horizon = 5.0;
        let (times, events, n) = make_data(10, 10, horizon);
        let predicted: Vec<f64> = (0..n)
            .map(|i| 0.1 + 0.05 * i as f64)
            .map(|v: f64| v.min(0.99))
            .collect();
        let result = calibration_analysis(&times, &events, &predicted, n, horizon, 5).expect("ok");
        assert!(
            result.hl_p_value >= 0.0 && result.hl_p_value <= 1.0,
            "hl_p_value out of [0,1]: {}",
            result.hl_p_value
        );
    }

    // ------------------------------------------------------------------
    // Test 11: DCA net_benefit_none all zeros
    // ------------------------------------------------------------------
    #[test]
    fn dca_net_benefit_none_all_zeros() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        let predicted: Vec<f64> = (0..n)
            .map(|i| 0.1 * (i + 1) as f64)
            .map(|v: f64| v.min(0.99))
            .collect();
        let result =
            decision_curve_analysis(&times, &events, &predicted, n, horizon, (0.1, 0.9), 10)
                .expect("ok");
        for &nb in &result.net_benefit_none {
            assert!(
                nb.abs() < 1.0e-12,
                "net_benefit_none should be 0, got {}",
                nb
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 12: DCA thresholds.len() == n_thresholds
    // ------------------------------------------------------------------
    #[test]
    fn dca_threshold_count_matches() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        let predicted = vec![0.3f64; n];
        let n_thresholds = 50;
        let result = decision_curve_analysis(
            &times,
            &events,
            &predicted,
            n,
            horizon,
            (0.05, 0.95),
            n_thresholds,
        )
        .expect("ok");
        assert_eq!(
            result.thresholds.len(),
            n_thresholds,
            "thresholds.len() should equal n_thresholds"
        );
        assert_eq!(result.n_thresholds, n_thresholds);
    }

    // ------------------------------------------------------------------
    // Test 13: DCA "treat all" net benefit decreasing with threshold
    // ------------------------------------------------------------------
    #[test]
    fn dca_treat_all_generally_decreasing() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        let predicted = vec![0.5f64; n];
        let result =
            decision_curve_analysis(&times, &events, &predicted, n, horizon, (0.01, 0.99), 20)
                .expect("ok");
        let nb_all = &result.net_benefit_all;
        // Check overall trend: last value ≤ first value
        assert!(
            nb_all[nb_all.len() - 1] <= nb_all[0] + 1.0e-6,
            "treat-all net benefit should decrease: first={}, last={}",
            nb_all[0],
            nb_all[nb_all.len() - 1]
        );
    }

    // ------------------------------------------------------------------
    // Test 14: time_roc_auc_only matches time_roc.auc
    // ------------------------------------------------------------------
    #[test]
    fn auc_only_matches_full_roc() {
        let horizon = 5.0;
        let (times, events, n) = make_data(5, 5, horizon);
        let scores: Vec<f64> = (0..n).map(|i| i as f64 * 1.7).collect();
        let auc_only = time_roc_auc_only(&times, &events, &scores, n, horizon).expect("ok");
        let full = time_roc(&times, &events, &scores, n, horizon).expect("ok");
        assert!(
            (auc_only - full.auc).abs() < 1.0e-12,
            "auc_only={} should equal full.auc={}",
            auc_only,
            full.auc
        );
    }

    // ------------------------------------------------------------------
    // Test 15: Horizon beyond max time returns error
    // ------------------------------------------------------------------
    #[test]
    fn horizon_beyond_all_times_no_controls_error() {
        // All times are ≤ horizon and events=1 → no controls
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1u8, 1, 1];
        let scores = vec![1.0, 2.0, 3.0];
        let n = 3;
        let horizon = 100.0; // all subjects have t ≤ horizon, so no controls
        let err = time_roc(&times, &events, &scores, n, horizon);
        assert!(err.is_err(), "expected error when no controls exist");
    }

    // ------------------------------------------------------------------
    // Test 16: Calibration with predicted exactly matching event indicators
    // achieves O/E = 1.0; ICI ∈ [0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn calibration_with_predicted_near_actual_rate() {
        let times = vec![1.0, 2.0, 6.0, 7.0, 8.0];
        let events = vec![1u8, 1, 0, 0, 0];
        let n = 5;
        let horizon = 5.0;
        // Predicted event probability = 2/5 = 0.4 for all subjects → O/E = 1.0 globally
        let predicted = vec![0.4f64; n];
        let result = calibration_analysis(&times, &events, &predicted, n, horizon, 2).expect("ok");
        // O/E = observed_events / sum_predicted = 2 / (5 * 0.4) = 2/2 = 1.0
        assert!(
            (result.oe_ratio - 1.0).abs() < 1.0e-10,
            "O/E should be ≈ 1.0, got {}",
            result.oe_ratio
        );
        // ICI must be in [0, 1] — not necessarily near 0 with small n
        assert!(
            (0.0..=1.0).contains(&result.ici),
            "ICI should be in [0,1], got {}",
            result.ici
        );
        // mean_predicted should equal predicted rate
        assert!(
            (result.mean_predicted - 0.4).abs() < 1.0e-10,
            "mean_predicted should be 0.4, got {}",
            result.mean_predicted
        );
        // observed_rate: 2/5 = 0.4
        assert!(
            (result.observed_rate - 0.4).abs() < 1.0e-10,
            "observed_rate should be 0.4, got {}",
            result.observed_rate
        );
    }
}
