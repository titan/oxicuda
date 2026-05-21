//! Survival meta-analysis: combining Kaplan-Meier curves and hazard ratios from
//! multiple independent studies.
//!
//! # Overview
//!
//! This module implements five complementary methods for aggregating survival
//! evidence across studies:
//!
//! 1. **Pooled KM** — inverse-variance weighting of `log(-log S(t))` on a merged
//!    time grid (the log-log transform stabilises variance near 0 and 1).
//! 2. **Fixed-effects meta-analysis** (Mantel-Haenszel / DerSimonian) of
//!    log-hazard-ratio summaries from K Cox or log-rank analyses.
//! 3. **Random-effects DerSimonian-Laird** estimator with between-study τ² and
//!    Cochran's Q / I² heterogeneity statistics.
//! 4. **Combined log-rank** (O'Brien-Fleming style) — aggregates O_k – E_k and
//!    hypergeometric variance terms across studies.
//! 5. **Guyot IPD reconstruction** — approximate digitized-KM → event/risk table
//!    reconstruction for subsequent re-analysis.
//!
//! All p-value computations use the Abramowitz & Stegun rational approximation
//! for Φ(z) (max error ≈ 1 × 10⁻⁵) and the Wilson-Hilferty normal approximation
//! for chi-square tail probabilities.

use crate::error::{SurvivalError, SurvivalResult};

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single study's Kaplan-Meier curve summary with Greenwood variance of
/// `log(-log S(t))` pre-computed at each event time.
#[derive(Debug, Clone)]
pub struct StudyKm {
    /// Human-readable study identifier.
    pub study_id: String,
    /// Unique event times (ascending, length = T).
    pub times: Vec<f64>,
    /// KM survival estimates S(t_i), length = T.
    pub survival: Vec<f64>,
    /// Number of events d_i at each time point, length = T.
    pub n_events: Vec<u64>,
    /// Number at risk n_i just before each time point, length = T.
    pub n_at_risk: Vec<u64>,
    /// Greenwood variance of `log(-log S(t_i))` at each time point, length = T.
    ///
    /// Formula: `Var[log(-log S(t))] = Σ_{t_j ≤ t} d_j / (n_j (n_j - d_j)) / (log S(t))²`
    pub ll_variance: Vec<f64>,
}

/// Study-level hazard ratio summary, e.g., from a fitted Cox model or log-rank.
#[derive(Debug, Clone)]
pub struct StudyHazardRatio {
    /// Human-readable study identifier.
    pub study_id: String,
    /// Log hazard ratio θ_k = log(HR_k).
    pub log_hr: f64,
    /// Variance V_k of the log hazard ratio.
    pub log_hr_variance: f64,
    /// Total number of events in the study (informational).
    pub n_events: u64,
}

/// Fixed-effects (inverse-variance weighted) meta-analysis result.
#[derive(Debug, Clone)]
pub struct FixedEffectsResult {
    /// Pooled log hazard ratio θ_FE.
    pub pooled_log_hr: f64,
    /// Pooled hazard ratio HR_FE = exp(θ_FE).
    pub pooled_hr: f64,
    /// 95% CI lower bound for HR.
    pub ci_lower: f64,
    /// 95% CI upper bound for HR.
    pub ci_upper: f64,
    /// Variance Var(θ_FE) = 1 / Σ_k w_k.
    pub variance: f64,
    /// z-statistic = θ_FE / sqrt(variance).
    pub z_stat: f64,
    /// Two-sided p-value from z_stat.
    pub p_value: f64,
}

/// Random-effects DerSimonian-Laird meta-analysis result.
#[derive(Debug, Clone)]
pub struct RandomEffectsResult {
    /// Pooled log hazard ratio θ_RE.
    pub pooled_log_hr: f64,
    /// Pooled hazard ratio HR_RE = exp(θ_RE).
    pub pooled_hr: f64,
    /// 95% CI lower bound for HR.
    pub ci_lower: f64,
    /// 95% CI upper bound for HR.
    pub ci_upper: f64,
    /// Variance Var(θ_RE) = 1 / Σ_k w_k*.
    pub variance: f64,
    /// Between-study variance τ² (≥ 0).
    pub tau_sq: f64,
    /// Cochran's Q heterogeneity statistic (df = K – 1).
    pub cochran_q: f64,
    /// I² = max(0, (Q – (K-1)) / Q) × 100 %, proportion of variance attributable
    /// to between-study heterogeneity.
    pub i_squared: f64,
    /// Two-sided p-value for Cochran's Q under H₀: τ² = 0.
    pub p_heterogeneity: f64,
}

/// Pooled Kaplan-Meier curve from multiple studies on a merged time grid.
#[derive(Debug, Clone)]
pub struct PooledKmResult {
    /// Merged and sorted time grid (union of all study event times).
    pub times: Vec<f64>,
    /// Pooled survival S_pooled(t) at each grid point.
    pub survival: Vec<f64>,
    /// 95% CI lower bound for pooled survival.
    pub ci_lower: Vec<f64>,
    /// 95% CI upper bound for pooled survival.
    pub ci_upper: Vec<f64>,
    /// Number of studies contributing a finite weight at each time point.
    pub n_studies: Vec<usize>,
}

/// Combined log-rank test from multiple independent studies.
#[derive(Debug, Clone)]
pub struct CombinedLogRankResult {
    /// z-statistic = Σ(O_k – E_k) / sqrt(Σ V_k).
    pub z_stat: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Combined (O – E) = Σ_k (O_k – E_k).
    pub combined_o_minus_e: f64,
    /// Combined variance = Σ_k V_k.
    pub combined_variance: f64,
}

/// Approximate individual-patient-data reconstructed from a digitized KM curve
/// and a published risk table (Guyot et al. 2012).
#[derive(Debug, Clone)]
pub struct GuyotReconstruction {
    /// Reconstructed event times (= input `km_times`).
    pub times: Vec<f64>,
    /// Reconstructed number of events at each time point.
    pub n_events: Vec<u64>,
    /// Reconstructed number at risk at each time point.
    pub n_at_risk: Vec<u64>,
    /// KM survival at each reconstructed time point (= input `km_survival`).
    pub survival: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// P-value helpers (Abramowitz & Stegun rational approximation)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF Φ(x) via Abramowitz & Stegun formula 26.2.16.
///
/// Max absolute error ≈ 1 × 10⁻⁵.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    // Handle extreme values
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    // A&S 26.2.16: Φ(x) ≈ 1 - φ(x)(b1 t + b2 t² + b3 t³), t = 1/(1 + 0.33267 x)
    // For x ≥ 0; for x < 0 use symmetry.
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.33267 * x_abs);
    const B1: f64 = 0.436_183_6;
    const B2: f64 = -0.120_167_6;
    const B3: f64 = 0.937_298_0;
    let phi = (-0.5 * x_abs * x_abs).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = phi * ((B1 + B2 * t + B3 * t * t) * t);
    // Φ(x) for x ≥ 0:  1 - poly
    // For x < 0:        poly
    0.5 + sign * (0.5 - poly)
}

/// Two-sided p-value from a z-statistic: `p = 2(1 - Φ(|z|))`.
#[inline]
fn p_from_z(z: f64) -> f64 {
    let p_upper = 1.0 - normal_cdf(z.abs());
    (2.0 * p_upper).min(1.0)
}

/// Chi-square tail probability P(χ²(df) > q) approximated via Wilson-Hilferty
/// normal transformation: `p ≈ P(Z > sqrt(2q) - sqrt(2df - 1))`.
///
/// Valid for df ≥ 1 and q ≥ 0.
#[inline]
fn p_from_chisq(q: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }
    if q <= 0.0 {
        return 1.0;
    }
    // Wilson-Hilferty: sqrt(2Q) - sqrt(2(K-1) - 1)
    let z = (2.0 * q).sqrt() - (2.0 * df as f64 - 1.0).max(0.0).sqrt();
    p_from_z(z)
}

// ─────────────────────────────────────────────────────────────────────────────
// Greenwood variance of log(-log S(t))
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Greenwood variance of `log(-log S(t))` from event and risk vectors.
///
/// Formula:
/// ```text
/// Var[log(-log S(t))] = [Σ_{i : t_i ≤ t} d_i / (n_i (n_i - d_i))] / (log S(t))²
/// ```
///
/// Returns a vector of length equal to `n_events.len()`.
fn greenwood_ll_variance(survival: &[f64], n_events: &[u64], n_at_risk: &[u64]) -> Vec<f64> {
    let m = survival.len();
    let mut variances = Vec::with_capacity(m);
    let mut cumsum = 0.0_f64;

    for i in 0..m {
        let d = n_events[i] as f64;
        let n = n_at_risk[i] as f64;
        let s = survival[i];

        // Accumulate Greenwood term: d / (n (n - d))
        if d > 0.0 && n > d {
            cumsum += d / (n * (n - d));
        }

        let log_s = if s > 0.0 { s.ln() } else { f64::NEG_INFINITY };
        let var = if log_s < 0.0 && log_s.is_finite() {
            cumsum / (log_s * log_s)
        } else {
            // S(t)=1 → log-log is undefined; or S(t)=0 → degenerate
            f64::INFINITY
        };
        variances.push(var);
    }

    variances
}

// ─────────────────────────────────────────────────────────────────────────────
// Public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`StudyKm`] from raw event-time summaries.
///
/// # Arguments
/// * `times`     — unique event times, ascending, length = T
/// * `n_events`  — events d_i at each time point, length = T
/// * `n_at_risk` — number at risk n_i at each time point, length = T
/// * `study_id`  — human-readable label
///
/// # Errors
/// - `EmptyDataset` if `times` is empty.
/// - `InvalidParameter` if arrays have mismatched lengths.
/// - `InvalidParameter` if any n_at_risk < n_events or n_at_risk == 0.
/// - `NumericalInstability` if survival would become negative.
pub fn compute_study_km(
    times: &[f64],
    n_events: &[u64],
    n_at_risk: &[u64],
    study_id: &str,
) -> SurvivalResult<StudyKm> {
    let m = times.len();
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if n_events.len() != m || n_at_risk.len() != m {
        return Err(SurvivalError::InvalidParameter(format!(
            "times ({}), n_events ({}), n_at_risk ({}) must all have the same length",
            m,
            n_events.len(),
            n_at_risk.len()
        )));
    }

    // Validate inputs
    for i in 0..m {
        if n_at_risk[i] == 0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "n_at_risk[{i}] is 0; at-risk count must be positive"
            )));
        }
        if n_events[i] > n_at_risk[i] {
            return Err(SurvivalError::InvalidParameter(format!(
                "n_events[{i}] = {} > n_at_risk[{i}] = {}",
                n_events[i], n_at_risk[i]
            )));
        }
        if times[i] < 0.0 {
            return Err(SurvivalError::NegativeTime(times[i]));
        }
    }

    // Compute KM survival S(t_i) = Π_{j ≤ i} (1 - d_j / n_j)
    let mut survival = Vec::with_capacity(m);
    let mut s_cur = 1.0_f64;
    for i in 0..m {
        let d = n_events[i] as f64;
        let n = n_at_risk[i] as f64;
        let factor = 1.0 - d / n;
        if factor < 0.0 {
            return Err(SurvivalError::NumericalInstability(format!(
                "survival factor {factor} < 0 at index {i}"
            )));
        }
        s_cur *= factor;
        survival.push(s_cur);
    }

    // Greenwood variance of log(-log S(t))
    let ll_variance = greenwood_ll_variance(&survival, n_events, n_at_risk);

    Ok(StudyKm {
        study_id: study_id.to_string(),
        times: times.to_vec(),
        survival,
        n_events: n_events.to_vec(),
        n_at_risk: n_at_risk.to_vec(),
        ll_variance,
    })
}

/// Evaluate a study's KM survival and log-log variance at a given time `t` by
/// last-observation-carried-forward (LOCF) interpolation.
///
/// Returns `(S(t), Var[log(-log S(t))])`.  If `t < times[0]`, returns `(1.0, INF)`.
/// If all times exceed `t`, returns the same.  The pair `(1.0, INF)` carries zero
/// weight in the inverse-variance pooling step, which is the desired behaviour.
fn interpolate_study_at(study: &StudyKm, t: f64) -> (f64, f64) {
    // Binary search for the largest index with times[idx] <= t
    let pos = study.times.partition_point(|&x| x <= t);
    if pos == 0 {
        // t is before the first event time → S(t) = 1
        return (1.0, f64::INFINITY);
    }
    let idx = pos - 1;
    let s = study.survival[idx];
    let v = study.ll_variance[idx];
    (s, v)
}

/// Pool Kaplan-Meier curves from multiple independent studies using
/// inverse-variance weighting of `log(-log S(t))` on a merged time grid.
///
/// # Algorithm
///
/// 1. Form the union of all event times across studies.
/// 2. At each grid point t, each study contributes weight `w_k(t) = 1 / Var_k(t)`
///    where `Var_k(t)` is the Greenwood variance of `log(-log S_k(t))`.
///    Studies with `S_k(t) = 1` (no events yet) or `S_k(t) = 0` (all events elapsed)
///    carry infinite variance and zero weight.
/// 3. Pooled log-log: `θ_pool(t) = Σ_k w_k θ_k / Σ_k w_k`
/// 4. Pooled survival: `S_pool(t) = exp(-exp(θ_pool(t)))`
/// 5. 95% CI on log-log scale: `θ ± 1.96 / sqrt(Σ_k w_k)`, back-transformed.
///
/// # Errors
/// - `EmptyDataset` if `studies` is empty.
/// - `NoEvents`     if no finite weights can be computed at any time point.
pub fn pool_km_curves(studies: &[StudyKm]) -> SurvivalResult<PooledKmResult> {
    if studies.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }

    // 1. Build the merged time grid (sorted union of all event times)
    let mut all_times: Vec<f64> = studies
        .iter()
        .flat_map(|s| s.times.iter().copied())
        .collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_times.dedup_by(|a, b| (*a - *b).abs() < 1e-12 * b.abs().max(1.0));

    if all_times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }

    let n_times = all_times.len();
    let mut pooled_survival = Vec::with_capacity(n_times);
    let mut ci_lower = Vec::with_capacity(n_times);
    let mut ci_upper = Vec::with_capacity(n_times);
    let mut n_contributing = Vec::with_capacity(n_times);

    for &t in &all_times {
        // Collect (log(-log S_k(t)), w_k = 1/Var_k) for studies with finite weight
        let mut sum_w = 0.0_f64;
        let mut sum_w_theta = 0.0_f64;
        let mut n_finite = 0usize;

        for study in studies {
            let (s_k, var_k) = interpolate_study_at(study, t);

            // Exclude degenerate cases
            if s_k <= 0.0 || s_k >= 1.0 || !var_k.is_finite() || var_k <= 0.0 {
                continue;
            }

            let log_log_s_k = (-s_k.ln()).ln(); // log(-log S_k(t))
            let w_k = 1.0 / var_k;

            sum_w += w_k;
            sum_w_theta += w_k * log_log_s_k;
            n_finite += 1;
        }

        n_contributing.push(n_finite);

        if sum_w <= 0.0 || n_finite == 0 {
            // No finite weight at this time: carry forward or use 1.0
            pooled_survival.push(1.0);
            ci_lower.push(1.0);
            ci_upper.push(1.0);
            continue;
        }

        // Pooled log-log and 95% CI on log-log scale
        let theta_pool = sum_w_theta / sum_w;
        let se_pool = 1.0 / sum_w.sqrt(); // SE(θ_pool) = 1 / sqrt(Σ w_k)

        const Z95: f64 = 1.959_963_985; // Φ⁻¹(0.975)
        let theta_lo = theta_pool - Z95 * se_pool;
        let theta_hi = theta_pool + Z95 * se_pool;

        // Back-transform: S(t) = exp(-exp(θ))
        let s_pool = (-theta_pool.exp()).exp().clamp(0.0, 1.0);
        // Note: larger θ → smaller S → CI inverts
        let s_lo = (-theta_hi.exp()).exp().clamp(0.0, 1.0);
        let s_hi = (-theta_lo.exp()).exp().clamp(0.0, 1.0);

        pooled_survival.push(s_pool);
        ci_lower.push(s_lo);
        ci_upper.push(s_hi);
    }

    Ok(PooledKmResult {
        times: all_times,
        survival: pooled_survival,
        ci_lower,
        ci_upper,
        n_studies: n_contributing,
    })
}

/// Fixed-effects (inverse-variance) meta-analysis of log hazard ratios.
///
/// # Algorithm
///
/// Given K studies with `(θ_k, V_k)` (log HR, variance):
/// - `w_k = 1 / V_k`
/// - `θ_FE = Σ w_k θ_k / Σ w_k`
/// - `Var(θ_FE) = 1 / Σ w_k`
/// - 95% CI: `θ_FE ± 1.96 √Var(θ_FE)`, back-transformed to HR scale
///
/// # Errors
/// - `EmptyDataset` if `studies` is empty.
/// - `InvalidParameter` if any variance is non-positive.
/// - `NumericalInstability` if the sum of weights is zero.
pub fn fixed_effects_meta(studies: &[StudyHazardRatio]) -> SurvivalResult<FixedEffectsResult> {
    if studies.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }

    let mut sum_w = 0.0_f64;
    let mut sum_w_theta = 0.0_f64;

    for (k, study) in studies.iter().enumerate() {
        if study.log_hr_variance <= 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "study {k} has non-positive variance {}",
                study.log_hr_variance
            )));
        }
        let w = 1.0 / study.log_hr_variance;
        sum_w += w;
        sum_w_theta += w * study.log_hr;
    }

    if sum_w <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "sum of inverse-variance weights is zero".to_string(),
        ));
    }

    let theta_fe = sum_w_theta / sum_w;
    let var_fe = 1.0 / sum_w;
    let se_fe = var_fe.sqrt();

    const Z95: f64 = 1.959_963_985;
    let log_ci_lo = theta_fe - Z95 * se_fe;
    let log_ci_hi = theta_fe + Z95 * se_fe;

    let z_stat = theta_fe / se_fe;
    let p_value = p_from_z(z_stat);

    Ok(FixedEffectsResult {
        pooled_log_hr: theta_fe,
        pooled_hr: theta_fe.exp(),
        ci_lower: log_ci_lo.exp(),
        ci_upper: log_ci_hi.exp(),
        variance: var_fe,
        z_stat,
        p_value,
    })
}

/// Random-effects DerSimonian-Laird meta-analysis of log hazard ratios.
///
/// # Algorithm
///
/// 1. Compute fixed-effects θ_FE.
/// 2. Cochran's Q = Σ_k w_k (θ_k – θ_FE)².
/// 3. Between-study variance:
///    `τ² = max(0, (Q – (K–1)) / (Σ w_k – Σ w_k² / Σ w_k))`.
/// 4. RE weights `w_k* = 1 / (V_k + τ²)`.
/// 5. RE estimate `θ_RE = Σ w_k* θ_k / Σ w_k*`, `Var(θ_RE) = 1 / Σ w_k*`.
/// 6. I² = max(0, (Q – (K–1)) / Q) × 100%.
///
/// # Errors
/// Same as [`fixed_effects_meta`].
pub fn random_effects_meta(studies: &[StudyHazardRatio]) -> SurvivalResult<RandomEffectsResult> {
    if studies.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }

    // ── Step 1: Fixed-effects denominator pieces ──────────────────────────────
    let k = studies.len();
    let mut sum_w = 0.0_f64;
    let mut sum_w2 = 0.0_f64;
    let mut sum_w_theta = 0.0_f64;

    for (idx, study) in studies.iter().enumerate() {
        if study.log_hr_variance <= 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "study {idx} has non-positive variance {}",
                study.log_hr_variance
            )));
        }
        let w = 1.0 / study.log_hr_variance;
        sum_w += w;
        sum_w2 += w * w;
        sum_w_theta += w * study.log_hr;
    }

    if sum_w <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "sum of inverse-variance weights is zero".to_string(),
        ));
    }

    // ── Step 2: Fixed-effects estimate and Cochran's Q ────────────────────────
    let theta_fe = sum_w_theta / sum_w;
    let cochran_q: f64 = studies
        .iter()
        .map(|s| {
            let w = 1.0 / s.log_hr_variance;
            let diff = s.log_hr - theta_fe;
            w * diff * diff
        })
        .sum();

    // ── Step 3: DerSimonian-Laird τ² ──────────────────────────────────────────
    let df = (k as f64) - 1.0;
    let c_factor = sum_w - sum_w2 / sum_w; // C = Σ w_k - Σ w_k² / Σ w_k
    let tau_sq = if c_factor > 0.0 {
        ((cochran_q - df) / c_factor).max(0.0)
    } else {
        0.0
    };

    // ── Step 4: Random-effects pooled estimate ────────────────────────────────
    let mut sum_w_re = 0.0_f64;
    let mut sum_w_re_theta = 0.0_f64;
    for study in studies {
        let w_re = 1.0 / (study.log_hr_variance + tau_sq);
        sum_w_re += w_re;
        sum_w_re_theta += w_re * study.log_hr;
    }

    if sum_w_re <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "sum of random-effects weights is zero".to_string(),
        ));
    }

    let theta_re = sum_w_re_theta / sum_w_re;
    let var_re = 1.0 / sum_w_re;
    let se_re = var_re.sqrt();

    const Z95: f64 = 1.959_963_985;
    let log_ci_lo = theta_re - Z95 * se_re;
    let log_ci_hi = theta_re + Z95 * se_re;

    // ── Step 5: Heterogeneity statistics ─────────────────────────────────────
    let i_squared = if cochran_q > 0.0 && k > 1 {
        ((cochran_q - df) / cochran_q * 100.0).max(0.0)
    } else {
        0.0
    };

    let p_het = if k > 1 {
        p_from_chisq(cochran_q, k - 1)
    } else {
        1.0
    };

    Ok(RandomEffectsResult {
        pooled_log_hr: theta_re,
        pooled_hr: theta_re.exp(),
        ci_lower: log_ci_lo.exp(),
        ci_upper: log_ci_hi.exp(),
        variance: var_re,
        tau_sq,
        cochran_q,
        i_squared,
        p_heterogeneity: p_het,
    })
}

/// Combine log-rank test statistics from K independent studies.
///
/// # Arguments
/// * `o_minus_e` — K-vector of (O_k – E_k) values from each study's log-rank test.
/// * `variances` — K-vector of hypergeometric variances V_k.
///
/// # Algorithm
///
/// ```text
/// Z = Σ_k (O_k – E_k) / sqrt(Σ_k V_k)
/// p = 2(1 – Φ(|Z|))
/// ```
///
/// # Errors
/// - `EmptyDataset` if slices are empty.
/// - `InvalidParameter` if lengths differ or any variance is non-positive.
/// - `NumericalInstability` if combined variance is non-positive.
pub fn combined_log_rank(
    o_minus_e: &[f64],
    variances: &[f64],
) -> SurvivalResult<CombinedLogRankResult> {
    let k = o_minus_e.len();
    if k == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if variances.len() != k {
        return Err(SurvivalError::InvalidParameter(format!(
            "o_minus_e length ({k}) != variances length ({})",
            variances.len()
        )));
    }

    for (i, &v) in variances.iter().enumerate() {
        if v < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "variances[{i}] = {v} is negative"
            )));
        }
    }

    let combined_oe: f64 = o_minus_e.iter().sum();
    let combined_var: f64 = variances.iter().sum();

    if combined_var <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "combined log-rank variance is non-positive".to_string(),
        ));
    }

    let z_stat = combined_oe / combined_var.sqrt();
    let p_value = p_from_z(z_stat);

    Ok(CombinedLogRankResult {
        z_stat,
        p_value,
        combined_o_minus_e: combined_oe,
        combined_variance: combined_var,
    })
}

/// Reconstruct approximate event/risk counts from a digitized KM curve and a
/// published risk table using the Guyot et al. (2012) algorithm.
///
/// # Arguments
/// * `km_times`    — digitized KM step times t₁ < t₂ < … < tₙ (non-negative).
/// * `km_survival` — KM step values S(tᵢ) ∈ (0, 1], non-increasing.
/// * `risk_times`  — time points at which n_at_risk was published (sorted).
/// * `n_at_risk`   — published at-risk counts at each `risk_times` entry.
///
/// # Algorithm
///
/// For each digitized KM step at time tᵢ:
/// 1. Find the published at-risk count from the preceding risk-table row via LOCF.
/// 2. Approximate events: `d_i = round(n_i * (1 - S_i / S_{i-1}))`.
/// 3. Reconstruct `n_{i+1}` from next risk-table boundary minus censored subjects.
///
/// # Errors
/// - `EmptyDataset` if either input slice is empty.
/// - `InvalidParameter` if lengths are inconsistent or survival is non-monotone.
pub fn guyot_reconstruct(
    km_times: &[f64],
    km_survival: &[f64],
    risk_times: &[f64],
    n_at_risk: &[u64],
) -> SurvivalResult<GuyotReconstruction> {
    let m = km_times.len();
    if m == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if km_survival.len() != m {
        return Err(SurvivalError::InvalidParameter(format!(
            "km_times length ({m}) != km_survival length ({})",
            km_survival.len()
        )));
    }
    if risk_times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if risk_times.len() != n_at_risk.len() {
        return Err(SurvivalError::InvalidParameter(format!(
            "risk_times length ({}) != n_at_risk length ({})",
            risk_times.len(),
            n_at_risk.len()
        )));
    }

    // Validate non-increasing survival
    for i in 1..m {
        if km_survival[i] > km_survival[i - 1] + 1e-9 {
            return Err(SurvivalError::InvalidParameter(format!(
                "km_survival is not non-increasing at index {i}: {} > {}",
                km_survival[i],
                km_survival[i - 1]
            )));
        }
    }

    // Helper: LOCF lookup of at-risk count at time t from risk table.
    // Returns the count at the last risk_times entry ≤ t, or n_at_risk[0] if
    // t < risk_times[0].
    let locf_n = |t: f64| -> u64 {
        let pos = risk_times.partition_point(|&rt| rt <= t);
        if pos == 0 {
            n_at_risk[0]
        } else {
            n_at_risk[pos - 1]
        }
    };

    let mut out_times = Vec::with_capacity(m);
    let mut out_n_events = Vec::with_capacity(m);
    let mut out_n_at_risk = Vec::with_capacity(m);
    let mut out_survival = Vec::with_capacity(m);

    // Running at-risk tracker: seeded from the risk table.
    // We use the at-risk count from the risk-table LOCF at each km time.
    let mut running_n: u64 = locf_n(km_times[0]);

    for i in 0..m {
        let t_i = km_times[i];
        let s_i = km_survival[i];
        let s_prev = if i == 0 { 1.0 } else { km_survival[i - 1] };

        // Resolve n_i from risk table (LOCF), then cap by running tracker
        let n_table = locf_n(t_i);
        // The at-risk at t_i is the MINIMUM of the risk-table value (which may
        // reflect censoring between risk-table rows) and our running estimate.
        let n_i = running_n.min(n_table);

        // Approximate number of events: d_i = round(n_i * (1 - S_i / S_prev))
        let ratio = if s_prev > 0.0 { s_i / s_prev } else { 0.0 };
        let frac_events = (1.0 - ratio) * n_i as f64;
        let d_i = frac_events.round() as u64;
        // Clamp to [0, n_i]
        let d_i = d_i.min(n_i);

        out_times.push(t_i);
        out_n_at_risk.push(n_i);
        out_n_events.push(d_i);
        out_survival.push(s_i);

        // Update running_n: survivors = n_i - d_i; subtract any censored between
        // t_i and t_{i+1} using the next risk-table entry when available.
        let survivors = n_i.saturating_sub(d_i);

        if i + 1 < m {
            let n_next_table = locf_n(km_times[i + 1]);
            // Subjects leaving between step i and i+1 = survivors - n_next_table
            // but we cannot have more at risk than survivors.
            running_n = survivors.min(n_next_table);
        } else {
            running_n = survivors;
        }
    }

    Ok(GuyotReconstruction {
        times: out_times,
        n_events: out_n_events,
        n_at_risk: out_n_at_risk,
        survival: out_survival,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b} ± {tol}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    fn make_study(study_id: &str, times: &[f64], n_events: &[u64], n_at_risk: &[u64]) -> StudyKm {
        compute_study_km(times, n_events, n_at_risk, study_id).expect("compute_study_km")
    }

    // ── Test 1: basic single-study KM ─────────────────────────────────────────
    #[test]
    fn compute_study_km_basic() {
        // 10 subjects, 2 events at t=1 (n=10), 3 events at t=2 (n=8)
        let times = [1.0, 2.0];
        let n_events = [2u64, 3];
        let n_at_risk = [10u64, 8];
        let study = compute_study_km(&times, &n_events, &n_at_risk, "study_a").expect("ok");

        // S(1) = 1 - 2/10 = 0.8
        assert_close(study.survival[0], 0.8, 1e-12);
        // S(2) = 0.8 * (1 - 3/8) = 0.8 * 0.625 = 0.5
        assert_close(study.survival[1], 0.5, 1e-12);

        assert_eq!(study.study_id, "study_a");
        assert_eq!(study.times.len(), 2);
    }

    // ── Test 2: survival is non-increasing ────────────────────────────────────
    #[test]
    fn compute_study_km_decreasing() {
        let times = [1.0, 2.0, 3.0, 5.0, 7.0];
        let n_events = [1u64, 2, 1, 3, 1];
        let n_at_risk = [20u64, 19, 17, 16, 12];
        let study = compute_study_km(&times, &n_events, &n_at_risk, "s").expect("ok");

        for i in 1..study.survival.len() {
            assert!(
                study.survival[i] <= study.survival[i - 1] + 1e-12,
                "survival not non-increasing at {i}: {} > {}",
                study.survival[i],
                study.survival[i - 1]
            );
        }
    }

    // ── Test 3: pooling a single study returns its own curve ──────────────────
    #[test]
    fn pool_km_curves_single_study() {
        let study = make_study("s1", &[1.0, 2.0, 3.0], &[1, 1, 1], &[10, 9, 8]);
        let pooled = pool_km_curves(std::slice::from_ref(&study)).expect("ok");

        // All time points should appear
        assert_eq!(pooled.times.len(), 3);

        // For a single study: pooled S should equal the study's S at each event time
        for (i, &t) in pooled.times.iter().enumerate() {
            let (s_study, _) = interpolate_study_at(&study, t);
            if pooled.n_studies[i] > 0 {
                // The pooled survival at the study's own times should match closely
                assert_close(pooled.survival[i], s_study, 1e-6);
            }
        }
    }

    // ── Test 4: two identical studies produce the same curve ──────────────────
    #[test]
    fn pool_km_curves_two_identical() {
        let study_a = make_study("a", &[1.0, 2.0, 3.0], &[1, 1, 1], &[10, 9, 8]);
        let study_b = make_study("b", &[1.0, 2.0, 3.0], &[1, 1, 1], &[10, 9, 8]);
        let pooled = pool_km_curves(&[study_a.clone(), study_b]).expect("ok");

        // Pooled must match individual study at each shared time
        for (i, &t) in pooled.times.iter().enumerate() {
            let (s_study, _) = interpolate_study_at(&study_a, t);
            if pooled.n_studies[i] >= 2 {
                assert_close(pooled.survival[i], s_study, 1e-5);
            }
        }
    }

    // ── Test 5: pooled survival stays in (0, 1] ───────────────────────────────
    #[test]
    fn pool_km_survival_range() {
        let s1 = make_study("a", &[1.0, 3.0, 5.0], &[2, 3, 2], &[30, 28, 24]);
        let s2 = make_study("b", &[2.0, 4.0, 6.0], &[1, 2, 1], &[20, 19, 16]);
        let pooled = pool_km_curves(&[s1, s2]).expect("ok");

        for &s in &pooled.survival {
            assert!((0.0..=1.0).contains(&s), "survival {s} out of [0,1]");
        }
        for (&lo, &hi) in pooled.ci_lower.iter().zip(pooled.ci_upper.iter()) {
            assert!(lo <= hi + 1e-9, "CI inverted: lo={lo}, hi={hi}");
        }
    }

    // ── Test 6: fixed-effects with a single study ─────────────────────────────
    #[test]
    fn fixed_effects_meta_single() {
        let study = StudyHazardRatio {
            study_id: "s1".to_string(),
            log_hr: 0.5,
            log_hr_variance: 0.04,
            n_events: 100,
        };
        let result = fixed_effects_meta(&[study]).expect("ok");

        assert_close(result.pooled_log_hr, 0.5, 1e-12);
        assert_close(result.pooled_hr, 0.5_f64.exp(), 1e-12);
        assert_close(result.variance, 0.04, 1e-12);
    }

    // ── Test 7: fixed-effects with two studies — HR between the two ───────────
    #[test]
    fn fixed_effects_meta_two() {
        let s1 = StudyHazardRatio {
            study_id: "s1".to_string(),
            log_hr: 0.2,
            log_hr_variance: 0.1,
            n_events: 50,
        };
        let s2 = StudyHazardRatio {
            study_id: "s2".to_string(),
            log_hr: 0.6,
            log_hr_variance: 0.1,
            n_events: 50,
        };
        let result = fixed_effects_meta(&[s1, s2]).expect("ok");

        // Equal variances → equal weights → pooled = mean of 0.2 and 0.6 = 0.4
        assert_close(result.pooled_log_hr, 0.4, 1e-10);
        // HR is between the two individual HRs
        assert!(result.pooled_hr > 0.2_f64.exp());
        assert!(result.pooled_hr < 0.6_f64.exp());
    }

    // ── Test 8: equal variance → equal weights → simple mean ─────────────────
    #[test]
    fn fixed_effects_meta_equal_weights() {
        let studies: Vec<StudyHazardRatio> = (0..4)
            .map(|i| StudyHazardRatio {
                study_id: format!("s{i}"),
                log_hr: i as f64 * 0.1, // 0.0, 0.1, 0.2, 0.3
                log_hr_variance: 0.25,  // all equal
                n_events: 40,
            })
            .collect();
        let result = fixed_effects_meta(&studies).expect("ok");

        // Equal weights → pooled = arithmetic mean = (0 + 0.1 + 0.2 + 0.3) / 4 = 0.15
        assert_close(result.pooled_log_hr, 0.15, 1e-10);
    }

    // ── Test 9: random-effects with zero heterogeneity → τ² = 0 ──────────────
    #[test]
    fn random_effects_meta_zero_het() {
        // All studies have the same log HR → Q = 0 → τ² = 0
        let studies: Vec<StudyHazardRatio> = (0..5)
            .map(|i| StudyHazardRatio {
                study_id: format!("s{i}"),
                log_hr: 0.3,
                log_hr_variance: 0.05,
                n_events: 80,
            })
            .collect();
        let result = random_effects_meta(&studies).expect("ok");

        assert_close(result.tau_sq, 0.0, 1e-10);
        assert_close(result.cochran_q, 0.0, 1e-10);
        assert_close(result.i_squared, 0.0, 1e-10);
    }

    // ── Test 10: random-effects I² increases with spread ─────────────────────
    #[test]
    fn random_effects_meta_high_het() {
        // Low heterogeneity
        let low_het: Vec<StudyHazardRatio> = (0..4)
            .map(|i| StudyHazardRatio {
                study_id: format!("lo{i}"),
                log_hr: 0.3 + (i as f64 - 1.5) * 0.02,
                log_hr_variance: 0.05,
                n_events: 60,
            })
            .collect();
        let r_low = random_effects_meta(&low_het).expect("ok");

        // High heterogeneity
        let high_het: Vec<StudyHazardRatio> = (0..4)
            .map(|i| StudyHazardRatio {
                study_id: format!("hi{i}"),
                log_hr: (i as f64 - 1.5) * 1.5, // large spread
                log_hr_variance: 0.05,
                n_events: 60,
            })
            .collect();
        let r_high = random_effects_meta(&high_het).expect("ok");

        assert!(
            r_high.i_squared > r_low.i_squared,
            "expected higher I² for high heterogeneity: {} vs {}",
            r_high.i_squared,
            r_low.i_squared
        );
        assert!(r_high.tau_sq > r_low.tau_sq);
    }

    // ── Test 11: random-effects ≈ fixed-effects when τ² = 0 ──────────────────
    #[test]
    fn random_effects_falls_back_to_fe_when_tau_zero() {
        let studies: Vec<StudyHazardRatio> = (0..3)
            .map(|i| StudyHazardRatio {
                study_id: format!("s{i}"),
                log_hr: 0.25,
                log_hr_variance: 0.1 + i as f64 * 0.05,
                n_events: 50,
            })
            .collect();

        let fe = fixed_effects_meta(&studies).expect("fe");
        let re = random_effects_meta(&studies).expect("re");

        // When τ²=0, RE weights = FE weights → estimates coincide
        if re.tau_sq < 1e-9 {
            assert_close(re.pooled_log_hr, fe.pooled_log_hr, 1e-8);
        }
    }

    // ── Test 12: combined log-rank significant ────────────────────────────────
    #[test]
    fn combined_log_rank_significance() {
        // Large consistent O – E across studies → small p-value
        let oe = [10.0, 12.0, 8.0, 15.0];
        let vars = [4.0, 5.0, 3.5, 6.0];
        let result = combined_log_rank(&oe, &vars).expect("ok");

        assert!(
            result.p_value < 0.05,
            "expected p < 0.05, got {}",
            result.p_value
        );
        assert!(result.z_stat > 0.0);
    }

    // ── Test 13: combined log-rank null (near-zero O – E) ─────────────────────
    #[test]
    fn combined_log_rank_null() {
        // O – E near zero → p near 1
        let oe = [0.1, -0.1, 0.05, -0.05];
        let vars = [4.0, 4.0, 4.0, 4.0];
        let result = combined_log_rank(&oe, &vars).expect("ok");

        assert!(
            result.p_value > 0.05,
            "expected p > 0.05, got {}",
            result.p_value
        );
    }

    // ── Test 14: Guyot reconstruction — survival non-increasing ───────────────
    #[test]
    fn guyot_reconstruct_monotone() {
        // Simulate a digitized KM with 5 steps
        let km_times = [0.5, 1.0, 2.0, 3.0, 4.5];
        let km_surv = [0.90, 0.80, 0.65, 0.50, 0.35];
        let risk_t = [0.0, 2.0, 4.0];
        let risk_n = [100u64, 85, 60];

        let rec = guyot_reconstruct(&km_times, &km_surv, &risk_t, &risk_n).expect("ok");

        assert_eq!(rec.survival.len(), km_times.len());
        for i in 1..rec.survival.len() {
            assert!(
                rec.survival[i] <= rec.survival[i - 1] + 1e-9,
                "reconstructed survival not monotone at {i}"
            );
        }
    }

    // ── Test 15: Guyot event counts non-negative ──────────────────────────────
    #[test]
    fn guyot_reconstruct_event_counts_positive() {
        let km_times = [1.0, 2.0, 3.0, 4.0];
        let km_surv = [0.85, 0.70, 0.55, 0.40];
        let risk_t = [0.0, 2.5];
        let risk_n = [50u64, 35];

        let rec = guyot_reconstruct(&km_times, &km_surv, &risk_t, &risk_n).expect("ok");

        for (i, &d) in rec.n_events.iter().enumerate() {
            // n_events is u64 so always ≥ 0, but also ≤ n_at_risk
            assert!(
                d <= rec.n_at_risk[i],
                "events {d} > at-risk {} at step {i}",
                rec.n_at_risk[i]
            );
        }
    }

    // ── Test 16: fixed-effects CI contains the pooled HR ─────────────────────
    #[test]
    fn fixed_effects_ci_contains_true_hr() {
        let studies = vec![
            StudyHazardRatio {
                study_id: "a".to_string(),
                log_hr: 0.4,
                log_hr_variance: 0.08,
                n_events: 80,
            },
            StudyHazardRatio {
                study_id: "b".to_string(),
                log_hr: 0.6,
                log_hr_variance: 0.12,
                n_events: 60,
            },
            StudyHazardRatio {
                study_id: "c".to_string(),
                log_hr: 0.3,
                log_hr_variance: 0.06,
                n_events: 100,
            },
        ];
        let result = fixed_effects_meta(&studies).expect("ok");

        // The 95% CI on the HR scale must bracket the pooled HR itself
        assert!(
            result.ci_lower <= result.pooled_hr,
            "CI lower {} > pooled HR {}",
            result.ci_lower,
            result.pooled_hr
        );
        assert!(
            result.ci_upper >= result.pooled_hr,
            "CI upper {} < pooled HR {}",
            result.ci_upper,
            result.pooled_hr
        );
    }

    // ── Bonus: p-value helper sanity ──────────────────────────────────────────
    #[test]
    fn normal_cdf_known_values() {
        // Φ(0) = 0.5
        assert_close(normal_cdf(0.0), 0.5, 1e-5);
        // Φ(1.96) ≈ 0.975
        assert_close(normal_cdf(1.96), 0.975, 1e-3);
        // Φ(-1.96) ≈ 0.025
        assert_close(normal_cdf(-1.96), 0.025, 1e-3);
    }

    #[test]
    fn p_from_z_two_sided() {
        // |z| = 1.96 → p ≈ 0.05
        assert_close(p_from_z(1.96), 0.05, 0.002);
        // |z| = 0 → p = 1.0
        assert_close(p_from_z(0.0), 1.0, 1e-4);
    }

    #[test]
    fn compute_study_km_rejects_empty() {
        let result = compute_study_km(&[], &[], &[], "x");
        assert!(result.is_err());
    }

    #[test]
    fn compute_study_km_rejects_mismatch() {
        let result = compute_study_km(&[1.0, 2.0], &[1], &[10, 9], "x");
        assert!(result.is_err());
    }

    #[test]
    fn pool_km_curves_rejects_empty() {
        let result = pool_km_curves(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn combined_log_rank_rejects_mismatch() {
        let result = combined_log_rank(&[1.0, 2.0], &[3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn guyot_reconstruct_rejects_non_monotone() {
        // Survival increases at step 2 — must be rejected
        let result = guyot_reconstruct(
            &[1.0, 2.0, 3.0],
            &[0.8, 0.9, 0.7], // 0.9 > 0.8 is non-monotone
            &[0.0],
            &[50],
        );
        assert!(result.is_err());
    }
}
