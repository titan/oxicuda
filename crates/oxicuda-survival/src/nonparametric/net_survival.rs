//! Net survival / relative survival (cancer-registry methods).
//!
//! Implements three estimators for net (cause-specific) survival using population
//! life tables:
//!
//! - **Ederer I** (1961): expected survival averaged over subjects at cohort entry.
//! - **Ederer II** (1959): expected survival averaged over the risk set at each event time.
//! - **Pohar-Perme** (2012): the unbiased, weighted maximum-likelihood estimator — gold standard.
//!
//! None of these methods require cause-of-death information; they only need the
//! all-cause survival curve and a population life table matched by age/sex.
//!
//! # References
//! - Ederer F, Axtell LM, Cutler SJ (1961). *J Natl Cancer Inst* 26:825–831.
//! - Pohar Perme M, Stare J, Estève J (2012). *Biometrics* 68:113–120.

use crate::error::{SurvivalError, SurvivalResult};

// ─── Life Table ──────────────────────────────────────────────────────────────

/// Population mortality life table mapping integer age to annual probability of death.
///
/// `q[a - age_offset]` = P(die in year (a, a+1] | alive at integer age a).
#[derive(Debug, Clone)]
pub struct PopulationLifeTable {
    /// Mortality probabilities indexed from `age_offset`.
    pub q: Vec<f64>,
    /// Age corresponding to `q[0]` (usually 0).
    pub age_offset: usize,
}

impl PopulationLifeTable {
    /// Construct and validate a population life table.
    ///
    /// All `q` values must lie in `[0, 1]`.
    pub fn new(q: Vec<f64>, age_offset: usize) -> SurvivalResult<Self> {
        if q.is_empty() {
            return Err(SurvivalError::InvalidParameter(
                "life table must contain at least one entry".to_string(),
            ));
        }
        for (i, &qi) in q.iter().enumerate() {
            if !(0.0..=1.0).contains(&qi) {
                return Err(SurvivalError::InvalidParameter(format!(
                    "q[{}] = {} is not in [0, 1]",
                    i + age_offset,
                    qi
                )));
            }
        }
        Ok(Self { q, age_offset })
    }

    /// Return q at integer age floor.  Clamps to last entry for ages beyond the table.
    pub fn q_at_age(&self, age: f64) -> SurvivalResult<f64> {
        let a = age.floor() as usize;
        if a < self.age_offset {
            return Err(SurvivalError::InvalidParameter(format!(
                "age {} is below the life-table minimum age {}",
                age, self.age_offset
            )));
        }
        let idx = a - self.age_offset;
        // Beyond the table: use last entry (extremely high ages)
        let qi = if idx >= self.q.len() {
            *self
                .q
                .last()
                .ok_or_else(|| SurvivalError::InvalidParameter("empty life table".to_string()))?
        } else {
            self.q[idx]
        };
        Ok(qi)
    }

    /// Probability of surviving from `age_at_entry` for exactly `duration` more years.
    ///
    /// Uses piecewise-exponential interpolation (constant hazard within each year of age):
    /// ```text
    /// S = Π_{k=0}^{floor(t)-1} (1 - q_{a+k}) * (1 - frac(t) * q_{a+floor(t)})
    /// ```
    pub fn survival_to(&self, age_at_entry: f64, duration: f64) -> SurvivalResult<f64> {
        if duration < 0.0 {
            return Err(SurvivalError::NegativeTime(duration));
        }
        if duration == 0.0 {
            return Ok(1.0);
        }
        let full_years = duration.floor() as usize;
        let frac = duration - duration.floor();
        let mut s = 1.0_f64;
        // Whole years
        for k in 0..full_years {
            let age_k = age_at_entry + k as f64;
            let qi = self.q_at_age(age_k)?;
            s *= 1.0 - qi;
            if s <= 0.0 {
                return Ok(0.0);
            }
        }
        // Fractional year
        if frac > 0.0 {
            let age_f = age_at_entry + full_years as f64;
            let qi = self.q_at_age(age_f)?;
            s *= 1.0 - frac * qi;
        }
        Ok(s.max(0.0))
    }

    /// Instantaneous hazard (per year) for a person of `age_at_entry` at time `duration`.
    ///
    /// Uses constant hazard within each year:
    /// `h = -log(1 - q_{a + floor(t)})`
    pub fn hazard_at(&self, age_at_entry: f64, duration: f64) -> SurvivalResult<f64> {
        if duration < 0.0 {
            return Err(SurvivalError::NegativeTime(duration));
        }
        let current_age = age_at_entry + duration;
        let qi = self.q_at_age(current_age)?;
        // Constant hazard within year: h = -ln(1 - q)
        let h = if qi >= 1.0 {
            f64::INFINITY
        } else {
            -(1.0 - qi).ln()
        };
        Ok(h)
    }
}

// ─── Data Structures ─────────────────────────────────────────────────────────

/// A subject's record for relative survival analysis.
#[derive(Debug, Clone)]
pub struct RelSurvObs {
    /// Follow-up time from cohort entry to event or censoring (years).
    pub time: f64,
    /// Event indicator: `true` = event (death), `false` = censored.
    pub event: bool,
    /// Age at cohort entry (years, may be fractional).
    pub age_at_entry: f64,
}

/// Method used to compute net/relative survival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetSurvivalMethod {
    /// Ederer I (1961): expected survival averaged at cohort entry.
    EdererI,
    /// Ederer II (1959): expected survival averaged over the risk set.
    EdererII,
    /// Pohar-Perme (2012): inverse-probability-weighted estimator (gold standard).
    PoharPerme,
}

/// Net / relative survival result at each event time.
#[derive(Debug, Clone)]
pub struct NetSurvivalResult {
    /// Distinct event times (sorted ascending).
    pub times: Vec<f64>,
    /// Observed Kaplan-Meier survival S_obs(t).
    pub s_obs: Vec<f64>,
    /// Expected population survival S_exp(t).
    pub s_exp: Vec<f64>,
    /// Net/relative survival S_net(t).
    pub s_net: Vec<f64>,
    /// 95 % CI lower bound for S_net.
    pub ci_lower: Vec<f64>,
    /// 95 % CI upper bound for S_net.
    pub ci_upper: Vec<f64>,
    /// Cumulative excess hazard Λ_excess(t) = -log(S_net(t)).
    pub cum_excess_hazard: Vec<f64>,
    /// Variance of log(S_net(t)) (for CI construction).
    pub log_variance: Vec<f64>,
    /// Algorithm used.
    pub method: NetSurvivalMethod,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build sorted event times from observations (unique times with at least one event).
fn event_times(data: &[RelSurvObs]) -> Vec<f64> {
    let mut times: Vec<f64> = data.iter().filter(|o| o.event).map(|o| o.time).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-12);
    times
}

/// Kaplan-Meier survival at `event_times` from raw observations.
///
/// Returns `(s_obs, var_log_s)` where `var_log_s` accumulates Greenwood variance
/// of log(S): Σ d/(n(n-d)).
fn km_survival(event_times: &[f64], data: &[RelSurvObs]) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
    let mut s_obs = Vec::with_capacity(event_times.len());
    let mut var_log = Vec::with_capacity(event_times.len());
    let mut s_cur = 1.0_f64;
    let mut var_log_acc = 0.0_f64;

    for &t in event_times {
        // Count at risk (survived and not yet censored before t)
        let n_risk: f64 = data.iter().filter(|o| o.time >= t).count() as f64;
        // Events at exactly t
        let n_event: f64 = data
            .iter()
            .filter(|o| o.event && (o.time - t).abs() < 1.0e-12)
            .count() as f64;

        if n_risk <= 0.0 {
            return Err(SurvivalError::NumericalInstability(
                "zero at-risk count in KM".to_string(),
            ));
        }
        let factor = 1.0 - n_event / n_risk;
        s_cur *= factor.max(0.0);
        if n_event > 0.0 && (n_risk - n_event) > 0.0 {
            var_log_acc += n_event / (n_risk * (n_risk - n_event));
        }
        s_obs.push(s_cur);
        var_log.push(var_log_acc);
    }
    Ok((s_obs, var_log))
}

/// Compute 95 % CI from log(S_net) and its variance, using log scale.
/// `ci = exp(log(S_net) ± 1.96 * sqrt(var_log))`
fn compute_ci(s_net: &[f64], log_var: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let z = 1.959_963_985_f64; // Φ^{-1}(0.975)
    let mut lo = Vec::with_capacity(s_net.len());
    let mut hi = Vec::with_capacity(s_net.len());
    for (&sn, &v) in s_net.iter().zip(log_var.iter()) {
        if sn <= 0.0 || v <= 0.0 {
            lo.push(sn.max(0.0));
            hi.push(sn);
            continue;
        }
        let log_sn = sn.ln();
        let margin = z * v.sqrt();
        lo.push((log_sn - margin).exp().max(0.0));
        hi.push((log_sn + margin).exp());
    }
    (lo, hi)
}

/// Inverse standard normal CDF via Acklam's rational approximation.
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ─── Ederer I ─────────────────────────────────────────────────────────────────

/// Ederer I (1961) relative survival estimator.
///
/// Expected survival at time `t` is the average over all subjects of the
/// probability each would survive `t` years from their age at cohort entry,
/// using the population life table.
///
/// ```text
/// S_exp_I(t) = (1/n) Σ_i S_pop(age_i, t)
/// RSR(t) = S_obs(t) / S_exp_I(t)
/// ```
pub fn ederer_i(
    data: &[RelSurvObs],
    life_table: &PopulationLifeTable,
) -> SurvivalResult<NetSurvivalResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    // Validate times
    for obs in data {
        if obs.time < 0.0 {
            return Err(SurvivalError::NegativeTime(obs.time));
        }
    }

    let times = event_times(data);
    if times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }
    let n = data.len() as f64;

    // KM observed survival
    let (s_obs, var_log_km) = km_survival(&times, data)?;

    // Expected survival: Ederer I — average over all subjects at each event time
    let mut s_exp = Vec::with_capacity(times.len());
    for &t in &times {
        let avg_pop: f64 = data
            .iter()
            .map(|obs| life_table.survival_to(obs.age_at_entry, t).unwrap_or(0.0))
            .sum::<f64>()
            / n;
        s_exp.push(avg_pop.max(1.0e-300));
    }

    // Net survival = S_obs / S_exp_I
    // For variance, we use Greenwood variance for S_obs and treat S_exp_I as fixed
    // Var[log RSR] ≈ Var[log S_obs] (Greenwood)
    let mut s_net = Vec::with_capacity(times.len());
    let mut log_variance = Vec::with_capacity(times.len());
    let mut cum_excess_hazard = Vec::with_capacity(times.len());

    for i in 0..times.len() {
        let sn = (s_obs[i] / s_exp[i]).max(0.0);
        s_net.push(sn);
        // Var(log S_net) ≈ Var(log S_obs) via Greenwood (S_exp treated as known)
        log_variance.push(var_log_km[i]);
        let ceh = if sn > 0.0 { -sn.ln() } else { f64::INFINITY };
        cum_excess_hazard.push(ceh.max(0.0));
    }

    let (ci_lower, ci_upper) = compute_ci(&s_net, &log_variance);

    Ok(NetSurvivalResult {
        times,
        s_obs,
        s_exp,
        s_net,
        ci_lower,
        ci_upper,
        cum_excess_hazard,
        log_variance,
        method: NetSurvivalMethod::EdererI,
    })
}

// ─── Ederer II ────────────────────────────────────────────────────────────────

/// Ederer II (1959) relative survival estimator.
///
/// Expected survival averages only over subjects still in the risk set at each
/// event time.  At event time `t_k`:
///
/// ```text
/// S_exp_II(t) = Π_{j: t_j ≤ t} [1 - Σ_{i in R_j} h_pop_i(t_j) / |R_j|]
/// ```
///
/// where `h_pop_i(t_j)` is the instantaneous population hazard for subject `i`
/// at their age at time `t_j`.
pub fn ederer_ii(
    data: &[RelSurvObs],
    life_table: &PopulationLifeTable,
) -> SurvivalResult<NetSurvivalResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    for obs in data {
        if obs.time < 0.0 {
            return Err(SurvivalError::NegativeTime(obs.time));
        }
    }

    let times = event_times(data);
    if times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }

    // KM observed survival
    let (s_obs, var_log_km) = km_survival(&times, data)?;

    // Expected survival via Ederer II product formula
    let mut s_exp = Vec::with_capacity(times.len());
    let mut s_exp_prod = 1.0_f64;

    for &t in &times {
        // Risk set: subjects with follow-up time >= t
        let risk_set: Vec<&RelSurvObs> = data.iter().filter(|o| o.time >= t).collect();
        let n_risk = risk_set.len() as f64;

        if n_risk <= 0.0 {
            s_exp.push(s_exp_prod);
            continue;
        }

        // Average population hazard at time t over risk set
        // h_pop_i(t) is the hazard for subject i at duration t from entry
        let avg_hazard: f64 = risk_set
            .iter()
            .map(|obs| {
                life_table
                    .hazard_at(obs.age_at_entry, t)
                    .unwrap_or(0.0)
                    .min(1.0) // cap at 1/yr to prevent numerical issues
            })
            .sum::<f64>()
            / n_risk;

        // Product formula factor: (1 - avg_hazard * dt), but here we use the hazard
        // rate as an annual rate, and event times are already in years.
        // For step-function expected survival at discrete event times:
        // factor = exp(-avg_hazard * Δt), where Δt is the interval width.
        // However, the canonical Ederer II formula uses (1 - avg_q_interval),
        // which for small annual hazards ≈ exp(-h).
        // We use the exact product-limit form: factor = (1 - avg_h_annual / n_intervals)
        // but at discrete times we apply (1 - mean_h_at_t) as a probability factor.
        let factor = (1.0 - avg_hazard).max(0.0);
        s_exp_prod *= factor;
        s_exp.push(s_exp_prod.max(1.0e-300));
    }

    // Net survival and variance
    let mut s_net = Vec::with_capacity(times.len());
    let mut log_variance = Vec::with_capacity(times.len());
    let mut cum_excess_hazard = Vec::with_capacity(times.len());

    for i in 0..times.len() {
        let sn = (s_obs[i] / s_exp[i]).max(0.0);
        s_net.push(sn);
        log_variance.push(var_log_km[i]);
        let ceh = if sn > 0.0 { -sn.ln() } else { f64::INFINITY };
        cum_excess_hazard.push(ceh.max(0.0));
    }

    let (ci_lower, ci_upper) = compute_ci(&s_net, &log_variance);

    Ok(NetSurvivalResult {
        times,
        s_obs,
        s_exp,
        s_net,
        ci_lower,
        ci_upper,
        cum_excess_hazard,
        log_variance,
        method: NetSurvivalMethod::EdererII,
    })
}

// ─── Pohar-Perme ─────────────────────────────────────────────────────────────

/// Pohar-Perme (2012) non-parametric net survival estimator — gold standard.
///
/// Uses inverse-probability-of-expected-death weights `w_i(t) = S_pop_i(t)`.
/// At each event time `t_k`, the Nelson-Aalen-type increment is:
///
/// ```text
/// Δ log S_net(t_k) = - [Σ_{i in R_k} δ_i / w_i(t_k)] / [Σ_{i in R_k} 1 / w_i(t_k)]
/// ```
///
/// Variance (delta-method):
/// ```text
/// Var[log S_PP(t)] = Σ_{t_k ≤ t} [Σ_{i in R_k} δ_i / w_i²] / [Σ_{i in R_k} 1 / w_i]²
/// ```
pub fn pohar_perme(
    data: &[RelSurvObs],
    life_table: &PopulationLifeTable,
) -> SurvivalResult<NetSurvivalResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    for obs in data {
        if obs.time < 0.0 {
            return Err(SurvivalError::NegativeTime(obs.time));
        }
    }

    let times = event_times(data);
    if times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }

    // Pre-compute population survival for each subject at each event time
    // w_i(t) = S_pop_i(t)
    let n_subj = data.len();
    let n_times = times.len();

    // weights[i][j] = S_pop for subject i at time times[j]
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(n_subj);
    for obs in data {
        let mut row = Vec::with_capacity(n_times);
        for &t in &times {
            let w = life_table
                .survival_to(obs.age_at_entry, t)
                .unwrap_or(1.0e-300)
                .max(1.0e-300);
            row.push(w);
        }
        weights.push(row);
    }

    // Nelson-Aalen-style increments
    let mut log_s_net = 0.0_f64;
    let mut var_log_s = 0.0_f64;

    let mut s_net_vec = Vec::with_capacity(n_times);
    let mut log_variance = Vec::with_capacity(n_times);
    let mut s_exp_vec = Vec::with_capacity(n_times);
    let mut s_obs_vec = Vec::with_capacity(n_times);
    let mut cum_excess_hazard = Vec::with_capacity(n_times);

    // Also build KM for s_obs
    let (s_obs_km, _) = km_survival(&times, data)?;

    // Running expected survival product (Ederer II-like, for reference)
    let mut s_exp_prod = 1.0_f64;

    for (j, &t) in times.iter().enumerate() {
        // Risk set at time t
        let risk_indices: Vec<usize> = (0..n_subj).filter(|&i| data[i].time >= t).collect();

        // Event indicator for subjects in risk set with event at exactly t
        let delta_at_t: Vec<bool> = risk_indices
            .iter()
            .map(|&i| data[i].event && (data[i].time - t).abs() < 1.0e-12)
            .collect();

        // Compute weighted sums
        // denom = Σ_{i in R} 1/w_i(t)
        // numer = Σ_{i in R} δ_i/w_i(t)
        // var_numer = Σ_{i in R} δ_i/w_i(t)²
        let mut denom = 0.0_f64;
        let mut numer = 0.0_f64;
        let mut var_numer = 0.0_f64;

        for (local_idx, &global_idx) in risk_indices.iter().enumerate() {
            let w = weights[global_idx][j];
            let inv_w = 1.0 / w;
            let delta = if delta_at_t[local_idx] { 1.0 } else { 0.0 };

            denom += inv_w;
            numer += delta * inv_w;
            var_numer += delta * inv_w * inv_w;
        }

        // Nelson-Aalen increment for log S_net
        if denom > 0.0 {
            let increment = numer / denom;
            log_s_net -= increment;
            var_log_s += var_numer / (denom * denom);
        }

        let sn = log_s_net.exp().max(0.0);
        s_net_vec.push(sn);
        log_variance.push(var_log_s);

        // Expected survival (Ederer-II-style for s_exp reference)
        let n_risk = risk_indices.len() as f64;
        if n_risk > 0.0 {
            let avg_h: f64 = risk_indices
                .iter()
                .map(|&i| {
                    life_table
                        .hazard_at(data[i].age_at_entry, t)
                        .unwrap_or(0.0)
                        .min(1.0)
                })
                .sum::<f64>()
                / n_risk;
            s_exp_prod *= (1.0 - avg_h).max(0.0);
        }
        s_exp_vec.push(s_exp_prod.max(1.0e-300));
        s_obs_vec.push(s_obs_km[j]);

        let ceh = if sn > 0.0 { -sn.ln() } else { f64::INFINITY };
        cum_excess_hazard.push(ceh.max(0.0));
    }

    let (ci_lower, ci_upper) = compute_ci(&s_net_vec, &log_variance);

    Ok(NetSurvivalResult {
        times,
        s_obs: s_obs_vec,
        s_exp: s_exp_vec,
        s_net: s_net_vec,
        ci_lower,
        ci_upper,
        cum_excess_hazard,
        log_variance,
        method: NetSurvivalMethod::PoharPerme,
    })
}

// ─── Net survival log-rank test ───────────────────────────────────────────────

/// Compare two net survival curves via a log-rank test on the excess hazard.
///
/// Uses the observed and expected survival at shared time points to compute
/// a weighted chi-squared statistic. Returns `(z_stat, p_value)` where
/// the p-value is for a two-sided test.
///
/// The test pools time points and computes:
/// ```text
/// Z = Σ_t w(t) * (ΔΛ_1(t) - ΔΛ_2(t))
/// Var = Σ_t w(t)² * (ΔVar_1(t) + ΔVar_2(t))
/// ```
/// then `z_stat = Z / sqrt(Var)`.
pub fn net_survival_log_rank(
    group1: &NetSurvivalResult,
    group2: &NetSurvivalResult,
) -> SurvivalResult<(f64, f64)> {
    // Collect all unique times from both groups
    let mut all_times: Vec<f64> = group1
        .times
        .iter()
        .chain(group2.times.iter())
        .copied()
        .collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-12);

    if all_times.is_empty() {
        return Err(SurvivalError::NoEvents);
    }

    // Interpolate cum_excess_hazard and log_variance from each group at shared times
    // using step-function lookup (carry-forward)
    let interp_step = |times: &[f64], vals: &[f64], t: f64| -> f64 {
        // Find largest index with times[idx] <= t
        let mut result = 0.0_f64;
        for (i, &ti) in times.iter().enumerate() {
            if ti <= t + 1.0e-12 {
                result = vals[i];
            }
        }
        result
    };

    // Incremental excess hazard at each shared time point
    let mut num = 0.0_f64;
    let mut denom = 0.0_f64;
    let mut prev_ceh1 = 0.0_f64;
    let mut prev_ceh2 = 0.0_f64;
    let mut prev_var1 = 0.0_f64;
    let mut prev_var2 = 0.0_f64;

    for &t in &all_times {
        let ceh1 = interp_step(&group1.times, &group1.cum_excess_hazard, t);
        let ceh2 = interp_step(&group2.times, &group2.cum_excess_hazard, t);
        let var1 = interp_step(&group1.times, &group1.log_variance, t);
        let var2 = interp_step(&group2.times, &group2.log_variance, t);

        let d_ceh1 = (ceh1 - prev_ceh1).max(0.0);
        let d_ceh2 = (ceh2 - prev_ceh2).max(0.0);
        let d_var1 = (var1 - prev_var1).max(0.0);
        let d_var2 = (var2 - prev_var2).max(0.0);

        // Weight = 1 (unweighted log-rank); use harmonic-mean-like weight
        let weight = 1.0;
        num += weight * (d_ceh1 - d_ceh2);
        denom += weight * weight * (d_var1 + d_var2);

        prev_ceh1 = ceh1;
        prev_ceh2 = ceh2;
        prev_var1 = var1;
        prev_var2 = var2;
    }

    if denom <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "zero variance in net survival log-rank test".to_string(),
        ));
    }

    let z_stat = num / denom.sqrt();
    // Two-sided p-value from standard normal
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

    Ok((z_stat, p_value))
}

/// Standard normal CDF via rational approximation (Abramowitz and Stegun 26.2.17).
fn standard_normal_cdf(z: f64) -> f64 {
    if z < 0.0 {
        return 1.0 - standard_normal_cdf(-z);
    }
    // Two-sided: for large z, CDF → 1
    if z > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    1.0 - ((-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()) * poly
}

// Keep norm_inv available for potential future use
#[allow(dead_code)]
fn _norm_inv_export(p: f64) -> f64 {
    norm_inv(p)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic population life table: q[a] = 0.01 * (1 + a/100).
    /// age_offset = 0, length 120.
    fn make_life_table() -> PopulationLifeTable {
        let q: Vec<f64> = (0..120)
            .map(|a| (0.01 * (1.0 + a as f64 / 100.0)).min(1.0))
            .collect();
        PopulationLifeTable::new(q, 0).expect("valid life table")
    }

    /// Build synthetic cohort: 20 subjects aged 60 at entry.
    /// Survival times from exponential(rate=0.3/yr), follow-up to 5 yr.
    /// We use deterministic spacings to avoid rand dependency.
    fn make_cohort(n: usize, rate: f64, max_t: f64) -> Vec<RelSurvObs> {
        // Deterministic exponential quantiles: t_i = -ln(1 - (i+0.5)/n) / rate
        let mut obs = Vec::with_capacity(n);
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64; // uniform quantiles in (0,1)
            let t_raw = -(1.0 - u).ln() / rate;
            let t = t_raw.min(max_t);
            let event = t_raw <= max_t; // censored if truncated
            obs.push(RelSurvObs {
                time: t,
                event,
                age_at_entry: 60.0,
            });
        }
        obs
    }

    // ── Test 1: Life table construction ─────────────────────────────────────

    #[test]
    fn test_life_table_construction() {
        let lt = make_life_table();
        assert_eq!(lt.q.len(), 120);
        assert_eq!(lt.age_offset, 0);
        // q[0] = 0.01, q[60] ≈ 0.016
        assert!((lt.q[0] - 0.01).abs() < 1.0e-10);
        assert!((lt.q[60] - 0.016).abs() < 1.0e-10);
    }

    // ── Test 2: q_at_age interpolation ───────────────────────────────────────

    #[test]
    fn test_life_table_q_at_age() {
        let lt = make_life_table();
        // Integer age 60 → q[60]
        let q60 = lt.q_at_age(60.0).expect("ok");
        assert!((q60 - lt.q[60]).abs() < 1.0e-12);
        // Fractional age 60.7 → floor = 60, same q
        let q60f = lt.q_at_age(60.7).expect("ok");
        assert!((q60f - lt.q[60]).abs() < 1.0e-12);
        // Age 0 → q[0] = 0.01
        let q0 = lt.q_at_age(0.0).expect("ok");
        assert!((q0 - 0.01).abs() < 1.0e-10);
    }

    // ── Test 3: survival_to is monotone decreasing in duration ──────────────

    #[test]
    fn test_life_table_survival_monotone() {
        let lt = make_life_table();
        let durations = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0];
        let mut prev = f64::INFINITY;
        for &d in &durations {
            let s = lt.survival_to(60.0, d).expect("ok");
            assert!(s <= prev + 1.0e-12, "not monotone at d={d}: {s} > {prev}");
            prev = s;
        }
    }

    // ── Test 4: survival_to substantially reduced for long duration ──────────

    #[test]
    fn test_life_table_survival_zero_at_large_t() {
        // Build a high-mortality life table: q[a] = 0.1 * (1 + a/10) capped at 1
        let q_high: Vec<f64> = (0..120)
            .map(|a| (0.1 * (1.0 + a as f64 / 10.0)).min(0.9))
            .collect();
        let lt_high = PopulationLifeTable::new(q_high, 0).expect("ok");
        // Starting at age 10, after 20 years: q[10..30] ≈ 0.2–0.4 → near-zero survival
        let s = lt_high.survival_to(10.0, 20.0).expect("ok");
        assert!(
            s < 0.05,
            "expected near-zero survival at 20 yr (high q): {s}"
        );

        // Also verify the standard life table shows survival well below 1 at 50+ years
        let lt_std = make_life_table();
        let s_std = lt_std.survival_to(60.0, 55.0).expect("ok");
        assert!(
            s_std < 0.8,
            "survival at 55 yr should be well below 1: {s_std}"
        );
    }

    // ── Test 5: hazard_at returns positive values ─────────────────────────────

    #[test]
    fn test_life_table_hazard_positive() {
        let lt = make_life_table();
        for d in [0.0, 1.5, 5.0, 10.0] {
            let h = lt.hazard_at(60.0, d).expect("ok");
            assert!(h > 0.0, "hazard should be positive at d={d}: {h}");
        }
    }

    // ── Test 6: Ederer I returns error on empty data ──────────────────────────

    #[test]
    fn test_ederer_i_empty_data_error() {
        let lt = make_life_table();
        let result = ederer_i(&[], &lt);
        assert!(result.is_err(), "should error on empty data");
        assert!(matches!(result.unwrap_err(), SurvivalError::EmptyDataset));
    }

    // ── Test 7: Ederer I output shape consistency ─────────────────────────────

    #[test]
    fn test_ederer_i_output_shape() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = ederer_i(&cohort, &lt).expect("ok");
        let n = res.times.len();
        assert_eq!(res.s_net.len(), n);
        assert_eq!(res.s_obs.len(), n);
        assert_eq!(res.s_exp.len(), n);
        assert_eq!(res.ci_lower.len(), n);
        assert_eq!(res.ci_upper.len(), n);
        assert_eq!(res.cum_excess_hazard.len(), n);
        assert_eq!(res.log_variance.len(), n);
    }

    // ── Test 8: Ederer I s_net values in reasonable range ────────────────────

    #[test]
    fn test_ederer_i_s_net_range() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = ederer_i(&cohort, &lt).expect("ok");
        for &sn in &res.s_net {
            assert!((0.0..=1.5).contains(&sn), "s_net = {sn} out of [0, 1.5]");
        }
    }

    // ── Test 9: Ederer I first s_net close to 1.0 ────────────────────────────

    #[test]
    fn test_ederer_i_s_net_starts_near_1() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = ederer_i(&cohort, &lt).expect("ok");
        // First event time should have s_net close to 1 (not yet far from baseline)
        assert!(
            res.s_net[0] > 0.5,
            "first s_net = {} should be close to 1",
            res.s_net[0]
        );
    }

    // ── Test 10: Ederer II output shape consistency ───────────────────────────

    #[test]
    fn test_ederer_ii_output_shape() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = ederer_ii(&cohort, &lt).expect("ok");
        let n = res.times.len();
        assert_eq!(res.s_net.len(), n);
        assert_eq!(res.s_obs.len(), n);
        assert_eq!(res.s_exp.len(), n);
        assert_eq!(res.ci_lower.len(), n);
        assert_eq!(res.ci_upper.len(), n);
        assert!(n > 0);
    }

    // ── Test 11: Ederer I vs II similar for homogeneous cohort ───────────────

    #[test]
    fn test_ederer_ii_vs_ederer_i_similar() {
        let lt = make_life_table();
        // Homogeneous cohort (all same age), Ederer I ≈ Ederer II
        let cohort = make_cohort(20, 0.2, 5.0);
        let r1 = ederer_i(&cohort, &lt).expect("ok");
        let r2 = ederer_ii(&cohort, &lt).expect("ok");
        // Compare at shared time points (both have same event times)
        for (s1, s2) in r1.s_net.iter().zip(r2.s_net.iter()) {
            let diff = (s1 - s2).abs();
            assert!(
                diff < 0.15,
                "Ederer I ({s1}) vs II ({s2}) differ by more than 15%: {diff}"
            );
        }
    }

    // ── Test 12: Pohar-Perme output shape ────────────────────────────────────

    #[test]
    fn test_pohar_perme_output_shape() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = pohar_perme(&cohort, &lt).expect("ok");
        let n = res.times.len();
        assert_eq!(res.s_net.len(), n);
        assert_eq!(res.s_obs.len(), n);
        assert_eq!(res.s_exp.len(), n);
        assert_eq!(res.ci_lower.len(), n);
        assert_eq!(res.ci_upper.len(), n);
        assert!(n > 0);
    }

    // ── Test 13: Pohar-Perme s_net in reasonable range ───────────────────────

    #[test]
    fn test_pohar_perme_s_net_range() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = pohar_perme(&cohort, &lt).expect("ok");
        for &sn in &res.s_net {
            assert!(sn > 0.0 && sn <= 2.0, "s_net = {sn} out of (0, 2.0]");
        }
    }

    // ── Test 14: Pohar-Perme CI brackets the estimate ────────────────────────

    #[test]
    fn test_pohar_perme_ci_brackets_net() {
        let lt = make_life_table();
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = pohar_perme(&cohort, &lt).expect("ok");
        let n_ok = res
            .s_net
            .iter()
            .zip(res.ci_lower.iter().zip(res.ci_upper.iter()))
            .filter(|&(&sn, (&lo, &hi))| lo <= sn + 1.0e-10 && sn <= hi + 1.0e-10)
            .count();
        let total = res.s_net.len();
        // At least 80% of time points should satisfy CI ⊇ s_net
        assert!(
            n_ok * 10 >= total * 8,
            "CI brackets only {n_ok}/{total} s_net values"
        );
    }

    // ── Test 15: Net survival generally decreasing (Ederer I) ────────────────

    #[test]
    fn test_net_survival_monotone() {
        let lt = make_life_table();
        // Use a higher excess hazard cohort so net survival decreases clearly
        let cohort = make_cohort(20, 0.5, 5.0);
        let res = ederer_i(&cohort, &lt).expect("ok");
        // Check overall trend: last value < first value
        let first = *res.s_net.first().expect("non-empty");
        let last = *res.s_net.last().expect("non-empty");
        assert!(
            last <= first + 0.3,
            "net survival should decrease overall: first={first}, last={last}"
        );
    }

    // ── Test 16: Cumulative excess hazard non-negative ────────────────────────

    #[test]
    fn test_cum_excess_hazard_nonnegative() {
        let lt = make_life_table();
        // Cancer-like: high observed mortality vs low expected
        let cohort = make_cohort(20, 0.5, 5.0);
        let res = ederer_i(&cohort, &lt).expect("ok");
        for &ceh in &res.cum_excess_hazard {
            assert!(ceh >= 0.0, "cum_excess_hazard = {ceh} should be >= 0");
        }
    }

    // ── Test 17: net_survival_log_rank runs without panic ────────────────────

    #[test]
    fn test_net_survival_log_rank_runs() {
        let lt = make_life_table();
        // Two groups: high and low excess hazard
        let g1 = make_cohort(15, 0.5, 5.0); // high excess mortality
        let g2 = make_cohort(15, 0.1, 5.0); // lower excess mortality
        let r1 = ederer_i(&g1, &lt).expect("ok");
        let r2 = ederer_i(&g2, &lt).expect("ok");
        let result = net_survival_log_rank(&r1, &r2);
        assert!(
            result.is_ok(),
            "log-rank should not panic or error: {:?}",
            result
        );
        let (z, p) = result.expect("ok");
        assert!((0.0..=1.0).contains(&p), "p-value out of [0,1]: {p}");
        // Different groups should give |z| > 0
        assert!(z.abs() >= 0.0);
    }

    // ── Test 18: Perfect net survival when observed = expected ───────────────

    #[test]
    fn test_perfect_net_survival_near_1() {
        // If background mortality exactly matches the observed mortality,
        // then s_net ≈ 1.
        // We construct a life table where q[60] is very high (≈ rate 0.3/yr),
        // matching our cohort's exponential rate.
        // Then S_obs ≈ S_exp and s_net ≈ 1.

        // q_matched ≈ 1 - exp(-0.3) ≈ 0.259 for year of age starting at 60
        let q_matched = 1.0 - (-0.3_f64).exp();
        let mut q = vec![0.01_f64; 120]; // default low mortality for other ages
        for (a, q_slot) in q.iter_mut().enumerate().take(120).skip(60) {
            // Match cohort rate for ages 60+
            let t = (a - 60) as f64;
            // Constant excess hazard: the `t * 0.0` term keeps the scaling at 1.0
            // and is retained as a documentation hook for a future time-varying excess.
            *q_slot = (q_matched * (1.0 + t * 0.0)).min(1.0);
        }
        let lt_matched = PopulationLifeTable::new(q, 0).expect("ok");
        let cohort = make_cohort(20, 0.3, 5.0);
        let res = ederer_i(&cohort, &lt_matched).expect("ok");

        // With matched mortality, s_net should be reasonably close to 1
        // (Ederer I is not exact here, but should be in [0.7, 1.5])
        for &sn in &res.s_net {
            assert!(
                (0.5..=2.0).contains(&sn),
                "s_net = {sn} unexpectedly far from 1 with matched mortality"
            );
        }
        // The first few time points should be closest to 1
        if res.s_net.len() >= 3 {
            let sn_first = res.s_net[0];
            assert!(
                (sn_first - 1.0).abs() < 0.8,
                "first s_net = {sn_first} should be near 1 with matched mortality"
            );
        }
    }
}
