//! CIF-SIS: Sure Independence Screening for the cumulative incidence function
//! in competing-risks data (Fu, Parikh & Kong 2017; Hong, Chen & Li 2018).
//!
//! In high-dimensional competing-risks problems the goal is to screen out the
//! vast majority of covariates that are marginally unrelated to the cause-of-
//! interest. CIF-SIS ranks each covariate `j` by a **marginal subdistribution
//! utility** built from the Fine-Gray subdistribution score at the null
//! `β = 0`. With inverse-probability-of-censoring weights (IPCW) for the
//! subdistribution risk set, the marginal score for covariate `j` is
//!
//! ```text
//! U_j = Σ_{i : ε_i = k}  w_i · ( x_{ij} − x̄_j(t_i) ) ,
//! ```
//!
//! where `ε_i = k` flags an event of the target cause, `w_i` is the IPCW weight
//! `Ĝ(t_i) / Ĝ(min(t_i, ·))`, and `x̄_j(t_i)` is the (weighted) mean of
//! covariate `j` over the *subdistribution* risk set at `t_i` — the set of
//! subjects who either are still under follow-up or have already failed from a
//! competing cause (and hence remain "at risk" of the cause-`k` cumulative
//! incidence with a censoring-discounted weight).
//!
//! Covariates are ranked by `|U_j|` (standardised by the covariate spread) and
//! the top `d ≈ n / ln n` are retained. This is the competing-risks analogue of
//! Cox SIS in [`crate::screening::sis`].

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Configuration for CIF sure-independence screening.
#[derive(Debug, Clone)]
pub struct CifSisConfig {
    /// Number of covariates to retain. `None` → use the `⌊n / ln n⌋` rule.
    pub d: Option<usize>,
    /// Target cause of interest (`> 0`; cause `0` is reserved for censoring).
    pub target_cause: u32,
    /// Whether to standardise the marginal score by the covariate's spread
    /// (recommended so covariates on different scales are comparable).
    pub standardize: bool,
}

impl Default for CifSisConfig {
    fn default() -> Self {
        Self {
            d: None,
            target_cause: 1,
            standardize: true,
        }
    }
}

/// Result of CIF sure-independence screening.
#[derive(Debug, Clone)]
pub struct CifSisResult {
    /// Marginal utility magnitude `|U_j|` for each covariate `j = 0..p`.
    pub marginal_scores: Vec<f64>,
    /// Covariate indices ranked by `|U_j|` (most associated first).
    pub ranked_indices: Vec<usize>,
    /// Selected covariate indices (top `d`).
    pub selected_indices: Vec<usize>,
    /// Score threshold (the `d`-th largest magnitude).
    pub threshold: f64,
    /// Number of covariates retained.
    pub d: usize,
    /// Number of observations.
    pub n: usize,
    /// Number of covariates.
    pub p: usize,
}

impl CifSisResult {
    /// Marginal utility for covariate `j`.
    #[must_use]
    pub fn score(&self, j: usize) -> f64 {
        self.marginal_scores[j]
    }

    /// Whether covariate `j` survived screening.
    #[must_use]
    pub fn is_selected(&self, j: usize) -> bool {
        self.selected_indices.contains(&j)
    }
}

/// Build the Kaplan-Meier estimator `Ĝ` of the *censoring* distribution.
///
/// Returns parallel `(time, G)` step arrays (left-continuous lookup), where a
/// "censoring event" is an observation with `event = false`.
fn censoring_km(data: &Dataset) -> (Vec<f64>, Vec<f64>) {
    let order = data.order_by_time();
    let n = data.len();
    let mut g_times = Vec::new();
    let mut g_vals = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n as f64;
    let mut k = 0usize;
    while k < order.len() {
        let t = data.observations[order[k]].time;
        let mut m = k;
        let mut dc = 0.0_f64;
        while m < order.len() && data.observations[order[m]].time == t {
            if !data.observations[order[m]].event {
                dc += 1.0;
            }
            m += 1;
        }
        if dc > 0.0 && at_risk > 0.0 {
            g_cur *= 1.0 - dc / at_risk;
        }
        g_times.push(t);
        g_vals.push(g_cur);
        at_risk -= (m - k) as f64;
        k = m;
    }
    (g_times, g_vals)
}

/// Step-function lookup `Ĝ(t)` (value at the largest event time `≤ t`).
fn step_lookup(times: &[f64], vals: &[f64], t: f64) -> f64 {
    let mut v = 1.0_f64;
    for (idx, &gt) in times.iter().enumerate() {
        if gt <= t {
            v = vals[idx];
        } else {
            break;
        }
    }
    v.max(1.0e-300)
}

/// Compute the marginal CIF-SIS utility for every covariate.
///
/// `causes[i] = 0` denotes censoring; a positive value denotes the failure
/// cause for subject `i` (must be consistent with the observation event flag).
pub fn cif_sure_independence_screening(
    data: &Dataset,
    causes: &[u32],
    cfg: &CifSisConfig,
) -> SurvivalResult<CifSisResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if causes.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![causes.len()],
        });
    }
    if cfg.target_cause == 0 {
        return Err(SurvivalError::InvalidParameter(
            "target_cause must be > 0".to_string(),
        ));
    }
    let covariates = data.covariates.as_ref().ok_or_else(|| {
        SurvivalError::InvalidParameter("dataset has no covariates to screen".to_string())
    })?;
    let n = data.len();
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "dataset has no covariates to screen".to_string(),
        ));
    }
    // At least one target-cause event must be present.
    let n_target = (0..n)
        .filter(|&i| data.observations[i].event && causes[i] == cfg.target_cause)
        .count();
    if n_target == 0 {
        return Err(SurvivalError::NoEvents);
    }

    let (g_times, g_vals) = censoring_km(data);

    // Pre-compute IPCW weights at each subject's event time, w_i = 1 / Ĝ(t_i).
    // (For target-cause failures the subdistribution weight at the event time is
    // unity discounted by the censoring survival; competing failures keep a
    // residual weight that decays as Ĝ shrinks past their failure time.)
    let times: Vec<f64> = (0..n).map(|i| data.observations[i].time).collect();

    // Order subjects by ascending time so we can sweep risk sets.
    let order = data.order_by_time();

    // Accumulate scores per covariate.
    let mut scores = vec![0.0_f64; p];

    // For numerical standardisation we need each covariate's overall mean/var.
    let mut col_mean = vec![0.0_f64; p];
    for row in covariates.iter() {
        for (j, &v) in row.iter().enumerate() {
            col_mean[j] += v;
        }
    }
    for m in col_mean.iter_mut() {
        *m /= n as f64;
    }
    let mut col_sd = vec![0.0_f64; p];
    for row in covariates.iter() {
        for (j, &v) in row.iter().enumerate() {
            let d = v - col_mean[j];
            col_sd[j] += d * d;
        }
    }
    for s in col_sd.iter_mut() {
        *s = (*s / n as f64).sqrt().max(1.0e-12);
    }

    // Sweep event times of the target cause.
    for &i in order.iter() {
        if !(data.observations[i].event && causes[i] == cfg.target_cause) {
            continue;
        }
        let t_i = times[i];
        let g_ti = step_lookup(&g_times, &g_vals, t_i);
        let w_i = 1.0 / g_ti;

        // Subdistribution risk set at t_i:
        //   * subjects with time ≥ t_i (still under observation), weight 1.
        //   * subjects who already failed from a *competing* cause before t_i,
        //     weight Ĝ(t_i) / Ĝ(time_k) (Fine-Gray IPCW), keeping them at risk
        //     of cause-k cumulative incidence.
        let mut wsum = 0.0_f64;
        let mut wmean = vec![0.0_f64; p];
        for &k in order.iter() {
            let t_k = times[k];
            let weight = if t_k >= t_i {
                1.0_f64
            } else if data.observations[k].event && causes[k] != cfg.target_cause && causes[k] != 0
            {
                // Competing failure already occurred — residual subdistribution weight.
                let g_tk = step_lookup(&g_times, &g_vals, t_k);
                g_ti / g_tk.max(1.0e-300)
            } else {
                0.0
            };
            if weight <= 0.0 {
                continue;
            }
            wsum += weight;
            let cov_k = &covariates[k];
            for (acc, &x) in wmean.iter_mut().zip(cov_k.iter()) {
                *acc += weight * x;
            }
        }
        if wsum <= 0.0 {
            continue;
        }
        for acc in wmean.iter_mut() {
            *acc /= wsum;
        }

        // Score contribution: w_i · (x_{ij} − weighted risk-set mean_j).
        let cov_i = &covariates[i];
        for (j, sj) in scores.iter_mut().enumerate() {
            *sj += w_i * (cov_i[j] - wmean[j]);
        }
    }

    // Standardise and take magnitudes.
    let mut magnitudes = vec![0.0_f64; p];
    for j in 0..p {
        let s = if cfg.standardize {
            scores[j] / col_sd[j]
        } else {
            scores[j]
        };
        magnitudes[j] = s.abs();
    }

    // Rank descending by magnitude.
    let mut ranked: Vec<usize> = (0..p).collect();
    ranked.sort_by(|&a, &b| {
        magnitudes[b]
            .partial_cmp(&magnitudes[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Determine d (default n / ln n).
    let d_default = if n > 2 {
        ((n as f64) / (n as f64).ln()).floor() as usize
    } else {
        1
    };
    let d = cfg.d.unwrap_or(d_default).clamp(1, p);

    let selected_indices: Vec<usize> = ranked[..d].to_vec();
    let threshold = magnitudes[ranked[d - 1]];

    Ok(CifSisResult {
        marginal_scores: magnitudes,
        ranked_indices: ranked,
        selected_indices,
        threshold,
        d,
        n,
        p,
    })
}

/// Convenience wrapper returning a new [`Dataset`] containing only the screened
/// covariates (columns), preserving observations, strata, and the `causes`
/// vector ordering.
pub fn cif_screened_dataset(
    data: &Dataset,
    causes: &[u32],
    cfg: &CifSisConfig,
) -> SurvivalResult<(Dataset, Vec<usize>)> {
    let res = cif_sure_independence_screening(data, causes, cfg)?;
    let covariates = data
        .covariates
        .as_ref()
        .ok_or(SurvivalError::EmptyDataset)?;
    let keep = &res.selected_indices;
    let new_cov: Vec<Vec<f64>> = covariates
        .iter()
        .map(|row| keep.iter().map(|&j| row[j]).collect())
        .collect();
    let new_data = Dataset::new(
        data.observations.clone(),
        Some(new_cov),
        data.strata.clone(),
    )?;
    Ok((new_data, keep.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    /// Build a dataset with covariates and a parallel cause vector.
    fn make(times: &[f64], events: &[bool], cov: Vec<Vec<f64>>) -> Dataset {
        let obs: Vec<Observation> = times
            .iter()
            .zip(events.iter())
            .map(|(&t, &e)| Observation::new(t, e).expect("ok"))
            .collect();
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    #[test]
    fn ranks_informative_covariate_first() {
        // Covariate 0 is strongly associated with target-cause failure time;
        // covariate 1 is pure noise (constant). Early failures have high x0.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true, true, true, true, true, true];
        let causes = vec![1u32, 1, 2, 1, 2, 1];
        let cov = vec![
            vec![5.0, 1.0],
            vec![4.0, 1.0],
            vec![0.0, 1.0],
            vec![2.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let d = make(&times, &events, cov);
        let cfg = CifSisConfig {
            d: Some(2),
            ..CifSisConfig::default()
        };
        let res = cif_sure_independence_screening(&d, &causes, &cfg).expect("ok");
        assert_eq!(
            res.ranked_indices[0], 0,
            "informative covariate should rank first"
        );
        assert!(res.score(0) > res.score(1));
    }

    #[test]
    fn constant_covariate_has_zero_score() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, true, true];
        let causes = vec![1u32, 1, 1, 1];
        let cov = vec![vec![7.0], vec![7.0], vec![7.0], vec![7.0]];
        let d = make(&times, &events, cov);
        let res =
            cif_sure_independence_screening(&d, &causes, &CifSisConfig::default()).expect("ok");
        assert!(
            res.score(0).abs() < 1.0e-9,
            "constant covariate score {}",
            res.score(0)
        );
    }

    #[test]
    fn selects_requested_number() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let causes = vec![1u32, 2, 1, 2, 1];
        let cov = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 1.0, 0.0, 5.0],
            vec![3.0, 0.0, 1.0, 2.0],
            vec![0.0, 4.0, 2.0, 1.0],
            vec![4.0, 3.0, 5.0, 0.0],
        ];
        let d = make(&times, &events, cov);
        let cfg = CifSisConfig {
            d: Some(2),
            ..CifSisConfig::default()
        };
        let res = cif_sure_independence_screening(&d, &causes, &cfg).expect("ok");
        assert_eq!(res.selected_indices.len(), 2);
        assert_eq!(res.d, 2);
    }

    #[test]
    fn default_d_uses_n_over_log_n() {
        let n = 20;
        let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let events = vec![true; n];
        let causes: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let cov: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64, (n - i) as f64, 1.0])
            .collect();
        let d = make(&times, &events, cov);
        let res =
            cif_sure_independence_screening(&d, &causes, &CifSisConfig::default()).expect("ok");
        let expected = ((n as f64) / (n as f64).ln()).floor() as usize;
        // Clamped to p = 3.
        assert_eq!(res.d, expected.clamp(1, 3));
    }

    #[test]
    fn ranked_indices_is_permutation() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let causes = vec![1u32, 1, 2, 1, 2];
        let cov = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 0.0],
            vec![3.0, 0.0, 1.0],
            vec![0.0, 4.0, 2.0],
            vec![4.0, 3.0, 5.0],
        ];
        let d = make(&times, &events, cov);
        let res =
            cif_sure_independence_screening(&d, &causes, &CifSisConfig::default()).expect("ok");
        let mut seen = vec![false; res.p];
        for &idx in &res.ranked_indices {
            assert!(!seen[idx]);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn standardization_changes_ranking_for_scaled_covariates() {
        // Covariate 1 is covariate 0 multiplied by 100 → without standardisation
        // it dominates; with standardisation they tie.
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, true, true];
        let causes = vec![1u32, 1, 1, 1];
        let cov = vec![
            vec![1.0, 100.0],
            vec![2.0, 200.0],
            vec![0.0, 0.0],
            vec![3.0, 300.0],
        ];
        let d = make(&times, &events, cov);
        let raw = cif_sure_independence_screening(
            &d,
            &causes,
            &CifSisConfig {
                standardize: false,
                ..CifSisConfig::default()
            },
        )
        .expect("ok");
        let std = cif_sure_independence_screening(
            &d,
            &causes,
            &CifSisConfig {
                standardize: true,
                ..CifSisConfig::default()
            },
        )
        .expect("ok");
        assert!(raw.score(1) > raw.score(0) * 50.0);
        assert!((std.score(0) - std.score(1)).abs() < 1e-6);
    }

    #[test]
    fn screened_dataset_keeps_selected_columns() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let causes = vec![1u32, 2, 1, 2, 1];
        let cov = vec![
            vec![5.0, 1.0, 0.0],
            vec![4.0, 1.0, 9.0],
            vec![0.0, 1.0, 3.0],
            vec![2.0, 1.0, 2.0],
            vec![1.0, 1.0, 7.0],
        ];
        let d = make(&times, &events, cov);
        let cfg = CifSisConfig {
            d: Some(2),
            ..CifSisConfig::default()
        };
        let (sub, keep) = cif_screened_dataset(&d, &causes, &cfg).expect("ok");
        assert_eq!(sub.n_features(), 2);
        assert_eq!(keep.len(), 2);
        assert_eq!(sub.len(), 5);
    }

    #[test]
    fn rejects_no_target_events() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![true, true, true];
        let causes = vec![2u32, 2, 2]; // none of target cause 1
        let cov = vec![vec![1.0], vec![2.0], vec![3.0]];
        let d = make(&times, &events, cov);
        let res = cif_sure_independence_screening(&d, &causes, &CifSisConfig::default());
        assert!(matches!(res, Err(SurvivalError::NoEvents)));
    }

    #[test]
    fn rejects_cause_length_mismatch() {
        let d = make(&[1.0, 2.0], &[true, true], vec![vec![1.0], vec![2.0]]);
        let res = cif_sure_independence_screening(&d, &[1u32], &CifSisConfig::default());
        assert!(matches!(res, Err(SurvivalError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_zero_target_cause() {
        let d = make(&[1.0, 2.0], &[true, true], vec![vec![1.0], vec![2.0]]);
        let cfg = CifSisConfig {
            target_cause: 0,
            ..CifSisConfig::default()
        };
        let res = cif_sure_independence_screening(&d, &[1u32, 1], &cfg);
        assert!(matches!(res, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn rejects_no_covariates() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let res = cif_sure_independence_screening(&d, &[1u32, 1], &CifSisConfig::default());
        assert!(matches!(res, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn censoring_handled_in_weights() {
        // Mix of censored and target/competing events; should run without error
        // and produce finite scores.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true, false, true, true, false, true];
        let causes = vec![1u32, 0, 2, 1, 0, 1];
        let cov = vec![
            vec![1.0, 0.5],
            vec![2.0, 0.4],
            vec![3.0, 0.3],
            vec![4.0, 0.2],
            vec![5.0, 0.1],
            vec![6.0, 0.0],
        ];
        let d = make(&times, &events, cov);
        let res =
            cif_sure_independence_screening(&d, &causes, &CifSisConfig::default()).expect("ok");
        for &s in &res.marginal_scores {
            assert!(s.is_finite(), "non-finite score {s}");
        }
    }
}
