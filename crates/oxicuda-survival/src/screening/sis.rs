//! Sure Independence Screening (SIS) for Cox proportional-hazards models.
//!
//! Implements the marginal partial-likelihood score screening of Fan & Song (2010).
//! For each covariate j, computes the marginal score at β=0:
//!
//! ```text
//! U_j = Σ_{i: event_i=1} [ x_{ij} - mean_{k ∈ R(t_i)} x_{kj} ]
//! ```
//!
//! where R(t) is the risk set {k : time_k ≥ t_i}. Features are ranked by |U_j|
//! and the top d are selected, with d = min(p, max(1, floor(n / ln(n)))) by default.

use crate::data::{Dataset, Observation};
use crate::error::{SurvivalError, SurvivalResult};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Tie-handling method for SIS partial-likelihood scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SisTieMethod {
    /// Breslow approximation (default).
    Breslow,
    /// Efron approximation.
    Efron,
}

/// Configuration for sure independence screening.
#[derive(Debug, Clone)]
pub struct SisConfig {
    /// Number of features to select. `None` → use n / ln(n) rule.
    pub d: Option<usize>,
    /// Tie-handling method.
    pub tie_method: SisTieMethod,
}

impl Default for SisConfig {
    fn default() -> Self {
        Self {
            d: None,
            tie_method: SisTieMethod::Breslow,
        }
    }
}

/// Result of sure independence screening.
#[derive(Debug, Clone)]
pub struct SisResult {
    /// Marginal score magnitude |U_j| for each feature j=0..p.
    pub marginal_scores: Vec<f64>,
    /// Feature indices sorted by |U_j| descending (most important first).
    pub ranked_indices: Vec<usize>,
    /// Selected feature indices (top d from ranked_indices).
    pub selected_indices: Vec<usize>,
    /// Score threshold = |U_{d-th feature}|.
    pub threshold: f64,
    /// Number of features selected.
    pub d: usize,
    /// Number of observations.
    pub n: usize,
    /// Number of features.
    pub p: usize,
}

impl SisResult {
    /// Score magnitude for feature `j`.
    #[must_use]
    pub fn score(&self, j: usize) -> f64 {
        self.marginal_scores[j]
    }

    /// Whether feature `j` was selected.
    #[must_use]
    pub fn is_selected(&self, j: usize) -> bool {
        self.selected_indices.contains(&j)
    }
}

// ─── Core algorithm ───────────────────────────────────────────────────────────

/// Compute marginal partial-likelihood scores for all features (Breslow version).
///
/// Algorithm:
/// 1. Sort observations by time ascending.
/// 2. For each event time t_i, compute risk-set mean of each feature.
/// 3. For each event at t_i: U_j += x_{event,j} - risk_set_mean_j.
fn compute_scores_breslow(
    covariates: &[Vec<f64>],
    sorted_indices: &[usize],
    observations: &[Observation],
    p: usize,
) -> Vec<f64> {
    let n = sorted_indices.len();
    let mut scores = vec![0.0_f64; p];

    let mut event_idx = 0usize;
    while event_idx < n {
        let obs_i = sorted_indices[event_idx];
        let obs = &observations[obs_i];

        // Skip censored observations.
        if !obs.event {
            event_idx += 1;
            continue;
        }

        let t_i = obs.time;

        // Identify all tied events at t_i.
        let tie_end = {
            let mut k = event_idx;
            while k < n && (observations[sorted_indices[k]].time - t_i).abs() < 1e-12 {
                k += 1;
            }
            k
        };

        // Risk set = all obs with time >= t_i.
        // In sorted order, risk_set starts from the first obs with time >= t_i.
        // Since sorted ascending, find the first index whose time >= t_i.
        // All from that point onwards are in the risk set.
        let risk_start = {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if observations[sorted_indices[mid]].time < t_i {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let risk_size = n - risk_start;

        if risk_size == 0 {
            event_idx = tie_end;
            continue;
        }

        // Compute risk-set mean for each feature.
        let mut risk_means = vec![0.0_f64; p];
        for &obs_k in sorted_indices[risk_start..n].iter() {
            let cov_k = &covariates[obs_k];
            for (mean_j, &x_kj) in risk_means.iter_mut().zip(cov_k.iter()) {
                *mean_j += x_kj;
            }
        }
        for mean_j in risk_means.iter_mut() {
            *mean_j /= risk_size as f64;
        }

        // Accumulate score for each event at t_i.
        for &obs_e in sorted_indices[event_idx..tie_end].iter() {
            if observations[obs_e].event {
                let cov_e = &covariates[obs_e];
                for ((score_j, &x_ej), &mu_j) in
                    scores.iter_mut().zip(cov_e.iter()).zip(risk_means.iter())
                {
                    *score_j += x_ej - mu_j;
                }
            }
        }

        event_idx = tie_end;
    }

    scores
}

/// Compute marginal partial-likelihood scores using the Efron tie correction.
///
/// For each distinct event time t_i with events E_i (set of event indices),
/// the Efron correction modifies the risk-set mean:
/// At the m-th event (m=1...|E_i|), the effective mean is:
///   μ_j(m) = risk_mean_j - (m-1)/|E_i| * event_mean_j
fn compute_scores_efron(
    covariates: &[Vec<f64>],
    sorted_indices: &[usize],
    observations: &[Observation],
    p: usize,
) -> Vec<f64> {
    let n = sorted_indices.len();
    let mut scores = vec![0.0_f64; p];

    let mut event_idx = 0usize;
    while event_idx < n {
        let obs_i = sorted_indices[event_idx];
        let obs = &observations[obs_i];

        if !obs.event {
            event_idx += 1;
            continue;
        }

        let t_i = obs.time;

        // Tied events at t_i.
        let tie_end = {
            let mut k = event_idx;
            while k < n && (observations[sorted_indices[k]].time - t_i).abs() < 1e-12 {
                k += 1;
            }
            k
        };

        // Risk set start.
        let risk_start = {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if observations[sorted_indices[mid]].time < t_i {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let risk_size = n - risk_start;
        if risk_size == 0 {
            event_idx = tie_end;
            continue;
        }

        // Risk-set mean.
        let mut risk_means = vec![0.0_f64; p];
        for &obs_k in sorted_indices[risk_start..n].iter() {
            let cov_k = &covariates[obs_k];
            for (mean_j, &x_kj) in risk_means.iter_mut().zip(cov_k.iter()) {
                *mean_j += x_kj;
            }
        }
        for mean_j in risk_means.iter_mut() {
            *mean_j /= risk_size as f64;
        }

        // Collect event indices at this time.
        let event_obs_at_t: Vec<usize> = sorted_indices[event_idx..tie_end]
            .iter()
            .filter(|&&obs_pos| observations[obs_pos].event)
            .copied()
            .collect();
        let d_i = event_obs_at_t.len();

        if d_i == 0 {
            event_idx = tie_end;
            continue;
        }

        // Event mean for Efron correction.
        let mut event_means = vec![0.0_f64; p];
        for &obs_e in &event_obs_at_t {
            let cov_e = &covariates[obs_e];
            for (emean_j, &x_ej) in event_means.iter_mut().zip(cov_e.iter()) {
                *emean_j += x_ej;
            }
        }
        for emean_j in event_means.iter_mut() {
            *emean_j /= d_i as f64;
        }

        // Score contribution: Efron-corrected.
        for (m, &obs_e) in event_obs_at_t.iter().enumerate() {
            let frac = m as f64 / d_i as f64;
            let cov_e = &covariates[obs_e];
            for (((score_j, &x_ej), &mu_j), &emean_j) in scores
                .iter_mut()
                .zip(cov_e.iter())
                .zip(risk_means.iter())
                .zip(event_means.iter())
            {
                let efron_mean = mu_j - frac * emean_j;
                *score_j += x_ej - efron_mean;
            }
        }

        event_idx = tie_end;
    }

    scores
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Perform sure independence screening on a survival dataset.
///
/// # Errors
/// - `EmptyDataset` if the dataset has no observations.
/// - `NoEvents` if there are no events.
/// - `InvalidConfiguration` if the dataset has no covariates.
pub fn sure_independence_screening(
    data: &Dataset,
    config: &SisConfig,
) -> SurvivalResult<SisResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidConfiguration(
            "dataset must have covariates for SIS".to_owned(),
        ));
    }

    let n = data.len();
    let covariates = data.covariates.as_ref().ok_or_else(|| {
        SurvivalError::InvalidConfiguration("dataset must have covariates for SIS".to_owned())
    })?;
    let sorted_indices = data.order_by_time();

    // Compute raw scores U_j.
    let raw_scores = match config.tie_method {
        SisTieMethod::Breslow => {
            compute_scores_breslow(covariates, &sorted_indices, &data.observations, p)
        }
        SisTieMethod::Efron => {
            compute_scores_efron(covariates, &sorted_indices, &data.observations, p)
        }
    };

    // Take absolute values.
    let marginal_scores: Vec<f64> = raw_scores.iter().map(|s| s.abs()).collect();

    // Determine d.
    let d = config.d.unwrap_or_else(|| {
        let ln_n = (n as f64).ln().max(1.0);
        let d_auto = (n as f64 / ln_n) as usize;
        d_auto.max(1).min(p)
    });
    let d = d.min(p).max(1);

    // Rank features by |U_j| descending.
    let mut ranked_indices: Vec<usize> = (0..p).collect();
    ranked_indices.sort_by(|&a, &b| {
        marginal_scores[b]
            .partial_cmp(&marginal_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let selected_indices: Vec<usize> = ranked_indices[..d].to_vec();

    // Threshold = score of the last selected feature.
    let threshold = selected_indices
        .last()
        .map(|&j| marginal_scores[j])
        .unwrap_or(0.0);

    Ok(SisResult {
        marginal_scores,
        ranked_indices,
        selected_indices,
        threshold,
        d,
        n,
        p,
    })
}

/// Build a new dataset restricted to the features selected by SIS.
///
/// The returned dataset has `n_features() == sis.selected_indices.len()`.
/// Observations (times, events) and strata are unchanged; only the covariates
/// are projected onto the selected columns.
///
/// # Errors
/// - `InvalidConfiguration` if the original dataset has no covariates.
pub fn screened_dataset(data: &Dataset, sis: &SisResult) -> SurvivalResult<Dataset> {
    let covariates = data.covariates.as_ref().ok_or_else(|| {
        SurvivalError::InvalidConfiguration("original dataset has no covariates".to_owned())
    })?;

    let new_covs: Vec<Vec<f64>> = covariates
        .iter()
        .map(|row| sis.selected_indices.iter().map(|&j| row[j]).collect())
        .collect();

    let obs_clone = data.observations.clone();
    let strata_clone = data.strata.clone();

    Dataset::new(obs_clone, Some(new_covs), strata_clone)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(times: &[f64], events: &[bool], covs: &[Vec<f64>]) -> Dataset {
        let obs: Vec<_> = times
            .iter()
            .zip(events)
            .map(|(&t, &e)| Observation::new(t, e).expect("new should succeed"))
            .collect();
        Dataset::new(obs, Some(covs.to_vec()), None).expect("value should be present")
    }

    // Test 1: True signal selected — x_1 separates early/late events.
    #[test]
    fn true_signal_selected() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, true, true, true, true];
        // x_1: strong separator (late events have high x_1).
        // x_2: noise.
        let covs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(1),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        // Feature 0 (x_1) should have strictly larger |U| than feature 1 (x_2).
        assert!(
            result.marginal_scores[0] > result.marginal_scores[1],
            "signal score={}, noise score={}",
            result.marginal_scores[0],
            result.marginal_scores[1]
        );
        assert_eq!(result.selected_indices, vec![0]);
    }

    // Test 2: d=1 selects exactly 1 feature.
    #[test]
    fn d1_selects_one_feature() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, true, true, true, true];
        let covs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(1),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        assert_eq!(result.selected_indices.len(), 1);
        assert_eq!(result.d, 1);
    }

    // Test 3: d=p selects all features.
    #[test]
    fn dp_selects_all_features() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let covs = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(3),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        assert_eq!(result.selected_indices.len(), 3);
        assert_eq!(result.d, 3);
    }

    // Test 4: d=None with n=100, p=10 → auto d capped at p.
    #[test]
    fn auto_d_capped_at_p() {
        // n=100, ln(100) ≈ 4.605, floor(100/4.605) = 21 → capped at p=10.
        let mut times = Vec::new();
        let mut events = Vec::new();
        let mut covs = Vec::new();
        for i in 0..100usize {
            times.push((i + 1) as f64);
            events.push(true);
            covs.push(vec![i as f64; 10]);
        }
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig::default();
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        assert_eq!(result.d, 10);
        assert_eq!(result.selected_indices.len(), 10);
    }

    // Test 5: ranked_indices.len() == p.
    #[test]
    fn ranked_indices_length_equals_p() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, false, true];
        let covs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default())
            .expect("value should be present");
        assert_eq!(result.ranked_indices.len(), 4);
        assert_eq!(result.p, 4);
    }

    // Test 6: selected_indices.len() == min(d, p).
    #[test]
    fn selected_indices_length_correct() {
        let times = [1.0, 2.0];
        let events = [true, true];
        let covs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(5),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        // d=5 > p=2 → selected = 2.
        assert_eq!(result.selected_indices.len(), 2);
    }

    // Test 7: All marginal_scores are non-negative.
    #[test]
    fn scores_non_negative() {
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, false, true, true];
        let covs = vec![
            vec![-5.0, 3.0],
            vec![2.0, -1.0],
            vec![0.0, 4.0],
            vec![-2.0, -3.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default())
            .expect("value should be present");
        for &s in &result.marginal_scores {
            assert!(s >= 0.0, "score {} < 0", s);
        }
    }

    // Test 8: ranked_indices is a permutation of 0..p.
    #[test]
    fn ranked_indices_is_permutation() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let covs = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default())
            .expect("value should be present");
        let mut sorted = result.ranked_indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    // Test 9: threshold == marginal_scores[selected_indices.last()].
    #[test]
    fn threshold_equals_last_selected_score() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, true, true, true, true];
        let covs = vec![
            vec![1.0, 0.5, 0.1, 2.0],
            vec![2.0, 1.0, 0.2, 1.5],
            vec![3.0, 1.5, 0.3, 1.0],
            vec![4.0, 2.0, 0.4, 0.5],
            vec![5.0, 2.5, 0.5, 0.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(2),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        let last_selected = result
            .selected_indices
            .last()
            .copied()
            .expect("copied should succeed");
        assert!(
            (result.threshold - result.marginal_scores[last_selected]).abs() < 1e-12,
            "threshold={} vs score[{}]={}",
            result.threshold,
            last_selected,
            result.marginal_scores[last_selected]
        );
    }

    // Test 10: screened_dataset returns correct n_features.
    #[test]
    fn screened_dataset_n_features() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let covs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(2),
            ..Default::default()
        };
        let sis = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        let screened = screened_dataset(&ds, &sis).expect("screened_dataset should succeed");
        assert_eq!(screened.n_features(), 2);
    }

    // Test 11: screened_dataset observations unchanged.
    #[test]
    fn screened_dataset_observations_unchanged() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, false, true];
        let covs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(1),
            ..Default::default()
        };
        let sis = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        let screened = screened_dataset(&ds, &sis).expect("screened_dataset should succeed");
        for (i, obs) in screened.observations.iter().enumerate() {
            assert!((obs.time - ds.observations[i].time).abs() < 1e-15);
            assert_eq!(obs.event, ds.observations[i].event);
        }
    }

    // Test 12: Null covariate (all zeros) → |U_j| = 0 → ranked last.
    #[test]
    fn null_covariate_ranked_last() {
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, true, true, true];
        let covs = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![4.0, 0.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default())
            .expect("value should be present");
        // Feature 1 is all-zero → score 0 → should be last in ranked.
        assert!(
            result.marginal_scores[1] < 1e-10,
            "null covariate score should be ~0, got {}",
            result.marginal_scores[1]
        );
        assert_eq!(
            *result.ranked_indices.last().expect("last should succeed"),
            1
        );
    }

    // Test 13: All-censored dataset → NoEvents.
    #[test]
    fn all_censored_returns_no_events() {
        let times = [1.0, 2.0, 3.0];
        let events = [false, false, false];
        let covs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default());
        assert!(matches!(result, Err(SurvivalError::NoEvents)));
    }

    // Test 14: Empty dataset → EmptyDataset.
    #[test]
    fn empty_dataset_returns_error() {
        // Use from_arrays on empty → error from Dataset, so build a wrapper.
        // Actually Dataset::new rejects empty, so we test the error path
        // by verifying Dataset::new rejects it.
        let result = Dataset::new(vec![], None, None);
        assert!(matches!(result, Err(SurvivalError::EmptyDataset)));
    }

    // Test 15: No covariates → InvalidConfiguration.
    #[test]
    fn no_covariates_returns_invalid_configuration() {
        let ds = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true, false, true])
            .expect("from_arrays should succeed");
        let result = sure_independence_screening(&ds, &SisConfig::default());
        assert!(matches!(
            result,
            Err(SurvivalError::InvalidConfiguration(_))
        ));
    }

    // Test 16: Single event.
    #[test]
    fn single_event_computes_correctly() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [false, false, true, false, false];
        let covs = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let result = sure_independence_screening(&ds, &SisConfig::default())
            .expect("value should be present");
        // With 1 event at t=3, risk set = {3,4,5} (indices 2,3,4).
        // risk_mean_0 = (3+4+5)/3 = 4.0; U_0 = 3.0 - 4.0 = -1.0 → |U_0| = 1.0
        // risk_mean_1 = (30+40+50)/3 = 40.0; U_1 = 30.0 - 40.0 = -10.0 → |U_1| = 10.0
        assert!((result.marginal_scores[0] - 1.0).abs() < 1e-9);
        assert!((result.marginal_scores[1] - 10.0).abs() < 1e-9);
    }

    // Test 17: Determinism.
    #[test]
    fn determinism_same_result() {
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, false, true, true];
        let covs = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig::default();
        let r1 = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        let r2 = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        assert_eq!(r1.marginal_scores, r2.marginal_scores);
        assert_eq!(r1.ranked_indices, r2.ranked_indices);
        assert_eq!(r1.selected_indices, r2.selected_indices);
    }

    // Test 18: Identical covariates → all |U_j| equal → all selected when d=p.
    #[test]
    fn identical_covariates_all_equal_scores() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let val = 5.0_f64;
        let covs = vec![
            vec![val, val, val],
            vec![val, val, val],
            vec![val, val, val],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(3),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        // All scores equal (all zero since constant covariates).
        let s0 = result.marginal_scores[0];
        for &s in &result.marginal_scores {
            assert!((s - s0).abs() < 1e-12);
        }
        assert_eq!(result.selected_indices.len(), 3);
    }

    // Test 19: Single feature dataset → d=1 selects that feature.
    #[test]
    fn single_feature_selects_it() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let covs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(1),
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        assert_eq!(result.selected_indices, vec![0]);
        assert_eq!(result.p, 1);
    }

    // Test 20: screened_dataset covariate index j maps to original feature selected_indices[j].
    #[test]
    fn screened_dataset_correct_column_mapping() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, true, true, true, true];
        // Feature 0: strong trend, Feature 1: anticorrelated, Feature 2: constant.
        let covs = vec![
            vec![0.0, 5.0, 7.0],
            vec![1.0, 4.0, 7.0],
            vec![2.0, 3.0, 7.0],
            vec![3.0, 2.0, 7.0],
            vec![4.0, 1.0, 7.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            d: Some(2),
            ..Default::default()
        };
        let sis = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        let screened = screened_dataset(&ds, &sis).expect("screened_dataset should succeed");
        // Verify each column in screened matches original column selected_indices[j].
        let orig_covs = ds.covariates.as_ref().expect("as_ref should succeed");
        let scr_covs = screened.covariates.as_ref().expect("as_ref should succeed");
        for i in 0..ds.len() {
            for (j, &orig_j) in sis.selected_indices.iter().enumerate() {
                assert!(
                    (scr_covs[i][j] - orig_covs[i][orig_j]).abs() < 1e-15,
                    "screened[{i}][{j}] != orig[{i}][{orig_j}]"
                );
            }
        }
    }

    // Bonus: Efron tie method compiles and produces non-negative scores.
    #[test]
    fn efron_tie_method_works() {
        let times = [1.0, 1.0, 2.0, 3.0];
        let events = [true, true, true, true];
        let covs = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let ds = make_dataset(&times, &events, &covs);
        let config = SisConfig {
            tie_method: SisTieMethod::Efron,
            ..Default::default()
        };
        let result = sure_independence_screening(&ds, &config)
            .expect("sure_independence_screening should succeed");
        for &s in &result.marginal_scores {
            assert!(s >= 0.0);
        }
    }
}
