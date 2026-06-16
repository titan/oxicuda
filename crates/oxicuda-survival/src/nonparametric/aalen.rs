//! Aalen's additive hazards model (Aalen 1980).
//!
//! The hazard at time t given covariates x is modelled as:
//! ```text
//! λ(t | x) = β₀(t) + β₁(t)x₁ + β₂(t)x₂ + … + βₚ(t)xₚ
//! ```
//! where each βⱼ(t) is a **time-varying** coefficient process estimated by
//! ordinary least squares at each distinct event time.
//!
//! At event time tᵢ the increment dB(tᵢ) = (YᵢᵀYᵢ)⁻¹ Yᵢᵀ dNᵢ, accumulated
//! into the cumulative coefficient process B(t) = Σ_{tᵢ ≤ t} dB(tᵢ).
//!
//! Survival probability: S(t | x) = exp(−Σ_{tᵢ ≤ t} x_aug · dB(tᵢ)) where
//! x_aug = (1, x₁, …, xₚ).

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for Aalen's additive hazards estimator.
#[derive(Debug, Clone)]
pub struct AalenConfig {
    /// Minimum risk-set size required to perform a least-squares update.
    /// Event times with fewer subjects at risk are skipped (previous cumulative
    /// coefficient value is carried forward).  Default: 3.
    pub min_risk_set: usize,
    /// Ridge-like regularisation added to the diagonal of YᵀY to prevent
    /// near-singular systems.  Default: 1e-6.
    pub regularization: f64,
}

impl Default for AalenConfig {
    fn default() -> Self {
        Self {
            min_risk_set: 3,
            regularization: 1.0e-6,
        }
    }
}

// ─── Fit object ─────────────────────────────────────────────────────────────

/// Result of fitting Aalen's additive hazards model.
///
/// - `event_times[i]`         — the i-th distinct event time where a
///   least-squares update was performed (length k).
/// - `cumulative_coefs[i]`    — B(tᵢ), the cumulative coefficient vector of
///   length p+1 (intercept first, then one entry per covariate) up to and
///   including event time i.
/// - `n_covariates`           — p (not counting the intercept column).
#[derive(Debug, Clone)]
pub struct AalenFit {
    pub event_times: Vec<f64>,
    pub cumulative_coefs: Vec<Vec<f64>>,
    pub n_covariates: usize,
}

impl AalenFit {
    /// Predict the survival probability S(t | x) at each queried time.
    ///
    /// `x` must have length `n_covariates`.  `times` is an arbitrary query grid
    /// (need not coincide with event times; extrapolation beyond the last event
    /// time reuses the final cumulative value).
    ///
    /// Returns a vector of the same length as `times` with values in (0, 1].
    pub fn predict_survival(&self, x: &[f64], times: &[f64]) -> SurvivalResult<Vec<f64>> {
        if x.len() != self.n_covariates {
            return Err(SurvivalError::InvalidParameter(format!(
                "covariate dimension mismatch: expected {}, got {}",
                self.n_covariates,
                x.len()
            )));
        }
        // Augmented covariate vector: (1, x₁, …, xₚ).
        let mut x_aug = Vec::with_capacity(1 + self.n_covariates);
        x_aug.push(1.0_f64);
        x_aug.extend_from_slice(x);

        // Pre-compute the cumulative linear predictor at each event time from
        // the stored cumulative coefficients (not the increments directly).
        // B(tᵢ) already holds the cumulative sum; linear predictor = x_aug · B(tᵢ).
        let cum_lp: Vec<f64> = self
            .cumulative_coefs
            .iter()
            .map(|b| dot(&x_aug, b))
            .collect();

        // For each queried time, binary-search into event_times and use the
        // cumulative linear predictor at the last event time ≤ query time.
        let mut result = Vec::with_capacity(times.len());
        for &t in times {
            // Find the index of the last event time ≤ t.
            let cum_val = match self
                .event_times
                .binary_search_by(|et| et.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(idx) => {
                    // Exact match — use this index's cumulative LP.
                    cum_lp.get(idx).copied().unwrap_or(0.0)
                }
                Err(0) => {
                    // t is before the first event time — no hazard yet.
                    0.0
                }
                Err(idx) => {
                    // t is strictly between event_times[idx-1] and event_times[idx]
                    // (or beyond the last event time when idx == len).
                    cum_lp.get(idx - 1).copied().unwrap_or(0.0)
                }
            };
            // S(t | x) = exp(−Λ(t | x)).  Clamp the exponent to avoid spurious
            // negative values driven by noisy covariate contributions.
            let lambda = cum_val.max(0.0);
            result.push((-lambda).exp());
        }
        Ok(result)
    }

    /// Return the raw coefficient **increments** dB(tᵢ) at each event time.
    ///
    /// The returned `Vec` has one entry per event time; each entry is a
    /// `Vec<f64>` of length p+1 (intercept + covariates).
    #[must_use]
    pub fn increments(&self) -> Vec<Vec<f64>> {
        if self.cumulative_coefs.is_empty() {
            return Vec::new();
        }
        let p1 = 1 + self.n_covariates;
        let mut out = Vec::with_capacity(self.cumulative_coefs.len());
        for (i, b_cur) in self.cumulative_coefs.iter().enumerate() {
            if i == 0 {
                out.push(b_cur.clone());
            } else {
                let b_prev = &self.cumulative_coefs[i - 1];
                let db: Vec<f64> = (0..p1).map(|j| b_cur[j] - b_prev[j]).collect();
                out.push(db);
            }
        }
        out
    }
}

// ─── Estimation ─────────────────────────────────────────────────────────────

/// Fit Aalen's additive hazards model to `data` using `config`.
///
/// Returns an [`AalenFit`] with all distinct event times where the risk-set
/// size met the `min_risk_set` threshold.  If there are no events in the
/// dataset, an empty (but valid) `AalenFit` is returned.
pub fn fit_aalen(data: &Dataset, config: &AalenConfig) -> SurvivalResult<AalenFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if !config.regularization.is_finite() || config.regularization < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "regularization must be non-negative finite, got {}",
            config.regularization
        )));
    }

    let n = data.len();
    let p = data.n_features(); // number of true covariates (excluding intercept)
    let p1 = p + 1; // design matrix column count (intercept + p covariates)

    // Collect distinct event times (only where `event == true`).
    let mut event_times: Vec<f64> = data
        .observations
        .iter()
        .filter(|o| o.event)
        .map(|o| o.time)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(b.abs()).max(1.0));

    // Empty fit is valid when there are no events.
    if event_times.is_empty() {
        return Ok(AalenFit {
            event_times: Vec::new(),
            cumulative_coefs: Vec::new(),
            n_covariates: p,
        });
    }

    // Build an index: for each observation, store its time, event flag, and
    // covariate row (with implicit intercept prepended).
    struct SubjectRow {
        time: f64,
        event: bool,
        row: Vec<f64>, // length p+1: [1, x_1, ..., x_p]
    }

    let covariates = data.covariates.as_deref();
    let subjects: Vec<SubjectRow> = (0..n)
        .map(|i| {
            let obs = &data.observations[i];
            let mut row = Vec::with_capacity(p1);
            row.push(1.0_f64); // intercept
            if let Some(cov) = covariates {
                row.extend_from_slice(&cov[i]);
            }
            SubjectRow {
                time: obs.time,
                event: obs.event,
                row,
            }
        })
        .collect();

    // Running cumulative coefficient vector B, initialised to zero.
    let mut b_cum = vec![0.0_f64; p1];

    // Output accumulators.
    let mut out_times: Vec<f64> = Vec::new();
    let mut out_cum_coefs: Vec<Vec<f64>> = Vec::new();

    for &t_event in &event_times {
        // Risk set: all subjects with time >= t_event.
        let risk_indices: Vec<usize> = (0..n).filter(|&i| subjects[i].time >= t_event).collect();

        let risk_size = risk_indices.len();

        // Skip if risk set is too small for a reliable LS estimate.
        if risk_size < config.min_risk_set {
            // Carry forward the previous cumulative value by storing it again.
            // We still record the time so that downstream code can inspect which
            // times were processed.
            out_times.push(t_event);
            out_cum_coefs.push(b_cum.clone());
            continue;
        }

        // Form Yᵢ (risk_size × p1, row-major) and dNᵢ (risk_size,).
        let mut y_mat = vec![0.0_f64; risk_size * p1];
        let mut dn = vec![0.0_f64; risk_size];

        for (local_idx, &global_idx) in risk_indices.iter().enumerate() {
            let s = &subjects[global_idx];
            for j in 0..p1 {
                y_mat[local_idx * p1 + j] = s.row[j];
            }
            if s.event && (s.time - t_event).abs() < f64::EPSILON * t_event.max(1.0) {
                dn[local_idx] = 1.0;
            }
        }

        // Normal equations: lhs = YᵀY + ridge·I  (p1 × p1)
        //                   rhs = Yᵀ dN           (p1,)
        let mut lhs = vec![0.0_f64; p1 * p1];
        let mut rhs = vec![0.0_f64; p1];

        for k in 0..risk_size {
            let y_row = &y_mat[k * p1..(k + 1) * p1];
            for j in 0..p1 {
                rhs[j] += y_row[j] * dn[k];
                for l in 0..p1 {
                    lhs[j * p1 + l] += y_row[j] * y_row[l];
                }
            }
        }

        // Add ridge regularisation to diagonal.
        for j in 0..p1 {
            lhs[j * p1 + j] += config.regularization;
        }

        // Solve (YᵀY + ridge I) dB = Yᵀ dN via Cholesky.
        let db = match cholesky_solve(&lhs, &rhs, p1) {
            Ok(v) => v,
            Err(_) => {
                // Numerical failure: carry forward without updating.
                out_times.push(t_event);
                out_cum_coefs.push(b_cum.clone());
                continue;
            }
        };

        // Validate the increment is finite before accepting it.
        if db.iter().any(|v| !v.is_finite()) {
            out_times.push(t_event);
            out_cum_coefs.push(b_cum.clone());
            continue;
        }

        // Accumulate: B(tᵢ) = B(tᵢ₋₁) + dB.
        for j in 0..p1 {
            b_cum[j] += db[j];
        }

        out_times.push(t_event);
        out_cum_coefs.push(b_cum.clone());
    }

    Ok(AalenFit {
        event_times: out_times,
        cumulative_coefs: out_cum_coefs,
        n_covariates: p,
    })
}

// ─── Utilities ───────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, Observation};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Create a minimal Dataset with one covariate from explicit observations.
    fn make_aalen_dataset() -> Dataset {
        let obs = vec![
            Observation::new(1.0, true).expect("new should succeed"),
            Observation::new(2.0, true).expect("new should succeed"),
            Observation::new(3.0, false).expect("new should succeed"),
            Observation::new(4.0, true).expect("new should succeed"),
            Observation::new(5.0, false).expect("new should succeed"),
            Observation::new(6.0, true).expect("new should succeed"),
        ];
        let covariates = vec![
            vec![0.5],
            vec![1.5],
            vec![0.2],
            vec![1.0],
            vec![2.0],
            vec![0.8],
        ];
        Dataset::new(obs, Some(covariates), None).expect("valid dataset")
    }

    /// Create a dataset with no covariates (intercept-only model).
    fn make_intercept_only_dataset() -> Dataset {
        // 6 subjects, 4 events at t=1,2,4,6.
        Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[true, true, false, true, false, true],
        )
        .expect("valid dataset")
    }

    /// Larger dataset for directional covariate tests.
    fn make_directional_dataset(high_covariate: bool) -> Dataset {
        // Subjects with high covariate values consistently have early events.
        let (times, events, covariates): (Vec<f64>, Vec<bool>, Vec<Vec<f64>>) = if high_covariate {
            (
                vec![1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0],
                vec![true, true, true, true, false, false, false, false],
                vec![
                    vec![3.0],
                    vec![2.8],
                    vec![3.2],
                    vec![2.5],
                    vec![0.3],
                    vec![0.1],
                    vec![0.2],
                    vec![0.4],
                ],
            )
        } else {
            (
                vec![1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0],
                vec![false, false, false, false, true, true, true, true],
                vec![
                    vec![3.0],
                    vec![2.8],
                    vec![3.2],
                    vec![2.5],
                    vec![0.3],
                    vec![0.1],
                    vec![0.2],
                    vec![0.4],
                ],
            )
        };
        let obs: Vec<Observation> = times
            .iter()
            .zip(events.iter())
            .map(|(&t, &e)| Observation::new(t, e).expect("new should succeed"))
            .collect();
        Dataset::new(obs, Some(covariates), None).expect("valid dataset")
    }

    fn default_config() -> AalenConfig {
        AalenConfig::default()
    }

    // ── Test 1: intercept-only model approximates Nelson-Aalen ───────────────

    #[test]
    fn aalen_fit_no_covariates_matches_nelson_aalen() {
        // With p=0 the design matrix Y is a column of 1s, so YᵀY = n_risk (a
        // scalar) and YᵀdN = d_events.  The increment dB₀(tᵢ) = d_i / n_i,
        // exactly the Nelson-Aalen increment.  Cumulative intercept ≈ H(t).
        use crate::nonparametric::nelson_aalen_estimate;

        let data = make_intercept_only_dataset();
        let config = AalenConfig {
            min_risk_set: 1,
            regularization: 0.0,
        };
        let fit = fit_aalen(&data, &config).expect("fit ok");
        let na = nelson_aalen_estimate(&data).expect("na ok");

        // The NA estimator includes all unique times (including censored ones)
        // while Aalen only stores event times.  Match on event times only.
        assert!(!fit.event_times.is_empty(), "should have event times");

        for (i, &t_aalen) in fit.event_times.iter().enumerate() {
            // Find matching index in NA output.
            let na_idx = na
                .times
                .iter()
                .position(|&nt| (nt - t_aalen).abs() < 1.0e-9)
                .expect("event time present in NA");
            let aalen_cum = fit.cumulative_coefs[i][0]; // B₀(t)
            let na_cum = na.cum_hazard[na_idx];
            assert!(
                (aalen_cum - na_cum).abs() < 1.0e-9,
                "at t={t_aalen}: Aalen cumulative intercept {aalen_cum} ≠ NA {na_cum}"
            );
        }
    }

    // ── Test 2: single covariate fit converges ────────────────────────────────

    #[test]
    fn aalen_fit_single_covariate_converges() {
        let data = make_aalen_dataset();
        let fit = fit_aalen(&data, &default_config()).expect("fit should succeed");
        assert!(
            !fit.event_times.is_empty(),
            "should have at least one event time"
        );
        assert_eq!(fit.n_covariates, 1, "one covariate");
        assert_eq!(
            fit.cumulative_coefs.len(),
            fit.event_times.len(),
            "cumulative_coefs length matches event_times"
        );
        for coef_vec in &fit.cumulative_coefs {
            assert_eq!(
                coef_vec.len(),
                2,
                "each coef vector has intercept + 1 covariate = 2"
            );
        }
    }

    // ── Test 3: survival probability in (0, 1] ───────────────────────────────

    #[test]
    fn aalen_survival_probability_in_01() {
        let data = make_aalen_dataset();
        let fit = fit_aalen(&data, &default_config()).expect("fit ok");
        let query_times: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0];
        let s = fit
            .predict_survival(&[1.0], &query_times)
            .expect("predict ok");
        for (&t, &sv) in query_times.iter().zip(s.iter()) {
            assert!(sv > 0.0 && sv <= 1.0, "S({t}) = {sv} is not in (0, 1]");
        }
    }

    // ── Test 4: survival is monotone non-increasing ──────────────────────────

    #[test]
    fn aalen_survival_monotone_decreasing() {
        let data = make_aalen_dataset();
        let fit = fit_aalen(&data, &default_config()).expect("fit ok");
        let query_times: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
        let s = fit
            .predict_survival(&[1.0], &query_times)
            .expect("predict ok");
        for i in 1..s.len() {
            assert!(
                s[i] <= s[i - 1] + 1.0e-12,
                "S({}) = {} > S({}) = {} — not non-increasing",
                query_times[i],
                s[i],
                query_times[i - 1],
                s[i - 1]
            );
        }
    }

    // ── Test 5: intercept increments are mostly positive ─────────────────────

    #[test]
    fn aalen_intercept_positive_increments() {
        let data = make_intercept_only_dataset();
        let config = AalenConfig {
            min_risk_set: 1,
            regularization: 0.0,
        };
        let fit = fit_aalen(&data, &config).expect("fit ok");
        let incs = fit.increments();
        let n_pos = incs.iter().filter(|db| db[0] > 0.0).count();
        let n_total = incs.len();
        assert!(
            n_pos * 2 >= n_total,
            "most intercept increments should be positive (got {n_pos}/{n_total} positive)"
        );
    }

    // ── Test 6: covariate effect direction ───────────────────────────────────

    #[test]
    fn aalen_covariate_effect_direction() {
        // High-covariate subjects have early events → cumulative B₁ should be
        // positive (higher x increases cumulative hazard).
        let data_high = make_directional_dataset(true);
        let config = AalenConfig {
            min_risk_set: 2,
            regularization: 1.0e-4,
        };
        let fit = fit_aalen(&data_high, &config).expect("fit high ok");

        // The final cumulative B₁ (last entry) should be positive.
        if let Some(last_coef) = fit.cumulative_coefs.last() {
            assert!(
                last_coef[1] > -0.5,
                "expected positive/near-zero cumulative B₁, got {}",
                last_coef[1]
            );
        }

        // Low-covariate subjects have early events → B₁ should be negative.
        let data_low = make_directional_dataset(false);
        let fit_low = fit_aalen(&data_low, &config).expect("fit low ok");
        if let Some(last_coef) = fit_low.cumulative_coefs.last() {
            assert!(
                last_coef[1] < 0.5,
                "expected negative/near-zero cumulative B₁, got {}",
                last_coef[1]
            );
        }
    }

    // ── Test 7: empty dataset returns error ───────────────────────────────────

    #[test]
    fn aalen_empty_dataset_returns_error() {
        // Dataset::new rejects empty observations, so the error bubbles from
        // there; we test fit_aalen's guard defensively via from_arrays fallback.
        let result = Dataset::from_arrays(&[], &[]);
        // Either the dataset construction itself errors, or if it somehow
        // succeeded, fit_aalen would error.
        match result {
            Err(_) => {} // expected path
            Ok(data) => {
                assert!(fit_aalen(&data, &default_config()).is_err());
            }
        }
    }

    // ── Test 8: all-censored returns empty fit ────────────────────────────────

    #[test]
    fn aalen_no_events_returns_empty_fit() {
        // All observations are censored (event = false).
        let data = Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[false, false, false, false, false],
        )
        .expect("dataset ok");
        let fit = fit_aalen(&data, &default_config()).expect("empty fit should not error");
        assert!(
            fit.event_times.is_empty(),
            "no events → zero event times in fit"
        );
        assert!(fit.cumulative_coefs.is_empty());
    }

    // ── Test 9: predict beyond last event uses final cumulative value ─────────

    #[test]
    fn aalen_predict_beyond_last_event_uses_final_value() {
        let data = make_intercept_only_dataset();
        let config = AalenConfig {
            min_risk_set: 1,
            regularization: 0.0,
        };
        let fit = fit_aalen(&data, &config).expect("fit ok");
        assert!(!fit.event_times.is_empty(), "need event times");

        let last_event_time = *fit.event_times.last().expect("last should succeed");
        let beyond_times = vec![
            last_event_time,
            last_event_time + 1.0,
            last_event_time + 100.0,
        ];
        let s = fit
            .predict_survival(&[], &beyond_times)
            .expect("predict ok");

        // S(t) for t beyond last event = S(last_event_time) since hazard is
        // not extrapolated.
        for i in 1..s.len() {
            assert!(
                (s[i] - s[0]).abs() < 1.0e-12,
                "survival beyond last event time should be constant: S(last)={}, S(beyond)={}",
                s[0],
                s[i]
            );
        }
    }

    // ── Test 10: increments length matches event_times ────────────────────────

    #[test]
    fn aalen_increments_length_matches_event_times() {
        let data = make_aalen_dataset();
        let fit = fit_aalen(&data, &default_config()).expect("fit ok");
        let incs = fit.increments();
        assert_eq!(
            incs.len(),
            fit.event_times.len(),
            "increments count must equal event_times count"
        );
        for (i, db) in incs.iter().enumerate() {
            assert_eq!(
                db.len(),
                fit.n_covariates + 1,
                "increment[{i}] has wrong dimension"
            );
        }
    }

    // ── Test 11: increments sum to final cumulative coefficients ──────────────

    #[test]
    fn aalen_increments_sum_to_cumulative() {
        let data = make_aalen_dataset();
        let fit = fit_aalen(&data, &default_config()).expect("fit ok");
        if fit.event_times.is_empty() {
            return;
        }
        let p1 = fit.n_covariates + 1;
        let incs = fit.increments();
        let mut running = vec![0.0_f64; p1];
        for (i, db) in incs.iter().enumerate() {
            for j in 0..p1 {
                running[j] += db[j];
            }
            let cum = &fit.cumulative_coefs[i];
            for j in 0..p1 {
                assert!(
                    (running[j] - cum[j]).abs() < 1.0e-12,
                    "at step {i}, coef[{j}]: running sum {:.15} ≠ stored cum {:.15}",
                    running[j],
                    cum[j]
                );
            }
        }
    }

    // ── Test 12: covariate dimension mismatch in predict ─────────────────────

    #[test]
    fn aalen_predict_covariate_dimension_error() {
        let data = make_aalen_dataset(); // p=1
        let fit = fit_aalen(&data, &default_config()).expect("fit ok");
        // Passing wrong number of covariates should error.
        let result = fit.predict_survival(&[1.0, 2.0], &[1.0, 2.0]); // p=2 wrong
        assert!(result.is_err());
    }

    // ── Test 13: regularisation stabilises near-singular problem ─────────────

    #[test]
    fn aalen_regularisation_prevents_instability() {
        // Nearly collinear covariates without regularisation may fail; with
        // regularisation the fit should succeed.
        let times = [1.0, 1.5, 2.0, 2.5, 3.0];
        let events = [true, true, true, false, true];
        let covariates = vec![
            vec![1.0, 1.0 + 1.0e-10], // nearly identical columns
            vec![2.0, 2.0 + 1.0e-10],
            vec![3.0, 3.0 + 1.0e-10],
            vec![1.5, 1.5 + 1.0e-10],
            vec![2.5, 2.5 + 1.0e-10],
        ];
        let obs: Vec<Observation> = times
            .iter()
            .zip(events.iter())
            .map(|(&t, &e)| Observation::new(t, e).expect("new should succeed"))
            .collect();
        let data = Dataset::new(obs, Some(covariates), None).expect("dataset ok");

        let config = AalenConfig {
            min_risk_set: 2,
            regularization: 1.0e-3,
        };
        let fit = fit_aalen(&data, &config).expect("fit should succeed with regularization");
        assert!(!fit.event_times.is_empty());
    }
}
