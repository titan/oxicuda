//! Unified prediction interface for fitted Cox proportional-hazards models.
//!
//! Provides:
//! - [`SurvivalPredict`]: trait that all fitted Cox models can implement.
//! - Blanket implementation for [`CoxFitResult`] from `cox_builder`.
//! - [`predict_survival_curve`]: produce a [`StepFunction`] for a given covariate vector.

use crate::cox::cox_builder::CoxFitResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::plot::step_functions::StepFunction;

// ─── Trait ────────────────────────────────────────────────────────────────────

/// Unified prediction interface for fitted Cox PH models.
pub trait SurvivalPredict {
    /// Predicted log-hazard ratio `xᵀβ` for a covariate vector `x`.
    ///
    /// # Errors
    ///
    /// Returns [`SurvivalError::DimensionMismatch`] when `x.len()` does not
    /// match the number of model coefficients.
    fn predict_log_hazard_ratio(&self, x: &[f64]) -> SurvivalResult<f64>;

    /// Predicted hazard ratio `exp(xᵀβ)` for `x`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SurvivalPredict::predict_log_hazard_ratio`].
    fn predict_hazard_ratio(&self, x: &[f64]) -> SurvivalResult<f64> {
        Ok(self.predict_log_hazard_ratio(x)?.exp())
    }

    /// Predicted survival function `S(t | x)` at time `t`.
    ///
    /// Uses the Breslow baseline cumulative hazard:
    /// `S(t | x) = exp(-H₀(t) · exp(xᵀβ))`.
    ///
    /// `baseline_times`  — monotone-increasing time points of H₀.
    /// `baseline_cumhaz` — baseline cumulative hazard values `H₀(t)`.
    ///
    /// # Errors
    ///
    /// Returns [`SurvivalError::DimensionMismatch`] when baseline arrays have
    /// different lengths.  Propagates errors from [`SurvivalPredict::predict_log_hazard_ratio`].
    fn predict_survival(
        &self,
        x: &[f64],
        t: f64,
        baseline_times: &[f64],
        baseline_cumhaz: &[f64],
    ) -> SurvivalResult<f64>;

    /// Predicted cumulative hazard `H(t | x) = H₀(t) · exp(xᵀβ)`.
    ///
    /// # Errors
    ///
    /// Returns [`SurvivalError::DimensionMismatch`] when baseline arrays have
    /// different lengths.  Propagates errors from [`SurvivalPredict::predict_log_hazard_ratio`].
    fn predict_cumulative_hazard(
        &self,
        x: &[f64],
        t: f64,
        baseline_times: &[f64],
        baseline_cumhaz: &[f64],
    ) -> SurvivalResult<f64>;
}

// ─── Internal baseline lookup ─────────────────────────────────────────────────

/// Evaluate H₀(t) from sorted `baseline_times` and `baseline_cumhaz` by
/// finding the largest baseline time ≤ t (step-function interpolation).
///
/// Returns `0.0` if `t` is before the first baseline time point.
fn baseline_cumhaz_at(t: f64, baseline_times: &[f64], baseline_cumhaz: &[f64]) -> f64 {
    // Linear scan from the right for the first entry ≤ t.
    for (k, &bt) in baseline_times.iter().enumerate().rev() {
        if bt <= t {
            return baseline_cumhaz[k];
        }
    }
    0.0 // t is before all event times → H₀(t) = 0
}

// ─── impl SurvivalPredict for CoxFitResult ────────────────────────────────────

impl SurvivalPredict for CoxFitResult {
    fn predict_log_hazard_ratio(&self, x: &[f64]) -> SurvivalResult<f64> {
        if x.len() != self.coef.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: self.coef.len(),
                b: x.len(),
            });
        }
        let lhr: f64 = x.iter().zip(self.coef.iter()).map(|(xi, bi)| xi * bi).sum();
        Ok(lhr)
    }

    fn predict_survival(
        &self,
        x: &[f64],
        t: f64,
        baseline_times: &[f64],
        baseline_cumhaz: &[f64],
    ) -> SurvivalResult<f64> {
        if baseline_times.len() != baseline_cumhaz.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: baseline_times.len(),
                b: baseline_cumhaz.len(),
            });
        }
        let lhr = self.predict_log_hazard_ratio(x)?;
        let h0t = baseline_cumhaz_at(t, baseline_times, baseline_cumhaz);
        // S(t | x) = exp(-H₀(t) · exp(xᵀβ))
        Ok((-h0t * lhr.exp()).exp())
    }

    fn predict_cumulative_hazard(
        &self,
        x: &[f64],
        t: f64,
        baseline_times: &[f64],
        baseline_cumhaz: &[f64],
    ) -> SurvivalResult<f64> {
        if baseline_times.len() != baseline_cumhaz.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: baseline_times.len(),
                b: baseline_cumhaz.len(),
            });
        }
        let lhr = self.predict_log_hazard_ratio(x)?;
        let h0t = baseline_cumhaz_at(t, baseline_times, baseline_cumhaz);
        Ok(h0t * lhr.exp())
    }
}

// ─── Blanket utility ──────────────────────────────────────────────────────────

/// Compute the survival function at each time in `time_grid` for covariate
/// vector `x` using a [`SurvivalPredict`] model.
///
/// Returns a [`StepFunction`] whose `times` are the non-empty subset of
/// `time_grid` and whose `values` are `S(t | x)` for each grid point.
///
/// # Errors
///
/// - [`SurvivalError::InvalidParameter`] when `time_grid` is empty.
/// - [`SurvivalError::DimensionMismatch`] when baseline arrays have different lengths.
/// - Propagates errors from [`SurvivalPredict::predict_survival`].
pub fn predict_survival_curve(
    model: &dyn SurvivalPredict,
    x: &[f64],
    time_grid: &[f64],
    baseline_times: &[f64],
    baseline_cumhaz: &[f64],
) -> SurvivalResult<StepFunction> {
    if time_grid.is_empty() {
        return Err(SurvivalError::InvalidParameter(
            "time_grid must be non-empty".to_string(),
        ));
    }
    if baseline_times.len() != baseline_cumhaz.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: baseline_times.len(),
            b: baseline_cumhaz.len(),
        });
    }

    let mut times = Vec::with_capacity(time_grid.len());
    let mut values = Vec::with_capacity(time_grid.len());

    for &t in time_grid {
        let s = model.predict_survival(x, t, baseline_times, baseline_cumhaz)?;
        times.push(t);
        values.push(s);
    }

    Ok(StepFunction {
        times,
        values,
        stderr: None,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal CoxFitResult with known coefficients.
    fn make_cox(coef: Vec<f64>) -> CoxFitResult {
        CoxFitResult {
            coef,
            log_lik: -10.0,
            n_iter: 5,
            converged: true,
            n_events: 10,
            n_subjects: 50,
        }
    }

    /// Baseline cumulative hazard: H₀(t) = 0.1*t (linear).
    fn linear_baseline(n: usize) -> (Vec<f64>, Vec<f64>) {
        let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let cumhaz: Vec<f64> = times.iter().map(|&t| 0.1 * t).collect();
        (times, cumhaz)
    }

    // ── Test 1: predict_log_hazard_ratio is x·beta ───────────────────────────
    #[test]
    fn log_hazard_ratio_dot_product() {
        let cox = make_cox(vec![0.5, -1.0, 2.0]);
        let x = vec![1.0, 2.0, 3.0];
        // 0.5*1 + (-1.0)*2 + 2.0*3 = 0.5 - 2 + 6 = 4.5
        let lhr = cox.predict_log_hazard_ratio(&x).unwrap();
        assert!((lhr - 4.5).abs() < 1e-12, "expected 4.5, got {lhr}");
    }

    // ── Test 2: hazard_ratio = exp(log_hazard_ratio) ─────────────────────────
    #[test]
    fn hazard_ratio_is_exp_lhr() {
        let cox = make_cox(vec![1.0, 0.5]);
        let x = vec![1.0, 2.0];
        let lhr = cox.predict_log_hazard_ratio(&x).unwrap();
        let hr = cox.predict_hazard_ratio(&x).unwrap();
        assert!((hr - lhr.exp()).abs() < 1e-12, "HR must be exp(LHR)");
    }

    // ── Test 3: predict_survival at t=0 = 1.0 ────────────────────────────────
    #[test]
    fn predict_survival_at_zero() {
        let cox = make_cox(vec![0.2]);
        let x = vec![1.0];
        let (bt, bh) = linear_baseline(5);
        // At t=0: H₀(0)=0 → S = exp(0) = 1.0
        let s = cox.predict_survival(&x, 0.0, &bt, &bh).unwrap();
        assert!((s - 1.0).abs() < 1e-12, "S(0|x) must be 1.0");
    }

    // ── Test 4: predict_survival is in (0, 1] ────────────────────────────────
    #[test]
    fn predict_survival_range() {
        let cox = make_cox(vec![0.3, -0.1]);
        let x = vec![1.5, -0.5];
        let (bt, bh) = linear_baseline(10);
        for &t in &[1.0, 3.0, 5.0, 10.0] {
            let s = cox.predict_survival(&x, t, &bt, &bh).unwrap();
            assert!(s > 0.0 && s <= 1.0, "S({t}|x)={s} not in (0,1]");
        }
    }

    // ── Test 5: predict_cumulative_hazard = -log(predict_survival) ───────────
    #[test]
    fn cumulative_hazard_is_neg_log_survival() {
        let cox = make_cox(vec![0.4]);
        let x = vec![1.0];
        let (bt, bh) = linear_baseline(5);
        for &t in &[1.0, 2.0, 3.0, 5.0] {
            let s = cox.predict_survival(&x, t, &bt, &bh).unwrap();
            let h = cox.predict_cumulative_hazard(&x, t, &bt, &bh).unwrap();
            let neg_log_s = -s.ln();
            assert!(
                (h - neg_log_s).abs() < 1e-10,
                "H({t}|x)={h} != -log S({t}|x)={neg_log_s}"
            );
        }
    }

    // ── Test 6: predict_log_hazard_ratio dim mismatch → error ────────────────
    #[test]
    fn log_hazard_ratio_dim_mismatch_error() {
        let cox = make_cox(vec![0.5, 1.0]);
        let x = vec![1.0]; // only 1 feature, but model has 2
        assert!(cox.predict_log_hazard_ratio(&x).is_err());
    }

    // ── Test 7: predict_survival baseline dim mismatch → error ───────────────
    #[test]
    fn predict_survival_baseline_mismatch_error() {
        let cox = make_cox(vec![0.5]);
        let x = vec![1.0];
        // Mismatched baseline arrays
        let result = cox.predict_survival(&x, 2.0, &[1.0, 2.0], &[0.1]);
        assert!(result.is_err());
    }

    // ── Test 8: predict_survival_curve returns correct shape ─────────────────
    #[test]
    fn predict_survival_curve_shape() {
        let cox = make_cox(vec![0.2]);
        let x = vec![1.0];
        let (bt, bh) = linear_baseline(10);
        let time_grid: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let sf = predict_survival_curve(&cox, &x, &time_grid, &bt, &bh).unwrap();
        assert_eq!(sf.times.len(), 10);
        assert_eq!(sf.values.len(), 10);
    }

    // ── Test 9: predict_survival_curve empty grid → error ────────────────────
    #[test]
    fn predict_survival_curve_empty_grid_error() {
        let cox = make_cox(vec![0.2]);
        let x = vec![1.0];
        let (bt, bh) = linear_baseline(5);
        assert!(predict_survival_curve(&cox, &x, &[], &bt, &bh).is_err());
    }

    // ── Test 10: predict_survival_curve values are non-increasing ────────────
    #[test]
    fn predict_survival_curve_nonincreasing() {
        let cox = make_cox(vec![0.5]);
        let x = vec![1.0];
        // Baseline: monotone increasing cumhaz → S should be non-increasing.
        let baseline_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let baseline_cumhaz = vec![0.05, 0.12, 0.22, 0.35, 0.50];
        let time_grid = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sf = predict_survival_curve(&cox, &x, &time_grid, &baseline_times, &baseline_cumhaz)
            .unwrap();
        for i in 1..sf.values.len() {
            assert!(
                sf.values[i] <= sf.values[i - 1] + 1e-10,
                "S(t) should be non-increasing: S[{}]={} > S[{}]={}",
                i,
                sf.values[i],
                i - 1,
                sf.values[i - 1]
            );
        }
    }

    // ── Test 11: beta=0 → hazard ratio = 1 ───────────────────────────────────
    #[test]
    fn zero_beta_hazard_ratio_one() {
        let cox = make_cox(vec![0.0, 0.0]);
        let x = vec![3.0, -2.0];
        let hr = cox.predict_hazard_ratio(&x).unwrap();
        assert!((hr - 1.0).abs() < 1e-12, "zero beta → HR=1, got {hr}");
    }

    // ── Test 12: predict_survival monotone with t (given increasing baseline) ─
    #[test]
    fn predict_survival_monotone_in_t() {
        let cox = make_cox(vec![0.3]);
        let x = vec![1.0];
        let (bt, bh) = linear_baseline(20);
        let mut prev = 1.0_f64;
        for i in 1..=20 {
            let s = cox.predict_survival(&x, i as f64, &bt, &bh).unwrap();
            assert!(s <= prev + 1e-12, "S({i}) > S({}) — not monotone", i - 1);
            prev = s;
        }
    }

    // ── Test 13: predict_log_hazard_ratio with negative beta ─────────────────
    #[test]
    fn negative_beta_reduces_hazard() {
        let cox = make_cox(vec![-1.0]);
        let x = vec![2.0];
        let lhr = cox.predict_log_hazard_ratio(&x).unwrap();
        assert!(lhr < 0.0, "negative beta × positive x → negative lhr");
        let hr = cox.predict_hazard_ratio(&x).unwrap();
        assert!(hr < 1.0, "negative lhr → HR < 1");
    }

    // ── Test 14: SurvivalPredict blanket hazard_ratio is default impl ─────────
    #[test]
    fn hazard_ratio_default_impl_is_exp_lhr() {
        let cox = make_cox(vec![0.0]);
        let x = vec![0.0];
        let lhr = cox.predict_log_hazard_ratio(&x).unwrap();
        let hr = cox.predict_hazard_ratio(&x).unwrap();
        assert!((hr - lhr.exp()).abs() < 1e-12);
        assert!((hr - 1.0).abs() < 1e-12);
    }
} // end mod tests
