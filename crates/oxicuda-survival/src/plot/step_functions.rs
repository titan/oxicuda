//! Piecewise-constant step-function helpers for survival plotting.
//!
//! Provides:
//! - [`StepFunction`]: piecewise-constant step function with optional standard errors.
//! - Conversion helpers from KM, Nelson-Aalen and CIF output arrays.
//! - RMST and median survival computation from a [`StepFunction`].
//! - `step_plot_arrays`: produces sentinel-padded arrays for matplotlib-compatible rendering.

use crate::error::{SurvivalError, SurvivalResult};

// ─── Inverse normal (Acklam's rational approximation) ────────────────────────

/// Approximate inverse standard-normal CDF via Acklam's algorithm.
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
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
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (C[0] + q * (C[1] + q * (C[2] + q * (C[3] + q * (C[4] + q * C[5])))))
            / (1.0 + q * (D[0] + q * (D[1] + q * (D[2] + q * D[3]))))
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (q * (A[0] + r * (A[1] + r * (A[2] + r * (A[3] + r * (A[4] + r * A[5]))))))
            / (B[0] + r * (B[1] + r * (B[2] + r * (B[3] + r * (B[4] + r)))))
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(C[0] + q * (C[1] + q * (C[2] + q * (C[3] + q * (C[4] + q * C[5])))))
            / (1.0 + q * (D[0] + q * (D[1] + q * (D[2] + q * D[3]))))
    }
}

// ─── StepFunction ─────────────────────────────────────────────────────────────

/// A piecewise-constant step function (suitable for KM, Nelson-Aalen, CIF).
///
/// Semantics:
/// - `values[i]` is the function value on the half-open interval
///   `[times[i], times[i+1])`.
/// - Before `times[0]`, the function is implicitly at `values[0]` (for survival,
///   typically 1.0).
/// - After the last time point, the function remains at `values.last()`.
///
/// `times` must be strictly increasing and non-empty.
#[derive(Clone, Debug)]
pub struct StepFunction {
    /// Strictly-increasing time points.
    pub times: Vec<f64>,
    /// Function value BEFORE each time-point change; `values[i]` applies on
    /// `[times[i], times[i+1])`.
    pub values: Vec<f64>,
    /// Optional standard errors at each time point.
    pub stderr: Option<Vec<f64>>,
}

impl StepFunction {
    /// Evaluate the step function at time `t`.
    ///
    /// - Returns `values[0]` for `t < times[0]`.
    /// - Returns `values.last()` for `t >= times.last()`.
    pub fn eval(&self, t: f64) -> f64 {
        if t < self.times[0] {
            return self.values[0];
        }
        // Find the last index where times[k] <= t.
        let mut idx = 0usize;
        for (k, &tk) in self.times.iter().enumerate() {
            if tk <= t {
                idx = k;
            } else {
                break;
            }
        }
        self.values[idx]
    }

    /// Evaluate the step function at each time in `query_times` (need not be sorted).
    pub fn eval_batch(&self, query_times: &[f64]) -> Vec<f64> {
        query_times.iter().map(|&t| self.eval(t)).collect()
    }

    /// Compute pointwise 95% normal-approximation confidence bands.
    ///
    /// Returns `(lower, upper)` [`StepFunction`] pairs, or `None` when no
    /// standard errors are available.
    ///
    /// `alpha` is the significance level (e.g. 0.05 for a 95% CI).
    pub fn confidence_band(&self, alpha: f64) -> Option<(StepFunction, StepFunction)> {
        let stderr = self.stderr.as_ref()?;
        let z = norm_inv(1.0 - alpha / 2.0);
        let mut lower_vals = Vec::with_capacity(self.values.len());
        let mut upper_vals = Vec::with_capacity(self.values.len());
        for (&v, &se) in self.values.iter().zip(stderr.iter()) {
            lower_vals.push((v - z * se).clamp(0.0, 1.0));
            upper_vals.push((v + z * se).clamp(0.0, 1.0));
        }
        let lower = StepFunction {
            times: self.times.clone(),
            values: lower_vals,
            stderr: None,
        };
        let upper = StepFunction {
            times: self.times.clone(),
            values: upper_vals,
            stderr: None,
        };
        Some((lower, upper))
    }

    /// Resample the step function to a regular grid of `n_points` between
    /// `times[0]` and `times.last()`.
    ///
    /// Returns `(grid_times, grid_values)`.
    ///
    /// # Errors
    ///
    /// Returns [`SurvivalError::InvalidParameter`] when `n_points < 2`.
    pub fn to_regular_grid(&self, n_points: usize) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        if n_points < 2 {
            return Err(SurvivalError::InvalidParameter(
                "n_points must be >= 2".to_string(),
            ));
        }
        let t_min = *self.times.first().ok_or_else(|| {
            SurvivalError::InvalidParameter("step function has no time points".to_string())
        })?;
        let t_max = *self.times.last().ok_or_else(|| {
            SurvivalError::InvalidParameter("step function has no time points".to_string())
        })?;
        let step = (t_max - t_min) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| t_min + i as f64 * step).collect();
        let vals: Vec<f64> = grid.iter().map(|&t| self.eval(t)).collect();
        Ok((grid, vals))
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

/// Convert Kaplan-Meier output arrays into a [`StepFunction`].
///
/// `km_times`    — event times (strictly increasing).
/// `km_survival` — Ŝ(t) values at those times.
/// `km_stderr`   — optional standard errors (√Greenwood variance).
///
/// # Errors
///
/// Returns [`SurvivalError::InvalidParameter`] when the input arrays are empty
/// or have mismatched lengths.
pub fn km_to_step_function(
    km_times: &[f64],
    km_survival: &[f64],
    km_stderr: Option<&[f64]>,
) -> SurvivalResult<StepFunction> {
    if km_times.is_empty() {
        return Err(SurvivalError::InvalidParameter(
            "km_times is empty".to_string(),
        ));
    }
    if km_times.len() != km_survival.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: km_times.len(),
            b: km_survival.len(),
        });
    }
    if let Some(se) = km_stderr {
        if se.len() != km_times.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: km_times.len(),
                b: se.len(),
            });
        }
    }
    Ok(StepFunction {
        times: km_times.to_vec(),
        values: km_survival.to_vec(),
        stderr: km_stderr.map(|s| s.to_vec()),
    })
}

/// Convert Nelson-Aalen cumulative hazard arrays into a [`StepFunction`].
///
/// # Errors
///
/// Returns [`SurvivalError::InvalidParameter`] / [`SurvivalError::DimensionMismatch`]
/// when inputs are empty or mismatched.
pub fn na_to_step_function(times: &[f64], cum_hazard: &[f64]) -> SurvivalResult<StepFunction> {
    if times.is_empty() {
        return Err(SurvivalError::InvalidParameter(
            "times is empty".to_string(),
        ));
    }
    if times.len() != cum_hazard.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: times.len(),
            b: cum_hazard.len(),
        });
    }
    Ok(StepFunction {
        times: times.to_vec(),
        values: cum_hazard.to_vec(),
        stderr: None,
    })
}

/// Convert a Cumulative Incidence Function (CIF) into a [`StepFunction`].
///
/// # Errors
///
/// Returns [`SurvivalError::InvalidParameter`] / [`SurvivalError::DimensionMismatch`]
/// when inputs are empty or mismatched.
pub fn cif_to_step_function(times: &[f64], cif: &[f64]) -> SurvivalResult<StepFunction> {
    if times.is_empty() {
        return Err(SurvivalError::InvalidParameter(
            "times is empty".to_string(),
        ));
    }
    if times.len() != cif.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: times.len(),
            b: cif.len(),
        });
    }
    Ok(StepFunction {
        times: times.to_vec(),
        values: cif.to_vec(),
        stderr: None,
    })
}

// ─── Analytical functions on StepFunction ────────────────────────────────────

/// Compute the restricted mean survival time (RMST) from a KM [`StepFunction`]
/// up to horizon `t_star`.
///
/// RMST = ∫₀^{t*} S(t) dt, approximated by the area under the piecewise-constant
/// step function (left-Riemann sum over each segment).
///
/// # Errors
///
/// Returns [`SurvivalError::InvalidParameter`] when `t_star <= 0`.
pub fn rmst_from_step(sf: &StepFunction, t_star: f64) -> SurvivalResult<f64> {
    if t_star <= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "t_star must be > 0, got {t_star}"
        )));
    }
    let mut area = 0.0_f64;
    let n = sf.times.len();

    // Breakpoints clipped to [0, t_star].
    // Build a merged list of breakpoints: 0.0, times[0], times[1], ..., t_star
    let t0 = sf.times[0];

    // Segment [0, min(t0, t_star)) — value is sf.values[0] (pre-first-event value).
    let seg_end = t0.min(t_star);
    if seg_end > 0.0 {
        area += sf.values[0] * seg_end;
    }
    if t_star <= t0 {
        return Ok(area);
    }

    // Segments [times[k], times[k+1]) for k = 0..n-2
    for k in 0..n {
        let left = sf.times[k];
        let right = if k + 1 < n { sf.times[k + 1] } else { t_star };
        let seg_left = left.max(t0);
        let seg_right = right.min(t_star);
        if seg_right <= seg_left {
            continue;
        }
        area += sf.values[k] * (seg_right - seg_left);
        if right >= t_star {
            break;
        }
    }
    Ok(area)
}

/// Find the median survival time from a KM [`StepFunction`].
///
/// Returns the first time `t` where `S(t) ≤ 0.5`, or `None` if the survival
/// function never drops to or below 0.5 within the observed time range.
pub fn median_survival(sf: &StepFunction) -> Option<f64> {
    for (&t, &s) in sf.times.iter().zip(sf.values.iter()) {
        if s <= 0.5 {
            return Some(t);
        }
    }
    None
}

/// Produce arrays suitable for matplotlib-compatible step rendering.
///
/// Prepends `(0, prepend_value)` (typically 1.0 for survival, 0.0 for hazard)
/// and appends `(t_max + epsilon, last_value)` so the plot extends to the end
/// of the observation window.
///
/// Returns `(times, values)` with `times.len() == values.len() == sf.times.len() + 2`.
pub fn step_plot_arrays(sf: &StepFunction, prepend_value: f64) -> (Vec<f64>, Vec<f64>) {
    let n = sf.times.len();
    let mut t_out = Vec::with_capacity(n + 2);
    let mut v_out = Vec::with_capacity(n + 2);

    // Prepend sentinel at t=0.
    t_out.push(0.0);
    v_out.push(prepend_value);

    // Main step-function values.
    t_out.extend_from_slice(&sf.times);
    v_out.extend_from_slice(&sf.values);

    // Append sentinel at t_max.
    let t_max = *sf.times.last().unwrap_or(&0.0);
    let last_val = *sf.values.last().unwrap_or(&prepend_value);
    t_out.push(t_max);
    v_out.push(last_val);

    (t_out, v_out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple KM-like step function with 5 events.
    fn simple_km() -> StepFunction {
        // S(t): 1.0 → 0.8 → 0.6 → 0.4 → 0.2
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![0.8, 0.6, 0.4, 0.2, 0.1];
        let stderr = Some(vec![0.02, 0.03, 0.04, 0.05, 0.06]);
        StepFunction {
            times,
            values,
            stderr,
        }
    }

    // ── Test 1: km_to_step_function produces correct length ─────────────────
    #[test]
    fn km_to_step_function_length() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let surv = vec![0.9, 0.8, 0.6, 0.4, 0.2];
        let sf =
            km_to_step_function(&times, &surv, None).expect("km_to_step_function should succeed");
        assert_eq!(sf.times.len(), 5);
        assert_eq!(sf.values.len(), 5);
        assert!(sf.stderr.is_none());
    }

    // ── Test 2: StepFunction.eval returns first value before first event ─────
    #[test]
    fn eval_before_first_event() {
        let sf = simple_km();
        // Before time=1: should return values[0] = 0.8
        let v = sf.eval(0.0);
        assert!(
            (v - 0.8).abs() < 1e-12,
            "eval before t=1 should return 0.8, got {v}"
        );
        let v05 = sf.eval(0.5);
        assert!(
            (v05 - 0.8).abs() < 1e-12,
            "eval at t=0.5 should return 0.8, got {v05}"
        );
    }

    // ── Test 3: eval returns correct values at and after events ─────────────
    #[test]
    fn eval_at_event_times() {
        let sf = simple_km();
        assert!((sf.eval(1.0) - 0.8).abs() < 1e-12);
        assert!((sf.eval(2.0) - 0.6).abs() < 1e-12);
        assert!((sf.eval(3.5) - 0.4).abs() < 1e-12, "mid-interval value");
        assert!((sf.eval(5.0) - 0.1).abs() < 1e-12);
    }

    // ── Test 4: eval at t > t_max returns last value ─────────────────────────
    #[test]
    fn eval_beyond_last_time() {
        let sf = simple_km();
        let last = *sf.values.last().expect("last should succeed");
        let v = sf.eval(100.0);
        assert!(
            (v - last).abs() < 1e-12,
            "eval beyond last time should return last value {last}, got {v}"
        );
    }

    // ── Test 5: eval_batch handles all times ─────────────────────────────────
    #[test]
    fn eval_batch_all_times() {
        let sf = simple_km();
        let queries = vec![0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        let results = sf.eval_batch(&queries);
        assert_eq!(results.len(), queries.len());
        // spot-check
        assert!((results[0] - 0.8).abs() < 1e-12); // before t=1
        assert!((results[3] - 0.6).abs() < 1e-12); // at t=2
        assert!((results[6] - 0.1).abs() < 1e-12); // beyond t_max
    }

    // ── Test 6: confidence_band produces lower ≤ upper ───────────────────────
    #[test]
    fn confidence_band_lower_le_upper() {
        let sf = simple_km();
        let (lower, upper) = sf
            .confidence_band(0.05)
            .expect("confidence_band should succeed");
        for (&lo, &hi) in lower.values.iter().zip(upper.values.iter()) {
            assert!(lo <= hi, "lower={lo} > upper={hi}");
        }
    }

    // ── Test 7: confidence_band is None when no stderr ───────────────────────
    #[test]
    fn confidence_band_none_without_stderr() {
        let sf = StepFunction {
            times: vec![1.0, 2.0],
            values: vec![0.8, 0.6],
            stderr: None,
        };
        assert!(sf.confidence_band(0.05).is_none());
    }

    // ── Test 8: to_regular_grid output shape ─────────────────────────────────
    #[test]
    fn to_regular_grid_shape() {
        let sf = simple_km();
        let (grid_t, grid_v) = sf
            .to_regular_grid(100)
            .expect("to_regular_grid should succeed");
        assert_eq!(grid_t.len(), 100);
        assert_eq!(grid_v.len(), 100);
        // Endpoints match
        assert!((grid_t[0] - 1.0).abs() < 1e-12);
        assert!((grid_t[99] - 5.0).abs() < 1e-10);
    }

    // ── Test 9: to_regular_grid n_points<2 returns error ────────────────────
    #[test]
    fn to_regular_grid_too_few_points_error() {
        let sf = simple_km();
        assert!(sf.to_regular_grid(1).is_err());
    }

    // ── Test 10: rmst_from_step on simple step function ──────────────────────
    #[test]
    fn rmst_simple_step_function() {
        // StepFunction: values[0]=0.5 applies to t in [0, times[1]=2).
        // In our eval() convention: eval(t) = values[0] = 0.5 for all t < times[1].
        // RMST up to t*=2:
        //   [0, times[0]=1):  value=0.5, width=1 → 0.5
        //   [times[0]=1, times[1]=2): value=0.5, width=1 → 0.5
        // Total = 1.0
        let sf = StepFunction {
            times: vec![1.0, 2.0],
            values: vec![0.5, 0.0],
            stderr: None,
        };
        let rmst = rmst_from_step(&sf, 2.0).expect("rmst_from_step should succeed");
        assert!((rmst - 1.0).abs() < 1e-10, "expected RMST=1.0, got {rmst}");
    }

    // ── Test 11: rmst t_star=0 returns error ─────────────────────────────────
    #[test]
    fn rmst_zero_horizon_error() {
        let sf = simple_km();
        assert!(rmst_from_step(&sf, 0.0).is_err());
    }

    // ── Test 12: median_survival finds crossing at 0.5 ───────────────────────
    #[test]
    fn median_survival_correct() {
        // S(t): drops to 0.4 at t=3 → first time <= 0.5 is t=3.
        let sf = simple_km();
        // values: [0.8, 0.6, 0.4, 0.2, 0.1] at times [1,2,3,4,5]
        let med = median_survival(&sf);
        assert_eq!(med, Some(3.0), "median should be at t=3");
    }

    // ── Test 13: median_survival returns None when S never hits 0.5 ──────────
    #[test]
    fn median_survival_none_when_high() {
        let sf = StepFunction {
            times: vec![1.0, 2.0, 3.0],
            values: vec![0.9, 0.8, 0.7],
            stderr: None,
        };
        assert!(median_survival(&sf).is_none());
    }

    // ── Test 14: step_plot_arrays prepends and appends correctly ─────────────
    #[test]
    fn step_plot_arrays_structure() {
        let sf = simple_km();
        let (t_arr, v_arr) = step_plot_arrays(&sf, 1.0);
        // Length = n + 2 = 5 + 2 = 7
        assert_eq!(t_arr.len(), 7);
        assert_eq!(v_arr.len(), 7);
        // First element: t=0, v=1.0 (prepend_value)
        assert!((t_arr[0] - 0.0).abs() < 1e-12);
        assert!((v_arr[0] - 1.0).abs() < 1e-12);
        // Last element: t=t_max=5, v=last_value=0.1
        assert!((t_arr[6] - 5.0).abs() < 1e-12);
        assert!((v_arr[6] - 0.1).abs() < 1e-12);
    }

    // ── Test 15: na_to_step_function correct ─────────────────────────────────
    #[test]
    fn na_to_step_function_correct() {
        let times = vec![1.0, 2.0, 3.0];
        let cum_haz = vec![0.1, 0.25, 0.5];
        let sf = na_to_step_function(&times, &cum_haz).expect("na_to_step_function should succeed");
        assert_eq!(sf.times.len(), 3);
        assert!((sf.eval(2.0) - 0.25).abs() < 1e-12);
    }

    // ── Test 16: cif_to_step_function correct ────────────────────────────────
    #[test]
    fn cif_to_step_function_correct() {
        let times = vec![1.0, 2.0, 3.0];
        let cif = vec![0.05, 0.15, 0.30];
        let sf = cif_to_step_function(&times, &cif).expect("cif_to_step_function should succeed");
        assert_eq!(sf.times.len(), 3);
        assert!((sf.eval(3.0) - 0.30).abs() < 1e-12);
    }

    // ── Test 17: km_to_step_function with stderr ──────────────────────────────
    #[test]
    fn km_to_step_function_with_stderr() {
        let times = vec![1.0, 2.0, 3.0];
        let surv = vec![0.9, 0.7, 0.5];
        let se = vec![0.05, 0.08, 0.12];
        let sf = km_to_step_function(&times, &surv, Some(&se)).expect("value should be present");
        assert!(sf.stderr.is_some());
        assert_eq!(sf.stderr.as_ref().expect("as_ref should succeed").len(), 3);
    }

    // ── Test 18: rmst_from_step within step interval ──────────────────────────
    #[test]
    fn rmst_step_at_midpoint() {
        // StepFunction: times=[2,10], values=[0.5,0.2].
        // eval() returns values[0]=0.5 for t in [0, times[1]=10).
        // BUT: values[0]=0.5 for t in [0, 2] (eval at t<2 and at t=2).
        //      values[1]=0.2 for t in (2, 10].
        // RMST to t*=4:
        //   [0, times[0]=2): value=0.5, width=2 → 1.0  (initial segment)
        //   [times[0]=2, t*=4): value=values[0]=0.5, width=2 → 1.0
        //   Total = 2.0
        // Actually the loop for k=0 covers [times[0]=2, times[1]=10), capped to t*=4:
        //   value=values[0]=0.5, width=4-2=2 → 1.0
        // Total = 1.0 (init) + 1.0 (loop k=0) = 2.0
        let sf = StepFunction {
            times: vec![2.0, 10.0],
            values: vec![0.5, 0.2],
            stderr: None,
        };
        let rmst = rmst_from_step(&sf, 4.0).expect("rmst_from_step should succeed");
        // [0,2): 0.5*2=1.0; [2,4): 0.5*2=1.0; Total=2.0
        assert!((rmst - 2.0).abs() < 1e-10, "expected 2.0, got {rmst}");
    }

    // ── Test 19: confidence_band: lower clamps to 0 ──────────────────────────
    #[test]
    fn confidence_band_lower_clamps_to_zero() {
        let sf = StepFunction {
            times: vec![1.0],
            values: vec![0.05],
            stderr: Some(vec![0.10]),
        };
        let (lower, upper) = sf
            .confidence_band(0.05)
            .expect("confidence_band should succeed");
        // Lower: 0.05 - 1.96*0.10 ≈ -0.146 → clamped to 0.0
        assert!(lower.values[0] >= 0.0);
        // Upper: 0.05 + 1.96*0.10 ≈ 0.246 → in [0,1]
        assert!(upper.values[0] > lower.values[0]);
    }

    // ── Test 20: empty times → km_to_step_function error ─────────────────────
    #[test]
    fn km_to_step_function_empty_error() {
        let result = km_to_step_function(&[], &[], None);
        assert!(result.is_err());
    }
} // end mod tests
