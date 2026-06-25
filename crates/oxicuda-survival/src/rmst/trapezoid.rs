//! Trapezoidal RMST integration and rectangle-vs-trapezoid accuracy verification.
//!
//! The default RMST estimator ([`crate::rmst::restricted_mean::restricted_mean_from_curve`])
//! integrates a *right-continuous step function* `S(t)` via the left-rectangle rule.
//! For a genuine Kaplan-Meier step function that rule is **exact**, because `S` is
//! piecewise constant between event times.
//!
//! When the underlying survivor curve is *smooth* (e.g. a parametric AFT survivor,
//! a Royston-Parmar spline survivor, or a finely-resampled curve) but is only
//! available on a grid `t_0 < t_1 < … < t_m`, the left-rectangle rule incurs an
//! `O(h)` quadrature bias while the composite **trapezoidal** rule is `O(h²)`:
//!
//! ```text
//!   ∫_a^b S(t) dt ≈ Σ_k (t_{k+1} − t_k) · (S(t_k) + S(t_{k+1})) / 2 .
//! ```
//!
//! This module supplies the trapezoidal integrator, a rectangle integrator over the
//! same grid for parity, and a [`QuadratureComparison`] helper that reports both
//! estimates against an analytic reference so the accuracy claim is *verifiable on the
//! CPU* (mirroring the `rmst_integrate` PTX kernel, whose on-device numerical accuracy
//! must match this reference within tolerance).

use crate::error::{SurvivalError, SurvivalResult};

/// Validate a quadrature grid: finite, non-decreasing, with `> 0` width and matching
/// survivor sample length.
fn validate_grid(grid: &[f64], values: &[f64]) -> SurvivalResult<()> {
    if grid.len() != values.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![grid.len()],
            got: vec![values.len()],
        });
    }
    if grid.len() < 2 {
        return Err(SurvivalError::InvalidParameter(
            "quadrature grid needs at least 2 points".to_string(),
        ));
    }
    for w in grid.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() {
            return Err(SurvivalError::InvalidParameter(
                "non-finite grid point".to_string(),
            ));
        }
        if w[1] <= w[0] {
            return Err(SurvivalError::InvalidParameter(
                "grid must be strictly increasing".to_string(),
            ));
        }
    }
    for v in values {
        if !v.is_finite() {
            return Err(SurvivalError::InvalidParameter(
                "non-finite survivor value".to_string(),
            ));
        }
    }
    Ok(())
}

/// Composite **trapezoidal** integral of a survivor curve sampled on `grid`.
///
/// `grid` and `values` must have equal length `≥ 2`, with `grid` strictly increasing.
/// `values[k] = S(grid[k])`. The integral is taken over `[grid[0], grid[last]]`.
///
/// `O(h²)` accuracy for a twice-differentiable survivor.
pub fn trapezoidal_rmst_from_grid(grid: &[f64], values: &[f64]) -> SurvivalResult<f64> {
    validate_grid(grid, values)?;
    let mut area = 0.0_f64;
    for k in 0..grid.len() - 1 {
        let h = grid[k + 1] - grid[k];
        area += 0.5 * h * (values[k] + values[k + 1]);
    }
    Ok(area)
}

/// Composite **left-rectangle** integral of a survivor curve sampled on `grid`.
///
/// Identical grid contract to [`trapezoidal_rmst_from_grid`]. Uses the value at the
/// *left* endpoint of each panel — the step-function convention of the KM RMST
/// estimator. `O(h)` accuracy for a smooth survivor (exact for a step survivor whose
/// jumps coincide with grid points).
pub fn rectangle_rmst_from_grid(grid: &[f64], values: &[f64]) -> SurvivalResult<f64> {
    validate_grid(grid, values)?;
    let mut area = 0.0_f64;
    for k in 0..grid.len() - 1 {
        let h = grid[k + 1] - grid[k];
        area += h * values[k];
    }
    Ok(area)
}

/// Side-by-side rectangle / trapezoid accuracy report against an analytic reference.
#[derive(Debug, Clone)]
pub struct QuadratureComparison {
    /// Upper integration limit τ (= last grid point).
    pub tau: f64,
    /// Number of panels (= grid length − 1).
    pub panels: usize,
    /// Left-rectangle RMST estimate.
    pub rectangle: f64,
    /// Trapezoidal RMST estimate.
    pub trapezoid: f64,
    /// Analytic reference value `∫₀^τ S(t) dt` (caller-supplied exact integral).
    pub reference: f64,
    /// `|rectangle − reference|`.
    pub rectangle_abs_error: f64,
    /// `|trapezoid − reference|`.
    pub trapezoid_abs_error: f64,
}

impl QuadratureComparison {
    /// Whether the trapezoidal estimate is at least as accurate as the rectangle
    /// estimate (true whenever `S` is smooth and the grid resolves it).
    #[must_use]
    pub fn trapezoid_is_better(&self) -> bool {
        self.trapezoid_abs_error <= self.rectangle_abs_error + 1.0e-15
    }
}

/// Sample an analytic survivor `s_fn` on a uniform grid of `panels + 1` points over
/// `[0, tau]`, integrate it with both the rectangle and trapezoid rules, and compare
/// against the caller-supplied analytic `reference = ∫₀^τ S(t) dt`.
///
/// This is the CPU verification harness for the `rmst_integrate` quadrature: it
/// quantifies the `O(h)` vs `O(h²)` bias on a known curve so a downstream GPU run of
/// the PTX kernel can be checked against the same reference numbers.
///
/// # Errors
/// Returns [`SurvivalError::InvalidParameter`] if `tau ≤ 0`, `panels == 0`, or
/// `reference` is non-finite.
pub fn compare_quadrature<F>(
    s_fn: F,
    tau: f64,
    panels: usize,
    reference: f64,
) -> SurvivalResult<QuadratureComparison>
where
    F: Fn(f64) -> f64,
{
    if !tau.is_finite() || tau <= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "tau must be > 0, got {tau}"
        )));
    }
    if panels == 0 {
        return Err(SurvivalError::InvalidParameter(
            "panels must be >= 1".to_string(),
        ));
    }
    if !reference.is_finite() {
        return Err(SurvivalError::InvalidParameter(
            "reference integral must be finite".to_string(),
        ));
    }
    let n = panels + 1;
    let h = tau / panels as f64;
    let mut grid = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for k in 0..n {
        let t = h * k as f64;
        grid.push(t);
        values.push(s_fn(t));
    }
    let rectangle = rectangle_rmst_from_grid(&grid, &values)?;
    let trapezoid = trapezoidal_rmst_from_grid(&grid, &values)?;
    Ok(QuadratureComparison {
        tau,
        panels,
        rectangle,
        trapezoid,
        reference,
        rectangle_abs_error: (rectangle - reference).abs(),
        trapezoid_abs_error: (trapezoid - reference).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a smooth exponential survivor S(t)=exp(-t), the trapezoidal rule must be
    /// strictly more accurate than the left-rectangle rule, and both must converge.
    #[test]
    fn trapezoid_beats_rectangle_on_exponential() {
        let tau = 4.0_f64;
        // ∫₀^τ e^{-t} dt = 1 − e^{-τ}.
        let reference = 1.0 - (-tau).exp();
        let cmp = compare_quadrature(|t| (-t).exp(), tau, 64, reference).expect("ok");
        assert!(cmp.trapezoid_is_better());
        // Trapezoid should be far tighter than rectangle here.
        assert!(cmp.trapezoid_abs_error < cmp.rectangle_abs_error * 0.05);
        assert!(cmp.trapezoid_abs_error < 1.0e-3);
    }

    /// Doubling the panel count quarters the trapezoidal error (O(h²) confirmation).
    #[test]
    fn trapezoid_second_order_convergence() {
        let tau = 3.0_f64;
        let reference = 1.0 - (-tau).exp();
        let coarse = compare_quadrature(|t| (-t).exp(), tau, 32, reference).expect("ok");
        let fine = compare_quadrature(|t| (-t).exp(), tau, 64, reference).expect("ok");
        // Halving h should shrink the trapezoidal error by ~4× (allow generous slack).
        let ratio = coarse.trapezoid_abs_error / fine.trapezoid_abs_error.max(1.0e-30);
        assert!(ratio > 3.0, "expected ~4x error reduction, got {ratio}");
    }

    /// On a linear survivor S(t)=1−t/τ the trapezoidal rule is exact (integrates a
    /// straight line with no error), so the error must be at machine precision.
    #[test]
    fn trapezoid_exact_on_linear() {
        let tau = 2.0;
        // ∫₀^τ (1 − t/τ) dt = τ/2.
        let reference = tau / 2.0;
        let cmp = compare_quadrature(|t| 1.0 - t / tau, tau, 10, reference).expect("ok");
        assert!(cmp.trapezoid_abs_error < 1.0e-12);
        // Rectangle rule is biased high on a decreasing curve.
        assert!(cmp.rectangle_abs_error > 1.0e-3);
    }

    /// Trapezoidal grid integral matches a hand-computed value exactly.
    #[test]
    fn trapezoid_hand_value() {
        // Grid t=[0,1,3], S=[1.0,0.5,0.25].
        // Panel 1: width 1, (1.0+0.5)/2 = 0.75.
        // Panel 2: width 2, (0.5+0.25)/2 = 0.375 ⇒ 0.75. Total = 1.5.
        let grid = [0.0, 1.0, 3.0];
        let vals = [1.0, 0.5, 0.25];
        let area = trapezoidal_rmst_from_grid(&grid, &vals).expect("ok");
        assert!((area - 1.5).abs() < 1.0e-12);
    }

    /// Rectangle grid integral matches a hand-computed value exactly.
    #[test]
    fn rectangle_hand_value() {
        // Grid t=[0,1,3], S=[1.0,0.5,0.25].
        // Panel 1: width 1 × 1.0 = 1.0. Panel 2: width 2 × 0.5 = 1.0. Total = 2.0.
        let grid = [0.0, 1.0, 3.0];
        let vals = [1.0, 0.5, 0.25];
        let area = rectangle_rmst_from_grid(&grid, &vals).expect("ok");
        assert!((area - 2.0).abs() < 1.0e-12);
    }

    /// On a genuine step survivor sampled at its own jump points, the rectangle rule
    /// is exact — confirming the default KM RMST convention is unbiased there.
    #[test]
    fn rectangle_exact_on_step_at_jumps() {
        // S = 1 on [0,1), 0.5 on [1,2), 0 on [2,∞). Sample at the jump points.
        let grid = [0.0, 1.0, 2.0];
        let vals = [1.0, 0.5, 0.0];
        // ∫₀² S = 1·1 + 1·0.5 = 1.5 (rectangle, left value) — exact for the step.
        let area = rectangle_rmst_from_grid(&grid, &vals).expect("ok");
        assert!((area - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_bad_grid() {
        assert!(trapezoidal_rmst_from_grid(&[0.0], &[1.0]).is_err());
        assert!(trapezoidal_rmst_from_grid(&[0.0, 0.0], &[1.0, 1.0]).is_err());
        assert!(trapezoidal_rmst_from_grid(&[1.0, 0.0], &[1.0, 1.0]).is_err());
        assert!(trapezoidal_rmst_from_grid(&[0.0, 1.0], &[1.0]).is_err());
        assert!(rectangle_rmst_from_grid(&[0.0, f64::NAN], &[1.0, 0.5]).is_err());
    }

    #[test]
    fn rejects_bad_compare_args() {
        let reference = 1.0;
        assert!(compare_quadrature(|t| (-t).exp(), 0.0, 8, reference).is_err());
        assert!(compare_quadrature(|t| (-t).exp(), 4.0, 0, reference).is_err());
        assert!(compare_quadrature(|t| (-t).exp(), 4.0, 8, f64::NAN).is_err());
    }
}
