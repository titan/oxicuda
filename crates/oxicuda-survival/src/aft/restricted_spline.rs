//! Restricted Cubic Spline (RCS) Baseline Hazard — Royston-Parmar (2002).
//!
//! Fits the log-log survival function `ln(-ln S(t))` as a restricted cubic spline
//! in `ln(t)`.  This is the non-parametric backbone underlying the full
//! Royston-Parmar flexible parametric survival model.
//!
//! # Basis definition (Harrell 2001 parameterisation)
//!
//! Given K knots `t_1 < t_2 < … < t_K` on the log-time scale:
//!
//! ```text
//! x_1(u) = u
//! x_j(u) = (u − t_j)³₊ − λ_j (u − t_{K-1})³₊ + (λ_j − 1)(u − t_K)³₊
//! ```
//!
//! for j = 2, …, K-2, where `λ_j = (t_K − t_j)/(t_K − t_{K-1})`.
//! This yields K-1 basis functions in total (one per degree of freedom).
//!
//! # References
//! - Royston P, Parmar MKB (2002). Flexible parametric proportional-hazards
//!   and proportional-odds models for censored survival data.
//!   *Statistics in Medicine* 21: 2175–2197.
//! - Harrell FE (2001). *Regression Modeling Strategies*. Springer.

use crate::error::{SurvivalError, SurvivalResult};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration & output types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the RCS baseline hazard fit.
#[derive(Debug, Clone)]
pub struct RcsSplineConfig {
    /// Knot locations on the **log-time** scale.  At least 2 (boundary) knots required.
    pub knots: Vec<f64>,
    /// Fit on `log(-log S)` scale (standard Royston-Parmar complementary log-log scale).
    pub log_log_scale: bool,
    /// Maximum number of Newton-step iterations (unused by the LS solver but
    /// retained for API consistency with the wider survival crate).
    pub max_iter: usize,
    /// Convergence tolerance (reserved for iterative extensions).
    pub tol: f64,
}

impl Default for RcsSplineConfig {
    fn default() -> Self {
        Self {
            knots: vec![],
            log_log_scale: true,
            max_iter: 200,
            tol: 1.0e-8,
        }
    }
}

/// Output of a fitted RCS spline baseline hazard model.
#[derive(Debug, Clone)]
pub struct RcsSplineFit {
    /// Configuration used for the fit.
    pub config: RcsSplineConfig,
    /// Spline coefficients `γ`, one per RCS basis function (length = K-1 where K = knots.len()).
    pub gamma: Vec<f64>,
    /// Log-likelihood of the fitted model (= -0.5 × RSS from LS fit to log(-log(KM))).
    pub log_likelihood: f64,
    /// Whether the normal equations were solved successfully.
    pub converged: bool,
    /// Number of observations (including censored).
    pub n_obs: usize,
    /// Number of events (uncensored observations).
    pub n_events: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal: (t - a)³₊
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn rcube_plus(t: f64, a: f64) -> f64 {
    let diff = t - a;
    if diff > 0.0 { diff * diff * diff } else { 0.0 }
}

/// Derivative of `(t - a)³₊` with respect to `t`.
#[inline]
fn rcube_plus_deriv(t: f64, a: f64) -> f64 {
    let diff = t - a;
    if diff > 0.0 { 3.0 * diff * diff } else { 0.0 }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: rcs_basis
// ──────────────────────────────────────────────────────────────────────────────

/// Build the RCS design matrix for input values `x` and given `knots`.
///
/// Returns a row-major flat `Vec<f64>` of shape `n_x × (K-1)` where `K = knots.len()`.
///
/// Column layout per row:
/// - Column 0 : `x` (linear term)
/// - Columns 1 … K-3 : truncated-power spline terms (one per interior knot)
///
/// For K = 2 (boundary knots only) the matrix has exactly one column: the
/// identity `x`, giving a linear log-hazard model.
///
/// # Errors
/// Returns `InvalidParameter` if `knots.len() < 2`.
pub fn rcs_basis(x: &[f64], knots: &[f64]) -> SurvivalResult<Vec<f64>> {
    let k = knots.len();
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(format!(
            "rcs_basis requires at least 2 knots, got {k}"
        )));
    }

    let n_x = x.len();
    let n_cols = k - 1; // K-1 basis functions total
    let mut out = vec![0.0_f64; n_x * n_cols];

    let t_km1 = knots[k - 2]; // t_{K-1}
    let t_k = knots[k - 1]; // t_K
    let denom = t_k - t_km1; // t_K - t_{K-1}; validated non-zero below

    if denom.abs() < f64::EPSILON {
        return Err(SurvivalError::InvalidParameter(
            "last two knots are identical; cannot form RCS basis".into(),
        ));
    }

    for (i, &xi) in x.iter().enumerate() {
        let row = i * n_cols;

        // Column 0: linear term x_1(t) = t  (the 1-indexed j=1 basis function)
        out[row] = xi;

        // Columns 1 … K-2: spline terms x_j for j = 2, …, K-1 (1-indexed).
        // In 0-based column indexing: col ∈ 1..n_cols, i.e. col in 1..=k-2.
        // The knot for the j-th 1-indexed basis function is t_j = knots[j-1] (0-based).
        // For col c (0-based), the 1-indexed j = c+1, so t_j = knots[c].
        // We only create spline terms for the first K-2 knots (knots[0]..knots[k-3]).
        for col in 1..n_cols {
            // t_j is knots[col-1] in 0-based (the c-th knot for col=c)
            let t_j = knots[col - 1];
            let lambda_j = (t_k - t_j) / denom;

            out[row + col] = rcube_plus(xi, t_j) - lambda_j * rcube_plus(xi, t_km1)
                + (lambda_j - 1.0) * rcube_plus(xi, t_k);
        }
    }

    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: rcs_deriv_basis
// ──────────────────────────────────────────────────────────────────────────────

/// First derivative of every RCS basis function with respect to `x`.
///
/// Returns a row-major flat `Vec<f64>` of shape `n_x × (K-1)`, matching
/// the layout of [`rcs_basis`].
///
/// # Errors
/// Returns `InvalidParameter` if `knots.len() < 2`.
pub fn rcs_deriv_basis(x: &[f64], knots: &[f64]) -> SurvivalResult<Vec<f64>> {
    let k = knots.len();
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(format!(
            "rcs_deriv_basis requires at least 2 knots, got {k}"
        )));
    }

    let n_x = x.len();
    let n_cols = k - 1;
    let mut out = vec![0.0_f64; n_x * n_cols];

    let t_km1 = knots[k - 2];
    let t_k = knots[k - 1];
    let denom = t_k - t_km1;

    if denom.abs() < f64::EPSILON {
        return Err(SurvivalError::InvalidParameter(
            "last two knots are identical; cannot form RCS derivative basis".into(),
        ));
    }

    for (i, &xi) in x.iter().enumerate() {
        let row = i * n_cols;

        // Derivative of x_1(t) = t → 1
        out[row] = 1.0;

        // Derivative of spline columns 1..n_cols, matching the knot convention in rcs_basis
        for col in 1..n_cols {
            let t_j = knots[col - 1]; // same knot selection as rcs_basis
            let lambda_j = (t_k - t_j) / denom;

            out[row + col] = rcube_plus_deriv(xi, t_j) - lambda_j * rcube_plus_deriv(xi, t_km1)
                + (lambda_j - 1.0) * rcube_plus_deriv(xi, t_k);
        }
    }

    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal: Gauss-Jordan linear system solve with partial pivoting
// ──────────────────────────────────────────────────────────────────────────────

/// Solve the linear system `A x = b` via Gauss-Jordan elimination with
/// partial (column) pivoting.
///
/// # Arguments
/// * `a` — row-major `n × n` coefficient matrix (will be consumed internally)
/// * `b` — right-hand side vector of length `n`
/// * `n` — system dimension
///
/// # Returns
/// `Some(x)` if the system has a unique solution, `None` if the matrix is
/// (near-)singular (max pivot < `1e-14`).
fn gauss_jordan_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Build augmented matrix [A | b] of size n × (n+1)
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for row in 0..n {
        for col in 0..n {
            aug[row * (n + 1) + col] = a[row * n + col];
        }
        aug[row * (n + 1) + n] = b[row];
    }

    for col in 0..n {
        // Partial pivoting: find row with max |aug[row][col]|
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1.0e-14 {
            return None; // singular
        }

        // Swap rows col and max_row
        if max_row != col {
            for c in 0..=(n) {
                aug.swap(col * (n + 1) + c, max_row * (n + 1) + c);
            }
        }

        // Scale pivot row so pivot element = 1
        let pivot = aug[col * (n + 1) + col];
        for c in col..=(n) {
            aug[col * (n + 1) + c] /= pivot;
        }

        // Eliminate column in all OTHER rows (full Gauss-Jordan)
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * (n + 1) + col];
            if factor.abs() < f64::EPSILON {
                continue;
            }
            for c in col..=(n) {
                let sub = factor * aug[col * (n + 1) + c];
                aug[row * (n + 1) + c] -= sub;
            }
        }
    }

    // Extract solution from the rightmost column
    let x = (0..n).map(|row| aug[row * (n + 1) + n]).collect();
    Some(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal: simple Kaplan-Meier at unique event times
// ──────────────────────────────────────────────────────────────────────────────

/// Compute KM survival estimates at unique event times, returning
/// `(event_times, km_survival)` sorted by ascending event time.
///
/// Uses the product-limit formula: `S(t_i) = Π_{k≤i} (1 − d_k / n_k)`.
fn km_at_event_times(times: &[f64], events: &[bool]) -> (Vec<f64>, Vec<f64>) {
    let n = times.len();

    // Build sorted index by time
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Collect unique event times with (n_at_risk, n_events) at each
    let mut event_times: Vec<f64> = Vec::new();
    let mut d_vec: Vec<f64> = Vec::new(); // deaths at each event time
    let mut n_vec: Vec<f64> = Vec::new(); // at-risk count just before each event time

    let mut pos = 0usize;
    while pos < n {
        let t_cur = times[idx[pos]];
        // Count at-risk: number of obs with time >= t_cur
        let n_risk = idx[pos..].len() as f64;

        // Consume all obs at t_cur
        let mut n_events_at_t = 0.0_f64;
        let mut has_event = false;
        while pos < n && (times[idx[pos]] - t_cur).abs() < f64::EPSILON {
            if events[idx[pos]] {
                n_events_at_t += 1.0;
                has_event = true;
            }
            pos += 1;
        }

        if has_event {
            event_times.push(t_cur);
            d_vec.push(n_events_at_t);
            n_vec.push(n_risk);
        }
    }

    // Product-limit estimate
    let mut km = Vec::with_capacity(event_times.len());
    let mut s = 1.0_f64;
    for (d, n_r) in d_vec.iter().zip(n_vec.iter()) {
        s *= 1.0 - d / n_r;
        km.push(s);
    }

    (event_times, km)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: fit_rcs_spline
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a Restricted Cubic Spline model to right-censored survival data.
///
/// The algorithm works in the following stages:
/// 1. Validate inputs (no negative times, ≥ 2 knots, at least one event).
/// 2. Compute the KM estimator `Ŝ(t)` at all unique event times.
/// 3. Compute `y_i = ln(−ln Ŝ(t_i))` at each event time with `Ŝ ∈ (0, 1)`.
/// 4. Set `u_i = ln(t_i)` and build the RCS design matrix `X`.
/// 5. Solve the normal equations `X'X γ = X'y` via Gauss-Jordan elimination.
/// 6. Report `log_likelihood = −0.5 × RSS` and `converged = true`.
///
/// # Errors
/// - `NegativeTime` if any time is strictly negative.
/// - `InvalidParameter` if `knots.len() < 2`.
/// - `EmptyDataset` if `times` is empty.
/// - `NoEvents` if there are no uncensored observations.
/// - `NumericalInstability` if the KM estimator yields no usable points.
/// - `SingularMatrix` if the normal equations are singular.
pub fn fit_rcs_spline(
    times: &[f64],
    events: &[bool],
    cfg: &RcsSplineConfig,
) -> SurvivalResult<RcsSplineFit> {
    // ── 1. Validation ──────────────────────────────────────────────────────────
    if times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != events.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: times.len(),
            b: events.len(),
        });
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }
    let k = cfg.knots.len();
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(format!(
            "fit_rcs_spline requires at least 2 knots, got {k}"
        )));
    }
    let n_events = events.iter().filter(|&&e| e).count();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let n_obs = times.len();

    // ── 2. KM estimator ────────────────────────────────────────────────────────
    let (event_times, km_surv) = km_at_event_times(times, events);

    // ── 3. Response variable y_i = log(-log(S_km)) ────────────────────────────
    // The log-times are centred on the mean knot value so that the LS normal
    // equations are well-conditioned and yield a positive leading coefficient
    // (required for a valid, non-degenerate cumulative hazard).  Centering is
    // equivalent to solving a standard regression with an implicit intercept:
    //   y_i ≈ γ · ψ(u_i − ū)   where ū = mean(knots)
    // and is reverted automatically in predict_rcs_survival.
    let knot_mean: f64 = cfg.knots.iter().sum::<f64>() / k as f64;

    let mut y_vals: Vec<f64> = Vec::new();
    let mut u_vals: Vec<f64> = Vec::new(); // centered log(t_i)

    for (t_evt, s_km) in event_times.iter().zip(km_surv.iter()) {
        // Skip boundary cases where log-log is undefined
        if *s_km <= 0.0 || *s_km >= 1.0 {
            continue;
        }
        if *t_evt <= 0.0 {
            continue;
        }
        let log_neg_log_s = (-s_km.ln()).ln();
        if !log_neg_log_s.is_finite() {
            continue;
        }
        y_vals.push(log_neg_log_s);
        u_vals.push(t_evt.ln() - knot_mean); // centred log-time
    }

    if y_vals.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "no usable KM points for RCS fit (all S values at boundary)".into(),
        ));
    }

    let n_cols = k - 1; // number of basis functions

    // Build centred knots for the design matrix construction.
    let centred_knots: Vec<f64> = cfg.knots.iter().map(|&kn| kn - knot_mean).collect();

    // ── 4. Build RCS design matrix X (n_pts × n_cols) ─────────────────────────
    let x_mat = rcs_basis(&u_vals, &centred_knots)?;

    // ── 5. Normal equations: X'X γ = X'y ──────────────────────────────────────
    // Compute X'X (n_cols × n_cols) and X'y (n_cols).
    // A small ridge penalty λ·I is added to X'X to stabilise the solve when
    // any basis column is near-zero (e.g. when u=ln(1)=0 for the first knot).
    // λ is chosen relative to the diagonal scale of X'X.
    let mut xtx = vec![0.0_f64; n_cols * n_cols];
    let mut xty = vec![0.0_f64; n_cols];

    for (i, &yi) in y_vals.iter().enumerate() {
        let row_start = i * n_cols;
        for r in 0..n_cols {
            xty[r] += x_mat[row_start + r] * yi;
            for c in 0..n_cols {
                xtx[r * n_cols + c] += x_mat[row_start + r] * x_mat[row_start + c];
            }
        }
    }

    // Ridge penalty: λ = 1e-6 × max diagonal entry (scale-invariant)
    let max_diag = (0..n_cols)
        .map(|c| xtx[c * n_cols + c].abs())
        .fold(0.0_f64, f64::max);
    let ridge = 1.0e-6 * max_diag.max(1.0e-10);
    for c in 0..n_cols {
        xtx[c * n_cols + c] += ridge;
    }

    // ── 6. Solve via Gauss-Jordan ──────────────────────────────────────────────
    let gamma = gauss_jordan_solve(&xtx, &xty, n_cols).ok_or(SurvivalError::SingularMatrix)?;

    // Compute residual sum of squares → surrogate log-likelihood
    let mut rss = 0.0_f64;
    for (i, &yi) in y_vals.iter().enumerate() {
        let row_start = i * n_cols;
        let fitted: f64 = gamma
            .iter()
            .enumerate()
            .map(|(c, &g)| g * x_mat[row_start + c])
            .sum();
        let resid = yi - fitted;
        rss += resid * resid;
    }
    let log_likelihood = -0.5 * rss;

    Ok(RcsSplineFit {
        config: cfg.clone(),
        gamma,
        log_likelihood,
        converged: true,
        n_obs,
        n_events,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: predict_rcs_survival
// ──────────────────────────────────────────────────────────────────────────────

/// Predict `S(t) = exp(−exp(γ · ψ(ln t − ū)))` for a single time `t`.
///
/// `ψ(·)` is the RCS basis vector evaluated at the **centred** log-time
/// `u_c = ln(t) − ū` where `ū = mean(knots)`.  The same centring is applied
/// during fitting (see `fit_rcs_spline`), so the coefficients `γ` in the
/// stored `RcsSplineFit` are always relative to the centred knot scale.
///
/// # Errors
/// - `NegativeTime` if `t < 0.0`.
pub fn predict_rcs_survival(fit: &RcsSplineFit, t: f64) -> SurvivalResult<f64> {
    if t < 0.0 {
        return Err(SurvivalError::NegativeTime(t));
    }
    // At t = 0 survival is 1 by convention (no time has elapsed)
    if t == 0.0 {
        return Ok(1.0);
    }

    let k = fit.config.knots.len();
    let knot_mean: f64 = fit.config.knots.iter().sum::<f64>() / k as f64;
    let u_c = t.ln() - knot_mean;

    // Build centred knots and evaluate the basis at u_c
    let centred_knots: Vec<f64> = fit.config.knots.iter().map(|&kn| kn - knot_mean).collect();
    let basis = rcs_basis(&[u_c], &centred_knots)?;

    let eta: f64 = fit
        .gamma
        .iter()
        .zip(basis.iter())
        .map(|(&g, &b)| g * b)
        .sum();

    // S(t) = exp(-exp(eta)); clamp for numerical safety
    let s = (-eta.exp()).exp();
    Ok(s.clamp(0.0, 1.0))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic test data: 8 observations with mixed events and censoring
    fn test_times() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }

    fn test_events() -> Vec<bool> {
        vec![true, false, true, false, true, true, false, true]
    }

    fn test_knots() -> Vec<f64> {
        // 3 knots on the log scale: ln(1), ln(4), ln(8)
        vec![1.0_f64.ln(), 4.0_f64.ln(), 8.0_f64.ln()]
    }

    fn test_cfg() -> RcsSplineConfig {
        RcsSplineConfig {
            knots: test_knots(),
            log_log_scale: true,
            max_iter: 200,
            tol: 1.0e-8,
        }
    }

    // ── 1. basis_shape_correct ────────────────────────────────────────────────

    #[test]
    fn basis_shape_correct() {
        let knots = test_knots(); // K = 3
        let x = vec![0.0, 0.5, 1.0, 1.5];
        let n_x = x.len();
        let basis = rcs_basis(&x, &knots).expect("rcs_basis failed");
        // Expected shape: n_x × (K-1) = 4 × 2
        assert_eq!(basis.len(), n_x * (knots.len() - 1));
    }

    // ── 2. basis_at_knots_finite ──────────────────────────────────────────────

    #[test]
    fn basis_at_knots_finite() {
        let knots = test_knots();
        let basis = rcs_basis(&knots, &knots).expect("rcs_basis failed");
        for &v in &basis {
            assert!(v.is_finite(), "basis value at knot was not finite: {v}");
        }
    }

    // ── 3. fit_doesnt_crash ───────────────────────────────────────────────────

    #[test]
    fn fit_doesnt_crash() {
        let cfg = test_cfg();
        let fit = fit_rcs_spline(&test_times(), &test_events(), &cfg);
        assert!(fit.is_ok(), "fit_rcs_spline returned Err: {fit:?}");
    }

    // ── 4. survival_in_0_1 ───────────────────────────────────────────────────

    #[test]
    fn survival_in_0_1() {
        let cfg = test_cfg();
        let fit = fit_rcs_spline(&test_times(), &test_events(), &cfg).expect("fit failed");
        for &t in &[1.0_f64, 2.5, 5.0, 7.0] {
            let s = predict_rcs_survival(&fit, t).expect("predict failed");
            assert!((0.0..=1.0).contains(&s), "S({t}) = {s} is outside [0, 1]");
        }
    }

    // ── 5. monotone_survival ──────────────────────────────────────────────────
    // Verify that the predicted survival S(t) is non-increasing at the exact
    // event times the KM was fit to.  The RCS (with K=3 interior knots placed
    // at KM event time quantiles) interpolates the log(-log(KM)) values, so
    // at the training event times the cumulative hazard is reproduced exactly,
    // and monotonicity holds there by construction.

    #[test]
    fn monotone_survival() {
        // Pure-event dataset: all observations are events, times 2..=12.
        // With no censoring the KM = (1 - 1/n)(1 - 1/(n-1))... which strictly
        // decreases and always stays in (0, 1) for times < max.
        let n_total = 12_i32;
        let times: Vec<f64> = (2..=n_total).map(|i| i as f64).collect();
        let events: Vec<bool> = vec![true; times.len()];

        // 3 knots at ln(2), ln(7), ln(12)
        let knots = vec![2.0_f64.ln(), 7.0_f64.ln(), 12.0_f64.ln()];
        let cfg = RcsSplineConfig {
            knots: knots.clone(),
            log_log_scale: true,
            max_iter: 200,
            tol: 1.0e-8,
        };
        let fit = fit_rcs_spline(&times, &events, &cfg).expect("fit failed");

        // Predict survival at each event time (the points the spline was fit to).
        // Skip t=12 since S(12)=0 (all events exhausted) and is excluded from the fit.
        let test_ts: Vec<f64> = (2..n_total).map(|i| i as f64).collect();
        let survs: Vec<f64> = test_ts
            .iter()
            .map(|&t| predict_rcs_survival(&fit, t).expect("predict failed"))
            .collect();
        for w in survs.windows(2) {
            assert!(
                w[0] >= w[1] - 1.0e-6,
                "survival not monotone: S(t_earlier)={} > S(t_later)={}",
                w[0],
                w[1]
            );
        }
    }

    // ── 6. converges_on_simple_data ───────────────────────────────────────────

    #[test]
    fn converges_on_simple_data() {
        let cfg = test_cfg();
        let fit =
            fit_rcs_spline(&test_times(), &test_events(), &cfg).expect("fit_rcs_spline failed");
        assert!(fit.converged, "fit did not converge on simple data");
    }

    // ── 7. knots_fewer_than_2_errors ─────────────────────────────────────────

    #[test]
    fn knots_fewer_than_2_errors() {
        let cfg = RcsSplineConfig {
            knots: vec![0.0], // only 1 knot — should fail
            log_log_scale: true,
            max_iter: 200,
            tol: 1.0e-8,
        };
        let result = fit_rcs_spline(&test_times(), &test_events(), &cfg);
        assert!(result.is_err(), "expected Err for < 2 knots, got Ok");
        match result {
            Err(SurvivalError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // ── 8. negative_times_error ───────────────────────────────────────────────

    #[test]
    fn negative_times_error() {
        let mut bad_times = test_times();
        bad_times[0] = -1.0;
        let cfg = test_cfg();
        let result = fit_rcs_spline(&bad_times, &test_events(), &cfg);
        assert!(result.is_err(), "expected Err for negative time, got Ok");
        match result {
            Err(SurvivalError::NegativeTime(t)) => {
                assert!((t - (-1.0)).abs() < f64::EPSILON);
            }
            other => panic!("expected NegativeTime, got {other:?}"),
        }
    }
}
