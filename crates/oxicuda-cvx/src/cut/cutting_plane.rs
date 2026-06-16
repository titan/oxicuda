//! Kelley's cutting-plane method for convex minimisation over a box.
//!
//! Minimises a convex function `f : ℝᵈ → ℝ` over the box `l ≤ x ≤ u` by outer
//! approximation, following Kelley (1960), "The Cutting-Plane Method for Solving
//! Convex Programs". The epigraph of `f` is approximated from below by a finite
//! set of supporting hyperplanes
//!
//! ```text
//!   f(xₖ) + gₖᵀ (x − xₖ) ≤ η,      gₖ = ∇f(xₖ),
//! ```
//!
//! and the master linear program
//!
//! ```text
//!   minimize   η
//!   subject to η ≥ f(xₖ) + gₖᵀ (x − xₖ)   for all cuts k,
//!              l ≤ x ≤ u
//! ```
//!
//! is solved at each iteration. Because every cut is a valid global underestimator
//! of the convex `f`, the master optimum `η*` is a lower bound on `min f`, and it
//! is **monotone non-decreasing** as cuts accumulate. Evaluating `f` at the master
//! minimiser yields a feasible point and hence an upper bound; the gap
//! `f(x_best) − η*` shrinks to zero. The master LP is solved with a self-contained
//! bounded-variable simplex tailored to this constraint structure.

use crate::error::{CvxError, CvxResult};

/// Termination status of [`kelley_cutting_plane`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuttingPlaneStatus {
    /// The optimality gap fell below tolerance.
    Converged,
    /// The iteration limit was reached before convergence.
    MaxIter,
}

/// Configuration for the cutting-plane method.
#[derive(Debug, Clone)]
pub struct CuttingPlaneConfig {
    /// Maximum number of cutting-plane iterations (`≥ 1`).
    pub max_iter: usize,
    /// Absolute tolerance on the gap `f(x_best) − η_master`.
    pub tol: f64,
    /// Maximum iterations for the inner master-LP simplex (`≥ 1`).
    pub master_max_iter: usize,
}

impl Default for CuttingPlaneConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1.0e-7,
            master_max_iter: 10_000,
        }
    }
}

/// Result of a cutting-plane solve.
#[derive(Debug, Clone)]
pub struct CuttingPlaneResult {
    /// Best feasible point found (the incumbent minimiser of `f`).
    pub x: Vec<f64>,
    /// Best (lowest) objective value `f(x)` attained.
    pub objective: f64,
    /// Final master lower bound `η*`.
    pub lower_bound: f64,
    /// Final optimality gap `objective − lower_bound`.
    pub gap: f64,
    /// History of master lower bounds, one per iteration (monotone
    /// non-decreasing).
    pub lower_bound_history: Vec<f64>,
    /// History of best upper bounds, one per iteration (monotone
    /// non-increasing).
    pub upper_bound_history: Vec<f64>,
    /// Number of cutting-plane iterations performed.
    pub iterations: usize,
    /// Number of cuts in the final model.
    pub cut_count: usize,
    /// Termination status.
    pub status: CuttingPlaneStatus,
}

/// A supporting cut `f(xₖ) + gₖᵀ (x − xₖ) ≤ η`, stored in the affine form
/// `η ≥ a + gᵀ x` with `a = f(xₖ) − gₖᵀ xₖ`.
struct Cut {
    intercept: f64,
    grad: Vec<f64>,
}

/// Minimise a convex `f` over the box `l ≤ x ≤ u` by Kelley's method.
///
/// * `f` — closure returning the (convex) objective value at a point.
/// * `grad_f` — closure returning `∇f` at a point (length `d`).
/// * `lower`, `upper` — length-`d` box bounds with `lower[i] ≤ upper[i]`.
/// * `cfg` — solver configuration.
///
/// The dimension `d` is taken from `lower.len()`.
///
/// # Errors
///
/// Returns [`CvxError::InvalidParameter`] for an empty or inconsistent box or a
/// zero iteration budget, [`CvxError::DimensionMismatch`] if a returned gradient
/// has the wrong length, and propagates any error raised by `f`/`grad_f` or the
/// master LP.
pub fn kelley_cutting_plane<F, G>(
    f: F,
    grad_f: G,
    lower: &[f64],
    upper: &[f64],
    cfg: &CuttingPlaneConfig,
) -> CvxResult<CuttingPlaneResult>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    let d = lower.len();
    if d == 0 {
        return Err(CvxError::InvalidParameter(
            "cutting-plane requires d ≥ 1".to_string(),
        ));
    }
    if upper.len() != d {
        return Err(CvxError::DimensionMismatch {
            a: upper.len(),
            b: d,
        });
    }
    if cfg.max_iter == 0 || cfg.master_max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "cutting-plane requires max_iter ≥ 1 and master_max_iter ≥ 1".to_string(),
        ));
    }
    for i in 0..d {
        if !(lower[i].is_finite() && upper[i].is_finite()) {
            return Err(CvxError::InvalidParameter(format!(
                "cutting-plane box bound {i} must be finite (Kelley needs a bounded master)"
            )));
        }
        if lower[i] > upper[i] {
            return Err(CvxError::InvalidParameter(format!(
                "cutting-plane lower[{i}]={} exceeds upper[{i}]={}",
                lower[i], upper[i]
            )));
        }
    }

    // Initial query point: the box centre.
    let mut x_query: Vec<f64> = (0..d).map(|i| 0.5 * (lower[i] + upper[i])).collect();

    let mut cuts: Vec<Cut> = Vec::new();
    let mut best_obj = f64::INFINITY;
    let mut best_x = x_query.clone();
    let mut lower_bound = f64::NEG_INFINITY;
    let mut lower_bound_history = Vec::new();
    let mut upper_bound_history = Vec::new();
    let mut iterations = 0usize;
    let mut status = CuttingPlaneStatus::MaxIter;

    for it in 0..cfg.max_iter {
        iterations = it + 1;

        // Evaluate f and the supporting cut at the current query point.
        let fval = f(&x_query)?;
        let g = grad_f(&x_query)?;
        if g.len() != d {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: d });
        }
        // Update incumbent (upper bound).
        if fval < best_obj {
            best_obj = fval;
            best_x = x_query.clone();
        }
        // New cut: intercept a = f(x) − gᵀ x.
        let mut intercept = fval;
        for i in 0..d {
            intercept -= g[i] * x_query[i];
        }
        cuts.push(Cut { intercept, grad: g });

        // Solve the master LP: minimise η s.t. η ≥ aₖ + gₖᵀ x, l ≤ x ≤ u.
        let (master_eta, master_x) = solve_master(&cuts, lower, upper, cfg.master_max_iter)?;
        // The master optimum is a valid lower bound; enforce monotonicity
        // explicitly to absorb tiny LP round-off.
        if master_eta > lower_bound {
            lower_bound = master_eta;
        }

        lower_bound_history.push(lower_bound);
        upper_bound_history.push(best_obj);

        let gap = best_obj - lower_bound;
        if gap.abs() <= cfg.tol {
            status = CuttingPlaneStatus::Converged;
            // Evaluate the master point once more so the incumbent reflects it.
            let fmx = f(&master_x)?;
            if fmx < best_obj {
                best_obj = fmx;
                best_x = master_x.clone();
                if let Some(last) = upper_bound_history.last_mut() {
                    *last = best_obj;
                }
            }
            break;
        }

        // Next query point is the master minimiser.
        x_query = master_x;
    }

    let gap = best_obj - lower_bound;
    Ok(CuttingPlaneResult {
        x: best_x,
        objective: best_obj,
        lower_bound,
        gap,
        lower_bound_history,
        upper_bound_history,
        iterations,
        cut_count: cuts.len(),
        status,
    })
}

/// Solve the master LP `min η  s.t.  η ≥ aₖ + gₖᵀ x, l ≤ x ≤ u`.
///
/// Returns `(η*, x*)`. The free variable `η` is split into `η = η⁺ − η⁻` and the
/// box variables are shifted to the origin, yielding a standard-form LP solved by
/// a bounded-variable two-phase simplex.
fn solve_master(
    cuts: &[Cut],
    lower: &[f64],
    upper: &[f64],
    max_iter: usize,
) -> CvxResult<(f64, Vec<f64>)> {
    let d = lower.len();
    let m = cuts.len();
    if m == 0 {
        // No cuts: η unbounded below; return the box centre with −∞ guarded.
        let centre: Vec<f64> = (0..d).map(|i| 0.5 * (lower[i] + upper[i])).collect();
        return Ok((f64::NEG_INFINITY, centre));
    }

    // Decision vector ordering for the simplex (all variables ≥ 0):
    //   y_i = x_i − l_i ∈ [0, w_i]    (i = 0..d), w_i = u_i − l_i
    //   η̂ = η − η_lo ≥ 0  with a finite lower bound η_lo on η over the box.
    //
    // η is bounded below over the box: each cut value aₖ + gₖᵀx attains its
    // minimum over [l, u] at the termwise extreme of gₖᵀx, so
    //   η_lo = min_k ( aₖ + Σ_i min(gₖᵢ l_i, gₖᵢ u_i) ).
    // Since η ≥ max_k(aₖ + gₖᵀx) ≥ η_lo for every feasible x, the shift η̂ ≥ 0 is
    // valid and removes the free-variable splitting (and its unbounded ray).
    //
    // Constraint k:  η ≥ aₖ + gₖᵀ x   ⇔   η̂ − gₖᵀ y − sₖ = bₖ,
    //   bₖ = aₖ + gₖᵀ l − η_lo ≥ 0,  surplus sₖ ≥ 0.
    // Upper bounds on y_i are extra rows  y_i + t_i = w_i  (t_i ≥ 0).
    //
    // Big-M single-phase bounded simplex over variables
    //   [ y(0..d), eta_hat, s(0..m), t(0..d) ] with artificials per row.

    // Valid finite lower bound on η over the box.
    let mut eta_lo = f64::INFINITY;
    for cut in cuts {
        let mut val = cut.intercept;
        for i in 0..d {
            val += (cut.grad[i] * lower[i]).min(cut.grad[i] * upper[i]);
        }
        if val < eta_lo {
            eta_lo = val;
        }
    }

    // Build a dense standard-form LP: minimise cᵀz s.t. A z = rhs, z ≥ 0.
    // Rows: m cut rows + d upper-bound rows.
    let n_struct = d + 1 + m + d; // y, eta_hat, surplus, slack(t)
    let rows = m + d;

    // Column index helpers.
    let idx_y = 0usize;
    let idx_eta = d;
    let idx_s = d + 1;
    let idx_t = d + 1 + m;

    // Right-hand side and the structural matrix (artificials appended later).
    let mut rhs = vec![0.0_f64; rows];
    let total_cols = n_struct + rows;
    let mut a = vec![0.0_f64; rows * total_cols]; // includes artificial block

    for (k, cut) in cuts.iter().enumerate() {
        // bₖ = aₖ + gₖᵀ l − η_lo (≥ 0 by construction).
        let mut bk = cut.intercept - eta_lo;
        for (gi, li) in cut.grad.iter().zip(lower.iter()) {
            bk += gi * li;
        }
        // Row k: η̂ − gₖᵀ y − sₖ = bₖ.
        let row = k * total_cols;
        a[row + idx_eta] = 1.0;
        for (i, gi) in cut.grad.iter().enumerate() {
            a[row + idx_y + i] = -gi;
        }
        a[row + idx_s + k] = -1.0;
        rhs[k] = bk;
    }
    // Upper-bound rows: y_i + t_i = w_i.
    for (i, (li, ui)) in lower.iter().zip(upper.iter()).enumerate() {
        let r = m + i;
        let row = r * total_cols;
        a[row + idx_y + i] = 1.0;
        a[row + idx_t + i] = 1.0;
        rhs[r] = ui - li;
    }

    // Normalise each row so its rhs is ≥ 0 (so an artificial = rhs is feasible).
    for (r, rhs_r) in rhs.iter_mut().enumerate() {
        if *rhs_r < 0.0 {
            *rhs_r = -*rhs_r;
            let row = r * total_cols;
            for j in 0..total_cols {
                a[row + j] = -a[row + j];
            }
        }
    }

    // Append one artificial per row (identity block) and set up the Big-M cost.
    // Objective: minimise η = η_lo + η̂  ⇒ (up to the constant η_lo) minimise η̂.
    let big_m = 1.0e7
        * (1.0
            + cuts
                .iter()
                .map(|c| c.intercept.abs() + c.grad.iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max));
    let mut cost = vec![0.0_f64; total_cols];
    cost[idx_eta] = 1.0;
    for r in 0..rows {
        let art_col = n_struct + r;
        let row = r * total_cols;
        a[row + art_col] = 1.0;
        cost[art_col] = big_m;
    }

    // Initial basis: the artificials.
    let basis: Vec<usize> = (0..rows).map(|r| n_struct + r).collect();

    let (z, _obj, final_basis) =
        bounded_simplex(&a, rows, total_cols, &rhs, &cost, &basis, max_iter)?;

    // Recover η and x (undo the η_lo shift).
    let eta = eta_lo + z[idx_eta];
    let mut x = vec![0.0_f64; d];
    for i in 0..d {
        x[i] = lower[i] + z[idx_y + i];
        // Clamp to the box to remove tiny LP excursions.
        if x[i] < lower[i] {
            x[i] = lower[i];
        }
        if x[i] > upper[i] {
            x[i] = upper[i];
        }
    }

    // If an artificial remains basic at a positive level, the master is
    // infeasible — which cannot happen for a well-posed bounded box, but guard
    // anyway.
    for (pos, &col) in final_basis.iter().enumerate() {
        if col >= n_struct && z.get(col).copied().unwrap_or(0.0) > 1.0e-6 {
            return Err(CvxError::Infeasible(format!(
                "cutting-plane master LP infeasible (artificial basic at row {pos})"
            )));
        }
    }

    Ok((eta, x))
}

/// Phase-style dense simplex (Bland's rule) for `min cᵀz s.t. A z = rhs, z ≥ 0`.
///
/// `a` is row-major `rows × cols` (with artificial columns already present),
/// `basis` is the starting basis (one column per row, `z_B = B⁻¹ rhs ≥ 0`).
/// Returns `(z, objective, basis)`.
fn bounded_simplex(
    a: &[f64],
    rows: usize,
    cols: usize,
    rhs: &[f64],
    cost: &[f64],
    basis: &[usize],
    max_iter: usize,
) -> CvxResult<(Vec<f64>, f64, Vec<usize>)> {
    if a.len() != rows * cols {
        return Err(CvxError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![a.len()],
        });
    }
    // Dense tableau in canonical form. Build [A | rhs] and pivot.
    let width = cols + 1;
    let mut tab = vec![0.0_f64; rows * width];
    for r in 0..rows {
        for j in 0..cols {
            tab[r * width + j] = a[r * cols + j];
        }
        tab[r * width + cols] = rhs[r];
    }
    let mut basis = basis.to_vec();

    // Reduce the tableau so each basic column is a unit vector.
    for (r, &bcol) in basis.iter().enumerate() {
        normalise_pivot(&mut tab, rows, width, r, bcol)?;
    }

    for _ in 0..max_iter {
        // Compute reduced costs c̄_j = c_j − c_B B⁻¹ A_j using the tableau.
        // With the tableau in canonical form, c_B·(column j of tableau) gives
        // the implicit z_j contribution. Reduced cost:
        //   c̄_j = c_j − Σ_r c_{basis[r]} · tab[r, j].
        let mut entering: Option<usize> = None;
        for j in 0..cols {
            if basis.contains(&j) {
                continue;
            }
            let mut cbar = cost[j];
            for r in 0..rows {
                cbar -= cost[basis[r]] * tab[r * width + j];
            }
            if cbar < -1.0e-9 {
                entering = Some(j);
                break; // Bland's rule: lowest index.
            }
        }
        let entering = match entering {
            Some(j) => j,
            None => break, // optimal
        };

        // Ratio test: choose leaving row with min rhs / tab[r, entering] over
        // positive entries.
        let mut leaving: Option<usize> = None;
        let mut best_ratio = f64::INFINITY;
        for r in 0..rows {
            let col_val = tab[r * width + entering];
            if col_val > 1.0e-12 {
                let ratio = tab[r * width + cols] / col_val;
                if ratio < best_ratio - 1.0e-12
                    || (ratio < best_ratio + 1.0e-12 && {
                        // Bland tie-break: smallest basis index.
                        match leaving {
                            Some(lr) => basis[r] < basis[lr],
                            None => true,
                        }
                    })
                {
                    best_ratio = ratio;
                    leaving = Some(r);
                }
            }
        }
        let leaving = match leaving {
            Some(r) => r,
            None => {
                return Err(CvxError::Unbounded(
                    "master LP unbounded during simplex".to_string(),
                ));
            }
        };

        // Pivot: make `entering` basic in row `leaving`.
        normalise_pivot(&mut tab, rows, width, leaving, entering)?;
        basis[leaving] = entering;
    }

    // Read off the solution.
    let mut z = vec![0.0_f64; cols];
    for (r, &bcol) in basis.iter().enumerate() {
        z[bcol] = tab[r * width + cols];
    }
    let objective: f64 = z.iter().zip(cost.iter()).map(|(zi, ci)| zi * ci).sum();
    Ok((z, objective, basis))
}

/// Scale row `r` so `tab[r, col] = 1`, then eliminate `col` from every other row.
fn normalise_pivot(
    tab: &mut [f64],
    rows: usize,
    width: usize,
    r: usize,
    col: usize,
) -> CvxResult<()> {
    let pivot = tab[r * width + col];
    if pivot.abs() < 1.0e-12 {
        return Err(CvxError::SingularMatrix(format!(
            "simplex pivot {pivot} too small at row {r}, col {col}"
        )));
    }
    let inv = 1.0 / pivot;
    for j in 0..width {
        tab[r * width + j] *= inv;
    }
    for rr in 0..rows {
        if rr == r {
            continue;
        }
        let factor = tab[rr * width + col];
        if factor != 0.0 {
            for j in 0..width {
                let v = tab[r * width + j];
                tab[rr * width + j] -= factor * v;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CuttingPlaneConfig {
        CuttingPlaneConfig {
            max_iter: 300,
            tol: 1.0e-6,
            master_max_iter: 20_000,
        }
    }

    #[test]
    fn quadratic_2d_converges_to_minimiser() {
        // f(x) = ‖x − a‖², a = (1, −2), box [−5, 5]². Minimiser at a.
        let a = [1.0_f64, -2.0];
        let f = move |x: &[f64]| -> CvxResult<f64> {
            Ok((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2))
        };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])])
        };
        let lower = [-5.0, -5.0];
        let upper = [5.0, 5.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        assert_eq!(res.status, CuttingPlaneStatus::Converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] + 2.0).abs() < 1.0e-3, "x1 = {}", res.x[1]);
        assert!(res.objective.abs() < 1.0e-3, "obj = {}", res.objective);
    }

    #[test]
    fn lower_bound_is_monotone_non_decreasing() {
        let a = [0.5_f64, 0.5];
        let f = move |x: &[f64]| -> CvxResult<f64> {
            Ok((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2))
        };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])])
        };
        let lower = [-2.0, -2.0];
        let upper = [2.0, 2.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        for w in res.lower_bound_history.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-9,
                "lower bound decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn upper_bound_is_monotone_non_increasing() {
        let a = [-1.0_f64, 1.5];
        let f = move |x: &[f64]| -> CvxResult<f64> {
            Ok((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2))
        };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])])
        };
        let lower = [-3.0, -3.0];
        let upper = [3.0, 3.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        for w in res.upper_bound_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1.0e-9,
                "upper bound increased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn gap_converges_and_cuts_grow() {
        let a = [2.0_f64];
        let f = move |x: &[f64]| -> CvxResult<f64> { Ok((x[0] - a[0]).powi(2)) };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![2.0 * (x[0] - a[0])]) };
        let lower = [-10.0];
        let upper = [10.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        assert_eq!(res.status, CuttingPlaneStatus::Converged);
        assert!(res.gap.abs() < 1.0e-5, "gap = {}", res.gap);
        // At least two cuts are required to box in a 1-D quadratic.
        assert!(res.cut_count >= 2, "cut_count = {}", res.cut_count);
    }

    #[test]
    fn one_dimensional_absolute_value() {
        // f(x) = |x − 3| over [0, 10]. Minimiser at 3, value 0.
        let f = |x: &[f64]| -> CvxResult<f64> { Ok((x[0] - 3.0).abs()) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            let s = if x[0] >= 3.0 { 1.0 } else { -1.0 };
            Ok(vec![s])
        };
        let lower = [0.0];
        let upper = [10.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        assert_eq!(res.status, CuttingPlaneStatus::Converged);
        assert!((res.x[0] - 3.0).abs() < 1.0e-3, "x = {}", res.x[0]);
        assert!(res.objective.abs() < 1.0e-3, "obj = {}", res.objective);
    }

    #[test]
    fn constrained_optimum_on_box_face() {
        // f(x) = ‖x − a‖² with a = (5, 5) but box [−1, 1]². Optimum at the
        // corner (1, 1) on the box face; f = (1−5)² + (1−5)² = 32.
        let a = [5.0_f64, 5.0];
        let f = move |x: &[f64]| -> CvxResult<f64> {
            Ok((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2))
        };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])])
        };
        let lower = [-1.0, -1.0];
        let upper = [1.0, 1.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        assert_eq!(res.status, CuttingPlaneStatus::Converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1.0e-3, "x1 = {}", res.x[1]);
        assert!(
            (res.objective - 32.0).abs() < 1.0e-2,
            "obj = {}",
            res.objective
        );
    }

    #[test]
    fn master_lower_bound_below_optimum() {
        // The lower-bound history must never exceed the true optimum f* = 0.
        let a = [1.0_f64, 1.0];
        let f = move |x: &[f64]| -> CvxResult<f64> {
            Ok((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2))
        };
        let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])])
        };
        let lower = [-2.0, -2.0];
        let upper = [2.0, 2.0];
        let res = kelley_cutting_plane(f, g, &lower, &upper, &cfg()).expect("solve");
        for &lb in &res.lower_bound_history {
            assert!(lb <= 1.0e-6, "lower bound {lb} exceeded optimum 0");
        }
    }

    #[test]
    fn err_empty_box() {
        let f = |_: &[f64]| -> CvxResult<f64> { Ok(0.0) };
        let g = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![]) };
        assert!(matches!(
            kelley_cutting_plane(f, g, &[], &[], &cfg()),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_inconsistent_bounds() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x[0]) };
        let g = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![1.0]) };
        assert!(matches!(
            kelley_cutting_plane(f, g, &[2.0], &[1.0], &cfg()),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_infinite_bounds() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x[0]) };
        let g = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![1.0]) };
        assert!(matches!(
            kelley_cutting_plane(f, g, &[f64::NEG_INFINITY], &[1.0], &cfg()),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn master_simplex_solves_tiny_lp() {
        // Sanity check the inner master directly: a single cut η ≥ 2 + 0·x over
        // box x ∈ [0, 1] → η* = 2.
        let cuts = vec![Cut {
            intercept: 2.0,
            grad: vec![0.0],
        }];
        let (eta, x) = solve_master(&cuts, &[0.0], &[1.0], 10_000).expect("master");
        assert!((eta - 2.0).abs() < 1.0e-6, "eta = {eta}");
        assert!((0.0..=1.0).contains(&x[0]));
    }
}
