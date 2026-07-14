//! Smoothed-aggregation Algebraic Multigrid (AMG) solver for general
//! symmetric positive-definite matrices stored as dense `n × n` arrays.
//!
//! # Algorithm overview
//!
//! ## Setup phase ([`AmgSolver::setup`])
//!
//! 1. Store the fine-level matrix `A` as level 0.
//! 2. Repeat for each subsequent level:
//!    - **Greedy aggregation**: nodes are partitioned into aggregates based on a
//!      strength-of-connection criterion `|A[i,j]| > threshold * max_k |A[i,k]|`.
//!      Each aggregate becomes a single coarse-grid degree of freedom.
//!    - **Tentative prolongation**: `P_tent[i, agg(i)] = 1` (indicator).
//!    - **Restriction**: `R = P^T`.
//!    - **Galerkin coarse operator**: `A_c = R * A * P`.
//!    - Stop when coarse size ≤ 4 or no coarsening occurs.
//!
//! ## Solve phase ([`AmgSolver::solve`])
//!
//! Applies a V-cycle AMG iteration until the relative residual
//! `||b - Ax|| / ||b||` falls below `tol` or `max_outer_iter` is reached.
//!
//! Dense matrix-vector products: `(Ax)[i] = Σ_j A[i*n + j] * x[j]`.

use crate::error::{PdeError, PdeResult};

/// Configuration for the AMG solver.
#[derive(Debug, Clone)]
pub struct AmgConfig {
    /// Maximum number of multigrid levels (including the finest).
    pub max_levels: usize,
    /// Strength-of-connection threshold for aggregation (e.g. 0.25).
    pub agg_threshold: f64,
    /// Number of pre- and post-smoothing sweeps per level in the V-cycle.
    pub nu_smooth: usize,
    /// Relative residual tolerance for the outer iteration.
    pub tol: f64,
    /// Maximum number of outer V-cycle iterations.
    pub max_outer_iter: usize,
}

/// One level of the AMG hierarchy.
#[derive(Debug, Clone)]
pub struct AmgLevel {
    /// Number of unknowns at this level.
    pub n: usize,
    /// Matrix `A` at this level, stored row-major `[n × n]`.
    pub a: Vec<f64>,
    /// Restriction operator `R`, row-major `[n_coarse × n]`.
    pub r: Vec<f64>,
    /// Prolongation operator `P`, row-major `[n × n_coarse]`.
    pub p: Vec<f64>,
    /// Number of coarse nodes at the next level.
    pub n_coarse: usize,
}

/// AMG solver holding the full multigrid hierarchy.
#[derive(Debug, Clone)]
pub struct AmgSolver {
    levels: Vec<AmgLevel>,
    config: AmgConfig,
}

impl AmgSolver {
    /// Build the AMG hierarchy from the fine-level matrix `A` (row-major, `n × n`).
    ///
    /// # Errors
    ///
    /// Returns `PdeError::InvalidGrid` if `n == 0`, or
    /// `PdeError::ShapeMismatch` if `a.len() != n * n`.
    pub fn setup(a: &[f64], n: usize, config: AmgConfig) -> PdeResult<Self> {
        if n == 0 {
            return Err(PdeError::InvalidGrid(
                "amg: matrix dimension n must be > 0".into(),
            ));
        }
        if a.len() != n * n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n * n],
                got: vec![a.len()],
            });
        }
        let mut levels: Vec<AmgLevel> = Vec::with_capacity(config.max_levels);
        let mut current_n = n;
        let mut current_a = a.to_vec();

        for _ in 1..config.max_levels {
            if current_n <= 4 {
                break;
            }
            // Build aggregates for the current matrix
            let agg = build_aggregates(&current_a, current_n, config.agg_threshold);
            let n_agg = agg.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
            if n_agg == 0 || n_agg >= current_n {
                // No coarsening occurred — stop.
                break;
            }
            // Build tentative prolongation P [current_n × n_agg]
            let p_mat = build_prolongation(&agg, current_n, n_agg);
            // Restriction R = P^T  [n_agg × current_n]
            let r_mat = transpose(&p_mat, current_n, n_agg);
            // Galerkin coarse operator A_c = R * A * P  [n_agg × n_agg]
            let a_coarse = galerkin_product(&r_mat, &current_a, &p_mat, n_agg, current_n);

            // Push the current level now that its R/P/n_coarse are fully known
            // — no need to look it back up afterwards to patch it in.
            levels.push(AmgLevel {
                n: current_n,
                a: current_a,
                r: r_mat,
                p: p_mat,
                n_coarse: n_agg,
            });

            current_n = n_agg;
            current_a = a_coarse;
        }

        // Push the coarsest level reached: nothing coarser exists below it, so
        // its R and P stay empty (see `v_cycle`'s `n_coarse == 0` terminal check).
        levels.push(AmgLevel {
            n: current_n,
            a: current_a,
            r: Vec::new(),
            p: Vec::new(),
            n_coarse: 0,
        });

        Ok(Self { levels, config })
    }

    /// Number of levels in the multigrid hierarchy.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Solve `A x = b` using the AMG V-cycle.
    ///
    /// Returns the approximate solution vector of length `n` (level 0).
    ///
    /// # Errors
    ///
    /// Returns `PdeError::InvalidGrid` if `b.len()` does not match `n`.
    pub fn solve(&self, b: &[f64]) -> PdeResult<Vec<f64>> {
        let n = self.levels[0].n;
        if b.len() != n {
            return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
        }
        let mut x = vec![0.0_f64; n];
        let b_norm = vec_norm(b);
        let tol = if b_norm < 1e-300 {
            self.config.tol
        } else {
            self.config.tol * b_norm
        };
        for _ in 0..self.config.max_outer_iter {
            v_cycle(&mut x, b, &self.levels, 0, self.config.nu_smooth)?;
            let res = self.residual(&x, b);
            if res <= tol {
                break;
            }
        }
        Ok(x)
    }

    /// Compute `||b - A x||_2` using the fine-level matrix `A = levels[0].a`.
    pub fn residual(&self, x: &[f64], b: &[f64]) -> f64 {
        let n = self.levels[0].n;
        if x.len() != n || b.len() != n {
            return f64::INFINITY;
        }
        let a = &self.levels[0].a;
        let ax = matvec(a, x, n);
        let mut sum = 0.0_f64;
        for i in 0..n {
            let r = b[i] - ax[i];
            sum += r * r;
        }
        sum.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Internal dense linear-algebra helpers
// ---------------------------------------------------------------------------

/// Dense matrix-vector product: `(A x)[i] = Σ_j A[i*n + j] * x[j]`.
fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Dense matrix-matrix product: `C = A * B` where `A` is `[m × k]`, `B` is `[k × p]`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * p];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i * p + j] += a_il * b[l * p + j];
            }
        }
    }
    c
}

/// Transpose a matrix `A [rows × cols]` to produce `A^T [cols × rows]`.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut at = vec![0.0_f64; cols * rows];
    for i in 0..rows {
        for j in 0..cols {
            at[j * rows + i] = a[i * cols + j];
        }
    }
    at
}

/// Compute the Galerkin triple product `A_c = R * A * P`.
///
/// - `r`: `[n_c × n]`
/// - `a`: `[n × n]`
/// - `p`: `[n × n_c]`
/// - Returns: `[n_c × n_c]`
fn galerkin_product(r: &[f64], a: &[f64], p: &[f64], n_c: usize, n: usize) -> Vec<f64> {
    // tmp = A * P  [n × n_c]
    let ap = matmul(a, p, n, n, n_c);
    // A_c = R * (A*P)  [n_c × n_c]
    matmul(r, &ap, n_c, n, n_c)
}

/// L2 norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Greedy aggregation based on strength of connection.
///
/// Returns `agg[i]` = aggregate index (0-based) for node `i`.
fn build_aggregates(a: &[f64], n: usize, threshold: f64) -> Vec<usize> {
    let mut agg = vec![usize::MAX; n]; // usize::MAX means unassigned
    let mut n_agg = 0_usize;

    // Precompute per-row max |off-diagonal|
    let mut row_max = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            if j != i {
                let v = a[i * n + j].abs();
                if v > row_max[i] {
                    row_max[i] = v;
                }
            }
        }
    }

    // Greedy pass: for each unassigned node create a new aggregate with
    // itself and its strongly-connected unassigned neighbours.
    for i in 0..n {
        if agg[i] != usize::MAX {
            continue;
        }
        agg[i] = n_agg;
        for j in 0..n {
            if j != i && agg[j] == usize::MAX {
                let strength = a[i * n + j].abs();
                // Strong connection if |A[i,j]| > threshold * max_k |A[i,k]|
                if strength > threshold * row_max[i] {
                    agg[j] = n_agg;
                }
            }
        }
        n_agg += 1;
    }
    agg
}

/// Build the tentative prolongation matrix `P [n × n_agg]`.
///
/// `P[i, agg[i]] = 1`, all other entries 0.
fn build_prolongation(agg: &[usize], n: usize, n_agg: usize) -> Vec<f64> {
    let mut p = vec![0.0_f64; n * n_agg];
    for (i, &a_i) in agg.iter().enumerate().take(n) {
        if a_i < n_agg {
            p[i * n_agg + a_i] = 1.0;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Smoothing for general dense matrix
// ---------------------------------------------------------------------------

/// Weighted-Jacobi smoother for the general dense system `A x = b`.
///
/// `x_new[i] = x[i] + omega * (b[i] - (Ax)[i]) / A[i,i]`
fn dense_jacobi_smooth(x: &mut [f64], a: &[f64], b: &[f64], n: usize, omega: f64, sweeps: usize) {
    for _ in 0..sweeps {
        let ax = matvec(a, x, n);
        for i in 0..n {
            let diag = a[i * n + i];
            if diag.abs() > 1e-300 {
                x[i] += omega * (b[i] - ax[i]) / diag;
            }
        }
    }
}

/// Direct solve on the coarsest grid using many Jacobi iterations.
fn coarsest_solve(x: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    dense_jacobi_smooth(x, a, b, n, 0.5, 500);
}

// ---------------------------------------------------------------------------
// V-cycle
// ---------------------------------------------------------------------------

/// Apply one AMG V-cycle starting from level `lv`.
///
/// # Errors
///
/// Returns `PdeError::DimensionMismatch` on internal shape mismatches (should
/// not occur for a correctly built hierarchy).
fn v_cycle(
    x: &mut [f64],
    b: &[f64],
    levels: &[AmgLevel],
    lv: usize,
    nu_smooth: usize,
) -> PdeResult<()> {
    let n = levels[lv].n;
    let a = &levels[lv].a;

    // Coarsest level: direct solve.
    if lv + 1 >= levels.len() || levels[lv].n_coarse == 0 {
        coarsest_solve(x, a, b, n);
        return Ok(());
    }

    // Pre-smooth
    dense_jacobi_smooth(x, a, b, n, 0.5, nu_smooth);

    // Compute residual r = b - A x
    let ax = matvec(a, x, n);
    let r_fine: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();

    // Restrict residual to coarse grid: r_c = R * r_fine
    let r_ref = &levels[lv].r;
    let n_c = levels[lv].n_coarse;
    if r_ref.len() != n_c * n {
        return Err(PdeError::DimensionMismatch {
            a: r_ref.len(),
            b: n_c * n,
        });
    }
    let r_coarse = matvec_rect(r_ref, &r_fine, n_c, n);

    // Recurse on coarse defect equation
    let mut e_coarse = vec![0.0_f64; n_c];
    v_cycle(&mut e_coarse, &r_coarse, levels, lv + 1, nu_smooth)?;

    // Prolongate correction: e_fine = P * e_coarse
    let p_ref = &levels[lv].p;
    if p_ref.len() != n * n_c {
        return Err(PdeError::DimensionMismatch {
            a: p_ref.len(),
            b: n * n_c,
        });
    }
    let e_fine = matvec_rect(p_ref, &e_coarse, n, n_c);

    // Correct x
    for i in 0..n {
        x[i] += e_fine[i];
    }

    // Post-smooth
    dense_jacobi_smooth(x, a, b, n, 0.5, nu_smooth);
    Ok(())
}

/// Matrix-vector product for a non-square `[m × k]` matrix stored row-major.
fn matvec_rect(a: &[f64], x: &[f64], m: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..k {
            s += a[i * k + j] * x[j];
        }
        y[i] = s;
    }
    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the 1D Laplacian matrix `-Δ` with homogeneous Dirichlet BCs,
    /// stored as a dense `n × n` row-major array.  The mesh spacing is
    /// `h = 1 / (n - 1)` and the operator is `(2u_i - u_{i-1} - u_{i+1}) / h²`.
    fn laplacian_1d(n: usize) -> Vec<f64> {
        assert!(n >= 2, "n must be >= 2");
        let h = 1.0 / (n - 1) as f64;
        let ih2 = 1.0 / (h * h);
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0 * ih2;
            if i > 0 {
                a[i * n + (i - 1)] = -ih2;
            }
            if i + 1 < n {
                a[i * n + (i + 1)] = -ih2;
            }
        }
        // Fix boundary rows to identity (Dirichlet)
        for j in 0..n {
            a[j] = if j == 0 { 1.0 } else { 0.0 };
            a[(n - 1) * n + j] = if j == n - 1 { 1.0 } else { 0.0 };
        }
        a
    }

    fn default_config() -> AmgConfig {
        AmgConfig {
            max_levels: 4,
            agg_threshold: 0.25,
            nu_smooth: 3,
            tol: 1e-6,
            max_outer_iter: 50,
        }
    }

    #[test]
    fn setup_builds_hierarchy_n16() {
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        assert!(
            solver.n_levels() >= 2,
            "expected >= 2 levels, got {}",
            solver.n_levels()
        );
    }

    #[test]
    fn solve_1d_laplacian_n8() {
        let n = 8;
        let a = laplacian_1d(n);
        let h = 1.0 / (n - 1) as f64;
        let ih2 = 1.0 / (h * h);
        // RHS: f = 2/h² at interior nodes (so exact u = x(1-x))
        let mut b = vec![2.0 * ih2; n];
        b[0] = 0.0; // Dirichlet
        b[n - 1] = 0.0;
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        let x = solver.solve(&b).expect("solve ok");
        let res = solver.residual(&x, &b);
        let b_norm = vec_norm(&b);
        assert!(
            res / b_norm.max(1e-300) < 1e-4,
            "relative residual {:.2e} not < 1e-4",
            res / b_norm.max(1e-300)
        );
    }

    #[test]
    fn solve_1d_laplacian_n16() {
        let n = 16;
        let a = laplacian_1d(n);
        let h = 1.0 / (n - 1) as f64;
        let ih2 = 1.0 / (h * h);
        let mut b = vec![2.0 * ih2; n];
        b[0] = 0.0;
        b[n - 1] = 0.0;
        let cfg = AmgConfig {
            max_levels: 4,
            agg_threshold: 0.25,
            nu_smooth: 5,
            tol: 1e-6,
            max_outer_iter: 100,
        };
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        let x = solver.solve(&b).expect("solve ok");
        let res = solver.residual(&x, &b);
        let b_norm = vec_norm(&b);
        assert!(
            res / b_norm.max(1e-300) < 1e-4,
            "relative residual {:.2e} not < 1e-4",
            res / b_norm.max(1e-300)
        );
    }

    #[test]
    fn n_levels_bounded() {
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let max_lv = cfg.max_levels;
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        assert!(
            solver.n_levels() <= max_lv,
            "n_levels {} > max_levels {}",
            solver.n_levels(),
            max_lv
        );
    }

    #[test]
    fn residual_finite() {
        let n = 8;
        let a = laplacian_1d(n);
        let h = 1.0 / (n - 1) as f64;
        let ih2 = 1.0 / (h * h);
        let mut b = vec![2.0 * ih2; n];
        b[0] = 0.0;
        b[n - 1] = 0.0;
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        let x = solver.solve(&b).expect("solve ok");
        let res = solver.residual(&x, &b);
        assert!(res.is_finite() && res >= 0.0, "residual = {res}");
    }

    #[test]
    fn coarsest_level_small() {
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        let last = solver.levels.last().expect("at least one level");
        assert!(
            last.n <= 8,
            "coarsest level has n={}, expected <= 8",
            last.n
        );
    }

    #[test]
    fn restriction_shape() {
        // Level 0 stores R and P for the transition to level 1.
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        if solver.n_levels() >= 2 {
            let lv0 = &solver.levels[0];
            let n_c = lv0.n_coarse;
            assert_eq!(
                lv0.p.len(),
                lv0.n * n_c,
                "P shape mismatch: {} != {}",
                lv0.p.len(),
                lv0.n * n_c
            );
            assert_eq!(
                lv0.r.len(),
                n_c * lv0.n,
                "R shape mismatch: {} != {}",
                lv0.r.len(),
                n_c * lv0.n
            );
        }
    }

    #[test]
    fn prolongation_col_sums() {
        // Each column of P (prolongation) should sum to the number of nodes
        // assigned to that aggregate. With our binary 0/1 P, each row sums
        // to 1.0 (partition of unity for the rows, i.e. each fine node maps
        // to exactly one coarse node).
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        if solver.n_levels() >= 2 {
            let lv0 = &solver.levels[0];
            let n_c = lv0.n_coarse;
            // Check row sums of P equal 1.0 (each fine node in exactly one agg).
            for i in 0..lv0.n {
                let row_sum: f64 = (0..n_c).map(|j| lv0.p[i * n_c + j]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-12,
                    "P row {i} sums to {row_sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn galerkin_triple_check() {
        // Verify that levels[1].a == R * levels[0].a * P (Galerkin condition).
        let n = 16;
        let a = laplacian_1d(n);
        let cfg = default_config();
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup ok");
        if solver.n_levels() >= 2 {
            let lv0 = &solver.levels[0];
            let lv1 = &solver.levels[1];
            let n_c = lv0.n_coarse;
            let a_c = galerkin_product(&lv0.r, &lv0.a, &lv0.p, n_c, lv0.n);
            for (k, (&computed, &stored)) in a_c.iter().zip(lv1.a.iter()).enumerate() {
                assert!(
                    (computed - stored).abs() < 1e-9,
                    "A_c mismatch at [{},{}]: computed={computed} stored={stored}",
                    k / n_c,
                    k % n_c
                );
            }
        }
    }

    #[test]
    fn empty_input_error() {
        let cfg = default_config();
        let result = AmgSolver::setup(&[], 0, cfg);
        assert!(
            matches!(result, Err(PdeError::InvalidGrid(_))),
            "expected InvalidGrid for n=0"
        );
    }

    #[test]
    fn setup_n4_laplacian() {
        let n = 4;
        let a = laplacian_1d(n);
        let cfg = AmgConfig {
            max_levels: 2,
            agg_threshold: 0.25,
            nu_smooth: 3,
            tol: 1e-6,
            max_outer_iter: 50,
        };
        let solver = AmgSolver::setup(&a, n, cfg).expect("setup did not panic");
        assert!(solver.n_levels() >= 1);
    }
}
