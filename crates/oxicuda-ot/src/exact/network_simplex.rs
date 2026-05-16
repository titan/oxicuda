//! Network-simplex (transportation-problem) algorithm for exact OT.
//!
//! Implements a textbook transportation-simplex tailored to dense, balanced
//! transportation tableaux:
//!
//! 1. **Northwest-corner rule** to seed an initial basic-feasible solution
//!    with `m + n − 1` basic variables.
//! 2. **Dual potentials** `(u_i, v_j)` recovered from `c_ij = u_i + v_j` on
//!    the basis (one potential is anchored at zero to make the system
//!    determined).
//! 3. **Reduced costs** `c̄_ij = c_ij − u_i − v_j` evaluated on non-basic
//!    cells; the most-negative reduced cost selects the pivot
//!    (Bland's rule for ties).
//! 4. **Stepping-stone cycle** is detected on the basis (which forms a
//!    spanning tree of the transportation graph), the minimum mass on the
//!    `θ−` legs is shifted, and the basis is updated.
//!
//! The implementation is sufficient for problems with `m, n ≤ 64`; it is
//! quadratic in the basis size per pivot, which is fine at this scale and
//! avoids the bookkeeping of a tree-based implementation.

use crate::error::{OtError, OtResult};

/// Configuration for the network-simplex solver.
#[derive(Debug, Clone)]
pub struct NsConfig {
    /// Maximum number of pivot iterations.
    pub max_iter: usize,
}

impl Default for NsConfig {
    fn default() -> Self {
        Self { max_iter: 10_000 }
    }
}

/// Output of the network-simplex solver.
#[derive(Debug, Clone)]
pub struct NsResult {
    /// Transport plan, shape `[m × n]` row-major.
    pub plan: Vec<f32>,
    /// Transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Number of pivot iterations executed.
    pub iters: usize,
}

/// Validate inputs and ensure mass balance within `1e-4`.
fn validate(c: &[f32], a: &[f32], b: &[f32], m: usize, n: usize) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if c.len() != m * n || a.len() != m || b.len() != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    let mut sum_a = 0.0_f32;
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sum_a += ai;
    }
    let mut sum_b = 0.0_f32;
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sum_b += bj;
    }
    if (sum_a - sum_b).abs() > 1e-4 {
        return Err(OtError::MassImbalance { sum_a, sum_b });
    }
    Ok(())
}

/// Northwest-corner-rule initial basic feasible solution.
///
/// Fills cells `(0,0), (0,1), …, (m−1,n−1)` greedily, exhausting the
/// minimum of remaining row supply and column demand at each step. The
/// returned `basis` lists exactly `m + n − 1` cells (with degenerate
/// `0`-valued entries inserted to break degeneracy when required).
fn northwest_corner(
    plan: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) -> Vec<(usize, usize)> {
    let mut a_rem = a.to_vec();
    let mut b_rem = b.to_vec();
    let mut basis: Vec<(usize, usize)> = Vec::with_capacity(m + n - 1);
    let mut i = 0_usize;
    let mut j = 0_usize;
    while i < m && j < n {
        let amount = a_rem[i].min(b_rem[j]);
        plan[i * n + j] = amount;
        basis.push((i, j));
        a_rem[i] -= amount;
        b_rem[j] -= amount;
        let row_done = a_rem[i] <= 1e-9;
        let col_done = b_rem[j] <= 1e-9;
        if row_done && col_done {
            // Degenerate step: advance both, but insert a zero-cell if room.
            if i + 1 < m && j + 1 < n {
                i += 1;
                basis.push((i, j));
                j += 1;
            } else if i + 1 < m {
                i += 1;
            } else {
                j += 1;
            }
        } else if row_done {
            i += 1;
        } else {
            j += 1;
        }
    }
    while basis.len() < m + n - 1 {
        // Pad with the last cell if degeneracy under-counted.
        let last = basis.last().copied().unwrap_or((0, 0));
        basis.push(last);
    }
    basis
}

/// Solve the dual `c_ij = u_i + v_j` on the basis (anchoring `u_0 = 0`).
fn solve_potentials(
    basis: &[(usize, usize)],
    c: &[f32],
    m: usize,
    n: usize,
) -> OtResult<(Vec<f32>, Vec<f32>)> {
    let mut u = vec![f32::NAN; m];
    let mut v = vec![f32::NAN; n];
    u[0] = 0.0;
    let total = m + n;
    let mut iter = 0_usize;
    let max_passes = total * 2 + 8;
    while iter < max_passes {
        iter += 1;
        let mut changed = false;
        for &(i, j) in basis {
            let u_known = !u[i].is_nan();
            let v_known = !v[j].is_nan();
            let c_ij = c[i * n + j];
            if u_known && !v_known {
                v[j] = c_ij - u[i];
                changed = true;
            } else if v_known && !u_known {
                u[i] = c_ij - v[j];
                changed = true;
            }
        }
        let all_u = u.iter().all(|x| !x.is_nan());
        let all_v = v.iter().all(|x| !x.is_nan());
        if all_u && all_v {
            return Ok((u, v));
        }
        if !changed {
            // Anchor an unanchored row/col explicitly.
            if let Some((idx, _)) = u.iter().enumerate().find(|(_, x)| x.is_nan()) {
                u[idx] = 0.0;
            } else if let Some((idx, _)) = v.iter().enumerate().find(|(_, x)| x.is_nan()) {
                v[idx] = 0.0;
            } else {
                break;
            }
        }
    }
    if u.iter().any(|x| x.is_nan()) || v.iter().any(|x| x.is_nan()) {
        return Err(OtError::Internal {
            msg: "potentials did not propagate over basis".to_string(),
        });
    }
    Ok((u, v))
}

/// Locate the entering variable: the non-basic cell with most-negative reduced cost.
fn select_entering(
    basis_set: &[bool],
    c: &[f32],
    u: &[f32],
    v: &[f32],
    m: usize,
    n: usize,
) -> Option<(usize, usize, f32)> {
    let mut best: Option<(usize, usize, f32)> = None;
    for i in 0..m {
        for j in 0..n {
            if basis_set[i * n + j] {
                continue;
            }
            let reduced = c[i * n + j] - u[i] - v[j];
            if reduced < -1e-7 {
                match best {
                    None => best = Some((i, j, reduced)),
                    Some((_, _, b)) if reduced < b => best = Some((i, j, reduced)),
                    Some((bi, bj, b)) if (reduced - b).abs() <= 1e-9 && (i, j) < (bi, bj) => {
                        best = Some((i, j, reduced));
                    }
                    _ => {}
                }
            }
        }
    }
    best
}

/// DFS to find a closed cycle in the basis that includes the entering cell.
fn find_cycle(
    basis: &[(usize, usize)],
    enter: (usize, usize),
    m: usize,
    n: usize,
) -> Option<Vec<(usize, usize)>> {
    // Build adjacency: for each row, list cells; for each column, list cells.
    let mut by_row: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
    let mut by_col: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for &(i, j) in basis {
        by_row[i].push((i, j));
        by_col[j].push((i, j));
    }
    by_row[enter.0].push(enter);
    by_col[enter.1].push(enter);

    // DFS alternating row/column moves starting from `enter` looking for `enter`.
    // Path is even-length, alternating axes. We track visited cells per axis-state.
    let mut path: Vec<(usize, usize)> = vec![enter];
    let mut found: Option<Vec<(usize, usize)>> = None;
    fn dfs(
        path: &mut Vec<(usize, usize)>,
        target: (usize, usize),
        next_axis_row: bool,
        by_row: &[Vec<(usize, usize)>],
        by_col: &[Vec<(usize, usize)>],
        found: &mut Option<Vec<(usize, usize)>>,
    ) {
        if found.is_some() {
            return;
        }
        let cur = match path.last() {
            Some(&c) => c,
            None => return,
        };
        let candidates: &[(usize, usize)] = if next_axis_row {
            &by_row[cur.0]
        } else {
            &by_col[cur.1]
        };
        for &cell in candidates {
            if cell == cur {
                continue;
            }
            if cell == target && path.len() >= 3 && path.len() % 2 == 1 {
                let mut closed = path.clone();
                closed.push(cell);
                *found = Some(closed);
                return;
            }
            if path.contains(&cell) {
                continue;
            }
            path.push(cell);
            dfs(path, target, !next_axis_row, by_row, by_col, found);
            if found.is_some() {
                return;
            }
            path.pop();
        }
    }
    dfs(&mut path, enter, true, &by_row, &by_col, &mut found);
    if found.is_none() {
        // Try starting along columns first (allows alternative routes).
        path.clear();
        path.push(enter);
        dfs(&mut path, enter, false, &by_row, &by_col, &mut found);
    }
    found.map(|mut c| {
        // Drop trailing duplicate of entering cell.
        if c.len() >= 2 && c[0] == c[c.len() - 1] {
            c.pop();
        }
        c
    })
}

/// Run the network-simplex algorithm.
pub fn network_simplex(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &NsConfig,
) -> OtResult<NsResult> {
    validate(c, a, b, m, n)?;
    let mut plan = vec![0.0_f32; m * n];
    let mut basis = northwest_corner(&mut plan, a, b, m, n);
    let mut basis_set = vec![false; m * n];
    for &(i, j) in &basis {
        basis_set[i * n + j] = true;
    }

    let mut iters: usize;
    for it in 0..cfg.max_iter {
        iters = it + 1;
        let (u, v) = solve_potentials(&basis, c, m, n)?;
        let entering = select_entering(&basis_set, c, &u, &v, m, n);
        let (ei, ej, _) = match entering {
            Some(t) => t,
            None => {
                // Optimal.
                let cost = plan.iter().zip(c.iter()).map(|(&p, &cij)| p * cij).sum();
                return Ok(NsResult { plan, cost, iters });
            }
        };
        let cycle = match find_cycle(&basis, (ei, ej), m, n) {
            Some(c) => c,
            None => {
                return Err(OtError::Internal {
                    msg: "could not close cycle for entering variable".to_string(),
                });
            }
        };
        // Even-indexed positions in the cycle are `+`, odd are `−`.
        let mut theta = f32::INFINITY;
        let mut leaving = (usize::MAX, usize::MAX);
        for (k, &(i, j)) in cycle.iter().enumerate() {
            if k % 2 == 1 {
                let val = plan[i * n + j];
                if val < theta {
                    theta = val;
                    leaving = (i, j);
                }
            }
        }
        if !theta.is_finite() {
            return Err(OtError::Internal {
                msg: "no leaving variable identified".to_string(),
            });
        }
        for (k, &(i, j)) in cycle.iter().enumerate() {
            if k % 2 == 0 {
                plan[i * n + j] += theta;
            } else {
                plan[i * n + j] -= theta;
            }
        }
        // Update basis: remove `leaving`, add `entering`.
        if let Some(idx) = basis.iter().position(|&c| c == leaving) {
            basis.swap_remove(idx);
        }
        basis_set[leaving.0 * n + leaving.1] = false;
        basis.push((ei, ej));
        basis_set[ei * n + ej] = true;
    }
    Err(OtError::NotConverged {
        iter: cfg.max_iter,
        tol: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn diagonal_zero_cost_for_equal_marginals() {
        let m = 3;
        let n = 3;
        let c = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let res = network_simplex(&c, &a, &b, m, n, &NsConfig::default()).expect("ok");
        assert!(approx(res.cost, 0.0, 1e-5), "cost={} should be 0", res.cost);
    }

    #[test]
    fn marginals_satisfied() {
        let m = 3;
        let n = 2;
        let c = vec![1.0, 4.0, 2.0, 3.0, 5.0, 1.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.6_f32, 0.4];
        let res = network_simplex(&c, &a, &b, m, n, &NsConfig::default()).expect("ok");
        for (i, &ai) in a.iter().enumerate() {
            let row_sum: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row_sum, ai, 1e-4), "row {i} {row_sum} ≠ {ai}");
        }
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col_sum, bj, 1e-4));
        }
    }

    #[test]
    fn mass_imbalance_rejected() {
        let c = vec![1.0_f32; 4];
        let a = vec![0.6_f32, 0.4];
        let b = vec![0.5_f32, 0.4];
        let res = network_simplex(&c, &a, &b, 2, 2, &NsConfig::default());
        assert!(matches!(res, Err(OtError::MassImbalance { .. })));
    }

    #[test]
    fn agrees_with_sinkhorn_at_small_eps() {
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.4_f32, 0.4, 0.2];
        let exact = network_simplex(&c, &a, &b, m, n, &NsConfig::default()).expect("ok");
        let cfg = crate::sinkhorn::sinkhorn::SinkhornConfig {
            eps: 0.5,
            max_iter: 5000,
            tol: 1e-4,
        };
        let approx_res = crate::sinkhorn::sinkhorn::sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        let diff = (exact.cost - approx_res.cost).abs();
        assert!(
            diff < 0.5,
            "exact={} sinkhorn={}",
            exact.cost,
            approx_res.cost
        );
    }

    #[test]
    fn shape_mismatch_rejected() {
        let c = vec![1.0_f32; 6];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let res = network_simplex(&c, &a, &b, 2, 2, &NsConfig::default());
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }
}
