//! Revised simplex method (Bland's rule for anti-cycling).
//!
//! Solves standard form: `min cᵀx  s.t. A x = b, x ≥ 0`,
//! where `A` is m × n with m ≤ n (assume A has full row rank).
//!
//! Maintains a basis index set `B` (size m), non-basis `N`, with `x_B = B⁻¹ b ≥ 0`,
//! `x_N = 0`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::solve::solve_dense;

/// Status of the simplex solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplexStatus {
    Optimal,
    Unbounded,
    MaxIter,
}

/// Revised simplex result.
#[derive(Debug, Clone)]
pub struct SimplexResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub basis: Vec<usize>,
    pub status: SimplexStatus,
    pub iter: usize,
}

/// Solve `min cᵀx s.t. Ax = b, x ≥ 0`.
///
/// `initial_basis` must be an m-element index set producing a feasible starting basis;
/// caller is responsible for Phase-1 or supplying it.  `b ≥ 0` assumed.
pub fn revised_simplex(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    c: &[f64],
    initial_basis: &[usize],
    max_iter: usize,
) -> CvxResult<SimplexResult> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    if c.len() != n {
        return Err(CvxError::DimensionMismatch { a: c.len(), b: n });
    }
    if initial_basis.len() != m {
        return Err(CvxError::DimensionMismatch {
            a: initial_basis.len(),
            b: m,
        });
    }
    for &j in initial_basis {
        if j >= n {
            return Err(CvxError::IndexOutOfBounds { index: j, len: n });
        }
    }
    let mut basis = initial_basis.to_vec();
    for it in 0..max_iter {
        // Build B (m × m): column j of A_B is A[:, basis[j]].
        let b_mat = extract_columns(a, m, n, &basis);
        // Compute x_B = B⁻¹ b.
        let x_b = solve_dense(&b_mat, m, b)?;
        // Feasibility: x_B ≥ 0 (allow small tol).
        if x_b.iter().any(|&v| v < -1.0e-8) {
            return Err(CvxError::Infeasible(format!(
                "simplex iter {it}: negative x_B encountered"
            )));
        }
        // Compute y = (Bᵀ)⁻¹ c_B (multipliers).
        let c_b: Vec<f64> = basis.iter().map(|&j| c[j]).collect();
        let b_t_mat = transpose_square(&b_mat, m);
        let y = solve_dense(&b_t_mat, m, &c_b)?;
        // Reduced costs c̄_N = c_N − Aᵀ_N y. Find min (or Bland's lowest-index negative).
        let mut entering: Option<usize> = None;
        for j in 0..n {
            if basis.contains(&j) {
                continue;
            }
            let mut aty = 0.0_f64;
            for i in 0..m {
                aty += a[i * n + j] * y[i];
            }
            let rc = c[j] - aty;
            if rc < -1.0e-9 {
                // Bland: pick smallest j.
                entering = Some(j);
                break;
            }
        }
        let entering = match entering {
            Some(j) => j,
            None => {
                // Optimal.
                let mut x = vec![0.0_f64; n];
                for (i, &j) in basis.iter().enumerate() {
                    x[j] = x_b[i];
                }
                let obj: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
                return Ok(SimplexResult {
                    x,
                    objective: obj,
                    basis,
                    status: SimplexStatus::Optimal,
                    iter: it,
                });
            }
        };
        // Compute direction d = B⁻¹ A_:,entering.
        let mut a_col = vec![0.0_f64; m];
        for i in 0..m {
            a_col[i] = a[i * n + entering];
        }
        let d = solve_dense(&b_mat, m, &a_col)?;
        // Ratio test: leaving = argmin {x_B[i] / d[i] : d[i] > 0}; Bland breaks ties by basis[i].
        let mut leaving_idx: Option<usize> = None;
        let mut min_ratio = f64::INFINITY;
        let mut leaving_basis_value: usize = usize::MAX;
        for i in 0..m {
            if d[i] > 1.0e-12 {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio - 1.0e-12
                    || (ratio < min_ratio + 1.0e-12 && basis[i] < leaving_basis_value)
                {
                    min_ratio = ratio;
                    leaving_idx = Some(i);
                    leaving_basis_value = basis[i];
                }
            }
        }
        let leaving = match leaving_idx {
            Some(i) => i,
            None => {
                // d ≤ 0 → can move infinitely → unbounded.
                let mut x = vec![0.0_f64; n];
                for (i, &j) in basis.iter().enumerate() {
                    x[j] = x_b[i];
                }
                return Ok(SimplexResult {
                    x,
                    objective: f64::NEG_INFINITY,
                    basis,
                    status: SimplexStatus::Unbounded,
                    iter: it,
                });
            }
        };
        basis[leaving] = entering;
    }
    // Max iter exhausted.
    let b_mat = extract_columns(a, m, n, &basis);
    let x_b = solve_dense(&b_mat, m, b)?;
    let mut x = vec![0.0_f64; n];
    for (i, &j) in basis.iter().enumerate() {
        x[j] = x_b[i];
    }
    let obj: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
    Ok(SimplexResult {
        x,
        objective: obj,
        basis,
        status: SimplexStatus::MaxIter,
        iter: max_iter,
    })
}

fn extract_columns(a: &[f64], m: usize, n: usize, cols: &[usize]) -> Vec<f64> {
    let p = cols.len();
    let mut out = vec![0.0_f64; m * p];
    for i in 0..m {
        for (k, &c) in cols.iter().enumerate() {
            out[i * p + k] = a[i * n + c];
        }
    }
    out
}

fn transpose_square(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[i * n + j] = a[j * n + i];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_min_negx_negy() {
        // min -x - y  s.t.  x + y + s = 1; x, y, s ≥ 0.
        // Equivalent to maximising x + y subject to x + y ≤ 1.
        // Optimum: (x,y) = (1, 0) or (0, 1), objective = -1.
        // Standard form: variables [x, y, s]; A = [[1, 1, 1]]; b = [1]; c = [-1, -1, 0].
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let c = vec![-1.0_f64, -1.0, 0.0];
        // Initial basis: slack {2}.
        let basis = vec![2usize];
        let res = revised_simplex(&a, 1, 3, &b, &c, &basis, 100).expect("ok");
        assert_eq!(res.status, SimplexStatus::Optimal);
        assert!((res.objective + 1.0).abs() < 1.0e-9);
        // x + y = 1.
        assert!((res.x[0] + res.x[1] - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn simplex_unbounded_detected() {
        // min -x  s.t.  -x + s = 1, x, s ≥ 0  → x can grow without bound.
        // A = [[-1, 1]]; b=[1]; c=[-1, 0]; start basis {1}.
        let a = vec![-1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let c = vec![-1.0_f64, 0.0];
        let basis = vec![1usize];
        let res = revised_simplex(&a, 1, 2, &b, &c, &basis, 50).expect("ok");
        assert_eq!(res.status, SimplexStatus::Unbounded);
    }
}
