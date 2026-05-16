//! Active-set QP for `min ½ x^T P x + q^T x  s.t. A_eq x = b_eq, A_ineq x ≤ b_ineq`.
//!
//! Maintains a working set of active inequality indices.  At each iteration:
//! 1. Solve the equality-constrained KKT subproblem.
//! 2. Compute step direction `d = x_kkt − x`.
//! 3. If `d ≈ 0`: compute Lagrange multipliers; if all ≥ 0, optimal; else drop most negative.
//! 4. Else: take maximum feasible step; add blocking constraint.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// Active-set QP result.
#[derive(Debug, Clone)]
pub struct QpResult {
    pub x: Vec<f64>,
    pub lambda_eq: Vec<f64>,
    pub mu_ineq: Vec<f64>,
    pub objective: f64,
    pub iter: usize,
    pub active_set: Vec<usize>,
}

/// Active-set method. `x0` must be a feasible starting point.
#[allow(clippy::too_many_arguments)]
pub fn active_set_qp(
    p_mat: &[f64],
    n: usize,
    q: &[f64],
    a_eq: &[f64],
    m_eq: usize,
    b_eq: &[f64],
    a_ineq: &[f64],
    m_ineq: usize,
    b_ineq: &[f64],
    x0: &[f64],
    max_iter: usize,
) -> CvxResult<QpResult> {
    if p_mat.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p_mat.len()],
        });
    }
    if q.len() != n || x0.len() != n {
        return Err(CvxError::DimensionMismatch { a: q.len(), b: n });
    }
    if a_eq.len() != m_eq * n || b_eq.len() != m_eq {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m_eq, n],
            got: vec![a_eq.len()],
        });
    }
    if a_ineq.len() != m_ineq * n || b_ineq.len() != m_ineq {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m_ineq, n],
            got: vec![a_ineq.len()],
        });
    }
    let mut x = x0.to_vec();
    // Determine initial active set: ineq i is active if a_i x = b_i.
    let mut active: Vec<usize> = Vec::new();
    for i in 0..m_ineq {
        let mut row_dot = 0.0_f64;
        for j in 0..n {
            row_dot += a_ineq[i * n + j] * x[j];
        }
        if (row_dot - b_ineq[i]).abs() < 1.0e-9 {
            active.push(i);
        }
    }
    let mut iters = 0usize;
    let mut final_mu_ineq = vec![0.0_f64; m_ineq];
    let mut final_lambda_eq = vec![0.0_f64; m_eq];
    for it in 0..max_iter {
        // Build A_active = [A_eq; A_ineq[active]] and b_active = [b_eq; b_ineq[active]].
        let m_act = m_eq + active.len();
        let mut a_act = vec![0.0_f64; m_act * n];
        let mut b_act = vec![0.0_f64; m_act];
        for i in 0..m_eq {
            b_act[i] = b_eq[i];
            for j in 0..n {
                a_act[i * n + j] = a_eq[i * n + j];
            }
        }
        for (k, &idx) in active.iter().enumerate() {
            b_act[m_eq + k] = b_ineq[idx];
            for j in 0..n {
                a_act[(m_eq + k) * n + j] = a_ineq[idx * n + j];
            }
        }
        // Solve KKT system [[P, A^T], [A, 0]] [x; λ] = [-q; b_act].
        let kkt_n = n + m_act;
        let mut kkt = vec![0.0_f64; kkt_n * kkt_n];
        for i in 0..n {
            for j in 0..n {
                kkt[i * kkt_n + j] = p_mat[i * n + j];
            }
            for k in 0..m_act {
                kkt[i * kkt_n + n + k] = a_act[k * n + i];
            }
        }
        for k in 0..m_act {
            for j in 0..n {
                kkt[(n + k) * kkt_n + j] = a_act[k * n + j];
            }
        }
        let mut rhs = vec![0.0_f64; kkt_n];
        for i in 0..n {
            rhs[i] = -q[i];
        }
        rhs[n..(n + m_act)].copy_from_slice(&b_act[..m_act]);
        let sol = solve_dense(&kkt, kkt_n, &rhs)?;
        let x_star: Vec<f64> = sol[..n].to_vec();
        let multipliers: Vec<f64> = sol[n..].to_vec();
        // Direction d = x_star - x.
        let mut d = vec![0.0_f64; n];
        for j in 0..n {
            d[j] = x_star[j] - x[j];
        }
        let d_nrm = norm2(&d);
        if d_nrm < 1.0e-10 {
            // Check Lagrange multipliers for inequalities (last `active.len()` entries).
            let n_eq = m_eq;
            let mut min_mu = 0.0_f64;
            let mut drop_pos: Option<usize> = None;
            for (k, &_idx) in active.iter().enumerate() {
                let mu = multipliers[n_eq + k];
                if mu < min_mu {
                    min_mu = mu;
                    drop_pos = Some(k);
                }
            }
            if let Some(pos) = drop_pos {
                active.remove(pos);
                iters = it + 1;
                continue;
            }
            // Optimal.
            final_lambda_eq[..m_eq].copy_from_slice(&multipliers[..m_eq]);
            for (k, &idx) in active.iter().enumerate() {
                final_mu_ineq[idx] = multipliers[m_eq + k];
            }
            let obj = quad_obj(p_mat, n, q, &x);
            return Ok(QpResult {
                x,
                lambda_eq: final_lambda_eq,
                mu_ineq: final_mu_ineq,
                objective: obj,
                iter: it + 1,
                active_set: active,
            });
        }
        // Step length: maximal alpha ∈ (0, 1] keeping a_i x ≤ b_i for all inactive.
        let mut alpha = 1.0_f64;
        let mut blocker: Option<usize> = None;
        for i in 0..m_ineq {
            if active.contains(&i) {
                continue;
            }
            let mut a_d = 0.0_f64;
            let mut a_x = 0.0_f64;
            for j in 0..n {
                a_d += a_ineq[i * n + j] * d[j];
                a_x += a_ineq[i * n + j] * x[j];
            }
            if a_d > 1.0e-12 {
                let cap = (b_ineq[i] - a_x) / a_d;
                if cap < alpha {
                    alpha = cap;
                    blocker = Some(i);
                }
            }
        }
        if alpha < 0.0 {
            alpha = 0.0;
        }
        for j in 0..n {
            x[j] += alpha * d[j];
        }
        if let Some(bidx) = blocker {
            if alpha < 1.0 - 1.0e-12 && !active.contains(&bidx) {
                active.push(bidx);
            }
        }
        iters = it + 1;
    }
    let _ = mat_vec; // suppress unused
    Err(CvxError::NotConverged {
        iter: iters,
        residual: f64::NAN,
    })
}

fn quad_obj(p_mat: &[f64], n: usize, q: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..n {
        let mut row = 0.0_f64;
        for j in 0..n {
            row += p_mat[i * n + j] * x[j];
        }
        sum += 0.5 * x[i] * row + q[i] * x[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qp_identity_with_unit_sum_constraint() {
        // min 0.5 ||x||² s.t. x_1 + x_2 = 1.  Optimum x = (0.5, 0.5).
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64; 2];
        let a_eq = vec![1.0_f64, 1.0];
        let b_eq = vec![1.0_f64];
        let a_ineq: Vec<f64> = Vec::new();
        let b_ineq: Vec<f64> = Vec::new();
        let res = active_set_qp(
            &p_mat,
            2,
            &q,
            &a_eq,
            1,
            &b_eq,
            &a_ineq,
            0,
            &b_ineq,
            &[1.0, 0.0],
            50,
        )
        .expect("ok");
        assert!((res.x[0] - 0.5).abs() < 1.0e-8);
        assert!((res.x[1] - 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn qp_identity_with_unit_x() {
        // min 0.5 ||x||² s.t. x_i = 1 ∀ i → x = (1, 1, 1).
        let n = 3;
        let p_mat = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64; n];
        let mut a_eq = vec![0.0_f64; n * n];
        for i in 0..n {
            a_eq[i * n + i] = 1.0;
        }
        let b_eq = vec![1.0_f64; n];
        let a_ineq: Vec<f64> = Vec::new();
        let b_ineq: Vec<f64> = Vec::new();
        let res = active_set_qp(
            &p_mat,
            n,
            &q,
            &a_eq,
            n,
            &b_eq,
            &a_ineq,
            0,
            &b_ineq,
            &[1.0, 1.0, 1.0],
            50,
        )
        .expect("ok");
        for &xi in &res.x {
            assert!((xi - 1.0).abs() < 1.0e-8);
        }
    }

    #[test]
    fn qp_with_inequality() {
        // min 0.5 (x_1² + x_2²) - x_1 - x_2 s.t. x_1 + x_2 ≤ 1, x_1, x_2 ≥ 0.
        // Unconstrained optimum: (1, 1). With x_1 + x_2 ≤ 1, optimum is on boundary, x = (0.5, 0.5).
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![-1.0_f64, -1.0];
        let a_eq: Vec<f64> = Vec::new();
        let b_eq: Vec<f64> = Vec::new();
        let a_ineq = vec![1.0_f64, 1.0, -1.0, 0.0, 0.0, -1.0];
        let b_ineq = vec![1.0_f64, 0.0, 0.0];
        let res = active_set_qp(
            &p_mat,
            2,
            &q,
            &a_eq,
            0,
            &b_eq,
            &a_ineq,
            3,
            &b_ineq,
            &[0.0, 0.0],
            50,
        )
        .expect("ok");
        assert!((res.x[0] - 0.5).abs() < 1.0e-6);
        assert!((res.x[1] - 0.5).abs() < 1.0e-6);
    }
}
