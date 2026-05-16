//! Low-level log-stabilised Sinkhorn building blocks.
//!
//! These functions expose the half-iteration row and column updates used by
//! the entropic OT solver, so callers can implement custom schedules
//! (e.g. acceleration, warm-restart, IPF coupling) without re-deriving the
//! log-domain arithmetic.

use crate::error::{OtError, OtResult};

/// Stable log-sum-exp over a slice; returns `f32::NEG_INFINITY` if empty.
fn logsumexp(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &x in slice {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f32;
    for &x in slice {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

/// Validate that `eps > 0` and that all slices have the expected shape.
fn validate(c: &[f32], eps: f32, m: usize, n: usize, u_len: usize, v_len: usize) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps });
    }
    if c.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: u_len,
            b_len: v_len,
        });
    }
    if u_len != m || v_len != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: u_len,
            b_len: v_len,
        });
    }
    Ok(())
}

/// Single half-iteration row update: `u_i ← ε · log a_i − ε · LSE_j ((v_j − C_ij)/ε)`.
///
/// `log_a` is the precomputed `log a_i`. `c` is the cost matrix `m × n` row-major.
pub fn log_sinkhorn_step_row(
    c: &[f32],
    log_a: &[f32],
    u: &mut [f32],
    v: &[f32],
    eps: f32,
    m: usize,
    n: usize,
) -> OtResult<()> {
    validate(c, eps, m, n, u.len(), v.len())?;
    if log_a.len() != m {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: log_a.len(),
            b_len: v.len(),
        });
    }
    let mut buf = vec![0.0_f32; n];
    for (i, u_val) in u.iter_mut().enumerate() {
        let row_off = i * n;
        for (j, b) in buf.iter_mut().enumerate() {
            *b = (v[j] - c[row_off + j]) / eps;
        }
        let lse = logsumexp(&buf);
        *u_val = eps * log_a[i] - eps * lse;
    }
    Ok(())
}

/// Single half-iteration column update: `v_j ← ε · log b_j − ε · LSE_i ((u_i − C_ij)/ε)`.
pub fn log_sinkhorn_step_col(
    c: &[f32],
    log_b: &[f32],
    v: &mut [f32],
    u: &[f32],
    eps: f32,
    m: usize,
    n: usize,
) -> OtResult<()> {
    validate(c, eps, m, n, u.len(), v.len())?;
    if log_b.len() != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: u.len(),
            b_len: log_b.len(),
        });
    }
    let mut buf = vec![0.0_f32; m];
    for (j, v_val) in v.iter_mut().enumerate() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (u[i] - c[i * n + j]) / eps;
        }
        let lse = logsumexp(&buf);
        *v_val = eps * log_b[j] - eps * lse;
    }
    Ok(())
}

/// Materialise the transport plan `P_ij = exp((u_i + v_j − C_ij)/ε)`.
pub fn log_to_plan(
    c: &[f32],
    u: &[f32],
    v: &[f32],
    eps: f32,
    m: usize,
    n: usize,
) -> OtResult<Vec<f32>> {
    validate(c, eps, m, n, u.len(), v.len())?;
    let mut plan = vec![0.0_f32; m * n];
    for (i, &u_i) in u.iter().enumerate() {
        let row_off = i * n;
        for (j, &v_j) in v.iter().enumerate() {
            plan[row_off + j] = ((u_i + v_j - c[row_off + j]) / eps).exp();
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_step_normalises_marginals_after_one_iteration_pair() {
        // Two-iteration manual cycle on uniform marginals.
        let m = 2;
        let n = 2;
        let c = vec![0.0_f32, 1.0, 1.0, 0.0];
        let log_a = vec![(0.5_f32).ln(); 2];
        let log_b = vec![(0.5_f32).ln(); 2];
        let mut u = vec![0.0_f32; m];
        let mut v = vec![0.0_f32; n];
        let eps = 0.1_f32;
        log_sinkhorn_step_row(&c, &log_a, &mut u, &v, eps, m, n).expect("ok");
        log_sinkhorn_step_col(&c, &log_b, &mut v, &u, eps, m, n).expect("ok");
        for _ in 0..200 {
            log_sinkhorn_step_row(&c, &log_a, &mut u, &v, eps, m, n).expect("ok");
            log_sinkhorn_step_col(&c, &log_b, &mut v, &u, eps, m, n).expect("ok");
        }
        let plan = log_to_plan(&c, &u, &v, eps, m, n).expect("ok");
        for i in 0..m {
            let row_sum: f32 = (0..n).map(|j| plan[i * n + j]).sum();
            assert!((row_sum - 0.5_f32).abs() < 1e-3);
        }
        for j in 0..n {
            let col_sum: f32 = (0..m).map(|i| plan[i * n + j]).sum();
            assert!((col_sum - 0.5_f32).abs() < 1e-3);
        }
    }

    #[test]
    fn log_to_plan_rejects_bad_shapes() {
        let c = vec![0.0_f32; 4];
        let u = vec![0.0_f32; 1];
        let v = vec![0.0_f32; 2];
        let res = log_to_plan(&c, &u, &v, 0.1, 2, 2);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn step_rejects_bad_eps() {
        let c = vec![0.0_f32; 4];
        let log_a = vec![0.0_f32; 2];
        let mut u = vec![0.0_f32; 2];
        let v = vec![0.0_f32; 2];
        let res = log_sinkhorn_step_row(&c, &log_a, &mut u, &v, 0.0, 2, 2);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn step_rejects_empty() {
        let res = log_sinkhorn_step_row(&[], &[], &mut [], &[], 0.1, 0, 0);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }
}
