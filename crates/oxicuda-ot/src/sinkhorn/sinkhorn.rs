//! Log-domain Sinkhorn-Knopp algorithm — entropic regularised OT.
//!
//! Solves
//!
//! ```text
//! min_P  <C, P> + ε · KL(P ‖ a ⊗ b)
//!       s.t.  P 1 = a,  Pᵀ 1 = b,  P ≥ 0
//! ```
//!
//! by alternating Bregman projections. The implementation runs entirely in
//! log-space using subtract-max log-sum-exp tricks to remain numerically
//! stable for very small `eps`.

use crate::error::{OtError, OtResult};

/// Configuration for the Sinkhorn-Knopp solver.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Entropic regularisation strength ε (must be > 0).
    pub eps: f32,
    /// Maximum number of outer Sinkhorn iterations.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance (`max_i |Σ_j P_ij − a_i|`).
    pub tol: f32,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-4,
        }
    }
}

/// Output of the Sinkhorn-Knopp solver.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// Transport plan, shape `[m × n]` row-major (length `m·n`).
    pub plan: Vec<f32>,
    /// Row-side log-domain dual potentials, length `m`.
    pub u: Vec<f32>,
    /// Column-side log-domain dual potentials, length `n`.
    pub v: Vec<f32>,
    /// Transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Number of completed Sinkhorn iterations.
    pub iters: usize,
}

/// Tiny clamp used to evaluate `log(0)` safely as the smallest finite log.
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Stable log-sum-exp on a slice; returns `f32::NEG_INFINITY` if the slice is empty.
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

/// Validate Sinkhorn inputs and return `Ok(())` if all checks pass.
fn validate_inputs(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &SinkhornConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if c.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != m || b.len() != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

/// Run the log-stabilised Sinkhorn-Knopp algorithm.
///
/// `c` is the cost matrix, shape `[m × n]` row-major. `a` is the source
/// histogram (length `m`), `b` is the target histogram (length `n`). The
/// algorithm iterates row-then-column log-domain updates until the maximum
/// row-marginal residual falls below `cfg.tol`.
pub fn sinkhorn(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &SinkhornConfig,
) -> OtResult<SinkhornResult> {
    validate_inputs(c, a, b, m, n, cfg)?;
    let eps = cfg.eps;

    let mut u = vec![0.0_f32; m];
    let mut v = vec![0.0_f32; n];
    for (i, &ai) in a.iter().enumerate() {
        u[i] = eps * safe_ln(ai);
    }
    for (j, &bj) in b.iter().enumerate() {
        v[j] = eps * safe_ln(bj);
    }

    let mut buf = vec![0.0_f32; m.max(n)];

    let mut completed = 0_usize;
    for it in 0..cfg.max_iter {
        // Row update: u_i ← ε log(a_i) − ε · logsumexp_j ((v_j − C_ij)/ε)
        // (After this update, row marginals of P are exactly `a`.)
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                buf[j] = (v[j] - c[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf[..n]);
            u[i] = eps * safe_ln(a[i]) - eps * lse;
        }

        // Convergence is measured on the column residual *after* the row
        // update: column marginals are not enforced, so `max_j |Σ_i P_ij − b_j|`
        // is the natural Sinkhorn error and decays exponentially.
        let mut max_residual = 0.0_f32;
        for (j, &v_j) in v.iter().enumerate() {
            let mut col_sum = 0.0_f32;
            for (i, &u_i) in u.iter().enumerate() {
                col_sum += ((u_i + v_j - c[i * n + j]) / eps).exp();
            }
            let r = (col_sum - b[j]).abs();
            if r > max_residual {
                max_residual = r;
            }
        }
        completed = it + 1;
        if max_residual < cfg.tol {
            // Run one more column update to make col-marginals exact too.
            for j in 0..n {
                for i in 0..m {
                    buf[i] = (u[i] - c[i * n + j]) / eps;
                }
                let lse = logsumexp(&buf[..m]);
                v[j] = eps * safe_ln(b[j]) - eps * lse;
            }
            break;
        }

        // Column update: v_j ← ε log(b_j) − ε · logsumexp_i ((u_i − C_ij)/ε)
        for j in 0..n {
            for i in 0..m {
                buf[i] = (u[i] - c[i * n + j]) / eps;
            }
            let lse = logsumexp(&buf[..m]);
            v[j] = eps * safe_ln(b[j]) - eps * lse;
        }

        if it + 1 == cfg.max_iter && max_residual >= cfg.tol {
            return Err(OtError::NotConverged {
                iter: cfg.max_iter,
                tol: cfg.tol,
            });
        }
    }

    let mut plan = vec![0.0_f32; m * n];
    let mut cost = 0.0_f32;
    for (i, &u_i) in u.iter().enumerate() {
        let row_off = i * n;
        for (j, &v_j) in v.iter().enumerate() {
            let p_ij = ((u_i + v_j - c[row_off + j]) / eps).exp();
            plan[row_off + j] = p_ij;
            cost += p_ij * c[row_off + j];
        }
    }

    Ok(SinkhornResult {
        plan,
        u,
        v,
        cost,
        iters: completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn marginals_satisfied_after_convergence() {
        // 3 sources, 3 sinks, identity-like cost.
        let m = 3;
        let n = 3;
        let c = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.4_f32, 0.4, 0.2];
        let cfg = SinkhornConfig {
            eps: 0.3,
            max_iter: 2000,
            tol: 1e-4,
        };
        let res = sinkhorn(&c, &a, &b, m, n, &cfg).expect("converges");
        // Check row marginals.
        for (i, &ai) in a.iter().enumerate() {
            let row_sum: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row_sum, ai, 5e-3), "row {i} sum {row_sum} != {ai} ");
        }
        // Check column marginals.
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col_sum, bj, 5e-3));
        }
    }

    #[test]
    fn large_eps_yields_near_uniform_plan() {
        let m = 2;
        let n = 2;
        let c = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let cfg = SinkhornConfig {
            eps: 50.0,
            max_iter: 2000,
            tol: 1e-5,
        };
        let res = sinkhorn(&c, &a, &b, m, n, &cfg).expect("converges");
        let uniform = 0.25_f32;
        for &p in &res.plan {
            assert!(approx(p, uniform, 5e-3), "plan entry {p} not near uniform");
        }
    }

    #[test]
    fn diagonal_for_zero_diagonal_cost() {
        // For a == b and zero-diag cost, plan should be approximately diagonal.
        let m = 3;
        let n = 3;
        let c = vec![0.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 0.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = SinkhornConfig {
            eps: 0.5,
            max_iter: 2000,
            tol: 1e-4,
        };
        let res = sinkhorn(&c, &a, &b, m, n, &cfg).expect("converges");
        for i in 0..3 {
            assert!(res.plan[i * 3 + i] > 0.30, "diagonal entry {i} too small");
        }
    }

    #[test]
    fn bad_epsilon_returns_error() {
        let cfg = SinkhornConfig {
            eps: 0.0,
            max_iter: 10,
            tol: 1e-3,
        };
        let res = sinkhorn(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let cfg = SinkhornConfig::default();
        let res = sinkhorn(&[0.0_f32; 6], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn negative_weights_rejected() {
        let cfg = SinkhornConfig::default();
        let c = vec![0.0_f32; 4];
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32, 0.5];
        let res = sinkhorn(&c, &a, &b, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn empty_input_rejected() {
        let cfg = SinkhornConfig::default();
        let res = sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn logsumexp_empty_returns_neg_inf() {
        assert_eq!(logsumexp(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn logsumexp_known_value() {
        let v = [0.0_f32, 0.0_f32];
        let expected = (2.0_f32).ln();
        assert!((logsumexp(&v) - expected).abs() < 1e-6);
    }
}
