#![allow(clippy::needless_range_loop)]
//! Static Schrödinger Bridge via log-domain Iterative Proportional Fitting.
//!
//! ## Algorithm
//!
//! The Gibbs kernel `K_ij = exp(−C_ij / ε)` is alternately rescaled to satisfy
//! the row and column marginal constraints
//!
//! ```text
//! u^{(k+1)}_i = a_i / (K v^{(k)})_i ,
//! v^{(k+1)}_j = b_j / (Kᵀ u^{(k+1)})_j .
//! ```
//!
//! Convergence is monitored by the maximum absolute marginal residual.
//! Numerical stability is preserved by carrying the logarithms `f_i = log u_i`
//! and `g_j = log v_j` and replacing the divisions/multiplications by
//! log-domain log-sum-exp updates. This is mathematically equivalent to the
//! Sinkhorn-Knopp solver in `crate::sinkhorn::sinkhorn`, but exposes the IPF
//! perspective — only marginal constraints, no explicit reference to a
//! transport plan.

use crate::error::{OtError, OtResult};

/// Configuration for the IPF Schrödinger Bridge solver.
#[derive(Debug, Clone)]
pub struct SchrodingerConfig {
    /// Entropic regularisation strength ε (> 0).
    pub eps: f32,
    /// Maximum number of IPF half-iterations.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance.
    pub tol: f32,
}

impl Default for SchrodingerConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 500,
            tol: 1e-4,
        }
    }
}

/// Output of `schrodinger_bridge`.
#[derive(Debug, Clone)]
pub struct SchrodingerResult {
    /// Joint plan `P_ij`, shape `[m × n]` row-major (length `m·n`).
    pub plan: Vec<f32>,
    /// Number of completed full IPF iterations (row + column update pair).
    pub iters: usize,
}

/// Stable log-sum-exp on a slice; returns `f32::NEG_INFINITY` if empty.
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

/// `log(x)` clamped from below by `log(f32::MIN_POSITIVE)`.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Validate cost / marginals shape and parameters.
fn validate(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &SchrodingerConfig,
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

/// Solve the static Schrödinger Bridge problem with IPF in log-domain.
///
/// `c` is the cost matrix shape `[m × n]` row-major; `a` and `b` are the
/// prescribed source/target marginals. The returned plan satisfies the row and
/// column marginal constraints up to `cfg.tol`.
pub fn schrodinger_bridge(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &SchrodingerConfig,
) -> OtResult<SchrodingerResult> {
    validate(c, a, b, m, n, cfg)?;

    let eps = cfg.eps;
    // Carry `f_i = ε · log u_i` and `g_j = ε · log v_j` so that
    // `log P_ij = (f_i + g_j − C_ij) / ε`.
    let mut f = vec![0.0_f32; m];
    let mut g = vec![0.0_f32; n];
    for (i, &ai) in a.iter().enumerate() {
        f[i] = eps * safe_ln(ai);
    }
    for (j, &bj) in b.iter().enumerate() {
        g[j] = eps * safe_ln(bj);
    }

    let mut buf = vec![0.0_f32; m.max(n)];
    let mut iters = 0_usize;

    for it in 0..cfg.max_iter {
        // Row IPF: f_i = ε · log a_i − ε · LSE_j ((g_j − C_ij)/ε)
        //          ⇔  u_i = a_i / (K v)_i.
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                buf[j] = (g[j] - c[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf[..n]);
            f[i] = eps * safe_ln(a[i]) - eps * lse;
        }

        // Column IPF: g_j = ε · log b_j − ε · LSE_i ((f_i − C_ij)/ε)
        //             ⇔  v_j = b_j / (Kᵀ u)_j.
        for j in 0..n {
            for i in 0..m {
                buf[i] = (f[i] - c[i * n + j]) / eps;
            }
            let lse = logsumexp(&buf[..m]);
            g[j] = eps * safe_ln(b[j]) - eps * lse;
        }

        iters = it + 1;

        // Convergence: max marginal residual after full IPF cycle.
        let mut max_residual = 0.0_f32;
        for (i, &ai) in a.iter().enumerate() {
            let row_off = i * n;
            let mut row_sum = 0.0_f32;
            for j in 0..n {
                row_sum += ((f[i] + g[j] - c[row_off + j]) / eps).exp();
            }
            let r = (row_sum - ai).abs();
            if r > max_residual {
                max_residual = r;
            }
        }
        for (j, &bj) in b.iter().enumerate() {
            let mut col_sum = 0.0_f32;
            for i in 0..m {
                col_sum += ((f[i] + g[j] - c[i * n + j]) / eps).exp();
            }
            let r = (col_sum - bj).abs();
            if r > max_residual {
                max_residual = r;
            }
        }
        if max_residual < cfg.tol {
            break;
        }
        if it + 1 == cfg.max_iter && max_residual >= cfg.tol {
            return Err(OtError::NotConverged {
                iter: cfg.max_iter,
                tol: cfg.tol,
            });
        }
    }

    // Materialise the joint plan.
    let mut plan = vec![0.0_f32; m * n];
    for i in 0..m {
        let row_off = i * n;
        for j in 0..n {
            plan[row_off + j] = ((f[i] + g[j] - c[row_off + j]) / eps).exp();
        }
    }

    Ok(SchrodingerResult { plan, iters })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn schrodinger_satisfies_marginals() {
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.4_f32, 0.4, 0.2];
        let cfg = SchrodingerConfig {
            eps: 0.3,
            max_iter: 2000,
            tol: 1e-4,
        };
        let res = schrodinger_bridge(&c, &a, &b, m, n, &cfg).expect("converges");
        for (i, &ai) in a.iter().enumerate() {
            let row_sum: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row_sum, ai, 5e-3), "row {i}: {row_sum} != {ai}");
        }
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col_sum, bj, 5e-3), "col {j}: {col_sum} != {bj}");
        }
    }

    #[test]
    fn schrodinger_agrees_with_sinkhorn() {
        let m = 4;
        let n = 4;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let d = i as f32 - j as f32;
                c[i * n + j] = d * d;
            }
        }
        let a = vec![0.25_f32; m];
        let b = vec![0.25_f32; n];
        let cfg = SchrodingerConfig {
            eps: 0.4,
            max_iter: 5000,
            tol: 1e-5,
        };
        let sk_cfg = SinkhornConfig {
            eps: 0.4,
            max_iter: 5000,
            tol: 1e-5,
        };
        let bridge = schrodinger_bridge(&c, &a, &b, m, n, &cfg).expect("ok");
        let sk = sinkhorn(&c, &a, &b, m, n, &sk_cfg).expect("ok");
        for k in 0..m * n {
            assert!(
                approx(bridge.plan[k], sk.plan[k], 5e-3),
                "entry {k}: bridge {} sinkhorn {}",
                bridge.plan[k],
                sk.plan[k]
            );
        }
    }

    #[test]
    fn schrodinger_uniform_for_constant_cost() {
        let m = 3;
        let n = 3;
        let c = vec![1.0_f32; m * n];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = SchrodingerConfig {
            eps: 0.5,
            max_iter: 500,
            tol: 1e-5,
        };
        let res = schrodinger_bridge(&c, &a, &b, m, n, &cfg).expect("ok");
        let expected = 1.0_f32 / 9.0;
        for &p in &res.plan {
            assert!(approx(p, expected, 1e-3));
        }
    }

    #[test]
    fn schrodinger_rejects_bad_eps() {
        let cfg = SchrodingerConfig {
            eps: 0.0,
            max_iter: 10,
            tol: 1e-3,
        };
        let res = schrodinger_bridge(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn schrodinger_rejects_shape_mismatch() {
        let cfg = SchrodingerConfig::default();
        let res = schrodinger_bridge(&[0.0_f32; 6], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn schrodinger_rejects_negative_marginal() {
        let cfg = SchrodingerConfig::default();
        let c = vec![0.0_f32; 4];
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32, 0.5];
        let res = schrodinger_bridge(&c, &a, &b, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn schrodinger_rejects_empty() {
        let cfg = SchrodingerConfig::default();
        let res = schrodinger_bridge(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }
}
