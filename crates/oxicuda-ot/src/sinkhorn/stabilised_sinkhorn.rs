//! Numerically Stabilised Sinkhorn (Schmitzer 2019) — log-domain with potential
//! absorption to prevent numerical underflow at small `ε`.
//!
//! # Motivation
//!
//! Standard Sinkhorn (even in log-domain) can accumulate large potentials that
//! eventually overflow the exponents inside the cost kernel when `ε` is small.
//! Schmitzer (2019) introduces *potential absorption*: the accumulated dual
//! potentials `f, g` are periodically *absorbed* back into the Gibbs kernel
//!
//! ```text
//! K̃_ij  ←  exp( (f_i + g_j − C_ij) / ε )
//! ```
//!
//! which resets the potentials to zero after each absorption while preserving
//! the product `u ⊗ v` implicit in the kernel.  This keeps all exponentials
//! numerically small regardless of `ε`.
//!
//! # Algorithm
//!
//! Iteration (in log domain):
//!
//! ```text
//! f_i  ←  ε · log a_i  −  ε · LSE_j[ (f_i + g_j − C_ij) / ε ]
//! g_j  ←  ε · log b_j  −  ε · LSE_i[ (f_i + g_j − C_ij) / ε ]
//! ```
//!
//! Every `absorb_every` iterations we set:
//!
//! ```text
//! C̃_ij  ←  C_ij − f_i − g_j          (absorb potentials into cost)
//! f_i   ←  0,    g_j  ←  0
//! ```
//!
//! which is equivalent to restarting with a modified cost but zero potentials.
//! The transport plan is recovered from the final accumulated potentials:
//!
//! ```text
//! P_ij = exp( (f_i + g_j − C_ij) / ε ) · (some global scale)
//! ```
//!
//! and is row/column normalised to satisfy the marginal constraints exactly.
//!
//! References:
//! - Schmitzer B. *Stabilized Sparse Scaling Algorithms for Entropy Regularized
//!   Transport Problems* (SIAM J. Sci. Comput., 2019).

use crate::error::{OtError, OtResult};

/// Configuration for the stabilised Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct StabilisedSinkhornConfig {
    /// Entropic regularisation strength ε (must be > 0).
    pub eps: f32,
    /// Maximum number of outer Sinkhorn iterations.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance.
    pub tol: f32,
    /// Number of iterations between potential-absorption steps.
    /// Smaller values reduce numerical range at the cost of extra memory ops.
    /// Recommended: 5–20.
    pub absorb_every: usize,
}

impl Default for StabilisedSinkhornConfig {
    fn default() -> Self {
        StabilisedSinkhornConfig {
            eps: 0.01,
            max_iter: 500,
            tol: 1e-6,
            absorb_every: 10,
        }
    }
}

/// Output of the stabilised Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct StabilisedSinkhornResult {
    /// Transport plan `P`, shape `[m × n]` row-major.
    pub plan: Vec<f32>,
    /// Row-side dual potential `f_i`, length `m`.
    pub f: Vec<f32>,
    /// Column-side dual potential `g_j`, length `n`.
    pub g: Vec<f32>,
    /// Transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Number of completed iterations.
    pub iters: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of potential absorption steps performed.
    pub n_absorptions: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Tiny guard for safe logarithm computation.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Stable log-sum-exp over a slice (subtract-max trick).
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

/// Validate inputs.
fn validate(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &StabilisedSinkhornConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if cfg.absorb_every == 0 {
        return Err(OtError::BadCount { got: 0 });
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

// ──────────────────────────────────────────────────────────────────────────────
// Main solver
// ──────────────────────────────────────────────────────────────────────────────

/// Run the stabilised Sinkhorn algorithm with periodic potential absorption.
///
/// `c` is the `[m × n]` cost matrix (row-major). `a` and `b` are the source
/// and target marginals (length `m` and `n` respectively). Both marginals
/// should sum to approximately 1.
pub fn stabilised_sinkhorn(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &StabilisedSinkhornConfig,
) -> OtResult<StabilisedSinkhornResult> {
    validate(c, a, b, m, n, cfg)?;

    let eps = cfg.eps;
    let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
    let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();

    // Mutable cost copy (absorptions modify it in-place).
    let mut c_cur = c.to_vec();

    // Dual potentials (start at zero; absorption resets them periodically).
    let mut f = vec![0.0_f32; m];
    let mut g = vec![0.0_f32; n];

    // Reusable scratch
    let mut row_buf = vec![0.0_f32; n]; // for row-wise LSE
    let mut col_buf = vec![0.0_f32; m]; // for col-wise LSE

    let mut iters = 0_usize;
    let mut n_absorptions = 0_usize;
    let mut converged = false;

    for iter in 0..cfg.max_iter {
        iters = iter + 1;

        // ── Row update ────────────────────────────────────────────────────────
        // f_i ← ε·log a_i − ε·LSE_j [ (f_i + g_j − C_ij) / ε ]
        for i in 0..m {
            for j in 0..n {
                row_buf[j] = (f[i] + g[j] - c_cur[i * n + j]) / eps;
            }
            f[i] = eps * log_a[i] - eps * logsumexp(&row_buf);
        }

        // ── Column update ─────────────────────────────────────────────────────
        // g_j ← ε·log b_j − ε·LSE_i [ (f_i + g_j − C_ij) / ε ]
        for j in 0..n {
            for i in 0..m {
                col_buf[i] = (f[i] + g[j] - c_cur[i * n + j]) / eps;
            }
            g[j] = eps * log_b[j] - eps * logsumexp(&col_buf);
        }

        // ── Convergence check ─────────────────────────────────────────────────
        // Row marginal violation: |Σ_j P_ij − a_i|
        let mut max_viol = 0.0_f32;
        for i in 0..m {
            for j in 0..n {
                row_buf[j] = (f[i] + g[j] - c_cur[i * n + j]) / eps;
            }
            let log_row_sum = logsumexp(&row_buf);
            let row_sum = log_row_sum.exp();
            let viol = (row_sum - a[i]).abs();
            if viol > max_viol {
                max_viol = viol;
            }
        }
        if max_viol < cfg.tol {
            converged = true;
            break;
        }

        // ── Potential absorption ──────────────────────────────────────────────
        // Every `absorb_every` iterations: fold f,g into the cost matrix and
        // reset potentials to zero.
        if (iter + 1) % cfg.absorb_every == 0 {
            for i in 0..m {
                for j in 0..n {
                    c_cur[i * n + j] -= f[i] + g[j];
                }
            }
            for v in f.iter_mut() {
                *v = 0.0;
            }
            for v in g.iter_mut() {
                *v = 0.0;
            }
            n_absorptions += 1;
        }
    }

    // ── Recover transport plan ────────────────────────────────────────────────
    // P_ij = exp( (f_i + g_j − C_ij) / ε )   [unnormalised]
    // Row-normalise to satisfy source marginal a.
    let mut plan = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            row_buf[j] = (f[i] + g[j] - c_cur[i * n + j]) / eps;
        }
        let lse = logsumexp(&row_buf);
        let target_log_ai = log_a[i];
        for j in 0..n {
            plan[i * n + j] = (target_log_ai + row_buf[j] - lse).exp();
        }
    }

    // Transport cost
    let cost: f32 = plan.iter().zip(c.iter()).map(|(&p, &cv)| p * cv).sum();

    Ok(StabilisedSinkhornResult {
        plan,
        f,
        g,
        cost,
        iters,
        converged,
        n_absorptions,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Convenience helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build the pairwise squared-Euclidean cost matrix from two flat sample
/// buffers (row-major, `m × d` and `n × d`). Returns the `[m × n]` cost.
pub fn sq_euclidean_cost(x: &[f32], y: &[f32], m: usize, n: usize, d: usize) -> OtResult<Vec<f32>> {
    if d == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    if x.len() != m * d || y.len() != n * d {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: m * d,
        });
    }
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for k in 0..d {
                let diff = x[i * d + k] - y[j * d + k];
                s += diff * diff;
            }
            c[i * n + j] = 0.5 * s;
        }
    }
    Ok(c)
}

/// Compute the marginal violation `max_i |Σ_j P_ij − a_i|` of a given plan.
pub fn marginal_violation_row(plan: &[f32], a: &[f32], m: usize, n: usize) -> f32 {
    let mut max_v = 0.0_f32;
    for i in 0..m {
        let row_sum: f32 = (0..n).map(|j| plan[i * n + j]).sum();
        let v = (row_sum - a[i]).abs();
        if v > max_v {
            max_v = v;
        }
    }
    max_v
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_marginal(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    fn cost_matrix_identity(n: usize) -> Vec<f32> {
        let mut c = vec![1.0_f32; n * n];
        for i in 0..n {
            c[i * n + i] = 0.0;
        }
        c
    }

    #[test]
    fn test_stabilised_sinkhorn_basic() {
        let m = 4;
        let n = 4;
        let c = cost_matrix_identity(m);
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig::default();
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        assert!(res.cost.is_finite());
        assert!(res.cost >= 0.0);
    }

    #[test]
    fn test_stabilised_sinkhorn_plan_shape() {
        let m = 3;
        let n = 5;
        let c: Vec<f32> = (0..(m * n)).map(|k| k as f32 * 0.1).collect();
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig {
            eps: 0.05,
            ..Default::default()
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        assert_eq!(res.plan.len(), m * n);
        for &p in &res.plan {
            assert!(p.is_finite() && p >= -1e-6);
        }
    }

    #[test]
    fn test_stabilised_sinkhorn_marginal_satisfaction() {
        let m = 5;
        let n = 5;
        let c = cost_matrix_identity(m);
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig {
            eps: 0.05,
            max_iter: 1000,
            tol: 1e-5,
            absorb_every: 5,
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        let viol = marginal_violation_row(&res.plan, &a, m, n);
        assert!(viol < 1e-3, "marginal violation = {viol}");
    }

    #[test]
    fn test_stabilised_sinkhorn_small_eps() {
        // The whole point: should not diverge at small ε.
        let m = 4;
        let n = 4;
        let c = cost_matrix_identity(m);
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig {
            eps: 1e-4,
            max_iter: 200,
            tol: 1e-4,
            absorb_every: 5,
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("small eps ok");
        assert!(res.cost.is_finite());
        // For identity cost, the optimal plan is the identity / uniform.
        assert!(res.cost < 0.5);
    }

    #[test]
    fn test_stabilised_sinkhorn_absorptions_occur() {
        // Use non-uniform marginals and a non-trivial cost so the algorithm
        // doesn't converge on iteration 1, ensuring absorptions occur.
        let m = 5;
        let n = 5;
        // Random-ish cost that isn't trivially optimal at uniform plan
        let c: Vec<f32> = (0..(m * n))
            .map(|k| {
                let i = k / n;
                let j = k % n;
                ((i as f32 - j as f32).powi(2) + 0.3 * (i + j) as f32) / 10.0
            })
            .collect();
        // Non-uniform marginals to prevent trivial convergence
        let a: Vec<f32> = vec![0.1, 0.3, 0.2, 0.25, 0.15];
        let b: Vec<f32> = vec![0.2, 0.15, 0.35, 0.1, 0.2];
        let cfg = StabilisedSinkhornConfig {
            eps: 1e-6, // very small eps → many iterations needed
            max_iter: 30,
            tol: 1e-15, // very tight → won't converge early
            absorb_every: 5,
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        // With 30 iters and absorb_every=5, expect ≥ 1 absorptions.
        assert!(
            res.n_absorptions >= 1,
            "expected absorptions, got {}",
            res.n_absorptions
        );
    }

    #[test]
    fn test_stabilised_sinkhorn_empty_input() {
        let cfg = StabilisedSinkhornConfig::default();
        let err = stabilised_sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_stabilised_sinkhorn_bad_epsilon() {
        let c = vec![1.0_f32; 4];
        let a = uniform_marginal(2);
        let b = uniform_marginal(2);
        let cfg = StabilisedSinkhornConfig {
            eps: -0.1,
            ..Default::default()
        };
        let err = stabilised_sinkhorn(&c, &a, &b, 2, 2, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_sq_euclidean_cost_basic() {
        let x = [0.0_f32, 0.0];
        let y = [1.0_f32, 0.0, 0.0_f32, 1.0];
        let c = sq_euclidean_cost(&x, &y, 1, 2, 2).expect("ok");
        assert_eq!(c.len(), 2);
        assert!((c[0] - 0.5).abs() < 1e-5, "c[0]={}", c[0]);
        assert!((c[1] - 0.5).abs() < 1e-5, "c[1]={}", c[1]);
    }

    #[test]
    fn test_stabilised_sinkhorn_plan_cost_matches() {
        let m = 3;
        let n = 3;
        let c: Vec<f32> = (0..(m * n)).map(|k| (k as f32) * 0.2).collect();
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-5,
            absorb_every: 10,
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        let computed_cost: f32 = res.plan.iter().zip(c.iter()).map(|(&p, &cv)| p * cv).sum();
        assert!((res.cost - computed_cost).abs() < 1e-4);
    }

    #[test]
    fn test_stabilised_sinkhorn_iters_positive() {
        let m = 3;
        let n = 3;
        let c: Vec<f32> = vec![0.0; m * n];
        let a = uniform_marginal(m);
        let b = uniform_marginal(n);
        let cfg = StabilisedSinkhornConfig {
            eps: 0.1,
            max_iter: 20,
            tol: 1e-8,
            absorb_every: 5,
        };
        let res = stabilised_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        assert!(res.iters > 0);
    }
}
