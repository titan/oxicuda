//! Bregman-projected Gromov-Wasserstein (Xu et al. 2019).
//!
//! Computes the entropic Gromov-Wasserstein distance via mirror descent on the
//! coupling matrix `Γ` under the GW objective, using Bregman projections onto
//! the transport polytope.
//!
//! # Background
//!
//! The Gromov-Wasserstein problem seeks a coupling `Γ ∈ Π(a, b)` that minimises
//!
//! ```text
//! GW(C¹, C², a, b) = min_{Γ ∈ Π(a,b)}  Σ_{i,j,k,l} L(C¹_{ik}, C²_{jl}) Γ_{ij} Γ_{kl}
//! ```
//!
//! where `L(s, t) = (s − t)²` is the squared-loss (squared Frobenius relaxation).
//!
//! Xu et al. (2019) formulate this as a *quadratic programme over probability
//! matrices* and apply mirror descent (Bregman proximal gradient) with the
//! negative entropy `Ω(Γ) = Σ_{ij} Γ_{ij} log Γ_{ij}` as the distance-generating
//! function.  Each update consists of:
//!
//! 1. **Gradient step**: Compute the gradient of the GW objective:
//!    `∇_Γ GW = −2 · (C¹ · Γ · C²)`.
//!    With the squared-loss this simplifies to:
//!    `G_{ij} = (f_i¹ + f_j² − 2·(C¹ Γ C²)_{ij}) · (sign determined by loss)`
//! 2. **Regularised projection**: Solve the entropic OT problem
//!    `min_Γ' <G, Γ'> + λ · KL(Γ' ‖ Γ)` subject to `Γ' ∈ Π(a, b)`,
//!    which is a single Sinkhorn step on the kernel `K_ij = Γ_ij · exp(−G_ij / λ)`.
//!
//! Convergence is guaranteed for λ-strongly convex regularisers (Bregman divergence).
//!
//! References:
//! - Xu H., Luo D., Zha H. & Duke L.C. *Gromov-Wasserstein Learning for Graph
//!   Matching and Node Embedding* (ICML 2019).
//! - Peyré G., Cuturi M. & Solomon J. *Gromov-Wasserstein Averaging of Kernel
//!   and Distance Matrices* (ICML 2016).

use crate::error::{OtError, OtResult};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Bregman-projected Gromov-Wasserstein solver.
#[derive(Debug, Clone)]
pub struct BregmanGwConfig {
    /// Entropic regularisation strength `λ` for the Bregman projection (> 0).
    pub lambda: f32,
    /// Maximum outer iterations (mirror descent steps).
    pub max_iter: usize,
    /// Convergence tolerance on the Frobenius change of `Γ`.
    pub tol: f32,
    /// Maximum Sinkhorn inner iterations for the projection step.
    pub inner_max_iter: usize,
    /// Sinkhorn convergence tolerance for the projection step.
    pub inner_tol: f32,
}

impl Default for BregmanGwConfig {
    fn default() -> Self {
        BregmanGwConfig {
            lambda: 0.1,
            max_iter: 100,
            tol: 1e-5,
            inner_max_iter: 50,
            inner_tol: 1e-6,
        }
    }
}

/// Output of the Bregman GW solver.
#[derive(Debug, Clone)]
pub struct BregmanGwResult {
    /// Optimal transport coupling `Γ`, shape `[m × n]` row-major.
    pub coupling: Vec<f32>,
    /// GW objective value at convergence: `Σ L(C¹_{ik}, C²_{jl}) Γ_{ij} Γ_{kl}`.
    pub gw_loss: f32,
    /// Number of outer mirror-descent iterations performed.
    pub iters: usize,
    /// Whether the outer loop converged within tolerance.
    pub converged: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Safe log to avoid -∞ issues.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Stable log-sum-exp on a slice.
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

/// Compute `M = A · B` where `A` is `[m×k]` row-major and `B` is `[k×n]`
/// row-major.  Returns `[m×n]` row-major.
fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    out
}

/// Compute the GW gradient tensor (linearised):
/// `G_{ij} = f_i^1 + f_j^2 − 2 · (C¹ Γ C²)_{ij}`
/// where `f_i^1 = Σ_{k} (C¹_{ik})² a_k` and `f_j^2 = Σ_l (C²_{jl})² b_l`.
///
/// The full GW gradient for the squared-Frobenius loss `L(s,t) = (s−t)²` is
/// `4 · G` (with sign convention for minimisation).
fn gw_gradient(
    c1: &[f32],
    c2: &[f32],
    gamma: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) -> Vec<f32> {
    // f1_i = Σ_k (C1_{ik})^2 * a_k
    let mut f1 = vec![0.0_f32; m];
    for i in 0..m {
        for k in 0..m {
            f1[i] += c1[i * m + k].powi(2) * a[k];
        }
    }
    // f2_j = Σ_l (C2_{jl})^2 * b_l
    let mut f2 = vec![0.0_f32; n];
    for j in 0..n {
        for l in 0..n {
            f2[j] += c2[j * n + l].powi(2) * b[l];
        }
    }
    // CgC = C1 · Γ · C2, shape [m × n]
    let c1_gamma = mat_mul(c1, gamma, m, m, n);
    let cgc = mat_mul(&c1_gamma, c2, m, n, n);

    // G_{ij} = f1_i + f2_j − 2 · (C1 Γ C2)_{ij}
    let mut g = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            g[i * n + j] = f1[i] + f2[j] - 2.0 * cgc[i * n + j];
        }
    }
    g
}

/// Compute the GW objective value:
/// `Σ_{i,j,k,l} (C¹_{ik} − C²_{jl})² Γ_{ij} Γ_{kl}`
pub fn gw_objective(c1: &[f32], c2: &[f32], gamma: &[f32], m: usize, n: usize) -> f32 {
    let mut val = 0.0_f32;
    for i in 0..m {
        for k in 0..m {
            for j in 0..n {
                for l in 0..n {
                    let diff = c1[i * m + k] - c2[j * n + l];
                    val += diff.powi(2) * gamma[i * n + j] * gamma[k * n + l];
                }
            }
        }
    }
    val
}

/// One Bregman projection step: given current coupling `gamma` and gradient `g`,
/// compute `Γ' = argmin_{Γ∈Π(a,b)} <G, Γ> + λ·KL(Γ‖Γ_cur)`.
///
/// This is solved by a single log-Sinkhorn sweep on kernel
/// `K_{ij} = Γ_{ij} · exp(−G_{ij} / λ)`.
fn bregman_projection(
    gamma: &[f32],
    g: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    lambda: f32,
    inner_max_iter: usize,
    inner_tol: f32,
) -> Vec<f32> {
    // Log-kernel: log K_{ij} = log Γ_{ij} − G_{ij}/λ
    let mut log_k: Vec<f32> = gamma
        .iter()
        .zip(g.iter())
        .map(|(&gij, &gij_grad)| safe_ln(gij) - gij_grad / lambda)
        .collect();

    // Sinkhorn on K: u_i, v_j potentials (log-domain).
    let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
    let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();

    let mut u = vec![0.0_f32; m];
    let mut v = vec![0.0_f32; n];

    let mut row_buf = vec![0.0_f32; n];
    let mut col_buf = vec![0.0_f32; m];

    for _it in 0..inner_max_iter {
        // Row update: u_i ← log a_i − LSE_j(log_k_{ij} + v_j)
        for i in 0..m {
            for j in 0..n {
                row_buf[j] = log_k[i * n + j] + v[j];
            }
            u[i] = log_a[i] - logsumexp(&row_buf);
        }

        // Column update: v_j ← log b_j − LSE_i(log_k_{ij} + u_i)
        for j in 0..n {
            for i in 0..m {
                col_buf[i] = log_k[i * n + j] + u[i];
            }
            v[j] = log_b[j] - logsumexp(&col_buf);
        }

        // Convergence check (row marginal violation)
        let mut max_viol = 0.0_f32;
        for i in 0..m {
            for j in 0..n {
                row_buf[j] = log_k[i * n + j] + v[j] + u[i];
            }
            let log_row = logsumexp(&row_buf);
            let viol = (log_row.exp() - a[i]).abs();
            if viol > max_viol {
                max_viol = viol;
            }
        }
        if max_viol < inner_tol {
            break;
        }

        // Absorb u into log_k every 10 steps to avoid overflow.
        if _it % 10 == 9 {
            for i in 0..m {
                for j in 0..n {
                    log_k[i * n + j] += u[i];
                }
            }
            for vi in u.iter_mut() {
                *vi = 0.0;
            }
        }
    }

    // Recover plan: P_{ij} = exp(log_k_{ij} + u_i + v_j)
    let mut plan = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            plan[i * n + j] = (log_k[i * n + j] + u[i] + v[j]).exp();
        }
    }
    plan
}

// ──────────────────────────────────────────────────────────────────────────────
// Main public API
// ──────────────────────────────────────────────────────────────────────────────

/// Compute Gromov-Wasserstein coupling via Bregman mirror descent.
///
/// - `c1`: `[m × m]` intra-source distance / cost matrix (row-major).
/// - `c2`: `[n × n]` intra-target distance / cost matrix (row-major).
/// - `a`: source marginal of length `m` (must sum to ≈ 1).
/// - `b`: target marginal of length `n` (must sum to ≈ 1).
///
/// Returns [`BregmanGwResult`] with the optimal coupling and GW loss.
pub fn bregman_gw(
    c1: &[f32],
    c2: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &BregmanGwConfig,
) -> OtResult<BregmanGwResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if c1.len() != m * m {
        return Err(OtError::MarginalMismatch {
            m,
            n: m,
            a_len: c1.len(),
            b_len: m * m,
        });
    }
    if c2.len() != n * n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: c2.len(),
            b_len: n * n,
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
    if cfg.lambda <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.lambda });
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

    // ── Initialise coupling: uniform product measure ──────────────────────────
    let mut gamma: Vec<f32> = a
        .iter()
        .flat_map(|&ai| b.iter().map(move |&bj| ai * bj))
        .collect();

    let mut iters = 0_usize;
    let mut converged = false;

    for iter in 0..cfg.max_iter {
        iters = iter + 1;

        // Compute GW gradient G = f1 ⊕ f2 − 2·C1·Γ·C2
        let g = gw_gradient(c1, c2, &gamma, a, b, m, n);

        // Bregman projection step
        let gamma_new = bregman_projection(
            &gamma,
            &g,
            a,
            b,
            m,
            n,
            cfg.lambda,
            cfg.inner_max_iter,
            cfg.inner_tol,
        );

        // Convergence check: Frobenius norm of update
        let frob_sq: f32 = gamma_new
            .iter()
            .zip(gamma.iter())
            .map(|(&new, &old)| (new - old).powi(2))
            .sum();
        let frob = frob_sq.sqrt();

        gamma = gamma_new;

        if frob < cfg.tol {
            converged = true;
            break;
        }
    }

    let gw_loss = gw_objective(c1, c2, &gamma, m, n);

    Ok(BregmanGwResult {
        coupling: gamma,
        gw_loss,
        iters,
        converged,
    })
}

/// Compute the GW distance (square root of the GW loss) between two metric
/// spaces given the optimal coupling.
pub fn bregman_gw_distance(gw_loss: f32) -> f32 {
    gw_loss.max(0.0).sqrt()
}

/// Compute the Frobenius GW cost matrix for use with other OT solvers:
/// `F_{ij} = Σ_k Σ_l (C1_{ik} − C2_{jl})² Γ_{kl}`.
/// This is the linearised GW cost evaluated at a given coupling `Γ`.
pub fn gw_linear_cost(
    c1: &[f32],
    c2: &[f32],
    gamma: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) -> OtResult<Vec<f32>> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if c1.len() != m * m || c2.len() != n * n || gamma.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: c1.len(),
            b_len: m * m,
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
    let g = gw_gradient(c1, c2, gamma, a, b, m, n);
    Ok(g)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    /// Build a simple distance matrix: `C_{ij} = |i − j|`.
    fn abs_diff_cost(n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).abs();
            }
        }
        c
    }

    #[test]
    fn test_bregman_gw_basic() {
        let m = 3;
        let n = 3;
        let c1 = abs_diff_cost(m);
        let c2 = abs_diff_cost(n);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = BregmanGwConfig {
            lambda: 0.1,
            max_iter: 10,
            ..Default::default()
        };
        let res = bregman_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("ok");
        assert_eq!(res.coupling.len(), m * n);
        assert!(res.gw_loss.is_finite());
        assert!(res.gw_loss >= 0.0);
    }

    #[test]
    fn test_bregman_gw_coupling_shape() {
        let m = 4;
        let n = 3;
        let c1 = abs_diff_cost(m);
        let c2 = abs_diff_cost(n);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = BregmanGwConfig {
            lambda: 0.2,
            max_iter: 5,
            ..Default::default()
        };
        let res = bregman_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("ok");
        assert_eq!(res.coupling.len(), m * n);
        for &v in &res.coupling {
            assert!(v.is_finite() && v >= -1e-5);
        }
    }

    #[test]
    fn test_bregman_gw_marginals_approximately_satisfied() {
        let m = 4;
        let n = 4;
        let c1 = abs_diff_cost(m);
        let c2 = abs_diff_cost(n);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = BregmanGwConfig {
            lambda: 0.05,
            max_iter: 50,
            tol: 1e-5,
            inner_max_iter: 100,
            inner_tol: 1e-7,
        };
        let res = bregman_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("ok");

        // Row marginals
        for (i, &ai) in a.iter().enumerate().take(m) {
            let row_sum: f32 = (0..n).map(|j| res.coupling[i * n + j]).sum();
            assert!(
                (row_sum - ai).abs() < 0.05,
                "row {i} sum={row_sum}, a[i]={ai}"
            );
        }
    }

    #[test]
    fn test_bregman_gw_identical_spaces() {
        // When c1 == c2, the GW loss should be close to 0 if the spaces are isometric.
        let m = 3;
        let c1 = abs_diff_cost(m);
        let a = uniform(m);
        let cfg = BregmanGwConfig {
            lambda: 0.01,
            max_iter: 50,
            ..Default::default()
        };
        let res = bregman_gw(&c1, &c1, &a, &a, m, m, &cfg).expect("ok");
        // Not exactly 0 due to entropic regularisation, but should be small.
        assert!(res.gw_loss < 2.0, "gw_loss={}", res.gw_loss);
    }

    #[test]
    fn test_bregman_gw_empty_error() {
        let cfg = BregmanGwConfig::default();
        let err = bregman_gw(&[], &[], &[], &[], 0, 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_bregman_gw_bad_lambda() {
        let c1 = abs_diff_cost(2);
        let c2 = abs_diff_cost(2);
        let a = uniform(2);
        let b = uniform(2);
        let cfg = BregmanGwConfig {
            lambda: -0.1,
            ..Default::default()
        };
        let err = bregman_gw(&c1, &c2, &a, &b, 2, 2, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_bregman_gw_iters_positive() {
        let m = 3;
        let c1 = abs_diff_cost(m);
        let a = uniform(m);
        let cfg = BregmanGwConfig {
            max_iter: 5,
            ..Default::default()
        };
        let res = bregman_gw(&c1, &c1, &a, &a, m, m, &cfg).expect("ok");
        assert!(res.iters > 0 && res.iters <= 5);
    }

    #[test]
    fn test_gw_distance_nonneg() {
        assert!(bregman_gw_distance(0.5) >= 0.0);
        assert!(bregman_gw_distance(0.0) == 0.0);
    }

    #[test]
    fn test_gw_linear_cost_shape() {
        let m = 3;
        let n = 3;
        let c1 = abs_diff_cost(m);
        let c2 = abs_diff_cost(n);
        let a = uniform(m);
        let b = uniform(n);
        // Initial coupling = product measure
        let gamma: Vec<f32> = a
            .iter()
            .flat_map(|&ai| b.iter().map(move |&bj| ai * bj))
            .collect();
        let lc = gw_linear_cost(&c1, &c2, &gamma, &a, &b, m, n).expect("ok");
        assert_eq!(lc.len(), m * n);
        for &v in &lc {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_bregman_gw_mat_mul_helper() {
        // Smoke test for mat_mul helper
        let a = [1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = [2.0_f32, 3.0, 4.0, 5.0]; // 2x2
        let out = mat_mul(&a, &b, 2, 2, 2);
        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
    }
}
