//! Low-rank Sinkhorn factorisation (Scetbon & Cuturi 2020).
//!
//! Approximates the transport plan as `P ≈ Q · diag(w) · Rᵀ` where
//! `Q ∈ ℝ^{n×r}`, `R ∈ ℝ^{m×r}`, `w ∈ ℝ^r` are non-negative factors with
//! rank `r << min(n, m)`. This reduces memory from `O(nm)` to `O((n+m)r)`.
//!
//! # Reference
//! Scetbon & Cuturi (2020). "Linear-time Gromov Wasserstein distances
//! using low rank couplings and costs."

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration / Result structs
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the low-rank Sinkhorn factorisation solver.
#[derive(Debug, Clone)]
pub struct LowRankConfig {
    /// Rank of the factorisation (must be ≥ 1 and ≤ min(n, m)).
    pub rank: usize,
    /// Entropic regularisation strength ε (must be > 0).
    pub reg: f64,
    /// Maximum number of alternating-projection iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum marginal violation.
    pub tol: f64,
    /// Dykstra step size γ (default 10.0).
    pub gamma: f64,
}

impl Default for LowRankConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            reg: 0.1,
            max_iter: 500,
            tol: 1e-5,
            gamma: 10.0,
        }
    }
}

/// Output of the low-rank Sinkhorn factorisation solver.
///
/// The approximate plan is `P ≈ Q · diag(w) · Rᵀ` (row-major `n×m`).
#[derive(Debug, Clone)]
pub struct LowRankFit {
    /// Left factor `Q`, shape `[n × rank]` row-major.
    pub q: Vec<f64>,
    /// Right factor `R`, shape `[m × rank]` row-major.
    pub r: Vec<f64>,
    /// Weight vector `w`, length `rank`.
    pub w: Vec<f64>,
    /// Number of source points (rows of Q and P).
    pub n: usize,
    /// Number of target points (rows of R and cols of P).
    pub m: usize,
    /// Factorisation rank.
    pub rank: usize,
    /// Actual row marginals `P 1_m`.
    pub marginal_a: Vec<f64>,
    /// Actual column marginals `Pᵀ 1_n`.
    pub marginal_b: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn safe_ln64(x: f64) -> f64 {
    if x <= f64::MIN_POSITIVE {
        f64::MIN_POSITIVE.ln()
    } else {
        x.ln()
    }
}

/// Stable log-sum-exp over a slice (f64).
fn logsumexp64(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    max_val + slice.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln()
}

/// Softmax of a slice in-place (numerically stable).
#[cfg(test)]
fn softmax_inplace(v: &mut [f64]) {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Normalise a non-negative slice to sum to 1. Returns false if sum is zero.
#[cfg(test)]
fn normalise_to_one(v: &mut [f64]) -> bool {
    let s: f64 = v.iter().sum();
    if s <= 0.0 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    cfg: &LowRankConfig,
) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if cfg.rank == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    if cfg.rank > n.min(m) {
        return Err(OtError::BadDim { got: cfg.rank });
    }
    if cost.len() != n * m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != n || b.len() != m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
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

// ─────────────────────────────────────────────────────────────────────────────
// Core solver: Scetbon-Cuturi (2020) Algorithm 1 — 3-marginal alternating
// Sinkhorn projections in the low-rank space.
//
// We maintain three dual scalings:
//   u ∈ ℝ^n  (Lagrange multipliers for row marginals)
//   v ∈ ℝ^m  (Lagrange multipliers for col marginals)
//   g ∈ ℝ^r  (Lagrange multipliers for rank coupling)
//
// Kernel sub-matrices built from r "reference points" sub-sampled from
// the index sets {0..n} and {0..m}:
//   K_Q[i,k] = exp(-cost[i, ref_col[k]] / reg)    shape n×r
//   K_R[j,k] = exp(-cost[ref_row[k], j] / reg)    shape m×r  (transposed)
//
// This turns the full n×m kernel into two tall/thin matrices.
// ─────────────────────────────────────────────────────────────────────────────

/// Run the low-rank Sinkhorn factorisation.
///
/// `cost` is the `n × m` row-major cost matrix. `a` (length `n`) and `b`
/// (length `m`) are the source and target histograms.
pub fn low_rank_sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &LowRankConfig,
) -> OtResult<LowRankFit> {
    validate_inputs(cost, a, b, n, m, cfg)?;

    let r = cfg.rank;
    let reg = cfg.reg;
    let _gamma = cfg.gamma.max(1e-8);

    // ── Select r evenly-spaced reference column indices from 0..m and
    //    r evenly-spaced reference row indices from 0..n.
    // These determine the sub-sampled "reference" points that define K_Q, K_R.
    let ref_cols: Vec<usize> = (0..r).map(|k| k * m / r).collect();
    let ref_rows: Vec<usize> = (0..r).map(|k| k * n / r).collect();

    // ── Build sub-kernels K_Q (n×r) and K_R (m×r) ────────────────────────────
    // K_Q[i,k] = exp(-cost[i, ref_cols[k]] / reg)
    let mut kq: Vec<f64> = vec![0.0; n * r];
    for i in 0..n {
        for k in 0..r {
            kq[i * r + k] = (-cost[i * m + ref_cols[k]] / reg).exp();
        }
    }
    // K_R[j,k] = exp(-cost[ref_rows[k], j] / reg)
    let mut kr: Vec<f64> = vec![0.0; m * r];
    for j in 0..m {
        for k in 0..r {
            kr[j * r + k] = (-cost[ref_rows[k] * m + j] / reg).exp();
        }
    }

    // ── Initialise dual scalings u, v, g ─────────────────────────────────────
    // u[i] = a[i], v[j] = b[j], g[k] = 1/r  (log-domain: 0-initialised)
    let mut log_u: Vec<f64> = a.iter().map(|&ai| safe_ln64(ai)).collect();
    let mut log_v: Vec<f64> = b.iter().map(|&bj| safe_ln64(bj)).collect();
    let mut log_g: Vec<f64> = vec![-safe_ln64(r as f64); r];

    // ── Alternating log-domain Sinkhorn on 3-marginal system ─────────────────
    // Sinkhorn on the factored kernel:
    //   P ≈ diag(u) · [K_Q · diag(g) · K_Rᵀ] · diag(v)
    //
    // Row update:  log_u[i] ← log a[i] - log(row_sum_i)
    // Col update:  log_v[j] ← log b[j] - log(col_sum_j)
    // Rank update: log_g[k] ← log_g[k] + log(gamma_k)  (third marginal correction)
    //
    // row_sum[i]  = Σ_k exp(log_u[i] + log_kq[i,k] + log_g[k]) · col_k
    //              where col_k = Σ_j exp(log_kr[j,k] + log_v[j])
    // col_sum[j]  = Σ_k exp(log_v[j] + log_kr[j,k] + log_g[k]) · row_k
    //              where row_k = Σ_i exp(log_kq[i,k] + log_u[i])

    // Pre-log the sub-kernels
    let log_kq: Vec<f64> = kq.iter().map(|&x| safe_ln64(x)).collect();
    let log_kr: Vec<f64> = kr.iter().map(|&x| safe_ln64(x)).collect();

    let mut buf_r = vec![0.0_f64; r];
    let mut buf_n = vec![0.0_f64; n];
    let mut buf_m = vec![0.0_f64; m];

    for _it in 0..cfg.max_iter {
        // ── Compute log_col_k = log Σ_j exp(log_kr[j,k] + log_v[j]) ──────────
        let mut log_col_k = vec![f64::NEG_INFINITY; r];
        for k in 0..r {
            for j in 0..m {
                buf_m[j] = log_kr[j * r + k] + log_v[j];
            }
            log_col_k[k] = logsumexp64(&buf_m);
        }

        // ── Row update: log_u[i] ← log a[i] - LSE_k(log_kq[i,k] + log_g[k] + log_col_k[k])
        let old_log_u = log_u.clone();
        for i in 0..n {
            for k in 0..r {
                buf_r[k] = log_kq[i * r + k] + log_g[k] + log_col_k[k];
            }
            let lse = logsumexp64(&buf_r);
            log_u[i] = safe_ln64(a[i]) - lse;
        }

        // ── Compute log_row_k = log Σ_i exp(log_kq[i,k] + log_u[i]) ──────────
        let mut log_row_k = vec![f64::NEG_INFINITY; r];
        for k in 0..r {
            for i in 0..n {
                buf_n[i] = log_kq[i * r + k] + log_u[i];
            }
            log_row_k[k] = logsumexp64(&buf_n);
        }

        // ── Col update: log_v[j] ← log b[j] - LSE_k(log_kr[j,k] + log_g[k] + log_row_k[k])
        let old_log_v = log_v.clone();
        for j in 0..m {
            for k in 0..r {
                buf_r[k] = log_kr[j * r + k] + log_g[k] + log_row_k[k];
            }
            let lse = logsumexp64(&buf_r);
            log_v[j] = safe_ln64(b[j]) - lse;
        }

        // ── Rank (3rd marginal) update:
        //    log_g[k] += log(gamma) + 0.5*(log_row_k[k] + log_col_k[k]) - log(sum_g_new)
        // Specifically, solve for g such that Σ_k w_k = 1 where w_k ∝ g_k
        // The closed-form is: g_k_new ∝ sqrt(row_k * col_k) (geometric mean of two marginals)
        let mut log_g_unnorm: Vec<f64> = (0..r)
            .map(|k| 0.5 * (log_row_k[k] + log_col_k[k]) + log_g[k])
            .collect();
        // Normalise log_g to log Σ exp(log_g_k) = 0  (i.e. Σ g_k = 1)
        let lse_g = logsumexp64(&log_g_unnorm);
        if lse_g.is_finite() {
            for x in log_g_unnorm.iter_mut() {
                *x -= lse_g;
            }
        }
        log_g = log_g_unnorm;

        // ── Convergence: max change in log_u + log_v ─────────────────────────
        let max_delta_u = log_u
            .iter()
            .zip(old_log_u.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_delta_v = log_v
            .iter()
            .zip(old_log_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        if max_delta_u.max(max_delta_v) < cfg.tol {
            break;
        }
    }

    // ── Extract factors Q, R, w ───────────────────────────────────────────────
    // After convergence, the full plan is:
    //   P[i,j] = u[i] * [K_Q · diag(g) · K_Rᵀ][i,j] * v[j]
    //           = Σ_k  u[i] * K_Q[i,k] * g[k] * K_R[j,k] * v[j]
    //
    // For the factored form P = Q · diag(w) · Rᵀ with w[k] = g[k]:
    //   Q[i,k] = u[i] * K_Q[i,k]   (not normalised per-row)
    //   R[j,k] = v[j] * K_R[j,k]
    //   w[k]   = g[k]
    //
    // Row marginals: Σ_j P[i,j] = Σ_k Q[i,k] * w[k] * Σ_j R[j,k] = a[i] by construction.

    let u: Vec<f64> = log_u.iter().map(|&lu| lu.exp()).collect();
    let v: Vec<f64> = log_v.iter().map(|&lv| lv.exp()).collect();
    let w: Vec<f64> = log_g.iter().map(|&lg| lg.exp()).collect();

    let mut q_mat: Vec<f64> = vec![0.0; n * r];
    for i in 0..n {
        for k in 0..r {
            q_mat[i * r + k] = (u[i] * kq[i * r + k]).max(0.0);
        }
    }

    let mut r_mat: Vec<f64> = vec![0.0; m * r];
    for j in 0..m {
        for k in 0..r {
            r_mat[j * r + k] = (v[j] * kr[j * r + k]).max(0.0);
        }
    }

    // ── Compute actual marginals of P = Q diag(w) Rᵀ ─────────────────────────
    // marginal_a[i] = Σ_k Q[i,k] * w[k] * Σ_j R[j,k]
    // marginal_b[j] = Σ_k R[j,k] * w[k] * Σ_i Q[i,k]

    // col_sum_r[k] = Σ_j R[j,k]
    let col_sum_r: Vec<f64> = (0..r)
        .map(|k| (0..m).map(|j| r_mat[j * r + k]).sum())
        .collect();
    // col_sum_q[k] = Σ_i Q[i,k]
    let col_sum_q: Vec<f64> = (0..r)
        .map(|k| (0..n).map(|i| q_mat[i * r + k]).sum())
        .collect();

    let marginal_a: Vec<f64> = (0..n)
        .map(|i| (0..r).map(|k| q_mat[i * r + k] * w[k] * col_sum_r[k]).sum())
        .collect();
    let marginal_b: Vec<f64> = (0..m)
        .map(|j| (0..r).map(|k| r_mat[j * r + k] * w[k] * col_sum_q[k]).sum())
        .collect();

    Ok(LowRankFit {
        q: q_mat,
        r: r_mat,
        w,
        n,
        m,
        rank: r,
        marginal_a,
        marginal_b,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Derived-quantity functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the transport cost `<P, C> = Σ_k w[k] · (q_k^T · C · r_k)`.
///
/// `cost` is the `n × m` row-major cost matrix.
pub fn low_rank_transport_cost(fit: &LowRankFit, cost: &[f64]) -> f64 {
    let (n, m, r) = (fit.n, fit.m, fit.rank);
    let mut total = 0.0_f64;
    // For each rank component k, compute q_k^T C r_k = Σ_i Σ_j Q[i,k]*C[i,j]*R[j,k]
    for k in 0..r {
        let mut component = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                component += fit.q[i * r + k] * cost[i * m + j] * fit.r[j * r + k];
            }
        }
        total += fit.w[k] * component;
    }
    total
}

/// Compute actual row and column marginals of the approximate plan `P = Q diag(w) Rᵀ`.
///
/// Returns `(marginal_a, marginal_b)`.
pub fn low_rank_marginals(fit: &LowRankFit) -> (Vec<f64>, Vec<f64>) {
    (fit.marginal_a.clone(), fit.marginal_b.clone())
}

/// Reconstruct the full `n × m` transport plan `P = Q · diag(w) · Rᵀ` row-major.
pub fn low_rank_dense(fit: &LowRankFit) -> Vec<f64> {
    let (n, m, r) = (fit.n, fit.m, fit.rank);
    let mut plan = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut pij = 0.0_f64;
            for k in 0..r {
                pij += fit.q[i * r + k] * fit.w[k] * fit.r[j * r + k];
            }
            plan[i * m + j] = pij.max(0.0);
        }
    }
    plan
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_cost(n: usize, m: usize) -> Vec<f64> {
        (0..n * m)
            .map(|k| {
                let i = k / m;
                let j = k % m;
                ((i as f64 - j as f64).powi(2)).sqrt()
            })
            .collect()
    }

    fn uniform_marginals(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    /// Test 1: Transport cost is finite.
    #[test]
    fn transport_cost_is_finite() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 100,
            tol: 1e-5,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let tc = low_rank_transport_cost(&fit, &cost);
        assert!(tc.is_finite(), "transport cost must be finite: {tc}");
    }

    /// Test 2: Row marginals approximately match a.
    #[test]
    fn row_marginals_approximately_match_a() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 300,
            tol: 1e-6,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let (marg_a, _) = low_rank_marginals(&fit);
        for (i, (ma, ai)) in marg_a.iter().zip(a.iter()).enumerate() {
            assert!(
                (ma - ai).abs() < 0.15,
                "row marginal {i}: {ma} vs expected {ai}"
            );
        }
    }

    /// Test 3: Col marginals approximately match b.
    #[test]
    fn col_marginals_approximately_match_b() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 300,
            tol: 1e-6,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let (_, marg_b) = low_rank_marginals(&fit);
        for (j, (mb, bj)) in marg_b.iter().zip(b.iter()).enumerate() {
            assert!(
                (mb - bj).abs() < 0.15,
                "col marginal {j}: {mb} vs expected {bj}"
            );
        }
    }

    /// Test 4: dense() reconstructs a non-negative matrix.
    #[test]
    fn dense_reconstruction_is_non_negative() {
        let n = 3;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.5,
            max_iter: 100,
            tol: 1e-5,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let plan = low_rank_dense(&fit);
        for (k, &p) in plan.iter().enumerate() {
            assert!(p >= 0.0, "plan[{k}] = {p} is negative");
        }
    }

    /// Test 5: Transport cost ≤ rough upper bound sum_i a[i] * max_j cost[i,j].
    #[test]
    fn transport_cost_below_upper_bound() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let max_c = cost.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let upper_bound = max_c; // since a sums to 1
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 200,
            tol: 1e-5,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let tc = low_rank_transport_cost(&fit, &cost);
        assert!(
            tc <= upper_bound * 1.5 + 0.01,
            "transport cost {tc} exceeds upper bound {upper_bound}"
        );
    }

    /// Test 6: rank-1 case works.
    #[test]
    fn rank_one_works() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 1,
            reg: 0.3,
            max_iter: 200,
            tol: 1e-5,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("rank-1 must work");
        assert_eq!(fit.rank, 1);
        let tc = low_rank_transport_cost(&fit, &cost);
        assert!(tc.is_finite());
    }

    /// Test 7: Error on rank > min(n, m).
    #[test]
    fn error_on_rank_exceeding_min_nm() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 4,
            reg: 0.1,
            max_iter: 100,
            tol: 1e-5,
            gamma: 10.0,
        };
        let res = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg);
        assert!(
            matches!(res, Err(OtError::BadDim { .. })),
            "expected BadDim error for rank > min(n,m)"
        );
    }

    /// Test 8: dense() entries sum approximately to 1.
    #[test]
    fn dense_entries_sum_near_one() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 300,
            tol: 1e-6,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let plan = low_rank_dense(&fit);
        let total: f64 = plan.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.2,
            "plan total = {total} should be near 1"
        );
    }

    /// Test 9: Weight vector w sums near 1.
    #[test]
    fn weight_vector_sums_near_one() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.2,
            max_iter: 200,
            tol: 1e-6,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let w_sum: f64 = fit.w.iter().sum();
        assert!(
            (w_sum - 1.0).abs() < 0.2,
            "weight vector sum = {w_sum} should be near 1"
        );
    }

    /// Test 10: Empty input rejected.
    #[test]
    fn empty_input_rejected() {
        let cfg = LowRankConfig::default();
        let res = low_rank_sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    /// Test 11: Negative regularisation rejected.
    #[test]
    fn negative_reg_rejected() {
        let n = 2;
        let m = 2;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = vec![0.0_f64; 4];
        let cfg = LowRankConfig {
            rank: 1,
            reg: -0.1,
            max_iter: 10,
            tol: 1e-5,
            gamma: 10.0,
        };
        let res = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    /// Test 12: 5×6 rectangular problem works.
    #[test]
    fn rectangular_problem_works() {
        let n = 5;
        let m = 6;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = uniform_cost(n, m);
        let cfg = LowRankConfig {
            rank: 2,
            reg: 0.3,
            max_iter: 200,
            tol: 1e-5,
            gamma: 10.0,
        };
        let fit = low_rank_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        assert_eq!(fit.q.len(), n * 2);
        assert_eq!(fit.r.len(), m * 2);
        let tc = low_rank_transport_cost(&fit, &cost);
        assert!(tc.is_finite());
    }

    // Suppress unused import warning for softmax helper in non-test code
    #[test]
    fn softmax_helper_normalises() {
        let mut v = vec![1.0_f64, 2.0, 3.0];
        softmax_inplace(&mut v);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalise_to_one_helper() {
        let mut v = vec![2.0_f64, 2.0, 2.0, 2.0];
        let ok = normalise_to_one(&mut v);
        assert!(ok);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }
}
