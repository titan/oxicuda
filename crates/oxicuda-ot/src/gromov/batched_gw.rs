//! Batched entropic Gromov-Wasserstein for hyperparameter sweeps,
//! ensembling, and batched graph matching.
//!
//! Standard entropic GW solves a single problem `(C_s, C_t, a, b)` per call.
//! Many practical settings — ensembling over marginals, hyperparameter
//! sweeps, batched graph alignment — require `k` independent GW problems
//! that share the same intra-domain cost matrices `C_s ∈ R^{n×n}` and
//! `C_t ∈ R^{m×m}` but differ in per-batch marginals `(a_b, b_b)`.
//!
//! This module amortises the shared work:
//!
//! 1. Pre-compute `C_s²` and `C_t²` once.
//! 2. The expensive cross term `C_s · T_b · C_t` is per-batch; the cheap
//!    constant terms `(C_s² · 1) aᵀ` and `(C_t² · 1) bᵀ` use the precomputed
//!    squared matrices.
//! 3. Optionally warm-start each batch's inner Sinkhorn with the previous
//!    batch's converged dual potentials so similar marginals converge in
//!    fewer outer iterations.
//!
//! Implementation note on linearised cost: the GW gradient at a current plan
//! `T` is
//!
//! ```text
//! ∇_T L = C_s² · row_sums(T) · 1ᵀ + 1 · col_sums(T)ᵀ · C_t² − 2 · C_s · T · C_t
//! ```
//!
//! and the GW loss is `L(T) = <∇_T L, T> − (constant part absorbed in C_s², C_t²)`.

use crate::error::{OtError, OtResult};

/// Configuration for [`batched_gromov_wasserstein`].
#[derive(Debug, Clone)]
pub struct BatchedGwConfig {
    /// Entropic regularisation strength `ε` for the inner Sinkhorn (must be > 0).
    pub epsilon: f64,
    /// Number of outer mirror-descent / Bregman iterations.
    pub outer_iter: usize,
    /// Maximum inner Sinkhorn iterations per outer step.
    pub inner_iter: usize,
    /// Convergence tolerance on the relative Frobenius change of the plan.
    pub tol: f64,
    /// If true, re-use the previous batch's converged log-potentials as the
    /// warm start for the next batch. Significantly accelerates similar
    /// marginals; harmless when marginals differ.
    pub warm_start: bool,
}

impl Default for BatchedGwConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            outer_iter: 50,
            inner_iter: 200,
            tol: 1e-4,
            warm_start: true,
        }
    }
}

/// Output of [`batched_gromov_wasserstein`].
#[derive(Debug, Clone)]
pub struct BatchedGwResult {
    /// One `n × m` transport plan per batch element (row-major, length `n·m`).
    pub plans: Vec<Vec<f64>>,
    /// Final GW cost per batch.
    pub costs: Vec<f64>,
    /// Outer iterations executed per batch.
    pub outer_iterations: Vec<usize>,
    /// Convergence flag per batch (true ⇔ relative plan change fell below `tol`).
    pub converged: Vec<bool>,
}

#[inline]
fn safe_ln(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

#[inline]
fn logsumexp(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut max_val = f64::NEG_INFINITY;
    for &x in slice {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f64;
    for &x in slice {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

fn validate_inputs(
    cost_source: &[f64],
    n: usize,
    cost_target: &[f64],
    m: usize,
    weights_a: &[Vec<f64>],
    weights_b: &[Vec<f64>],
    cfg: &BatchedGwConfig,
) -> OtResult<usize> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cost_source.len() != n * n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: cost_source.len(),
            b_len: n * n,
        });
    }
    if cost_target.len() != m * m {
        return Err(OtError::MarginalMismatch {
            m,
            n: m,
            a_len: cost_target.len(),
            b_len: m * m,
        });
    }
    let k = weights_a.len();
    if k == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    if weights_b.len() != k {
        return Err(OtError::IncompatibleLength {
            a: weights_a.len(),
            b: weights_b.len(),
        });
    }
    if cfg.epsilon <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.epsilon as f32,
        });
    }
    if cfg.outer_iter == 0 || cfg.inner_iter == 0 {
        return Err(OtError::NotConverged {
            iter: 0,
            tol: cfg.tol as f32,
        });
    }
    if cfg.tol <= 0.0 {
        return Err(OtError::Internal {
            msg: format!(
                "batched_gromov_wasserstein: tol must be > 0, got {}",
                cfg.tol
            ),
        });
    }
    for (b, wa) in weights_a.iter().enumerate() {
        if wa.len() != n {
            return Err(OtError::MarginalMismatch {
                m: n,
                n,
                a_len: wa.len(),
                b_len: n,
            });
        }
        for &v in wa {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
        let _ = b;
    }
    for wb in weights_b {
        if wb.len() != m {
            return Err(OtError::MarginalMismatch {
                m,
                n: m,
                a_len: wb.len(),
                b_len: m,
            });
        }
        for &v in wb {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
    }
    Ok(k)
}

/// Returns `(C + Cᵀ) / 2`. Asymmetric self-cost matrices are symmetrised.
fn symmetrise(c: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = 0.5 * (c[i * k + j] + c[j * k + i]);
        }
    }
    out
}

/// Element-wise square of a matrix-shaped slice.
fn squared_matrix(c: &[f64]) -> Vec<f64> {
    c.iter().map(|x| x * x).collect()
}

/// Compute the gradient linearisation of the GW objective at plan `T`.
///
/// The full linearised cost decomposes into the constant part
/// `cs2_a_i + ct2_b_j` and the cross part `−2 · (C_s · T · C_t)_ij`. The
/// constant additive offset cancels under entropic OT modulo a per-batch
/// constant absorbed in the dual potentials, but we keep it explicit so the
/// resulting `L` is the literal cost used by the Sinkhorn inner solve.
fn linearised_cost(cs2_a: &[f64], n: usize, ct2_b: &[f64], m: usize, cs_t_ct: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * m];
    for (i, &cs_i) in cs2_a.iter().enumerate().take(n) {
        let off = i * m;
        for (j, &ct_j) in ct2_b.iter().enumerate().take(m) {
            out[off + j] = cs_i + ct_j - 2.0 * cs_t_ct[off + j];
        }
    }
    out
}

/// Compute `C_s · T · C_t` via two `O(n²m + nm²)` matrix multiplications.
fn cs_t_ct(cost_source: &[f64], n: usize, cost_target: &[f64], m: usize, plan: &[f64]) -> Vec<f64> {
    let mut tmp = vec![0.0_f64; n * m];
    for i in 0..n {
        let cs_row = i * n;
        let out_row = i * m;
        for k in 0..n {
            let cs_ik = cost_source[cs_row + k];
            if cs_ik == 0.0 {
                continue;
            }
            let pl_row = k * m;
            for l in 0..m {
                tmp[out_row + l] += cs_ik * plan[pl_row + l];
            }
        }
    }
    let mut out = vec![0.0_f64; n * m];
    for i in 0..n {
        let tmp_row = i * m;
        let out_row = i * m;
        for j in 0..m {
            let ct_row = j * m;
            let mut acc = 0.0_f64;
            for l in 0..m {
                acc += tmp[tmp_row + l] * cost_target[ct_row + l];
            }
            out[out_row + j] = acc;
        }
    }
    out
}

/// In-place log-Sinkhorn solve in `f64`. Modifies `f` and `g` (warm-started
/// dual potentials) and writes the plan into `plan`. Returns the iteration
/// count used.
fn log_sinkhorn_inplace(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    eps: f64,
    max_iter: usize,
    tol: f64,
    f: &mut [f64],
    g: &mut [f64],
    plan: &mut [f64],
) -> usize {
    let log_a: Vec<f64> = a.iter().map(|&v| safe_ln(v)).collect();
    let log_b: Vec<f64> = b.iter().map(|&v| safe_ln(v)).collect();
    let mut buf = vec![0.0_f64; n.max(m)];
    let mut completed = 0_usize;
    for it in 0..max_iter {
        let f_prev = f.to_vec();
        for j in 0..m {
            for i in 0..n {
                buf[i] = (f[i] - cost[i * m + j]) / eps;
            }
            let lse = logsumexp(&buf[..n]);
            g[j] = eps * log_b[j] - eps * lse;
        }
        for i in 0..n {
            let row_off = i * m;
            for j in 0..m {
                buf[j] = (g[j] - cost[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf[..m]);
            f[i] = eps * log_a[i] - eps * lse;
        }
        completed = it + 1;
        let mut max_change = 0.0_f64;
        for (a_v, b_v) in f.iter().zip(f_prev.iter()) {
            let d = (a_v - b_v).abs();
            if d > max_change {
                max_change = d;
            }
        }
        if max_change < tol {
            break;
        }
    }
    for (i, &fi) in f.iter().enumerate().take(n) {
        let row_off = i * m;
        for (j, &gj) in g.iter().enumerate().take(m) {
            plan[row_off + j] = ((fi + gj - cost[row_off + j]) / eps).exp();
        }
    }
    completed
}

/// Frobenius norm of the difference of two equal-length slices.
fn frob_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc.sqrt()
}

/// Compute the literal GW cost
/// `L(T) = Σ_{ij} L(C_s, C_t, T)_ij · T_ij = Σ T cs2_a + Σ T ct2_b − 2 <T, C_s T C_t>`.
fn gw_cost(
    cs2_a: &[f64],
    n: usize,
    ct2_b: &[f64],
    m: usize,
    cs_t_ct_mat: &[f64],
    plan: &[f64],
) -> f64 {
    let mut term1 = 0.0_f64;
    let mut term2 = 0.0_f64;
    let mut cross = 0.0_f64;
    for (i, &cs_i) in cs2_a.iter().enumerate().take(n) {
        let off = i * m;
        for (j, &ct_j) in ct2_b.iter().enumerate().take(m) {
            let p = plan[off + j];
            term1 += cs_i * p;
            term2 += ct_j * p;
            cross += cs_t_ct_mat[off + j] * p;
        }
    }
    term1 + term2 - 2.0 * cross
}

/// Solve `k` independent entropic Gromov-Wasserstein problems that share the
/// intra-domain costs `C_s` and `C_t`, batching the precomputable work.
///
/// # Arguments
///
/// * `cost_source` — `n × n` symmetric cost matrix (row-major). If asymmetric,
///   silently symmetrised to `(C + Cᵀ) / 2` before use.
/// * `cost_target` — `m × m` symmetric cost matrix (row-major).
/// * `weights_a` — `k` source marginals, each length `n`.
/// * `weights_b` — `k` target marginals, each length `m`.
///
/// # Returns
///
/// A [`BatchedGwResult`] with one transport plan, GW cost, outer-iteration
/// count, and convergence flag per batch element.
pub fn batched_gromov_wasserstein(
    cost_source: &[f64],
    n: usize,
    cost_target: &[f64],
    m: usize,
    weights_a: &[Vec<f64>],
    weights_b: &[Vec<f64>],
    cfg: &BatchedGwConfig,
) -> OtResult<BatchedGwResult> {
    let k = validate_inputs(cost_source, n, cost_target, m, weights_a, weights_b, cfg)?;

    let cs = symmetrise(cost_source, n);
    let ct = symmetrise(cost_target, m);
    let cs2 = squared_matrix(&cs);
    let ct2 = squared_matrix(&ct);

    let mut plans = Vec::with_capacity(k);
    let mut costs = Vec::with_capacity(k);
    let mut outer_iters = Vec::with_capacity(k);
    let mut converged = Vec::with_capacity(k);

    let mut warm_f: Option<Vec<f64>> = None;
    let mut warm_g: Option<Vec<f64>> = None;

    for batch in 0..k {
        let a = &weights_a[batch];
        let b = &weights_b[batch];

        let cs2_a = simpler_cs2_a(&cs2, a, n);
        let ct2_b = simpler_ct2_b(&ct2, b, m);

        let mut plan = vec![0.0_f64; n * m];
        for (i, &ai) in a.iter().enumerate().take(n) {
            let row = i * m;
            for (j, &bj) in b.iter().enumerate().take(m) {
                plan[row + j] = ai * bj;
            }
        }

        let mut f: Vec<f64> = if cfg.warm_start && warm_f.is_some() {
            warm_f.clone().unwrap()
        } else {
            vec![0.0_f64; n]
        };
        let mut g: Vec<f64> = if cfg.warm_start && warm_g.is_some() {
            warm_g.clone().unwrap()
        } else {
            vec![0.0_f64; m]
        };

        let mut total_outer = 0_usize;
        let mut conv = false;
        let mut last_cs_t_ct = vec![0.0_f64; n * m];
        for it in 0..cfg.outer_iter {
            let ct_t_mat = cs_t_ct(&cs, n, &ct, m, &plan);
            last_cs_t_ct = ct_t_mat.clone();
            let cost = linearised_cost(&cs2_a, n, &ct2_b, m, &ct_t_mat);

            let mut new_plan = vec![0.0_f64; n * m];
            let _used = log_sinkhorn_inplace(
                &cost,
                a,
                b,
                n,
                m,
                cfg.epsilon,
                cfg.inner_iter,
                cfg.tol,
                &mut f,
                &mut g,
                &mut new_plan,
            );

            let denom = frob_norm(&plan).max(1e-12);
            let delta = frob_diff(&plan, &new_plan) / denom;
            plan = new_plan;
            total_outer = it + 1;
            if delta < cfg.tol {
                conv = true;
                break;
            }
        }

        let cs_t_ct_final = if total_outer > 0 {
            last_cs_t_ct
        } else {
            cs_t_ct(&cs, n, &ct, m, &plan)
        };
        let cost = gw_cost(&cs2_a, n, &ct2_b, m, &cs_t_ct_final, &plan);

        if cfg.warm_start {
            warm_f = Some(f.clone());
            warm_g = Some(g.clone());
        }

        plans.push(plan);
        costs.push(cost);
        outer_iters.push(total_outer);
        converged.push(conv);
    }

    Ok(BatchedGwResult {
        plans,
        costs,
        outer_iterations: outer_iters,
        converged,
    })
}

/// `cs2_a[i] = Σ_k (C_s²)_ik · a_k` — the squared-source potential offset.
fn simpler_cs2_a(cs2: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        let row = i * n;
        let mut s = 0.0_f64;
        for (k, &ak) in a.iter().enumerate().take(n) {
            s += cs2[row + k] * ak;
        }
        *slot = s;
    }
    out
}

/// `ct2_b[j] = Σ_l (C_t²)_jl · b_l` — the squared-target potential offset.
fn simpler_ct2_b(ct2: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m];
    for (j, slot) in out.iter_mut().enumerate().take(m) {
        let row = j * m;
        let mut s = 0.0_f64;
        for (l, &bl) in b.iter().enumerate().take(m) {
            s += ct2[row + l] * bl;
        }
        *slot = s;
    }
    out
}

fn frob_norm(a: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for &x in a {
        s += x * x;
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn make_symmetric(values: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; n * n];
        for &(i, j, v) in values {
            c[i * n + j] = v;
            c[j * n + i] = v;
        }
        c
    }

    fn scalar_gw_single_batch(
        cs: &[f64],
        n: usize,
        ct: &[f64],
        m: usize,
        a: &[f64],
        b: &[f64],
        cfg: &BatchedGwConfig,
    ) -> BatchedGwResult {
        batched_gromov_wasserstein(cs, n, ct, m, &[a.to_vec()], &[b.to_vec()], cfg)
            .expect("single batch ok")
    }

    #[test]
    fn single_batch_yields_finite_plan() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)], n);
        let ct = cs.clone();
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let cfg = BatchedGwConfig {
            epsilon: 0.1,
            outer_iter: 50,
            inner_iter: 500,
            tol: 1e-4,
            warm_start: false,
        };
        let res = scalar_gw_single_batch(&cs, n, &ct, m, &a, &b, &cfg);
        assert_eq!(res.plans.len(), 1);
        assert_eq!(res.costs.len(), 1);
        for &p in &res.plans[0] {
            assert!(p.is_finite() && p >= -1e-9);
        }
    }

    #[test]
    fn two_identical_batches_produce_identical_plans() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)], n);
        let ct = cs.clone();
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let cfg = BatchedGwConfig {
            epsilon: 0.1,
            outer_iter: 30,
            inner_iter: 300,
            tol: 1e-4,
            warm_start: true,
        };
        let res = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a.clone(), a.clone()],
            &[b.clone(), b.clone()],
            &cfg,
        )
        .expect("ok");
        for k in 0..res.plans[0].len() {
            assert!(
                approx(res.plans[0][k], res.plans[1][k], 1e-3),
                "plan mismatch at idx {k}: {} vs {}",
                res.plans[0][k],
                res.plans[1][k]
            );
        }
        assert!(approx(res.costs[0], res.costs[1], 1e-4));
    }

    #[test]
    fn warm_start_saves_iterations_on_similar_marginals() {
        let n = 4;
        let m = 4;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], n);
        let ct = cs.clone();
        let a = vec![0.25_f64; 4];
        let b = vec![0.25_f64; 4];
        let cfg_warm = BatchedGwConfig {
            epsilon: 0.1,
            outer_iter: 30,
            inner_iter: 200,
            tol: 1e-3,
            warm_start: true,
        };
        let cfg_cold = BatchedGwConfig {
            warm_start: false,
            ..cfg_warm.clone()
        };
        let res_warm = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a.clone(), a.clone(), a.clone()],
            &[b.clone(), b.clone(), b.clone()],
            &cfg_warm,
        )
        .expect("ok");
        let res_cold = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a.clone(), a.clone(), a.clone()],
            &[b.clone(), b.clone(), b.clone()],
            &cfg_cold,
        )
        .expect("ok");
        let warm_total: usize = res_warm.outer_iterations.iter().sum();
        let cold_total: usize = res_cold.outer_iterations.iter().sum();
        assert!(
            warm_total <= cold_total,
            "warm start should not require more iterations (warm={warm_total}, cold={cold_total})"
        );
    }

    #[test]
    fn per_batch_convergence_flag_recorded() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0)], n);
        let ct = cs.clone();
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let cfg = BatchedGwConfig {
            epsilon: 0.1,
            outer_iter: 200,
            inner_iter: 500,
            tol: 1e-5,
            warm_start: false,
        };
        let res = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            &cfg,
        )
        .expect("ok");
        assert_eq!(res.converged.len(), 1);
        assert_eq!(res.outer_iterations.len(), 1);
        if res.converged[0] {
            assert!(res.outer_iterations[0] < cfg.outer_iter);
        }
    }

    #[test]
    fn invalid_epsilon_rejected() {
        let cs = vec![0.0_f64; 4];
        let ct = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig {
            epsilon: 0.0,
            outer_iter: 10,
            inner_iter: 10,
            tol: 1e-3,
            warm_start: false,
        };
        let r = batched_gromov_wasserstein(
            &cs,
            2,
            &ct,
            2,
            std::slice::from_ref(&a),
            std::slice::from_ref(&a),
            &cfg,
        );
        assert!(matches!(r, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn invalid_outer_iter_rejected() {
        let cs = vec![0.0_f64; 4];
        let ct = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig {
            epsilon: 0.1,
            outer_iter: 0,
            inner_iter: 10,
            tol: 1e-3,
            warm_start: false,
        };
        let r = batched_gromov_wasserstein(
            &cs,
            2,
            &ct,
            2,
            std::slice::from_ref(&a),
            std::slice::from_ref(&a),
            &cfg,
        );
        assert!(matches!(r, Err(OtError::NotConverged { .. })));
    }

    #[test]
    fn dim_mismatch_rejected() {
        let cs = vec![0.0_f64; 3];
        let ct = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig::default();
        let r = batched_gromov_wasserstein(
            &cs,
            2,
            &ct,
            2,
            std::slice::from_ref(&a),
            std::slice::from_ref(&a),
            &cfg,
        );
        assert!(matches!(r, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn batch_count_mismatch_rejected() {
        let cs = vec![0.0_f64; 4];
        let ct = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig::default();
        let r = batched_gromov_wasserstein(
            &cs,
            2,
            &ct,
            2,
            &[a.clone(), a.clone()],
            std::slice::from_ref(&b),
            &cfg,
        );
        assert!(matches!(r, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn marginal_length_mismatch_rejected() {
        let cs = vec![0.0_f64; 4];
        let ct = vec![0.0_f64; 4];
        let a = vec![0.3_f64, 0.3, 0.4];
        let b = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig::default();
        let r = batched_gromov_wasserstein(&cs, 2, &ct, 2, &[a], &[b], &cfg);
        assert!(matches!(r, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn asymmetric_cost_is_symmetrised_silently() {
        let n = 2;
        let m = 2;
        let cs_asym = vec![0.0_f64, 1.0, 0.5, 0.0];
        let cs_sym = vec![0.0_f64, 0.75, 0.75, 0.0];
        let ct = vec![0.0_f64, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig {
            epsilon: 0.5,
            outer_iter: 30,
            inner_iter: 200,
            tol: 1e-4,
            warm_start: false,
        };
        let r1 = batched_gromov_wasserstein(
            &cs_asym,
            n,
            &ct,
            m,
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            &cfg,
        )
        .expect("ok");
        let r2 = batched_gromov_wasserstein(
            &cs_sym,
            n,
            &ct,
            m,
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            &cfg,
        )
        .expect("ok");
        for k in 0..r1.plans[0].len() {
            assert!(
                approx(r1.plans[0][k], r2.plans[0][k], 1e-3),
                "asymmetric must equal pre-symmetrised at idx {k}"
            );
        }
    }

    #[test]
    fn marginals_satisfied_per_batch() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0)], n);
        let ct = cs.clone();
        let a1 = vec![0.5_f64, 0.3, 0.2];
        let b1 = vec![0.4_f64, 0.4, 0.2];
        let a2 = vec![1.0_f64 / 3.0; 3];
        let b2 = vec![1.0_f64 / 3.0; 3];
        let cfg = BatchedGwConfig {
            epsilon: 0.3,
            outer_iter: 30,
            inner_iter: 500,
            tol: 1e-4,
            warm_start: false,
        };
        let res = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a1.clone(), a2.clone()],
            &[b1.clone(), b2.clone()],
            &cfg,
        )
        .expect("ok");
        for (b_idx, (a, b)) in [(a1, b1), (a2, b2)].iter().enumerate() {
            let plan = &res.plans[b_idx];
            for i in 0..n {
                let row: f64 = (0..m).map(|j| plan[i * m + j]).sum();
                assert!(
                    approx(row, a[i], 5e-2),
                    "batch {b_idx} row {i}: {row} vs {}",
                    a[i]
                );
            }
            for j in 0..m {
                let col: f64 = (0..n).map(|i| plan[i * m + j]).sum();
                assert!(
                    approx(col, b[j], 5e-2),
                    "batch {b_idx} col {j}: {col} vs {}",
                    b[j]
                );
            }
        }
    }

    #[test]
    fn output_arrays_have_length_k() {
        let n = 2;
        let m = 2;
        let cs = vec![0.0_f64, 1.0, 1.0, 0.0];
        let ct = vec![0.0_f64, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig {
            epsilon: 0.3,
            outer_iter: 10,
            inner_iter: 100,
            tol: 1e-3,
            warm_start: false,
        };
        let k = 4;
        let weights_a: Vec<Vec<f64>> = (0..k).map(|_| a.clone()).collect();
        let weights_b: Vec<Vec<f64>> = (0..k).map(|_| b.clone()).collect();
        let res =
            batched_gromov_wasserstein(&cs, n, &ct, m, &weights_a, &weights_b, &cfg).expect("ok");
        assert_eq!(res.plans.len(), k);
        assert_eq!(res.costs.len(), k);
        assert_eq!(res.outer_iterations.len(), k);
        assert_eq!(res.converged.len(), k);
    }

    #[test]
    fn warm_vs_cold_converge_to_same_plan() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0)], n);
        let ct = cs.clone();
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let cfg_warm = BatchedGwConfig {
            epsilon: 0.2,
            outer_iter: 80,
            inner_iter: 400,
            tol: 1e-5,
            warm_start: true,
        };
        let cfg_cold = BatchedGwConfig {
            warm_start: false,
            ..cfg_warm.clone()
        };
        let res_warm = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a.clone(), a.clone()],
            &[b.clone(), b.clone()],
            &cfg_warm,
        )
        .expect("ok");
        let res_cold = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            &[a.clone(), a.clone()],
            &[b.clone(), b.clone()],
            &cfg_cold,
        )
        .expect("ok");
        for k in 0..res_warm.plans[1].len() {
            assert!(
                approx(res_warm.plans[1][k], res_cold.plans[1][k], 1e-2),
                "warm vs cold mismatch at idx {k}: {} vs {}",
                res_warm.plans[1][k],
                res_cold.plans[1][k]
            );
        }
    }

    #[test]
    fn small_eps_produces_sharply_peaked_plan() {
        let n = 3;
        let m = 3;
        let cs = make_symmetric(&[(0, 1, 1.0), (1, 2, 1.0)], n);
        let ct = cs.clone();
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let cfg = BatchedGwConfig {
            epsilon: 0.005,
            outer_iter: 30,
            inner_iter: 1500,
            tol: 1e-5,
            warm_start: false,
        };
        let res = batched_gromov_wasserstein(
            &cs,
            n,
            &ct,
            m,
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            &cfg,
        )
        .expect("ok");
        let plan = &res.plans[0];
        let max_p = plan.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_p > 0.2, "expected sharply peaked plan, got max={max_p}");
    }

    #[test]
    fn negative_weights_rejected() {
        let cs = vec![0.0_f64; 4];
        let ct = vec![0.0_f64; 4];
        let a = vec![-0.1_f64, 0.6];
        let b = vec![0.5_f64, 0.5];
        let cfg = BatchedGwConfig::default();
        let r = batched_gromov_wasserstein(&cs, 2, &ct, 2, &[a], &[b], &cfg);
        assert!(matches!(r, Err(OtError::NegativeWeight)));
    }
}
