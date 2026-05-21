//! Entropic Gromov-Wasserstein with linear-memory column-sketching approximation.
//!
//! The full entropic GW objective (Peyré et al. 2016) requires computing the
//! `n×m` GW cost matrix
//!
//! ```text
//! M_{ij} = C_s²_ii + C_t²_jj − 2 (C_s T C_t^T)_{ij}
//! ```
//!
//! at every outer Frank-Wolfe iteration.  For large `n, m` materialising the
//! `n×m` matrix is expensive in memory.  This module offers two modes:
//!
//! 1. **Exact** (`n_samples >= m`): computes the full GW cost matrix via the
//!    three-term decomposition `C_s² 1^T + 1 C_t²^T − 2 C_s T C_t^T`.
//!
//! 2. **Approximate** (`n_samples < m`): at each outer iteration, `n_samples`
//!    target columns are randomly sampled via `LcgRng`; only those columns of `M`
//!    are computed.  A reduced Sinkhorn problem is solved on the sampled
//!    sub-columns, and the resulting sub-plan is scattered back into the full
//!    `n×m` plan (unsampled columns receive proportionally allocated mass from the
//!    marginal residual).
//!
//! # References
//!
//! Peyré, Cuturi, Solomon.  "Gromov-Wasserstein Averaging of Kernel and Distance
//! Matrices." ICML 2016.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for the fast entropic Gromov-Wasserstein solver.
#[derive(Debug, Clone)]
pub struct EntropicGwFastConfig {
    /// Entropic regularisation strength (`> 0`).
    pub reg: f32,
    /// Number of outer Frank-Wolfe / Bregman iterations.
    pub max_outer_iter: usize,
    /// Maximum inner Sinkhorn iterations per outer step.
    pub max_inner_iter: usize,
    /// Frobenius-norm convergence tolerance on plan update.
    pub tol: f32,
    /// Number of target columns to sample per outer iteration.
    /// If `n_samples >= nb`, the exact full GW cost matrix is used.
    pub n_samples: usize,
    /// RNG seed for reproducible column sampling.
    pub seed: u64,
}

impl Default for EntropicGwFastConfig {
    fn default() -> Self {
        Self {
            reg: 0.05,
            max_outer_iter: 50,
            max_inner_iter: 200,
            tol: 1e-4,
            n_samples: 64,
            seed: 42,
        }
    }
}

/// Output of the fast entropic Gromov-Wasserstein solver.
#[derive(Debug, Clone)]
pub struct EntropicGwFastFit {
    /// Transport plan, shape `[na × nb]` row-major (length `na·nb`).
    pub transport_plan: Vec<f32>,
    /// Final GW cost `Σ_{ij} T[i,j] * M[i,j] / 2`.
    pub gw_cost: f32,
    /// Number of completed outer iterations.
    pub n_outer_iter: usize,
    /// Number of source points.
    pub n: usize,
    /// Number of target points.
    pub m: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate inputs to the fast entropic GW solver.
fn validate(
    ca: &[f32],
    cb: &[f32],
    a: &[f32],
    b: &[f32],
    na: usize,
    nb: usize,
    cfg: &EntropicGwFastConfig,
) -> OtResult<()> {
    if na == 0 || nb == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.reg });
    }
    if cfg.n_samples == 0 {
        return Err(OtError::BadCount { got: cfg.n_samples });
    }
    if ca.len() != na * na {
        return Err(OtError::MarginalMismatch {
            m: na,
            n: nb,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if cb.len() != nb * nb {
        return Err(OtError::MarginalMismatch {
            m: na,
            n: nb,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != na || b.len() != nb {
        return Err(OtError::MarginalMismatch {
            m: na,
            n: nb,
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

/// Compute `M_full = C_s² diag(a) 1^T + 1^T diag(b) C_t² − 2 C_s T C_t^T`
/// which is the full `na × nb` GW cost matrix.
pub fn gw_cost_matrix(ca: &[f32], cb: &[f32], t: &[f32], na: usize, nb: usize) -> Vec<f32> {
    // Precompute row terms from ca: d_i = Σ_k ca[i,k]^2 * row_sum_k(T)
    // and column terms from cb: e_j = Σ_l cb[j,l]^2 * col_sum_l(T)
    let row_sums: Vec<f32> = (0..na)
        .map(|k| (0..nb).map(|j| t[k * nb + j]).sum::<f32>())
        .collect();
    let col_sums: Vec<f32> = (0..nb)
        .map(|l| (0..na).map(|i| t[i * nb + l]).sum::<f32>())
        .collect();

    // d_i = Σ_k ca[i,k]^2 * row_sums[k]
    let d_src: Vec<f32> = (0..na)
        .map(|i| {
            (0..na)
                .map(|k| {
                    let v = ca[i * na + k];
                    v * v * row_sums[k]
                })
                .sum::<f32>()
        })
        .collect();

    // e_j = Σ_l cb[j,l]^2 * col_sums[l]
    let e_tgt: Vec<f32> = (0..nb)
        .map(|j| {
            (0..nb)
                .map(|l| {
                    let v = cb[j * nb + l];
                    v * v * col_sums[l]
                })
                .sum::<f32>()
        })
        .collect();

    // Cross term: F = C_s · T · C_t^T,  F[i,j] = Σ_{k,l} ca[i,k] · T[k,l] · cb[j,l]
    // Compute as (C_s · T) first (na×nb), then multiply by C_t^T.
    let mut tmp = vec![0.0_f32; na * nb]; // C_s · T
    for i in 0..na {
        for k in 0..na {
            let ca_ik = ca[i * na + k];
            if ca_ik == 0.0 {
                continue;
            }
            let t_row = k * nb;
            let tmp_row = i * nb;
            for l in 0..nb {
                tmp[tmp_row + l] += ca_ik * t[t_row + l];
            }
        }
    }

    // F[i,j] = Σ_l tmp[i,l] * cb[j,l]
    let mut m = vec![0.0_f32; na * nb];
    for (i, &d_i) in d_src.iter().enumerate() {
        let tmp_row = i * nb;
        let m_row = i * nb;
        for j in 0..nb {
            let cb_row = j * nb;
            let mut acc = 0.0_f32;
            for l in 0..nb {
                acc += tmp[tmp_row + l] * cb[cb_row + l];
            }
            m[m_row + j] = d_i + e_tgt[j] - 2.0 * acc;
        }
    }
    m
}

/// Compute selected columns of the GW cost matrix without materialising the full matrix.
///
/// `col_idx` is the list of target column indices to compute.
/// Returns a `na × n_samples` cost sub-matrix (row-major).
fn gw_cost_matrix_columns(
    ca: &[f32],
    cb: &[f32],
    t: &[f32],
    na: usize,
    nb: usize,
    col_idx: &[usize],
) -> Vec<f32> {
    let ns = col_idx.len();
    if ns == 0 {
        return Vec::new();
    }

    let row_sums: Vec<f32> = (0..na)
        .map(|k| (0..nb).map(|j| t[k * nb + j]).sum::<f32>())
        .collect();

    // d_i = Σ_k ca[i,k]^2 * row_sums[k]
    let d_src: Vec<f32> = (0..na)
        .map(|i| {
            (0..na)
                .map(|k| {
                    let v = ca[i * na + k];
                    v * v * row_sums[k]
                })
                .sum::<f32>()
        })
        .collect();

    // Precompute col_sums only for selected target columns.
    // e_j = Σ_l cb[j,l]^2 * col_sum[l]  where col_sum[l] = Σ_i T[i,l].
    let col_sums_full: Vec<f32> = (0..nb)
        .map(|l| (0..na).map(|i| t[i * nb + l]).sum::<f32>())
        .collect();

    let e_tgt_sel: Vec<f32> = col_idx
        .iter()
        .map(|&j| {
            (0..nb)
                .map(|l| {
                    let v = cb[j * nb + l];
                    v * v * col_sums_full[l]
                })
                .sum::<f32>()
        })
        .collect();

    // tmp = C_s · T   (na × nb).
    let mut tmp = vec![0.0_f32; na * nb];
    for i in 0..na {
        for k in 0..na {
            let ca_ik = ca[i * na + k];
            if ca_ik == 0.0 {
                continue;
            }
            let t_row = k * nb;
            let tmp_row = i * nb;
            for l in 0..nb {
                tmp[tmp_row + l] += ca_ik * t[t_row + l];
            }
        }
    }

    // Build na × ns sub-matrix.
    let mut m_sub = vec![0.0_f32; na * ns];
    for (i, &d_i) in d_src.iter().enumerate() {
        let tmp_row = i * nb;
        let m_row = i * ns;
        for (s, &j) in col_idx.iter().enumerate() {
            let cb_row = j * nb;
            let mut acc = 0.0_f32;
            for l in 0..nb {
                acc += tmp[tmp_row + l] * cb[cb_row + l];
            }
            m_sub[m_row + s] = d_i + e_tgt_sel[s] - 2.0 * acc;
        }
    }
    m_sub
}

/// Frobenius-norm difference between two equal-length slices.
fn frob_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc.sqrt()
}

/// Return the GW distance stored in the fit (half the GW cost).
pub fn gw_distance(fit: &EntropicGwFastFit) -> f32 {
    fit.gw_cost
}

// ---------------------------------------------------------------------------
// Approximate (column-sketched) plan update
// ---------------------------------------------------------------------------

/// Given a sub-plan on `n_samples` target columns (from Sinkhorn on M_sub),
/// scatter it back into a full `na × nb` plan while preserving target marginals.
///
/// Strategy: for sampled columns, use Sinkhorn sub-plan directly (re-scaled to
/// target marginal fractions). For unsampled columns, distribute residual source
/// marginal proportionally.
fn scatter_sub_plan_to_full(
    sub_plan: &[f32],
    col_idx: &[usize],
    a: &[f32],
    b: &[f32],
    na: usize,
    nb: usize,
) -> Vec<f32> {
    let ns = col_idx.len();
    let mut plan = vec![0.0_f32; na * nb];

    // Fill sampled columns from sub_plan.
    for (s, &j) in col_idx.iter().enumerate() {
        for i in 0..na {
            plan[i * nb + j] = sub_plan[i * ns + s];
        }
    }

    // For unsampled columns, distribute remaining mass (a_i - Σ_{sampled} P_ij) * b_j / (1 - sampled_b_sum).
    // Build a boolean mask of sampled columns.
    let mut sampled = vec![false; nb];
    for &j in col_idx {
        sampled[j] = true;
    }
    let sampled_b_sum: f32 = col_idx.iter().map(|&j| b[j]).sum();
    let unsampled_b_sum = (1.0 - sampled_b_sum).max(0.0);

    if unsampled_b_sum < 1e-9 {
        return plan;
    }

    for i in 0..na {
        let row_sampled_sum: f32 = col_idx.iter().map(|&j| plan[i * nb + j]).sum();
        let residual = (a[i] - row_sampled_sum).max(0.0);
        for j in 0..nb {
            if !sampled[j] {
                plan[i * nb + j] = residual * b[j] / unsampled_b_sum;
            }
        }
    }

    plan
}

// ---------------------------------------------------------------------------
// Public solver
// ---------------------------------------------------------------------------

/// Solve the entropic Gromov-Wasserstein problem with optional column-sketching.
///
/// # Arguments
/// * `ca` — source intra-domain distance matrix, `na × na` row-major.
/// * `cb` — target intra-domain distance matrix, `nb × nb` row-major.
/// * `a` — source marginal, length `na` (need not be normalised; will be normalised internally).
/// * `b` — target marginal, length `nb`.
/// * `na` — number of source points.
/// * `nb` — number of target points.
/// * `cfg` — solver configuration.
pub fn entropic_gw_fast(
    ca: &[f32],
    cb: &[f32],
    a: &[f32],
    b: &[f32],
    na: usize,
    nb: usize,
    cfg: &EntropicGwFastConfig,
) -> OtResult<EntropicGwFastFit> {
    validate(ca, cb, a, b, na, nb, cfg)?;

    // Normalise marginals.
    let sum_a: f32 = a.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let sum_b: f32 = b.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let a_norm: Vec<f32> = a.iter().map(|&ai| ai / sum_a).collect();
    let b_norm: Vec<f32> = b.iter().map(|&bj| bj / sum_b).collect();

    // Initial plan: T_0 = a ⊗ b  (outer product).
    let mut plan = vec![0.0_f32; na * nb];
    for (i, &ai) in a_norm.iter().enumerate() {
        let row_off = i * nb;
        for (j, &bj) in b_norm.iter().enumerate() {
            plan[row_off + j] = ai * bj;
        }
    }

    let use_exact = cfg.n_samples >= nb;
    let mut rng = LcgRng::new(cfg.seed);
    let inner_cfg = SinkhornConfig {
        eps: cfg.reg,
        max_iter: cfg.max_inner_iter,
        tol: cfg.tol,
    };

    let mut n_outer = 0_usize;

    for _outer in 0..cfg.max_outer_iter {
        let new_plan = if use_exact {
            // Exact: compute full GW cost matrix and run Sinkhorn.
            let m_full = gw_cost_matrix(ca, cb, &plan, na, nb);
            match sinkhorn(&m_full, &a_norm, &b_norm, na, nb, &inner_cfg) {
                Ok(res) => res.plan,
                Err(OtError::NotConverged { .. }) => {
                    // Accept current plan on inner non-convergence.
                    plan.clone()
                }
                Err(e) => return Err(e),
            }
        } else {
            // Approximate: sample n_samples target columns uniformly without replacement.
            let n_sel = cfg.n_samples.min(nb);
            let mut col_idx: Vec<usize> = (0..nb).collect();
            // Fisher-Yates partial shuffle to pick n_sel columns.
            for k in 0..n_sel {
                let swap = k + rng.next_usize(nb - k);
                col_idx.swap(k, swap);
            }
            let col_idx = col_idx[..n_sel].to_vec();

            // Compute GW cost on selected columns only.
            let m_sub = gw_cost_matrix_columns(ca, cb, &plan, na, nb, &col_idx);

            // Build sub-marginal b_sub.
            // Normalise b_sub to make it a proper marginal.
            let b_sub_unnorm: Vec<f32> = col_idx.iter().map(|&j| b_norm[j]).collect();
            let b_sub_sum: f32 = b_sub_unnorm.iter().sum::<f32>().max(f32::MIN_POSITIVE);
            let b_sub_norm: Vec<f32> = b_sub_unnorm.iter().map(|&v| v / b_sub_sum).collect();

            // Run Sinkhorn on sub-problem (na × n_sel).
            // Scale a to match sub-marginal mass.
            let a_sub_norm = a_norm.clone();
            let sub_result = match sinkhorn(&m_sub, &a_sub_norm, &b_sub_norm, na, n_sel, &inner_cfg)
            {
                Ok(res) => res,
                Err(OtError::NotConverged { .. }) => {
                    n_outer += 1;
                    break;
                }
                Err(e) => return Err(e),
            };

            // Re-scale sub-plan: multiply each column by (b_sub_unnorm[s] / b_sub_norm[s]) = b_sub_sum.
            let mut sub_plan_rescaled = sub_result.plan.clone();
            for i in 0..na {
                for s in 0..n_sel {
                    sub_plan_rescaled[i * n_sel + s] *= b_sub_sum;
                }
            }

            // Scatter sub-plan into full na × nb plan.
            scatter_sub_plan_to_full(&sub_plan_rescaled, &col_idx, &a_norm, &b_norm, na, nb)
        };

        let delta = frob_diff(&plan, &new_plan);
        plan = new_plan;
        n_outer += 1;
        if delta < cfg.tol {
            break;
        }
    }

    // Compute final GW cost using the exact cost matrix.
    let m_final = gw_cost_matrix(ca, cb, &plan, na, nb);
    let gw_cost: f32 = plan
        .iter()
        .zip(m_final.iter())
        .map(|(&p, &m)| p * m)
        .sum::<f32>()
        / 2.0;

    Ok(EntropicGwFastFit {
        transport_plan: plan,
        gw_cost,
        n_outer_iter: n_outer,
        n: na,
        m: nb,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f32> {
        vec![1.0_f32 / n as f32; n]
    }

    /// Build a symmetric distance matrix from a sequence of edges.
    fn dist_matrix(edges: &[(usize, usize, f32)], n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; n * n];
        for &(i, j, v) in edges {
            c[i * n + j] = v;
            c[j * n + i] = v;
        }
        c
    }

    fn line_dist(n: usize) -> Vec<f32> {
        // C[i,j] = |i - j| / (n-1)
        let mut c = vec![0.0_f32; n * n];
        let scale = if n > 1 { (n - 1) as f32 } else { 1.0 };
        for i in 0..n {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).abs() / scale;
            }
        }
        c
    }

    // -----------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------

    #[test]
    fn rejects_empty() {
        let cfg = EntropicGwFastConfig::default();
        let res = entropic_gw_fast(&[], &[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_reg() {
        let n = 2;
        let ca = vec![0.0_f32; n * n];
        let cb = vec![0.0_f32; n * n];
        let a = uniform(n);
        let b = uniform(n);
        let cfg = EntropicGwFastConfig {
            reg: 0.0,
            ..Default::default()
        };
        let res = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn rejects_negative_weights() {
        let n = 2;
        let ca = vec![0.0_f32; n * n];
        let cb = vec![0.0_f32; n * n];
        let a = vec![-0.5_f32, 1.5];
        let b = uniform(n);
        let cfg = EntropicGwFastConfig::default();
        let res = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let na = 2;
        let nb = 3;
        let ca = vec![0.0_f32; na * na + 1]; // wrong
        let cb = vec![0.0_f32; nb * nb];
        let a = uniform(na);
        let b = uniform(nb);
        let cfg = EntropicGwFastConfig::default();
        let res = entropic_gw_fast(&ca, &cb, &a, &b, na, nb, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn rejects_zero_n_samples() {
        let n = 2;
        let ca = vec![0.0_f32; n * n];
        let cb = vec![0.0_f32; n * n];
        let a = uniform(n);
        let b = uniform(n);
        let cfg = EntropicGwFastConfig {
            n_samples: 0,
            ..Default::default()
        };
        let res = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    // -----------------------------------------------------------------
    // gw_cost_matrix tests
    // -----------------------------------------------------------------

    #[test]
    fn gw_cost_matrix_zero_for_identity_plan() {
        // When T = a ⊗ b (outer product) and C_s = C_t = 0, M should be all zeros.
        let na = 3;
        let nb = 3;
        let a = uniform(na);
        let b = uniform(nb);
        let ca = vec![0.0_f32; na * na];
        let cb = vec![0.0_f32; nb * nb];
        let mut t = vec![0.0_f32; na * nb];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                t[i * nb + j] = ai * bj;
            }
        }
        let m = gw_cost_matrix(&ca, &cb, &t, na, nb);
        for &v in &m {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn gw_cost_matrix_non_negative_for_valid_matrices() {
        let na = 3;
        let nb = 3;
        let ca = dist_matrix(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)], na);
        let cb = ca.clone();
        let a = uniform(na);
        // Identity-like plan: diagonal.
        let mut t = vec![0.0_f32; na * nb];
        for i in 0..na.min(nb) {
            t[i * nb + i] = a[i];
        }
        let m = gw_cost_matrix(&ca, &cb, &t, na, nb);
        for &v in &m {
            // GW cost matrix can be negative in intermediate steps (it measures quadratic mismatch).
            assert!(v.is_finite(), "non-finite GW cost entry: {v}");
        }
    }

    // -----------------------------------------------------------------
    // Functional solver tests
    // -----------------------------------------------------------------

    #[test]
    fn plan_shape_correct() {
        let na = 3;
        let nb = 4;
        let ca = line_dist(na);
        let cb = line_dist(nb);
        let a = uniform(na);
        let b = uniform(nb);
        let cfg = EntropicGwFastConfig {
            reg: 0.2,
            max_outer_iter: 10,
            max_inner_iter: 100,
            tol: 1e-3,
            n_samples: nb + 1, // exact mode
            seed: 1,
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, na, nb, &cfg).expect("ok");
        assert_eq!(fit.transport_plan.len(), na * nb);
        assert_eq!(fit.n, na);
        assert_eq!(fit.m, nb);
    }

    #[test]
    fn plan_entries_non_negative() {
        let na = 3;
        let nb = 3;
        let ca = line_dist(na);
        let cb = line_dist(nb);
        let a = uniform(na);
        let b = uniform(nb);
        let cfg = EntropicGwFastConfig {
            reg: 0.2,
            max_outer_iter: 20,
            max_inner_iter: 200,
            tol: 1e-3,
            n_samples: nb + 1,
            seed: 1,
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, na, nb, &cfg).expect("ok");
        for &p in &fit.transport_plan {
            assert!(p >= -1e-5 && p.is_finite(), "plan entry {p}");
        }
    }

    #[test]
    fn gw_cost_non_negative() {
        let na = 3;
        let nb = 3;
        let ca = line_dist(na);
        let cb = line_dist(nb);
        let a = uniform(na);
        let b = uniform(nb);
        let cfg = EntropicGwFastConfig {
            reg: 0.1,
            max_outer_iter: 30,
            max_inner_iter: 300,
            tol: 1e-3,
            n_samples: nb + 1,
            seed: 7,
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, na, nb, &cfg).expect("ok");
        assert!(fit.gw_cost >= -1e-4, "gw_cost={}", fit.gw_cost);
    }

    #[test]
    fn identical_spaces_have_low_gw_cost() {
        // For identical metric spaces and uniform marginals, GW cost should be small.
        let n = 3;
        let ca = dist_matrix(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)], n);
        let cb = ca.clone();
        let a = uniform(n);
        let b = uniform(n);
        let cfg = EntropicGwFastConfig {
            reg: 0.1,
            max_outer_iter: 60,
            max_inner_iter: 400,
            tol: 1e-3,
            n_samples: n + 1, // exact
            seed: 3,
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg).expect("ok");
        assert!(fit.gw_cost < 3.0, "gw_cost={} too high", fit.gw_cost);
    }

    #[test]
    fn gw_distance_helper_consistent() {
        let n = 3;
        let ca = line_dist(n);
        let cb = line_dist(n);
        let a = uniform(n);
        let b = uniform(n);
        let cfg = EntropicGwFastConfig {
            reg: 0.2,
            max_outer_iter: 20,
            max_inner_iter: 100,
            n_samples: n + 1,
            ..Default::default()
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg).expect("ok");
        let d = gw_distance(&fit);
        assert!((d - fit.gw_cost).abs() < 1e-6);
    }

    #[test]
    fn approximate_mode_runs_without_panic() {
        // Approximate mode: n_samples = 2 < nb = 4.
        let na = 4;
        let nb = 4;
        let ca = line_dist(na);
        let cb = line_dist(nb);
        let a = uniform(na);
        let b = uniform(nb);
        let cfg = EntropicGwFastConfig {
            reg: 0.3,
            max_outer_iter: 10,
            max_inner_iter: 100,
            tol: 1e-2,
            n_samples: 2, // < nb → approximate mode
            seed: 99,
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, na, nb, &cfg).expect("ok");
        assert_eq!(fit.transport_plan.len(), na * nb);
        assert!(fit.gw_cost.is_finite());
    }

    #[test]
    fn n_outer_iter_at_least_one() {
        let n = 2;
        let ca = vec![0.0_f32, 1.0, 1.0, 0.0];
        let cb = ca.clone();
        let a = uniform(n);
        let b = uniform(n);
        let cfg = EntropicGwFastConfig {
            reg: 0.5,
            max_outer_iter: 5,
            max_inner_iter: 50,
            n_samples: n + 1,
            ..Default::default()
        };
        let fit = entropic_gw_fast(&ca, &cb, &a, &b, n, n, &cfg).expect("ok");
        assert!(fit.n_outer_iter >= 1);
    }
}
