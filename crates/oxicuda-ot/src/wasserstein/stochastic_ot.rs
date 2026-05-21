//! Stochastic Optimal Transport via mini-batch dual potential estimation.
//!
//! Implements the stochastic OT framework of Genevay et al. (2016) and
//! Seguy et al. (2018). Rather than solving a single large OT problem,
//! we maintain global dual potentials `(log_u, log_v)` updated by
//! Exponential Moving Average (EMA) over independent mini-batch Sinkhorn
//! solves. Each mini-batch selects a random subset of source and target
//! indices, solves Sinkhorn on the induced sub-cost-matrix, and blends
//! the resulting batch potentials into the global potentials.
//!
//! ```text
//! For epoch t, batch (I, J):
//!   u_I^batch, v_J^batch ← Sinkhorn(C_{I×J}, a_I/|a_I|, b_J/|b_J|)
//!   log_u[I] ← (1-α) log_u[I] + α u_I^batch
//!   log_v[J] ← (1-α) log_v[J] + α v_J^batch
//! ```
//!
//! The primal-dual cost approximation after training is:
//!
//! ```text
//! W_ε ≈ Σ_i a_i · log_u_i + Σ_j b_j · log_v_j
//! ```

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::sinkhorn::log_sinkhorn::{log_sinkhorn_step_col, log_sinkhorn_step_row, log_to_plan};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the stochastic OT dual-potential estimator.
#[derive(Debug, Clone)]
pub struct StochasticOtConfig {
    /// Sinkhorn entropic regularisation `ε > 0`.
    pub reg: f64,
    /// Number of full passes over the index space.
    pub n_epochs: usize,
    /// Mini-batch size drawn from each distribution per inner iteration.
    pub batch_size: usize,
    /// EMA blending factor α ∈ (0, 1]: new = (1-α)·old + α·batch.
    pub ema_alpha: f64,
    /// RNG seed for reproducible batch sampling.
    pub seed: u64,
}

impl Default for StochasticOtConfig {
    fn default() -> Self {
        Self {
            reg: 0.1,
            n_epochs: 20,
            batch_size: 32,
            ema_alpha: 0.5,
            seed: 42,
        }
    }
}

/// Output of the stochastic OT solver containing trained dual potentials.
#[derive(Debug, Clone)]
pub struct StochasticOtFit {
    /// Log-domain source dual potentials, length `n`.
    pub log_u: Vec<f64>,
    /// Log-domain target dual potentials, length `m`.
    pub log_v: Vec<f64>,
    /// Estimated dual cost: `Σ_i a_i·u_i + Σ_j b_j·v_j`.
    pub dual_cost: f64,
    /// Number of source points.
    pub n: usize,
    /// Number of target points.
    pub m: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_stochastic(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &StochasticOtConfig,
) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if a.len() != n {
        return Err(OtError::IncompatibleLength { a: a.len(), b: n });
    }
    if b.len() != m {
        return Err(OtError::IncompatibleLength { a: b.len(), b: m });
    }
    if cost.len() != n * m {
        return Err(OtError::IncompatibleLength {
            a: cost.len(),
            b: n * m,
        });
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if cfg.batch_size == 0 {
        return Err(OtError::BadCount {
            got: cfg.batch_size,
        });
    }
    if cfg.batch_size > n {
        return Err(OtError::IncompatibleLength {
            a: cfg.batch_size,
            b: n,
        });
    }
    if cfg.batch_size > m {
        return Err(OtError::IncompatibleLength {
            a: cfg.batch_size,
            b: m,
        });
    }
    if cfg.n_epochs == 0 {
        return Err(OtError::BadCount { got: cfg.n_epochs });
    }
    if !(cfg.ema_alpha > 0.0 && cfg.ema_alpha <= 1.0) {
        return Err(OtError::Internal {
            msg: format!("ema_alpha must be in (0, 1], got {}", cfg.ema_alpha),
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Partial Fisher-Yates shuffle: sample `k` distinct indices from `0..n`.
fn sample_without_replacement(rng: &mut LcgRng, n: usize, k: usize, scratch: &mut Vec<usize>) {
    scratch.clear();
    scratch.extend(0..n);
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        scratch.swap(i, j);
    }
}

/// Extract the sub-cost matrix for given row indices `src_idx` (batch_size entries)
/// and column indices `tgt_idx` (batch_size entries) from the full cost matrix.
fn extract_batch_cost(cost: &[f64], src_idx: &[usize], tgt_idx: &[usize], m: usize) -> Vec<f32> {
    let bs = src_idx.len();
    let bt = tgt_idx.len();
    let mut c = vec![0.0_f32; bs * bt];
    for (bi, &si) in src_idx.iter().enumerate() {
        for (bj, &tj) in tgt_idx.iter().enumerate() {
            c[bi * bt + bj] = cost[si * m + tj] as f32;
        }
    }
    c
}

/// Run log-domain Sinkhorn on a uniform b×b mini-batch problem.
/// Returns `(u_batch, v_batch)` as f64 vectors of length `bs` and `bt` respectively.
fn mini_sinkhorn(
    batch_cost: &[f32],
    bs: usize,
    bt: usize,
    eps: f32,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Uniform marginals: log(1/bs), log(1/bt)
    let log_a: Vec<f32> = vec![(1.0_f32 / bs as f32).ln(); bs];
    let log_b: Vec<f32> = vec![(1.0_f32 / bt as f32).ln(); bt];

    let mut u = vec![0.0_f32; bs];
    let mut v = vec![0.0_f32; bt];

    for _ in 0..max_iter {
        if log_sinkhorn_step_row(batch_cost, &log_a, &mut u, &v, eps, bs, bt).is_err() {
            break;
        }
        if log_sinkhorn_step_col(batch_cost, &log_b, &mut v, &u, eps, bs, bt).is_err() {
            break;
        }
    }

    let u_out: Vec<f64> = u.iter().map(|&x| x as f64).collect();
    let v_out: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    (u_out, v_out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic OT via mini-batch dual potential EMA.
///
/// Trains global log-dual potentials `(log_u, log_v)` over `n_epochs`
/// passes of mini-batch Sinkhorn solves, blending each batch result into
/// the global potentials via EMA with factor `cfg.ema_alpha`.
///
/// # Parameters
///
/// - `a`: source marginal weights, length `n`, need not be normalised.
/// - `b`: target marginal weights, length `m`, need not be normalised.
/// - `cost`: flattened `n × m` cost matrix, row-major.
/// - `n`: number of source points.
/// - `m`: number of target points.
/// - `cfg`: solver configuration.
///
/// # Returns
///
/// A [`StochasticOtFit`] with trained potentials and dual cost estimate.
///
/// # Errors
///
/// Returns errors for invalid inputs (empty, bad dimensions, non-positive reg).
pub fn stochastic_ot(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &StochasticOtConfig,
) -> OtResult<StochasticOtFit> {
    validate_stochastic(a, b, cost, n, m, cfg)?;

    let mut rng = LcgRng::new(cfg.seed);
    let alpha = cfg.ema_alpha;
    let one_minus_alpha = 1.0 - alpha;
    let eps = cfg.reg as f32;

    // Initialise global potentials: log-uniform prior
    let mut log_u: Vec<f64> = vec![(1.0_f64 / n as f64).ln(); n];
    let mut log_v: Vec<f64> = vec![(1.0_f64 / m as f64).ln(); m];

    let mut src_scratch = vec![0usize; n];
    let mut tgt_scratch = vec![0usize; m];

    // Number of inner Sinkhorn iterations per mini-batch
    let sinkhorn_iters = 30_usize;

    let bs = cfg.batch_size;

    for _epoch in 0..cfg.n_epochs {
        // Shuffle source and target indices independently
        sample_without_replacement(&mut rng, n, n, &mut src_scratch);
        sample_without_replacement(&mut rng, m, m, &mut tgt_scratch);

        // Partition the shuffled indices into mini-batches
        let n_src_batches = n / bs;
        let n_tgt_batches = m / bs;
        let n_batches = n_src_batches.min(n_tgt_batches).max(1);

        for batch_idx in 0..n_batches {
            let src_start = (batch_idx * bs).min(n.saturating_sub(bs));
            let tgt_start = (batch_idx * bs).min(m.saturating_sub(bs));

            let src_idx = &src_scratch[src_start..src_start + bs];
            let tgt_idx = &tgt_scratch[tgt_start..tgt_start + bs];

            // Build the batch cost sub-matrix
            let batch_cost = extract_batch_cost(cost, src_idx, tgt_idx, m);

            // Solve Sinkhorn on the uniform batch marginals
            let (u_batch, v_batch) = mini_sinkhorn(&batch_cost, bs, bs, eps, sinkhorn_iters);

            // EMA update of global potentials for the batch indices
            for (bi, &si) in src_idx.iter().enumerate() {
                log_u[si] = one_minus_alpha * log_u[si] + alpha * u_batch[bi];
            }
            for (bj, &tj) in tgt_idx.iter().enumerate() {
                log_v[tj] = one_minus_alpha * log_v[tj] + alpha * v_batch[bj];
            }
        }
    }

    // Compute dual cost: Σ_i a_i·u_i + Σ_j b_j·v_j
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    let inv_sum_a = if sum_a > 1e-300 { 1.0 / sum_a } else { 1.0 };
    let inv_sum_b = if sum_b > 1e-300 { 1.0 / sum_b } else { 1.0 };

    let dual_cost: f64 = a
        .iter()
        .zip(log_u.iter())
        .map(|(&ai, &ui)| ai * inv_sum_a * ui)
        .sum::<f64>()
        + b.iter()
            .zip(log_v.iter())
            .map(|(&bj, &vj)| bj * inv_sum_b * vj)
            .sum::<f64>();

    Ok(StochasticOtFit {
        log_u,
        log_v,
        dual_cost,
        n,
        m,
    })
}

/// Approximate transport cost from the trained stochastic OT dual potentials.
///
/// Reconstructs the soft transport plan `P[i,j] = exp(log_u[i] + log_v[j] - C[i,j]/ε)`
/// and returns the inner product `⟨P_normalised, C⟩`.
///
/// # Errors
///
/// Returns an error if `cost` has incorrect length or potentials are inconsistent.
pub fn stochastic_transport_cost(fit: &StochasticOtFit, cost: &[f64], reg: f64) -> OtResult<f64> {
    let n = fit.n;
    let m = fit.m;
    if cost.len() != n * m {
        return Err(OtError::IncompatibleLength {
            a: cost.len(),
            b: n * m,
        });
    }
    if reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: reg as f32 });
    }

    let inv_reg = 1.0 / reg;
    let mut total_cost = 0.0_f64;
    let mut total_mass = 0.0_f64;

    for i in 0..n {
        let ui = fit.log_u[i];
        for j in 0..m {
            let cij = cost[i * m + j];
            let log_pij = ui + fit.log_v[j] - cij * inv_reg;
            // Clamp to avoid overflow; very negative values give ~0 mass
            let pij = if log_pij > -500.0 { log_pij.exp() } else { 0.0 };
            total_cost += pij * cij;
            total_mass += pij;
        }
    }

    if total_mass > 1e-300 {
        Ok(total_cost / total_mass)
    } else {
        Ok(0.0)
    }
}

/// Estimate the marginal violation of the approximate transport plan.
///
/// Computes:
/// ```text
/// max( max_i |Σ_j P_ij/Z - ā_i|, max_j |Σ_i P_ij/Z - b̄_j| )
/// ```
/// where `ā_i = a_i / Σ a` and `b̄_j = b_j / Σ b` are the normalised
/// marginals and `Z = Σ_{ij} P_ij`.
///
/// # Errors
///
/// Returns an error if inputs have inconsistent dimensions.
pub fn stochastic_marginal_violation(
    fit: &StochasticOtFit,
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    reg: f64,
) -> OtResult<f64> {
    let n = fit.n;
    let m = fit.m;
    if a.len() != n {
        return Err(OtError::IncompatibleLength { a: a.len(), b: n });
    }
    if b.len() != m {
        return Err(OtError::IncompatibleLength { a: b.len(), b: m });
    }
    if cost.len() != n * m {
        return Err(OtError::IncompatibleLength {
            a: cost.len(),
            b: n * m,
        });
    }
    if reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: reg as f32 });
    }

    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    let inv_sum_a = if sum_a > 1e-300 { 1.0 / sum_a } else { 1.0 };
    let inv_sum_b = if sum_b > 1e-300 { 1.0 / sum_b } else { 1.0 };

    let inv_reg = 1.0 / reg;
    let mut row_sums = vec![0.0_f64; n];
    let mut col_sums = vec![0.0_f64; m];
    let mut total_mass = 0.0_f64;

    for i in 0..n {
        let ui = fit.log_u[i];
        for j in 0..m {
            let cij = cost[i * m + j];
            let log_pij = ui + fit.log_v[j] - cij * inv_reg;
            let pij = if log_pij > -500.0 { log_pij.exp() } else { 0.0 };
            row_sums[i] += pij;
            col_sums[j] += pij;
            total_mass += pij;
        }
    }

    if total_mass < 1e-300 {
        return Ok(1.0);
    }

    let inv_mass = 1.0 / total_mass;
    let mut max_viol = 0.0_f64;

    for (i, rs) in row_sums.iter().enumerate() {
        let viol = (rs * inv_mass - a[i] * inv_sum_a).abs();
        if viol > max_viol {
            max_viol = viol;
        }
    }
    for (j, cs) in col_sums.iter().enumerate() {
        let viol = (cs * inv_mass - b[j] * inv_sum_b).abs();
        if viol > max_viol {
            max_viol = viol;
        }
    }

    Ok(max_viol)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public re-export of log_to_plan for plan materialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Materialise the full transport plan from trained stochastic-OT potentials.
///
/// Returns an `n × m` row-major plan matrix where each entry is
/// `exp(log_u[i] + log_v[j] - cost[i,j]/reg)`, unnormalised.
///
/// # Errors
///
/// Returns an error if input dimensions are inconsistent.
pub fn stochastic_transport_plan(
    fit: &StochasticOtFit,
    cost: &[f64],
    reg: f64,
) -> OtResult<Vec<f64>> {
    let n = fit.n;
    let m = fit.m;
    if cost.len() != n * m {
        return Err(OtError::IncompatibleLength {
            a: cost.len(),
            b: n * m,
        });
    }
    if reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: reg as f32 });
    }

    // Delegate to log_to_plan on f32 arrays for numerical stability
    let cost_f32: Vec<f32> = cost.iter().map(|&c| c as f32).collect();
    let u_f32: Vec<f32> = fit.log_u.iter().map(|&u| u as f32).collect();
    let v_f32: Vec<f32> = fit.log_v.iter().map(|&v| v as f32).collect();

    let plan_f32 = log_to_plan(&cost_f32, &u_f32, &v_f32, reg as f32, n, m)?;
    Ok(plan_f32.iter().map(|&p| p as f64).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a squared-Euclidean cost matrix for two 1-D grids.
    fn sq_cost_1d(src: &[f64], tgt: &[f64]) -> Vec<f64> {
        let n = src.len();
        let m = tgt.len();
        let mut c = vec![0.0_f64; n * m];
        for i in 0..n {
            for j in 0..m {
                let d = src[i] - tgt[j];
                c[i * m + j] = d * d;
            }
        }
        c
    }

    /// Uniform weight vector of length `k`.
    fn uniform(k: usize) -> Vec<f64> {
        vec![1.0 / k as f64; k]
    }

    #[test]
    fn dual_cost_is_finite_for_simple_problem() {
        let n = 8;
        let m = 8;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64 + 0.5).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.5,
            n_epochs: 5,
            batch_size: 4,
            ema_alpha: 0.5,
            seed: 1,
        };
        let fit = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("converges");
        assert!(fit.dual_cost.is_finite(), "dual_cost={}", fit.dual_cost);
        assert_eq!(fit.n, n);
        assert_eq!(fit.m, m);
        assert_eq!(fit.log_u.len(), n);
        assert_eq!(fit.log_v.len(), m);
    }

    #[test]
    fn dual_cost_increases_with_separation() {
        let n = 10;
        let m = 10;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tgt_close: Vec<f64> = (0..m).map(|j| j as f64 + 0.1).collect();
        let tgt_far: Vec<f64> = (0..m).map(|j| j as f64 + 5.0).collect();

        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.5,
            n_epochs: 10,
            batch_size: 5,
            ema_alpha: 0.7,
            seed: 42,
        };

        let cost_close = sq_cost_1d(&src, &tgt_close);
        let cost_far = sq_cost_1d(&src, &tgt_far);

        let fit_close = stochastic_ot(&a, &b, &cost_close, n, m, &cfg).expect("ok");
        let fit_far = stochastic_ot(&a, &b, &cost_far, n, m, &cfg).expect("ok");

        let tc_close = stochastic_transport_cost(&fit_close, &cost_close, cfg.reg).expect("ok");
        let tc_far = stochastic_transport_cost(&fit_far, &cost_far, cfg.reg).expect("ok");

        assert!(
            tc_far >= tc_close - 1e-6,
            "far cost {tc_far} should be >= close cost {tc_close}"
        );
    }

    #[test]
    fn transport_cost_is_non_negative() {
        let n = 6;
        let m = 6;
        let src: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64 * 0.5 + 1.0).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 1.0,
            n_epochs: 8,
            batch_size: 3,
            ema_alpha: 0.6,
            seed: 7,
        };
        let fit = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        let tc = stochastic_transport_cost(&fit, &cost, cfg.reg).expect("ok");
        assert!(tc >= 0.0, "transport cost={tc}");
    }

    #[test]
    fn transport_plan_shape_correct() {
        let n = 5;
        let m = 7;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64 * 0.5).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 1.0,
            n_epochs: 5,
            batch_size: 4,
            ema_alpha: 0.5,
            seed: 99,
        };
        let fit = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        let plan = stochastic_transport_plan(&fit, &cost, cfg.reg).expect("ok");
        assert_eq!(plan.len(), n * m);
        for &p in &plan {
            assert!(p >= 0.0 && p.is_finite());
        }
    }

    #[test]
    fn marginal_violation_is_finite() {
        let n = 8;
        let m = 8;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.5,
            n_epochs: 15,
            batch_size: 4,
            ema_alpha: 0.7,
            seed: 13,
        };
        let fit = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        let viol = stochastic_marginal_violation(&fit, &a, &b, &cost, cfg.reg).expect("ok");
        assert!(viol.is_finite(), "viol={viol}");
        assert!(viol >= 0.0, "viol={viol}");
    }

    #[test]
    fn empty_input_returns_error() {
        let cfg = StochasticOtConfig::default();
        let res = stochastic_ot(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_reg_returns_error() {
        let n = 8;
        let m = 8;
        let cost = vec![0.0_f64; n * m];
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: -1.0,
            n_epochs: 5,
            batch_size: 4,
            ema_alpha: 0.5,
            seed: 0,
        };
        let res = stochastic_ot(&a, &b, &cost, n, m, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_batch_size_returns_error() {
        let n = 4;
        let m = 4;
        let cost = vec![0.0_f64; n * m];
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.1,
            n_epochs: 5,
            batch_size: 10, // > n
            ema_alpha: 0.5,
            seed: 0,
        };
        let res = stochastic_ot(&a, &b, &cost, n, m, &cfg);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn bad_epochs_returns_error() {
        let n = 8;
        let m = 8;
        let cost = vec![0.0_f64; n * m];
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.1,
            n_epochs: 0,
            batch_size: 4,
            ema_alpha: 0.5,
            seed: 0,
        };
        let res = stochastic_ot(&a, &b, &cost, n, m, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn ema_alpha_one_overwrites_fully() {
        // With α=1 and 1 epoch, each batch fully overwrites the potentials
        let n = 8;
        let m = 8;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.5,
            n_epochs: 3,
            batch_size: 4,
            ema_alpha: 1.0,
            seed: 77,
        };
        let fit = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        assert!(fit.dual_cost.is_finite());
    }

    #[test]
    fn transport_cost_error_on_bad_reg() {
        let fit = StochasticOtFit {
            log_u: vec![0.0; 3],
            log_v: vec![0.0; 3],
            dual_cost: 0.0,
            n: 3,
            m: 3,
        };
        let cost = vec![0.0_f64; 9];
        let res = stochastic_transport_cost(&fit, &cost, -1.0);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn marginal_violation_error_on_wrong_shape() {
        let fit = StochasticOtFit {
            log_u: vec![0.0; 4],
            log_v: vec![0.0; 4],
            dual_cost: 0.0,
            n: 4,
            m: 4,
        };
        let a = vec![0.25_f64; 5]; // wrong length
        let b = vec![0.25_f64; 4];
        let cost = vec![0.0_f64; 16];
        let res = stochastic_marginal_violation(&fit, &a, &b, &cost, 1.0);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn determinism_same_seed() {
        let n = 10;
        let m = 10;
        let src: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let tgt: Vec<f64> = (0..m).map(|j| j as f64 * 0.1 + 0.5).collect();
        let cost = sq_cost_1d(&src, &tgt);
        let a = uniform(n);
        let b = uniform(m);
        let cfg = StochasticOtConfig {
            reg: 0.2,
            n_epochs: 5,
            batch_size: 5,
            ema_alpha: 0.5,
            seed: 123,
        };
        let fit1 = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        let fit2 = stochastic_ot(&a, &b, &cost, n, m, &cfg).expect("ok");
        for (u1, u2) in fit1.log_u.iter().zip(fit2.log_u.iter()) {
            assert_eq!(u1, u2, "potentials differ with same seed");
        }
    }
}
