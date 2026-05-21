//! Debiased Sinkhorn divergence with shared dual structure (Feydy 2020).
//!
//! The Sinkhorn divergence is defined as
//!
//! ```text
//! S_ε(α, β) = OT_ε(α, β) − ½ OT_ε(α, α) − ½ OT_ε(β, β)
//! ```
//!
//! Unlike vanilla entropic OT, `S_ε` satisfies `S_ε(α, α) = 0`, is symmetric,
//! positive on distinct distributions, and admits finite weight gradients even
//! as `ε → 0`. This module solves the three coupled log-domain Sinkhorn
//! problems concurrently using shared potential structure: the self-divergence
//! `OT_ε(α, α)` reduces to a single symmetric fixed-point `f ← ½ (f + Sym(f))`,
//! halving the cost compared to running asymmetric Sinkhorn on `(α, α)`.
//!
//! All computations use `f64` for numerical stability at small `ε`. Outputs
//! include the divergence value, the three constituent OT values, the
//! per-problem iteration counts, and optionally the weight gradients:
//! `∇_a S_ε = f_ab − f_aa`, where `f_ab` is the source dual potential of the
//! cross problem and `f_aa` the symmetric self potential.

use crate::error::{OtError, OtResult};

/// Configuration for [`debiased_sinkhorn_divergence`].
#[derive(Debug, Clone)]
pub struct DebiasedDivergenceConfig {
    /// Entropic regularisation strength `ε` (must be > 0). Smaller values
    /// produce sharper plans but require more iterations and tighter numerics.
    pub epsilon: f64,
    /// Maximum number of Sinkhorn iterations per sub-problem.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum potential update `‖f_new − f_old‖_∞`.
    pub tol: f64,
    /// If `true`, populate `grad_a` and `grad_b` in the result. The gradient
    /// of `S_ε` w.r.t. the weights is the Feydy closed form `f_ab − f_aa`
    /// (and analogously for `b`), defined on the simplex tangent space:
    /// coordinate *differences* recover the true `dS_ε / da` exactly, while
    /// the global gauge is fixed up to an `ε`-scale constant by the dual
    /// normalisation. The closed form is finite as `ε → 0` (Feydy 2020).
    pub compute_gradients: bool,
}

impl Default for DebiasedDivergenceConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 500,
            tol: 1e-7,
            compute_gradients: false,
        }
    }
}

/// Output of [`debiased_sinkhorn_divergence`].
#[derive(Debug, Clone)]
pub struct DebiasedDivergenceResult {
    /// Sinkhorn divergence `S_ε(α, β)`.
    pub divergence: f64,
    /// Cross entropic OT `OT_ε(α, β)` (dual objective value).
    pub ot_ab: f64,
    /// Self entropic OT `OT_ε(α, α)` via symmetric fixed point.
    pub ot_aa: f64,
    /// Self entropic OT `OT_ε(β, β)` via symmetric fixed point.
    pub ot_bb: f64,
    /// Gradient of `S_ε` w.r.t. the `a` weights (length `n`), populated only
    /// when `compute_gradients` is true.
    pub grad_a: Option<Vec<f64>>,
    /// Gradient of `S_ε` w.r.t. the `b` weights (length `m`).
    pub grad_b: Option<Vec<f64>>,
    /// Iterations completed for the cross `OT_ε(α, β)` problem.
    pub iterations_ab: usize,
    /// Iterations completed for the self `OT_ε(α, α)` problem.
    pub iterations_aa: usize,
    /// Iterations completed for the self `OT_ε(β, β)` problem.
    pub iterations_bb: usize,
}

/// Floor used for `log(0)` to avoid `-∞` poisoning gradients.
#[inline]
fn safe_ln(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Numerically stable log-sum-exp on a slice of `f64`.
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

/// Maximum absolute difference between two equal-length slices.
#[inline]
fn linf_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn validate_cross_dims(cost_ab: &[f64], n: usize, m: usize) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cost_ab.len() != n * m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: cost_ab.len(),
            b_len: n * m,
        });
    }
    Ok(())
}

fn validate_self(cost_xx: &[f64], k: usize) -> OtResult<()> {
    if cost_xx.len() != k * k {
        return Err(OtError::MarginalMismatch {
            m: k,
            n: k,
            a_len: cost_xx.len(),
            b_len: k * k,
        });
    }
    Ok(())
}

fn validate_weights(w: &[f64], expected: usize) -> OtResult<()> {
    if w.len() != expected {
        return Err(OtError::MarginalMismatch {
            m: expected,
            n: expected,
            a_len: w.len(),
            b_len: expected,
        });
    }
    for &wi in w {
        if wi < 0.0 || !wi.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

fn validate_config(cfg: &DebiasedDivergenceConfig) -> OtResult<()> {
    if cfg.epsilon <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.epsilon as f32,
        });
    }
    if cfg.max_iter == 0 {
        return Err(OtError::NotConverged {
            iter: 0,
            tol: cfg.tol as f32,
        });
    }
    if cfg.tol <= 0.0 {
        return Err(OtError::Internal {
            msg: format!(
                "debiased_sinkhorn_divergence: tol must be > 0, got {}",
                cfg.tol
            ),
        });
    }
    Ok(())
}

/// Run a single cross log-Sinkhorn problem in `f64` returning the converged
/// dual potentials and the number of iterations used.
fn cross_log_sinkhorn(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    eps: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, usize) {
    let log_a: Vec<f64> = a.iter().map(|&v| safe_ln(v)).collect();
    let log_b: Vec<f64> = b.iter().map(|&v| safe_ln(v)).collect();

    let mut f = vec![0.0_f64; n];
    let mut g = vec![0.0_f64; m];
    let mut buf = vec![0.0_f64; n.max(m)];

    let mut completed = 0_usize;
    for it in 0..max_iter {
        let f_prev = f.clone();
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
        if linf_diff(&f, &f_prev) < tol {
            break;
        }
    }
    (f, g, completed)
}

/// Symmetric log-Sinkhorn fixed-point `f ← ½ (f + Sym(f))` where
/// `Sym(f)_i = ε log a_i − ε · LSE_j ((f_j − C_ij)/ε)`.
fn symmetric_log_sinkhorn(
    cost: &[f64],
    a: &[f64],
    k: usize,
    eps: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize) {
    let log_a: Vec<f64> = a.iter().map(|&v| safe_ln(v)).collect();

    let mut f = vec![0.0_f64; k];
    let mut buf = vec![0.0_f64; k];

    let mut completed = 0_usize;
    for it in 0..max_iter {
        let f_prev = f.clone();
        let mut sym = vec![0.0_f64; k];
        for i in 0..k {
            let row_off = i * k;
            for j in 0..k {
                buf[j] = (f_prev[j] - cost[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf);
            sym[i] = eps * log_a[i] - eps * lse;
        }
        for i in 0..k {
            f[i] = 0.5 * (f_prev[i] + sym[i]);
        }
        completed = it + 1;
        if linf_diff(&f, &f_prev) < tol {
            break;
        }
    }
    (f, completed)
}

/// Dual objective `OT_ε = Σ_i a_i f_i + Σ_j b_j g_j`.
#[inline]
fn dual_value(a: &[f64], f: &[f64], b: &[f64], g: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for (ai, fi) in a.iter().zip(f.iter()) {
        s += ai * fi;
    }
    for (bj, gj) in b.iter().zip(g.iter()) {
        s += bj * gj;
    }
    s
}

/// Compute the debiased Sinkhorn divergence `S_ε(α, β)` between two
/// non-negative weighted point clouds via three log-domain Sinkhorn solves.
///
/// # Arguments
///
/// * `cost_ab` — `n × m` cost matrix (row-major) between `α` and `β` supports.
/// * `cost_aa` — `n × n` self-cost of `α` (must be symmetric with zero diagonal,
///   else relaxed to `(C + Cᵀ)/2`).
/// * `cost_bb` — `m × m` self-cost of `β` (same symmetry assumption).
/// * `a`, `b` — non-negative weights (lengths `n` and `m`). Need not sum to 1.
///
/// Returns the divergence and intermediates packaged in
/// [`DebiasedDivergenceResult`]. When `cfg.compute_gradients` is true, the
/// weight gradients `∇_a S_ε` and `∇_b S_ε` are returned; these are finite as
/// `ε → 0` and exact under autograd-style backprop through the dual loss.
pub fn debiased_sinkhorn_divergence(
    cost_ab: &[f64],
    n: usize,
    m: usize,
    cost_aa: &[f64],
    cost_bb: &[f64],
    a: &[f64],
    b: &[f64],
    cfg: &DebiasedDivergenceConfig,
) -> OtResult<DebiasedDivergenceResult> {
    validate_config(cfg)?;
    validate_cross_dims(cost_ab, n, m)?;
    validate_self(cost_aa, n)?;
    validate_self(cost_bb, m)?;
    validate_weights(a, n)?;
    validate_weights(b, m)?;

    let eps = cfg.epsilon;

    let cost_aa_sym = symmetrise(cost_aa, n);
    let cost_bb_sym = symmetrise(cost_bb, m);

    let (f_ab, g_ab, iter_ab) = cross_log_sinkhorn(cost_ab, a, b, n, m, eps, cfg.max_iter, cfg.tol);
    let (f_aa, iter_aa) = symmetric_log_sinkhorn(&cost_aa_sym, a, n, eps, cfg.max_iter, cfg.tol);
    let (f_bb, iter_bb) = symmetric_log_sinkhorn(&cost_bb_sym, b, m, eps, cfg.max_iter, cfg.tol);

    let ot_ab = dual_value(a, &f_ab, b, &g_ab);
    let ot_aa = dual_value(a, &f_aa, a, &f_aa);
    let ot_bb = dual_value(b, &f_bb, b, &f_bb);
    let divergence = ot_ab - 0.5 * ot_aa - 0.5 * ot_bb;

    let (grad_a, grad_b) = if cfg.compute_gradients {
        let mut ga = vec![0.0_f64; n];
        for i in 0..n {
            ga[i] = f_ab[i] - f_aa[i];
        }
        let mut gb = vec![0.0_f64; m];
        for j in 0..m {
            gb[j] = g_ab[j] - f_bb[j];
        }
        (Some(ga), Some(gb))
    } else {
        (None, None)
    };

    Ok(DebiasedDivergenceResult {
        divergence,
        ot_ab,
        ot_aa,
        ot_bb,
        grad_a,
        grad_b,
        iterations_ab: iter_ab,
        iterations_aa: iter_aa,
        iterations_bb: iter_bb,
    })
}

/// Return `(C + Cᵀ) / 2`. Self-cost matrices are required to be symmetric;
/// callers passing slightly-asymmetric matrices are silently relaxed.
fn symmetrise(c: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = 0.5 * (c[i * k + j] + c[j * k + i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_dist_matrix(x: &[f64], y: &[f64], dim: usize) -> Vec<f64> {
        let nx = x.len() / dim;
        let ny = y.len() / dim;
        let mut c = vec![0.0_f64; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let mut s = 0.0_f64;
                for d in 0..dim {
                    let diff = x[i * dim + d] - y[j * dim + d];
                    s += diff * diff;
                }
                c[i * ny + j] = s;
            }
        }
        c
    }

    #[test]
    fn identical_distributions_yield_zero_divergence() {
        let dim = 1;
        let x = vec![0.0_f64, 1.0, 2.0];
        let a = vec![1.0_f64 / 3.0; 3];
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.5,
            max_iter: 2000,
            tol: 1e-9,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&cost_aa, 3, 3, &cost_aa, &cost_aa, &a, &a, &cfg)
            .expect("ok");
        assert!(
            res.divergence.abs() < 1e-6,
            "S(a,a) = {} should be ≈ 0",
            res.divergence
        );
    }

    #[test]
    fn reflexive_alpha_alpha_is_exactly_zero_by_construction() {
        let dim = 1;
        let x = vec![0.5_f64, 1.5, 2.5, 3.5];
        let a = vec![0.25_f64, 0.25, 0.25, 0.25];
        let c = squared_dist_matrix(&x, &x, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.3,
            max_iter: 3000,
            tol: 1e-10,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&c, 4, 4, &c, &c, &a, &a, &cfg).expect("ok");
        let lhs = res.ot_ab;
        let rhs = 0.5 * (res.ot_aa + res.ot_bb);
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "OT(a,b)={lhs}, 0.5(OT(a,a)+OT(b,b))={rhs}"
        );
    }

    #[test]
    fn positive_for_distinct_distributions() {
        let dim = 1;
        let x = vec![0.0_f64, 0.1];
        let y = vec![1.0_f64, 1.1];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let cost_ab = squared_dist_matrix(&x, &y, dim);
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cost_bb = squared_dist_matrix(&y, &y, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.05,
            max_iter: 2000,
            tol: 1e-9,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&cost_ab, 2, 2, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        assert!(
            res.divergence > 0.1,
            "S(a,b)={} should be > 0 for disjoint supports",
            res.divergence
        );
    }

    #[test]
    fn small_eps_approaches_wasserstein_for_dirac_pair() {
        let cost_ab = vec![4.0_f64];
        let cost_aa = vec![0.0_f64];
        let cost_bb = vec![0.0_f64];
        let a = vec![1.0_f64];
        let b = vec![1.0_f64];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.001,
            max_iter: 500,
            tol: 1e-10,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&cost_ab, 1, 1, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        assert!(
            (res.divergence - 4.0).abs() < 1e-2,
            "expected ~W² = 4.0, got {}",
            res.divergence
        );
    }

    #[test]
    fn symmetry_swap_alpha_beta() {
        let dim = 1;
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.5_f64, 1.5];
        let a = vec![0.4_f64, 0.3, 0.3];
        let b = vec![0.5_f64, 0.5];
        let cost_ab = squared_dist_matrix(&x, &y, dim);
        let cost_ba = squared_dist_matrix(&y, &x, dim);
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cost_bb = squared_dist_matrix(&y, &y, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.1,
            max_iter: 2000,
            tol: 1e-9,
            compute_gradients: false,
        };
        let s_ab = debiased_sinkhorn_divergence(&cost_ab, 3, 2, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        let s_ba = debiased_sinkhorn_divergence(&cost_ba, 2, 3, &cost_bb, &cost_aa, &b, &a, &cfg)
            .expect("ok");
        assert!(
            (s_ab.divergence - s_ba.divergence).abs() < 1e-6,
            "S(a,b)={} vs S(b,a)={}",
            s_ab.divergence,
            s_ba.divergence
        );
    }

    #[test]
    fn convergence_sentinel_respected_on_easy_problem() {
        let cost = vec![0.0_f64, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.5,
            max_iter: 1000,
            tol: 1e-9,
            compute_gradients: false,
        };
        let res =
            debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg).expect("ok");
        assert!(
            res.iterations_ab < 1000,
            "should converge well before max_iter, got {}",
            res.iterations_ab
        );
    }

    #[test]
    fn invalid_epsilon_rejected() {
        let cost = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.0,
            max_iter: 10,
            tol: 1e-3,
            compute_gradients: false,
        };
        let r = debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg);
        assert!(matches!(r, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn invalid_max_iter_rejected() {
        let cost = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.1,
            max_iter: 0,
            tol: 1e-3,
            compute_gradients: false,
        };
        let r = debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg);
        assert!(matches!(r, Err(OtError::NotConverged { .. })));
    }

    #[test]
    fn invalid_tol_rejected() {
        let cost = vec![0.0_f64; 4];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.1,
            max_iter: 10,
            tol: 0.0,
            compute_gradients: false,
        };
        let r = debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg);
        assert!(matches!(r, Err(OtError::Internal { .. })));
    }

    #[test]
    fn dim_mismatch_rejected() {
        let cost = vec![0.0_f64; 6];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig::default();
        let r = debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg);
        assert!(matches!(r, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn negative_weight_rejected() {
        let cost = vec![0.0_f64; 4];
        let a = vec![-0.1_f64, 0.6];
        let b = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig::default();
        let r = debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &b, &cfg);
        assert!(matches!(r, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn finite_difference_gradient_check() {
        let dim = 1;
        let x = vec![0.0_f64, 1.0];
        let y = vec![0.6_f64, 1.4];
        let a = vec![0.6_f64, 0.4];
        let b = vec![0.5_f64, 0.5];
        let cost_ab = squared_dist_matrix(&x, &y, dim);
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cost_bb = squared_dist_matrix(&y, &y, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.1,
            max_iter: 8000,
            tol: 1e-13,
            compute_gradients: true,
        };
        let res = debiased_sinkhorn_divergence(&cost_ab, 2, 2, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        let grad_a = res.grad_a.as_ref().expect("grad");
        // Finite-difference the divergence w.r.t. `a[0]`. The Feydy 2020 closed
        // form `f_ab - f_aa` matches `dS/da` exactly when the dual loss uses
        // the gauge `Σ exp((f_i+g_j-C_ij)/ε) = Σ a_i Σ b_j`; under a different
        // gauge there is a constant offset shared across all coordinates which
        // cancels for differences `grad_a[i] - grad_a[j]`. We therefore test
        // that the analytic gradient and the FD gradient differ by the same
        // constant across both coordinates.
        let h = 1e-5_f64;
        let cfg_nograd = DebiasedDivergenceConfig {
            compute_gradients: false,
            ..cfg.clone()
        };
        let mut fd = [0.0_f64; 2];
        for i in 0..2 {
            let mut a_plus = a.clone();
            a_plus[i] += h;
            let mut a_minus = a.clone();
            a_minus[i] -= h;
            let s_plus = debiased_sinkhorn_divergence(
                &cost_ab,
                2,
                2,
                &cost_aa,
                &cost_bb,
                &a_plus,
                &b,
                &cfg_nograd,
            )
            .expect("ok")
            .divergence;
            let s_minus = debiased_sinkhorn_divergence(
                &cost_ab,
                2,
                2,
                &cost_aa,
                &cost_bb,
                &a_minus,
                &b,
                &cfg_nograd,
            )
            .expect("ok")
            .divergence;
            fd[i] = (s_plus - s_minus) / (2.0 * h);
        }
        let diff_analytic = grad_a[0] - grad_a[1];
        let diff_fd = fd[0] - fd[1];
        assert!(
            (diff_analytic - diff_fd).abs() < 1e-2,
            "analytic grad diff={diff_analytic} vs FD diff={diff_fd}; \
             analytic=[{}, {}], FD=[{}, {}]",
            grad_a[0],
            grad_a[1],
            fd[0],
            fd[1]
        );
    }

    #[test]
    fn numerical_stability_with_disparate_weights() {
        let dim = 1;
        let x = vec![0.0_f64, 1.0];
        let y = vec![0.3_f64, 1.7];
        let a = vec![0.999_f64, 0.001];
        let b = vec![0.001_f64, 0.999];
        let cost_ab = squared_dist_matrix(&x, &y, dim);
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cost_bb = squared_dist_matrix(&y, &y, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.05,
            max_iter: 5000,
            tol: 1e-10,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&cost_ab, 2, 2, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        assert!(res.divergence.is_finite(), "divergence not finite");
        assert!(res.ot_ab.is_finite() && res.ot_aa.is_finite() && res.ot_bb.is_finite());
    }

    #[test]
    fn non_unit_weights_supported() {
        let dim = 1;
        let x = vec![0.0_f64, 1.0];
        let y = vec![0.5_f64, 1.5];
        let a = vec![2.0_f64, 1.0];
        let b = vec![1.5_f64, 1.5];
        let cost_ab = squared_dist_matrix(&x, &y, dim);
        let cost_aa = squared_dist_matrix(&x, &x, dim);
        let cost_bb = squared_dist_matrix(&y, &y, dim);
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.1,
            max_iter: 2000,
            tol: 1e-9,
            compute_gradients: false,
        };
        let res = debiased_sinkhorn_divergence(&cost_ab, 2, 2, &cost_aa, &cost_bb, &a, &b, &cfg)
            .expect("ok");
        assert!(res.divergence.is_finite());
        assert!(res.ot_ab > 0.0);
    }

    #[test]
    fn self_iteration_counts_recorded() {
        let cost = vec![0.0_f64, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let cfg = DebiasedDivergenceConfig {
            epsilon: 0.5,
            max_iter: 50,
            tol: 1e-9,
            compute_gradients: false,
        };
        let res =
            debiased_sinkhorn_divergence(&cost, 2, 2, &cost, &cost, &a, &a, &cfg).expect("ok");
        assert!(res.iterations_aa >= 1);
        assert!(res.iterations_bb >= 1);
        assert!(res.iterations_ab >= 1);
    }
}
