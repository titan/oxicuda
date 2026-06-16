//! Levenberg-Marquardt nonlinear least-squares solver.
//!
//! Minimises `½‖r(p)‖²` where `r: ℝⁿ → ℝᵐ` with `m ≥ n`.
//!
//! The damped normal equations `(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr` are solved
//! each iteration via LU decomposition.  On accepted steps the damping factor
//! λ is reduced; on rejected steps it is increased.
//!
//! Both an analytic-Jacobian variant (`levenberg_marquardt`) and a
//! numerical-Jacobian variant using forward differences
//! (`levenberg_marquardt_numerical`) are provided.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

// ── Public types ─────────────────────────────────────────────────────────────

/// Configuration for the Levenberg-Marquardt algorithm.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Maximum number of iterations (default 1000).
    pub max_iter: usize,
    /// Relative-cost tolerance (default 1e-8).
    pub ftol: f64,
    /// Gradient (Jᵀr) infinity-norm tolerance (default 1e-8).
    pub gtol: f64,
    /// Step-norm tolerance (default 1e-8).
    pub xtol: f64,
    /// Initial damping factor λ (default 1e-3).
    pub lambda_init: f64,
    /// Multiplicative factor to increase λ on rejection (default 10.0).
    pub lambda_up: f64,
    /// Multiplicative factor to decrease λ on acceptance (default 10.0).
    pub lambda_down: f64,
    /// Forward-difference step size for numerical Jacobian (default 1e-6).
    pub fd_step: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            ftol: 1.0e-8,
            gtol: 1.0e-8,
            xtol: 1.0e-8,
            lambda_init: 1.0e-3,
            lambda_up: 10.0,
            lambda_down: 10.0,
            fd_step: 1.0e-6,
        }
    }
}

/// Result produced by the Levenberg-Marquardt solver.
#[derive(Debug, Clone)]
pub struct LmResult {
    /// Best parameter vector found.
    pub params: Vec<f64>,
    /// Final cost `½‖r(params)‖²`.
    pub cost: f64,
    /// Final residual norm `‖r(params)‖`.
    pub residual_norm: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether a convergence criterion was met.
    pub converged: bool,
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Compute `½‖r‖²`.
#[inline]
fn half_sq_norm(r: &[f64]) -> f64 {
    0.5 * r.iter().map(|v| v * v).sum::<f64>()
}

/// Build `JᵀJ` (n×n) and `Jᵀr` (n×1) from a row-major Jacobian J (m×n).
fn jtj_jtr(j: &[f64], r: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jtj = vec![0.0_f64; n * n];
    let mut jtr = vec![0.0_f64; n];
    for k in 0..m {
        for i in 0..n {
            let ji = j[k * n + i];
            jtr[i] += ji * r[k];
            for jj in 0..n {
                jtj[i * n + jj] += ji * j[k * n + jj];
            }
        }
    }
    (jtj, jtr)
}

/// Build the damped normal equations matrix `A = JᵀJ + λ · diag(max(JᵀJ_ii, 1))`.
fn build_damped_matrix(jtj: &[f64], n: usize, lambda: f64) -> Vec<f64> {
    let mut a = jtj.to_vec();
    for i in 0..n {
        let d = jtj[i * n + i].max(1.0);
        a[i * n + i] += lambda * d;
    }
    a
}

/// Core LM iteration.  `jac_fn` returns the m×n row-major Jacobian as a Vec<f64>.
fn lm_core<R, JF>(residual: R, jac_fn: JF, p0: &[f64], cfg: &LmConfig) -> NumericResult<LmResult>
where
    R: Fn(&[f64]) -> NumericResult<Vec<f64>>,
    JF: Fn(&[f64]) -> NumericResult<Vec<f64>>,
{
    // ── Input validation ────────────────────────────────────────────────────
    if p0.is_empty() {
        return Err(NumericError::InvalidParameter(
            "p0 must be non-empty".into(),
        ));
    }
    let n = p0.len();

    // Evaluate residual at initial point.
    let r0 = residual(p0)?;
    let m = r0.len();
    if m < n {
        return Err(NumericError::DimensionMismatch { a: m, b: n });
    }

    let mut p = p0.to_vec();
    let mut r = r0;
    let mut cost = half_sq_norm(&r);
    let mut lambda = cfg.lambda_init;
    let mut converged = false;
    let mut final_iter = cfg.max_iter;

    for iter in 0..cfg.max_iter {
        // ── Jacobian ────────────────────────────────────────────────────────
        let j = jac_fn(&p)?;
        if j.len() != m * n {
            return Err(NumericError::DimensionMismatch {
                a: j.len(),
                b: m * n,
            });
        }

        // ── Normal equations ─────────────────────────────────────────────────
        let (jtj, jtr) = jtj_jtr(&j, &r, m, n);

        // Check gradient convergence (before damping).
        let g_inf = jtr.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < cfg.gtol {
            converged = true;
            let residual_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            return Ok(LmResult {
                params: p,
                cost,
                residual_norm,
                iterations: iter,
                converged,
            });
        }

        // ── Attempt a step ───────────────────────────────────────────────────
        // Build rhs = −Jᵀr
        let rhs: Vec<f64> = jtr.iter().map(|v| -v).collect();

        // Try to solve (JᵀJ + λD)δ = −Jᵀr.
        // If singular at large λ, give up.
        let a = build_damped_matrix(&jtj, n, lambda);
        let delta = match lu_decompose(&a, n) {
            Ok((lu, piv, _)) => match lu_solve(&lu, &piv, n, &rhs) {
                Ok(d) => d,
                Err(_) => {
                    if lambda > 1.0e14 {
                        return Err(NumericError::NotConverged {
                            iter,
                            residual: cost.sqrt() * std::f64::consts::SQRT_2,
                        });
                    }
                    lambda *= cfg.lambda_up;
                    lambda = lambda.min(1.0e16);
                    continue;
                }
            },
            Err(_) => {
                if lambda > 1.0e14 {
                    return Err(NumericError::NotConverged {
                        iter,
                        residual: cost.sqrt() * std::f64::consts::SQRT_2,
                    });
                }
                lambda *= cfg.lambda_up;
                lambda = lambda.min(1.0e16);
                continue;
            }
        };

        // ── Trial step ───────────────────────────────────────────────────────
        let p_new: Vec<f64> = p.iter().zip(delta.iter()).map(|(pi, di)| pi + di).collect();
        let r_new = residual(&p_new)?;
        if r_new.len() != m {
            return Err(NumericError::DimensionMismatch {
                a: r_new.len(),
                b: m,
            });
        }
        let cost_new = half_sq_norm(&r_new);

        if cost_new < cost {
            // ── Accept ───────────────────────────────────────────────────────
            let cost_prev = cost;
            p = p_new;
            r = r_new;
            cost = cost_new;
            lambda = (lambda / cfg.lambda_down).max(1.0e-16);

            // ── Convergence checks (after accepted step) ─────────────────────
            // ftol: relative cost reduction
            let rel_cost = (cost_prev - cost).abs() / (cost_prev + 1.0e-15);
            let ftol_met = rel_cost < cfg.ftol;

            // xtol: step norm
            let step_norm: f64 = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            let p_norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            let xtol_met = step_norm < cfg.xtol * (p_norm + cfg.xtol);

            if ftol_met || xtol_met {
                converged = true;
                final_iter = iter + 1;
                break;
            }
        } else {
            // ── Reject ───────────────────────────────────────────────────────
            lambda *= cfg.lambda_up;
            lambda = lambda.min(1.0e16);
        }
    }

    let residual_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(LmResult {
        params: p,
        cost,
        residual_norm,
        iterations: final_iter,
        converged,
    })
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Levenberg-Marquardt with an analytic Jacobian.
///
/// `jacobian` must return a row-major `m × n` matrix as a `Vec<f64>` of
/// length `m * n` where `m = residual.len()` and `n = p0.len()`.
pub fn levenberg_marquardt<R, J>(
    residual: R,
    jacobian: J,
    p0: &[f64],
    cfg: &LmConfig,
) -> NumericResult<LmResult>
where
    R: Fn(&[f64]) -> NumericResult<Vec<f64>>,
    J: Fn(&[f64]) -> NumericResult<Vec<f64>>,
{
    lm_core(residual, jacobian, p0, cfg)
}

/// Levenberg-Marquardt with a numerical Jacobian (forward differences).
///
/// The Jacobian column `j` is approximated as
/// `(r(p + h·eⱼ) − r(p)) / h` where `h = cfg.fd_step`.
pub fn levenberg_marquardt_numerical<R>(
    residual: R,
    p0: &[f64],
    cfg: &LmConfig,
) -> NumericResult<LmResult>
where
    R: Fn(&[f64]) -> NumericResult<Vec<f64>>,
{
    let fd_step = cfg.fd_step;
    // We need to share `residual` between the closure and lm_core.
    // Since both the residual and the Jacobian approximation call it,
    // we wrap it in a reference.
    let jac_fn = |p: &[f64]| -> NumericResult<Vec<f64>> {
        let r0 = residual(p)?;
        let m = r0.len();
        let n = p.len();
        let mut j = vec![0.0_f64; m * n];
        let mut p_pert = p.to_vec();
        for jj in 0..n {
            let orig = p_pert[jj];
            p_pert[jj] = orig + fd_step;
            let r_pert = residual(&p_pert)?;
            if r_pert.len() != m {
                return Err(NumericError::DimensionMismatch {
                    a: r_pert.len(),
                    b: m,
                });
            }
            for i in 0..m {
                j[i * n + jj] = (r_pert[i] - r0[i]) / fd_step;
            }
            p_pert[jj] = orig;
        }
        Ok(j)
    };
    lm_core(&residual, jac_fn, p0, cfg)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: linear LSQ 5×3 ──────────────────────────────────────────────
    #[test]
    fn lm_linear_lsq_5x3() {
        // Fixed 5×3 matrix with known condition number
        let a: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 1.0, 0.5, 3.0, 2.5, 2.0, 0.0, 4.0,
        ];
        let b = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        // Exact normal-equation solution via LU
        let mut ata = vec![0.0_f64; 9];
        let mut atb = vec![0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..5 {
                    ata[i * 3 + j] += a[k * 3 + i] * a[k * 3 + j];
                }
            }
            for k in 0..5 {
                atb[i] += a[k * 3 + i] * b[k];
            }
        }
        let (lu, piv, _) = crate::linalg::lu_decomp::lu_decompose(&ata, 3).expect("lu");
        let exact = crate::linalg::lu_decomp::lu_solve(&lu, &piv, 3, &atb).expect("solve");

        let a_clone = a.clone();
        let residual = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let mut r = vec![0.0_f64; 5];
            for i in 0..5 {
                for j in 0..3 {
                    r[i] += a_clone[i * 3 + j] * p[j];
                }
                r[i] -= b[i];
            }
            Ok(r)
        };
        let a_clone2 = a.clone();
        let jacobian = move |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(a_clone2.clone()) };
        let cfg = LmConfig {
            max_iter: 200,
            ftol: 1.0e-12,
            gtol: 1.0e-12,
            xtol: 1.0e-12,
            ..Default::default()
        };
        let result = levenberg_marquardt(residual, jacobian, &[0.0, 0.0, 0.0], &cfg).expect("lm");
        for (i, (got, ex)) in result.params.iter().zip(exact.iter()).enumerate() {
            assert!(
                (got - ex).abs() < 1.0e-8,
                "param[{i}]: got={got:.10} exact={ex:.10}"
            );
        }
    }

    // ── Test 2: exponential fit a*exp(b*t) ──────────────────────────────────
    #[test]
    fn lm_exponential_fit() {
        // True params: a=2, b=0.5
        let ts: Vec<f64> = (0..10).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = ts.iter().map(|&t| 2.0 * (0.5 * t).exp()).collect();

        let ts2 = ts.clone();
        let ys2 = ys.clone();
        let residual = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let a = p[0];
            let b = p[1];
            Ok(ts2
                .iter()
                .zip(ys2.iter())
                .map(|(&t, &y)| a * (b * t).exp() - y)
                .collect())
        };
        let ts3 = ts.clone();
        let jacobian = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let a = p[0];
            let b = p[1];
            let mut j = vec![0.0_f64; ts3.len() * 2];
            for (i, &t) in ts3.iter().enumerate() {
                j[i * 2] = (b * t).exp(); // ∂r_i/∂a
                j[i * 2 + 1] = a * t * (b * t).exp(); // ∂r_i/∂b
            }
            Ok(j)
        };
        let cfg = LmConfig::default();
        let result = levenberg_marquardt(residual, jacobian, &[1.0, 0.1], &cfg).expect("exp fit");
        assert!(
            (result.params[0] - 2.0).abs() < 1.0e-4,
            "a={}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 0.5).abs() < 1.0e-4,
            "b={}",
            result.params[1]
        );
    }

    // ── Test 3: Rosenbrock as LSQ, analytic Jacobian ─────────────────────────
    #[test]
    fn lm_rosenbrock_analytic() {
        // r = [10*(p1 - p0^2), 1 - p0], minimum at (1,1) with cost 0
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        };
        let jacobian = |p: &[f64]| -> NumericResult<Vec<f64>> {
            // J is 2×2 row-major
            Ok(vec![
                -20.0 * p[0],
                10.0, // row 0
                -1.0,
                0.0, // row 1
            ])
        };
        let cfg = LmConfig {
            max_iter: 2000,
            ftol: 1.0e-12,
            gtol: 1.0e-12,
            xtol: 1.0e-12,
            ..Default::default()
        };
        let result =
            levenberg_marquardt(residual, jacobian, &[-1.0, 1.0], &cfg).expect("rosenbrock");
        assert!(
            (result.params[0] - 1.0).abs() < 1.0e-8,
            "p0={:.10}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 1.0e-8,
            "p1={:.10}",
            result.params[1]
        );
        assert!(result.cost < 1.0e-12, "cost={:.2e}", result.cost);
    }

    // ── Test 4: Gaussian peak fit ────────────────────────────────────────────
    #[test]
    fn lm_gaussian_peak_fit() {
        // True: A=3, c=2, w=0.5
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 3.0 * (-(x - 2.0).powi(2) / (2.0 * 0.25)).exp())
            .collect();
        let xs2 = xs.clone();
        let ys2 = ys.clone();
        let residual = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let (amp, ctr, wid) = (p[0], p[1], p[2]);
            Ok(xs2
                .iter()
                .zip(ys2.iter())
                .map(|(&x, &y)| amp * (-(x - ctr).powi(2) / (2.0 * wid * wid)).exp() - y)
                .collect())
        };
        let xs3 = xs.clone();
        let jacobian = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let (amp, ctr, wid) = (p[0], p[1], p[2]);
            let m = xs3.len();
            let mut j = vec![0.0_f64; m * 3];
            for (i, &x) in xs3.iter().enumerate() {
                let exponent = -(x - ctr).powi(2) / (2.0 * wid * wid);
                let g = exponent.exp();
                j[i * 3] = g; // ∂/∂A
                j[i * 3 + 1] = amp * g * (x - ctr) / (wid * wid); // ∂/∂c
                j[i * 3 + 2] = amp * g * (x - ctr).powi(2) / (wid * wid * wid); // ∂/∂w
            }
            Ok(j)
        };
        let cfg = LmConfig::default();
        let result =
            levenberg_marquardt(residual, jacobian, &[2.0, 1.5, 0.3], &cfg).expect("gaussian");
        assert!(
            (result.params[0] - 3.0).abs() < 1.0e-4,
            "A={}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 2.0).abs() < 1.0e-4,
            "c={}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 0.5).abs() < 1.0e-4,
            "w={}",
            result.params[2]
        );
    }

    // ── Test 5: analytic vs numerical Jacobian agree ─────────────────────────
    #[test]
    fn lm_analytic_vs_numerical_jacobian() {
        let ts: Vec<f64> = (0..10).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = ts.iter().map(|&t| 2.0 * (0.5 * t).exp()).collect();

        let make_res = || {
            let ts = ts.clone();
            let ys = ys.clone();
            move |p: &[f64]| -> NumericResult<Vec<f64>> {
                let a = p[0];
                let b = p[1];
                Ok(ts
                    .iter()
                    .zip(ys.iter())
                    .map(|(&t, &y)| a * (b * t).exp() - y)
                    .collect())
            }
        };

        let ts2 = ts.clone();
        let jacobian = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let a = p[0];
            let b = p[1];
            let mut j = vec![0.0_f64; ts2.len() * 2];
            for (i, &t) in ts2.iter().enumerate() {
                j[i * 2] = (b * t).exp();
                j[i * 2 + 1] = a * t * (b * t).exp();
            }
            Ok(j)
        };

        let cfg = LmConfig::default();
        let r_ana = levenberg_marquardt(make_res(), jacobian, &[1.0, 0.1], &cfg).expect("analytic");
        let r_num = levenberg_marquardt_numerical(make_res(), &[1.0, 0.1], &cfg).expect("numeric");

        let diff: f64 = r_ana
            .params
            .iter()
            .zip(r_num.params.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1.0e-6, "analytic vs numeric diff = {diff:.2e}");
    }

    // ── Test 6: Rosenbrock from bad initial guess (5, -5) ────────────────────
    #[test]
    fn lm_rosenbrock_bad_initial() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        };
        let jacobian =
            |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-20.0 * p[0], 10.0, -1.0, 0.0]) };
        let cfg = LmConfig {
            max_iter: 5000,
            ftol: 1.0e-10,
            gtol: 1.0e-10,
            xtol: 1.0e-10,
            ..Default::default()
        };
        let result =
            levenberg_marquardt(residual, jacobian, &[5.0, -5.0], &cfg).expect("rosenbrock bad");
        assert!(
            (result.params[0] - 1.0).abs() < 1.0e-6,
            "p0={:.8}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 1.0e-6,
            "p1={:.8}",
            result.params[1]
        );
    }

    // ── Test 7: converged=true when gtol met ─────────────────────────────────
    #[test]
    fn lm_converged_gtol() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 3.0]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let cfg = LmConfig {
            gtol: 1.0e-6,
            ftol: 0.0,
            xtol: 0.0,
            ..Default::default()
        };
        let result = levenberg_marquardt(residual, jacobian, &[0.0], &cfg).expect("gtol");
        assert!(result.converged, "should have converged via gtol");
    }

    // ── Test 8: converged=true when ftol met ─────────────────────────────────
    #[test]
    fn lm_converged_ftol() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 3.0]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let cfg = LmConfig {
            ftol: 1.0e-6,
            gtol: 0.0,
            xtol: 0.0,
            ..Default::default()
        };
        let result = levenberg_marquardt(residual, jacobian, &[0.0], &cfg).expect("ftol");
        assert!(result.converged, "should have converged via ftol");
    }

    // ── Test 9: converged=true when xtol met ─────────────────────────────────
    #[test]
    fn lm_converged_xtol() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 3.0]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let cfg = LmConfig {
            xtol: 1.0e-6,
            ftol: 0.0,
            gtol: 0.0,
            ..Default::default()
        };
        let result = levenberg_marquardt(residual, jacobian, &[0.0], &cfg).expect("xtol");
        assert!(result.converged, "should have converged via xtol");
    }

    // ── Test 10: max_iter exhausted → converged=false, Ok ────────────────────
    #[test]
    fn lm_max_iter_not_err() {
        // Use a problem that doesn't converge in 2 steps
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            // Hard oscillatory residual
            Ok(vec![p[0].sin() - 0.5, p[1].cos() - 0.3])
        };
        let jacobian =
            |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0].cos(), 0.0, 0.0, -p[1].sin()]) };
        let cfg = LmConfig {
            max_iter: 2,
            ftol: 0.0,
            gtol: 0.0,
            xtol: 0.0,
            ..Default::default()
        };
        let result =
            levenberg_marquardt(residual, jacobian, &[10.0, 10.0], &cfg).expect("max_iter ok");
        assert!(!result.converged, "should not have converged in 2 steps");
    }

    // ── Test 11: determinism ─────────────────────────────────────────────────
    #[test]
    fn lm_deterministic() {
        fn make_residual(p: &[f64]) -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        }
        fn make_jacobian(p: &[f64]) -> NumericResult<Vec<f64>> {
            Ok(vec![-20.0 * p[0], 10.0, -1.0, 0.0])
        }
        let cfg = LmConfig::default();
        let r1 = levenberg_marquardt(make_residual, make_jacobian, &[-1.0, 1.0], &cfg).expect("r1");
        let r2 = levenberg_marquardt(make_residual, make_jacobian, &[-1.0, 1.0], &cfg).expect("r2");
        assert_eq!(r1.params, r2.params);
        assert_eq!(r1.iterations, r2.iterations);
    }

    // ── Test 12: p0 empty → InvalidParameter ─────────────────────────────────
    #[test]
    fn lm_empty_p0_err() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0]]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let res = levenberg_marquardt(residual, jacobian, &[], &LmConfig::default());
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    // ── Test 13: m < n → DimensionMismatch ───────────────────────────────────
    #[test]
    fn lm_underdetermined_err() {
        // p has 3 elements but residual returns 2
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 1.0, p[1] - 2.0]) };
        let jacobian =
            |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) };
        let res = levenberg_marquardt(residual, jacobian, &[0.0, 0.0, 0.0], &LmConfig::default());
        assert!(matches!(res, Err(NumericError::DimensionMismatch { .. })));
    }

    // ── Test 14: Jacobian length mismatch → DimensionMismatch ────────────────
    #[test]
    fn lm_jacobian_length_mismatch_err() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 1.0, p[1] - 2.0]) };
        let bad_jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![1.0, 0.0]) // should be 2*2=4 elements but returns 2
        };
        let res = levenberg_marquardt(residual, bad_jacobian, &[0.0, 0.0], &LmConfig::default());
        assert!(matches!(res, Err(NumericError::DimensionMismatch { .. })));
    }

    // ── Test 15: near-zero residual — no NaN ─────────────────────────────────
    #[test]
    fn lm_near_zero_residual_stable() {
        // Already at minimum — residual ≈ 0
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 1.0, p[1] - 1.0]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0, 0.0, 0.0, 1.0]) };
        let cfg = LmConfig::default();
        let result = levenberg_marquardt(residual, jacobian, &[1.0, 1.0], &cfg).expect("near zero");
        assert!(result.cost.is_finite(), "cost is NaN or Inf");
        assert!(
            result.params.iter().all(|v| v.is_finite()),
            "params contain NaN/Inf"
        );
    }

    // ── Test 16: lambda adaptation — grows on bad steps ──────────────────────
    #[test]
    fn lm_lambda_adaptation() {
        // Start far from optimum on Rosenbrock — early steps are bad so λ grows.
        // We track via a counter that at some point lambda must increase.
        // Since we can't inspect internal state, we just verify convergence still happens.
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        };
        let jacobian =
            |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-20.0 * p[0], 10.0, -1.0, 0.0]) };
        // Very bad starting point — forces many lambda increases
        let cfg = LmConfig {
            max_iter: 10000,
            lambda_init: 1.0e-6, // start too small → bad steps → lambda grows
            ..Default::default()
        };
        let result =
            levenberg_marquardt(residual, jacobian, &[10.0, -10.0], &cfg).expect("lambda adapt");
        // If lambda adaptation works correctly, we still reach the minimum.
        assert!(
            (result.params[0] - 1.0).abs() < 1.0e-4,
            "p0={:.8}",
            result.params[0]
        );
    }

    // ── Test 17: linear 1-D, r=[p-3] → converges to 3 in few iterations ─────
    #[test]
    fn lm_linear_1d() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![p[0] - 3.0]) };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let cfg = LmConfig {
            max_iter: 10,
            ftol: 1.0e-12,
            gtol: 1.0e-12,
            xtol: 1.0e-12,
            ..Default::default()
        };
        let result = levenberg_marquardt(residual, jacobian, &[0.0], &cfg).expect("1d linear");
        assert!(
            (result.params[0] - 3.0).abs() < 1.0e-10,
            "p={:.12}",
            result.params[0]
        );
    }

    // ── Test 18: Rosenbrock from (-1,-1) → global minimum ────────────────────
    #[test]
    fn lm_rosenbrock_from_minus1_minus1() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        };
        let jacobian =
            |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-20.0 * p[0], 10.0, -1.0, 0.0]) };
        let cfg = LmConfig {
            max_iter: 2000,
            ftol: 1.0e-12,
            gtol: 1.0e-12,
            xtol: 1.0e-12,
            ..Default::default()
        };
        let result =
            levenberg_marquardt(residual, jacobian, &[-1.0, -1.0], &cfg).expect("rosenbrock -1,-1");
        assert!(
            (result.params[0] - 1.0).abs() < 1.0e-8,
            "p0={:.10}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 1.0e-8,
            "p1={:.10}",
            result.params[1]
        );
    }

    // ── Test 19: residual returning NumericError propagates ──────────────────
    #[test]
    fn lm_residual_error_propagates() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        let residual = move |_p: &[f64]| -> NumericResult<Vec<f64>> {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            if n >= 3 {
                Err(NumericError::NumericalInstability("mock failure".into()))
            } else {
                Ok(vec![_p[0] - 1.0, _p[1] - 1.0])
            }
        };
        let jacobian = |_p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0, 0.0, 0.0, 1.0]) };
        let res = levenberg_marquardt(residual, jacobian, &[0.0, 0.0], &LmConfig::default());
        assert!(res.is_err(), "expected error to propagate");
    }

    // ── Test 20: params stay finite ──────────────────────────────────────────
    #[test]
    fn lm_params_stay_finite() {
        let residual = |p: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])
        };
        let jacobian =
            |p: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-20.0 * p[0], 10.0, -1.0, 0.0]) };
        let cfg = LmConfig::default();
        let result = levenberg_marquardt(residual, jacobian, &[-1.0, 1.0], &cfg).expect("finite");
        assert!(result.params.iter().all(|v| v.is_finite()));
        assert!(result.cost.is_finite());
        assert!(result.residual_norm.is_finite());
    }

    // ── Test 21: fd_step sensitivity ─────────────────────────────────────────
    #[test]
    fn lm_fd_step_sensitivity() {
        let ts: Vec<f64> = (0..10).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = ts.iter().map(|&t| 2.0 * (0.5 * t).exp()).collect();

        let make_res = || {
            let ts = ts.clone();
            let ys = ys.clone();
            move |p: &[f64]| -> NumericResult<Vec<f64>> {
                Ok(ts
                    .iter()
                    .zip(ys.iter())
                    .map(|(&t, &y)| p[0] * (p[1] * t).exp() - y)
                    .collect())
            }
        };

        let cfg_coarse = LmConfig {
            fd_step: 1.0e-4,
            ..Default::default()
        };
        let cfg_fine = LmConfig {
            fd_step: 1.0e-8,
            ..Default::default()
        };

        let r_coarse =
            levenberg_marquardt_numerical(make_res(), &[1.0, 0.1], &cfg_coarse).expect("coarse");
        let r_fine =
            levenberg_marquardt_numerical(make_res(), &[1.0, 0.1], &cfg_fine).expect("fine");

        let diff: f64 = r_coarse
            .params
            .iter()
            .zip(r_fine.params.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1.0e-4, "fd_step sensitivity diff = {diff:.2e}");
    }

    // ── Test 22: 4-param sine fit A*sin(Bx+C)+D ──────────────────────────────
    #[test]
    fn lm_sine_4param_fit() {
        // True: A=2, B=1.5, C=0.5, D=1.0
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.4).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 2.0 * (1.5 * x + 0.5).sin() + 1.0)
            .collect();
        let xs2 = xs.clone();
        let ys2 = ys.clone();
        let residual = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let (amp, freq, phase, offset) = (p[0], p[1], p[2], p[3]);
            Ok(xs2
                .iter()
                .zip(ys2.iter())
                .map(|(&x, &y)| amp * (freq * x + phase).sin() + offset - y)
                .collect())
        };
        let xs3 = xs.clone();
        let jacobian = move |p: &[f64]| -> NumericResult<Vec<f64>> {
            let (amp, freq, phase, _offset) = (p[0], p[1], p[2], p[3]);
            let m = xs3.len();
            let mut j = vec![0.0_f64; m * 4];
            for (i, &x) in xs3.iter().enumerate() {
                let arg = freq * x + phase;
                j[i * 4] = arg.sin(); // ∂/∂A
                j[i * 4 + 1] = amp * x * arg.cos(); // ∂/∂B
                j[i * 4 + 2] = amp * arg.cos(); // ∂/∂C
                j[i * 4 + 3] = 1.0; // ∂/∂D
            }
            Ok(j)
        };
        let cfg = LmConfig {
            max_iter: 5000,
            ..Default::default()
        };
        // Reasonable initial guess
        let result =
            levenberg_marquardt(residual, jacobian, &[1.5, 1.0, 0.0, 0.5], &cfg).expect("sine fit");
        // Verify fit quality (residual norm small, not necessarily exact params due to local min)
        assert!(
            result.residual_norm < 0.1,
            "sine fit residual norm = {:.4}",
            result.residual_norm
        );
        // The recovered A*sin(Bx+C)+D should produce values close to true data
        let xs_check: Vec<f64> = (0..5).map(|i| i as f64 * 0.4).collect();
        let p = &result.params;
        for &x in &xs_check {
            let pred = p[0] * (p[1] * x + p[2]).sin() + p[3];
            let exact = 2.0 * (1.5 * x + 0.5).sin() + 1.0;
            assert!(
                (pred - exact).abs() < 1.0e-3,
                "x={x:.1} pred={pred:.4} exact={exact:.4}"
            );
        }
    }
}
