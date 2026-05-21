//! Newton's method on the (concave) dual function.
//!
//! Many convex problems admit a smooth concave dual whose gradient and Hessian
//! are cheap to evaluate. This module implements a damped/regularised Newton
//! ascent on such a dual `g(λ)`, optionally with an Armijo backtracking line
//! search (Boyd-Vandenberghe §9.5, Nocedal-Wright §3.1).
//!
//! The Newton system at step k is
//!   `(H(λ) + μI) Δ = ∇g(λ)`
//! where `H(λ)` is the Hessian of the function `−g` (SPD when `g` is concave).
//! `μ ≥ 0` regularises a numerically stiff Hessian; the resulting matrix is
//! still SPD so we reuse `linalg::solve_dense`. The step is
//!   `λ ← λ + α · Δ`
//! with `α = 1` (Full) or chosen by backtracking on the Armijo sufficient-ascent
//!   `g(λ + α Δ) ≥ g(λ) + c1 · α · ∇g(λ)ᵀ Δ`.
//!
//! Reference: Boyd & Vandenberghe, "Convex Optimization", §5.4 and §9.5.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;
use crate::linalg::solve::solve_dense;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Step-length strategy for the dual Newton iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepKind {
    /// Full Newton step `α = 1`.
    Full,
    /// Backtracking Armijo line search with sufficient-ascent constant `c1`,
    /// contraction `beta`, and a cap of `max_steps` halvings per outer step.
    BacktrackingArmijo {
        c1: f64,
        beta: f64,
        max_steps: usize,
    },
}

/// Configuration for dual Newton.
#[derive(Debug, Clone)]
pub struct DualNewtonConfig {
    /// Maximum number of Newton iterations.
    pub max_iter: usize,
    /// Stop when `||∇g(λ)||_2 < tol`.
    pub tol: f64,
    /// Damping `μ ≥ 0` added to the diagonal of `H` (regularises a stiff Hessian).
    pub hessian_regularization: f64,
    /// Step-length strategy.
    pub step_kind: StepKind,
    /// Length of the dual variable.
    pub n_dual: usize,
}

/// State returned by `newton_on_dual`.
#[derive(Debug, Clone)]
pub struct DualNewtonState {
    /// Final dual iterate.
    pub lambda: Vec<f64>,
    /// Number of Newton iterations completed.
    pub iter: usize,
    /// Value of `g(λ)` recorded at the END of each accepted step (length =
    /// `iter`). Monotonically non-decreasing under Armijo.
    pub dual_objective_history: Vec<f64>,
    /// `||∇g(λ)||_2` recorded at the END of each iteration (length = `iter`).
    pub dual_grad_norm_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_cfg(cfg: &DualNewtonConfig) -> CvxResult<()> {
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter("max_iter must be >= 1".into()));
    }
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0 and finite, got {}",
            cfg.tol
        )));
    }
    if cfg.hessian_regularization < 0.0 || !cfg.hessian_regularization.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "hessian_regularization must be >= 0 and finite, got {}",
            cfg.hessian_regularization
        )));
    }
    if cfg.n_dual == 0 {
        return Err(CvxError::EmptyInput);
    }
    if let StepKind::BacktrackingArmijo {
        c1,
        beta,
        max_steps,
    } = cfg.step_kind
    {
        if !(c1 > 0.0 && c1 < 1.0) {
            return Err(CvxError::InvalidParameter(format!(
                "Armijo c1 must be in (0, 1), got {c1}"
            )));
        }
        if !(beta > 0.0 && beta < 1.0) {
            return Err(CvxError::InvalidParameter(format!(
                "Armijo beta must be in (0, 1), got {beta}"
            )));
        }
        if max_steps == 0 {
            return Err(CvxError::InvalidParameter(
                "Armijo max_steps must be >= 1".into(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64], b: &[f64]) -> CvxResult<f64> {
    if a.len() != b.len() {
        return Err(CvxError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let mut s = 0.0_f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        s += ai * bi;
    }
    Ok(s)
}

/// Build `H + μI` in a freshly-allocated row-major buffer.
fn regularised_hessian(h: &[f64], n: usize, mu: f64) -> CvxResult<Vec<f64>> {
    if h.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![h.len()],
        });
    }
    let mut out = h.to_vec();
    let out_len = out.len();
    if mu > 0.0 {
        for i in 0..n {
            let idx = i * n + i;
            let cell = out.get_mut(idx).ok_or(CvxError::IndexOutOfBounds {
                index: idx,
                len: out_len,
            })?;
            *cell += mu;
        }
    }
    Ok(out)
}

fn check_finite_vec(v: &[f64], label: &str) -> CvxResult<()> {
    for (i, &val) in v.iter().enumerate() {
        if !val.is_finite() {
            return Err(CvxError::NumericalInstability(format!(
                "{label}[{i}] is not finite (got {val})"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Newton's method on the concave dual `g(λ)`.
///
/// - `initial_lambda`: starting dual iterate, length `cfg.n_dual`.
/// - `dual_objective(λ) -> f64`: the value of `g(λ)`.
/// - `dual_gradient(λ) -> Vec<f64>`: `∇g(λ)` (length `n_dual`).
/// - `dual_hessian(λ) -> Vec<f64>`: Hessian of `−g`, row-major `n_dual²`, SPD
///   when `g` is concave.
///
/// Solves the Newton system `(H(λ) + μI) Δ = ∇g(λ)` (ascent direction on `g`,
/// equivalently descent on `−g`) at every iteration. Backtracking Armijo
/// performs `α ← α·β` until `g(λ + α Δ) ≥ g(λ) + c1 · α · ∇g(λ)ᵀ Δ`. Stops when
/// `||∇g(λ)||_2 < cfg.tol`.
pub fn newton_on_dual<F, G, H>(
    initial_lambda: &[f64],
    dual_objective: F,
    dual_gradient: G,
    dual_hessian: H,
    cfg: &DualNewtonConfig,
) -> CvxResult<DualNewtonState>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<f64>,
{
    validate_cfg(cfg)?;
    if initial_lambda.len() != cfg.n_dual {
        return Err(CvxError::DimensionMismatch {
            a: initial_lambda.len(),
            b: cfg.n_dual,
        });
    }
    let n = cfg.n_dual;
    let mu = cfg.hessian_regularization;

    let mut lambda = initial_lambda.to_vec();
    let mut obj_hist = Vec::with_capacity(cfg.max_iter);
    let mut grad_hist = Vec::with_capacity(cfg.max_iter);

    for it in 0..cfg.max_iter {
        let grad = dual_gradient(&lambda);
        if grad.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: grad.len(),
                b: n,
            });
        }
        check_finite_vec(&grad, "dual_gradient")?;
        let gnorm = norm2(&grad);
        if gnorm < cfg.tol {
            // Already converged; record the current state and exit.
            let cur_obj = dual_objective(&lambda);
            if !cur_obj.is_finite() {
                return Err(CvxError::NumericalInstability(
                    "dual_objective is not finite at converged point".into(),
                ));
            }
            obj_hist.push(cur_obj);
            grad_hist.push(gnorm);
            return Ok(DualNewtonState {
                lambda,
                iter: it + 1,
                dual_objective_history: obj_hist,
                dual_grad_norm_history: grad_hist,
            });
        }

        let hess = dual_hessian(&lambda);
        if hess.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![hess.len()],
            });
        }
        check_finite_vec(&hess, "dual_hessian")?;
        let hess_reg = regularised_hessian(&hess, n, mu)?;

        // Solve (H + μI) Δ = ∇g  → Δ is an ASCENT direction on g.
        let delta = solve_dense(&hess_reg, n, &grad)?;
        check_finite_vec(&delta, "newton_direction")?;

        let dir_dot_grad = dot(&grad, &delta)?;
        // For an SPD (H + μI) and a non-zero gradient, ∇gᵀ Δ > 0 (ascent).
        // If somehow zero / negative (numerical edge), fall back to gradient
        // ascent direction.
        let use_grad_fallback = dir_dot_grad <= 0.0;
        let (effective_delta, effective_dir_dot_grad) = if use_grad_fallback {
            let g_dot_g = dot(&grad, &grad)?;
            (grad.clone(), g_dot_g)
        } else {
            (delta, dir_dot_grad)
        };

        let cur_obj = dual_objective(&lambda);
        if !cur_obj.is_finite() {
            return Err(CvxError::NumericalInstability(
                "dual_objective is not finite at current λ".into(),
            ));
        }

        let alpha = match cfg.step_kind {
            StepKind::Full => 1.0_f64,
            StepKind::BacktrackingArmijo {
                c1,
                beta,
                max_steps,
            } => {
                let mut a = 1.0_f64;
                let mut accepted = false;
                for _ in 0..max_steps {
                    let mut trial = lambda.clone();
                    let trial_len = trial.len();
                    for i in 0..n {
                        let di = *effective_delta.get(i).ok_or(CvxError::IndexOutOfBounds {
                            index: i,
                            len: effective_delta.len(),
                        })?;
                        let cell = trial.get_mut(i).ok_or(CvxError::IndexOutOfBounds {
                            index: i,
                            len: trial_len,
                        })?;
                        *cell += a * di;
                    }
                    let trial_obj = dual_objective(&trial);
                    if trial_obj.is_finite()
                        && trial_obj >= cur_obj + c1 * a * effective_dir_dot_grad
                    {
                        accepted = true;
                        break;
                    }
                    a *= beta;
                }
                if !accepted {
                    return Err(CvxError::LineSearchFailed(format!(
                        "Armijo failed after {max_steps} backtracking steps at iter {it}"
                    )));
                }
                a
            }
        };

        // λ ← λ + α · Δ
        for i in 0..n {
            let di = *effective_delta.get(i).ok_or(CvxError::IndexOutOfBounds {
                index: i,
                len: effective_delta.len(),
            })?;
            let cell = lambda
                .get_mut(i)
                .ok_or(CvxError::IndexOutOfBounds { index: i, len: n })?;
            *cell += alpha * di;
        }
        let new_obj = dual_objective(&lambda);
        if !new_obj.is_finite() {
            return Err(CvxError::NumericalInstability(
                "dual_objective is not finite after Newton update".into(),
            ));
        }
        obj_hist.push(new_obj);
        let new_grad = dual_gradient(&lambda);
        if new_grad.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: new_grad.len(),
                b: n,
            });
        }
        let new_gnorm = norm2(&new_grad);
        grad_hist.push(new_gnorm);
        if new_gnorm < cfg.tol {
            return Ok(DualNewtonState {
                lambda,
                iter: it + 1,
                dual_objective_history: obj_hist,
                dual_grad_norm_history: grad_hist,
            });
        }
    }
    Ok(DualNewtonState {
        lambda,
        iter: cfg.max_iter,
        dual_objective_history: obj_hist,
        dual_grad_norm_history: grad_hist,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
mod tests {
    use super::*;

    // For a quadratic dual g(λ) = −½ λᵀM λ + qᵀλ with M SPD:
    //   ∇g(λ) = q − M λ
    //   Hess of −g = M (SPD).
    //   Optimum: λ* = M⁻¹ q.
    fn make_m(d: usize, seed: f64) -> Vec<f64> {
        let mut a = vec![0.0_f64; d * d];
        let mut basis = vec![0.0_f64; d * d];
        for i in 0..d {
            for j in 0..d {
                basis[i * d + j] = ((i as f64 + 1.0) * (j as f64 + 0.7) + seed).cos();
            }
        }
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0_f64;
                for k in 0..d {
                    s += basis[i * d + k] * basis[j * d + k];
                }
                a[i * d + j] = s;
                if i == j {
                    a[i * d + j] += (d as f64) + 0.5;
                }
            }
        }
        a
    }

    fn make_q(d: usize, seed: f64) -> Vec<f64> {
        (0..d)
            .map(|i| ((i as f64 + 1.0) * 0.4 + seed).sin())
            .collect()
    }

    fn quadratic_dual_oracles(
        m: Vec<f64>,
        q: Vec<f64>,
        d: usize,
    ) -> (
        impl Fn(&[f64]) -> f64,
        impl Fn(&[f64]) -> Vec<f64>,
        impl Fn(&[f64]) -> Vec<f64>,
    ) {
        let m_obj = m.clone();
        let q_obj = q.clone();
        let m_g = m.clone();
        let q_g = q.clone();
        let m_h = m;
        let obj = move |lam: &[f64]| -> f64 {
            // g(λ) = −½ λᵀ M λ + qᵀ λ
            let mut quad = 0.0_f64;
            for i in 0..d {
                let mut row = 0.0_f64;
                for j in 0..d {
                    row += m_obj[i * d + j] * lam[j];
                }
                quad += lam[i] * row;
            }
            let lin: f64 = lam.iter().zip(q_obj.iter()).map(|(l, q)| l * q).sum();
            -0.5 * quad + lin
        };
        let grad = move |lam: &[f64]| -> Vec<f64> {
            // ∇g = q − M λ
            (0..d)
                .map(|i| {
                    let mut s = 0.0_f64;
                    for j in 0..d {
                        s += m_g[i * d + j] * lam[j];
                    }
                    q_g[i] - s
                })
                .collect()
        };
        let hess = move |_lam: &[f64]| -> Vec<f64> {
            // Hessian of −g is M.
            m_h.clone()
        };
        (obj, grad, hess)
    }

    #[test]
    fn full_step_one_iter_to_optimum() {
        let d = 4usize;
        let m = make_m(d, 0.1);
        let q = make_q(d, 0.3);
        let lambda_star = solve_dense(&m, d, &q).expect("ref");

        let cfg = DualNewtonConfig {
            max_iter: 5,
            tol: 1.0e-9,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        // First Newton step solves Mx = q exactly → converges in 1 iter.
        assert!(state.iter <= 2, "got {} iters", state.iter);
        for i in 0..d {
            assert!(
                (state.lambda[i] - lambda_star[i]).abs() < 1.0e-9,
                "coord {i}: nd={} lstar={}",
                state.lambda[i],
                lambda_star[i]
            );
        }
    }

    #[test]
    fn damped_newton_converges_with_mu() {
        let d = 4usize;
        let m = make_m(d, 0.6);
        let q = make_q(d, 0.5);
        let lambda_star = solve_dense(&m, d, &q).expect("ref");

        let cfg = DualNewtonConfig {
            max_iter: 200,
            tol: 1.0e-8,
            hessian_regularization: 0.1,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        for i in 0..d {
            assert!(
                (state.lambda[i] - lambda_star[i]).abs() < 1.0e-5,
                "coord {i}: damped={} lstar={}",
                state.lambda[i],
                lambda_star[i]
            );
        }
        // Damped should take more than 1 iter.
        assert!(state.iter >= 2);
    }

    #[test]
    fn armijo_objective_history_non_decreasing() {
        let d = 4usize;
        let m = make_m(d, 0.4);
        let q = make_q(d, 0.2);
        let cfg = DualNewtonConfig {
            max_iter: 50,
            tol: 1.0e-9,
            hessian_regularization: 0.0,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 1.0e-4,
                beta: 0.5,
                max_steps: 50,
            },
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![1.0; d], obj, grad, hess, &cfg).expect("ok");
        for w in state.dual_objective_history.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-12,
                "objective decreased: prev={} next={}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn full_step_well_conditioned() {
        let d = 3usize;
        // Identity-ish M for a well-conditioned problem.
        let m = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let q = vec![1.0, -2.0, 3.0];
        let cfg = DualNewtonConfig {
            max_iter: 5,
            tol: 1.0e-10,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q.clone(), d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        let expected = [0.5, -1.0, 1.5];
        for i in 0..d {
            assert!((state.lambda[i] - expected[i]).abs() < 1.0e-9);
        }
    }

    #[test]
    fn regularisation_stabilises_stiff_problem() {
        let d = 3usize;
        // Stiff but SPD M: condition ≈ 1e3 yet finite.
        let m = vec![1.0e3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-3];
        let q = vec![1.0, 1.0, 1.0];
        let cfg = DualNewtonConfig {
            max_iter: 200,
            tol: 1.0e-8,
            hessian_regularization: 1.0e-2,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        // Last gradient norm should drop below tol.
        if let Some(&g) = state.dual_grad_norm_history.last() {
            assert!(g < 1.0e-7, "final grad={g}");
        }
    }

    #[test]
    fn armijo_increases_objective_each_step() {
        // Quadratic-plus-linear dual: monotone increases along accepted Armijo steps.
        let d = 3usize;
        let m = make_m(d, 0.9);
        let q = make_q(d, 0.7);
        let cfg = DualNewtonConfig {
            max_iter: 30,
            tol: 1.0e-8,
            hessian_regularization: 0.05,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 1.0e-4,
                beta: 0.5,
                max_steps: 30,
            },
            n_dual: d,
        };
        let (obj_fn, grad, hess) = quadratic_dual_oracles(m, q, d);
        let init = vec![5.0_f64, -3.0, 2.0];
        let g0 = obj_fn(&init);
        let state = newton_on_dual(&init, obj_fn, grad, hess, &cfg).expect("ok");
        // At least the first recorded objective must be >= initial.
        let first = *state
            .dual_objective_history
            .first()
            .expect("at least 1 history entry");
        assert!(first >= g0 - 1.0e-10);
    }

    #[test]
    fn newton_deterministic_repeated_runs() {
        let d = 3usize;
        let m = make_m(d, 0.2);
        let q = make_q(d, 0.1);
        let cfg = DualNewtonConfig {
            max_iter: 30,
            tol: 1.0e-10,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj1, grad1, hess1) = quadratic_dual_oracles(m.clone(), q.clone(), d);
        let s1 = newton_on_dual(&vec![0.0; d], obj1, grad1, hess1, &cfg).expect("ok");
        let (obj2, grad2, hess2) = quadratic_dual_oracles(m, q, d);
        let s2 = newton_on_dual(&vec![0.0; d], obj2, grad2, hess2, &cfg).expect("ok");
        assert_eq!(s1.iter, s2.iter);
        for i in 0..d {
            assert!((s1.lambda[i] - s2.lambda[i]).abs() < 1.0e-15);
        }
    }

    #[test]
    fn gradient_at_converged_point_below_tol() {
        let d = 4usize;
        let m = make_m(d, 0.55);
        let q = make_q(d, 0.66);
        let cfg = DualNewtonConfig {
            max_iter: 20,
            tol: 1.0e-9,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        let last = state
            .dual_grad_norm_history
            .last()
            .copied()
            .unwrap_or(f64::INFINITY);
        assert!(last < cfg.tol);
    }

    #[test]
    fn err_max_iter_zero() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 0,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_tol_nonpositive() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 0.0,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_mu_negative() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: -0.1,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_armijo_c1_out_of_range() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 1.0,
                beta: 0.5,
                max_steps: 10,
            },
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_armijo_beta_out_of_range() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 0.5,
                beta: 1.5,
                max_steps: 10,
            },
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_armijo_max_steps_zero() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 0.1,
                beta: 0.5,
                max_steps: 0,
            },
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_initial_lambda_wrong_dim() {
        let d = 3usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; 2],
            |_l| 0.0,
            |_l| vec![0.0; d],
            |_l| vec![1.0; d * d],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_nan_in_gradient() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            |_l| vec![f64::NAN; d],
            |_l| vec![1.0, 0.0, 0.0, 1.0],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn err_nan_in_hessian() {
        let d = 2usize;
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let res = newton_on_dual(
            &vec![0.0; d],
            |_l| 0.0,
            // Non-zero gradient to ensure we get past the early-exit check.
            |_l| vec![1.0, 1.0],
            |_l| vec![f64::NAN; 4],
            &cfg,
        );
        assert!(res.is_err());
    }

    #[test]
    fn history_lengths_match_iter() {
        let d = 3usize;
        let m = make_m(d, 0.31);
        let q = make_q(d, 0.41);
        let cfg = DualNewtonConfig {
            max_iter: 5,
            tol: 1.0e-15,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        assert_eq!(state.dual_objective_history.len(), state.iter);
        assert_eq!(state.dual_grad_norm_history.len(), state.iter);
    }

    #[test]
    fn n_dual_zero_errors() {
        let cfg = DualNewtonConfig {
            max_iter: 10,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::Full,
            n_dual: 0,
        };
        let res = newton_on_dual(&vec![], |_l| 0.0, |_l| vec![], |_l| vec![], &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn armijo_converges_quadratic() {
        let d = 4usize;
        let m = make_m(d, 0.12);
        let q = make_q(d, 0.34);
        let lambda_star = solve_dense(&m, d, &q).expect("ref");
        let cfg = DualNewtonConfig {
            max_iter: 50,
            tol: 1.0e-8,
            hessian_regularization: 0.0,
            step_kind: StepKind::BacktrackingArmijo {
                c1: 1.0e-4,
                beta: 0.5,
                max_steps: 30,
            },
            n_dual: d,
        };
        let (obj, grad, hess) = quadratic_dual_oracles(m, q, d);
        let state = newton_on_dual(&vec![0.0; d], obj, grad, hess, &cfg).expect("ok");
        for i in 0..d {
            assert!(
                (state.lambda[i] - lambda_star[i]).abs() < 1.0e-6,
                "coord {i}: arm={} lstar={}",
                state.lambda[i],
                lambda_star[i]
            );
        }
    }
}
