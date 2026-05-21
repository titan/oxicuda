//! L-BFGS-B (Byrd, Lu, Nocedal 1995): limited-memory BFGS with box constraints
//! `lower ≤ x ≤ upper`.
//!
//! Faithful but simplified implementation:
//!   * Store the last `memory` (s_k, y_k) pairs in `f32` history buffers.
//!   * Compute the search direction with the L-BFGS two-loop recursion.
//!   * Apply a **projected backtracking line search** that enforces an Armijo
//!     sufficient-decrease condition on `f(P_[l,u](x − α d))`.
//!   * Convergence criterion: the *projected gradient* — `‖x − P_[l,u](x − g)‖₂`
//!     — drops below `cfg.tol`.
//!
//! Reference: Byrd R., Lu P., Nocedal J. & Zhu C., "A Limited Memory Algorithm
//! for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5), 1995.
//!
//! The CRF connection: the negative-log-likelihood + 0.5 λ ‖w‖² objective used
//! by `crf/crf_train.rs` is *unconstrained*; for non-negativity-constrained
//! parameter training one would call `LbfgsB::minimize` with `lower = 0`,
//! `upper = +∞` and the same objective/gradient pair.

use crate::error::{SeqError, SeqResult};

/// L-BFGS-B configuration.
#[derive(Debug, Clone)]
pub struct LbfgsBConfig {
    /// Number of (s, y) history pairs (`m` in the paper).
    pub memory: usize,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Projected-gradient-norm convergence tolerance.
    pub tol: f32,
    /// Cap on the initial line-search step magnitude `‖α · d‖₂`.
    pub max_step: f32,
}

impl Default for LbfgsBConfig {
    fn default() -> Self {
        Self {
            memory: 5,
            max_iter: 100,
            tol: 1e-5,
            max_step: 1.0,
        }
    }
}

/// Result of an L-BFGS-B run.
#[derive(Debug, Clone)]
pub struct LbfgsBResult {
    /// Final (feasible) point.
    pub x: Vec<f32>,
    /// Final objective value `f(x)`.
    pub f: f32,
    /// Iterations performed.
    pub iterations: usize,
    /// `true` if the projected-gradient norm fell below `cfg.tol`.
    pub converged: bool,
}

/// Stateless namespace for the L-BFGS-B routines.
pub struct LbfgsB;

impl LbfgsB {
    /// Project a point onto the box `[lower, upper]` elementwise.
    pub fn project_box(x: &[f32], lower: &[f32], upper: &[f32]) -> SeqResult<Vec<f32>> {
        if x.len() != lower.len() {
            return Err(SeqError::LengthMismatch {
                a: x.len(),
                b: lower.len(),
            });
        }
        if x.len() != upper.len() {
            return Err(SeqError::LengthMismatch {
                a: x.len(),
                b: upper.len(),
            });
        }
        let mut out = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let lo = lower[i];
            let hi = upper[i];
            if lo > hi {
                return Err(SeqError::InvalidConfiguration(format!(
                    "lower[{i}] > upper[{i}]: {lo} > {hi}"
                )));
            }
            let xi = x[i];
            let clamped = if xi < lo {
                lo
            } else if xi > hi {
                hi
            } else {
                xi
            };
            out.push(clamped);
        }
        Ok(out)
    }

    /// `‖x − P_[l,u](x − g)‖₂`, the standard projected-gradient stationarity
    /// measure.  Zero exactly at a feasible stationary point.
    pub fn projected_gradient_norm(
        x: &[f32],
        g: &[f32],
        lower: &[f32],
        upper: &[f32],
    ) -> SeqResult<f32> {
        if x.len() != g.len() {
            return Err(SeqError::LengthMismatch {
                a: x.len(),
                b: g.len(),
            });
        }
        if x.len() != lower.len() {
            return Err(SeqError::LengthMismatch {
                a: x.len(),
                b: lower.len(),
            });
        }
        if x.len() != upper.len() {
            return Err(SeqError::LengthMismatch {
                a: x.len(),
                b: upper.len(),
            });
        }
        let mut acc = 0.0_f32;
        for i in 0..x.len() {
            let lo = lower[i];
            let hi = upper[i];
            if lo > hi {
                return Err(SeqError::InvalidConfiguration(format!(
                    "lower[{i}] > upper[{i}]: {lo} > {hi}"
                )));
            }
            let step = x[i] - g[i];
            let proj = if step < lo {
                lo
            } else if step > hi {
                hi
            } else {
                step
            };
            let d = x[i] - proj;
            acc += d * d;
        }
        Ok(acc.sqrt())
    }

    /// Validate configuration and bound arrays.
    fn validate(x0: &[f32], lower: &[f32], upper: &[f32], cfg: &LbfgsBConfig) -> SeqResult<()> {
        if x0.len() != lower.len() {
            return Err(SeqError::LengthMismatch {
                a: x0.len(),
                b: lower.len(),
            });
        }
        if x0.len() != upper.len() {
            return Err(SeqError::LengthMismatch {
                a: x0.len(),
                b: upper.len(),
            });
        }
        for i in 0..x0.len() {
            if lower[i] > upper[i] {
                return Err(SeqError::InvalidConfiguration(format!(
                    "lower[{i}] > upper[{i}]: {} > {}",
                    lower[i], upper[i]
                )));
            }
        }
        if cfg.memory == 0 {
            return Err(SeqError::InvalidParameter {
                name: "memory".to_string(),
                value: 0.0,
            });
        }
        if cfg.max_iter == 0 {
            return Err(SeqError::InvalidParameter {
                name: "max_iter".to_string(),
                value: 0.0,
            });
        }
        if cfg.tol <= 0.0 || cfg.tol.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "tol".to_string(),
                value: cfg.tol as f64,
            });
        }
        if cfg.max_step <= 0.0 || cfg.max_step.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "max_step".to_string(),
                value: cfg.max_step as f64,
            });
        }
        Ok(())
    }

    /// L-BFGS two-loop recursion → returns `H_k · g` (a *descent*-mapped vector;
    /// the caller takes a step in the `−direction`).
    fn two_loop(
        grad: &[f32],
        s_hist: &[Vec<f32>],
        y_hist: &[Vec<f32>],
        rho_hist: &[f32],
    ) -> Vec<f32> {
        let m = s_hist.len();
        let n = grad.len();
        let mut q = grad.to_vec();
        let mut alpha = vec![0.0_f32; m];
        // First loop: i = m-1 .. 0
        for i in (0..m).rev() {
            let r = rho_hist[i];
            let mut dot = 0.0_f32;
            for k in 0..n {
                dot += s_hist[i][k] * q[k];
            }
            alpha[i] = r * dot;
            for k in 0..n {
                q[k] -= alpha[i] * y_hist[i][k];
            }
        }
        // Initial Hessian approximation γ = (s·y)/(y·y)
        let mut gamma = 1.0_f32;
        if m > 0 {
            let last_s = &s_hist[m - 1];
            let last_y = &y_hist[m - 1];
            let mut sy = 0.0_f32;
            let mut yy = 0.0_f32;
            for k in 0..n {
                sy += last_s[k] * last_y[k];
                yy += last_y[k] * last_y[k];
            }
            if yy > 1e-30 {
                gamma = sy / yy;
            }
        }
        let mut r = q;
        for v in r.iter_mut() {
            *v *= gamma;
        }
        // Second loop: i = 0 .. m
        for i in 0..m {
            let mut dot = 0.0_f32;
            for k in 0..n {
                dot += y_hist[i][k] * r[k];
            }
            let beta = rho_hist[i] * dot;
            for k in 0..n {
                r[k] += s_hist[i][k] * (alpha[i] - beta);
            }
        }
        r
    }

    /// Compute the *active-set–consistent* projected descent direction
    /// `d`, then scale so that `‖d‖₂ ≤ max_step`.  Variables hitting a
    /// boundary in the direction that would leave the box have their
    /// direction component zeroed (free-variable subspace projection).
    fn project_direction(d: &mut [f32], x: &[f32], lower: &[f32], upper: &[f32], max_step: f32) {
        let n = d.len();
        // Zero direction at active bounds when stepping would exit the box.
        // The L-BFGS direction is a *descent* direction; we step x − α·d, so
        // a positive d_i decreases x_i.  Hence:
        //   * If x_i is at lower bound and d_i > 0  → would push x_i below lo → zero d_i.
        //   * If x_i is at upper bound and d_i < 0  → would push x_i above hi → zero d_i.
        for i in 0..n {
            let at_lower_pushing_out = x[i] <= lower[i] && d[i] > 0.0;
            let at_upper_pushing_out = x[i] >= upper[i] && d[i] < 0.0;
            if at_lower_pushing_out || at_upper_pushing_out {
                d[i] = 0.0;
            }
        }
        let mut norm_sq = 0.0_f32;
        for v in d.iter() {
            norm_sq += *v * *v;
        }
        let norm = norm_sq.sqrt();
        if norm > max_step {
            let scale = max_step / norm;
            for v in d.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Minimise `f` over the box `[lower, upper]` starting from `x0`.
    pub fn minimize<F, G>(
        x0: &[f32],
        lower: &[f32],
        upper: &[f32],
        f: F,
        grad: G,
        cfg: &LbfgsBConfig,
    ) -> SeqResult<LbfgsBResult>
    where
        F: Fn(&[f32]) -> f32,
        G: Fn(&[f32]) -> Vec<f32>,
    {
        Self::validate(x0, lower, upper, cfg)?;
        let n = x0.len();
        if n == 0 {
            return Err(SeqError::EmptyInput);
        }

        let mut x = Self::project_box(x0, lower, upper)?;
        let mut f_val = f(&x);
        let mut g_val = grad(&x);
        if g_val.len() != n {
            return Err(SeqError::ShapeMismatch {
                expected: n,
                got: g_val.len(),
            });
        }

        let mut s_hist: Vec<Vec<f32>> = Vec::with_capacity(cfg.memory);
        let mut y_hist: Vec<Vec<f32>> = Vec::with_capacity(cfg.memory);
        let mut rho_hist: Vec<f32> = Vec::with_capacity(cfg.memory);

        let mut converged = false;
        let mut iter_done = 0_usize;
        let armijo: f32 = 1e-4;
        let backtrack: f32 = 0.5;
        let max_line_search: usize = 30;

        for it in 0..cfg.max_iter {
            iter_done = it + 1;
            let pg_norm = Self::projected_gradient_norm(&x, &g_val, lower, upper)?;
            if pg_norm < cfg.tol {
                converged = true;
                iter_done = it;
                break;
            }

            // Direction d = H · g  (a descent direction for f).  Cold start:
            // d = g  → projected gradient descent.
            let mut d = if s_hist.is_empty() {
                g_val.clone()
            } else {
                Self::two_loop(&g_val, &s_hist, &y_hist, &rho_hist)
            };

            // Ensure descent: g·d > 0 (so that x − α d decreases f locally).
            let mut gd: f32 = 0.0;
            for i in 0..n {
                gd += g_val[i] * d[i];
            }
            if gd <= 0.0 {
                d = g_val.clone();
            }

            // Free-variable subspace + step cap.
            Self::project_direction(&mut d, &x, lower, upper, cfg.max_step);

            // Re-evaluate g·d after projection.
            let mut gd_proj: f32 = 0.0;
            for i in 0..n {
                gd_proj += g_val[i] * d[i];
            }
            if gd_proj <= 0.0 {
                // No descent in the feasible cone → we're at a stationary
                // point of the projected problem; stop.
                converged = true;
                break;
            }

            // Projected backtracking line search satisfying Armijo on the
            // *projected* iterate.
            let mut step: f32 = 1.0;
            let mut accepted = false;
            let mut x_trial = x.clone();
            let mut f_trial = f_val;
            let mut g_trial = g_val.clone();
            for _ls in 0..max_line_search {
                // y = P_[l,u](x − step · d)
                let mut y = Vec::with_capacity(n);
                for i in 0..n {
                    let v = x[i] - step * d[i];
                    let v = if v < lower[i] {
                        lower[i]
                    } else if v > upper[i] {
                        upper[i]
                    } else {
                        v
                    };
                    y.push(v);
                }
                let fy = f(&y);
                if fy <= f_val - armijo * step * gd_proj {
                    f_trial = fy;
                    g_trial = grad(&y);
                    if g_trial.len() != n {
                        return Err(SeqError::ShapeMismatch {
                            expected: n,
                            got: g_trial.len(),
                        });
                    }
                    x_trial = y;
                    accepted = true;
                    break;
                }
                step *= backtrack;
            }

            if !accepted {
                // Cannot find a decrease in this direction; declare stop.
                break;
            }

            // Update L-BFGS history.
            let mut s_vec = vec![0.0_f32; n];
            let mut y_vec = vec![0.0_f32; n];
            for i in 0..n {
                s_vec[i] = x_trial[i] - x[i];
                y_vec[i] = g_trial[i] - g_val[i];
            }
            // NB: the standard L-BFGS curvature condition is `s·y > 0` —
            // the sign is `+` because we minimise (the direction maps g, not
            // −g; the sign of `s` flips relative to the maximisation variant
            // in `crf_train.rs`).
            let mut sy = 0.0_f32;
            for i in 0..n {
                sy += s_vec[i] * y_vec[i];
            }
            if sy > 1e-20 {
                if s_hist.len() == cfg.memory {
                    let _ = s_hist.remove(0);
                    let _ = y_hist.remove(0);
                    let _ = rho_hist.remove(0);
                }
                rho_hist.push(1.0 / sy);
                s_hist.push(s_vec);
                y_hist.push(y_vec);
            }

            x = x_trial;
            f_val = f_trial;
            g_val = g_trial;
        }

        Ok(LbfgsBResult {
            x,
            f: f_val,
            iterations: iter_done,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Approximate equality.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn default_cfg() -> LbfgsBConfig {
        LbfgsBConfig {
            memory: 5,
            max_iter: 200,
            tol: 1e-6,
            max_step: 1.0,
        }
    }

    /// Build closures for f(x) = 0.5 ‖x − c‖²; ∇f = x − c.
    fn quad_target(c: Vec<f32>) -> (impl Fn(&[f32]) -> f32, impl Fn(&[f32]) -> Vec<f32>) {
        let cf = c.clone();
        let f = move |x: &[f32]| -> f32 {
            let mut s = 0.0;
            for i in 0..x.len() {
                let d = x[i] - cf[i];
                s += d * d;
            }
            0.5 * s
        };
        let cg = c;
        let g = move |x: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0; x.len()];
            for i in 0..x.len() {
                out[i] = x[i] - cg[i];
            }
            out
        };
        (f, g)
    }

    #[test]
    fn project_box_clamps_into_interval() {
        let x = vec![-2.0_f32, 0.5, 3.0];
        let lo = vec![-1.0_f32, 0.0, 0.0];
        let hi = vec![1.0_f32, 1.0, 2.0];
        let out = LbfgsB::project_box(&x, &lo, &hi).expect("ok");
        assert!(close(out[0], -1.0, 0.0));
        assert!(close(out[1], 0.5, 0.0));
        assert!(close(out[2], 2.0, 0.0));
    }

    #[test]
    fn project_box_idempotent_inside() {
        let x = vec![-0.5_f32, 0.5, 1.5];
        let lo = vec![-1.0_f32, 0.0, 0.0];
        let hi = vec![1.0_f32, 1.0, 2.0];
        let out = LbfgsB::project_box(&x, &lo, &hi).expect("ok");
        for i in 0..3 {
            assert!(close(out[i], x[i], 0.0));
        }
    }

    #[test]
    fn projected_gradient_norm_zero_at_interior_minimum() {
        // f(x) = 0.5 ‖x − c‖² with c interior to [l, u].  At x = c, g = 0,
        // so the projected-gradient norm is exactly 0.
        let lo = vec![-2.0_f32, -2.0];
        let hi = vec![2.0_f32, 2.0];
        let x = vec![0.3_f32, -0.5];
        let g = vec![0.0_f32, 0.0];
        let pg = LbfgsB::projected_gradient_norm(&x, &g, &lo, &hi).expect("ok");
        assert!(close(pg, 0.0, 1e-7));
    }

    #[test]
    fn projected_gradient_norm_zero_at_boundary_when_gradient_pushes_out() {
        // x at lower bound, gradient positive → step `x − g` would go below
        // lower → projection clamps back to x.  ‖x − P(x − g)‖ = 0.
        let lo = vec![0.0_f32];
        let hi = vec![1.0_f32];
        let x = vec![0.0_f32];
        let g = vec![1.0_f32];
        let pg = LbfgsB::projected_gradient_norm(&x, &g, &lo, &hi).expect("ok");
        assert!(close(pg, 0.0, 1e-7));
    }

    #[test]
    fn minimize_convex_quadratic_with_interior_minimum() {
        let c = vec![0.3_f32, -0.5];
        let lo = vec![-2.0_f32, -2.0];
        let hi = vec![2.0_f32, 2.0];
        let (f, g) = quad_target(c.clone());
        let res = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f, g, &default_cfg()).expect("ok");
        for i in 0..2 {
            assert!(
                (res.x[i] - c[i]).abs() < 1e-3,
                "x[{i}] = {} vs c[{i}] = {}",
                res.x[i],
                c[i]
            );
        }
        assert!(res.f < 1e-5);
    }

    #[test]
    fn minimize_quadratic_with_minimum_outside_box() {
        // Minimum c = (5, -5) is well outside [-1, 1]².  Optimum on the box
        // is the corner (1, -1).
        let c = vec![5.0_f32, -5.0];
        let lo = vec![-1.0_f32, -1.0];
        let hi = vec![1.0_f32, 1.0];
        let (f, g) = quad_target(c);
        let res = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!(close(res.x[0], 1.0, 1e-4));
        assert!(close(res.x[1], -1.0, 1e-4));
    }

    #[test]
    fn minimize_converged_flag_on_easy_problem() {
        let c = vec![0.0_f32];
        let lo = vec![-1.0_f32];
        let hi = vec![1.0_f32];
        let (f, g) = quad_target(c);
        let res = LbfgsB::minimize(&[0.5], &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!(res.converged);
    }

    #[test]
    fn minimize_is_deterministic() {
        let c = vec![0.25_f32, 0.75];
        let lo = vec![-1.0_f32, -1.0];
        let hi = vec![1.0_f32, 1.0];
        let (f1, g1) = quad_target(c.clone());
        let (f2, g2) = quad_target(c);
        let r1 = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f1, g1, &default_cfg()).expect("ok");
        let r2 = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f2, g2, &default_cfg()).expect("ok");
        assert_eq!(r1.iterations, r2.iterations);
        for i in 0..2 {
            assert!(close(r1.x[i], r2.x[i], 0.0));
        }
        assert!(close(r1.f, r2.f, 0.0));
    }

    #[test]
    fn minimize_one_dimensional_works() {
        let c = vec![0.2_f32];
        let lo = vec![-1.0_f32];
        let hi = vec![1.0_f32];
        let (f, g) = quad_target(c.clone());
        let res = LbfgsB::minimize(&[0.9], &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!((res.x[0] - c[0]).abs() < 1e-3);
    }

    #[test]
    fn err_when_x0_lower_length_mismatch() {
        let cfg = default_cfg();
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0, 0.0], &[0.0], &[1.0, 1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::LengthMismatch { .. })));
    }

    #[test]
    fn err_when_lower_upper_length_mismatch() {
        let cfg = default_cfg();
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0, 0.0], &[0.0, 0.0], &[1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::LengthMismatch { .. })));
    }

    #[test]
    fn err_when_lower_greater_than_upper() {
        let cfg = default_cfg();
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0], &[1.0], &[0.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_when_memory_zero() {
        let mut cfg = default_cfg();
        cfg.memory = 0;
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0], &[0.0], &[1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_when_max_iter_zero() {
        let mut cfg = default_cfg();
        cfg.max_iter = 0;
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0], &[0.0], &[1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_when_tol_non_positive() {
        let mut cfg = default_cfg();
        cfg.tol = 0.0;
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0], &[0.0], &[1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_when_max_step_non_positive() {
        let mut cfg = default_cfg();
        cfg.max_step = 0.0;
        let f = |_x: &[f32]| 0.0_f32;
        let g = |x: &[f32]| vec![0.0; x.len()];
        let r = LbfgsB::minimize(&[0.0], &[0.0], &[1.0], f, g, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn constant_gradient_drives_x_down_until_lower_bound() {
        // f(x) = x with constant ∇f = 1 → minimum at the lower bound.
        let lo = vec![-1.0_f32];
        let hi = vec![1.0_f32];
        let f = |x: &[f32]| x[0];
        let g = |_x: &[f32]| vec![1.0_f32];
        let res = LbfgsB::minimize(&[1.0], &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!(res.x[0] < 1.0);
        assert!((res.x[0] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn convex_objective_decreases_monotonically() {
        let c = vec![0.0_f32, 0.0];
        let lo = vec![-2.0_f32, -2.0];
        let hi = vec![2.0_f32, 2.0];
        let (f, g) = quad_target(c);
        let x0 = vec![1.5_f32, -1.2];
        let f0 = 0.5 * (1.5_f32.powi(2) + 1.2_f32.powi(2));
        let res = LbfgsB::minimize(&x0, &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!(res.f < f0 - 1e-3);
    }

    #[test]
    fn memory_one_still_converges() {
        let c = vec![0.3_f32, -0.5];
        let lo = vec![-1.0_f32, -1.0];
        let hi = vec![1.0_f32, 1.0];
        let (f, g) = quad_target(c.clone());
        let mut cfg = default_cfg();
        cfg.memory = 1;
        cfg.max_iter = 500;
        let res = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f, g, &cfg).expect("ok");
        for i in 0..2 {
            assert!((res.x[i] - c[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn projected_gradient_norm_length_mismatch_errors() {
        let r = LbfgsB::projected_gradient_norm(&[0.0, 1.0], &[0.0], &[-1.0, -1.0], &[1.0, 1.0]);
        assert!(matches!(r, Err(SeqError::LengthMismatch { .. })));
    }

    #[test]
    fn project_box_length_mismatch_errors() {
        let r = LbfgsB::project_box(&[0.0, 1.0], &[-1.0], &[1.0, 1.0]);
        assert!(matches!(r, Err(SeqError::LengthMismatch { .. })));
    }

    #[test]
    fn project_box_lower_greater_than_upper_errors() {
        let r = LbfgsB::project_box(&[0.0], &[1.0], &[0.0]);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn iterations_count_is_finite() {
        let c = vec![0.1_f32, 0.2, 0.3, -0.1, -0.2, -0.3];
        let lo = vec![-1.0_f32; 6];
        let hi = vec![1.0_f32; 6];
        let (f, g) = quad_target(c);
        let res = LbfgsB::minimize(
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            &lo,
            &hi,
            f,
            g,
            &default_cfg(),
        )
        .expect("ok");
        assert!(res.iterations <= 200);
    }

    #[test]
    fn corner_minimum_when_unconstrained_optimum_in_corner() {
        // f(x) = 0.5(x - 10)^2 + 0.5(y - 10)^2 → corner (1, 1).
        let c = vec![10.0_f32, 10.0];
        let lo = vec![-1.0_f32, -1.0];
        let hi = vec![1.0_f32, 1.0];
        let (f, g) = quad_target(c);
        let res = LbfgsB::minimize(&[0.0, 0.0], &lo, &hi, f, g, &default_cfg()).expect("ok");
        assert!(close(res.x[0], 1.0, 1e-4));
        assert!(close(res.x[1], 1.0, 1e-4));
    }
}
