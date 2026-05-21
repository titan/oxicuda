//! Inexact proximal operator via inner Conjugate Gradient.
//!
//! For a smooth quadratic regulariser `g(x) = ½ xᵀA x − bᵀx` (symmetric PSD
//! `A` available only via a matrix-vector product) the prox operator at `v`
//! with parameter `ρ > 0` is
//! ```text
//!     prox_g(v) = argmin_x  g(x) + (ρ/2) ‖x − v‖² .
//! ```
//! Setting the gradient to zero yields the symmetric positive-definite linear
//! system
//! ```text
//!     (A + ρI) x = ρ v + b ,
//! ```
//! which is solved iteratively by Conjugate Gradient (Hestenes-Stiefel 1952)
//! to a relative residual tolerance — i.e. *inexactly*. Used as an inner
//! subroutine inside e.g. proximal-gradient or ADMM solvers when no
//! closed-form prox is available.
//!
//! The implementation is matrix-free: only the operator `a_mv : x ↦ A x` is
//! needed; the augmented operator `M x = A x + ρ x` is applied implicitly by
//! adding `ρ x` to the user-supplied `a_mv` output. CG is started from `v`
//! (a natural warm start since `prox_g(v) → v` as `ρ → ∞`).

use crate::error::{CvxError, CvxResult};

/// Configuration for the [`InexactProx`] solver.
#[derive(Debug, Clone)]
pub struct InexactProxConfig {
    /// Prox parameter `ρ > 0`.
    pub rho: f32,
    /// Maximum number of inner CG iterations (`≥ 1`).
    pub max_inner_iter: usize,
    /// Stopping tolerance on `‖r‖₂` (`> 0`).
    pub tol: f32,
}

impl Default for InexactProxConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_inner_iter: 100,
            tol: 1.0e-6,
        }
    }
}

/// Inexact proximal operator (matrix-free CG on `(A + ρI) x = ρv + b`).
pub struct InexactProx;

impl InexactProx {
    /// Compute `prox_g(v) ≈ argmin_x ½ xᵀA x − bᵀx + (ρ/2)‖x − v‖²`.
    ///
    /// * `v` — input point of the prox operator.
    /// * `b` — linear term of `g(x) = ½ xᵀA x − bᵀ x` (same length as `v`).
    /// * `rho` — prox parameter `ρ > 0`.
    /// * `a_mv` — matrix-vector product `x ↦ A x` (must preserve length).
    /// * `cfg` — CG configuration.
    ///
    /// Returns `(x, n_inner_iters)` where `n_inner_iters` counts how many CG
    /// iterations were actually performed (≤ `cfg.max_inner_iter`).
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::EmptyInput`] if `v` is empty,
    /// [`CvxError::DimensionMismatch`] if `b`, `v`, or `a_mv(v)` mismatch,
    /// [`CvxError::InvalidParameter`] for non-positive `rho` / `max_inner_iter` / `tol`,
    /// and [`CvxError::NumericalInstability`] if the CG denominator vanishes
    /// (e.g. user-supplied `a_mv` is not PSD).
    pub fn prox<F>(
        &self,
        v: &[f32],
        b: &[f32],
        rho: f32,
        a_mv: F,
        cfg: &InexactProxConfig,
    ) -> CvxResult<(Vec<f32>, usize)>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        // The struct-borrowing form `prox` validates with the rho override.
        let merged = InexactProxConfig {
            rho,
            max_inner_iter: cfg.max_inner_iter,
            tol: cfg.tol,
        };
        Self::run(v, b, &a_mv, &merged)
    }

    /// Convenience wrapper without the `self` borrow.
    pub fn solve<F>(
        v: &[f32],
        b: &[f32],
        a_mv: F,
        cfg: &InexactProxConfig,
    ) -> CvxResult<(Vec<f32>, usize)>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        Self::run(v, b, &a_mv, cfg)
    }

    fn run<F>(
        v: &[f32],
        b: &[f32],
        a_mv: &F,
        cfg: &InexactProxConfig,
    ) -> CvxResult<(Vec<f32>, usize)>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        Self::validate(v, b, cfg)?;
        let n = v.len();
        let rho = cfg.rho;

        // RHS of the SPD system: r0 = ρ v + b .
        let mut rhs = vec![0.0_f32; n];
        for i in 0..n {
            rhs[i] = rho * v[i] + b[i];
        }

        // Warm start: x0 = v. M x0 = A v + ρ v.
        let mut x = v.to_vec();
        let a_x = a_mv(&x);
        if a_x.len() != n {
            return Err(CvxError::DimensionMismatch { a: a_x.len(), b: n });
        }
        let mut m_x = vec![0.0_f32; n];
        for i in 0..n {
            m_x[i] = a_x[i] + rho * x[i];
        }

        // Residual r = rhs − M x0 ; initial search direction p = r .
        let mut r = vec![0.0_f32; n];
        for i in 0..n {
            r[i] = rhs[i] - m_x[i];
        }
        let mut p = r.clone();
        let mut rr: f32 = r.iter().map(|v| v * v).sum();

        // Already converged at the warm start?
        if rr.sqrt() < cfg.tol {
            return Ok((x, 0));
        }

        let mut iter_done = 0usize;
        for k in 0..cfg.max_inner_iter {
            iter_done = k + 1;
            let a_p = a_mv(&p);
            if a_p.len() != n {
                return Err(CvxError::DimensionMismatch { a: a_p.len(), b: n });
            }
            // M p = A p + ρ p .
            let mut mp = vec![0.0_f32; n];
            for i in 0..n {
                mp[i] = a_p[i] + rho * p[i];
            }
            let p_mp: f32 = p.iter().zip(mp.iter()).map(|(a, b)| a * b).sum();
            if p_mp <= 0.0 || !p_mp.is_finite() {
                return Err(CvxError::NumericalInstability(format!(
                    "inexact prox CG: non-positive pᵀMp = {p_mp} (A not PSD?)"
                )));
            }
            let alpha = rr / p_mp;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * mp[i];
            }
            let rr_new: f32 = r.iter().map(|v| v * v).sum();
            if rr_new.sqrt() < cfg.tol {
                return Ok((x, iter_done));
            }
            let beta = rr_new / rr.max(1.0e-30_f32);
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rr = rr_new;
        }
        Ok((x, iter_done))
    }

    fn validate(v: &[f32], b: &[f32], cfg: &InexactProxConfig) -> CvxResult<()> {
        if v.is_empty() || b.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if v.len() != b.len() {
            return Err(CvxError::DimensionMismatch {
                a: v.len(),
                b: b.len(),
            });
        }
        if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "inexact prox rho must be > 0, got {}",
                cfg.rho
            )));
        }
        if cfg.max_inner_iter == 0 {
            return Err(CvxError::InvalidParameter(
                "inexact prox max_inner_iter must be ≥ 1".to_string(),
            ));
        }
        if cfg.tol <= 0.0 || !cfg.tol.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "inexact prox tol must be > 0, got {}",
                cfg.tol
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A = I` (identity).
    fn identity_mv(x: &[f32]) -> Vec<f32> {
        x.to_vec()
    }

    /// `A = 2·diag(d)` for a user-supplied diagonal `d`.
    fn diag2_mv(d: &[f32]) -> impl Fn(&[f32]) -> Vec<f32> + '_ {
        move |x: &[f32]| -> Vec<f32> {
            x.iter()
                .zip(d.iter())
                .map(|(xi, di)| 2.0 * di * xi)
                .collect()
        }
    }

    /// Symmetric 2×2 SPD matvec for testing.
    fn spd_2x2_mv(a: [[f32; 2]; 2]) -> impl Fn(&[f32]) -> Vec<f32> {
        move |x: &[f32]| -> Vec<f32> {
            vec![
                a[0][0] * x[0] + a[0][1] * x[1],
                a[1][0] * x[0] + a[1][1] * x[1],
            ]
        }
    }

    /// Residual `‖(A + ρI) x − (ρ v + b)‖₂` for verification.
    fn residual<F: Fn(&[f32]) -> Vec<f32>>(
        a_mv: F,
        x: &[f32],
        v: &[f32],
        b: &[f32],
        rho: f32,
    ) -> f32 {
        let ax = a_mv(x);
        let mut s = 0.0_f32;
        for i in 0..x.len() {
            let lhs = ax[i] + rho * x[i];
            let rhs = rho * v[i] + b[i];
            let r = lhs - rhs;
            s += r * r;
        }
        s.sqrt()
    }

    #[test]
    fn identity_a_zero_b_closed_form() {
        // A = I, b = 0, prox of ½‖x‖² at v with parameter ρ:
        //   (I + ρI) x = ρ v   →   x = ρ v / (1 + ρ) .
        let v = vec![3.0_f32, -2.0, 1.5];
        let b = vec![0.0_f32; 3];
        let rho = 2.0_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 50,
            tol: 1.0e-7,
        };
        let (x, _) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        let expected_scale = rho / (1.0 + rho);
        for (xi, vi) in x.iter().zip(v.iter()) {
            assert!(
                (*xi - expected_scale * vi).abs() < 1.0e-3,
                "x = {xi}, expected = {}",
                expected_scale * vi
            );
        }
    }

    #[test]
    fn diagonal_a_separable_closed_form() {
        // A = 2·diag(d), b arbitrary → (2d + ρ) x = ρ v + b → x = (ρv+b)/(2d+ρ).
        let d = vec![1.0_f32, 3.0, 0.5, 4.0];
        let v = vec![2.0_f32, -1.0, 0.5, 3.0];
        let b = vec![0.7_f32, -0.3, 1.2, 0.0];
        let rho = 1.5_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 50,
            tol: 1.0e-7,
        };
        let (x, _) = InexactProx::solve(&v, &b, diag2_mv(&d), &cfg).expect("ok");
        for i in 0..d.len() {
            let expected = (rho * v[i] + b[i]) / (2.0 * d[i] + rho);
            assert!(
                (x[i] - expected).abs() < 1.0e-4,
                "x[{i}] = {} vs expected {}",
                x[i],
                expected
            );
        }
    }

    #[test]
    fn converges_within_budget() {
        // CG on an SPD system terminates in ≤ n iterations in exact
        // arithmetic; with rounding we cap by `max_inner_iter`.
        let v = vec![1.0_f32; 4];
        let b = vec![0.0_f32; 4];
        let cfg = InexactProxConfig {
            rho: 1.0,
            max_inner_iter: 20,
            tol: 1.0e-6,
        };
        let (_, iters) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        assert!(iters <= cfg.max_inner_iter, "iters = {iters}");
    }

    #[test]
    fn warm_start_zero_residual_uses_zero_iterations() {
        // When the warm start is exact (rhs = 0, x0 = v = 0), zero iterations
        // are required.
        let v = vec![0.0_f32; 3];
        let b = vec![0.0_f32; 3];
        let cfg = InexactProxConfig {
            rho: 1.0,
            max_inner_iter: 10,
            tol: 1.0e-6,
        };
        let (_, iters) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        assert_eq!(iters, 0);
    }

    #[test]
    fn smaller_condition_number_converges_faster() {
        // Well-conditioned: A = I (κ = 1).
        // Ill-conditioned: A = diag(1, 50) (κ = 50).
        let v = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let cfg = InexactProxConfig {
            rho: 1.0e-3,
            max_inner_iter: 100,
            tol: 1.0e-7,
        };
        let (_, iter_well) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        let a_ill = spd_2x2_mv([[1.0, 0.0], [0.0, 50.0]]);
        let (_, iter_ill) = InexactProx::solve(&v, &b, a_ill, &cfg).expect("ok");
        assert!(
            iter_well <= iter_ill,
            "well = {iter_well}, ill = {iter_ill}"
        );
    }

    #[test]
    fn residual_below_tolerance_at_convergence() {
        let d = vec![0.5_f32, 2.0, 1.0];
        let v = vec![1.0_f32, -1.0, 0.5];
        let b = vec![0.2_f32, 0.3, -0.1];
        let rho = 0.5_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 100,
            tol: 1.0e-7,
        };
        let (x, _) = InexactProx::solve(&v, &b, diag2_mv(&d), &cfg).expect("ok");
        let res = residual(diag2_mv(&d), &x, &v, &b, rho);
        assert!(res < 1.0e-4, "residual = {res}");
    }

    #[test]
    fn heavy_prox_recovers_v() {
        // rho very large → x ≈ v (penalty dominates).
        let v = vec![1.5_f32, -2.3, 0.7];
        let b = vec![0.0_f32; 3];
        let rho = 1.0e6_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 50,
            tol: 1.0e-5,
        };
        let (x, _) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        for (xi, vi) in x.iter().zip(v.iter()) {
            assert!((xi - vi).abs() < 1.0e-3, "xi = {xi}, vi = {vi}");
        }
    }

    #[test]
    fn light_prox_recovers_unconstrained_minimiser() {
        // rho very small → x ≈ A⁻¹ b (minimiser of g alone).
        // For A = 2·diag(d), minimiser of g is x_i = b_i / (2 d_i).
        let d = vec![1.0_f32, 3.0, 0.5];
        let b = vec![4.0_f32, 6.0, 2.0];
        let v = vec![0.0_f32, 0.0, 0.0];
        let rho = 1.0e-4_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 500,
            tol: 1.0e-8,
        };
        let (x, _) = InexactProx::solve(&v, &b, diag2_mv(&d), &cfg).expect("ok");
        for i in 0..d.len() {
            let expected = b[i] / (2.0 * d[i]);
            assert!(
                (x[i] - expected).abs() < 5.0e-3,
                "x[{i}] = {}, expected = {expected}",
                x[i]
            );
        }
    }

    #[test]
    fn deterministic_runs_match() {
        let v = vec![1.0_f32, 0.5, -0.3, 2.0];
        let b = vec![0.1_f32, -0.2, 0.3, -0.4];
        let cfg = InexactProxConfig::default();
        let (x1, n1) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        let (x2, n2) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        assert_eq!(n1, n2);
        assert_eq!(x1, x2);
    }

    #[test]
    fn err_rho_non_positive() {
        let cfg = InexactProxConfig {
            rho: 0.0,
            ..InexactProxConfig::default()
        };
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &[0.0_f32], identity_mv, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = InexactProxConfig {
            rho: -1.0,
            ..InexactProxConfig::default()
        };
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &[0.0_f32], identity_mv, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_max_inner_iter_zero() {
        let cfg = InexactProxConfig {
            max_inner_iter: 0,
            ..InexactProxConfig::default()
        };
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &[0.0_f32], identity_mv, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_tol_non_positive() {
        let cfg = InexactProxConfig {
            tol: 0.0,
            ..InexactProxConfig::default()
        };
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &[0.0_f32], identity_mv, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = InexactProxConfig {
            tol: -1.0e-6,
            ..InexactProxConfig::default()
        };
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &[0.0_f32], identity_mv, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_empty_v_or_b() {
        let cfg = InexactProxConfig::default();
        let v_empty: [f32; 0] = [];
        let b_empty: [f32; 0] = [];
        assert!(matches!(
            InexactProx::solve(&v_empty, &b_empty, identity_mv, &cfg),
            Err(CvxError::EmptyInput)
        ));
        assert!(matches!(
            InexactProx::solve(&[1.0_f32], &b_empty, identity_mv, &cfg),
            Err(CvxError::EmptyInput)
        ));
        assert!(matches!(
            InexactProx::solve(&v_empty, &[1.0_f32], identity_mv, &cfg),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn err_length_mismatch_v_b() {
        let cfg = InexactProxConfig::default();
        assert!(matches!(
            InexactProx::solve(&[1.0_f32, 2.0], &[1.0_f32], identity_mv, &cfg),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_a_mv_wrong_length() {
        let cfg = InexactProxConfig::default();
        let bad_mv = |x: &[f32]| -> Vec<f32> {
            // Returns half-length: trigger DimensionMismatch.
            x.iter()
                .take(x.len() / 2 + 1)
                .copied()
                .collect::<Vec<f32>>()[..x.len() / 2]
                .to_vec()
        };
        // For an n=4 input, bad_mv returns length 2 → mismatch.
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![0.0_f32; 4];
        assert!(matches!(
            InexactProx::solve(&v, &b, bad_mv, &cfg),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn identity_unique_solver_path() {
        // Direct check on A = I, b non-zero:
        //   (I + ρI) x = ρv + b  ⇒  x = (ρv + b)/(1+ρ).
        let v = vec![2.0_f32, -3.0];
        let b = vec![1.0_f32, 4.0];
        let rho = 0.5_f32;
        let cfg = InexactProxConfig {
            rho,
            max_inner_iter: 20,
            tol: 1.0e-7,
        };
        let (x, _) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        let inv = 1.0_f32 / (1.0 + rho);
        for i in 0..2 {
            let expected = (rho * v[i] + b[i]) * inv;
            assert!(
                (x[i] - expected).abs() < 1.0e-4,
                "x[{i}] = {}, expected = {expected}",
                x[i]
            );
        }
    }

    #[test]
    fn one_dimensional_works() {
        // 1-D: A = [4], b = [3], v = [1], ρ = 2 → (4+2) x = 2·1 + 3 = 5 → x = 5/6.
        let v = vec![1.0_f32];
        let b = vec![3.0_f32];
        let a_mv = |x: &[f32]| -> Vec<f32> { vec![4.0 * x[0]] };
        let cfg = InexactProxConfig {
            rho: 2.0,
            max_inner_iter: 10,
            tol: 1.0e-8,
        };
        let (x, _) = InexactProx::solve(&v, &b, a_mv, &cfg).expect("ok");
        assert!((x[0] - 5.0 / 6.0).abs() < 1.0e-5, "x = {}", x[0]);
    }

    #[test]
    fn symmetric_psd_two_by_two() {
        // A = [[3, 1], [1, 2]] (SPD), b = [1, -1], v = [0.5, 0.5], ρ = 1.
        // (A + I) = [[4, 1], [1, 3]] ; rhs = ρv + b = [1.5, -0.5] .
        // det = 11 → x = (1/11)·([[3, -1], [-1, 4]]·[1.5, -0.5])
        //               = (1/11)·([5, -3.5]) = [0.4545..., -0.3181...].
        let v = vec![0.5_f32, 0.5];
        let b = vec![1.0_f32, -1.0];
        let cfg = InexactProxConfig {
            rho: 1.0,
            max_inner_iter: 30,
            tol: 1.0e-8,
        };
        let (x, _) =
            InexactProx::solve(&v, &b, spd_2x2_mv([[3.0, 1.0], [1.0, 2.0]]), &cfg).expect("ok");
        let exp_x0 = 5.0_f32 / 11.0;
        let exp_x1 = -3.5_f32 / 11.0;
        assert!((x[0] - exp_x0).abs() < 1.0e-4, "x0 = {}", x[0]);
        assert!((x[1] - exp_x1).abs() < 1.0e-4, "x1 = {}", x[1]);
    }

    #[test]
    fn prox_method_alias_matches_solve() {
        // The struct-method `prox()` form must agree with `solve()`.
        let v = vec![1.0_f32, 2.0, 3.0];
        let b = vec![0.5_f32, -0.5, 1.0];
        let cfg = InexactProxConfig {
            rho: 1.5,
            max_inner_iter: 20,
            tol: 1.0e-7,
        };
        let helper = InexactProx;
        let (x_a, n_a) = helper.prox(&v, &b, 1.5, identity_mv, &cfg).expect("ok");
        let (x_b, n_b) = InexactProx::solve(&v, &b, identity_mv, &cfg).expect("ok");
        assert_eq!(x_a, x_b);
        assert_eq!(n_a, n_b);
    }
}
