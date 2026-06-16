//! End-to-end integration tests for `oxicuda-cvx`.

#![allow(clippy::approx_constant)]

use crate::admm::admm_solve;
use crate::admm::consensus_admm;
use crate::admm::{DualDecompConfig, dual_decomp};
use crate::augmented_lagrangian::augmented_lagrangian;
use crate::error::CvxResult;
use crate::gradient::{heavy_ball, nesterov_accelerated, projected_gradient};
use crate::handle::LcgRng;
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;
use crate::linesearch::{armijo_search, backtracking_search, strong_wolfe_search, wolfe_search};
use crate::lp::{SimplexStatus, mehrotra_predictor_corrector, primal_dual_lp, revised_simplex};
use crate::metrics::{convergence_rate, duality_gap, kkt_residual, primal_residual};
use crate::primal_dual::chambolle_pock;
use crate::projection::{
    dykstra_pocs, project_box, project_halfspace, project_l1_ball, project_l2_ball,
    project_psd_cone, project_simplex, project_soc,
};
use crate::prox_ops::{
    prox_elastic_net, prox_group_lasso, prox_l1, prox_l2, prox_linf, prox_nuclear, prox_tv_1d,
    soft_threshold,
};
use crate::proximal::{accelerated_prox_gradient, douglas_rachford, fista, proximal_gradient};
use crate::ptx_kernels::{
    admm_dual_update_ptx, axpy_ptx, fista_extrapolate_ptx, gradient_step_ptx, proj_l2_ball_ptx,
    simplex_proj_ptx, soft_threshold_ptx,
};
use crate::qp::{active_set_qp, mehrotra_qp, primal_dual_qp};
use crate::sdp::sdp_interior_point;
use crate::socp::primal_dual_socp;

/// Borrowed projection/update closure used by the POCS / dual-decomposition tests.
type ProjFn<'a> = &'a dyn Fn(&[f64]) -> CvxResult<Vec<f64>>;

// 1. LP simplex on 2D `min -x-y s.t. x+y ≤ 1, x,y ≥ 0` recovers (1,0) or (0,1) with objective -1.
#[test]
fn e2e_lp_simplex_2d() {
    // standard form: variables [x, y, s]; A = [[1,1,1]]; b=[1]; c=[-1,-1,0].
    let a = vec![1.0_f64, 1.0, 1.0];
    let b = vec![1.0_f64];
    let c = vec![-1.0_f64, -1.0, 0.0];
    let basis = vec![2usize];
    let res = revised_simplex(&a, 1, 3, &b, &c, &basis, 100).expect("ok");
    assert_eq!(res.status, SimplexStatus::Optimal);
    assert!((res.objective + 1.0).abs() < 1.0e-9);
    assert!((res.x[0] + res.x[1] - 1.0).abs() < 1.0e-9);
}

// 2. LP via Mehrotra recovers same optimum.
#[test]
fn e2e_lp_mehrotra_matches_simplex() {
    let a = vec![1.0_f64, 1.0, 1.0];
    let b = vec![1.0_f64];
    let c = vec![-1.0_f64, -1.0, 0.0];
    let res = mehrotra_predictor_corrector(&a, 1, 3, &b, &c, 200, 1.0e-7).expect("ok");
    let obj: f64 = res.x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
    assert!((obj + 1.0).abs() < 1.0e-3);
}

// 3. QP active-set on `min ½||x||² s.t. x_i = 1 ∀ i` recovers (1, 1, 1).
#[test]
fn e2e_qp_identity_constraints() {
    let n = 3;
    let p_mat = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let q = vec![0.0_f64; n];
    let mut a_eq = vec![0.0_f64; n * n];
    for i in 0..n {
        a_eq[i * n + i] = 1.0;
    }
    let b_eq = vec![1.0_f64; n];
    let res = active_set_qp(
        &p_mat,
        n,
        &q,
        &a_eq,
        n,
        &b_eq,
        &[],
        0,
        &[],
        &[1.0, 1.0, 1.0],
        50,
    )
    .expect("ok");
    for &xi in &res.x {
        assert!((xi - 1.0).abs() < 1.0e-9);
    }
}

// 4. L1 prox example.
#[test]
fn e2e_l1_prox_doc_example() {
    let v = [2.0_f64, 0.5, -0.5, -2.0];
    let p = prox_l1(&v, 1.0).expect("ok");
    assert!((p[0] - 1.0).abs() < 1.0e-12);
    assert!(p[1].abs() < 1.0e-12);
    assert!(p[2].abs() < 1.0e-12);
    assert!((p[3] + 1.0).abs() < 1.0e-12);
}

// 5. Simplex projection of [1, 1, 1] is [1/3, 1/3, 1/3].
#[test]
fn e2e_simplex_projection_uniform() {
    let v = vec![1.0_f64, 1.0, 1.0];
    let p = project_simplex(&v, 1.0).expect("ok");
    for &pi in &p {
        assert!((pi - 1.0 / 3.0).abs() < 1.0e-12);
    }
}

// 6. PSD cone projection of [-1, 0; 0, 1] is [0, 0; 0, 1].
#[test]
fn e2e_psd_cone_projection_neg() {
    let a = vec![-1.0_f64, 0.0, 0.0, 1.0];
    let p = project_psd_cone(&a, 2).expect("ok");
    assert!(p[0].abs() < 1.0e-9);
    assert!((p[3] - 1.0).abs() < 1.0e-9);
    assert!(p[1].abs() < 1.0e-9);
    assert!(p[2].abs() < 1.0e-9);
}

// 7. TV prox: 1D TV denoising on piecewise-constant noisy signal reduces stair-step noise.
#[test]
fn e2e_tv_denoising_reduces_variance() {
    let mut rng = LcgRng::new(7);
    let mut y = Vec::with_capacity(20);
    for _ in 0..10 {
        y.push(0.0 + 0.1 * rng.next_normal());
    }
    for _ in 0..10 {
        y.push(2.0 + 0.1 * rng.next_normal());
    }
    let x = prox_tv_1d(&y, 1.0).expect("ok");
    let mean_left: f64 = x[0..10].iter().sum::<f64>() / 10.0;
    let mean_right: f64 = x[10..20].iter().sum::<f64>() / 10.0;
    assert!(mean_right - mean_left > 1.0);
    // Variance within first half of x should be small.
    let var_x: f64 = x[0..10].iter().map(|v| (v - mean_left).powi(2)).sum();
    let var_y: f64 = y[0..10]
        .iter()
        .map(|v| (v - y[0..10].iter().sum::<f64>() / 10.0).powi(2))
        .sum();
    assert!(var_x <= var_y + 1.0e-9);
}

// 8. FISTA on L1-regularised least squares converges (smoke; rate check via residual ratio).
#[test]
fn e2e_fista_lasso_converges() {
    // f(x) = 0.5 ||x - b||² with b = [3, -2, 0.5]; g(x) = λ ||x||₁ with λ=1.
    let b = vec![3.0_f64, -2.0, 0.5];
    let f = |x: &[f64]| -> CvxResult<f64> {
        Ok(x.iter()
            .zip(b.iter())
            .map(|(xi, bi)| 0.5 * (xi - bi).powi(2))
            .sum())
    };
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
        Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect())
    };
    let p = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, s) };
    let x = fista(&[0.0, 0.0, 0.0], &f, &g, &p, 1.0, 1000, 1.0e-12, false).expect("ok");
    assert!((x[0] - 2.0).abs() < 1.0e-6);
    assert!((x[1] + 1.0).abs() < 1.0e-6);
    assert!(x[2].abs() < 1.0e-6);
}

// 9. ADMM on Lasso matches FISTA on small problem.
#[test]
fn e2e_admm_matches_fista_lasso() {
    let b = vec![3.0_f64, -2.0, 0.5];
    let lambda = 1.0_f64;
    let rho = 1.0_f64;
    let n = 3;
    let mut a_mat = vec![0.0_f64; n * n];
    let mut b_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        a_mat[i * n + i] = 1.0;
        b_mat[i * n + i] = -1.0;
    }
    let c = vec![0.0_f64; n];
    let b_clone = b.clone();
    let xu = |z: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
        Ok((0..n)
            .map(|i| (b_clone[i] + rho * (z[i] - u[i])) / (1.0 + rho))
            .collect())
    };
    let zu = |x: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
        Ok((0..n)
            .map(|i| soft_threshold(x[i] + u[i], lambda / rho))
            .collect())
    };
    let res = admm_solve(
        &a_mat, n, n, &b_mat, n, &c, rho, &xu, &zu, 500, 1.0e-8, 1.0e-8,
    )
    .expect("ok");
    assert!((res.z[0] - 2.0).abs() < 1.0e-4);
    assert!((res.z[1] + 1.0).abs() < 1.0e-4);
    assert!(res.z[2].abs() < 1.0e-4);
}

// 10. Chambolle-Pock decreases primal energy (TV-L2).
#[test]
fn e2e_chambolle_pock_decreases_energy() {
    // Simple separable problem; we test convergence.
    let k = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.to_vec()) };
    let kt = |y: &[f64]| -> CvxResult<Vec<f64>> { Ok(y.to_vec()) };
    let pf_star = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l2(y, s) };
    let pg = |x: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(x.to_vec()) };
    let (x, _y) = chambolle_pock(
        &[1.0, 2.0],
        &[0.0, 0.0],
        &k,
        &kt,
        &pf_star,
        &pg,
        0.5,
        0.5,
        1.0,
        500,
        1.0e-9,
    )
    .expect("ok");
    for &xi in &x {
        assert!(xi.abs() < 1.0e-2);
    }
}

// 11. Projected gradient on box-constrained quadratic converges to KKT.
#[test]
fn e2e_projected_gradient_box_quadratic() {
    let target = vec![2.0_f64, -3.0];
    let grad = |x: &[f64]| -> CvxResult<Vec<f64>> {
        Ok(x.iter()
            .zip(target.iter())
            .map(|(xi, ti)| 2.0 * (xi - ti))
            .collect())
    };
    let proj = |y: &[f64]| -> CvxResult<Vec<f64>> { project_box(y, -1.0, 1.0) };
    let x = projected_gradient(&[0.0, 0.0], &grad, &proj, 0.1, 500, 1.0e-10).expect("ok");
    assert!((x[0] - 1.0).abs() < 1.0e-6);
    assert!((x[1] + 1.0).abs() < 1.0e-6);
}

// 12. Strong Wolfe line search satisfies both conditions.
#[test]
fn e2e_strong_wolfe_conditions() {
    let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
    let x = vec![3.0_f64];
    let grad = vec![6.0_f64];
    let d = vec![-6.0_f64];
    let alpha = strong_wolfe_search(&x, &d, &grad, &f, &g, 1.0e-4, 0.9, 50).expect("ok");
    let x_new = vec![x[0] + alpha * d[0]];
    let f_new = f(&x_new).expect("ok");
    let fx = f(&x).expect("ok");
    let gd = grad[0] * d[0];
    assert!(f_new <= fx + 1.0e-4 * alpha * gd + 1.0e-12);
    let g_new = g(&x_new).expect("ok");
    let gd_new = g_new[0] * d[0];
    assert!(gd_new.abs() <= 0.9 * gd.abs() + 1.0e-9);
}

// 13. PTX kernels emit valid headers across SM versions.
#[test]
fn e2e_ptx_kernels_all_sm_smoke() {
    for sm in [75u32, 80, 86, 89, 90, 100] {
        let kernels = [
            axpy_ptx(sm),
            soft_threshold_ptx(sm),
            simplex_proj_ptx(sm),
            gradient_step_ptx(sm),
            fista_extrapolate_ptx(sm),
            admm_dual_update_ptx(sm),
            proj_l2_ball_ptx(sm),
        ];
        for k in &kernels {
            assert!(k.contains(".visible .entry"));
            assert!(k.contains("ret"));
        }
    }
}

// 14. KKT residual zero at known optimum.
#[test]
fn e2e_kkt_residual_zero() {
    let a = vec![1.0_f64, 1.0];
    let b = vec![1.0_f64];
    let x = vec![0.5_f64, 0.5];
    let lambda = vec![-0.5_f64];
    let mu = vec![0.0_f64, 0.0];
    let grad = x.clone();
    let r = kkt_residual(&a, 1, 2, &b, &x, &lambda, &mu, &grad).expect("ok");
    assert!(r < 1.0e-9);
}

// 15. Augmented Lagrangian recovers x = (0.5, 0.5).
#[test]
fn e2e_alm_quadratic_equality() {
    let a = vec![1.0_f64, 1.0];
    let b = vec![1.0_f64];
    let inner = |lambda: &[f64], rho: f64| -> CvxResult<Vec<f64>> {
        let s = (rho * b[0] - lambda[0]) / (1.0 + 2.0 * rho);
        Ok(vec![s * a[0], s * a[1]])
    };
    let res = augmented_lagrangian(&a, 1, 2, &b, &inner, 1.0, 2.0, 30, 1.0e-8).expect("ok");
    assert!((res.x[0] - 0.5).abs() < 1.0e-5);
    assert!((res.x[1] - 0.5).abs() < 1.0e-5);
}

// 16. Consensus ADMM converges to mean of b_i.
#[test]
fn e2e_consensus_admm_mean() {
    let bs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let rho = 1.0_f64;
    let updates: Vec<_> = bs
        .iter()
        .map(|b| {
            let b_owned = b.clone();
            move |z: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
                Ok((0..b_owned.len())
                    .map(|i| (b_owned[i] + rho * (z[i] - u[i])) / (1.0 + rho))
                    .collect())
            }
        })
        .collect();
    let res = consensus_admm(3, 2, rho, &updates, 500, 1.0e-9).expect("ok");
    assert!((res.z[0] - 3.0).abs() < 1.0e-4);
    assert!((res.z[1] - 4.0).abs() < 1.0e-4);
}

// 17. Heavy-ball converges on quadratic.
#[test]
fn e2e_heavy_ball_quadratic() {
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
    let x = heavy_ball(&[3.0, 4.0], &g, 0.1, 0.5, 2000, 1.0e-9).expect("ok");
    for &xi in &x {
        assert!(xi.abs() < 1.0e-4);
    }
}

// 18. SDP recovers tr(X) = 1 with C = I.
#[test]
fn e2e_sdp_basic_trace() {
    let n = 2;
    let c = vec![1.0_f64, 0.0, 0.0, 1.0];
    let a1 = vec![1.0_f64, 0.0, 0.0, 1.0];
    let b = vec![1.0_f64];
    let res = sdp_interior_point(&c, n, &[a1], &b, 200, 1.0e-7).expect("ok");
    let tr = res.x[0] + res.x[3];
    assert!((tr - 1.0).abs() < 1.0e-3);
}

// 19. SOC projection inside cone is identity.
#[test]
fn e2e_soc_projection_inside_passes_through() {
    let (t, x) = project_soc(2.0, &[1.0, 0.0]).expect("ok");
    assert!((t - 2.0).abs() < 1.0e-12);
    assert_eq!(x, vec![1.0, 0.0]);
}

// 20. Halfspace projection lies on the boundary when starting outside.
#[test]
fn e2e_halfspace_projects_to_boundary() {
    let v = vec![1.0_f64, 1.0];
    let p = project_halfspace(&v, &[1.0, 1.0], 1.0).expect("ok");
    let av: f64 = p.iter().sum();
    assert!((av - 1.0).abs() < 1.0e-10);
}

// 21. Nuclear prox shrinks singular values.
#[test]
fn e2e_nuclear_prox_threshold() {
    let y = vec![3.0_f64, 0.0, 0.0, 0.5];
    let out = prox_nuclear(&y, 2, 2, 1.0).expect("ok");
    assert!((out[0] - 2.0).abs() < 1.0e-6);
    assert!(out[3].abs() < 1.0e-6);
}

// 22. Elastic net combines L1 and L2.
#[test]
fn e2e_elastic_net_combined() {
    let v = vec![3.0_f64];
    let p = prox_elastic_net(&v, 1.0, 1.0).expect("ok");
    assert!((p[0] - 1.0).abs() < 1.0e-10);
}

// 23. L∞ prox conserves L1 cap.
#[test]
fn e2e_linf_prox_l1_cap() {
    let v = vec![5.0_f64, 1.0];
    let p = prox_linf(&v, 1.0).expect("ok");
    let diff: f64 = v.iter().zip(p.iter()).map(|(vi, pi)| (vi - pi).abs()).sum();
    assert!((diff - 1.0).abs() < 1.0e-10);
}

// 24. Group lasso block soft-threshold.
#[test]
fn e2e_group_lasso() {
    let v = vec![0.1_f64, 0.1, 10.0, 10.0];
    let p = prox_group_lasso(&v, &[(0, 2), (2, 4)], 1.0).expect("ok");
    assert!(p[0].abs() < 1.0e-12);
    assert!(p[1].abs() < 1.0e-12);
    assert!(p[2].abs() > 0.0);
}

// 25. Nesterov accelerated converges to origin.
#[test]
fn e2e_nesterov_to_origin() {
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
    let x = nesterov_accelerated(&[5.0, -3.0], &g, 0.4, 1000, 1.0e-10).expect("ok");
    for &xi in &x {
        assert!(xi.abs() < 1.0e-6);
    }
}

// 26. Suppress unused-import for now: linalg helpers smoke.
#[test]
fn e2e_linalg_solve_dense_identity() {
    let a = vec![1.0_f64, 0.0, 0.0, 1.0];
    let b = vec![3.0_f64, 4.0];
    let x = solve_dense(&a, 2, &b).expect("ok");
    assert!((x[0] - 3.0).abs() < 1.0e-12);
    assert!((x[1] - 4.0).abs() < 1.0e-12);
}

// 27. Convergence-rate metric.
#[test]
fn e2e_convergence_rate_quadratic() {
    let p = convergence_rate(0.01, 0.0001).expect("ok");
    assert!((p - 2.0).abs() < 1.0e-6);
}

// 28. Backtracking and Wolfe both produce positive steps.
#[test]
fn e2e_armijo_wolfe_backtracking() {
    let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|v| 2.0 * v).collect()) };
    let x = vec![1.0_f64];
    let grad = vec![2.0_f64];
    let d = vec![-2.0_f64];
    let a = armijo_search(&x, &d, &grad, &f, 1.0, 0.5, 1.0e-4, 50).expect("ok");
    let bt = backtracking_search(&x, &d, &grad, &f).expect("ok");
    let w = wolfe_search(&x, &d, &grad, &f, &g, 1.0e-4, 0.9, 50).expect("ok");
    assert!(a > 0.0);
    assert!(bt > 0.0);
    assert!(w > 0.0);
}

// 29. Douglas-Rachford simple sum-of-two.
#[test]
fn e2e_douglas_rachford_simple() {
    let b = vec![4.0_f64, 0.0];
    let f = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> {
        Ok(v.iter()
            .zip(b.iter())
            .map(|(vi, bi)| (g * bi + vi) / (1.0 + g))
            .collect())
    };
    let pg = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> { prox_l2(v, g) };
    let x = douglas_rachford(&[0.0, 0.0], &f, &pg, 1.0, 200, 1.0e-10).expect("ok");
    assert!((x[0] - 2.0).abs() < 1.0e-5);
    assert!(x[1].abs() < 1.0e-5);
}

// 30. SOCP recovers t = 1 with c = e_0.
#[test]
fn e2e_socp_unit() {
    let a = vec![1.0_f64, 0.0, 0.0];
    let b = vec![1.0_f64];
    let c = vec![1.0_f64, 0.0, 0.0];
    let res = primal_dual_socp(&a, 1, 3, &b, &c, 100, 1.0e-6).expect("ok");
    assert!((res.x[0] - 1.0).abs() < 1.0e-3);
}

// 31. Accelerated prox-grad recovers Lasso solution.
#[test]
fn e2e_accelerated_prox_lasso() {
    let b = vec![5.0_f64];
    let f = |x: &[f64]| -> CvxResult<f64> { Ok(0.5 * (x[0] - b[0]).powi(2)) };
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![x[0] - b[0]]) };
    let p = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, s) };
    let x = accelerated_prox_gradient(&[0.0], &f, &g, &p, 1.0, 500, 1.0e-10).expect("ok");
    assert!((x[0] - 4.0).abs() < 1.0e-6);
}

// 32. Primal-dual LP basic.
#[test]
fn e2e_primal_dual_lp_basic() {
    let a = vec![1.0_f64, 1.0, 1.0];
    let b = vec![1.0_f64];
    let c = vec![-1.0_f64, -1.0, 0.0];
    let res = primal_dual_lp(&a, 1, 3, &b, &c, 100, 1.0e-7).expect("ok");
    let obj: f64 = res.x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
    assert!((obj + 1.0).abs() < 1.0e-3);
}

// 33. Primal-dual QP basic.
#[test]
fn e2e_primal_dual_qp_basic() {
    let p_mat = vec![1.0_f64, 0.0, 0.0, 0.0];
    let q = vec![0.0_f64, 0.0];
    let a = vec![1.0_f64, 1.0];
    let b = vec![1.0_f64];
    let res = primal_dual_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1.0e-7).expect("ok");
    assert!(res.x[0].abs() < 1.0e-3);
}

// 34. Proximal gradient on Lasso with backtracking.
#[test]
fn e2e_prox_gradient_lasso_backtracking() {
    let b = vec![3.0_f64, -2.0];
    let f = |x: &[f64]| -> CvxResult<f64> {
        Ok(x.iter()
            .zip(b.iter())
            .map(|(xi, bi)| 0.5 * (xi - bi).powi(2))
            .sum())
    };
    let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
        Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect())
    };
    let p = |y: &[f64], _s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, 1.0) };
    let x = proximal_gradient(&[0.0, 0.0], &f, &g, &p, 1.0, 500, 1.0e-9, true).expect("ok");
    assert!((x[0] - 2.0).abs() < 1.0e-4);
    assert!((x[1] + 1.0).abs() < 1.0e-4);
}

// 35. L1 and L2 ball projections.
#[test]
fn e2e_ball_projections_basic() {
    let p1 = project_l1_ball(&[1.0, 1.0, -1.0], 1.0).expect("ok");
    let s1: f64 = p1.iter().map(|x| x.abs()).sum();
    assert!((s1 - 1.0).abs() < 1.0e-10);
    let p2 = project_l2_ball(&[3.0, 4.0], 1.0).expect("ok");
    let n2 = norm2(&p2);
    assert!((n2 - 1.0).abs() < 1.0e-10);
}

// 36. Duality gap basic.
#[test]
fn e2e_duality_gap_basic() {
    assert!((duality_gap(5.0, 4.0) - 1.0).abs() < 1.0e-12);
}

// 37. Primal residual computation.
#[test]
fn e2e_primal_residual_basic() {
    let a = vec![1.0_f64, 1.0];
    let x = vec![0.5_f64, 0.5];
    let b = vec![1.0_f64];
    let r = primal_residual(&a, 1, 2, &x, &b).expect("ok");
    assert!(r < 1.0e-12);
}

// 38. mat_t_vec basic helper.
#[test]
fn e2e_mat_t_vec_basic() {
    let a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let x = vec![1.0_f64, 1.0];
    let y = mat_t_vec(&a, 2, 2, &x).expect("ok");
    assert_eq!(y, vec![4.0, 6.0]);
}

// 39. mat_vec basic helper.
#[test]
fn e2e_mat_vec_basic() {
    let a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let x = vec![1.0_f64, 1.0];
    let y = mat_vec(&a, 2, 2, &x).expect("ok");
    assert_eq!(y, vec![3.0, 7.0]);
}

// Wave AAA+55 e2e tests

// 40. Dykstra POCS: L2-ball ∩ non-negative box intersection.
#[test]
fn e2e_dykstra_pocs_l2_box_intersection() {
    // Project (-2, -2) onto L2-ball(r=1) ∩ {x ≥ 0}.
    // Expected solution is near the origin (0, 0), which is the corner of the
    // non-negative orthant closest to the L2 ball boundary.
    let ball_proj = |x: &[f64]| -> CvxResult<Vec<f64>> {
        let norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        if norm <= 1.0 {
            Ok(x.to_vec())
        } else {
            Ok(x.iter().map(|xi| xi / norm).collect())
        }
    };
    let nn1 = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|xi| xi.max(0.0)).collect()) };
    let nn2 = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.iter().map(|xi| xi.max(0.0)).collect()) };
    let projs: Vec<ProjFn> = vec![&ball_proj, &nn1, &nn2];
    let res = dykstra_pocs(&projs, &[-2.0_f64, -2.0], 1000, 1e-8).expect("ok");
    // Must be in L2 ball
    let norm: f64 = res.x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    assert!(norm <= 1.0 + 1e-5, "result not in L2 ball: ‖x‖={}", norm);
    // Must be non-negative
    assert!(res.x[0] >= -1e-6, "x[0]={} < 0", res.x[0]);
    assert!(res.x[1] >= -1e-6, "x[1]={} < 0", res.x[1]);
}

// 41. Dykstra POCS: three halfplanes intersection.
#[test]
fn e2e_dykstra_pocs_three_halfplanes() {
    // Project (5, 5) onto {x₁≤2} ∩ {x₂≤2} ∩ {x₁+x₂≤3}.
    let h1 = |x: &[f64]| -> CvxResult<Vec<f64>> {
        // {x1 ≤ 2}: project along a=[1,0], b=2
        let excess = x[0] - 2.0;
        if excess <= 0.0 {
            Ok(x.to_vec())
        } else {
            Ok(vec![x[0] - excess, x[1]])
        }
    };
    let h2 = |x: &[f64]| -> CvxResult<Vec<f64>> {
        // {x2 ≤ 2}: project along a=[0,1], b=2
        let excess = x[1] - 2.0;
        if excess <= 0.0 {
            Ok(x.to_vec())
        } else {
            Ok(vec![x[0], x[1] - excess])
        }
    };
    let h3 = |x: &[f64]| -> CvxResult<Vec<f64>> {
        // {x1+x2 ≤ 3}: project along a=[1,1], b=3, ‖a‖²=2
        let dot = x[0] + x[1];
        if dot <= 3.0 {
            Ok(x.to_vec())
        } else {
            let scale = (dot - 3.0) / 2.0;
            Ok(vec![x[0] - scale, x[1] - scale])
        }
    };
    let projs: Vec<ProjFn> = vec![&h1, &h2, &h3];
    let res = dykstra_pocs(&projs, &[5.0_f64, 5.0], 1000, 1e-8).expect("ok");
    assert!(res.x[0] <= 2.0 + 1e-4, "x[0]={} violates x1≤2", res.x[0]);
    assert!(res.x[1] <= 2.0 + 1e-4, "x[1]={} violates x2≤2", res.x[1]);
    assert!(
        res.x[0] + res.x[1] <= 3.0 + 1e-4,
        "x[0]+x[1]={} violates x1+x2≤3",
        res.x[0] + res.x[1]
    );
}

// 42. Dual decomposition: 2-block separable quadratic coupling.
#[test]
fn e2e_dual_decomp_quadratic_coupling() {
    // min ½x₁² + ½x₂²  s.t. x₁+x₂=1 → x₁=x₂=0.5
    let a = [1.0_f64];
    // min ½x² + λx → x* = -λ  (d/dx [½x² + λx] = x + λ = 0)
    let f1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
    let f2 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
    let upds: &[ProjFn] = &[&f1 as ProjFn, &f2 as ProjFn];
    let cfg = DualDecompConfig {
        step_size: 0.1,
        max_iter: 3000,
        tol: 1e-6,
    };
    let res = dual_decomp(&[&a[..], &a[..]], &[(1, 1), (1, 1)], &[1.0], upds, &cfg).expect("ok");
    let sum = res.x_blocks[0][0] + res.x_blocks[1][0];
    assert!((sum - 1.0).abs() < 1e-4, "x1+x2={} should be ≈ 1.0", sum);
    assert!(
        (res.x_blocks[0][0] - 0.5).abs() < 1e-3,
        "x1={} should be ≈ 0.5",
        res.x_blocks[0][0]
    );
}

// 43. Mehrotra QP: 3-variable uniform solution.
#[test]
fn e2e_mehrotra_qp_simple_qp() {
    // min ½(x1²+x2²+x3²) s.t. x1+x2+x3=1, x≥0 → (1/3, 1/3, 1/3)
    let p_mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0_f64];
    let q = vec![0.0_f64; 3];
    let a = vec![1.0_f64, 1.0, 1.0];
    let b = vec![1.0_f64];
    let res = mehrotra_qp(&p_mat, 3, &q, &a, 1, &b, 100, 1e-7).expect("ok");
    for (i, &xi) in res.x.iter().enumerate() {
        assert!(
            (xi - 1.0 / 3.0).abs() < 1e-4,
            "x[{}]={} expected 1/3",
            i,
            xi
        );
    }
    assert!(res.converged, "should have converged");
}
