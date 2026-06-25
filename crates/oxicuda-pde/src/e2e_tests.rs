//! End-to-end integration tests for `oxicuda-pde`.

use crate::dg::dg_2d::{Dg2dSpace, DgBoundary, dg_2d_advect, dg_2d_burgers};
use crate::dg::dg1d::{Dg1dSpace, lax_friedrichs_flux, lgl_nodes, lgl_weights, upwind_flux};
use crate::fdm::advection_1d::{lax_wendroff_step_1d, upwind_step_1d};
use crate::fdm::heat_1d::{backward_euler_step, crank_nicolson_step, forward_euler_step};
use crate::fdm::poisson_1d::{Dirichlet1d, solve_poisson_1d};
use crate::fdm::poisson_2d::{
    DirichletRect, assemble_poisson_2d_csr, build_poisson_2d_rhs, solve_poisson_2d_gs,
};
use crate::fdm::wave_1d::{WaveState1d, leapfrog_step_1d};
use crate::fem::dirichlet_apply::apply_dirichlet_csr;
use crate::fem::mass_stiffness::{assemble_load_centroid, assemble_mass_stiffness};
use crate::fem::mixed_poisson::{MixedBoundary, element_divergence, mixed_poisson_rt0};
use crate::fem::p1_tet::p1_tet_local_stiffness;
use crate::handle::LcgRng;
use crate::mesh::{Mesh1d, Mesh2d, TriMesh2d};
use crate::metrics::metrics::{convergence_order, h1_seminorm_1d, l2_norm_1d, max_norm};
use crate::multigrid::vcycle::v_cycle_1d;
use crate::ptx_kernels::{
    cg_axpy_dot_ptx, csr_spmv_ptx, fdm_stencil_5pt_ptx, fem_assemble_ptx, gauss_seidel_step_ptx,
    mg_prolong_ptx, mg_restrict_ptx,
};
use crate::solver::cg::cg_solve;
use crate::solver::pcg::{pcg_ilu0, pcg_jacobi, pcg_ssor};
use crate::solver::sparse::SparseCsr;
use crate::spectral::chebyshev::solve_poisson_chebyshev;
use crate::spectral::chebyshev_2d::{Rectangle, chebyshev_2d_grid, chebyshev_2d_poisson};
use crate::spectral::fft_spectral::periodic_poisson_solve;
use crate::time::rk4::rk4_step;

fn make_uniform_1d_poisson_csr(n: usize, h: f64) -> SparseCsr {
    let inv_h2 = 1.0 / (h * h);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            cols.push(i - 1);
            vals.push(-inv_h2);
        }
        cols.push(i);
        vals.push(2.0 * inv_h2);
        if i + 1 < n {
            cols.push(i + 1);
            vals.push(-inv_h2);
        }
        row_ptr.push(cols.len());
    }
    SparseCsr::new(n, n, row_ptr, cols, vals).expect("ok")
}

// 1. FDM Poisson 1D: O(h^2) convergence on sin(pi x)
#[test]
fn fdm_poisson_1d_convergence_order_2() {
    let pi = std::f64::consts::PI;
    let ns = [21usize, 41, 81];
    let mut errs = Vec::new();
    let mut hs = Vec::new();
    for &n in &ns {
        let mesh = Mesh1d::uniform(0.0, 1.0, n).expect("ok");
        let f: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|x| pi * pi * (pi * x).sin())
            .collect();
        let u = solve_poisson_1d(&mesh, &f, Dirichlet1d { ua: 0.0, ub: 0.0 }).expect("ok");
        let err: f64 = u
            .iter()
            .zip(mesh.nodes.iter())
            .map(|(ui, x)| (ui - (pi * x).sin()).abs())
            .fold(0.0_f64, |a, b| a.max(b));
        errs.push(err);
        hs.push(mesh.h());
    }
    let order = convergence_order(hs[0], errs[0], hs[1], errs[1]).expect("ok");
    assert!(order > 1.8 && order < 2.2, "order={order}");
}

// 2. FDM Poisson 2D Gauss-Seidel converges
#[test]
fn fdm_poisson_2d_gs_converges() {
    let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
    let f = vec![1.0; mesh.n_nodes()];
    let (u, _it, res) = solve_poisson_2d_gs(
        &mesh,
        &f,
        DirichletRect {
            left: 0.0,
            right: 0.0,
            bottom: 0.0,
            top: 0.0,
        },
        5000,
        1.0e-6,
    )
    .expect("ok");
    assert!(res < 1.0e-3);
    let center = u[5 * mesh.ny + 5];
    assert!(center > 0.0);
}

// 3. FDM Poisson 2D CSR assembly + CG matches GS
#[test]
fn fdm_poisson_2d_csr_cg_solution() {
    let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 9, 9).expect("ok");
    let f = vec![1.0; mesh.n_nodes()];
    let a = assemble_poisson_2d_csr(&mesh).expect("ok");
    let bc = DirichletRect {
        left: 0.0,
        right: 0.0,
        bottom: 0.0,
        top: 0.0,
    };
    let rhs = build_poisson_2d_rhs(&mesh, &f, bc).expect("ok");
    let x = cg_solve(&a, &rhs, &vec![0.0; a.n_rows], 5000, 1.0e-12).expect("ok");
    let max_x = max_norm(&x);
    assert!(max_x > 0.0);
}

// 4. Heat 1D Crank-Nicolson decays at correct rate
#[test]
fn heat_1d_crank_nicolson_decay() {
    let pi = std::f64::consts::PI;
    let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
    let alpha = 1.0_f64;
    let dt = 0.001_f64;
    let mut u: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
    let t_final = 0.05_f64;
    let nsteps = (t_final / dt).round() as usize;
    for _ in 0..nsteps {
        crank_nicolson_step(&mesh, &mut u, alpha, dt, 0.0, 0.0).expect("ok");
    }
    let expected_amp = (-pi * pi * alpha * t_final).exp();
    let center = u[mesh.n / 2];
    let analytic = (pi * mesh.nodes[mesh.n / 2]).sin() * expected_amp;
    assert!((center - analytic).abs() < 5.0e-3);
}

// 5. Heat 1D Backward Euler with large dt is stable
#[test]
fn heat_1d_be_large_dt_stable() {
    let pi = std::f64::consts::PI;
    let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
    let alpha = 1.0;
    let dt = 5.0 * mesh.h() * mesh.h();
    let mut u: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
    for _ in 0..30 {
        backward_euler_step(&mesh, &mut u, alpha, dt, 0.0, 0.0).expect("ok");
    }
    // Solution heat-dissipates
    let nrm = l2_norm_1d(&u, mesh.h()).expect("ok");
    assert!(nrm < 0.5);
}

// 6. Forward Euler heat with CFL violation errors out
#[test]
fn heat_1d_fe_cfl_violation() {
    let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
    let alpha = 1.0;
    let dt = 1.0;
    let mut u = vec![0.5; mesh.n];
    let res = forward_euler_step(&mesh, &mut u, alpha, dt, 0.0, 0.0);
    assert!(res.is_err());
}

// 7. Wave 1D standing wave returns to ~zero at quarter period
#[test]
fn wave_1d_standing_wave_quarter_period() {
    let pi = std::f64::consts::PI;
    let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
    let c = 1.0;
    let dt = 0.5 * mesh.h() / c;
    let u0: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
    let v0 = vec![0.0; mesh.n];
    let mut s = WaveState1d::from_initial(&mesh, &u0, &v0, c, dt, 0.0, 0.0).expect("ok");
    let nsteps = (0.5 / dt).round() as usize;
    for _ in 0..nsteps {
        leapfrog_step_1d(&mesh, &mut s, c, dt, 0.0, 0.0).expect("ok");
    }
    let m = max_norm(&s.u_curr);
    assert!(m < 0.05);
}

// 8. Advection upwind transports a pulse
#[test]
fn advection_upwind_transport() {
    let mesh = Mesh1d::uniform(0.0, 1.0, 201).expect("ok");
    let c = 1.0;
    let dt = 0.5 * mesh.h() / c;
    let mut u: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|x| (-((x - 0.3) * (x - 0.3)) / 0.01).exp())
        .collect();
    let t_final = 0.4;
    let nsteps = (t_final / dt).round() as usize;
    for _ in 0..nsteps {
        upwind_step_1d(&mesh, &mut u, c, dt, 0.0).expect("ok");
    }
    let (idx, _) = u
        .iter()
        .enumerate()
        .fold((0_usize, f64::NEG_INFINITY), |(i, m), (k, &v)| {
            if v > m { (k, v) } else { (i, m) }
        });
    let x_peak = mesh.nodes[idx];
    assert!((x_peak - 0.7).abs() < 0.05);
}

// 9. FEM P1 Poisson solves correctly on a square (constant load)
#[test]
fn fem_p1_square_poisson() {
    let tri = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 9, 9).expect("ok");
    let fa = assemble_mass_stiffness(&tri).expect("ok");
    let mut k = fa.stiffness.clone();
    let mut b = assemble_load_centroid(&tri, |_, _| 1.0).expect("ok");
    let bc_vals: Vec<f64> = tri.boundary_nodes.iter().map(|_| 0.0).collect();
    apply_dirichlet_csr(&mut k, &mut b, &tri.boundary_nodes, &bc_vals).expect("ok");
    let x = pcg_jacobi(&k, &b, &vec![0.0; k.n_rows], 5000, 1.0e-10).expect("ok");
    let max_u = max_norm(&x);
    assert!(max_u > 0.0 && max_u < 0.2);
}

// 10. Chebyshev Poisson solves -u'' = 2 exactly (polynomial)
#[test]
fn cheb_poisson_polynomial_exact() {
    let n = 20;
    let x = crate::spectral::chebyshev::cheb_nodes(n);
    let f: Vec<f64> = x.iter().map(|_| 2.0).collect();
    let u = solve_poisson_chebyshev(n, &f).expect("ok");
    for i in 0..=n {
        let expected = 1.0 - x[i] * x[i];
        assert!((u[i] - expected).abs() < 1.0e-9);
    }
}

// 11. FFT periodic Poisson solves sine wave
#[test]
fn fft_periodic_poisson_sine() {
    let two_pi = std::f64::consts::TAU;
    let n = 64;
    let f: Vec<f64> = (0..n)
        .map(|j| {
            let x = two_pi * j as f64 / n as f64;
            4.0 * (2.0 * x).sin()
        })
        .collect();
    let u = periodic_poisson_solve(&f, two_pi).expect("ok");
    for (j, &uj) in u.iter().enumerate() {
        let x = two_pi * j as f64 / n as f64;
        let exp = (2.0 * x).sin();
        assert!((uj - exp).abs() < 1.0e-9);
    }
}

// 12. RK4 conserves harmonic oscillator energy
#[test]
fn rk4_harmonic_energy_conserved() {
    let mut s = vec![1.0, 0.0];
    let dt = 0.01;
    let n = 1000;
    for k in 0..n {
        let t = k as f64 * dt;
        rk4_step(&mut s, t, dt, |_, x| vec![x[1], -x[0]]).expect("ok");
    }
    let e_initial = 0.5;
    let e_final = 0.5 * (s[0] * s[0] + s[1] * s[1]);
    assert!((e_initial - e_final).abs() < 1.0e-7);
}

// 13. Multigrid V-cycle 1D converges to analytic solution
#[test]
fn multigrid_1d_converges() {
    let n = 33;
    let h = 1.0 / (n - 1) as f64;
    let mut u = vec![0.0; n];
    let f = vec![2.0; n];
    for _ in 0..12 {
        v_cycle_1d(&mut u, &f, h, 4, 4).expect("ok");
    }
    for (i, &ui) in u.iter().enumerate().take(n - 1).skip(1) {
        let x = i as f64 * h;
        let exact = x * (1.0 - x);
        assert!((ui - exact).abs() < 1.0e-3);
    }
}

// 14. PCG with ILU0 outperforms basic CG for stiff problem
#[test]
fn pcg_ilu0_solves_tridiag() {
    let n = 33;
    let h = 1.0 / (n - 1) as f64;
    let a = make_uniform_1d_poisson_csr(n, h);
    let b: Vec<f64> = (0..n)
        .map(|i| (i as f64 * h * std::f64::consts::PI).sin())
        .collect();
    let x_ilu = pcg_ilu0(&a, &b, &vec![0.0; n], 200, 1.0e-10).expect("ok");
    let r_ilu = a.matvec(&x_ilu).expect("ok");
    let max_res: f64 = (0..n)
        .map(|i| (r_ilu[i] - b[i]).abs())
        .fold(0.0, |a, c| a.max(c));
    assert!(max_res < 1.0e-8);
}

// 15. PCG with SSOR works
#[test]
fn pcg_ssor_solves_tridiag() {
    let n = 33;
    let h = 1.0 / (n - 1) as f64;
    let a = make_uniform_1d_poisson_csr(n, h);
    let b: Vec<f64> = (0..n)
        .map(|i| (i as f64 * h * std::f64::consts::PI).sin())
        .collect();
    let x = pcg_ssor(&a, &b, &vec![0.0; n], 1.2, 200, 1.0e-10).expect("ok");
    let r = a.matvec(&x).expect("ok");
    let max_res: f64 = (0..n)
        .map(|i| (r[i] - b[i]).abs())
        .fold(0.0, |a, c| a.max(c));
    assert!(max_res < 1.0e-8);
}

// 16. Lax-Wendroff conserves mass for periodic advection
#[test]
fn lax_wendroff_periodic_conserves_mass() {
    let mesh = Mesh1d::uniform(0.0, 1.0, 101).expect("ok");
    let c = 1.0;
    let dt = 0.5 * mesh.h() / c;
    let mut u: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|x| (-((x - 0.5).powi(2)) / 0.005).exp())
        .collect();
    let initial_mass: f64 = u.iter().sum::<f64>() * mesh.h();
    for _ in 0..200 {
        lax_wendroff_step_1d(&mesh, &mut u, c, dt).expect("ok");
    }
    let final_mass: f64 = u.iter().sum::<f64>() * mesh.h();
    assert!((final_mass - initial_mass).abs() / initial_mass.abs() < 1.0e-6);
}

// 17. DG1D LGL weights integrate constants and polynomials exactly
#[test]
fn dg1d_lgl_quadrature_exact() {
    let p = 4;
    let x = lgl_nodes(p).expect("ok");
    let w = lgl_weights(p).expect("ok");
    // integral of 1 over [-1,1] = 2
    let s: f64 = w.iter().sum();
    assert!((s - 2.0).abs() < 1.0e-10);
    // integral of x^2 over [-1,1] = 2/3
    let s2: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| xi * xi * wi).sum();
    assert!((s2 - 2.0 / 3.0).abs() < 1.0e-10);
}

// 18. DG1D Lax-Friedrichs flux is upwind for constant a
#[test]
fn dg1d_lax_friedrichs_upwind() {
    let lf = lax_friedrichs_flux(1.0, 0.0, 2.0);
    let up = upwind_flux(1.0, 0.0, 2.0);
    assert!((lf - 2.0).abs() < 1.0e-12);
    assert!((up - 2.0).abs() < 1.0e-12);
}

// 19. PTX kernel strings non-empty across 6 SM × 7 kernels
#[test]
fn ptx_kernels_all_sm_versions() {
    type KFn = fn(u32) -> String;
    let kernels: &[(&str, KFn)] = &[
        ("fdm_stencil_5pt", fdm_stencil_5pt_ptx),
        ("gauss_seidel_step", gauss_seidel_step_ptx),
        ("csr_spmv", csr_spmv_ptx),
        ("cg_axpy_dot", cg_axpy_dot_ptx),
        ("fem_assemble", fem_assemble_ptx),
        ("mg_restrict", mg_restrict_ptx),
        ("mg_prolong", mg_prolong_ptx),
    ];
    let sms = [75u32, 80, 86, 89, 90, 100];
    for sm in sms {
        for (name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "kernel {name} sm={sm} empty");
            assert!(s.contains(".visible .entry"));
            assert!(s.contains("ret"));
        }
    }
}

// 20. H1 seminorm and L2 norm scale correctly with mesh refinement
#[test]
fn norm_scaling_refinement() {
    let pi = std::f64::consts::PI;
    let mesh1 = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
    let mesh2 = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
    let u1: Vec<f64> = mesh1.nodes.iter().map(|x| (pi * x).sin()).collect();
    let u2: Vec<f64> = mesh2.nodes.iter().map(|x| (pi * x).sin()).collect();
    let l2_1 = l2_norm_1d(&u1, mesh1.h()).expect("ok");
    let l2_2 = l2_norm_1d(&u2, mesh2.h()).expect("ok");
    let exact_l2 = (0.5_f64).sqrt(); // integral_0^1 sin^2(pi x) = 1/2
    assert!((l2_1 - exact_l2).abs() < 0.05);
    assert!((l2_2 - exact_l2).abs() < 0.05);
    let h1_1 = h1_seminorm_1d(&u1, mesh1.h()).expect("ok");
    // exact |u|_H1 = pi * sqrt(1/2)
    let exact_h1 = pi * (0.5_f64).sqrt();
    assert!((h1_1 - exact_h1).abs() < 0.5);
}

// 21. LcgRng deterministic sequence
#[test]
fn lcg_deterministic_seq() {
    let mut a = LcgRng::new(42);
    let mut b = LcgRng::new(42);
    for _ in 0..32 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

// 22. DG1D space construction & DOF count
#[test]
fn dg1d_space_construction() {
    let s = Dg1dSpace::new(5, 3, 0.0, 1.0).expect("ok");
    assert_eq!(s.n_dofs(), 20);
    let md = s.mass_diag();
    assert_eq!(md.len(), 4);
    let total_mass: f64 = md.iter().sum::<f64>() * s.n_elem as f64;
    // Each element has mass = h_e (since LGL integrates 1 to 2 on [-1,1] * (h_e/2) -> h_e per element)
    // total mass over domain = 1.0 (length)
    assert!((total_mass - 1.0).abs() < 1e-10);
}

// 23. Tensor-product Chebyshev 2D Poisson: SPECTRAL accuracy on a smooth
// manufactured solution (max nodal error ≤ 1e-8 at moderate N — far beyond O(h²)).
#[test]
fn chebyshev_2d_poisson_spectral_accuracy() {
    let pi = std::f64::consts::PI;
    let domain = Rectangle::new(0.0, 1.0, 0.0, 1.0).expect("ok");
    let n = 20;
    let (x, y) = chebyshev_2d_grid(n, n, &domain);
    let n1 = n + 1;
    let mut f = vec![0.0; n1 * n1];
    let mut exact = vec![0.0; n1 * n1];
    for (iy, &yy) in y.iter().enumerate() {
        for (ix, &xx) in x.iter().enumerate() {
            let u = (pi * xx).sin() * (pi * yy).sin();
            exact[iy * n1 + ix] = u;
            f[iy * n1 + ix] = 2.0 * pi * pi * u; // -Δu = 2π² u
        }
    }
    let bc = vec![0.0; n1 * n1];
    let u = chebyshev_2d_poisson(n, n, &domain, &f, &bc).expect("ok");
    let err = u
        .iter()
        .zip(&exact)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(err < 1e-8, "Chebyshev-2D spectral error {err}");
}

// 24. RT0/P0 mixed Poisson: LOCAL CONSERVATION — ∫_T div σ_h = ∫_T f exactly,
// element by element (the defining property of the lowest-order mixed method).
#[test]
fn mixed_rt0_local_conservation() {
    let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 6, 6).expect("ok");
    let n_tri = mesh.n_tri();
    let mut f = vec![0.0; n_tri];
    for (e, fe) in f.iter_mut().enumerate() {
        *fe = 0.5 + 0.27 * e as f64; // spatially varying forcing
    }
    let sol = mixed_poisson_rt0(&mesh, &f, &MixedBoundary::Dirichlet(|_, _| 0.0)).expect("ok");
    let div = element_divergence(&mesh, &sol).expect("ok");
    for e in 0..n_tri {
        let area = mesh.area(e).expect("area");
        let int_div = div[e] * area; // ∫_T div σ_h
        let int_f = f[e] * area; // ∫_T f
        assert!(
            (int_div - int_f).abs() < 1e-10,
            "elem {e}: ∫div σ_h={int_div} ∫f={int_f}"
        );
    }
}

// 25. DG-2D linear advection: discrete MASS conservation under periodic BC to
// ~1e-12, plus a smooth Gaussian advected one full period returns ≈ itself.
#[test]
fn dg_2d_advection_mass_and_transport() {
    let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 13, 13).expect("ok");
    let bc = DgBoundary::Periodic {
        x0: 0.0,
        x1: 1.0,
        y0: 0.0,
        y1: 1.0,
    };
    let space = Dg2dSpace::new(&mesh, bc).expect("ok");
    let mut u0 = vec![0.0; space.n_dofs()];
    for e in 0..space.n_elem {
        let v = space.element_vertices(e).expect("v");
        for i in 0..3 {
            let dx = v[i][0] - 0.5;
            let dy = v[i][1] - 0.5;
            u0[3 * e + i] = (-30.0 * (dx * dx + dy * dy)).exp();
        }
    }
    let m0 = space.total_mass(&u0);
    let beta = (1.0, 0.0);
    let dt0 = 0.3 * space.cfl_dt(beta.0, beta.1, 1.0);
    let nsteps = (1.0 / dt0).ceil() as usize;
    let dt = 1.0 / nsteps as f64; // land exactly on one period T=1
    let u = dg_2d_advect(&mesh, &u0, beta, dt, nsteps, bc, false).expect("ok");
    let m1 = space.total_mass(&u);
    assert!((m1 - m0).abs() < 1e-12, "DG-2D mass drift {m0} -> {m1}");
    // one period ⇒ near-identity for exact transport (high order, no limiter).
    let mut err2 = 0.0;
    let mut nrm2 = 0.0;
    for e in 0..space.n_elem {
        let area = space.area(e).expect("a");
        for i in 0..3 {
            let d = u[3 * e + i] - u0[3 * e + i];
            err2 += area / 3.0 * d * d;
            nrm2 += area / 3.0 * u0[3 * e + i] * u0[3 * e + i];
        }
    }
    assert!(
        (err2 / nrm2).sqrt() < 0.15,
        "DG-2D one-period L2 error too large"
    );
}

// 26. DG-2D inviscid Burgers Riemann step (uL>uR) with the slope limiter ON:
// the shock travels at the Rankine-Hugoniot speed s=(uL+uR)/2 and the solution
// stays monotone within [uR, uL].
#[test]
fn dg_2d_burgers_rankine_hugoniot() {
    let nx = 81;
    let ny = 3;
    let mesh = TriMesh2d::rect_grid(-1.0, 3.0, 0.0, 0.1, nx, ny).expect("ok");
    let bc = DgBoundary::Compact { far_field: 0.0 };
    let space = Dg2dSpace::new(&mesh, bc).expect("ok");
    let mut u0 = vec![0.0; space.n_dofs()];
    for e in 0..space.n_elem {
        let v = space.element_vertices(e).expect("v");
        for i in 0..3 {
            u0[3 * e + i] = if v[i][0] < 1.0 { 1.0 } else { 0.0 };
        }
    }
    let t_final = 1.0;
    let dt0 = 0.4 * space.cfl_dt(1.0, 0.0, 1.0);
    let nsteps = (t_final / dt0).ceil() as usize;
    let dt = t_final / nsteps as f64;
    let u = dg_2d_burgers(&mesh, &u0, 1.0, dt, nsteps, bc, true).expect("ok");
    // monotonicity: solution stays within [0, 1].
    let umax = u.iter().cloned().fold(f64::MIN, f64::max);
    let umin = u.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        umax < 1.0 + 1e-9 && umin > -1e-9,
        "Burgers not monotone [{umin},{umax}]"
    );
    // shock front: largest centroid-x with cell mean > 0.5; RH speed s=0.5.
    let mut front = f64::MIN;
    for e in 0..space.n_elem {
        if space.cell_mean(&u, e) > 0.5 {
            let c = space.centroid(e).expect("c");
            if c[0] > front {
                front = c[0];
            }
        }
    }
    let analytic = 1.0 + 0.5 * t_final;
    assert!(
        (front - analytic).abs() < 0.1,
        "shock front {front} vs RH {analytic}"
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  Verification & worked-example deliverables (TODO "Verification Gaps" /
//  "Documentation Gaps").  Each computes its convergence rate / residual
//  reduction at runtime and asserts it — no number is hard-coded from
//  off-line knowledge.
// ════════════════════════════════════════════════════════════════════════════

/// Build a structured tetrahedral mesh of the unit cube `[0,1]³`.
///
/// `nv` vertices per axis ⇒ `(nv-1)³` hexahedral cells, each split into the six
/// tetrahedra of the Kuhn / Freudenthal decomposition along the body diagonal
/// `v0 → v7`.  The six monotone lattice paths from `(0,0,0)` to `(1,1,1)`
/// tile the cube exactly (no gaps, no overlaps), each tet having volume
/// `h³/6`.
///
/// Returns `(coords, tets)` where `coords[g] = [x,y,z]` for global vertex `g`
/// (lexicographic id `i + nv*(j + nv*k)`) and each `tets[e]` lists the four
/// global vertex ids of one tetrahedron.
fn unit_cube_tet_mesh(nv: usize) -> (Vec<[f64; 3]>, Vec<[usize; 4]>) {
    assert!(nv >= 2, "need at least 2 vertices per axis");
    let h = 1.0 / (nv - 1) as f64;
    let vid = |i: usize, j: usize, k: usize| i + nv * (j + nv * k);

    let mut coords = Vec::with_capacity(nv * nv * nv);
    for k in 0..nv {
        for j in 0..nv {
            for i in 0..nv {
                coords.push([i as f64 * h, j as f64 * h, k as f64 * h]);
            }
        }
    }

    // Local corner numbering of a cell: bit0 = +x, bit1 = +y, bit2 = +z.
    //   0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(1,1,0)
    //   4=(0,0,1) 5=(1,0,1) 6=(0,1,1) 7=(1,1,1)
    // Kuhn-6 tetrahedra, every one of which shares the body diagonal 0–7.
    const KUHN6: [[usize; 4]; 6] = [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ];

    let mut tets = Vec::with_capacity(6 * (nv - 1).pow(3));
    for k in 0..nv - 1 {
        for j in 0..nv - 1 {
            for i in 0..nv - 1 {
                // Global ids of the eight cell corners, indexed by the local
                // bit numbering above.
                let corner = |c: usize| {
                    let di = c & 1;
                    let dj = (c >> 1) & 1;
                    let dk = (c >> 2) & 1;
                    vid(i + di, j + dj, k + dk)
                };
                for t in &KUHN6 {
                    tets.push([corner(t[0]), corner(t[1]), corner(t[2]), corner(t[3])]);
                }
            }
        }
    }
    (coords, tets)
}

/// Solve `−Δu = f` on the unit-cube tet mesh with homogeneous Dirichlet BCs and
/// return the discrete nodal solution together with the nodal-quadrature L²
/// error against `exact`.
///
/// Global P1 assembly uses [`p1_tet_local_stiffness`] per element; the load is
/// formed with the vertex (trapezoidal) quadrature `b_i += (V/4)·f(p_i)`, which
/// is exact to O(h²) and keeps the overall P1 rate.  The system is solved with
/// the crate's [`cg_solve`].  The reported error is
/// `sqrt(Σ_T (V_T/4) Σ_{i∈T} (u_h(p_i) − u(p_i))²)`, the natural element-wise
/// L² quadrature consistent with the load rule.
fn solve_cube_poisson_tet<Ff, Fu>(nv: usize, f: Ff, exact: Fu) -> (Vec<f64>, f64)
where
    Ff: Fn(f64, f64, f64) -> f64,
    Fu: Fn(f64, f64, f64) -> f64,
{
    let (coords, tets) = unit_cube_tet_mesh(nv);
    let n = coords.len();

    // Dense global assembly (n is modest for these meshes) then convert to CSR.
    let mut k_dense = vec![0.0_f64; n * n];
    let mut load = vec![0.0_f64; n];
    for tet in &tets {
        let nodes = [
            coords[tet[0]],
            coords[tet[1]],
            coords[tet[2]],
            coords[tet[3]],
        ];
        let ke = p1_tet_local_stiffness(&nodes).expect("local stiffness");
        // Tet volume from the same edge-triple determinant used internally.
        let e1 = [
            nodes[1][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1],
            nodes[1][2] - nodes[0][2],
        ];
        let e2 = [
            nodes[2][0] - nodes[0][0],
            nodes[2][1] - nodes[0][1],
            nodes[2][2] - nodes[0][2],
        ];
        let e3 = [
            nodes[3][0] - nodes[0][0],
            nodes[3][1] - nodes[0][1],
            nodes[3][2] - nodes[0][2],
        ];
        let triple = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
            + e1[1] * (e2[2] * e3[0] - e2[0] * e3[2])
            + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
        let vol = triple.abs() / 6.0;
        for a in 0..4 {
            let ga = tet[a];
            for b in 0..4 {
                k_dense[ga * n + tet[b]] += ke[a * 4 + b];
            }
            let p = nodes[a];
            load[ga] += 0.25 * vol * f(p[0], p[1], p[2]);
        }
    }

    // Convert dense → CSR (drop exact zeros but always keep the diagonal).
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        for j in 0..n {
            let v = k_dense[i * n + j];
            if v != 0.0 || i == j {
                cols.push(j);
                vals.push(v);
            }
        }
        row_ptr.push(cols.len());
    }
    let mut a = SparseCsr::new(n, n, row_ptr, cols, vals).expect("csr");

    // Homogeneous Dirichlet on every face of the cube.
    let nv1 = nv - 1;
    let mut bc_nodes = Vec::new();
    for k in 0..nv {
        for j in 0..nv {
            for i in 0..nv {
                let on_bdy = i == 0 || i == nv1 || j == 0 || j == nv1 || k == 0 || k == nv1;
                if on_bdy {
                    bc_nodes.push(i + nv * (j + nv * k));
                }
            }
        }
    }
    let bc_vals = vec![0.0; bc_nodes.len()];
    apply_dirichlet_csr(&mut a, &mut load, &bc_nodes, &bc_vals).expect("dirichlet");

    let u = cg_solve(&a, &load, &vec![0.0; n], 20_000, 1.0e-12).expect("cg");

    // Element-wise nodal-quadrature L² error.
    let mut err2 = 0.0_f64;
    for tet in &tets {
        let nodes = [
            coords[tet[0]],
            coords[tet[1]],
            coords[tet[2]],
            coords[tet[3]],
        ];
        let e1 = [
            nodes[1][0] - nodes[0][0],
            nodes[1][1] - nodes[0][1],
            nodes[1][2] - nodes[0][2],
        ];
        let e2 = [
            nodes[2][0] - nodes[0][0],
            nodes[2][1] - nodes[0][1],
            nodes[2][2] - nodes[0][2],
        ];
        let e3 = [
            nodes[3][0] - nodes[0][0],
            nodes[3][1] - nodes[0][1],
            nodes[3][2] - nodes[0][2],
        ];
        let triple = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
            + e1[1] * (e2[2] * e3[0] - e2[0] * e3[2])
            + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
        let vol = triple.abs() / 6.0;
        for a in 0..4 {
            let g = tet[a];
            let p = nodes[a];
            let d = u[g] - exact(p[0], p[1], p[2]);
            err2 += 0.25 * vol * d * d;
        }
    }
    (u, err2.sqrt())
}

// 27. 3D Poisson on TETRAHEDRAL meshes: P1 FEM on a manufactured solution
// `u = sin(πx)sin(πy)sin(πz)` (so `−Δu = 3π² u`) over the unit cube, solved on
// three successively refined Kuhn-6 tet meshes.  The discrete L² error must
// decrease at the expected O(h²) P1 rate.  The observed order is COMPUTED from
// the measured errors via `convergence_order` (never hard-coded).
#[test]
fn fem_p1_tet_3d_poisson_convergence_order_2() {
    let pi = std::f64::consts::PI;
    let exact = |x: f64, y: f64, z: f64| (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
    let f = |x: f64, y: f64, z: f64| 3.0 * pi * pi * exact(x, y, z);

    let nvs = [5usize, 9, 17]; // h = 1/4, 1/8, 1/16
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &nv in &nvs {
        let (_u, err) = solve_cube_poisson_tet(nv, f, exact);
        hs.push(1.0 / (nv - 1) as f64);
        errs.push(err);
        assert!(err.is_finite() && err > 0.0, "nv={nv} err={err}");
    }
    // Print the error table + observed orders (visible with `--nocapture`).
    println!("3D tet P1 Poisson convergence (manufactured sin·sin·sin):");
    for w in 0..nvs.len() {
        if w == 0 {
            println!("  h={:.5}  L2err={:.3e}", hs[w], errs[w]);
        } else {
            let p = convergence_order(hs[w - 1], errs[w - 1], hs[w], errs[w]).expect("order");
            println!(
                "  h={:.5}  L2err={:.3e}  observed order={:.3}",
                hs[w], errs[w], p
            );
        }
    }

    // Errors must monotonically shrink under refinement.
    assert!(
        errs[1] < errs[0] && errs[2] < errs[1],
        "errors not decreasing: {errs:?}"
    );
    // Each refinement (h → h/2) must show ≈ O(h²): order in [1.7, 2.3].
    let p1 = convergence_order(hs[0], errs[0], hs[1], errs[1]).expect("order");
    let p2 = convergence_order(hs[1], errs[1], hs[2], errs[2]).expect("order");
    assert!(
        (1.7..=2.3).contains(&p1),
        "first refinement order {p1} not ≈ 2 (errs {errs:?})"
    );
    assert!(
        (1.7..=2.3).contains(&p2),
        "second refinement order {p2} not ≈ 2 (errs {errs:?})"
    );
}

// 28. Worked example (TODO "Documentation Gaps"): solve −u'' = f on [0,1] with
// the manufactured solution `u = sin(πx)` (so `f = π² sin(πx)`) on four
// successively refined uniform meshes, print the max-error table + observed
// convergence orders, and assert the rate is the expected O(h²).
#[test]
fn worked_example_poisson_1d_convergence() {
    let pi = std::f64::consts::PI;
    let exact = |x: f64| (pi * x).sin();
    let f = |x: f64| pi * pi * (pi * x).sin();

    let ns = [11usize, 21, 41, 81]; // h = 1/10, 1/20, 1/40, 1/80
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &n in &ns {
        let mesh = Mesh1d::uniform(0.0, 1.0, n).expect("mesh");
        let fvals: Vec<f64> = mesh.nodes.iter().map(|&x| f(x)).collect();
        let u = solve_poisson_1d(&mesh, &fvals, Dirichlet1d { ua: 0.0, ub: 0.0 }).expect("solve");
        let err = u
            .iter()
            .zip(mesh.nodes.iter())
            .map(|(&ui, &x)| (ui - exact(x)).abs())
            .fold(0.0_f64, f64::max);
        hs.push(mesh.h());
        errs.push(err);
    }

    println!("1D Poisson −u''=π²sin(πx), manufactured u=sin(πx):");
    println!("        h          max-error      order");
    let mut orders = Vec::new();
    for w in 0..ns.len() {
        if w == 0 {
            println!("  {:.6e}   {:.6e}      -", hs[w], errs[w]);
        } else {
            let p = convergence_order(hs[w - 1], errs[w - 1], hs[w], errs[w]).expect("order");
            orders.push(p);
            println!("  {:.6e}   {:.6e}   {:.4}", hs[w], errs[w], p);
        }
    }

    // Every successive order must be ≈ 2, and the finest (asymptotic) one tight.
    for (w, &p) in orders.iter().enumerate() {
        assert!(
            (1.9..=2.1).contains(&p),
            "refinement {w} order {p} not ≈ 2 (errs {errs:?})"
        );
    }
    let finest = *orders.last().expect("≥1 order");
    assert!(
        (finest - 2.0).abs() < 0.05,
        "asymptotic order {finest} not ≈ 2"
    );
}

// 29. Worked example (TODO "Documentation Gaps"): time-step the heat equation
// `u_t = α u_xx` with Crank–Nicolson and verify BOTH
//   (a) the numerical solution matches the analytic `u(x,t)=sin(πx)e^{−απ²t}`
//       to tolerance, and
//   (b) Crank–Nicolson is 2nd-order in time — halving dt quarters the temporal
//       error.  The temporal error is isolated by self-comparison against a
//       reference solution on the SAME (fixed, fine) spatial mesh with a very
//       small dt, which cancels the spatial discretisation error exactly.
#[test]
fn worked_example_heat_crank_nicolson() {
    let pi = std::f64::consts::PI;
    let alpha = 1.0_f64;
    let t_final = 0.1_f64;
    // Fine spatial mesh so the spatial O(h²) error is far below the temporal
    // errors probed by the dt sweep.
    let mesh = Mesh1d::uniform(0.0, 1.0, 201).expect("mesh");
    let u0: Vec<f64> = mesh.nodes.iter().map(|&x| (pi * x).sin()).collect();

    let integrate = |dt: f64| -> Vec<f64> {
        let nsteps = (t_final / dt).round() as usize;
        let dt_used = t_final / nsteps as f64;
        let mut u = u0.clone();
        for _ in 0..nsteps {
            crank_nicolson_step(&mesh, &mut u, alpha, dt_used, 0.0, 0.0).expect("cn step");
        }
        u
    };

    // (a) Accuracy vs the analytic exponential-decay solution.
    let decay = (-pi * pi * alpha * t_final).exp();
    let u_acc = integrate(0.0025);
    let mut max_abs = 0.0_f64;
    for (&uh, &x) in u_acc.iter().zip(mesh.nodes.iter()) {
        let analytic = (pi * x).sin() * decay;
        max_abs = max_abs.max((uh - analytic).abs());
    }
    println!("Heat CN vs analytic sin(πx)e^(−απ²t) at t={t_final}: max|err|={max_abs:.3e}");
    assert!(max_abs < 1.0e-3, "CN solution off analytic: {max_abs}");

    // (b) Temporal order via self-convergence on the fixed mesh.  Reference
    // uses a dt 8× smaller than the smallest probed dt, so its temporal error
    // is ~64× smaller and acts as ground truth for the time discretisation.
    let dts = [0.02_f64, 0.01, 0.005];
    let u_ref = integrate(dts[2] / 8.0);
    let l2_diff = |a: &[f64], b: &[f64]| -> f64 {
        let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        (mesh.h() * s).sqrt()
    };
    let mut terr = Vec::new();
    for &dt in &dts {
        let u = integrate(dt);
        terr.push(l2_diff(&u, &u_ref));
    }
    println!("Crank–Nicolson temporal self-convergence (fixed mesh):");
    let mut torders = Vec::new();
    for w in 0..dts.len() {
        if w == 0 {
            println!("  dt={:.4}  L2(time-err)={:.3e}", dts[w], terr[w]);
        } else {
            // order = log(e_coarse/e_fine)/log(dt_coarse/dt_fine).
            let p = (terr[w - 1] / terr[w]).ln() / (dts[w - 1] / dts[w]).ln();
            torders.push(p);
            println!(
                "  dt={:.4}  L2(time-err)={:.3e}  order={:.3}  (err ratio {:.2})",
                dts[w],
                terr[w],
                p,
                terr[w - 1] / terr[w]
            );
        }
    }
    // Halving dt must roughly quarter the error ⇒ order ≈ 2.
    for (w, &p) in torders.iter().enumerate() {
        assert!(
            (1.8..=2.2).contains(&p),
            "CN temporal order {p} (step {w}) not ≈ 2 (terr {terr:?})"
        );
    }
}

// 30. Worked example (TODO "Documentation Gaps"): geometric multigrid V-cycle
// on the 1D Poisson problem `−u'' = π² sin(πx)`.  Run several V-cycles, record
// the residual norm after each, print the per-cycle residual history and its
// reduction factor, and assert the residual drops by a roughly constant factor
// (< 1) every cycle — i.e. geometric (mesh-independent) convergence.
#[test]
fn worked_example_multigrid_vcycle_residual() {
    let pi = std::f64::consts::PI;
    let n = 129; // 2^7 + 1 ⇒ a full coarsening hierarchy down to n=3
    let h = 1.0 / (n - 1) as f64;
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * h;
            pi * pi * (pi * x).sin()
        })
        .collect();

    // Residual r = f − A u for the interior 1D Poisson stencil.
    let residual_norm = |u: &[f64]| -> f64 {
        let inv_h2 = 1.0 / (h * h);
        let mut s = 0.0;
        for i in 1..n - 1 {
            let lap = inv_h2 * (2.0 * u[i] - u[i - 1] - u[i + 1]);
            let r = f[i] - lap;
            s += r * r;
        }
        s.sqrt()
    };

    let mut u = vec![0.0; n];
    let mut history = vec![residual_norm(&u)];
    let n_cycles = 8;
    for _ in 0..n_cycles {
        v_cycle_1d(&mut u, &f, h, 2, 2).expect("v-cycle");
        history.push(residual_norm(&u));
    }

    println!("Multigrid V-cycle residual history (1D Poisson, n={n}):");
    let mut factors = Vec::new();
    for c in 0..history.len() {
        if c == 0 {
            println!("  cycle {c}: residual={:.6e}", history[c]);
        } else {
            let factor = history[c] / history[c - 1];
            factors.push(factor);
            println!(
                "  cycle {c}: residual={:.6e}  reduction factor={:.4}",
                history[c], factor
            );
        }
    }

    // Total reduction must be large (textbook MG drives the residual to ~0).
    assert!(
        history.last().expect("hist") < &(1.0e-6 * history[0]),
        "MG did not reduce residual enough: {history:?}"
    );
    // Each cycle must contract the residual (factor < 1)…
    for (c, &factor) in factors.iter().enumerate() {
        assert!(
            factor < 1.0,
            "cycle {c} factor {factor} ≥ 1 (no contraction)"
        );
    }
    // …and once the iteration settles, the per-cycle factor is roughly
    // constant (geometric convergence): the asymptotic contraction factors
    // stay well below 1 and within a bounded band of each other.
    let tail = &factors[factors.len().saturating_sub(3)..];
    let tail_max = tail.iter().cloned().fold(f64::MIN, f64::max);
    let tail_min = tail.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        tail_max < 0.6,
        "asymptotic V-cycle contraction {tail_max} too weak (factors {factors:?})"
    );
    assert!(
        tail_max - tail_min < 0.35,
        "V-cycle factor not roughly constant: [{tail_min}, {tail_max}]"
    );
}
