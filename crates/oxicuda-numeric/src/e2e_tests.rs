//! End-to-end integration tests for `oxicuda-numeric`.
//!
//! Exercises each public-facing module against well-known reference values.

#![allow(clippy::approx_constant)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

use crate::cubature::genz_malik::genz_malik_cubature;
use crate::cubature::monte_carlo::monte_carlo_integrate;
use crate::cubature::quasi_monte_carlo_sobol::{sobol_integrate, sobol_point};
use crate::cubature::tensor_product_gauss::tensor_product_gauss_integrate;
use crate::diff::central_difference::central_difference;
use crate::diff::complex_step::{CDual, complex_step_derivative};
use crate::diff::richardson_extrapolation::richardson_derivative;
use crate::error::NumericResult;
use crate::handle::LcgRng;
use crate::interp::akima::akima_interpolate;
use crate::interp::barycentric::{barycentric_eval, barycentric_weights};
use crate::interp::cubic_spline::{natural_cubic_spline, spline_eval};
use crate::interp::hermite::hermite_interpolate;
use crate::interp::lagrange::lagrange_interpolate;
use crate::interp::linear::linear_interpolate;
use crate::interp::pchip::pchip_interpolate;
use crate::ode::bdf12::{bdf1, bdf2};
use crate::ode::dopri5::dopri5;
use crate::ode::explicit_euler::explicit_euler;
use crate::ode::heun::heun;
use crate::ode::imex_euler::imex_euler;
use crate::ode::rk4::rk4;
use crate::ode::rosenbrock_w::rosenbrock_w;
use crate::poly::durand_kerner::durand_kerner;
use crate::poly::horner_eval::horner;
use crate::poly::jenkins_traub::jenkins_traub_real;
use crate::ptx_kernels::{
    bessel_recurrence_ptx, bisection_step_ptx, central_diff_ptx, gauss_quad_accumulate_ptx,
    horner_eval_ptx, rk4_stage_ptx, spline_eval_ptx,
};
use crate::quadrature::adaptive_simpson::adaptive_simpson;
use crate::quadrature::clenshaw_curtis::clenshaw_curtis;
use crate::quadrature::gauss_chebyshev::gauss_chebyshev_integrate;
use crate::quadrature::gauss_kronrod::gauss_kronrod_g7k15;
use crate::quadrature::gauss_legendre::gauss_legendre_integrate;
use crate::quadrature::romberg::romberg;
use crate::root::aberth_all_roots::aberth_all_roots;
use crate::root::bisection::bisection;
use crate::root::brent::brent;
use crate::root::halley::halley;
use crate::root::newton::newton;
use crate::root::secant::secant;
use crate::special::airy::airy_ai;
use crate::special::bessel_jy::{bessel_j0, bessel_jn};
use crate::special::elliptic_ke::{elliptic_e, elliptic_k};
use crate::special::lambert_w::lambert_w0;
use crate::special::zeta::zeta;

const PI: f64 = std::f64::consts::PI;

// 1. Bisection finds π/2 as root of cos on [0, π].
#[test]
fn e2e_bisection_cos_pi_over_two() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.cos()) };
    let r = bisection(f, 0.0, PI, 1.0e-12, 100).expect("ok");
    assert!((r - PI / 2.0).abs() < 1.0e-10);
}

// 2. Newton converges to 2^(1/3) in <15 iterations.
#[test]
fn e2e_newton_cube_root_two() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 2.0) };
    let g = |x: f64| -> NumericResult<f64> { Ok(3.0 * x * x) };
    let r = newton(f, g, 1.0, 1.0e-14, 15).expect("ok");
    assert!((r - 2.0_f64.powf(1.0 / 3.0)).abs() < 1.0e-10);
}

// 3. Brent finds π as root of sin on [3, 4].
#[test]
fn e2e_brent_sin_pi() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
    let r = brent(f, 3.0, 4.0, 1.0e-14, 50).expect("ok");
    assert!((r - PI).abs() < 1.0e-10);
}

// 4. Secant on log(x) = 0 finds 1.
#[test]
fn e2e_secant_log_root() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.ln()) };
    let r = secant(f, 0.5, 2.0, 1.0e-12, 100).expect("ok");
    assert!((r - 1.0).abs() < 1.0e-8);
}

// 5. Halley on x³ - 2 = 0 converges cubically.
#[test]
fn e2e_halley_cube_root_two() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 2.0) };
    let g = |x: f64| -> NumericResult<f64> { Ok(3.0 * x * x) };
    let h = |x: f64| -> NumericResult<f64> { Ok(6.0 * x) };
    let r = halley(f, g, h, 1.0, 1.0e-14, 15).expect("ok");
    assert!((r - 2.0_f64.powf(1.0 / 3.0)).abs() < 1.0e-12);
}

// 6. Romberg integrates 1/(1+x²) on [0,1] to π/4.
#[test]
fn e2e_romberg_arctan() {
    let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
    let r = romberg(f, 0.0, 1.0, 1.0e-12, 12).expect("ok");
    assert!((r - PI / 4.0).abs() < 1.0e-10);
}

// 7. Gauss-Legendre n=5 integrates x⁹ exactly (= 0).
#[test]
fn e2e_gauss_legendre_x9_exact() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(9)) };
    let r = gauss_legendre_integrate(f, -1.0, 1.0, 5).expect("ok");
    assert!(r.abs() < 1.0e-12);
}

// 8. Adaptive Simpson handles 1/√x on (1e-12, 1] with tol 1e-6, yielding ≈ 2.
#[test]
fn e2e_adaptive_simpson_sqrt_singular() {
    let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / x.sqrt()) };
    let r = adaptive_simpson(f, 1.0e-12, 1.0, 1.0e-6, 50).expect("ok");
    assert!((r - 2.0).abs() < 1.0e-3);
}

// 9. Gauss-Kronrod G7K15 returns a tight error estimate on smooth integrands.
#[test]
fn e2e_gauss_kronrod_smooth() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.exp()) };
    let (v, err) = gauss_kronrod_g7k15(f, 0.0, 1.0).expect("ok");
    assert!((v - (std::f64::consts::E - 1.0)).abs() < 1.0e-12);
    assert!(err < 1.0e-3);
}

// 10. Clenshaw-Curtis integrates a polynomial exactly.
#[test]
fn e2e_clenshaw_curtis_polynomial() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(4)) };
    let r = clenshaw_curtis(f, 0.0, 1.0, 8).expect("ok");
    assert!((r - 0.2).abs() < 1.0e-10);
}

// 11. Gauss-Chebyshev integrates the constant 1/√(1-x²) on (-1, 1) yielding π.
#[test]
fn e2e_gauss_chebyshev_pi() {
    let f = |_x: f64| -> NumericResult<f64> { Ok(1.0) };
    let r = gauss_chebyshev_integrate(f, 6).expect("ok");
    assert!((r - PI).abs() < 1.0e-12);
}

// 12. bessel_j0(0) = 1, j0(2.4048...) ≈ 0.
#[test]
fn e2e_bessel_j0_known_values() {
    assert!((bessel_j0(0.0) - 1.0).abs() < 1.0e-12);
    assert!(bessel_j0(2.404_825_557_695_773).abs() < 1.0e-7);
}

// 13. Airy Ai(0) = 1 / (3^(2/3) Γ(2/3)).
#[test]
fn e2e_airy_ai_zero() {
    let r = airy_ai(0.0).expect("ok");
    let expected = 1.0 / (3.0_f64.powf(2.0 / 3.0) * 1.354_117_939_426_400_4);
    assert!((r - expected).abs() < 1.0e-10);
}

// 14. Lambert W₀(e) = 1.
#[test]
fn e2e_lambert_w0_of_e() {
    let r = lambert_w0(std::f64::consts::E).expect("ok");
    assert!((r - 1.0).abs() < 1.0e-10);
}

// 15. Elliptic K(0) = π/2.
#[test]
fn e2e_elliptic_k_zero() {
    let r = elliptic_k(0.0).expect("ok");
    assert!((r - PI / 2.0).abs() < 1.0e-12);
}

// 16. Elliptic E(0) = π/2 and E(1) = 1.
#[test]
fn e2e_elliptic_e_endpoints() {
    let z0 = elliptic_e(0.0).expect("ok");
    let z1 = elliptic_e(1.0).expect("ok");
    assert!((z0 - PI / 2.0).abs() < 1.0e-12);
    assert!((z1 - 1.0).abs() < 1.0e-12);
}

// 17. ζ(2) = π²/6.
#[test]
fn e2e_zeta_two_basel() {
    let r = zeta(2.0).expect("ok");
    assert!((r - PI * PI / 6.0).abs() < 1.0e-8);
}

// 18. RK4 on y' = -y matches exp(-t) to 1e-4 with h = 0.01.
#[test]
fn e2e_rk4_exponential_decay() {
    let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
    let (t_arr, ys) = rk4(f, 0.0, 1.0, &[1.0], 0.01).expect("ok");
    for (t, yvec) in t_arr.iter().zip(ys.iter()) {
        assert!((yvec[0] - (-t).exp()).abs() < 1.0e-4);
    }
}

// 19. DOPRI5 adaptive step on harmonic oscillator conserves energy.
#[test]
fn e2e_dopri5_energy_conservation() {
    let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![y[1], -y[0]]) };
    let (_t, ys) = dopri5(f, 0.0, 8.0, &[1.0, 0.0], 0.1, 1.0e-10, 1.0e-12, 100_000).expect("ok");
    for yvec in ys.iter() {
        let e = yvec[0] * yvec[0] + yvec[1] * yvec[1];
        assert!((e - 1.0).abs() < 1.0e-6);
    }
}

// 20. Explicit Euler, Heun, BDF1/BDF2 all approach the analytic decay solution.
#[test]
fn e2e_euler_family_decay() {
    let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
    let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
    let (_t, ys) = explicit_euler(&f, 0.0, 1.0, &[1.0], 1.0e-4).expect("ok");
    let euler_last = ys.last().expect("non-empty")[0];
    assert!((euler_last - (-1.0_f64).exp()).abs() < 1.0e-3);
    let (_t2, ys2) = heun(&f, 0.0, 1.0, &[1.0], 0.01).expect("ok");
    let heun_last = ys2.last().expect("non-empty")[0];
    assert!((heun_last - (-1.0_f64).exp()).abs() < 1.0e-3);
    let (_t3, ys3) = bdf1(&f, &j, 0.0, 1.0, &[1.0], 0.001, 1.0e-12, 30).expect("ok");
    let bdf1_last = ys3.last().expect("non-empty")[0];
    assert!((bdf1_last - (-1.0_f64).exp()).abs() < 1.0e-2);
    let (_t4, ys4) = bdf2(&f, &j, 0.0, 1.0, &[1.0], 0.001, 1.0e-12, 30).expect("ok");
    let bdf2_last = ys4.last().expect("non-empty")[0];
    assert!((bdf2_last - (-1.0_f64).exp()).abs() < 1.0e-3);
}

// 21. Rosenbrock-W and IMEX Euler also handle linear decay.
#[test]
fn e2e_rosenbrock_imex_decay() {
    let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
    let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
    let (_t, ys) = rosenbrock_w(&f, &j, 0.0, 1.0, &[1.0], 0.01).expect("ok");
    let last = ys.last().expect("non-empty")[0];
    assert!((last - (-1.0_f64).exp()).abs() < 1.0e-3);
    let s_stiff = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
    let s_non = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![0.0]) };
    let (_t, ys) =
        imex_euler(s_stiff, s_non, &j, 0.0, 1.0, &[1.0], 0.005, 1.0e-12, 30).expect("ok");
    let last = ys.last().expect("non-empty")[0];
    assert!((last - (-1.0_f64).exp()).abs() < 1.0e-2);
}

// 22. Cubic spline through (0,0)(1,1)(2,8)(3,27) at 1.5 is approximately 3.375.
#[test]
fn e2e_cubic_spline_cubic_data() {
    let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
    let ys = vec![0.0_f64, 1.0, 8.0, 27.0];
    let spl = natural_cubic_spline(&xs, &ys).expect("ok");
    let v = spline_eval(&spl, 1.5).expect("ok");
    // natural cubic spline boundary effect → relaxed tolerance
    assert!((v - 3.375).abs() < 0.5);
}

// 23. PCHIP preserves monotonicity on [0, 1, 2, 4, 8].
#[test]
fn e2e_pchip_monotone() {
    let xs = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
    let ys = vec![0.0_f64, 1.0, 2.0, 4.0, 8.0];
    let mut prev = pchip_interpolate(&xs, &ys, 0.0).expect("ok");
    let mut t = 0.0_f64;
    while t <= 8.0 {
        let v = pchip_interpolate(&xs, &ys, t).expect("ok");
        assert!(v >= prev - 1.0e-12);
        prev = v;
        t += 0.05;
    }
}

// 24. Durand-Kerner finds all roots of (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6.
#[test]
fn e2e_durand_kerner_cubic() {
    let coeffs = vec![-6.0_f64, 11.0, -6.0, 1.0];
    let roots = durand_kerner(&coeffs, 1.0e-10, 400).expect("ok");
    let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
    assert!((reals[0] - 1.0).abs() < 1.0e-4);
    assert!((reals[1] - 2.0).abs() < 1.0e-4);
    assert!((reals[2] - 3.0).abs() < 1.0e-4);
}

// 25. Sobol sequence's first dimension matches van-der-Corput-base-2.
#[test]
fn e2e_sobol_first_dim_vdc() {
    let p1 = sobol_point(0, 1).expect("ok");
    let p2 = sobol_point(1, 1).expect("ok");
    let p3 = sobol_point(2, 1).expect("ok");
    assert!((p1[0] - 0.5).abs() < 1.0e-12);
    assert!((p2[0] - 0.25).abs() < 1.0e-12);
    assert!((p3[0] - 0.75).abs() < 1.0e-12);
}

// 26. Aberth method finds roots of x³ - 6x² + 11x - 6.
#[test]
fn e2e_aberth_cubic() {
    let coeffs = vec![-6.0_f64, 11.0, -6.0, 1.0];
    let roots = aberth_all_roots(&coeffs, 1.0e-10, 300).expect("ok");
    let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
    assert!((reals[0] - 1.0).abs() < 1.0e-6);
    assert!((reals[1] - 2.0).abs() < 1.0e-6);
    assert!((reals[2] - 3.0).abs() < 1.0e-6);
}

// 27. Horner evaluation matches direct.
#[test]
fn e2e_horner_matches_direct() {
    let p = vec![1.0_f64, 2.0, 3.0, 4.0];
    let x = 1.5_f64;
    let r = horner(&p, x).expect("ok");
    let direct = p[0] + p[1] * x + p[2] * x * x + p[3] * x * x * x;
    assert!((r - direct).abs() < 1.0e-12);
}

// 28. Jenkins-Traub finds the real roots of (x-1)(x-2)(x-3).
#[test]
fn e2e_jenkins_traub_real_roots() {
    let p = vec![-6.0_f64, 11.0, -6.0, 1.0];
    let mut roots = jenkins_traub_real(&p, 1.0e-10, 200).expect("ok");
    roots.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
    assert!((roots[0] - 1.0).abs() < 1.0e-6);
    assert!((roots[1] - 2.0).abs() < 1.0e-6);
    assert!((roots[2] - 3.0).abs() < 1.0e-6);
}

// 29. Central + Richardson agree on f'(x) = cos(x) at π/3 → 0.5.
#[test]
fn e2e_central_richardson_agree() {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
    let d1 = central_difference(&f, std::f64::consts::FRAC_PI_3, 1.0e-5).expect("ok");
    let d2 = richardson_derivative(&f, std::f64::consts::FRAC_PI_3, 0.05, 4).expect("ok");
    assert!((d1 - 0.5).abs() < 1.0e-6);
    assert!((d2 - 0.5).abs() < 1.0e-10);
}

// 30. Complex-step derivative gives machine precision for sin.
#[test]
fn e2e_complex_step_sin() {
    let f = |z: CDual| -> NumericResult<CDual> { Ok(z.sin()) };
    let d = complex_step_derivative(f, std::f64::consts::FRAC_PI_3, 1.0e-30).expect("ok");
    assert!((d - 0.5).abs() < 1.0e-12);
}

// 31. Linear / Lagrange / barycentric / Hermite all agree at midpoint of (0,0),(1,1),(2,4).
#[test]
fn e2e_interp_agreement() {
    let xs = vec![0.0_f64, 1.0, 2.0];
    let ys = vec![0.0_f64, 1.0, 4.0];
    let v_lin = linear_interpolate(&xs, &ys, 1.5).expect("ok");
    let v_lag = lagrange_interpolate(&xs, &ys, 1.5).expect("ok");
    let ws = barycentric_weights(&xs).expect("ok");
    let v_bary = barycentric_eval(&xs, &ys, &ws, 1.5).expect("ok");
    let v_h = hermite_interpolate(1.0, 2.0, 1.0, 4.0, 2.0, 4.0, 1.5).expect("ok");
    // linear should give 2.5 (midpoint of 1 and 4); Lagrange = 2.25 (= 1.5²)
    assert!((v_lin - 2.5).abs() < 1.0e-12);
    assert!((v_lag - 2.25).abs() < 1.0e-12);
    assert!((v_bary - 2.25).abs() < 1.0e-12);
    // Hermite cubic with v0=1,v1=4,dv0=2,dv1=4 over [1,2] at 1.5 should match 1.5²=2.25
    assert!((v_h - 2.25).abs() < 1.0e-12);
}

// 32. Akima passes through nodes.
#[test]
fn e2e_akima_passes_through_nodes() {
    let xs = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let ys = vec![0.0_f64, 1.0, 4.0, 9.0, 16.0];
    for (x, y) in xs.iter().zip(ys.iter()) {
        let v = akima_interpolate(&xs, &ys, *x).expect("ok");
        assert!((v - y).abs() < 1.0e-10);
    }
}

// 33. Monte Carlo on unit disc area ≈ π.
#[test]
fn e2e_mc_circle_area() {
    let f = |x: &[f64]| -> NumericResult<f64> {
        Ok(if x[0] * x[0] + x[1] * x[1] < 1.0 {
            1.0
        } else {
            0.0
        })
    };
    let mut rng = LcgRng::new(7);
    let (v, _err) =
        monte_carlo_integrate(f, &[-1.0, -1.0], &[1.0, 1.0], 80_000, &mut rng).expect("ok");
    assert!((v - PI).abs() < 0.05);
}

// 34. Tensor-product Gauss exact for product polynomials.
#[test]
fn e2e_tp_gauss_polynomial() {
    let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0].powi(3) * x[1].powi(3)) };
    let v = tensor_product_gauss_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 4).expect("ok");
    assert!((v - 1.0 / 16.0).abs() < 1.0e-10);
}

// 35. Genz-Malik converges on smooth 2-D integrand.
#[test]
fn e2e_gm_smooth_2d() {
    let f = |x: &[f64]| -> NumericResult<f64> { Ok((x[0] + x[1]).exp()) };
    let v = genz_malik_cubature(f, &[0.0, 0.0], &[1.0, 1.0], 1.0e-6, 200).expect("ok");
    // ∫_0^1 ∫_0^1 e^{x+y} = (e - 1)²
    let exact = (std::f64::consts::E - 1.0).powi(2);
    assert!((v - exact).abs() < 1.0e-3);
}

// 36. Sobol QMC gives accurate quadrature.
#[test]
fn e2e_sobol_gives_accurate_integration() {
    let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] * x[0] + x[1] * x[1]) };
    // ∫_0^1 ∫_0^1 (x² + y²) dx dy = 2/3
    let v = sobol_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 4096).expect("ok");
    assert!((v - 2.0 / 3.0).abs() < 5.0e-3);
}

// 37. bessel_jn(3, 5) consistency check.
#[test]
fn e2e_bessel_jn_consistency() {
    let v = bessel_jn(3, 5.0).expect("ok");
    assert!((v - 0.364_831_230_613_667).abs() < 1.0e-4);
}

// 38. All 7 PTX kernels produce non-empty strings on all 6 SM versions.
#[test]
fn e2e_ptx_kernels_all_sm() {
    type KernelFn = fn(u32) -> String;
    let kernels: [(&str, KernelFn); 7] = [
        ("horner_eval", horner_eval_ptx),
        ("rk4_stage", rk4_stage_ptx),
        ("bisection_step", bisection_step_ptx),
        ("gauss_quad_accumulate", gauss_quad_accumulate_ptx),
        ("spline_eval", spline_eval_ptx),
        ("central_diff", central_diff_ptx),
        ("bessel_recurrence", bessel_recurrence_ptx),
    ];
    for sm in [75_u32, 80, 86, 89, 90, 100] {
        for (name, k) in kernels.iter() {
            let s = k(sm);
            assert!(!s.is_empty(), "kernel {name} sm={sm} was empty");
            assert!(s.contains(".visible .entry"));
        }
    }
}
