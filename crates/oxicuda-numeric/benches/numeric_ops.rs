//! Benches for `oxicuda-numeric`.

#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_numeric::error::NumericResult;
use oxicuda_numeric::handle::LcgRng;
use oxicuda_numeric::interp::cubic_spline::{natural_cubic_spline, spline_eval};
use oxicuda_numeric::ode::rk4::rk4;
use oxicuda_numeric::poly::durand_kerner::durand_kerner;
use oxicuda_numeric::poly::horner_eval::horner;
use oxicuda_numeric::ptx_kernels::{
    bessel_recurrence_ptx, bisection_step_ptx, central_diff_ptx, gauss_quad_accumulate_ptx,
    horner_eval_ptx, rk4_stage_ptx, spline_eval_ptx,
};
use oxicuda_numeric::quadrature::adaptive_simpson::adaptive_simpson;
use oxicuda_numeric::quadrature::gauss_legendre::gauss_legendre_integrate;
use oxicuda_numeric::quadrature::romberg::romberg;
use oxicuda_numeric::root::brent::brent;
use oxicuda_numeric::root::newton::newton;
use oxicuda_numeric::special::bessel_jy::bessel_j0;
use oxicuda_numeric::special::elliptic_ke::elliptic_k;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75_u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("horner_eval", horner_eval_ptx),
        ("rk4_stage", rk4_stage_ptx),
        ("bisection_step", bisection_step_ptx),
        ("gauss_quad_accumulate", gauss_quad_accumulate_ptx),
        ("spline_eval", spline_eval_ptx),
        ("central_diff", central_diff_ptx),
        ("bessel_recurrence", bessel_recurrence_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_horner(c: &mut Criterion) {
    let mut rng = LcgRng::new(11);
    let coeffs: Vec<f64> = (0..32).map(|_| rng.next_normal()).collect();
    c.bench_function("horner_n32", |b| {
        b.iter(|| horner(&coeffs, 1.234).expect("ok"))
    });
}

fn bench_root_newton(c: &mut Criterion) {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3) - 2.0) };
    let g = |x: f64| -> NumericResult<f64> { Ok(3.0 * x * x) };
    c.bench_function("newton_cube_root_two", |b| {
        b.iter(|| newton(&f, &g, 1.0, 1.0e-12, 30).expect("ok"))
    });
}

fn bench_root_brent(c: &mut Criterion) {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
    c.bench_function("brent_sin_pi", |b| {
        b.iter(|| brent(&f, 3.0, 4.0, 1.0e-12, 50).expect("ok"))
    });
}

fn bench_romberg(c: &mut Criterion) {
    let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
    c.bench_function("romberg_arctan", |b| {
        b.iter(|| romberg(&f, 0.0, 1.0, 1.0e-10, 10).expect("ok"))
    });
}

fn bench_gauss_legendre(c: &mut Criterion) {
    let f = |x: f64| -> NumericResult<f64> { Ok(x.exp()) };
    c.bench_function("gauss_legendre_n8", |b| {
        b.iter(|| gauss_legendre_integrate(&f, 0.0, 1.0, 8).expect("ok"))
    });
}

fn bench_adaptive_simpson(c: &mut Criterion) {
    let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
    c.bench_function("adaptive_simpson_arctan", |b| {
        b.iter(|| adaptive_simpson(&f, 0.0, 1.0, 1.0e-8, 30).expect("ok"))
    });
}

fn bench_rk4_decay(c: &mut Criterion) {
    let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
    c.bench_function("rk4_decay_1000_steps", |b| {
        b.iter(|| rk4(&f, 0.0, 1.0, &[1.0], 0.001).expect("ok"))
    });
}

fn bench_special_bessel(c: &mut Criterion) {
    c.bench_function("bessel_j0_eval", |b| b.iter(|| bessel_j0(5.0)));
    c.bench_function("elliptic_k_half", |b| {
        b.iter(|| elliptic_k(0.5).expect("ok"))
    });
}

fn bench_durand_kerner(c: &mut Criterion) {
    let coeffs = vec![-6.0_f64, 11.0, -6.0, 1.0];
    c.bench_function("durand_kerner_cubic", |b| {
        b.iter(|| durand_kerner(&coeffs, 1.0e-10, 200).expect("ok"))
    });
}

fn bench_cubic_spline(c: &mut Criterion) {
    let mut rng = LcgRng::new(2);
    let n = 64_usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let ys: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    c.bench_function("cubic_spline_build_n64", |b| {
        b.iter(|| natural_cubic_spline(&xs, &ys).expect("ok"))
    });
    let spl = natural_cubic_spline(&xs, &ys).expect("ok");
    c.bench_function("cubic_spline_eval", |b| {
        b.iter(|| spline_eval(&spl, 12.34).expect("ok"))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_horner,
    bench_root_newton,
    bench_root_brent,
    bench_romberg,
    bench_gauss_legendre,
    bench_adaptive_simpson,
    bench_rk4_decay,
    bench_special_bessel,
    bench_durand_kerner,
    bench_cubic_spline,
);
criterion_main!(benches);
