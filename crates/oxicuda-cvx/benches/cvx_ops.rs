#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_cvx::error::CvxResult;
use oxicuda_cvx::handle::LcgRng;
use oxicuda_cvx::lp::revised_simplex;
use oxicuda_cvx::projection::{project_l1_ball, project_psd_cone, project_simplex};
use oxicuda_cvx::prox_ops::{prox_l1, prox_tv_1d};
use oxicuda_cvx::proximal::fista;
use oxicuda_cvx::ptx_kernels::{
    admm_dual_update_ptx, axpy_ptx, fista_extrapolate_ptx, gradient_step_ptx, proj_l2_ball_ptx,
    simplex_proj_ptx, soft_threshold_ptx,
};

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("axpy", axpy_ptx),
        ("soft_threshold", soft_threshold_ptx),
        ("simplex_proj", simplex_proj_ptx),
        ("gradient_step", gradient_step_ptx),
        ("fista_extrapolate", fista_extrapolate_ptx),
        ("admm_dual_update", admm_dual_update_ptx),
        ("proj_l2_ball", proj_l2_ball_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_prox_l1(c: &mut Criterion) {
    let mut rng = LcgRng::new(11);
    let v: Vec<f64> = (0..1024).map(|_| rng.next_normal()).collect();
    c.bench_function("prox_l1_n1024", |b| {
        b.iter(|| prox_l1(&v, 0.1).expect("ok"))
    });
}

fn bench_prox_tv(c: &mut Criterion) {
    let mut rng = LcgRng::new(13);
    let y: Vec<f64> = (0..512).map(|_| rng.next_normal()).collect();
    c.bench_function("prox_tv_1d_n512", |b| {
        b.iter(|| prox_tv_1d(&y, 0.5).expect("ok"))
    });
}

fn bench_simplex_proj(c: &mut Criterion) {
    let mut rng = LcgRng::new(17);
    let v: Vec<f64> = (0..512).map(|_| rng.next_normal()).collect();
    c.bench_function("simplex_proj_n512", |b| {
        b.iter(|| project_simplex(&v, 1.0).expect("ok"))
    });
}

fn bench_l1_ball_proj(c: &mut Criterion) {
    let mut rng = LcgRng::new(19);
    let v: Vec<f64> = (0..512).map(|_| rng.next_normal()).collect();
    c.bench_function("l1_ball_proj_n512", |b| {
        b.iter(|| project_l1_ball(&v, 1.0).expect("ok"))
    });
}

fn bench_psd_proj(c: &mut Criterion) {
    let mut rng = LcgRng::new(23);
    let n = 8usize;
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = rng.next_normal();
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    c.bench_function("psd_proj_n8", |b| {
        b.iter(|| project_psd_cone(&m, n).expect("ok"))
    });
}

fn bench_lp_simplex(c: &mut Criterion) {
    let a = vec![1.0_f64, 1.0, 1.0];
    let b = vec![1.0_f64];
    let cc = vec![-1.0_f64, -1.0, 0.0];
    let basis = vec![2usize];
    c.bench_function("lp_simplex_2x3", |b_iter| {
        b_iter.iter(|| revised_simplex(&a, 1, 3, &b, &cc, &basis, 100).expect("ok"))
    });
}

fn bench_fista(c: &mut Criterion) {
    let b = vec![3.0_f64, -2.0, 0.5, 1.0, -1.0];
    let f = move |x: &[f64]| -> CvxResult<f64> {
        Ok(x.iter()
            .zip(b.iter())
            .map(|(xi, bi)| 0.5 * (xi - bi).powi(2))
            .sum())
    };
    let b2 = vec![3.0_f64, -2.0, 0.5, 1.0, -1.0];
    let g = move |x: &[f64]| -> CvxResult<Vec<f64>> {
        Ok(x.iter().zip(b2.iter()).map(|(xi, bi)| xi - bi).collect())
    };
    let p = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, s) };
    c.bench_function("fista_lasso_n5", |b_iter| {
        b_iter.iter(|| {
            fista(
                &[0.0, 0.0, 0.0, 0.0, 0.0],
                &f,
                &g,
                &p,
                1.0,
                100,
                1.0e-9,
                false,
            )
            .expect("ok")
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_prox_l1,
    bench_prox_tv,
    bench_simplex_proj,
    bench_l1_ball_proj,
    bench_psd_proj,
    bench_lp_simplex,
    bench_fista
);
criterion_main!(benches);
