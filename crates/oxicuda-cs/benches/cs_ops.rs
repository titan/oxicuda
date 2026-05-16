#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_cs::amp::amp;
use oxicuda_cs::basis_pursuit::basis_pursuit;
use oxicuda_cs::greedy::{cosamp, omp};
use oxicuda_cs::handle::LcgRng;
use oxicuda_cs::lasso::{coord_descent_lasso, fista_lasso};
use oxicuda_cs::matrix_completion::svt;
use oxicuda_cs::measurement::gaussian_matrix;
use oxicuda_cs::ptx_kernels::{
    amp_onsager_ptx, correlate_ptx, hard_threshold_ptx, iht_step_ptx, soft_threshold_ptx,
    svt_threshold_ptx, tv_grad_ptx,
};
use oxicuda_cs::thresholding::iht;
use oxicuda_cs::tv::tv_1d_chambolle;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("correlate", correlate_ptx),
        ("hard_threshold", hard_threshold_ptx),
        ("soft_threshold", soft_threshold_ptx),
        ("iht_step", iht_step_ptx),
        ("amp_onsager", amp_onsager_ptx),
        ("svt_threshold", svt_threshold_ptx),
        ("tv_grad", tv_grad_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn build_problem(seed: u64, m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
    let mut x = vec![0.0_f64; n];
    for i in 0..k {
        let idx = (i * 13 + 7) % n;
        x[idx] = if i % 2 == 0 { 1.0 } else { -0.6 };
    }
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        for j in 0..n {
            y[i] += phi[i * n + j] * x[j];
        }
    }
    (phi, x, y)
}

fn bench_omp(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(11, 20, 50, 3);
    c.bench_function("omp_m20_n50_k3", |b| {
        b.iter(|| omp(&phi, 20, 50, &y, 3, 1.0e-7).expect("ok"))
    });
}

fn bench_cosamp(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(13, 20, 50, 3);
    c.bench_function("cosamp_m20_n50_k3", |b| {
        b.iter(|| cosamp(&phi, 20, 50, &y, 3, 50, 1.0e-7).expect("ok"))
    });
}

fn bench_iht(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(17, 24, 48, 3);
    c.bench_function("iht_m24_n48_k3", |b| {
        b.iter(|| iht(&phi, 24, 48, &y, 3, 1.0, 200, 1.0e-9).expect("ok"))
    });
}

fn bench_amp(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(19, 24, 48, 3);
    c.bench_function("amp_m24_n48", |b| {
        b.iter(|| amp(&phi, 24, 48, &y, 1.4, 50, 1.0e-9).expect("ok"))
    });
}

fn bench_basis_pursuit(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(23, 20, 32, 3);
    c.bench_function("basis_pursuit_m20_n32", |b| {
        b.iter(|| basis_pursuit(&phi, 20, 32, &y, 2.0, 100, 1.0e-6).expect("ok"))
    });
}

fn bench_cd_lasso(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(29, 16, 24, 2);
    c.bench_function("cd_lasso_m16_n24", |b| {
        b.iter(|| coord_descent_lasso(&phi, 16, 24, &y, 0.05, None, 200, 1.0e-9).expect("ok"))
    });
}

fn bench_fista_lasso(c: &mut Criterion) {
    let (phi, _x, y) = build_problem(31, 16, 24, 2);
    c.bench_function("fista_lasso_m16_n24", |b| {
        b.iter(|| fista_lasso(&phi, 16, 24, &y, 0.05, None, 200, 1.0e-9).expect("ok"))
    });
}

fn bench_tv(c: &mut Criterion) {
    let mut rng = LcgRng::new(37);
    let y: Vec<f64> = (0..128)
        .map(|i| if i < 64 { 1.0 } else { 5.0 } + 0.1 * rng.next_normal())
        .collect();
    c.bench_function("tv_1d_n128", |b| {
        b.iter(|| tv_1d_chambolle(&y, 0.3, 200, 1.0e-9).expect("ok"))
    });
}

fn bench_svt(c: &mut Criterion) {
    let m = vec![
        1.0_f64, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0,
    ];
    let mask = vec![true; 16];
    c.bench_function("svt_4x4", |b| {
        b.iter(|| svt(&m, &mask, 4, 4, 0.5, 1.5, 80, 1.0e-7).expect("ok"))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_omp,
    bench_cosamp,
    bench_iht,
    bench_amp,
    bench_basis_pursuit,
    bench_cd_lasso,
    bench_fista_lasso,
    bench_tv,
    bench_svt
);
criterion_main!(benches);
