use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_pinn::handle::LcgRng;
use oxicuda_pinn::prelude::*;

// ─── PTX kernel benchmarks ───────────────────────────────────────────────────

fn bench_pinn_residual_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("pinn_residual_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(pinn_residual_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_spectral_conv_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("spectral_conv_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(spectral_conv_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_dual_op_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("dual_op_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(dual_op_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_adjoint_ode_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("adjoint_ode_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(adjoint_ode_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_branch_trunk_dot_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("branch_trunk_dot_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(branch_trunk_dot_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_siren_forward_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("siren_forward_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(siren_forward_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_lhs_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("lhs_sample_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(lhs_sample_ptx(sm)))
        });
    }
    g.finish();
}

// ─── Algorithm benchmarks ────────────────────────────────────────────────────

fn bench_rk4_step_d64(c: &mut Criterion) {
    fn linear_decay(_t: f32, y: &[f32], dy: &mut [f32]) {
        for (dyi, &yi) in dy.iter_mut().zip(y.iter()) {
            *dyi = -yi;
        }
    }
    let y0: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    c.bench_function("rk4_step_d64", |b| {
        b.iter(|| {
            std::hint::black_box(rk4_step(
                &linear_decay,
                std::hint::black_box(0.0_f32),
                std::hint::black_box(&y0),
                std::hint::black_box(0.01_f32),
            ))
        })
    });
}

fn bench_dopri45_step_d32(c: &mut Criterion) {
    fn linear_decay(_t: f32, y: &[f32], dy: &mut [f32]) {
        for (dyi, &yi) in dy.iter_mut().zip(y.iter()) {
            *dyi = -yi;
        }
    }
    let y0: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    c.bench_function("dopri45_step_d32", |b| {
        b.iter(|| {
            std::hint::black_box(dopri45_step(
                &linear_decay,
                std::hint::black_box(0.0_f32),
                std::hint::black_box(&y0),
                std::hint::black_box(0.01_f32),
            ))
        })
    });
}

fn bench_fno1d_forward_n32(c: &mut Criterion) {
    let mut rng = LcgRng::new(1);
    let cfg = Fno1dConfig {
        d_in: 1,
        d_out: 1,
        width: 16,
        k_max: 8,
        n_blocks: 2,
    };
    let fno = Fno1d::new(cfg, &mut rng);
    let input = vec![0.5_f32; 32];
    c.bench_function("fno1d_forward_n32", |b| {
        b.iter(|| {
            std::hint::black_box(
                fno.forward(std::hint::black_box(&input), std::hint::black_box(32))
                    .unwrap(),
            )
        })
    });
}

fn bench_dft_n32(c: &mut Criterion) {
    let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin()).collect();
    c.bench_function("dft_n32", |b| {
        b.iter(|| std::hint::black_box(dft_1d(std::hint::black_box(&x))))
    });
}

fn bench_lhs_sample_d4_n256(c: &mut Criterion) {
    c.bench_function("lhs_sample_d4_n256", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(42);
            std::hint::black_box(latin_hypercube_sample(
                std::hint::black_box(256),
                std::hint::black_box(4),
                &mut rng,
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_pinn_residual_ptx,
    bench_spectral_conv_ptx,
    bench_dual_op_ptx,
    bench_adjoint_ode_ptx,
    bench_branch_trunk_dot_ptx,
    bench_siren_forward_ptx,
    bench_lhs_ptx,
    bench_rk4_step_d64,
    bench_dopri45_step_d32,
    bench_fno1d_forward_n32,
    bench_dft_n32,
    bench_lhs_sample_d4_n256,
);
criterion_main!(benches);
