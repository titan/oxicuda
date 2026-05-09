use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_adversarial::handle::LcgRng;
use oxicuda_adversarial::prelude::*;

fn bench_fgsm_step_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("fgsm_step_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(fgsm_step_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_pgd_proj_l_inf_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("pgd_proj_l_inf_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(pgd_proj_l_inf_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_pgd_proj_l2_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("pgd_proj_l2_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(pgd_proj_l2_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_smoothing_noise_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("smoothing_noise_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(smoothing_noise_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_grad_sign_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("grad_sign_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(grad_sign_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_certified_radius_reduce_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("certified_radius_reduce_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(certified_radius_reduce_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_attack_loss_grad_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("attack_loss_grad_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(attack_loss_grad_ptx(sm)))
        });
    }
    g.finish();
}

/// Synthetic gradient generator: gradient = x − target.
fn make_quadratic_grad(
    target: Vec<f32>,
) -> impl Fn(&[f32]) -> oxicuda_adversarial::error::AdvResult<Vec<f32>> {
    move |x: &[f32]| Ok(x.iter().zip(target.iter()).map(|(a, b)| a - b).collect())
}

fn bench_fgsm_attack(c: &mut Criterion) {
    let dim = 1024;
    let target = vec![0.5_f32; dim];
    let x = vec![0.6_f32; dim];
    let grad = make_quadratic_grad(target);
    c.bench_function("fgsm_attack_d1024", |b| {
        b.iter(|| {
            std::hint::black_box(
                fgsm_attack(std::hint::black_box(&x), 0.05, 0.0, 1.0, &grad).expect("ok"),
            )
        })
    });
}

fn bench_pgd_l_inf_attack(c: &mut Criterion) {
    let dim = 512;
    let target = vec![0.5_f32; dim];
    let x = vec![0.6_f32; dim];
    let grad = make_quadratic_grad(target);
    let cfg = PgdConfig::new(0.05, 0.01, 10, false).expect("ok");
    let mut rng = LcgRng::new(0);
    c.bench_function("pgd_l_inf_attack_d512_n10", |b| {
        b.iter(|| {
            std::hint::black_box(
                pgd_attack_l_inf(std::hint::black_box(&x), 0.0, 1.0, &cfg, &mut rng, &grad)
                    .expect("ok"),
            )
        })
    });
}

fn bench_trades_loss(c: &mut Criterion) {
    let n = 64;
    let k = 10;
    let mut rng = LcgRng::new(1);
    let mut clean = vec![0.0_f32; n * k];
    let mut adv = vec![0.0_f32; n * k];
    rng.fill_normal(&mut clean);
    rng.fill_normal(&mut adv);
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let cfg = TradesConfig::new(6.0).expect("ok");
    c.bench_function("trades_loss_b64_k10", |b| {
        b.iter(|| {
            std::hint::black_box(
                trades_loss(
                    std::hint::black_box(&clean),
                    std::hint::black_box(&adv),
                    std::hint::black_box(&labels),
                    n,
                    k,
                    &cfg,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_ibp_propagate(c: &mut Criterion) {
    let in_dim = 64;
    let out_dim = 32;
    let mut rng = LcgRng::new(2);
    let mut w = vec![0.0_f32; in_dim * out_dim];
    let mut bias = vec![0.0_f32; out_dim];
    rng.fill_normal(&mut w);
    rng.fill_normal(&mut bias);
    let bounds_in: Vec<IntervalBound> = (0..in_dim)
        .map(|_| IntervalBound::new(-0.1, 0.1).expect("ok"))
        .collect();
    c.bench_function("ibp_propagate_64x32", |b| {
        b.iter(|| {
            std::hint::black_box(
                ibp_propagate(
                    std::hint::black_box(&bounds_in),
                    std::hint::black_box(&w),
                    std::hint::black_box(&bias),
                    in_dim,
                    out_dim,
                )
                .expect("ok"),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_fgsm_step_ptx,
    bench_pgd_proj_l_inf_ptx,
    bench_pgd_proj_l2_ptx,
    bench_smoothing_noise_ptx,
    bench_grad_sign_ptx,
    bench_certified_radius_reduce_ptx,
    bench_attack_loss_grad_ptx,
    bench_fgsm_attack,
    bench_pgd_l_inf_attack,
    bench_trades_loss,
    bench_ibp_propagate,
);
criterion_main!(benches);
