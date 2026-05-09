use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_continual::handle::LcgRng;
use oxicuda_continual::prelude::*;

// ─── PTX kernel benchmarks ───────────────────────────────────────────────────

fn bench_ewc_penalty_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ewc_penalty_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(ewc_penalty_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_fisher_diag_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("fisher_diag_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(fisher_diag_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gradient_project_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gradient_project_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gradient_project_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_mask_apply_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("mask_apply_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(mask_apply_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_si_omega_update_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("si_omega_update_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(si_omega_update_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_logit_distill_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("logit_distill_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(logit_distill_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_replay_sample_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("replay_sample_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(replay_sample_ptx(sm)))
        });
    }
    g.finish();
}

// ─── Algorithm benchmarks ────────────────────────────────────────────────────

fn bench_ewc_loss_d1024(c: &mut Criterion) {
    let d = 1024;
    let mut rng = LcgRng::new(0);
    let mut params = vec![0.0_f32; d];
    rng.fill_normal(&mut params);
    let mut anchor = vec![0.0_f32; d];
    rng.fill_normal(&mut anchor);
    let mut fisher_params = vec![0.0_f32; d];
    rng.fill_normal(&mut fisher_params);
    for v in &mut fisher_params {
        *v = v.abs();
    }
    let fisher = FisherDiag {
        params: fisher_params,
    };
    let mut reg = EwcRegularizer::new();
    ewc_add_task(&mut reg, anchor, fisher);
    let cfg = EwcConfig {
        lambda: 1.0,
        n_tasks: 10,
    };
    c.bench_function("ewc_loss_d1024", |b| {
        b.iter(|| {
            std::hint::black_box(
                ewc_loss(
                    std::hint::black_box(&params),
                    std::hint::black_box(&reg),
                    std::hint::black_box(&cfg),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_fisher_diag_accumulate(c: &mut Criterion) {
    let d = 1024;
    let n_samples = 32;
    let mut rng = LcgRng::new(1);
    let mut gradients = vec![0.0_f32; d * n_samples];
    rng.fill_normal(&mut gradients);
    c.bench_function("fisher_diag_accumulate_d1024_n32", |b| {
        b.iter(|| {
            std::hint::black_box(
                compute_fisher_empirical(
                    std::hint::black_box(&gradients),
                    std::hint::black_box(n_samples),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_gem_project_d512(c: &mut Criterion) {
    let d = 512;
    let n_tasks = 8;
    let mut rng = LcgRng::new(2);
    let mut grad = vec![0.0_f32; d];
    rng.fill_normal(&mut grad);
    let mut mem_grads = Vec::with_capacity(n_tasks);
    for _ in 0..n_tasks {
        let mut mg = vec![0.0_f32; d];
        rng.fill_normal(&mut mg);
        mem_grads.push(mg);
    }
    c.bench_function("gem_project_d512_k8", |b| {
        b.iter(|| {
            std::hint::black_box(
                gem_project_gradient(
                    std::hint::black_box(&grad),
                    std::hint::black_box(&mem_grads),
                    std::hint::black_box(0.0),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_er_sample_b32(c: &mut Criterion) {
    let mut rng = LcgRng::new(3);
    let mut buf = er_buffer_new(1024).unwrap();
    for i in 0..1024_usize {
        er_add(&mut buf, vec![i as f32; 32], (i % 10) as u32, &mut rng);
    }
    c.bench_function("er_sample_b32_cap1024", |b| {
        b.iter(|| {
            let mut sample_rng = LcgRng::new(99);
            std::hint::black_box(
                er_sample_batch(
                    std::hint::black_box(&buf),
                    std::hint::black_box(32),
                    std::hint::black_box(&mut sample_rng),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_packnet_prune_d1024(c: &mut Criterion) {
    let d = 1024;
    let mut rng = LcgRng::new(4);
    let mut weights = vec![0.0_f32; d];
    rng.fill_normal(&mut weights);
    c.bench_function("packnet_prune_d1024_s05", |b| {
        b.iter(|| {
            std::hint::black_box(
                prune_weights_l1(
                    std::hint::black_box(&weights),
                    std::hint::black_box(0.5),
                    std::hint::black_box(0),
                )
                .unwrap(),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_ewc_penalty_ptx,
    bench_fisher_diag_ptx,
    bench_gradient_project_ptx,
    bench_mask_apply_ptx,
    bench_si_omega_update_ptx,
    bench_logit_distill_ptx,
    bench_replay_sample_ptx,
    bench_ewc_loss_d1024,
    bench_fisher_diag_accumulate,
    bench_gem_project_d512,
    bench_er_sample_b32,
    bench_packnet_prune_d1024,
);
criterion_main!(benches);
