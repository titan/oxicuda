use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_ssl::handle::LcgRng;
use oxicuda_ssl::prelude::*;

fn bench_nt_xent_softmax_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("nt_xent_softmax_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(nt_xent_softmax_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_momentum_update_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("momentum_update_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(momentum_update_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_byol_cosine_loss_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("byol_cosine_loss_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(byol_cosine_loss_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_barlow_cross_corr_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("barlow_cross_corr_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(barlow_cross_corr_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_random_mask_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("random_mask_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(random_mask_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_cosine_similarity_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("cosine_similarity_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(cosine_similarity_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gather_features_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gather_features_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gather_features_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_simclr_loss(c: &mut Criterion) {
    let n = 64;
    let d = 128;
    let mut rng = LcgRng::new(0);
    let mut z_a = vec![0.0_f32; n * d];
    let mut z_b = vec![0.0_f32; n * d];
    rng.fill_normal(&mut z_a);
    rng.fill_normal(&mut z_b);
    let cfg = SimClrConfig::default();
    c.bench_function("simclr_loss_b64_d128", |b| {
        b.iter(|| {
            std::hint::black_box(
                simclr_loss(
                    std::hint::black_box(&z_a),
                    std::hint::black_box(&z_b),
                    n,
                    d,
                    &cfg,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_moco_loss(c: &mut Criterion) {
    let n = 16;
    let d = 64;
    let queue_size = 256;
    let mut rng = LcgRng::new(1);
    let mut q_buf = vec![0.0_f32; n * d];
    let mut k_buf = vec![0.0_f32; n * d];
    rng.fill_normal(&mut q_buf);
    rng.fill_normal(&mut k_buf);
    let mut queue = MocoQueue::new(queue_size, d).expect("ok");
    let mut neg = vec![0.0_f32; queue_size * d];
    rng.fill_normal(&mut neg);
    queue.enqueue(&neg).expect("ok");
    c.bench_function("moco_loss_b16_d64_q256", |b| {
        b.iter(|| {
            std::hint::black_box(
                moco_loss(
                    std::hint::black_box(&q_buf),
                    std::hint::black_box(&k_buf),
                    n,
                    d,
                    &queue,
                    0.07,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_barlow_cross_corr(c: &mut Criterion) {
    let n = 256;
    let d = 64;
    let mut rng = LcgRng::new(2);
    let mut z_a = vec![0.0_f32; n * d];
    let mut z_b = vec![0.0_f32; n * d];
    rng.fill_normal(&mut z_a);
    rng.fill_normal(&mut z_b);
    let cfg = BarlowTwinsConfig::default();
    c.bench_function("barlow_loss_b256_d64", |b| {
        b.iter(|| {
            std::hint::black_box(
                barlow_twins_loss(
                    std::hint::black_box(&z_a),
                    std::hint::black_box(&z_b),
                    n,
                    d,
                    &cfg,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_mae_mask(c: &mut Criterion) {
    let mut rng = LcgRng::new(3);
    c.bench_function("mae_mask_p196_r075", |b| {
        b.iter(|| std::hint::black_box(random_patch_mask(196, 0.75, &mut rng).expect("ok")))
    });
}

fn bench_dino_loss(c: &mut Criterion) {
    let n = 64;
    let k = 128;
    let mut rng = LcgRng::new(4);
    let mut s = vec![0.0_f32; n * k];
    let mut t = vec![0.0_f32; n * k];
    rng.fill_normal(&mut s);
    rng.fill_normal(&mut t);
    let centre = vec![0.0_f32; k];
    let cfg = DinoConfig::default();
    c.bench_function("dino_loss_b64_k128", |b| {
        b.iter(|| {
            std::hint::black_box(
                dino_loss(
                    std::hint::black_box(&s),
                    std::hint::black_box(&t),
                    std::hint::black_box(&centre),
                    n,
                    k,
                    &cfg,
                )
                .expect("ok"),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_nt_xent_softmax_ptx,
    bench_momentum_update_ptx,
    bench_byol_cosine_loss_ptx,
    bench_barlow_cross_corr_ptx,
    bench_random_mask_ptx,
    bench_cosine_similarity_ptx,
    bench_gather_features_ptx,
    bench_simclr_loss,
    bench_moco_loss,
    bench_barlow_cross_corr,
    bench_mae_mask,
    bench_dino_loss,
);
criterion_main!(benches);
