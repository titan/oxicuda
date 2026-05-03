//! Criterion benchmarks for `oxicuda-vision` PTX kernel generation and
//! CPU-reference forward passes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_vision::{
    clip::contrastive::info_nce_loss,
    handle::LcgRng,
    ptx_kernels::{
        adaptive_avg_pool_ptx, bilinear_interp_ptx, contrastive_loss_ptx, focal_loss_ptx,
        image_normalize_ptx, patch_embed_ptx, roi_align_ptx,
    },
    vit::{ViTConfig, ViTModel},
};
use std::hint::black_box;

const SM_VERSIONS: &[u32] = &[75, 80, 90, 120];

// ─── PTX kernel generation benchmarks ────────────────────────────────────────

fn bench_patch_embed_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_embed_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| patch_embed_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_bilinear_interp_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("bilinear_interp_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| bilinear_interp_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_contrastive_loss_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("contrastive_loss_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| contrastive_loss_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_roi_align_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("roi_align_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| roi_align_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_image_normalize_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_normalize_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| image_normalize_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_adaptive_avg_pool_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_avg_pool_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| adaptive_avg_pool_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_focal_loss_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("focal_loss_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| focal_loss_ptx(black_box(sm)));
        });
    }
    group.finish();
}

// ─── Forward pass benchmarks ──────────────────────────────────────────────────

fn bench_vit_forward_tiny(c: &mut Criterion) {
    let cfg = ViTConfig::tiny();
    let mut rng = LcgRng::new(42);
    let model = ViTModel::new(cfg, &mut rng).expect("model ok");
    let image = vec![0.5f32; 3 * 32 * 32];

    c.bench_function("vit_tiny_forward", |b| {
        b.iter(|| model.forward(black_box(&image)).expect("forward ok"))
    });
}

fn bench_clip_info_nce(c: &mut Criterion) {
    let embed_dim = 64;
    let batch = 8;
    let mut rng = LcgRng::new(7);
    let mut img_e = vec![0.0f32; batch * embed_dim];
    let mut txt_e = vec![0.0f32; batch * embed_dim];
    rng.fill_normal(&mut img_e);
    rng.fill_normal(&mut txt_e);

    c.bench_function("clip_info_nce_b8_d64", |b| {
        b.iter(|| {
            info_nce_loss(black_box(&img_e), black_box(&txt_e), embed_dim, 0.07).expect("loss ok")
        })
    });
}

criterion_group!(
    benches,
    bench_patch_embed_ptx,
    bench_bilinear_interp_ptx,
    bench_contrastive_loss_ptx,
    bench_roi_align_ptx,
    bench_image_normalize_ptx,
    bench_adaptive_avg_pool_ptx,
    bench_focal_loss_ptx,
    bench_vit_forward_tiny,
    bench_clip_info_nce,
);
criterion_main!(benches);
