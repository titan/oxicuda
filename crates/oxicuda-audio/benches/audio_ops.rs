//! Criterion benchmarks for `oxicuda-audio` PTX kernel generation and
//! CPU-reference forward passes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_audio::{
    ctc::ctc_forward_log,
    encoder::{ConformerConfig, ConformerEncoder},
    handle::LcgRng,
    ptx_kernels::{
        ctc_alpha_ptx, depthwise_conv1d_ptx, dilated_conv1d_ptx, rel_pos_bias_ptx,
        spec_augment_mask_ptx, stats_pool_ptx, stride_conv1d_ptx,
    },
    vocoder::{WaveNetConfig, WaveNetStack},
};
use std::hint::black_box;

const SM_VERSIONS: &[u32] = &[75, 80, 90, 120];

// ─── PTX kernel generation benchmarks ────────────────────────────────────────

fn bench_stride_conv1d_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_conv1d_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| stride_conv1d_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_dilated_conv1d_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("dilated_conv1d_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| dilated_conv1d_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_ctc_alpha_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("ctc_alpha_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| ctc_alpha_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_spec_augment_mask_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("spec_augment_mask_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| spec_augment_mask_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_depthwise_conv1d_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("depthwise_conv1d_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| depthwise_conv1d_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_rel_pos_bias_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("rel_pos_bias_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| rel_pos_bias_ptx(black_box(sm)));
        });
    }
    group.finish();
}

fn bench_stats_pool_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_pool_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| stats_pool_ptx(black_box(sm)));
        });
    }
    group.finish();
}

// ─── Forward pass benchmarks ─────────────────────────────────────────────────

fn bench_conformer_forward_tiny(c: &mut Criterion) {
    let cfg = ConformerConfig::tiny();
    let embed_dim = cfg.embed_dim;
    let mut rng = LcgRng::new(42);
    let enc = ConformerEncoder::new(cfg, &mut rng).expect("conformer build ok");
    let t = 100usize;
    let mut x = vec![0.0f32; t * embed_dim];
    rng.fill_normal(&mut x);

    c.bench_function("conformer_tiny_forward_t100", |b| {
        b.iter(|| {
            enc.forward(black_box(&x), black_box(t))
                .expect("forward ok")
        })
    });
}

fn bench_ctc_forward_log(c: &mut Criterion) {
    let t = 200usize;
    let v = 32usize;
    let blank = 0usize;
    let target: Vec<usize> = (1..21usize).collect();
    let mut rng = LcgRng::new(7);
    let mut log_probs = vec![0.0f32; t * v];
    rng.fill_normal(&mut log_probs);
    for row in 0..t {
        let base = &mut log_probs[row * v..(row + 1) * v];
        let max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = base.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        for lp in base.iter_mut() {
            *lp = (*lp - max) - s;
        }
    }

    c.bench_function("ctc_forward_log_t200_v32", |b| {
        b.iter(|| {
            ctc_forward_log(
                black_box(&log_probs),
                black_box(t),
                black_box(v),
                black_box(&target),
                black_box(blank),
            )
            .expect("ctc ok")
        })
    });
}

fn bench_wavenet_stack(c: &mut Criterion) {
    let cfg = WaveNetConfig::tiny();
    let residual_channels = cfg.residual_channels;
    let mut rng = LcgRng::new(99);
    let stack = WaveNetStack::new(cfg, &mut rng).expect("wavenet build ok");
    let t = 100usize;
    let x = vec![0.05f32; residual_channels * t];

    c.bench_function("wavenet_tiny_stack_t100", |b| {
        b.iter(|| {
            stack
                .forward(black_box(&x), black_box(t))
                .expect("forward ok")
        })
    });
}

criterion_group!(
    benches,
    bench_stride_conv1d_ptx,
    bench_dilated_conv1d_ptx,
    bench_ctc_alpha_ptx,
    bench_spec_augment_mask_ptx,
    bench_depthwise_conv1d_ptx,
    bench_rel_pos_bias_ptx,
    bench_stats_pool_ptx,
    bench_conformer_forward_tiny,
    bench_ctc_forward_log,
    bench_wavenet_stack,
);
criterion_main!(benches);
