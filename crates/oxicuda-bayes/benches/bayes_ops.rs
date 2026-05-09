use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_bayes::handle::LcgRng;
use oxicuda_bayes::prelude::*;

fn bench_kl_gaussian_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("kl_gaussian_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(kl_gaussian_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_mc_dropout_mask_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("mc_dropout_mask_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(mc_dropout_mask_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_local_reparam_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("local_reparam_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(local_reparam_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_ece_bucket_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ece_bucket_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(ece_bucket_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_ensemble_aggregate_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ensemble_aggregate_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(ensemble_aggregate_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_flipout_perturb_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("flipout_perturb_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(flipout_perturb_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_temp_scale_logits_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("temp_scale_logits_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(temp_scale_logits_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_temperature_scaling_fit(c: &mut Criterion) {
    let n = 256usize;
    let k = 10usize;
    let mut rng = LcgRng::new(0);
    let mut logits = vec![0.0_f32; n * k];
    rng.fill_normal(&mut logits);
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    c.bench_function("temperature_scaling_fit_n256_k10", |b| {
        b.iter(|| {
            std::hint::black_box(
                TemperatureScaler::fit_default(
                    std::hint::black_box(&logits),
                    std::hint::black_box(&labels),
                    k,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_isotonic_pav_fit(c: &mut Criterion) {
    let n = 1024usize;
    let mut rng = LcgRng::new(1);
    let mut scores = vec![0.0_f32; n];
    for s in scores.iter_mut() {
        *s = rng.next_f32();
    }
    let targets: Vec<f32> = scores
        .iter()
        .map(|&s| if s > 0.5 { 1.0 } else { 0.0 })
        .collect();
    c.bench_function("isotonic_pav_fit_n1024", |b| {
        b.iter(|| {
            std::hint::black_box(
                IsotonicRegressor::fit(
                    std::hint::black_box(&scores),
                    std::hint::black_box(&targets),
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_ece_compute(c: &mut Criterion) {
    let n = 4096usize;
    let mut rng = LcgRng::new(2);
    let mut probs = vec![0.0_f32; n * 10];
    for p in probs.iter_mut() {
        *p = rng.next_f32();
    }
    // re-normalise rows
    for chunk in probs.chunks_mut(10) {
        let s: f32 = chunk.iter().sum();
        if s > 0.0 {
            let inv = 1.0 / s;
            for v in chunk.iter_mut() {
                *v *= inv;
            }
        }
    }
    let labels: Vec<usize> = (0..n).map(|i| i % 10).collect();
    let (conf, ok) = top1_confidences(&probs, &labels, 10).expect("ok");
    c.bench_function("ece_n4096_k10_15bins", |b| {
        b.iter(|| {
            std::hint::black_box(expected_calibration_error(
                std::hint::black_box(&conf),
                std::hint::black_box(&ok),
                15,
            ))
        })
    });
}

fn bench_swag_sample(c: &mut Criterion) {
    let mut handle = BayesHandle::default_handle();
    let dim = 256usize;
    let max_rank = 20usize;
    let mut posterior = SwagPosterior::new(dim, max_rank).expect("ok");
    let mut buf = vec![0.0_f32; dim];
    for _ in 0..(max_rank + 4) {
        handle.rng_mut().fill_normal(&mut buf);
        posterior.update(&buf).expect("ok");
    }
    c.bench_function("swag_sample_d256_k20", |b| {
        b.iter(|| std::hint::black_box(posterior.sample(handle.rng_mut()).expect("ok")))
    });
}

fn bench_deep_ensemble_aggregate(c: &mut Criterion) {
    let m = 16usize;
    let k = 100usize;
    let mut rng = LcgRng::new(3);
    let mut preds: Vec<Vec<f32>> = Vec::with_capacity(m);
    for _ in 0..m {
        let mut p = vec![0.0_f32; k];
        for v in p.iter_mut() {
            *v = rng.next_f32();
        }
        let s: f32 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= s;
        }
        preds.push(p);
    }
    let ensemble = DeepEnsemble::new(preds).expect("ok");
    c.bench_function("deep_ensemble_aggregate_m16_k100", |b| {
        b.iter(|| std::hint::black_box(ensemble.aggregate()))
    });
}

criterion_group!(
    benches,
    bench_kl_gaussian_ptx,
    bench_mc_dropout_mask_ptx,
    bench_local_reparam_ptx,
    bench_ece_bucket_ptx,
    bench_ensemble_aggregate_ptx,
    bench_flipout_perturb_ptx,
    bench_temp_scale_logits_ptx,
    bench_temperature_scaling_fit,
    bench_isotonic_pav_fit,
    bench_ece_compute,
    bench_swag_sample,
    bench_deep_ensemble_aggregate,
);
criterion_main!(benches);
