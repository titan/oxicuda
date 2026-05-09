use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_anomaly::handle::LcgRng;
use oxicuda_anomaly::prelude::*;

// ─── PTX kernel benchmarks ───────────────────────────────────────────────────

fn bench_svdd_loss_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("svdd_loss_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(svdd_loss_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_recon_score_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("recon_score_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(recon_score_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_lof_reach_dist_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("lof_reach_dist_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(lof_reach_dist_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_copod_ecdf_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("copod_ecdf_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(copod_ecdf_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_mahal_dist_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("mahal_dist_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(mahal_dist_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_iforest_score_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("iforest_score_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(iforest_score_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_ensemble_normalize_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ensemble_normalize_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(ensemble_normalize_ptx(sm)))
        });
    }
    g.finish();
}

// ─── Algorithm benchmarks ────────────────────────────────────────────────────

fn bench_ae_score_batch(c: &mut Criterion) {
    let n = 256_usize;
    let d = 32_usize;
    let cfg = AeConfig {
        encoder_dims: vec![d, 16, 8],
        decoder_dims: vec![8, 16, d],
    };
    let mut rng = LcgRng::new(0);
    let ae = AutoencoderAnomaly::new(cfg, &mut rng).unwrap();
    let mut data = vec![0.0_f32; n * d];
    rng.fill_normal(&mut data);

    c.bench_function("bench_ae_score_batch_256x32", |b| {
        b.iter(|| std::hint::black_box(ae.score_batch(&data, n).unwrap()))
    });
}

fn bench_lof_fit_score(c: &mut Criterion) {
    let n_train = 512_usize;
    let n_query = 32_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(1);
    let mut train_data = vec![0.0_f32; n_train * d];
    rng.fill_normal(&mut train_data);
    let mut query_data = vec![0.0_f32; n_query * d];
    rng.fill_normal(&mut query_data);

    c.bench_function("bench_lof_fit_score_512x32", |b| {
        b.iter(|| {
            let mut lof = Lof::new(5);
            lof.fit(&train_data, n_train, d).unwrap();
            std::hint::black_box(lof.score_batch(&query_data, n_query).unwrap())
        })
    });
}

fn bench_copod_score(c: &mut Criterion) {
    let n = 256_usize;
    let d = 8_usize;
    let mut rng = LcgRng::new(2);
    let mut train_data = vec![0.0_f32; n * d];
    rng.fill_normal(&mut train_data);

    let mut copod = Copod::new();
    copod.fit(&train_data, n, d).unwrap();

    let mut query = vec![0.0_f32; n * d];
    rng.fill_normal(&mut query);

    c.bench_function("bench_copod_score_256", |b| {
        b.iter(|| std::hint::black_box(copod.score_batch(&query, n).unwrap()))
    });
}

fn bench_mahalanobis(c: &mut Criterion) {
    let n = 512_usize;
    let d = 16_usize;
    let mut rng = LcgRng::new(3);
    let mut train_data = vec![0.0_f32; n * d];
    rng.fill_normal(&mut train_data);

    let mut det = MahalanobisDetector::new();
    det.fit(&train_data, n, d).unwrap();

    let mut query = vec![0.0_f32; n * d];
    rng.fill_normal(&mut query);

    c.bench_function("bench_mahalanobis_512x16", |b| {
        b.iter(|| std::hint::black_box(det.score_batch(&query, n).unwrap()))
    });
}

fn bench_iforest(c: &mut Criterion) {
    let n = 512_usize;
    let d = 8_usize;
    let mut rng = LcgRng::new(4);
    let mut train_data = vec![0.0_f32; n * d];
    rng.fill_normal(&mut train_data);

    let mut scorer = IsolationScorer::new(100, &mut rng);
    scorer.fit(&train_data, n, d, &mut rng).unwrap();

    let mut query = vec![0.0_f32; n * d];
    rng.fill_normal(&mut query);

    c.bench_function("bench_iforest_512", |b| {
        b.iter(|| std::hint::black_box(scorer.score_batch(&query, n).unwrap()))
    });
}

criterion_group!(
    ptx_benches,
    bench_svdd_loss_ptx,
    bench_recon_score_ptx,
    bench_lof_reach_dist_ptx,
    bench_copod_ecdf_ptx,
    bench_mahal_dist_ptx,
    bench_iforest_score_ptx,
    bench_ensemble_normalize_ptx,
);

criterion_group!(
    algo_benches,
    bench_ae_score_batch,
    bench_lof_fit_score,
    bench_copod_score,
    bench_mahalanobis,
    bench_iforest,
);

criterion_main!(ptx_benches, algo_benches);
