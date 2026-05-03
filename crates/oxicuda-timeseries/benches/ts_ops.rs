use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_timeseries::handle::LcgRng;
use oxicuda_timeseries::ptx_kernels::*;

fn bench_moving_average_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("moving_average_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(moving_average_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_patch_embed_1d_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("patch_embed_1d_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(patch_embed_1d_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_causal_temporal_conv_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("causal_temporal_conv_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(causal_temporal_conv_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_auto_correlation_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("auto_correlation_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(auto_correlation_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_revin_normalize_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("revin_normalize_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(revin_normalize_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_multirate_pool_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("multirate_pool_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(multirate_pool_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_period_detect_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("period_detect_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(period_detect_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_tcn_forward(c: &mut Criterion) {
    use oxicuda_timeseries::tcn::{TcnConfig, TcnEncoder};
    let mut rng = LcgRng::new(42);
    let cfg = TcnConfig::tiny();
    let enc = TcnEncoder::new(cfg.clone(), &mut rng).expect("ok");
    let t = 100usize;
    let x = vec![0.1_f32; t * cfg.in_channels];
    c.bench_function("tcn_tiny_forward", |b| {
        b.iter(|| std::hint::black_box(enc.forward(std::hint::black_box(&x), t).expect("ok")))
    });
}

fn bench_nhits_forward(c: &mut Criterion) {
    use oxicuda_timeseries::nhits::{NHits, NHitsConfig};
    let mut rng = LcgRng::new(42);
    let t = 96usize;
    let c_var = 4usize;
    let horizon = 24usize;
    let cfg = NHitsConfig::tiny(t, c_var, horizon);
    let model = NHits::new(cfg, &mut rng).expect("ok");
    let x = vec![0.1_f32; t * c_var];
    c.bench_function("nhits_tiny_forward", |b| {
        b.iter(|| std::hint::black_box(model.forward(std::hint::black_box(&x)).expect("ok")))
    });
}

fn bench_patchtst_forward(c: &mut Criterion) {
    use oxicuda_timeseries::patchtst::{PatchTst, PatchTstConfig};
    let mut rng = LcgRng::new(42);
    let t = 96usize;
    let c_var = 4usize;
    let horizon = 24usize;
    let cfg = PatchTstConfig::tiny(c_var, t, horizon);
    let model = PatchTst::new(cfg, &mut rng).expect("ok");
    let x = vec![0.1_f32; t * c_var];
    c.bench_function("patchtst_tiny_forward", |b| {
        b.iter(|| std::hint::black_box(model.forward(std::hint::black_box(&x)).expect("ok")))
    });
}

fn bench_timesnet_forward(c: &mut Criterion) {
    use oxicuda_timeseries::timesnet::{TimesNet, TimesNetConfig};
    let mut rng = LcgRng::new(42);
    let t = 64usize;
    let c_var = 4usize;
    let horizon = 16usize;
    let cfg = TimesNetConfig::tiny(c_var, t, horizon);
    let model = TimesNet::new(cfg, &mut rng).expect("ok");
    let x = vec![0.1_f32; t * c_var];
    c.bench_function("timesnet_tiny_forward", |b| {
        b.iter(|| std::hint::black_box(model.forward(std::hint::black_box(&x)).expect("ok")))
    });
}

fn bench_itransformer_forward(c: &mut Criterion) {
    use oxicuda_timeseries::itransformer::{ITransformer, ITransformerConfig};
    let mut rng = LcgRng::new(42);
    let t = 96usize;
    let c_var = 4usize;
    let horizon = 24usize;
    let cfg = ITransformerConfig::tiny(c_var, t, horizon);
    let model = ITransformer::new(cfg, &mut rng).expect("ok");
    let x = vec![0.1_f32; t * c_var];
    c.bench_function("itransformer_tiny_forward", |b| {
        b.iter(|| std::hint::black_box(model.forward(std::hint::black_box(&x)).expect("ok")))
    });
}

criterion_group!(
    ptx_benches,
    bench_moving_average_ptx,
    bench_patch_embed_1d_ptx,
    bench_causal_temporal_conv_ptx,
    bench_auto_correlation_ptx,
    bench_revin_normalize_ptx,
    bench_multirate_pool_ptx,
    bench_period_detect_ptx,
);

criterion_group!(
    arch_benches,
    bench_tcn_forward,
    bench_nhits_forward,
    bench_patchtst_forward,
    bench_timesnet_forward,
    bench_itransformer_forward,
);

criterion_main!(ptx_benches, arch_benches);
