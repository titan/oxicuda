use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_privacy::handle::{LcgRng, PrivacyHandle};
use oxicuda_privacy::mechanism::exponential::{ExponentialConfig, exponential_sample};
use oxicuda_privacy::ptx_kernels::{
    clip_gradient_ptx, exponential_sample_ptx, gaussian_noise_ptx, laplace_noise_ptx,
    oue_encode_ptx, prv_convolve_ptx, svt_threshold_ptx,
};

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("privacy_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("exponential_sample_sm{sm}"), |b| {
            b.iter(|| exponential_sample_ptx(sm))
        });
        g.bench_function(format!("laplace_noise_sm{sm}"), |b| {
            b.iter(|| laplace_noise_ptx(sm))
        });
        g.bench_function(format!("gaussian_noise_sm{sm}"), |b| {
            b.iter(|| gaussian_noise_ptx(sm))
        });
        g.bench_function(format!("clip_gradient_sm{sm}"), |b| {
            b.iter(|| clip_gradient_ptx(sm))
        });
        g.bench_function(format!("svt_threshold_sm{sm}"), |b| {
            b.iter(|| svt_threshold_ptx(sm))
        });
        g.bench_function(format!("prv_convolve_sm{sm}"), |b| {
            b.iter(|| prv_convolve_ptx(sm))
        });
        g.bench_function(format!("oue_encode_sm{sm}"), |b| {
            b.iter(|| oue_encode_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("privacy_algo");
    let mut rng = LcgRng::new(42);
    let scores: Vec<f64> = (0..256).map(|i| i as f64 * 0.01).collect();
    let cfg = ExponentialConfig::new(1.0, 1.0).expect("valid config");
    g.bench_function("exponential_256", |bx| {
        bx.iter(|| exponential_sample(&scores, &cfg, &mut rng))
    });
    g.finish();
}

fn bench_handle(c: &mut Criterion) {
    let mut g = c.benchmark_group("privacy_handle");
    let mut handle = PrivacyHandle::new(80, 42);
    g.bench_function("gaussian_noise_1024", |b| {
        b.iter(|| handle.generate_gaussian_noise(1.0, 1024))
    });
    g.bench_function("laplace_noise_1024", |b| {
        b.iter(|| handle.generate_laplace_noise(1.0, 1024))
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo, bench_handle);
criterion_main!(benches);
