use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_snn::handle::LcgRng;
use oxicuda_snn::neuron::lif::{LifConfig, LifState, lif_step};
use oxicuda_snn::ptx_kernels::{
    bptt_accum_ptx, lif_step_ptx, poisson_sample_ptx, rate_encode_ptx, spike_conv_ptx,
    stdp_update_ptx, surrogate_grad_ptx,
};
use oxicuda_snn::surrogate::sigmoid::sigmoid_grad;

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("snn_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("lif_step_sm{sm}"), |b| b.iter(|| lif_step_ptx(sm)));
        g.bench_function(format!("surrogate_grad_sm{sm}"), |b| {
            b.iter(|| surrogate_grad_ptx(sm))
        });
        g.bench_function(format!("stdp_update_sm{sm}"), |b| {
            b.iter(|| stdp_update_ptx(sm))
        });
        g.bench_function(format!("spike_conv_sm{sm}"), |b| {
            b.iter(|| spike_conv_ptx(sm))
        });
        g.bench_function(format!("rate_encode_sm{sm}"), |b| {
            b.iter(|| rate_encode_ptx(sm))
        });
        g.bench_function(format!("poisson_sample_sm{sm}"), |b| {
            b.iter(|| poisson_sample_ptx(sm))
        });
        g.bench_function(format!("bptt_accum_sm{sm}"), |b| {
            b.iter(|| bptt_accum_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("snn_algo");
    let n = 256_usize;
    let cfg = LifConfig::default();
    let current = vec![0.5_f32; n];
    g.bench_function("lif_step_256", |bx| {
        let mut state = LifState::new(n);
        let mut spikes = vec![0.0_f32; n];
        bx.iter(|| {
            let _ = lif_step(&mut state, &current, &cfg, &mut spikes);
        });
    });

    let mut rng = LcgRng::new(7);
    let mut v = vec![0.0_f32; n];
    rng.fill_normal(&mut v);
    let mut grad = vec![0.0_f32; n];
    g.bench_function("sigmoid_grad_256", |bx| {
        bx.iter(|| {
            let _ = sigmoid_grad(&v, 1.0, 4.0, &mut grad);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
