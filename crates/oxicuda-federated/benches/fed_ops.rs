use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_federated::handle::LcgRng;
use oxicuda_federated::prelude::*;

fn bench_aggregate_mean_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("aggregate_mean_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(aggregate_mean_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_dp_clip_gradient_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("dp_clip_gradient_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(dp_clip_gradient_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_fedavg_weighted_sum_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("fedavg_weighted_sum_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(fedavg_weighted_sum_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gaussian_noise_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gaussian_noise_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gaussian_noise_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_pairwise_mask_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("pairwise_mask_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(pairwise_mask_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_qsgd_quantize_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("qsgd_quantize_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(qsgd_quantize_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_topk_mask_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("topk_mask_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(topk_mask_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_fedavg_aggregate(c: &mut Criterion) {
    let n_params = 4096usize;
    let n_clients = 16usize;
    let mut rng = LcgRng::new(0);
    let mut updates: Vec<(Vec<f32>, f32)> = Vec::with_capacity(n_clients);
    for _ in 0..n_clients {
        let mut p = vec![0.0_f32; n_params];
        rng.fill_normal(&mut p);
        updates.push((p, 1.0));
    }
    c.bench_function("fedavg_aggregate_p4096_n16", |b| {
        b.iter(|| {
            let mut state = FedAvgState::new(n_params);
            state.aggregate(std::hint::black_box(&updates)).expect("ok");
            std::hint::black_box(state)
        })
    });
}

fn bench_topk_sparsify(c: &mut Criterion) {
    let n = 8192usize;
    let mut rng = LcgRng::new(1);
    let mut grad = vec![0.0_f32; n];
    rng.fill_normal(&mut grad);
    c.bench_function("topk_sparsify_n8192_k512", |b| {
        b.iter(|| {
            std::hint::black_box(topk_sparsify(std::hint::black_box(&grad), 512).expect("ok"))
        })
    });
}

fn bench_qsgd_quantize(c: &mut Criterion) {
    let n = 8192usize;
    let mut rng = LcgRng::new(2);
    let mut grad = vec![0.0_f32; n];
    rng.fill_normal(&mut grad);
    let mut local_rng = LcgRng::new(7);
    c.bench_function("qsgd_quantize_n8192_4bit", |b| {
        b.iter(|| {
            std::hint::black_box(
                stochastic_quantize(std::hint::black_box(&grad), 4, &mut local_rng).expect("ok"),
            )
        })
    });
}

fn bench_shamir_share_reconstruct(c: &mut Criterion) {
    let n = 16usize;
    let threshold = 3usize;
    let cfg = ShamirConfig::new(threshold, n).expect("ok");
    let secret = 42_u64;
    let mut rng = LcgRng::new(3);
    c.bench_function("shamir_share_n16_t3", |b| {
        b.iter(|| std::hint::black_box(share_scalar(secret, &cfg, &mut rng).expect("ok")))
    });
    let shares = share_scalar(secret, &cfg, &mut rng).expect("ok");
    c.bench_function("shamir_reconstruct_n16_t3", |b| {
        b.iter(|| {
            std::hint::black_box(
                reconstruct_scalar(std::hint::black_box(&shares[..threshold]), threshold)
                    .expect("ok"),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_aggregate_mean_ptx,
    bench_dp_clip_gradient_ptx,
    bench_fedavg_weighted_sum_ptx,
    bench_gaussian_noise_ptx,
    bench_pairwise_mask_ptx,
    bench_qsgd_quantize_ptx,
    bench_topk_mask_ptx,
    bench_fedavg_aggregate,
    bench_topk_sparsify,
    bench_qsgd_quantize,
    bench_shamir_share_reconstruct,
);
criterion_main!(benches);
