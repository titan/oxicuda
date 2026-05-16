use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_hdc::distance::cosine::cosine_binary;
use oxicuda_hdc::handle::{HdcHandle, LcgRng};
use oxicuda_hdc::ops::binding::binary_bind;
use oxicuda_hdc::ops::bundling::bundle_binary;
use oxicuda_hdc::ptx_kernels::{
    bundle_majority_ptx, complex_bind_ptx, cosine_sim_ptx, cyclic_shift_ptx, hamming_dist_ptx,
    hd_classify_ptx, xor_bind_ptx,
};

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("hdc_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("xor_bind_sm{sm}"), |b| b.iter(|| xor_bind_ptx(sm)));
        g.bench_function(format!("bundle_major_sm{sm}"), |b| {
            b.iter(|| bundle_majority_ptx(sm))
        });
        g.bench_function(format!("cyclic_shift_sm{sm}"), |b| {
            b.iter(|| cyclic_shift_ptx(sm))
        });
        g.bench_function(format!("cosine_sim_sm{sm}"), |b| {
            b.iter(|| cosine_sim_ptx(sm))
        });
        g.bench_function(format!("hamming_dist_sm{sm}"), |b| {
            b.iter(|| hamming_dist_ptx(sm))
        });
        g.bench_function(format!("complex_bind_sm{sm}"), |b| {
            b.iter(|| complex_bind_ptx(sm))
        });
        g.bench_function(format!("hd_classify_sm{sm}"), |b| {
            b.iter(|| hd_classify_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("hdc_algo");
    let dim = 10_000_usize;
    let mut rng = LcgRng::new(42);
    let a: Vec<i8> = (0..dim)
        .map(|_| if rng.next_bool() { 1 } else { -1 })
        .collect();
    let b: Vec<i8> = (0..dim)
        .map(|_| if rng.next_bool() { 1 } else { -1 })
        .collect();
    g.bench_function("binary_bind_10k", |bx| bx.iter(|| binary_bind(&a, &b)));
    g.bench_function("cosine_binary_10k", |bx| bx.iter(|| cosine_binary(&a, &b)));
    let hvs: Vec<Vec<i8>> = (0..16)
        .map(|_| {
            (0..dim)
                .map(|_| if rng.next_bool() { 1 } else { -1 })
                .collect()
        })
        .collect();
    g.bench_function("bundle_binary_16x10k", |bx| {
        bx.iter(|| bundle_binary(&hvs, &mut rng))
    });
    // Also benchmark the handle API
    let mut handle = HdcHandle::new(80, 99);
    g.bench_function("handle_random_binary_10k", |bx| {
        bx.iter(|| handle.random_binary_hv(dim))
    });
    g.bench_function("handle_random_complex_10k", |bx| {
        bx.iter(|| handle.random_complex_hv(dim))
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
