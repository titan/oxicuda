//! Criterion benchmarks for oxicuda-mamba PTX kernel generation.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_mamba::ptx_kernels::{
    depthwise_conv1d_ptx, hippo_legendre_ptx, parallel_scan_ptx, rms_norm_silu_ptx,
    selective_scan_ptx, ssd_chunk_ptx, wkv_forward_ptx,
};

const SM_VERSIONS: &[u32] = &[75, 80, 90, 120];

fn bench_selective_scan_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("selective_scan_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| selective_scan_ptx(sm));
        });
    }
    group.finish();
}

fn bench_parallel_scan_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scan_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| parallel_scan_ptx(sm));
        });
    }
    group.finish();
}

fn bench_depthwise_conv1d_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("depthwise_conv1d_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| depthwise_conv1d_ptx(sm));
        });
    }
    group.finish();
}

fn bench_wkv_forward_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("wkv_forward_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| wkv_forward_ptx(sm));
        });
    }
    group.finish();
}

fn bench_ssd_chunk_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssd_chunk_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| ssd_chunk_ptx(sm));
        });
    }
    group.finish();
}

fn bench_hippo_legendre_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("hippo_legendre_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| hippo_legendre_ptx(sm));
        });
    }
    group.finish();
}

fn bench_rms_norm_silu_ptx(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm_silu_ptx");
    for &sm in SM_VERSIONS {
        group.bench_with_input(BenchmarkId::from_parameter(sm), &sm, |b, &sm| {
            b.iter(|| rms_norm_silu_ptx(sm));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_selective_scan_ptx,
    bench_parallel_scan_ptx,
    bench_depthwise_conv1d_ptx,
    bench_wkv_forward_ptx,
    bench_ssd_chunk_ptx,
    bench_hippo_legendre_ptx,
    bench_rms_norm_silu_ptx,
);
criterion_main!(benches);
