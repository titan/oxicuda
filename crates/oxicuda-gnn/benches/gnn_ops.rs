//! Criterion benchmarks for `oxicuda-gnn` operations.

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_gnn::graph::coo::CooGraph;
use oxicuda_gnn::graph::csr::CsrGraph;
use oxicuda_gnn::layers::gcn::{GcnConfig, GcnLayer};
use oxicuda_gnn::message_passing::scatter::scatter_add;
use oxicuda_gnn::pooling::global_pool::global_mean_pool;

// ─── Graph construction ───────────────────────────────────────────────────────

fn bench_csr_from_edges(c: &mut Criterion) {
    let n = 1024usize;
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
        .collect();

    c.bench_function("csr_from_edges_n1024", |b| {
        b.iter(|| {
            let g = CsrGraph::from_edges(std::hint::black_box(n), std::hint::black_box(&edges))
                .unwrap();
            std::hint::black_box(g.n_edges())
        });
    });
}

fn bench_coo_to_csr(c: &mut Criterion) {
    let n = 512usize;
    let src: Vec<usize> = (0..n).flat_map(|i| [i, (i + 1) % n]).collect();
    let dst: Vec<usize> = (0..n).flat_map(|i| [(i + 1) % n, i]).collect();
    let coo = CooGraph::new(n, src, dst).unwrap();

    c.bench_function("coo_to_csr_n512", |b| {
        b.iter(|| {
            let csr = coo.to_csr().unwrap();
            std::hint::black_box(csr.n_edges())
        });
    });
}

// ─── SpMV ─────────────────────────────────────────────────────────────────────

fn bench_spmv(c: &mut Criterion) {
    let n = 512usize;
    let feat_dim = 64usize;
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
        .collect();
    let g = CsrGraph::from_edges(n, &edges).unwrap();
    let x = vec![0.1_f32; n * feat_dim];

    c.bench_function("spmv_ring_n512_fd64", |b| {
        b.iter(|| {
            let y = g
                .spmv(std::hint::black_box(&x), std::hint::black_box(feat_dim))
                .unwrap();
            std::hint::black_box(y.len())
        });
    });
}

// ─── GCN forward ─────────────────────────────────────────────────────────────

fn bench_gcn_forward(c: &mut Criterion) {
    let n = 256usize;
    let in_f = 64usize;
    let out_f = 64usize;
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
        .collect();
    let g = CsrGraph::from_edges(n, &edges).unwrap();
    let feats = vec![0.1_f32; n * in_f];
    let weight = vec![0.01_f32; in_f * out_f];
    let layer = GcnLayer::new(GcnConfig {
        in_features: in_f,
        out_features: out_f,
        bias: false,
        normalize: true,
    })
    .unwrap();

    c.bench_function("gcn_forward_ring_n256_fd64", |b| {
        b.iter(|| {
            let out = layer
                .forward(
                    std::hint::black_box(&g),
                    std::hint::black_box(&feats),
                    std::hint::black_box(&weight),
                    None,
                )
                .unwrap();
            std::hint::black_box(out.len())
        });
    });
}

// ─── Scatter-add ─────────────────────────────────────────────────────────────

fn bench_scatter_add(c: &mut Criterion) {
    let n_edges = 8192usize;
    let feat_dim = 32usize;
    let n_nodes = 512usize;
    let src: Vec<f32> = vec![0.1_f32; n_edges * feat_dim];
    let idx: Vec<usize> = (0..n_edges).map(|i| i % n_nodes).collect();

    c.bench_function("scatter_add_8k_edges_fd32", |b| {
        b.iter(|| {
            let out = scatter_add(
                std::hint::black_box(&src),
                std::hint::black_box(&idx),
                std::hint::black_box(n_nodes),
                std::hint::black_box(feat_dim),
            )
            .unwrap();
            std::hint::black_box(out.len())
        });
    });
}

// ─── Global mean pool ─────────────────────────────────────────────────────────

fn bench_global_mean_pool(c: &mut Criterion) {
    let n = 2048usize;
    let feat_dim = 128usize;
    let x = vec![0.3_f32; n * feat_dim];

    c.bench_function("global_mean_pool_n2048_fd128", |b| {
        b.iter(|| {
            let out = global_mean_pool(
                std::hint::black_box(&x),
                std::hint::black_box(n),
                std::hint::black_box(feat_dim),
            )
            .unwrap();
            std::hint::black_box(out.len())
        });
    });
}

// ─── Registration ────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_csr_from_edges,
    bench_coo_to_csr,
    bench_spmv,
    bench_gcn_forward,
    bench_scatter_add,
    bench_global_mean_pool,
);
criterion_main!(benches);
