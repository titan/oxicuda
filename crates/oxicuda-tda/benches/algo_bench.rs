//! Extended algorithm benchmarks on standard parametric datasets.
//!
//! Unlike `tda_ops.rs` (which times PTX *string* generation and a tiny Vietoris–Rips
//! run), this suite exercises the full host-side persistence pipeline end to end on the
//! topological surfaces TDA libraries are usually validated against: the circle `S¹`, the
//! 2-torus `T²` and the 2-sphere `S²`, each sampled deterministically (no `rand`
//! dependency). It also pits the chunk-parallel boundary reducer against the sequential
//! ELZ reducer and benchmarks the diagram-distance back end on the resulting diagrams.
//!
//! Everything here is CPU-side: no GPU, no driver, no fabricated device timings. The new
//! `gpu_reduction` PTX emitters are timed only as *code generation*, exactly like the
//! existing kernel benches.

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_tda::complex::filtration::Filtration;
use oxicuda_tda::homology::boundary::BoundaryMatrix;
use oxicuda_tda::homology::gpu_reduction::{
    batched_column_reduce_ptx, chunked_parallel_reduce, vietoris_rips_edges_ptx,
    wasserstein_auction_ptx,
};
use oxicuda_tda::homology::persistent::extract_persistence_pairs;
use oxicuda_tda::homology::reduction::reduce_boundary_matrix;
use oxicuda_tda::persistence::diagram::PersistenceDiagram;
use oxicuda_tda::persistence::distance::{bottleneck_distance, wasserstein_1};

// ---------------------------------------------------------------------------
// Deterministic parametric point clouds (flat row-major [x, y, z, ...]).
// ---------------------------------------------------------------------------

/// `n` points equally spaced on the unit circle `S¹` in the plane.
fn circle(n: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|i| {
            let t = std::f64::consts::TAU * (i as f64) / (n as f64);
            [t.cos(), t.sin()]
        })
        .collect()
}

/// `m × m` points on a 2-torus `T²` in `R³` (tube radius `r`, centre radius `R`).
fn torus(m: usize) -> Vec<f64> {
    let big_r = 2.0_f64;
    let small_r = 0.8_f64;
    let mut pts = Vec::with_capacity(m * m * 3);
    for i in 0..m {
        let u = std::f64::consts::TAU * (i as f64) / (m as f64);
        for j in 0..m {
            let v = std::f64::consts::TAU * (j as f64) / (m as f64);
            let ring = big_r + small_r * v.cos();
            pts.push(ring * u.cos());
            pts.push(ring * u.sin());
            pts.push(small_r * v.sin());
        }
    }
    pts
}

/// Roughly `n` points on the unit 2-sphere `S²` via the deterministic Fibonacci spiral.
fn sphere(n: usize) -> Vec<f64> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut pts = Vec::with_capacity(n * 3);
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / (n as f64);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let theta = golden * (i as f64);
        pts.push(r * theta.cos());
        pts.push(r * theta.sin());
        pts.push(z);
    }
    pts
}

// ---------------------------------------------------------------------------
// Pipeline helpers.
// ---------------------------------------------------------------------------

/// Full persistence pipeline: Vietoris–Rips → boundary matrix → sequential reduction →
/// persistence pairs.  Returns the pairs so the bencher cannot optimise the work away.
fn rips_pipeline(
    points: &[f64],
    n_dims: usize,
    max_radius: f64,
    max_dim: usize,
) -> Vec<oxicuda_tda::homology::persistent::PersistencePair> {
    let filt = Filtration::vietoris_rips_from_points(points, n_dims, max_radius, max_dim)
        .expect("vietoris-rips");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("boundary matrix");
    reduce_boundary_matrix(&mut bm);
    extract_persistence_pairs(&bm, &filt).expect("persistence pairs")
}

// ---------------------------------------------------------------------------
// Benches.
// ---------------------------------------------------------------------------

fn bench_parametric_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("rips_pipeline");

    let circle_pts = circle(48);
    group.bench_function("circle_S1_48pts", |b| {
        b.iter(|| rips_pipeline(&circle_pts, 2, 0.6, 2))
    });

    let torus_pts = torus(8); // 64 points on T²
    group.bench_function("torus_T2_64pts", |b| {
        b.iter(|| rips_pipeline(&torus_pts, 3, 1.6, 2))
    });

    let sphere_pts = sphere(60);
    group.bench_function("sphere_S2_60pts", |b| {
        b.iter(|| rips_pipeline(&sphere_pts, 3, 0.9, 2))
    });

    group.finish();
}

fn bench_reduction_strategies(c: &mut Criterion) {
    // A single shared filtration so both reducers face identical work.
    let pts = circle(56);
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 0.7, 2).expect("rips");

    let mut group = c.benchmark_group("reduction_strategy");
    group.bench_function("sequential_elz", |b| {
        b.iter(|| {
            let mut bm = BoundaryMatrix::from_filtration(&filt).expect("bm");
            reduce_boundary_matrix(&mut bm);
            bm
        })
    });
    group.bench_function("chunk_parallel", |b| {
        b.iter(|| {
            let mut bm = BoundaryMatrix::from_filtration(&filt).expect("bm");
            chunked_parallel_reduce(&mut bm);
            bm
        })
    });
    group.finish();
}

fn bench_diagram_distances(c: &mut Criterion) {
    // Two diagrams from a circle and a perturbed circle, in H1.
    let a = rips_pipeline(&circle(40), 2, 0.7, 2);
    let mut perturbed = circle(40);
    for (k, slot) in perturbed.iter_mut().enumerate() {
        // Deterministic small perturbation, no rng needed.
        *slot += 0.02 * (((k * 131) % 17) as f64 / 17.0 - 0.5);
    }
    let b = rips_pipeline(&perturbed, 2, 0.7, 2);

    let diag_a = PersistenceDiagram::from_pairs_by_dim(&a, 1).remove(1);
    let diag_b = PersistenceDiagram::from_pairs_by_dim(&b, 1).remove(1);

    let mut group = c.benchmark_group("diagram_distance");
    group.bench_function("bottleneck_H1", |bch| {
        bch.iter(|| bottleneck_distance(&diag_a, &diag_b).expect("bottleneck"))
    });
    group.bench_function("wasserstein1_H1", |bch| {
        bch.iter(|| wasserstein_1(&diag_a, &diag_b).expect("wasserstein"))
    });
    group.finish();
}

fn bench_gpu_reduction_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    type KernelEntry = (&'static str, fn(u32) -> String);
    let kernels: &[KernelEntry] = &[
        ("batched_column_reduce", batched_column_reduce_ptx),
        ("vietoris_rips_edges", vietoris_rips_edges_ptx),
        ("wasserstein_auction", wasserstein_auction_ptx),
    ];
    let mut group = c.benchmark_group("gpu_reduction_ptx");
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            group.bench_function(format!("{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_parametric_pipeline,
    bench_reduction_strategies,
    bench_diagram_distances,
    bench_gpu_reduction_ptx
);
criterion_main!(benches);
