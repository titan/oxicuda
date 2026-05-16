use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_tda::ptx_kernels::{
    betti_count_ptx, boundary_reduce_ptx, diagram_match_ptx, filtration_sort_ptx,
    mapper_cluster_ptx, pairwise_dist_ptx, witness_dist_ptx,
};

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("pairwise_dist", pairwise_dist_ptx),
        ("filtration_sort", filtration_sort_ptx),
        ("boundary_reduce", boundary_reduce_ptx),
        ("diagram_match", diagram_match_ptx),
        ("witness_dist", witness_dist_ptx),
        ("betti_count", betti_count_ptx),
        ("mapper_cluster", mapper_cluster_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_vietoris_rips(c: &mut Criterion) {
    use oxicuda_tda::complex::filtration::Filtration;
    c.bench_function("vietoris_rips_20pts_2d", |b| {
        b.iter(|| {
            let mut pts = vec![0.0f64; 40];
            for i in 0..20 {
                pts[2 * i] = (i as f64) / 20.0;
                pts[2 * i + 1] = ((i * 7) % 20) as f64 / 20.0;
            }
            Filtration::vietoris_rips_from_points(&pts, 2, 0.5, 2).expect("ok")
        })
    });
}

criterion_group!(benches, bench_ptx, bench_vietoris_rips);
criterion_main!(benches);
