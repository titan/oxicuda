use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_manifold::handle::LcgRng;
use oxicuda_manifold::linalg::jacobi_eig::jacobi_eigh;
use oxicuda_manifold::linear::pca::pca_fit;
use oxicuda_manifold::neighbor::knn_brute::knn_brute;
use oxicuda_manifold::ptx_kernels::{
    knn_topk_ptx, mds_double_center_ptx, pairwise_dist_sq_ptx, pca_center_ptx, random_proj_ptx,
    tsne_grad_ptx, umap_step_ptx,
};

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("pairwise_dist_sq", pairwise_dist_sq_ptx),
        ("knn_topk", knn_topk_ptx),
        ("tsne_grad", tsne_grad_ptx),
        ("umap_step", umap_step_ptx),
        ("pca_center", pca_center_ptx),
        ("mds_double_center", mds_double_center_ptx),
        ("random_proj", random_proj_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_pca(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let n = 16;
    let dim = 6;
    let mut x = vec![0.0; n * dim];
    for v in &mut x {
        *v = rng.next_normal();
    }
    c.bench_function("pca_16x6_to_3", |b| {
        b.iter(|| pca_fit(&x, n, dim, 3).expect("ok"))
    });
}

fn bench_eigh(c: &mut Criterion) {
    let mut rng = LcgRng::new(11);
    let n = 12;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = rng.next_normal();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    c.bench_function("jacobi_eigh_12x12", |b| {
        b.iter(|| jacobi_eigh(&a, n).expect("ok"))
    });
}

fn bench_knn(c: &mut Criterion) {
    let mut rng = LcgRng::new(13);
    let n = 32;
    let dim = 4;
    let mut x = vec![0.0; n * dim];
    for v in &mut x {
        *v = rng.next_normal();
    }
    c.bench_function("knn_brute_32x4_k4", |b| {
        b.iter(|| knn_brute(&x, n, dim, 4).expect("ok"))
    });
}

criterion_group!(benches, bench_ptx, bench_pca, bench_eigh, bench_knn);
criterion_main!(benches);
