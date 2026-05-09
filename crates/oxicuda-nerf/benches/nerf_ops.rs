//! Benchmarks for oxicuda-nerf operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_nerf::prelude::*;

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [80_u32, 90, 100, 120];
    #[allow(clippy::type_complexity)]
    let kernel_fns: &[(&str, fn(u32) -> String)] = &[
        ("positional_encoding_ptx", positional_encoding_ptx),
        ("volume_render_ptx", volume_render_ptx),
        ("hash_grid_lookup_ptx", hash_grid_lookup_ptx),
        ("ray_march_ptx", ray_march_ptx),
        ("sh_to_rgb_ptx", sh_to_rgb_ptx),
        ("occupancy_update_ptx", occupancy_update_ptx),
        ("importance_resample_ptx", importance_resample_ptx),
    ];
    let mut g = c.benchmark_group("ptx_generation");
    for (name, gen_fn) in kernel_fns {
        for &sm in &sm_versions {
            g.bench_with_input(BenchmarkId::new(*name, sm), &sm, |b, &sm| {
                b.iter(|| gen_fn(sm))
            });
        }
    }
    g.finish();
}

fn bench_pos_enc(c: &mut Criterion) {
    let cfg = PosEncConfig {
        n_freq: 10,
        include_input: true,
        input_dim: 3,
    };
    let input = vec![0.5_f32; 1024 * 3];
    let mut g = c.benchmark_group("positional_encoding");
    g.bench_function("1024_pts_L10", |b| {
        b.iter(|| positional_encode(&input, &cfg))
    });
    g.finish();
}

fn bench_hash_grid(c: &mut Criterion) {
    let cfg = HashGridConfig {
        n_levels: 16,
        n_features_per_level: 2,
        log2_hashmap_size: 14,
        base_resolution: 16,
        max_resolution: 2048,
    };
    let mut rng = LcgRng::new(42);
    let grid = HashGrid::new(cfg, &mut rng).unwrap();
    let pts: Vec<f32> = (0..1024)
        .flat_map(|i| {
            let v = (i as f32) / 1024.0;
            [v, 1.0 - v, v * 0.5]
        })
        .collect();

    let mut g = c.benchmark_group("hash_grid");
    g.bench_function("query_batch_1024", |b| {
        b.iter(|| grid.query_batch(&pts, 1024))
    });
    g.finish();
}

fn bench_volume_render(c: &mut Criterion) {
    let n_rays = 64_usize;
    let n_samp = 64_usize;
    let sigma = vec![0.05_f32; n_rays * n_samp];
    let color = vec![0.5_f32; n_rays * n_samp * 3];
    let t: Vec<f32> = (0..n_rays * n_samp)
        .map(|i| 0.01 + i as f32 * 0.01)
        .collect();

    let mut g = c.benchmark_group("volume_render");
    g.bench_function("64_rays_64_samples", |b| {
        b.iter(|| volume_render_batch(&sigma, &color, &t, n_rays, n_samp))
    });
    g.finish();
}

fn bench_stratified_sample(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let mut g = c.benchmark_group("stratified_sample");
    g.bench_function("128_samples", |b| {
        b.iter(|| stratified_sample(0.1, 10.0, 128, &mut rng))
    });
    g.finish();
}

fn bench_tensorf(c: &mut Criterion) {
    let cfg = TensorRfConfig {
        rank: 16,
        grid_dim: 32,
        n_color_feat: 27,
    };
    let mut rng = LcgRng::new(314);
    let tf = TensorRf::new(cfg, &mut rng).unwrap();

    let pts: Vec<[f32; 3]> = (0..1024)
        .map(|i| {
            let v = (i as f32) / 512.0 - 1.0;
            [v, -v, v * 0.5]
        })
        .collect();

    let mut g = c.benchmark_group("tensorf");
    g.bench_function("density_query_1024", |b| {
        b.iter(|| {
            for &xyz in &pts {
                let _ = tf.query_density(xyz);
            }
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_ptx,
    bench_pos_enc,
    bench_hash_grid,
    bench_volume_render,
    bench_stratified_sample,
    bench_tensorf
);
criterion_main!(benches);
