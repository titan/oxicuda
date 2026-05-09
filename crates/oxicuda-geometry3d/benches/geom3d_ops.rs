use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_geometry3d::handle::LcgRng;
use oxicuda_geometry3d::prelude::*;

// ─── PTX kernel benchmarks ───────────────────────────────────────────────────

fn bench_fps_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("fps_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(farthest_point_sample_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_ball_query_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ball_query_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(ball_query_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gather_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gather_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gather_points_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_voxelize_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("voxelize_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(voxelize_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_chamfer_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("chamfer_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(chamfer_distance_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gaussian_project_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gaussian_project_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gaussian_project_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_sh_eval_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("sh_eval_ptx");
    for &sm in &[80_u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(sh_eval_ptx(sm)))
        });
    }
    g.finish();
}

// ─── Algorithm benchmarks ────────────────────────────────────────────────────

fn make_points(n: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut pts = vec![0.0_f32; n * 3];
    for v in &mut pts {
        *v = rng.next_f32() * 2.0 - 1.0;
    }
    pts
}

fn bench_fps_n4096_m512(c: &mut Criterion) {
    let n = 4096;
    let m = 512;
    let mut rng = LcgRng::new(0);
    let pts = make_points(n, &mut rng);
    c.bench_function("fps_n4096_m512", |b| {
        b.iter(|| {
            std::hint::black_box(
                farthest_point_sample(
                    std::hint::black_box(&pts),
                    std::hint::black_box(n),
                    std::hint::black_box(m),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_knn_n2048_k16(c: &mut Criterion) {
    let n = 2048;
    let k = 16;
    let mut rng = LcgRng::new(1);
    let pts = make_points(n, &mut rng);
    let queries = make_points(64, &mut rng);
    c.bench_function("knn_n2048_k16", |b| {
        b.iter(|| {
            std::hint::black_box(
                knn(
                    std::hint::black_box(&queries),
                    std::hint::black_box(64),
                    std::hint::black_box(&pts),
                    std::hint::black_box(n),
                    std::hint::black_box(k),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_chamfer_na1024_nb1024(c: &mut Criterion) {
    let na = 1024;
    let nb = 1024;
    let mut rng = LcgRng::new(2);
    let a = make_points(na, &mut rng);
    let b = make_points(nb, &mut rng);
    c.bench_function("chamfer_na1024_nb1024", |b_bench| {
        b_bench.iter(|| {
            std::hint::black_box(
                chamfer_distance(
                    std::hint::black_box(&a),
                    std::hint::black_box(na),
                    std::hint::black_box(&b),
                    std::hint::black_box(nb),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_pointnet_forward_n512(c: &mut Criterion) {
    let n = 512;
    let mut rng = LcgRng::new(3);
    let cfg = PointNetConfig {
        n_points: n,
        n_classes: 40,
    };
    let net = PointNet::new(cfg, &mut rng);
    let pts = make_points(n, &mut rng);
    c.bench_function("pointnet_forward_n512", |b| {
        b.iter(|| std::hint::black_box(net.forward(std::hint::black_box(&pts)).unwrap()))
    });
}

fn bench_kdtree_build_n4096(c: &mut Criterion) {
    let n = 4096;
    let mut rng = LcgRng::new(4);
    let pts = make_points(n, &mut rng);
    c.bench_function("kdtree_build_n4096", |b| {
        b.iter(|| {
            std::hint::black_box(
                KdTree::build(std::hint::black_box(&pts), std::hint::black_box(n)).unwrap(),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_fps_ptx,
    bench_ball_query_ptx,
    bench_gather_ptx,
    bench_voxelize_ptx,
    bench_chamfer_ptx,
    bench_gaussian_project_ptx,
    bench_sh_eval_ptx,
    bench_fps_n4096_m512,
    bench_knn_n2048_k16,
    bench_chamfer_na1024_nb1024,
    bench_pointnet_forward_n512,
    bench_kdtree_build_n4096,
);
criterion_main!(benches);
