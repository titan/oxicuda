use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_geom2d::closest_pair::brute_force::closest_pair_brute;
use oxicuda_geom2d::closest_pair::divide_conquer::closest_pair_dc;
use oxicuda_geom2d::enclosing::welzl_smallest_circle::welzl_smallest_circle;
use oxicuda_geom2d::handle::LcgRng;
use oxicuda_geom2d::hull::andrew_monotone_chain::andrew_monotone_chain;
use oxicuda_geom2d::hull::graham_scan::graham_scan;
use oxicuda_geom2d::hull::quickhull::quickhull;
use oxicuda_geom2d::index::kd_tree_2d::KdTree2d;
use oxicuda_geom2d::primitives::point::Point;
use oxicuda_geom2d::ptx_kernels::{
    convex_hull_step_ptx, cross_product_ptx, kd_tree_traverse_ptx, orientation_test_ptx,
    point_in_aabb_ptx, polygon_area_ptx, segment_intersection_ptx,
};
use oxicuda_geom2d::triangulation::bowyer_watson_delaunay::bowyer_watson;

type KernelEntry = (&'static str, fn(u32) -> String);

fn random_points(n: usize, seed: u64) -> Vec<Point> {
    let mut r = LcgRng::new(seed);
    (0..n)
        .map(|_| Point::new(r.next_f64() * 100.0, r.next_f64() * 100.0))
        .collect()
}

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("orientation_test", orientation_test_ptx),
        ("cross_product", cross_product_ptx),
        ("point_in_aabb", point_in_aabb_ptx),
        ("segment_intersection", segment_intersection_ptx),
        ("convex_hull_step", convex_hull_step_ptx),
        ("kd_tree_traverse", kd_tree_traverse_ptx),
        ("polygon_area", polygon_area_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_hull(c: &mut Criterion) {
    let pts = random_points(100, 7);
    c.bench_function("graham_scan_100", |b| {
        b.iter(|| graham_scan(&pts).expect("ok"))
    });
    c.bench_function("andrew_monotone_chain_100", |b| {
        b.iter(|| andrew_monotone_chain(&pts).expect("ok"))
    });
    c.bench_function("quickhull_100", |b| b.iter(|| quickhull(&pts).expect("ok")));
}

fn bench_closest_pair(c: &mut Criterion) {
    let pts = random_points(200, 1);
    c.bench_function("closest_pair_brute_200", |b| {
        b.iter(|| closest_pair_brute(&pts).expect("ok"))
    });
    c.bench_function("closest_pair_dc_200", |b| {
        b.iter(|| closest_pair_dc(&pts).expect("ok"))
    });
}

fn bench_delaunay(c: &mut Criterion) {
    let pts = random_points(50, 3);
    c.bench_function("bowyer_watson_50", |b| {
        b.iter(|| bowyer_watson(&pts).expect("ok"))
    });
}

fn bench_welzl(c: &mut Criterion) {
    let pts = random_points(100, 11);
    c.bench_function("welzl_smallest_circle_100", |b| {
        b.iter(|| welzl_smallest_circle(&pts, 0).expect("ok"))
    });
}

fn bench_kdtree(c: &mut Criterion) {
    let pts = random_points(500, 5);
    let kd = KdTree2d::build(&pts);
    c.bench_function("kdtree_build_500", |b| b.iter(|| KdTree2d::build(&pts)));
    c.bench_function("kdtree_knn5_500", |b| {
        b.iter(|| kd.knn(Point::new(50.0, 50.0), 5))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_hull,
    bench_closest_pair,
    bench_delaunay,
    bench_welzl,
    bench_kdtree
);
criterion_main!(benches);
