use criterion::{Criterion, criterion_group, criterion_main};

fn bench_ptx_kernels(c: &mut Criterion) {
    let mut g = c.benchmark_group("ann_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("l2_distance_batch_sm{sm}"), |b| {
            b.iter(|| oxicuda_ann::ptx_kernels::l2_distance_batch_ptx(sm))
        });
        g.bench_function(format!("topk_select_sm{sm}"), |b| {
            b.iter(|| oxicuda_ann::ptx_kernels::topk_select_ptx(sm))
        });
    }
    g.finish();
}

fn bench_flat(c: &mut Criterion) {
    use oxicuda_ann::flat::flat::FlatIndex;
    use oxicuda_ann::handle::LcgRng;
    let mut rng = LcgRng::new(42);
    let dim = 64;
    let n = 1000;
    let mut idx = FlatIndex::new(dim);
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        idx.add(&v);
    }
    let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
    let mut g = c.benchmark_group("ann_algo");
    g.bench_function("flat_search_1000x64", |b| {
        b.iter(|| idx.search_l2(&query, 10).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench_ptx_kernels, bench_flat);
criterion_main!(benches);
