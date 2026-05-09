use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_recsys::handle::LcgRng;
use oxicuda_recsys::ptx_kernels;

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("recsys_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("als_step_sm{sm}"), |b| {
            b.iter(|| ptx_kernels::als_step_ptx(sm))
        });
        g.bench_function(format!("dot_score_sm{sm}"), |b| {
            b.iter(|| ptx_kernels::dot_score_ptx(sm))
        });
    }
    g.finish();
}

fn bench_metrics(c: &mut Criterion) {
    use oxicuda_recsys::metrics::recsys_metrics::ndcg_at_k;
    use std::collections::HashSet;
    let recommended: Vec<usize> = (0..100).collect();
    let relevant: HashSet<usize> = (0..10).collect();
    let mut g = c.benchmark_group("recsys_algo");
    g.bench_function("ndcg_at_10", |b| {
        b.iter(|| ndcg_at_k(&recommended, &relevant, 10))
    });
    g.finish();
}

fn bench_rng(c: &mut Criterion) {
    let mut g = c.benchmark_group("recsys_rng");
    g.bench_function("lcg_next_f32_1000", |b| {
        let mut rng = LcgRng::new(42);
        b.iter(|| {
            let mut sum = 0.0_f32;
            for _ in 0..1000 {
                sum += rng.next_f32();
            }
            sum
        })
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_metrics, bench_rng);
criterion_main!(benches);
