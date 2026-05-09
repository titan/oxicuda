use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_causal::ptx_kernels;

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("causal_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("notears_loss_sm{sm}"), |b| {
            b.iter(|| ptx_kernels::notears_loss_ptx(sm))
        });
        g.bench_function(format!("dml_residual_sm{sm}"), |b| {
            b.iter(|| ptx_kernels::dml_residual_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    use oxicuda_causal::dag::dag::Dag;
    let mut g = c.benchmark_group("causal_algo");
    g.bench_function("dag_topo_sort_100", |b| {
        b.iter(|| {
            let mut dag = Dag::new(100);
            for i in 0..99 {
                dag.add_edge(i, i + 1).unwrap();
            }
            dag.topo_sort().unwrap()
        })
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
