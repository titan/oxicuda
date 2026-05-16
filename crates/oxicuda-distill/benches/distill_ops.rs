use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_distill::logit::hinton_kd::{HintonKdConfig, kd_loss};
use oxicuda_distill::ptx_kernels::{gram_matrix_ptx, kd_loss_ptx};

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("distill_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("kd_loss_sm{sm}"), |b| b.iter(|| kd_loss_ptx(sm)));
        g.bench_function(format!("gram_matrix_sm{sm}"), |b| {
            b.iter(|| gram_matrix_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("distill_algo");
    let cfg = HintonKdConfig {
        temperature: 4.0,
        alpha: 0.5,
    };
    let s: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let t: Vec<f32> = vec![1.1, 1.9, 3.1, 4.2, 4.9];
    g.bench_function("hinton_kd_loss_5class", |b| {
        b.iter(|| kd_loss(&s, &t, 2, &cfg).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
