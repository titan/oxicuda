use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_ot::ptx_kernels::{
    barycenter_update_ptx, cost_matrix_ptx, gromov_grad_ptx, sinkhorn_step_ptx, sliced_proj_ptx,
    transport_apply_ptx, unbalanced_step_ptx,
};
use oxicuda_ot::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ot_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("sinkhorn_step_sm{sm}"), |b| {
            b.iter(|| sinkhorn_step_ptx(sm))
        });
        g.bench_function(format!("cost_matrix_sm{sm}"), |b| {
            b.iter(|| cost_matrix_ptx(sm))
        });
        g.bench_function(format!("transport_apply_sm{sm}"), |b| {
            b.iter(|| transport_apply_ptx(sm))
        });
        g.bench_function(format!("sliced_proj_sm{sm}"), |b| {
            b.iter(|| sliced_proj_ptx(sm))
        });
        g.bench_function(format!("gromov_grad_sm{sm}"), |b| {
            b.iter(|| gromov_grad_ptx(sm))
        });
        g.bench_function(format!("unbalanced_step_sm{sm}"), |b| {
            b.iter(|| unbalanced_step_ptx(sm))
        });
        g.bench_function(format!("barycenter_update_sm{sm}"), |b| {
            b.iter(|| barycenter_update_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("ot_algo");
    let m = 16_usize;
    let n = 16_usize;
    let cost: Vec<f32> = (0..m * n)
        .map(|k| {
            let i = (k / n) as f32;
            let j = (k % n) as f32;
            (i - j).powi(2)
        })
        .collect();
    let a = vec![1.0_f32 / m as f32; m];
    let b = vec![1.0_f32 / n as f32; n];
    let cfg = SinkhornConfig {
        eps: 0.5,
        max_iter: 200,
        tol: 1e-4,
    };
    g.bench_function("sinkhorn_16x16_eps05", |bx| {
        bx.iter(|| {
            let _ = sinkhorn(&cost, &a, &b, m, n, &cfg);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
