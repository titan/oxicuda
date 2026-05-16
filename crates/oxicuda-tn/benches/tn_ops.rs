use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::mps::mps::Mps;
use oxicuda_tn::ptx_kernels::{
    dmrg_local_apply_ptx, hosvd_unfold_ptx, mpo_apply_ptx, svd_jacobi_step_ptx,
    tensor_contract_ptx, trotter_step_ptx, tt_round_ptx,
};
use oxicuda_tn::svd::svd_jacobi;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("tensor_contract", tensor_contract_ptx),
        ("svd_jacobi_step", svd_jacobi_step_ptx),
        ("dmrg_local_apply", dmrg_local_apply_ptx),
        ("mpo_apply", mpo_apply_ptx),
        ("trotter_step", trotter_step_ptx),
        ("hosvd_unfold", hosvd_unfold_ptx),
        ("tt_round", tt_round_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_svd(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let m = 12;
    let n = 12;
    let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
    c.bench_function("svd_jacobi_12x12", |b| {
        b.iter(|| svd_jacobi(&mat, m, n).expect("ok"))
    });
}

fn bench_mps(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let mps = Mps::random_mps(4, 2, 4, &mut rng).expect("ok");
    c.bench_function("mps_norm_squared_L4_chi4", |b| {
        b.iter(|| mps.norm_squared().expect("ok"))
    });
}

criterion_group!(benches, bench_ptx, bench_svd, bench_mps);
criterion_main!(benches);
