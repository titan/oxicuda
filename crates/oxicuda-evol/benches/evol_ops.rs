use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_evol::handle::LcgRng;
use oxicuda_evol::ptx_kernels::{
    cmaes_sample_ptx, de_mutate_ptx, fitness_eval_ptx, gaussian_mutate_ptx, nsga_crowding_ptx,
    pso_update_ptx, tournament_select_ptx,
};

fn bench_ptx(c: &mut Criterion) {
    type KernelFn = fn(u32) -> String;
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[(&str, KernelFn)] = &[
        ("fitness_eval", fitness_eval_ptx),
        ("tournament_select", tournament_select_ptx),
        ("gaussian_mutate", gaussian_mutate_ptx),
        ("nsga_crowding", nsga_crowding_ptx),
        ("pso_update", pso_update_ptx),
        ("de_mutate", de_mutate_ptx),
        ("cmaes_sample", cmaes_sample_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_cmaes_sphere(c: &mut Criterion) {
    use oxicuda_evol::evolution::cmaes::{CmaEsConfig, CmaEsState};
    c.bench_function("cmaes_sphere_5d", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(42);
            let cfg = CmaEsConfig::new(5).expect("ok");
            let mut state = CmaEsState::new(vec![2.0; 5], &cfg).expect("ok");
            let _ = state.run(|x| x.iter().map(|v| v * v).sum::<f64>(), &cfg, &mut rng);
        })
    });
}

criterion_group!(benches, bench_ptx, bench_cmaes_sphere);
criterion_main!(benches);
