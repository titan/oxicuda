//! Extended algorithm benchmarks: NSGA-II on the standard ZDT and DTLZ multi-objective
//! test problems.
//!
//! Each benchmark drives `run_nsga2_benchmark` (real-coded SBX + polynomial mutation,
//! fast non-dominated sort, crowding-distance selection) end to end on a canonical
//! analytic problem and reports the per-run wall-clock cost. The deterministic `LcgRng`
//! seed makes every sample reproducible. Front-quality convergence (GD / IGD / hypervolume
//! against the analytic Pareto front) is asserted separately in the crate's `#[test]` suite
//! (`benchmarks::bbob::tests::nsga2_*`).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_evol::{dtlz1, dtlz2, run_nsga2_benchmark, zdt1, zdt2, zdt3};

fn bench_nsga2_mo(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga2_mo");
    // NSGA-II runs are heavier than the PTX string benches; keep the sample count modest.
    group.sample_size(20);

    // ── Two-objective ZDT family (decision space [0,1]^10) ───────────────────
    group.bench_function("zdt1_10d", |b| {
        b.iter(|| {
            black_box(
                run_nsga2_benchmark(zdt1, 10, 2, 40, 60, 0x2DA1, "zdt1", vec![1.1, 1.1])
                    .expect("zdt1 bench"),
            )
        })
    });
    group.bench_function("zdt2_10d", |b| {
        b.iter(|| {
            black_box(
                run_nsga2_benchmark(zdt2, 10, 2, 40, 60, 0x5AE2, "zdt2", vec![1.1, 1.1])
                    .expect("zdt2 bench"),
            )
        })
    });
    group.bench_function("zdt3_10d", |b| {
        b.iter(|| {
            black_box(
                run_nsga2_benchmark(zdt3, 10, 2, 40, 60, 0x3D73, "zdt3", vec![1.1, 1.5])
                    .expect("zdt3 bench"),
            )
        })
    });

    // ── Three-objective DTLZ family (decision space [0,1]^n) ──────────────────
    group.bench_function("dtlz1_6d", |b| {
        b.iter(|| {
            black_box(
                run_nsga2_benchmark(
                    dtlz1,
                    6,
                    3,
                    60,
                    60,
                    0xD711,
                    "dtlz1",
                    vec![500.0, 500.0, 500.0],
                )
                .expect("dtlz1 bench"),
            )
        })
    });
    group.bench_function("dtlz2_12d", |b| {
        b.iter(|| {
            black_box(
                run_nsga2_benchmark(dtlz2, 12, 3, 60, 60, 0xD712, "dtlz2", vec![2.0, 2.0, 2.0])
                    .expect("dtlz2 bench"),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_nsga2_mo);
criterion_main!(benches);
