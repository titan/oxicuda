use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_pde::fdm::poisson_1d::{Dirichlet1d, solve_poisson_1d};
use oxicuda_pde::handle::LcgRng;
use oxicuda_pde::mesh::Mesh1d;
use oxicuda_pde::multigrid::vcycle::v_cycle_1d;
use oxicuda_pde::ptx_kernels::{
    cg_axpy_dot_ptx, csr_spmv_ptx, fdm_stencil_5pt_ptx, fem_assemble_ptx, gauss_seidel_step_ptx,
    mg_prolong_ptx, mg_restrict_ptx,
};
use oxicuda_pde::solver::cg::cg_solve;
use oxicuda_pde::solver::sparse::SparseCsr;
use oxicuda_pde::spectral::chebyshev::solve_poisson_chebyshev;
use oxicuda_pde::spectral::fft_spectral::periodic_poisson_solve;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("fdm_stencil_5pt", fdm_stencil_5pt_ptx),
        ("gauss_seidel_step", gauss_seidel_step_ptx),
        ("csr_spmv", csr_spmv_ptx),
        ("cg_axpy_dot", cg_axpy_dot_ptx),
        ("fem_assemble", fem_assemble_ptx),
        ("mg_restrict", mg_restrict_ptx),
        ("mg_prolong", mg_prolong_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_fdm_poisson(c: &mut Criterion) {
    let pi = std::f64::consts::PI;
    let mesh = Mesh1d::uniform(0.0, 1.0, 101).expect("ok");
    let f: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|x| pi * pi * (pi * x).sin())
        .collect();
    c.bench_function("fdm_poisson_1d_n101", |b| {
        b.iter(|| solve_poisson_1d(&mesh, &f, Dirichlet1d { ua: 0.0, ub: 0.0 }).expect("ok"))
    });
}

fn bench_chebyshev_poisson(c: &mut Criterion) {
    let n = 20;
    let x = oxicuda_pde::spectral::chebyshev::cheb_nodes(n);
    let f: Vec<f64> = x.iter().map(|_| 2.0).collect();
    c.bench_function("chebyshev_poisson_n20", |b| {
        b.iter(|| solve_poisson_chebyshev(n, &f).expect("ok"))
    });
}

fn bench_fft_poisson(c: &mut Criterion) {
    let two_pi = std::f64::consts::TAU;
    let n = 64;
    let f: Vec<f64> = (0..n)
        .map(|j| {
            let x = two_pi * j as f64 / n as f64;
            4.0 * (2.0 * x).sin()
        })
        .collect();
    c.bench_function("fft_periodic_poisson_n64", |b| {
        b.iter(|| periodic_poisson_solve(&f, two_pi).expect("ok"))
    });
}

fn bench_cg(c: &mut Criterion) {
    let n = 64;
    let mut rng = LcgRng::new(7);
    let h = 1.0 / (n - 1) as f64;
    let inv_h2 = 1.0 / (h * h);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            cols.push(i - 1);
            vals.push(-inv_h2);
        }
        cols.push(i);
        vals.push(2.0 * inv_h2);
        if i + 1 < n {
            cols.push(i + 1);
            vals.push(-inv_h2);
        }
        row_ptr.push(cols.len());
    }
    let a = SparseCsr::new(n, n, row_ptr, cols, vals).expect("ok");
    let b: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    c.bench_function("cg_solve_n64", |b_bench| {
        b_bench.iter(|| cg_solve(&a, &b, &vec![0.0; n], 200, 1.0e-10).expect("ok"))
    });
}

fn bench_v_cycle(c: &mut Criterion) {
    let n = 65;
    let h = 1.0 / (n - 1) as f64;
    let f = vec![2.0; n];
    c.bench_function("multigrid_v_cycle_n65", |b| {
        b.iter(|| {
            let mut u = vec![0.0; n];
            v_cycle_1d(&mut u, &f, h, 2, 2).expect("ok");
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_fdm_poisson,
    bench_chebyshev_poisson,
    bench_fft_poisson,
    bench_cg,
    bench_v_cycle
);
criterion_main!(benches);
