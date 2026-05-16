use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_stats::descriptive::summary::{mean, sample_var};
use oxicuda_stats::distributions::normal::Normal;
use oxicuda_stats::distributions::student_t::StudentT;
use oxicuda_stats::handle::LcgRng;
use oxicuda_stats::parametric::t_test::two_sample_t;
use oxicuda_stats::ptx_kernels::{
    bootstrap_resample_ptx, chi2_cell_ptx, histogram_bin_ptx, lr_normal_eq_ptx, mean_var_ptx,
    permute_labels_ptx, rank_assign_ptx,
};
use oxicuda_stats::regression::linear::ols;
use oxicuda_stats::resampling::bootstrap::bootstrap;
use oxicuda_stats::special::betainc::betainc;
use oxicuda_stats::special::erf::erf;
use oxicuda_stats::special::gammaln::lgamma;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("mean_var", mean_var_ptx),
        ("rank_assign", rank_assign_ptx),
        ("histogram_bin", histogram_bin_ptx),
        ("bootstrap_resample", bootstrap_resample_ptx),
        ("permute_labels", permute_labels_ptx),
        ("chi2_cell", chi2_cell_ptx),
        ("lr_normal_eq", lr_normal_eq_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_special(c: &mut Criterion) {
    c.bench_function("erf_pointwise", |b| b.iter(|| erf(1.5)));
    c.bench_function("lgamma_pointwise", |b| b.iter(|| lgamma(10.5)));
    c.bench_function("betainc_pointwise", |b| {
        b.iter(|| betainc(2.5, 4.0, 0.3).expect("ok"))
    });
}

fn bench_distributions(c: &mut Criterion) {
    let n = Normal::standard();
    c.bench_function("normal_cdf", |b| b.iter(|| n.cdf(1.5)));
    c.bench_function("normal_ppf", |b| b.iter(|| n.ppf(0.95).expect("ok")));
    let t = StudentT::new(15.0).expect("ok");
    c.bench_function("student_t_cdf_df15", |b| b.iter(|| t.cdf(1.5).expect("ok")));
}

fn bench_descriptive(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let data: Vec<f64> = (0..1000).map(|_| rng.next_normal()).collect();
    c.bench_function("mean_n1000", |b| b.iter(|| mean(&data).expect("ok")));
    c.bench_function("sample_var_n1000", |b| {
        b.iter(|| sample_var(&data).expect("ok"))
    });
}

fn bench_parametric(c: &mut Criterion) {
    let mut rng = LcgRng::new(11);
    let x1: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
    let x2: Vec<f64> = (0..200).map(|_| rng.next_normal() + 0.5).collect();
    c.bench_function("two_sample_t_n200", |b| {
        b.iter(|| two_sample_t(&x1, &x2).expect("ok"))
    });
}

fn bench_regression(c: &mut Criterion) {
    let mut rng = LcgRng::new(3);
    let n = 100;
    let p = 3;
    let mut x_mat = vec![0.0; n * p];
    let mut y = vec![0.0; n];
    for i in 0..n {
        x_mat[i * p] = 1.0;
        let v1 = rng.next_normal();
        let v2 = rng.next_normal();
        x_mat[i * p + 1] = v1;
        x_mat[i * p + 2] = v2;
        y[i] = 1.0 + 2.0 * v1 - 0.5 * v2 + 0.1 * rng.next_normal();
    }
    c.bench_function("ols_n100_p3", |b| {
        b.iter(|| ols(&x_mat, &y, n, p).expect("ok"))
    });
}

fn bench_bootstrap(c: &mut Criterion) {
    let data: Vec<f64> = (1..=50).map(|v| v as f64).collect();
    c.bench_function("bootstrap_mean_b100_n50", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(0);
            bootstrap(&data, 100, 0.95, mean, &mut rng).expect("ok")
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_special,
    bench_distributions,
    bench_descriptive,
    bench_parametric,
    bench_regression,
    bench_bootstrap
);
criterion_main!(benches);
