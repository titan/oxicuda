use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_survival::aft::{AftFamily, fit_aft};
use oxicuda_survival::calibration::brier_score::brier_score_at;
use oxicuda_survival::concordance::harrell_c_index;
use oxicuda_survival::cox::{CoxPhConfig, fit_cox_ph};
use oxicuda_survival::data::{Dataset, Observation};
use oxicuda_survival::handle::LcgRng;
use oxicuda_survival::nonparametric::kaplan_meier_estimate;
use oxicuda_survival::ptx_kernels::{
    brier_score_ptx, cox_info_ptx, cox_risk_sum_ptx, cox_score_ptx, km_step_ptx, logrank_oe_ptx,
    rmst_integrate_ptx,
};
use oxicuda_survival::rmst::rmst_from_dataset;
use oxicuda_survival::test::log_rank_test;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("km_step", km_step_ptx),
        ("cox_risk_sum", cox_risk_sum_ptx),
        ("cox_score", cox_score_ptx),
        ("cox_info", cox_info_ptx),
        ("logrank_oe", logrank_oe_ptx),
        ("brier_score", brier_score_ptx),
        ("rmst_integrate", rmst_integrate_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn synthetic_dataset(n: usize, beta_true: f64, seed: u64) -> Dataset {
    let mut rng = LcgRng::new(seed);
    let mut obs = Vec::with_capacity(n);
    let mut cov = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.next_normal();
        let lambda = (beta_true * x).exp();
        let t = rng.next_exponential(lambda).max(1.0e-6);
        obs.push(Observation::new(t, true).expect("ok"));
        cov.push(vec![x]);
    }
    Dataset::new(obs, Some(cov), None).expect("ok")
}

fn bench_kaplan_meier(c: &mut Criterion) {
    let d = synthetic_dataset(200, 0.5, 7);
    c.bench_function("km_n200", |b| {
        b.iter(|| kaplan_meier_estimate(&d).expect("ok"))
    });
}

fn bench_cox_fit(c: &mut Criterion) {
    let d = synthetic_dataset(150, 0.5, 11);
    c.bench_function("cox_n150", |b| {
        b.iter(|| fit_cox_ph(&d, CoxPhConfig::default()).expect("ok"))
    });
}

fn bench_log_rank(c: &mut Criterion) {
    let d = synthetic_dataset(120, 0.0, 3);
    let mut rng = LcgRng::new(99);
    let groups: Vec<usize> = (0..120)
        .map(|_| if rng.next_bool() { 1 } else { 0 })
        .collect();
    c.bench_function("logrank_n120", |b| {
        b.iter(|| log_rank_test(&d, &groups).expect("ok"))
    });
}

fn bench_brier(c: &mut Criterion) {
    let d = synthetic_dataset(100, 0.0, 5);
    let s_pred = vec![0.5; 100];
    c.bench_function("brier_n100", |b| {
        b.iter(|| brier_score_at(&d, &s_pred, 1.0).expect("ok"))
    });
}

fn bench_c_index(c: &mut Criterion) {
    let d = synthetic_dataset(80, 0.5, 17);
    let eta: Vec<f64> = (0..80).map(|i| 0.01 * i as f64).collect();
    c.bench_function("c_index_n80", |b| {
        b.iter(|| harrell_c_index(&d, &eta).expect("ok"))
    });
}

fn bench_rmst(c: &mut Criterion) {
    let d = synthetic_dataset(100, 0.0, 21);
    c.bench_function("rmst_n100", |b| {
        b.iter(|| rmst_from_dataset(&d, 1.0).expect("ok"))
    });
}

fn bench_aft_weibull(c: &mut Criterion) {
    let d = synthetic_dataset(100, 0.0, 23);
    c.bench_function("weibull_fit_n100", |b| {
        b.iter(|| fit_aft(&d, AftFamily::Weibull).expect("ok"))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_kaplan_meier,
    bench_cox_fit,
    bench_log_rank,
    bench_brier,
    bench_c_index,
    bench_rmst,
    bench_aft_weibull
);
criterion_main!(benches);
