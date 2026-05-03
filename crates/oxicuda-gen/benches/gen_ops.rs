//! Criterion benchmarks for oxicuda-gen core operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_gen::guidance::cfg::{CfgConfig, CfgGuidance};
use oxicuda_gen::handle::LcgRng;
use oxicuda_gen::lora::adapter::{LoraConfig, LoraLinear};
use oxicuda_gen::scheduler::beta_schedule::BetaSchedule;
use oxicuda_gen::scheduler::ddpm::DdpmScheduler;

fn bench_beta_schedule_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_schedule_linear");
    for steps in [100_usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &n| {
            b.iter(|| BetaSchedule::linear(n, 0.0001, 0.02).unwrap());
        });
    }
    group.finish();
}

fn bench_beta_schedule_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_schedule_cosine");
    for steps in [100_usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &n| {
            b.iter(|| BetaSchedule::cosine(n, 0.008).unwrap());
        });
    }
    group.finish();
}

fn bench_cfg_apply(c: &mut Criterion) {
    let config = CfgConfig::new(7.5).unwrap();
    let guide = CfgGuidance::new(config);
    let mut rng = LcgRng::new(42);
    let mut group = c.benchmark_group("cfg_apply");
    for n in [256_usize, 4096, 65536] {
        let mut cond = vec![0.0_f32; n];
        let mut uncond = vec![0.0_f32; n];
        rng.fill_normal(&mut cond);
        rng.fill_normal(&mut uncond);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| guide.apply(&cond, &uncond).unwrap());
        });
    }
    group.finish();
}

fn bench_ddpm_step(c: &mut Criterion) {
    let sched = DdpmScheduler::new(1000).unwrap();
    let mut rng = LcgRng::new(99);
    let mut group = c.benchmark_group("ddpm_step");
    for n in [256_usize, 4096, 65536] {
        let mut eps = vec![0.0_f32; n];
        let mut x_t = vec![0.0_f32; n];
        let mut noise = vec![0.0_f32; n];
        rng.fill_normal(&mut eps);
        rng.fill_normal(&mut x_t);
        rng.fill_normal(&mut noise);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sched.step(&eps, &x_t, 500, &noise).unwrap());
        });
    }
    group.finish();
}

fn bench_lora_forward(c: &mut Criterion) {
    let config = LoraConfig::new(16, 16.0).unwrap();
    let mut rng = LcgRng::new(7);
    let lora = LoraLinear::new(512, 512, &config, &mut rng).unwrap();
    let mut group = c.benchmark_group("lora_forward");
    for batch in [1_usize, 4, 16] {
        let x = vec![0.1_f32; batch * 512];
        let base = vec![0.0_f32; batch * 512];
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, _| {
            b.iter(|| lora.forward(&x, &base, batch).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_beta_schedule_linear,
    bench_beta_schedule_cosine,
    bench_cfg_apply,
    bench_ddpm_step,
    bench_lora_forward,
);
criterion_main!(benches);
