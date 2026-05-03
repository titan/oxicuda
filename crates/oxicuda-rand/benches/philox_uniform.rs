//! P7 performance verification harness — Philox 100M uniform F32 samples.
//!
//! Target: ≥ 95 % of cuRAND `curandGenerateUniform` throughput on the same
//! hardware (Vol.5 Sec 9, Performance Requirements).  Reported as
//! `Throughput::Elements(100_000_000)` so criterion prints
//! `<value> Gelem/s` directly comparable to cuRAND.
//!
//! Two scenarios are exercised inside the same group:
//!
//! * `optimized` — `RngGenerator::generate_uniform_f32_optimized` (Philox
//!   4-values-per-thread, grid-stride loop). This is the **headline P7
//!   number** because it is the variant tuned to compete with cuRAND.
//! * `baseline` — `RngGenerator::generate_uniform_f32` (1-value-per-thread).
//!   Useful for ratio-checking the optimization gain.
//!
//! Allocation, generator construction and PTX compilation happen **outside
//! `b.iter`**; only the actual fill call is timed.  The fill call already
//! calls `stream.synchronize()` internally
//! (`generator.rs::compile_and_launch_uniform`), so no extra sync is
//! required for accurate timing.
//!
//! On macOS / no-GPU hosts every bench function early-returns after
//! emitting `skip: no GPU` to stderr — keeping `cargo bench --no-run`
//! green on CI runners that lack a CUDA driver.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_driver::{Context, Device, init};
use oxicuda_memory::DeviceBuffer;
use oxicuda_rand::generator::{RngEngine, RngGenerator};

/// Total samples per draw — fixed at 100M (P7 spec).
const SAMPLES: usize = 100_000_000;

/// RNG seed (deterministic; arbitrary fixed value).
const SEED: u64 = 0x4F78_4943_5544_4101;

/// Build a CUDA context, returning `None` on no-GPU / driver-missing hosts.
///
/// Matches the existing `oxicuda-memory/benches/bandwidth_copy.rs` pattern:
/// `init()` + `Device::count()` + `Device::get(0)` + `Context::new`.
/// Wrapped in `Arc` because `RngGenerator::new` takes `&Arc<Context>`.
fn try_setup_context() -> Option<Arc<Context>> {
    init().ok()?;
    if Device::count().ok()? <= 0 {
        return None;
    }
    let device = Device::get(0).ok()?;
    Some(Arc::new(Context::new(&device).ok()?))
}

/// P7 headline: Philox optimized uniform F32, 100M samples.
fn bench_philox_uniform_optimized(c: &mut Criterion) {
    let ctx = match try_setup_context() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut rng = match RngGenerator::new(RngEngine::Philox, SEED, &ctx) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("skip: RngGenerator::new failed: {err}");
            return;
        }
    };

    let mut buf = match DeviceBuffer::<f32>::alloc(SAMPLES) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip: device allocation of {SAMPLES} f32 failed: {err}");
            return;
        }
    };

    // Warm-up call: ensures the PTX module is JIT-compiled before timing.
    if let Err(err) = rng.generate_uniform_f32_optimized(&mut buf) {
        eprintln!("skip: warm-up generate_uniform_f32_optimized failed: {err}");
        return;
    }

    let mut group = c.benchmark_group("philox_uniform_100m_f32");
    group.throughput(Throughput::Elements(SAMPLES as u64));
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("optimized", SAMPLES),
        &SAMPLES,
        |b, &_n| {
            b.iter(|| {
                if let Err(err) = rng.generate_uniform_f32_optimized(&mut buf) {
                    eprintln!("generate_uniform_f32_optimized failed: {err}");
                }
                black_box(buf.len())
            });
        },
    );

    group.finish();
}

/// Baseline: Philox 1-per-thread uniform F32, 100M samples.
fn bench_philox_uniform_baseline(c: &mut Criterion) {
    let ctx = match try_setup_context() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut rng = match RngGenerator::new(RngEngine::Philox, SEED, &ctx) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("skip: RngGenerator::new failed: {err}");
            return;
        }
    };

    let mut buf = match DeviceBuffer::<f32>::alloc(SAMPLES) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip: device allocation of {SAMPLES} f32 failed: {err}");
            return;
        }
    };

    if let Err(err) = rng.generate_uniform_f32(&mut buf) {
        eprintln!("skip: warm-up generate_uniform_f32 failed: {err}");
        return;
    }

    let mut group = c.benchmark_group("philox_uniform_100m_f32");
    group.throughput(Throughput::Elements(SAMPLES as u64));
    group.sample_size(10);

    group.bench_with_input(BenchmarkId::new("baseline", SAMPLES), &SAMPLES, |b, &_n| {
        b.iter(|| {
            if let Err(err) = rng.generate_uniform_f32(&mut buf) {
                eprintln!("generate_uniform_f32 failed: {err}");
            }
            black_box(buf.len())
        });
    });

    group.finish();
}

criterion_group!(
    philox_uniform_benches,
    bench_philox_uniform_optimized,
    bench_philox_uniform_baseline,
);
criterion_main!(philox_uniform_benches);
