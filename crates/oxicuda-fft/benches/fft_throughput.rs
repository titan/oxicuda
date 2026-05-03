//! FFT throughput benchmarks for OxiCUDA blueprint targets P1-P2.
//!
//! Measures end-to-end FFT execution throughput in GFLOPS for two
//! Vol.5 Sec 9 performance gates:
//!
//! | Target | Workload                | Acceptance        |
//! |--------|-------------------------|-------------------|
//! | P1     | 1-D C2C, N = 2^20       | >= 90% cuFFT      |
//! | P2     | 2-D C2C, 1024 x 1024    | >= 85% cuFFT      |
//!
//! Throughput formula (Cooley-Tukey FLOP count):
//! `GFLOPS = 5 * N * log2(N) / time_seconds / 1e9`
//!
//! On hosts without an NVIDIA GPU (e.g. macOS CI, no-driver Linux),
//! the benches print `skip: no GPU` and return successfully so the
//! workspace `cargo bench --no-run` continues to compile.
//!
//! Verification of P1/P2 thresholds requires a Linux + NVIDIA driver
//! 525+ host; the harness records the wall-clock time per execute call
//! with the plan creation amortised outside the iter loop and a
//! `Stream::synchronize` at the end of every iteration so criterion
//! captures actual GPU completion (not just the async dispatch).
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex32;
use oxicuda_driver::{Context, Device, init};
use oxicuda_fft::prelude::*;
use oxicuda_memory::DeviceBuffer;
use oxicuda_memory::copy::copy_htod;

/// 1-D C2C target size: N = 2^20 = 1,048,576 complex elements.
const N_1D: usize = 1 << 20;

/// 2-D C2C target dimensions: 1024 x 1024 complex elements.
const N_2D_X: usize = 1024;
const N_2D_Y: usize = 1024;

/// Initialises CUDA, picks device 0, and constructs an FFT handle.
///
/// Returns `None` on any failure (no driver, no device, context init
/// error, etc.) so that the bench can issue a friendly skip message
/// without polluting CI logs with stack traces.
fn setup_handle() -> Option<(Arc<Context>, FftHandle)> {
    init().ok()?;
    if Device::count().ok()? <= 0 {
        return None;
    }
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = FftHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

/// Builds a deterministic complex input vector of `n` elements.
///
/// Uses a simple non-random pattern so successive bench runs operate
/// on identical data and any GFLOPS variance is attributable to the
/// device, not the input distribution.
fn deterministic_input(n: usize) -> Vec<Complex32> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let re = ((i % 257) as f32) * 0.001 - 0.128;
        let im = ((i % 263) as f32) * 0.001 - 0.131;
        data.push(Complex32::new(re, im));
    }
    data
}

/// Bench: 1-D C2C FFT, N = 2^20.
///
/// Records GFLOPS using the Cooley-Tukey count `5 * N * log2(N)` per
/// transform. The plan and device buffers are created once; only the
/// `execute` + `synchronize` round-trip is timed.
fn bench_fft_1d_c2c_2_20(c: &mut Criterion) {
    let (_ctx, handle) = match setup_handle() {
        Some(x) => x,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let plan = match FftPlan::new_1d(N_1D, FftType::C2C, 1) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: plan creation failed: {e}");
            return;
        }
    };

    let host_input = deterministic_input(N_1D);

    let mut device_input = match DeviceBuffer::<Complex32>::alloc(N_1D) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("skip: device alloc input failed: {e}");
            return;
        }
    };
    let device_output = match DeviceBuffer::<Complex32>::alloc(N_1D) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("skip: device alloc output failed: {e}");
            return;
        }
    };

    if let Err(e) = copy_htod(&mut device_input, host_input.as_slice()) {
        eprintln!("skip: htod copy failed: {e}");
        return;
    }

    let in_ptr = device_input.as_device_ptr();
    let out_ptr = device_output.as_device_ptr();

    let mut group = c.benchmark_group("fft_1d_c2c_2_20");
    group.sample_size(20);
    // FLOP count for one C2C of length N = 5 * N * log2(N).
    let flops_per_iter: u64 = 5u64
        .saturating_mul(N_1D as u64)
        .saturating_mul((N_1D as u64).trailing_zeros() as u64);
    group.throughput(Throughput::Elements(flops_per_iter));

    group.bench_function("c2c_n_1048576", |b| {
        b.iter(|| {
            if handle
                .execute(
                    &plan,
                    black_box(in_ptr),
                    black_box(out_ptr),
                    FftDirection::Forward,
                )
                .is_err()
            {
                return;
            }
            // Ensure the GPU work has completed before stopping the
            // criterion timer; otherwise we measure dispatch only.
            let _ = handle.stream().synchronize();
        });
    });

    group.finish();
}

/// Bench: 2-D C2C FFT, 1024 x 1024.
///
/// Records GFLOPS using `5 * Nrows * Ncols * log2(Nrows*Ncols)` per
/// transform. Plan and buffers are created once; only `execute` +
/// `synchronize` is timed.
fn bench_fft_2d_1024(c: &mut Criterion) {
    let (_ctx, handle) = match setup_handle() {
        Some(x) => x,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let plan = match FftPlan::new_2d(N_2D_X, N_2D_Y, FftType::C2C, 1) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: plan creation failed: {e}");
            return;
        }
    };

    let total = N_2D_X * N_2D_Y;
    let host_input = deterministic_input(total);

    let mut device_input = match DeviceBuffer::<Complex32>::alloc(total) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("skip: device alloc input failed: {e}");
            return;
        }
    };
    let device_output = match DeviceBuffer::<Complex32>::alloc(total) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("skip: device alloc output failed: {e}");
            return;
        }
    };

    if let Err(e) = copy_htod(&mut device_input, host_input.as_slice()) {
        eprintln!("skip: htod copy failed: {e}");
        return;
    }

    let in_ptr = device_input.as_device_ptr();
    let out_ptr = device_output.as_device_ptr();

    let mut group = c.benchmark_group("fft_2d_1024");
    group.sample_size(20);
    // FLOP count for one 2-D C2C of size Nx*Ny:
    //   5 * Nx * Ny * log2(Nx*Ny)
    let log2_total = (total as u64).trailing_zeros() as u64;
    let flops_per_iter: u64 = 5u64.saturating_mul(total as u64).saturating_mul(log2_total);
    group.throughput(Throughput::Elements(flops_per_iter));

    group.bench_function("c2c_1024x1024", |b| {
        b.iter(|| {
            if handle
                .execute(
                    &plan,
                    black_box(in_ptr),
                    black_box(out_ptr),
                    FftDirection::Forward,
                )
                .is_err()
            {
                return;
            }
            let _ = handle.stream().synchronize();
        });
    });

    group.finish();
}

criterion_group!(
    fft_throughput_benches,
    bench_fft_1d_c2c_2_20,
    bench_fft_2d_1024
);
criterion_main!(fft_throughput_benches);
