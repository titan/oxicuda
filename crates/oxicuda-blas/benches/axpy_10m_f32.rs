//! AXPY 10M F32 benchmark — target: P6 ≥ 95% cuBLAS throughput.
//!
//! Computes `y = alpha * x + y` over `10_000_000` single-precision
//! elements. AXPY is purely memory-bound on every contemporary GPU —
//! 2 reads (`x[i]`, `y[i]`) + 1 write (`y[i]`) per element — so the
//! relevant figure of merit is bandwidth, not flops.
//!
//! Throughput is reported as `Throughput::Bytes(N * 3 * 4)`: three F32
//! transfers per element (two reads + one write) at 4 bytes each.
//!
//! ## Memory budget
//!
//! Two F32 buffers of `10_000_000 × 4 B ≈ 38 MiB` each → ~77 MiB
//! resident. Fits on any contemporary GPU.
//!
//! ## Platform behaviour
//!
//! * **Linux / Windows with NVIDIA driver 525+** — full benchmark runs.
//! * **macOS / no GPU / no driver** — bench function returns immediately
//!   after `eprintln!("skip: no GPU")`. The skip guard is the **first**
//!   statement of every bench function in this file.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-blas --bench axpy_10m_f32
//! ```
//!
//! Verifies P-gate `P6` (axpy, 10M F32 elements) once executed on
//! Linux+NVIDIA hardware.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level1::axpy;
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// Number of F32 elements in the working vectors.
const N: u32 = 10_000_000;

/// Bytes transferred per AXPY call: 2 reads + 1 write per element × 4 B.
const BYTES_PER_CALL: u64 = (N as u64) * 3u64 * (std::mem::size_of::<f32>() as u64);

/// GPU resources shared across iterations of the criterion loop.
struct AxpyHarness {
    /// Owns the CUDA context for the lifetime of all device buffers and the
    /// `BlasHandle`.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Source vector `x`.
    x: DeviceBuffer<f32>,
    /// Destination vector `y` (modified in-place each iteration).
    y: DeviceBuffer<f32>,
}

/// Initialise the driver, allocate buffers, and prepare a `BlasHandle`.
///
/// Returns `None` on any platform without a usable CUDA driver (macOS,
/// CI without a GPU, etc.). The caller must skip the benchmark in that case.
fn try_setup() -> Option<AxpyHarness> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;

    let elements = N as usize;

    // Deterministic host-side fill — small magnitudes keep the running
    // `y = alpha * x + y` well within F32 precision over many iterations.
    let host_x: Vec<f32> = (0..elements).map(|i| (i as f32) * 1.0e-7).collect();
    let host_y: Vec<f32> = (0..elements).map(|i| (i as f32) * 1.0e-8).collect();

    let x = DeviceBuffer::<f32>::from_host(&host_x).ok()?;
    let y = DeviceBuffer::<f32>::from_host(&host_y).ok()?;

    Some(AxpyHarness {
        _ctx: ctx,
        handle,
        x,
        y,
    })
}

/// Measures steady-state AXPY bandwidth on 10 M F32 elements.
fn bench_axpy_10m_f32(c: &mut Criterion) {
    let mut harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    // A small but non-trivial alpha so the kernel actually fires (alpha == 0
    // short-circuits to a no-op in `axpy`).
    let alpha: f32 = 1.0e-4;

    let mut group = c.benchmark_group("axpy_10m_f32");
    group.throughput(Throughput::Bytes(BYTES_PER_CALL));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = axpy::<f32>(
                black_box(&harness.handle),
                N,
                alpha,
                black_box(&harness.x),
                1,
                &mut harness.y,
                1,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_axpy_10m_f32);
criterion_main!(benches);
