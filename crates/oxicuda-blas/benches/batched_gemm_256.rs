//! Strided batched GEMM 1000 × (256³) benchmark — target: P4 ≥ 90% cuBLAS.
//!
//! Computes `D[i] = alpha * A[i] * B[i] + beta * C[i]` for `i ∈ 0..1000`
//! with each per-batch matrix `M = N = K = 256`. The `gemm_strided_batched`
//! entry point places consecutive matrices at a fixed element stride so a
//! single device allocation backs the whole batch.
//!
//! Throughput in TFLOPS is reported via the
//! `Throughput::Elements(BATCH * 2 * M * N * K)` annotation.
//!
//! ## Memory budget
//!
//! Four F32 buffers of `1000 × 256 × 256 × 4 B = 256 MiB` each → ~1 GiB
//! total resident. Comfortably fits on A100/H100; small consumer GPUs may
//! be tight if the harness shares the device with other processes.
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
//! cargo bench -p oxicuda-blas --bench batched_gemm_256
//! ```
//!
//! Verifies P-gate `P4` (Batched GEMM, 1000 × 256³) once executed on
//! Linux+NVIDIA hardware.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_blas::batched::gemm_strided_batched;
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::types::Transpose;
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// Per-matrix edge length.
const M: u32 = 256;
/// Per-matrix edge length.
const N: u32 = 256;
/// Per-matrix edge length.
const K: u32 = 256;
/// Number of independent matrices in the batch.
const BATCH: u32 = 1000;

/// Total flop count for one full batched GEMM (BATCH · 2 · M · N · K).
const FLOPS_PER_BATCH: u64 = (BATCH as u64) * 2u64 * (M as u64) * (N as u64) * (K as u64);

/// GPU resources shared across iterations of the criterion loop.
struct BatchedHarness {
    /// Owns the CUDA context for the lifetime of all device buffers.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Backing storage for every A[i] (laid out at element stride `M*K`).
    a_buf: DeviceBuffer<f32>,
    /// Backing storage for every B[i] (laid out at element stride `K*N`).
    b_buf: DeviceBuffer<f32>,
    /// Backing storage for every C[i] (laid out at element stride `M*N`).
    c_buf: DeviceBuffer<f32>,
    /// Backing storage for every D[i] (laid out at element stride `M*N`).
    d_buf: DeviceBuffer<f32>,
}

/// Initialise the driver, allocate all matrices, and prepare a `BlasHandle`.
///
/// Returns `None` on any platform without a usable CUDA driver (macOS,
/// CI without a GPU, etc.). The caller must skip the benchmark in that case.
fn try_setup() -> Option<BatchedHarness> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;

    let mat_elements = (M as usize) * (K as usize); // matches all four shapes.
    let total_elements = mat_elements * (BATCH as usize);

    // Deterministic host-side fill, small magnitudes to keep the F32
    // accumulator within range across the 256-element-per-row reduction.
    let host_a: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 1.0e-6).collect();
    let host_b: Vec<f32> = (0..total_elements)
        .map(|i| ((i % 256) as f32) * 1.0e-6)
        .collect();

    let a_buf = DeviceBuffer::<f32>::from_host(&host_a).ok()?;
    let b_buf = DeviceBuffer::<f32>::from_host(&host_b).ok()?;
    let c_buf = DeviceBuffer::<f32>::zeroed(total_elements).ok()?;
    let d_buf = DeviceBuffer::<f32>::zeroed(total_elements).ok()?;

    Some(BatchedHarness {
        _ctx: ctx,
        handle,
        a_buf,
        b_buf,
        c_buf,
        d_buf,
    })
}

/// Measures steady-state strided batched GEMM throughput.
fn bench_batched_gemm_256(c: &mut Criterion) {
    let harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    // Pre-compute element strides (matrix sizes, in elements).
    let stride_a: i64 = (M as i64) * (K as i64);
    let stride_b: i64 = (K as i64) * (N as i64);
    let stride_c: i64 = (M as i64) * (N as i64);
    let stride_d: i64 = (M as i64) * (N as i64);

    // Leading dimensions for row-major-style storage where each tile is
    // tightly packed with the major dimension equal to `M`.
    let lda: u32 = M;
    let ldb: u32 = K;
    let ldc: u32 = M;
    let ldd: u32 = M;

    let a_ptr = harness.a_buf.as_device_ptr();
    let b_ptr = harness.b_buf.as_device_ptr();
    let c_ptr = harness.c_buf.as_device_ptr();
    let d_ptr = harness.d_buf.as_device_ptr();

    let mut group = c.benchmark_group("batched_gemm_256");
    group.throughput(Throughput::Elements(FLOPS_PER_BATCH));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = gemm_strided_batched::<f32>(
                black_box(&harness.handle),
                Transpose::NoTrans,
                Transpose::NoTrans,
                M,
                N,
                K,
                1.0_f32,
                a_ptr,
                lda,
                stride_a,
                b_ptr,
                ldb,
                stride_b,
                0.0_f32,
                c_ptr,
                ldc,
                stride_c,
                d_ptr,
                ldd,
                stride_d,
                BATCH,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_batched_gemm_256);
criterion_main!(benches);
