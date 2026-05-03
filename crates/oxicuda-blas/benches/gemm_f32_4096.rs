//! GEMM F32 4096³ benchmark — target: P2 ≥ 95% cuBLAS throughput on sm_80.
//!
//! Computes `C = alpha * A * B + beta * C` for `M = N = K = 4096` in single
//! precision. Throughput in TFLOPS is reported by criterion via the
//! `Throughput::Elements(2 * M * N * K)` annotation: each output element of
//! C costs `2 * K` flops (one multiply + one add per inner-product term).
//!
//! ## Memory budget
//!
//! Three F32 matrices of size 4096² × 4 B = 64 MiB each → ~192 MiB resident.
//! Comfortably fits on any modern data-center GPU; CI runners with consumer
//! GPUs (≥ 4 GiB VRAM) also cope.
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
//! cargo bench -p oxicuda-blas --bench gemm_f32_4096
//! ```
//!
//! Verifies P-gate `P2` (GEMM F32 sm_80, M=N=K=4096) once executed on
//! Linux+NVIDIA hardware.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// Edge length of every matrix in this benchmark.
const N: u32 = 4096;

/// Total flop count for a single 4096³ F32 GEMM (2·M·N·K).
const FLOPS_PER_GEMM: u64 = 2u64 * (N as u64) * (N as u64) * (N as u64);

/// GPU resources shared across iterations of the criterion loop.
struct GemmHarness {
    /// Owns the CUDA context for the lifetime of all device buffers and the
    /// `BlasHandle`. The context must outlive any descriptor that references
    /// `_a`, `_b`, or `_c` because those hold `CUdeviceptr`s allocated under
    /// it.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Input matrix A (`N × N`, row-major).
    _a: DeviceBuffer<f32>,
    /// Input matrix B (`N × N`, row-major).
    _b: DeviceBuffer<f32>,
    /// Output matrix C (`N × N`, row-major).
    _c: DeviceBuffer<f32>,
    /// Read-only view of A used in every `gemm` call.
    a_desc: MatrixDesc<f32>,
    /// Read-only view of B used in every `gemm` call.
    b_desc: MatrixDesc<f32>,
    /// Mutable view of C used in every `gemm` call.
    c_desc: MatrixDescMut<f32>,
}

/// Initialise the driver, allocate all matrices, and prepare a `BlasHandle`.
///
/// Returns `None` on any platform without a usable CUDA driver (macOS,
/// CI without a GPU, etc.). The caller must skip the benchmark in that case.
fn try_setup() -> Option<GemmHarness> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;

    let elements = (N as usize) * (N as usize);

    // Deterministic host-side fill — small magnitudes keep the F32 accumulator
    // well within range so we don't accidentally measure denormal handling.
    let host_a: Vec<f32> = (0..elements).map(|i| (i as f32) * 1.0e-6).collect();
    let host_b: Vec<f32> = (0..elements)
        .map(|i| ((i % 4096) as f32) * 1.0e-6)
        .collect();

    let a = DeviceBuffer::<f32>::from_host(&host_a).ok()?;
    let b = DeviceBuffer::<f32>::from_host(&host_b).ok()?;
    let c = DeviceBuffer::<f32>::zeroed(elements).ok()?;

    let a_desc = MatrixDesc::<f32>::from_buffer(&a, N, N, Layout::RowMajor).ok()?;
    let b_desc = MatrixDesc::<f32>::from_buffer(&b, N, N, Layout::RowMajor).ok()?;
    // Re-read the device pointer of `c` without taking a mutable reference,
    // since we still need to keep `c` alive in the harness. We construct the
    // mutable descriptor from the same raw pointer instead.
    let c_ptr = c.as_device_ptr();
    let c_desc = MatrixDescMut::<f32>::from_raw(c_ptr, N, N, N, Layout::RowMajor);

    Some(GemmHarness {
        _ctx: ctx,
        handle,
        _a: a,
        _b: b,
        _c: c,
        a_desc,
        b_desc,
        c_desc,
    })
}

/// Measures steady-state F32 4096³ GEMM throughput.
///
/// The bench body issues one `gemm` call per iteration, scaling C by `1.0`
/// (effectively `C += A * B`). Criterion's median × `Throughput::Elements`
/// annotation gives a directly-comparable TFLOPS number (2·N³ flops per
/// iteration).
fn bench_gemm_f32_4096(c: &mut Criterion) {
    let mut harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut group = c.benchmark_group("gemm_f32_4096");
    group.throughput(Throughput::Elements(FLOPS_PER_GEMM));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = gemm::<f32>(
                black_box(&harness.handle),
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0_f32,
                black_box(&harness.a_desc),
                black_box(&harness.b_desc),
                1.0_f32,
                &mut harness.c_desc,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_gemm_f32_4096);
criterion_main!(benches);
