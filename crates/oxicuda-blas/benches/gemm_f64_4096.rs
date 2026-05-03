//! GEMM F64 4096³ benchmark — target: P3 ≥ 95% cuBLAS throughput on sm_80.
//!
//! Computes `C = alpha * A * B + beta * C` for `M = N = K = 4096` in double
//! precision. Throughput in TFLOPS is reported via the
//! `Throughput::Elements(2 * M * N * K)` annotation: each output element of
//! C costs `2 * K` flops (one multiply + one add per inner-product term).
//!
//! ## Memory budget
//!
//! Three F64 matrices of size 4096² × 8 B = 128 MiB each → ~384 MiB resident.
//! Fits comfortably on data-center GPUs; small consumer cards (≤ 4 GiB)
//! should still cope but will share the device with criterion's harness.
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
//! cargo bench -p oxicuda-blas --bench gemm_f64_4096
//! ```
//!
//! Verifies P-gate `P3` (GEMM F64 sm_80, M=N=K=4096) once executed on
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

/// Total flop count for a single 4096³ F64 GEMM (2·M·N·K).
const FLOPS_PER_GEMM: u64 = 2u64 * (N as u64) * (N as u64) * (N as u64);

/// GPU resources shared across iterations of the criterion loop.
struct GemmHarness {
    /// Owns the CUDA context for the lifetime of all device buffers and the
    /// `BlasHandle`. Must outlive any `MatrixDesc*` constructed below.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Input matrix A (`N × N`, row-major).
    _a: DeviceBuffer<f64>,
    /// Input matrix B (`N × N`, row-major).
    _b: DeviceBuffer<f64>,
    /// Output matrix C (`N × N`, row-major).
    _c: DeviceBuffer<f64>,
    /// Read-only view of A used in every `gemm` call.
    a_desc: MatrixDesc<f64>,
    /// Read-only view of B used in every `gemm` call.
    b_desc: MatrixDesc<f64>,
    /// Mutable view of C used in every `gemm` call.
    c_desc: MatrixDescMut<f64>,
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

    // Deterministic, well-conditioned host-side fill.
    let host_a: Vec<f64> = (0..elements).map(|i| (i as f64) * 1.0e-9).collect();
    let host_b: Vec<f64> = (0..elements)
        .map(|i| ((i % 4096) as f64) * 1.0e-9)
        .collect();

    let a = DeviceBuffer::<f64>::from_host(&host_a).ok()?;
    let b = DeviceBuffer::<f64>::from_host(&host_b).ok()?;
    let c = DeviceBuffer::<f64>::zeroed(elements).ok()?;

    let a_desc = MatrixDesc::<f64>::from_buffer(&a, N, N, Layout::RowMajor).ok()?;
    let b_desc = MatrixDesc::<f64>::from_buffer(&b, N, N, Layout::RowMajor).ok()?;
    let c_ptr = c.as_device_ptr();
    let c_desc = MatrixDescMut::<f64>::from_raw(c_ptr, N, N, N, Layout::RowMajor);

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

/// Measures steady-state F64 4096³ GEMM throughput.
fn bench_gemm_f64_4096(c: &mut Criterion) {
    let mut harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut group = c.benchmark_group("gemm_f64_4096");
    group.throughput(Throughput::Elements(FLOPS_PER_GEMM));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = gemm::<f64>(
                black_box(&harness.handle),
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0_f64,
                black_box(&harness.a_desc),
                black_box(&harness.b_desc),
                1.0_f64,
                &mut harness.c_desc,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_gemm_f64_4096);
criterion_main!(benches);
