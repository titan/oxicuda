//! GEMM F16 4096³ benchmark — target: P1 ≥ 95% cuBLAS throughput on sm_80.
//!
//! Computes `C = alpha * A * B + beta * C` for `M = N = K = 4096` in IEEE
//! half precision (`half::f16`). Throughput in TFLOPS is reported via the
//! `Throughput::Elements(2 * M * N * K)` annotation: each output element of
//! C costs `2 * K` flops (one multiply + one add per inner-product term).
//!
//! ## Memory budget
//!
//! Three F16 matrices of size 4096² × 2 B = 32 MiB each → ~96 MiB resident.
//! Tiny by data-center standards; consumer cards (≥ 4 GiB) easily cope.
//!
//! ## Platform behaviour
//!
//! * **Linux / Windows with NVIDIA driver 525+** — full benchmark runs.
//! * **macOS / no GPU / no driver** — bench function returns immediately
//!   after `eprintln!("skip: no GPU")`. The skip guard is the **first**
//!   statement of every bench function in this file.
//!
//! This bench is feature-gated behind `f16` because `half::f16` is an
//! optional dependency of `oxicuda-blas`. The corresponding `[[bench]]`
//! entry in `Cargo.toml` declares `required-features = ["f16"]`.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-blas --features f16 --bench gemm_f16_4096
//! ```
//!
//! Verifies P-gate `P1` (GEMM F16 sm_80, M=N=K=4096) once executed on
//! Linux+NVIDIA hardware.

#![cfg(feature = "f16")]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// Edge length of every matrix in this benchmark.
const N: u32 = 4096;

/// Total flop count for a single 4096³ F16 GEMM (2·M·N·K).
const FLOPS_PER_GEMM: u64 = 2u64 * (N as u64) * (N as u64) * (N as u64);

/// GPU resources shared across iterations of the criterion loop.
struct GemmHarness {
    /// Owns the CUDA context for the lifetime of all device buffers and the
    /// `BlasHandle`.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Input matrix A (`N × N`, row-major).
    _a: DeviceBuffer<f16>,
    /// Input matrix B (`N × N`, row-major).
    _b: DeviceBuffer<f16>,
    /// Output matrix C (`N × N`, row-major).
    _c: DeviceBuffer<f16>,
    /// Read-only view of A used in every `gemm` call.
    a_desc: MatrixDesc<f16>,
    /// Read-only view of B used in every `gemm` call.
    b_desc: MatrixDesc<f16>,
    /// Mutable view of C used in every `gemm` call.
    c_desc: MatrixDescMut<f16>,
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

    // Deterministic host-side fill — small magnitudes keep the F32
    // accumulator well within the 4096-element-per-row reduction range.
    let host_a: Vec<f16> = (0..elements)
        .map(|i| f16::from_f32((i as f32) * 1.0e-4))
        .collect();
    let host_b: Vec<f16> = (0..elements)
        .map(|i| f16::from_f32(((i % 4096) as f32) * 1.0e-4))
        .collect();

    let a = DeviceBuffer::<f16>::from_host(&host_a).ok()?;
    let b = DeviceBuffer::<f16>::from_host(&host_b).ok()?;
    let c = DeviceBuffer::<f16>::zeroed(elements).ok()?;

    let a_desc = MatrixDesc::<f16>::from_buffer(&a, N, N, Layout::RowMajor).ok()?;
    let b_desc = MatrixDesc::<f16>::from_buffer(&b, N, N, Layout::RowMajor).ok()?;
    let c_ptr = c.as_device_ptr();
    let c_desc = MatrixDescMut::<f16>::from_raw(c_ptr, N, N, N, Layout::RowMajor);

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

/// Measures steady-state F16 4096³ GEMM throughput.
fn bench_gemm_f16_4096(c: &mut Criterion) {
    let mut harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let alpha = f16::from_f32(1.0);
    let beta = f16::from_f32(1.0);

    let mut group = c.benchmark_group("gemm_f16_4096");
    group.throughput(Throughput::Elements(FLOPS_PER_GEMM));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = gemm::<f16>(
                black_box(&harness.handle),
                Transpose::NoTrans,
                Transpose::NoTrans,
                alpha,
                black_box(&harness.a_desc),
                black_box(&harness.b_desc),
                beta,
                &mut harness.c_desc,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_gemm_f16_4096);
criterion_main!(benches);
