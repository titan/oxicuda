//! Row-wise softmax 4096² benchmark — target: P5 ≥ 90% cuDNN throughput.
//!
//! Computes a numerically stable row-wise softmax of a `4096 × 4096` F32
//! matrix. The path exercised here (`cols > 1024`) is the multi-block
//! `reduce + finalize` pipeline that exchanges per-block `(max, sum_exp)`
//! pairs through a global scratch buffer.
//!
//! Throughput is reported as `Throughput::Bytes(rows * cols * 2 * 4)` —
//! softmax is fundamentally memory-bound, and the kernel must read the
//! input row and write the output row (factor of 2). At 4096 × 4096 ×
//! 4 bytes (F32) the working set is 64 MiB read + 64 MiB written = 128 MiB
//! per invocation.
//!
//! ## Memory budget
//!
//! Two F32 buffers of size `4096 × 4096 × 4 B = 64 MiB` each → ~128 MiB
//! resident plus a small per-row block scratch. Comfortably fits on any
//! modern GPU.
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
//! cargo bench -p oxicuda-blas --bench softmax_4096
//! ```
//!
//! Verifies P-gate `P5` (Softmax, 4096 × 4096) once executed on
//! Linux+NVIDIA hardware.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::reduction::softmax;
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

/// Number of rows in the softmax input.
const ROWS: u32 = 4096;
/// Number of columns per row.
const COLS: u32 = 4096;

/// Bytes touched per softmax call: one read and one write of the full matrix.
const BYTES_PER_CALL: u64 =
    (ROWS as u64) * (COLS as u64) * 2u64 * (std::mem::size_of::<f32>() as u64);

/// GPU resources shared across iterations of the criterion loop.
struct SoftmaxHarness {
    /// Owns the CUDA context for the lifetime of all device buffers and the
    /// `BlasHandle`.
    _ctx: Arc<Context>,
    /// BLAS dispatch handle bound to a default stream on `_ctx`.
    handle: BlasHandle,
    /// Input matrix (`ROWS × COLS`, row-major).
    input: DeviceBuffer<f32>,
    /// Output matrix (`ROWS × COLS`, row-major).
    output: DeviceBuffer<f32>,
}

/// Initialise the driver, allocate all matrices, and prepare a `BlasHandle`.
///
/// Returns `None` on any platform without a usable CUDA driver (macOS,
/// CI without a GPU, etc.). The caller must skip the benchmark in that case.
fn try_setup() -> Option<SoftmaxHarness> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;

    let elements = (ROWS as usize) * (COLS as usize);

    // Deterministic host-side fill; magnitudes well within `expf` range so
    // the numerically stable `(x - max)` shift always yields finite values.
    let host_input: Vec<f32> = (0..elements)
        .map(|i| ((i % 4096) as f32) * 1.0e-3)
        .collect();

    let input = DeviceBuffer::<f32>::from_host(&host_input).ok()?;
    let output = DeviceBuffer::<f32>::zeroed(elements).ok()?;

    Some(SoftmaxHarness {
        _ctx: ctx,
        handle,
        input,
        output,
    })
}

/// Measures steady-state row-wise softmax throughput on a 4096² matrix.
fn bench_softmax_4096(c: &mut Criterion) {
    let mut harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut group = c.benchmark_group("softmax_4096");
    group.throughput(Throughput::Bytes(BYTES_PER_CALL));
    group.bench_function("oxicuda", |b| {
        b.iter(|| {
            let r = softmax::<f32>(
                black_box(&harness.handle),
                ROWS,
                COLS,
                black_box(&harness.input),
                &mut harness.output,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_softmax_4096);
criterion_main!(benches);
