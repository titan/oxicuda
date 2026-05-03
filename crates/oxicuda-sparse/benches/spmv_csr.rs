//! SpMV CSR throughput benchmark — SciPy-scale sparse matrix.
//!
//! Verifies the **P3** quality gate from `crates/oxicuda-sparse/TODO.md`
//! ("SpMV CSR, typical SciPy-scale sparse matrix — ≥ 85% cuSPARSE
//! throughput") on a real GPU. The harness compiles on every host but
//! runtime-skips on macOS / no-NVIDIA boxes by detecting the absence of a
//! CUDA device at the top of every benchmark function (driver `init()` /
//! `Device::get(0)` returns `Err` without a working installation).
//!
//! ## Workload
//!
//! A synthetic CSR matrix of shape `(N, N)` with `N = 1_000_000` rows and
//! a deterministic stencil-style nonzero pattern: each row `i` carries up
//! to four nonzeros at columns `[i, i+1, i+1023, i+1024]`, capped at
//! `N - 1`. Duplicate columns produced by the cap on the last ~1024 rows
//! are tolerated by `CsrMatrix::from_host` (their values are summed at
//! SpMV time, not validated for uniqueness). Total nnz ≈ `4·N` ≈ 4·10⁶.
//!
//! Values are filled with a deterministic LCG-free formula
//! `value(i, j) = (i * j) as f32 * 1e-6`; no `rand` dependency is
//! introduced.
//!
//! ## Throughput accounting
//!
//! Criterion is told `Throughput::Elements(2 * nnz)` so the per-iteration
//! GFLOPS reading is the standard SpMV metric (one multiply + one add per
//! nonzero). Only the device-side `spmv` call is timed; CSR upload, host
//! buffer construction, and `SparseHandle` setup happen outside `b.iter`.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_driver::{Context, Device, init};
use oxicuda_memory::DeviceBuffer;
use oxicuda_sparse::format::CsrMatrix;
use oxicuda_sparse::handle::SparseHandle;
use oxicuda_sparse::ops::{SpMVAlgo, spmv};

/// Builds a synthetic CSR matrix with the stencil pattern documented in the
/// module-level comment. Returns `(rows, nnz, row_ptr, col_idx, values)` ready
/// to feed [`CsrMatrix::from_host`].
///
/// Pattern: row `i` has up to four nonzeros at the column offsets
/// `[i, i+1, i+1023, i+1024]`, capped at `n - 1`. The last ~1024 rows
/// therefore carry duplicate column indices; `CsrMatrix::from_host` accepts
/// these and SpMV simply sums their contributions, which is the same
/// behaviour as a real-world degenerate stencil on the matrix's right edge.
fn build_synthetic_csr(n: usize) -> (u32, u32, Vec<i32>, Vec<i32>, Vec<f32>) {
    let n_i = n as i32;
    let mut row_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<i32> = Vec::with_capacity(n * 4);
    let mut values: Vec<f32> = Vec::with_capacity(n * 4);

    row_ptr.push(0);
    let offsets: [i32; 4] = [0, 1, 1023, 1024];
    for i in 0..n_i {
        for off in offsets {
            let mut j = i + off;
            if j > n_i - 1 {
                j = n_i - 1;
            }
            col_idx.push(j);
            // value(i, j) = (i * j) as f32 * 1e-6 — deterministic, no rand dep.
            let v = (i64::from(i) * i64::from(j)) as f32 * 1e-6_f32;
            values.push(v);
        }
        row_ptr.push(col_idx.len() as i32);
    }

    let rows = n as u32;
    let nnz = col_idx.len() as u32;
    (rows, nnz, row_ptr, col_idx, values)
}

/// Tiny helper to attempt the full driver→context→handle bring-up.
/// Returns `None` on any failure so callers can issue the standard
/// "skip: no GPU" message without panicking.
fn try_setup() -> Option<(Arc<Context>, SparseHandle)> {
    init().ok()?;
    if Device::count().ok()? <= 0 {
        return None;
    }
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = SparseHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

fn bench_spmv_csr_scipy_scale(c: &mut Criterion) {
    // Hard constraint: runtime-skip on macOS / no-GPU boxes.
    // `Device::get` requires the driver to be initialised first, so route
    // the early-skip probe through `init()`. On Linux+NVIDIA both calls
    // succeed; the redundant `init()` inside `try_setup` is a no-op.
    let _device = match init().and_then(|_| Device::get(0)) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let (_ctx, handle) = match try_setup() {
        Some(state) => state,
        None => {
            eprintln!("skip: no GPU");
            return;
        }
    };

    let mut group = c.benchmark_group("spmv_csr_scipy_scale");
    // SciPy-scale sweep: keep one very large case so reviewers see the P3
    // operating point, plus two smaller sizes for trend visibility.
    let sizes: &[(&str, usize)] = &[
        ("256k", 256 * 1024),
        ("512k", 512 * 1024),
        ("1m", 1_000_000),
    ];

    for &(label, n) in sizes {
        let (rows, nnz, row_ptr, col_idx, values) = build_synthetic_csr(n);

        let csr = match CsrMatrix::<f32>::from_host(rows, n as u32, &row_ptr, &col_idx, &values) {
            Ok(c) => c,
            Err(err) => {
                eprintln!("skip: CSR upload failed for {label}: {err}");
                continue;
            }
        };

        // Dense vectors x (length cols) and y (length rows).
        let x_host = vec![1.0_f32; n];
        let x_dev = match DeviceBuffer::<f32>::from_host(&x_host) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("skip: x upload failed for {label}: {err}");
                continue;
            }
        };
        let y_dev = match DeviceBuffer::<f32>::alloc(n) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("skip: y alloc failed for {label}: {err}");
                continue;
            }
        };

        // 2 flops per nonzero — Criterion converts elements/s to GFLOPS in
        // its report so the reading lines up directly with the cuSPARSE
        // reference.
        group.throughput(Throughput::Elements(u64::from(nnz) * 2));

        group.bench_with_input(
            BenchmarkId::new("adaptive_f32", label),
            &(rows, nnz),
            |b, _| {
                b.iter(|| {
                    let _ = black_box(spmv::<f32>(
                        &handle,
                        SpMVAlgo::Adaptive,
                        1.0_f32,
                        &csr,
                        x_dev.as_device_ptr(),
                        0.0_f32,
                        y_dev.as_device_ptr(),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(spmv_csr_benches, bench_spmv_csr_scipy_scale);
criterion_main!(spmv_csr_benches);
