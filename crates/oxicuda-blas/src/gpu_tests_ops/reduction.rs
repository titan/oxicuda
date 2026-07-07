//! On-device numeric validation for the BLAS **reduction** kernels.
//!
//! The production ops in [`crate::reduction`] (`sum`, `mean`, `variance`,
//! `max`, `min`, `softmax`, `reduce_axis`) each JIT-compile a PTX kernel via
//! `Module::from_ptx` and launch it on the device. Prior coverage only asserted
//! the emitted PTX *as a string*; this module instead drives the **real
//! production op** end-to-end — upload, launch on the live GPU, copy back — and
//! asserts equivalence against an independent CPU re-derivation of the math.
//! These are exactly the paths a convergence audit flagged as launched-in-
//! production-but-never-validated (the same gap that hid the SpMM "wrote 1 of 4
//! columns" and CSR5 "dropped beta" bugs elsewhere).
//!
//! ## What this suite uncovered
//!
//! Every reduction / block-softmax PTX template in `oxicuda-ptx` emits the
//! `.maxntid` performance-tuning directive **inside** the kernel body, which
//! `ptxas` rejects (`Parsing error near '.maxntid'`), so *none* of these
//! kernels loaded on device. The BLAS-layer launchers now route generated PTX
//! through [`super::super::reduction`]'s `ptx_fixup::relocate_perf_directives`
//! (a stopgap that hoists the directive above the body `{`); the root-cause fix
//! belongs in `oxicuda-ptx` and is reported separately.
//!
//! A second, deeper `oxicuda-ptx` defect remains **out of scope** and is
//! reported, not worked around: the non-axis `ReductionTemplate` (and the
//! `Scale` elementwise template) declare `.reg .f32` registers but use them
//! with `.f64` instructions, and the f64 reduction warp-shuffles a 64-bit
//! accumulator with a 32-bit `shfl`. That makes **f64 `sum`/`mean`/`variance`/
//! `max`/`min` un-loadable**; this suite validates those in f32 and validates
//! f64 only on the per-axis path (whose template uses correctly-typed
//! registers). See `sum_f64_nonaxis_blocked_by_upstream_ptx_bug`.
//!
//! Every test returns early (skips) when no CUDA device is present.

use super::{Lcg, assert_close_f32, assert_close_f64, gpu_fixture};

use oxicuda_memory::DeviceBuffer;

use crate::handle::BlasHandle;
use crate::reduction::{ReductionOp, max, mean, min, reduce_axis, softmax, sum, variance};
use crate::types::GpuFloat;

// ---------------------------------------------------------------------------
// Shared constants & helpers
// ---------------------------------------------------------------------------

/// Block size of the two-phase reduction kernels (`REDUCE_BLOCK_SIZE`).
const RB: u32 = 256;

/// `ceil(n / RB)` clamped to at least 1 — the partial-buffer length the
/// two-phase `sum`/`mean`/`variance`/`max`/`min` reductions require.
fn n_blocks(n: u32) -> usize {
    n.div_ceil(RB).max(1) as usize
}

/// Deterministic `Vec<f32>` of `n` samples uniform in `[lo, hi)`.
fn gen_f32(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.range_f32(lo, hi)).collect()
}

/// Deterministic `Vec<f64>` of `n` samples uniform in `[lo, hi)`.
fn gen_f64(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.range_f64(lo, hi)).collect()
}

/// Drives a scalar f32 reduction (`sum`/`max`/`min`) on device and returns
/// `result[0]`. The result buffer is sized to hold the per-block partials so
/// the multi-block (two-phase) path is exercised whenever `n > RB`.
fn run_scalar_f32(
    h: &BlasHandle,
    host: &[f32],
    n: u32,
    launch: impl FnOnce(
        &BlasHandle,
        u32,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> crate::error::BlasResult<()>,
) -> f32 {
    let x = DeviceBuffer::from_host(host).expect("upload input");
    let nb = n_blocks(n);
    let mut r = DeviceBuffer::<f32>::zeroed(nb).expect("alloc result");
    launch(h, n, &x, &mut r).expect("reduction launch");
    h.stream().synchronize().expect("sync");
    let mut got = vec![0f32; nb];
    r.copy_to_host(&mut got).expect("copy result");
    got[0]
}

/// Drives the production `variance` (3-pass: mean, squared-diff, sum) on device.
fn run_variance_f32(h: &BlasHandle, host: &[f32], n: u32) -> f32 {
    let x = DeviceBuffer::from_host(host).expect("upload input");
    let nb = n_blocks(n);
    let mut work = DeviceBuffer::<f32>::zeroed(n as usize).expect("alloc work");
    let mut r = DeviceBuffer::<f32>::zeroed(nb).expect("alloc result");
    variance(h, n, &x, &mut work, &mut r).expect("variance launch");
    h.stream().synchronize().expect("sync");
    let mut got = vec![0f32; nb];
    r.copy_to_host(&mut got).expect("copy result");
    got[0]
}

/// Drives the production row-wise `softmax` on device, returning the full
/// `rows * cols` output (row-major).
fn run_softmax_f32(h: &BlasHandle, host: &[f32], rows: u32, cols: u32) -> Vec<f32> {
    let total = (rows as usize) * (cols as usize);
    let x = DeviceBuffer::from_host(host).expect("upload input");
    let mut out = DeviceBuffer::<f32>::zeroed(total).expect("alloc output");
    softmax(h, rows, cols, &x, &mut out).expect("softmax launch");
    h.stream().synchronize().expect("sync");
    let mut got = vec![0f32; total];
    out.copy_to_host(&mut got).expect("copy output");
    got
}

/// Drives the production per-axis reduction on device (generic over precision).
fn run_axis<T: GpuFloat>(
    h: &BlasHandle,
    op: ReductionOp,
    outer: u32,
    axis_len: u32,
    inner: u32,
    host: &[T],
) -> Vec<T> {
    let total = (outer as usize) * (inner as usize);
    let x = DeviceBuffer::from_host(host).expect("upload input");
    let mut out = DeviceBuffer::<T>::zeroed(total).expect("alloc output");
    reduce_axis(h, op, outer, axis_len, inner, &x, &mut out).expect("reduce_axis launch");
    h.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); total];
    out.copy_to_host(&mut got).expect("copy output");
    got
}

// ---------------------------------------------------------------------------
// CPU oracles (independent re-derivation of the reduction math, in f64)
// ---------------------------------------------------------------------------

/// `sum(host[..n])` accumulated in f64 (tight reference for f32 GPU sums).
fn oracle_sum(host: &[f32], n: usize) -> f64 {
    host[..n].iter().map(|&v| f64::from(v)).sum()
}

/// `max(host[..n])` — exact (no rounding).
fn oracle_max(host: &[f32], n: usize) -> f32 {
    host[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// `min(host[..n])` — exact (no rounding).
fn oracle_min(host: &[f32], n: usize) -> f32 {
    host[..n].iter().copied().fold(f32::INFINITY, f32::min)
}

/// Population variance `E[(x-mean)^2]` accumulated in f64.
fn oracle_variance(host: &[f32], n: usize) -> f64 {
    let mean = oracle_sum(host, n) / n as f64;
    host[..n]
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64
}

/// Numerically-stable row-wise softmax oracle (computed in f64, cast to f32).
fn oracle_softmax(host: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row
            .iter()
            .map(|&v| (f64::from(v) - f64::from(m)).exp())
            .collect();
        let s: f64 = exps.iter().sum();
        for c in 0..cols {
            out[r * cols + c] = (exps[c] / s) as f32;
        }
    }
    out
}

/// Per-axis reduction oracle over a tensor viewed as `[outer, axis_len, inner]`.
fn oracle_axis(
    input: &[f64],
    outer: usize,
    axis_len: usize,
    inner: usize,
    op: ReductionOp,
) -> Vec<f64> {
    let mut out = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = match op {
                ReductionOp::Sum | ReductionOp::Mean => 0.0,
                ReductionOp::Product => 1.0,
                ReductionOp::Max => f64::NEG_INFINITY,
                ReductionOp::Min => f64::INFINITY,
            };
            for k in 0..axis_len {
                let v = input[o * axis_len * inner + k * inner + i];
                acc = match op {
                    ReductionOp::Sum | ReductionOp::Mean => acc + v,
                    ReductionOp::Product => acc * v,
                    ReductionOp::Max => acc.max(v),
                    ReductionOp::Min => acc.min(v),
                };
            }
            if op == ReductionOp::Mean {
                acc /= axis_len as f64;
            }
            out[o * inner + i] = acc;
        }
    }
    out
}

// ===========================================================================
// sum
// ===========================================================================

#[test]
fn sum_f32_single_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 200u32; // <= RB: single-block path
    let host = gen_f32(0x5117, n as usize, 0.25, 1.75);
    let got = run_scalar_f32(&h, &host, n, sum);
    let want = oracle_sum(&host, n as usize) as f32;
    // A no-op kernel (returning the identity 0) would be ~200 away from `want`.
    assert_close_f32(&[got], &[want], 1e-4, 1e-2, "sum f32 single-block");
}

#[test]
fn sum_f32_multi_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 4096u32; // 16 blocks -> phase-1 partials + phase-2 reduce
    assert_eq!(n_blocks(n), 16);
    let host = gen_f32(0x5234, n as usize, 0.25, 1.75);
    let got = run_scalar_f32(&h, &host, n, sum);
    let want = oracle_sum(&host, n as usize) as f32;
    // Dropping any single block would shift the sum by ~256 (>> tolerance).
    assert_close_f32(&[got], &[want], 1e-4, 1e-1, "sum f32 multi-block");
}

#[test]
fn sum_f32_full_two_phase_cap_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 65536u32; // 256 blocks: the top of the two-phase capacity
    assert_eq!(n_blocks(n), 256);
    let host = gen_f32(0x5360, n as usize, 0.25, 1.75);
    let got = run_scalar_f32(&h, &host, n, sum);
    let want = oracle_sum(&host, n as usize) as f32;
    assert_close_f32(&[got], &[want], 1e-4, 2.0, "sum f32 two-phase cap");
}

#[test]
fn sum_f64_nonaxis_blocked_by_upstream_ptx_bug() {
    // f64 NON-axis reductions are currently un-loadable on device: the shared
    // `oxicuda-ptx` `ReductionTemplate` declares `.reg .f32 %f<8>` yet emits
    // `mov.f64`/`ld.global.f64` into those registers and warp-shuffles the f64
    // accumulator with a 32-bit `shfl.sync.down.b32`. ptxas rejects this
    // ("Arguments mismatch"). The defect is in oxicuda-ptx (OUT OF SCOPE here)
    // and is reported separately; the BLAS-layer `.maxntid` relocation cannot
    // fix it. This test stays correct in both worlds — it validates the number
    // if upstream is fixed, and otherwise asserts the documented load failure
    // (never green-washing a wrong result).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 300u32;
    let host = gen_f64(0x5F64, n as usize, -3.0, 3.0);
    let x = DeviceBuffer::from_host(&host).expect("upload");
    let nb = n_blocks(n);
    let mut r = DeviceBuffer::<f64>::zeroed(nb).expect("alloc result");
    match sum(&h, n, &x, &mut r) {
        Ok(()) => {
            h.stream().synchronize().expect("sync");
            let mut got = vec![0f64; nb];
            r.copy_to_host(&mut got).expect("copy result");
            let want: f64 = host.iter().sum();
            assert!(
                (got[0] - want).abs() <= 1e-9 * want.abs() + 1e-9,
                "f64 sum now loads but is numerically wrong: got {} want {want}",
                got[0]
            );
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("module load")
                    || msg.contains("Arguments")
                    || msg.contains("ptx")
                    || msg.contains("PTX"),
                "expected the known oxicuda-ptx f64 module-load rejection, got: {msg}"
            );
        }
    }
}

// ===========================================================================
// mean
// ===========================================================================

#[test]
fn mean_f32_single_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 250u32;
    let host = gen_f32(0x3EA1, n as usize, 0.25, 1.75);
    let got = run_scalar_f32(&h, &host, n, mean);
    let want = (oracle_sum(&host, n as usize) / f64::from(n)) as f32;
    assert_close_f32(&[got], &[want], 1e-4, 1e-4, "mean f32 single-block");
}

#[test]
fn mean_f32_multi_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 5000u32; // 20 blocks
    assert_eq!(n_blocks(n), 20);
    let host = gen_f32(0x3EB2, n as usize, 0.25, 1.75);
    let got = run_scalar_f32(&h, &host, n, mean);
    let want = (oracle_sum(&host, n as usize) / f64::from(n)) as f32;
    assert_close_f32(&[got], &[want], 1e-4, 1e-4, "mean f32 multi-block");
}

// ===========================================================================
// variance
// ===========================================================================

#[test]
fn variance_f32_single_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 256u32; // single block exactly
    let host = gen_f32(0x7A01, n as usize, -2.0, 2.0);
    let got = run_variance_f32(&h, &host, n);
    let want = oracle_variance(&host, n as usize) as f32;
    assert_close_f32(&[got], &[want], 2e-3, 1e-3, "variance f32 single-block");
}

#[test]
fn variance_f32_multi_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 2048u32; // 8 blocks
    assert_eq!(n_blocks(n), 8);
    let host = gen_f32(0x7A12, n as usize, -2.0, 2.0);
    let got = run_variance_f32(&h, &host, n);
    let want = oracle_variance(&host, n as usize) as f32;
    assert_close_f32(&[got], &[want], 2e-3, 1e-3, "variance f32 multi-block");
}

// ===========================================================================
// max / min  (inject a known interior extreme to prove every block contributes)
// ===========================================================================

#[test]
fn max_f32_multi_block_finds_injected_extreme() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 5000u32;
    let mut host = gen_f32(0x9A01, n as usize, -5.0, 5.0);
    host[1234] = 42.0; // interior of block 4 (1234/256); not a boundary lane
    let got = run_scalar_f32(&h, &host, n, max);
    let want = oracle_max(&host, n as usize);
    assert_close_f32(&[got], &[want], 0.0, 1e-6, "max f32 multi-block");
    assert!(
        (got - 42.0).abs() < 1e-6,
        "injected max must be found: {got}"
    );
}

#[test]
fn min_f32_multi_block_finds_injected_extreme() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 5000u32;
    let mut host = gen_f32(0x9B01, n as usize, -5.0, 5.0);
    host[3777] = -42.0;
    let got = run_scalar_f32(&h, &host, n, min);
    let want = oracle_min(&host, n as usize);
    assert_close_f32(&[got], &[want], 0.0, 1e-6, "min f32 multi-block");
    assert!(
        (got + 42.0).abs() < 1e-6,
        "injected min must be found: {got}"
    );
}

#[test]
fn max_f32_single_block_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let n = 100u32;
    let host = gen_f32(0x9C01, n as usize, -5.0, 5.0);
    let got = run_scalar_f32(&h, &host, n, max);
    let want = oracle_max(&host, n as usize);
    assert_close_f32(&[got], &[want], 0.0, 1e-6, "max f32 single-block");
}

// ===========================================================================
// softmax  (looser tolerance: built on the base-2 `ex2.approx.f32` pipeline)
// ===========================================================================

#[test]
fn softmax_f32_warp_path_matches_oracle() {
    // cols <= 32 -> warp-shuffle strategy (one warp per row).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (rows, cols) = (64u32, 16u32);
    let host = gen_f32(0x50F1, (rows * cols) as usize, -4.0, 4.0);
    let got = run_softmax_f32(&h, &host, rows, cols);
    let want = oracle_softmax(&host, rows as usize, cols as usize);
    assert_close_f32(&got, &want, 1e-2, 1e-3, "softmax f32 warp");
    // Each output row must sum to 1 (normalization sanity).
    for r in 0..rows as usize {
        let s: f32 = got[r * cols as usize..(r + 1) * cols as usize].iter().sum();
        assert!((s - 1.0).abs() < 5e-3, "softmax row {r} sum = {s}");
    }
}

#[test]
fn softmax_f32_warp_boundary_cols32_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (rows, cols) = (40u32, 32u32);
    let host = gen_f32(0x50F2, (rows * cols) as usize, -4.0, 4.0);
    let got = run_softmax_f32(&h, &host, rows, cols);
    let want = oracle_softmax(&host, rows as usize, cols as usize);
    assert_close_f32(&got, &want, 1e-2, 1e-3, "softmax f32 cols=32");
}

#[test]
fn softmax_f32_block_path_matches_oracle() {
    // 32 < cols <= 1024 -> shared-memory block reduction (one block per row).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    for cols in [100u32, 512u32] {
        let rows = 24u32;
        let host = gen_f32(0x5B00 ^ u64::from(cols), (rows * cols) as usize, -4.0, 4.0);
        let got = run_softmax_f32(&h, &host, rows, cols);
        let want = oracle_softmax(&host, rows as usize, cols as usize);
        assert_close_f32(&got, &want, 1e-2, 1e-3, "softmax f32 block");
    }
}

#[test]
fn softmax_f32_multi_block_path_matches_oracle() {
    // cols > 1024 -> MultiBlockSoftmaxPtx reduce + finalize pipeline (f32 only).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (rows, cols) = (8u32, 2048u32);
    let host = gen_f32(0x5C0F_FEE0, (rows * cols) as usize, -4.0, 4.0);
    let got = run_softmax_f32(&h, &host, rows, cols);
    let want = oracle_softmax(&host, rows as usize, cols as usize);
    assert_close_f32(&got, &want, 1e-2, 1e-3, "softmax f32 multi-block");
    for r in 0..rows as usize {
        let s: f32 = got[r * cols as usize..(r + 1) * cols as usize].iter().sum();
        assert!(
            (s - 1.0).abs() < 5e-3,
            "multi-block softmax row {r} sum = {s}"
        );
    }
}

#[test]
fn softmax_f32_uniform_row_is_uniform() {
    // A constant row must map to the uniform distribution 1/cols everywhere —
    // a strong check that the max-subtraction and normalization are correct.
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (rows, cols) = (4u32, 100u32);
    let host = vec![1.5f32; (rows * cols) as usize];
    let got = run_softmax_f32(&h, &host, rows, cols);
    let want = vec![1.0f32 / cols as f32; (rows * cols) as usize];
    assert_close_f32(&got, &want, 1e-3, 1e-4, "softmax uniform row");
}

// ===========================================================================
// reduce_axis  (the strided `inner > 1` path is the non-trivial one)
// ===========================================================================

#[test]
fn axis_sum_f32_contiguous_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (outer, axis_len, inner) = (4u32, 5u32, 1u32);
    let host = gen_f32(0xA001, (outer * axis_len * inner) as usize, 0.25, 1.75);
    let got = run_axis(&h, ReductionOp::Sum, outer, axis_len, inner, &host);
    let input64: Vec<f64> = host.iter().map(|&v| f64::from(v)).collect();
    let want: Vec<f32> = oracle_axis(
        &input64,
        outer as usize,
        axis_len as usize,
        inner as usize,
        ReductionOp::Sum,
    )
    .iter()
    .map(|&v| v as f32)
    .collect();
    assert_close_f32(&got, &want, 1e-4, 1e-4, "axis sum f32 contiguous");
}

#[test]
fn axis_sum_f32_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    // inner = 4 -> each block strides over the axis by `inner`; a contiguous
    // (stride-ignoring) kernel would mix columns and fail.
    let (outer, axis_len, inner) = (3u32, 6u32, 4u32);
    let host = gen_f32(0xA002, (outer * axis_len * inner) as usize, 0.25, 1.75);
    let got = run_axis(&h, ReductionOp::Sum, outer, axis_len, inner, &host);
    let input64: Vec<f64> = host.iter().map(|&v| f64::from(v)).collect();
    let want: Vec<f32> = oracle_axis(
        &input64,
        outer as usize,
        axis_len as usize,
        inner as usize,
        ReductionOp::Sum,
    )
    .iter()
    .map(|&v| v as f32)
    .collect();
    assert_close_f32(&got, &want, 1e-4, 1e-4, "axis sum f32 strided");
}

#[test]
fn axis_max_min_mean_product_f32_strided_match_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (outer, axis_len, inner) = (2u32, 5u32, 3u32);
    let count = (outer * axis_len * inner) as usize;

    for op in [ReductionOp::Max, ReductionOp::Min, ReductionOp::Mean] {
        let host = gen_f32(0xB100 + op as u64, count, -5.0, 5.0);
        let got = run_axis(&h, op, outer, axis_len, inner, &host);
        let input64: Vec<f64> = host.iter().map(|&v| f64::from(v)).collect();
        let want: Vec<f32> = oracle_axis(
            &input64,
            outer as usize,
            axis_len as usize,
            inner as usize,
            op,
        )
        .iter()
        .map(|&v| v as f32)
        .collect();
        let tol = if op == ReductionOp::Mean { 1e-4 } else { 0.0 };
        assert_close_f32(&got, &want, tol, 1e-5, "axis max/min/mean f32 strided");
    }

    // Product kept near 1 with a short axis to stay well-conditioned.
    let host = gen_f32(0xB1FF, count, 0.7, 1.3);
    let got = run_axis(&h, ReductionOp::Product, outer, axis_len, inner, &host);
    let input64: Vec<f64> = host.iter().map(|&v| f64::from(v)).collect();
    let want: Vec<f32> = oracle_axis(
        &input64,
        outer as usize,
        axis_len as usize,
        inner as usize,
        ReductionOp::Product,
    )
    .iter()
    .map(|&v| v as f32)
    .collect();
    assert_close_f32(&got, &want, 1e-4, 1e-5, "axis product f32 strided");
}

#[test]
fn axis_sum_f32_long_axis_exercises_accumulation_loop() {
    // axis_len > block_size forces the in-kernel `$AXIS_LOOP` to iterate
    // (k = tid, tid+256, ...). inner = 2 keeps it strided.
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (outer, axis_len, inner) = (1u32, 300u32, 2u32);
    let host = gen_f32(0xC001, (outer * axis_len * inner) as usize, 0.25, 1.75);
    let got = run_axis(&h, ReductionOp::Sum, outer, axis_len, inner, &host);
    let input64: Vec<f64> = host.iter().map(|&v| f64::from(v)).collect();
    let want: Vec<f32> = oracle_axis(
        &input64,
        outer as usize,
        axis_len as usize,
        inner as usize,
        ReductionOp::Sum,
    )
    .iter()
    .map(|&v| v as f32)
    .collect();
    assert_close_f32(&got, &want, 1e-4, 1e-3, "axis sum f32 long axis");
}

#[test]
fn axis_sum_f64_strided_matches_oracle() {
    // The PER-AXIS template uses correctly-typed f64 registers (unlike the
    // non-axis ReductionTemplate), so f64 axis reductions DO load and run.
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (outer, axis_len, inner) = (3u32, 4u32, 4u32);
    let host = gen_f64(0xD001, (outer * axis_len * inner) as usize, -3.0, 3.0);
    let got = run_axis(&h, ReductionOp::Sum, outer, axis_len, inner, &host);
    let want = oracle_axis(
        &host,
        outer as usize,
        axis_len as usize,
        inner as usize,
        ReductionOp::Sum,
    );
    assert_close_f64(&got, &want, 1e-10, 1e-12, "axis sum f64 strided");
}

#[test]
fn axis_max_min_f64_strided_match_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let h = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (outer, axis_len, inner) = (2u32, 6u32, 3u32);
    let count = (outer * axis_len * inner) as usize;
    for op in [ReductionOp::Max, ReductionOp::Min] {
        let host = gen_f64(0xD100 + op as u64, count, -5.0, 5.0);
        let got = run_axis(&h, op, outer, axis_len, inner, &host);
        let want = oracle_axis(&host, outer as usize, axis_len as usize, inner as usize, op);
        assert_close_f64(&got, &want, 0.0, 1e-12, "axis max/min f64 strided");
    }
}
