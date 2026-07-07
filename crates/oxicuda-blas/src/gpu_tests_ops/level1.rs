//! On-device numeric validation for the BLAS **Level-1** kernels.
//!
//! The production ops in [`crate::level1`] (`dot`, `nrm2`, `asum`, `axpy`,
//! `scal`, `swap`, `copy_vec`, `iamax`) each JIT-compile a PTX kernel via
//! `Module::from_ptx` and launch it on the device. Prior coverage only checked
//! the emitted PTX *as a string*; this module instead drives the **real
//! production op** end-to-end — upload, launch on the live GPU, copy back — and
//! asserts equivalence against an independent CPU re-derivation of the BLAS
//! math.
//!
//! Every kernel here is **numerically validated against a CPU oracle** (none are
//! fragments). The suite deliberately exercises the paths that hid real bugs in
//! sibling kernels elsewhere in the workspace:
//!
//! * **Strided access** (`incx`/`incy` != 1) — off-stride buffer slots are
//!   seeded with a poison value so a kernel that mistakenly reads/writes
//!   contiguously produces a wildly wrong result.
//! * **Non-trivial `alpha`** for `axpy`/`scal` (a dropped scalar is exactly the
//!   class of "CSR5 dropped beta" bug).
//! * **The multi-block reduction path** and, for the two-phase reductions, the
//!   phase-2 accumulation loop (`num_blocks > blockDim`).
//! * **Corruption probes** proving the launch genuinely reads device memory
//!   (non-vacuous).
//!
//! Each test returns early (skips) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use super::*;

use oxicuda_memory::DeviceBuffer;

use crate::handle::BlasHandle;
use crate::types::GpuFloat;

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/// Builds a strided host buffer: places `vals[i]` at index `i * inc`, filling
/// every other (off-stride) slot with `poison`. With `inc > 1`, a kernel that
/// reads contiguously instead of by stride would pick up `poison` and mismatch.
fn strided<T: Copy>(vals: &[T], inc: usize, poison: T) -> Vec<T> {
    if vals.is_empty() {
        return Vec::new();
    }
    let mut out = vec![poison; 1 + (vals.len() - 1) * inc];
    for (i, &v) in vals.iter().enumerate() {
        out[i * inc] = v;
    }
    out
}

// ---------------------------------------------------------------------------
// Generic device runners (drive the production op)
// ---------------------------------------------------------------------------

/// Runs the production `dot` on device and returns the scalar result.
fn device_dot<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &[T],
    incx: i32,
    y: &[T],
    incy: i32,
) -> T {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let dy = DeviceBuffer::from_host(y).expect("upload y");
    let mut dr = DeviceBuffer::<T>::zeroed(1).expect("alloc result");
    crate::level1::dot(handle, n, &dx, incx, &dy, incy, &mut dr).expect("dot launch");
    handle.stream().synchronize().expect("sync");
    let mut got = [T::gpu_zero()];
    dr.copy_to_host(&mut got).expect("d2h result");
    got[0]
}

/// Runs the production `nrm2` on device and returns the scalar result.
fn device_nrm2<T: GpuFloat>(handle: &BlasHandle, n: u32, x: &[T], incx: i32) -> T {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dr = DeviceBuffer::<T>::zeroed(1).expect("alloc result");
    crate::level1::nrm2(handle, n, &dx, incx, &mut dr).expect("nrm2 launch");
    handle.stream().synchronize().expect("sync");
    let mut got = [T::gpu_zero()];
    dr.copy_to_host(&mut got).expect("d2h result");
    got[0]
}

/// Runs the production `asum` on device and returns the scalar result.
fn device_asum<T: GpuFloat>(handle: &BlasHandle, n: u32, x: &[T], incx: i32) -> T {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dr = DeviceBuffer::<T>::zeroed(1).expect("alloc result");
    crate::level1::asum(handle, n, &dx, incx, &mut dr).expect("asum launch");
    handle.stream().synchronize().expect("sync");
    let mut got = [T::gpu_zero()];
    dr.copy_to_host(&mut got).expect("d2h result");
    got[0]
}

/// Runs the production `axpy` on device and returns the updated `y` buffer.
fn device_axpy<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    alpha: T,
    x: &[T],
    incx: i32,
    y: &[T],
    incy: i32,
) -> Vec<T> {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dy = DeviceBuffer::from_host(y).expect("upload y");
    crate::level1::axpy(handle, n, alpha, &dx, incx, &mut dy, incy).expect("axpy launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); y.len()];
    dy.copy_to_host(&mut got).expect("d2h y");
    got
}

/// Runs the production `scal` on device and returns the updated `x` buffer.
fn device_scal<T: GpuFloat>(handle: &BlasHandle, n: u32, alpha: T, x: &[T], incx: i32) -> Vec<T> {
    let mut dx = DeviceBuffer::from_host(x).expect("upload x");
    crate::level1::scal(handle, n, alpha, &mut dx, incx).expect("scal launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); x.len()];
    dx.copy_to_host(&mut got).expect("d2h x");
    got
}

/// Runs the production `swap` on device and returns the updated `(x, y)`.
fn device_swap<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &[T],
    incx: i32,
    y: &[T],
    incy: i32,
) -> (Vec<T>, Vec<T>) {
    let mut dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dy = DeviceBuffer::from_host(y).expect("upload y");
    crate::level1::swap(handle, n, &mut dx, incx, &mut dy, incy).expect("swap launch");
    handle.stream().synchronize().expect("sync");
    let mut gx = vec![T::gpu_zero(); x.len()];
    let mut gy = vec![T::gpu_zero(); y.len()];
    dx.copy_to_host(&mut gx).expect("d2h x");
    dy.copy_to_host(&mut gy).expect("d2h y");
    (gx, gy)
}

/// Runs the production `copy_vec` on device and returns the updated `y`.
fn device_copy<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    x: &[T],
    incx: i32,
    y: &[T],
    incy: i32,
) -> Vec<T> {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dy = DeviceBuffer::from_host(y).expect("upload y");
    crate::level1::copy_vec(handle, n, &dx, incx, &mut dy, incy).expect("copy launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); y.len()];
    dy.copy_to_host(&mut got).expect("d2h y");
    got
}

/// Runs the production `iamax` on device and returns the argmax index.
fn device_iamax<T: GpuFloat>(handle: &BlasHandle, n: u32, x: &[T], incx: i32) -> u32 {
    let dx = DeviceBuffer::from_host(x).expect("upload x");
    let mut dr = DeviceBuffer::<u32>::zeroed(1).expect("alloc idx");
    crate::level1::iamax(handle, n, &dx, incx, &mut dr).expect("iamax launch");
    handle.stream().synchronize().expect("sync");
    let mut got = [0u32];
    dr.copy_to_host(&mut got).expect("d2h idx");
    got[0]
}

// ---------------------------------------------------------------------------
// CPU oracles (independent re-derivation of the BLAS math)
// ---------------------------------------------------------------------------

fn oracle_dot_f32(x: &[f32], incx: usize, y: &[f32], incy: usize, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += x[i * incx] * y[i * incy];
    }
    acc
}

fn oracle_dot_f64(x: &[f64], incx: usize, y: &[f64], incy: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += x[i * incx] * y[i * incy];
    }
    acc
}

fn oracle_nrm2_f32(x: &[f32], incx: usize, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += x[i * incx] * x[i * incx];
    }
    acc.sqrt()
}

fn oracle_nrm2_f64(x: &[f64], incx: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += x[i * incx] * x[i * incx];
    }
    acc.sqrt()
}

fn oracle_asum_f32(x: &[f32], incx: usize, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += x[i * incx].abs();
    }
    acc
}

fn oracle_asum_f64(x: &[f64], incx: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += x[i * incx].abs();
    }
    acc
}

/// 0-based argmax of `|x[i*incx]|` over `i in 0..n`, smallest index on ties
/// (standard BLAS IAMAX convention).
fn oracle_iamax_f32(x: &[f32], incx: usize, n: usize) -> u32 {
    let mut best = 0usize;
    let mut best_abs = x[0].abs();
    for i in 1..n {
        let a = x[i * incx].abs();
        if a > best_abs {
            best_abs = a;
            best = i;
        }
    }
    best as u32
}

// ===========================================================================
// Fixture sanity
// ===========================================================================

/// The fixture's detected SM version and its stream are both live: a
/// [`BlasHandle`] built from the same context targets the same architecture the
/// fixture reported, and the fixture stream is a valid, synchronisable stream.
/// This also makes the shared `GpuFixture::{sm, stream}` fields observed by the
/// Level-1 suite (the GEMM/handle path owns its own stream).
#[test]
fn fixture_sm_matches_handle_and_stream_syncs() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    assert_eq!(
        handle.sm_version(),
        fx.sm,
        "handle SM must match the fixture-detected SM"
    );
    fx.stream
        .synchronize()
        .expect("fixture stream synchronises");
}

// ===========================================================================
// DOT
// ===========================================================================

#[test]
fn dot_f32_contiguous_crosses_block_boundary() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // n = 257 spans two 256-thread blocks (exercises the multi-block path).
    let n = 257usize;
    let mut rng = Lcg::new(0x1111_2222);
    let x: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    let got = device_dot(&handle, n as u32, &x, 1, &y, 1);
    let want = oracle_dot_f32(&x, 1, &y, 1, n);
    assert_close_f32(&[got], &[want], 1e-4, 1e-3, "dot_f32_contiguous");
}

#[test]
fn dot_f32_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 100usize;
    let (incx, incy) = (2usize, 3usize);
    let mut rng = Lcg::new(0x3333_4444);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
    let yv: Vec<f32> = (0..n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
    // Poison the off-stride slots: a contiguous-read bug would read 1000.0.
    let x = strided(&xv, incx, 1000.0f32);
    let y = strided(&yv, incy, -1000.0f32);

    let got = device_dot(&handle, n as u32, &x, incx as i32, &y, incy as i32);
    let want = oracle_dot_f32(&x, incx, &y, incy, n);
    assert_close_f32(&[got], &[want], 1e-4, 1e-3, "dot_f32_strided");
}

#[test]
fn dot_f64_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 96usize;
    let (incx, incy) = (3usize, 2usize);
    let mut rng = Lcg::new(0x5555_6666);
    let xv: Vec<f64> = (0..n).map(|_| rng.range_f64(-2.0, 2.0)).collect();
    let yv: Vec<f64> = (0..n).map(|_| rng.range_f64(-2.0, 2.0)).collect();
    let x = strided(&xv, incx, 1e9f64);
    let y = strided(&yv, incy, -1e9f64);

    let got = device_dot(&handle, n as u32, &x, incx as i32, &y, incy as i32);
    let want = oracle_dot_f64(&x, incx, &y, incy, n);
    assert_close_f64(&[got], &[want], 1e-10, 1e-10, "dot_f64_strided");
}

#[test]
fn dot_f32_multiblock_phase2_loop_is_exact() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // n = 100_000 -> num_blocks = ceil(100000/256) = 391 > 256, so phase-2 must
    // run its accumulation loop (each thread sums multiple partials). All-ones
    // inputs make the exact dot product = n, independent of reduction order.
    let n = 100_000usize;
    let x = vec![1.0f32; n];
    let y = vec![1.0f32; n];
    let got = device_dot(&handle, n as u32, &x, 1, &y, 1);
    assert_close_f32(&[got], &[n as f32], 1e-5, 1e-2, "dot_f32_phase2_loop");
}

/// Non-vacuous: perturbing an input element must change the dot result.
#[test]
fn dot_f32_corruption_probe_is_detected() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 300usize;
    let mut rng = Lcg::new(0x7777_8888);
    let x: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    let clean = device_dot(&handle, n as u32, &x, 1, &y, 1);
    let mut x_bad = x.clone();
    x_bad[123] += 5.0;
    let dirty = device_dot(&handle, n as u32, &x_bad, 1, &y, 1);
    assert!(
        (clean - dirty).abs() > 1e-3,
        "corrupting an input element did not change the dot product \
         (clean={clean}, dirty={dirty}) — kernel may not read device memory"
    );
}

// ===========================================================================
// NRM2
// ===========================================================================

#[test]
fn nrm2_f32_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 120usize;
    let incx = 3usize;
    let mut rng = Lcg::new(0x9999_AAAA);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.5, 1.5)).collect();
    let x = strided(&xv, incx, 1000.0f32);

    let got = device_nrm2(&handle, n as u32, &x, incx as i32);
    let want = oracle_nrm2_f32(&x, incx, n);
    assert_close_f32(&[got], &[want], 1e-4, 1e-3, "nrm2_f32_strided");
}

#[test]
fn nrm2_f64_contiguous_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 257usize;
    let mut rng = Lcg::new(0xBBBB_CCCC);
    let x: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();

    let got = device_nrm2(&handle, n as u32, &x, 1);
    let want = oracle_nrm2_f64(&x, 1, n);
    assert_close_f64(&[got], &[want], 1e-10, 1e-10, "nrm2_f64");
}

#[test]
fn nrm2_f32_multiblock_ones_is_sqrt_n() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // ||1||_2 = sqrt(n); n=100_000 exercises the phase-2 loop and sum-of-squares.
    let n = 100_000usize;
    let x = vec![1.0f32; n];
    let got = device_nrm2(&handle, n as u32, &x, 1);
    let want = (n as f32).sqrt();
    assert_close_f32(&[got], &[want], 1e-4, 1e-2, "nrm2_f32_ones");
}

// ===========================================================================
// ASUM
// ===========================================================================

#[test]
fn asum_f32_strided_with_negatives_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 130usize;
    let incx = 2usize;
    let mut rng = Lcg::new(0xDDDD_EEEE);
    // Deliberately signed so the |.| step actually matters.
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-3.0, 3.0)).collect();
    let x = strided(&xv, incx, 1000.0f32);

    let got = device_asum(&handle, n as u32, &x, incx as i32);
    let want = oracle_asum_f32(&x, incx, n);
    assert_close_f32(&[got], &[want], 1e-4, 1e-3, "asum_f32_strided");
}

#[test]
fn asum_f64_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 88usize;
    let incx = 4usize;
    let mut rng = Lcg::new(0x0123_4567);
    let xv: Vec<f64> = (0..n).map(|_| rng.range_f64(-3.0, 3.0)).collect();
    let x = strided(&xv, incx, -1e9f64);

    let got = device_asum(&handle, n as u32, &x, incx as i32);
    let want = oracle_asum_f64(&x, incx, n);
    assert_close_f64(&[got], &[want], 1e-10, 1e-10, "asum_f64_strided");
}

#[test]
fn asum_f32_multiblock_ones_is_n() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 100_000usize;
    let x = vec![-1.0f32; n]; // |-1| = 1, so sum |x| = n exactly.
    let got = device_asum(&handle, n as u32, &x, 1);
    assert_close_f32(&[got], &[n as f32], 1e-5, 1e-2, "asum_f32_ones");
}

// ===========================================================================
// AXPY — y = alpha*x + y
// ===========================================================================

#[test]
fn axpy_f32_alpha_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 200usize;
    let (incx, incy) = (1usize, 2usize);
    let alpha = 2.5f32;
    let mut rng = Lcg::new(0x2468_ACE0);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let yv: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let x = strided(&xv, incx, 1000.0f32);
    let y = strided(&yv, incy, 777.0f32); // off-stride y must stay untouched.

    let got = device_axpy(&handle, n as u32, alpha, &x, incx as i32, &y, incy as i32);

    // Oracle: fused alpha*x + y at stride positions; off-stride unchanged.
    let mut want = y.clone();
    for i in 0..n {
        want[i * incy] = alpha.mul_add(x[i * incx], y[i * incy]);
    }
    assert_close_f32(&got, &want, 1e-4, 1e-4, "axpy_f32_strided");
    // Explicit untouched-slot probe: index 1 (incy=2) was never a stride target.
    assert_close_f32(&[got[1]], &[777.0f32], 0.0, 0.0, "axpy_f32_off_stride");
}

#[test]
fn axpy_f64_alpha_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 150usize;
    let alpha = -1.75f64;
    let mut rng = Lcg::new(0x1357_9BDF);
    let x: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect();

    let got = device_axpy(&handle, n as u32, alpha, &x, 1, &y, 1);
    let mut want = y.clone();
    for i in 0..n {
        want[i] = alpha.mul_add(x[i], y[i]);
    }
    assert_close_f64(&got, &want, 1e-12, 1e-12, "axpy_f64");
}

#[test]
fn axpy_f32_multiblock_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 5000usize; // ~20 blocks
    let alpha = 0.3f32;
    let mut rng = Lcg::new(0xFEED_FACE);
    let x: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    let got = device_axpy(&handle, n as u32, alpha, &x, 1, &y, 1);
    let mut want = y.clone();
    for i in 0..n {
        want[i] = alpha.mul_add(x[i], y[i]);
    }
    assert_close_f32(&got, &want, 1e-4, 1e-4, "axpy_f32_multiblock");
}

// ===========================================================================
// SCAL — x = alpha*x
// ===========================================================================

#[test]
fn scal_f32_alpha_strided_leaves_off_stride_untouched() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 180usize;
    let incx = 2usize;
    let alpha = 1.5f32;
    let mut rng = Lcg::new(0x0BAD_F00D);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
    let poison = 333.0f32;
    let x = strided(&xv, incx, poison);

    let got = device_scal(&handle, n as u32, alpha, &x, incx as i32);

    let mut want = x.clone();
    for i in 0..n {
        want[i * incx] = alpha * x[i * incx];
    }
    assert_close_f32(&got, &want, 1e-5, 1e-5, "scal_f32_strided");
    // Off-stride slot (index 1) must be byte-for-byte unchanged.
    assert_close_f32(&[got[1]], &[poison], 0.0, 0.0, "scal_f32_off_stride");
}

#[test]
fn scal_f64_alpha_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 160usize;
    let alpha = -0.625f64;
    let mut rng = Lcg::new(0xC0FF_EE00);
    let x: Vec<f64> = (0..n).map(|_| rng.range_f64(-2.0, 2.0)).collect();

    let got = device_scal(&handle, n as u32, alpha, &x, 1);
    let want: Vec<f64> = x.iter().map(|&v| alpha * v).collect();
    assert_close_f64(&got, &want, 1e-12, 1e-12, "scal_f64");
}

#[test]
fn scal_f32_multiblock_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 4096usize; // 16 blocks
    let alpha = 3.0f32;
    let mut rng = Lcg::new(0xABCD_1234);
    let x: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    let got = device_scal(&handle, n as u32, alpha, &x, 1);
    let want: Vec<f32> = x.iter().map(|&v| alpha * v).collect();
    assert_close_f32(&got, &want, 1e-5, 1e-5, "scal_f32_multiblock");
}

// ===========================================================================
// SWAP — x <-> y
// ===========================================================================

#[test]
fn swap_f32_strided_exchanges_only_stride_positions() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 64usize;
    let (incx, incy) = (2usize, 3usize);
    let mut rng = Lcg::new(0x5A5A_6B6B);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-5.0, 5.0)).collect();
    let yv: Vec<f32> = (0..n).map(|_| rng.range_f32(-5.0, 5.0)).collect();
    let x = strided(&xv, incx, 111.0f32);
    let y = strided(&yv, incy, 222.0f32);

    let (gx, gy) = device_swap(&handle, n as u32, &x, incx as i32, &y, incy as i32);

    let mut want_x = x.clone();
    let mut want_y = y.clone();
    for i in 0..n {
        want_x[i * incx] = y[i * incy];
        want_y[i * incy] = x[i * incx];
    }
    assert_close_f32(&gx, &want_x, 0.0, 0.0, "swap_f32_x");
    assert_close_f32(&gy, &want_y, 0.0, 0.0, "swap_f32_y");
}

// ===========================================================================
// COPY — y = x
// ===========================================================================

#[test]
fn copy_f32_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 70usize;
    let (incx, incy) = (3usize, 2usize);
    let mut rng = Lcg::new(0x7C7C_8D8D);
    let xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-4.0, 4.0)).collect();
    let x = strided(&xv, incx, 555.0f32);
    let poison = 999.0f32;
    let y = strided(&vec![0.0f32; n], incy, poison);

    let got = device_copy(&handle, n as u32, &x, incx as i32, &y, incy as i32);

    let mut want = y.clone();
    for i in 0..n {
        want[i * incy] = x[i * incx];
    }
    assert_close_f32(&got, &want, 0.0, 0.0, "copy_f32_strided");
    // Off-stride destination slot (index 1, incy=2) untouched.
    assert_close_f32(&[got[1]], &[poison], 0.0, 0.0, "copy_f32_off_stride");
}

// ===========================================================================
// IAMAX — argmax |x|
// ===========================================================================

#[test]
fn iamax_f32_picks_largest_absolute_value() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // abs = [1, 2, 3, 9, 4, 5, 6, 2] -> max |.| = 9 at index 3.
    let x = vec![1.0f32, -2.0, 3.0, -9.0, 4.0, 5.0, -6.0, 2.0];
    let got = device_iamax(&handle, x.len() as u32, &x, 1);
    let want = oracle_iamax_f32(&x, 1, x.len());
    assert_eq!(got, want, "iamax basic: got {got}, want {want}");
    assert_eq!(got, 3, "iamax must select the |.|-max element");
}

#[test]
fn iamax_f32_ties_select_smallest_index() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // Three elements share the max |.| = 3; BLAS picks the smallest index (0).
    let x = vec![3.0f32, -3.0, 1.0, 3.0, -2.0];
    let got = device_iamax(&handle, x.len() as u32, &x, 1);
    assert_eq!(got, 0, "iamax tie-break must choose the smallest index");
}

#[test]
fn iamax_f32_strided_matches_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let n = 50usize;
    let incx = 3usize;
    let mut rng = Lcg::new(0x9E9E_AFAF);
    let mut xv: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    xv[37] = -8.5; // unambiguous winner at logical index 37
    // Poison off-stride slots with a HUGE magnitude: a contiguous-read bug
    // would pick a poison index instead of 37.
    let x = strided(&xv, incx, 1e9f32);

    let got = device_iamax(&handle, n as u32, &x, incx as i32);
    let want = oracle_iamax_f32(&x, incx, n);
    assert_eq!(got, want, "iamax strided: got {got}, want {want}");
    assert_eq!(
        got, 37,
        "iamax strided must report the logical (not raw) index"
    );
}

#[test]
fn iamax_f32_multiblock_finds_winner_in_later_block() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    // n = 1000 -> 4 blocks; place the winner in the 4th block (index 777).
    let n = 1000usize;
    let mut rng = Lcg::new(0xB1B1_C2C2);
    let mut x: Vec<f32> = (0..n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    x[777] = 12.0;
    let got = device_iamax(&handle, n as u32, &x, 1);
    assert_eq!(
        got, 777,
        "iamax multiblock must find the cross-block maximum"
    );
}
