//! On-device validation for the BLAS level2 kernels.
//!
//! These suites drive the *production* level-2 ops (`gemv`, `ger`, `syr`,
//! `symv`, `trsv`, `trmv`), each of which JIT-compiles its PTX and launches it
//! on the live CUDA device. The result is copied back and compared against an
//! independent CPU oracle that re-derives the BLAS math from first principles.
//!
//! Coverage deliberately exercises the non-trivial paths that hid real bugs in
//! sibling kernels (an SpMM that wrote only one output column, a CSR5 that
//! dropped `beta`): strided vectors, transpose, Upper/Lower fill, Unit/NonUnit
//! diagonals, `alpha`/`beta != 1` *and* `beta = 0`, padded leading dimensions,
//! the symmetric mirror reads in SYMV, and the multi-block TRSV dispatch path.
//!
//! Every test returns early (skips) when no CUDA device is present.

use oxicuda_memory::DeviceBuffer;

use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Transpose};

use super::*;

// ---------------------------------------------------------------------------
// Numeric helper trait: bridges f32/f64 for the generic oracles + asserts.
// ---------------------------------------------------------------------------

/// A real floating type usable by the generic level-2 oracles. Bundles the
/// arithmetic the oracles need on top of [`GpuFloat`], plus a precision-aware
/// closeness assertion so the test bodies stay precision-agnostic.
trait OracleNum:
    GpuFloat
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    /// Builds a value from an `f64` (rounding to the target precision).
    fn from_f64(v: f64) -> Self;
    /// Precision-appropriate slice comparison against a CPU oracle.
    fn assert_slice_close(got: &[Self], exp: &[Self], rel: f64, abs: f64, tag: &str);
}

impl OracleNum for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn assert_slice_close(got: &[Self], exp: &[Self], rel: f64, abs: f64, tag: &str) {
        assert_close_f32(got, exp, rel as f32, abs as f32, tag);
    }
}

impl OracleNum for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn assert_slice_close(got: &[Self], exp: &[Self], rel: f64, abs: f64, tag: &str) {
        assert_close_f64(got, exp, rel, abs, tag);
    }
}

// ---------------------------------------------------------------------------
// Deterministic host-data builders.
// ---------------------------------------------------------------------------

/// Length of the backing buffer for a logical `n`-element vector with stride
/// `inc`: the last element lives at index `(n - 1) * inc`.
fn slen(n: usize, inc: usize) -> usize {
    if n == 0 { 0 } else { 1 + (n - 1) * inc }
}

/// A deterministic random vector of `len` elements in `[lo, hi)`.
fn rand_vec<T: OracleNum>(len: usize, seed: u64, lo: f64, hi: f64) -> Vec<T> {
    let mut rng = Lcg::new(seed);
    (0..len)
        .map(|_| T::from_f64(rng.range_f64(lo, hi)))
        .collect()
}

/// A row-major `rows x cols` matrix with leading dimension `lda` (`lda >=
/// cols`). The padding columns `[cols, lda)` are left as zero and never read.
fn rand_mat<T: OracleNum>(
    rows: usize,
    cols: usize,
    lda: usize,
    seed: u64,
    lo: f64,
    hi: f64,
) -> Vec<T> {
    let mut rng = Lcg::new(seed);
    let mut v = vec![T::gpu_zero(); rows * lda];
    for i in 0..rows {
        for j in 0..cols {
            v[i * lda + j] = T::from_f64(rng.range_f64(lo, hi));
        }
    }
    v
}

/// A diagonally-dominant, genuinely triangular `n x n` row-major matrix.
///
/// Only the triangle selected by `uplo` is populated; the opposite triangle is
/// exactly zero so the matrix is a true triangular operator. The diagonal is
/// set to `diag_val` (use a large sentinel for Unit-diagonal tests to prove the
/// kernel ignores the stored diagonal); off-diagonals are scaled by `1/n` to
/// keep the system extremely well-conditioned.
fn tri_mat<T: OracleNum>(n: usize, lda: usize, uplo: FillMode, seed: u64, diag_val: f64) -> Vec<T> {
    let mut rng = Lcg::new(seed);
    let mut v = vec![T::gpu_zero(); n * lda];
    for i in 0..n {
        for j in 0..n {
            let referenced = match uplo {
                FillMode::Upper => j >= i,
                _ => j <= i,
            };
            if !referenced {
                continue;
            }
            let val = if i == j {
                diag_val
            } else {
                rng.range_f64(-0.5, 0.5) / n as f64
            };
            v[i * lda + j] = T::from_f64(val);
        }
    }
    v
}

// ===========================================================================
// GEMV — y = alpha * op(A) * x + beta * y
// ===========================================================================

struct GemvCase<T: OracleNum> {
    trans: Transpose,
    m: u32,
    n: u32,
    alpha: T,
    beta: T,
    a: Vec<T>,
    lda: u32,
    x: Vec<T>,
    incx: i32,
    y: Vec<T>,
    incy: i32,
}

fn oracle_gemv<T: OracleNum>(c: &GemvCase<T>) -> Vec<T> {
    let lda = c.lda as usize;
    let incx = c.incx as usize;
    let incy = c.incy as usize;
    let (out_len, inner) = match c.trans {
        Transpose::NoTrans => (c.m as usize, c.n as usize),
        _ => (c.n as usize, c.m as usize),
    };
    let mut out = c.y.clone();
    for i in 0..out_len {
        let mut acc = T::gpu_zero();
        for k in 0..inner {
            let a_elem = match c.trans {
                Transpose::NoTrans => c.a[i * lda + k],
                _ => c.a[k * lda + i],
            };
            acc = acc + a_elem * c.x[k * incx];
        }
        out[i * incy] = c.alpha * acc + c.beta * c.y[i * incy];
    }
    out
}

fn run_gemv<T: OracleNum>(fx: &GpuFixture, c: &GemvCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let mut d_y = DeviceBuffer::from_host(&c.y).expect("y h2d");
    let a_desc = MatrixDesc::<T>::from_raw(d_a.as_device_ptr(), c.m, c.n, c.lda, Layout::RowMajor);
    crate::level2::gemv::gemv(
        &handle, c.trans, c.m, c.n, c.alpha, &a_desc, &d_x, c.incx, c.beta, &mut d_y, c.incy,
    )
    .expect("gemv launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.y.len()];
    d_y.copy_to_host(&mut got).expect("y d2h");
    got
}

fn check_gemv<T: OracleNum>(fx: &GpuFixture, c: &GemvCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_gemv(fx, c);
    let exp = oracle_gemv(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

#[test]
fn gemv_f32_notrans_basic() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (6usize, 5usize);
    let case = GemvCase::<f32> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.0,
        beta: 0.0,
        a: rand_mat(m, n, n, 0x1001, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x1002, -1.0, 1.0),
        incx: 1,
        y: rand_vec(m, 0x1003, -0.5, 0.5),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_notrans_basic");
}

#[test]
fn gemv_f32_notrans_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (7usize, 4usize);
    let case = GemvCase::<f32> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 2.0,
        beta: -0.5,
        a: rand_mat(m, n, n, 0x1101, -1.5, 1.5),
        lda: n as u32,
        x: rand_vec(n, 0x1102, -1.0, 1.0),
        incx: 1,
        y: rand_vec(m, 0x1103, -1.0, 1.0),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_notrans_alpha_beta");
}

#[test]
fn gemv_f32_trans_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // A is m x n; Trans => output length n, inner length m.
    let (m, n) = (5usize, 6usize);
    let case = GemvCase::<f32> {
        trans: Transpose::Trans,
        m: m as u32,
        n: n as u32,
        alpha: 0.75,
        beta: 1.25,
        a: rand_mat(m, n, n, 0x1201, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(m, 0x1202, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x1203, -1.0, 1.0),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_trans_alpha_beta");
}

#[test]
fn gemv_f32_conjtrans_real_equals_trans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (4usize, 5usize);
    let case = GemvCase::<f32> {
        trans: Transpose::ConjTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.5,
        beta: -0.25,
        a: rand_mat(m, n, n, 0x1301, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(m, 0x1302, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x1303, -1.0, 1.0),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_conjtrans");
}

#[test]
fn gemv_f32_strided() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (6usize, 5usize);
    let (incx, incy) = (2usize, 3usize);
    let case = GemvCase::<f32> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.25,
        beta: 0.5,
        a: rand_mat(m, n, n, 0x1401, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(slen(n, incx), 0x1402, -1.0, 1.0),
        incx: incx as i32,
        y: rand_vec(slen(m, incy), 0x1403, -1.0, 1.0),
        incy: incy as i32,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_strided");
}

#[test]
fn gemv_f32_padded_lda() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Leading dimension larger than the column count: stride bugs surface here.
    let (m, n, lda) = (5usize, 4usize, 7usize);
    let case = GemvCase::<f32> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.0,
        beta: 0.0,
        a: rand_mat(m, n, lda, 0x1501, -1.0, 1.0),
        lda: lda as u32,
        x: rand_vec(n, 0x1502, -1.0, 1.0),
        incx: 1,
        y: rand_vec(m, 0x1503, -0.5, 0.5),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_padded_lda");
}

#[test]
fn gemv_f32_trans_beta_zero_ignores_old_y() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // beta = 0: existing finite garbage in y must not influence the result.
    let (m, n) = (5usize, 5usize);
    let case = GemvCase::<f32> {
        trans: Transpose::Trans,
        m: m as u32,
        n: n as u32,
        alpha: 1.0,
        beta: 0.0,
        a: rand_mat(m, n, n, 0x1601, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(m, 0x1602, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x1603, 10.0, 20.0),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-4, 1e-4, "gemv_f32_trans_beta_zero");
}

#[test]
fn gemv_f64_notrans_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (8usize, 12usize);
    let case = GemvCase::<f64> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.3,
        beta: 0.4,
        a: rand_mat(m, n, n, 0x1701, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x1702, -1.0, 1.0),
        incx: 1,
        y: rand_vec(m, 0x1703, -0.5, 0.5),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-10, 1e-10, "gemv_f64_notrans");
}

#[test]
fn gemv_f64_trans_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (9usize, 6usize);
    let case = GemvCase::<f64> {
        trans: Transpose::Trans,
        m: m as u32,
        n: n as u32,
        alpha: -0.7,
        beta: 2.0,
        a: rand_mat(m, n, n, 0x1801, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(m, 0x1802, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x1803, -1.0, 1.0),
        incy: 1,
    };
    check_gemv(&fx, &case, 1e-10, 1e-10, "gemv_f64_trans");
}

/// Non-vacuous probe: perturbing one input matrix element must change the
/// output, proving the kernel genuinely reads device memory.
#[test]
fn gemv_f32_reads_device_memory_nonvacuous() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (8usize, 8usize);
    let mut case = GemvCase::<f32> {
        trans: Transpose::NoTrans,
        m: m as u32,
        n: n as u32,
        alpha: 1.0,
        beta: 0.0,
        a: rand_mat(m, n, n, 0x1901, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x1902, 0.25, 1.0),
        incx: 1,
        y: rand_vec(m, 0x1903, -0.5, 0.5),
        incy: 1,
    };
    let clean = run_gemv(&fx, &case);
    case.a[0] += 5.0; // perturb A[0,0] which feeds y[0]
    let dirty = run_gemv(&fx, &case);
    let changed = clean
        .iter()
        .zip(dirty.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-3);
    assert!(
        changed,
        "gemv output unchanged after corrupting A[0,0] — kernel may not read device memory"
    );
}

// ===========================================================================
// GER — A = alpha * x * y^T + A   (rank-1 update)
// ===========================================================================

struct GerCase<T: OracleNum> {
    m: u32,
    n: u32,
    alpha: T,
    x: Vec<T>,
    incx: i32,
    y: Vec<T>,
    incy: i32,
    a: Vec<T>,
    lda: u32,
}

fn oracle_ger<T: OracleNum>(c: &GerCase<T>) -> Vec<T> {
    let (m, n, lda, incx, incy) = (
        c.m as usize,
        c.n as usize,
        c.lda as usize,
        c.incx as usize,
        c.incy as usize,
    );
    let mut out = c.a.clone();
    for i in 0..m {
        for j in 0..n {
            out[i * lda + j] = c.a[i * lda + j] + c.alpha * c.x[i * incx] * c.y[j * incy];
        }
    }
    out
}

fn run_ger<T: OracleNum>(fx: &GpuFixture, c: &GerCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let d_y = DeviceBuffer::from_host(&c.y).expect("y h2d");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let mut a_desc =
        MatrixDescMut::<T>::from_raw(d_a.as_device_ptr(), c.m, c.n, c.lda, Layout::RowMajor);
    crate::level2::ger::ger(
        &handle,
        c.m,
        c.n,
        c.alpha,
        &d_x,
        c.incx,
        &d_y,
        c.incy,
        &mut a_desc,
    )
    .expect("ger launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.a.len()];
    d_a.copy_to_host(&mut got).expect("a d2h");
    got
}

fn check_ger<T: OracleNum>(fx: &GpuFixture, c: &GerCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_ger(fx, c);
    let exp = oracle_ger(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

#[test]
fn ger_f32_basic() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (4usize, 5usize);
    let case = GerCase::<f32> {
        m: m as u32,
        n: n as u32,
        alpha: 1.0,
        x: rand_vec(m, 0x2001, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x2002, -1.0, 1.0),
        incy: 1,
        a: rand_mat(m, n, n, 0x2003, -1.0, 1.0),
        lda: n as u32,
    };
    check_ger(&fx, &case, 1e-4, 1e-4, "ger_f32_basic");
}

#[test]
fn ger_f32_alpha_strided() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (5usize, 4usize);
    let (incx, incy) = (3usize, 2usize);
    let case = GerCase::<f32> {
        m: m as u32,
        n: n as u32,
        alpha: 2.5,
        x: rand_vec(slen(m, incx), 0x2101, -1.0, 1.0),
        incx: incx as i32,
        y: rand_vec(slen(n, incy), 0x2102, -1.0, 1.0),
        incy: incy as i32,
        a: rand_mat(m, n, n, 0x2103, -1.0, 1.0),
        lda: n as u32,
    };
    check_ger(&fx, &case, 1e-4, 1e-4, "ger_f32_alpha_strided");
}

#[test]
fn ger_f32_padded_lda() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n, lda) = (4usize, 4usize, 6usize);
    let case = GerCase::<f32> {
        m: m as u32,
        n: n as u32,
        alpha: -1.5,
        x: rand_vec(m, 0x2201, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x2202, -1.0, 1.0),
        incy: 1,
        a: rand_mat(m, n, lda, 0x2203, -1.0, 1.0),
        lda: lda as u32,
    };
    check_ger(&fx, &case, 1e-4, 1e-4, "ger_f32_padded_lda");
}

#[test]
fn ger_f64_basic() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (m, n) = (6usize, 5usize);
    let case = GerCase::<f64> {
        m: m as u32,
        n: n as u32,
        alpha: 1.7,
        x: rand_vec(m, 0x2301, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x2302, -1.0, 1.0),
        incy: 1,
        a: rand_mat(m, n, n, 0x2303, -1.0, 1.0),
        lda: n as u32,
    };
    check_ger(&fx, &case, 1e-10, 1e-10, "ger_f64_basic");
}

// ===========================================================================
// SYR — A = alpha * x * x^T + A   (symmetric rank-1, one triangle)
// ===========================================================================

struct SyrCase<T: OracleNum> {
    uplo: FillMode,
    n: u32,
    alpha: T,
    x: Vec<T>,
    incx: i32,
    a: Vec<T>,
    lda: u32,
}

fn oracle_syr<T: OracleNum>(c: &SyrCase<T>) -> Vec<T> {
    let (n, lda, incx) = (c.n as usize, c.lda as usize, c.incx as usize);
    let mut out = c.a.clone();
    for i in 0..n {
        for j in 0..n {
            let referenced = match c.uplo {
                FillMode::Upper => j >= i,
                _ => i >= j,
            };
            if referenced {
                out[i * lda + j] = c.a[i * lda + j] + c.alpha * c.x[i * incx] * c.x[j * incx];
            }
        }
    }
    out
}

fn run_syr<T: OracleNum>(fx: &GpuFixture, c: &SyrCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let mut a_desc =
        MatrixDescMut::<T>::from_raw(d_a.as_device_ptr(), c.n, c.n, c.lda, Layout::RowMajor);
    crate::level2::syr::syr(&handle, c.uplo, c.n, c.alpha, &d_x, c.incx, &mut a_desc)
        .expect("syr launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.a.len()];
    d_a.copy_to_host(&mut got).expect("a d2h");
    got
}

fn check_syr<T: OracleNum>(fx: &GpuFixture, c: &SyrCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_syr(fx, c);
    // The oracle leaves the non-stored triangle untouched; a full-matrix
    // compare therefore also verifies the kernel does NOT touch it.
    let exp = oracle_syr(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

#[test]
fn syr_f32_upper() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 5usize;
    let case = SyrCase::<f32> {
        uplo: FillMode::Upper,
        n: n as u32,
        alpha: 1.0,
        x: rand_vec(n, 0x3001, -1.0, 1.0),
        incx: 1,
        a: rand_mat(n, n, n, 0x3002, -1.0, 1.0),
        lda: n as u32,
    };
    check_syr(&fx, &case, 1e-4, 1e-4, "syr_f32_upper");
}

#[test]
fn syr_f32_lower_alpha() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 6usize;
    let case = SyrCase::<f32> {
        uplo: FillMode::Lower,
        n: n as u32,
        alpha: -1.5,
        x: rand_vec(n, 0x3101, -1.0, 1.0),
        incx: 1,
        a: rand_mat(n, n, n, 0x3102, -1.0, 1.0),
        lda: n as u32,
    };
    check_syr(&fx, &case, 1e-4, 1e-4, "syr_f32_lower_alpha");
}

#[test]
fn syr_f32_upper_strided() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 5usize;
    let incx = 2usize;
    let case = SyrCase::<f32> {
        uplo: FillMode::Upper,
        n: n as u32,
        alpha: 2.0,
        x: rand_vec(slen(n, incx), 0x3201, -1.0, 1.0),
        incx: incx as i32,
        a: rand_mat(n, n, n, 0x3202, -1.0, 1.0),
        lda: n as u32,
    };
    check_syr(&fx, &case, 1e-4, 1e-4, "syr_f32_upper_strided");
}

#[test]
fn syr_f64_lower() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 7usize;
    let case = SyrCase::<f64> {
        uplo: FillMode::Lower,
        n: n as u32,
        alpha: 0.9,
        x: rand_vec(n, 0x3301, -1.0, 1.0),
        incx: 1,
        a: rand_mat(n, n, n, 0x3302, -1.0, 1.0),
        lda: n as u32,
    };
    check_syr(&fx, &case, 1e-10, 1e-10, "syr_f64_lower");
}

// ===========================================================================
// SYMV — y = alpha * A * x + beta * y   (A symmetric, one triangle stored)
// ===========================================================================

struct SymvCase<T: OracleNum> {
    uplo: FillMode,
    n: u32,
    alpha: T,
    beta: T,
    a: Vec<T>,
    lda: u32,
    x: Vec<T>,
    incx: i32,
    y: Vec<T>,
    incy: i32,
}

fn oracle_symv<T: OracleNum>(c: &SymvCase<T>) -> Vec<T> {
    let (n, lda, incx, incy) = (
        c.n as usize,
        c.lda as usize,
        c.incx as usize,
        c.incy as usize,
    );
    let mut out = c.y.clone();
    for i in 0..n {
        let mut acc = T::gpu_zero();
        for j in 0..n {
            // Read the symmetric element from the stored triangle.
            let (r, col) = match c.uplo {
                FillMode::Upper => {
                    if i <= j {
                        (i, j)
                    } else {
                        (j, i)
                    }
                }
                _ => {
                    if i >= j {
                        (i, j)
                    } else {
                        (j, i)
                    }
                }
            };
            acc = acc + c.a[r * lda + col] * c.x[j * incx];
        }
        out[i * incy] = c.alpha * acc + c.beta * c.y[i * incy];
    }
    out
}

fn run_symv<T: OracleNum>(fx: &GpuFixture, c: &SymvCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let mut d_y = DeviceBuffer::from_host(&c.y).expect("y h2d");
    let a_desc = MatrixDesc::<T>::from_raw(d_a.as_device_ptr(), c.n, c.n, c.lda, Layout::RowMajor);
    crate::level2::symv::symv(
        &handle, c.uplo, c.n, c.alpha, &a_desc, &d_x, c.incx, c.beta, &mut d_y, c.incy,
    )
    .expect("symv launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.y.len()];
    d_y.copy_to_host(&mut got).expect("y d2h");
    got
}

fn check_symv<T: OracleNum>(fx: &GpuFixture, c: &SymvCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_symv(fx, c);
    let exp = oracle_symv(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

#[test]
fn symv_f32_upper_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 6usize;
    let case = SymvCase::<f32> {
        uplo: FillMode::Upper,
        n: n as u32,
        alpha: 1.5,
        beta: -0.5,
        a: rand_mat(n, n, n, 0x4001, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x4002, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x4003, -1.0, 1.0),
        incy: 1,
    };
    check_symv(&fx, &case, 1e-4, 1e-4, "symv_f32_upper_alpha_beta");
}

#[test]
fn symv_f32_lower_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 7usize;
    let case = SymvCase::<f32> {
        uplo: FillMode::Lower,
        n: n as u32,
        alpha: 0.8,
        beta: 1.2,
        a: rand_mat(n, n, n, 0x4101, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x4102, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x4103, -1.0, 1.0),
        incy: 1,
    };
    check_symv(&fx, &case, 1e-4, 1e-4, "symv_f32_lower_alpha_beta");
}

#[test]
fn symv_f32_upper_strided() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 5usize;
    let (incx, incy) = (2usize, 3usize);
    let case = SymvCase::<f32> {
        uplo: FillMode::Upper,
        n: n as u32,
        alpha: 1.0,
        beta: 0.5,
        a: rand_mat(n, n, n, 0x4201, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(slen(n, incx), 0x4202, -1.0, 1.0),
        incx: incx as i32,
        y: rand_vec(slen(n, incy), 0x4203, -1.0, 1.0),
        incy: incy as i32,
    };
    check_symv(&fx, &case, 1e-4, 1e-4, "symv_f32_upper_strided");
}

#[test]
fn symv_f32_lower_beta_zero() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 6usize;
    let case = SymvCase::<f32> {
        uplo: FillMode::Lower,
        n: n as u32,
        alpha: 1.0,
        beta: 0.0,
        a: rand_mat(n, n, n, 0x4301, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x4302, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x4303, 5.0, 9.0),
        incy: 1,
    };
    check_symv(&fx, &case, 1e-4, 1e-4, "symv_f32_lower_beta_zero");
}

#[test]
fn symv_f64_lower() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 8usize;
    let case = SymvCase::<f64> {
        uplo: FillMode::Lower,
        n: n as u32,
        alpha: 1.1,
        beta: -0.3,
        a: rand_mat(n, n, n, 0x4401, -1.0, 1.0),
        lda: n as u32,
        x: rand_vec(n, 0x4402, -1.0, 1.0),
        incx: 1,
        y: rand_vec(n, 0x4403, -1.0, 1.0),
        incy: 1,
    };
    check_symv(&fx, &case, 1e-10, 1e-10, "symv_f64_lower");
}

// ===========================================================================
// TRSV — solve op(A) * x = b   (triangular, in-place over x)
// ===========================================================================

struct TrsvCase<T: OracleNum> {
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: u32,
    a: Vec<T>,
    lda: u32,
    x: Vec<T>,
    incx: i32,
}

fn oracle_trsv<T: OracleNum>(c: &TrsvCase<T>) -> Vec<T> {
    let (n, lda, incx) = (c.n as usize, c.lda as usize, c.incx as usize);
    let use_trans = !matches!(c.trans, Transpose::NoTrans);
    let forward = matches!(c.uplo, FillMode::Upper) == use_trans;
    let unit = matches!(c.diag, DiagType::Unit);
    let op = |i: usize, j: usize| -> T {
        if use_trans {
            c.a[j * lda + i]
        } else {
            c.a[i * lda + j]
        }
    };
    let mut x = c.x.clone();
    if forward {
        for i in 0..n {
            let mut s = T::gpu_zero();
            for j in 0..i {
                s = s + op(i, j) * x[j * incx];
            }
            let diff = x[i * incx] - s;
            x[i * incx] = if unit { diff } else { diff / op(i, i) };
        }
    } else {
        for i in (0..n).rev() {
            let mut s = T::gpu_zero();
            for j in (i + 1)..n {
                s = s + op(i, j) * x[j * incx];
            }
            let diff = x[i * incx] - s;
            x[i * incx] = if unit { diff } else { diff / op(i, i) };
        }
    }
    x
}

fn run_trsv<T: OracleNum>(fx: &GpuFixture, c: &TrsvCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let mut d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let a_desc = MatrixDesc::<T>::from_raw(d_a.as_device_ptr(), c.n, c.n, c.lda, Layout::RowMajor);
    crate::level2::trsv::trsv(
        &handle, c.uplo, c.trans, c.diag, c.n, &a_desc, &mut d_x, c.incx,
    )
    .expect("trsv launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.x.len()];
    d_x.copy_to_host(&mut got).expect("x d2h");
    got
}

fn check_trsv<T: OracleNum>(fx: &GpuFixture, c: &TrsvCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_trsv(fx, c);
    let exp = oracle_trsv(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

/// Builds a TRSV case with a well-conditioned triangular `A` and random RHS.
fn trsv_case<T: OracleNum>(
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: usize,
    seed: u64,
) -> TrsvCase<T> {
    // Unit-diagonal tests store a large sentinel diagonal that the kernel must
    // ignore; non-unit tests store a dominant diagonal of 2.0.
    let diag_val = if matches!(diag, DiagType::Unit) {
        9.0
    } else {
        2.0
    };
    TrsvCase {
        uplo,
        trans,
        diag,
        n: n as u32,
        a: tri_mat(n, n, uplo, seed, diag_val),
        lda: n as u32,
        x: rand_vec(n, seed ^ 0xABCD, -1.0, 1.0),
        incx: 1,
    }
}

#[test]
fn trsv_f32_lower_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trsv_case::<f32>(
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        48,
        0x5001,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_lower_notrans_nonunit");
}

#[test]
fn trsv_f32_upper_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trsv_case::<f32>(
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        48,
        0x5101,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_upper_notrans_nonunit");
}

#[test]
fn trsv_f32_lower_notrans_unit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trsv_case::<f32>(
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::Unit,
        40,
        0x5201,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_lower_notrans_unit");
}

#[test]
fn trsv_f32_upper_trans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Upper + Trans => forward substitution over U^T.
    let case = trsv_case::<f32>(
        FillMode::Upper,
        Transpose::Trans,
        DiagType::NonUnit,
        48,
        0x5301,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_upper_trans_nonunit");
}

#[test]
fn trsv_f32_lower_trans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Lower + Trans => backward substitution over L^T.
    let case = trsv_case::<f32>(
        FillMode::Lower,
        Transpose::Trans,
        DiagType::NonUnit,
        48,
        0x5401,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_lower_trans_nonunit");
}

#[test]
fn trsv_f32_upper_trans_unit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trsv_case::<f32>(
        FillMode::Upper,
        Transpose::Trans,
        DiagType::Unit,
        40,
        0x5501,
    );
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_upper_trans_unit");
}

#[test]
fn trsv_f32_strided_lower_notrans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 40usize;
    let incx = 2usize;
    let case = TrsvCase::<f32> {
        uplo: FillMode::Lower,
        trans: Transpose::NoTrans,
        diag: DiagType::NonUnit,
        n: n as u32,
        a: tri_mat(n, n, FillMode::Lower, 0x5601, 2.0),
        lda: n as u32,
        x: rand_vec(slen(n, incx), 0x5602, -1.0, 1.0),
        incx: incx as i32,
    };
    check_trsv(&fx, &case, 1e-3, 1e-3, "trsv_f32_strided_lower_notrans");
}

#[test]
fn trsv_f64_upper_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trsv_case::<f64>(
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        64,
        0x5701,
    );
    check_trsv(&fx, &case, 1e-10, 1e-10, "trsv_f64_upper_notrans_nonunit");
}

/// Independent residual check: solving `L x = b` and then forming `L x` must
/// recover `b`. Computed in f64 (the inputs are f64) so it validates the device
/// solve against the defining equation, not just against another substitution.
#[test]
fn trsv_f64_residual_lower_notrans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 56usize;
    let a = tri_mat::<f64>(n, n, FillMode::Lower, 0x5801, 2.0);
    let b = rand_vec::<f64>(n, 0x5802, -1.0, 1.0);
    let case = TrsvCase::<f64> {
        uplo: FillMode::Lower,
        trans: Transpose::NoTrans,
        diag: DiagType::NonUnit,
        n: n as u32,
        a: a.clone(),
        lda: n as u32,
        x: b.clone(),
        incx: 1,
    };
    let x = run_trsv(&fx, &case);
    for i in 0..n {
        let mut s = 0.0f64;
        for j in 0..=i {
            s += a[i * n + j] * x[j];
        }
        assert!(
            (s - b[i]).abs() <= 1e-9 + 1e-9 * b[i].abs(),
            "trsv residual mismatch at row {i}: (L x)={s} vs b={}",
            b[i]
        );
    }
}

/// Multi-block dispatch path (`n > TRSV_SINGLE_BLOCK_MAX = 4096`). Validates
/// the blocked diagonal-solve + off-diagonal GEMV-update pointer math executes
/// correctly end-to-end on the device.
#[test]
fn trsv_f32_blocked_lower_notrans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 4097usize; // one past the single-block ceiling
    let case = TrsvCase::<f32> {
        uplo: FillMode::Lower,
        trans: Transpose::NoTrans,
        diag: DiagType::NonUnit,
        n: n as u32,
        a: tri_mat(n, n, FillMode::Lower, 0x5901, 2.0),
        lda: n as u32,
        x: rand_vec(n, 0x5902, -1.0, 1.0),
        incx: 1,
    };
    check_trsv(&fx, &case, 2e-3, 2e-3, "trsv_f32_blocked_lower_notrans");
}

// ===========================================================================
// TRMV — x = op(A) * x   (triangular, in-place over x)
// ===========================================================================

struct TrmvCase<T: OracleNum> {
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: u32,
    a: Vec<T>,
    lda: u32,
    x: Vec<T>,
    incx: i32,
}

fn oracle_trmv<T: OracleNum>(c: &TrmvCase<T>) -> Vec<T> {
    let (n, lda, incx) = (c.n as usize, c.lda as usize, c.incx as usize);
    let use_trans = !matches!(c.trans, Transpose::NoTrans);
    let iter_upper = matches!(c.uplo, FillMode::Upper) != use_trans;
    let unit = matches!(c.diag, DiagType::Unit);
    let op = |i: usize, j: usize| -> T {
        if use_trans {
            c.a[j * lda + i]
        } else {
            c.a[i * lda + j]
        }
    };
    let mut out = c.x.clone();
    for i in 0..n {
        let mut acc = T::gpu_zero();
        if iter_upper {
            for j in i..n {
                if unit && j == i {
                    continue;
                }
                acc = acc + op(i, j) * c.x[j * incx];
            }
        } else {
            for j in 0..=i {
                if unit && j == i {
                    continue;
                }
                acc = acc + op(i, j) * c.x[j * incx];
            }
        }
        if unit {
            acc = acc + c.x[i * incx];
        }
        out[i * incx] = acc;
    }
    out
}

fn run_trmv<T: OracleNum>(fx: &GpuFixture, c: &TrmvCase<T>) -> Vec<T> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&c.a).expect("a h2d");
    let mut d_x = DeviceBuffer::from_host(&c.x).expect("x h2d");
    let a_desc = MatrixDesc::<T>::from_raw(d_a.as_device_ptr(), c.n, c.n, c.lda, Layout::RowMajor);
    crate::level2::trmv::trmv(
        &handle, c.uplo, c.trans, c.diag, c.n, &a_desc, &mut d_x, c.incx,
    )
    .expect("trmv launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![T::gpu_zero(); c.x.len()];
    d_x.copy_to_host(&mut got).expect("x d2h");
    got
}

fn check_trmv<T: OracleNum>(fx: &GpuFixture, c: &TrmvCase<T>, rel: f64, abs: f64, tag: &str) {
    let got = run_trmv(fx, c);
    let exp = oracle_trmv(c);
    T::assert_slice_close(&got, &exp, rel, abs, tag);
}

/// Builds a TRMV case. Unit-diagonal tests store a large sentinel diagonal that
/// must be ignored in favour of the implicit 1.
fn trmv_case<T: OracleNum>(
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: usize,
    seed: u64,
) -> TrmvCase<T> {
    let diag_val = if matches!(diag, DiagType::Unit) {
        7.0
    } else {
        2.0
    };
    TrmvCase {
        uplo,
        trans,
        diag,
        n: n as u32,
        a: tri_mat(n, n, uplo, seed, diag_val),
        lda: n as u32,
        x: rand_vec(n, seed ^ 0x1234, -1.0, 1.0),
        incx: 1,
    }
}

#[test]
fn trmv_f32_upper_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        32,
        0x6001,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_upper_notrans_nonunit");
}

#[test]
fn trmv_f32_lower_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        32,
        0x6101,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_lower_notrans_nonunit");
}

#[test]
fn trmv_f32_upper_trans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Upper,
        Transpose::Trans,
        DiagType::NonUnit,
        32,
        0x6201,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_upper_trans_nonunit");
}

#[test]
fn trmv_f32_lower_trans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Lower,
        Transpose::Trans,
        DiagType::NonUnit,
        32,
        0x6301,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_lower_trans_nonunit");
}

#[test]
fn trmv_f32_upper_notrans_unit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::Unit,
        32,
        0x6401,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_upper_notrans_unit");
}

#[test]
fn trmv_f32_lower_trans_unit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f32>(
        FillMode::Lower,
        Transpose::Trans,
        DiagType::Unit,
        32,
        0x6501,
    );
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_lower_trans_unit");
}

#[test]
fn trmv_f32_strided_upper_notrans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 24usize;
    let incx = 2usize;
    let case = TrmvCase::<f32> {
        uplo: FillMode::Upper,
        trans: Transpose::NoTrans,
        diag: DiagType::NonUnit,
        n: n as u32,
        a: tri_mat(n, n, FillMode::Upper, 0x6601, 2.0),
        lda: n as u32,
        x: rand_vec(slen(n, incx), 0x6602, -1.0, 1.0),
        incx: incx as i32,
    };
    check_trmv(&fx, &case, 1e-3, 1e-3, "trmv_f32_strided_upper_notrans");
}

#[test]
fn trmv_f64_lower_notrans_nonunit() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = trmv_case::<f64>(
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        40,
        0x6701,
    );
    check_trmv(&fx, &case, 1e-10, 1e-10, "trmv_f64_lower_notrans_nonunit");
}
