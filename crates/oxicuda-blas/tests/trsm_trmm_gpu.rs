//! GPU numerical tests for TRSM (triangular solve) and TRMM (triangular
//! multiply).
//!
//! Each test builds a triangular matrix A and a right-hand-side / operand
//! matrix B on the host, runs the GPU routine, and cross-checks the device
//! result against a straightforward host reference implementation.
//!
//! Following the crate convention for device-requiring code (see
//! `benches/gemm_f32_4096.rs`), every test acquires the GPU through
//! [`try_handle`], which returns `None` on any platform without a usable
//! CUDA driver — the test then skips instead of failing.

use std::sync::Arc;

use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::level3::{trmm, trsm};
use oxicuda_blas::types::{DiagType, FillMode, Layout, MatrixDesc, MatrixDescMut, Side, Transpose};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// GPU acquisition helper
// ---------------------------------------------------------------------------

/// Attempts to initialise the driver and build a [`BlasHandle`].
///
/// Returns `None` on any platform without a working CUDA driver, so the
/// caller can skip the test gracefully. The returned [`Arc<Context>`] must
/// be kept alive for as long as any device buffer is in use.
fn try_handle() -> Option<(Arc<Context>, BlasHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = BlasHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

// ---------------------------------------------------------------------------
// Host helpers (row-major, all `f32` unless noted)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random value in `[-0.5, 0.5)` from a linear index.
fn pseudo(i: usize, salt: u64) -> f32 {
    // A small splitmix-style hash keeps the matrices reproducible.
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    let frac = ((x >> 40) as f32) / ((1u64 << 24) as f32);
    frac - 0.5
}

/// Builds an `n x n` row-major triangular matrix.
///
/// `upper` selects which triangle carries data; the off-triangle is zeroed.
/// The diagonal is offset to keep the matrix well away from singular so the
/// solve is numerically stable.
fn make_triangular(n: usize, upper: bool, salt: u64) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let in_tri = if upper { c >= r } else { c <= r };
            if !in_tri {
                continue;
            }
            if r == c {
                // Diagonal: magnitude in [2.0, 3.0).
                a[r * n + c] = 2.0 + (pseudo(r * n + c, salt) + 0.5);
            } else {
                a[r * n + c] = pseudo(r * n + c, salt);
            }
        }
    }
    a
}

/// Builds an `rows x cols` row-major dense matrix of pseudo-random values.
fn make_dense(rows: usize, cols: usize, salt: u64) -> Vec<f32> {
    (0..rows * cols).map(|i| pseudo(i, salt)).collect()
}

/// Reads `op(A)[i, j]` from a row-major `n x n` matrix, honouring the
/// transpose flag and a unit diagonal.
fn op_a(a: &[f32], n: usize, i: usize, j: usize, transposed: bool, unit_diag: bool) -> f32 {
    if unit_diag && i == j {
        return 1.0;
    }
    if transposed {
        a[j * n + i]
    } else {
        a[i * n + j]
    }
}

/// Host reference TRSM: solves `op(A) X = alpha B` (Left) or
/// `X op(A) = alpha B` (Right). `b` is overwritten with the solution.
#[allow(clippy::too_many_arguments)]
fn ref_trsm(
    side: Side,
    upper: bool,
    transposed: bool,
    unit_diag: bool,
    n: usize,
    rhs: usize,
    alpha: f32,
    a: &[f32],
    b: &mut [f32],
) {
    // `op(A)` is effectively upper for (upper && !trans) || (!upper && trans).
    let op_upper = upper ^ transposed;

    match side {
        Side::Left => {
            // B is n x rhs (row-major).
            for c in 0..rhs {
                if op_upper {
                    // Back substitution.
                    for i in (0..n).rev() {
                        let mut acc = alpha * b[i * rhs + c];
                        for k in (i + 1)..n {
                            acc -= op_a(a, n, i, k, transposed, unit_diag) * b[k * rhs + c];
                        }
                        let diag = if unit_diag {
                            1.0
                        } else {
                            op_a(a, n, i, i, transposed, false)
                        };
                        b[i * rhs + c] = acc / diag;
                    }
                } else {
                    // Forward substitution.
                    for i in 0..n {
                        let mut acc = alpha * b[i * rhs + c];
                        for k in 0..i {
                            acc -= op_a(a, n, i, k, transposed, unit_diag) * b[k * rhs + c];
                        }
                        let diag = if unit_diag {
                            1.0
                        } else {
                            op_a(a, n, i, i, transposed, false)
                        };
                        b[i * rhs + c] = acc / diag;
                    }
                }
            }
        }
        Side::Right => {
            // B is rhs x n (row-major); solve each row independently.
            for r in 0..rhs {
                if op_upper {
                    // Forward over columns.
                    for i in 0..n {
                        let mut acc = alpha * b[r * n + i];
                        for k in 0..i {
                            acc -= b[r * n + k] * op_a(a, n, k, i, transposed, unit_diag);
                        }
                        let diag = if unit_diag {
                            1.0
                        } else {
                            op_a(a, n, i, i, transposed, false)
                        };
                        b[r * n + i] = acc / diag;
                    }
                } else {
                    // Back over columns.
                    for i in (0..n).rev() {
                        let mut acc = alpha * b[r * n + i];
                        for k in (i + 1)..n {
                            acc -= b[r * n + k] * op_a(a, n, k, i, transposed, unit_diag);
                        }
                        let diag = if unit_diag {
                            1.0
                        } else {
                            op_a(a, n, i, i, transposed, false)
                        };
                        b[r * n + i] = acc / diag;
                    }
                }
            }
        }
    }
}

/// Host reference TRMM: `alpha op(A) B` (Left) or `alpha B op(A)` (Right).
///
/// The off-triangle of `a` is assumed already zeroed (as produced by
/// [`make_triangular`]), so a plain dense multiply yields the triangular
/// result and no explicit `fill_mode` is needed.
#[allow(clippy::too_many_arguments)]
fn ref_trmm(
    side: Side,
    transposed: bool,
    unit_diag: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    match side {
        Side::Left => {
            // A is m x m, B is m x n.
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0.0f32;
                    for k in 0..m {
                        acc += op_a(a, m, r, k, transposed, unit_diag) * b[k * n + c];
                    }
                    out[r * n + c] = alpha * acc;
                }
            }
        }
        Side::Right => {
            // A is n x n, B is m x n.
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0.0f32;
                    for k in 0..n {
                        acc += b[r * n + k] * op_a(a, n, k, c, transposed, unit_diag);
                    }
                    out[r * n + c] = alpha * acc;
                }
            }
        }
    }
    out
}

/// Asserts every element of `got` is within a relative+absolute tolerance of
/// `want`.
fn assert_close(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1.0e-3 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{label}: element {idx} mismatch: got {g}, want {w} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// TRSM GPU tests
// ---------------------------------------------------------------------------

/// Drives one TRSM configuration end to end and cross-checks the result.
#[allow(clippy::too_many_arguments)]
fn run_trsm_case(
    handle: &BlasHandle,
    side: Side,
    fill: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: usize,
    rhs: usize,
    alpha: f32,
) {
    let upper = fill == FillMode::Upper;
    let transposed = trans != Transpose::NoTrans;
    let unit = diag == DiagType::Unit;

    let a_host = make_triangular(n, upper, 0x1234 ^ (n as u64));
    // B dimensions depend on the side: Left -> n x rhs, Right -> rhs x n.
    let (b_rows, b_cols) = match side {
        Side::Left => (n, rhs),
        Side::Right => (rhs, n),
    };
    let b_host = make_dense(b_rows, b_cols, 0xABCD ^ (rhs as u64));

    // Host reference solution.
    let mut want = b_host.clone();
    ref_trsm(
        side, upper, transposed, unit, n, rhs, alpha, &a_host, &mut want,
    );

    // Device buffers.
    let a_dev = DeviceBuffer::<f32>::from_host(&a_host).expect("alloc A");
    let mut b_dev = DeviceBuffer::<f32>::from_host(&b_host).expect("alloc B");

    let a_desc =
        MatrixDesc::<f32>::from_buffer(&a_dev, n as u32, n as u32, Layout::RowMajor).expect("A");
    let mut b_desc = MatrixDescMut::<f32>::from_buffer(
        &mut b_dev,
        b_rows as u32,
        b_cols as u32,
        Layout::RowMajor,
    )
    .expect("B");

    trsm::<f32>(handle, side, fill, trans, diag, alpha, &a_desc, &mut b_desc).expect("trsm launch");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; b_host.len()];
    b_dev.copy_to_host(&mut got).expect("copy back");

    let label =
        format!("trsm side={side:?} fill={fill:?} trans={trans:?} diag={diag:?} n={n} rhs={rhs}");
    assert_close(&got, &want, &label);
}

#[test]
fn trsm_left_lower_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        96,
        40,
        1.0,
    );
}

#[test]
fn trsm_left_upper_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        96,
        40,
        1.0,
    );
}

#[test]
fn trsm_left_lower_trans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::Trans,
        DiagType::NonUnit,
        80,
        24,
        1.0,
    );
}

#[test]
fn trsm_left_upper_trans_unit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::Trans,
        DiagType::Unit,
        72,
        17,
        1.0,
    );
}

#[test]
fn trsm_left_lower_notrans_unit_alpha() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    // Non-unit alpha exercises the up-front B scaling path.
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::Unit,
        64,
        32,
        2.5,
    );
}

#[test]
fn trsm_right_upper_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Right,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        88,
        36,
        1.0,
    );
}

#[test]
fn trsm_right_lower_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Right,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        88,
        36,
        1.0,
    );
}

#[test]
fn trsm_right_lower_trans_nonunit_alpha() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Right,
        FillMode::Lower,
        Transpose::Trans,
        DiagType::NonUnit,
        70,
        21,
        0.75,
    );
}

#[test]
fn trsm_single_block_small() {
    // n <= block size exercises the single-diagonal-block path with no
    // trailing GEMM.
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trsm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        32,
        16,
        1.0,
    );
}

// ---------------------------------------------------------------------------
// TRMM GPU tests
// ---------------------------------------------------------------------------

/// Drives one TRMM configuration end to end and cross-checks the result.
#[allow(clippy::too_many_arguments)]
fn run_trmm_case(
    handle: &BlasHandle,
    side: Side,
    fill: FillMode,
    trans: Transpose,
    diag: DiagType,
    m: usize,
    n: usize,
    alpha: f32,
) {
    let upper = fill == FillMode::Upper;
    let transposed = trans != Transpose::NoTrans;
    let unit = diag == DiagType::Unit;

    // A is m x m for Left, n x n for Right.
    let tri_n = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let a_host = make_triangular(tri_n, upper, 0x55AA ^ (tri_n as u64));
    let b_host = make_dense(m, n, 0x3C3C ^ (n as u64));

    let want = ref_trmm(side, transposed, unit, m, n, alpha, &a_host, &b_host);

    let a_dev = DeviceBuffer::<f32>::from_host(&a_host).expect("alloc A");
    let mut b_dev = DeviceBuffer::<f32>::from_host(&b_host).expect("alloc B");

    let a_desc =
        MatrixDesc::<f32>::from_buffer(&a_dev, tri_n as u32, tri_n as u32, Layout::RowMajor)
            .expect("A");
    let mut b_desc =
        MatrixDescMut::<f32>::from_buffer(&mut b_dev, m as u32, n as u32, Layout::RowMajor)
            .expect("B");

    trmm::<f32>(handle, side, fill, trans, diag, alpha, &a_desc, &mut b_desc).expect("trmm launch");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; b_host.len()];
    b_dev.copy_to_host(&mut got).expect("copy back");

    let label =
        format!("trmm side={side:?} fill={fill:?} trans={trans:?} diag={diag:?} m={m} n={n}");
    assert_close(&got, &want, &label);
}

#[test]
fn trmm_left_upper_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        64,
        48,
        1.0,
    );
}

#[test]
fn trmm_left_lower_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        64,
        48,
        1.0,
    );
}

#[test]
fn trmm_left_upper_trans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::Trans,
        DiagType::NonUnit,
        50,
        37,
        1.0,
    );
}

#[test]
fn trmm_left_lower_trans_unit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::Trans,
        DiagType::Unit,
        48,
        29,
        1.0,
    );
}

#[test]
fn trmm_left_upper_notrans_unit_alpha() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::Unit,
        40,
        40,
        -1.5,
    );
}

#[test]
fn trmm_right_lower_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Right,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        44,
        56,
        1.0,
    );
}

#[test]
fn trmm_right_upper_notrans_nonunit() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Right,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        44,
        56,
        1.0,
    );
}

#[test]
fn trmm_right_upper_trans_unit_alpha() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };
    run_trmm_case(
        &handle,
        Side::Right,
        FillMode::Upper,
        Transpose::Trans,
        DiagType::Unit,
        33,
        41,
        3.0,
    );
}

// ---------------------------------------------------------------------------
// TRSM <-> TRMM round-trip
// ---------------------------------------------------------------------------

/// Solving `op(A) X = B` and then multiplying `op(A) X` must recover `B`.
#[test]
fn trsm_then_trmm_round_trip() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };

    let n = 64usize;
    let rhs = 48usize;
    let a_host = make_triangular(n, false, 0x7777);
    let b_host = make_dense(n, rhs, 0x9999);

    let a_dev = DeviceBuffer::<f32>::from_host(&a_host).expect("alloc A");
    let mut x_dev = DeviceBuffer::<f32>::from_host(&b_host).expect("alloc B");

    let a_desc =
        MatrixDesc::<f32>::from_buffer(&a_dev, n as u32, n as u32, Layout::RowMajor).expect("A");

    // Step 1: X = A^{-1} B  (overwrites x_dev).
    {
        let mut x_desc =
            MatrixDescMut::<f32>::from_buffer(&mut x_dev, n as u32, rhs as u32, Layout::RowMajor)
                .expect("X");
        trsm::<f32>(
            &handle,
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
            1.0,
            &a_desc,
            &mut x_desc,
        )
        .expect("trsm");
    }

    // Step 2: X := A X  -> should recover the original B.
    {
        let mut x_desc =
            MatrixDescMut::<f32>::from_buffer(&mut x_dev, n as u32, rhs as u32, Layout::RowMajor)
                .expect("X");
        trmm::<f32>(
            &handle,
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
            1.0,
            &a_desc,
            &mut x_desc,
        )
        .expect("trmm");
    }
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; b_host.len()];
    x_dev.copy_to_host(&mut got).expect("copy back");
    assert_close(&got, &b_host, "trsm->trmm round trip");
}

// ---------------------------------------------------------------------------
// f64 coverage
// ---------------------------------------------------------------------------

/// f64 TRSM cross-checked against an `f64` host solve.
#[test]
fn trsm_f64_left_lower_notrans() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };

    let n = 48usize;
    let rhs = 24usize;

    // Build an f64 lower-triangular matrix with a strong diagonal.
    let mut a_host = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..=r {
            a_host[r * n + c] = if r == c {
                2.0 + f64::from(pseudo(r * n + c, 0x11) + 0.5)
            } else {
                f64::from(pseudo(r * n + c, 0x11))
            };
        }
    }
    let b_host: Vec<f64> = (0..n * rhs).map(|i| f64::from(pseudo(i, 0x22))).collect();

    // Host reference (forward substitution).
    let mut want = b_host.clone();
    for c in 0..rhs {
        for i in 0..n {
            let mut acc = want[i * rhs + c];
            for k in 0..i {
                acc -= a_host[i * n + k] * want[k * rhs + c];
            }
            want[i * rhs + c] = acc / a_host[i * n + i];
        }
    }

    let a_dev = DeviceBuffer::<f64>::from_host(&a_host).expect("alloc A");
    let mut b_dev = DeviceBuffer::<f64>::from_host(&b_host).expect("alloc B");

    let a_desc =
        MatrixDesc::<f64>::from_buffer(&a_dev, n as u32, n as u32, Layout::RowMajor).expect("A");
    let mut b_desc =
        MatrixDescMut::<f64>::from_buffer(&mut b_dev, n as u32, rhs as u32, Layout::RowMajor)
            .expect("B");

    trsm::<f64>(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        1.0,
        &a_desc,
        &mut b_desc,
    )
    .expect("trsm f64");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f64; b_host.len()];
    b_dev.copy_to_host(&mut got).expect("copy back");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1.0e-9 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "trsm f64 element {idx}: got {g}, want {w} (tol {tol})"
        );
    }
}

/// f64 TRMM cross-checked against an `f64` host multiply.
#[test]
fn trmm_f64_left_upper_notrans() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };

    let n = 40usize;
    let cols = 28usize;

    let mut a_host = vec![0.0f64; n * n];
    for r in 0..n {
        for c in r..n {
            a_host[r * n + c] = if r == c {
                1.0 + f64::from(pseudo(r * n + c, 0x33) + 0.5)
            } else {
                f64::from(pseudo(r * n + c, 0x33))
            };
        }
    }
    let b_host: Vec<f64> = (0..n * cols).map(|i| f64::from(pseudo(i, 0x44))).collect();

    // Host reference: out = A * B (A upper).
    let mut want = vec![0.0f64; n * cols];
    for r in 0..n {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for k in r..n {
                acc += a_host[r * n + k] * b_host[k * cols + c];
            }
            want[r * cols + c] = acc;
        }
    }

    let a_dev = DeviceBuffer::<f64>::from_host(&a_host).expect("alloc A");
    let mut b_dev = DeviceBuffer::<f64>::from_host(&b_host).expect("alloc B");

    let a_desc =
        MatrixDesc::<f64>::from_buffer(&a_dev, n as u32, n as u32, Layout::RowMajor).expect("A");
    let mut b_desc =
        MatrixDescMut::<f64>::from_buffer(&mut b_dev, n as u32, cols as u32, Layout::RowMajor)
            .expect("B");

    trmm::<f64>(
        &handle,
        Side::Left,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        1.0,
        &a_desc,
        &mut b_desc,
    )
    .expect("trmm f64");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f64; b_host.len()];
    b_dev.copy_to_host(&mut got).expect("copy back");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1.0e-9 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "trmm f64 element {idx}: got {g}, want {w} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Column-major coverage
// ---------------------------------------------------------------------------

/// TRMM with column-major operands exercises the layout-aware strides.
#[test]
fn trmm_col_major_left_lower() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skip: no GPU");
        return;
    };

    let n = 48usize;
    let cols = 32usize;

    // Build A and B in row-major logical form, then transpose into
    // column-major storage so the descriptors can use Layout::ColMajor.
    let a_row = make_triangular(n, false, 0xC0FE);
    let b_row = make_dense(n, cols, 0xBEEF);
    let want = ref_trmm(
        Side::Left,
        false, // transposed
        false, // unit diagonal
        n,
        cols,
        1.0,
        &a_row,
        &b_row,
    );

    // Column-major storage: element (r, c) at c * rows + r.
    let mut a_col = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            a_col[c * n + r] = a_row[r * n + c];
        }
    }
    let mut b_col = vec![0.0f32; n * cols];
    for r in 0..n {
        for c in 0..cols {
            b_col[c * n + r] = b_row[r * cols + c];
        }
    }

    let a_dev = DeviceBuffer::<f32>::from_host(&a_col).expect("alloc A");
    let mut b_dev = DeviceBuffer::<f32>::from_host(&b_col).expect("alloc B");

    let a_desc =
        MatrixDesc::<f32>::from_buffer(&a_dev, n as u32, n as u32, Layout::ColMajor).expect("A");
    let mut b_desc =
        MatrixDescMut::<f32>::from_buffer(&mut b_dev, n as u32, cols as u32, Layout::ColMajor)
            .expect("B");

    trmm::<f32>(
        &handle,
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::NonUnit,
        1.0,
        &a_desc,
        &mut b_desc,
    )
    .expect("trmm col-major");
    handle.stream().synchronize().expect("sync");

    // Read back column-major storage and compare against the row-major
    // reference element by element.
    let mut got_col = vec![0.0f32; b_col.len()];
    b_dev.copy_to_host(&mut got_col).expect("copy back");
    for r in 0..n {
        for c in 0..cols {
            let g = got_col[c * n + r];
            let w = want[r * cols + c];
            let tol = 1.0e-3 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "trmm col-major ({r},{c}): got {g}, want {w} (tol {tol})"
            );
        }
    }
}
