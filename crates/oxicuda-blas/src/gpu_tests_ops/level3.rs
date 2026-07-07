//! On-device validation for the BLAS Level-3 symmetric-rank kernels (SYRK,
//! SYR2K).
//!
//! A convergence audit found that the production tensor-core path for f32
//! SYRK/SYR2K on Ampere (`syrk_tc` / `syr2k_tc`) was an incomplete placeholder
//! computing only the `k = 0` term — silently wrong for `K > 1`. That path is
//! now gated off (it falls back to the correct GEMM-based implementation); these
//! tests drive the public ops with `n >= 32` and `K > 1` (so the f32 TC gate
//! *would* have fired) and assert equivalence to an independent CPU oracle,
//! covering Upper/Lower fill, alpha/beta != 1, and the untouched off-triangle.

use oxicuda_memory::DeviceBuffer;

use super::{Lcg, assert_close_f32, assert_close_f64, gpu_fixture};
use crate::handle::BlasHandle;
use crate::level3::gemm_api::gemm;
use crate::types::{FillMode, Layout, MatrixDesc, MatrixDescMut, Transpose};

/// CPU oracle for `C = alpha * op(A) * op(B) + beta * C`, tight row-major,
/// where `ta`/`tb` select whether `A`/`B` are transposed. Physical storage:
/// `op(A)` is `m x k` (so untransposed `A` is `m x k` and transposed `A` is
/// `k x m`); `op(B)` is `k x n` (untransposed `B` is `k x n`, transposed is
/// `n x k`); `C` is `m x n`.
#[allow(clippy::too_many_arguments)]
fn gemm_oracle_f64(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    ta: bool,
    tb: bool,
    alpha: f64,
    beta: f64,
    c_in: &[f64],
) -> Vec<f64> {
    let mut c = c_in.to_vec();
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                let av = if ta { a[p * m + i] } else { a[i * k + p] };
                let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                acc += av * bv;
            }
            c[i * n + j] = alpha * acc + beta * c_in[i * n + j];
        }
    }
    c
}

/// Drives the public `gemm` for one `(trans_a, trans_b)` combination on f32 and
/// asserts CPU-vs-GPU equivalence. Regression guard for the bug where
/// transposed GEMM silently computed the untransposed product.
#[test]
fn gemm_f32_all_transpose_combos_match_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (m, n, k) = (48usize, 40usize, 24usize);
    let mut rng = Lcg::new(0x6E11_0001);
    let (alpha, beta) = (0.85_f32, -0.35_f32);

    for &ta in &[false, true] {
        for &tb in &[false, true] {
            // Physical A: m x k (NoTrans) or k x m (Trans); B: k x n / n x k.
            let a: Vec<f32> = (0..m * k).map(|_| rng.range_f32(-1.0, 1.0)).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
            let c0: Vec<f32> = (0..m * n).map(|_| rng.range_f32(-0.5, 0.5)).collect();

            let (a_rows, a_cols) = if ta { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if tb { (n, k) } else { (k, n) };

            let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
            let d_b = DeviceBuffer::from_host(&b).expect("b h2d");
            let d_c = DeviceBuffer::from_host(&c0).expect("c h2d");

            let a_desc = MatrixDesc::<f32>::from_raw(
                d_a.as_device_ptr(),
                a_rows as u32,
                a_cols as u32,
                a_cols as u32,
                Layout::RowMajor,
            );
            let b_desc = MatrixDesc::<f32>::from_raw(
                d_b.as_device_ptr(),
                b_rows as u32,
                b_cols as u32,
                b_cols as u32,
                Layout::RowMajor,
            );
            let mut c_desc = MatrixDescMut::<f32>::from_raw(
                d_c.as_device_ptr(),
                m as u32,
                n as u32,
                n as u32,
                Layout::RowMajor,
            );

            let ta_flag = if ta {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            let tb_flag = if tb {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            gemm(
                &handle,
                ta_flag,
                tb_flag,
                alpha,
                &a_desc,
                &b_desc,
                beta,
                &mut c_desc,
            )
            .expect("gemm");
            handle.stream().synchronize().expect("sync");

            let mut got = vec![0.0f32; m * n];
            d_c.copy_to_host(&mut got).expect("c d2h");

            let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
            let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
            let c64: Vec<f64> = c0.iter().map(|&x| x as f64).collect();
            let want64 =
                gemm_oracle_f64(&a64, &b64, m, n, k, ta, tb, alpha as f64, beta as f64, &c64);
            let want: Vec<f32> = want64.iter().map(|&x| x as f32).collect();
            assert_close_f32(&got, &want, 1e-3, 1e-3, "gemm_f32_transpose");
        }
    }
}

/// Same coverage on f64 for numerical strictness.
#[test]
fn gemm_f64_all_transpose_combos_match_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (m, n, k) = (40usize, 36usize, 28usize);
    let mut rng = Lcg::new(0x6E11_F64A);
    let (alpha, beta) = (1.2_f64, 0.5_f64);

    for &ta in &[false, true] {
        for &tb in &[false, true] {
            let a: Vec<f64> = (0..m * k).map(|_| rng.range_f64(-1.0, 1.0)).collect();
            let b: Vec<f64> = (0..k * n).map(|_| rng.range_f64(-1.0, 1.0)).collect();
            let c0: Vec<f64> = (0..m * n).map(|_| rng.range_f64(-0.5, 0.5)).collect();

            let (a_rows, a_cols) = if ta { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if tb { (n, k) } else { (k, n) };

            let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
            let d_b = DeviceBuffer::from_host(&b).expect("b h2d");
            let d_c = DeviceBuffer::from_host(&c0).expect("c h2d");

            let a_desc = MatrixDesc::<f64>::from_raw(
                d_a.as_device_ptr(),
                a_rows as u32,
                a_cols as u32,
                a_cols as u32,
                Layout::RowMajor,
            );
            let b_desc = MatrixDesc::<f64>::from_raw(
                d_b.as_device_ptr(),
                b_rows as u32,
                b_cols as u32,
                b_cols as u32,
                Layout::RowMajor,
            );
            let mut c_desc = MatrixDescMut::<f64>::from_raw(
                d_c.as_device_ptr(),
                m as u32,
                n as u32,
                n as u32,
                Layout::RowMajor,
            );

            let ta_flag = if ta {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            let tb_flag = if tb {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            gemm(
                &handle,
                ta_flag,
                tb_flag,
                alpha,
                &a_desc,
                &b_desc,
                beta,
                &mut c_desc,
            )
            .expect("gemm");
            handle.stream().synchronize().expect("sync");

            let mut got = vec![0.0f64; m * n];
            d_c.copy_to_host(&mut got).expect("c d2h");
            let want = gemm_oracle_f64(&a, &b, m, n, k, ta, tb, alpha, beta, &c0);
            assert_close_f64(&got, &want, 1e-10, 1e-10, "gemm_f64_transpose");
        }
    }
}

/// CPU oracle for `C = alpha * A * A^T + beta * C` (row-major, NoTrans),
/// writing only the `fill` triangle and leaving the other triangle as `c_in`.
fn syrk_oracle_f64(
    a: &[f64],
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    c_in: &[f64],
    upper: bool,
) -> Vec<f64> {
    let mut c = c_in.to_vec();
    for i in 0..n {
        for j in 0..n {
            let write = if upper { j >= i } else { i >= j };
            if !write {
                continue;
            }
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * a[j * k + p];
            }
            c[i * n + j] = alpha * acc + beta * c_in[i * n + j];
        }
    }
    c
}

/// CPU oracle for `C = alpha * (A * B^T + B * A^T) + beta * C` (row-major,
/// NoTrans), writing only the `fill` triangle.
#[allow(clippy::too_many_arguments)]
fn syr2k_oracle_f64(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    c_in: &[f64],
    upper: bool,
) -> Vec<f64> {
    let mut c = c_in.to_vec();
    for i in 0..n {
        for j in 0..n {
            let write = if upper { j >= i } else { i >= j };
            if !write {
                continue;
            }
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p] + b[i * k + p] * a[j * k + p];
            }
            c[i * n + j] = alpha * acc + beta * c_in[i * n + j];
        }
    }
    c
}

#[test]
fn syrk_f32_matches_cpu_upper_and_lower() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    // n >= 32 and K > 1 so the (now-gated) f32 tensor-core path would have
    // fired; this proves the fallback is selected and correct.
    let (n, k) = (40usize, 24usize);
    let mut rng = Lcg::new(0x5147_0001);
    let a: Vec<f32> = (0..n * k).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let c0: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-0.5, 0.5)).collect();
    let (alpha, beta) = (0.75_f32, -1.25_f32);

    for upper in [true, false] {
        let fill = if upper {
            FillMode::Upper
        } else {
            FillMode::Lower
        };
        let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
        let d_c = DeviceBuffer::from_host(&c0).expect("c h2d");
        let a_desc = MatrixDesc::<f32>::from_raw(
            d_a.as_device_ptr(),
            n as u32,
            k as u32,
            k as u32,
            Layout::RowMajor,
        );
        let mut c_desc = MatrixDescMut::<f32>::from_raw(
            d_c.as_device_ptr(),
            n as u32,
            n as u32,
            n as u32,
            Layout::RowMajor,
        );
        crate::level3::syrk(
            &handle,
            fill,
            Transpose::NoTrans,
            alpha,
            &a_desc,
            beta,
            &mut c_desc,
        )
        .expect("syrk");
        handle.stream().synchronize().expect("sync");

        let mut got = vec![0.0f32; n * n];
        d_c.copy_to_host(&mut got).expect("c d2h");

        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let c64: Vec<f64> = c0.iter().map(|&x| x as f64).collect();
        let want64 = syrk_oracle_f64(&a64, n, k, alpha as f64, beta as f64, &c64, upper);
        let want: Vec<f32> = want64.iter().map(|&x| x as f32).collect();
        assert_close_f32(
            &got,
            &want,
            1e-3,
            1e-3,
            if upper {
                "syrk_f32_upper"
            } else {
                "syrk_f32_lower"
            },
        );
    }
}

#[test]
fn syrk_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (n, k) = (36usize, 20usize);
    let mut rng = Lcg::new(0x5147_F64A);
    let a: Vec<f64> = (0..n * k).map(|_| rng.range_f64(-1.0, 1.0)).collect();
    let c0: Vec<f64> = (0..n * n).map(|_| rng.range_f64(-0.5, 0.5)).collect();
    let (alpha, beta) = (1.3_f64, 0.4_f64);

    let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
    let d_c = DeviceBuffer::from_host(&c0).expect("c h2d");
    let a_desc = MatrixDesc::<f64>::from_raw(
        d_a.as_device_ptr(),
        n as u32,
        k as u32,
        k as u32,
        Layout::RowMajor,
    );
    let mut c_desc = MatrixDescMut::<f64>::from_raw(
        d_c.as_device_ptr(),
        n as u32,
        n as u32,
        n as u32,
        Layout::RowMajor,
    );
    crate::level3::syrk(
        &handle,
        FillMode::Upper,
        Transpose::NoTrans,
        alpha,
        &a_desc,
        beta,
        &mut c_desc,
    )
    .expect("syrk f64");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f64; n * n];
    d_c.copy_to_host(&mut got).expect("c d2h");
    let want = syrk_oracle_f64(&a, n, k, alpha, beta, &c0, true);
    assert_close_f64(&got, &want, 1e-10, 1e-10, "syrk_f64_upper");
}

#[test]
fn syr2k_f32_matches_cpu_upper_and_lower() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let (n, k) = (40usize, 24usize);
    let mut rng = Lcg::new(0x5232_0001);
    let a: Vec<f32> = (0..n * k).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let c0: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-0.5, 0.5)).collect();
    let (alpha, beta) = (0.9_f32, -0.6_f32);

    for upper in [true, false] {
        let fill = if upper {
            FillMode::Upper
        } else {
            FillMode::Lower
        };
        let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
        let d_b = DeviceBuffer::from_host(&b).expect("b h2d");
        let d_c = DeviceBuffer::from_host(&c0).expect("c h2d");
        let a_desc = MatrixDesc::<f32>::from_raw(
            d_a.as_device_ptr(),
            n as u32,
            k as u32,
            k as u32,
            Layout::RowMajor,
        );
        let b_desc = MatrixDesc::<f32>::from_raw(
            d_b.as_device_ptr(),
            n as u32,
            k as u32,
            k as u32,
            Layout::RowMajor,
        );
        let mut c_desc = MatrixDescMut::<f32>::from_raw(
            d_c.as_device_ptr(),
            n as u32,
            n as u32,
            n as u32,
            Layout::RowMajor,
        );
        crate::level3::syr2k(
            &handle,
            fill,
            Transpose::NoTrans,
            alpha,
            &a_desc,
            &b_desc,
            beta,
            &mut c_desc,
        )
        .expect("syr2k");
        handle.stream().synchronize().expect("sync");

        let mut got = vec![0.0f32; n * n];
        d_c.copy_to_host(&mut got).expect("c d2h");

        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        let c64: Vec<f64> = c0.iter().map(|&x| x as f64).collect();
        let want64 = syr2k_oracle_f64(&a64, &b64, n, k, alpha as f64, beta as f64, &c64, upper);
        let want: Vec<f32> = want64.iter().map(|&x| x as f32).collect();
        assert_close_f32(
            &got,
            &want,
            2e-3,
            2e-3,
            if upper {
                "syr2k_f32_upper"
            } else {
                "syr2k_f32_lower"
            },
        );
    }
}
