//! On-device validation for the strided batched GEMM.
//!
//! A convergence/production audit found that `gemm_strided_batched` launched an
//! 8-parameter kernel with a 17-field argument tuple — a wild-pointer bug that
//! corrupted arbitrary CUDA-context memory. The op now iterates the batch,
//! computing per-batch pointer offsets and launching the naive NoTrans GEMM
//! kernel once per batch. These tests drive the public entry point with
//! `batch_count >= 2` and distinct per-batch data and assert equivalence to an
//! independent CPU oracle, covering both the in-place `D == C` case and the
//! separate-`D` (`beta * C`) epilogue.

use oxicuda_memory::DeviceBuffer;

use super::{Lcg, assert_close_f32, gpu_fixture};
use crate::batched::strided_gemm::gemm_strided_batched;
use crate::handle::BlasHandle;
use crate::types::Transpose;

/// CPU oracle for one batch: `D = alpha * A * B + beta * C`, all tight
/// row-major, `A` is `m x k`, `B` is `k x n`, `C`/`D` are `m x n`.
#[allow(clippy::too_many_arguments)]
fn batch_oracle_f64(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut d = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            d[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
    }
    d
}

/// Separate-`D` variant: `D[i] = alpha * A[i] * B[i] + beta * C[i]`, distinct
/// data per batch, `beta != 0` so the C epilogue is exercised.
#[test]
fn strided_batched_f32_separate_d_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let (m, n, k) = (12usize, 10usize, 8usize);
    let batch = 3usize;
    let (alpha, beta) = (0.9_f32, -0.4_f32);
    let mut rng = Lcg::new(0xB47C_0001);

    // Distinct per-batch data laid out contiguously (tight strides).
    let a: Vec<f32> = (0..batch * m * k)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let b: Vec<f32> = (0..batch * k * n)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let c: Vec<f32> = (0..batch * m * n)
        .map(|_| rng.range_f32(-0.5, 0.5))
        .collect();
    let d_init: Vec<f32> = vec![f32::NAN; batch * m * n]; // must be overwritten

    let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
    let d_b = DeviceBuffer::from_host(&b).expect("b h2d");
    let d_c = DeviceBuffer::from_host(&c).expect("c h2d");
    let d_d = DeviceBuffer::from_host(&d_init).expect("d h2d");

    gemm_strided_batched::<f32>(
        &handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        d_a.as_device_ptr(),
        k as u32,
        (m * k) as i64,
        d_b.as_device_ptr(),
        n as u32,
        (k * n) as i64,
        beta,
        d_c.as_device_ptr(),
        n as u32,
        (m * n) as i64,
        d_d.as_device_ptr(),
        n as u32,
        (m * n) as i64,
        batch as u32,
    )
    .expect("gemm_strided_batched");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; batch * m * n];
    d_d.copy_to_host(&mut got).expect("d d2h");

    let mut want = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_i: Vec<f64> = a[bi * m * k..(bi + 1) * m * k]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let b_i: Vec<f64> = b[bi * k * n..(bi + 1) * k * n]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let c_i: Vec<f64> = c[bi * m * n..(bi + 1) * m * n]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let d_i = batch_oracle_f64(&a_i, &b_i, &c_i, m, n, k, alpha as f64, beta as f64);
        for (dst, &v) in want[bi * m * n..(bi + 1) * m * n]
            .iter_mut()
            .zip(d_i.iter())
        {
            *dst = v as f32;
        }
    }
    assert_close_f32(&got, &want, 1e-3, 1e-3, "strided_batched_f32_separate_d");
}

/// In-place variant: `C == D` (same buffer, same stride), so the kernel updates
/// each batch in place. Uses `batch_count = 2` with distinct data.
#[test]
fn strided_batched_f32_in_place_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");

    let (m, n, k) = (10usize, 14usize, 9usize);
    let batch = 2usize;
    let (alpha, beta) = (1.1_f32, 0.6_f32);
    let mut rng = Lcg::new(0xB47C_1002);

    let a: Vec<f32> = (0..batch * m * k)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let b: Vec<f32> = (0..batch * k * n)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let c: Vec<f32> = (0..batch * m * n)
        .map(|_| rng.range_f32(-0.5, 0.5))
        .collect();

    let d_a = DeviceBuffer::from_host(&a).expect("a h2d");
    let d_b = DeviceBuffer::from_host(&b).expect("b h2d");
    let d_c = DeviceBuffer::from_host(&c).expect("c h2d");

    // D aliases C exactly (same pointer + strides): in-place update.
    gemm_strided_batched::<f32>(
        &handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        d_a.as_device_ptr(),
        k as u32,
        (m * k) as i64,
        d_b.as_device_ptr(),
        n as u32,
        (k * n) as i64,
        beta,
        d_c.as_device_ptr(),
        n as u32,
        (m * n) as i64,
        d_c.as_device_ptr(),
        n as u32,
        (m * n) as i64,
        batch as u32,
    )
    .expect("gemm_strided_batched in place");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; batch * m * n];
    d_c.copy_to_host(&mut got).expect("c d2h");

    let mut want = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_i: Vec<f64> = a[bi * m * k..(bi + 1) * m * k]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let b_i: Vec<f64> = b[bi * k * n..(bi + 1) * k * n]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let c_i: Vec<f64> = c[bi * m * n..(bi + 1) * m * n]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let d_i = batch_oracle_f64(&a_i, &b_i, &c_i, m, n, k, alpha as f64, beta as f64);
        for (dst, &v) in want[bi * m * n..(bi + 1) * m * n]
            .iter_mut()
            .zip(d_i.iter())
        {
            *dst = v as f32;
        }
    }
    assert_close_f32(&got, &want, 1e-3, 1e-3, "strided_batched_f32_in_place");
}
