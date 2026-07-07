//! On-device GPU validation for the `norm` subsystem of `oxicuda-dnn`.
//!
//! Every kernel in this cluster is driven on the live CUDA device and its
//! output compared against an independent CPU re-derivation. Two launch
//! strategies are used:
//!
//! * **Public op API** for kernels with a clean public launcher
//!   (`layer_norm`, `rms_norm`, `fused_add_rms_norm`, `batch_norm_forward`,
//!   `group_norm`, `fused_layer_norm_relu`, `fused_rms_norm_silu`).
//! * **Direct PTX** via the published `*Plan` structs for the kernels that
//!   only expose PTX (`instance_norm` fwd/bwd, `power_norm`, `scale_norm`).
//!
//! ## Bugs fixed in the owned source during this pass
//!
//! 1. `.maxntid` placement (all 8 non-`layer_norm` files): the kernel
//!    directive was emitted *inside* the body (first statement after `{`, with
//!    a trailing `;`). `ptxas` rejects that with a parse error, so none of
//!    those kernels assembled. Moved it after the param list and before `{`
//!    with no semicolon (matching the working `layer_norm`).
//! 2. `batch_norm_forward` launched with `block_size = next_pow2(batch*spatial)`
//!    while the PTX bakes `next_pow2(spatial*32)` into the strided-loop stride
//!    and the reduction-tree width. Any `batch != 32` corrupted the reduction.
//!    The host now launches with the PTX-baked block size.
//! 3. `instance_norm` backward reduced `sum_dy` then `sum_dy_xhat` with the
//!    same helper, whose `%f15`/`%f16` scratch clobbered the still-needed
//!    `sum_dy_xhat` partial held in `%f16`. The partial is now stashed in
//!    `%f23` before the first reduction.
//!
//! Each test returns early (skips) when no CUDA device is present.

use super::{Lcg, assert_close_f32, gpu_fixture, load_kernel};

use oxicuda_launch::LaunchParams;
use oxicuda_memory::DeviceBuffer;

use crate::norm::instance_norm::{InstanceNormConfig, InstanceNormPlan};
use crate::norm::power_norm::{PowerNormConfig, PowerNormPlan};
use crate::norm::scale_norm::{ScaleNormConfig, ScaleNormPlan};
use crate::types::{TensorDesc, TensorDescMut};

// ---------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------

/// Uploads a host slice to a fresh device buffer.
fn dbuf(data: &[f32]) -> DeviceBuffer<f32> {
    DeviceBuffer::<f32>::from_host(data).expect("from_host")
}

/// Allocates a zeroed device buffer of `n` elements.
fn dzeros(n: usize) -> DeviceBuffer<f32> {
    DeviceBuffer::<f32>::from_host(&vec![0.0f32; n]).expect("from_host zeros")
}

/// Copies a device buffer back to a host vector.
fn to_host(buf: &DeviceBuffer<f32>, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; n];
    buf.copy_to_host(&mut host).expect("copy_to_host");
    host
}

/// Deterministic vector of `n` values in `[lo, hi)`.
fn rand_vec(rng: &mut Lcg, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..n).map(|_| rng.range_f32(lo, hi)).collect()
}

// ---------------------------------------------------------------------------
// CPU oracles (computed in f64 for stability; compared with float tolerances)
// ---------------------------------------------------------------------------

fn cpu_layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
    let inv_std = 1.0 / (var + eps as f64).sqrt();
    (0..d)
        .map(|i| ((x[i] as f64 - mean) * inv_std * gamma[i] as f64 + beta[i] as f64) as f32)
        .collect()
}

fn cpu_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean_sq = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
    let rms = (mean_sq + eps as f64).sqrt();
    (0..d)
        .map(|i| (x[i] as f64 / rms * gamma[i] as f64) as f32)
        .collect()
}

fn cpu_scale_norm(x: &[f32], g: f32, eps: f32) -> Vec<f32> {
    let sumsq = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    let inv = 1.0 / (sumsq + eps as f64).sqrt();
    x.iter()
        .map(|&v| (g as f64 * v as f64 * inv) as f32)
        .collect()
}

fn cpu_power_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32, power: f32) -> Vec<f32> {
    let d = x.len();
    let p = power as f64;
    let e = eps as f64;
    let is_p2 = (power - 2.0).abs() < 1e-6;
    let is_p1 = (power - 1.0).abs() < 1e-6;
    let acc: f64 = x
        .iter()
        .map(|&v| {
            let a = (v as f64).abs();
            if is_p2 {
                a * a
            } else if is_p1 {
                a
            } else {
                // mirrors the eps-guarded lg2/ex2 path: (|x| + eps)^p
                (a + e).powf(p)
            }
        })
        .sum();
    let mean = acc / d as f64;
    let pm = if is_p2 {
        mean.sqrt()
    } else if is_p1 {
        mean
    } else {
        (mean + e).powf(1.0 / p)
    };
    let inv = 1.0 / (pm + e);
    (0..d)
        .map(|i| (gamma[i] as f64 * x[i] as f64 * inv + beta[i] as f64) as f32)
        .collect()
}

fn cpu_fused_ln_relu(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    cpu_layer_norm(x, gamma, beta, eps)
        .into_iter()
        .map(|v| v.max(0.0))
        .collect()
}

fn cpu_fused_rms_silu(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean_sq = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();
    (0..d)
        .map(|i| {
            let xn = x[i] as f64 * inv_rms * gamma[i] as f64;
            let sig = 1.0 / (1.0 + (-xn).exp());
            (xn * sig) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// layer_norm (public API)
// ---------------------------------------------------------------------------

fn run_layer_norm(num_rows: u32, d: u32, seed: u64) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(seed);
    let n = (num_rows * d) as usize;
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, d as usize, -0.5, 0.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);
    let mut d_out = dzeros(n);

    let input = TensorDesc::<f32>::matrix(&d_in, num_rows, d).expect("input desc");
    let mut output = TensorDescMut::<f32>::matrix(&mut d_out, num_rows, d).expect("output desc");
    crate::norm::layer_norm::layer_norm(&fx.handle, &input, &d_gamma, &d_beta, &mut output, eps)
        .expect("layer_norm launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        let out = cpu_layer_norm(row, &gamma, &beta, eps);
        cpu[r * d as usize..(r + 1) * d as usize].copy_from_slice(&out);
    }
    assert_close_f32(&gpu, &cpu, 1e-3, 1e-4, "layer_norm");
}

#[test]
fn layer_norm_warp_d32() {
    run_layer_norm(8, 32, 0x1111);
}

#[test]
fn layer_norm_block_d256() {
    run_layer_norm(4, 256, 0x2222);
}

#[test]
fn layer_norm_block_d768() {
    run_layer_norm(3, 768, 0x3333);
}

// ---------------------------------------------------------------------------
// rms_norm + fused_add_rms_norm (public API)
// ---------------------------------------------------------------------------

fn run_rms_norm(num_rows: u32, d: u32, seed: u64) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(seed);
    let n = (num_rows * d) as usize;
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let mut d_out = dzeros(n);

    let input = TensorDesc::<f32>::matrix(&d_in, num_rows, d).expect("input desc");
    let mut output = TensorDescMut::<f32>::matrix(&mut d_out, num_rows, d).expect("output desc");
    crate::norm::rms_norm::rms_norm(&fx.handle, &input, &d_gamma, &mut output, eps)
        .expect("rms_norm launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        cpu[r * d as usize..(r + 1) * d as usize].copy_from_slice(&cpu_rms_norm(row, &gamma, eps));
    }
    assert_close_f32(&gpu, &cpu, 1e-3, 1e-4, "rms_norm");
}

#[test]
fn rms_norm_warp_d32() {
    run_rms_norm(8, 32, 0x4444);
}

#[test]
fn rms_norm_block_d256() {
    run_rms_norm(4, 256, 0x5555);
}

#[test]
fn fused_add_rms_norm_block_d128() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (num_rows, d) = (3u32, 128u32);
    let n = (num_rows * d) as usize;
    let mut rng = Lcg::new(0x6666);
    let x = rand_vec(&mut rng, n, -2.0, 2.0);
    let resid = rand_vec(&mut rng, n, -2.0, 2.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let mut d_resid = dbuf(&resid);
    let mut d_out = dzeros(n);

    let input = TensorDesc::<f32>::matrix(&d_in, num_rows, d).expect("input desc");
    let mut residual = TensorDescMut::<f32>::matrix(&mut d_resid, num_rows, d).expect("resid desc");
    let mut output = TensorDescMut::<f32>::matrix(&mut d_out, num_rows, d).expect("output desc");
    crate::norm::rms_norm::fused_add_rms_norm(
        &fx.handle,
        &input,
        &mut residual,
        &d_gamma,
        &mut output,
        eps,
    )
    .expect("fused_add_rms_norm launch");
    fx.stream().synchronize().expect("sync");

    let gpu_out = to_host(&d_out, n);
    let gpu_resid = to_host(&d_resid, n);

    let mut cpu_out = vec![0.0f32; n];
    let mut cpu_resid = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let lo = r * d as usize;
        let hi = lo + d as usize;
        let summed: Vec<f32> = (lo..hi).map(|i| x[i] + resid[i]).collect();
        cpu_resid[lo..hi].copy_from_slice(&summed);
        cpu_out[lo..hi].copy_from_slice(&cpu_rms_norm(&summed, &gamma, eps));
    }
    assert_close_f32(
        &gpu_resid,
        &cpu_resid,
        1e-4,
        1e-5,
        "fused_add_rms_norm residual",
    );
    assert_close_f32(&gpu_out, &cpu_out, 1e-3, 1e-4, "fused_add_rms_norm output");
}

// ---------------------------------------------------------------------------
// batch_norm (public API): training + inference
// ---------------------------------------------------------------------------

#[test]
fn batch_norm_training() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Large batch (N=64) deliberately diverges `batch*spatial` from the
    // PTX-baked `spatial*32`, so the previous host block-size formula would
    // drop reduction partials here. This is a regression guard for that fix.
    let (n, c, h, w) = (64u32, 5u32, 2u32, 3u32);
    let spatial = (h * w) as usize;
    let total = (n * c * h * w) as usize;
    let mut rng = Lcg::new(0x7777);
    let x = rand_vec(&mut rng, total, -2.0, 4.0);
    let gamma = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, c as usize, -0.5, 0.5);
    let rmean0 = rand_vec(&mut rng, c as usize, -0.2, 0.2);
    let rvar0 = rand_vec(&mut rng, c as usize, 0.8, 1.2);
    let eps = 1e-5f32;
    let mom = 0.1f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);
    let mut d_rmean = dbuf(&rmean0);
    let mut d_rvar = dbuf(&rvar0);
    let mut d_out = dzeros(total);
    let mut d_save_mean = dzeros(c as usize);
    let mut d_save_invvar = dzeros(c as usize);

    let input = TensorDesc::<f32>::nchw(&d_in, n, c, h, w).expect("input desc");
    let mut output = TensorDescMut::<f32>::nchw(&mut d_out, n, c, h, w).expect("output desc");
    crate::norm::batch_norm::batch_norm_forward(
        &fx.handle,
        &input,
        &d_gamma,
        &d_beta,
        &mut d_rmean,
        &mut d_rvar,
        &mut output,
        eps,
        mom,
        true,
        Some(&mut d_save_mean),
        Some(&mut d_save_invvar),
    )
    .expect("batch_norm training launch");
    fx.stream().synchronize().expect("sync");

    let gpu_out = to_host(&d_out, total);
    let gpu_save_mean = to_host(&d_save_mean, c as usize);
    let gpu_save_invvar = to_host(&d_save_invvar, c as usize);
    let gpu_rmean = to_host(&d_rmean, c as usize);
    let gpu_rvar = to_host(&d_rvar, c as usize);

    // CPU oracle: per-channel stats over N*spatial.
    let idx = |ni: usize, ci: usize, hw: usize| ((ni * c as usize + ci) * spatial) + hw;
    let mut cpu_out = vec![0.0f32; total];
    let mut cpu_mean = vec![0.0f32; c as usize];
    let mut cpu_invvar = vec![0.0f32; c as usize];
    let mut cpu_rmean = vec![0.0f32; c as usize];
    let mut cpu_rvar = vec![0.0f32; c as usize];
    let count = (n as usize) * spatial;
    for ci in 0..c as usize {
        let mut sum = 0.0f64;
        for ni in 0..n as usize {
            for hw in 0..spatial {
                sum += x[idx(ni, ci, hw)] as f64;
            }
        }
        let mean = sum / count as f64;
        let mut var = 0.0f64;
        for ni in 0..n as usize {
            for hw in 0..spatial {
                var += (x[idx(ni, ci, hw)] as f64 - mean).powi(2);
            }
        }
        var /= count as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        cpu_mean[ci] = mean as f32;
        cpu_invvar[ci] = inv_std as f32;
        cpu_rmean[ci] = ((1.0 - mom as f64) * rmean0[ci] as f64 + mom as f64 * mean) as f32;
        cpu_rvar[ci] = ((1.0 - mom as f64) * rvar0[ci] as f64 + mom as f64 * var) as f32;
        for ni in 0..n as usize {
            for hw in 0..spatial {
                let i = idx(ni, ci, hw);
                cpu_out[i] =
                    ((x[i] as f64 - mean) * inv_std * gamma[ci] as f64 + beta[ci] as f64) as f32;
            }
        }
    }
    assert_close_f32(
        &gpu_save_mean,
        &cpu_mean,
        1e-3,
        1e-4,
        "batch_norm save_mean",
    );
    assert_close_f32(
        &gpu_save_invvar,
        &cpu_invvar,
        1e-3,
        1e-4,
        "batch_norm save_invvar",
    );
    assert_close_f32(
        &gpu_rmean,
        &cpu_rmean,
        1e-3,
        1e-4,
        "batch_norm running_mean",
    );
    assert_close_f32(&gpu_rvar, &cpu_rvar, 1e-3, 1e-4, "batch_norm running_var");
    assert_close_f32(&gpu_out, &cpu_out, 1e-3, 1e-4, "batch_norm training output");
}

#[test]
fn batch_norm_inference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, h, w) = (4u32, 6u32, 2u32, 2u32);
    let spatial = (h * w) as usize;
    let total = (n * c * h * w) as usize;
    let mut rng = Lcg::new(0x8888);
    let x = rand_vec(&mut rng, total, -2.0, 2.0);
    let gamma = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, c as usize, -0.5, 0.5);
    let rmean = rand_vec(&mut rng, c as usize, -1.0, 1.0);
    let rvar = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);
    let mut d_rmean = dbuf(&rmean);
    let mut d_rvar = dbuf(&rvar);
    let mut d_out = dzeros(total);

    let input = TensorDesc::<f32>::nchw(&d_in, n, c, h, w).expect("input desc");
    let mut output = TensorDescMut::<f32>::nchw(&mut d_out, n, c, h, w).expect("output desc");
    crate::norm::batch_norm::batch_norm_forward(
        &fx.handle,
        &input,
        &d_gamma,
        &d_beta,
        &mut d_rmean,
        &mut d_rvar,
        &mut output,
        eps,
        0.1,
        false,
        None,
        None,
    )
    .expect("batch_norm inference launch");
    fx.stream().synchronize().expect("sync");

    let gpu_out = to_host(&d_out, total);
    let idx = |ni: usize, ci: usize, hw: usize| ((ni * c as usize + ci) * spatial) + hw;
    let mut cpu_out = vec![0.0f32; total];
    for ci in 0..c as usize {
        let inv_std = 1.0 / (rvar[ci] as f64 + eps as f64).sqrt();
        for ni in 0..n as usize {
            for hw in 0..spatial {
                let i = idx(ni, ci, hw);
                cpu_out[i] = ((x[i] as f64 - rmean[ci] as f64) * inv_std * gamma[ci] as f64
                    + beta[ci] as f64) as f32;
            }
        }
    }
    assert_close_f32(
        &gpu_out,
        &cpu_out,
        1e-3,
        1e-4,
        "batch_norm inference output",
    );
}

// ---------------------------------------------------------------------------
// group_norm (public API)
// ---------------------------------------------------------------------------

#[test]
fn group_norm_basic() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, h, w) = (2u32, 8u32, 3u32, 3u32);
    let num_groups = 2u32;
    let cpg = (c / num_groups) as usize;
    let spatial = (h * w) as usize;
    let total = (n * c * h * w) as usize;
    let mut rng = Lcg::new(0x9999);
    let x = rand_vec(&mut rng, total, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, c as usize, -0.5, 0.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);
    let mut d_out = dzeros(total);

    let input = TensorDesc::<f32>::nchw(&d_in, n, c, h, w).expect("input desc");
    let mut output = TensorDescMut::<f32>::nchw(&mut d_out, n, c, h, w).expect("output desc");
    crate::norm::group_norm::group_norm(
        &fx.handle,
        &input,
        num_groups,
        &d_gamma,
        &d_beta,
        &mut output,
        eps,
    )
    .expect("group_norm launch");
    fx.stream().synchronize().expect("sync");

    let gpu_out = to_host(&d_out, total);
    let idx = |ni: usize, ci: usize, hw: usize| ((ni * c as usize + ci) * spatial) + hw;
    let mut cpu_out = vec![0.0f32; total];
    let group_size = cpg * spatial;
    for ni in 0..n as usize {
        for g in 0..num_groups as usize {
            let mut sum = 0.0f64;
            for cc in 0..cpg {
                let ci = g * cpg + cc;
                for hw in 0..spatial {
                    sum += x[idx(ni, ci, hw)] as f64;
                }
            }
            let mean = sum / group_size as f64;
            let mut var = 0.0f64;
            for cc in 0..cpg {
                let ci = g * cpg + cc;
                for hw in 0..spatial {
                    var += (x[idx(ni, ci, hw)] as f64 - mean).powi(2);
                }
            }
            var /= group_size as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for cc in 0..cpg {
                let ci = g * cpg + cc;
                for hw in 0..spatial {
                    let i = idx(ni, ci, hw);
                    cpu_out[i] = ((x[i] as f64 - mean) * inv_std * gamma[ci] as f64
                        + beta[ci] as f64) as f32;
                }
            }
        }
    }
    assert_close_f32(&gpu_out, &cpu_out, 1e-3, 1e-4, "group_norm output");
}

// ---------------------------------------------------------------------------
// fused_norm (public API)
// ---------------------------------------------------------------------------

#[test]
fn fused_layer_norm_relu_block() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (num_rows, d) = (4u32, 128u32);
    let n = (num_rows * d) as usize;
    let mut rng = Lcg::new(0xABCD);
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, d as usize, -1.0, 1.0);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);
    let mut d_out = dzeros(n);

    let input = TensorDesc::<f32>::matrix(&d_in, num_rows, d).expect("input desc");
    let mut output = TensorDescMut::<f32>::matrix(&mut d_out, num_rows, d).expect("output desc");
    crate::norm::fused_norm::fused_layer_norm_relu(
        &fx.handle,
        &input,
        &d_gamma,
        &d_beta,
        &mut output,
        eps,
    )
    .expect("fused_layer_norm_relu launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        cpu[r * d as usize..(r + 1) * d as usize]
            .copy_from_slice(&cpu_fused_ln_relu(row, &gamma, &beta, eps));
    }
    assert_close_f32(&gpu, &cpu, 1e-3, 1e-4, "fused_layer_norm_relu");
}

#[test]
fn fused_rms_norm_silu_block() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (num_rows, d) = (4u32, 128u32);
    let n = (num_rows * d) as usize;
    let mut rng = Lcg::new(0xBEEF);
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let eps = 1e-5f32;

    let d_in = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let mut d_out = dzeros(n);

    let input = TensorDesc::<f32>::matrix(&d_in, num_rows, d).expect("input desc");
    let mut output = TensorDescMut::<f32>::matrix(&mut d_out, num_rows, d).expect("output desc");
    crate::norm::fused_norm::fused_rms_norm_silu(&fx.handle, &input, &d_gamma, &mut output, eps)
        .expect("fused_rms_norm_silu launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        cpu[r * d as usize..(r + 1) * d as usize]
            .copy_from_slice(&cpu_fused_rms_silu(row, &gamma, eps));
    }
    // SiLU uses ex2.approx for the sigmoid exponent: a looser tolerance.
    assert_close_f32(&gpu, &cpu, 3e-3, 2e-4, "fused_rms_norm_silu");
}

// ---------------------------------------------------------------------------
// instance_norm forward (direct PTX via InstanceNormPlan)
// ---------------------------------------------------------------------------

#[test]
fn instance_norm_forward_affine() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, spatial) = (2u32, 3u32, 64u32);
    let total = (n * c * spatial) as usize;
    let mut rng = Lcg::new(0xC0DE);
    let x = rand_vec(&mut rng, total, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, c as usize, -0.5, 0.5);
    let eps = 1e-5f32;

    let config = InstanceNormConfig {
        num_channels: c,
        spatial_size: spatial,
        epsilon: eps,
        affine: true,
        track_running_stats: false,
    };
    let plan = InstanceNormPlan::new::<f32>(config, fx.sm).expect("instance plan");
    let entry = format!("instance_norm_fwd_f32_s{spatial}");
    let kernel = load_kernel(plan.forward_ptx(), &entry);

    let d_in = dbuf(&x);
    let d_out = dzeros(total);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);

    let block = spatial.next_power_of_two().clamp(32, 1024);
    let params = LaunchParams::new(n * c, block);
    let args = (
        d_in.as_device_ptr(),
        d_out.as_device_ptr(),
        d_gamma.as_device_ptr(),
        d_beta.as_device_ptr(),
        n,
        c,
        spatial,
        eps.to_bits(),
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("instance fwd launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, total);
    let sp = spatial as usize;
    let mut cpu = vec![0.0f32; total];
    for ni in 0..n as usize {
        for ci in 0..c as usize {
            let base = (ni * c as usize + ci) * sp;
            let slice = &x[base..base + sp];
            let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / sp as f64;
            let var = slice
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / sp as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for (k, &xv) in slice.iter().enumerate() {
                cpu[base + k] =
                    ((xv as f64 - mean) * inv_std * gamma[ci] as f64 + beta[ci] as f64) as f32;
            }
        }
    }
    assert_close_f32(&gpu, &cpu, 1e-3, 1e-4, "instance_norm forward");
}

// ---------------------------------------------------------------------------
// instance_norm backward (direct PTX via InstanceNormPlan)
// ---------------------------------------------------------------------------

#[test]
fn instance_norm_backward_dx() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, spatial) = (2u32, 3u32, 64u32);
    let total = (n * c * spatial) as usize;
    let mut rng = Lcg::new(0xD00D);
    let x = rand_vec(&mut rng, total, -3.0, 3.0);
    let dy = rand_vec(&mut rng, total, -1.5, 1.5);
    let gamma = rand_vec(&mut rng, c as usize, 0.5, 1.5);
    let eps = 1e-5f32;

    let config = InstanceNormConfig {
        num_channels: c,
        spatial_size: spatial,
        epsilon: eps,
        affine: true,
        track_running_stats: false,
    };
    let plan = InstanceNormPlan::new::<f32>(config, fx.sm).expect("instance plan");
    let entry = format!("instance_norm_bwd_f32_s{spatial}");
    let kernel = load_kernel(plan.backward_ptx(), &entry);

    let d_dy = dbuf(&dy);
    let d_x = dbuf(&x);
    let d_gamma = dbuf(&gamma);
    let d_dx = dzeros(total);

    let block = spatial.next_power_of_two().clamp(32, 1024);
    let params = LaunchParams::new(n * c, block);
    // ABI: grad_output, input, gamma, grad_input, batch, channels, spatial, epsilon_bits.
    let args = (
        d_dy.as_device_ptr(),
        d_x.as_device_ptr(),
        d_gamma.as_device_ptr(),
        d_dx.as_device_ptr(),
        n,
        c,
        spatial,
        eps.to_bits(),
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("instance bwd launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_dx, total);
    let sp = spatial as usize;
    let mut cpu = vec![0.0f32; total];
    for ni in 0..n as usize {
        for ci in 0..c as usize {
            let base = (ni * c as usize + ci) * sp;
            let xs = &x[base..base + sp];
            let dys = &dy[base..base + sp];
            let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / sp as f64;
            let var = xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / sp as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            let xhat: Vec<f64> = xs.iter().map(|&v| (v as f64 - mean) * inv_std).collect();
            let sum_dy: f64 = dys.iter().map(|&v| v as f64).sum();
            let sum_dy_xhat: f64 = dys
                .iter()
                .zip(xhat.iter())
                .map(|(&g, &xh)| g as f64 * xh)
                .sum();
            for k in 0..sp {
                let dxk = gamma[ci] as f64
                    * inv_std
                    * (dys[k] as f64 - (sum_dy + xhat[k] * sum_dy_xhat) / sp as f64);
                cpu[base + k] = dxk as f32;
            }
        }
    }
    assert_close_f32(&gpu, &cpu, 2e-3, 1e-4, "instance_norm backward dx");
}

// ---------------------------------------------------------------------------
// power_norm forward (direct PTX via PowerNormPlan)
// ---------------------------------------------------------------------------

fn run_power_norm(num_rows: u32, d: u32, power: f32, rel: f32, abs: f32, seed: u64) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = (num_rows * d) as usize;
    let mut rng = Lcg::new(seed);
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let gamma = rand_vec(&mut rng, d as usize, 0.5, 1.5);
    let beta = rand_vec(&mut rng, d as usize, -0.5, 0.5);
    let eps = 1e-5f32;

    let config = PowerNormConfig {
        hidden_size: d,
        epsilon: eps,
        power,
    };
    let plan = PowerNormPlan::new::<f32>(config, fx.sm).expect("power plan");
    let entry = format!("power_norm_fwd_f32_d{d}");
    let kernel = load_kernel(plan.forward_ptx(), &entry);

    let d_in = dbuf(&x);
    let d_out = dzeros(n);
    let d_gamma = dbuf(&gamma);
    let d_beta = dbuf(&beta);

    let block = if d <= 1024 {
        d.next_power_of_two().min(1024)
    } else {
        1024
    };
    let params = LaunchParams::new(num_rows, block);
    let args = (
        d_in.as_device_ptr(),
        d_out.as_device_ptr(),
        d_gamma.as_device_ptr(),
        d_beta.as_device_ptr(),
        num_rows,
        d,
        eps.to_bits(),
        power.to_bits(),
        (1.0f32 / power).to_bits(),
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("power launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        cpu[r * d as usize..(r + 1) * d as usize]
            .copy_from_slice(&cpu_power_norm(row, &gamma, &beta, eps, power));
    }
    assert_close_f32(&gpu, &cpu, rel, abs, "power_norm");
}

#[test]
fn power_norm_p2() {
    run_power_norm(3, 128, 2.0, 1e-3, 1e-4, 0xE001);
}

#[test]
fn power_norm_p1() {
    run_power_norm(3, 128, 1.0, 1e-3, 1e-4, 0xE002);
}

#[test]
fn power_norm_p1_5() {
    // General power path uses lg2.approx/ex2.approx -> looser tolerance.
    run_power_norm(3, 64, 1.5, 3e-2, 2e-3, 0xE003);
}

// ---------------------------------------------------------------------------
// scale_norm forward (direct PTX via ScaleNormPlan)
// ---------------------------------------------------------------------------

fn run_scale_norm(num_rows: u32, d: u32, seed: u64) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = (num_rows * d) as usize;
    let mut rng = Lcg::new(seed);
    let x = rand_vec(&mut rng, n, -3.0, 3.0);
    let g = rng.range_f32(0.5, 2.0);
    let eps = 1e-6f32;

    let config = ScaleNormConfig {
        hidden_size: d,
        epsilon: eps,
    };
    let plan = ScaleNormPlan::new::<f32>(config, fx.sm).expect("scale plan");
    let entry = format!("scale_norm_fwd_f32_d{d}");
    let kernel = load_kernel(plan.forward_ptx(), &entry);

    let d_in = dbuf(&x);
    let d_out = dzeros(n);
    let d_g = dbuf(&[g]);

    let block = if d <= 1024 {
        d.next_power_of_two().min(1024)
    } else {
        1024
    };
    let params = LaunchParams::new(num_rows, block);
    let args = (
        d_in.as_device_ptr(),
        d_out.as_device_ptr(),
        d_g.as_device_ptr(),
        num_rows,
        d,
        eps.to_bits(),
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("scale launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host(&d_out, n);
    let mut cpu = vec![0.0f32; n];
    for r in 0..num_rows as usize {
        let row = &x[r * d as usize..(r + 1) * d as usize];
        cpu[r * d as usize..(r + 1) * d as usize].copy_from_slice(&cpu_scale_norm(row, g, eps));
    }
    assert_close_f32(&gpu, &cpu, 1e-3, 1e-4, "scale_norm");
}

#[test]
fn scale_norm_warp_d32() {
    run_scale_norm(4, 32, 0xF001);
}

#[test]
fn scale_norm_block_d128() {
    run_scale_norm(3, 128, 0xF002);
}

#[test]
fn scale_norm_block_d512() {
    run_scale_norm(2, 512, 0xF003);
}
