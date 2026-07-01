//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies results back,
//! and asserts numerical equivalence to an independent CPU re-derivation of the
//! kernel's documented arithmetic. The launch ABI follows the same convention as
//! `oxicuda-snn` / `oxicuda-sparse`: device buffers are passed as their
//! `CUdeviceptr` (`.param .u64`), scalars as the matching Rust scalar type, in
//! declared parameter order.
//!
//! ## Oracle tiers (honest accounting)
//!
//! All seven kernels in this crate are **fully implemented** (every loop body
//! contains real `ld.global` / `st.global` / arithmetic — none is a hollow
//! stub), so every test is a genuine **CPU-vs-GPU numerical-equivalence** check
//! against an independent host re-derivation of the kernel's documented layout
//! and math:
//!
//! * `moving_average`  — symmetric clamped moving average over `[N, T]`.
//! * `patch_embed_1d`  — overlapping 1-D patch gather `[N, T] -> [N, P, L]`.
//! * `causal_temporal_conv` — dilated causal 1-D conv `[N, C_in, T] -> [N, C_out, T]`.
//! * `auto_correlation` — complex magnitude squared `re^2 + im^2`.
//! * `revin_normalize` — per-(n,c) reversible instance normalisation.
//! * `multirate_pool`  — average pool at a variable stride.
//! * `period_detect`   — mean magnitude across the `(N, C)` batch.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let Ok(dev) = Device::get(0) else {
        return None;
    };
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = (major * 10 + minor) as u32;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// JIT-compile `ptx` for the live device and look up `entry`.
///
/// A failure here means ptxas rejected the PTX (a real, structural bug) — the
/// test panics rather than silently skipping.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a 1-D grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

/// Worst absolute / relative divergence over two equal-length slices.
fn worst_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        let a = (g - c).abs();
        if a > worst_abs {
            worst_abs = a;
        }
        let denom = g.abs().max(c.abs());
        if denom > 0.0 {
            let r = a / denom;
            if r > worst_rel {
                worst_rel = r;
            }
        }
    }
    (worst_rel, worst_abs)
}

/// Assert element-wise closeness with a relative-plus-absolute tolerance.
fn assert_close(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length mismatch");
    let (wr, wa) = worst_diff(gpu, cpu);
    for k in 0..gpu.len() {
        let tol = rel * gpu[k].abs().max(cpu[k].abs()) + abs;
        assert!(
            (gpu[k] - cpu[k]).abs() <= tol,
            "{what}: mismatch at {k}: gpu={} cpu={} (worst rel={wr:e} abs={wa:e})",
            gpu[k],
            cpu[k]
        );
    }
}

/// Deterministic, reproducible pseudo-random f32 in `[-1, 1)` from an index.
///
/// A self-contained splitmix-style hash (independent of the crate's `LcgRng`) so
/// the inputs are fixed and the oracle is unambiguous.
fn det_f32(seed: u64, idx: u64) -> f32 {
    let mut z = seed
        .wrapping_add(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 24-bit mantissa -> [0, 1) -> [-1, 1)
    let u = ((z >> 8) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0_f32;
    2.0_f32 * u - 1.0_f32
}

/// Like [`det_f32`] but in `[0, 1)` (for non-negative magnitudes).
fn det_unit(seed: u64, idx: u64) -> f32 {
    0.5_f32 * (det_f32(seed, idx) + 1.0_f32)
}

// ===========================================================================
// 1. moving_average — CPU-vs-GPU equivalence (symmetric clamped moving average)
// ===========================================================================
//
// Kernel: out[n, t] = (1/K) * Σ_{k=0}^{K-1} in[n, clamp(t - K/2 + k, 0, T-1)]
// over a [N, T] tensor; one thread per flat index n*T + t.

#[test]
fn moving_average_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rows = 5_usize;
    let t_len = 17_usize;
    let k = 5_u32;
    let half = (k / 2) as i32;

    let input: Vec<f32> = (0..n_rows * t_len)
        .map(|i| det_f32(0xA1, i as u64))
        .collect();

    // Host re-derivation in the kernel's [N, T] layout.
    let mut cpu = vec![0.0_f32; n_rows * t_len];
    let inv_k = 1.0_f32 / k as f32;
    for n in 0..n_rows {
        for t in 0..t_len {
            let mut acc = 0.0_f32;
            for kk in 0..k as i32 {
                let src = (t as i32 + kk - half).clamp(0, t_len as i32 - 1) as usize;
                acc += input[n * t_len + src];
            }
            cpu[n * t_len + t] = acc * inv_k;
        }
    }

    let ptx = crate::ptx_kernels::moving_average_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "moving_average");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rows * t_len]).expect("d_out");

    let total = (n_rows * t_len) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n_rows as u32,
                t_len as u32,
                k,
            ),
        )
        .expect("launch moving_average");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_rows * t_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    assert_close(&gpu, &cpu, 1e-4, 1e-5, "moving_average");
}

// ===========================================================================
// 2. patch_embed_1d — CPU-vs-GPU equivalence (overlapping patch gather)
// ===========================================================================
//
// Kernel: out[n, p, l] = in[n, p*stride + l]  over [N, T] -> [N, P, L].
// One thread per output element.

#[test]
fn patch_embed_1d_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rows = 4_usize;
    let t_len = 20_usize;
    let patch_len = 6_usize;
    let stride = 3_usize;
    let num_patches = (t_len - patch_len) / stride + 1; // = 5

    let input: Vec<f32> = (0..n_rows * t_len)
        .map(|i| det_f32(0xB2, i as u64))
        .collect();

    // Host re-derivation; every (p, l) here has t_src < T (full patches).
    let mut cpu = vec![0.0_f32; n_rows * num_patches * patch_len];
    for n in 0..n_rows {
        for p in 0..num_patches {
            for l in 0..patch_len {
                let t_src = p * stride + l;
                cpu[(n * num_patches + p) * patch_len + l] = input[n * t_len + t_src];
            }
        }
    }

    let ptx = crate::ptx_kernels::patch_embed_1d_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "patch_embed_1d");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let out_len = n_rows * num_patches * patch_len;
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(out_len as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n_rows as u32,
                t_len as u32,
                patch_len as u32,
                stride as u32,
                num_patches as u32,
            ),
        )
        .expect("launch patch_embed_1d");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // Exact gather: must be bit-identical.
    for k in 0..out_len {
        assert_eq!(
            gpu[k].to_bits(),
            cpu[k].to_bits(),
            "patch_embed_1d: mismatch at {k}: gpu={} cpu={}",
            gpu[k],
            cpu[k]
        );
    }
}

// ===========================================================================
// 3. causal_temporal_conv — CPU-vs-GPU equivalence (dilated causal 1-D conv)
// ===========================================================================
//
// Kernel: y[n,c_out,t] = b[c_out] + Σ_{c_in,k} w[c_out,c_in,k] * x[n,c_in,t-k*d]
// with causal skip when t - k*d < 0.  Shapes [N, C_in, T] -> [N, C_out, T].

#[test]
fn causal_temporal_conv_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_b = 2_usize;
    let c_in = 3_usize;
    let c_out = 4_usize;
    let t_len = 11_usize;
    let k = 3_usize;
    let dilation = 2_usize;

    let input: Vec<f32> = (0..n_b * c_in * t_len)
        .map(|i| det_f32(0xC3, i as u64))
        .collect();
    let weights: Vec<f32> = (0..c_out * c_in * k)
        .map(|i| det_f32(0xC4, i as u64))
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|i| det_f32(0xC5, i as u64)).collect();

    // Host re-derivation.
    let mut cpu = vec![0.0_f32; n_b * c_out * t_len];
    for n in 0..n_b {
        for co in 0..c_out {
            for t in 0..t_len {
                let mut acc = bias[co];
                for ci in 0..c_in {
                    for kk in 0..k {
                        let ts = t as i64 - (kk * dilation) as i64;
                        if ts < 0 {
                            continue;
                        }
                        let x = input[(n * c_in + ci) * t_len + ts as usize];
                        let w = weights[(co * c_in + ci) * k + kk];
                        acc += x * w;
                    }
                }
                cpu[(n * c_out + co) * t_len + t] = acc;
            }
        }
    }

    let ptx = crate::ptx_kernels::causal_temporal_conv_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "causal_temporal_conv");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let out_len = n_b * c_out * t_len;
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_b = DeviceBuffer::<f32>::from_host(&bias).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(out_len as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_w.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n_b as u32,
                c_in as u32,
                c_out as u32,
                t_len as u32,
                k as u32,
                dilation as u32,
            ),
        )
        .expect("launch causal_temporal_conv");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    assert_close(&gpu, &cpu, 1e-4, 1e-5, "causal_temporal_conv");
}

// ===========================================================================
// 4. auto_correlation — CPU-vs-GPU equivalence (|FFT|^2 magnitude step)
// ===========================================================================
//
// Kernel: out[i] = re[i]^2 + im[i]^2.

#[test]
fn auto_correlation_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let total = 257_usize;
    let re: Vec<f32> = (0..total).map(|i| det_f32(0xD6, i as u64) * 4.0).collect();
    let im: Vec<f32> = (0..total).map(|i| det_f32(0xD7, i as u64) * 4.0).collect();

    let cpu: Vec<f32> = re.iter().zip(&im).map(|(&r, &i)| r * r + i * i).collect();

    let ptx = crate::ptx_kernels::auto_correlation_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "auto_correlation");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im).expect("d_im");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                d_out.as_device_ptr(),
                total as u32,
            ),
        )
        .expect("launch auto_correlation");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    assert_close(&gpu, &cpu, 1e-5, 1e-5, "auto_correlation");
}

// ===========================================================================
// 5. revin_normalize — CPU-vs-GPU equivalence (reversible instance norm)
// ===========================================================================
//
// Kernel: y[n,c,t] = (x[n,c,t] - mean[n,c]) / (std[n,c] + eps) * gamma[c] + beta[c]
// over [N, C, T]; mean/std length N*C, gamma/beta length C; eps = 1e-5.

#[test]
fn revin_normalize_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_b = 3_usize;
    let c = 4_usize;
    let t_len = 9_usize;
    let eps = 1e-5_f32;

    let x: Vec<f32> = (0..n_b * c * t_len)
        .map(|i| det_f32(0xE8, i as u64) * 3.0)
        .collect();
    let mean: Vec<f32> = (0..n_b * c).map(|i| det_f32(0xE9, i as u64)).collect();
    // Strictly positive std so (std + eps) is well away from zero.
    let std: Vec<f32> = (0..n_b * c)
        .map(|i| 0.5_f32 + det_unit(0xEA, i as u64))
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 0.5_f32 + det_unit(0xEB, i as u64)).collect();
    let beta: Vec<f32> = (0..c).map(|i| det_f32(0xEC, i as u64)).collect();

    let mut cpu = vec![0.0_f32; n_b * c * t_len];
    for n in 0..n_b {
        for ci in 0..c {
            let nc = n * c + ci;
            for t in 0..t_len {
                let idx = (n * c + ci) * t_len + t;
                let norm = (x[idx] - mean[nc]) / (std[nc] + eps);
                cpu[idx] = norm * gamma[ci] + beta[ci];
            }
        }
    }

    let ptx = crate::ptx_kernels::revin_normalize_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "revin_normalize");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let len = n_b * c * t_len;
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mean = DeviceBuffer::<f32>::from_host(&mean).expect("d_mean");
    let d_std = DeviceBuffer::<f32>::from_host(&std).expect("d_std");
    let d_gamma = DeviceBuffer::<f32>::from_host(&gamma).expect("d_gamma");
    let d_beta = DeviceBuffer::<f32>::from_host(&beta).expect("d_beta");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; len]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(len as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_std.as_device_ptr(),
                d_gamma.as_device_ptr(),
                d_beta.as_device_ptr(),
                d_out.as_device_ptr(),
                n_b as u32,
                c as u32,
                t_len as u32,
            ),
        )
        .expect("launch revin_normalize");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // div.approx.f32 carries ~2^-23 relative error; 5e-4 is generous yet still
    // catches any gross formula error.
    assert_close(&gpu, &cpu, 5e-4, 1e-5, "revin_normalize");
}

// ===========================================================================
// 6. multirate_pool — CPU-vs-GPU equivalence (average pool at variable stride)
// ===========================================================================
//
// Kernel: out[n,c,to] = (1/stride) * Σ_{k=0}^{stride-1} in[n,c, to*stride + k]
// over [N, C, T] -> [N, C, T_out], T_out = T / stride.

#[test]
fn multirate_pool_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_b = 2_usize;
    let c = 3_usize;
    let t_len = 16_usize;
    let stride = 4_usize;
    let t_out = t_len / stride; // = 4

    let input: Vec<f32> = (0..n_b * c * t_len)
        .map(|i| det_f32(0xF1, i as u64))
        .collect();

    let inv_ps = 1.0_f32 / stride as f32;
    let mut cpu = vec![0.0_f32; n_b * c * t_out];
    for n in 0..n_b {
        for ci in 0..c {
            for to in 0..t_out {
                let mut acc = 0.0_f32;
                for kk in 0..stride {
                    let ts = to * stride + kk;
                    acc += input[(n * c + ci) * t_len + ts];
                }
                cpu[(n * c + ci) * t_out + to] = acc * inv_ps;
            }
        }
    }

    let ptx = crate::ptx_kernels::multirate_pool_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "multirate_pool");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let out_len = n_b * c * t_out;
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(out_len as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n_b as u32,
                c as u32,
                t_len as u32,
                stride as u32,
            ),
        )
        .expect("launch multirate_pool");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    assert_close(&gpu, &cpu, 1e-4, 1e-5, "multirate_pool");
}

// ===========================================================================
// 7. period_detect — CPU-vs-GPU equivalence (mean magnitude across batch)
// ===========================================================================
//
// Kernel: out[f] = (1/NC) * Σ_{nc=0}^{NC-1} mag[nc*F + f]  over [NC, F] -> [F].

#[test]
fn period_detect_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let nc = 6_usize;
    let f = 13_usize;

    // Non-negative magnitudes.
    let mag: Vec<f32> = (0..nc * f)
        .map(|i| det_unit(0x1A, i as u64) * 5.0)
        .collect();

    let inv_nc = 1.0_f32 / nc as f32;
    let mut cpu = vec![0.0_f32; f];
    for ff in 0..f {
        let mut acc = 0.0_f32;
        for n in 0..nc {
            acc += mag[n * f + ff];
        }
        cpu[ff] = acc * inv_nc;
    }

    let ptx = crate::ptx_kernels::period_detect_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "period_detect");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mag = DeviceBuffer::<f32>::from_host(&mag).expect("d_mag");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; f]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(f as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mag.as_device_ptr(),
                d_out.as_device_ptr(),
                nc as u32,
                f as u32,
            ),
        )
        .expect("launch period_detect");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; f];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    assert_close(&gpu, &cpu, 1e-4, 1e-5, "period_detect");
}
