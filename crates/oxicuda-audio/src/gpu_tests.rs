//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-snn` / `oxicuda-ot`
//! canaries: device buffers are passed as their `CUdeviceptr` (a `.param .u64`),
//! scalars are passed as the matching Rust scalar (`.param .u32` / `.param .f32`)
//! in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared bit-for-bit / within FP32 tol to a
//!   `pub` CPU function the kernel mirrors:
//!   `ctc_alpha_kernel` terminal ↔ [`crate::ctc::ctc_forward_log`],
//!   `rel_pos_bias_kernel` ↔ [`crate::attention::RelPosEncoding::bias_matrix`],
//!   `stats_pool_kernel` mean ↔ the first `C` outputs of
//!   [`crate::speaker::stats_pool`].
//! * **Independent host re-derivation** — the op has no standalone `pub fn` (it
//!   is fused into a larger routine, or the kernel uses a *documented* variant
//!   that intentionally diverges from the crate convention), so the oracle is an
//!   independent Rust re-implementation of the kernel's documented arithmetic:
//!   `stride_conv1d_kernel` (matches the private `stride_conv1d` inside
//!   `Wav2VecCnnEncoder`), `depthwise_conv1d_kernel` (matches the private
//!   `depthwise_causal_conv1d` inside `ConvModule`), `dilated_conv1d_kernel`
//!   (depthwise causal dilated conv), `spec_augment_mask_kernel` (deterministic
//!   union of one time- and one freq-band), the full CTC alpha matrix, and the
//!   `stats_pool_kernel` std (population variance + `1e-8`, which diverges from
//!   the crate's Bessel-corrected `/(T-1)` + `1e-10` clamp — the mean still
//!   matches the crate exactly). These still genuinely fail if ptxas miscompiles
//!   or the PTX has a wrong constant / shift / index / base, because the host
//!   code is independent of the JIT-compiled PTX.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

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

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length slices,
/// ignoring positions where both values are `-inf`.
fn worst_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        if g == f32::NEG_INFINITY && c == f32::NEG_INFINITY {
            continue;
        }
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

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A failure here (ptxas rejecting the PTX, or the entry being absent) is a real
/// kernel bug for this SM and is surfaced as a panic with the driver message.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

/// Natural, max-stabilised log-sum-exp of two log-domain values, with the same
/// `-inf` semantics as [`crate::ctc::ctc_forward_log`]'s `log_sum_exp2`.
fn host_lse2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (1.0_f32 + (a.min(b) - m).exp()).ln()
}

// ===========================================================================
// 1. stride_conv1d  —  INDEPENDENT HOST RE-DERIVATION
//    (matches the private `stride_conv1d` inside `Wav2VecCnnEncoder`)
// ===========================================================================

#[test]
fn stride_conv1d_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let in_chans = 2_usize;
    let in_len = 16_usize;
    let out_chans = 3_usize;
    let kernel_size = 4_usize;
    let stride = 2_usize;
    let out_len = (in_len - kernel_size) / stride + 1; // 7

    let mut rng = LcgRng::new(0x57C0_11D5);
    let input: Vec<f32> = (0..in_chans * in_len)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let weights: Vec<f32> = (0..out_chans * in_chans * kernel_size)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let bias: Vec<f32> = (0..out_chans).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host re-derivation: acc = bias[oc]; ic outer, k inner (kernel's order).
    let mut out_host = vec![0.0_f32; out_chans * out_len];
    for oc in 0..out_chans {
        for pos in 0..out_len {
            let t_start = pos * stride;
            let mut acc = bias[oc];
            for ic in 0..in_chans {
                let w_off = (oc * in_chans + ic) * kernel_size;
                let x_off = ic * in_len + t_start;
                for k in 0..kernel_size {
                    acc += weights[w_off + k] * input[x_off + k];
                }
            }
            out_host[oc * out_len + pos] = acc;
        }
    }

    let ptx = crate::ptx_kernels::stride_conv1d_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "stride_conv1d_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_b = DeviceBuffer::<f32>::from_host(&bias).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_chans * out_len]).expect("d_out");

    let total = (out_chans * out_len) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_w.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                in_chans as u32,
                in_len as u32,
                out_chans as u32,
                kernel_size as u32,
                stride as u32,
                out_len as u32,
            ),
        )
        .expect("launch stride_conv1d_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; out_chans * out_len];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "stride_conv1d out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 2. dilated_conv1d  —  INDEPENDENT HOST RE-DERIVATION
//    (depthwise causal dilated conv with separate filter + gate outputs)
// ===========================================================================

#[test]
fn dilated_conv1d_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let channels = 3_usize;
    let length = 10_usize;
    let kernel_size = 3_usize;
    let dilation = 2_usize;

    let mut rng = LcgRng::new(0xD11A_7ED0);
    let input: Vec<f32> = (0..channels * length)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let filter_w: Vec<f32> = (0..channels * kernel_size)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let gate_w: Vec<f32> = (0..channels * kernel_size)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let filter_b: Vec<f32> = (0..channels).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let gate_b: Vec<f32> = (0..channels).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host re-derivation of the documented per-(ch,t) causal dilated conv.
    let mut filt_host = vec![0.0_f32; channels * length];
    let mut gate_host = vec![0.0_f32; channels * length];
    for ch in 0..channels {
        for t in 0..length {
            let mut f_acc = filter_b[ch];
            let mut g_acc = gate_b[ch];
            for k in 0..kernel_size {
                let src = t as isize - (k * dilation) as isize;
                let x = if src < 0 {
                    0.0_f32
                } else {
                    input[ch * length + src as usize]
                };
                f_acc += filter_w[ch * kernel_size + k] * x;
                g_acc += gate_w[ch * kernel_size + k] * x;
            }
            filt_host[ch * length + t] = f_acc;
            gate_host[ch * length + t] = g_acc;
        }
    }

    let ptx = crate::ptx_kernels::dilated_conv1d_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dilated_conv1d_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_fw = DeviceBuffer::<f32>::from_host(&filter_w).expect("d_fw");
    let d_gw = DeviceBuffer::<f32>::from_host(&gate_w).expect("d_gw");
    let d_fb = DeviceBuffer::<f32>::from_host(&filter_b).expect("d_fb");
    let d_gb = DeviceBuffer::<f32>::from_host(&gate_b).expect("d_gb");
    let d_of = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; channels * length]).expect("d_of");
    let d_og = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; channels * length]).expect("d_og");

    let total = (channels * length) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_fw.as_device_ptr(),
                d_gw.as_device_ptr(),
                d_fb.as_device_ptr(),
                d_gb.as_device_ptr(),
                d_of.as_device_ptr(),
                d_og.as_device_ptr(),
                length as u32,
                channels as u32,
                kernel_size as u32,
                dilation as u32,
            ),
        )
        .expect("launch dilated_conv1d_kernel");
    stream.synchronize().expect("sync");

    let mut filt_gpu = vec![0.0_f32; channels * length];
    let mut gate_gpu = vec![0.0_f32; channels * length];
    d_of.copy_to_host(&mut filt_gpu).expect("copy filt");
    d_og.copy_to_host(&mut gate_gpu).expect("copy gate");

    let (rel_f, abs_f) = worst_diff(&filt_gpu, &filt_host);
    for k in 0..filt_gpu.len() {
        assert!(
            close(filt_gpu[k], filt_host[k], 1e-4, 1e-5),
            "dilated filter[{k}] mismatch: gpu={} host={} (worst rel={rel_f:e} abs={abs_f:e})",
            filt_gpu[k],
            filt_host[k]
        );
    }
    let (rel_g, abs_g) = worst_diff(&gate_gpu, &gate_host);
    for k in 0..gate_gpu.len() {
        assert!(
            close(gate_gpu[k], gate_host[k], 1e-4, 1e-5),
            "dilated gate[{k}] mismatch: gpu={} host={} (worst rel={rel_g:e} abs={abs_g:e})",
            gate_gpu[k],
            gate_host[k]
        );
    }
}

// ===========================================================================
// 3. ctc_alpha  —  CRATE ORACLE (ctc::ctc_forward_log) + full-matrix re-derivation
//    LOG-DOMAIN: log-sum-exp via ex2/lg2 must be base-e (×log2(e) before ex2).
// ===========================================================================

/// Build the blank-interleaved extended target `l'` of length `2|target|+1`.
fn extended_target(target: &[u32], blank: u32) -> Vec<u32> {
    let mut l = vec![blank; 2 * target.len() + 1];
    for (i, &lbl) in target.iter().enumerate() {
        l[2 * i + 1] = lbl;
    }
    l
}

/// Row-normalised log-softmax `[T, V]` from deterministic normal logits.
fn make_log_probs(t: usize, v: usize, seed: u64) -> Vec<f32> {
    let mut rng = LcgRng::new(seed);
    let mut lp = vec![0.0_f32; t * v];
    rng.fill_normal(&mut lp);
    for row in 0..t {
        let base = &mut lp[row * v..(row + 1) * v];
        let max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = base.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        for x in base.iter_mut() {
            *x = (*x - max) - s;
        }
    }
    lp
}

/// Full-matrix CTC forward in the log domain (base-e), mirroring the kernel's
/// recursion exactly and storing every `alpha[t, s]`. The `t = 0` row is the
/// same initialisation the test uploads to the device.
fn host_ctc_alpha(log_probs: &[f32], t: usize, v: usize, l_prime: &[u32], blank: u32) -> Vec<f32> {
    let s_len = l_prime.len();
    let mut alpha = vec![f32::NEG_INFINITY; t * s_len];
    // t = 0 initialisation (only the first blank and first label are reachable).
    alpha[0] = log_probs[l_prime[0] as usize];
    if s_len > 1 {
        alpha[1] = log_probs[l_prime[1] as usize];
    }
    for ts in 1..t {
        for s in 0..s_len {
            let a_s = alpha[(ts - 1) * s_len + s];
            let a_s1 = if s >= 1 {
                alpha[(ts - 1) * s_len + s - 1]
            } else {
                f32::NEG_INFINITY
            };
            let mut val = host_lse2(a_s, a_s1);
            if s >= 2 && l_prime[s] != blank && l_prime[s] != l_prime[s - 2] {
                val = host_lse2(val, alpha[(ts - 1) * s_len + s - 2]);
            }
            let emit = log_probs[ts * v + l_prime[s] as usize];
            alpha[ts * s_len + s] = val + emit;
        }
    }
    alpha
}

fn run_ctc_case(fx: &GpuFixture, target: &[u32], t: usize, v: usize, seed: u64) {
    let blank = 0_u32;
    let l_prime = extended_target(target, blank);
    let s_len = l_prime.len();

    let log_probs = make_log_probs(t, v, seed);

    // Host full-matrix oracle (base-e), and its terminal log-likelihood.
    let alpha_host = host_ctc_alpha(&log_probs, t, v, &l_prime, blank);
    let ll_host = host_lse2(
        alpha_host[(t - 1) * s_len + s_len - 1],
        alpha_host[(t - 1) * s_len + s_len - 2],
    );

    // CRATE ORACLE cross-check: independent crate routine must agree at terminal.
    let target_usize: Vec<usize> = target.iter().map(|&x| x as usize).collect();
    let ll_crate = crate::ctc::ctc_forward_log(&log_probs, t, v, &target_usize, blank as usize)
        .expect("ctc_forward_log");
    assert!(
        (ll_host - ll_crate).abs() < 1e-4,
        "host CTC oracle disagrees with crate ctc_forward_log: host={ll_host} crate={ll_crate}"
    );

    // Upload alpha with row 0 initialised (kernel computes rows 1..T).
    let mut alpha0 = vec![f32::NEG_INFINITY; t * s_len];
    alpha0[0] = log_probs[l_prime[0] as usize];
    if s_len > 1 {
        alpha0[1] = log_probs[l_prime[1] as usize];
    }

    let ptx = crate::ptx_kernels::ctc_alpha_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ctc_alpha_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_lp = DeviceBuffer::<f32>::from_host(&log_probs).expect("d_lp");
    let d_alpha = DeviceBuffer::<f32>::from_host(&alpha0).expect("d_alpha");
    let d_ext = DeviceBuffer::<u32>::from_host(&l_prime).expect("d_ext");

    // grid = 1, block = S: the time recursion uses a block-wide `bar.sync`, so
    // every label position must live in the same block.
    let params = LaunchParams::new(1u32, s_len as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_lp.as_device_ptr(),
                d_alpha.as_device_ptr(),
                d_ext.as_device_ptr(),
                t as u32,
                v as u32,
                s_len as u32,
            ),
        )
        .expect("launch ctc_alpha_kernel");
    stream.synchronize().expect("sync");

    let mut alpha_gpu = vec![0.0_f32; t * s_len];
    d_alpha.copy_to_host(&mut alpha_gpu).expect("copy alpha");

    // Cell-by-cell: finite cells within tol; `-inf` cells must stay `-inf`.
    for ts in 0..t {
        for s in 0..s_len {
            let idx = ts * s_len + s;
            let h = alpha_host[idx];
            let g = alpha_gpu[idx];
            if h == f32::NEG_INFINITY {
                assert!(
                    g <= -1.0e30 || g == f32::NEG_INFINITY,
                    "ctc alpha[t={ts},s={s}] should be -inf, gpu={g}"
                );
            } else {
                assert!(
                    g.is_finite(),
                    "ctc alpha[t={ts},s={s}] non-finite on gpu: {g} (host={h})"
                );
                assert!(
                    close(g, h, 2e-3, 2e-3),
                    "ctc alpha[t={ts},s={s}] mismatch: gpu={g} host={h}"
                );
            }
        }
    }

    // Terminal log-likelihood vs the crate oracle.
    let ll_gpu = host_lse2(
        alpha_gpu[(t - 1) * s_len + s_len - 1],
        alpha_gpu[(t - 1) * s_len + s_len - 2],
    );
    assert!(
        close(ll_gpu, ll_crate, 2e-3, 2e-3),
        "ctc terminal log-likelihood mismatch: gpu={ll_gpu} crate={ll_crate}"
    );
}

#[test]
fn ctc_alpha_single_label_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // S = 3, no `s-2` diagonal transitions: isolates the base-e LSE of two
    // finite operands (the base-2 exp bug shows up here).
    run_ctc_case(&fx, &[1], 5, 4, 0x0C7C_0001);
}

#[test]
fn ctc_alpha_two_label_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // S = 5: exercises the `s-2` diagonal transition and an unreachable `-inf`
    // cell (alpha[1, 4]), so both the base-e LSE and the `lse(-inf, -inf) = -inf`
    // robustness are validated against the crate oracle.
    run_ctc_case(&fx, &[1, 2], 6, 5, 0x0C7C_0002);
}

// ===========================================================================
// 4. spec_augment_mask  —  INDEPENDENT HOST RE-DERIVATION (deterministic bands)
// ===========================================================================

#[test]
fn spec_augment_mask_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let t = 6_usize;
    let f = 8_usize;
    let t_start = 2_u32;
    let t_len = 2_u32; // frames 2, 3
    let f_start = 5_u32;
    let f_len = 2_u32; // bins 5, 6

    let mut rng = LcgRng::new(0x5AE6_0A11);
    let mel: Vec<f32> = (0..t * f).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // Host re-derivation: zero where (t in time-band) OR (f in freq-band).
    let mut host = mel.clone();
    for ti in 0..t {
        for fi in 0..f {
            let in_t = ti as u32 >= t_start && (ti as u32) < t_start + t_len;
            let in_f = fi as u32 >= f_start && (fi as u32) < f_start + f_len;
            if in_t || in_f {
                host[ti * f + fi] = 0.0;
            }
        }
    }

    let ptx = crate::ptx_kernels::spec_augment_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "spec_augment_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mel = DeviceBuffer::<f32>::from_host(&mel).expect("d_mel");
    let total = (t * f) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mel.as_device_ptr(),
                t as u32,
                f as u32,
                t_start,
                t_len,
                f_start,
                f_len,
            ),
        )
        .expect("launch spec_augment_mask_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; t * f];
    d_mel.copy_to_host(&mut gpu).expect("copy mel");

    // Masking is either a zero or an exact copy, so the result is bit-exact.
    for k in 0..gpu.len() {
        assert_eq!(
            gpu[k].to_bits(),
            host[k].to_bits(),
            "spec_augment out[{k}] mismatch: gpu={} host={}",
            gpu[k],
            host[k]
        );
    }
}

// ===========================================================================
// 5. depthwise_conv1d  —  INDEPENDENT HOST RE-DERIVATION
//    (matches the private `depthwise_causal_conv1d` inside `ConvModule`)
// ===========================================================================

#[test]
fn depthwise_conv1d_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let channels = 4_usize;
    let length = 12_usize;
    let kernel_size = 5_usize;

    let mut rng = LcgRng::new(0xDE97_1234);
    let input: Vec<f32> = (0..channels * length)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let weights: Vec<f32> = (0..channels * kernel_size)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let bias: Vec<f32> = (0..channels).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host re-derivation: causal left-pad of (kernel_size - 1); idx = t - pad + k.
    let pad = kernel_size - 1;
    let mut host = vec![0.0_f32; channels * length];
    for ch in 0..channels {
        for t in 0..length {
            let mut acc = bias[ch];
            for k in 0..kernel_size {
                let idx = t as isize - pad as isize + k as isize;
                if idx < 0 {
                    continue;
                }
                acc += weights[ch * kernel_size + k] * input[ch * length + idx as usize];
            }
            host[ch * length + t] = acc;
        }
    }

    let ptx = crate::ptx_kernels::depthwise_conv1d_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "depthwise_conv1d_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_b = DeviceBuffer::<f32>::from_host(&bias).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; channels * length]).expect("d_out");

    let total = (channels * length) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_w.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                channels as u32,
                length as u32,
                kernel_size as u32,
            ),
        )
        .expect("launch depthwise_conv1d_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; channels * length];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    let (rel, abs) = worst_diff(&gpu, &host);
    for k in 0..gpu.len() {
        assert!(
            close(gpu[k], host[k], 1e-4, 1e-5),
            "depthwise out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            host[k]
        );
    }
}

// ===========================================================================
// 6. rel_pos_bias  —  CRATE ORACLE (attention::RelPosEncoding::bias_matrix)
// ===========================================================================

#[test]
fn rel_pos_bias_matches_crate() {
    use crate::attention::RelPosEncoding;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // `max_len` deliberately smaller than the query/key span so that some
    // relative displacements underflow the table's negative range
    // (k - q < -(max_len-1)). The crate clamps those to index 0; a u32-modular
    // kernel instead wraps them to the *maximum* index — exactly the divergence
    // this test must catch. In-range and positive-overflow displacements are
    // also covered.
    let max_len = 4_usize;
    let q_len = 7_usize;
    let k_len = 7_usize;
    let table_len = 2 * max_len - 1; // 7

    let mut rng = LcgRng::new(0x4E10_B1A5);
    let table: Vec<f32> = (0..table_len).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CRATE ORACLE: index = clamp((k - q) + (max_len - 1), 0, 2*max_len-2),
    // computed in signed arithmetic. Negative relative positions (k < q) must
    // map to small positive table indices, NOT collapse to the last entry.
    let enc = RelPosEncoding {
        table: table.clone(),
        max_len,
    };
    let expected = enc.bias_matrix(q_len, k_len);

    let ptx = crate::ptx_kernels::rel_pos_bias_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rel_pos_bias_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_table = DeviceBuffer::<f32>::from_host(&table).expect("d_table");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; q_len * k_len]).expect("d_out");

    let total = (q_len * k_len) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_table.as_device_ptr(),
                d_out.as_device_ptr(),
                q_len as u32,
                k_len as u32,
                max_len as u32,
            ),
        )
        .expect("launch rel_pos_bias_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; q_len * k_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // Pure gather: the result must be bit-exact with the crate's table look-up.
    for q in 0..q_len {
        for k in 0..k_len {
            let idx = q * k_len + k;
            assert_eq!(
                gpu[idx].to_bits(),
                expected[idx].to_bits(),
                "rel_pos_bias[q={q},k={k}] mismatch: gpu={} crate={} \
                 (k-q={})",
                gpu[idx],
                expected[idx],
                k as isize - q as isize
            );
        }
    }
}

// ===========================================================================
// 7. stats_pool  —  CRATE ORACLE (mean) + INDEPENDENT HOST RE-DERIVATION (std)
//    Honest divergence: the kernel's std uses POPULATION variance (/T) + 1e-8,
//    while the crate's `stats_pool` uses Bessel /(T-1) + a 1e-10 clamp; only the
//    mean is compared to the crate. The kernel's intra-block reduction must also
//    be race-free (one writer per channel).
// ===========================================================================

#[test]
fn stats_pool_matches_oracles() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let t = 50_usize;
    let c = 8_usize;
    let eps = 1.0e-8_f32;

    // Input is laid out [C, T] (channel-major), matching the kernel's
    // `input[c * T + t]` indexing.
    let mut rng = LcgRng::new(0x57A7_5009);
    let mut input = vec![0.0_f32; c * t];
    rng.fill_normal(&mut input);

    // CRATE ORACLE for the mean: `stats_pool` expects a [T, C] tensor, so build
    // the transpose and take its first C outputs (per-channel mean over time).
    let mut tc = vec![0.0_f32; t * c];
    for ch in 0..c {
        for ti in 0..t {
            tc[ti * c + ch] = input[ch * t + ti];
        }
    }
    let crate_out = crate::speaker::stats_pool(&tc, t, c).expect("stats_pool");
    let mean_crate = &crate_out[0..c];

    // Independent host re-derivation matching the kernel's documented math:
    // mean = (1/T) Σ x; std = sqrt((1/T) Σ (x-mean)^2 + 1e-8)  (population var).
    let mut mean_host = vec![0.0_f32; c];
    let mut std_host = vec![0.0_f32; c];
    for ch in 0..c {
        let mut sum = 0.0_f32;
        for ti in 0..t {
            sum += input[ch * t + ti];
        }
        let mean = sum / t as f32;
        let mut var = 0.0_f32;
        for ti in 0..t {
            let d = input[ch * t + ti] - mean;
            var += d * d;
        }
        mean_host[ch] = mean;
        std_host[ch] = (var / t as f32 + eps).sqrt();
    }

    let ptx = crate::ptx_kernels::stats_pool_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "stats_pool_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_mean = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; c]).expect("d_mean");
    let d_std = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; c]).expect("d_std");

    // One block per channel; block = 128 threads (4 warps) exercises the
    // cross-warp shared-memory reduction.
    let params = LaunchParams::new(c as u32, 128u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_std.as_device_ptr(),
                t as u32,
                c as u32,
            ),
        )
        .expect("launch stats_pool_kernel");
    stream.synchronize().expect("sync");

    let mut mean_gpu = vec![0.0_f32; c];
    let mut std_gpu = vec![0.0_f32; c];
    d_mean.copy_to_host(&mut mean_gpu).expect("copy mean");
    d_std.copy_to_host(&mut std_gpu).expect("copy std");

    // Mean: compare to the crate oracle (and, equivalently, the host mean).
    let (rel_m, abs_m) = worst_diff(&mean_gpu, mean_crate);
    for ch in 0..c {
        assert!(
            close(mean_gpu[ch], mean_crate[ch], 1e-4, 1e-5),
            "stats_pool mean[{ch}] vs crate mismatch: gpu={} crate={} (worst rel={rel_m:e} abs={abs_m:e})",
            mean_gpu[ch],
            mean_crate[ch]
        );
        assert!(
            close(mean_gpu[ch], mean_host[ch], 1e-4, 1e-5),
            "stats_pool mean[{ch}] vs host mismatch: gpu={} host={}",
            mean_gpu[ch],
            mean_host[ch]
        );
    }

    // Std: compare to the independent host re-derivation (population variance).
    let (rel_s, abs_s) = worst_diff(&std_gpu, &std_host);
    for ch in 0..c {
        assert!(
            close(std_gpu[ch], std_host[ch], 1e-4, 1e-5),
            "stats_pool std[{ch}] mismatch: gpu={} host={} (worst rel={rel_s:e} abs={abs_s:e})",
            std_gpu[ch],
            std_host[ch]
        );
    }
}
