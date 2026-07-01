//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` canaries: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars are passed as the matching Rust scalar
//! (`.param .u32` / `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   `ia3_scale_kernel` ↔ [`crate::ia3::ia3::Ia3Vector::apply`],
//!   `nf4_dequant_kernel` ↔ [`crate::quant::nf4_quant::dequantize_nf4`]
//!   (per-block, bit-for-bit table lookup),
//!   `lora_merge_kernel` ↔ [`crate::lora::lora::LoraLinear::merge_into_w`],
//!   `prompt_concat_kernel` ↔
//!   [`crate::prefix::prompt_tuning::SoftPrompt::prepend_to_sequence`].
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function (the op is fused into a larger routine or the kernel uses a
//!   different neuron/normalisation path on the CPU), so the oracle is an
//!   independent Rust re-implementation of the kernel's *documented* arithmetic:
//!   `lora_matmul_kernel` (fused `B·(A·x)`), `prefix_expand_kernel` (tiling),
//!   and `adapter_forward_kernel` (bottleneck FFN + tanh-GELU + residual, using
//!   the crate's own [`crate::adapter::houlsby::gelu`] for the activation; the
//!   kernel deliberately omits the LayerNorm that `HoulsbyAdapter::forward`
//!   applies, so a whole-layer crate oracle is impossible). These still
//!   genuinely fail if ptxas miscompiles or the PTX has a wrong constant / shift
//!   / index, because the host code is independent of the JIT-compiled PTX.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.
//!
//! ## PTX bug audit result
//!
//! All seven kernels were validated on a real RTX A4000 (sm_86, CUDA 12.4).
//! Every kernel JIT-compiles (ptxas accepts the PTX — no invalid-PTX bugs) and
//! every kernel reproduces its CPU oracle within FP32 tolerance — including the
//! base-2 exp/log trap: `adapter_forward_kernel`'s tanh-GELU correctly scales
//! its argument by `log2(e)` before `ex2.approx.f32`, so it matches the libm
//! `tanh` reference rather than being ~40 % off. No bugs were found; no kernel
//! is a hollow stub (every kernel issues real `st.global` stores of computed
//! values, verified by the oracle comparisons below).

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, which both creates the context and makes
/// it current on the calling thread; the returned `Arc<Context>` must be kept
/// alive for the whole test (nextest runs each test in its own process, so a
/// per-test context is fine).
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

/// Worst (relative, absolute) divergence over two equal-length slices.
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

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A failure here means ptxas rejected the PTX — a real, must-fix bug.
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

// ===========================================================================
// 1. lora_matmul  —  INDEPENDENT HOST RE-DERIVATION (fused B·(A·x))
// ===========================================================================

#[test]
fn lora_matmul_matches_host() {
    // `lora_matmul_kernel` computes the pure low-rank product `B·(A·x)` (no base
    // weight), one thread per output row. The crate's `LoraLinear::forward` adds
    // `W·x` on top, so the oracle here is an independent host re-derivation of
    // exactly the kernel's documented arithmetic.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 7_usize; // in_dim
    let r = 5_usize; // rank
    let m = 12_usize; // out_dim

    let mut rng = LcgRng::new(0x10A_3A11);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let a: Vec<f32> = (0..r * n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..m * r).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host: out[row] = Σ_ri B[row,ri] · (Σ_j A[ri,j]·x[j]).
    let mut out_host = vec![0.0_f32; m];
    for (row, slot) in out_host.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for ri in 0..r {
            let mut tmp = 0.0_f32;
            for j in 0..n {
                tmp += a[ri * n + j] * x[j];
            }
            acc += b[row * r + ri] * tmp;
        }
        *slot = acc;
    }

    let ptx = crate::ptx_kernels::lora_matmul_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lora_matmul_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m]).expect("d_out");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(m as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                r as u32,
                m as u32,
            ),
        )
        .expect("launch lora_matmul_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; m];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // GPU fuses each product with `fma.rn`; the host uses plain mul/add. Over
    // r·n = 35 mixed-sign terms the divergence is a few ulp (~1e-6 relative);
    // 1e-4 is comfortable yet still flags any wrong index / missing term.
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..m {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "lora_matmul out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 2. ia3_scale  —  CRATE ORACLE (ia3::ia3::Ia3Vector::apply)
// ===========================================================================

#[test]
fn ia3_scale_matches_cpu() {
    use crate::ia3::ia3::{Ia3Placement, Ia3Vector};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 100_usize;
    let mut rng = LcgRng::new(0x1A3_5CA1);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let scale: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: the crate's own element-wise IA³ scaling.
    let mut iv = Ia3Vector::new(n, Ia3Placement::FeedForward);
    iv.scale = scale.clone();
    let out_cpu = iv.apply(&x);

    let ptx = crate::ptx_kernels::ia3_scale_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ia3_scale_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_scale = DeviceBuffer::<f32>::from_host(&scale).expect("d_scale");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_scale.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch ia3_scale_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Plain `mul.f32` on the GPU vs Rust `*`: identical single rounding, so this
    // is bit-exact. A tiny tolerance still guards against any future reorder.
    for k in 0..n {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-6, 1e-7),
            "ia3_scale out[{k}] mismatch: gpu={} cpu={}",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ===========================================================================
// 3. prefix_expand  —  INDEPENDENT HOST RE-DERIVATION (tile over batch)
// ===========================================================================

#[test]
fn prefix_expand_matches_host() {
    // `prefix_expand_kernel` tiles prefix[seq, dim] into out[batch*seq, dim] via
    // `out[b*seq + s, c] = prefix[s, c]`. Pure data movement → bit-exact oracle.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let batch = 3_usize;
    let seq = 4_usize;
    let dim = 5_usize;

    let mut rng = LcgRng::new(0x9E_F1C);
    let prefix: Vec<f32> = (0..seq * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host: out row r = (b, s) with s = r % seq tiled over the batch.
    let total = batch * seq * dim;
    let mut out_host = vec![0.0_f32; total];
    for (idx, slot) in out_host.iter_mut().enumerate() {
        let row = idx / dim;
        let col = idx % dim;
        let src_row = row % seq;
        *slot = prefix[src_row * dim + col];
    }

    let ptx = crate::ptx_kernels::prefix_expand_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "prefix_expand_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_prefix = DeviceBuffer::<f32>::from_host(&prefix).expect("d_prefix");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_prefix.as_device_ptr(),
                d_out.as_device_ptr(),
                batch as u32,
                seq as u32,
                dim as u32,
            ),
        )
        .expect("launch prefix_expand_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Pure copy: must be bit-exact.
    for k in 0..total {
        assert_eq!(
            out_gpu[k].to_bits(),
            out_host[k].to_bits(),
            "prefix_expand out[{k}] mismatch: gpu={} host={}",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 4. adapter_forward  —  HOST RE-DERIVATION using crate gelu (no LayerNorm)
// ===========================================================================

#[test]
fn adapter_forward_matches_host() {
    // `adapter_forward_kernel` is a bottleneck FFN: down-proj + bias → tanh-GELU
    // → up-proj + bias + residual. It deliberately omits the LayerNorm that
    // `HoulsbyAdapter::forward` applies first, so the oracle is an independent
    // host re-derivation that reuses the crate's own `gelu` (identical tanh
    // formula). This is the base-2 trap test: the kernel's `ex2.approx`-based
    // tanh must scale by log2(e), or it would diverge ~40 % from libm `tanh`.
    use crate::adapter::houlsby::gelu;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize; // in_dim
    let bot = 4_usize; // bottleneck
    let seq = 3_usize; // tokens

    // Keep magnitudes small so the GELU argument stays in `ex2`'s accurate
    // domain (and far from the `e^{2y}` overflow that makes the rational tanh
    // approximation degenerate for large positive arguments).
    let mut rng = LcgRng::new(0xADA_F00D);
    let x: Vec<f32> = (0..seq * n).map(|_| rng.next_f32() - 0.5).collect();
    let w_down: Vec<f32> = (0..bot * n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();
    let w_up: Vec<f32> = (0..n * bot).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();
    let b_down: Vec<f32> = (0..bot).map(|_| rng.next_f32() - 0.5).collect();
    let b_up: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

    // Host re-derivation (no LayerNorm): for each (tok, out_col).
    let mut out_host = vec![0.0_f32; seq * n];
    for tok in 0..seq {
        for out_col in 0..n {
            let residual = x[tok * n + out_col];
            let mut acc = b_up[out_col];
            for bi in 0..bot {
                let mut h = b_down[bi];
                for j in 0..n {
                    h += w_down[bi * n + j] * x[tok * n + j];
                }
                acc += w_up[out_col * bot + bi] * gelu(h);
            }
            out_host[tok * n + out_col] = acc + residual;
        }
    }

    let ptx = crate::ptx_kernels::adapter_forward_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "adapter_forward_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_wd = DeviceBuffer::<f32>::from_host(&w_down).expect("d_wd");
    let d_wu = DeviceBuffer::<f32>::from_host(&w_up).expect("d_wu");
    let d_bd = DeviceBuffer::<f32>::from_host(&b_down).expect("d_bd");
    let d_bu = DeviceBuffer::<f32>::from_host(&b_up).expect("d_bu");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; seq * n]).expect("d_out");

    let total = (seq * n) as u32;
    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_wd.as_device_ptr(),
                d_wu.as_device_ptr(),
                d_bd.as_device_ptr(),
                d_bu.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                bot as u32,
                seq as u32,
            ),
        )
        .expect("launch adapter_forward_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; seq * n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // The GPU tanh uses `ex2.approx.f32` (~2 ulp) + `div.rn`, the host uses libm
    // `tanh` (<1 ulp). Over the small-magnitude inputs the per-element GELU
    // agrees to ~1e-5 relative; 1e-3 is comfortable yet flags a base-2 error
    // (which would be tens of percent) or any wrong weight index immediately.
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-3, 1e-4),
            "adapter_forward out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 5. nf4_dequant  —  CRATE ORACLE (quant::nf4_quant::dequantize_nf4, per block)
// ===========================================================================

#[test]
fn nf4_dequant_matches_cpu() {
    use crate::quant::nf4_quant::dequantize_nf4;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_blocks = 3_usize;
    let block_size = 8_usize; // even, as the kernel documents
    let bytes_per_block = block_size / 2;

    // Deterministic NF4 codes (0..15) for every element, packed two per byte:
    // low nibble = even element, high nibble = odd element (matches both the
    // kernel and `dequantize_nf4`'s unpacking).
    let n_elems = n_blocks * block_size;
    let codes_idx: Vec<u8> = (0..n_elems).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let mut packed = vec![0_u8; n_blocks * bytes_per_block];
    for (i, &c) in codes_idx.iter().enumerate() {
        if i % 2 == 0 {
            packed[i / 2] = c;
        } else {
            packed[i / 2] |= c << 4;
        }
    }
    // Distinct positive per-block scales.
    let absmax: Vec<f32> = (0..n_blocks).map(|k| 0.5 + 0.75 * k as f32).collect();

    // CPU reference: the crate's own dequantiser, called per block.
    let mut out_cpu = vec![0.0_f32; n_elems];
    for k in 0..n_blocks {
        let block_bytes = &packed[k * bytes_per_block..(k + 1) * bytes_per_block];
        let deq = dequantize_nf4(block_bytes, absmax[k], block_size);
        out_cpu[k * block_size..(k + 1) * block_size].copy_from_slice(&deq);
    }

    let ptx = crate::ptx_kernels::nf4_dequant_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "nf4_dequant_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_codes = DeviceBuffer::<u8>::from_host(&packed).expect("d_codes");
    let d_absmax = DeviceBuffer::<f32>::from_host(&absmax).expect("d_absmax");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_elems]).expect("d_out");

    let n_bytes = (n_blocks * bytes_per_block) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_bytes, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_codes.as_device_ptr(),
                d_absmax.as_device_ptr(),
                d_out.as_device_ptr(),
                n_blocks as u32,
                block_size as u32,
            ),
        )
        .expect("launch nf4_dequant_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_elems];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Both sides are `NF4_QUANTS[code] * absmax` with a single multiply, so the
    // result is bit-exact; a tiny tolerance guards against rounding-mode drift.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..n_elems {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-5, 1e-6),
            "nf4_dequant out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ===========================================================================
// 6. lora_merge  —  CRATE ORACLE (lora::lora::LoraLinear::merge_into_w)
// ===========================================================================

#[test]
fn lora_merge_matches_cpu() {
    use crate::lora::lora::{LoraConfig, LoraLinear};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize; // in_dim
    let r = 4_usize; // rank
    let m = 6_usize; // out_dim
    let scale = 0.25_f32;

    let mut rng = LcgRng::new(0x10A_E26E);
    let w0: Vec<f32> = (0..m * n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let a: Vec<f32> = (0..r * n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..m * r).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: load the same factors into a LoraLinear and merge.
    let cfg = LoraConfig {
        r,
        alpha: scale * r as f32, // so scale = alpha / r
        init_scale: 0.0,
    };
    let mut lora = LoraLinear::new(n, m, &cfg, &mut rng);
    lora.w = w0.clone();
    lora.a = a.clone();
    lora.b = b.clone();
    lora.scale = scale;
    lora.merge_into_w();
    let w_expected = lora.w.clone();

    let ptx = crate::ptx_kernels::lora_merge_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lora_merge_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w0).expect("d_w");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");

    let total = (m * n) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                scale,
                n as u32,
                r as u32,
                m as u32,
            ),
        )
        .expect("launch lora_merge_kernel");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; m * n];
    d_w.copy_to_host(&mut w_gpu).expect("copy w");

    // GPU fuses `B·A` with `fma.rn` then one `mul`+`add`; the CPU sums with
    // mul/add. Over r = 4 terms the divergence is a few ulp (~1e-6 relative).
    let (rel, abs) = worst_diff(&w_gpu, &w_expected);
    for k in 0..w_gpu.len() {
        assert!(
            close(w_gpu[k], w_expected[k], 1e-4, 1e-6),
            "lora_merge w[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            w_gpu[k],
            w_expected[k]
        );
    }
}

// ===========================================================================
// 7. prompt_concat  —  CRATE ORACLE (prefix::prompt_tuning::SoftPrompt)
// ===========================================================================

#[test]
fn prompt_concat_matches_cpu() {
    use crate::prefix::prompt_tuning::SoftPrompt;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let p = 2_usize; // prompt tokens
    let s = 4_usize; // sequence tokens
    let d = 5_usize; // embed dim

    let mut rng = LcgRng::new(0x9_C047);
    let prompt: Vec<f32> = (0..p * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let seq: Vec<f32> = (0..s * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: the crate's own soft-prompt prepend.
    let sp = SoftPrompt {
        num_tokens: p,
        embed_dim: d,
        embeddings: prompt.clone(),
    };
    let out_cpu = sp.prepend_to_sequence(&seq, s);

    let ptx = crate::ptx_kernels::prompt_concat_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "prompt_concat_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let total = (p + s) * d;
    let d_prompt = DeviceBuffer::<f32>::from_host(&prompt).expect("d_prompt");
    let d_seq = DeviceBuffer::<f32>::from_host(&seq).expect("d_seq");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_prompt.as_device_ptr(),
                d_seq.as_device_ptr(),
                d_out.as_device_ptr(),
                p as u32,
                s as u32,
                d as u32,
            ),
        )
        .expect("launch prompt_concat_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Pure concatenation copy: must be bit-exact.
    for k in 0..total {
        assert_eq!(
            out_gpu[k].to_bits(),
            out_cpu[k].to_bits(),
            "prompt_concat out[{k}] mismatch: gpu={} cpu={}",
            out_gpu[k],
            out_cpu[k]
        );
    }
}
