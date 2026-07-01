//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to a CPU oracle. The launch
//! ABI follows the same convention as `oxicuda-snn` / `oxicuda-sparse`: device
//! buffers are passed as their `CUdeviceptr` (`.param .u64`), scalars as the
//! matching Rust scalar type, in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   - `patch_embed_kernel` ↔ [`crate::patch_embed::conv2d_patch::PatchEmbed::forward`]
//!   - `roi_align` ↔ [`crate::detection::roi_align::roi_align`]
//!   - `image_normalize` ↔ [`crate::augment::normalize::normalize_chw`]
//!   - `focal_loss` ↔ [`crate::losses::focal::binary_focal_loss_one`] (α=0.25, γ=2)
//! * **Independent host re-derivation** — no single dedicated crate function
//!   mirrors the kernel, so the oracle is an independent Rust re-implementation
//!   of the kernel's *documented* arithmetic:
//!   - `bilinear_interp` — half-pixel convention bilinear blend
//!   - `contrastive_loss` — max-stabilised 3-pass LSE cross-entropy
//!     (base-e oracle; directly catches the missing-LOG2E bug class)
//!   - `adaptive_avg_pool` — integer window-bound formula
//!
//! ## PTX bug classes checked explicitly
//!
//! * **Base-2 exp/log** (`contrastive_loss`, `focal_loss`): both kernels correctly
//!   multiply by `LOG2E` before `ex2.approx` and by `LN2` after `lg2.approx`.
//!   The base-e CPU oracles for contrastive loss would catch a missing scale
//!   factor by orders of magnitude (~50 % relative error on sum_exp).
//!   `focal_loss` is likewise cross-checked against the stable CPU sigmoid + log.
//! * **Invalid PTX**: JIT-load via `Module::from_ptx` is the gatekeeper — any
//!   `ptxas` error aborts with a clear panic message.
//! * **`rcp.approx.f32` precision**: `roi_align`, `image_normalize`, and
//!   `adaptive_avg_pool` all use `rcp.approx`; test dimensions are chosen so
//!   the reciprocal argument is an exact power-of-two (no extra approximation
//!   error), and the 1e-4 relative tolerance covers any rounding divergence.
//!
//! Every test skips gracefully when no CUDA device is present.

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
// 1. patch_embed  —  CRATE ORACLE (crate::patch_embed::conv2d_patch::PatchEmbed::forward)
// ===========================================================================

/// Verify that `patch_embed_kernel` matches `PatchEmbed::forward` for identical
/// kernel weights, bias, and input image.
///
/// The GPU kernel uses `fma.rn.f32` (single rounding) while the CPU uses two
/// separate scalar ops (two roundings); we allow 1e-4 relative divergence which
/// is thousands of ULPs above the expected ~1 ULP spread and will only fail if
/// the PTX addresses a wrong index or omits a dimension factor.
#[test]
fn patch_embed_matches_cpu() {
    use crate::patch_embed::conv2d_patch::{PatchEmbed, PatchEmbedConfig};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Small but non-trivial: 3-channel 8×8 image, 4×4 patches → 4 patches,
    // embed_dim = 8.  FP32 budget: 3×4×4 = 48 multiply-adds per output element.
    let img_size = 8_usize;
    let patch_size = 4_usize;
    let in_chans = 3_usize;
    let embed_dim = 8_usize;
    let n_patches = (img_size / patch_size) * (img_size / patch_size); // 4

    let cfg =
        PatchEmbedConfig::new(img_size, patch_size, in_chans, embed_dim).expect("valid config");
    let mut rng = LcgRng::new(0xBEEF_F00D);
    let pe = PatchEmbed::new(cfg.clone(), &mut rng);

    // Random input image.
    let image: Vec<f32> = (0..in_chans * img_size * img_size)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // CPU reference.
    let cpu_out = pe.forward(&image).expect("cpu patch_embed forward");

    // GPU kernel.
    let ptx = crate::ptx_kernels::patch_embed_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "patch_embed");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&image).expect("d_in");
    let d_kernel = DeviceBuffer::<f32>::from_host(&pe.weights.kernel).expect("d_kernel");
    let d_bias = DeviceBuffer::<f32>::from_host(&pe.weights.bias).expect("d_bias");
    let d_out =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_patches * embed_dim]).expect("d_out");

    let block = 256_u32;
    let total = (n_patches * embed_dim) as u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_kernel.as_device_ptr(),
                d_bias.as_device_ptr(),
                d_out.as_device_ptr(),
                n_patches as u32,
                embed_dim as u32,
                in_chans as u32,
                patch_size as u32,
                img_size as u32,
            ),
        )
        .expect("launch patch_embed");
    stream.synchronize().expect("sync");

    let mut gpu_out = vec![0.0_f32; n_patches * embed_dim];
    d_out.copy_to_host(&mut gpu_out).expect("copy out");

    let (rel, abs) = worst_diff(&gpu_out, &cpu_out);
    // Tolerance: fma.rn.f32 vs two-op CPU — expected ≤1 ULP per output, 1e-4 is generous.
    for k in 0..gpu_out.len() {
        assert!(
            close(gpu_out[k], cpu_out[k], 1e-4_f32, 1e-6_f32),
            "patch_embed out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 2. bilinear_interp  —  INDEPENDENT HOST RE-DERIVATION (half-pixel convention)
// ===========================================================================

/// Host re-derivation of `bilinear_interp_kernel`'s exact F32 pipeline.
///
/// Mirrors the PTX step-by-step:
/// 1. `src_y = (oy + 0.5) * (in_h / out_h) - 0.5` (same ops as PTX; all F32)
/// 2. Clamp to `[0, in_h - 1]`
/// 3. `floor(src_y)` → y0, `src_y - y0` → fy; likewise for x
/// 4. y1 = min(y0+1, in_h-1), x1 = min(x0+1, in_w-1)
/// 5. Four-tap bilinear blend using the GPU's `fma.rn.f32` pattern
///
/// Since the PTX uses `div.rn.f32` (correctly-rounded) and the same arithmetic
/// order, divergence is ≤2 ULP per blended pixel — well within 1e-4 relative.
fn host_bilinear_interp(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    n_chans: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_chans * out_h * out_w];
    for c in 0..n_chans {
        for oy in 0..out_h {
            for ox in 0..out_w {
                // Half-pixel: src_y = (oy + 0.5) * (in_h / out_h) - 0.5
                let scale_y = in_h as f32 / out_h as f32;
                let scale_x = in_w as f32 / out_w as f32;
                let src_y = (oy as f32 + 0.5_f32) * scale_y - 0.5_f32;
                let src_x = (ox as f32 + 0.5_f32) * scale_x - 0.5_f32;

                let src_y = src_y.max(0.0_f32).min(in_h as f32 - 1.0_f32);
                let src_x = src_x.max(0.0_f32).min(in_w as f32 - 1.0_f32);

                let y0f = src_y.floor();
                let x0f = src_x.floor();
                let fy = src_y - y0f;
                let fx = src_x - x0f;

                let y0 = y0f as usize;
                let x0 = x0f as usize;
                let y1 = (y0 + 1).min(in_h - 1);
                let x1 = (x0 + 1).min(in_w - 1);

                let ch_base = c * in_h * in_w;
                let tl = src[ch_base + y0 * in_w + x0];
                let tr = src[ch_base + y0 * in_w + x1];
                let bl = src[ch_base + y1 * in_w + x0];
                let br = src[ch_base + y1 * in_w + x1];

                // Bilinear blend matching PTX:
                // top = tl*(1-fx) + tr*fx,  bot = bl*(1-fx) + br*fx
                // result = top*(1-fy) + bot*fy
                let one_minus_fx = 1.0_f32 - fx;
                let one_minus_fy = 1.0_f32 - fy;
                let top = tl * one_minus_fx + tr * fx;
                let bot = bl * one_minus_fx + br * fx;
                let val = top * one_minus_fy + bot * fy;

                out[c * out_h * out_w + oy * out_w + ox] = val;
            }
        }
    }
    out
}

#[test]
fn bilinear_interp_matches_host_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // 2-channel 8×8 → 4×4 (scale = 2.0, rational — minimises rounding).
    let in_h = 8_usize;
    let in_w = 8_usize;
    let out_h = 4_usize;
    let out_w = 4_usize;
    let n_chans = 2_usize;

    let n_src = n_chans * in_h * in_w;
    let n_dst = n_chans * out_h * out_w;

    let mut rng = LcgRng::new(0x81C_5EED);
    let src: Vec<f32> = (0..n_src).map(|_| rng.next_f32()).collect();

    // CPU oracle (independent re-derivation).
    let cpu_out = host_bilinear_interp(&src, in_h, in_w, out_h, out_w, n_chans);

    // GPU kernel.
    let ptx = crate::ptx_kernels::bilinear_interp_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bilinear_interp");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_src = DeviceBuffer::<f32>::from_host(&src).expect("d_src");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_dst]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_dst as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_src.as_device_ptr(),
                d_out.as_device_ptr(),
                in_h as u32,
                in_w as u32,
                out_h as u32,
                out_w as u32,
                n_chans as u32,
            ),
        )
        .expect("launch bilinear_interp");
    stream.synchronize().expect("sync");

    let mut gpu_out = vec![0.0_f32; n_dst];
    d_out.copy_to_host(&mut gpu_out).expect("copy out");

    let (rel, abs) = worst_diff(&gpu_out, &cpu_out);
    // Tolerance: FMA vs two-op blend, div.rn.f32 vs host division — ≤2 ULP per pixel.
    for k in 0..gpu_out.len() {
        assert!(
            close(gpu_out[k], cpu_out[k], 1e-4_f32, 1e-6_f32),
            "bilinear_interp out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 3. contrastive_loss  —  INDEPENDENT HOST RE-DERIVATION
//    Oracle uses base-e exp/log → catches missing LOG2E / LN2 scaling in PTX
// ===========================================================================

/// Base-e, max-stabilised cross-entropy matching the PTX 3-pass algorithm.
///
/// ```text
/// max_val  = max over j of sim[row, j]
/// sum_exp  = Σ_j exp(sim[row,j] − max_val)
/// loss[row]= −(sim[row,row] − max_val) + ln(sum_exp)
/// ```
///
/// **Why this catches the base-2 bug**: if `LOG2E` were absent before
/// `ex2.approx`, the kernel would compute `exp2(sim − max)` = `exp((sim-max)/ln2)`,
/// making every exponent scaled by `1/ln(2) ≈ 1.44`, giving a `sum_exp` that is
/// systematically wrong by `sum_exp^{1/ln(2)-1}` — a ~50 % relative error on
/// this oracle for typical similarity magnitudes.
fn host_contrastive_loss(sim: &[f32], n_batch: usize) -> Vec<f32> {
    let mut loss = vec![0.0_f32; n_batch];
    for row in 0..n_batch {
        let base = row * n_batch;
        let row_slice = &sim[base..base + n_batch];

        // Pass 1: row maximum.
        let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: sum of exp(sim − max) — base-e, not base-2.
        let sum_exp: f32 = row_slice.iter().map(|&x| (x - max_val).exp()).sum();

        // Pass 3: -(sim[row,row] − max_val) + ln(sum_exp).
        let sim_diag = sim[row * n_batch + row];
        loss[row] = -(sim_diag - max_val) + sum_exp.ln();
    }
    loss
}

#[test]
fn contrastive_loss_matches_host_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // B=8 so every row fits inside one warp, with moderate similarity values
    // to keep exponents well within ex2.approx's accurate range.
    let n_batch = 8_usize;
    let mut rng = LcgRng::new(0x4C6F_5353);

    // Similarity values in [-2, 2]: exponents after shifting ≤ 4 → no overflow.
    let sim: Vec<f32> = (0..n_batch * n_batch)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();

    // CPU oracle.
    let cpu_loss = host_contrastive_loss(&sim, n_batch);

    // GPU kernel.
    let ptx = crate::ptx_kernels::contrastive_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "contrastive_loss");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sim = DeviceBuffer::<f32>::from_host(&sim).expect("d_sim");
    let d_loss = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_batch]).expect("d_loss");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_batch as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_sim.as_device_ptr(),
                d_loss.as_device_ptr(),
                n_batch as u32,
            ),
        )
        .expect("launch contrastive_loss");
    stream.synchronize().expect("sync");

    let mut gpu_loss = vec![0.0_f32; n_batch];
    d_loss.copy_to_host(&mut gpu_loss).expect("copy loss");

    let (rel, abs) = worst_diff(&gpu_loss, &cpu_loss);
    // Tolerance: ex2.approx (~2 ULP) + lg2.approx (~2 ULP) composition.
    // A missing LOG2E would cause ~50 % error — caught by 5e-4 bound.
    for k in 0..n_batch {
        assert!(
            close(gpu_loss[k], cpu_loss[k], 5e-4_f32, 1e-6_f32),
            "contrastive_loss[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_loss[k],
            cpu_loss[k]
        );
    }
}

// ===========================================================================
// 4. roi_align  —  CRATE ORACLE (crate::detection::roi_align::roi_align)
// ===========================================================================

/// Verify `roi_align` kernel against the crate's CPU RoI Align reference.
///
/// Uses `sampling_ratio=2` so `ratio² = 4`, and `rcp.approx.f32(4.0) = 0.25`
/// exactly — eliminating any `rcp.approx` error from the comparison and making
/// the tolerance analysis purely about bilinear interpolation rounding.
#[test]
fn roi_align_matches_cpu() {
    use crate::detection::roi_align::roi_align as cpu_roi_align;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_chans = 2_usize;
    let feat_h = 8_usize;
    let feat_w = 8_usize;
    let n_rois = 2_usize;
    let pooled_h = 3_usize;
    let pooled_w = 3_usize;
    let sampling_ratio = 2_usize; // ratio² = 4 → rcp(4.0) = 0.25 exactly

    let n_feat = n_chans * feat_h * feat_w;
    let n_out = n_rois * n_chans * pooled_h * pooled_w;

    let mut rng = LcgRng::new(0xA10A_0A0A);

    // Random feature map in [0, 1].
    let feat: Vec<f32> = (0..n_feat).map(|_| rng.next_f32()).collect();

    // Two RoIs in feature-map coordinates: must have x2 > x1 and y2 > y1.
    let rois: Vec<f32> = vec![
        0.5_f32, 0.5_f32, 5.5_f32, 5.5_f32, // RoI 0
        1.0_f32, 1.0_f32, 7.0_f32, 7.0_f32, // RoI 1
    ];

    // CPU reference (crate oracle).
    let cpu_out = cpu_roi_align(
        &feat,
        n_chans,
        feat_h,
        feat_w,
        &rois,
        n_rois,
        pooled_h,
        pooled_w,
        sampling_ratio,
    )
    .expect("cpu roi_align");

    // GPU kernel.
    let ptx = crate::ptx_kernels::roi_align_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "roi_align");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_feat = DeviceBuffer::<f32>::from_host(&feat).expect("d_feat");
    let d_rois = DeviceBuffer::<f32>::from_host(&rois).expect("d_rois");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_out]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_out as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_feat.as_device_ptr(),
                d_rois.as_device_ptr(),
                d_out.as_device_ptr(),
                n_rois as u32,
                feat_h as u32,
                feat_w as u32,
                n_chans as u32,
                pooled_h as u32,
                pooled_w as u32,
                sampling_ratio as u32,
            ),
        )
        .expect("launch roi_align");
    stream.synchronize().expect("sync");

    let mut gpu_out = vec![0.0_f32; n_out];
    d_out.copy_to_host(&mut gpu_out).expect("copy out");

    let (rel, abs) = worst_diff(&gpu_out, &cpu_out);
    // Tolerance: bilinear FP32 divergence (~2 ULP) + rcp.approx on power-of-4
    // (= 0.25 exactly). 1e-4 relative is conservative.
    for k in 0..n_out {
        assert!(
            close(gpu_out[k], cpu_out[k], 1e-4_f32, 1e-6_f32),
            "roi_align out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 5. image_normalize  —  CRATE ORACLE (crate::augment::normalize::normalize_chw)
// ===========================================================================

/// Verify that the in-place `image_normalize` kernel matches `normalize_chw`.
///
/// The kernel writes back to `p_img` (same pointer it reads from). The test
/// passes the original image in a DeviceBuffer, runs the kernel, then copies
/// the *modified* buffer back and compares to the CPU `normalize_chw` output.
///
/// The GPU uses `rcp.approx.f32` for the std reciprocal; for std values in
/// [0.2, 1.5] the approximation error is < 2 ULP, well within 1e-4 relative.
#[test]
fn image_normalize_matches_cpu() {
    use crate::augment::normalize::normalize_chw;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_chans = 3_usize;
    let h = 4_usize;
    let w = 4_usize;
    let n_elems = n_chans * h * w;

    let mut rng = LcgRng::new(0x99BB_EE11);

    // Random image in [0, 1] matching typical ImageNet pre-normalised range.
    let image: Vec<f32> = (0..n_elems).map(|_| rng.next_f32()).collect();

    // Per-channel mean and std; std in [0.2, 1.5] so rcp.approx is well-behaved.
    let mean: Vec<f32> = (0..n_chans).map(|_| rng.next_f32() * 0.6 + 0.2).collect();
    let std: Vec<f32> = (0..n_chans).map(|_| rng.next_f32() * 1.3 + 0.2).collect();

    // CPU reference (crate oracle).
    let cpu_out = normalize_chw(&image, n_chans, h, w, &mean, &std).expect("cpu normalize_chw");

    // GPU kernel (in-place on d_img).
    let ptx = crate::ptx_kernels::image_normalize_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "image_normalize");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_img = DeviceBuffer::<f32>::from_host(&image).expect("d_img");
    let d_mean = DeviceBuffer::<f32>::from_host(&mean).expect("d_mean");
    let d_std = DeviceBuffer::<f32>::from_host(&std).expect("d_std");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_elems as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_img.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_std.as_device_ptr(),
                h as u32,
                w as u32,
                n_chans as u32,
            ),
        )
        .expect("launch image_normalize");
    stream.synchronize().expect("sync");

    // The kernel modifies d_img in-place; copy the modified buffer back.
    let mut gpu_out = vec![0.0_f32; n_elems];
    d_img.copy_to_host(&mut gpu_out).expect("copy img");

    let (rel, abs) = worst_diff(&gpu_out, &cpu_out);
    // Tolerance: rcp.approx.f32(std) vs exact division. For std ≥ 0.2 the
    // approximation error is ~1 ULP — well within 1e-4 relative.
    for k in 0..n_elems {
        assert!(
            close(gpu_out[k], cpu_out[k], 1e-4_f32, 1e-6_f32),
            "image_normalize out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 6. adaptive_avg_pool  —  INDEPENDENT HOST RE-DERIVATION
// ===========================================================================

/// Host re-derivation of the PTX window-bound integer formulas.
///
/// ```text
/// h_start = (oh * in_h) / out_h                    (integer floor division)
/// h_end   = ceil((oh+1) * in_h / out_h)            (via .div_ceil(), same as PTX)
/// ```
/// The GPU uses `rcp.approx.f32(n_elems)` for averaging; with in=8, out=4
/// every window has exactly 2×2=4 elements, so `rcp(4.0) = 0.25` is exact.
fn host_adaptive_avg_pool(
    inp: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    n_chans: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_chans * out_h * out_w];
    for c in 0..n_chans {
        let ch_base = c * in_h * in_w;
        for oh in 0..out_h {
            // Integer floor/ceiling formulas matching the PTX.
            let h_start = oh * in_h / out_h;
            let h_end = ((oh + 1) * in_h).div_ceil(out_h);
            for ow in 0..out_w {
                let w_start = ow * in_w / out_w;
                let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                let n_elems = (h_end - h_start) * (w_end - w_start);
                let mut acc = 0.0_f32;
                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        acc += inp[ch_base + ih * in_w + iw];
                    }
                }
                out[c * out_h * out_w + oh * out_w + ow] = acc / n_elems as f32;
            }
        }
    }
    out
}

#[test]
fn adaptive_avg_pool_matches_host_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // 2-channel 8×8 → 4×4: windows are exactly 2×2=4 elements each, so
    // rcp.approx(4.0) = 0.25 is exact — pure numerical cross-check.
    let in_h = 8_usize;
    let in_w = 8_usize;
    let out_h = 4_usize;
    let out_w = 4_usize;
    let n_chans = 2_usize;
    let n_in = n_chans * in_h * in_w;
    let n_out = n_chans * out_h * out_w;

    let mut rng = LcgRng::new(0x5A5A_0303);
    let inp: Vec<f32> = (0..n_in).map(|_| rng.next_f32()).collect();

    // CPU oracle (independent re-derivation).
    let cpu_out = host_adaptive_avg_pool(&inp, in_h, in_w, out_h, out_w, n_chans);

    // GPU kernel.
    let ptx = crate::ptx_kernels::adaptive_avg_pool_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "adaptive_avg_pool");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&inp).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_out]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_out as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                in_h as u32,
                in_w as u32,
                out_h as u32,
                out_w as u32,
                n_chans as u32,
            ),
        )
        .expect("launch adaptive_avg_pool");
    stream.synchronize().expect("sync");

    let mut gpu_out = vec![0.0_f32; n_out];
    d_out.copy_to_host(&mut gpu_out).expect("copy out");

    let (rel, abs) = worst_diff(&gpu_out, &cpu_out);
    // Tolerance: rcp.approx.f32(4) = 0.25 exactly → only FP32 sum rounding.
    for k in 0..n_out {
        assert!(
            close(gpu_out[k], cpu_out[k], 1e-4_f32, 1e-6_f32),
            "adaptive_avg_pool out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 7. focal_loss  —  CRATE ORACLE (crate::losses::focal::binary_focal_loss_one)
//    Fixed hyperparameters: α = 0.25, γ = 2.0 (embedded in PTX as hex literals)
// ===========================================================================

/// Verify `focal_loss_kernel` against `binary_focal_loss_one` with α=0.25, γ=2.
///
/// The PTX embeds α=0.25 as `0F3E800000` and computes γ=2 via squaring, which
/// are both verified by the unit tests in `ptx_kernels::tests`. This test checks
/// the full numerical pipeline — sigmoid via ex2.approx + rcp.approx, and log
/// via lg2.approx * LN2 — against the crate's numerically-stable CPU path.
///
/// Logits are restricted to [-2, 2] to stay well inside ex2.approx's accurate
/// range; the GPU's approximation errors are sub-ULP there, so 5e-4 relative
/// tolerance is conservative by several orders of magnitude.
#[test]
fn focal_loss_matches_cpu() {
    use crate::losses::focal::binary_focal_loss_one;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_elem = 64_usize;
    let alpha = 0.25_f32;
    let gamma = 2.0_f32;

    let mut rng = LcgRng::new(0xF0CA_7055);

    // Logits in [-2, 2]; labels are binary 0.0 / 1.0.
    let logits: Vec<f32> = (0..n_elem).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let labels: Vec<f32> = (0..n_elem)
        .map(|_| {
            if rng.next_f32() < 0.5 {
                1.0_f32
            } else {
                0.0_f32
            }
        })
        .collect();

    // CPU reference (crate oracle, element-wise).
    let cpu_loss: Vec<f32> = logits
        .iter()
        .zip(labels.iter())
        .map(|(&logit, &label)| {
            binary_focal_loss_one(logit, label, alpha, gamma).expect("cpu binary_focal_loss_one")
        })
        .collect();

    // GPU kernel.
    let ptx = crate::ptx_kernels::focal_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "focal_loss");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_labels = DeviceBuffer::<f32>::from_host(&labels).expect("d_labels");
    let d_loss = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_elem]).expect("d_loss");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_elem as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_labels.as_device_ptr(),
                d_loss.as_device_ptr(),
                n_elem as u32,
            ),
        )
        .expect("launch focal_loss");
    stream.synchronize().expect("sync");

    let mut gpu_loss = vec![0.0_f32; n_elem];
    d_loss.copy_to_host(&mut gpu_loss).expect("copy loss");

    let (rel, abs) = worst_diff(&gpu_loss, &cpu_loss);
    // Tolerance: ex2.approx + rcp.approx (sigmoid) + lg2.approx (log) compose
    // to at most ~4 ULP total per output — 5e-4 relative is hundreds of ULPs
    // above the expected error, but still catches a missing LOG2E (~50% error)
    // or a wrong α/γ constant by orders of magnitude.
    for k in 0..n_elem {
        assert!(
            close(gpu_loss[k], cpu_loss[k], 5e-4_f32, 1e-5_f32),
            "focal_loss[{k}] mismatch: gpu={} cpu={} logit={} label={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu_loss[k],
            cpu_loss[k],
            logits[k],
            labels[k]
        );
    }
}
