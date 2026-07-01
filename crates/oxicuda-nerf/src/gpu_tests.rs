//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies the results
//! back, and asserts numerical equivalence to a CPU reference. The launch ABI
//! mirrors the `oxicuda-snn` / `oxicuda-sparse` convention: device buffers are
//! passed as their `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust
//! scalar (`.param .u32` / `.param .f32` / `.param .u64`), in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! All seven kernels here carry a **real CPU-vs-GPU numerical-equivalence**
//! assertion — there are no hollow stubs in this crate's PTX surface.
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   `pe_kernel` → [`crate::encoding::positional::positional_encode`],
//!   `volume_render_kernel` → [`crate::rendering::volume_render::volume_render`],
//!   `hash_grid_kernel` → [`crate::encoding::hash_grid::HashGrid::query_batch`],
//!   `sh_eval_nerf_kernel` → [`crate::encoding::spherical_harmonics::ShEncoder::sh_basis`],
//!   `occupancy_update_kernel` → element-wise `density > threshold`.
//! * **Independent host re-derivation** — the kernel embeds a per-thread LCG
//!   (no standalone crate function): `ray_march_kernel` (stratified jitter),
//!   `importance_resample_kernel` (inverse-CDF pick). The host code re-implements
//!   the kernel's documented integer/float pipeline independently of the JIT
//!   PTX, so it genuinely fails on a miscompile or wrong constant/shift.
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs`)
//!
//! 1. **`volume_render_kernel` — base-2 vs base-e exponential.** The alpha
//!    compositing computed `alpha = 1 - 2^(-sigma*delta)` (`ex2.approx.f32`
//!    applied directly to `-sigma*delta`) instead of `alpha = 1 - exp(-sigma*delta)`.
//!    The kernel even carried a comment admitting the approximation. Because the
//!    opacity still saturates to ~1 on the last (infinite-delta) sample, a
//!    shape/sum check misses it; only the base-e CPU oracle catches the ~25%
//!    per-sample error. FIX: multiply the exponent by `log2(e)` before `ex2`.
//! 2. **`ray_march_kernel` — stratified jitter capped at half-stratum.** The
//!    23-bit mantissa was scaled by `2^-24`, yielding jitter in `[0, 0.5)` rather
//!    than `[0, 1)` (every sample biased toward the lower edge of its stratum).
//!    The sibling `importance_resample_kernel` uses the correct `2^-23` for the
//!    identical 23-bit extraction. FIX: scale by `2^-23`.
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

/// JIT-compile `ptx` for the live device and look up `entry`, returning a
/// launchable kernel. A `Module::from_ptx` failure means ptxas rejected the
/// PTX — a real bug, surfaced as a test panic rather than a skip.
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

/// Knuth MMIX LCG constants, matching the PTX immediates in `ray_march` /
/// `importance_resample`.
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

// ===========================================================================
// 1. pe_kernel  —  CRATE ORACLE (encoding::positional::positional_encode)
// ===========================================================================

#[test]
fn positional_encoding_matches_cpu() {
    use crate::encoding::positional::{PosEncConfig, positional_encode};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_pts = 5_usize;
    let input_dim = 3_usize;
    let n_freq = 4_usize;
    let cfg = PosEncConfig {
        n_freq,
        include_input: false,
        input_dim,
    };

    // Deterministic inputs in [-0.5, 0.5]; the largest encoding argument is
    // 2^(L-1) * pi * 0.5 ≈ 12.6 rad, well inside sin/cos.approx accuracy.
    let input: Vec<f32> = (0..n_pts * input_dim)
        .map(|k| (k as f32 * 0.137).sin() * 0.5)
        .collect();

    let cpu = positional_encode(&input, &cfg).expect("cpu positional_encode");

    let ptx = crate::ptx_kernels::positional_encoding_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pe_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let out_len = n_pts * n_freq * 2 * input_dim;
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");

    let total = (n_pts * n_freq * input_dim) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n_pts as u32,
                n_freq as u32,
                input_dim as u32,
            ),
        )
        .expect("launch pe_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    let (rel, abs) = worst_diff(&gpu, &cpu);
    for k in 0..out_len {
        assert!(
            close(gpu[k], cpu[k], 1e-3, 1e-3),
            "pe[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            cpu[k]
        );
    }
}

// ===========================================================================
// 2. volume_render_kernel  —  CRATE ORACLE (rendering::volume_render)
//    Validates the base-2 → base-e exponential FIX.
// ===========================================================================

#[test]
fn volume_render_matches_cpu() {
    use crate::rendering::volume_render::volume_render;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rays = 3_usize;
    let n_samples = 8_usize;

    // Moderate densities (alpha ~0.18 per sample) so transmittance never drops
    // below the 1e-4 early-termination floor before the final sample, and the
    // base-2 vs base-e error (~25% per alpha) shows up clearly in the result.
    let mut sigma = vec![0.0_f32; n_rays * n_samples];
    let mut color = vec![0.0_f32; n_rays * n_samples * 3];
    let mut t_vals = vec![0.0_f32; n_rays * n_samples];
    for r in 0..n_rays {
        for s in 0..n_samples {
            let idx = r * n_samples + s;
            sigma[idx] = 0.25 + 0.30 * ((idx as f32 * 0.31).sin() * 0.5 + 0.5);
            // Strictly increasing t along each ray (delta > 0 like real sampling).
            t_vals[idx] = 0.1 + s as f32 * 0.5 + r as f32 * 0.05;
            color[idx * 3] = (idx as f32 * 0.13).sin() * 0.5 + 0.5;
            color[idx * 3 + 1] = (idx as f32 * 0.27).cos() * 0.5 + 0.5;
            color[idx * 3 + 2] = (idx as f32 * 0.07).sin() * 0.5 + 0.5;
        }
    }

    // CPU reference, per ray.
    let mut rgb_cpu = vec![0.0_f32; n_rays * 3];
    let mut depth_cpu = vec![0.0_f32; n_rays];
    let mut opacity_cpu = vec![0.0_f32; n_rays];
    for r in 0..n_rays {
        let s0 = r * n_samples;
        let c0 = r * n_samples * 3;
        let res = volume_render(
            &sigma[s0..s0 + n_samples],
            &color[c0..c0 + n_samples * 3],
            &t_vals[s0..s0 + n_samples],
        )
        .expect("cpu volume_render");
        rgb_cpu[r * 3] = res.rgb[0];
        rgb_cpu[r * 3 + 1] = res.rgb[1];
        rgb_cpu[r * 3 + 2] = res.rgb[2];
        depth_cpu[r] = res.depth;
        opacity_cpu[r] = res.opacity;
    }

    let ptx = crate::ptx_kernels::volume_render_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "volume_render_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sigma = DeviceBuffer::<f32>::from_host(&sigma).expect("d_sigma");
    let d_color = DeviceBuffer::<f32>::from_host(&color).expect("d_color");
    let d_t = DeviceBuffer::<f32>::from_host(&t_vals).expect("d_t");
    let d_rgb = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rays * 3]).expect("d_rgb");
    let d_depth = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rays]).expect("d_depth");
    let d_opacity = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rays]).expect("d_opacity");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_rays as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_sigma.as_device_ptr(),
                d_color.as_device_ptr(),
                d_t.as_device_ptr(),
                d_rgb.as_device_ptr(),
                d_depth.as_device_ptr(),
                d_opacity.as_device_ptr(),
                n_rays as u32,
                n_samples as u32,
            ),
        )
        .expect("launch volume_render_kernel");
    stream.synchronize().expect("sync");

    let mut rgb_gpu = vec![0.0_f32; n_rays * 3];
    let mut depth_gpu = vec![0.0_f32; n_rays];
    let mut opacity_gpu = vec![0.0_f32; n_rays];
    d_rgb.copy_to_host(&mut rgb_gpu).expect("copy rgb");
    d_depth.copy_to_host(&mut depth_gpu).expect("copy depth");
    d_opacity
        .copy_to_host(&mut opacity_gpu)
        .expect("copy opacity");

    let (rel, abs) = worst_diff(&rgb_gpu, &rgb_cpu);
    for k in 0..n_rays * 3 {
        assert!(
            close(rgb_gpu[k], rgb_cpu[k], 3e-3, 2e-3),
            "volume_render rgb[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            rgb_gpu[k],
            rgb_cpu[k]
        );
    }
    for r in 0..n_rays {
        assert!(
            close(depth_gpu[r], depth_cpu[r], 3e-3, 3e-3),
            "volume_render depth[{r}] mismatch: gpu={} cpu={}",
            depth_gpu[r],
            depth_cpu[r]
        );
        assert!(
            close(opacity_gpu[r], opacity_cpu[r], 3e-3, 2e-3),
            "volume_render opacity[{r}] mismatch: gpu={} cpu={}",
            opacity_gpu[r],
            opacity_cpu[r]
        );
    }
}

// ===========================================================================
// 3. hash_grid_kernel  —  CRATE ORACLE (encoding::hash_grid::HashGrid)
// ===========================================================================

#[test]
fn hash_grid_matches_cpu() {
    use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
    use crate::handle::LcgRng;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let cfg = HashGridConfig {
        n_levels: 3,
        n_features_per_level: 2,
        log2_hashmap_size: 6, // T = 64
        base_resolution: 4,
        max_resolution: 16,
    };
    let mut rng = LcgRng::new(0x4E_5E_5F);
    let mut grid = HashGrid::new(cfg.clone(), &mut rng).expect("grid");

    // Overwrite the tiny U(-1e-4, 1e-4) init with a deterministic O(1) pattern so
    // the comparison is discriminating (a sign/index error in the hash, the
    // table stride, or the trilinear weights would produce a large mismatch).
    for (i, v) in grid.data.iter_mut().enumerate() {
        *v = (i as f32 * 0.123).sin() - 0.5 * (i as f32 * 0.047).cos();
    }

    let n_pts = 4_usize;
    // Interior query coordinates (avoid the exact 0/1 grid edges).
    let xyz: Vec<f32> = vec![
        0.13, 0.71, 0.42, //
        0.55, 0.08, 0.93, //
        0.37, 0.62, 0.18, //
        0.84, 0.49, 0.26, //
    ];

    let cpu = grid.query_batch(&xyz, n_pts).expect("cpu query_batch");

    let level_res: Vec<u32> = grid.level_resolutions().iter().map(|&n| n as u32).collect();

    let ptx = crate::ptx_kernels::hash_grid_lookup_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hash_grid_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let out_len = n_pts * grid.output_dim();
    let d_xyz = DeviceBuffer::<f32>::from_host(&xyz).expect("d_xyz");
    let d_data = DeviceBuffer::<f32>::from_host(&grid.data).expect("d_data");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");
    let d_res = DeviceBuffer::<u32>::from_host(&level_res).expect("d_res");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_pts as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_xyz.as_device_ptr(),
                d_data.as_device_ptr(),
                d_out.as_device_ptr(),
                d_res.as_device_ptr(),
                n_pts as u32,
                cfg.n_levels as u32,
                cfg.n_features_per_level as u32,
                cfg.log2_hashmap_size as u32,
            ),
        )
        .expect("launch hash_grid_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    let (rel, abs) = worst_diff(&gpu, &cpu);
    for k in 0..out_len {
        assert!(
            close(gpu[k], cpu[k], 2e-3, 1e-4),
            "hash_grid out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            cpu[k]
        );
    }
}

// ===========================================================================
// 4. ray_march_kernel  —  INDEPENDENT HOST RE-DERIVATION (stratified jitter)
//    Validates the jitter-range FIX ([0,1) not [0,0.5)).
// ===========================================================================

/// Re-derive one stratified sample position exactly as the (fixed) PTX does:
/// one LCG step on `tid ^ seed`, top-23-bit mantissa scaled by 2^-23 → jitter in
/// `[0, 1)`, then `t = near + (sample + jitter) / n_samples * (far - near)`.
fn ray_march_host(tid: u32, sample: u32, n_samples: u32, near: f32, far: f32, seed: u64) -> f32 {
    let mut state = (tid as u64) ^ seed;
    state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    let bits = ((state >> 41) & 0x7F_FF_FF) as u32; // 23-bit mantissa
    let jitter = (bits as f32) * (1.0_f32 / 8_388_608.0_f32); // 2^-23 → [0, 1)
    let span = far - near;
    let mut v = sample as f32 + jitter;
    v /= n_samples as f32;
    v *= span;
    v + near
}

#[test]
fn ray_march_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rays = 4_usize;
    let n_samples = 8_usize;
    let seed = 0x0BAD_F00D_1234_5678_u64;

    let t_near: Vec<f32> = (0..n_rays).map(|r| 0.2 + r as f32 * 0.1).collect();
    let t_far: Vec<f32> = (0..n_rays).map(|r| 3.0 + r as f32 * 0.5).collect();

    let total = n_rays * n_samples;
    let mut host = vec![0.0_f32; total];
    for (idx, h) in host.iter_mut().enumerate() {
        let ray = idx / n_samples;
        let sample = idx % n_samples;
        *h = ray_march_host(
            idx as u32,
            sample as u32,
            n_samples as u32,
            t_near[ray],
            t_far[ray],
            seed,
        );
    }

    let ptx = crate::ptx_kernels::ray_march_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ray_march_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_near = DeviceBuffer::<f32>::from_host(&t_near).expect("d_near");
    let d_far = DeviceBuffer::<f32>::from_host(&t_far).expect("d_far");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_near.as_device_ptr(),
                d_far.as_device_ptr(),
                d_out.as_device_ptr(),
                n_rays as u32,
                n_samples as u32,
                seed,
            ),
        )
        .expect("launch ray_march_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // Bit-for-bit (within FP32 tol) match to the independent host re-derivation.
    let (rel, abs) = worst_diff(&gpu, &host);
    for (idx, (&g, &h)) in gpu.iter().zip(host.iter()).enumerate() {
        assert!(
            close(g, h, 1e-5, 1e-4),
            "ray_march t[{idx}] mismatch: gpu={g} host={h} (worst rel={rel:e} abs={abs:e})"
        );
    }

    // Discriminating jitter-range check: recover jitter = (t-near)/span*N - sample.
    // The buggy 2^-24 scale caps jitter at <0.5; the fix must let it reach the
    // upper half of a stratum, while never reaching or exceeding 1.0.
    let mut max_jitter = 0.0_f32;
    for (idx, &g) in gpu.iter().enumerate() {
        let ray = idx / n_samples;
        let sample = (idx % n_samples) as f32;
        let span = t_far[ray] - t_near[ray];
        let jitter = (g - t_near[ray]) / span * n_samples as f32 - sample;
        assert!(
            (-1e-4..1.0 + 1e-4).contains(&jitter),
            "ray_march jitter[{idx}] = {jitter} outside [0, 1)"
        );
        if jitter > max_jitter {
            max_jitter = jitter;
        }
    }
    assert!(
        max_jitter > 0.55,
        "ray_march jitter never exceeded 0.55 (max={max_jitter}); the [0,0.5) \
         half-stratum scaling bug appears un-fixed"
    );
}

// ===========================================================================
// 5. sh_eval_nerf_kernel  —  CRATE ORACLE (spherical_harmonics::ShEncoder)
// ===========================================================================

#[test]
fn sh_to_rgb_matches_cpu() {
    use crate::encoding::spherical_harmonics::ShEncoder;
    use crate::handle::LcgRng;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rays = 4_usize;
    let n_coeffs = 16_usize; // degree 3
    let n_channels = 3_usize;

    // Unit-length directions: the kernel evaluates the SH basis on the raw
    // direction (no normalisation), so we feed unit vectors and use the crate's
    // own `sh_basis` on the identical components as the oracle.
    let raw_dirs: [[f32; 3]; 4] = [
        [0.3, -0.6, 0.74],
        [-0.5, 0.2, 0.84],
        [0.1, 0.95, -0.29],
        [-0.7, -0.4, 0.59],
    ];
    let mut dirs = vec![0.0_f32; n_rays * 3];
    let mut unit = vec![[0.0_f32; 3]; n_rays];
    for r in 0..n_rays {
        let d = raw_dirs[r];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        for c in 0..3 {
            unit[r][c] = d[c] / len;
            dirs[r * 3 + c] = d[c] / len;
        }
    }

    let mut rng = LcgRng::new(0x05C0_FFEE);
    let coeffs: Vec<f32> = (0..n_rays * n_coeffs * n_channels)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // CPU oracle: basis on the same (already unit) components, then the identical
    // interleaved [coeff*3 + channel] dot product the kernel performs.
    let mut cpu = vec![0.0_f32; n_rays * n_channels];
    for r in 0..n_rays {
        let basis = ShEncoder::sh_basis(unit[r][0], unit[r][1], unit[r][2], 3).expect("sh_basis");
        for c in 0..n_channels {
            let mut acc = 0.0_f32;
            for (i, &b) in basis.iter().enumerate() {
                acc += coeffs[(r * n_coeffs + i) * n_channels + c] * b;
            }
            cpu[r * n_channels + c] = acc;
        }
    }

    let ptx = crate::ptx_kernels::sh_to_rgb_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sh_eval_nerf_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dir = DeviceBuffer::<f32>::from_host(&dirs).expect("d_dir");
    let d_coeff = DeviceBuffer::<f32>::from_host(&coeffs).expect("d_coeff");
    let d_rgb = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rays * 3]).expect("d_rgb");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_rays as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_dir.as_device_ptr(),
                d_coeff.as_device_ptr(),
                d_rgb.as_device_ptr(),
                n_rays as u32,
            ),
        )
        .expect("launch sh_eval_nerf_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_rays * 3];
    d_rgb.copy_to_host(&mut gpu).expect("copy rgb");

    let (rel, abs) = worst_diff(&gpu, &cpu);
    for k in 0..n_rays * 3 {
        assert!(
            close(gpu[k], cpu[k], 2e-3, 1e-4),
            "sh rgb[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            cpu[k]
        );
    }
}

// ===========================================================================
// 6. occupancy_update_kernel  —  CRATE-EQUIVALENT ORACLE (density > threshold)
// ===========================================================================

#[test]
fn occupancy_update_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_voxels = 64_usize;
    let threshold = 0.5_f32;
    // Deterministic densities, a mix above and below the threshold but always at
    // least 0.05 away from it so the `>` comparison is never on a FP knife-edge.
    let density: Vec<f32> = (0..n_voxels)
        .map(|i| {
            let s = (i as f32 * 0.211).sin();
            if s >= 0.0 {
                0.55 + 0.4 * s // (0.5, 0.95]
            } else {
                0.45 + 0.4 * s // [0.05, 0.45)
            }
        })
        .collect();
    for &d in &density {
        assert!(
            (d - threshold).abs() > 1e-2,
            "test setup: density on knife-edge"
        );
    }

    let cpu: Vec<u8> = density
        .iter()
        .map(|&d| if d > threshold { 1_u8 } else { 0_u8 })
        .collect();

    let ptx = crate::ptx_kernels::occupancy_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "occupancy_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_density = DeviceBuffer::<f32>::from_host(&density).expect("d_density");
    let d_occ = DeviceBuffer::<u8>::from_host(&vec![0_u8; n_voxels]).expect("d_occ");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_voxels as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_density.as_device_ptr(),
                d_occ.as_device_ptr(),
                threshold,
                n_voxels as u32,
            ),
        )
        .expect("launch occupancy_update_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0_u8; n_voxels];
    d_occ.copy_to_host(&mut gpu).expect("copy occ");

    for k in 0..n_voxels {
        assert_eq!(
            gpu[k], cpu[k],
            "occupancy[{k}] mismatch: gpu={} cpu={} (density={})",
            gpu[k], cpu[k], density[k]
        );
    }
}

// ===========================================================================
// 7. importance_resample_kernel  —  INDEPENDENT HOST RE-DERIVATION (inverse CDF)
// ===========================================================================

/// Re-derive the kernel's per-thread uniform `u ∈ [0,1)` exactly: one LCG step
/// on `fine_idx ^ seed`, top-23-bit mantissa scaled by 2^-23.
fn irs_uniform(fine_idx: u32, seed: u64) -> f32 {
    let mut state = (fine_idx as u64) ^ seed;
    state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    let bits = ((state >> 41) & 0x7F_FF_FF) as u32;
    (bits as f32) * (1.0_f32 / 8_388_608.0_f32)
}

#[test]
fn importance_resample_matches_host_inverse_cdf() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_coarse = 6_usize;
    let n_fine = 12_usize;
    let seed = 0xFEED_FACE_C0DE_0001_u64;
    let eps = 1e-5_f32;

    // Well-separated weights and strictly increasing coarse positions so the
    // inverse-CDF pick is unambiguous (no FP knife-edge between strata).
    let weights: Vec<f32> = vec![3.0, 1.0, 4.0, 1.5, 2.5, 2.0];
    let coarse_t: Vec<f32> = vec![0.10, 0.35, 0.55, 0.80, 1.20, 1.75];

    // Host inverse-CDF, mirroring the exact PTX float pipeline.
    let total = {
        let mut acc = 0.0_f32;
        for &w in &weights {
            acc += w.max(0.0) + eps;
        }
        acc
    };
    let mut host = vec![0.0_f32; n_fine];
    for (f, h) in host.iter_mut().enumerate() {
        let u = irs_uniform(f as u32, seed);
        let target = u * total;
        let mut accum = 0.0_f32;
        let mut idx = n_coarse - 1; // fallback: last coarse t
        for (r, &w) in weights.iter().enumerate() {
            accum += w.max(0.0) + eps;
            if accum >= target {
                idx = r;
                break;
            }
        }
        *h = coarse_t[idx];
    }

    let ptx = crate::ptx_kernels::importance_resample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "importance_resample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_coarse = DeviceBuffer::<f32>::from_host(&coarse_t).expect("d_coarse");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_fine = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_fine]).expect("d_fine");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_fine as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_coarse.as_device_ptr(),
                d_w.as_device_ptr(),
                d_fine.as_device_ptr(),
                n_coarse as u32,
                n_fine as u32,
                seed,
            ),
        )
        .expect("launch importance_resample_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_fine];
    d_fine.copy_to_host(&mut gpu).expect("copy fine");

    // Every output must be one of the coarse positions, and exactly the one the
    // independent host inverse-CDF selected.
    for (f, (&g, &h)) in gpu.iter().zip(host.iter()).enumerate() {
        assert!(
            coarse_t.iter().any(|&t| (t - g).abs() < 1e-6),
            "importance_resample fine[{f}] = {g} is not a coarse position"
        );
        assert!(
            (g - h).abs() < 1e-4,
            "importance_resample fine[{f}] mismatch: gpu={g} host={h}"
        );
    }
}
