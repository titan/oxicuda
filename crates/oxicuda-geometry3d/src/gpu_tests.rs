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
//!   CPU function the kernel mirrors:
//!   `project_kernel` ↔ [`crate::gaussian::project::project_gaussian`]
//!   (screen `xy`, `depth`, `valid`, and the full EWA `cov2d`); and
//!   `sh_eval_kernel` ↔ [`crate::gaussian::gaussian::Gaussian3d::sh_color`]
//!   (the full 9-term L=0..2 RGB evaluation).
//! * **Independent host re-derivation** — the op has no standalone `pub fn`, so
//!   the oracle is an independent Rust re-implementation of the kernel's
//!   *documented* arithmetic, computed independently of the JIT-compiled PTX:
//!   `fps_kernel` (squared-distance min update), `ball_query_kernel`
//!   (radius-bounded neighbour list), `gather_kernel` (indexed gather),
//!   `voxelize_kernel` (voxel scatter-add of features + counts), and
//!   `chamfer_kernel` (mean nearest-neighbour squared distance).
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs` history)
//!
//! * `project_kernel`: an `add.u64 %rd9, %rd7, %r4;` mixed a 64-bit and a
//!   32-bit register — ptxas rejected the whole module ("invalid PTX"). The
//!   redundant line was removed (the following `cvt.u64.u32` + `add.u64`
//!   already widen the index correctly). INVALID-PTX class.
//! * `sh_eval_kernel`: `mul.f32 %f32, %f13, %f20;` used `%f32`, one past the
//!   `.reg .f32 %f<32>` declaration (valid indices are `%f0..%f31`) — ptxas
//!   rejected the module. The line was dead (the next two instructions
//!   recompute the value into `%f27`), so it was removed. INVALID-PTX class.
//! * `sh_eval_kernel`: the first `L=1` term in all three colour channels was
//!   `coeff * Y11` with the **direction component omitted** (the other two
//!   terms correctly multiply by `dy` and `dz`; the code comment even read
//!   "using x approx"). Fixed to `coeff * Y11 * dx`, matching the L=1 basis
//!   `{Y11·x, Y11·y, Y11·z}`. WRONG-MATH class.
//!
//! ## Completeness notes
//!
//! * `project_kernel` now emits all four documented outputs — `out_xy`,
//!   `out_cov2d`, `out_depth`, `out_valid`. The 2D covariance is the full EWA
//!   Jacobian form `Σ_2d = J·R·Σ_3d·Rᵀ·Jᵀ + 0.3·I`, validated against
//!   `project_gaussian` with a genuinely non-identity `Σ_3d` per Gaussian.
//! * `sh_eval_kernel` now evaluates the full L=0..2 basis (all nine
//!   coefficients per channel, with the `m=0` term `Y20·(3·dz²−1)` and the four
//!   higher `L=2` terms `Y21A·xz`, `Y21A·yz`, `Y22A·(x²−y²)`, `Y22A·xy`),
//!   validated against the crate's full
//!   [`crate::gaussian::gaussian::Gaussian3d::sh_color`].
//!
//! Every test skips (returns early) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.

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

/// Squared Euclidean distance between two 3-vectors at `a[ai*3..]`, `b[bi*3..]`.
fn sq_dist3(a: &[f32], ai: usize, b: &[f32], bi: usize) -> f32 {
    let dx = a[ai * 3] - b[bi * 3];
    let dy = a[ai * 3 + 1] - b[bi * 3 + 1];
    let dz = a[ai * 3 + 2] - b[bi * 3 + 2];
    dx * dx + dy * dy + dz * dz
}

// ===========================================================================
// 1. fps_kernel  —  HOST RE-DERIVATION (squared-distance min update)
// ===========================================================================
//
// Each thread (grid-stride) loads point `i`, computes its squared distance to
// the single last-selected point, and writes `out_dist[i] = min(out_dist[i],
// sq_dist)`. This is one distance-update step of farthest point sampling; the
// max-reduce that picks the next centroid is performed host-side in the crate.

#[test]
fn fps_distance_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200_usize;
    let mut rng = LcgRng::new(0xF95_0001);
    let points: Vec<f32> = (0..n * 3).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let last: Vec<f32> = vec![0.3_f32, -0.4, 0.7];
    // Mixed initial distances: some larger than sq_dist (→ updated), some
    // smaller (→ kept), so the `min` is exercised in both directions.
    let dist_init: Vec<f32> = (0..n).map(|_| rng.next_f32() * 6.0).collect();

    // ---- host reference ----
    let mut dist_host = dist_init.clone();
    for (i, slot) in dist_host.iter_mut().enumerate() {
        let sq = sq_dist3(&points, i, &last, 0);
        *slot = slot.min(sq);
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::farthest_point_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fps_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_dist = DeviceBuffer::<f32>::from_host(&dist_init).expect("d_dist");
    let d_last = DeviceBuffer::<f32>::from_host(&last).expect("d_last");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_dist.as_device_ptr(),
                d_last.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch fps_kernel");
    stream.synchronize().expect("sync");

    let mut dist_gpu = vec![0.0_f32; n];
    d_dist.copy_to_host(&mut dist_gpu).expect("copy dist");

    let (rel, abs) = worst_diff(&dist_gpu, &dist_host);
    for i in 0..n {
        assert!(
            close(dist_gpu[i], dist_host[i], 1e-5, 1e-6),
            "fps dist[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            dist_gpu[i],
            dist_host[i]
        );
    }
}

// ===========================================================================
// 2. ball_query_kernel  —  HOST RE-DERIVATION (radius-bounded neighbour list)
// ===========================================================================
//
// One thread per query. For each query the kernel scans points in ascending
// index order, and for every point with `sq_dist < radius_sq` (and while the
// slot count `< k_max`) writes the point index into
// `out_idx[query*k_max + slot]` and increments the per-query count. Unfilled
// slots are left at the host-initialised sentinel `0xFFFF_FFFF`.

#[test]
fn ball_query_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let np = 10_usize;
    let nq = 4_usize;
    let k_max = 4_u32;
    // Points evenly spaced on the x-axis: point i = (i, 0, 0).
    let mut points = vec![0.0_f32; np * 3];
    for i in 0..np {
        points[i * 3] = i as f32;
    }
    // Queries chosen so neighbour counts are 2,2,2,3 and never hit a distance
    // exactly equal to the radius (so the strict `<` decision is unambiguous on
    // both CPU and GPU): radius² = 2.0.
    let queries: Vec<f32> = vec![
        0.0, 0.0, 0.0, // q0 → points 0,1
        4.5, 0.0, 0.0, // q1 → points 4,5
        9.0, 0.0, 0.0, // q2 → points 8,9
        2.0, 0.0, 0.0, // q3 → points 1,2,3
    ];
    let radius_sq = 2.0_f32;

    // ---- host reference ----
    let mut idx_host = vec![u32::MAX; nq * k_max as usize];
    let mut cnt_host = vec![0_u32; nq];
    for q in 0..nq {
        let mut slot = 0_u32;
        for p in 0..np {
            if slot >= k_max {
                break;
            }
            if sq_dist3(&queries, q, &points, p) < radius_sq {
                idx_host[q * k_max as usize + slot as usize] = p as u32;
                slot += 1;
            }
        }
        cnt_host[q] = slot;
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::ball_query_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ball_query_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_queries = DeviceBuffer::<f32>::from_host(&queries).expect("d_queries");
    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_idx =
        DeviceBuffer::<u32>::from_host(&vec![u32::MAX; nq * k_max as usize]).expect("d_idx");
    let d_cnt = DeviceBuffer::<u32>::from_host(&vec![0_u32; nq]).expect("d_cnt");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(nq as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_queries.as_device_ptr(),
                d_points.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_cnt.as_device_ptr(),
                radius_sq,
                k_max,
                nq as u32,
                np as u32,
            ),
        )
        .expect("launch ball_query_kernel");
    stream.synchronize().expect("sync");

    let mut idx_gpu = vec![0_u32; nq * k_max as usize];
    let mut cnt_gpu = vec![0_u32; nq];
    d_idx.copy_to_host(&mut idx_gpu).expect("copy idx");
    d_cnt.copy_to_host(&mut cnt_gpu).expect("copy cnt");

    for q in 0..nq {
        assert_eq!(
            cnt_gpu[q], cnt_host[q],
            "ball_query count[{q}] mismatch: gpu={} host={}",
            cnt_gpu[q], cnt_host[q]
        );
    }
    for k in 0..nq * k_max as usize {
        assert_eq!(
            idx_gpu[k], idx_host[k],
            "ball_query idx[{k}] mismatch: gpu={} host={}",
            idx_gpu[k], idx_host[k]
        );
    }
}

// ===========================================================================
// 3. gather_kernel  —  HOST RE-DERIVATION (indexed feature gather)
// ===========================================================================
//
// `tid` maps to `(k_i = tid / c, c_j = tid % c)`; for `k_i < k` the kernel
// writes `out[k_i*c + c_j] = in[idx[k_i]*c + c_j]`.

#[test]
fn gather_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 6_usize;
    let c = 3_usize;
    let k = 4_usize;
    let mut rng = LcgRng::new(0x6A78_0003);
    let input: Vec<f32> = (0..n * c).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let idx: Vec<u32> = vec![5, 0, 3, 2];

    // ---- host reference ----
    let mut out_host = vec![0.0_f32; k * c];
    for (k_i, &src_row) in idx.iter().enumerate() {
        for c_j in 0..c {
            out_host[k_i * c + c_j] = input[src_row as usize * c + c_j];
        }
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::gather_points_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gather_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_idx = DeviceBuffer::<u32>::from_host(&idx).expect("d_idx");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k * c]).expect("d_out");

    let block = 256_u32;
    let total = (k * c) as u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_out.as_device_ptr(),
                c as u32,
                k as u32,
            ),
        )
        .expect("launch gather_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; k * c];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for j in 0..k * c {
        assert_eq!(
            out_gpu[j].to_bits(),
            out_host[j].to_bits(),
            "gather out[{j}] mismatch: gpu={} host={}",
            out_gpu[j],
            out_host[j]
        );
    }
}

// ===========================================================================
// 4. voxelize_kernel  —  HOST RE-DERIVATION (voxel scatter-add)
// ===========================================================================
//
// Each point computes its voxel `(ix,iy,iz)` via `floor((p - origin)/size)`,
// bounds-checks against the grid dims, then `atom.add`s its features into
// `vox_feat[vox*c + ch]` and `atom.add`s 1 into `vox_cnt[vox]`. Points are
// placed at distinct voxel centres so the float atomics see no contention and
// the accumulated sums are exact regardless of execution order.

#[test]
fn voxelize_scatter_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize;
    let c = 2_usize;
    let (gx, gy, gz) = (4_u32, 4_u32, 4_u32);
    let v_size = 1.0_f32;
    let (ox, oy, oz) = (0.0_f32, 0.0_f32, 0.0_f32);
    let n_vox = (gx * gy * gz) as usize;

    // One point per distinct voxel: voxel (k%4, k/4, 0), centre at +0.5.
    let mut points = vec![0.0_f32; n * 3];
    for k in 0..n {
        points[k * 3] = (k % 4) as f32 + 0.5;
        points[k * 3 + 1] = (k / 4) as f32 + 0.5;
        points[k * 3 + 2] = 0.5;
    }
    let mut rng = LcgRng::new(0x402E_0004);
    let features: Vec<f32> = (0..n * c).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- host reference ----
    let mut feat_host = vec![0.0_f32; n_vox * c];
    let mut cnt_host = vec![0_u32; n_vox];
    for k in 0..n {
        let ix = ((points[k * 3] - ox) / v_size).floor() as i32;
        let iy = ((points[k * 3 + 1] - oy) / v_size).floor() as i32;
        let iz = ((points[k * 3 + 2] - oz) / v_size).floor() as i32;
        if ix < 0 || iy < 0 || iz < 0 {
            continue;
        }
        let (ux, uy, uz) = (ix as u32, iy as u32, iz as u32);
        if ux >= gx || uy >= gy || uz >= gz {
            continue;
        }
        let vox = (ux * gy * gz + uy * gz + uz) as usize;
        cnt_host[vox] += 1;
        for ch in 0..c {
            feat_host[vox * c + ch] += features[k * c + ch];
        }
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::voxelize_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "voxelize_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_feat = DeviceBuffer::<f32>::from_host(&features).expect("d_feat");
    let d_vox_feat = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_vox * c]).expect("d_vox_feat");
    let d_vox_cnt = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_vox]).expect("d_vox_cnt");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_feat.as_device_ptr(),
                d_vox_feat.as_device_ptr(),
                d_vox_cnt.as_device_ptr(),
                v_size,
                ox,
                oy,
                oz,
                gx,
                gy,
                gz,
                c as u32,
                n as u32,
            ),
        )
        .expect("launch voxelize_kernel");
    stream.synchronize().expect("sync");

    let mut feat_gpu = vec![0.0_f32; n_vox * c];
    let mut cnt_gpu = vec![0_u32; n_vox];
    d_vox_feat
        .copy_to_host(&mut feat_gpu)
        .expect("copy vox_feat");
    d_vox_cnt.copy_to_host(&mut cnt_gpu).expect("copy vox_cnt");

    for v in 0..n_vox {
        assert_eq!(
            cnt_gpu[v], cnt_host[v],
            "voxelize count[{v}] mismatch: gpu={} host={}",
            cnt_gpu[v], cnt_host[v]
        );
    }
    let (rel, abs) = worst_diff(&feat_gpu, &feat_host);
    for j in 0..n_vox * c {
        assert!(
            close(feat_gpu[j], feat_host[j], 1e-5, 1e-6),
            "voxelize feat[{j}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            feat_gpu[j],
            feat_host[j]
        );
    }
}

// ===========================================================================
// 5. chamfer_kernel  —  HOST RE-DERIVATION (mean nearest-neighbour sq dist)
// ===========================================================================
//
// One thread per point in A. Each finds `min_b sq_dist(a, b)`, multiplies by
// `inv_na`, and `atom.add`s it into the single scalar output, so the result is
// `(1/na) · Σ_a min_b ‖a−b‖²` — the one-directional Chamfer term A→B.

#[test]
fn chamfer_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let na = 6_usize;
    let nb = 5_usize;
    let mut rng = LcgRng::new(0xC4A_0005);
    let a: Vec<f32> = (0..na * 3).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let b: Vec<f32> = (0..nb * 3).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let inv_na = 1.0_f32 / na as f32;

    // ---- host reference ----
    let mut acc = 0.0_f32;
    for i in 0..na {
        let mut best = f32::INFINITY;
        for j in 0..nb {
            best = best.min(sq_dist3(&a, i, &b, j));
        }
        acc += best * inv_na;
    }
    let host = acc;

    // ---- GPU ----
    let ptx = crate::ptx_kernels::chamfer_distance_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "chamfer_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(na as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                na as u32,
                nb as u32,
                inv_na,
            ),
        )
        .expect("launch chamfer_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert!(
        close(out_gpu[0], host, 1e-4, 1e-4),
        "chamfer mismatch: gpu={} host={}",
        out_gpu[0],
        host
    );
}

// ===========================================================================
// 6. project_kernel  —  CRATE ORACLE (gaussian::project::project_gaussian)
// ===========================================================================
//
// Validates the three outputs the PTX actually writes — screen `xy`, `depth`,
// and the `valid` flag — against the crate's `project_gaussian`. The Gaussians
// are placed comfortably in front of the camera (Z_cam > near for all), so the
// CPU oracle takes its valid branch and the screen projection is well defined.
// The kernel does NOT populate `out_cov2d`, so that output is not checked here
// (documented in the module header).

#[test]
fn project_xy_depth_cov2d_valid_matches_crate() {
    use crate::gaussian::gaussian::Gaussian3d;
    use crate::gaussian::project::{CameraIntrinsics, project_gaussian};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 16_usize;
    // A genuine rotation about the y-axis (θ = 0.3) plus a +z translation keeps
    // every Gaussian in front of the camera while exercising the full R·p + t
    // matrix-vector product (not just the diagonal).
    let theta = 0.3_f32;
    let (s, co) = (theta.sin(), theta.cos());
    let view: [f32; 12] = [
        co, 0.0, s, 0.2, // row0: R[0] | t_x
        0.0, 1.0, 0.0, -0.1, // row1: R[1] | t_y
        -s, 0.0, co, 4.0, // row2: R[2] | t_z
    ];
    let cam = CameraIntrinsics {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        near: 0.1,
    };

    let mut rng = LcgRng::new(0x9301_0006);
    // Build Gaussians with random means plus a genuinely non-identity rotation
    // and anisotropic scale, so the 3D covariance Σ fed to the kernel is a real
    // SPD matrix — this exercises the full W·Σ·Wᵀ contraction (a wrong Σ index,
    // a transposed W, or a dropped term would all show up), not merely W·Wᵀ.
    let mut gaussians: Vec<Gaussian3d> = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = [
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
        ];
        let rot = [
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
        ];
        let scale = [
            (rng.next_f32() * 0.5 + 0.5).ln(),
            (rng.next_f32() * 0.5 + 0.5).ln(),
            (rng.next_f32() * 0.5 + 0.5).ln(),
        ];
        gaussians.push(Gaussian3d {
            pos,
            rot,
            scale,
            opacity: 0.0,
            sh: vec![0.0_f32; 27],
        });
    }

    let mut means = vec![0.0_f32; n * 3];
    for (i, g) in gaussians.iter().enumerate() {
        means[i * 3] = g.pos[0];
        means[i * 3 + 1] = g.pos[1];
        means[i * 3 + 2] = g.pos[2];
    }

    // ---- crate oracle ----
    let mut xy_cpu = vec![0.0_f32; n * 2];
    let mut depth_cpu = vec![0.0_f32; n];
    let mut cov2d_cpu = vec![0.0_f32; n * 4];
    let mut valid_cpu = vec![0_u8; n];
    // Σ_3d per Gaussian, fed verbatim to the kernel's p_cov3d buffer.
    let mut cov3d = vec![0.0_f32; n * 9];
    for (i, g) in gaussians.iter().enumerate() {
        let sigma = g.covariance3d().expect("covariance3d");
        cov3d[i * 9..i * 9 + 9].copy_from_slice(&sigma);
        let pg = project_gaussian(g, &view, &cam).expect("project_gaussian");
        xy_cpu[i * 2] = pg.xy[0];
        xy_cpu[i * 2 + 1] = pg.xy[1];
        depth_cpu[i] = pg.depth;
        cov2d_cpu[i * 4..i * 4 + 4].copy_from_slice(&pg.cov2d);
        valid_cpu[i] = u8::from(pg.valid);
        assert!(
            pg.valid,
            "test setup: Gaussian {i} must be in front of camera"
        );
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::gaussian_project_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "project_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_means = DeviceBuffer::<f32>::from_host(&means).expect("d_means");
    let d_cov3d = DeviceBuffer::<f32>::from_host(&cov3d).expect("d_cov3d");
    let d_view = DeviceBuffer::<f32>::from_host(&view).expect("d_view");
    let intrinsics = vec![cam.fx, cam.fy, cam.cx, cam.cy, cam.near];
    let d_intr = DeviceBuffer::<f32>::from_host(&intrinsics).expect("d_intr");
    let d_xy = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 2]).expect("d_xy");
    let d_cov2d = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 4]).expect("d_cov2d");
    let d_depth = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_depth");
    let d_valid = DeviceBuffer::<u8>::from_host(&vec![0xFF_u8; n]).expect("d_valid");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_means.as_device_ptr(),
                d_cov3d.as_device_ptr(),
                d_view.as_device_ptr(),
                d_intr.as_device_ptr(),
                d_xy.as_device_ptr(),
                d_cov2d.as_device_ptr(),
                d_depth.as_device_ptr(),
                d_valid.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch project_kernel");
    stream.synchronize().expect("sync");

    let mut xy_gpu = vec![0.0_f32; n * 2];
    let mut depth_gpu = vec![0.0_f32; n];
    let mut cov2d_gpu = vec![0.0_f32; n * 4];
    let mut valid_gpu = vec![0_u8; n];
    d_xy.copy_to_host(&mut xy_gpu).expect("copy xy");
    d_depth.copy_to_host(&mut depth_gpu).expect("copy depth");
    d_cov2d.copy_to_host(&mut cov2d_gpu).expect("copy cov2d");
    d_valid.copy_to_host(&mut valid_gpu).expect("copy valid");

    let (rel_xy, abs_xy) = worst_diff(&xy_gpu, &xy_cpu);
    for j in 0..n * 2 {
        assert!(
            close(xy_gpu[j], xy_cpu[j], 1e-4, 1e-3),
            "project xy[{j}] mismatch: gpu={} cpu={} (worst rel={rel_xy:e} abs={abs_xy:e})",
            xy_gpu[j],
            xy_cpu[j]
        );
    }
    for i in 0..n {
        assert!(
            close(depth_gpu[i], depth_cpu[i], 1e-5, 1e-5),
            "project depth[{i}] mismatch: gpu={} cpu={}",
            depth_gpu[i],
            depth_cpu[i]
        );
        assert_eq!(
            valid_gpu[i], valid_cpu[i],
            "project valid[{i}] mismatch: gpu={} cpu={}",
            valid_gpu[i], valid_cpu[i]
        );
    }
    // Full EWA 2D covariance Σ_2d = J·R·Σ·Rᵀ·Jᵀ + 0.3·I, now emitted by the kernel.
    let (rel_cov, abs_cov) = worst_diff(&cov2d_gpu, &cov2d_cpu);
    for j in 0..n * 4 {
        assert!(
            close(cov2d_gpu[j], cov2d_cpu[j], 1e-3, 1e-2),
            "project cov2d[{j}] mismatch: gpu={} cpu={} (worst rel={rel_cov:e} abs={abs_cov:e})",
            cov2d_gpu[j],
            cov2d_cpu[j]
        );
    }
}

// ===========================================================================
// 7. sh_eval_kernel  —  CRATE ORACLE (gaussian::gaussian::Gaussian3d::sh_color)
// ===========================================================================
//
// The kernel now evaluates the FULL L=0..2 spherical-harmonics basis (all 9
// coefficients per channel), so it is validated directly against the crate's
// own `Gaussian3d::sh_color`, the reference it mirrors:
//   out[ch] = Σ_{i=0..8} sh[ch*9 + i] · basis_i(dir),
// with basis = [Y00, Y11·x, Y11·y, Y11·z, Y20·(3z²−1), Y21A·xz, Y21A·yz,
//               Y22A·(x²−y²), Y22A·xy].

#[test]
fn sh_eval_full_matches_crate() {
    use crate::gaussian::gaussian::Gaussian3d;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 32_usize;
    let mut rng = LcgRng::new(0x58E_0007);
    let sh: Vec<f32> = (0..n * 27).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let dir: Vec<f32> = (0..n * 3).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- crate oracle: full 9-term SH per channel ----
    let mut out_host = vec![0.0_f32; n * 3];
    for g in 0..n {
        let gauss = Gaussian3d {
            pos: [0.0, 0.0, 0.0],
            rot: [1.0, 0.0, 0.0, 0.0],
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
            sh: sh[g * 27..g * 27 + 27].to_vec(),
        };
        let color = gauss
            .sh_color([dir[g * 3], dir[g * 3 + 1], dir[g * 3 + 2]])
            .expect("sh_color");
        out_host[g * 3] = color[0];
        out_host[g * 3 + 1] = color[1];
        out_host[g * 3 + 2] = color[2];
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::sh_eval_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sh_eval_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sh = DeviceBuffer::<f32>::from_host(&sh).expect("d_sh");
    let d_dir = DeviceBuffer::<f32>::from_host(&dir).expect("d_dir");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 3]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_sh.as_device_ptr(),
                d_dir.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch sh_eval_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n * 3];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for j in 0..n * 3 {
        assert!(
            close(out_gpu[j], out_host[j], 1e-4, 1e-5),
            "sh_eval out[{j}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[j],
            out_host[j]
        );
    }
}
