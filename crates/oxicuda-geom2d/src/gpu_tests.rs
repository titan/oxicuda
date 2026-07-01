//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's own CPU reference (or, where the per-element op has no
//! standalone `pub fn`, an independent host re-derivation of the kernel's
//! documented arithmetic). The launch ABI mirrors the working `oxicuda-snn`
//! canary: device buffers are passed as their `CUdeviceptr` (a `.param .u64`),
//! scalars are passed as the matching Rust scalar (`.param .f64` / `.param .u32`)
//! in the kernel's declared parameter order.
//!
//! ## Kernel inventory (all 7 are LAUNCHABLE on sm_86 and VALIDATED)
//!
//! All kernels operate on `f64` and use only plain IEEE arithmetic (no `ex2` /
//! `lg2`), so the base-2 exp/log bug class does not apply here. Every kernel is
//! compared element-wise to a CPU oracle — none are hollow stubs:
//!
//! * `orientation_test_kernel`  ↔ [`crate::predicate::orientation::orient_value`]
//! * `cross_product_kernel`     ↔ `ax*by − ay*bx` (host re-derivation)
//! * `point_in_aabb_kernel`     ↔ [`crate::primitives::aabb::Aabb::contains`]
//! * `segment_intersection_kernel` ↔ proper sign-difference crossing test
//!   (host re-derivation of the kernel's exact `o1·o2 < 0 ∧ o3·o4 < 0` logic)
//! * `convex_hull_step_kernel`  ↔ `sign(orient_value)` (+1 / 0 / −1)
//! * `kd_tree_traverse_kernel`  ↔ [`crate::primitives::point::Point::distance_sq`]
//! * `polygon_area_kernel`      ↔ per-edge shoelace term `px[i]·py[j] − px[j]·py[i]`
//!
//! ## PTX bug found and fixed
//!
//! `point_in_aabb_kernel` declared `.reg .u32 %r<8>;` (which already creates
//! `%r0..%r7`) and then redundantly re-declared `.reg .u32 %r5;`. ptxas rejects
//! this on the real RTX A4000 (sm_86) with `Duplicate definition of variable
//! '%r5'`, so the kernel had never loaded on any GPU. Fix applied in
//! `ptx_kernels.rs`: removed the redundant `%r5` declaration.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;
use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

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

/// Relative-with-absolute-floor closeness test for FP64 comparisons.
fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length slices.
fn worst_diff(gpu: &[f64], cpu: &[f64]) -> (f64, f64) {
    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
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

/// JIT-compile `ptx` for the live device and look up `entry`.
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

/// Build six per-triangle coordinate arrays `(ax, ay, bx, by, cx, cy)`.
///
/// Indices 0..3 are seeded as explicit collinear / CCW / CW triangles so both
/// the raw orientation value and the sign branch (+1 / 0 / −1) are exercised;
/// the remainder are well-separated pseudo-random points whose orientation
/// magnitude is `O(1)`, far from any degenerate knife-edge.
#[allow(clippy::type_complexity)]
fn triangle_inputs(
    n: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut ax = vec![0.0_f64; n];
    let mut ay = vec![0.0_f64; n];
    let mut bx = vec![0.0_f64; n];
    let mut by = vec![0.0_f64; n];
    let mut cx = vec![0.0_f64; n];
    let mut cy = vec![0.0_f64; n];

    // Exactly-representable special cases (orientation is bit-exact 0 / +1 / −1).
    let specials: [[f64; 6]; 3] = [
        // collinear: a=(0,0) b=(1,0) c=(2,0) -> orient = 0
        [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        // CCW: a=(0,0) b=(1,0) c=(0,1) -> orient = +1
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        // CW: a=(0,0) b=(0,1) c=(1,0) -> orient = −1
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ];

    let mut rng = LcgRng::new(seed);
    for i in 0..n {
        if i < specials.len() {
            let s = specials[i];
            ax[i] = s[0];
            ay[i] = s[1];
            bx[i] = s[2];
            by[i] = s[3];
            cx[i] = s[4];
            cy[i] = s[5];
        } else {
            ax[i] = rng.next_f64() * 4.0 - 2.0;
            ay[i] = rng.next_f64() * 4.0 - 2.0;
            bx[i] = rng.next_f64() * 4.0 - 2.0;
            by[i] = rng.next_f64() * 4.0 - 2.0;
            cx[i] = rng.next_f64() * 4.0 - 2.0;
            cy[i] = rng.next_f64() * 4.0 - 2.0;
        }
    }
    (ax, ay, bx, by, cx, cy)
}

// ===========================================================================
// 1. orientation_test  —  CRATE ORACLE (predicate::orientation::orient_value)
// ===========================================================================

#[test]
fn orientation_test_matches_cpu() {
    use crate::predicate::orientation::orient_value;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let (ax, ay, bx, by, cx, cy) = triangle_inputs(n, 0x0_4156);

    // CPU reference: the crate's own twice-signed-area predicate.
    let mut out_cpu = vec![0.0_f64; n];
    for i in 0..n {
        let a = Point::new(ax[i], ay[i]);
        let b = Point::new(bx[i], by[i]);
        let c = Point::new(cx[i], cy[i]);
        out_cpu[i] = orient_value(a, b, c);
    }

    let ptx = crate::ptx_kernels::orientation_test_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "orientation_test_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ax = DeviceBuffer::<f64>::from_host(&ax).expect("d_ax");
    let d_ay = DeviceBuffer::<f64>::from_host(&ay).expect("d_ay");
    let d_bx = DeviceBuffer::<f64>::from_host(&bx).expect("d_bx");
    let d_by = DeviceBuffer::<f64>::from_host(&by).expect("d_by");
    let d_cx = DeviceBuffer::<f64>::from_host(&cx).expect("d_cx");
    let d_cy = DeviceBuffer::<f64>::from_host(&cy).expect("d_cy");
    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ax.as_device_ptr(),
                d_ay.as_device_ptr(),
                d_bx.as_device_ptr(),
                d_by.as_device_ptr(),
                d_cx.as_device_ptr(),
                d_cy.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch orientation_test_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f64; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Pure f64 `mul`/`sub`; the only divergence is a possible single-rounding
    // ptxas fma contraction (~1 ulp ≈ 2e-16 relative). 1e-9 relative is a
    // comfortable bound that still flags any wrong index / sign / formula.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_cpu[i], 1e-9, 1e-12),
            "orientation out[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_cpu[i]
        );
    }
}

// ===========================================================================
// 2. cross_product  —  HOST RE-DERIVATION (ax·by − ay·bx)  cross-checked vs cross2
// ===========================================================================

#[test]
fn cross_product_matches_cpu() {
    use crate::predicate::dot_cross::cross2;
    use crate::primitives::vector::Vector;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0xC_2055);
    let ax: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let ay: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let bx: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let by: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();

    // CPU reference: the crate's own 2-D cross helper, `a.x*b.y − a.y*b.x`.
    let mut out_cpu = vec![0.0_f64; n];
    for i in 0..n {
        out_cpu[i] = cross2(Vector::new(ax[i], ay[i]), Vector::new(bx[i], by[i]));
    }

    let ptx = crate::ptx_kernels::cross_product_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cross_product_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ax = DeviceBuffer::<f64>::from_host(&ax).expect("d_ax");
    let d_ay = DeviceBuffer::<f64>::from_host(&ay).expect("d_ay");
    let d_bx = DeviceBuffer::<f64>::from_host(&bx).expect("d_bx");
    let d_by = DeviceBuffer::<f64>::from_host(&by).expect("d_by");
    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ax.as_device_ptr(),
                d_ay.as_device_ptr(),
                d_bx.as_device_ptr(),
                d_by.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch cross_product_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f64; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_cpu[i], 1e-9, 1e-12),
            "cross_product out[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_cpu[i]
        );
    }
}

// ===========================================================================
// 3. point_in_aabb  —  CRATE ORACLE (primitives::aabb::Aabb::contains)
// ===========================================================================
//
// PTX BUG FIXED here (see module docs): the original kernel re-declared `%r5`
// after `.reg .u32 %r<8>;`, which ptxas rejects as a duplicate definition on
// sm_86. With the fix the kernel loads and produces a correct in/out mask.

#[test]
fn point_in_aabb_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    // Closed box [0,0]–[1,1]; points spread over [−0.5, 1.5) so roughly half are
    // inside and the inputs never land exactly on a boundary (random f64), so the
    // `>=` / `<=` comparisons are unambiguous and identical on host and device.
    let aabb = Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
    let xmin = aabb.min.x;
    let ymin = aabb.min.y;
    let xmax = aabb.max.x;
    let ymax = aabb.max.y;

    let mut rng = LcgRng::new(0xA_ABB0);
    let px: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 0.5).collect();
    let py: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 0.5).collect();

    // CPU reference: the crate's own closed-box membership test.
    let out_cpu: Vec<u32> = (0..n)
        .map(|i| u32::from(aabb.contains(Point::new(px[i], py[i]))))
        .collect();

    let ptx = crate::ptx_kernels::point_in_aabb_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "point_in_aabb_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_px = DeviceBuffer::<f64>::from_host(&px).expect("d_px");
    let d_py = DeviceBuffer::<f64>::from_host(&py).expect("d_py");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_px.as_device_ptr(),
                d_py.as_device_ptr(),
                xmin,
                ymin,
                xmax,
                ymax,
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch point_in_aabb_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // At least one inside and one outside, so the test is non-trivial.
    let inside = out_cpu.iter().filter(|&&v| v == 1).count();
    assert!(
        inside > 0 && inside < n,
        "test setup: expected a mix of inside/outside, got {inside}/{n} inside"
    );

    for i in 0..n {
        assert_eq!(
            out_gpu[i], out_cpu[i],
            "point_in_aabb out[{i}] mismatch: gpu={} cpu={} (px={}, py={})",
            out_gpu[i], out_cpu[i], px[i], py[i]
        );
    }
}

// ===========================================================================
// 4. segment_intersection  —  HOST RE-DERIVATION of the proper-crossing test
// ===========================================================================
//
// The kernel implements the strict sign-difference (proper) crossing test:
//   o1 = (p2−p1) × (q1−p1),  o2 = (p2−p1) × (q2−p1)
//   o3 = (q2−q1) × (p1−q1),  o4 = (q2−q1) × (p2−q1)
//   out = 1  iff  (o1·o2 < 0) ∧ (o3·o4 < 0)
// (collinear / endpoint-touching cases yield 0). The host oracle re-derives the
// identical arithmetic so the boolean is bit-exact for well-separated inputs.

/// 2-D cross of `(bx−ax, by−ay)` with `(cx−ax, cy−ay)`.
fn cross_about(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

#[test]
fn segment_intersection_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Deterministic mix: explicit clear crossing, explicit clear miss, then
    // well-separated random segments. Endpoints stay O(1) so the products are
    // far from zero and a 1-ulp fma contraction cannot flip a decision.
    let mut p1x = vec![0.0_f64];
    let mut p1y = vec![0.0_f64];
    let mut p2x = vec![1.0_f64];
    let mut p2y = vec![1.0_f64];
    let mut q1x = vec![0.0_f64];
    let mut q1y = vec![1.0_f64];
    let mut q2x = vec![1.0_f64];
    let mut q2y = vec![0.0_f64];
    // index 1: parallel-ish clear miss
    p1x.push(0.0);
    p1y.push(0.0);
    p2x.push(1.0);
    p2y.push(0.0);
    q1x.push(0.0);
    q1y.push(1.0);
    q2x.push(1.0);
    q2y.push(1.0);

    let mut rng = LcgRng::new(0x5_E600);
    let n_rand = 254_usize;
    for _ in 0..n_rand {
        p1x.push(rng.next_f64() * 2.0 - 1.0);
        p1y.push(rng.next_f64() * 2.0 - 1.0);
        p2x.push(rng.next_f64() * 2.0 - 1.0);
        p2y.push(rng.next_f64() * 2.0 - 1.0);
        q1x.push(rng.next_f64() * 2.0 - 1.0);
        q1y.push(rng.next_f64() * 2.0 - 1.0);
        q2x.push(rng.next_f64() * 2.0 - 1.0);
        q2y.push(rng.next_f64() * 2.0 - 1.0);
    }
    let n = p1x.len();

    // Host oracle: identical strict sign-difference test.
    let mut out_host = vec![0_u32; n];
    for i in 0..n {
        let o1 = cross_about(p1x[i], p1y[i], p2x[i], p2y[i], q1x[i], q1y[i]);
        let o2 = cross_about(p1x[i], p1y[i], p2x[i], p2y[i], q2x[i], q2y[i]);
        let o3 = cross_about(q1x[i], q1y[i], q2x[i], q2y[i], p1x[i], p1y[i]);
        let o4 = cross_about(q1x[i], q1y[i], q2x[i], q2y[i], p2x[i], p2y[i]);
        out_host[i] = u32::from((o1 * o2 < 0.0) && (o3 * o4 < 0.0));
    }

    let ptx = crate::ptx_kernels::segment_intersection_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "segment_intersection_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p1x = DeviceBuffer::<f64>::from_host(&p1x).expect("d_p1x");
    let d_p1y = DeviceBuffer::<f64>::from_host(&p1y).expect("d_p1y");
    let d_p2x = DeviceBuffer::<f64>::from_host(&p2x).expect("d_p2x");
    let d_p2y = DeviceBuffer::<f64>::from_host(&p2y).expect("d_p2y");
    let d_q1x = DeviceBuffer::<f64>::from_host(&q1x).expect("d_q1x");
    let d_q1y = DeviceBuffer::<f64>::from_host(&q1y).expect("d_q1y");
    let d_q2x = DeviceBuffer::<f64>::from_host(&q2x).expect("d_q2x");
    let d_q2y = DeviceBuffer::<f64>::from_host(&q2y).expect("d_q2y");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p1x.as_device_ptr(),
                d_p1y.as_device_ptr(),
                d_p2x.as_device_ptr(),
                d_p2y.as_device_ptr(),
                d_q1x.as_device_ptr(),
                d_q1y.as_device_ptr(),
                d_q2x.as_device_ptr(),
                d_q2y.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch segment_intersection_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // The two seeded cases anchor the test: a proper crossing and a clear miss.
    assert_eq!(out_host[0], 1, "seeded crossing must intersect");
    assert_eq!(out_host[1], 0, "seeded parallel pair must not intersect");

    for i in 0..n {
        assert_eq!(
            out_gpu[i], out_host[i],
            "segment_intersection out[{i}] mismatch: gpu={} host={}",
            out_gpu[i], out_host[i]
        );
    }
}

// ===========================================================================
// 5. convex_hull_step  —  CRATE ORACLE (sign of predicate::orientation::orient_value)
// ===========================================================================
//
// The kernel writes the orientation sign as a 32-bit integer: +1 (CCW), 0
// (collinear), or 0xFFFFFFFF (−1, CW). It is read back as `i32`.

#[test]
fn convex_hull_step_matches_cpu() {
    use crate::predicate::orientation::orient_value;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let (ax, ay, bx, by, cx, cy) = triangle_inputs(n, 0x0_C811);

    // CPU reference: sign of the crate's orientation value. The seeded indices
    // 0/1/2 deterministically exercise the 0 / +1 / −1 branches.
    let mut sign_cpu = vec![0_i32; n];
    for i in 0..n {
        let v = orient_value(
            Point::new(ax[i], ay[i]),
            Point::new(bx[i], by[i]),
            Point::new(cx[i], cy[i]),
        );
        sign_cpu[i] = if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        };
    }

    let ptx = crate::ptx_kernels::convex_hull_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "convex_hull_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ax = DeviceBuffer::<f64>::from_host(&ax).expect("d_ax");
    let d_ay = DeviceBuffer::<f64>::from_host(&ay).expect("d_ay");
    let d_bx = DeviceBuffer::<f64>::from_host(&bx).expect("d_bx");
    let d_by = DeviceBuffer::<f64>::from_host(&by).expect("d_by");
    let d_cx = DeviceBuffer::<f64>::from_host(&cx).expect("d_cx");
    let d_cy = DeviceBuffer::<f64>::from_host(&cy).expect("d_cy");
    let d_out = DeviceBuffer::<i32>::from_host(&vec![0_i32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ax.as_device_ptr(),
                d_ay.as_device_ptr(),
                d_bx.as_device_ptr(),
                d_by.as_device_ptr(),
                d_cx.as_device_ptr(),
                d_cy.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch convex_hull_step_kernel");
    stream.synchronize().expect("sync");

    let mut sign_gpu = vec![0_i32; n];
    d_out.copy_to_host(&mut sign_gpu).expect("copy out");

    // Sanity: the three seeded branches landed where expected.
    assert_eq!(sign_cpu[0], 0, "seeded collinear must be 0");
    assert_eq!(sign_cpu[1], 1, "seeded CCW must be +1");
    assert_eq!(sign_cpu[2], -1, "seeded CW must be −1");

    for i in 0..n {
        assert_eq!(
            sign_gpu[i], sign_cpu[i],
            "convex_hull_step sign[{i}] mismatch: gpu={} cpu={}",
            sign_gpu[i], sign_cpu[i]
        );
    }
}

// ===========================================================================
// 6. kd_tree_traverse  —  CRATE ORACLE (primitives::point::Point::distance_sq)
// ===========================================================================

#[test]
fn kd_tree_traverse_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let qx = 0.37_f64;
    let qy = -0.81_f64;
    let query = Point::new(qx, qy);

    let mut rng = LcgRng::new(0x4_D7EE);
    let cx: Vec<f64> = (0..n).map(|_| rng.next_f64() * 6.0 - 3.0).collect();
    let cy: Vec<f64> = (0..n).map(|_| rng.next_f64() * 6.0 - 3.0).collect();

    // CPU reference: the crate's own squared-distance helper.
    let mut out_cpu = vec![0.0_f64; n];
    for i in 0..n {
        out_cpu[i] = Point::new(cx[i], cy[i]).distance_sq(query);
    }

    let ptx = crate::ptx_kernels::kd_tree_traverse_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "kd_tree_traverse_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_cx = DeviceBuffer::<f64>::from_host(&cx).expect("d_cx");
    let d_cy = DeviceBuffer::<f64>::from_host(&cy).expect("d_cy");
    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                qx,
                qy,
                d_cx.as_device_ptr(),
                d_cy.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch kd_tree_traverse_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f64; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_cpu[i], 1e-9, 1e-12),
            "kd_tree dist_sq[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_cpu[i]
        );
    }
}

// ===========================================================================
// 7. polygon_area  —  HOST RE-DERIVATION (per-edge shoelace term, ring sum = 2·area)
// ===========================================================================
//
// `out[i] = px[i]·py[j] − px[j]·py[i]` with `j = (i+1) mod n`. The sum over all
// edges equals twice the signed polygon area, which we additionally cross-check
// against the crate's `Polygon::signed_area`.

#[test]
fn polygon_area_matches_host() {
    use crate::primitives::polygon::Polygon;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // A convex, CCW polygon (regular-ish ring) so the wrap-around term and the
    // ring sum are both meaningful and the signed area is positive.
    let n = 12_usize;
    let mut px = vec![0.0_f64; n];
    let mut py = vec![0.0_f64; n];
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        px[i] = 2.0 * theta.cos() + 0.25;
        py[i] = 1.5 * theta.sin() - 0.5;
    }

    // Host oracle: the kernel's documented per-edge term with `j = (i+1) % n`.
    let mut out_host = vec![0.0_f64; n];
    for i in 0..n {
        let j = (i + 1) % n;
        out_host[i] = px[i] * py[j] - px[j] * py[i];
    }

    let ptx = crate::ptx_kernels::polygon_area_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "polygon_area_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_px = DeviceBuffer::<f64>::from_host(&px).expect("d_px");
    let d_py = DeviceBuffer::<f64>::from_host(&py).expect("d_py");
    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_px.as_device_ptr(),
                d_py.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch polygon_area_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f64; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Per-edge equivalence.
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_host[i], 1e-9, 1e-12),
            "polygon_area edge[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_host[i]
        );
    }

    // Cross-check: 0.5·Σ edge terms == the crate's signed polygon area.
    let gpu_signed_area = 0.5 * out_gpu.iter().sum::<f64>();
    let pts: Vec<Point> = (0..n).map(|i| Point::new(px[i], py[i])).collect();
    let poly = Polygon::new(pts).expect("polygon");
    let crate_area = poly.signed_area();
    assert!(
        close(gpu_signed_area, crate_area, 1e-9, 1e-9),
        "polygon_area: 0.5·Σ(gpu edge terms)={gpu_signed_area} != Polygon::signed_area()={crate_area}"
    );
}
