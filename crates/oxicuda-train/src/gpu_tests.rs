//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`] and [`crate::amp`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU oracle. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` harnesses: every pointer parameter is passed as its
//! `CUdeviceptr` (a `.param .u64`); scalars are passed as the matching Rust
//! scalar (`.param .f32` ⇒ `f32`, `.param .u64` ⇒ `u64`) in the kernel's
//! declared parameter order. Note that **every length parameter in these
//! kernels is `.param .u64`**, so element counts are passed as `u64`.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to the
//!   crate's own `gpu_optimizer::*::apply_update` host reference, which is the
//!   CPU twin of the fused kernel:
//!   `adam_update_f32` ↔ [`crate::gpu_optimizer::adam`] (with `wd = 0`, since the
//!   Adam kernel takes no weight-decay parameter), `adamw_update_f32` (decoupled
//!   weight decay), and `lion_update_f32` ↔ [`crate::gpu_optimizer::lion`].
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine (or has no standalone `pub fn`), so the oracle is an independent
//!   Rust re-implementation of the kernel's *documented* arithmetic:
//!   `sgd_update_f32` (the documented SGD-with-momentum step, both Nesterov on
//!   and off), `came_row_factor_f32` / `came_col_factor_f32` (per-row / per-col
//!   Σ g²), `norm_sq_partial_f32` (block Σ g² via warp-shuffle + shared-memory
//!   reduction), `scale_inplace_f32` (`x *= s`) and `add_inplace_f32`
//!   (`acc += src`). These still genuinely fail if ptxas miscompiles or the PTX
//!   has a wrong constant / shift / index, because the host code is independent
//!   of the JIT-compiled PTX.
//!
//! ## PTX bug found and fixed
//!
//! ### `sgd_update_f32` — parameter update silently cancelled (WRONG MATH)
//!
//! The original update epilogue was:
//! ```text
//!     fma.rn.f32   %p,   %lr_r, %upd, %p;   // p = p + lr*upd
//!     mul.f32      %upd, %lr_r, %upd;       // upd = lr*upd
//!     sub.f32      %p,   %p,   %upd;        // p = (p + lr*upd) - lr*upd = p
//! ```
//! The stray leading `fma` adds `lr*upd` and the following `mul`+`sub` subtract
//! exactly the same quantity, so the stored parameter equals its *input* — SGD
//! never moved the weights at all. The velocity buffer was updated correctly,
//! which is why a velocity-only check would have missed it; only a parameter
//! CPU-vs-GPU comparison exposes it. Fixed in `ptx_kernels.rs` by removing the
//! stray `fma`, leaving the correct `p -= lr*upd`.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

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
    sm: SmVersion,
    /// Numeric SM value (`major * 10 + minor`, e.g. `86` for `sm_86`), used by
    /// the `amp` PTX generators which take a plain `u32` SM rather than a
    /// [`SmVersion`].
    sm_num: u32,
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
    let sm = SmVersion::from_compute_capability(major, minor)?;
    let sm_num = (major * 10 + minor) as u32;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
        sm_num,
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug, so we
/// panic (fail the test) rather than skip.
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
// 1. adam_update_f32  —  CRATE ORACLE (gpu_optimizer::adam, wd = 0)
// ===========================================================================

#[test]
fn adam_update_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let step_size = 1.0e-3_f32;
    let bc2_rsqrt = 1.0_f32; // 1/√(1-β₂ᵗ): pick the converged value for simplicity
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1.0e-8_f32;

    let mut rng = LcgRng::new(0xADA_0001);
    let param: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let m1: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
    // Second moment in [0.5, 1.5] keeps √v̂ ≥ ~0.7 so the rcp.approx denominator
    // is well-conditioned (never near the eps cliff).
    let m2: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32()).collect();

    // ---- CPU reference: the kernel's documented per-element Adam (no wd) ----
    let c1 = 1.0_f32 - beta1;
    let c2 = 1.0_f32 - beta2;
    let mut p_cpu = param.clone();
    let mut m1_cpu = m1.clone();
    let mut m2_cpu = m2.clone();
    for k in 0..n {
        let g = grad[k];
        let nm1 = beta1 * m1[k] + c1 * g;
        let nm2 = beta2 * m2[k] + c2 * g * g;
        let denom = nm2.sqrt() * bc2_rsqrt + eps;
        p_cpu[k] = param[k] - step_size * nm1 / denom;
        m1_cpu[k] = nm1;
        m2_cpu[k] = nm2;
    }

    let ptx = crate::ptx_kernels::adam_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "adam_update_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&param).expect("d_p");
    let d_g = DeviceBuffer::<f32>::from_host(&grad).expect("d_g");
    let d_m1 = DeviceBuffer::<f32>::from_host(&m1).expect("d_m1");
    let d_m2 = DeviceBuffer::<f32>::from_host(&m2).expect("d_m2");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_g.as_device_ptr(),
                d_m1.as_device_ptr(),
                d_m2.as_device_ptr(),
                step_size,
                bc2_rsqrt,
                beta1,
                beta2,
                eps,
                n as u64,
            ),
        )
        .expect("launch adam_update_f32");
    stream.synchronize().expect("sync");

    let mut p_gpu = vec![0.0_f32; n];
    let mut m1_gpu = vec![0.0_f32; n];
    let mut m2_gpu = vec![0.0_f32; n];
    d_p.copy_to_host(&mut p_gpu).expect("copy p");
    d_m1.copy_to_host(&mut m1_gpu).expect("copy m1");
    d_m2.copy_to_host(&mut m2_gpu).expect("copy m2");

    // sqrt.approx + rcp.approx ⇒ a few ulp (~1e-6 rel); 2e-3 is comfortable and
    // still flags any gross formula error (e.g. a missing bias correction).
    let (rel, abs) = worst_diff(&p_gpu, &p_cpu);
    for k in 0..n {
        assert!(
            close(p_gpu[k], p_cpu[k], 2e-3, 1e-4),
            "adam param[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            p_gpu[k],
            p_cpu[k]
        );
    }
    // Moments are exact fma ⇒ tight tolerance.
    for k in 0..n {
        assert!(
            close(m1_gpu[k], m1_cpu[k], 1e-5, 1e-6),
            "adam m1[{k}] mismatch: gpu={} cpu={}",
            m1_gpu[k],
            m1_cpu[k]
        );
        assert!(
            close(m2_gpu[k], m2_cpu[k], 1e-5, 1e-6),
            "adam m2[{k}] mismatch: gpu={} cpu={}",
            m2_gpu[k],
            m2_cpu[k]
        );
    }
}

// ===========================================================================
// 2. adamw_update_f32  —  CRATE ORACLE (gpu_optimizer::adamw, decoupled wd)
// ===========================================================================

#[test]
fn adamw_update_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let step_size = 1.0e-3_f32;
    let bc2_rsqrt = 1.0_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1.0e-8_f32;
    let lr_wd = 1.0e-2_f32; // lr*λ (decoupled weight-decay product)

    let mut rng = LcgRng::new(0xADA_0002);
    let param: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let m1: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
    let m2: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32()).collect();

    // ---- CPU reference: p *= (1-lr*wd) first, then the no-L2 Adam step ----
    let c1 = 1.0_f32 - beta1;
    let c2 = 1.0_f32 - beta2;
    let wdf = 1.0_f32 - lr_wd;
    let mut p_cpu = param.clone();
    for k in 0..n {
        let g = grad[k];
        let p_decayed = param[k] * wdf;
        let nm1 = beta1 * m1[k] + c1 * g;
        let nm2 = beta2 * m2[k] + c2 * g * g;
        let denom = nm2.sqrt() * bc2_rsqrt + eps;
        p_cpu[k] = p_decayed - step_size * nm1 / denom;
    }

    let ptx = crate::ptx_kernels::adamw_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "adamw_update_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&param).expect("d_p");
    let d_g = DeviceBuffer::<f32>::from_host(&grad).expect("d_g");
    let d_m1 = DeviceBuffer::<f32>::from_host(&m1).expect("d_m1");
    let d_m2 = DeviceBuffer::<f32>::from_host(&m2).expect("d_m2");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_g.as_device_ptr(),
                d_m1.as_device_ptr(),
                d_m2.as_device_ptr(),
                step_size,
                bc2_rsqrt,
                beta1,
                beta2,
                eps,
                lr_wd,
                n as u64,
            ),
        )
        .expect("launch adamw_update_f32");
    stream.synchronize().expect("sync");

    let mut p_gpu = vec![0.0_f32; n];
    d_p.copy_to_host(&mut p_gpu).expect("copy p");

    let (rel, abs) = worst_diff(&p_gpu, &p_cpu);
    for k in 0..n {
        assert!(
            close(p_gpu[k], p_cpu[k], 2e-3, 1e-4),
            "adamw param[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            p_gpu[k],
            p_cpu[k]
        );
    }
}

// ===========================================================================
// 3. sgd_update_f32  —  INDEPENDENT HOST RE-DERIVATION (momentum + Nesterov)
//    THIS TEST CATCHES THE FIXED "parameter never updated" BUG.
// ===========================================================================

fn run_sgd_case(fx: &GpuFixture, nesterov: bool) {
    let n = 1024_usize;
    let lr = 0.05_f32;
    let momentum = 0.9_f32;
    let dampening = 0.1_f32;
    let weight_decay = 1.0e-3_f32;
    let one_damp = 1.0_f32 - dampening;
    let nes_flag = if nesterov { 1.0_f32 } else { 0.0_f32 };

    let mut rng = LcgRng::new(0x56D_0000 ^ u64::from(nesterov));
    let param: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let vel: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

    // ---- Independent host re-derivation of the documented SGD step ----
    let mut p_cpu = param.clone();
    let mut v_cpu = vel.clone();
    for k in 0..n {
        let gp = grad[k] + weight_decay * param[k];
        let v_new = momentum * vel[k] + one_damp * gp;
        let upd = if nesterov {
            gp + momentum * v_new
        } else {
            v_new
        };
        p_cpu[k] = param[k] - lr * upd;
        v_cpu[k] = v_new;
    }

    let ptx = crate::ptx_kernels::sgd_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sgd_update_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&param).expect("d_p");
    let d_g = DeviceBuffer::<f32>::from_host(&grad).expect("d_g");
    let d_v = DeviceBuffer::<f32>::from_host(&vel).expect("d_v");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_g.as_device_ptr(),
                d_v.as_device_ptr(),
                lr,
                momentum,
                dampening,
                weight_decay,
                nes_flag,
                n as u64,
            ),
        )
        .expect("launch sgd_update_f32");
    stream.synchronize().expect("sync");

    let mut p_gpu = vec![0.0_f32; n];
    let mut v_gpu = vec![0.0_f32; n];
    d_p.copy_to_host(&mut p_gpu).expect("copy p");
    d_v.copy_to_host(&mut v_gpu).expect("copy v");

    // All ops are exact fma/mul/sub ⇒ ~1 ulp.
    let (rel, abs) = worst_diff(&p_gpu, &p_cpu);
    for k in 0..n {
        assert!(
            close(p_gpu[k], p_cpu[k], 1e-5, 1e-6),
            "sgd (nesterov={nesterov}) param[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e}) — if gpu==input the update cancelled",
            p_gpu[k],
            p_cpu[k]
        );
        assert!(
            close(v_gpu[k], v_cpu[k], 1e-5, 1e-6),
            "sgd (nesterov={nesterov}) vel[{k}] mismatch: gpu={} cpu={}",
            v_gpu[k],
            v_cpu[k]
        );
    }

    // Guard against a silent no-op regression: with these inputs the parameter
    // MUST actually move (the pre-fix kernel left it identical to the input).
    let mut moved = 0usize;
    for k in 0..n {
        if (p_gpu[k] - param[k]).abs() > 1e-6 {
            moved += 1;
        }
    }
    assert!(
        moved > n / 2,
        "sgd (nesterov={nesterov}): only {moved}/{n} params changed — update appears cancelled"
    );
}

#[test]
fn sgd_update_plain_momentum_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_sgd_case(&fx, false);
}

#[test]
fn sgd_update_nesterov_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_sgd_case(&fx, true);
}

// ===========================================================================
// 4. lion_update_f32  —  CRATE ORACLE (gpu_optimizer::lion)
// ===========================================================================

#[test]
fn lion_update_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let lr = 1.0e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.99_f32;
    let weight_decay = 1.0e-2_f32;
    let wd_factor = 1.0_f32 - lr * weight_decay;

    let mut rng = LcgRng::new(0x110_0003);
    let param: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let m: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

    // ---- CPU reference matching gpu_optimizer::lion::apply_update ----
    let c1 = 1.0_f32 - beta1;
    let c2 = 1.0_f32 - beta2;
    let mut p_cpu = param.clone();
    let mut m_cpu = m.clone();
    for k in 0..n {
        let g = grad[k];
        let mi = m[k];
        let c = beta1 * mi + c1 * g;
        let sgn = if c > 0.0_f32 {
            1.0_f32
        } else if c < 0.0_f32 {
            -1.0_f32
        } else {
            0.0_f32
        };
        p_cpu[k] = param[k] * wd_factor - lr * sgn;
        m_cpu[k] = beta2 * mi + c2 * g;
    }

    let ptx = crate::ptx_kernels::lion_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lion_update_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&param).expect("d_p");
    let d_g = DeviceBuffer::<f32>::from_host(&grad).expect("d_g");
    let d_m = DeviceBuffer::<f32>::from_host(&m).expect("d_m");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_g.as_device_ptr(),
                d_m.as_device_ptr(),
                lr,
                beta1,
                beta2,
                weight_decay,
                n as u64,
            ),
        )
        .expect("launch lion_update_f32");
    stream.synchronize().expect("sync");

    let mut p_gpu = vec![0.0_f32; n];
    let mut m_gpu = vec![0.0_f32; n];
    d_p.copy_to_host(&mut p_gpu).expect("copy p");
    d_m.copy_to_host(&mut m_gpu).expect("copy m");

    let (rel, abs) = worst_diff(&p_gpu, &p_cpu);
    for k in 0..n {
        assert!(
            close(p_gpu[k], p_cpu[k], 1e-5, 1e-6),
            "lion param[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            p_gpu[k],
            p_cpu[k]
        );
        assert!(
            close(m_gpu[k], m_cpu[k], 1e-5, 1e-6),
            "lion m[{k}] mismatch: gpu={} cpu={}",
            m_gpu[k],
            m_cpu[k]
        );
    }
}

// ===========================================================================
// 5. came_row_factor_f32  —  INDEPENDENT HOST RE-DERIVATION (per-row Σ g²)
// ===========================================================================

#[test]
fn came_row_factor_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rows = 17_usize;
    let n_cols = 23_usize;

    let mut rng = LcgRng::new(0xCA3_0005);
    // g² buffer (already squared gradients), row-major n_rows × n_cols.
    let g2: Vec<f32> = (0..n_rows * n_cols)
        .map(|_| {
            let g = rng.next_f32() * 2.0 - 1.0;
            g * g
        })
        .collect();

    // Host: row[i] = Σ_j g²[i, j] in the kernel's column-ascending order.
    let mut row_host = vec![0.0_f32; n_rows];
    for (i, slot) in row_host.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for j in 0..n_cols {
            acc += g2[i * n_cols + j];
        }
        *slot = acc;
    }

    let ptx = crate::ptx_kernels::came_row_factor_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "came_row_factor_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_g2 = DeviceBuffer::<f32>::from_host(&g2).expect("d_g2");
    // Initialise to zero: the kernel overwrites each row factor with the sum.
    let d_row = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_rows]).expect("d_row");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_rows as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_g2.as_device_ptr(),
                d_row.as_device_ptr(),
                n_cols as u64,
                n_rows as u64,
            ),
        )
        .expect("launch came_row_factor_f32");
    stream.synchronize().expect("sync");

    let mut row_gpu = vec![0.0_f32; n_rows];
    d_row.copy_to_host(&mut row_gpu).expect("copy row");

    let (rel, abs) = worst_diff(&row_gpu, &row_host);
    for i in 0..n_rows {
        assert!(
            close(row_gpu[i], row_host[i], 1e-5, 1e-5),
            "came_row factor[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            row_gpu[i],
            row_host[i]
        );
    }
}

// ===========================================================================
// 6. came_col_factor_f32  —  INDEPENDENT HOST RE-DERIVATION (per-col Σ g²)
// ===========================================================================

#[test]
fn came_col_factor_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rows = 19_usize;
    let n_cols = 13_usize;

    let mut rng = LcgRng::new(0xCA3_0006);
    let g2: Vec<f32> = (0..n_rows * n_cols)
        .map(|_| {
            let g = rng.next_f32() * 2.0 - 1.0;
            g * g
        })
        .collect();

    // Host: col[j] = Σ_i g²[i, j] in the kernel's row-ascending order.
    let mut col_host = vec![0.0_f32; n_cols];
    for (j, slot) in col_host.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for i in 0..n_rows {
            acc += g2[i * n_cols + j];
        }
        *slot = acc;
    }

    let ptx = crate::ptx_kernels::came_col_factor_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "came_col_factor_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_g2 = DeviceBuffer::<f32>::from_host(&g2).expect("d_g2");
    let d_col = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_cols]).expect("d_col");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_cols as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_g2.as_device_ptr(),
                d_col.as_device_ptr(),
                n_cols as u64,
                n_rows as u64,
            ),
        )
        .expect("launch came_col_factor_f32");
    stream.synchronize().expect("sync");

    let mut col_gpu = vec![0.0_f32; n_cols];
    d_col.copy_to_host(&mut col_gpu).expect("copy col");

    let (rel, abs) = worst_diff(&col_gpu, &col_host);
    for j in 0..n_cols {
        assert!(
            close(col_gpu[j], col_host[j], 1e-5, 1e-5),
            "came_col factor[{j}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            col_gpu[j],
            col_host[j]
        );
    }
}

// ===========================================================================
// 7. norm_sq_partial_f32  —  INDEPENDENT HOST RE-DERIVATION (block Σ g²)
//    Exercises the full warp-shuffle butterfly + shared-memory cross-warp reduce.
// ===========================================================================

#[test]
fn norm_sq_partial_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Single block, 256 threads (8 warps) ⇒ the shared-memory cross-warp reduce
    // path is fully exercised. n a multiple of the block keeps the grid-stride
    // trip count uniform within every warp (no shfl.sync divergence).
    let n = 2048_usize;
    let block = 256_u32;
    let n_blocks = 1_u32;

    let mut rng = LcgRng::new(0x0070_5007);
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host: full block sum of g² (the single block reduces all n elements).
    let host_sum: f32 = grad.iter().map(|&g| g * g).sum();

    let ptx = crate::ptx_kernels::norm_sq_partial_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "norm_sq_partial_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_g = DeviceBuffer::<f32>::from_host(&grad).expect("d_g");
    let d_partial =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_blocks as usize]).expect("d_partial");

    let params = LaunchParams::new(n_blocks, block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_g.as_device_ptr(), d_partial.as_device_ptr(), n as u64),
        )
        .expect("launch norm_sq_partial_f32");
    stream.synchronize().expect("sync");

    let mut partial_gpu = vec![0.0_f32; n_blocks as usize];
    d_partial
        .copy_to_host(&mut partial_gpu)
        .expect("copy partial");

    // Tree reduction vs sequential host sum over 2048 positive terms: a few ulp
    // accumulated, ~1e-5 relative. The block sum is O(170) so a 1e-4 relative
    // bound (~1.7e-2 absolute) is comfortable yet still flags any structural
    // reduction error (e.g. a dropped warp or smem-offset bug → way off).
    let gpu = partial_gpu[0];
    assert!(
        close(gpu, host_sum, 1e-4, 1e-2),
        "norm_sq_partial block sum mismatch: gpu={gpu} host={host_sum} \
         (rel={:e})",
        ((gpu - host_sum) / host_sum).abs()
    );
}

// ===========================================================================
// 8. scale_inplace_f32  —  INDEPENDENT HOST RE-DERIVATION (x *= s)
// ===========================================================================

#[test]
fn scale_inplace_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let scale = 0.375_f32;

    let mut rng = LcgRng::new(0x5CA_0008);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    let x_host: Vec<f32> = x.iter().map(|&v| v * scale).collect();

    let ptx = crate::ptx_kernels::scale_inplace_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "scale_inplace_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_x.as_device_ptr(), scale, n as u64))
        .expect("launch scale_inplace_f32");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    for k in 0..n {
        assert!(
            close(x_gpu[k], x_host[k], 1e-6, 1e-7),
            "scale_inplace x[{k}] mismatch: gpu={} host={}",
            x_gpu[k],
            x_host[k]
        );
    }
}

// ===========================================================================
// 9. add_inplace_f32  —  INDEPENDENT HOST RE-DERIVATION (acc += src)
// ===========================================================================

#[test]
fn add_inplace_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;

    let mut rng = LcgRng::new(0xADD_0009);
    let acc: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let src: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    let acc_host: Vec<f32> = acc.iter().zip(&src).map(|(&a, &s)| a + s).collect();

    let ptx = crate::ptx_kernels::add_inplace_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "add_inplace_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_acc = DeviceBuffer::<f32>::from_host(&acc).expect("d_acc");
    let d_src = DeviceBuffer::<f32>::from_host(&src).expect("d_src");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_acc.as_device_ptr(), d_src.as_device_ptr(), n as u64),
        )
        .expect("launch add_inplace_f32");
    stream.synchronize().expect("sync");

    let mut acc_gpu = vec![0.0_f32; n];
    d_acc.copy_to_host(&mut acc_gpu).expect("copy acc");

    for k in 0..n {
        assert!(
            close(acc_gpu[k], acc_host[k], 1e-6, 1e-7),
            "add_inplace acc[{k}] mismatch: gpu={} host={}",
            acc_gpu[k],
            acc_host[k]
        );
    }
}

// ===========================================================================
// 10. unscale_inplace  —  INDEPENDENT HOST RE-DERIVATION (data[i] *= inv_scale)
//     AMP gradient unscale, from `crate::amp::unscale_ptx`.
//
//     This is the kernel the convergence audit flagged: its only prior test
//     was a `ptx.contains("unscale_inplace")` string assertion that never
//     invoked ptxas, hiding the fact that the PTX did not compile at all (the
//     user registers `%tid`/`%ntid`/`%ctaid`/`%nctaid` collided with the PTX
//     built-in special registers, so `mov.u32 %tid, %tid.x` was rejected by
//     ptxas as an illegal video selector). Fixed in `amp.rs` by renaming the
//     work registers to `%t`/`%nt`/`%bid`/`%nc`.
//
//     Launched deliberately under-provisioned (grid*block < n) so the
//     grid-stride advance is exercised across multiple iterations per thread —
//     a stuck grid-stride would leave the buffer tail unscaled and fail here.
// ===========================================================================

fn run_unscale_case(fx: &GpuFixture, inv_scale: f32, seed: u64) {
    let n = 4096_usize;

    let mut rng = LcgRng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * 8.0 - 4.0).collect();

    // Host oracle: every element multiplied by inv_scale (= 1 / scale_factor).
    let host: Vec<f32> = data.iter().map(|&v| v * inv_scale).collect();

    let ptx = crate::amp::unscale_ptx(fx.sm_num);
    let kernel = load_kernel(&ptx, "unscale_inplace");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");

    // 4 blocks × 128 = 512 threads for 4096 elements ⇒ each thread strides 8
    // times. n is passed as `.u32` (the amp kernel's length type, unlike the
    // `.u64` lengths in `ptx_kernels`).
    let block = 128_u32;
    let grid = 4_u32;
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), n as u32, inv_scale),
        )
        .expect("launch unscale_inplace");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_data.copy_to_host(&mut gpu).expect("copy data");

    // A single exact `mul.rn.f32` ⇒ ~1 ulp.
    let (rel, abs) = worst_diff(&gpu, &host);
    for k in 0..n {
        assert!(
            close(gpu[k], host[k], 1e-6, 1e-7),
            "unscale (inv_scale={inv_scale}) data[{k}] mismatch: gpu={} host={} \
             (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            host[k]
        );
    }
}

#[test]
fn unscale_inplace_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // The canonical AMP inverse scale (1 / 2¹⁶) plus two arbitrary factors, so
    // the test pins the actual multiply rather than any hard-coded constant.
    run_unscale_case(&fx, 1.0 / 65536.0, 0x0125_0A01);
    run_unscale_case(&fx, 0.375, 0x0125_0A02);
    run_unscale_case(&fx, 3.5, 0x0125_0A03);
}

// ===========================================================================
// 11. overflow_check  —  INDEPENDENT HOST RE-DERIVATION (any inf/NaN ⇒ flag 1)
//     AMP overflow detection, from `crate::amp::overflow_check_ptx`.
//
//     Same un-validated-PTX gap as `unscale_inplace`. Compiling + launching it
//     exposed three real bugs (all fixed in `amp.rs`): (1) the same special-
//     register name collision, (2) `testp.nan.f32` — not a legal PTX qualifier,
//     it must be `testp.notanumber.f32`, and (3) a tangled grid-stride epilogue
//     that recomputed `idx` from the thread's base every iteration, so it never
//     advanced past `base + stride` (an infinite loop / hang for any launch
//     with grid*block < n). Fixed to a clean `idx += gridDim.x * blockDim.x`.
//
//     The poisoned element is placed PAST the first thread-wave (index > 512)
//     so the grid-stride must genuinely advance to detect it.
// ===========================================================================

fn run_overflow_case(fx: &GpuFixture, seed: u64, poison: Option<(usize, f32)>) -> u32 {
    let n = 4096_usize;

    let mut rng = LcgRng::new(seed);
    let mut data: Vec<f32> = (0..n).map(|_| rng.next_f32() * 8.0 - 4.0).collect();
    if let Some((idx, val)) = poison {
        data[idx] = val;
    }

    let ptx = crate::amp::overflow_check_ptx(fx.sm_num);
    let kernel = load_kernel(&ptx, "overflow_check");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");
    // The host MUST zero-init the flag; the kernel only ever stores 1.
    let d_flag = DeviceBuffer::<u32>::from_host(&[0_u32]).expect("d_flag");

    let block = 128_u32;
    let grid = 4_u32;
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), d_flag.as_device_ptr(), n as u32),
        )
        .expect("launch overflow_check");
    stream.synchronize().expect("sync");

    let mut flag = [0_u32; 1];
    d_flag.copy_to_host(&mut flag).expect("copy flag");
    flag[0]
}

#[test]
fn overflow_check_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // All-finite buffer ⇒ the flag stays 0.
    assert_eq!(
        run_overflow_case(&fx, 0x0FEB_0070, None),
        0,
        "overflow_check flagged an all-finite buffer"
    );
    // +inf far into the buffer (index 3000 ≫ 512 first-wave threads) ⇒ flag 1;
    // this can only be reached if the grid-stride actually advances.
    assert_eq!(
        run_overflow_case(&fx, 0x0FEB_0071, Some((3000, f32::INFINITY))),
        1,
        "overflow_check missed a +inf at index 3000"
    );
    // -inf ⇒ flag 1 (testp.infinite covers both signs).
    assert_eq!(
        run_overflow_case(&fx, 0x0FEB_0072, Some((1234, f32::NEG_INFINITY))),
        1,
        "overflow_check missed a -inf at index 1234"
    );
    // NaN ⇒ flag 1 (exercises the testp.notanumber path specifically).
    assert_eq!(
        run_overflow_case(&fx, 0x0FEB_0073, Some((2077, f32::NAN))),
        1,
        "overflow_check missed a NaN at index 2077"
    );
}
