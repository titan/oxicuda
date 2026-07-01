//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the proven `oxicuda-snn` /
//! `oxicuda-ot` canaries: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! Every kernel in this crate performs a well-defined, branch-free arithmetic
//! task (a tensor contraction, a Givens rotation, a mode-k unfolding, or an
//! identity copy) whose result is fully determined by the kernel's own
//! docstring. None of them have a single dedicated `pub` CPU function that
//! mirrors the *exact* index convention the PTX uses (the crate's CPU paths fuse
//! these ops into larger routines and, for the unfolding, follow a different
//! ordering convention). Each test therefore uses an **independent host
//! re-derivation** of the kernel's *documented* arithmetic as the oracle:
//!
//! * `tensor_contract`, `dmrg_local_apply`, `mpo_apply`, `trotter_step` — the
//!   contraction is re-implemented in plain Rust with the same index formulas;
//!   the GPU `fma.rn` (single rounding) and host `mul`/`add` (two roundings)
//!   agree to a few ulp, so a tight FP32 tolerance still catches any wrong
//!   constant / shift / index while passing the legitimate rounding gap.
//! * `svd_jacobi_step` — the per-row Givens rotation is re-derived exactly;
//!   compared within FP32 tolerance.
//! * `hosvd_unfold` (all 3 modes) and `tt_round` — pure data movement (no
//!   arithmetic), so the GPU output is compared **bit-exact** against the host
//!   re-derivation.
//!
//! The host code is written independently of the JIT-compiled PTX, so a
//! mismatch genuinely indicates a ptxas miscompile or a real PTX bug. Every
//! test skips (returns early) when no CUDA device is present, so the suite stays
//! green on CPU-only machines.

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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug.
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

/// Deterministic signed-`f32` vector in roughly `[-1, 1)` from the crate LCG.
fn signed_vec(rng: &mut LcgRng, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| (rng.next_f64() as f32) * 2.0 - 1.0)
        .collect()
}

/// Deterministic positive-`f32` vector in `[0.1, 1.1)` from the crate LCG.
fn positive_vec(rng: &mut LcgRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| 0.1 + (rng.next_f64() as f32)).collect()
}

// ===========================================================================
// 1. tensor_contract  —  INDEPENDENT HOST RE-DERIVATION
//    c[i,l] = sum_{j,k} a[i,j,k] * b[j,k,l]
// ===========================================================================

#[test]
fn tensor_contract_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_i = 3_usize;
    let n_j = 4_usize;
    let n_k = 2_usize;
    let n_l = 5_usize;

    let mut rng = LcgRng::new(0x7E11_5023);
    let a = signed_vec(&mut rng, n_i * n_j * n_k);
    let b = signed_vec(&mut rng, n_j * n_k * n_l);

    // Independent host re-derivation in the kernel's accumulation order.
    let mut c_host = vec![0.0_f32; n_i * n_l];
    for i in 0..n_i {
        for l in 0..n_l {
            let mut s = 0.0_f32;
            for j in 0..n_j {
                for k in 0..n_k {
                    let av = a[(i * n_j + j) * n_k + k];
                    let bv = b[(j * n_k + k) * n_l + l];
                    s += av * bv;
                }
            }
            c_host[i * n_l + l] = s;
        }
    }

    let ptx = crate::ptx_kernels::tensor_contract_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tensor_contract_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_i * n_l]).expect("d_c");

    // Grid = (ceil(n_l/16), ceil(n_i/16)), Block = (16, 16).
    let block = (16_u32, 16_u32);
    let grid = (grid_1d(n_l as u32, 16), grid_1d(n_i as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                n_i as u32,
                n_j as u32,
                n_k as u32,
                n_l as u32,
            ),
        )
        .expect("launch tensor_contract_kernel");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; n_i * n_l];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");

    let (rel, abs) = worst_diff(&c_gpu, &c_host);
    for k in 0..c_gpu.len() {
        assert!(
            close(c_gpu[k], c_host[k], 1e-4, 1e-5),
            "tensor_contract c[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            c_gpu[k],
            c_host[k]
        );
    }
}

// ===========================================================================
// 2. svd_jacobi_step  —  INDEPENDENT HOST RE-DERIVATION (per-row Givens)
//    [new_p, new_q] = [c*a_p + s*a_q, -s*a_p + c*a_q]
// ===========================================================================

#[test]
fn svd_jacobi_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_rows = 5_usize;
    let n_cols = 4_usize;
    let p = 1_usize;
    let q = 3_usize;
    let angle = 0.6_f32;
    let c = angle.cos();
    let s = angle.sin();

    let mut rng = LcgRng::new(0x53D1_AC0B);
    let a = signed_vec(&mut rng, n_rows * n_cols);

    // Independent host re-derivation: rotate columns p and q in place.
    let mut a_host = a.clone();
    for r in 0..n_rows {
        let ap = a[r * n_cols + p];
        let aq = a[r * n_cols + q];
        a_host[r * n_cols + p] = c * ap + s * aq;
        a_host[r * n_cols + q] = -s * ap + c * aq;
    }

    let ptx = crate::ptx_kernels::svd_jacobi_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "svd_jacobi_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");

    // Block = (32, 1, 1), one thread per row.
    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_rows as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                n_rows as u32,
                n_cols as u32,
                p as u32,
                q as u32,
                c,
                s,
            ),
        )
        .expect("launch svd_jacobi_step_kernel");
    stream.synchronize().expect("sync");

    let mut a_gpu = vec![0.0_f32; n_rows * n_cols];
    d_a.copy_to_host(&mut a_gpu).expect("copy a");

    let (rel, abs) = worst_diff(&a_gpu, &a_host);
    for k in 0..a_gpu.len() {
        assert!(
            close(a_gpu[k], a_host[k], 1e-4, 1e-6),
            "svd_jacobi a[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            a_gpu[k],
            a_host[k]
        );
    }
}

// ===========================================================================
// 3. dmrg_local_apply  —  INDEPENDENT HOST RE-DERIVATION
//    out[a,p1,p2,b] = sum_{p1',p2'} h[p1,p2,p1',p2'] * psi[a,p1',p2',b]
// ===========================================================================

#[test]
fn dmrg_local_apply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let d_l = 2_usize;
    let d_p1 = 2_usize;
    let d_p2 = 2_usize;
    let d_r = 2_usize;
    let total = d_l * d_p1 * d_p2 * d_r;

    let mut rng = LcgRng::new(0x0D33_8150);
    let psi = signed_vec(&mut rng, d_l * d_p1 * d_p2 * d_r);
    let h = signed_vec(&mut rng, d_p1 * d_p2 * d_p1 * d_p2);

    // Independent host re-derivation, matching the kernel's index formulas:
    //   gid       = ((a*d_p1 + p1)*d_p2 + p2)*d_r + b
    //   h index   = ((p1*d_p2 + p2)*d_p1 + p1')*d_p2 + p2'
    //   psi index = ((a*d_p1 + p1')*d_p2 + p2')*d_r + b
    let mut out_host = vec![0.0_f32; total];
    for a in 0..d_l {
        for p1 in 0..d_p1 {
            for p2 in 0..d_p2 {
                for b in 0..d_r {
                    let gid = ((a * d_p1 + p1) * d_p2 + p2) * d_r + b;
                    let mut acc = 0.0_f32;
                    for p1p in 0..d_p1 {
                        for p2p in 0..d_p2 {
                            let h_idx = ((p1 * d_p2 + p2) * d_p1 + p1p) * d_p2 + p2p;
                            let psi_idx = ((a * d_p1 + p1p) * d_p2 + p2p) * d_r + b;
                            acc += h[h_idx] * psi[psi_idx];
                        }
                    }
                    out_host[gid] = acc;
                }
            }
        }
    }

    let ptx = crate::ptx_kernels::dmrg_local_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dmrg_local_apply_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_psi = DeviceBuffer::<f32>::from_host(&psi).expect("d_psi");
    let d_h = DeviceBuffer::<f32>::from_host(&h).expect("d_h");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_psi.as_device_ptr(),
                d_h.as_device_ptr(),
                d_out.as_device_ptr(),
                d_l as u32,
                d_p1 as u32,
                d_p2 as u32,
                d_r as u32,
            ),
        )
        .expect("launch dmrg_local_apply_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "dmrg_local_apply out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 4. mpo_apply  —  INDEPENDENT HOST RE-DERIVATION
//    out[(a,wl),p,(b,wr)] = sum_{p_in} mpo[wl,p,p_in,wr] * mps[a,p_in,b]
// ===========================================================================

#[test]
fn mpo_apply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dl = 2_usize;
    let d = 2_usize;
    let dr = 2_usize;
    let wl = 2_usize;
    let wr = 2_usize;
    let total = dl * wl * d * dr * wr;

    let mut rng = LcgRng::new(0x0390_A771);
    let mps = signed_vec(&mut rng, dl * d * dr);
    let mpo = signed_vec(&mut rng, wl * d * d * wr);

    // Independent host re-derivation, matching the kernel's index formulas:
    //   gid       = ((((a*wl + wl_)*d + p)*dr + b)*wr + wr_
    //   mpo index = (((wl_*d + p)*d + p_in)*wr + wr_
    //   mps index = (a*d + p_in)*dr + b
    let mut out_host = vec![0.0_f32; total];
    for a in 0..dl {
        for wl_ in 0..wl {
            for p in 0..d {
                for b in 0..dr {
                    for wr_ in 0..wr {
                        let gid = ((((a * wl + wl_) * d + p) * dr + b) * wr) + wr_;
                        let mut acc = 0.0_f32;
                        for p_in in 0..d {
                            let mpo_idx = (((wl_ * d + p) * d + p_in) * wr) + wr_;
                            let mps_idx = (a * d + p_in) * dr + b;
                            acc += mpo[mpo_idx] * mps[mps_idx];
                        }
                        out_host[gid] = acc;
                    }
                }
            }
        }
    }

    let ptx = crate::ptx_kernels::mpo_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mpo_apply_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mps = DeviceBuffer::<f32>::from_host(&mps).expect("d_mps");
    let d_mpo = DeviceBuffer::<f32>::from_host(&mpo).expect("d_mpo");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mps.as_device_ptr(),
                d_mpo.as_device_ptr(),
                d_out.as_device_ptr(),
                dl as u32,
                d as u32,
                dr as u32,
                wl as u32,
                wr as u32,
            ),
        )
        .expect("launch mpo_apply_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "mpo_apply out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 5. trotter_step  —  INDEPENDENT HOST RE-DERIVATION
//    out[a,p1,p2,b] = sum_{p1',p2'} gate[p1,p2,p1',p2'] * theta[a,p1',p2',b]
// ===========================================================================

#[test]
fn trotter_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dl = 2_usize;
    let d = 2_usize;
    let dr = 2_usize;
    let total = dl * d * d * dr;

    let mut rng = LcgRng::new(0x7607_7E12);
    let theta = signed_vec(&mut rng, dl * d * d * dr);
    let gate = signed_vec(&mut rng, d * d * d * d);

    // Independent host re-derivation, matching the kernel's index formulas:
    //   gid         = ((a*d + p1)*d + p2)*dr + b
    //   gate index  = (((p1*d + p2)*d + p1')*d + p2'
    //   theta index = ((a*d + p1')*d + p2')*dr + b
    let mut out_host = vec![0.0_f32; total];
    for a in 0..dl {
        for p1 in 0..d {
            for p2 in 0..d {
                for b in 0..dr {
                    let gid = ((a * d + p1) * d + p2) * dr + b;
                    let mut acc = 0.0_f32;
                    for p1p in 0..d {
                        for p2p in 0..d {
                            let gate_idx = ((p1 * d + p2) * d + p1p) * d + p2p;
                            let theta_idx = ((a * d + p1p) * d + p2p) * dr + b;
                            acc += gate[gate_idx] * theta[theta_idx];
                        }
                    }
                    out_host[gid] = acc;
                }
            }
        }
    }

    let ptx = crate::ptx_kernels::trotter_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "trotter_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_theta = DeviceBuffer::<f32>::from_host(&theta).expect("d_theta");
    let d_gate = DeviceBuffer::<f32>::from_host(&gate).expect("d_gate");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_theta.as_device_ptr(),
                d_gate.as_device_ptr(),
                d_out.as_device_ptr(),
                dl as u32,
                d as u32,
                dr as u32,
            ),
        )
        .expect("launch trotter_step_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "trotter_step out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 6. hosvd_unfold  —  INDEPENDENT HOST RE-DERIVATION (bit-exact data movement)
//    mode 0: out[i*(d1*d2) + j*d2 + k] = a[i,j,k]
//    mode 1: out[j*(d0*d2) + i*d2 + k] = a[i,j,k]
//    mode 2: out[k*(d0*d1) + i*d1 + j] = a[i,j,k]
// ===========================================================================

fn run_hosvd_unfold_case(fx: &GpuFixture, mode: u32) {
    let d0 = 2_usize;
    let d1 = 3_usize;
    let d2 = 4_usize;
    let total = d0 * d1 * d2;

    let mut rng = LcgRng::new(0x405D_0000 ^ u64::from(mode));
    let a = signed_vec(&mut rng, total);

    // Independent host re-derivation of the documented unfolding map.
    let mut out_host = vec![0.0_f32; total];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let src = (i * d1 + j) * d2 + k;
                let dst = match mode {
                    0 => i * (d1 * d2) + j * d2 + k,
                    1 => j * (d0 * d2) + i * d2 + k,
                    _ => k * (d0 * d1) + i * d1 + j,
                };
                out_host[dst] = a[src];
            }
        }
    }

    let ptx = crate::ptx_kernels::hosvd_unfold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hosvd_unfold_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_out.as_device_ptr(),
                d0 as u32,
                d1 as u32,
                d2 as u32,
                mode,
            ),
        )
        .expect("launch hosvd_unfold_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Pure data movement (no arithmetic) ⇒ compare bit-exact.
    for k in 0..out_gpu.len() {
        assert_eq!(
            out_gpu[k].to_bits(),
            out_host[k].to_bits(),
            "hosvd_unfold mode {mode} out[{k}] mismatch: gpu={} host={}",
            out_gpu[k],
            out_host[k]
        );
    }
}

#[test]
fn hosvd_unfold_mode0_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_hosvd_unfold_case(&fx, 0);
}

#[test]
fn hosvd_unfold_mode1_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_hosvd_unfold_case(&fx, 1);
}

#[test]
fn hosvd_unfold_mode2_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_hosvd_unfold_case(&fx, 2);
}

// ===========================================================================
// 7. tt_round  —  INDEPENDENT HOST RE-DERIVATION (bit-exact identity copy)
//    core_out[g] = core_in[g]  for g in [0, r_l*n*r_r)
// ===========================================================================

#[test]
fn tt_round_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let r_l = 2_usize;
    let n = 3_usize;
    let r_r = 4_usize;
    let total = r_l * n * r_r;

    let mut rng = LcgRng::new(0x0770_C09E);
    let core_in = positive_vec(&mut rng, total);

    // Independent host re-derivation: an exact identity copy.
    let core_host = core_in.clone();

    let ptx = crate::ptx_kernels::tt_round_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tt_round_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&core_in).expect("d_in");
    // Initialise the output to a sentinel distinct from the input so a no-op
    // kernel would be caught (the assertion below requires a real copy).
    let d_out = DeviceBuffer::<f32>::from_host(&vec![-9.0_f32; total]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                r_l as u32,
                n as u32,
                r_r as u32,
            ),
        )
        .expect("launch tt_round_kernel");
    stream.synchronize().expect("sync");

    let mut core_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut core_gpu).expect("copy out");

    // Identity copy ⇒ bit-exact.
    for k in 0..core_gpu.len() {
        assert_eq!(
            core_gpu[k].to_bits(),
            core_host[k].to_bits(),
            "tt_round core_out[{k}] mismatch: gpu={} host={}",
            core_gpu[k],
            core_host[k]
        );
    }
}
