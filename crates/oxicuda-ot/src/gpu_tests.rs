//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-snn` canary: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars are
//! passed as the matching Rust scalar (`.param .u32` / `.param .f32`), in the
//! kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   `sinkhorn_step_kernel` ↔ [`crate::sinkhorn::log_sinkhorn::log_sinkhorn_step_row`],
//!   `cost_matrix_kernel` (L2²) ↔ `2 ·`[`crate::sinkhorn::stabilised_sinkhorn::sq_euclidean_cost`]
//!   (that helper carries a `0.5` factor the kernel omits, so the kernel must
//!   equal twice it), and `transport_apply_kernel` ↔
//!   [`crate::domain::mapping::barycentric_map`].
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine with no standalone `pub fn`, so the oracle is an independent Rust
//!   re-implementation of the kernel's *documented* arithmetic:
//!   `cost_matrix_kernel` (L1), `sliced_proj_kernel`, `gromov_grad_kernel`,
//!   `unbalanced_step_kernel` (matches the row half-update inside
//!   [`crate::unbalanced::unbalanced_ot`]), and `barycenter_update_kernel`
//!   (matches the support update inside
//!   [`crate::barycenter::free_support_barycenter`]). These still genuinely
//!   fail if ptxas miscompiles or the PTX has a wrong constant / shift / index,
//!   because the host code is independent of the JIT-compiled PTX.
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

/// Natural, max-stabilised log-sum-exp matching the crate's CPU reference
/// (`max + ln Σ exp(x − max)`); `NEG_INFINITY` for an empty / all-`-inf` slice.
fn host_logsumexp(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &x in slice {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f32;
    for &x in slice {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

// ===========================================================================
// 1. sinkhorn_step  —  CRATE ORACLE (sinkhorn::log_sinkhorn::log_sinkhorn_step_row)
// ===========================================================================

#[test]
fn sinkhorn_step_matches_cpu() {
    use crate::sinkhorn::log_sinkhorn::log_sinkhorn_step_row;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Well-conditioned single Sinkhorn row update.
    //
    // Conditioning rationale: with eps = 1.0 and costs / potentials of order 1,
    // every exponent z_j = (v_j − C_ij)/eps is O(1), so after subtracting the
    // per-row max the exponentials lie in (0, 1] and the sum lies in [1, n] — no
    // overflow / underflow on either side. A uniform marginal a_i = 1/m gives
    // log_a = −ln m ≈ −2.08, and the row update u_i = ε·log a_i − ε·LSE lands
    // comfortably in roughly [−4.7, −2.6], i.e. |u_i| ≫ 0, so the relative
    // comparison is never evaluated near a zero crossing.
    let m = 8_usize;
    let n = 8_usize;
    let eps = 1.0_f32;

    let mut rng = LcgRng::new(0x0007_C057);
    let c: Vec<f32> = (0..m * n).map(|_| rng.next_f32()).collect(); // [0, 1)
    let v: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect(); // [-0.5, 0.5)
    let log_a: Vec<f32> = vec![(1.0_f32 / m as f32).ln(); m]; // uniform marginal
    let log_b = vec![0.0_f32; n]; // declared but unused by the kernel

    // ---- CPU reference ----
    let mut u_cpu = vec![0.0_f32; m];
    log_sinkhorn_step_row(&c, &log_a, &mut u_cpu, &v, eps, m, n).expect("cpu sinkhorn step");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::sinkhorn_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sinkhorn_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c = DeviceBuffer::<f32>::from_host(&c).expect("d_c");
    let d_log_a = DeviceBuffer::<f32>::from_host(&log_a).expect("d_log_a");
    let d_log_b = DeviceBuffer::<f32>::from_host(&log_b).expect("d_log_b");
    let d_u = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m]).expect("d_u");
    let d_v = DeviceBuffer::<f32>::from_host(&v).expect("d_v");

    // Grid = (m, 1, 1), block = (1, 1, 1): one block per row of u.
    let params = LaunchParams::new(m as u32, 1u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c.as_device_ptr(),
                d_log_a.as_device_ptr(),
                d_log_b.as_device_ptr(),
                d_u.as_device_ptr(),
                d_v.as_device_ptr(),
                m as u32,
                n as u32,
                eps,
            ),
        )
        .expect("launch sinkhorn_step_kernel");
    stream.synchronize().expect("sync");

    let mut u_gpu = vec![0.0_f32; m];
    d_u.copy_to_host(&mut u_gpu).expect("copy u");

    // Tolerance justification: after the (fixed) base-conversion, the GPU LSE
    // uses `ex2.approx.f32` (~2 ulp) and `lg2.approx.f32` (~2–3 ulp) while the
    // CPU uses libm `exp`/`ln` (<1 ulp). With the shared max-subtraction the
    // exponentials are in (0, 1], so the LSE term carries only a few-ulp
    // relative error (~1e-6) which propagates to |u| ≈ 3 as ~1e-6 relative —
    // three orders of magnitude inside the 1e-4 bound, yet 1e-4 still flags any
    // gross formula error (e.g. a base-2 vs base-e logsumexp, ~19 % here).
    let (rel, abs) = worst_diff(&u_gpu, &u_cpu);
    for i in 0..m {
        assert!(
            close(u_gpu[i], u_cpu[i], 1e-4, 1e-6),
            "sinkhorn u[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            u_gpu[i],
            u_cpu[i]
        );
    }
}

// ===========================================================================
// 2. cost_matrix  —  CRATE ORACLE (L2² = 2·sq_euclidean_cost) + host re-derivation (L1)
// ===========================================================================

fn run_cost_matrix_case(fx: &GpuFixture, mode: u32) {
    let m = 8_usize;
    let n = 8_usize;
    let dim = 4_usize;

    let mut rng = LcgRng::new(0xC057_0007 ^ u64::from(mode));
    let x: Vec<f32> = (0..m * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let y: Vec<f32> = (0..n * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Independent host re-derivation of the documented per-element math, in the
    // kernel's accumulation order (d = 0..dim), so any FP difference is bounded
    // by a handful of ulp.
    let mut c_host = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for d in 0..dim {
                let diff = x[i * dim + d] - y[j * dim + d];
                s += if mode == 1 { diff.abs() } else { diff * diff };
            }
            c_host[i * n + j] = s;
        }
    }

    let ptx = crate::ptx_kernels::cost_matrix_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cost_matrix_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m * n]).expect("d_c");

    // Grid = (m, n, 1), block = (1, 1, 1): one block per (i, j) cell.
    let params = LaunchParams::new((m as u32, n as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                d_c.as_device_ptr(),
                m as u32,
                n as u32,
                dim as u32,
                mode,
            ),
        )
        .expect("launch cost_matrix_kernel");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; m * n];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");

    let (rel, abs) = worst_diff(&c_gpu, &c_host);
    for k in 0..c_gpu.len() {
        assert!(
            close(c_gpu[k], c_host[k], 1e-5, 1e-5),
            "cost_matrix mode {mode} c[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            c_gpu[k],
            c_host[k]
        );
    }

    // Extra CRATE cross-check for L2²: the only public point-set cost helper,
    // `sq_euclidean_cost`, multiplies the squared distance by 0.5, so the kernel
    // (which omits that factor) must equal exactly twice it. The 0.5 and 2.0
    // scalings are exact powers of two, so this agrees to a few ulp.
    if mode == 2 {
        let c_half = crate::sinkhorn::stabilised_sinkhorn::sq_euclidean_cost(&x, &y, m, n, dim)
            .expect("sq_euclidean_cost");
        for k in 0..c_gpu.len() {
            let expected = 2.0_f32 * c_half[k];
            assert!(
                close(c_gpu[k], expected, 1e-5, 1e-5),
                "cost_matrix L2² c[{k}] vs 2·sq_euclidean_cost mismatch: gpu={} crate={}",
                c_gpu[k],
                expected
            );
        }
    }
}

#[test]
fn cost_matrix_l2sq_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_cost_matrix_case(&fx, 2);
}

#[test]
fn cost_matrix_l1_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_cost_matrix_case(&fx, 1);
}

// ===========================================================================
// 3. transport_apply  —  CRATE ORACLE (domain::mapping::barycentric_map)
// ===========================================================================

#[test]
fn transport_apply_matches_cpu() {
    use crate::domain::mapping::barycentric_map;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 4_usize;
    let n = 8_usize;
    let dim = 3_usize;

    // Plan entries in [0.1, 1.0) ⇒ every row sum ≥ 0.8 ≫ 1e-12, so the kernel's
    // `row_sum + 1e-12` denominator and `barycentric_map`'s exact `1/row_sum`
    // agree to ~1e-12 relative, and the CPU's degenerate-row fallback (mean of
    // y) never triggers. Targets y in [0.5, 1.5) keep every mapped coordinate a
    // positive convex combination, away from any zero crossing.
    let mut rng = LcgRng::new(0x7A_43_05);
    let plan: Vec<f32> = (0..m * n).map(|_| 0.1 + 0.9 * rng.next_f32()).collect();
    let y: Vec<f32> = (0..n * dim).map(|_| 0.5 + rng.next_f32()).collect();

    // ---- CPU reference ----
    let out_cpu = barycentric_map(&plan, &y, m, n, dim).expect("barycentric_map");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::transport_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "transport_apply_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_plan = DeviceBuffer::<f32>::from_host(&plan).expect("d_plan");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m * dim]).expect("d_out");

    // Grid = (m, 1, 1), block = (1, 1, 1): one block per source row.
    let params = LaunchParams::new(m as u32, 1u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_plan.as_device_ptr(),
                d_y.as_device_ptr(),
                d_out.as_device_ptr(),
                m as u32,
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch transport_apply_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; m * dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tolerance: the kernel accumulates Σ_j P·y then divides once, while the CPU
    // scales each term by 1/row_sum and sums; the reorder over n = 8 positive
    // terms is bounded by ~n ulp (~5e-7 relative), plus the negligible 1e-12
    // denominator offset. 1e-5 relative is a comfortable, still-meaningful bound.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-5, 1e-6),
            "transport_apply out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ===========================================================================
// 4. sliced_proj  —  INDEPENDENT HOST RE-DERIVATION (per-direction dot product)
// ===========================================================================

#[test]
fn sliced_proj_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_proj = 4_usize;
    let n = 8_usize;
    let dim = 5_usize;

    // Unit (normalised) directions and positive samples keep each projection a
    // sum of positive terms, away from catastrophic cancellation, so a tight
    // tolerance stays valid while still catching any wrong index / constant.
    let mut rng = LcgRng::new(0x5_71C3D);
    let mut theta = vec![0.0_f32; n_proj * dim];
    for k in 0..n_proj {
        let mut norm = 0.0_f32;
        for d in 0..dim {
            let t = rng.next_f32() + 0.1; // (0.1, 1.1): strictly positive
            theta[k * dim + d] = t;
            norm += t * t;
        }
        let inv = 1.0_f32 / norm.sqrt();
        for d in 0..dim {
            theta[k * dim + d] *= inv;
        }
    }
    let x: Vec<f32> = (0..n * dim).map(|_| 0.2 + rng.next_f32()).collect(); // [0.2, 1.2)

    // Independent host re-derivation: proj[k, i] = Σ_d theta[k, d] · x[i, d].
    let mut proj_host = vec![0.0_f32; n_proj * n];
    for k in 0..n_proj {
        for i in 0..n {
            let mut s = 0.0_f32;
            for d in 0..dim {
                s += theta[k * dim + d] * x[i * dim + d];
            }
            proj_host[k * n + i] = s;
        }
    }

    let ptx = crate::ptx_kernels::sliced_proj_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sliced_proj_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_theta = DeviceBuffer::<f32>::from_host(&theta).expect("d_theta");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_proj = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_proj * n]).expect("d_proj");

    // Grid = (n_proj, n, 1), block = (1, 1, 1): one block per (direction, sample).
    let params = LaunchParams::new((n_proj as u32, n as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_theta.as_device_ptr(),
                d_x.as_device_ptr(),
                d_proj.as_device_ptr(),
                n_proj as u32,
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch sliced_proj_kernel");
    stream.synchronize().expect("sync");

    let mut proj_gpu = vec![0.0_f32; n_proj * n];
    d_proj.copy_to_host(&mut proj_gpu).expect("copy proj");

    // The GPU uses `fma.rn` (one rounding/term); the host does mul+add (two).
    // Over dim = 5 positive terms the divergence is a few ulp (~1e-6 relative).
    let (rel, abs) = worst_diff(&proj_gpu, &proj_host);
    for k in 0..proj_gpu.len() {
        assert!(
            close(proj_gpu[k], proj_host[k], 1e-5, 1e-6),
            "sliced_proj proj[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            proj_gpu[k],
            proj_host[k]
        );
    }
}

// ===========================================================================
// 5. gromov_grad  —  INDEPENDENT HOST RE-DERIVATION (-2·Σ_kl C1·T·C2)
// ===========================================================================

#[test]
fn gromov_grad_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 6_usize;
    let n = 6_usize;

    // Positive structural matrices and plan ⇒ every G[i,j] = −2·Σ(positive) is
    // strictly negative and bounded away from zero, so no cancellation.
    let mut rng = LcgRng::new(0x0067_0307);
    let c1: Vec<f32> = (0..m * m).map(|_| rng.next_f32() + 0.1).collect();
    let c2: Vec<f32> = (0..n * n).map(|_| rng.next_f32() + 0.1).collect();
    let t: Vec<f32> = (0..m * n).map(|_| rng.next_f32() + 0.1).collect();

    // Independent host re-derivation of the documented quartic contraction
    // G[i,j] = −2·Σ_{k,l} C1[i,k]·T[k,l]·C2[j,l] with the kernel's row-major
    // indexing (C1 is m×m, T is m×n, C2 is n×n).
    let mut g_host = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for k in 0..m {
                let c1_ik = c1[i * m + k];
                for l in 0..n {
                    acc += c1_ik * t[k * n + l] * c2[j * n + l];
                }
            }
            g_host[i * n + j] = -2.0_f32 * acc;
        }
    }

    let ptx = crate::ptx_kernels::gromov_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gromov_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c1 = DeviceBuffer::<f32>::from_host(&c1).expect("d_c1");
    let d_c2 = DeviceBuffer::<f32>::from_host(&c2).expect("d_c2");
    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_g = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m * n]).expect("d_g");

    // Grid = (m, n, 1), block = (1, 1, 1): one block per (i, j) gradient entry.
    let params = LaunchParams::new((m as u32, n as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c1.as_device_ptr(),
                d_c2.as_device_ptr(),
                d_t.as_device_ptr(),
                d_g.as_device_ptr(),
                m as u32,
                n as u32,
            ),
        )
        .expect("launch gromov_grad_kernel");
    stream.synchronize().expect("sync");

    let mut g_gpu = vec![0.0_f32; m * n];
    d_g.copy_to_host(&mut g_gpu).expect("copy g");

    // The kernel fuses the inner product with `fma.rn`; the host re-derivation
    // uses plain mul/add in a different grouping. Over m·n = 36 positive terms
    // the relative divergence is ~1e-5; 1e-4 is a comfortable, meaningful bound.
    let (rel, abs) = worst_diff(&g_gpu, &g_host);
    for k in 0..g_gpu.len() {
        assert!(
            close(g_gpu[k], g_host[k], 1e-4, 1e-4),
            "gromov_grad g[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            g_gpu[k],
            g_host[k]
        );
    }
}

// ===========================================================================
// 6. unbalanced_step  —  INDEPENDENT HOST RE-DERIVATION (matches unbalanced_ot row update)
// ===========================================================================

#[test]
fn unbalanced_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Same well-conditioning argument as `sinkhorn_step_matches_cpu`: eps = 1.0,
    // O(1) costs/potentials, uniform log_a. The KL contraction factor
    // tau/(tau+eps) = 0.5 scales the row update; f stays comfortably negative.
    let m = 8_usize;
    let n = 8_usize;
    let eps = 1.0_f32;
    let tau = 1.0_f32;
    let factor = tau / (tau + eps);

    let mut rng = LcgRng::new(0x07BA_1A7C);
    let c: Vec<f32> = (0..m * n).map(|_| rng.next_f32()).collect();
    let g: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
    let log_a: Vec<f32> = vec![(1.0_f32 / m as f32).ln(); m];

    // Independent host re-derivation of the documented KL-relaxed Sinkhorn
    // half-step: f_i = (τ/(τ+ε)) · (ε·log a_i − ε·LSE_j((g_j − C_ij)/ε)). This is
    // exactly the row update inside `unbalanced_ot`, computed independently of
    // the JIT-compiled PTX.
    let mut f_host = vec![0.0_f32; m];
    let mut buf = vec![0.0_f32; n];
    for i in 0..m {
        let row_off = i * n;
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = (g[j] - c[row_off + j]) / eps;
        }
        let lse = host_logsumexp(&buf);
        f_host[i] = factor * (eps * log_a[i] - eps * lse);
    }

    let ptx = crate::ptx_kernels::unbalanced_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "unbalanced_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c = DeviceBuffer::<f32>::from_host(&c).expect("d_c");
    let d_log_a = DeviceBuffer::<f32>::from_host(&log_a).expect("d_log_a");
    let d_g = DeviceBuffer::<f32>::from_host(&g).expect("d_g");
    let d_f = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m]).expect("d_f");

    // Grid = (m, 1, 1), block = (1, 1, 1): one block per row of f.
    let params = LaunchParams::new(m as u32, 1u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c.as_device_ptr(),
                d_log_a.as_device_ptr(),
                d_g.as_device_ptr(),
                d_f.as_device_ptr(),
                m as u32,
                n as u32,
                eps,
                tau,
            ),
        )
        .expect("launch unbalanced_step_kernel");
    stream.synchronize().expect("sync");

    let mut f_gpu = vec![0.0_f32; m];
    d_f.copy_to_host(&mut f_gpu).expect("copy f");

    // Same LSE tolerance argument as the balanced step (ex2/lg2 approx ~few ulp,
    // |f| ≈ 1.5), with an extra exact multiply by the 0.5 contraction factor.
    let (rel, abs) = worst_diff(&f_gpu, &f_host);
    for i in 0..m {
        assert!(
            close(f_gpu[i], f_host[i], 1e-4, 1e-6),
            "unbalanced f[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            f_gpu[i],
            f_host[i]
        );
    }
}

// ===========================================================================
// 7. barycenter_update  —  INDEPENDENT HOST RE-DERIVATION (weighted barycentric mean)
// ===========================================================================

#[test]
fn barycenter_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k_count = 2_usize;
    let m = 4_usize;
    let n_k = 5_usize;
    let dim = 3_usize;

    // Positive plans (each (k,i) row sum ≥ 0.5 ≫ 1e-12) and positive supports ⇒
    // the kernel's `+1e-12` denominators are negligible and every output is a
    // positive weighted mean, away from zero.
    let mut rng = LcgRng::new(0x0BA4_7C03);
    let t: Vec<f32> = (0..k_count * m * n_k)
        .map(|_| 0.1 + 0.9 * rng.next_f32())
        .collect();
    let x: Vec<f32> = (0..k_count * n_k * dim)
        .map(|_| 0.5 + rng.next_f32())
        .collect();
    let lambda: Vec<f32> = vec![0.6_f32, 0.4_f32];

    // Independent host re-derivation matching the kernel's documented arithmetic
    // (including its `+1e-12` floors on the row sum and the weight normaliser):
    //   y[i,d] = (Σ_k λ_k · (Σ_j T_k[i,j]·x_k[j,d]) / (Σ_j T_k[i,j] + 1e-12))
    //            / (Σ_k λ_k + 1e-12).
    let eps_d = 1e-12_f32;
    let mut y_host = vec![0.0_f32; m * dim];
    for i in 0..m {
        for d in 0..dim {
            let mut acc = 0.0_f32;
            let mut wnorm = 0.0_f32;
            for k in 0..k_count {
                let lam = lambda[k];
                let mut sum = 0.0_f32;
                let mut row_t = 0.0_f32;
                for j in 0..n_k {
                    let t_val = t[(k * m + i) * n_k + j];
                    let x_val = x[(k * n_k + j) * dim + d];
                    sum += t_val * x_val;
                    row_t += t_val;
                }
                row_t += eps_d;
                sum /= row_t;
                acc += lam * sum;
                wnorm += lam;
            }
            wnorm += eps_d;
            y_host[i * dim + d] = acc / wnorm;
        }
    }

    let ptx = crate::ptx_kernels::barycenter_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "barycenter_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_lambda = DeviceBuffer::<f32>::from_host(&lambda).expect("d_lambda");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m * dim]).expect("d_y");

    // Grid = (m, dim, 1), block = (1, 1, 1): ctaid.x = i, ctaid.y = d.
    let params = LaunchParams::new((m as u32, dim as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_t.as_device_ptr(),
                d_x.as_device_ptr(),
                d_lambda.as_device_ptr(),
                d_y.as_device_ptr(),
                m as u32,
                n_k as u32,
                k_count as u32,
                dim as u32,
            ),
        )
        .expect("launch barycenter_update_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; m * dim];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    // Mixed fma/div reductions over k·n_k = 10 positive terms: a few-ulp
    // divergence, ~1e-6 relative; 1e-4 is comfortable and still meaningful.
    let (rel, abs) = worst_diff(&y_gpu, &y_host);
    for k in 0..y_gpu.len() {
        assert!(
            close(y_gpu[k], y_host[k], 1e-4, 1e-6),
            "barycenter_update y[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[k],
            y_host[k]
        );
    }
}
