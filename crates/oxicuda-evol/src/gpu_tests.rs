//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to a CPU reference. The
//! launch ABI mirrors the working `oxicuda-snn` / `oxicuda-ot` canaries: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar (`.param .u32` / `.param .f64` / `.param .u64`), in the
//! kernel's declared parameter order. All EA kernels here operate in `f64`.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP64 tolerance to a `pub`
//!   CPU function the kernel mirrors: `fitness_eval_kernel` ↔
//!   [`crate::sphere`] (sum of squares).
//! * **Independent host re-derivation** — the kernel embeds its own counter /
//!   LCG schedule that differs from the crate's sequential `LcgRng`, so the
//!   oracle is an independent Rust re-implementation of the kernel's *documented*
//!   arithmetic (the inline Knuth-MMIX LCG + the documented update), computed
//!   independently of the JIT-compiled PTX: `tournament_select_kernel`,
//!   `pso_update_kernel`, `de_mutate_kernel`, `cmaes_sample_kernel`,
//!   `nsga_crowding_kernel`, and the Box–Muller `gaussian_mutate_kernel`.
//!
//! A `lg2.approx.f64` bug in `gaussian_mutate_kernel` (an instruction ptxas
//! rejects — `.approx` has no `.f64` form) was found and fixed; see the test.
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

// --- Inline Knuth-MMIX LCG re-derivation (matches the PTX immediates) -------

/// Knuth MMIX multiplier, identical to the PTX immediate `6364136223846793005`.
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
/// Knuth MMIX increment, identical to the PTX immediate `1442695040888963407`.
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

/// One LCG advance: `state = state * MUL + ADD` (wrapping `u64` math).
fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD)
}

/// Uniform `[0, 1)` from a 64-bit state via the top 53 bits, exactly matching
/// the PTX `shr.u64 r, state, 11; cvt.rn.f64.u64; mul.rn.f64 r, 0d3CA0…` (×2⁻⁵³).
fn unif_hi(state: u64) -> f64 {
    ((state >> 11) as f64) / ((1_u64 << 53) as f64)
}

// ===========================================================================
// 1. fitness_eval  —  CRATE ORACLE (crate::sphere, sum of squares)
// ===========================================================================

#[test]
fn fitness_eval_matches_sphere() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_dims = 6_usize;
    let pop_size = 64_usize;

    let mut rng = LcgRng::new(0x0F17_0E55);
    let x: Vec<f64> = (0..pop_size * n_dims)
        .map(|_| rng.next_f64() * 2.0 - 1.0)
        .collect();

    // CPU reference: the crate's own sphere fitness for each individual.
    let fit_cpu: Vec<f64> = (0..pop_size)
        .map(|i| crate::sphere(&x[i * n_dims..(i + 1) * n_dims]))
        .collect();

    let ptx = crate::ptx_kernels::fitness_eval_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fitness_eval_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f64>::from_host(&x).expect("d_x");
    let d_fit = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; pop_size]).expect("d_fit");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(pop_size as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_fit.as_device_ptr(),
                n_dims as u32,
                pop_size as u32,
            ),
        )
        .expect("launch fitness_eval_kernel");
    stream.synchronize().expect("sync");

    let mut fit_gpu = vec![0.0_f64; pop_size];
    d_fit.copy_to_host(&mut fit_gpu).expect("copy fit");

    // GPU uses `fma.rn.f64` (single rounding/term); the crate sums `xi*xi` (two
    // roundings). Over 6 terms the divergence is a few ulp; 1e-12 relative is
    // generous yet still flags a gross indexing/formula error.
    let (rel, abs) = worst_diff(&fit_gpu, &fit_cpu);
    for i in 0..pop_size {
        assert!(
            close(fit_gpu[i], fit_cpu[i], 1e-12, 1e-12),
            "fitness_eval[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            fit_gpu[i],
            fit_cpu[i]
        );
    }
}

// ===========================================================================
// 2. tournament_select  —  INDEPENDENT HOST RE-DERIVATION (inline LCG + k=2)
// ===========================================================================

#[test]
fn tournament_select_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let pop_size = 32_usize;
    let n_select = 32_usize;
    let seed = 0x57DB_C0DE_u64;

    // Distinct, well-separated fitness so the `<=` tie path is irrelevant and the
    // chosen winner index is unambiguous.
    let mut rng = LcgRng::new(0xF17_BEEF);
    let fitness: Vec<f64> = (0..pop_size).map(|_| rng.next_f64() * 100.0).collect();

    // Host re-derivation of the kernel: per thread t, state = seed ^ t; advance
    // once → candidate a = (state as u32) % pop; advance again → candidate b;
    // winner = lower-fitness candidate (ties → a).
    let mut sel_host = vec![0_u32; n_select];
    for (t, slot) in sel_host.iter_mut().enumerate() {
        let s1 = lcg_step(seed ^ t as u64);
        let cand_a = (s1 as u32) % pop_size as u32;
        let s2 = lcg_step(s1);
        let cand_b = (s2 as u32) % pop_size as u32;
        *slot = if fitness[cand_a as usize] <= fitness[cand_b as usize] {
            cand_a
        } else {
            cand_b
        };
    }

    let ptx = crate::ptx_kernels::tournament_select_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tournament_select_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_fit = DeviceBuffer::<f64>::from_host(&fitness).expect("d_fit");
    let d_sel = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_select]).expect("d_sel");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_select as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_fit.as_device_ptr(),
                d_sel.as_device_ptr(),
                pop_size as u32,
                n_select as u32,
                seed,
            ),
        )
        .expect("launch tournament_select_kernel");
    stream.synchronize().expect("sync");

    let mut sel_gpu = vec![0_u32; n_select];
    d_sel.copy_to_host(&mut sel_gpu).expect("copy sel");

    for t in 0..n_select {
        assert_eq!(
            sel_gpu[t], sel_host[t],
            "tournament_select[{t}] mismatch: gpu={} host={} \
             (fit[gpu]={}, fit[host]={})",
            sel_gpu[t], sel_host[t], fitness[sel_gpu[t] as usize], fitness[sel_host[t] as usize]
        );
    }
}

// ===========================================================================
// 3. gaussian_mutate  —  INDEPENDENT HOST RE-DERIVATION (Box–Muller)
// ===========================================================================
//
// PTX BUG FOUND AND FIXED: the original kernel computed `ln u1` with
// `lg2.approx.f64` — but `lg2.approx` exists ONLY for `.f32`, so ptxas rejected
// the PTX ("Unexpected instruction types specified for 'lg2'") and the kernel
// had never run on any GPU. Fix in ptx_kernels.rs: narrow `u1` to f32, take
// `lg2.approx.f32`, widen back, then `× ln 2`. The two tests below now both
// load AND validate the arithmetic.

/// Box–Muller test with `p_mut = 0`: the per-gene gate `u_gate >= p_mut` is
/// always true (u_gate ∈ [0,1) ≥ 0), so the kernel mutates nothing and the
/// genome must be byte-for-byte unchanged. Also proves the (fixed) PTX loads.
#[test]
fn gaussian_mutate_zero_prob_is_identity() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let sigma = 0.5_f64;
    let p_mut = 0.0_f64;
    let seed = 0xBADC_0FFEE_u64;

    let mut rng = LcgRng::new(0x6E11_3201);
    let genome0: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();

    let ptx = crate::ptx_kernels::gaussian_mutate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gaussian_mutate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_g = DeviceBuffer::<f64>::from_host(&genome0).expect("d_g");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_g.as_device_ptr(), n as u32, sigma, p_mut, seed),
        )
        .expect("launch gaussian_mutate_kernel");
    stream.synchronize().expect("sync");

    let mut genome_gpu = vec![0.0_f64; n];
    d_g.copy_to_host(&mut genome_gpu).expect("copy genome");

    for k in 0..n {
        assert_eq!(
            genome_gpu[k].to_bits(),
            genome0[k].to_bits(),
            "gaussian_mutate p_mut=0: gene[{k}] changed {} -> {}",
            genome0[k],
            genome_gpu[k]
        );
    }
}

/// Box–Muller test with `p_mut = 1`: every gene mutates, and the applied delta
/// must equal the mathematically-correct `sigma · sqrt(-2 ln u1) · cos(2π u2)`
/// for the (bit-exactly re-derived) per-gene uniforms `u1, u2`.
///
/// Tolerance rationale: the inline LCG + the `×2⁻⁵³` uniform scaling are
/// bit-exact integer/exact-power-of-two operations, so `u1, u2` are reproduced
/// exactly. The only GPU approximations are the f32 `lg2.approx` (≈3-ulp of the
/// log result) and the degree-10/11 octant-reduced cos/sin Taylor (<1.2e-10).
/// Propagating the f32 log through `r = sqrt(-2 ln u1)` bounds the absolute
/// error of `z` at ≈2e-6 even for `u1 ≈ 2⁻⁵³`, so an absolute floor of 5e-3 is
/// orders of magnitude clear of correct output yet still catches the base-2 /
/// missing-`ln2` class of bug (≈20–44 % error) and any gross scale mistake.
#[test]
fn gaussian_mutate_box_muller_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 2048_usize;
    let sigma = 0.5_f64;
    let p_mut = 1.0_f64;
    let seed = 0x0B0C_1234_u64;

    let genome0 = vec![0.0_f64; n];

    // Host re-derivation: replicate the inline LCG draw schedule (u_gate, u1, u2)
    // then the documented Box–Muller delta. With p_mut = 1 the gate never skips.
    let two_pi = std::f64::consts::TAU;
    let eps = (2.0_f64).powi(-53); // PTX clamp constant 0d3CA0…
    let mut delta_host = vec![0.0_f64; n];
    for (k, slot) in delta_host.iter_mut().enumerate() {
        let s1 = lcg_step(seed ^ k as u64); // u_gate draw (unused when p_mut=1)
        let _u_gate = unif_hi(s1);
        let s2 = lcg_step(s1);
        let u1 = unif_hi(s2).max(eps);
        let s3 = lcg_step(s2);
        let u2 = unif_hi(s3);
        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
        *slot = sigma * z;
    }

    let ptx = crate::ptx_kernels::gaussian_mutate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gaussian_mutate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_g = DeviceBuffer::<f64>::from_host(&genome0).expect("d_g");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_g.as_device_ptr(), n as u32, sigma, p_mut, seed),
        )
        .expect("launch gaussian_mutate_kernel");
    stream.synchronize().expect("sync");

    let mut genome_gpu = vec![0.0_f64; n];
    d_g.copy_to_host(&mut genome_gpu).expect("copy genome");

    // delta_gpu = genome_gpu - genome0 (genome0 is all zero, but keep it explicit).
    let delta_gpu: Vec<f64> = genome_gpu
        .iter()
        .zip(&genome0)
        .map(|(g, g0)| g - g0)
        .collect();

    // Every gene must actually have been mutated (no all-zero stub slipping by).
    let mutated = delta_gpu.iter().filter(|&&d| d != 0.0).count();
    assert!(
        mutated >= n - 2,
        "gaussian_mutate p_mut=1: only {mutated}/{n} genes mutated"
    );
    for &d in &delta_gpu {
        assert!(
            d.is_finite(),
            "gaussian_mutate produced non-finite delta {d}"
        );
    }

    let (rel, abs) = worst_diff(&delta_gpu, &delta_host);
    for k in 0..n {
        assert!(
            close(delta_gpu[k], delta_host[k], 5e-3, 5e-3),
            "gaussian_mutate delta[{k}] mismatch: gpu={} host={} \
             (worst rel={rel:e} abs={abs:e})",
            delta_gpu[k],
            delta_host[k]
        );
    }
}

// ===========================================================================
// 4. nsga_crowding  —  INDEPENDENT HOST RE-DERIVATION (boundary skip + atomic)
// ===========================================================================

#[test]
fn nsga_crowding_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let obj_range = 4.0_f64;

    let mut rng = LcgRng::new(0xC0FF_EE42);
    let sorted_obj: Vec<f64> = (0..n).map(|_| rng.next_f64() * obj_range).collect();
    // Non-zero initial crowding so the boundary "unchanged" check is meaningful.
    let crowd_init: Vec<f64> = vec![0.5_f64; n];

    // Host re-derivation: interior i (1 <= i <= n-2) gets
    // crowd[i] += (obj[i+1] - obj[i-1]) / range; boundaries (0, n-1) untouched.
    let mut crowd_host = crowd_init.clone();
    for i in 1..n - 1 {
        crowd_host[i] += (sorted_obj[i + 1] - sorted_obj[i - 1]) / obj_range;
    }

    let ptx = crate::ptx_kernels::nsga_crowding_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "nsga_crowding_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_obj = DeviceBuffer::<f64>::from_host(&sorted_obj).expect("d_obj");
    let d_crowd = DeviceBuffer::<f64>::from_host(&crowd_init).expect("d_crowd");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_obj.as_device_ptr(),
                d_crowd.as_device_ptr(),
                n as u32,
                obj_range,
            ),
        )
        .expect("launch nsga_crowding_kernel");
    stream.synchronize().expect("sync");

    let mut crowd_gpu = vec![0.0_f64; n];
    d_crowd.copy_to_host(&mut crowd_gpu).expect("copy crowd");

    // Boundaries must be byte-for-byte unchanged.
    assert_eq!(
        crowd_gpu[0].to_bits(),
        crowd_init[0].to_bits(),
        "nsga_crowding: boundary 0 modified"
    );
    assert_eq!(
        crowd_gpu[n - 1].to_bits(),
        crowd_init[n - 1].to_bits(),
        "nsga_crowding: boundary n-1 modified"
    );
    let (rel, abs) = worst_diff(&crowd_gpu, &crowd_host);
    for i in 0..n {
        assert!(
            close(crowd_gpu[i], crowd_host[i], 1e-12, 1e-12),
            "nsga_crowding[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            crowd_gpu[i],
            crowd_host[i]
        );
    }
}

// ===========================================================================
// 5. pso_update  —  INDEPENDENT HOST RE-DERIVATION (inline LCG + velocity rule)
// ===========================================================================

#[test]
fn pso_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 32_usize;
    let w = 0.729_f64;
    let c1 = 1.49445_f64;
    let c2 = 1.49445_f64;
    let seed = 0x9501_2345_u64;

    let mut rng = LcgRng::new(0x7501_BEEF);
    let pos0: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let vel0: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
    let pbest: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let gbest: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();

    // Host re-derivation: per thread d, state = d ^ seed; advance → r1 (top 53
    // bits ×2⁻⁵³); advance → r2. Then the documented velocity/position update.
    let mut vel_host = vec![0.0_f64; n];
    let mut pos_host = vec![0.0_f64; n];
    for d in 0..n {
        let s1 = lcg_step(d as u64 ^ seed);
        let r1 = unif_hi(s1);
        let s2 = lcg_step(s1);
        let r2 = unif_hi(s2);
        let x = pos0[d];
        let v_new = w * vel0[d] + r1 * (c1 * (pbest[d] - x)) + r2 * (c2 * (gbest[d] - x));
        vel_host[d] = v_new;
        pos_host[d] = x + v_new;
    }

    let ptx = crate::ptx_kernels::pso_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pso_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_pos = DeviceBuffer::<f64>::from_host(&pos0).expect("d_pos");
    let d_vel = DeviceBuffer::<f64>::from_host(&vel0).expect("d_vel");
    let d_pbest = DeviceBuffer::<f64>::from_host(&pbest).expect("d_pbest");
    let d_gbest = DeviceBuffer::<f64>::from_host(&gbest).expect("d_gbest");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_pos.as_device_ptr(),
                d_vel.as_device_ptr(),
                d_pbest.as_device_ptr(),
                d_gbest.as_device_ptr(),
                n as u32,
                w,
                c1,
                c2,
                seed,
            ),
        )
        .expect("launch pso_update_kernel");
    stream.synchronize().expect("sync");

    let mut vel_gpu = vec![0.0_f64; n];
    let mut pos_gpu = vec![0.0_f64; n];
    d_vel.copy_to_host(&mut vel_gpu).expect("copy vel");
    d_pos.copy_to_host(&mut pos_gpu).expect("copy pos");

    // All `mul.rn`/`add.rn`/`sub.rn` (no fma) in the same grouping as the host,
    // so the divergence is at most a couple of ulp; 1e-11 relative is generous.
    let (rel_v, abs_v) = worst_diff(&vel_gpu, &vel_host);
    let (rel_p, abs_p) = worst_diff(&pos_gpu, &pos_host);
    for d in 0..n {
        assert!(
            close(vel_gpu[d], vel_host[d], 1e-11, 1e-12),
            "pso_update vel[{d}] mismatch: gpu={} host={} (worst rel={rel_v:e} abs={abs_v:e})",
            vel_gpu[d],
            vel_host[d]
        );
        assert!(
            close(pos_gpu[d], pos_host[d], 1e-11, 1e-12),
            "pso_update pos[{d}] mismatch: gpu={} host={} (worst rel={rel_p:e} abs={abs_p:e})",
            pos_gpu[d],
            pos_host[d]
        );
    }
}

// ===========================================================================
// 6. de_mutate  —  INDEPENDENT HOST RE-DERIVATION (DE/rand/1 with inline LCG)
// ===========================================================================

#[test]
fn de_mutate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_dims = 6_usize;
    let pop_size = 8_usize;
    let f_scale = 0.7_f64;
    let target_idx = 2_u32;
    let seed = 0xDEED_4321_u64;

    let mut rng = LcgRng::new(0xDE17_9001);
    let pop: Vec<f64> = (0..pop_size * n_dims)
        .map(|_| rng.next_f64() * 4.0 - 2.0)
        .collect();

    // Host re-derivation of the kernel's per-dim index draws and DE/rand/1 rule.
    // Each thread = one dim; state = dim ^ seed. Indices use the LOW 32 bits
    // (cvt.u32.u64) % pop_size, with a single `+1 (mod pop)` bump when == target.
    let pick = |state: u64| -> (u64, u32) {
        let s = lcg_step(state);
        let mut idx = (s as u32) % pop_size as u32;
        if idx == target_idx {
            idx = (idx + 1) % pop_size as u32;
        }
        (s, idx)
    };

    let mut mutant_host = vec![0.0_f64; pop_size * n_dims];
    for dim in 0..n_dims {
        let st0 = dim as u64 ^ seed;
        let (st1, r1) = pick(st0);
        let (st2, r2) = pick(st1);
        let (_st3, r3) = pick(st2);
        let base = pop[r1 as usize * n_dims + dim];
        let diff = pop[r2 as usize * n_dims + dim] - pop[r3 as usize * n_dims + dim];
        mutant_host[target_idx as usize * n_dims + dim] = f_scale.mul_add(diff, base);
    }

    let ptx = crate::ptx_kernels::de_mutate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "de_mutate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_pop = DeviceBuffer::<f64>::from_host(&pop).expect("d_pop");
    let d_mutant =
        DeviceBuffer::<f64>::from_host(&vec![0.0_f64; pop_size * n_dims]).expect("d_mutant");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_dims as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_pop.as_device_ptr(),
                d_mutant.as_device_ptr(),
                n_dims as u32,
                pop_size as u32,
                f_scale,
                target_idx,
                seed,
            ),
        )
        .expect("launch de_mutate_kernel");
    stream.synchronize().expect("sync");

    let mut mutant_gpu = vec![0.0_f64; pop_size * n_dims];
    d_mutant.copy_to_host(&mut mutant_gpu).expect("copy mutant");

    // Single fma.rn vs host mul_add (also single rounding) → bit-exact in
    // practice; allow a 1-ulp cushion. Untouched rows stay exactly zero.
    let (rel, abs) = worst_diff(&mutant_gpu, &mutant_host);
    for k in 0..mutant_gpu.len() {
        assert!(
            close(mutant_gpu[k], mutant_host[k], 1e-12, 1e-12),
            "de_mutate mutant[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            mutant_gpu[k],
            mutant_host[k]
        );
    }
}

// ===========================================================================
// 7. cmaes_sample  —  INDEPENDENT HOST RE-DERIVATION (x = m + sigma·B·D·z)
// ===========================================================================

#[test]
fn cmaes_sample_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_dims = 4_usize;
    let pop_size = 5_usize;
    let sigma = 0.3_f64;

    let mut rng = LcgRng::new(0xC3A5_E501);
    let m: Vec<f64> = (0..n_dims).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
    let b_mat: Vec<f64> = (0..n_dims * n_dims)
        .map(|_| rng.next_f64() * 2.0 - 1.0)
        .collect();
    let d_vec: Vec<f64> = (0..n_dims).map(|_| 0.2 + rng.next_f64()).collect();
    let z: Vec<f64> = (0..pop_size * n_dims)
        .map(|_| rng.next_f64() * 2.0 - 1.0)
        .collect();

    // Host re-derivation: x[k][i] = m[i] + sigma · Σ_j B[i,j]·D[j]·z[k][j].
    let mut x_host = vec![0.0_f64; pop_size * n_dims];
    for k in 0..pop_size {
        for i in 0..n_dims {
            let mut acc = 0.0_f64;
            for j in 0..n_dims {
                acc += b_mat[i * n_dims + j] * (d_vec[j] * z[k * n_dims + j]);
            }
            x_host[k * n_dims + i] = m[i] + sigma * acc;
        }
    }

    let ptx = crate::ptx_kernels::cmaes_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cmaes_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_m = DeviceBuffer::<f64>::from_host(&m).expect("d_m");
    let d_b = DeviceBuffer::<f64>::from_host(&b_mat).expect("d_b");
    let d_d = DeviceBuffer::<f64>::from_host(&d_vec).expect("d_d");
    let d_z = DeviceBuffer::<f64>::from_host(&z).expect("d_z");
    let d_x = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; pop_size * n_dims]).expect("d_x");

    let total = (pop_size * n_dims) as u32;
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_m.as_device_ptr(),
                sigma,
                d_b.as_device_ptr(),
                d_d.as_device_ptr(),
                d_z.as_device_ptr(),
                d_x.as_device_ptr(),
                n_dims as u32,
                pop_size as u32,
            ),
        )
        .expect("launch cmaes_sample_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f64; pop_size * n_dims];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    // GPU fuses the inner product with `fma.rn`; host uses mul+add. Over 4 terms
    // the divergence is a few ulp; 1e-11 relative is generous yet meaningful.
    let (rel, abs) = worst_diff(&x_gpu, &x_host);
    for k in 0..x_gpu.len() {
        assert!(
            close(x_gpu[k], x_host[k], 1e-11, 1e-12),
            "cmaes_sample x[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            x_gpu[k],
            x_host[k]
        );
    }
}
