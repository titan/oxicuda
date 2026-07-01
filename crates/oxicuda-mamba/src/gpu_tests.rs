//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-snn` path: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars are
//! passed as the matching Rust scalar (`.param .u32` / `.param .f32`), in
//! declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   - `selective_scan` → [`crate::ssm::state_cache::SsmStateCache`] (state
//!     dim 1: the per-channel scalar recurrence `h = a_bar·h + b_bar·u`,
//!     `out = c·h`).
//!   - `parallel_scan`  → [`crate::ssm::parallel_scan::inclusive_scan`] (the
//!     associative `(a,b)` inclusive prefix scan).
//!   - `ssd_chunk`      → [`crate::mamba2::ssd::ssd_recurrent`] applied
//!     per-state-dimension column (the kernel treats each state dim `s`
//!     independently with its own `x[·,s]`).
//!   - `rms_norm_silu`  → [`crate::mamba::mamba_block::rms_norm`] (32-wide
//!     window) times [`crate::mamba::mamba_block::silu`].
//!   - `wkv_forward`    → an exact host re-derivation of the WKV recurrence in
//!     [`crate::rwkv::time_mixing::TimeMixingLayer::forward`] (the recurrence is
//!     fused into the larger layer, so it is lifted out verbatim here).
//! * **Independent host re-derivation** — the documented kernel arithmetic
//!   re-implemented in Rust, independent of the JIT-compiled PTX, so it still
//!   genuinely fails on a ptxas miscompile / wrong constant / wrong index:
//!   - `depthwise_conv1d` — channel-major causal conv (the crate's
//!     `causal_depthwise_conv1d` uses time-major layout + a bias; this re-derives
//!     the kernel's own `[C, L]`, bias-free arithmetic).
//!   - `hippo_legendre`   — the diagonal forward-Euler HiPPO-LegS update
//!     `c_n·(1 − Δ(n+1)) + Δ·√(2n+1)·u`; no crate function realises this exact
//!     diagonal approximation, so it is flagged ORACLE-LESS.
//!
//! ## Bugs found and fixed (see `ptx_kernels.rs` history)
//!
//! Two real PTX bugs were caught by base-e CPU oracles and fixed:
//! * `parallel_scan` read the *later* warp neighbour with `shfl.sync.down`; an
//!   inclusive prefix scan must read the *earlier* neighbour with
//!   `shfl.sync.up`. The old kernel matched neither a prefix nor a suffix scan
//!   (31/32 elements wrong).
//! * `wkv_forward` computed the per-step output with the state-update pivot
//!   `max(p+w, k)`, applying an extra `exp(w)` to the history term — a ~45%
//!   error. The output must use its own pivot `max(p, u+k)`.
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
    if oxicuda_driver::Device::count().ok()? == 0 {
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
/// A failure here is a real PTX / ptxas error for the live SM version.
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

/// Uniform `f32` vector in `[lo, hi)` from a fixed-seed LCG.
fn rand_vec(rng: &mut LcgRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| lo + (hi - lo) * rng.next_f32()).collect()
}

// ===========================================================================
// 1. selective_scan  —  CRATE ORACLE (SsmStateCache, state dim 1)
// ===========================================================================

#[test]
fn selective_scan_matches_cpu_recurrence() {
    use crate::ssm::state_cache::SsmStateCache;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let seq_len = 24_usize;
    let d_model = 64_usize;
    let mut rng = LcgRng::new(0x5E1EC7);

    // Build pre-discretized a_bar = exp(Δ·A) with base-e exp on the host (A < 0,
    // Δ > 0), exactly as `mamba::selective_scan` forms it. The kernel itself
    // performs ONLY the recurrence `h = a_bar·h + b_bar·u`, `out = c·h`, so the
    // base-2 exp bug class does not apply to this kernel — the exp is host-side.
    let total = seq_len * d_model;
    let u_in = rand_vec(&mut rng, total, -1.0, 1.0);
    let c_in = rand_vec(&mut rng, total, -1.0, 1.0);
    let mut a_bar = vec![0.0_f32; total];
    let mut b_bar = vec![0.0_f32; total];
    for i in 0..total {
        let a_log = rng.next_f32() * 1.5 - 0.5; // (-0.5, 1.0)
        let a_val = -(a_log.exp()); // A = -exp(a_log) < 0
        let dt = crate::mamba::selective_scan::softplus(rng.next_f32() * 2.0 - 0.5);
        a_bar[i] = (dt * a_val).exp(); // base-e ZOH discretization, in (0, 1)
        b_bar[i] = dt * (rng.next_f32() * 2.0 - 1.0); // Δ·B
    }

    // ---- CPU reference: the crate's own streaming recurrence with N = 1. ----
    let mut cache = SsmStateCache::new(d_model, 1).expect("cache");
    let out_cpu = cache
        .advance_chunk(&u_in, &a_bar, &b_bar, &c_in, seq_len)
        .expect("cpu selective scan");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::selective_scan_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "selective_scan");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_u = DeviceBuffer::<f32>::from_host(&u_in).expect("d_u");
    let d_delta = DeviceBuffer::<f32>::from_host(&u_in).expect("d_delta"); // unused by kernel
    let d_a = DeviceBuffer::<f32>::from_host(&a_bar).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b_bar).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c_in).expect("d_c");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(d_model as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_u.as_device_ptr(),
                d_delta.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                d_out.as_device_ptr(),
                seq_len as u32,
                d_model as u32,
            ),
        )
        .expect("launch selective_scan");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for (k, (&g, &c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 1e-4, 1e-5),
            "selective_scan[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 2. parallel_scan  —  CRATE ORACLE (inclusive_scan); BUG FOUND + FIXED
// ===========================================================================

#[test]
fn parallel_scan_matches_inclusive_scan() {
    use crate::ssm::parallel_scan::{ScanPair, inclusive_scan};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Exactly one full warp: all 32 lanes participate in every shfl.sync so the
    // inclusive scan covers the whole window. The kernel's documented monoid is
    // `(a,b)·(a',b') = (a·a', a·b' + b)`; with the (now corrected) `shfl.sync.up`
    // each lane combines with its EARLIER neighbour, yielding the true SSM scan.
    let n = 32_usize;
    let mut rng = LcgRng::new(0x5CA9);
    let a_in = rand_vec(&mut rng, n, 0.1, 0.95); // stable decays
    let b_in = rand_vec(&mut rng, n, -1.0, 1.0);

    // ---- CPU reference: sequential inclusive scan over ScanPair. ----
    let pairs: Vec<ScanPair> = a_in
        .iter()
        .zip(b_in.iter())
        .map(|(&a, &b)| ScanPair { a, b })
        .collect();
    let scanned = inclusive_scan(&pairs);
    let out_a_cpu: Vec<f32> = scanned.iter().map(|p| p.a).collect();
    let out_b_cpu: Vec<f32> = scanned.iter().map(|p| p.b).collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::parallel_scan_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "parallel_scan");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a_in).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b_in).expect("d_b");
    let d_oa = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_oa");
    let d_ob = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_ob");

    let params = LaunchParams::new(1u32, 32u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_oa.as_device_ptr(),
                d_ob.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch parallel_scan");
    stream.synchronize().expect("sync");

    let mut out_a_gpu = vec![0.0_f32; n];
    let mut out_b_gpu = vec![0.0_f32; n];
    d_oa.copy_to_host(&mut out_a_gpu).expect("copy a");
    d_ob.copy_to_host(&mut out_b_gpu).expect("copy b");

    let (rel_a, abs_a) = worst_diff(&out_a_gpu, &out_a_cpu);
    let (rel_b, abs_b) = worst_diff(&out_b_gpu, &out_b_cpu);
    for (k, (&g, &c)) in out_a_gpu.iter().zip(out_a_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 2e-4, 1e-6),
            "parallel_scan out_a[{k}] gpu={g} cpu={c} (worst rel={rel_a:e} abs={abs_a:e})"
        );
    }
    for (k, (&g, &c)) in out_b_gpu.iter().zip(out_b_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 2e-4, 1e-6),
            "parallel_scan out_b[{k}] gpu={g} cpu={c} (worst rel={rel_b:e} abs={abs_b:e})"
        );
    }
}

// ===========================================================================
// 3. depthwise_conv1d  —  INDEPENDENT HOST RE-DERIVATION (channel-major causal)
// ===========================================================================

#[test]
fn depthwise_conv1d_matches_host_causal_conv() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let channels = 5_usize;
    let seq_len = 12_usize;
    let kernel_size = 4_usize;
    let mut rng = LcgRng::new(0xDC04);

    // Channel-major `[C, L]` input and `[C, K]` weights, matching the kernel's
    // own indexing `x[c*L + t]`, `w[c*K + k]` (no bias).
    let x_in = rand_vec(&mut rng, channels * seq_len, -1.0, 1.0);
    let w_in = rand_vec(&mut rng, channels * kernel_size, -1.0, 1.0);

    // Independent host re-derivation: y[c, t] = Σ_{k≤t} w[c,k] * x[c, t-k].
    let mut y_cpu = vec![0.0_f32; channels * seq_len];
    for c in 0..channels {
        for t in 0..seq_len {
            let mut acc = 0.0_f32;
            for k in 0..kernel_size {
                if t >= k {
                    acc += w_in[c * kernel_size + k] * x_in[c * seq_len + (t - k)];
                }
            }
            y_cpu[c * seq_len + t] = acc;
        }
    }

    let ptx = crate::ptx_kernels::depthwise_conv1d_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "depthwise_conv1d");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x_in).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w_in).expect("d_w");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; channels * seq_len]).expect("d_y");

    let block = 256_u32;
    let total = (channels * seq_len) as u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_w.as_device_ptr(),
                d_y.as_device_ptr(),
                seq_len as u32,
                channels as u32,
                kernel_size as u32,
            ),
        )
        .expect("launch depthwise_conv1d");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; channels * seq_len];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &y_cpu);
    for (k, (&g, &c)) in y_gpu.iter().zip(y_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 1e-5, 1e-6),
            "depthwise_conv1d[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 4. wkv_forward  —  CRATE ORACLE (rwkv::time_mixing recurrence); BUG FOUND + FIXED
// ===========================================================================

/// Exact host re-derivation of the WKV recurrence inside
/// [`crate::rwkv::time_mixing::TimeMixingLayer::forward`] (lines computing
/// `wkv`). The output uses pivot `q = max(p, u+k)`; the state update uses pivot
/// `q2 = max(p+w, k)`. Base-e everywhere.
fn wkv_cpu(
    k: &[f32],
    v: &[f32],
    w: &[f32],
    u: &[f32],
    seq_len: usize,
    channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq_len * channels];
    for c in 0..channels {
        let w_c = w[c];
        let u_c = u[c];
        let mut a = 0.0_f32;
        let mut b = 0.0_f32;
        let mut p = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let kk = k[t * channels + c];
            let vv = v[t * channels + c];
            let q = p.max(u_c + kk);
            let num = (p - q).exp() * a + (u_c + kk - q).exp() * vv;
            let den = (p - q).exp() * b + (u_c + kk - q).exp();
            out[t * channels + c] = if den.abs() > 1e-30 { num / den } else { 0.0 };
            let q2 = (p + w_c).max(kk);
            let decay = (p + w_c - q2).exp();
            let inp = (kk - q2).exp();
            a = decay * a + inp * vv;
            b = decay * b + inp;
            p = q2;
        }
    }
    out
}

#[test]
fn wkv_forward_matches_cpu_recurrence() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let seq_len = 12_usize;
    let channels = 8_usize;
    let mut rng = LcgRng::new(0x77F0);

    let k_in = rand_vec(&mut rng, seq_len * channels, -1.5, 1.5);
    let v_in = rand_vec(&mut rng, seq_len * channels, -2.0, 2.0);
    let w_in = rand_vec(&mut rng, channels, 0.3, 2.5); // positive decay, as in the layer
    let u_in = rand_vec(&mut rng, channels, -0.5, 0.5);

    let out_cpu = wkv_cpu(&k_in, &v_in, &w_in, &u_in, seq_len, channels);

    let ptx = crate::ptx_kernels::wkv_forward_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "wkv_forward");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_k = DeviceBuffer::<f32>::from_host(&k_in).expect("d_k");
    let d_v = DeviceBuffer::<f32>::from_host(&v_in).expect("d_v");
    let d_w = DeviceBuffer::<f32>::from_host(&w_in).expect("d_w");
    let d_u = DeviceBuffer::<f32>::from_host(&u_in).expect("d_u");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; seq_len * channels]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(channels as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_k.as_device_ptr(),
                d_v.as_device_ptr(),
                d_w.as_device_ptr(),
                d_u.as_device_ptr(),
                d_out.as_device_ptr(),
                seq_len as u32,
                channels as u32,
            ),
        )
        .expect("launch wkv_forward");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; seq_len * channels];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tolerance covers ex2.approx + rcp.approx accumulated over the recurrence
    // (measured worst ≪ 1e-3); the fixed-vs-buggy gap is ~45%, so this still
    // catches the pivot bug by orders of magnitude.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for (k, (&g, &c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 3e-3, 1e-4),
            "wkv_forward[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 5. ssd_chunk  —  CRATE ORACLE (ssd_recurrent per state-dim column)
// ===========================================================================

#[test]
fn ssd_chunk_matches_ssd_recurrent_per_column() {
    use crate::mamba2::ssd::ssd_recurrent;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let chunk_len = 6_usize;
    let state_dim = 4_usize;
    let mut rng = LcgRng::new(0x55D0);

    let a_in = rand_vec(&mut rng, chunk_len, 0.3, 0.8); // stable scalar decays
    let b_in = rand_vec(&mut rng, chunk_len * state_dim, -1.0, 1.0);
    let c_in = rand_vec(&mut rng, chunk_len * state_dim, -1.0, 1.0);
    let x_in = rand_vec(&mut rng, chunk_len * state_dim, -1.0, 1.0);

    // CPU reference: the kernel computes each state dim `s` independently with
    // its own input `x[·,s]`, which is exactly `ssd_recurrent` over the column
    // (a, B[:,s], C[:,s], x[:,s]) with N = 1. Assemble Y[i,s] column by column.
    let mut out_cpu = vec![0.0_f32; chunk_len * state_dim];
    for s in 0..state_dim {
        let b_col: Vec<f32> = (0..chunk_len).map(|t| b_in[t * state_dim + s]).collect();
        let c_col: Vec<f32> = (0..chunk_len).map(|t| c_in[t * state_dim + s]).collect();
        let x_col: Vec<f32> = (0..chunk_len).map(|t| x_in[t * state_dim + s]).collect();
        let y_col = ssd_recurrent(&a_in, &b_col, &c_col, &x_col, chunk_len, 1).expect("ssd column");
        for (i, &y) in y_col.iter().enumerate() {
            out_cpu[i * state_dim + s] = y;
        }
    }

    let ptx = crate::ptx_kernels::ssd_chunk_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ssd_chunk");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a_in).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b_in).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c_in).expect("d_c");
    let d_x = DeviceBuffer::<f32>::from_host(&x_in).expect("d_x");
    let d_out =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; chunk_len * state_dim]).expect("d_out");

    let block = 256_u32;
    let total = (chunk_len * state_dim) as u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                d_x.as_device_ptr(),
                d_out.as_device_ptr(),
                chunk_len as u32,
                state_dim as u32,
            ),
        )
        .expect("launch ssd_chunk");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; chunk_len * state_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for (k, (&g, &c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 1e-4, 1e-5),
            "ssd_chunk[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 6. hippo_legendre  —  ORACLE-LESS (independent host re-derivation)
// ===========================================================================

#[test]
fn hippo_legendre_matches_host_forward_euler() {
    // HONEST SCOPE: the kernel realises a DIAGONAL forward-Euler HiPPO-LegS step
    // `c_n' = c_n·(1 − Δ(n+1)) + Δ·√(2n+1)·u`. The crate's HiPPO routines build
    // the full (dense, lower-triangular) LegS matrix, not this diagonal
    // approximation, so there is no crate equivalence oracle. We re-derive the
    // documented diagonal arithmetic independently; the test still genuinely
    // fails on a ptxas miscompile, a wrong `√(2n+1)` scaling, or an index error.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_coeffs = 64_usize;
    let delta = 0.01_f32;
    let u_val = 0.7_f32;
    let mut rng = LcgRng::new(0x41B0);
    let c_in = rand_vec(&mut rng, n_coeffs, -1.0, 1.0);

    let mut c_cpu = vec![0.0_f32; n_coeffs];
    for n in 0..n_coeffs {
        let np1 = (n + 1) as f32;
        let decay = 1.0_f32 - delta * np1;
        let coupling = delta * (2.0 * np1 - 1.0).sqrt() * u_val; // √(2n+1) = √(2(n+1)−1)
        c_cpu[n] = c_in[n] * decay + coupling;
    }

    let ptx = crate::ptx_kernels::hippo_legendre_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hippo_legendre");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c = DeviceBuffer::<f32>::from_host(&c_in).expect("d_c");
    let d_co = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_coeffs]).expect("d_co");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_coeffs as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c.as_device_ptr(),
                d_co.as_device_ptr(),
                u_val,
                delta,
                n_coeffs as u32,
            ),
        )
        .expect("launch hippo_legendre");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; n_coeffs];
    d_co.copy_to_host(&mut c_gpu).expect("copy c_out");

    let (rel, abs) = worst_diff(&c_gpu, &c_cpu);
    for (k, (&g, &c)) in c_gpu.iter().zip(c_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 1e-4, 1e-6),
            "hippo_legendre[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 7. rms_norm_silu  —  CRATE ORACLE (rms_norm × silu, 32-wide window)
// ===========================================================================

#[test]
fn rms_norm_silu_matches_cpu() {
    use crate::mamba::mamba_block::{rms_norm, silu};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Exactly one warp (32 elements): the kernel divides the warp sum-of-squares
    // by 32, so the RMS window is the full 32-element vector — matching
    // `rms_norm` over d_model = 32. All 32 lanes participate in every
    // shfl.bfly.sync (no divergence).
    let n = 32_usize;
    let eps = 1e-5_f32;
    let mut rng = LcgRng::new(0x121F);
    let x_in = rand_vec(&mut rng, n, -2.0, 2.0);
    let g_in = rand_vec(&mut rng, n, -1.5, 1.5);
    let z_in = rand_vec(&mut rng, n, -3.0, 3.0);

    // CPU reference: out[i] = (x[i]/rms · g[i]) · silu(z[i]).
    let normed = rms_norm(&x_in, &g_in, 1, n, eps).expect("rms_norm");
    let z_silu = silu(&z_in);
    let out_cpu: Vec<f32> = normed
        .iter()
        .zip(z_silu.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let ptx = crate::ptx_kernels::rms_norm_silu_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rms_norm_silu");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x_in).expect("d_x");
    let d_g = DeviceBuffer::<f32>::from_host(&g_in).expect("d_g");
    let d_z = DeviceBuffer::<f32>::from_host(&z_in).expect("d_z");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let params = LaunchParams::new(1u32, 32u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_g.as_device_ptr(),
                d_z.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                eps,
            ),
        )
        .expect("launch rms_norm_silu");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // sqrt.approx + rcp.approx + ex2.approx → ~5e-4 relative; a gross formula
    // error (e.g. base-2 exp without the log2(e) scale) would be ~20–50%.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for (k, (&g, &c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 5e-4, 1e-6),
            "rms_norm_silu[{k}] gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}
