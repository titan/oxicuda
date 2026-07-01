//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the proven `oxicuda-snn` /
//! `oxicuda-ot` harnesses: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars are passed as the matching Rust scalar
//! (`.param .u32` / `.param .f32` / `.param .u64`) in the kernel's declared
//! parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate cross-check** — `edit_dist_kernel`'s final corner cell is also
//!   compared against [`crate::metrics::edit_distance`] (the crate's own
//!   Levenshtein), on top of a full-table independent host DP.
//! * **Independent host re-derivation** — the remaining kernels each mirror a
//!   step fused into a larger CPU routine, so the oracle is an independent Rust
//!   re-implementation of the kernel's documented arithmetic: HMM forward
//!   log-sum-exp (`forward_pass`), Viterbi max+argmax (`viterbi_step`), the CRF
//!   per-cell emission+transition score (`crf_features`), the beam rank
//!   (`beam_topk`), the Kalman predict mat-vec / `A·P·Aᵀ+Q` (`kalman_predict`),
//!   and the Ising Gibbs single-site sigmoid resample with a bit-exact LCG
//!   (`mrf_gibbs`). These genuinely fail if ptxas miscompiles or the PTX has a
//!   wrong constant / shift / index, because the host code is independent of the
//!   JIT-compiled PTX.
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs`)
//!
//! * **`forward_pass_kernel`** — base-2 vs base-e error: the log-sum-exp used
//!   `ex2.approx.f32` for `exp` and `lg2.approx.f32` for `ln` with no base
//!   conversion, computing `max + log2(Σ 2^(x−max))` (~30 % wrong) instead of
//!   the natural `max + ln(Σ e^(x−max))`. Fixed by scaling the `ex2` argument by
//!   `log2(e)` and the `lg2` result by `ln(2)`.
//! * **`mrf_gibbs_kernel`** — same class: `p_up = 1/(1+exp(−2·field))` used
//!   `ex2(−2·field)` (a base-2 sigmoid). Fixed by scaling the exponent by
//!   `log2(e)` so the `ex2` evaluates `exp(−2·field)`.
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
/// A failure here means ptxas rejected the PTX — a real, invalid-PTX bug.
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

/// Uniform `f32` in `[0, 1)` from this crate's `LcgRng` (which exposes
/// `next_f64`, not `next_f32`).
fn unit(rng: &mut LcgRng) -> f32 {
    rng.next_f64() as f32
}

// ===========================================================================
// 1. forward_pass  —  INDEPENDENT HOST RE-DERIVATION (HMM forward log-sum-exp)
//                     + the base-2 → base-e fix is validated here.
// ===========================================================================

/// Natural, max-stabilised log-sum-exp matching the kernel's documented math
/// (`max + ln Σ exp(x − max)`).
fn host_logsumexp(vals: &[f32]) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for &x in vals {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f32;
    for &x in vals {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

#[test]
fn forward_pass_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // S states. Moderate-magnitude log-domain values keep every exponent O(1)
    // after max-subtraction, so the `ex2.approx` exponential stays well inside
    // its accurate domain and a libm-`exp` oracle is a few-ulp match — yet a
    // base-2 vs base-e error (~30 %) would blow past the 1e-3 tolerance.
    let s = 12_usize;
    let mut rng = LcgRng::new(0x4D31_0001);
    let alpha_prev: Vec<f32> = (0..s).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();
    // log_a[i*S + j] is the log-transition i -> j.
    let log_a: Vec<f32> = (0..s * s).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();
    let log_b_o: Vec<f32> = (0..s).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();

    // Independent host re-derivation:
    //   alpha_next[j] = logsumexp_i(alpha_prev[i] + log_a[i*S + j]) + log_b_o[j].
    let mut next_host = vec![0.0_f32; s];
    let mut col = vec![0.0_f32; s];
    for j in 0..s {
        for i in 0..s {
            col[i] = alpha_prev[i] + log_a[i * s + j];
        }
        next_host[j] = host_logsumexp(&col) + log_b_o[j];
    }

    let ptx = crate::ptx_kernels::forward_pass_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "forward_pass_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_prev = DeviceBuffer::<f32>::from_host(&alpha_prev).expect("d_prev");
    let d_next = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; s]).expect("d_next");
    let d_log_a = DeviceBuffer::<f32>::from_host(&log_a).expect("d_log_a");
    let d_log_b = DeviceBuffer::<f32>::from_host(&log_b_o).expect("d_log_b");

    let block = s as u32;
    let params = LaunchParams::new(grid_1d(s as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_prev.as_device_ptr(),
                d_next.as_device_ptr(),
                d_log_a.as_device_ptr(),
                d_log_b.as_device_ptr(),
                s as u32,
            ),
        )
        .expect("launch forward_pass_kernel");
    stream.synchronize().expect("sync");

    let mut next_gpu = vec![0.0_f32; s];
    d_next.copy_to_host(&mut next_gpu).expect("copy next");

    let (rel, abs) = worst_diff(&next_gpu, &next_host);
    for j in 0..s {
        assert!(
            close(next_gpu[j], next_host[j], 1e-3, 1e-4),
            "forward_pass alpha_next[{j}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            next_gpu[j],
            next_host[j]
        );
    }
}

// ===========================================================================
// 2. viterbi_step  —  INDEPENDENT HOST RE-DERIVATION (max + argmax)
// ===========================================================================

#[test]
fn viterbi_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let s = 12_usize;
    let mut rng = LcgRng::new(0x71B6_0002);
    // Distinct, well-separated additive values so the argmax is unambiguous (no
    // ties) and the GPU's single-rounding compare matches the host exactly.
    let delta_prev: Vec<f32> = (0..s).map(|_| unit(&mut rng) * 4.0 - 2.0).collect();
    let log_a: Vec<f32> = (0..s * s).map(|_| unit(&mut rng) * 4.0 - 2.0).collect();
    let log_b_o: Vec<f32> = (0..s).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();

    // Host: delta_next[j] = max_i(delta_prev[i] + log_a[i*S+j]) + log_b_o[j];
    //       psi[j] = argmax_i.
    let mut next_host = vec![0.0_f32; s];
    let mut psi_host = vec![-1_i32; s];
    for j in 0..s {
        let mut best = f32::NEG_INFINITY;
        let mut arg = -1_i32;
        for i in 0..s {
            let v = delta_prev[i] + log_a[i * s + j];
            if v > best {
                best = v;
                arg = i as i32;
            }
        }
        next_host[j] = best + log_b_o[j];
        psi_host[j] = arg;
    }

    let ptx = crate::ptx_kernels::viterbi_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "viterbi_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_prev = DeviceBuffer::<f32>::from_host(&delta_prev).expect("d_prev");
    let d_next = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; s]).expect("d_next");
    let d_log_a = DeviceBuffer::<f32>::from_host(&log_a).expect("d_log_a");
    let d_log_b = DeviceBuffer::<f32>::from_host(&log_b_o).expect("d_log_b");
    let d_psi = DeviceBuffer::<i32>::from_host(&vec![0_i32; s]).expect("d_psi");

    let block = s as u32;
    let params = LaunchParams::new(grid_1d(s as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_prev.as_device_ptr(),
                d_next.as_device_ptr(),
                d_log_a.as_device_ptr(),
                d_log_b.as_device_ptr(),
                d_psi.as_device_ptr(),
                s as u32,
            ),
        )
        .expect("launch viterbi_step_kernel");
    stream.synchronize().expect("sync");

    let mut next_gpu = vec![0.0_f32; s];
    let mut psi_gpu = vec![0_i32; s];
    d_next.copy_to_host(&mut next_gpu).expect("copy next");
    d_psi.copy_to_host(&mut psi_gpu).expect("copy psi");

    for j in 0..s {
        assert_eq!(
            psi_gpu[j], psi_host[j],
            "viterbi psi[{j}] (argmax) mismatch: gpu={} host={}",
            psi_gpu[j], psi_host[j]
        );
        assert!(
            close(next_gpu[j], next_host[j], 1e-5, 1e-6),
            "viterbi delta_next[{j}] mismatch: gpu={} host={}",
            next_gpu[j],
            next_host[j]
        );
    }
}

// ===========================================================================
// 3. crf_features  —  INDEPENDENT HOST RE-DERIVATION (emission dot + transition)
// ===========================================================================

#[test]
fn crf_features_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_labels = 6_usize;
    let n_features = 5_usize;
    let t_steps = 3_usize;
    let t = 1_u32; // which time step's features to score

    let mut rng = LcgRng::new(0xC4F0_0003);
    let emit: Vec<f32> = (0..n_labels * n_features)
        .map(|_| unit(&mut rng) * 2.0 - 1.0)
        .collect();
    let trans: Vec<f32> = (0..n_labels * n_labels)
        .map(|_| unit(&mut rng) * 2.0 - 1.0)
        .collect();
    let x_feat: Vec<f32> = (0..t_steps * n_features)
        .map(|_| unit(&mut rng) * 2.0 - 1.0)
        .collect();

    // Host: score[y_prev*L + y_cur] = dot(emit[y_cur,:], x_feat[t,:])
    //                                 + trans[y_prev*L + y_cur].
    let mut score_host = vec![0.0_f32; n_labels * n_labels];
    for y_prev in 0..n_labels {
        for y_cur in 0..n_labels {
            let mut emission = 0.0_f32;
            for k in 0..n_features {
                emission += emit[y_cur * n_features + k] * x_feat[t as usize * n_features + k];
            }
            score_host[y_prev * n_labels + y_cur] = emission + trans[y_prev * n_labels + y_cur];
        }
    }

    let ptx = crate::ptx_kernels::crf_features_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "crf_features_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_emit = DeviceBuffer::<f32>::from_host(&emit).expect("d_emit");
    let d_trans = DeviceBuffer::<f32>::from_host(&trans).expect("d_trans");
    let d_x = DeviceBuffer::<f32>::from_host(&x_feat).expect("d_x");
    let d_score =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_labels * n_labels]).expect("d_score");

    // Grid = (y_cur, y_prev): ctaid.x -> y_cur, ctaid.y -> y_prev, block = (1,1).
    let params = LaunchParams::new((n_labels as u32, n_labels as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_emit.as_device_ptr(),
                d_trans.as_device_ptr(),
                d_x.as_device_ptr(),
                d_score.as_device_ptr(),
                t,
                n_labels as u32,
                n_features as u32,
            ),
        )
        .expect("launch crf_features_kernel");
    stream.synchronize().expect("sync");

    let mut score_gpu = vec![0.0_f32; n_labels * n_labels];
    d_score.copy_to_host(&mut score_gpu).expect("copy score");

    let (rel, abs) = worst_diff(&score_gpu, &score_host);
    for k in 0..score_gpu.len() {
        assert!(
            close(score_gpu[k], score_host[k], 1e-5, 1e-6),
            "crf_features score[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            score_gpu[k],
            score_host[k]
        );
    }
}

// ===========================================================================
// 4. beam_topk  —  INDEPENDENT HOST RE-DERIVATION (strict-greater rank)
// ===========================================================================

#[test]
fn beam_topk_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 32_usize;
    let k = 8_u32;

    // Distinct scores (no ties) so the strict-greater rank is unambiguous and
    // the GPU's `setp.gt.f32` compare is bit-exact against the host.
    let scores: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 5.0).collect();
    // Deterministically shuffle to avoid an already-sorted input.
    let mut scores = scores;
    let mut rng = LcgRng::new(0xBEE7_0004);
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        scores.swap(i, j);
    }

    // Host: rank[i] = #{ j : scores[j] > scores[i] }; survivors keep rank, else -1.
    let mut rank_host = vec![0_i32; n];
    for i in 0..n {
        let mut rank = 0_u32;
        for j in 0..n {
            if scores[j] > scores[i] {
                rank += 1;
            }
        }
        rank_host[i] = if rank < k { rank as i32 } else { -1 };
    }

    let ptx = crate::ptx_kernels::beam_topk_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "beam_topk_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_scores = DeviceBuffer::<f32>::from_host(&scores).expect("d_scores");
    let d_rank = DeviceBuffer::<i32>::from_host(&vec![0_i32; n]).expect("d_rank");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_scores.as_device_ptr(),
                d_rank.as_device_ptr(),
                n as u32,
                k,
            ),
        )
        .expect("launch beam_topk_kernel");
    stream.synchronize().expect("sync");

    let mut rank_gpu = vec![0_i32; n];
    d_rank.copy_to_host(&mut rank_gpu).expect("copy rank");

    for i in 0..n {
        assert_eq!(
            rank_gpu[i], rank_host[i],
            "beam_topk rank[{i}] mismatch: gpu={} host={} (score={})",
            rank_gpu[i], rank_host[i], scores[i]
        );
    }
    // Exactly k survivors (rank >= 0) must exist.
    let survivors = rank_gpu.iter().filter(|&&r| r >= 0).count();
    assert_eq!(
        survivors, k as usize,
        "beam_topk: expected {k} survivors, got {survivors}"
    );
}

// ===========================================================================
// 5. edit_dist  —  FULL-TABLE HOST DP + crate::metrics::edit_distance corner
//                  cross-check (anti-diagonal wavefront, one launch per diag).
// ===========================================================================

#[test]
fn edit_dist_matches_host_and_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Two integer "strings"; the kernel compares characters with `setp.eq.s32`.
    let a: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let b: Vec<i32> = vec![3, 1, 5, 9, 2, 6, 5];
    let n_a = a.len();
    let n_b = b.len();
    let cols = n_b + 1;

    // Host Levenshtein DP table (the kernel fills exactly this).
    let mut dp_host = vec![0_i32; (n_a + 1) * cols];
    for i in 0..=n_a {
        dp_host[i * cols] = i as i32;
    }
    for j in 0..=n_b {
        dp_host[j] = j as i32;
    }
    for i in 1..=n_a {
        for j in 1..=n_b {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = dp_host[(i - 1) * cols + j] + 1;
            let ins = dp_host[i * cols + (j - 1)] + 1;
            let sub = dp_host[(i - 1) * cols + (j - 1)] + cost;
            dp_host[i * cols + j] = del.min(ins).min(sub);
        }
    }

    // Device DP buffer pre-initialised with the boundary row/column; the kernel
    // only fills interior cells (i,j) with 1 <= i <= n_a, 1 <= j <= n_b.
    let mut dp_init = vec![0_i32; (n_a + 1) * cols];
    for i in 0..=n_a {
        dp_init[i * cols] = i as i32;
    }
    for j in 0..=n_b {
        dp_init[j] = j as i32;
    }

    let ptx = crate::ptx_kernels::edit_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "edit_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dp = DeviceBuffer::<i32>::from_host(&dp_init).expect("d_dp");
    let d_a = DeviceBuffer::<i32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<i32>::from_host(&b).expect("d_b");

    // Anti-diagonal wavefront: diag = i + j runs 2 ..= n_a + n_b. Each launch
    // fills every interior cell on that diagonal (tid -> i = tid+1).
    let block = n_a as u32;
    for diag in 2..=(n_a + n_b) {
        let params = LaunchParams::new(1u32, block);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_dp.as_device_ptr(),
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    n_a as u32,
                    n_b as u32,
                    diag as u32,
                ),
            )
            .expect("launch edit_dist_kernel");
        stream.synchronize().expect("sync");
    }

    let mut dp_gpu = vec![0_i32; (n_a + 1) * cols];
    d_dp.copy_to_host(&mut dp_gpu).expect("copy dp");

    for idx in 0..dp_gpu.len() {
        assert_eq!(
            dp_gpu[idx],
            dp_host[idx],
            "edit_dist dp[{}][{}] mismatch: gpu={} host={}",
            idx / cols,
            idx % cols,
            dp_gpu[idx],
            dp_host[idx]
        );
    }

    // Crate cross-check: the final corner is the full edit distance.
    let crate_dist = crate::metrics::edit_distance(&a, &b) as i32;
    assert_eq!(
        dp_gpu[n_a * cols + n_b],
        crate_dist,
        "edit_dist corner vs crate::metrics::edit_distance: gpu={} crate={}",
        dp_gpu[n_a * cols + n_b],
        crate_dist
    );
}

// ===========================================================================
// 6. kalman_predict  —  INDEPENDENT HOST RE-DERIVATION (x_pred = A·x,
//                       P_pred = A·P·Aᵀ + Q)
// ===========================================================================

#[test]
fn kalman_predict_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize;
    let mut rng = LcgRng::new(0x4A15_0006);
    let x: Vec<f32> = (0..n).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();
    let a_mat: Vec<f32> = (0..n * n).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();
    let p_mat: Vec<f32> = (0..n * n).map(|_| unit(&mut rng) * 2.0 - 1.0).collect();
    let q_mat: Vec<f32> = (0..n * n).map(|_| unit(&mut rng) * 0.5).collect();

    // Host: x_pred[i] = Σ_k A[i,k]·x[k].
    let mut x_pred_host = vec![0.0_f32; n];
    for i in 0..n {
        let mut s = 0.0_f32;
        for k in 0..n {
            s += a_mat[i * n + k] * x[k];
        }
        x_pred_host[i] = s;
    }
    // Host: P_pred[i,j] = Σ_k A[i,k] (Σ_l P[k,l]·A[j,l]) + Q[i,j].
    let mut p_pred_host = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for k in 0..n {
                let mut inner = 0.0_f32;
                for l in 0..n {
                    inner += p_mat[k * n + l] * a_mat[j * n + l];
                }
                acc += a_mat[i * n + k] * inner;
            }
            p_pred_host[i * n + j] = acc + q_mat[i * n + j];
        }
    }

    let ptx = crate::ptx_kernels::kalman_predict_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "kalman_predict_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_x_pred = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_x_pred");
    let d_a = DeviceBuffer::<f32>::from_host(&a_mat).expect("d_a");
    let d_p = DeviceBuffer::<f32>::from_host(&p_mat).expect("d_p");
    let d_p_pred = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * n]).expect("d_p_pred");
    let d_q = DeviceBuffer::<f32>::from_host(&q_mat).expect("d_q");

    let block = n as u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_x_pred.as_device_ptr(),
                d_a.as_device_ptr(),
                d_p.as_device_ptr(),
                d_p_pred.as_device_ptr(),
                d_q.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch kalman_predict_kernel");
    stream.synchronize().expect("sync");

    let mut x_pred_gpu = vec![0.0_f32; n];
    let mut p_pred_gpu = vec![0.0_f32; n * n];
    d_x_pred.copy_to_host(&mut x_pred_gpu).expect("copy x_pred");
    d_p_pred.copy_to_host(&mut p_pred_gpu).expect("copy p_pred");

    let (rel_x, abs_x) = worst_diff(&x_pred_gpu, &x_pred_host);
    for i in 0..n {
        assert!(
            close(x_pred_gpu[i], x_pred_host[i], 1e-4, 1e-5),
            "kalman x_pred[{i}] mismatch: gpu={} host={} (worst rel={rel_x:e} abs={abs_x:e})",
            x_pred_gpu[i],
            x_pred_host[i]
        );
    }
    let (rel_p, abs_p) = worst_diff(&p_pred_gpu, &p_pred_host);
    for k in 0..p_pred_gpu.len() {
        assert!(
            close(p_pred_gpu[k], p_pred_host[k], 1e-4, 1e-5),
            "kalman P_pred[{k}] mismatch: gpu={} host={} (worst rel={rel_p:e} abs={abs_p:e})",
            p_pred_gpu[k],
            p_pred_host[k]
        );
    }
}

// ===========================================================================
// 7. mrf_gibbs  —  INDEPENDENT HOST RE-DERIVATION (Ising single-site sigmoid
//                  resample with bit-exact LCG); the base-2 → base-e fix is
//                  validated here.
// ===========================================================================

const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

/// Re-derive the kernel's inline counter-based uniform draw for cell `idx`.
///
/// `state = (idx ^ seed)·M + A`; `u = ((state >> 32) >> 8) · 2^-24`. The 24-bit
/// integer fits a float mantissa exactly and `2^-24` is an exact power of two,
/// so `u` is bit-identical to the GPU's `cvt.rn.f32.u32` + `mul.f32`.
fn gibbs_uniform(idx: u64, seed: u64) -> f32 {
    let state = (idx ^ seed).wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    let high = (state >> 32) as u32;
    let r = high >> 8;
    (r as f32) * (1.0_f32 / 16_777_216.0_f32)
}

#[test]
fn mrf_gibbs_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // KEY conditioning: coupling j = 0 decouples every site from its neighbours,
    // so `field = j·Σ(neighbours) + h = h` is a constant independent of the
    // (racy) concurrent spin writes — the kernel's output is deterministic. With
    // h chosen so p_up = sigmoid(2h) ≈ 0.6, the base-2 bug (which would yield
    // p_up ≈ 0.57) flips every cell whose uniform draw lands in the ~0.03 gap,
    // so this test fails loudly if the base conversion is wrong.
    let n_rows = 16_usize;
    let n_cols = 16_usize;
    let h = 0.5_f32 * 1.5_f32.ln(); // sigmoid(2h) = 0.6
    let j_coupling = 0.0_f32;
    // Seed chosen (offline search) so that, over the 256 cells, no uniform draw
    // lands within 1e-3 of p_up = 0.6. The draws form a low-discrepancy set
    // (the cell index spans only the low 8 bits XORed into the LCG input), so
    // the best achievable decision margin is ~2.5e-3 — still ~12× the strictest
    // 1e-3 floor here yet far below the ~0.03 base-2 error gap this test guards.
    let seed = 0x4612_8395_5D0E_D500_u64;

    let p_up = 1.0_f32 / (1.0_f32 + (-2.0_f32 * h).exp());

    // Host reference + decision-margin precondition (so the comparison is an
    // honest bit-exact check, not a knife-edge near u == p_up).
    let mut spins_host = vec![0_i32; n_rows * n_cols];
    let mut min_margin = f32::INFINITY;
    for row in 0..n_rows {
        for c in 0..n_cols {
            let idx = (row * n_cols + c) as u64;
            let u = gibbs_uniform(idx, seed);
            min_margin = min_margin.min((u - p_up).abs());
            spins_host[row * n_cols + c] = if u < p_up { 1 } else { -1 };
        }
    }
    assert!(
        min_margin > 1e-3,
        "test setup: a Gibbs cell sits within {min_margin} of p_up={p_up}; \
         pick another seed so the spin decision is unambiguous"
    );

    // Initial spins are irrelevant (multiplied by j = 0) but must be valid ±1.
    let spins_init = vec![1_i32; n_rows * n_cols];

    let ptx = crate::ptx_kernels::mrf_gibbs_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mrf_gibbs_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_spins = DeviceBuffer::<i32>::from_host(&spins_init).expect("d_spins");

    // Grid = (1,1), block = (n_cols, n_rows): tid.x -> col, tid.y -> row.
    let params = LaunchParams::new((1u32, 1u32), (n_cols as u32, n_rows as u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_spins.as_device_ptr(),
                h,
                j_coupling,
                n_rows as u32,
                n_cols as u32,
                seed,
            ),
        )
        .expect("launch mrf_gibbs_kernel");
    stream.synchronize().expect("sync");

    let mut spins_gpu = vec![0_i32; n_rows * n_cols];
    d_spins.copy_to_host(&mut spins_gpu).expect("copy spins");

    let mut flips = 0usize;
    for k in 0..spins_gpu.len() {
        assert!(
            spins_gpu[k] == 1 || spins_gpu[k] == -1,
            "mrf_gibbs spin[{k}] = {} not ±1",
            spins_gpu[k]
        );
        if spins_gpu[k] != spins_host[k] {
            flips += 1;
        }
    }
    assert_eq!(
        flips,
        0,
        "mrf_gibbs: {flips}/{} spins differ from the base-e host oracle (p_up={p_up})",
        spins_gpu.len()
    );
}
