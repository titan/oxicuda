//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's CPU reference (or an independent host re-derivation of the
//! kernel's documented arithmetic). The launch ABI mirrors the `oxicuda-snn`
//! canary: device buffers are passed as their `CUdeviceptr` (`.param .u64`),
//! scalars as the matching Rust scalar type, in the kernel's declared parameter
//! order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   `statevec_apply_1q` ↔ [`crate::statevec::apply_1q::apply_1q_inplace`],
//!   `statevec_apply_2q` ↔ [`crate::statevec::apply_2q::apply_2q_inplace`],
//!   `statevec_apply_cnot` ↔ [`crate::gates::controlled::apply_cnot`],
//!   `expval_pauli` ↔ [`crate::pauli::expval::expval_z_string`].
//! * **Independent host re-derivation** — the op has no single dedicated crate
//!   function (the diagonal phase / probability reduction is fused into larger
//!   routines on the CPU), so the oracle is an independent Rust re-implementation
//!   of the kernel's *documented* arithmetic: `partial_trace` (reduced-state
//!   diagonal), `trotter_step` (ZZ diagonal phase), `measure_prob` (outcome
//!   probability), `qft_butterfly` (twiddled Cooley-Tukey butterfly). These
//!   still genuinely fail if ptxas miscompiles or the PTX has a wrong index /
//!   constant, because the host code is independent of the JIT-compiled PTX.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use num_complex::Complex;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

type C = Complex<f32>;

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

/// Build a random complex state of `dim` amplitudes (re, im split) with `|ψ| = 1`.
fn random_state(dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let mut re = vec![0.0_f32; dim];
    let mut im = vec![0.0_f32; dim];
    let mut norm = 0.0_f32;
    for k in 0..dim {
        let r = rng.next_f32() * 2.0 - 1.0;
        let i = rng.next_f32() * 2.0 - 1.0;
        re[k] = r;
        im[k] = i;
        norm += r * r + i * i;
    }
    let inv = 1.0_f32 / norm.sqrt();
    for k in 0..dim {
        re[k] *= inv;
        im[k] *= inv;
    }
    (re, im)
}

/// Re-assemble (re, im) f32 vectors into a complex vector.
fn join(re: &[f32], im: &[f32]) -> Vec<C> {
    re.iter()
        .zip(im.iter())
        .map(|(&r, &i)| C::new(r, i))
        .collect()
}

/// Assert two complex states (held as gpu re/im vs cpu complex) agree.
fn assert_state_close(re_gpu: &[f32], im_gpu: &[f32], cpu: &[C], rel: f32, abs: f32, what: &str) {
    let re_cpu: Vec<f32> = cpu.iter().map(|z| z.re).collect();
    let im_cpu: Vec<f32> = cpu.iter().map(|z| z.im).collect();
    let (rr, ra) = worst_diff(re_gpu, &re_cpu);
    let (ir, ia) = worst_diff(im_gpu, &im_cpu);
    for k in 0..cpu.len() {
        assert!(
            close(re_gpu[k], re_cpu[k], rel, abs) && close(im_gpu[k], im_cpu[k], rel, abs),
            "{what}: amp[{k}] mismatch gpu=({}, {}) cpu=({}, {}) \
             (worst re rel={rr:e} abs={ra:e}; im rel={ir:e} abs={ia:e})",
            re_gpu[k],
            im_gpu[k],
            re_cpu[k],
            im_cpu[k]
        );
    }
}

// ===========================================================================
// 1. statevec_apply_1q  —  CRATE ORACLE (statevec::apply_1q::apply_1q_inplace)
// ===========================================================================

#[test]
fn statevec_apply_1q_matches_cpu() {
    use crate::statevec::apply_1q::apply_1q_inplace;
    use crate::statevec::state::StateVector;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 4_usize;
    let dim = 1usize << n_qubits;
    let qubit = 1_usize;
    let mask = 1u32 << qubit;
    let n_pairs = (dim / 2) as u32;

    let mut rng = LcgRng::new(0x1A00_u64 ^ 0xC0FFEE);
    let (re0, im0) = random_state(dim, &mut rng);

    // Arbitrary (non-unitary) 2x2 complex gate — a linear map is enough to
    // distinguish a correct kernel from a wrong index/arithmetic one.
    let g: [f32; 8] = std::array::from_fn(|_| rng.next_f32() * 2.0 - 1.0);
    let gate = [
        [C::new(g[0], g[1]), C::new(g[2], g[3])],
        [C::new(g[4], g[5]), C::new(g[6], g[7])],
    ];

    // ---- CPU reference ----
    let mut sv = StateVector {
        amps: join(&re0, &im0),
        n_qubits,
    };
    apply_1q_inplace(&mut sv, qubit, &gate).expect("cpu apply_1q");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::statevec_apply_1q_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "statevec_apply_1q");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_pairs, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                mask,
                g[0],
                g[1],
                g[2],
                g[3],
                g[4],
                g[5],
                g[6],
                g[7],
                n_pairs,
            ),
        )
        .expect("launch statevec_apply_1q");
    stream.synchronize().expect("sync");

    let mut re_gpu = vec![0.0_f32; dim];
    let mut im_gpu = vec![0.0_f32; dim];
    d_re.copy_to_host(&mut re_gpu).expect("copy re");
    d_im.copy_to_host(&mut im_gpu).expect("copy im");

    assert_state_close(&re_gpu, &im_gpu, &sv.amps, 1e-4, 1e-5, "apply_1q");
}

// ===========================================================================
// 2. statevec_apply_2q  —  CRATE ORACLE (statevec::apply_2q::apply_2q_inplace)
// ===========================================================================

#[test]
fn statevec_apply_2q_matches_cpu() {
    use crate::statevec::apply_2q::apply_2q_inplace;
    use crate::statevec::state::StateVector;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 4_usize;
    let dim = 1usize << n_qubits;
    let q0 = 0_usize;
    let q1 = 2_usize;
    let mask0 = 1u32 << q0;
    let mask1 = 1u32 << q1;
    let n_groups = (dim / 4) as u32;

    let mut rng = LcgRng::new(0x2A00_u64 ^ 0xBEEF);
    let (re0, im0) = random_state(dim, &mut rng);

    // Arbitrary 4x4 complex gate (row-major). gate[j][k] is the row-major entry.
    let gre: Vec<f32> = (0..16).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let gim: Vec<f32> = (0..16).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let mut gate = [[C::new(0.0, 0.0); 4]; 4];
    for j in 0..4 {
        for k in 0..4 {
            gate[j][k] = C::new(gre[j * 4 + k], gim[j * 4 + k]);
        }
    }

    // ---- CPU reference ----
    let mut sv = StateVector {
        amps: join(&re0, &im0),
        n_qubits,
    };
    apply_2q_inplace(&mut sv, q0, q1, &gate).expect("cpu apply_2q");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::statevec_apply_2q_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "statevec_apply_2q");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");
    let d_gre = DeviceBuffer::<f32>::from_host(&gre).expect("d_gre");
    let d_gim = DeviceBuffer::<f32>::from_host(&gim).expect("d_gim");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_groups, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                mask0,
                mask1,
                n_groups,
                d_gre.as_device_ptr(),
                d_gim.as_device_ptr(),
            ),
        )
        .expect("launch statevec_apply_2q");
    stream.synchronize().expect("sync");

    let mut re_gpu = vec![0.0_f32; dim];
    let mut im_gpu = vec![0.0_f32; dim];
    d_re.copy_to_host(&mut re_gpu).expect("copy re");
    d_im.copy_to_host(&mut im_gpu).expect("copy im");

    assert_state_close(&re_gpu, &im_gpu, &sv.amps, 1e-4, 1e-5, "apply_2q");
}

// ===========================================================================
// 3. statevec_apply_cnot  —  CRATE ORACLE (gates::controlled::apply_cnot)
// ===========================================================================

#[test]
fn statevec_apply_cnot_matches_cpu() {
    use crate::gates::controlled::apply_cnot;
    use crate::statevec::state::StateVector;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 3_usize;
    let dim = 1usize << n_qubits;
    let ctrl = 0_usize;
    let tgt = 1_usize;
    let ctrl_mask = 1u32 << ctrl;
    let tgt_mask = 1u32 << tgt;

    let mut rng = LcgRng::new(0xC07_u64 ^ 0x1234);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- CPU reference ----
    let mut sv = StateVector {
        amps: join(&re0, &im0),
        n_qubits,
    };
    apply_cnot(&mut sv, ctrl, tgt).expect("cpu cnot");

    // ---- GPU (one thread per amplitude) ----
    let ptx = crate::ptx_kernels::statevec_apply_cnot_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "statevec_apply_cnot");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");

    let block = 256_u32;
    let n = dim as u32;
    let params = LaunchParams::new(grid_1d(n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                ctrl_mask,
                tgt_mask,
                n,
            ),
        )
        .expect("launch statevec_apply_cnot");
    stream.synchronize().expect("sync");

    let mut re_gpu = vec![0.0_f32; dim];
    let mut im_gpu = vec![0.0_f32; dim];
    d_re.copy_to_host(&mut re_gpu).expect("copy re");
    d_im.copy_to_host(&mut im_gpu).expect("copy im");

    // CNOT is an exact permutation of amplitudes — must be bit-exact.
    assert_state_close(&re_gpu, &im_gpu, &sv.amps, 0.0, 0.0, "cnot");
}

// ===========================================================================
// 4. expval_pauli  —  CRATE ORACLE (pauli::expval::expval_z_string)
// ===========================================================================

#[test]
fn expval_pauli_matches_cpu() {
    use crate::pauli::expval::expval_z_string;
    use crate::pauli::pauli_string::PauliOp;
    use crate::statevec::state::StateVector;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // 256 amplitudes = exactly 8 full warps in one block, so every lane is
    // active through the warp-shuffle reduction (no partial-warp membermask UB).
    let n_qubits = 8_usize;
    let dim = 1usize << n_qubits;
    let z_positions = [0_usize, 3, 5];
    let zmask: u32 = z_positions.iter().fold(0u32, |a, &q| a | (1 << q));

    let mut rng = LcgRng::new(0xECA1_u64);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- CPU reference ----
    let amps = join(&re0, &im0);
    let sv = StateVector { amps, n_qubits };
    let ops: Vec<PauliOp> = (0..n_qubits)
        .map(|q| {
            if z_positions.contains(&q) {
                PauliOp::Z
            } else {
                PauliOp::I
            }
        })
        .collect();
    let e_cpu = expval_z_string(&sv, &ops).expect("cpu expval");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::expval_pauli_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "expval_pauli");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let n = dim as u32;
    let params = LaunchParams::new(grid_1d(n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                zmask,
                n,
                d_out.as_device_ptr(),
            ),
        )
        .expect("launch expval_pauli");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], e_cpu, 1e-3, 1e-3),
        "expval_pauli mismatch: gpu={} cpu={}",
        out[0],
        e_cpu
    );
}

// ===========================================================================
// 5. partial_trace  —  INDEPENDENT HOST RE-DERIVATION (reduced-state diagonal)
// ===========================================================================

/// Compact the kept (non-trace) bits of `idx` into a dense reduced index,
/// matching the kernel's COMPACT_LOOP (skip bits where `trace_mask` is set).
fn compact_keep_bits(idx: u32, trace_mask: u32, n_bits: u32) -> usize {
    let mut dense = 0u32;
    let mut pos = 0u32;
    for b in 0..n_bits {
        if (trace_mask >> b) & 1 == 1 {
            continue;
        }
        let bit = (idx >> b) & 1;
        dense |= bit << pos;
        pos += 1;
    }
    dense as usize
}

#[test]
fn partial_trace_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 4_u32;
    let dim = 1usize << n_qubits;
    // Trace out qubits 1 and 3; keep qubits 0 and 2.
    let trace_mask: u32 = (1 << 1) | (1 << 3);
    let n_keep = 2_u32;
    let reduced_dim = 1usize << n_keep;

    let mut rng = LcgRng::new(0x9747_u64);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- Host reference: out[dense] = Σ_{i: keep(i)=dense} |amp_i|^2 ----
    let mut out_host = vec![0.0_f32; reduced_dim];
    for i in 0..dim {
        let dense = compact_keep_bits(i as u32, trace_mask, n_qubits);
        out_host[dense] += re0[i] * re0[i] + im0[i] * im0[i];
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::partial_trace_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "partial_trace");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; reduced_dim]).expect("d_out");

    let block = 256_u32;
    let n_total = dim as u32;
    let params = LaunchParams::new(grid_1d(n_total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                n_total,
                trace_mask,
                n_keep,
                d_out.as_device_ptr(),
            ),
        )
        .expect("launch partial_trace");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; reduced_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Total probability is conserved (sanity) and each reduced diagonal matches.
    let total: f32 = out_gpu.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-3,
        "partial_trace: reduced diagonal does not sum to 1: {total}"
    );
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..reduced_dim {
        assert!(
            close(out_gpu[k], out_host[k], 1e-4, 1e-5),
            "partial_trace out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 6. trotter_step  —  INDEPENDENT HOST RE-DERIVATION (ZZ diagonal phase)
// ===========================================================================

#[test]
fn trotter_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 4_usize;
    let dim = 1usize << n_qubits;
    let zz_mask: u32 = (1 << 0) | (1 << 2);
    // Small angle keeps the kernel's truncated cos/sin polynomial within ~1e-4
    // of the exact rotation used by the host oracle.
    let theta = 0.3_f32;

    let mut rng = LcgRng::new(0x70BB_u64);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- Host reference: amp_i *= exp(±iθ) with sign from the ZZ parity ----
    // exp(-iθ Z⊗Z): even parity (eigenvalue +1) -> exp(-iθ); odd -> exp(+iθ).
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut cpu = join(&re0, &im0);
    for (i, amp) in cpu.iter_mut().enumerate() {
        let parity = (i as u32 & zz_mask).count_ones() & 1;
        let phase = if parity == 1 {
            C::new(cos_t, sin_t)
        } else {
            C::new(cos_t, -sin_t)
        };
        *amp *= phase;
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::trotter_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "trotter_step");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");

    let block = 256_u32;
    let n = dim as u32;
    let params = LaunchParams::new(grid_1d(n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                zz_mask,
                theta,
                n,
            ),
        )
        .expect("launch trotter_step");
    stream.synchronize().expect("sync");

    let mut re_gpu = vec![0.0_f32; dim];
    let mut im_gpu = vec![0.0_f32; dim];
    d_re.copy_to_host(&mut re_gpu).expect("copy re");
    d_im.copy_to_host(&mut im_gpu).expect("copy im");

    assert_state_close(&re_gpu, &im_gpu, &cpu, 1e-3, 1e-3, "trotter_step");
}

// ===========================================================================
// 7. measure_prob  —  INDEPENDENT HOST RE-DERIVATION (outcome probability)
// ===========================================================================

#[test]
fn measure_prob_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // 256 amplitudes = 8 full warps in one block. The measured qubit is bit 2,
    // which VARIES within each warp — so this test specifically exercises the
    // fixed (selp-masked) reduction that no longer diverges the warp.
    let n_qubits = 8_usize;
    let dim = 1usize << n_qubits;
    let qubit = 2_usize;
    let qubit_mask = 1u32 << qubit;
    let outcome = 1_u32;

    let mut rng = LcgRng::new(0x3EA5_u64);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- Host reference: P(outcome) = Σ_{i: bit==outcome} |amp_i|^2 ----
    let mut p_host = 0.0_f32;
    for i in 0..dim {
        let bit = u32::from((i as u32 & qubit_mask) != 0);
        if bit == outcome {
            p_host += re0[i] * re0[i] + im0[i] * im0[i];
        }
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::measure_prob_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "measure_prob");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let n = dim as u32;
    let params = LaunchParams::new(grid_1d(n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                qubit_mask,
                outcome,
                n,
                d_out.as_device_ptr(),
            ),
        )
        .expect("launch measure_prob");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], p_host, 1e-3, 1e-3),
        "measure_prob mismatch: gpu={} host={}",
        out[0],
        p_host
    );
}

// ===========================================================================
// 8. qft_butterfly  —  INDEPENDENT HOST RE-DERIVATION (twiddled CT butterfly)
// ===========================================================================

#[test]
fn qft_butterfly_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_qubits = 4_usize;
    let dim = 1usize << n_qubits;
    let qubit = 1_usize;
    let mask = 1u32 << qubit;
    let n_pairs = (dim / 2) as u32;
    // Small angle keeps the kernel's truncated cos/sin within ~1e-4 of exact.
    let theta = 0.4_f32;

    let mut rng = LcgRng::new(0x9F70_u64);
    let (re0, im0) = random_state(dim, &mut rng);

    // ---- Host reference: y0=(x0+w·x1)/√2, y1=(x0-w·x1)/√2, w=exp(iθ) ----
    let w = C::new(theta.cos(), theta.sin());
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut cpu = join(&re0, &im0);
    for i in 0..dim {
        if (i as u32 & mask) != 0 {
            continue;
        }
        let i0 = i;
        let i1 = i | (mask as usize);
        let x0 = cpu[i0];
        let x1 = cpu[i1];
        let wx1 = w * x1;
        cpu[i0] = (x0 + wx1) * inv_sqrt2;
        cpu[i1] = (x0 - wx1) * inv_sqrt2;
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::qft_butterfly_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "qft_butterfly");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_re = DeviceBuffer::<f32>::from_host(&re0).expect("d_re");
    let d_im = DeviceBuffer::<f32>::from_host(&im0).expect("d_im");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_pairs, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_re.as_device_ptr(),
                d_im.as_device_ptr(),
                mask,
                theta,
                n_pairs,
            ),
        )
        .expect("launch qft_butterfly");
    stream.synchronize().expect("sync");

    let mut re_gpu = vec![0.0_f32; dim];
    let mut im_gpu = vec![0.0_f32; dim];
    d_re.copy_to_host(&mut re_gpu).expect("copy re");
    d_im.copy_to_host(&mut im_gpu).expect("copy im");

    assert_state_close(&re_gpu, &im_gpu, &cpu, 1e-3, 1e-3, "qft_butterfly");
}
