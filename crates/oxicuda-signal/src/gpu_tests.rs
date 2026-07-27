//! On-device GPU validation for the hand-written PTX kernels in
//! `oxicuda-signal`.
//!
//! Unlike the domain crates that expose `NAME_ptx(sm)` builders in a single
//! `ptx_kernels.rs`, this crate emits PTX through per-subsystem `emit_*_kernel`
//! generators (window, DCT-II/III/IV, Haar DWT, FIR, Gaussian blur, Sobel,
//! morphology). Each test JIT-compiles a generator's PTX for the live device's
//! SM version via `Module::from_ptx`, launches the kernel on the real CUDA
//! device through `oxicuda-launch`, copies the results back, and asserts
//! numerical equivalence against the crate's own CPU reference (or, where the
//! op is fused on the host, an independent host re-derivation of the kernel's
//! documented arithmetic).
//!
//! The launch ABI mirrors the `oxicuda-cs` / `oxicuda-ot` canaries: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar in the kernel's declared parameter order. The signal
//! kernels take length/dimension scalars as `.param .u64` (DCT / DWT / FIR /
//! window) or `.param .u32` (image kernels), so the launch tuples widen
//! accordingly.
//!
//! ## PTX bugs found and fixed on a real RTX A4000 (sm_86)
//!
//! * **Braced predication** `@p { .. }` is not valid PTX (ptxas: *"Parsing
//!   error near '{'"*) — it was used in `dct2_permute`, `dct3_pretwiddle`,
//!   `dct3_unpermute`, and `fir_direct`, so none of those kernels had ever
//!   loaded on any GPU. Rewritten with per-instruction predication / a
//!   branch-and-skip.
//! * **`fir_direct` always-false guard:** the tap predicate was
//!   `setp.gt.u64 %p, %src, 0xFFFFFFFFFFFFFFFF` (nothing exceeds `u64::MAX`),
//!   ANDed into the accumulate guard, so every tap was skipped and the kernel
//!   produced all-zeros. Replaced with the correct single unsigned bound
//!   `src < n` (wrap-around covers `src >= 0`).
//! * **Precision-incorrect immediates:** `dct3_pretwiddle` (`0.5`) and
//!   `dct4_postscale` (`2.0`) emitted an `0f...` single-precision literal into
//!   an `.f64` instruction; switched to precision-correct `0d...` literals.
//!
//! Every test skips (returns early) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::types::SignalPrecision;

// ---------------------------------------------------------------------------
// Shared fixture + helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: SmVersion,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor)?;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// Deterministic LCG; unit floats are normalised by 2^32 (never 2^31).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.next_u32() as f32 / 4_294_967_296.0_f32;
        lo + (hi - lo) * u
    }
}

fn fill_f32(rng: &mut Lcg, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32(lo, hi)).collect()
}

/// Best-effort ptxas pre-screen: assembles `ptx` for `sm` with the standalone
/// `ptxas` binary (if found) so a malformed kernel fails with the assembler's
/// own diagnostic before we hand it to the JIT. A missing binary is not fatal —
/// `Module::from_ptx` still validates via the driver's JIT.
fn ptxas_prescreen(ptx: &str, entry: &str, sm: SmVersion) {
    let candidates = [
        "/usr/local/cuda/bin/ptxas",
        "/usr/local/cuda-12.4/bin/ptxas",
        "ptxas",
    ];
    let Some(bin) = candidates
        .iter()
        .find(|p| **p == "ptxas" || std::path::Path::new(*p).exists())
    else {
        return;
    };
    let dir = std::env::temp_dir();
    let path = dir.join(format!("oxicuda_signal_{entry}.ptx"));
    if std::fs::write(&path, ptx).is_err() {
        return;
    }
    let arch = format!("-arch={}", sm.as_ptx_str());
    let out = std::process::Command::new(bin)
        .arg(&arch)
        .arg(&path)
        .arg("-o")
        .arg(dir.join(format!("oxicuda_signal_{entry}.cubin")))
        .output();
    let _ = std::fs::remove_file(&path);
    if let Ok(out) = out {
        // ptxas writes its diagnostics to stdout, not stderr; reporting only
        // stderr leaves the panic message empty and the failure undiagnosable.
        assert!(
            out.status.success(),
            "ptxas rejected `{entry}` for {}:\n{}{}\n--- PTX ---\n{ptx}",
            sm.as_ptx_str(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Pre-screen with ptxas, JIT-compile `ptx`, and look up `entry`.
fn load(fx: &GpuFixture, ptx: &str, entry: &str) -> Kernel {
    ptxas_prescreen(ptx, entry, fx.sm);
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

fn close_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

fn assert_close_f32(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close_f32(g, c, rel, abs),
            "{label}[{i}] mismatch: gpu={g} cpu={c}"
        );
    }
}

// ===========================================================================
// 1. window_apply  —  HOST RE-DERIVATION (x[i] *= w[i])
// ===========================================================================

#[test]
fn window_apply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 300_usize; // ragged: 300 % 256 != 0
    let mut rng = Lcg::new(0x5157_0001);
    let x = fill_f32(&mut rng, n, -2.0, 2.0);
    let w = fill_f32(&mut rng, n, 0.0, 1.0);
    let expected: Vec<f32> = x.iter().zip(w.iter()).map(|(&a, &b)| a * b).collect();

    let ptx = crate::window::emit_window_apply_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "window_apply");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_w.as_device_ptr(), n as u64),
        )
        .expect("launch window_apply");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_x.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-6, 1e-6, "window_apply");
}

// ===========================================================================
// 2. dct2_permute  —  HOST RE-DERIVATION (even/odd gather)
// ===========================================================================

#[test]
fn dct2_permute_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC72_0002);
    let x = fill_f32(&mut rng, n, -3.0, 3.0);

    // y[tid/2] = x[tid] for even tid; y[N-1-tid/2] = x[tid] for odd tid.
    let mut expected = vec![0.0_f32; n];
    for (tid, &xv) in x.iter().enumerate() {
        let half = tid / 2;
        let out = if tid % 2 == 0 { half } else { n - 1 - half };
        expected[out] = xv;
    }

    let ptx = crate::dct::dct2::emit_permute_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct2_permute");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_y.as_device_ptr(), n as u64),
        )
        .expect("launch dct2_permute");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_y.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 0.0, 0.0, "dct2_permute");
}

// ===========================================================================
// 3. dct2_twiddle  —  HOST RE-DERIVATION (X[k] = Re*cw + Im*sw)
// ===========================================================================

#[test]
fn dct2_twiddle_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC72_0003);
    let fft = fill_f32(&mut rng, 2 * n, -1.0, 1.0); // interleaved [re, im]
    let tw = fill_f32(&mut rng, 2 * n, -1.0, 1.0); // interleaved [cw, -sw]

    let expected: Vec<f32> = (0..n)
        .map(|k| fft[2 * k] * tw[2 * k] + fft[2 * k + 1] * tw[2 * k + 1])
        .collect();

    let ptx = crate::dct::dct2::emit_twiddle_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct2_twiddle");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_fft = DeviceBuffer::<f32>::from_host(&fft).expect("d_fft");
    let d_tw = DeviceBuffer::<f32>::from_host(&tw).expect("d_tw");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_fft.as_device_ptr(),
                d_tw.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch dct2_twiddle");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-5, 1e-6, "dct2_twiddle");
}

// ===========================================================================
// 4. dct3_pretwiddle  —  HOST RE-DERIVATION (conjugate twiddle, X[0]*=0.5)
// ===========================================================================

#[test]
fn dct3_pretwiddle_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC73_0004);
    let x = fill_f32(&mut rng, n, -2.0, 2.0); // real input
    let tw = fill_f32(&mut rng, 2 * n, -1.0, 1.0); // interleaved [cw, -sw]

    // buf_re = xk*cw, buf_im = xk*(-(-sw)) = xk*sw_neg_negated; xk[0] *= 0.5.
    let mut expected = vec![0.0_f32; 2 * n];
    for k in 0..n {
        let xk = if k == 0 { x[k] * 0.5 } else { x[k] };
        let cw = tw[2 * k];
        let sw = tw[2 * k + 1];
        expected[2 * k] = xk * cw;
        expected[2 * k + 1] = xk * (-sw);
    }

    let ptx = crate::dct::emit_dct3_pretwiddle_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct3_pretwiddle");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_buf = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; 2 * n]).expect("d_buf");
    let d_tw = DeviceBuffer::<f32>::from_host(&tw).expect("d_tw");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_buf.as_device_ptr(),
                d_tw.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch dct3_pretwiddle");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; 2 * n];
    d_buf.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-5, 1e-6, "dct3_pretwiddle");
}

// ===========================================================================
// 5. dct3_unpermute  —  HOST RE-DERIVATION (inverse even/odd gather)
// ===========================================================================

#[test]
fn dct3_unpermute_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC73_0005);
    let y = fill_f32(&mut rng, n, -3.0, 3.0);

    // even tid: x[tid] = y[tid/2]; odd tid: x[tid] = y[(N-1-tid)/2].
    let mut expected = vec![0.0_f32; n];
    for (tid, e) in expected.iter_mut().enumerate() {
        let idx = if tid % 2 == 0 {
            tid / 2
        } else {
            (n - 1 - tid) / 2
        };
        *e = y[idx];
    }

    let ptx = crate::dct::emit_unpermute_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct3_unpermute");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_x = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_x");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_y.as_device_ptr(), d_x.as_device_ptr(), n as u64),
        )
        .expect("launch dct3_unpermute");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_x.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 0.0, 0.0, "dct3_unpermute");
}

// ===========================================================================
// 6. dct4_pretwiddle  —  HOST RE-DERIVATION (u[i] = x[i] * tw[i])
// ===========================================================================

#[test]
fn dct4_pretwiddle_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC74_0006);
    let x = fill_f32(&mut rng, n, -2.0, 2.0);
    let tw = fill_f32(&mut rng, n, -1.0, 1.0);
    let expected: Vec<f32> = x.iter().zip(tw.iter()).map(|(&a, &b)| a * b).collect();

    let ptx = crate::dct::emit_dct4_pretwiddle_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct4_pretwiddle");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_u = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_u");
    let d_tw = DeviceBuffer::<f32>::from_host(&tw).expect("d_tw");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_u.as_device_ptr(),
                d_tw.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch dct4_pretwiddle");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_u.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-6, 1e-6, "dct4_pretwiddle");
}

// ===========================================================================
// 7. dct4_postscale  —  HOST RE-DERIVATION (X[k] = 2 * tw[k] * u[k])
// ===========================================================================

#[test]
fn dct4_postscale_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let mut rng = Lcg::new(0xDC74_0007);
    let u = fill_f32(&mut rng, n, -2.0, 2.0);
    let tw = fill_f32(&mut rng, n, -1.0, 1.0);
    let expected: Vec<f32> = u
        .iter()
        .zip(tw.iter())
        .map(|(&uu, &tt)| 2.0 * tt * uu)
        .collect();

    let ptx = crate::dct::emit_postscale_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "dct4_postscale");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_u = DeviceBuffer::<f32>::from_host(&u).expect("d_u");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_tw = DeviceBuffer::<f32>::from_host(&tw).expect("d_tw");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_u.as_device_ptr(),
                d_out.as_device_ptr(),
                d_tw.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch dct4_postscale");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-6, 1e-6, "dct4_postscale");
}

// ===========================================================================
// 8. fir_direct  —  HOST RE-DERIVATION (causal FIR, zero boundary)
// ===========================================================================
//
// y[i] = Σ_{k=0}^{m-1} h[k]·x[i-k], x[<0] = 0. Before the fix the tap guard was
// always false and the kernel returned all zeros; the test now asserts a
// non-trivial convolution.

#[test]
fn fir_direct_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 96_usize;
    let m = 9_usize;
    let mut rng = Lcg::new(0xF12_0008);
    let x = fill_f32(&mut rng, n, -1.0, 1.0);
    let h = fill_f32(&mut rng, m, -1.0, 1.0);

    let mut expected = vec![0.0_f32; n];
    for (i, e) in expected.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (k, &hk) in h.iter().enumerate() {
            if i >= k {
                acc += hk * x[i - k];
            }
        }
        *e = acc;
    }
    // Guard against the all-zeros regression: the convolution must be live.
    assert!(
        expected.iter().any(|&v| v.abs() > 1e-3),
        "test setup produced a trivially-zero reference"
    );

    let ptx = crate::filter::emit_fir_direct_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "fir_direct");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");
    let d_h = DeviceBuffer::<f32>::from_host(&h).expect("d_h");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                d_h.as_device_ptr(),
                n as u64,
                m as u64,
            ),
        )
        .expect("launch fir_direct");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_y.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-5, 1e-5, "fir_direct");
}

// ===========================================================================
// 9. haar_forward_level  —  CRATE ORACLE (haar_forward)
// ===========================================================================

#[test]
fn haar_forward_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let half = 64_usize;
    let n = 2 * half;
    let mut rng = Lcg::new(0x4A20_0009);
    let x = fill_f32(&mut rng, n, -4.0, 4.0);

    // Crate reference operates in-place on a single [approx | detail] buffer.
    let mut ref_buf: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    crate::dwt::haar_forward(&mut ref_buf, n).expect("haar_forward");
    let approx_ref: Vec<f32> = ref_buf[..half].iter().map(|&v| v as f32).collect();
    let detail_ref: Vec<f32> = ref_buf[half..].iter().map(|&v| v as f32).collect();

    let ptx = crate::dwt::emit_haar_forward_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "haar_forward_level");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_ap = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; half]).expect("d_ap");
    let d_dt = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; half]).expect("d_dt");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(half as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_ap.as_device_ptr(),
                d_dt.as_device_ptr(),
                half as u64,
            ),
        )
        .expect("launch haar_forward_level");
    stream.synchronize().expect("sync");

    let mut ap = vec![0.0_f32; half];
    let mut dt = vec![0.0_f32; half];
    d_ap.copy_to_host(&mut ap).expect("copy ap");
    d_dt.copy_to_host(&mut dt).expect("copy dt");
    assert_close_f32(&ap, &approx_ref, 1e-5, 1e-5, "haar_forward.approx");
    assert_close_f32(&dt, &detail_ref, 1e-5, 1e-5, "haar_forward.detail");
}

// ===========================================================================
// 10. haar_inverse_level  —  CRATE ORACLE (haar_inverse)
// ===========================================================================

#[test]
fn haar_inverse_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let half = 64_usize;
    let n = 2 * half;
    let mut rng = Lcg::new(0x4A21_000A);
    // Independent approx/detail subbands.
    let approx = fill_f32(&mut rng, half, -4.0, 4.0);
    let detail = fill_f32(&mut rng, half, -4.0, 4.0);

    // Crate reference: pack [approx | detail] then haar_inverse in-place.
    let mut ref_buf: Vec<f64> = approx
        .iter()
        .chain(detail.iter())
        .map(|&v| f64::from(v))
        .collect();
    crate::dwt::haar_inverse(&mut ref_buf, n).expect("haar_inverse");
    let x_ref: Vec<f32> = ref_buf.iter().map(|&v| v as f32).collect();

    let ptx = crate::dwt::emit_haar_inverse_kernel(SignalPrecision::F32, fx.sm);
    let kernel = load(&fx, &ptx, "haar_inverse_level");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ap = DeviceBuffer::<f32>::from_host(&approx).expect("d_ap");
    let d_dt = DeviceBuffer::<f32>::from_host(&detail).expect("d_dt");
    let d_x = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_x");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(half as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ap.as_device_ptr(),
                d_dt.as_device_ptr(),
                d_x.as_device_ptr(),
                half as u64,
            ),
        )
        .expect("launch haar_inverse_level");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");
    assert_close_f32(&x_gpu, &x_ref, 1e-5, 1e-5, "haar_inverse");
}

// ===========================================================================
// 11/12. gaussian_blur_h / _v  —  CRATE ORACLE (gaussian_blur_h / _v)
// ===========================================================================

fn run_gaussian(fx: &GpuFixture, horizontal: bool) {
    let height = 12_usize;
    let width = 13_usize;
    let n = height * width;
    let radius = 2_usize;
    let mut rng = Lcg::new(if horizontal { 0x6A50_000B } else { 0x6A51_000C });
    let image = fill_f32(&mut rng, n, 0.0, 1.0);

    // Device loads the kernel as f32; widen the SAME f32 values for the oracle
    // so the only divergence is the f32-vs-f64 accumulation order.
    let kernel_f64 = crate::image::gaussian_kernel_1d(1.4, radius);
    let kernel_f32: Vec<f32> = kernel_f64.iter().map(|&v| v as f32).collect();
    let kernel_oracle: Vec<f64> = kernel_f32.iter().map(|&v| f64::from(v)).collect();

    let (ptx, entry, expected) = if horizontal {
        (
            crate::image::emit_gaussian_blur_h_kernel(fx.sm),
            "signal_gaussian_blur_h_f32",
            crate::image::gaussian_blur_h(&image, height, width, &kernel_oracle),
        )
    } else {
        (
            crate::image::emit_gaussian_blur_v_kernel(fx.sm),
            "signal_gaussian_blur_v_f32",
            crate::image::gaussian_blur_v(&image, height, width, &kernel_oracle),
        )
    };

    let kernel = load(fx, &ptx, entry);
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&image).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_k = DeviceBuffer::<f32>::from_host(&kernel_f32).expect("d_k");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                d_k.as_device_ptr(),
                height as u32,
                width as u32,
                radius as u32,
                n as u64,
            ),
        )
        .expect("launch gaussian blur");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-4, 1e-4, entry);
}

#[test]
fn gaussian_blur_h_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_gaussian(&fx, true);
}

#[test]
fn gaussian_blur_v_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_gaussian(&fx, false);
}

// ===========================================================================
// 13/14. sobel_gx / sobel_gy  —  CRATE ORACLE (sobel_x / sobel_y)
// ===========================================================================

fn run_sobel(fx: &GpuFixture, horizontal: bool) {
    let height = 11_usize;
    let width = 14_usize;
    let n = height * width;
    let mut rng = Lcg::new(if horizontal { 0x50B0_000D } else { 0x50B1_000E });
    let image = fill_f32(&mut rng, n, -1.0, 1.0);

    let (ptx, entry, expected) = if horizontal {
        (
            crate::image::emit_sobel_x_kernel(fx.sm),
            "signal_sobel_gx_f32",
            crate::image::sobel_x(&image, height, width).expect("sobel_x"),
        )
    } else {
        (
            crate::image::emit_sobel_y_kernel(fx.sm),
            "signal_sobel_gy_f32",
            crate::image::sobel_y(&image, height, width).expect("sobel_y"),
        )
    };

    let kernel = load(fx, &ptx, entry);
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&image).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                height as u32,
                width as u32,
                n as u64,
            ),
        )
        .expect("launch sobel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy");
    assert_close_f32(&out, &expected, 1e-4, 1e-4, entry);
}

#[test]
fn sobel_gx_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_sobel(&fx, true);
}

#[test]
fn sobel_gy_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_sobel(&fx, false);
}

// ===========================================================================
// 15/16. erosion / dilation  —  CRATE ORACLE (erode / dilate)
// ===========================================================================

fn run_morphology(fx: &GpuFixture, is_erosion: bool) {
    use crate::types::StructuringElement;

    let height = 12_usize;
    let width = 11_usize;
    let n = height * width;
    let se = StructuringElement::Rectangle {
        height: 3,
        width: 3,
    };
    let (mask, se_h, se_w) = crate::image::generate_se_mask(se);
    let mut rng = Lcg::new(if is_erosion { 0x401_000F } else { 0x401_0010 });
    let image = fill_f32(&mut rng, n, -2.0, 2.0);

    let (ptx, entry, expected) = if is_erosion {
        (
            crate::image::emit_erosion_kernel(fx.sm),
            "signal_erosion_f32",
            crate::image::erode(&image, height, width, se).expect("erode"),
        )
    } else {
        (
            crate::image::emit_dilation_kernel(fx.sm),
            "signal_dilation_f32",
            crate::image::dilate(&image, height, width, se).expect("dilate"),
        )
    };

    let kernel = load(fx, &ptx, entry);
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&image).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_mask = DeviceBuffer::<u8>::from_host(&mask).expect("d_mask");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                d_mask.as_device_ptr(),
                height as u32,
                width as u32,
                se_h as u32,
                se_w as u32,
                n as u64,
            ),
        )
        .expect("launch morphology");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy");
    // The reduction is a pure min/max selection of input values: bit-exact.
    for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{entry}[{i}] mismatch: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn erosion_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_morphology(&fx, true);
}

#[test]
fn dilation_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_morphology(&fx, false);
}

// ===========================================================================
// Non-vacuous probe: a deliberately corrupted kernel MUST fail the oracle.
// ===========================================================================
//
// Proves the device path is live: if we perturb the window_apply arithmetic
// (multiply -> add), the on-device result diverges from the host oracle, so a
// passing equivalence test above is meaningful rather than vacuously green.

#[test]
fn corruption_probe_is_detected() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 128_usize;
    let mut rng = Lcg::new(0xC0FF_EE11);
    let x = fill_f32(&mut rng, n, 0.5, 2.0);
    let w = fill_f32(&mut rng, n, 0.5, 2.0);
    let expected_mul: Vec<f32> = x.iter().zip(w.iter()).map(|(&a, &b)| a * b).collect();

    // Corrupt: replace the sole `mul.f32` product with `add.f32` (the only
    // f32 multiply in the kernel; `mul.lo.u64` address math is untouched).
    let good = crate::window::emit_window_apply_kernel(SignalPrecision::F32, fx.sm);
    assert_eq!(
        good.matches("mul.f32").count(),
        1,
        "probe assumption broken"
    );
    let bad = good.replace("mul.f32", "add.f32");
    assert_ne!(bad, good, "probe failed to mutate the kernel source");

    let kernel = load(&fx, &bad, "window_apply");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_w.as_device_ptr(), n as u64),
        )
        .expect("launch corrupted window_apply");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_x.copy_to_host(&mut out).expect("copy");

    // The corrupted kernel computes x+w; it must NOT match the x*w oracle.
    let mut diverged = false;
    for (&g, &e) in out.iter().zip(expected_mul.iter()) {
        if !close_f32(g, e, 1e-4, 1e-4) {
            diverged = true;
            break;
        }
    }
    assert!(
        diverged,
        "corruption probe did not diverge — device path may be vacuous"
    );
}
