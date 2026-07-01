//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's own CPU reference. The launch ABI mirrors the working
//! `oxicuda-snn` / `oxicuda-recsys` paths: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .s32` ← `u32`, `.param .f32` ← `f32`), in declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! Every kernel in this crate is validated by a **real CPU-vs-GPU numerical
//! equivalence** assertion — there are no stubs and no load-only kernels:
//!
//! * **Crate oracle** — `fake_quant_inplace` is compared to
//!   [`crate::qat::fake_quant::FakeQuantize::forward`] (zero-point 0), and
//!   `nf4_dequant` to the crate's [`crate::scheme::nf4::NF4_LUT`] table.
//! * **Independent host re-derivation of the kernel's documented arithmetic** —
//!   `int8_quant` (`clamp(round(x/scale), -127, 127)`), `int8_dequant`
//!   (`(i8)x * scale`) and `prune_by_mask` (`w *= (mask != 0)`). The host code
//!   is written independently of the JIT-compiled PTX, so a ptxas miscompile,
//!   a wrong constant, or a wrong index genuinely fails the assertion.
//!
//! Inputs are fully deterministic (no RNG); the fake-quant / int8-quant inputs
//! are constructed so the post-division value never lands on a rounding tie,
//! making the round-to-nearest comparison exact rather than knife-edge.
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs`)
//!
//! * **Undeclared registers (all five kernels)** — every kernel referenced the
//!   `%`-prefixed virtual registers `%r0..`, `%rd0..`, plus `%f0` / `%p0`
//!   (nf4, prune) without ever declaring them with a `.reg .TYPE %rN<COUNT>;`
//!   directive. ptxas rejected all five with "Arguments mismatch" / undefined
//!   identifier, so none of them had ever loaded on any GPU. Fix: added the
//!   missing `.reg` declarations at the top of each function.
//! * **`nf4_dequant` — illegal `[symbol + register]` memory operand** — the LUT
//!   gather used `ld.shared.f32 vlo, [lut + %r4]`, combining a shared-space
//!   symbol with a register inside the address, which is not a valid PTX
//!   addressing mode. Fix: materialise the LUT base with `mov.u32 %r, lut`,
//!   add the byte offset in a register, then dereference `[%r]`.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

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

/// JIT-compile `ptx` for the live device and look up `entry`.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX (invalid PTX) —
/// that is a real bug, so we panic loudly rather than skip.
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

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

// ===========================================================================
// 1. fake_quant_inplace  —  CRATE ORACLE (qat::fake_quant::FakeQuantize::forward)
// ===========================================================================

#[test]
fn fake_quant_matches_cpu() {
    use crate::qat::fake_quant::FakeQuantize;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let scale = 0.05_f32;
    // q_min / q_max as the kernel sees them: INT8 symmetric range [-128, 127],
    // matching `FakeQuantize`'s `quant_range` for `bits = 8, symmetric = true`.
    let q_min = -128.0_f32;
    let q_max = 127.0_f32;

    // Deterministic inputs: x = (q_target + delta) * scale with q_target spanning
    // below -128, the interior, and above +127, and delta = ±0.3 (never a tie),
    // so round(x/scale) == q_target exactly under both round-half-even (PTX
    // `cvt.rni`) and round-half-away (Rust `f32::round`).
    let data: Vec<f32> = (0..n)
        .map(|k| {
            let q_target = -135_i32 + 5 * (k as i32);
            let delta = if k % 2 == 0 { 0.3_f32 } else { -0.3_f32 };
            (q_target as f32 + delta) * scale
        })
        .collect();

    // ---- CPU reference (crate oracle) ----
    let fq = FakeQuantize::new(8, true, scale, 0).expect("FakeQuantize::new");
    let expected = fq.forward(&data);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::fake_quant_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fake_quant_inplace");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), n as u32, scale, q_min, q_max),
        )
        .expect("launch fake_quant_inplace");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_data.copy_to_host(&mut gpu).expect("copy data");

    for k in 0..n {
        assert!(
            close(gpu[k], expected[k], 1e-5, 1e-6),
            "fake_quant[{k}] mismatch: gpu={} cpu={} (input={})",
            gpu[k],
            expected[k],
            data[k]
        );
    }
}

// ===========================================================================
// 2. int8_quant  —  INDEPENDENT HOST RE-DERIVATION
//    out[i] = clamp(round(in[i] / scale), -127, 127)  (stored as i8)
// ===========================================================================

#[test]
fn int8_quant_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let scale = 0.05_f32;

    // Same tie-free construction as fake_quant; spans both clamp saturations.
    let data: Vec<f32> = (0..n)
        .map(|k| {
            let q_target = -135_i32 + 5 * (k as i32);
            let delta = if k % 2 == 0 { 0.3_f32 } else { -0.3_f32 };
            (q_target as f32 + delta) * scale
        })
        .collect();

    // ---- CPU reference ----
    // The kernel rounds first (cvt.rni → nearest even), then clamps in float to
    // [-127, 127], then converts to s32 / stores s8.  Re-derive identically.
    let expected: Vec<i8> = data
        .iter()
        .map(|&v| {
            let q = (v / scale).round_ties_even().clamp(-127.0, 127.0);
            q as i8
        })
        .collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::int8_quant_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "int8_quant");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&data).expect("d_in");
    let d_out = DeviceBuffer::<i8>::from_host(&vec![0_i8; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), n as u32, scale),
        )
        .expect("launch int8_quant");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0_i8; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    for k in 0..n {
        assert_eq!(
            gpu[k], expected[k],
            "int8_quant[{k}] mismatch: gpu={} cpu={} (input={})",
            gpu[k], expected[k], data[k]
        );
    }
}

// ===========================================================================
// 3. int8_dequant  —  INDEPENDENT HOST RE-DERIVATION
//    out[i] = (float) in[i] * scale
// ===========================================================================

#[test]
fn int8_dequant_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let scale = 0.0125_f32;

    // Deterministic i8 codes spanning the full signed-byte range.
    let codes: Vec<i8> = (0..n)
        .map(|k| {
            let v = -127_i32 + 4 * (k as i32);
            v.clamp(-127, 127) as i8
        })
        .collect();

    // ---- CPU reference ----
    let expected: Vec<f32> = codes.iter().map(|&c| f32::from(c) * scale).collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::int8_dequant_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "int8_dequant");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<i8>::from_host(&codes).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), n as u32, scale),
        )
        .expect("launch int8_dequant");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    for k in 0..n {
        assert!(
            close(gpu[k], expected[k], 1e-6, 1e-7),
            "int8_dequant[{k}] mismatch: gpu={} cpu={} (code={})",
            gpu[k],
            expected[k],
            codes[k]
        );
    }
}

// ===========================================================================
// 4. nf4_dequant  —  CRATE ORACLE (scheme::nf4::NF4_LUT)
//    out[2*i+0] = LUT[packed & 0xF] * absmax
//    out[2*i+1] = LUT[packed >> 4]  * absmax
// ===========================================================================

#[test]
fn nf4_dequant_matches_cpu() {
    use crate::scheme::nf4::NF4_LUT;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_bytes = 32_usize;
    let n_floats = n_bytes * 2; // kernel `n` parameter (number of output f32)
    let absmax = 2.5_f32;

    // Deterministic packed nibbles exercising every LUT entry in both positions.
    let packed: Vec<u8> = (0..n_bytes)
        .map(|k| {
            let lo = (k % 16) as u8;
            let hi = ((k * 7 + 3) % 16) as u8;
            (hi << 4) | lo
        })
        .collect();

    // ---- CPU reference (crate LUT) ----
    let mut expected = vec![0.0_f32; n_floats];
    for (k, &byte) in packed.iter().enumerate() {
        let lo = (byte & 0x0F) as usize;
        let hi = (byte >> 4) as usize;
        expected[2 * k] = NF4_LUT[lo] * absmax;
        expected[2 * k + 1] = NF4_LUT[hi] * absmax;
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::nf4_dequant_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "nf4_dequant");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_packed = DeviceBuffer::<u8>::from_host(&packed).expect("d_packed");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_floats]).expect("d_out");

    // Single block so thread 0 initialises the shared LUT for every active lane;
    // grid-stride loop walks the `n_bytes` packed bytes.
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n_bytes as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_packed.as_device_ptr(),
                d_out.as_device_ptr(),
                n_floats as u32,
                absmax,
            ),
        )
        .expect("launch nf4_dequant");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_floats];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    for k in 0..n_floats {
        assert!(
            close(gpu[k], expected[k], 1e-5, 1e-6),
            "nf4_dequant[{k}] mismatch: gpu={} cpu={}",
            gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 5. prune_by_mask  —  INDEPENDENT HOST RE-DERIVATION
//    weights[i] = (mask[i] != 0) ? weights[i] : 0
// ===========================================================================

#[test]
fn prune_by_mask_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;

    // Deterministic non-zero weights and a structured 0/1 mask.
    let weights: Vec<f32> = (0..n).map(|k| (k as f32 - 32.0) * 0.137 + 0.011).collect();
    let mask: Vec<u8> = (0..n).map(|k| u8::from(k % 3 != 0)).collect();

    // ---- CPU reference ----
    let expected: Vec<f32> = weights
        .iter()
        .zip(mask.iter())
        .map(|(&w, &m)| if m != 0 { w } else { 0.0 })
        .collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::prune_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "prune_by_mask");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_m = DeviceBuffer::<u8>::from_host(&mask).expect("d_m");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_w.as_device_ptr(), d_m.as_device_ptr(), n as u32),
        )
        .expect("launch prune_by_mask");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_w.copy_to_host(&mut gpu).expect("copy weights");

    // Pruning is exact (kept weights are byte-identical; pruned weights are 0.0).
    for k in 0..n {
        assert_eq!(
            gpu[k].to_bits(),
            expected[k].to_bits(),
            "prune_by_mask[{k}] mismatch: gpu={} cpu={} (mask={})",
            gpu[k],
            expected[k],
            mask[k]
        );
    }
}
