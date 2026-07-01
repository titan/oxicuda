//! On-device GPU validation for the hand-written PTX RNG kernels in this crate.
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! [`oxicuda_driver::Module::from_ptx`] (which runs `ptxas`), launches it on the
//! real CUDA device through `oxicuda-launch`, copies the results back, and
//! asserts numerical equivalence to a CPU oracle.
//!
//! ## Oracle strength (honest accounting)
//!
//! * **Bit-exact reproductions** — for the integer / fixed-point RNG kernels
//!   (`philox_*`, `mrg32k3a_*`, `xorwow_*`, `sobol`, `scrambled_sobol`,
//!   `halton`, `latin_hypercube`, `binomial` direct path, `multinomial`,
//!   `poisson_postprocess`) the oracle reproduces the kernel's exact 32-bit
//!   integer / `round-to-nearest` FP32 arithmetic, so the comparison is
//!   bit-exact (`cvt.rn.f32.u32` and `mul.rn.f32 *2^-32` are reproduced exactly
//!   by Rust's `as f32` + multiply).
//! * **Transcendental tolerance** — Box-Muller normals and the log/exp kernels
//!   use the SFU approximate instructions (`lg2.approx`, `ex2.approx`,
//!   `sin/cos.approx`, `sqrt.approx`); the oracle uses `std` transcendentals and
//!   compares with a relative-with-absolute-floor tolerance.
//! * **Crate AES oracle** — `aes_ctr_generate` is compared block-by-block to the
//!   crate's own FIPS-197-validated [`aes_encrypt_block`].
//! * **Load-only** — `aes_sbox_load` stages the S-box into shared memory and
//!   returns without writing global memory, so it is launch-validated only.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// Shared GPU fixture / helpers
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

/// Relative-with-absolute-floor closeness test for FP comparisons.
fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Reproduces the kernel's `cvt.rn.f32.u32` + `mul.rn.f32 *2^-32` exactly.
fn u32_to_unit_f32(x: u32) -> f32 {
    (x as f32) * f32::from_bits(0x2F80_0000)
}

// ===========================================================================
// CPU oracles for the bespoke RNG engines
// ===========================================================================

// --- MRG32k3a (this crate's bespoke single-/double-step variant) -----------
//
// NOTE on what is validated here: tracing the kernel's `emit_mrg32k3a_step`
// shows that for one or two outputs per thread the combined output `(s10-s20)
// mod m1` reads the *shifted-in* scramble words, never the modular-recurrence
// result `p1/p2` (those land in `s12`/`s22`, which a 1-/2-step kernel never
// reads). So the kernel's observable output is the wrapped difference of two
// clamped seed-scramble words. The oracle reproduces that exactly (bit-exact);
// the heavy modular-reduction path is genuinely dead for these entries and is
// not exercised by the output.

const MRG_M1_U32: u32 = 4_294_967_087;

/// Clamp-to-nonzero, as the kernel's `emit_clamp_nonzero` does.
fn mrg_clamp(v: u32) -> u32 {
    if v == 0 { 1 } else { v }
}

/// The kernel's combined output `(x - y) mod m1` using its exact u32 selp logic.
fn mrg_wrap_diff(x: u32, y: u32) -> u32 {
    if x >= y {
        x.wrapping_sub(y)
    } else {
        MRG_M1_U32.wrapping_sub(y.wrapping_sub(x))
    }
}

/// Bit-exact reproduction of the single-step `mrg32k3a_u32` / `_uniform` output.
fn mrg32k3a_u32_oracle(seed: u32, gid: u32) -> u32 {
    // After one step, s10 = clamped scramble #1, s20 = clamped scramble #4.
    let a1 = mrg_clamp(seed ^ gid.wrapping_mul(1_812_433_253));
    let b1 = mrg_clamp(seed ^ gid.wrapping_mul(214_013));
    mrg_wrap_diff(a1, b1)
}

/// Reproduces the two consecutive outputs the `mrg32k3a_normal` kernel uses.
fn mrg32k3a_u32_oracle_two_steps(seed: u32, gid: u32) -> (u32, u32) {
    let a1 = mrg_clamp(seed ^ gid.wrapping_mul(1_812_433_253));
    let b1 = mrg_clamp(seed ^ gid.wrapping_mul(214_013));
    let a2 = mrg_clamp(seed ^ gid.wrapping_mul(1_566_083_941));
    let b2 = mrg_clamp(seed ^ gid.wrapping_mul(2_531_011));
    (mrg_wrap_diff(a1, b1), mrg_wrap_diff(a2, b2))
}

// --- XORWOW (this crate's bespoke single-/double-step variant) -------------

const XORWOW_WEYL_INC: u32 = 362_437;

/// One XORWOW step; returns `(new_state, d)` and the combined `s4 + d` output.
fn xorwow_step(s: &mut [u32; 5], d: &mut u32) -> u32 {
    let t = s[0] ^ (s[0] >> 2);
    let s4_old = s[4];
    s[0] = s[1];
    s[1] = s[2];
    s[2] = s[3];
    s[3] = s[4];
    s[4] = (s4_old ^ (s4_old << 4)) ^ (t ^ (t << 1));
    *d = d.wrapping_add(XORWOW_WEYL_INC);
    s[4].wrapping_add(*d)
}

/// Initialises XORWOW state the way the kernels do (seed ^ gid scrambles).
fn xorwow_init(seed: u32, gid: u32) -> ([u32; 5], u32) {
    let mut s = [
        seed ^ gid,
        seed ^ gid.wrapping_mul(1_812_433_253),
        seed ^ gid.wrapping_mul(1_566_083_941),
        seed ^ gid.wrapping_mul(1_103_515_245),
        seed ^ gid.wrapping_mul(214_013),
    ];
    if s[0] | s[1] | s[2] | s[3] | s[4] == 0 {
        s[0] = 1;
    }
    (s, 0)
}

/// Bit-exact reproduction of `xorwow_uniform_f32` (one step, `s4 + d`).
fn xorwow_u32_oracle(seed: u32, gid: u32) -> u32 {
    let (mut s, mut d) = xorwow_init(seed, gid);
    xorwow_step(&mut s, &mut d)
}

/// Box-Muller z from two uniforms, matching the kernel's f32 approximate recipe.
fn box_muller_z(u1: f32, u2: f32) -> f32 {
    let eps = f32::from_bits(0x3380_0000); // ~5.96e-8
    let u1_safe = u1.max(eps);
    let two_pi = f32::from_bits(0x40C9_0FDB);
    let radius = (-2.0_f32 * u1_safe.ln()).sqrt();
    radius * (two_pi * u2).cos()
}

/// Box-Muller sine companion (for the optimized 4-output normal kernel).
fn box_muller_z_sin(u1: f32, u2: f32) -> f32 {
    let eps = f32::from_bits(0x3380_0000);
    let u1_safe = u1.max(eps);
    let two_pi = f32::from_bits(0x40C9_0FDB);
    let radius = (-2.0_f32 * u1_safe.ln()).sqrt();
    radius * (two_pi * u2).sin()
}

// ===========================================================================
// 1. Philox uniform / u32 — BIT-EXACT (crate oracle philox4x32_10)
// ===========================================================================

#[test]
fn philox_uniform_f32_matches_oracle() {
    use crate::engines::philox::{generate_philox_uniform_ptx, philox4x32_10};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u64 = 0x1234_5678_9ABC_DEF0;
    let key = [seed as u32, (seed >> 32) as u32];

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| {
            let c0 = philox4x32_10([gid, 0, 0, 0], key)[0];
            u32_to_unit_f32(c0)
        })
        .collect();

    let ptx = generate_philox_uniform_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_uniform_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32, // offset_lo
                0_u32, // offset_hi
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");

    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "philox uniform mismatch at {i}");
    }
}

#[test]
fn philox_u32_matches_oracle() {
    use crate::engines::philox::{generate_philox_u32_ptx, philox4x32_10};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u64 = 0xDEAD_BEEF_F00D_CAFE;
    let key = [seed as u32, (seed >> 32) as u32];

    let expected: Vec<u32> = (0..n as u32)
        .map(|gid| philox4x32_10([gid, 0, 0, 0], key)[0])
        .collect();

    let ptx = generate_philox_u32_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_u32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "philox_u32 stream mismatch");

    // Non-vacuous probe: a deliberately wrong key must NOT match.
    let wrong: Vec<u32> = (0..n as u32)
        .map(|gid| philox4x32_10([gid, 0, 0, 0], [key[0] ^ 1, key[1]])[0])
        .collect();
    assert_ne!(got, wrong, "probe: wrong-key oracle unexpectedly matched");
}

#[test]
fn philox_uniform_f64_matches_oracle() {
    use crate::engines::philox::{generate_philox_uniform_ptx, philox4x32_10};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let seed: u64 = 0x0BAD_C0DE_1234_5678;
    let key = [seed as u32, (seed >> 32) as u32];

    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| {
            let c = philox4x32_10([gid, 0, 0, 0], key);
            let part_hi = (c[1] as f64) * f64::from_bits(0x3DF0_0000_0000_0000);
            (c[0] as f64).mul_add(f64::from_bits(0x3BF0_0000_0000_0000), part_hi)
        })
        .collect();

    let ptx = generate_philox_uniform_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_uniform_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "philox uniform f64 mismatch at {i}"
        );
    }
}

// ===========================================================================
// 2. Philox normal — Box-Muller (transcendental tolerance)
// ===========================================================================

#[test]
fn philox_normal_f32_matches_oracle() {
    use crate::engines::philox::{generate_philox_normal_ptx, philox4x32_10};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u64 = 0xABCD_1234_5678_9F01;
    let key = [seed as u32, (seed >> 32) as u32];
    let mean = 1.5_f32;
    let stddev = 2.0_f32;

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| {
            let c = philox4x32_10([gid, 0, 0, 0], key);
            let u1 = u32_to_unit_f32(c[0]);
            let u2 = u32_to_unit_f32(c[1]);
            stddev.mul_add(box_muller_z(u1, u2), mean)
        })
        .collect();

    let ptx = generate_philox_normal_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_normal_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 3e-3, 3e-3),
            "philox normal mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 3. Philox optimized uniform / normal — grid-stride, 4 outputs/thread
// ===========================================================================

#[test]
fn philox_optimized_uniform_f32_matches_oracle() {
    use crate::engines::philox::philox4x32_10;
    use crate::engines::philox_optimized::generate_philox_optimized_uniform_f32_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u64 = 0x5151_2323_4545_6767;
    let key = [seed as u32, (seed >> 32) as u32];

    let expected: Vec<f32> = (0..n as u32)
        .map(|i| {
            let block = i / 4;
            let lane = (i % 4) as usize;
            let c = philox4x32_10([block, 0, 0, 0], key);
            u32_to_unit_f32(c[lane])
        })
        .collect();

    let ptx = generate_philox_optimized_uniform_f32_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_optimized_uniform_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    // n_div4 work-items; one thread handles 4 outputs, grid-stride loop.
    let params = LaunchParams::new(grid_1d((n as u32) / 4, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "philox-opt uniform mismatch at {i}"
        );
    }
}

#[test]
fn philox_optimized_normal_f32_matches_oracle() {
    use crate::engines::philox::philox4x32_10;
    use crate::engines::philox_optimized::generate_philox_optimized_normal_f32_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u64 = 0x9090_1212_3434_5656;
    let key = [seed as u32, (seed >> 32) as u32];
    let mean = -0.5_f32;
    let stddev = 1.25_f32;

    let expected: Vec<f32> = (0..n as u32)
        .map(|i| {
            let blk = i / 4;
            let lane = i % 4;
            let c = philox4x32_10([blk, 0, 0, 0], key);
            let u: Vec<f32> = c.iter().map(|&w| u32_to_unit_f32(w)).collect();
            let z = match lane {
                0 => box_muller_z(u[0], u[1]),
                1 => box_muller_z_sin(u[0], u[1]),
                2 => box_muller_z(u[2], u[3]),
                _ => box_muller_z_sin(u[2], u[3]),
            };
            stddev.mul_add(z, mean)
        })
        .collect();

    let ptx = generate_philox_optimized_normal_f32_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_optimized_normal_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d((n as u32) / 4, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 3e-3, 3e-3),
            "philox-opt normal mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 4. MRG32k3a uniform / u32 — BIT-EXACT recipe reproduction
// ===========================================================================

#[test]
fn mrg32k3a_u32_matches_oracle() {
    use crate::engines::mrg32k3a::generate_mrg32k3a_u32_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x1357_9BDF;

    let expected: Vec<u32> = (0..n as u32)
        .map(|gid| mrg32k3a_u32_oracle(seed, gid))
        .collect();

    let ptx = generate_mrg32k3a_u32_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "mrg32k3a_u32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, seed, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "mrg32k3a_u32 mismatch");
}

#[test]
fn mrg32k3a_uniform_f32_matches_oracle() {
    use crate::engines::mrg32k3a::generate_mrg32k3a_uniform_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x2468_ACE0;

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| u32_to_unit_f32(mrg32k3a_u32_oracle(seed, gid)))
        .collect();

    let ptx = generate_mrg32k3a_uniform_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "mrg32k3a_uniform_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, seed, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "mrg uniform mismatch at {i}");
    }
}

#[test]
fn mrg32k3a_normal_f32_matches_oracle() {
    use crate::engines::mrg32k3a::generate_mrg32k3a_normal_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The normal kernel runs two MRG steps; reproduce the exact sequence.
    let n = 1024_usize;
    let seed: u32 = 0x0F1E_2D3C;
    let mean = 0.25_f32;
    let stddev = 0.75_f32;

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| {
            let r1 = mrg32k3a_u32_oracle_two_steps(seed, gid);
            let u1 = u32_to_unit_f32(r1.0);
            let u2 = u32_to_unit_f32(r1.1);
            stddev.mul_add(box_muller_z(u1, u2), mean)
        })
        .collect();

    let ptx = generate_mrg32k3a_normal_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "mrg32k3a_normal_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 3e-3, 3e-3),
            "mrg normal mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 5. XORWOW uniform / normal — BIT-EXACT / transcendental
// ===========================================================================

#[test]
fn xorwow_uniform_f32_matches_oracle() {
    use crate::engines::xorwow::generate_xorwow_uniform_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x7777_3333;

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| u32_to_unit_f32(xorwow_u32_oracle(seed, gid)))
        .collect();

    let ptx = generate_xorwow_uniform_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "xorwow_uniform_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, seed, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "xorwow uniform mismatch at {i}");
    }

    // Non-vacuous probe: wrong seed must not match.
    let wrong: Vec<f32> = (0..n as u32)
        .map(|gid| u32_to_unit_f32(xorwow_u32_oracle(seed ^ 0xFF, gid)))
        .collect();
    assert_ne!(got, wrong, "probe: wrong-seed xorwow unexpectedly matched");
}

#[test]
fn xorwow_normal_f32_matches_oracle() {
    use crate::engines::xorwow::generate_xorwow_normal_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x1111_8888;
    let mean = 3.0_f32;
    let stddev = 0.5_f32;

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| {
            let (mut s, mut d) = xorwow_init(seed, gid);
            let u1_raw = xorwow_step(&mut s, &mut d);
            let u2_raw = xorwow_step(&mut s, &mut d);
            let u1 = u32_to_unit_f32(u1_raw);
            let u2 = u32_to_unit_f32(u2_raw);
            stddev.mul_add(box_muller_z(u1, u2), mean)
        })
        .collect();

    let ptx = generate_xorwow_normal_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "xorwow_normal_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 3e-3, 3e-3),
            "xorwow normal mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 6. AES-256-CTR — crate AES oracle (aes_encrypt_block)
// ===========================================================================

#[test]
fn aes_ctr_generate_matches_crate_aes() {
    use crate::engines::aes_ctr::{
        AesCtrConfig, aes_encrypt_block, build_counter_block, expand_key_256, generate_aes_ctr_ptx,
    };

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let key: [u8; 32] = [
        0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77,
        0x81, 0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14,
        0xdf, 0xf4,
    ];
    let nonce: [u8; 12] = [
        0xca, 0xfe, 0xba, 0xbe, 0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88,
    ];
    let initial_counter: u64 = 0x0102_0304;

    let num_blocks = 64_usize;
    let n = num_blocks * 4;

    let config = AesCtrConfig {
        key,
        nonce,
        initial_counter,
        threads_per_block: 256,
        sm_version: fx.sm,
    };

    let round_keys = expand_key_256(&key);
    let mut expected = vec![0_u32; n];
    for gid in 0..num_blocks as u64 {
        let counter = initial_counter.wrapping_add(gid);
        let block = build_counter_block(&nonce, counter);
        let out = aes_encrypt_block(&block, &round_keys);
        for w in 0..4 {
            expected[gid as usize * 4 + w] =
                u32::from_le_bytes([out[4 * w], out[4 * w + 1], out[4 * w + 2], out[4 * w + 3]]);
        }
    }

    let ptx = generate_aes_ctr_ptx(&config).expect("ptx");
    let kernel = load_kernel(&ptx, "aes_ctr_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");
    let block = 256_u32;
    let mut params = LaunchParams::new(1_u32, block);
    params.shared_mem_bytes = 256;
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(
        got, expected,
        "aes_ctr_generate keystream mismatch vs CPU AES"
    );
}

#[test]
fn aes_sbox_load_runs() {
    // Load-only: this kernel stages the S-box into shared memory and returns
    // without writing global output, so we validate that it JIT-compiles and
    // launches cleanly (a ptxas / launch failure would panic).
    use crate::engines::aes_ctr::{AES_SBOX, generate_sbox_load_ptx};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = generate_sbox_load_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "aes_sbox_load");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sbox = DeviceBuffer::<u8>::from_host(&AES_SBOX).expect("d_sbox");
    let block = 256_u32;
    let mut params = LaunchParams::new(1_u32, block);
    params.shared_mem_bytes = 256;
    kernel
        .launch(&params, &stream, &(d_sbox.as_device_ptr(),))
        .expect("launch aes_sbox_load");
    stream.synchronize().expect("sync");
}

// ===========================================================================
// 7. Binomial (direct path) — BIT-EXACT
// ===========================================================================

#[test]
fn binomial_generate_matches_oracle() {
    use crate::distributions::binomial::generate_binomial_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_trials = 12_u32; // < BTPE_THRESHOLD (20) -> direct inversion path
    let p = 0.35_f32;
    let count = 1024_usize;
    let seed: u64 = 0x33CC_55AA_77EE_1122;
    let seed_lo = seed as u32;

    let expected: Vec<u32> = (0..count as u32)
        .map(|gid| {
            let thread_state = seed_lo ^ gid;
            let mut successes = 0_u32;
            for trial in 0..n_trials {
                let mixed = thread_state ^ trial.wrapping_mul(0x9E37_79B9);
                let hashed = mixed.wrapping_mul(0x045D_9F3B);
                let u = u32_to_unit_f32(hashed);
                if u < p {
                    successes += 1;
                }
            }
            successes
        })
        .collect();

    let ptx = generate_binomial_ptx(n_trials, p, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "binomial_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; count]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(count as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                count as u32,
                seed as u32,
                (seed >> 32) as u32,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; count];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "binomial direct-path mismatch");
}

// ===========================================================================
// 8. Multinomial — BIT-EXACT category counting
// ===========================================================================

#[test]
fn multinomial_generate_matches_oracle() {
    use crate::distributions::multinomial::generate_multinomial_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k = 4_usize;
    let n_trials = 16_u32; // <= 32 unroll bound
    let num_samples = 512_usize;
    let seed: u64 = 0x1A2B_3C4D_5E6F_7080;
    let seed_lo = seed as u32;

    // Strictly-increasing cumulative probabilities, last == 1.0.
    let cum_probs: [f32; 4] = [0.2, 0.5, 0.85, 1.0];

    let expected: Vec<u32> = {
        let mut out = vec![0_u32; num_samples * k];
        for gid in 0..num_samples as u32 {
            let mut state = seed_lo ^ gid;
            let mut counts = [0_u32; 4];
            for trial in 0..32_u32 {
                let active = trial < n_trials;
                let mix = state ^ trial.wrapping_mul(0x9E37_79B9);
                let hashed = mix.wrapping_mul(0x045D_9F3B);
                let u = u32_to_unit_f32(hashed);
                for i in 0..k {
                    let cat = u < cum_probs[i];
                    let inc = if i == 0 {
                        active && cat
                    } else {
                        cat && (u >= cum_probs[i - 1]) && active
                    };
                    if inc {
                        counts[i] += 1;
                    }
                }
                state ^= hashed;
            }
            for i in 0..k {
                out[gid as usize * k + i] = counts[i];
            }
        }
        out
    };

    let ptx = generate_multinomial_ptx(k, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "multinomial_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; num_samples * k]).expect("d_out");
    let d_prob = DeviceBuffer::<f32>::from_host(&cum_probs).expect("d_prob");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(num_samples as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_prob.as_device_ptr(),
                num_samples as u32,
                n_trials,
                seed as u32,
                (seed >> 32) as u32,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; num_samples * k];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "multinomial counts mismatch");

    // Per-sample trial conservation: row sum must equal n_trials.
    for gid in 0..num_samples {
        let s: u32 = got[gid * k..gid * k + k].iter().sum();
        assert_eq!(
            s, n_trials,
            "multinomial row {gid} does not sum to n_trials"
        );
    }
}

// ===========================================================================
// 9. Geometric — inverse-CDF (transcendental, boundary-filtered)
// ===========================================================================

#[test]
fn geometric_generate_matches_oracle() {
    use crate::distributions::geometric::generate_geometric_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let p = 0.3_f32;
    let count = 2048_usize;
    let seed: u64 = 0x55AA_33CC_99BB_7711;
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let log_1mp = (1.0_f64 - f64::from(p)).ln();

    let ptx = generate_geometric_ptx(p, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "geometric_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; count]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(count as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), count as u32, seed_lo, seed_hi),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0_u32; count];
    d_out.copy_to_host(&mut got).expect("copy");

    // Oracle: reproduce the hash -> uniform exactly (bit-exact), then compute the
    // inverse-CDF in f64. Skip indices whose ratio sits within 0.02 of an integer
    // (where the SFU lg2/div approximations can flip the `ceil`).
    let mut checked = 0_usize;
    for (gid, &g) in got.iter().enumerate() {
        let mixed = seed_lo ^ gid as u32;
        let hashed = mixed.wrapping_mul(0x045D_9F3B);
        let mixed2 = hashed ^ seed_hi;
        let hashed2 = mixed2.wrapping_mul(0x27D4_EB2D);
        let u = f64::from(u32_to_unit_f32(hashed2)).max(f64::from(f32::from_bits(0x3380_0000)));
        let ratio = u.ln() / log_1mp;
        let frac = (ratio - ratio.round()).abs();
        let k_expected = ratio.ceil().max(1.0) as u32;
        assert!(g >= 1, "geometric value must be >= 1 at {gid}");
        if frac > 0.02 {
            assert_eq!(
                g, k_expected,
                "geometric mismatch at {gid}: u={u} ratio={ratio}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > count / 2,
        "too few geometric samples passed the oracle"
    );
}

// ===========================================================================
// 10. Truncated normal — wide bounds force attempt-0 acceptance
// ===========================================================================

#[test]
fn truncated_normal_generate_matches_oracle() {
    use crate::distributions::truncated_normal::generate_truncated_normal_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let count = 1024_usize;
    let mean = 0.0_f32;
    let stddev = 1.0_f32;
    // Bounds wide enough that the first Box-Muller candidate is always accepted.
    let lower = -1.0e30_f32;
    let upper = 1.0e30_f32;
    let seed: u64 = 0x2C2C_6E6E_1F1F_4A4A;
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;

    let expected: Vec<f32> = (0..count as u32)
        .map(|gid| {
            let state = seed_lo ^ gid;
            let h1 = state.wrapping_mul(0x045D_9F3B); // attempt 0, mix1 = state ^ 0
            let mix2 = state ^ 0x9E37_79B9_u32; // (0*2+1)*golden
            let h2 = mix2.wrapping_mul(0x045D_9F3B);
            let h1b = h1 ^ seed_hi;
            let u1 = u32_to_unit_f32(h1b);
            let u2 = u32_to_unit_f32(h2);
            stddev.mul_add(box_muller_z(u1, u2), mean)
        })
        .collect();

    let ptx = generate_truncated_normal_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "truncated_normal_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; count]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(count as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                count as u32,
                mean,
                stddev,
                lower,
                upper,
                seed_lo,
                seed_hi,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; count];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 3e-3, 3e-3),
            "truncated normal mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 11. Sobol / scrambled Sobol — BIT-EXACT (direction-number XOR)
// ===========================================================================

/// Deterministic direction-number table (van-der-Corput base-2 plus a couple of
/// perturbed lanes so the XOR accumulation is exercised non-trivially).
fn test_directions() -> [u32; 32] {
    let mut d = [0_u32; 32];
    for (k, slot) in d.iter_mut().enumerate() {
        *slot = (1_u32 << (31 - k)) ^ ((k as u32).wrapping_mul(0x9E37_79B9));
    }
    d
}

fn sobol_value(dir: &[u32; 32], index: u32) -> u32 {
    let gray = index ^ (index >> 1);
    let mut acc = 0_u32;
    for (bit, &dv) in dir.iter().enumerate() {
        if gray & (1_u32 << bit) != 0 {
            acc ^= dv;
        }
    }
    acc
}

#[test]
fn sobol_generate_matches_oracle() {
    use crate::quasi::sobol::generate_sobol_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let base_index = 7_u32;
    let dir = test_directions();

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| u32_to_unit_f32(sobol_value(&dir, base_index + gid)))
        .collect();

    let ptx = generate_sobol_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "sobol_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_dir = DeviceBuffer::<u32>::from_host(&dir).expect("d_dir");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_dir.as_device_ptr(),
                n as u32,
                base_index,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "sobol mismatch at {i}");
    }
}

#[test]
fn scrambled_sobol_generate_matches_oracle() {
    use crate::quasi::ScrambledSobolGenerator;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let base_index = 3_u32;
    let scramble_seed = 0x12AB_34CD_u32;
    let dir = test_directions();

    let expected: Vec<f32> = (0..n as u32)
        .map(|gid| {
            let sobol = sobol_value(&dir, base_index + gid);
            let scrambled1 = sobol ^ scramble_seed;
            let reversed = scrambled1.reverse_bits();
            let rotated_seed = scramble_seed.rotate_left(16);
            u32_to_unit_f32(reversed ^ rotated_seed)
        })
        .collect();

    let ss_gen = ScrambledSobolGenerator::new(2, 42).expect("gen");
    let ptx = ss_gen.generate_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "scrambled_sobol_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_dir = DeviceBuffer::<u32>::from_host(&dir).expect("d_dir");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_dir.as_device_ptr(),
                n as u32,
                base_index,
                scramble_seed,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "scrambled sobol mismatch at {i}");
    }
}

// ===========================================================================
// 12. Halton — BIT-EXACT f32 radical inverse
// ===========================================================================

#[test]
fn halton_generate_matches_oracle() {
    use crate::quasi::HaltonGenerator;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dims = 2_usize;
    let n = 512_usize;
    let base_index = 5_u32;
    let halton_gen = HaltonGenerator::new(dims).expect("gen");
    let primes: Vec<u32> = halton_gen.primes().to_vec();

    let radical = |index: u32, prime: u32| -> f32 {
        let mut result = 0.0_f32;
        let inv = 1.0_f32 / prime as f32;
        let mut factor = inv;
        let mut n_val = index;
        let max_iters = match prime {
            2 => 20,
            3 => 14,
            5 => 10,
            _ => 8,
        };
        for _ in 0..max_iters {
            if n_val != 0 {
                let digit = n_val % prime;
                let contrib = (digit as f32) * factor;
                result += contrib;
                factor *= inv;
                n_val /= prime;
            }
        }
        result
    };

    let mut expected = vec![0.0_f32; n * dims];
    for gid in 0..n as u32 {
        let index = base_index + gid + 1;
        for (d, &prime) in primes.iter().enumerate() {
            expected[gid as usize * dims + d] = radical(index, prime);
        }
    }

    let ptx = halton_gen.generate_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "halton_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * dims]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, base_index),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * dims];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "halton mismatch at {i}");
    }
}

// ===========================================================================
// 13. Latin hypercube — BIT-EXACT f32 hash-stratified
// ===========================================================================

#[test]
fn latin_hypercube_generate_matches_oracle() {
    use crate::quasi::LatinHypercubeSampler;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dims = 3_usize;
    let n = 257_usize; // non-power-of-two stresses the modulo
    let seed: u64 = 0x7E57_C0DE_0BAD_F00D;
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let inv_n = 1.0_f32 / n as f32;

    let mut expected = vec![0.0_f32; n * dims];
    for gid in 0..n as u32 {
        for d in 0..dims as u32 {
            let mix1 = gid ^ seed_lo.wrapping_add(d.wrapping_mul(0x9E37_79B9));
            let mix2 = mix1.wrapping_mul(0x045D_9F3B);
            let mix3 = mix2 ^ seed_hi.wrapping_add(d.wrapping_mul(0x85EB_CA6B));
            let mix4 = mix3.wrapping_mul(0xC2B2_AE35);
            let stratum = mix4 % n as u32;
            let jitter_hash = mix4 ^ 0x27D4_EB2F;
            let u_jitter = u32_to_unit_f32(jitter_hash);
            let val = ((stratum as f32) + u_jitter) * inv_n;
            expected[gid as usize * dims + d as usize] = val;
        }
    }

    let sampler = LatinHypercubeSampler::new(dims, seed);
    let ptx = sampler.generate_ptx(n, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "latin_hypercube_generate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * dims]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_out.as_device_ptr(), n as u32))
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * dims];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "latin hypercube mismatch at {i}");
    }
}

// ===========================================================================
// 14. log-normal exp post-process (ex2) and poisson post-process
// ===========================================================================

#[test]
fn log_normal_exp_f32_matches_oracle() {
    use crate::generator::generate_log_normal_exp_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    // Normal-ish inputs in [-3, 3).
    let input: Vec<f32> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f32) / (n as f32))
        .collect();

    let log2e = f32::from_bits(0x3FB8_AA3B);
    let expected: Vec<f32> = input.iter().map(|&x| (x * log2e).exp2()).collect();

    let ptx = generate_log_normal_exp_ptx(PtxType::F32, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "log_normal_exp_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_buf = DeviceBuffer::<f32>::from_host(&input).expect("d_buf");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_buf.as_device_ptr(), n as u32))
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_buf.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(f64::from(g), f64::from(e), 2e-3, 1e-6),
            "log_normal_exp mismatch at {i}: gpu={g} cpu={e} (in={})",
            input[i]
        );
    }
}

#[test]
fn poisson_postprocess_f32_matches_oracle() {
    use crate::generator::generate_poisson_postprocess_f32_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let input: Vec<f32> = vec![
        -2.7, -0.4, 0.0, 0.5, 0.49, 1.5, 2.5, 3.2, 9.9, -1.0, 7.5, 100.4,
    ];
    let n = input.len();

    // Oracle: round-to-nearest-even, clamp to >= 0, back to f32.
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| x.round_ties_even().max(0.0))
        .collect();

    let ptx = generate_poisson_postprocess_f32_ptx(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "poisson_postprocess_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_buf = DeviceBuffer::<f32>::from_host(&input).expect("d_buf");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_buf.as_device_ptr(), n as u32))
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_buf.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "poisson postprocess mismatch at {i}"
        );
    }
}

// ===========================================================================
// 15. f64 dtype variants — same algorithm as the validated f32 entries
// ===========================================================================

#[test]
fn philox_normal_f64_matches_oracle() {
    use crate::engines::philox::{generate_philox_normal_ptx, philox4x32_10};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let seed: u64 = 0x4242_9999_1717_8080;
    let key = [seed as u32, (seed >> 32) as u32];
    let mean = -2.0_f64;
    let stddev = 3.0_f64;

    // The kernel computes z in f32 (approx transcendentals) then widens to f64.
    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| {
            let c = philox4x32_10([gid, 0, 0, 0], key);
            let z = box_muller_z(u32_to_unit_f32(c[0]), u32_to_unit_f32(c[1]));
            stddev.mul_add(f64::from(z), mean)
        })
        .collect();

    let ptx = generate_philox_normal_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "philox_normal_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed as u32,
                (seed >> 32) as u32,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g, e, 3e-3, 3e-3),
            "philox normal f64 mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn mrg32k3a_uniform_f64_matches_oracle() {
    use crate::engines::mrg32k3a::generate_mrg32k3a_uniform_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x0246_8ACE;
    let scale = f64::from_bits(0x3DF0_0000_0000_0000); // 2^-32

    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| f64::from(mrg32k3a_u32_oracle(seed, gid)) * scale)
        .collect();

    let ptx = generate_mrg32k3a_uniform_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "mrg32k3a_uniform_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, seed, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "mrg uniform f64 mismatch at {i}");
    }
}

#[test]
fn mrg32k3a_normal_f64_matches_oracle() {
    use crate::engines::mrg32k3a::generate_mrg32k3a_normal_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x1020_3040;
    let mean = 5.0_f64;
    let stddev = 0.5_f64;

    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| {
            let (o1, o2) = mrg32k3a_u32_oracle_two_steps(seed, gid);
            let z = box_muller_z(u32_to_unit_f32(o1), u32_to_unit_f32(o2));
            stddev.mul_add(f64::from(z), mean)
        })
        .collect();

    let ptx = generate_mrg32k3a_normal_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "mrg32k3a_normal_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g, e, 3e-3, 3e-3),
            "mrg normal f64 mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn xorwow_uniform_f64_matches_oracle() {
    use crate::engines::xorwow::generate_xorwow_uniform_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0x3333_CCCC;
    let scale = f64::from_bits(0x3DF0_0000_0000_0000);

    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| f64::from(xorwow_u32_oracle(seed, gid)) * scale)
        .collect();

    let ptx = generate_xorwow_uniform_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "xorwow_uniform_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n as u32, seed, 0_u32, 0_u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "xorwow uniform f64 mismatch at {i}"
        );
    }
}

#[test]
fn xorwow_normal_f64_matches_oracle() {
    use crate::engines::xorwow::generate_xorwow_normal_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let seed: u32 = 0xABAB_CDCD;
    let mean = -1.0_f64;
    let stddev = 2.5_f64;

    let expected: Vec<f64> = (0..n as u32)
        .map(|gid| {
            let (mut s, mut d) = xorwow_init(seed, gid);
            let u1 = u32_to_unit_f32(xorwow_step(&mut s, &mut d));
            let u2 = u32_to_unit_f32(xorwow_step(&mut s, &mut d));
            let z = box_muller_z(u1, u2);
            stddev.mul_add(f64::from(z), mean)
        })
        .collect();

    let ptx = generate_xorwow_normal_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "xorwow_normal_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                n as u32,
                seed,
                0_u32,
                0_u32,
                mean,
                stddev,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g, e, 3e-3, 3e-3),
            "xorwow normal f64 mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn log_normal_exp_f64_matches_oracle() {
    use crate::generator::generate_log_normal_exp_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let input: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f64) / (n as f64))
        .collect();

    // The kernel narrows the f64 input to f32, does ex2 in f32, then widens.
    let log2e = f32::from_bits(0x3FB8_AA3B);
    let expected: Vec<f64> = input
        .iter()
        .map(|&x| f64::from(((x as f32) * log2e).exp2()))
        .collect();

    let ptx = generate_log_normal_exp_ptx(PtxType::F64, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "log_normal_exp_f64");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_buf = DeviceBuffer::<f64>::from_host(&input).expect("d_buf");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_buf.as_device_ptr(), n as u32))
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_buf.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g, e, 2e-3, 1e-6),
            "log_normal_exp f64 mismatch at {i}: gpu={g} cpu={e}"
        );
    }
}
