//! On-device GPU validation for the `quantize` subsystem of `oxicuda-dnn`.
//!
//! Coverage (run on the live sm_86 device; every test skips when no GPU):
//!
//! * **INT8** — `quantize_to_int8` (absmax + quantize) and
//!   `dequantize_from_int8`, validated against a CPU oracle. The absmax
//!   reduction kernel had an invalid-PTX shared-memory addressing bug
//!   (`[smem + r*4]`) that this pass fixed; the numeric test now exercises it.
//! * **FP8 E4M3** — `quantize_to_fp8` (absmax + real E4M3 encode) and
//!   `dequantize_from_fp8`, validated bit-exactly against the in-file E4M3
//!   model.
//! * **INT4 / NF4** — `quantize_to_int4` (scale + symmetric pack),
//!   `dequantize_int4`, and `quantize_to_nf4`, validated against CPU oracles.
//! * **Block-scaled FP4 proxy** — `quantize_block_scaled` (per-block absmax +
//!   biased-integer quantize). The per-block absmax reduction had the same
//!   `[smem + r*4]` bug, fixed here and exercised numerically.
//! * **QAT** — fake-quantize forward, STE backward, and the observer min/max
//!   reduction (also fixed for the `[smem + r*4]` bug), driven via direct PTX.
//! * **GPTQ / AWQ** — `dnn_gptq_dequantize_f32` and `dnn_fused_dequant_gemv_f32`
//!   validated numerically; the `Simplified`/proxy GPTQ-quantize, AWQ
//!   scale-search and AWQ-quantize packers are load+launch fault-free checks
//!   (their math is intentionally a non-faithful proxy — see the manifest).

use super::*;

use oxicuda_blas::GpuFloat;
use oxicuda_launch::LaunchParams;
use oxicuda_memory::DeviceBuffer;

use crate::quantize::block_scale::quantize_block_scaled;
use crate::quantize::fp8_quantize::{dequantize_from_fp8, quantize_to_fp8};
use crate::quantize::gptq_awq::{AwqConfig, GptqConfig, WeightQuantMethod, WeightQuantPlan};
use crate::quantize::int4_quantize::{
    Int4QuantConfig, dequantize_int4, quantize_to_int4, quantize_to_nf4,
};
use crate::quantize::int8_quantize::{dequantize_from_int8, quantize_to_int8};
use crate::quantize::qat::{
    FakeQuantize, ObserverMode, QatBitWidth, QatConfig, QatGranularity, QatSymmetry,
};
use crate::types::{TensorDesc, TensorDescMut, TensorLayout};

// ---------------------------------------------------------------------------
// CPU oracle helpers
// ---------------------------------------------------------------------------

/// IEEE round-to-nearest-even, matching PTX `cvt.rni.f32.f32`.
fn rne(x: f32) -> f32 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Bit-exact CPU model of the kernel's `emit_e4m3_encode` routine. Encodes an
/// `f32` (already clamped to `[-448, 448]`) into an E4M3 byte
/// `[sign:1 | exponent:4 | mantissa:3]`, exponent bias 7.
fn cpu_e4m3_encode(value: f32) -> u8 {
    let fbits = value.to_bits();
    let sign = ((fbits >> 24) & 0x80) as u8;
    let xbits = fbits & 0x7fff_ffff;

    if xbits < 0x3580_0000 {
        return sign;
    }
    if xbits >= 0x43E0_0000 {
        return sign | 0x7E;
    }

    let e32 = (xbits >> 23) & 0xff;
    let m32 = xbits & 0x7f_ffff;
    let mut e4 = e32 as i32 - 120;

    if e4 >= 1 {
        let mut m3 = m32 >> 20;
        let rest = m32 & 0x0f_ffff;
        let round = rest > 0x8_0000 || (rest == 0x8_0000 && (m3 & 1) == 1);
        if round {
            m3 += 1;
        }
        if m3 == 8 {
            m3 = 0;
            e4 += 1;
        }
        let m3 = m3 & 7;
        if e4 > 15 || (e4 == 15 && m3 == 7) {
            return sign | 0x7E;
        }
        sign | (((e4 as u32) << 3 | m3) as u8)
    } else {
        let full = m32 | 0x80_0000;
        let shift = (21 - e4) as u32;
        let mut sub_m = full >> shift;
        let rbit = (full >> (shift - 1)) & 1;
        let sticky = full & ((1u32 << (shift - 1)) - 1);
        let round = rbit != 0 && (sticky != 0 || (sub_m & 1) != 0);
        if round {
            sub_m += 1;
        }
        let mag = if sub_m >= 8 { 8 } else { sub_m };
        sign | (mag as u8)
    }
}

/// Bit-exact CPU model of the kernel's `emit_e4m3_decode` routine.
fn cpu_e4m3_decode(byte: u8) -> f32 {
    let sign = ((byte >> 7) & 1) as u32;
    let exp = ((byte >> 3) & 0xf) as u32;
    let mant = (byte & 7) as u32;
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        sign_f * (mant as f32) * (1.0 / 512.0)
    } else {
        let e32 = exp + 120;
        let m32 = mant << 20;
        let bits = (e32 << 23) | m32;
        sign_f * f32::from_bits(bits)
    }
}

/// NF4 lookup table (mirrors `int4_quantize::NF4_LOOKUP`).
const NF4_LOOKUP: [f64; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Builds a 1-D tensor descriptor over a device buffer.
fn desc1d<T: GpuFloat>(buf: &DeviceBuffer<T>, n: usize) -> TensorDesc<T> {
    TensorDesc::from_raw(
        buf.as_device_ptr(),
        vec![n as u32],
        vec![1],
        TensorLayout::Nchw,
    )
    .expect("1-D TensorDesc")
}

/// Builds a 1-D mutable tensor descriptor over a device buffer.
fn desc1d_mut<T: GpuFloat>(buf: &DeviceBuffer<T>, n: usize) -> TensorDescMut<T> {
    TensorDescMut::from_raw(
        buf.as_device_ptr(),
        vec![n as u32],
        vec![1],
        TensorLayout::Nchw,
    )
    .expect("1-D TensorDescMut")
}

/// Packs 4-bit values (LSB-first, 8 per `u32`) into packed words.
fn pack_4bit(nibbles: &[u32]) -> Vec<u32> {
    assert!(nibbles.len() % 8 == 0, "len must be a multiple of 8");
    let mut out = vec![0u32; nibbles.len() / 8];
    for (e, &q) in nibbles.iter().enumerate() {
        out[e / 8] |= (q & 0xF) << ((e % 8) * 4);
    }
    out
}

// ---------------------------------------------------------------------------
// INT8
// ---------------------------------------------------------------------------

#[test]
fn int8_quantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 300usize;
    let mut lcg = Lcg::new(0x1234_5678);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-4.0, 4.0)).collect();

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let mut out_q = DeviceBuffer::<i8>::zeroed(n).expect("q out");
    let mut scale = DeviceBuffer::<f32>::zeroed(1).expect("scale");

    let desc = desc1d(&input, n);
    quantize_to_int8(&fx.handle, &desc, &mut out_q, &mut scale).expect("quantize_to_int8");
    fx.stream().synchronize().expect("sync");

    let mut gpu_scale = [0f32; 1];
    scale.copy_to_host(&mut gpu_scale).expect("copy scale");
    let mut gpu_q = vec![0i8; n];
    out_q.copy_to_host(&mut gpu_q).expect("copy q");

    let absmax = data.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let cpu_scale = (absmax / 127.0).max(1e-12);
    assert!(
        close_f32(gpu_scale[0], cpu_scale, 1e-5, 1e-9),
        "int8 scale gpu={} cpu={}",
        gpu_scale[0],
        cpu_scale
    );

    let s = gpu_scale[0];
    for (i, (&x, &q)) in data.iter().zip(gpu_q.iter()).enumerate() {
        let expect = rne((x / s).clamp(-127.0, 127.0)) as i32 as i8;
        assert_eq!(q, expect, "int8 quant element {i}: x={x} scale={s}");
    }
}

#[test]
fn int8_dequantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 192usize;
    let mut lcg = Lcg::new(0xC0FF_EE01);
    let qvals: Vec<i8> = (0..n)
        .map(|_| ((lcg.next_u32() % 255) as i32 - 127) as i8)
        .collect();
    let s = 0.0375f32;

    let input = DeviceBuffer::<i8>::from_host(&qvals).expect("q in");
    let scale = DeviceBuffer::<f32>::from_host(&[s]).expect("scale");
    let out = DeviceBuffer::<f32>::zeroed(n).expect("out");
    let mut out_desc = desc1d_mut(&out, n);

    dequantize_from_int8(&fx.handle, &input, &scale, &mut out_desc, n as u32)
        .expect("dequantize_from_int8");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = qvals.iter().map(|&q| q as f32 * s).collect();
    assert_close_f32(&gpu, &cpu, 1e-6, 1e-9, "int8 dequant");
}

// ---------------------------------------------------------------------------
// FP8 E4M3
// ---------------------------------------------------------------------------

#[test]
fn fp8_quantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 257usize;
    let mut lcg = Lcg::new(0xBEEF_0042);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-6.0, 6.0)).collect();

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let mut out_q = DeviceBuffer::<u8>::zeroed(n).expect("q out");
    let mut scale = DeviceBuffer::<f32>::zeroed(1).expect("scale");

    let desc = desc1d(&input, n);
    quantize_to_fp8(&fx.handle, &desc, &mut out_q, &mut scale).expect("quantize_to_fp8");
    fx.stream().synchronize().expect("sync");

    let mut gpu_scale = [0f32; 1];
    scale.copy_to_host(&mut gpu_scale).expect("copy scale");
    let mut gpu_q = vec![0u8; n];
    out_q.copy_to_host(&mut gpu_q).expect("copy q");

    let absmax = data.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let cpu_scale = (absmax / 448.0).max(1e-12);
    assert!(
        close_f32(gpu_scale[0], cpu_scale, 1e-5, 1e-9),
        "fp8 scale gpu={} cpu={}",
        gpu_scale[0],
        cpu_scale
    );

    let s = gpu_scale[0];
    for (i, (&x, &q)) in data.iter().zip(gpu_q.iter()).enumerate() {
        let scaled = (x / s).clamp(-448.0, 448.0);
        let expect = cpu_e4m3_encode(scaled);
        assert_eq!(q, expect, "fp8 quant byte {i}: x={x} scale={s}");
    }
}

#[test]
fn fp8_dequantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Sweep every representable byte value 0..=255 (decode is a total function).
    let n = 256usize;
    let bytes: Vec<u8> = (0..n).map(|i| i as u8).collect();
    let s = 0.031f32;

    let input = DeviceBuffer::<u8>::from_host(&bytes).expect("byte in");
    let scale = DeviceBuffer::<f32>::from_host(&[s]).expect("scale");
    let out = DeviceBuffer::<f32>::zeroed(n).expect("out");
    let mut out_desc = desc1d_mut(&out, n);

    dequantize_from_fp8(&fx.handle, &input, &scale, &mut out_desc, n as u32)
        .expect("dequantize_from_fp8");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = bytes.iter().map(|&b| cpu_e4m3_decode(b) * s).collect();
    assert_close_f32(&gpu, &cpu, 1e-6, 1e-7, "fp8 dequant");
}

// ---------------------------------------------------------------------------
// INT4 / NF4
// ---------------------------------------------------------------------------

#[test]
fn int4_quantize_pack_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64usize;
    let group_size = 32usize;
    let num_groups = n / group_size;
    let packed_bytes = n / 2;

    let mut lcg = Lcg::new(0x4444_1111);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-5.0, 5.0)).collect();

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let mut out = DeviceBuffer::<u8>::zeroed(packed_bytes).expect("packed out");
    let mut scales = DeviceBuffer::<f32>::zeroed(num_groups).expect("scales");
    let mut zeros = DeviceBuffer::<f32>::zeroed(num_groups).expect("zeros");

    let cfg = Int4QuantConfig::new(group_size, true).expect("cfg");
    quantize_to_int4(
        &fx.handle,
        &input,
        &mut out,
        &mut scales,
        &mut zeros,
        n,
        &cfg,
    )
    .expect("quantize_to_int4");
    fx.stream().synchronize().expect("sync");

    let mut gpu_scales = vec![0f32; num_groups];
    scales.copy_to_host(&mut gpu_scales).expect("copy scales");
    let mut gpu_packed = vec![0u8; packed_bytes];
    out.copy_to_host(&mut gpu_packed).expect("copy packed");

    // Per-group scale = max(absmax / 7, 1e-12)  (kernel uses INT4_SYM_MAX=7).
    let mut cpu_scales = vec![0f32; num_groups];
    for g in 0..num_groups {
        let absmax = data[g * group_size..(g + 1) * group_size]
            .iter()
            .fold(0f32, |m, &x| m.max(x.abs()));
        cpu_scales[g] = (absmax / 7.0).max(1e-12);
    }
    for g in 0..num_groups {
        assert!(
            close_f32(gpu_scales[g], cpu_scales[g], 1e-5, 1e-9),
            "int4 group {g} scale gpu={} cpu={}",
            gpu_scales[g],
            cpu_scales[g]
        );
    }

    // Quantize each element using the read-back scale (truncation toward zero).
    let quant_one = |x: f32, s: f32| -> u32 {
        let scaled = (x / s).clamp(-8.0, 7.0) + 8.0;
        (scaled.trunc() as i64).clamp(0, 15) as u32
    };
    for byte_idx in 0..packed_bytes {
        let even = byte_idx * 2;
        let odd = even + 1;
        let s_even = gpu_scales[even / group_size];
        let s_odd = gpu_scales[odd / group_size];
        let q_even = quant_one(data[even], s_even);
        let q_odd = quant_one(data[odd], s_odd);
        let expect = ((q_odd << 4) | q_even) as u8;
        assert_eq!(gpu_packed[byte_idx], expect, "int4 packed byte {byte_idx}");
    }
}

#[test]
fn int4_dequantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64usize;
    let group_size = 32usize;
    let num_groups = n / group_size;
    let packed_bytes = n / 2;

    let mut lcg = Lcg::new(0x9090_5050);
    let nibbles: Vec<u32> = (0..n).map(|_| lcg.next_u32() % 16).collect();
    let scales_host: Vec<f32> = (0..num_groups).map(|_| lcg.range_f32(0.05, 0.5)).collect();

    // Pack two nibbles per byte (low = even, high = odd).
    let mut packed = vec![0u8; packed_bytes];
    for byte_idx in 0..packed_bytes {
        let lo = nibbles[byte_idx * 2] & 0xF;
        let hi = nibbles[byte_idx * 2 + 1] & 0xF;
        packed[byte_idx] = ((hi << 4) | lo) as u8;
    }

    let input = DeviceBuffer::<u8>::from_host(&packed).expect("packed in");
    let scales = DeviceBuffer::<f32>::from_host(&scales_host).expect("scales");
    let zeros = DeviceBuffer::<f32>::zeroed(num_groups).expect("zeros");
    let mut out_buf = DeviceBuffer::<f32>::zeroed(n).expect("out");

    let cfg = Int4QuantConfig::new(group_size, true).expect("cfg");
    dequantize_int4(&fx.handle, &input, &scales, &zeros, &mut out_buf, n, &cfg)
        .expect("dequantize_int4");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out_buf.copy_to_host(&mut gpu).expect("copy out");

    // Symmetric: out[i] = (nibble - 8) * scale[group].
    let cpu: Vec<f32> = (0..n)
        .map(|i| {
            let s = scales_host[i / group_size];
            (nibbles[i] as f32 - 8.0) * s
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-6, 1e-7, "int4 dequant");
}

#[test]
fn nf4_pack_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64usize;
    let group_size = 32usize;
    let num_groups = n / group_size;
    let packed_bytes = n / 2;

    let mut lcg = Lcg::new(0x5A5A_3C3C);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-3.0, 3.0)).collect();

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let mut out = DeviceBuffer::<u8>::zeroed(packed_bytes).expect("packed out");
    let mut scales = DeviceBuffer::<f32>::zeroed(num_groups).expect("scales");

    quantize_to_nf4(&fx.handle, &input, &mut out, &mut scales, n, group_size)
        .expect("quantize_to_nf4");
    fx.stream().synchronize().expect("sync");

    let mut gpu_scales = vec![0f32; num_groups];
    scales.copy_to_host(&mut gpu_scales).expect("copy scales");
    let mut gpu_packed = vec![0u8; packed_bytes];
    out.copy_to_host(&mut gpu_packed).expect("copy packed");

    // Midpoints between consecutive NF4 codewords (computed in f64, used as f32).
    let midpoints: Vec<f32> = (0..15)
        .map(|i| ((NF4_LOOKUP[i] + NF4_LOOKUP[i + 1]) / 2.0) as f32)
        .collect();
    let nf4_code = |x: f32, s: f32| -> u32 {
        let safe = s.max(1e-12);
        let normalized = x / safe;
        let mut code = 0u32;
        for (i, &mp) in midpoints.iter().enumerate() {
            if normalized > mp {
                code = (i + 1) as u32;
            }
        }
        code
    };

    for byte_idx in 0..packed_bytes {
        let even = byte_idx * 2;
        let odd = even + 1;
        let s_even = gpu_scales[even / group_size];
        let s_odd = gpu_scales[odd / group_size];
        let lo = nf4_code(data[even], s_even);
        let hi = nf4_code(data[odd], s_odd);
        let expect = ((hi << 4) | lo) as u8;
        assert_eq!(gpu_packed[byte_idx], expect, "nf4 packed byte {byte_idx}");
    }
}

// ---------------------------------------------------------------------------
// Block-scaled FP4 proxy
// ---------------------------------------------------------------------------

#[test]
fn block_quantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128usize;
    let block_size = 32u32;
    let num_blocks = n / block_size as usize;

    let mut lcg = Lcg::new(0x7777_2222);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-5.0, 5.0)).collect();

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let mut out = DeviceBuffer::<u8>::zeroed(n).expect("byte out");
    let mut scales = DeviceBuffer::<f32>::zeroed(num_blocks).expect("scales");

    let desc = desc1d(&input, n);
    quantize_block_scaled(&fx.handle, &desc, &mut out, &mut scales, block_size)
        .expect("quantize_block_scaled");
    fx.stream().synchronize().expect("sync");

    let mut gpu_scales = vec![0f32; num_blocks];
    scales.copy_to_host(&mut gpu_scales).expect("copy scales");
    let mut gpu_bytes = vec![0u8; n];
    out.copy_to_host(&mut gpu_bytes).expect("copy bytes");

    // Per-block scale = max(absmax / 6, 1e-12).
    let bs = block_size as usize;
    let mut cpu_scales = vec![0f32; num_blocks];
    for blk in 0..num_blocks {
        let absmax = data[blk * bs..(blk + 1) * bs]
            .iter()
            .fold(0f32, |m, &x| m.max(x.abs()));
        cpu_scales[blk] = (absmax / 6.0).max(1e-12);
    }
    for blk in 0..num_blocks {
        assert!(
            close_f32(gpu_scales[blk], cpu_scales[blk], 1e-5, 1e-9),
            "block {blk} scale gpu={} cpu={}",
            gpu_scales[blk],
            cpu_scales[blk]
        );
    }

    // out[i] = clamp(rne(clamp(x/scale, -6, 6)) + 6, 0, 255).
    for i in 0..n {
        let s = gpu_scales[i / bs];
        let q = rne((data[i] / s).clamp(-6.0, 6.0)) as i32 + 6;
        let expect = q.clamp(0, 255) as u8;
        assert_eq!(gpu_bytes[i], expect, "block byte {i}: x={}", data[i]);
    }
}

// ---------------------------------------------------------------------------
// QAT (direct PTX)
// ---------------------------------------------------------------------------

fn qat_config(sm: SmVersion, symmetry: QatSymmetry) -> QatConfig {
    QatConfig {
        bit_width: QatBitWidth::Int8,
        symmetry,
        granularity: QatGranularity::PerTensor,
        observer: ObserverMode::MinMax,
        sm_version: sm,
        float_type: oxicuda_ptx::ir::PtxType::F32,
    }
}

#[test]
fn qat_fake_quantize_symmetric_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 320usize;
    let mut lcg = Lcg::new(0x1357_9BDF);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-5.0, 5.0)).collect();
    let scale = 0.05f32;
    let zero_point = 0i32;
    let (qmin, qmax) = (-128i32, 127i32);

    let fq = FakeQuantize::new(qat_config(fx.sm, QatSymmetry::Symmetric)).expect("fq");
    let ptx = fq.generate_fake_quantize_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let out = DeviceBuffer::<f32>::zeroed(n).expect("out");

    let grid = ceil_div(n as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        input.as_device_ptr(),
        out.as_device_ptr(),
        scale,
        zero_point,
        n as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch fake_quantize");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = data
        .iter()
        .map(|&x| {
            let q = (rne(x / scale) as i32 + zero_point).clamp(qmin, qmax);
            (q - zero_point) as f32 * scale
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-6, "qat fake_quantize symmetric");
}

#[test]
fn qat_fake_quantize_asymmetric_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256usize;
    let mut lcg = Lcg::new(0x2468_ACE0);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-5.0, 5.0)).collect();
    let scale = 0.05f32;
    let zero_point = 10i32;
    let (qmin, qmax) = (-128i32, 127i32);

    let fq = FakeQuantize::new(qat_config(fx.sm, QatSymmetry::Asymmetric)).expect("fq");
    let ptx = fq.generate_fake_quantize_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let out = DeviceBuffer::<f32>::zeroed(n).expect("out");

    let grid = ceil_div(n as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        input.as_device_ptr(),
        out.as_device_ptr(),
        scale,
        zero_point,
        n as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch fake_quantize");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = data
        .iter()
        .map(|&x| {
            let q = (rne(x / scale) as i32 + zero_point).clamp(qmin, qmax);
            (q - zero_point) as f32 * scale
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-6, "qat fake_quantize asymmetric");
}

#[test]
fn qat_ste_backward_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 300usize;
    let mut lcg = Lcg::new(0x0F0F_E1E1);
    let x: Vec<f32> = (0..n).map(|_| lcg.range_f32(-5.0, 5.0)).collect();
    let grad_in: Vec<f32> = (0..n).map(|_| lcg.range_f32(-2.0, 2.0)).collect();
    let qmin_float = -2.0f32;
    let qmax_float = 3.0f32;

    let fq = FakeQuantize::new(qat_config(fx.sm, QatSymmetry::Symmetric)).expect("fq");
    let ptx = fq.generate_ste_backward_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let x_buf = DeviceBuffer::<f32>::from_host(&x).expect("x");
    let g_buf = DeviceBuffer::<f32>::from_host(&grad_in).expect("grad_in");
    let out = DeviceBuffer::<f32>::zeroed(n).expect("grad_out");

    let grid = ceil_div(n as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        x_buf.as_device_ptr(),
        g_buf.as_device_ptr(),
        out.as_device_ptr(),
        qmin_float,
        qmax_float,
        n as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch ste_backward");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; n];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = x
        .iter()
        .zip(grad_in.iter())
        .map(|(&xi, &gi)| {
            if xi >= qmin_float && xi <= qmax_float {
                gi
            } else {
                0.0
            }
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 0.0, 0.0, "qat ste_backward");
}

#[test]
fn qat_observer_minmax_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200usize;
    let mut lcg = Lcg::new(0xDEAD_1234);
    let data: Vec<f32> = (0..n).map(|_| lcg.range_f32(-7.0, 11.0)).collect();

    let fq = FakeQuantize::new(qat_config(fx.sm, QatSymmetry::Symmetric)).expect("fq");
    let ptx = fq.generate_observer_ptx().expect("ptx");
    // The reduction kernel previously emitted invalid PTX (`[smem + r*4]`);
    // a successful JIT here confirms the fix assembles on the device.
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let input = DeviceBuffer::<f32>::from_host(&data).expect("input");
    let out = DeviceBuffer::<f32>::zeroed(2).expect("out min/max");

    // Single block: out[0]/out[1] are written by block 0 only.
    let params = LaunchParams::new(1u32, 256u32);
    let args = (input.as_device_ptr(), out.as_device_ptr(), n as u32);
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch observer");
    fx.stream().synchronize().expect("sync");

    let mut gpu = [0f32; 2];
    out.copy_to_host(&mut gpu).expect("copy out");

    let cpu_min = data.iter().fold(f32::INFINITY, |m, &x| m.min(x));
    let cpu_max = data.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    assert!(
        close_f32(gpu[0], cpu_min, 0.0, 0.0),
        "observer min gpu={} cpu={cpu_min}",
        gpu[0]
    );
    assert!(
        close_f32(gpu[1], cpu_max, 0.0, 0.0),
        "observer max gpu={} cpu={cpu_max}",
        gpu[1]
    );
}

// ---------------------------------------------------------------------------
// GPTQ / AWQ
// ---------------------------------------------------------------------------

fn gptq_plan(rows: usize, cols: usize, group_size: usize) -> WeightQuantPlan {
    let cfg = GptqConfig {
        bits: 4,
        group_size,
        block_size: group_size,
        damp_percent: 0.01,
        symmetric: true,
        act_order: false,
        true_sequential: true,
    };
    WeightQuantPlan::new(WeightQuantMethod::Gptq(cfg), rows, cols).expect("gptq plan")
}

#[test]
fn gptq_dequantize_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let rows = 3usize;
    let cols = 64usize;
    let group_size = 32usize;
    let gpr = cols / group_size;
    let total = rows * cols;

    let mut lcg = Lcg::new(0xABCD_1234);
    let nibbles: Vec<u32> = (0..total).map(|_| lcg.next_u32() % 16).collect();
    let packed = pack_4bit(&nibbles);
    let scales_host: Vec<f32> = (0..rows * gpr).map(|_| lcg.range_f32(0.02, 0.4)).collect();

    let packed_buf = DeviceBuffer::<u32>::from_host(&packed).expect("packed");
    let scale_buf = DeviceBuffer::<f32>::from_host(&scales_host).expect("scales");
    let zero_buf = DeviceBuffer::<f32>::zeroed(rows * gpr).expect("zeros");
    let out_buf = DeviceBuffer::<f32>::zeroed(total).expect("out");

    let plan = gptq_plan(rows, cols, group_size);
    let ptx = plan.generate_gptq_dequantize_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(total as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        packed_buf.as_device_ptr(),
        scale_buf.as_device_ptr(),
        zero_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        total as u32,
        cols as u32,
        group_size as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch gptq dequantize");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; total];
    out_buf.copy_to_host(&mut gpu).expect("copy out");

    let cpu: Vec<f32> = (0..total)
        .map(|e| {
            let row = e / cols;
            let col = e % cols;
            let s_idx = row * gpr + col / group_size;
            nibbles[e] as f32 * scales_host[s_idx]
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-7, "gptq dequantize");
}

#[test]
fn fused_dequant_gemv_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let rows = 3usize;
    let cols = 64usize;
    let group_size = 32usize;
    let gpr = cols / group_size;
    let total = rows * cols;

    let mut lcg = Lcg::new(0x55AA_9933);
    let nibbles: Vec<u32> = (0..total).map(|_| lcg.next_u32() % 16).collect();
    let packed = pack_4bit(&nibbles);
    let scales_host: Vec<f32> = (0..rows * gpr).map(|_| lcg.range_f32(0.02, 0.3)).collect();
    let x_host: Vec<f32> = (0..cols).map(|_| lcg.range_f32(-2.0, 2.0)).collect();

    let packed_buf = DeviceBuffer::<u32>::from_host(&packed).expect("packed");
    let scale_buf = DeviceBuffer::<f32>::from_host(&scales_host).expect("scales");
    let zero_buf = DeviceBuffer::<f32>::zeroed(rows * gpr).expect("zeros");
    let x_buf = DeviceBuffer::<f32>::from_host(&x_host).expect("x");
    let y_buf = DeviceBuffer::<f32>::zeroed(rows).expect("y");

    let plan = gptq_plan(rows, cols, group_size);
    let ptx = plan.generate_fused_dequant_gemv_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(rows as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        packed_buf.as_device_ptr(),
        scale_buf.as_device_ptr(),
        zero_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        rows as u32,
        cols as u32,
        group_size as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch fused gemv");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0f32; rows];
    y_buf.copy_to_host(&mut gpu).expect("copy y");

    // y[row] = sum_col (q*scale) * x[col]; mirror the kernel's fma over cols.
    let cpu: Vec<f32> = (0..rows)
        .map(|row| {
            let mut acc = 0f32;
            for col in 0..cols {
                let e = row * cols + col;
                let s = scales_host[row * gpr + col / group_size];
                let dq = nibbles[e] as f32 * s;
                acc = dq.mul_add(x_host[col], acc);
            }
            acc
        })
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-4, 1e-4, "fused dequant gemv");
}

#[test]
fn gptq_quantize_load_launch() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The simplified GPTQ-quantize is a proxy (no inverse-Hessian / residual
    // propagation); only assert it assembles and launches fault-free.
    let rows = 4usize;
    let cols = 32usize;
    let group_size = 32usize;
    let gpr = cols / group_size;
    let total = rows * cols;
    let packed_words = total / 8;

    let mut lcg = Lcg::new(0x0102_0304);
    let weight: Vec<f32> = (0..total).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let hessian: Vec<f32> = (0..cols).map(|_| lcg.range_f32(0.1, 2.0)).collect();

    let w_buf = DeviceBuffer::<f32>::from_host(&weight).expect("weight");
    let h_buf = DeviceBuffer::<f32>::from_host(&hessian).expect("hessian");
    let out_buf = DeviceBuffer::<u32>::zeroed(packed_words).expect("packed (zeroed)");
    let scale_buf = DeviceBuffer::<f32>::zeroed(rows * gpr).expect("scales");
    let zero_buf = DeviceBuffer::<f32>::zeroed(rows * gpr).expect("zeros");

    let plan = gptq_plan(rows, cols, group_size);
    let ptx = plan.generate_gptq_quantize_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(rows as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        w_buf.as_device_ptr(),
        h_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        scale_buf.as_device_ptr(),
        zero_buf.as_device_ptr(),
        rows as u32,
        cols as u32,
        group_size as u32,
        group_size as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch gptq quantize");
    fx.stream()
        .synchronize()
        .expect("gptq quantize must run fault-free");
}

#[test]
fn awq_scale_search_load_launch() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // AWQ scale-search ignores its own alpha grid (inexact rsqrt proxy);
    // only assert it assembles and launches fault-free.
    let rows = 4usize;
    let cols = 32usize;
    let group_size = 32usize;
    let num_channels = cols;

    let mut lcg = Lcg::new(0x0A0B_0C0D);
    let act: Vec<f32> = (0..num_channels)
        .map(|_| lcg.range_f32(0.01, 4.0))
        .collect();
    let weight: Vec<f32> = (0..rows * cols).map(|_| lcg.range_f32(-1.0, 1.0)).collect();

    let act_buf = DeviceBuffer::<f32>::from_host(&act).expect("act");
    let w_buf = DeviceBuffer::<f32>::from_host(&weight).expect("weight");
    let scale_out = DeviceBuffer::<f32>::zeroed(num_channels).expect("scale_out");
    let best_alpha = DeviceBuffer::<f32>::zeroed(num_channels).expect("best_alpha");

    let cfg = AwqConfig {
        bits: 4,
        group_size,
        zero_point: true,
        search_alpha_min: 0.0,
        search_alpha_max: 1.0,
        search_steps: 20,
    };
    let plan = WeightQuantPlan::new(WeightQuantMethod::Awq(cfg), rows, cols).expect("awq plan");
    let ptx = plan.generate_awq_scale_search_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(num_channels as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        act_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        scale_out.as_device_ptr(),
        best_alpha.as_device_ptr(),
        num_channels as u32,
        cols as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch awq scale search");
    fx.stream()
        .synchronize()
        .expect("awq scale search must run fault-free");

    // sqrt(|act|) is well-defined; spot-check the scale output stays finite.
    let mut gpu = vec![0f32; num_channels];
    scale_out.copy_to_host(&mut gpu).expect("copy scale_out");
    assert!(
        gpu.iter().all(|v| v.is_finite() && *v >= 0.0),
        "awq scale outputs must be finite and non-negative"
    );
}

#[test]
fn awq_quantize_load_launch() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // AWQ-quantize reads group scales as input and hardcodes zero_point=0;
    // it is a proxy — only assert it assembles and launches fault-free.
    let rows = 4usize;
    let cols = 32usize;
    let group_size = 32usize;
    let gpr = cols / group_size;
    let total = rows * cols;
    let packed_words = total / 8;

    let mut lcg = Lcg::new(0x0E0F_1011);
    let weight: Vec<f32> = (0..total).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let ch_scales: Vec<f32> = (0..cols).map(|_| lcg.range_f32(0.5, 1.5)).collect();
    let g_scales: Vec<f32> = (0..rows * gpr).map(|_| lcg.range_f32(0.05, 0.5)).collect();

    let w_buf = DeviceBuffer::<f32>::from_host(&weight).expect("weight");
    let cs_buf = DeviceBuffer::<f32>::from_host(&ch_scales).expect("channel scales");
    let out_buf = DeviceBuffer::<u32>::zeroed(packed_words).expect("packed (zeroed)");
    let scale_buf = DeviceBuffer::<f32>::from_host(&g_scales).expect("group scales");
    let zero_buf = DeviceBuffer::<f32>::zeroed(rows * gpr).expect("zeros");

    let cfg = AwqConfig {
        bits: 4,
        group_size,
        zero_point: true,
        search_alpha_min: 0.0,
        search_alpha_max: 1.0,
        search_steps: 20,
    };
    let plan = WeightQuantPlan::new(WeightQuantMethod::Awq(cfg), rows, cols).expect("awq plan");
    let ptx = plan.generate_awq_quantize_ptx().expect("ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(total as u32, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        w_buf.as_device_ptr(),
        cs_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        scale_buf.as_device_ptr(),
        zero_buf.as_device_ptr(),
        total as u32,
        cols as u32,
        group_size as u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch awq quantize");
    fx.stream()
        .synchronize()
        .expect("awq quantize must run fault-free");
}
