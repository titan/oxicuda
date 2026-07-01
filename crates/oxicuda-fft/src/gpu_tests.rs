//! On-device GPU validation for the generated PTX kernels in `oxicuda-fft`.
//!
//! Mirrors the canonical harness in `oxicuda-cs`/`oxicuda-ot`: each test
//! JIT-compiles a kernel's PTX for the live device via `Module::from_ptx`,
//! launches it through `oxicuda-launch`, copies results back, and asserts
//! numerical equivalence to an independent CPU reference. Every test skips
//! (returns early) when no CUDA device is present.
//!
//! The single-kernel Stockham FFT, the batch FFT, and the large multi-pass
//! FFT already have CPU-DFT oracle tests in `tests/gpu_fft_numerical.rs`; this
//! module covers the remaining distinct kernel algorithms and provides the
//! invalid-PTX prescreen.
//!
//! ## PTX bugs found and fixed on the live RTX A4000 (sm_86)
//!
//! * **`precompute_window`** (callbacks): emitted `cos.approx.f64` (no such
//!   instruction — `cos.approx` is f32-only), a user register named `%tid`
//!   that shadows the `%tid` special register (so `mov %tid, %tid.x` parsed
//!   `.x` as a video selector), plus undeclared predicate/temp registers
//!   (`%p_done`, `%tmp1/2`, `%t1..4`). ptxas rejected the module outright.
//! * **`fft_fp16_butterfly_*` / `fft_fp16_twiddle_*`** (half_precision): the
//!   loads were `ld.global.f16` into registers the IR allocator declares as
//!   `.b16`, and — more fundamentally — the allocator routes F16 *and* F32 to
//!   one shared `%f` prefix declared with the first-used type (`.b16`), so the
//!   f32 math registers were declared 16-bit and every `cvt`/`ld` mismatched.
//!   Fixed by carrying the half values in a dedicated manually declared
//!   `.b16 %h` bank and using `.b16` global loads/stores, keeping the `%f`
//!   pool pure-f32.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

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

/// Map a numeric compute-capability (e.g. 86) to a builder `SmVersion`.
fn sm_version(sm: u32) -> SmVersion {
    let major = (sm / 10) as i32;
    let minor = (sm % 10) as i32;
    SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80)
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Small deterministic LCG. Normalisation is by 2^32 (never 2^31).
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = f64::from(self.next_u32()) / f64::from(u32::MAX); // /2^32 scale
        (f64::from(lo) + f64::from(hi - lo) * u) as f32
    }
}

/// Complex interleaving helpers for the `[re, im, re, im, ...]` layout.
fn interleave(signal: &[(f32, f32)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(signal.len() * 2);
    for &(re, im) in signal {
        out.push(re);
        out.push(im);
    }
    out
}

fn deinterleave(flat: &[f32]) -> Vec<(f32, f32)> {
    flat.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

// IEEE-754 binary16 codec (used only for the FP16 kernels). The oracle decodes
// the exact uploaded half bits before computing, so encoder rounding never
// causes a false mismatch; only the GPU's output rounding contributes error.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e: i32 = -1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127 + 1) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((u32::from(exp) + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn f16_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        return sign; // underflow -> signed zero (inputs avoid this regime)
    }
    let mant10 = (mant >> 13) as u16;
    let remainder = mant & 0x1fff;
    let mut h = sign | ((exp as u16) << 10) | mant10;
    let half = 0x1000;
    if remainder > half || (remainder == half && (mant10 & 1) == 1) {
        h += 1; // round to nearest, ties to even (carry into exp is correct)
    }
    h
}

// ===========================================================================
// PTX prescreen — generate + JIT-compile every kernel (no oracle).
// ===========================================================================

/// Build every kernel's `(label, entry_name, ptx)` for the given SM version.
fn all_kernels(sm: SmVersion) -> Vec<(&'static str, String, String)> {
    use crate::callbacks::{WindowFunction, generate_window_ptx};
    use crate::conv_fft::{ConvolutionMode, CrossCorrelationPlan, FftConv2dPlan, FftConvPlan};
    use crate::half_precision::{
        AccumulationMode, generate_fp16_butterfly_ptx, generate_fp16_twiddle_ptx,
    };
    use crate::inverse_scaling::{generate_fused_butterfly_scale_ptx, generate_scale_kernel_ptx};
    use crate::kernels::bank_conflict_free::BankConflictFreeStockham;
    use crate::kernels::batch_fft::generate_batch_fft_kernel;
    use crate::kernels::fused_batch::FusedBatchFft;
    use crate::kernels::large_fft::generate_large_fft_pass;
    use crate::kernels::stockham::{generate_single_kernel, generate_stage_kernel};
    use crate::kernels::transpose::generate_transpose_kernel;
    use crate::plan::FftStrategy;
    use crate::pruned::{PrunedFftConfig, PrunedStage, generate_pruned_butterfly_ptx};
    use crate::radix::pfa::PrimeFactorFft;
    use crate::transforms::real_fft::RealFft;

    let p = FftPrecision::Single;
    let mut out: Vec<(&'static str, String, String)> = Vec::new();

    let w = generate_window_ptx(&WindowFunction::Hann, 64, sm).expect("window ptx");
    out.push(("window_hann", "precompute_window".to_string(), w));

    let conv = FftConvPlan::new(40, 25, ConvolutionMode::Full, p).expect("conv plan");
    let pm = conv.generate_pointwise_multiply_ptx().expect("pm ptx");
    out.push(("conv_pointwise_mul", pm.entry_name.clone(), pm.source));
    let zp = conv.generate_zero_pad_ptx().expect("zp ptx");
    out.push(("conv_zero_pad", zp.entry_name.clone(), zp.source));
    let xc = CrossCorrelationPlan::new(40, 25, p).expect("xcorr plan");
    let cm = xc.generate_conj_multiply_ptx().expect("cm ptx");
    out.push(("conv_conj_mul", cm.entry_name.clone(), cm.source));
    let conv2d = FftConv2dPlan::new(12, 12, 5, 5, ConvolutionMode::Full, p).expect("conv2d plan");
    let zp2 = conv2d.generate_zero_pad_2d_ptx().expect("zp2 ptx");
    out.push(("conv_zero_pad_2d", zp2.entry_name.clone(), zp2.source));

    let sc = generate_scale_kernel_ptx(64, 1.0 / 64.0, p, sm).expect("scale ptx");
    out.push(("scale", "scale_fft_n64_f32".to_string(), sc));
    let fbs = generate_fused_butterfly_scale_ptx(2, 0.5, p, sm).expect("fbs ptx");
    out.push((
        "fused_butterfly_scale",
        "fused_butterfly_scale_r2_f32".to_string(),
        fbs,
    ));

    let pcfg = PrunedFftConfig {
        fft_size: 256,
        input_nonzero_count: 256,
        output_needed_count: 256,
        direction: FftDirection::Forward,
        precision: p,
        sm_version: sm,
    };
    let stage = PrunedStage {
        stage_index: 2,
        radix: 2,
        active_butterflies: 128,
        total_butterflies: 128,
        can_skip_entirely: false,
    };
    let pr = generate_pruned_butterfly_ptx(&pcfg, &stage).expect("pruned ptx");
    out.push((
        "pruned_butterfly",
        "pruned_butterfly_f32_n256_s2_a128".to_string(),
        pr,
    ));

    let fp16b = generate_fp16_butterfly_ptx(2, AccumulationMode::Fp32, sm).expect("fp16b ptx");
    out.push((
        "fp16_butterfly",
        "fft_fp16_butterfly_r2_fp32acc".to_string(),
        fp16b,
    ));
    let fp16t = generate_fp16_twiddle_ptx(64, sm).expect("fp16t ptx");
    out.push(("fp16_twiddle", "fft_fp16_twiddle_n64".to_string(), fp16t));

    let bf = generate_batch_fft_kernel(8, 4, p, FftDirection::Forward, sm).expect("batch ptx");
    out.push(("batch_fft", "fft_batch_f32_n8_b4_fwd".to_string(), bf));

    let lf = generate_large_fft_pass(64, 2, 1, p, FftDirection::Forward, sm).expect("large ptx");
    out.push((
        "large_fft_pass",
        "fft_large_pass_f32_n64_r2_s1_fwd".to_string(),
        lf,
    ));

    let bcf = BankConflictFreeStockham::new(64, p, FftDirection::Forward);
    let strat64 = FftStrategy {
        radices: vec![8, 8],
        strides: vec![1, 8],
        single_kernel: true,
    };
    let bcf_ptx = bcf.generate_kernel(&strat64, sm).expect("bcf ptx");
    out.push(("bank_conflict_free", "fft_bcf_f32_n64".to_string(), bcf_ptx));

    let strat8 = FftStrategy {
        radices: vec![2, 2, 2],
        strides: vec![1, 2, 4],
        single_kernel: true,
    };
    let sk =
        generate_single_kernel(8, &strat8, 1, p, FftDirection::Forward, sm).expect("single ptx");
    out.push((
        "stockham_single",
        "fft_stockham_f32_n8_b1_fwd".to_string(),
        sk,
    ));

    let stg =
        generate_stage_kernel(64, 2, 0, 6, 1, p, FftDirection::Forward, sm).expect("stage ptx");
    out.push((
        "stockham_stage",
        "fft_stockham_stage_f32_n64_r2_s0of6_fwd".to_string(),
        stg,
    ));

    let fused = FusedBatchFft::new(64, 16, p, FftDirection::Forward);
    let fpb = fused.ffts_per_block();
    let fused_ptx = fused.generate_kernel(sm).expect("fused ptx");
    out.push((
        "fused_batch",
        format!("fft_fused_f32_n64_fpb{fpb}"),
        fused_ptx,
    ));

    let tr = generate_transpose_kernel(64, 32, p, sm).expect("transpose ptx");
    out.push(("transpose", "transpose_f32_64x32".to_string(), tr));

    let pfa = PrimeFactorFft::new(15).expect("pfa plan");
    let pfa_ptx = pfa
        .generate_kernel(p, FftDirection::Forward, sm)
        .expect("pfa ptx");
    out.push(("pfa", "pfa_fft".to_string(), pfa_ptx));

    let rf = RealFft::new(16, p).expect("real fft");
    let pack = rf.generate_pack_kernel(sm).expect("pack ptx");
    out.push(("real_fft_pack", pack.entry_name.clone(), pack.source));
    let unpack = rf.generate_unpack_kernel(sm).expect("unpack ptx");
    out.push(("real_fft_unpack", unpack.entry_name.clone(), unpack.source));

    out
}

#[test]
fn prescreen_dump_ptx() {
    // Always runnable (no GPU): dump every kernel's sm_86 PTX for offline
    // `ptxas` inspection.
    let sm = SmVersion::Sm86;
    let dir = std::env::temp_dir().join("oxicuda_fft_ptx");
    let _ = std::fs::create_dir_all(&dir);
    for (label, entry, ptx) in all_kernels(sm) {
        let path = dir.join(format!("{label}__{entry}.ptx"));
        std::fs::write(&path, &ptx).expect("write ptx");
    }
    eprintln!("PTX dumped to {}", dir.display());
}

#[test]
fn prescreen_jit_all() {
    // On-device: every kernel must JIT-compile (ptxas-accept) for the live SM.
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let sm = sm_version(fx.sm);
    for (label, entry, ptx) in all_kernels(sm) {
        let module = Module::from_ptx(&ptx)
            .unwrap_or_else(|e| panic!("ptxas rejected `{label}` ({entry}): {e}"));
        Kernel::from_module(Arc::new(module), &entry)
            .unwrap_or_else(|e| panic!("entry `{entry}` missing for `{label}`: {e}"));
    }
}

// ===========================================================================
// 1. precompute_window — CRATE ORACLE (callbacks::window_coefficient)
// ===========================================================================

#[test]
fn window_hann_matches_crate() {
    use crate::callbacks::{WindowFunction, generate_window_ptx, window_coefficient};

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let ptx = generate_window_ptx(&WindowFunction::Hann, n, sm_version(fx.sm)).expect("window ptx");
    let kernel = load_kernel(&ptx, "precompute_window");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_out");
    // The kernel indexes purely by %tid.x (no block offset) -> single block.
    let params = LaunchParams::new(1_u32, n as u32);
    kernel
        .launch(&params, &stream, &(d_out.as_device_ptr(),))
        .expect("launch precompute_window");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f64; n];
    d_out.copy_to_host(&mut got).expect("copy out");

    for (i, &g) in got.iter().enumerate() {
        let want = window_coefficient(&WindowFunction::Hann, i, n);
        assert!((g - want).abs() < 1e-3, "Hann[{i}]: gpu={g} cpu={want}");
    }
}

// ===========================================================================
// 2/3. conv pointwise & conjugate complex multiply — HOST RE-DERIVATION
// ===========================================================================

fn run_complex_multiply(fx: &GpuFixture, conjugate: bool) {
    use crate::conv_fft::{ConvolutionMode, CrossCorrelationPlan, FftConvPlan};

    let n = 64_usize;
    let mut rng = LcgRng::new(if conjugate { 0xC0FE } else { 0x00A1 });
    let a: Vec<(f32, f32)> = (0..n)
        .map(|_| (rng.next_f32(-1.5, 1.5), rng.next_f32(-1.5, 1.5)))
        .collect();
    let b: Vec<(f32, f32)> = (0..n)
        .map(|_| (rng.next_f32(-1.5, 1.5), rng.next_f32(-1.5, 1.5)))
        .collect();

    let (ptx, entry) = if conjugate {
        let plan = CrossCorrelationPlan::new(40, 25, FftPrecision::Single).expect("xcorr");
        let m = plan.generate_conj_multiply_ptx().expect("cm");
        (m.source, m.entry_name)
    } else {
        let plan =
            FftConvPlan::new(40, 25, ConvolutionMode::Full, FftPrecision::Single).expect("conv");
        let m = plan.generate_pointwise_multiply_ptx().expect("pm");
        (m.source, m.entry_name)
    };

    // Expected: standard `a*b`, or conjugate `conj(a)*b`.
    let expected: Vec<(f32, f32)> = a
        .iter()
        .zip(b.iter())
        .map(|(&(ar, ai), &(br, bi))| {
            let ai = if conjugate { -ai } else { ai };
            (ar * br - ai * bi, ar * bi + ai * br)
        })
        .collect();

    let kernel = load_kernel(&ptx, &entry);
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&interleave(&a)).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&interleave(&b)).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 2]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch complex multiply");
    stream.synchronize().expect("sync");

    let mut flat = vec![0.0_f32; n * 2];
    d_out.copy_to_host(&mut flat).expect("copy");
    let got = deinterleave(&flat);

    for (i, (&(gr, gi), &(er, ei))) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(gr, er, 1e-5, 1e-5) && close(gi, ei, 1e-5, 1e-5),
            "mul[{i}] (conj={conjugate}): gpu=({gr},{gi}) cpu=({er},{ei})"
        );
    }
}

#[test]
fn conv_pointwise_multiply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_complex_multiply(&fx, false);
}

#[test]
fn conv_conjugate_multiply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_complex_multiply(&fx, true);
}

// ===========================================================================
// 4. conv zero-pad (1-D) — HOST RE-DERIVATION (copy first input_len, else 0)
// ===========================================================================

#[test]
fn conv_zero_pad_matches_host() {
    use crate::conv_fft::{ConvolutionMode, FftConvPlan};

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = FftConvPlan::new(40, 25, ConvolutionMode::Full, FftPrecision::Single).expect("plan");
    let m = plan.generate_zero_pad_ptx().expect("zp");

    // The kernel works on individual floats: copy `input_len`, zero up to
    // `total_len`.
    let total_len = 50_u32;
    let input_len = 30_u32;
    let mut rng = LcgRng::new(0x2EED);
    let input: Vec<f32> = (0..input_len).map(|_| rng.next_f32(-3.0, 3.0)).collect();

    let expected: Vec<f32> = (0..total_len)
        .map(|i| {
            if i < input_len {
                input[i as usize]
            } else {
                0.0
            }
        })
        .collect();

    let kernel = load_kernel(&m.source, &m.entry_name);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![7.0_f32; total_len as usize]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total_len, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                input_len,
                total_len,
            ),
        )
        .expect("launch zero_pad");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; total_len as usize];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "zero_pad[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 5. conv zero-pad (2-D) — HOST RE-DERIVATION (row/col copy-or-zero)
// ===========================================================================

#[test]
fn conv_zero_pad_2d_matches_host() {
    use crate::conv_fft::{ConvolutionMode, FftConv2dPlan};

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = FftConv2dPlan::new(12, 12, 5, 5, ConvolutionMode::Full, FftPrecision::Single)
        .expect("plan");
    let m = plan.generate_zero_pad_2d_ptx().expect("zp2");

    let out_h = 16_u32;
    let out_w = 16_u32;
    let in_h = 12_u32;
    let in_w = 12_u32;
    let total = out_h * out_w;
    let mut rng = LcgRng::new(0x2D2D);
    let input: Vec<f32> = (0..in_h * in_w).map(|_| rng.next_f32(-2.0, 2.0)).collect();

    let mut expected = vec![0.0_f32; total as usize];
    for r in 0..out_h {
        for c in 0..out_w {
            let idx = (r * out_w + c) as usize;
            expected[idx] = if r < in_h && c < in_w {
                input[(r * in_w + c) as usize]
            } else {
                0.0
            };
        }
    }

    let kernel = load_kernel(&m.source, &m.entry_name);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![9.0_f32; total as usize]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                in_h,
                in_w,
                out_w,
                total,
            ),
        )
        .expect("launch zero_pad_2d");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; total as usize];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "zero_pad_2d[{i}]: gpu={g} cpu={e}"
        );
    }
}

// ===========================================================================
// 6. scale — HOST RE-DERIVATION (out[i] = scale * data[i])
// ===========================================================================

#[test]
fn scale_matches_host() {
    use crate::inverse_scaling::generate_scale_kernel_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let scale = 1.0_f64 / 64.0;
    let ptx = generate_scale_kernel_ptx(n, scale, FftPrecision::Single, sm_version(fx.sm))
        .expect("scale ptx");

    let total_floats = n * 2;
    let mut rng = LcgRng::new(0x5CA1);
    let data: Vec<f32> = (0..total_floats).map(|_| rng.next_f32(-4.0, 4.0)).collect();
    let expected: Vec<f32> = data.iter().map(|&v| v * scale as f32).collect();

    let kernel = load_kernel(&ptx, "scale_fft_n64_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total_floats as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), total_floats as u32),
        )
        .expect("launch scale");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; total_floats];
    d_data.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(close(g, e, 1e-6, 1e-6), "scale[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 7. fused butterfly+scale — HOST RE-DERIVATION
//
// Despite the name the kernel currently applies *only* the per-leg scale (the
// "butterfly" is the documented fusion-pattern skeleton). For radix-2 with
// stride == n_val every complex element is scaled exactly once.
// ===========================================================================

#[test]
fn fused_butterfly_scale_matches_host() {
    use crate::inverse_scaling::generate_fused_butterfly_scale_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let scale = 0.5_f64;
    let ptx = generate_fused_butterfly_scale_ptx(2, scale, FftPrecision::Single, sm_version(fx.sm))
        .expect("fbs ptx");

    // radix-2, stride = n_val = 8 -> legs cover complex elements [0,16) once.
    let n_val = 8_u32;
    let stride = 8_u32;
    let n_complex = 16_usize;
    let mut rng = LcgRng::new(0xF5ED);
    let data: Vec<f32> = (0..n_complex * 2)
        .map(|_| rng.next_f32(-3.0, 3.0))
        .collect();
    let expected: Vec<f32> = data.iter().map(|&v| v * scale as f32).collect();

    let kernel = load_kernel(&ptx, "fused_butterfly_scale_r2_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_val, block), block);
    kernel
        .launch(&params, &stream, &(d_data.as_device_ptr(), n_val, stride))
        .expect("launch fbs");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n_complex * 2];
    d_data.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(close(g, e, 1e-6, 1e-6), "fbs[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 8. pruned radix-2 butterfly stage — HOST RE-DERIVATION
// ===========================================================================

#[test]
fn pruned_butterfly_stage_matches_host() {
    use crate::pruned::{PrunedFftConfig, PrunedStage, generate_pruned_butterfly_ptx};

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 256_usize;
    let stage_index = 2_u32; // half_group = 4, group_size = 8 (non-trivial twiddles)
    let cfg = PrunedFftConfig {
        fft_size: n,
        input_nonzero_count: n,
        output_needed_count: n,
        direction: FftDirection::Forward,
        precision: FftPrecision::Single,
        sm_version: sm_version(fx.sm),
    };
    let n_active = (n / 2) as u32; // all 128 butterflies of this stage
    let stage = PrunedStage {
        stage_index,
        radix: 2,
        active_butterflies: u64::from(n_active),
        total_butterflies: u64::from(n_active),
        can_skip_entirely: false,
    };
    let ptx = generate_pruned_butterfly_ptx(&cfg, &stage).expect("pruned ptx");
    let entry = format!("pruned_butterfly_f32_n{n}_s{stage_index}_a{n_active}");

    let mut rng = LcgRng::new(0x9B17);
    let input: Vec<(f32, f32)> = (0..n)
        .map(|_| (rng.next_f32(-1.0, 1.0), rng.next_f32(-1.0, 1.0)))
        .collect();

    // Oracle: replicate the kernel's exact index map + twiddle.
    let half_group = 1_usize << stage_index;
    let group_size = half_group * 2;
    let dir_sign = -1.0_f64; // forward
    let mut expected = vec![(0.0_f32, 0.0_f32); n];
    for tid in 0..n_active as usize {
        let group_idx = tid / half_group;
        let j = tid % half_group;
        let idx_a = group_idx * group_size + j;
        let idx_b = idx_a + half_group;
        let angle = dir_sign * 2.0 * std::f64::consts::PI / group_size as f64 * j as f64;
        let (tw_re, tw_im) = (angle.cos(), angle.sin());
        let (ar, ai) = (f64::from(input[idx_a].0), f64::from(input[idx_a].1));
        let (br, bi) = (f64::from(input[idx_b].0), f64::from(input[idx_b].1));
        let wb_re = tw_re * br - tw_im * bi;
        let wb_im = tw_re * bi + tw_im * br;
        expected[idx_a] = ((ar + wb_re) as f32, (ai + wb_im) as f32);
        expected[idx_b] = ((ar - wb_re) as f32, (ai - wb_im) as f32);
    }

    let kernel = load_kernel(&ptx, &entry);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&interleave(&input)).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 2]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_active, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), n_active),
        )
        .expect("launch pruned");
    stream.synchronize().expect("sync");

    let mut flat = vec![0.0_f32; n * 2];
    d_out.copy_to_host(&mut flat).expect("copy");
    let got = deinterleave(&flat);
    for (i, (&(gr, gi), &(er, ei))) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(gr, er, 2e-3, 2e-3) && close(gi, ei, 2e-3, 2e-3),
            "pruned[{i}]: gpu=({gr},{gi}) cpu=({er},{ei})"
        );
    }
}

// ===========================================================================
// 9. FP16 radix-2 butterfly (fp32-accumulate) — HOST RE-DERIVATION
// ===========================================================================

#[test]
fn fp16_butterfly_matches_host() {
    use crate::half_precision::{AccumulationMode, generate_fp16_butterfly_ptx};

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = generate_fp16_butterfly_ptx(2, AccumulationMode::Fp32, sm_version(fx.sm))
        .expect("fp16b ptx");

    // S independent radix-2 butterflies: data has 2*S complex elements,
    // idx_a in [0,S), idx_b = idx_a + S, twiddle[idx_a].
    let s = 4_u32;
    let n_complex = (2 * s) as usize;
    let mut rng = LcgRng::new(0x16B7);
    let a_f32: Vec<(f32, f32)> = (0..s)
        .map(|_| (rng.next_f32(-1.0, 1.0), rng.next_f32(-1.0, 1.0)))
        .collect();
    let b_f32: Vec<(f32, f32)> = (0..s)
        .map(|_| (rng.next_f32(-1.0, 1.0), rng.next_f32(-1.0, 1.0)))
        .collect();
    let tw_f32: Vec<(f32, f32)> = (0..s)
        .map(|k| {
            let ang = -2.0 * std::f64::consts::PI * f64::from(k) / 8.0;
            (ang.cos() as f32, ang.sin() as f32)
        })
        .collect();

    // Pack data buffer (complex f16): [a (S), b (S)]; twiddle buffer (S).
    let mut data16 = vec![0u16; n_complex * 2];
    for k in 0..s as usize {
        data16[k * 2] = f16_from_f32(a_f32[k].0);
        data16[k * 2 + 1] = f16_from_f32(a_f32[k].1);
        data16[(k + s as usize) * 2] = f16_from_f32(b_f32[k].0);
        data16[(k + s as usize) * 2 + 1] = f16_from_f32(b_f32[k].1);
    }
    let mut tw16 = vec![0u16; s as usize * 2];
    for k in 0..s as usize {
        tw16[k * 2] = f16_from_f32(tw_f32[k].0);
        tw16[k * 2 + 1] = f16_from_f32(tw_f32[k].1);
    }

    // Oracle on the exact uploaded half values (decode -> f32 math).
    let mut expected = vec![0.0_f32; n_complex * 2];
    for k in 0..s as usize {
        let ar = f16_to_f32(data16[k * 2]);
        let ai = f16_to_f32(data16[k * 2 + 1]);
        let br = f16_to_f32(data16[(k + s as usize) * 2]);
        let bi = f16_to_f32(data16[(k + s as usize) * 2 + 1]);
        let wr = f16_to_f32(tw16[k * 2]);
        let wi = f16_to_f32(tw16[k * 2 + 1]);
        let wb_re = wr * br - wi * bi;
        let wb_im = wr * bi + wi * br;
        expected[k * 2] = ar + wb_re;
        expected[k * 2 + 1] = ai + wb_im;
        expected[(k + s as usize) * 2] = ar - wb_re;
        expected[(k + s as usize) * 2 + 1] = ai - wb_im;
    }

    let kernel = load_kernel(&ptx, "fft_fp16_butterfly_r2_fp32acc");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_data = DeviceBuffer::<u16>::from_host(&data16).expect("d_data");
    let d_tw = DeviceBuffer::<u16>::from_host(&tw16).expect("d_tw");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(s, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_data.as_device_ptr(),
                d_tw.as_device_ptr(),
                s,    // stride
                s,    // n (bound on idx_a)
                0u32, // direction (unused by body)
            ),
        )
        .expect("launch fp16 butterfly");
    stream.synchronize().expect("sync");

    let mut out16 = vec![0u16; n_complex * 2];
    d_data.copy_to_host(&mut out16).expect("copy");
    for (i, (&h, &e)) in out16.iter().zip(expected.iter()).enumerate() {
        let g = f16_to_f32(h);
        // One f16 ULP at the result magnitude, plus a small floor.
        let tol = e.abs() / 512.0 + 5e-3;
        assert!((g - e).abs() <= tol, "fp16_bfly[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 10. FP16 twiddle apply — HOST RE-DERIVATION (out = x * W)
// ===========================================================================

#[test]
fn fp16_twiddle_matches_host() {
    use crate::half_precision::generate_fp16_twiddle_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 16_u32;
    let ptx = generate_fp16_twiddle_ptx(n as usize, sm_version(fx.sm)).expect("fp16t ptx");

    let mut rng = LcgRng::new(0x7D17);
    let mut data16 = vec![0u16; n as usize * 2];
    let mut tw16 = vec![0u16; n as usize * 2];
    for k in 0..n as usize {
        data16[k * 2] = f16_from_f32(rng.next_f32(-1.0, 1.0));
        data16[k * 2 + 1] = f16_from_f32(rng.next_f32(-1.0, 1.0));
        let ang = -2.0 * std::f64::consts::PI * k as f64 / f64::from(n);
        tw16[k * 2] = f16_from_f32(ang.cos() as f32);
        tw16[k * 2 + 1] = f16_from_f32(ang.sin() as f32);
    }

    let mut expected = vec![0.0_f32; n as usize * 2];
    for k in 0..n as usize {
        let xr = f16_to_f32(data16[k * 2]);
        let xi = f16_to_f32(data16[k * 2 + 1]);
        let wr = f16_to_f32(tw16[k * 2]);
        let wi = f16_to_f32(tw16[k * 2 + 1]);
        expected[k * 2] = xr * wr - xi * wi;
        expected[k * 2 + 1] = xr * wi + xi * wr;
    }

    let kernel = load_kernel(&ptx, "fft_fp16_twiddle_n16");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_data = DeviceBuffer::<u16>::from_host(&data16).expect("d_data");
    let d_tw = DeviceBuffer::<u16>::from_host(&tw16).expect("d_tw");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), d_tw.as_device_ptr(), n),
        )
        .expect("launch fp16 twiddle");
    stream.synchronize().expect("sync");

    let mut out16 = vec![0u16; n as usize * 2];
    d_data.copy_to_host(&mut out16).expect("copy");
    for (i, (&h, &e)) in out16.iter().zip(expected.iter()).enumerate() {
        let g = f16_to_f32(h);
        let tol = e.abs() / 512.0 + 5e-3;
        assert!((g - e).abs() <= tol, "fp16_tw[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 11. bank-conflict-free Stockham kernel — PASSTHROUGH oracle (HONEST)
//
// The generated `fft_bcf_*` kernel's butterfly-stage loop emits ONLY comments
// and `bar.sync` (the bank-conflict-free butterfly is an unimplemented
// skeleton), so on-device it is an identity copy: cooperative global -> padded
// shared load, no compute, then padded shared -> global store. We validate the
// load/store + bank-padding-index path it actually exercises (out == in) and
// do NOT pretend it computes a DFT.
// ===========================================================================

#[test]
fn bank_conflict_free_passthrough_matches_host() {
    use crate::kernels::bank_conflict_free::BankConflictFreeStockham;
    use crate::plan::FftStrategy;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let bcf = BankConflictFreeStockham::new(n, FftPrecision::Single, FftDirection::Forward);
    let strat = FftStrategy {
        radices: vec![8, 8],
        strides: vec![1, 8],
        single_kernel: true,
    };
    let ptx = bcf
        .generate_kernel(&strat, sm_version(fx.sm))
        .expect("bcf ptx");

    let signal: Vec<(f32, f32)> = (0..n)
        .map(|i| ((i as f32 * 0.11).sin(), (i as f32 * 0.07).cos()))
        .collect();

    let kernel = load_kernel(&ptx, "fft_bcf_f32_n64");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&interleave(&signal)).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * 2]).expect("d_out");

    // compute_block_size(64) = 64; single batch -> one block.
    let params = LaunchParams::new(1_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), 1u32, 0u32),
        )
        .expect("launch bcf");
    stream.synchronize().expect("sync");

    let mut flat = vec![0.0_f32; n * 2];
    d_out.copy_to_host(&mut flat).expect("copy");
    let want = interleave(&signal);
    for (i, (&g, &w)) in flat.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "bcf passthrough[{i}]: gpu={g} cpu={w}"
        );
    }
}

// ===========================================================================
// 12. fused batched kernel — PASSTHROUGH oracle (HONEST)
//
// Like the BCF kernel, `fft_fused_*`'s in-block Stockham stages are an
// unimplemented comment-only stub, so on-device each batch row is copied
// global -> shared -> global unchanged. We validate the per-row shared-memory
// load/store confinement (each row's bytes round-trip identically) and do NOT
// claim a DFT.
// ===========================================================================

#[test]
fn fused_batch_passthrough_matches_host() {
    use crate::kernels::fused_batch::FusedBatchFft;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let fused = FusedBatchFft::new(n, 1024, FftPrecision::Single, FftDirection::Forward);
    let fpb = fused.ffts_per_block();
    let batch = fpb; // a single block processes the whole batch
    let ptx = fused.generate_kernel(sm_version(fx.sm)).expect("fused ptx");
    let entry = format!("fft_fused_f32_n{n}_fpb{fpb}");

    let signals: Vec<Vec<(f32, f32)>> = (0..batch)
        .map(|bi| {
            (0..n)
                .map(|i| (((i + bi) % 7) as f32, ((i * 2 + bi) % 5) as f32 - 2.0))
                .collect()
        })
        .collect();
    let mut host_in: Vec<f32> = Vec::with_capacity(n * batch * 2);
    for s in &signals {
        host_in.extend(interleave(s));
    }

    let kernel = load_kernel(&ptx, &entry);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&host_in).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; host_in.len()]).expect("d_out");

    let params = LaunchParams::new(fused.grid_size(batch), fused.block_size());
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                batch as u32,
                0u32,
            ),
        )
        .expect("launch fused");
    stream.synchronize().expect("sync");

    let mut host_out = vec![0.0_f32; host_in.len()];
    d_out.copy_to_host(&mut host_out).expect("copy");
    for (i, (&g, &w)) in host_out.iter().zip(host_in.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "fused passthrough[{i}]: gpu={g} cpu={w}"
        );
    }
}

// ===========================================================================
// 13. complex tiled transpose — HOST RE-DERIVATION (out[j][i] = in[i][j])
// ===========================================================================

#[test]
fn transpose_matches_host() {
    use crate::kernels::transpose::generate_transpose_kernel;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let rows = 64_u32;
    let cols = 32_u32;
    let ptx = generate_transpose_kernel(rows, cols, FftPrecision::Single, sm_version(fx.sm))
        .expect("transpose ptx");

    let mut rng = LcgRng::new(0x713A);
    let input: Vec<(f32, f32)> = (0..(rows * cols) as usize)
        .map(|_| (rng.next_f32(-5.0, 5.0), rng.next_f32(-5.0, 5.0)))
        .collect();

    // out is cols x rows: out[r*rows + c] = in[c*cols + r].
    let mut expected = vec![(0.0_f32, 0.0_f32); (rows * cols) as usize];
    for r in 0..cols {
        for c in 0..rows {
            expected[(r * rows + c) as usize] = input[(c * cols + r) as usize];
        }
    }

    let kernel = load_kernel(&ptx, "transpose_f32_64x32");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&interleave(&input)).expect("d_in");
    let d_out =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; (rows * cols) as usize * 2]).expect("d_out");

    // TILE_DIM = 32, BLOCK_DIM = (32, 8); grid = (ceil(cols/32), ceil(rows/32)).
    let grid = (cols.div_ceil(32), rows.div_ceil(32));
    let params = LaunchParams::new(grid, (32_u32, 8_u32));
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), rows, cols),
        )
        .expect("launch transpose");
    stream.synchronize().expect("sync");

    let mut flat = vec![0.0_f32; (rows * cols) as usize * 2];
    d_out.copy_to_host(&mut flat).expect("copy");
    let got = deinterleave(&flat);
    for (i, (&(gr, gi), &(er, ei))) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            (gr.to_bits(), gi.to_bits()),
            (er.to_bits(), ei.to_bits()),
            "transpose[{i}]: gpu=({gr},{gi}) cpu=({er},{ei})"
        );
    }
}

// ===========================================================================
// 14. PFA kernel — HOST RE-DERIVATION (documented passthrough copy baseline)
//
// The current `pfa_fft` body is an explicit "passthrough complex copy
// baseline" (the CRT index mapping is not yet implemented), so the honest
// oracle is the identity copy — NOT a DFT. Verified as such.
// ===========================================================================

#[test]
fn pfa_passthrough_matches_host() {
    use crate::radix::pfa::PrimeFactorFft;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 15_usize;
    let batch = 3_u32;
    let pfa = PrimeFactorFft::new(n).expect("pfa");
    let ptx = pfa
        .generate_kernel(
            FftPrecision::Single,
            FftDirection::Forward,
            sm_version(fx.sm),
        )
        .expect("pfa ptx");

    let total = n * batch as usize;
    let mut rng = LcgRng::new(0x0FA1);
    let input: Vec<(f32, f32)> = (0..total)
        .map(|_| (rng.next_f32(-2.0, 2.0), rng.next_f32(-2.0, 2.0)))
        .collect();

    let kernel = load_kernel(&ptx, "pfa_fft");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&interleave(&input)).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total * 2]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), batch),
        )
        .expect("launch pfa");
    stream.synchronize().expect("sync");

    let mut flat = vec![0.0_f32; total * 2];
    d_out.copy_to_host(&mut flat).expect("copy");
    let got = deinterleave(&flat);
    for (i, (&(gr, gi), &(er, ei))) in got.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            (gr.to_bits(), gi.to_bits()),
            (er.to_bits(), ei.to_bits()),
            "pfa passthrough[{i}]"
        );
    }
}

// ===========================================================================
// 15. real-FFT pack — HOST RE-DERIVATION (z[k] = x[2k] + j x[2k+1])
// ===========================================================================

#[test]
fn real_fft_pack_matches_host() {
    use crate::transforms::real_fft::RealFft;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 16_usize; // N real samples -> N/2 complex
    let rf = RealFft::new(n, FftPrecision::Single).expect("real fft");
    let m = rf
        .generate_pack_kernel(sm_version(fx.sm))
        .expect("pack ptx");

    let mut rng = LcgRng::new(0x4EA1);
    let reals: Vec<f32> = (0..n).map(|_| rng.next_f32(-3.0, 3.0)).collect();
    // Packed complex buffer is byte-identical: out[i] = reals[i].
    let expected = reals.clone();

    let kernel = load_kernel(&m.source, &m.entry_name);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_in = DeviceBuffer::<f32>::from_host(&reals).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let half = (n / 2) as u32;
    // compute_block_size(half_n=8) = 32 -> the kernel's `.maxntid` is 32.
    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(half, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), n as u32),
        )
        .expect("launch pack");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "pack[{i}]: gpu={g} cpu={e}");
    }
}

// ===========================================================================
// 16. real-FFT unpack — LOAD-ONLY (documented incomplete skeleton)
//
// `generate_unpack_kernel` computes a per-bin twiddle angle but performs NO
// memory writes (the body is documented as the "PTX skeleton"), so there is no
// numeric output to validate. We confirm it JITs and launches without error
// on-device and is honestly NOT a numeric oracle test.
// ===========================================================================

#[test]
fn real_fft_unpack_load_only() {
    use crate::transforms::real_fft::RealFft;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 16_usize;
    let rf = RealFft::new(n, FftPrecision::Single).expect("real fft");
    let m = rf
        .generate_unpack_kernel(sm_version(fx.sm))
        .expect("unpack ptx");

    let kernel = load_kernel(&m.source, &m.entry_name);
    let stream = Stream::new(&fx.ctx).expect("stream");
    let half = n / 2;
    let d_in = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; (half + 1) * 2]).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; (half + 1) * 2]).expect("d_out");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d((half + 1) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), half as u32),
        )
        .expect("launch unpack");
    stream
        .synchronize()
        .expect("sync (unpack skeleton runs cleanly)");
}

// ===========================================================================
// Non-vacuous device probe
//
// Proves the device path is real (not a vacuous skip): the scale kernel's
// output must MATCH the correct `scale*x` AND clearly DIFFER from a corrupted
// expectation. If the kernel had silently not run, the output would still be
// the initial buffer contents and the "differs from wrong" half would fail.
// ===========================================================================

#[test]
fn nonvacuous_scale_probe() {
    use crate::inverse_scaling::generate_scale_kernel_ptx;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 64_usize;
    let scale = 0.25_f64; // exact in f32
    let ptx = generate_scale_kernel_ptx(n, scale, FftPrecision::Single, sm_version(fx.sm))
        .expect("scale ptx");

    let total_floats = n * 2;
    let mut rng = LcgRng::new(0x9E0B_E001);
    let data: Vec<f32> = (0..total_floats).map(|_| rng.next_f32(1.0, 4.0)).collect();
    let init = data.clone();

    let kernel = load_kernel(&ptx, "scale_fft_n64_f32");
    let stream = Stream::new(&fx.ctx).expect("stream");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total_floats as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_data.as_device_ptr(), total_floats as u32),
        )
        .expect("launch scale");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; total_floats];
    d_data.copy_to_host(&mut got).expect("copy");

    // (a) matches the correct transform...
    for (i, (&g, &x)) in got.iter().zip(init.iter()).enumerate() {
        assert!(
            close(g, x * scale as f32, 1e-6, 1e-6),
            "probe scale[{i}]: gpu={g} want={}",
            x * scale as f32
        );
    }
    // (b) ...and is unambiguously NOT the un-scaled input (the kernel ran).
    let unchanged = got
        .iter()
        .zip(init.iter())
        .all(|(&g, &x)| (g - x).abs() < 1e-6);
    assert!(
        !unchanged,
        "probe vacuous: output equals input — kernel did not execute"
    );
}
