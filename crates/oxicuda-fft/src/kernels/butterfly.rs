//! Shared Stockham auto-sort radix butterfly PTX emitter.
//!
//! This module provides a single, reusable PTX code generator for the
//! Stockham FFT butterfly stages.  It is wired into the three kernel
//! generators ([`stockham`](super::stockham),
//! [`batch_fft`](super::batch_fft) and [`large_fft`](super::large_fft))
//! so they all share one numerically-correct implementation.
//!
//! # The Stockham auto-sort algorithm
//!
//! The Stockham FFT is iterative and out-of-place: each stage reads from
//! one buffer and writes a naturally-sorted result into another, avoiding
//! the digit-reversal permutation that Cooley-Tukey requires.
//!
//! For a transform of size `N` decomposed into radices `r_0, r_1, ...`,
//! stage `s` uses radix `r = r_s` and a sub-transform length
//! `L = r_0 * r_1 * ... * r_{s-1}` (the product of all *previous* radices,
//! `L = 1` for the first stage).  Each stage performs `N / r` radix-`r`
//! butterflies.  Butterfly index `j` in `0 .. N/r` is split as:
//!
//! ```text
//!   p = j / L      (which length-L block, 0 .. N/(r*L))
//!   q = j % L      (position inside the block, 0 .. L)
//! ```
//!
//! The `r` input elements for that butterfly are read from the source
//! buffer at:
//!
//! ```text
//!   in[t] = src[ q + (p * r + t) * L ]      for t in 0 .. r
//! ```
//!
//! Each input leg `t` is multiplied by the twiddle factor
//! `W_N^{t * p * L}` where `W_N^k = exp(-2*pi*i*k/N)` for the forward
//! transform (the sign is flipped for the inverse transform).  After the
//! radix-`r` DFT the `r` results are scattered to the destination buffer
//! at:
//!
//! ```text
//!   out[t] = dst[ q + p * L * r ... ]  ->  dst[ q + p * L + t * (N / r) ]
//! ```
//!
//! Writing the results with the `t * (N/r)` stride is what makes the
//! transform self-sorting: after the final stage the data is already in
//! natural order.
//!
//! # Twiddle factors
//!
//! Twiddle exponents are known at PTX-generation time, so for fixed-`N`
//! single-block kernels they are emitted as immediate `cos`/`sin`
//! constants via `crate::ptx_helpers::load_twiddle_imm`.  For the
//! per-pass large-FFT kernels the exponent depends on a runtime thread
//! index, so the angle is computed at run time and `cos`/`sin` are
//! evaluated with `cos.approx.f32` / `sin.approx.f32` (f32) or a
//! range-reduced polynomial (f64 — PTX has no `sin.approx.f64`).

use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

use crate::ptx_helpers::ComplexRegs;
use crate::radix::radix2::{emit_radix2_butterfly, emit_radix2_butterfly_trivial};
use crate::radix::radix4::{emit_radix4_butterfly, emit_radix4_butterfly_trivial};
use crate::radix::radix8::{emit_radix8_butterfly, emit_radix8_butterfly_trivial};
use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// Direction handling
// ---------------------------------------------------------------------------

/// Returns the twiddle sign for a transform direction: `-1.0` for the
/// forward transform, `+1.0` for the inverse transform.
///
/// `W_N^k = exp(sign * 2*pi*i*k/N)`.
#[must_use]
pub(crate) fn direction_sign(direction: FftDirection) -> f64 {
    direction.sign()
}

// ---------------------------------------------------------------------------
// Radix factorisation
// ---------------------------------------------------------------------------

/// Factorises `n` into a Stockham radix sequence, preferring large
/// radices: `8`, then `4`, then `2`, then the odd radices `3`, `5`, `7`.
///
/// Any prime factor larger than `7` is appended verbatim and handled by
/// the generic direct-DFT butterfly.  The product of the returned
/// radices always equals `n` (for `n >= 1`); `n == 0` yields an empty
/// vector.
#[must_use]
pub(crate) fn factor_radices(mut n: usize) -> Vec<u32> {
    let mut radices = Vec::new();
    if n == 0 {
        return radices;
    }
    for &r in &[8usize, 4, 2, 3, 5, 7] {
        while n % r == 0 && n > 1 {
            radices.push(r as u32);
            n /= r;
        }
    }
    if n > 1 {
        radices.push(n as u32);
    }
    radices
}

// ---------------------------------------------------------------------------
// Stockham stage shape
// ---------------------------------------------------------------------------

/// The shape of a single Stockham radix stage: the full transform size,
/// the stage radix, the sub-transform length `L` (product of previous
/// radices) and the transform direction.
///
/// Bundling these four parameters keeps the stage-emission functions
/// within the workspace argument-count budget.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageShape {
    /// Full transform size `N`.
    pub n: usize,
    /// Radix of this stage.
    pub radix: usize,
    /// Sub-transform length `L` (product of all previous stage radices;
    /// `L = 1` for the first stage).
    pub l: usize,
    /// Transform direction (sets the twiddle sign).
    pub direction: FftDirection,
}

// ---------------------------------------------------------------------------
// Stockham index mapping (compile-time)
// ---------------------------------------------------------------------------

/// The compile-time index data for one radix butterfly of a Stockham stage.
#[derive(Debug, Clone)]
pub(crate) struct ButterflyIndices {
    /// Source-buffer element indices for the `r` input legs.
    pub input: Vec<usize>,
    /// Destination-buffer element indices for the `r` output legs.
    pub output: Vec<usize>,
    /// Twiddle exponent `k` shared by leg `t` (the actual leg twiddle is
    /// `W_N^{t*k}`); equals `i * (N / (r*L))`.
    pub twiddle_k: usize,
}

/// Computes the Stockham auto-sort index mapping for butterfly `b` of a
/// stage with radix `r` and sub-transform length `l` (`L`), for a
/// transform of size `n`.
///
/// `L` is the product of all *previous* stage radices (`L = 1` for the
/// first stage).  With `i = b % L` (position inside the sub-transform)
/// and `j = b / L` (butterfly group):
///
/// ```text
///   in[t]  = j*L + i + t*(N/r)            t in 0..r   (strided gather)
///   out[t] = j*L*r + i + t*L                          (contiguous scatter)
///   twiddle exponent for leg t = t * i * (N / (r*L))
/// ```
///
/// This is the decimation-in-time Stockham recurrence: the gather is
/// strided by `N/r` (the decimation step), the scatter is contiguous in
/// blocks of `r*L`, and the twiddle exponent depends on the in-block
/// position `i` with modulus `N/(r*L)`.  Swapping the gather/scatter
/// strides or using the wrong twiddle exponent silently corrupts the
/// spectrum after the first stage.
#[must_use]
pub(crate) fn stockham_indices(n: usize, r: usize, l: usize, j: usize) -> ButterflyIndices {
    let i = j % l;
    let group = j / l;
    let n_div_r = n / r;

    let mut input = Vec::with_capacity(r);
    let mut output = Vec::with_capacity(r);
    for t in 0..r {
        // Input: strided gather (decimation step N/r).
        input.push(group * l + i + t * n_div_r);
        // Output: contiguous scatter in blocks of r*L.
        output.push(group * l * r + i + t * l);
    }

    ButterflyIndices {
        input,
        output,
        // DIT Stockham twiddle exponent: i * (N / (r*L)).
        twiddle_k: i * (n / (r * l)),
    }
}

// ---------------------------------------------------------------------------
// Shared-memory complex element access
// ---------------------------------------------------------------------------

/// A handle to one of the two ping-pong buffers backing a Stockham FFT.
///
/// `base` is a `U64` register holding the byte address of the buffer's
/// first real component.  `elem_index_offset` is added to every logical
/// complex index — it is used by the batch kernel to confine each batch
/// row to its own slice of shared memory so indexing never crosses a
/// batch-row boundary.
#[derive(Debug, Clone)]
pub(crate) struct PingPongBuffer {
    /// Byte address of the buffer's first real component.
    pub base: Register,
    /// Logical-index bias applied to every complex access (batch-row base).
    pub elem_index_offset: usize,
}

/// Loads a complex value from a ping-pong buffer at logical complex index
/// `complex_index`.
pub(crate) fn load_shared_complex(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    buf: &PingPongBuffer,
    complex_index: usize,
) -> ComplexRegs {
    let elem_bytes = precision.element_bytes();
    let real_index = buf.elem_index_offset + complex_index;
    let re_byte = real_index * 2 * elem_bytes;
    let im_byte = re_byte + elem_bytes;

    match precision {
        FftPrecision::Single => {
            let re_addr = offset_addr(b, &buf.base, re_byte);
            let re = b.load_shared_f32(re_addr);
            let im_addr = offset_addr(b, &buf.base, im_byte);
            let im = b.load_shared_f32(im_addr);
            ComplexRegs { re, im }
        }
        FftPrecision::Double => {
            let re_addr = offset_addr(b, &buf.base, re_byte);
            let re = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("ld.shared.f64 {re}, [{re_addr}];"));
            let im_addr = offset_addr(b, &buf.base, im_byte);
            let im = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("ld.shared.f64 {im}, [{im_addr}];"));
            ComplexRegs { re, im }
        }
    }
}

/// Stores a complex value into a ping-pong buffer at logical complex index
/// `complex_index`.
pub(crate) fn store_shared_complex(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    buf: &PingPongBuffer,
    complex_index: usize,
    value: &ComplexRegs,
) {
    let elem_bytes = precision.element_bytes();
    let real_index = buf.elem_index_offset + complex_index;
    let re_byte = real_index * 2 * elem_bytes;
    let im_byte = re_byte + elem_bytes;

    match precision {
        FftPrecision::Single => {
            let re_addr = offset_addr(b, &buf.base, re_byte);
            b.store_shared_f32(re_addr, value.re.clone());
            let im_addr = offset_addr(b, &buf.base, im_byte);
            b.store_shared_f32(im_addr, value.im.clone());
        }
        FftPrecision::Double => {
            let re_addr = offset_addr(b, &buf.base, re_byte);
            b.raw_ptx(&format!("st.shared.f64 [{re_addr}], {};", value.re));
            let im_addr = offset_addr(b, &buf.base, im_byte);
            b.raw_ptx(&format!("st.shared.f64 [{im_addr}], {};", value.im));
        }
    }
}

/// Emits `dst = base + byte_offset` as a `U64` register.
///
/// When `byte_offset` is zero the base register is returned directly.
fn offset_addr(b: &mut BodyBuilder<'_>, base: &Register, byte_offset: usize) -> Register {
    if byte_offset == 0 {
        return base.clone();
    }
    let off = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {off}, {byte_offset};"));
    b.add_u64(base.clone(), off)
}

// ---------------------------------------------------------------------------
// Radix dispatch
// ---------------------------------------------------------------------------

/// Applies the radix-`r` DFT butterfly to `inputs`, returning the `r`
/// output legs.
///
/// `twiddle_k` is the shared twiddle exponent (`p*L`); when it is zero
/// the trivial (no-twiddle) variant is used.  `n` is the full transform
/// size, used as the twiddle modulus.
fn apply_radix_butterfly(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    radix: usize,
    inputs: &[ComplexRegs],
    twiddle_k: usize,
    n: usize,
    sign: f64,
) -> Vec<ComplexRegs> {
    let forward = sign < 0.0;
    let k = twiddle_k as u32;
    let n_u32 = n as u32;

    match radix {
        2 => {
            let out = if twiddle_k == 0 {
                emit_radix2_butterfly_trivial(b, precision, &inputs[0], &inputs[1])
            } else {
                emit_radix2_butterfly(b, precision, &inputs[0], &inputs[1], k, n_u32, sign)
            };
            vec![out.0, out.1]
        }
        4 => {
            let arr: [ComplexRegs; 4] = [
                inputs[0].clone(),
                inputs[1].clone(),
                inputs[2].clone(),
                inputs[3].clone(),
            ];
            let out = if twiddle_k == 0 {
                emit_radix4_butterfly_trivial(b, precision, &arr, forward)
            } else {
                emit_radix4_butterfly(b, precision, &arr, k, n_u32, sign)
            };
            out.to_vec()
        }
        8 => {
            let arr: [ComplexRegs; 8] = [
                inputs[0].clone(),
                inputs[1].clone(),
                inputs[2].clone(),
                inputs[3].clone(),
                inputs[4].clone(),
                inputs[5].clone(),
                inputs[6].clone(),
                inputs[7].clone(),
            ];
            let out = if twiddle_k == 0 {
                emit_radix8_butterfly_trivial(b, precision, &arr, sign)
            } else {
                emit_radix8_butterfly(b, precision, &arr, k, n_u32, sign)
            };
            out.to_vec()
        }
        _ => {
            // Generic radix (3, 5, 7, ...): direct DFT with per-leg twiddles.
            emit_generic_dft(b, precision, radix, inputs, twiddle_k, n, sign)
        }
    }
}

/// Emits a direct radix-`r` DFT for an arbitrary (non power-of-two) radix.
///
/// `out[u] = sum_t inputs[t] * W_N^{t*twiddle_k} * W_r^{t*u}`.
fn emit_generic_dft(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    radix: usize,
    inputs: &[ComplexRegs],
    twiddle_k: usize,
    n: usize,
    sign: f64,
) -> Vec<ComplexRegs> {
    b.comment(&format!("radix-{radix} direct DFT butterfly"));

    // Pre-apply the outer Stockham twiddle W_N^{t*twiddle_k}.
    let mut twiddled: Vec<ComplexRegs> = Vec::with_capacity(radix);
    for (t, leg) in inputs.iter().enumerate() {
        if t == 0 || twiddle_k == 0 {
            twiddled.push(leg.clone());
        } else {
            let angle = sign * 2.0 * std::f64::consts::PI * (t * twiddle_k) as f64 / n as f64;
            let tw = const_complex(b, precision, angle.cos(), angle.sin());
            twiddled.push(complex_mul_regs(b, precision, leg, &tw));
        }
    }

    // Direct DFT: out[u] = sum_t twiddled[t] * W_r^{t*u}.
    let mut outputs: Vec<ComplexRegs> = Vec::with_capacity(radix);
    for u in 0..radix {
        let mut acc = twiddled[0].clone();
        for (t, leg) in twiddled.iter().enumerate().skip(1) {
            let angle = sign * 2.0 * std::f64::consts::PI * (t * u) as f64 / radix as f64;
            let wr = const_complex(b, precision, angle.cos(), angle.sin());
            let term = complex_mul_regs(b, precision, leg, &wr);
            acc = complex_add_regs(b, precision, &acc, &term);
        }
        outputs.push(acc);
    }
    outputs
}

// ---------------------------------------------------------------------------
// Small complex-arithmetic helpers (precision-polymorphic)
// ---------------------------------------------------------------------------

/// Materialises a complex immediate `(re, im)` into a register pair.
fn const_complex(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    re: f64,
    im: f64,
) -> ComplexRegs {
    match precision {
        FftPrecision::Single => {
            let re_r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.b32 {re_r}, 0F{:08X};", (re as f32).to_bits()));
            let im_r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("mov.b32 {im_r}, 0F{:08X};", (im as f32).to_bits()));
            ComplexRegs { re: re_r, im: im_r }
        }
        FftPrecision::Double => {
            let re_r = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("mov.b64 {re_r}, 0D{:016X};", re.to_bits()));
            let im_r = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("mov.b64 {im_r}, 0D{:016X};", im.to_bits()));
            ComplexRegs { re: re_r, im: im_r }
        }
    }
}

/// Emits a precision-polymorphic complex multiply.
fn complex_mul_regs(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
    bv: &ComplexRegs,
) -> ComplexRegs {
    crate::ptx_helpers::complex_mul(b, precision, a, bv)
}

/// Emits a precision-polymorphic complex add.
fn complex_add_regs(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    a: &ComplexRegs,
    bv: &ComplexRegs,
) -> ComplexRegs {
    crate::ptx_helpers::complex_add(b, precision, a, bv)
}

// ---------------------------------------------------------------------------
// Runtime twiddle computation (large-FFT per-pass kernels)
// ---------------------------------------------------------------------------

/// Computes `(cos(angle), sin(angle))` at run time for a twiddle whose
/// exponent is only known at launch time.
///
/// For f32 this uses the hardware `cos.approx.f32` / `sin.approx.f32`
/// instructions.  For f64 — which PTX has no fast approximation for — it
/// uses the same range-reduced Taylor polynomial as
/// [`crate::pruned`]: reduce the angle to `[-pi, pi]` then evaluate
/// `sin`/`cos` to 5th / 4th order.
///
/// `angle_reg` must already contain the angle in radians (matching
/// `precision`).
pub(crate) fn runtime_cos_sin(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    angle_reg: &Register,
) -> (Register, Register) {
    match precision {
        FftPrecision::Single => {
            let cos_r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cos.approx.f32 {cos_r}, {angle_reg};"));
            let sin_r = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("sin.approx.f32 {sin_r}, {angle_reg};"));
            (cos_r, sin_r)
        }
        FftPrecision::Double => runtime_cos_sin_f64(b, angle_reg),
    }
}

/// f64 range-reduced `cos`/`sin` polynomial.
///
/// Mirrors the pattern in [`crate::pruned`]: `x = angle - round(angle/2pi)
/// * 2pi` puts `x` in `[-pi, pi]`, then
/// `sin(x) ~= x - x^3/6 + x^5/120` and `cos(x) ~= 1 - x^2/2 + x^4/24`.
fn runtime_cos_sin_f64(b: &mut BodyBuilder<'_>, angle_reg: &Register) -> (Register, Register) {
    let two_pi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {two_pi}, 0D{:016X};",
        (2.0 * std::f64::consts::PI).to_bits()
    ));
    let inv_two_pi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {inv_two_pi}, 0D{:016X};",
        (1.0 / (2.0 * std::f64::consts::PI)).to_bits()
    ));
    let one = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mov.b64 {one}, 0D{:016X};", 1.0_f64.to_bits()));
    let neg_half = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {neg_half}, 0D{:016X};",
        (-0.5_f64).to_bits()
    ));
    let one_over_24 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {one_over_24}, 0D{:016X};",
        (1.0_f64 / 24.0_f64).to_bits()
    ));
    let neg_one_over_6 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {neg_one_over_6}, 0D{:016X};",
        (-(1.0_f64 / 6.0_f64)).to_bits()
    ));
    let one_over_120 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "mov.b64 {one_over_120}, 0D{:016X};",
        (1.0_f64 / 120.0_f64).to_bits()
    ));

    // x = angle - round(angle / 2pi) * 2pi
    let scaled = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {scaled}, {angle_reg}, {inv_two_pi};"));
    let k_i64 = b.alloc_reg(PtxType::S64);
    b.raw_ptx(&format!("cvt.rni.s64.f64 {k_i64}, {scaled};"));
    let k_f64 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("cvt.rn.f64.s64 {k_f64}, {k_i64};"));
    let k_two_pi = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {k_two_pi}, {k_f64}, {two_pi};"));
    let x = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("sub.rn.f64 {x}, {angle_reg}, {k_two_pi};"));

    // Powers of x.
    let x2 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {x2}, {x}, {x};"));
    let x3 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {x3}, {x2}, {x};"));
    let x4 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {x4}, {x2}, {x2};"));
    let x5 = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!("mul.rn.f64 {x5}, {x3}, {x2};"));

    // sin(x) = x + (-1/6) x^3 + (1/120) x^5
    let sin_t = b.fma_f64(neg_one_over_6, x3, x.clone());
    let sin_r = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "fma.rn.f64 {sin_r}, {one_over_120}, {x5}, {sin_t};"
    ));

    // cos(x) = 1 + (-1/2) x^2 + (1/24) x^4
    let cos_t = b.fma_f64(neg_half, x2, one);
    let cos_r = b.alloc_reg(PtxType::F64);
    b.raw_ptx(&format!(
        "fma.rn.f64 {cos_r}, {one_over_24}, {x4}, {cos_t};"
    ));

    (cos_r, sin_r)
}

/// Multiplies a complex value by a twiddle whose `(cos, sin)` components
/// are supplied as registers — the runtime-twiddle counterpart of
/// [`crate::ptx_helpers::complex_mul`].
pub(crate) fn complex_mul_runtime_twiddle(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    value: &ComplexRegs,
    tw_cos: &Register,
    tw_sin: &Register,
) -> ComplexRegs {
    let tw = ComplexRegs {
        re: tw_cos.clone(),
        im: tw_sin.clone(),
    };
    crate::ptx_helpers::complex_mul(b, precision, value, &tw)
}

// ---------------------------------------------------------------------------
// Single-block Stockham stage emission
// ---------------------------------------------------------------------------

/// Emits one complete Stockham stage that operates entirely on shared
/// memory, reading from `src` and writing to `dst`.
///
/// `shape` describes the transform size, stage radix, sub-transform
/// length `L` and direction.  Every one of the `N / radix` butterflies
/// is unrolled at code-generation time, so the caller is responsible for
/// the surrounding `bar_sync` barriers.
///
/// The function does **not** synchronise — the caller must `bar_sync`
/// before and after so the ping-pong buffers are consistent.
pub(crate) fn emit_stockham_stage_shared(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    shape: StageShape,
    src: &PingPongBuffer,
    dst: &PingPongBuffer,
) {
    let StageShape {
        n,
        radix,
        l,
        direction,
    } = shape;
    let sign = direction_sign(direction);
    let butterflies = n / radix;
    b.comment(&format!(
        "Stockham stage: N={n}, radix={radix}, L={l}, {butterflies} butterflies"
    ));

    for j in 0..butterflies {
        let idx = stockham_indices(n, radix, l, j);

        // Load the radix-r input legs from the source buffer.
        let mut inputs: Vec<ComplexRegs> = Vec::with_capacity(radix);
        for &in_idx in &idx.input {
            inputs.push(load_shared_complex(b, precision, src, in_idx));
        }

        // Apply the radix-r DFT (with the shared Stockham twiddle).
        let outputs = apply_radix_butterfly(b, precision, radix, &inputs, idx.twiddle_k, n, sign);

        // Scatter the results into the destination buffer.
        for (out_idx, value) in idx.output.iter().zip(outputs.iter()) {
            store_shared_complex(b, precision, dst, *out_idx, value);
        }
    }
}

/// Emits **all** Stockham stages for an in-shared-memory transform of size
/// `n` decomposed into `radices`.
///
/// Every butterfly of every stage is unrolled, so the whole transform is
/// emitted as straight-line code meant to be executed by a **single
/// thread** — the caller must run it under a `tid == 0` guard.  Because
/// one thread runs every stage sequentially, program order alone makes
/// each stage observe the previous stage's shared-memory writes, so **no
/// inter-stage `bar_sync` is emitted**.  Emitting one here would in fact
/// deadlock the kernel: under the `tid == 0` guard only thread 0 would
/// reach it while the rest of the block waits at the kernel's
/// post-butterfly barrier — a barrier-count mismatch.  The caller is
/// responsible for the two barriers that bracket the whole single-thread
/// region (after the cooperative load, before the cooperative store).
///
/// `buffer_a` holds the input on entry.  The function returns the buffer
/// that holds the final result (`buffer_a` when the stage count is even,
/// `buffer_b` when it is odd) so the caller knows which buffer to copy
/// back to global memory.
pub(crate) fn emit_stockham_all_stages<'b>(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    n: usize,
    radices: &[u32],
    buffer_a: &'b PingPongBuffer,
    buffer_b: &'b PingPongBuffer,
    direction: FftDirection,
) -> &'b PingPongBuffer {
    let mut src = buffer_a;
    let mut dst = buffer_b;
    let mut l: usize = 1;

    for (stage_idx, &radix) in radices.iter().enumerate() {
        b.comment(&format!(
            "--- Stockham stage {stage_idx}/{} (radix-{radix}) ---",
            radices.len()
        ));
        let shape = StageShape {
            n,
            radix: radix as usize,
            l,
            direction,
        };
        emit_stockham_stage_shared(b, precision, shape, src, dst);

        l *= radix as usize;
        std::mem::swap(&mut src, &mut dst);
    }

    // After the final swap, `src` points at the buffer holding the result.
    src
}

// ---------------------------------------------------------------------------
// Global-memory Stockham pass emission (large multi-pass FFT)
// ---------------------------------------------------------------------------

/// Applies a radix-`r` DFT to legs that have **already** been multiplied
/// by their outer twiddles — i.e. the trivial (no-twiddle) butterfly.
fn apply_radix_butterfly_trivial(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    radix: usize,
    inputs: &[ComplexRegs],
    sign: f64,
) -> Vec<ComplexRegs> {
    apply_radix_butterfly(b, precision, radix, inputs, 0, 1, sign)
}

/// Loads a complex value from a global-memory complex array.
///
/// `array_base` is the `U64` byte address of element 0; `complex_index`
/// is a `U32` register giving the element index.  The real part is at
/// `base + index*2*elem_bytes`, the imaginary part one element further.
fn load_global_complex_indexed(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    array_base: &Register,
    complex_index: &Register,
) -> ComplexRegs {
    let elem_bytes = precision.element_bytes();
    let complex_bytes = (elem_bytes * 2) as u32;
    let stride_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {stride_reg}, {complex_bytes};"));
    let byte_off = b.mul_wide_u32_to_u64(complex_index.clone(), stride_reg);
    let re_addr = b.add_u64(array_base.clone(), byte_off);

    match precision {
        FftPrecision::Single => {
            let re = b.load_global_f32(re_addr.clone());
            let im_addr = bump_addr(b, &re_addr, elem_bytes);
            let im = b.load_global_f32(im_addr);
            ComplexRegs { re, im }
        }
        FftPrecision::Double => {
            let re = b.load_global_f64(re_addr.clone());
            let im_addr = bump_addr(b, &re_addr, elem_bytes);
            let im = b.load_global_f64(im_addr);
            ComplexRegs { re, im }
        }
    }
}

/// Stores a complex value into a global-memory complex array at the
/// `U32`-register element index `complex_index`.
fn store_global_complex_indexed(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    array_base: &Register,
    complex_index: &Register,
    value: &ComplexRegs,
) {
    let elem_bytes = precision.element_bytes();
    let complex_bytes = (elem_bytes * 2) as u32;
    let stride_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {stride_reg}, {complex_bytes};"));
    let byte_off = b.mul_wide_u32_to_u64(complex_index.clone(), stride_reg);
    let re_addr = b.add_u64(array_base.clone(), byte_off);

    match precision {
        FftPrecision::Single => {
            b.store_global_f32(re_addr.clone(), value.re.clone());
            let im_addr = bump_addr(b, &re_addr, elem_bytes);
            b.store_global_f32(im_addr, value.im.clone());
        }
        FftPrecision::Double => {
            b.store_global_f64(re_addr.clone(), value.re.clone());
            let im_addr = bump_addr(b, &re_addr, elem_bytes);
            b.store_global_f64(im_addr, value.im.clone());
        }
    }
}

/// Emits `dst = addr + delta_bytes` as a fresh `U64` register.
fn bump_addr(b: &mut BodyBuilder<'_>, addr: &Register, delta_bytes: usize) -> Register {
    let delta = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("mov.u64 {delta}, {delta_bytes};"));
    b.add_u64(addr.clone(), delta)
}

/// Emits one complete radix-`r` Stockham butterfly for the global-memory
/// multi-pass FFT, for the butterfly whose index is held in `j_reg`.
///
/// This is the runtime-indexed analogue of [`emit_stockham_stage_shared`]:
/// the butterfly index, the butterfly group, the in-block position `i`
/// and hence the twiddle exponent are all computed at run time.
///
/// * `shape` — the Stockham stage shape (`N`, radix, sub-transform
///   length `L`, direction).  The first pass uses `L = 1`.
/// * `j_reg` — `U32` register holding the butterfly index `b` in
///   `0 .. N/radix`.  The caller is responsible for the surrounding
///   bounds check.
/// * `input_ptr` / `output_ptr` — `U64` base addresses of the source and
///   destination complex arrays.
pub(crate) fn emit_stockham_pass_global(
    b: &mut BodyBuilder<'_>,
    precision: FftPrecision,
    shape: StageShape,
    j_reg: &Register,
    input_ptr: &Register,
    output_ptr: &Register,
) {
    let StageShape {
        n,
        radix,
        l,
        direction,
    } = shape;
    let sign = direction_sign(direction);
    let n_div_r = n / radix;
    b.comment(&format!(
        "Stockham global pass butterfly: N={n}, radix={radix}, L={l}"
    ));

    // i = b % L (in-block position), group = b / L (L is compile-time).
    let l_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {l_reg}, {l};"));
    let group_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {group_reg}, {j_reg}, {l_reg};"));
    let i_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {i_reg}, {j_reg}, {l_reg};"));

    // base_in = group*L + i   (the t=0 input index; legs add t*(N/r)).
    let base_in = b.mad_lo_u32(group_reg.clone(), l_reg.clone(), i_reg.clone());

    // base_out = group*L*r + i (the t=0 output index; legs add t*L).
    let l_r = (l * radix) as u32;
    let l_r_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {l_r_reg}, {l_r};"));
    let base_out = b.mad_lo_u32(group_reg, l_r_reg, i_reg.clone());

    // DIT Stockham twiddle base exponent k = i * (N / (r*L)); the
    // leg-t twiddle is W_N^{t*k}.  N/(r*L) is a compile-time constant.
    let tw_modulus = (n / (radix * l)) as u32;
    let tw_modulus_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {tw_modulus_reg}, {tw_modulus};"));
    let tw_k = b.mul_lo_u32(i_reg, tw_modulus_reg);
    // Convert k to float and pre-scale by sign*2pi/N once.
    let tw_k_f = match precision {
        FftPrecision::Single => {
            let f = b.alloc_reg(PtxType::F32);
            b.raw_ptx(&format!("cvt.rn.f32.u32 {f}, {tw_k};"));
            f
        }
        FftPrecision::Double => {
            let f = b.alloc_reg(PtxType::F64);
            b.raw_ptx(&format!("cvt.rn.f64.u32 {f}, {tw_k};"));
            f
        }
    };
    let two_pi_over_n = sign * 2.0 * std::f64::consts::PI / n as f64;

    // Load and twiddle the r input legs (strided gather: stride N/r).
    let mut legs: Vec<ComplexRegs> = Vec::with_capacity(radix);
    for t in 0..radix {
        // input index = base_in + t*(N/r)
        let in_idx = if t == 0 {
            base_in.clone()
        } else {
            let t_ndr = (t * n_div_r) as u32;
            let t_ndr_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {t_ndr_reg}, {t_ndr};"));
            b.add_u32(base_in.clone(), t_ndr_reg)
        };
        let raw = load_global_complex_indexed(b, precision, input_ptr, &in_idx);

        if t == 0 {
            // W_N^0 = 1 — no twiddle for leg 0.
            legs.push(raw);
        } else {
            // angle = (t * k) * (sign*2pi/N) = tw_k_f * (t * sign*2pi/N)
            let per_leg = (t as f64) * two_pi_over_n;
            let angle = match precision {
                FftPrecision::Single => {
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "mov.b32 {scale}, 0F{:08X};",
                        (per_leg as f32).to_bits()
                    ));
                    let a = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {a}, {tw_k_f}, {scale};"));
                    a
                }
                FftPrecision::Double => {
                    let scale = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mov.b64 {scale}, 0D{:016X};", per_leg.to_bits()));
                    let a = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mul.rn.f64 {a}, {tw_k_f}, {scale};"));
                    a
                }
            };
            let (cos_r, sin_r) = runtime_cos_sin(b, precision, &angle);
            let twiddled = complex_mul_runtime_twiddle(b, precision, &raw, &cos_r, &sin_r);
            legs.push(twiddled);
        }
    }

    // radix-r DFT on the twiddled legs.
    let outputs = apply_radix_butterfly_trivial(b, precision, radix, &legs, sign);

    // Scatter the r outputs to the destination array (contiguous: stride L).
    let l_u32 = l as u32;
    for (t, value) in outputs.iter().enumerate() {
        let out_idx = if t == 0 {
            base_out.clone()
        } else {
            let t_l = (t as u32) * l_u32;
            let t_l_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {t_l_reg}, {t_l};"));
            b.add_u32(base_out.clone(), t_l_reg)
        };
        store_global_complex_indexed(b, precision, output_ptr, &out_idx, value);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference CPU complex multiply.
    fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[test]
    fn direction_sign_values() {
        assert_eq!(direction_sign(FftDirection::Forward), -1.0);
        assert_eq!(direction_sign(FftDirection::Inverse), 1.0);
    }

    /// For the first stage (L=1) every butterfly has `i = 0`: the gather
    /// is strided by `N/r` (`in = [b, b+N/r]`) and the scatter contiguous
    /// (`out = [2b, 2b+1]`); the twiddle exponent is trivial.
    #[test]
    fn stockham_indices_first_stage_radix2() {
        let n = 8;
        let r = 2;
        let l = 1;
        // butterfly 0: i=0,group=0 -> inputs [0,4], outputs [0,1]
        let idx0 = stockham_indices(n, r, l, 0);
        assert_eq!(idx0.input, vec![0, 4]);
        assert_eq!(idx0.output, vec![0, 1]);
        assert_eq!(idx0.twiddle_k, 0);
        // butterfly 3: i=0,group=3 -> inputs [3,7], outputs [6,7]
        let idx3 = stockham_indices(n, r, l, 3);
        assert_eq!(idx3.input, vec![3, 7]);
        assert_eq!(idx3.output, vec![6, 7]);
        assert_eq!(idx3.twiddle_k, 0);
    }

    /// In a middle stage the twiddle exponent is `i * (N/(r*L))`.
    #[test]
    fn stockham_indices_middle_stage_twiddle() {
        // N=8, radix-2, second stage: L=2, N/(r*L)=2.
        let n = 8;
        let r = 2;
        let l = 2;
        // b=0: i=0 -> twiddle 0
        assert_eq!(stockham_indices(n, r, l, 0).twiddle_k, 0);
        // b=1: i=1 -> twiddle 1*2 = 2
        assert_eq!(stockham_indices(n, r, l, 1).twiddle_k, 2);
        // b=3: i=1 -> twiddle 1*2 = 2
        assert_eq!(stockham_indices(n, r, l, 3).twiddle_k, 2);
    }

    /// For the last stage (`L = N/r`) of a decimation-in-time Stockham
    /// FFT the twiddle modulus is `N/(r*L) = 1`, so the twiddle exponent
    /// equals `i = b`; the gather is `[b, b+N/r]` and the scatter lands
    /// in natural-order pairs `[b, b+N/r]`.
    #[test]
    fn stockham_indices_last_stage_radix2() {
        let n = 8;
        let r = 2;
        let l = 4; // last radix-2 stage of an 8-point transform
        for b in 0..n / r {
            let idx = stockham_indices(n, r, l, b);
            // twiddle modulus N/(r*L) == 1, so twiddle_k == i == b.
            assert_eq!(idx.twiddle_k, b, "last stage twiddle exponent is i");
            // gather legs are b and b + N/r
            assert_eq!(idx.input, vec![b, b + n / r]);
            // scatter lands at b and b + N/r (natural-order pair)
            assert_eq!(idx.output, vec![b, b + n / r]);
        }
    }

    /// The union of all output indices across all butterflies of a stage
    /// must be a permutation of `0..N` — no element is written twice and
    /// none is skipped (this is what guarantees no ping-pong aliasing).
    #[test]
    fn stockham_output_indices_are_a_permutation() {
        for &(n, r) in &[(8usize, 2usize), (16, 4), (64, 8), (256, 4), (512, 8)] {
            let mut l = 1usize;
            while l < n {
                let mut seen = vec![false; n];
                for j in 0..n / r {
                    let idx = stockham_indices(n, r, l, j);
                    for &o in &idx.output {
                        assert!(o < n, "output index {o} out of range for N={n}");
                        assert!(
                            !seen[o],
                            "output index {o} written twice (N={n}, r={r}, L={l})"
                        );
                        seen[o] = true;
                    }
                }
                assert!(
                    seen.iter().all(|&s| s),
                    "stage did not cover all N={n} elements (r={r}, L={l})"
                );
                l *= r;
            }
        }
    }

    /// Likewise the input indices of a stage must be a permutation of
    /// `0..N` — every source element is consumed exactly once.
    #[test]
    fn stockham_input_indices_are_a_permutation() {
        for &(n, r) in &[(8usize, 2usize), (16, 4), (64, 8)] {
            let mut l = 1usize;
            while l < n {
                let mut seen = vec![false; n];
                for j in 0..n / r {
                    let idx = stockham_indices(n, r, l, j);
                    for &i in &idx.input {
                        assert!(!seen[i], "input index {i} read twice");
                        seen[i] = true;
                    }
                }
                assert!(seen.iter().all(|&s| s), "stage skipped a source element");
                l *= r;
            }
        }
    }

    /// End-to-end CPU model of the Stockham auto-sort algorithm using the
    /// exact index mapping and twiddles of [`stockham_indices`].  The
    /// result must match a naive DFT — this validates the index mapping
    /// and twiddle exponents that the PTX emitter relies on.
    fn stockham_cpu(input: &[(f64, f64)], radices: &[usize], sign: f64) -> Vec<(f64, f64)> {
        let n = input.len();
        let mut buf_a = input.to_vec();
        let mut buf_b = vec![(0.0, 0.0); n];
        let mut l = 1usize;

        for &r in radices {
            for j in 0..n / r {
                let idx = stockham_indices(n, r, l, j);
                // Load + apply outer Stockham twiddle.
                let mut legs: Vec<(f64, f64)> = Vec::with_capacity(r);
                for (t, &in_idx) in idx.input.iter().enumerate() {
                    let v = buf_a[in_idx];
                    if t == 0 || idx.twiddle_k == 0 {
                        legs.push(v);
                    } else {
                        let ang = sign * 2.0 * std::f64::consts::PI * (t * idx.twiddle_k) as f64
                            / n as f64;
                        legs.push(cmul(v, (ang.cos(), ang.sin())));
                    }
                }
                // radix-r DFT.
                for (u, &out_idx) in idx.output.iter().enumerate() {
                    let mut acc = (0.0, 0.0);
                    for (t, &leg) in legs.iter().enumerate() {
                        let ang = sign * 2.0 * std::f64::consts::PI * (t * u) as f64 / r as f64;
                        let term = cmul(leg, (ang.cos(), ang.sin()));
                        acc = (acc.0 + term.0, acc.1 + term.1);
                    }
                    buf_b[out_idx] = acc;
                }
            }
            l *= r;
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
        buf_a
    }

    /// Naive O(N^2) DFT reference.
    fn naive_dft(input: &[(f64, f64)], sign: f64) -> Vec<(f64, f64)> {
        let n = input.len();
        let mut out = vec![(0.0, 0.0); n];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = (0.0, 0.0);
            for (m, &x) in input.iter().enumerate() {
                let ang = sign * 2.0 * std::f64::consts::PI * (k * m) as f64 / n as f64;
                let term = cmul(x, (ang.cos(), ang.sin()));
                acc = (acc.0 + term.0, acc.1 + term.1);
            }
            *slot = acc;
        }
        out
    }

    #[test]
    fn stockham_cpu_matches_dft_radix2() {
        let input: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 0.5 * i as f64 - 1.0)).collect();
        let got = stockham_cpu(&input, &[2, 2, 2], -1.0);
        let want = naive_dft(&input, -1.0);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g.0 - w.0).abs() < 1e-9, "re mismatch at {i}: {g:?} {w:?}");
            assert!((g.1 - w.1).abs() < 1e-9, "im mismatch at {i}: {g:?} {w:?}");
        }
    }

    #[test]
    fn stockham_cpu_matches_dft_radix4() {
        let input: Vec<(f64, f64)> = (0..16).map(|i| ((i % 5) as f64, (i % 3) as f64)).collect();
        let got = stockham_cpu(&input, &[4, 4], -1.0);
        let want = naive_dft(&input, -1.0);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g.0 - w.0).abs() < 1e-9, "re mismatch at {i}");
            assert!((g.1 - w.1).abs() < 1e-9, "im mismatch at {i}");
        }
    }

    #[test]
    fn stockham_cpu_matches_dft_radix8() {
        let input: Vec<(f64, f64)> = (0..64)
            .map(|i| ((i as f64).sin(), (i as f64).cos()))
            .collect();
        let got = stockham_cpu(&input, &[8, 8], -1.0);
        let want = naive_dft(&input, -1.0);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g.0 - w.0).abs() < 1e-8, "re mismatch at {i}");
            assert!((g.1 - w.1).abs() < 1e-8, "im mismatch at {i}");
        }
    }

    #[test]
    fn stockham_cpu_matches_dft_mixed_radix() {
        // N = 24 = 8 * 3 exercises the generic radix-3 path.
        let input: Vec<(f64, f64)> = (0..24).map(|i| (i as f64, -(i as f64))).collect();
        let got = stockham_cpu(&input, &[8, 3], -1.0);
        let want = naive_dft(&input, -1.0);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g.0 - w.0).abs() < 1e-8, "re mismatch at {i}");
            assert!((g.1 - w.1).abs() < 1e-8, "im mismatch at {i}");
        }
    }

    /// Forward followed by inverse (with 1/N scaling) must round-trip.
    #[test]
    fn stockham_cpu_forward_inverse_roundtrip() {
        let n = 32usize;
        let input: Vec<(f64, f64)> = (0..n)
            .map(|i| ((i * 7 % 11) as f64, (i * 3 % 5) as f64))
            .collect();
        let spectrum = stockham_cpu(&input, &[8, 4], -1.0);
        let back = stockham_cpu(&spectrum, &[8, 4], 1.0);
        for (i, (b, x)) in back.iter().zip(input.iter()).enumerate() {
            let re = b.0 / n as f64;
            let im = b.1 / n as f64;
            assert!((re - x.0).abs() < 1e-9, "roundtrip re mismatch at {i}");
            assert!((im - x.1).abs() < 1e-9, "roundtrip im mismatch at {i}");
        }
    }
}
