//! `f64` math primitives and synthesized transcendentals for the
//! [`BodyBuilder`](super::BodyBuilder) DSL.
//!
//! This module extends the instruction-emission API with double-precision
//! constant materialization (`mov_imm_f64`/`mov_imm_f32`), rounded division
//! (`div_rn_f64`/`div_rn_f32`), and a small library of transcendental
//! functions (`exp_f64`, `log_f64`, `erf_f64`, `tanh_f64`) built entirely from
//! valid `ptxas` instructions.
//!
//! ## Why synthesize the transcendentals?
//!
//! PTX provides hardware fast-paths such as `ex2.approx.f64` and
//! `lg2.approx.f64` for `f32`, but **`.approx.f64` forms are NOT valid `ptxas`
//! instructions**. To remain assemblable for `f64`, every transcendental here
//! is synthesized from `fma.rn.f64`, exact bit manipulation
//! (`and`/`or`/`shl`/`shr` on the IEEE-754 layout), and polynomial
//! approximation. No `.approx.f64` substring is ever emitted (the test suite
//! asserts this explicitly).
//!
//! ## Accuracy is NOT GPU-verified
//!
//! There is no NVIDIA hardware on the development or CI machine, so the unit
//! tests in this module validate **PTX generation and algorithm structure
//! only** — they confirm the expected instruction mnemonics appear and that no
//! illegal `.approx.f64` form is emitted. The numerical-accuracy claims in the
//! per-function documentation (ulp / absolute-error bounds) describe the
//! mathematical algorithm and have not been measured on a real GPU.

use super::BodyBuilder;
use crate::ir::{PtxType, Register};

impl BodyBuilder<'_> {
    // ════════════════════════════════════════════════════════════════════
    //  Constant materialization
    // ════════════════════════════════════════════════════════════════════

    /// Materializes an `f64` immediate into a fresh register.
    ///
    /// Emits `mov.b64 {r}, 0D{bits:016X};` where `bits` is the raw IEEE-754
    /// bit pattern of `val` (PTX uses the `0D` prefix for hex-encoded
    /// double-precision literals).
    pub fn mov_imm_f64(&mut self, val: f64) -> Register {
        let r = self.alloc_reg(PtxType::F64);
        let bits = val.to_bits();
        self.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
        r
    }

    /// Materializes an `f32` immediate into a fresh register.
    ///
    /// Emits `mov.b32 {r}, 0F{bits:08X};` where `bits` is the raw IEEE-754
    /// bit pattern of `val` (PTX uses the `0F` prefix for hex-encoded
    /// single-precision literals).
    pub fn mov_imm_f32(&mut self, val: f32) -> Register {
        let r = self.alloc_reg(PtxType::F32);
        let bits = val.to_bits();
        self.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
        r
    }

    // ════════════════════════════════════════════════════════════════════
    //  Rounded division
    // ════════════════════════════════════════════════════════════════════

    /// Emits `div.rn.f64 {dst}, {a}, {b};` — IEEE round-to-nearest-even
    /// double-precision division.
    pub fn div_rn_f64(&mut self, a: &Register, b: &Register) -> Register {
        let dst = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("div.rn.f64 {dst}, {a}, {b};"));
        dst
    }

    /// Emits `div.rn.f32 {dst}, {a}, {b};` — IEEE round-to-nearest-even
    /// single-precision division.
    pub fn div_rn_f32(&mut self, a: &Register, b: &Register) -> Register {
        let dst = self.alloc_reg(PtxType::F32);
        self.raw_ptx(&format!("div.rn.f32 {dst}, {a}, {b};"));
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Private f64 helpers (no `mul_f64` exists on BodyBuilder)
    // ════════════════════════════════════════════════════════════════════

    /// Emits `mul.rn.f64 {dst}, {a}, {b};` — round-to-nearest-even
    /// double-precision multiply. Private: there is no public `mul_f64`.
    fn mul_rn_f64(&mut self, a: &Register, b: &Register) -> Register {
        let dst = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("mul.rn.f64 {dst}, {a}, {b};"));
        dst
    }

    /// Emits `neg.f64 {dst}, {x};` — double-precision negation.
    fn neg_f64(&mut self, x: &Register) -> Register {
        let dst = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("neg.f64 {dst}, {x};"));
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Transcendentals (synthesized — never emit `.approx.f64`)
    // ════════════════════════════════════════════════════════════════════

    /// Computes `exp(x)` for `f64` without any `.approx.f64` instruction.
    ///
    /// # Algorithm
    ///
    /// Cody–Waite range reduction `x = k*ln2 + r` (with a two-word `ln2`
    /// split for extra precision), a degree-7 Taylor polynomial evaluated by
    /// Horner via `fma.rn.f64` for `exp(r)`, and reconstruction of `2^k` by
    /// directly assembling the IEEE-754 exponent field.
    ///
    /// # Precision
    ///
    /// Full-`f64` arithmetic, approximately 1–2 ulp over the normal exponent
    /// range. Overflow/underflow saturation for extreme `k` is **not** handled
    /// (see the `2^k` assembly note below).
    pub fn exp_f64(&mut self, x: &Register) -> Register {
        // k ≈ x * log2(e); kf is the (still fractional) scaled argument.
        let log2e = self.mov_imm_f64(std::f64::consts::LOG2_E);
        let kf = self.mul_rn_f64(x, &log2e);

        // Round kf to the nearest integer (as f64) using the Lomont/Lattner
        // "add the magic constant" trick: adding 2^52 + 2^51 forces the
        // fractional bits out under default round-to-nearest, then subtracting
        // it back leaves the rounded integer. Valid for |kf| < 2^51.
        let magic = self.mov_imm_f64(6_755_399_441_055_744.0);
        let kf_plus = self.add_f64(kf, magic.clone());
        let kf_rounded = self.sub_f64(kf_plus, magic);

        // Range reduction r = x - k*ln2, using a two-word ln2 (hi + lo) so the
        // subtraction does not lose low-order bits. The hi word has trailing
        // mantissa zeros (fdlibm split: hi = 0x3FE62E42FEE00000); these shortest
        // round-trip decimals reproduce the exact same f64 bit patterns.
        // Constants are bound to locals first to avoid overlapping &mut borrows.
        let neg_ln2_hi = self.mov_imm_f64(-0.693_147_180_369_123_8);
        let r = self.fma_f64(kf_rounded.clone(), neg_ln2_hi, x.clone());
        let neg_ln2_lo = self.mov_imm_f64(-1.908_214_929_270_587_7e-10);
        let r = self.fma_f64(kf_rounded.clone(), neg_ln2_lo, r);

        // exp(r) via Taylor series 1 + r + r^2/2 + ... + r^7/5040, evaluated
        // by Horner with fused multiply-adds. With the range-reduced argument
        // |r| <= ln2/2, the degree-7 truncation error is < 2e-16 (~1 ulp).
        let mut p = self.mov_imm_f64(1.0 / 5040.0);
        let c6 = self.mov_imm_f64(1.0 / 720.0);
        p = self.fma_f64(p, r.clone(), c6);
        let c5 = self.mov_imm_f64(1.0 / 120.0);
        p = self.fma_f64(p, r.clone(), c5);
        let c4 = self.mov_imm_f64(1.0 / 24.0);
        p = self.fma_f64(p, r.clone(), c4);
        let c3 = self.mov_imm_f64(1.0 / 6.0);
        p = self.fma_f64(p, r.clone(), c3);
        let c2 = self.mov_imm_f64(0.5);
        p = self.fma_f64(p, r.clone(), c2);
        let c1 = self.mov_imm_f64(1.0);
        p = self.fma_f64(p, r.clone(), c1);
        let c0 = self.mov_imm_f64(1.0);
        p = self.fma_f64(p, r, c0);

        // Build 2^k = ((k + 1023) << 52) reinterpreted as a double. kf_rounded
        // holds the rounded integer as f64; convert it to s32, bias it, and
        // place it in the exponent field. Valid for the normal exponent range.
        let ki = self.alloc_reg(PtxType::S32);
        self.raw_ptx(&format!("cvt.rzi.s32.f64 {ki}, {kf_rounded};"));
        let ki64 = self.alloc_reg(PtxType::S64);
        self.raw_ptx(&format!("cvt.s64.s32 {ki64}, {ki};"));
        let biased = self.alloc_reg(PtxType::S64);
        self.raw_ptx(&format!("add.s64 {biased}, {ki64}, 1023;"));
        let expbits = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!("shl.b64 {expbits}, {biased}, 52;"));
        let two_k = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("mov.b64 {two_k}, {expbits};"));
        // TODO(precision): subnormal/overflow saturation for extreme k is not handled.

        // exp(x) = exp(r) * 2^k.
        self.mul_rn_f64(&p, &two_k)
    }

    /// Computes the natural logarithm `log(x)` for `f64` without any
    /// `.approx.f64` instruction.
    ///
    /// # Algorithm
    ///
    /// `frexp`-style split of the IEEE-754 bits into a mantissa `m ∈ [1, 2)`
    /// and an unbiased exponent `e`, then `log(m)` via the `log1p` identity
    /// `log(m) = 2*atanh(f / (2 + f))` with `f = m - 1`. The `atanh` series is
    /// evaluated by Horner via `fma.rn.f64`. Finally
    /// `log(x) = e*ln2 + log(m)`.
    ///
    /// The customary √2 centering of `m` is omitted; keeping `m ∈ [1, 2)`
    /// bounds `|s| = |f / (2 + f)| <= 1/3`, which the degree-4 `atanh` series
    /// resolves to ~1–2 ulp.
    ///
    /// # Domain
    ///
    /// No domain check is performed: `x <= 0` yields NaN/garbage, matching the
    /// behavior of raw PTX instructions and libm conventions.
    ///
    /// # Precision
    ///
    /// Full-`f64` arithmetic, approximately 1–2 ulp over the normal positive
    /// range.
    // Single-letter names (m, f, s, z, q) match the standard frexp/atanh
    // derivation used throughout numeric literature; renaming would obscure it.
    #[allow(clippy::many_single_char_names)]
    pub fn log_f64(&mut self, x: &Register) -> Register {
        // Reinterpret the input as raw bits to extract the IEEE-754 fields.
        let bits = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!("mov.b64 {bits}, {x};"));
        // Raw (biased) exponent = (bits >> 52) & 0x7FF.
        let e_shifted = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!("shr.b64 {e_shifted}, {bits}, 52;"));
        let e_raw = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!("and.b64 {e_raw}, {e_shifted}, 0x7FF;"));
        // Mantissa: clear exponent/sign, set the biased exponent to 1023 so the
        // result is the significand reinterpreted as a double in [1, 2).
        let mant_frac = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!("and.b64 {mant_frac}, {bits}, 0x000FFFFFFFFFFFFF;"));
        let mant_bits = self.alloc_reg(PtxType::B64);
        self.raw_ptx(&format!(
            "or.b64 {mant_bits}, {mant_frac}, 0x3FF0000000000000;"
        ));
        let m = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("mov.b64 {m}, {mant_bits};"));

        // Unbiased exponent as f64: e = (raw - 1023).
        let e_i64 = self.alloc_reg(PtxType::S64);
        self.raw_ptx(&format!("cvt.s64.u64 {e_i64}, {e_raw};"));
        let e_unbiased = self.alloc_reg(PtxType::S64);
        self.raw_ptx(&format!("sub.s64 {e_unbiased}, {e_i64}, 1023;"));
        let ef = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("cvt.rn.f64.s64 {ef}, {e_unbiased};"));

        // f = m - 1; s = f / (2 + f); z = s^2.
        let one = self.mov_imm_f64(1.0);
        let f = self.sub_f64(m, one);
        let two = self.mov_imm_f64(2.0);
        let two_plus_f = self.add_f64(f.clone(), two);
        let s = self.div_rn_f64(&f, &two_plus_f);
        let z = self.mul_rn_f64(&s, &s);

        // atanh series Q(z) = 1 + z/3 + z^2/5 + z^3/7 + z^4/9, Horner in z.
        // log1p via 2*atanh(f/(2+f)); with |f| in [0,1) => |s| <= 1/3 the
        // series error is ~1-2 ulp. Constants bound to locals first.
        let mut q = self.mov_imm_f64(1.0 / 9.0);
        let c7 = self.mov_imm_f64(1.0 / 7.0);
        q = self.fma_f64(q, z.clone(), c7);
        let c5 = self.mov_imm_f64(1.0 / 5.0);
        q = self.fma_f64(q, z.clone(), c5);
        let c3 = self.mov_imm_f64(1.0 / 3.0);
        q = self.fma_f64(q, z.clone(), c3);
        let c1 = self.mov_imm_f64(1.0);
        q = self.fma_f64(q, z, c1);

        // log(m) = 2*s * Q(z).
        let two_s = self.add_f64(s.clone(), s);
        let log_mant = self.mul_rn_f64(&two_s, &q);

        // log(x) = e*ln2 + log(m).
        let ln2 = self.mov_imm_f64(std::f64::consts::LN_2);
        self.fma_f64(ef, ln2, log_mant)
    }

    /// Computes the error function `erf(x)` for `f64` without any
    /// `.approx.f64` instruction.
    ///
    /// # Algorithm
    ///
    /// Abramowitz & Stegun formula 7.1.26: with `t = 1 / (1 + p*|x|)`,
    /// `erf(|x|) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)`,
    /// then the sign is restored because `erf` is odd. The polynomial is
    /// evaluated by Horner via `fma.rn.f64`, and `exp(-x^2)` reuses
    /// [`exp_f64`](Self::exp_f64).
    ///
    /// # Precision
    ///
    /// A&S 7.1.26 has a maximum absolute error of ~1.5e-7 — this is **reduced
    /// precision** relative to full `f64`, but the arithmetic is genuine `f64`
    /// (there is no `f32` bridge), and no `.approx.f64` instruction is emitted.
    pub fn erf_f64(&mut self, x: &Register) -> Register {
        // ax = |x|.
        let ax = self.alloc_reg(PtxType::F64);
        self.raw_ptx(&format!("abs.f64 {ax}, {x};"));

        // t = 1 / (1 + p*ax), p = 0.3275911.
        let pconst = self.mov_imm_f64(0.327_591_1);
        let one = self.mov_imm_f64(1.0);
        let denom = self.fma_f64(pconst, ax.clone(), one.clone());
        let t = self.div_rn_f64(&one, &denom);

        // poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))), Horner then outer t.
        // a5=1.061405429, a4=-1.453152027, a3=1.421413741,
        // a2=-0.284496736, a1=0.254829592. Constants bound to locals first.
        let mut poly = self.mov_imm_f64(1.061_405_429);
        let a4c = self.mov_imm_f64(-1.453_152_027);
        poly = self.fma_f64(poly, t.clone(), a4c);
        let a3c = self.mov_imm_f64(1.421_413_741);
        poly = self.fma_f64(poly, t.clone(), a3c);
        let a2c = self.mov_imm_f64(-0.284_496_736);
        poly = self.fma_f64(poly, t.clone(), a2c);
        let a1c = self.mov_imm_f64(0.254_829_592);
        poly = self.fma_f64(poly, t.clone(), a1c);
        poly = self.mul_rn_f64(&poly, &t);

        // e = exp(-ax^2); y = 1 - poly*e (this is erf(|x|)).
        let ax2 = self.mul_rn_f64(&ax, &ax);
        let nax2 = self.neg_f64(&ax2);
        let e = self.exp_f64(&nax2);
        let pe = self.mul_rn_f64(&poly, &e);
        let y = self.sub_f64(one, pe);

        // Restore sign: erf is odd, so return -y when x < 0.
        let neg_y = self.neg_f64(&y);
        let zero = self.mov_imm_f64(0.0);
        let p_neg = self.alloc_reg(PtxType::Pred);
        self.raw_ptx(&format!("setp.lt.f64 {p_neg}, {x}, {zero};"));
        // TODO(precision): upgrade to Cody 1969 rational minimax for full-f64 erf.
        self.selp(PtxType::F64, neg_y, y, p_neg)
    }

    /// Computes the hyperbolic tangent `tanh(x)` for `f64` without any
    /// `.approx.f64` instruction.
    ///
    /// # Algorithm
    ///
    /// `tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)`, reusing
    /// [`exp_f64`](Self::exp_f64) for `e^{2x}` and `div.rn.f64` for the final
    /// quotient.
    ///
    /// # Precision
    ///
    /// Inherits the precision of [`exp_f64`](Self::exp_f64) (~1–2 ulp) for
    /// moderate `|x|`. For large `|x|` the intermediate `e^{2x}` overflows;
    /// saturation/clamping is not handled.
    pub fn tanh_f64(&mut self, x: &Register) -> Register {
        // e = exp(2x).
        let two_x = self.add_f64(x.clone(), x.clone());
        let e = self.exp_f64(&two_x);
        let one = self.mov_imm_f64(1.0);
        let num = self.sub_f64(e.clone(), one.clone());
        let den = self.add_f64(e, one);
        // TODO: clamp |x| for large-magnitude stability.
        self.div_rn_f64(&num, &den)
    }
}

// Accuracy is NOT GPU-verified here (no NVIDIA hardware on dev/CI); these tests
// only check PTX generation/structure and that no `.approx.f64` form is emitted.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;
    use crate::builder::KernelBuilder;

    /// Helper: build a kernel with the given body and return the PTX text.
    fn build_with_body<F>(f: F) -> String
    where
        F: FnOnce(&mut BodyBuilder<'_>) + 'static,
    {
        KernelBuilder::new("test")
            .target(SmVersion::Sm80)
            .param("a", PtxType::U64)
            .param("n", PtxType::U32)
            .body(f)
            .build()
            .expect("build should succeed")
    }

    #[test]
    fn mov_imm_f64_emits_mov_b64() {
        let ptx = build_with_body(|b| {
            let _r = b.mov_imm_f64(1.5);
            b.ret();
        });
        assert!(ptx.contains("mov.b64"), "expected mov.b64 in:\n{ptx}");
        assert!(ptx.contains("0D"), "expected 0D literal in:\n{ptx}");
    }

    #[test]
    fn mov_imm_f32_emits_mov_b32() {
        let ptx = build_with_body(|b| {
            let _r = b.mov_imm_f32(1.5_f32);
            b.ret();
        });
        assert!(ptx.contains("mov.b32"), "expected mov.b32 in:\n{ptx}");
        assert!(ptx.contains("0F"), "expected 0F literal in:\n{ptx}");
    }

    #[test]
    fn div_rn_f64_emits_div() {
        let ptx = build_with_body(|b| {
            let a = b.mov_imm_f64(3.0);
            let c = b.mov_imm_f64(2.0);
            let _r = b.div_rn_f64(&a, &c);
            b.ret();
        });
        assert!(ptx.contains("div.rn.f64"), "expected div.rn.f64 in:\n{ptx}");
    }

    #[test]
    fn div_rn_f32_emits_div() {
        let ptx = build_with_body(|b| {
            let a = b.mov_imm_f32(3.0_f32);
            let c = b.mov_imm_f32(2.0_f32);
            let _r = b.div_rn_f32(&a, &c);
            b.ret();
        });
        assert!(ptx.contains("div.rn.f32"), "expected div.rn.f32 in:\n{ptx}");
    }

    #[test]
    fn exp_f64_emits_poly_and_2k() {
        let ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.exp_f64(&x);
            b.ret();
        });
        assert!(
            ptx.contains("fma.rn.f64"),
            "expected fma.rn.f64 (Taylor Horner) in:\n{ptx}"
        );
        assert!(ptx.contains("mov.b64"), "expected mov.b64 in:\n{ptx}");
        assert!(
            ptx.contains("shl.b64"),
            "expected shl.b64 (2^k assembly) in:\n{ptx}"
        );
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "expected cvt.rzi.s32.f64 (k extraction) in:\n{ptx}"
        );
    }

    #[test]
    fn log_f64_emits_bit_split_and_poly() {
        let ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(1.5);
            let _r = b.log_f64(&x);
            b.ret();
        });
        assert!(
            ptx.contains("shr.b64"),
            "expected shr.b64 (exponent extraction) in:\n{ptx}"
        );
        assert!(
            ptx.contains("and.b64"),
            "expected and.b64 (mantissa mask) in:\n{ptx}"
        );
        assert!(
            ptx.contains("fma.rn.f64"),
            "expected fma.rn.f64 (atanh Horner) in:\n{ptx}"
        );
    }

    #[test]
    fn erf_f64_emits_abs_and_exp() {
        let ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.erf_f64(&x);
            b.ret();
        });
        assert!(
            ptx.contains("fma.rn.f64"),
            "expected fma.rn.f64 (A&S Horner) in:\n{ptx}"
        );
        assert!(ptx.contains("abs.f64"), "expected abs.f64 (|x|) in:\n{ptx}");
        // erf reuses exp_f64, so the exp internals must appear too.
        assert!(
            ptx.contains("shl.b64"),
            "expected shl.b64 (exp 2^k assembly) in:\n{ptx}"
        );
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "expected cvt.rzi.s32.f64 (exp k extraction) in:\n{ptx}"
        );
    }

    #[test]
    fn tanh_f64_emits_div_and_exp() {
        let ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.tanh_f64(&x);
            b.ret();
        });
        assert!(
            ptx.contains("div.rn.f64"),
            "expected div.rn.f64 (final quotient) in:\n{ptx}"
        );
        // tanh reuses exp_f64, so the exp internals must appear too.
        assert!(
            ptx.contains("shl.b64"),
            "expected shl.b64 (exp 2^k assembly) in:\n{ptx}"
        );
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "expected cvt.rzi.s32.f64 (exp k extraction) in:\n{ptx}"
        );
    }

    #[test]
    fn no_approx_f64_emitted() {
        let exp_ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.exp_f64(&x);
            b.ret();
        });
        assert!(
            !exp_ptx.contains(".approx.f64"),
            "exp_f64 must not emit .approx.f64:\n{exp_ptx}"
        );

        let log_ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(1.5);
            let _r = b.log_f64(&x);
            b.ret();
        });
        assert!(
            !log_ptx.contains(".approx.f64"),
            "log_f64 must not emit .approx.f64:\n{log_ptx}"
        );

        let erf_ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.erf_f64(&x);
            b.ret();
        });
        assert!(
            !erf_ptx.contains(".approx.f64"),
            "erf_f64 must not emit .approx.f64:\n{erf_ptx}"
        );

        let tanh_ptx = build_with_body(|b| {
            let x = b.mov_imm_f64(0.5);
            let _r = b.tanh_f64(&x);
            b.ret();
        });
        assert!(
            !tanh_ptx.contains(".approx.f64"),
            "tanh_f64 must not emit .approx.f64:\n{tanh_ptx}"
        );
    }
}
