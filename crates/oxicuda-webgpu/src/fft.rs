//! WGSL radix-2 Cooley-Tukey FFT (Cooley & Tukey 1965) — host plan + shader.
//!
//! Implements the P2 roadmap item `shader/fft_wgsl.rs`: a compute-only FFT
//! dispatch suitable for browsers, expressed as a single iterative
//! Cooley-Tukey butterfly kernel plus a host-side [`WgslFftPlan`] that carries
//! the bit-reversal permutation and the twiddle-factor table.
//!
//! The plan (bit-reversal indices, twiddle factors) is **pure CPU arithmetic**
//! and is fully unit-tested here.  The shader-source generation is likewise
//! CPU-testable structurally.  Actually *running* the kernel on a GPU is gated
//! on a real adapter and is not exercised here.
//!
//! # Layout
//!
//! Complex data is stored interleaved: element `i` occupies indices `2*i`
//! (real) and `2*i + 1` (imaginary).  An in-place radix-2 DIT FFT proceeds in
//! `log2(n)` stages; each stage applies butterflies with stride `m = 2^stage`.
//!
//! The host first bit-reverses the input into the natural-order scratch buffer
//! (or vice-versa), then dispatches one [`fft_stage_wgsl`] pass per stage with
//! a `FftStageParams` uniform carrying the current `half_size` and the
//! per-stage twiddle base.  Twiddles `W_N^k = exp(-2πi k / N)` are precomputed
//! by [`WgslFftPlan::twiddles`] so the shader needs no `sin`/`cos`.

use std::f64::consts::PI;

/// FFT transform direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward DFT: `W_N^k = exp(-2πi k / N)`.
    Forward,
    /// Inverse DFT: `W_N^k = exp(+2πi k / N)` (the host scales by `1/N`).
    Inverse,
}

/// A host-side plan for a power-of-two radix-2 Cooley-Tukey FFT of length `n`.
///
/// Precomputes the bit-reversal permutation and the twiddle-factor table so
/// the WGSL kernel can be a pure butterfly with no transcendental calls.
#[derive(Debug, Clone)]
pub struct WgslFftPlan {
    n: u32,
    log2n: u32,
    direction: FftDirection,
    bit_reversal: Vec<u32>,
    /// Interleaved `(re, im)` twiddles `W_N^k` for `k in 0..n/2`.
    twiddles: Vec<f32>,
}

impl WgslFftPlan {
    /// Build a plan for transform length `n` (must be a power of two `>= 1`).
    ///
    /// # Errors
    ///
    /// Returns an error string if `n` is zero or not a power of two.
    pub fn new(n: u32, direction: FftDirection) -> Result<Self, String> {
        if n == 0 {
            return Err("FFT length must be non-zero".to_string());
        }
        if !n.is_power_of_two() {
            return Err(format!("FFT length {n} is not a power of two"));
        }
        let log2n = n.trailing_zeros();
        let bit_reversal = (0..n).map(|i| reverse_bits(i, log2n)).collect();

        // Twiddles W_N^k = exp(sign * 2πi k / N) for k in 0..n/2.
        let sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };
        let half = (n / 2).max(1) as usize;
        let mut twiddles = Vec::with_capacity(half * 2);
        for k in 0..half {
            let angle = sign * 2.0 * PI * (k as f64) / (n as f64);
            twiddles.push(angle.cos() as f32);
            twiddles.push(angle.sin() as f32);
        }

        Ok(Self {
            n,
            log2n,
            direction,
            bit_reversal,
            twiddles,
        })
    }

    /// Transform length.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.n
    }

    /// Whether the transform length is zero (always `false` for a built plan).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of FFT stages, `log2(n)`.
    #[must_use]
    pub fn num_stages(&self) -> u32 {
        self.log2n
    }

    /// Transform direction.
    #[must_use]
    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    /// The bit-reversal permutation: `bit_reversal()[i]` is the source index
    /// that lands at position `i` after the reorder.
    #[must_use]
    pub fn bit_reversal(&self) -> &[u32] {
        &self.bit_reversal
    }

    /// The interleaved `(re, im)` twiddle table of length `n/2`.
    #[must_use]
    pub fn twiddles(&self) -> &[f32] {
        &self.twiddles
    }

    /// The normalisation scale the host applies after an inverse transform
    /// (`1/n` for [`FftDirection::Inverse`], `1.0` for forward).
    #[must_use]
    pub fn inverse_scale(&self) -> f32 {
        match self.direction {
            FftDirection::Inverse => 1.0 / self.n as f32,
            FftDirection::Forward => 1.0,
        }
    }

    /// The `half_size` (`m/2`) uniform value for stage `stage` (`0`-based).
    ///
    /// Stage `s` operates on butterfly groups of size `m = 2^(s+1)`; its half
    /// size is `2^s`.  Returns `0` if `stage >= num_stages`.
    #[must_use]
    pub fn stage_half_size(&self, stage: u32) -> u32 {
        if stage >= self.log2n {
            0
        } else {
            1u32 << stage
        }
    }
}

/// Reverse the low `bits` bits of `value`.
#[inline]
#[must_use]
pub fn reverse_bits(value: u32, bits: u32) -> u32 {
    if bits == 0 {
        return 0;
    }
    let mut v = value;
    let mut r = 0u32;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Generate WGSL for the bit-reversal reorder pass.
///
/// Reads `src` (interleaved complex) and writes each element to its
/// bit-reversed position in `dst`, using a precomputed permutation buffer
/// (the host uploads [`WgslFftPlan::bit_reversal`]).
#[must_use]
pub fn fft_bitreverse_wgsl() -> String {
    r#"
struct FftReorderParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       src:     array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:     array<f32>;
@group(0) @binding(2) var<storage, read>       perm:    array<u32>;
@group(0) @binding(3) var<uniform>             params:  FftReorderParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let j = perm[i];
    // Move complex element j -> i.
    dst[2u * i]      = src[2u * j];
    dst[2u * i + 1u] = src[2u * j + 1u];
}
"#
    .to_string()
}

/// Generate WGSL for a single radix-2 Cooley-Tukey butterfly stage (DIT).
///
/// Operates in-place on `data` (interleaved complex, already bit-reversed).
/// Each invocation handles one butterfly pair within a group of size
/// `2 * half_size`; the twiddle `W` is fetched from the precomputed `twiddles`
/// buffer at the stride-appropriate index.
///
/// `FftStageParams.half_size` is `m/2` for the current stage; the per-stage
/// twiddle stride is `n / (2 * half_size)`.
#[must_use]
pub fn fft_stage_wgsl() -> String {
    r#"
struct FftStageParams {
    n:            u32,
    half_size:    u32,  // m / 2 for this stage
    twiddle_step: u32,  // n / (2 * half_size)
    _pad:         u32,
}

@group(0) @binding(0) var<storage, read_write> data:     array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles: array<f32>;
@group(0) @binding(2) var<uniform>             params:   FftStageParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let total_pairs = params.n / 2u;
    if (pair >= total_pairs) { return; }

    let m = params.half_size * 2u;
    // Index of this butterfly within its group, and the group base.
    let k = pair % params.half_size;
    let group = pair / params.half_size;
    let base = group * m;
    let i0 = base + k;          // top of the butterfly
    let i1 = i0 + params.half_size; // bottom

    // Twiddle W = (wr, wi) for this k.
    let tw = k * params.twiddle_step;
    let wr = twiddles[2u * tw];
    let wi = twiddles[2u * tw + 1u];

    let ar = data[2u * i0];
    let ai = data[2u * i0 + 1u];
    let br = data[2u * i1];
    let bi = data[2u * i1 + 1u];

    // t = W * b  (complex multiply).
    let tr = wr * br - wi * bi;
    let ti = wr * bi + wi * br;

    // Butterfly: a' = a + t, b' = a - t.
    data[2u * i0]      = ar + tr;
    data[2u * i0 + 1u] = ai + ti;
    data[2u * i1]      = ar - tr;
    data[2u * i1 + 1u] = ai - ti;
}
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_bits_known_values() {
        // 3 bits: 0b001 (1) -> 0b100 (4).
        assert_eq!(reverse_bits(0b001, 3), 0b100);
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b000, 3), 0b000);
        assert_eq!(reverse_bits(0b111, 3), 0b111);
        // 0 bits -> 0.
        assert_eq!(reverse_bits(5, 0), 0);
    }

    #[test]
    fn plan_rejects_non_power_of_two() {
        assert!(WgslFftPlan::new(0, FftDirection::Forward).is_err());
        assert!(WgslFftPlan::new(3, FftDirection::Forward).is_err());
        assert!(WgslFftPlan::new(6, FftDirection::Forward).is_err());
        assert!(WgslFftPlan::new(7, FftDirection::Forward).is_err());
    }

    #[test]
    fn plan_accepts_powers_of_two() {
        for &n in &[1u32, 2, 4, 8, 16, 1024] {
            let plan = WgslFftPlan::new(n, FftDirection::Forward).expect("power of two");
            assert_eq!(plan.len(), n);
            assert_eq!(plan.num_stages(), n.trailing_zeros());
            assert!(!plan.is_empty());
        }
    }

    #[test]
    fn plan_bit_reversal_is_a_permutation() {
        let plan = WgslFftPlan::new(8, FftDirection::Forward).expect("plan");
        let perm = plan.bit_reversal();
        assert_eq!(perm.len(), 8);
        // Known 3-bit reversal of 0..8.
        assert_eq!(perm, &[0, 4, 2, 6, 1, 5, 3, 7]);
        // It is a genuine permutation (every index 0..8 appears once).
        let mut seen = perm.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0u32..8).collect::<Vec<_>>());
    }

    #[test]
    fn plan_twiddles_length_and_dc() {
        let plan = WgslFftPlan::new(8, FftDirection::Forward).expect("plan");
        // n/2 complex = n interleaved floats.
        assert_eq!(plan.twiddles().len(), 8);
        // W_N^0 = 1 + 0i.
        assert!((plan.twiddles()[0] - 1.0).abs() < 1e-6);
        assert!(plan.twiddles()[1].abs() < 1e-6);
    }

    #[test]
    fn plan_forward_twiddle_sign_is_negative() {
        // Forward: W_8^1 = exp(-2πi/8) = (cos(-45°), sin(-45°)) ≈ (0.707, -0.707).
        let plan = WgslFftPlan::new(8, FftDirection::Forward).expect("plan");
        let re = plan.twiddles()[2];
        let im = plan.twiddles()[3];
        assert!((re - 0.707_106_77).abs() < 1e-5);
        assert!((im + 0.707_106_77).abs() < 1e-5, "im was {im}");
    }

    #[test]
    fn plan_inverse_twiddle_sign_is_positive() {
        // Inverse: W_8^1 = exp(+2πi/8) ≈ (0.707, +0.707).
        let plan = WgslFftPlan::new(8, FftDirection::Inverse).expect("plan");
        let im = plan.twiddles()[3];
        assert!(
            im > 0.0,
            "inverse imaginary twiddle should be positive, got {im}"
        );
        assert!((plan.inverse_scale() - (1.0 / 8.0)).abs() < 1e-7);
    }

    #[test]
    fn plan_forward_inverse_scale() {
        let fwd = WgslFftPlan::new(16, FftDirection::Forward).expect("plan");
        assert!((fwd.inverse_scale() - 1.0).abs() < 1e-7);
        let inv = WgslFftPlan::new(16, FftDirection::Inverse).expect("plan");
        assert!((inv.inverse_scale() - 1.0 / 16.0).abs() < 1e-7);
    }

    #[test]
    fn plan_stage_half_sizes() {
        let plan = WgslFftPlan::new(8, FftDirection::Forward).expect("plan");
        // 3 stages: half sizes 1, 2, 4.
        assert_eq!(plan.stage_half_size(0), 1);
        assert_eq!(plan.stage_half_size(1), 2);
        assert_eq!(plan.stage_half_size(2), 4);
        // Out-of-range stage -> 0.
        assert_eq!(plan.stage_half_size(3), 0);
    }

    #[test]
    fn plan_n1_has_zero_stages() {
        let plan = WgslFftPlan::new(1, FftDirection::Forward).expect("plan");
        assert_eq!(plan.num_stages(), 0);
        // n/2 == 0, but we reserve at least 1 twiddle slot (DC).
        assert_eq!(plan.twiddles().len(), 2);
    }

    // ── shader source ─────────────────────────────────────────────────────

    #[test]
    fn wgsl_bitreverse_moves_by_permutation() {
        let src = fft_bitreverse_wgsl();
        assert!(src.contains("@compute @workgroup_size(256)"));
        assert!(src.contains("let j = perm[i];"));
        assert!(src.contains("dst[2u * i]      = src[2u * j];"));
        assert!(src.contains("dst[2u * i + 1u] = src[2u * j + 1u];"));
        // Permutation buffer binding.
        assert!(src.contains("perm:    array<u32>"));
    }

    #[test]
    fn wgsl_fft_stage_complex_butterfly() {
        let src = fft_stage_wgsl();
        assert!(src.contains("@compute @workgroup_size(256)"));
        // Complex multiply t = W * b.
        assert!(src.contains("let tr = wr * br - wi * bi;"));
        assert!(src.contains("let ti = wr * bi + wi * br;"));
        // Butterfly add/sub.
        assert!(src.contains("data[2u * i0]      = ar + tr;"));
        assert!(src.contains("data[2u * i1]      = ar - tr;"));
    }

    #[test]
    fn wgsl_fft_stage_uses_precomputed_twiddles() {
        let src = fft_stage_wgsl();
        // Twiddles come from a buffer, not from sin/cos in the shader.
        assert!(src.contains("twiddles: array<f32>"));
        assert!(src.contains("let wr = twiddles[2u * tw];"));
        assert!(src.contains("let wi = twiddles[2u * tw + 1u];"));
        assert!(!src.contains("sin("));
        assert!(!src.contains("cos("));
        // Stage params carry half_size and twiddle_step.
        assert!(src.contains("half_size:    u32"));
        assert!(src.contains("twiddle_step: u32"));
    }

    #[test]
    fn wgsl_fft_stage_has_bounds_guard() {
        let src = fft_stage_wgsl();
        assert!(src.contains("let total_pairs = params.n / 2u;"));
        assert!(src.contains("if (pair >= total_pairs) { return; }"));
    }
}
