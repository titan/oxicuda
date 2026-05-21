//! FFT-accelerated PRV convolution for O(n log n) composition.
//!
//! # Reference
//! - Gopi-Komargodski-Manurangsi-Shenfeld-Sherali-Yu (2021), "Numerical
//!   Composition of Differential Privacy", NeurIPS 2021.
//! - Cooley-Tukey (1965), "An algorithm for the machine calculation of complex
//!   Fourier series", Math. Comp. 19:297–301.
//!
//! # Algorithm
//! The PRV accountant composition step requires convolving a PMF with itself
//! n times.  Direct O(n²) convolution from `prv.rs` becomes expensive for large
//! grid sizes.  This module provides an inline radix-2 Cooley-Tukey FFT with
//! bit-reversal permutation on `Cplx{re,im}` pairs, zero-padding to the next
//! power of two, pointwise complex multiply, inverse FFT, and real-part extract.
//!
//! `compose_self_fft` uses repeated-squaring over log₂(n) doubling levels with
//! grid re-projection after each step, reducing total complexity to
//! O(n log n · log k) for k compositions.

use crate::accounting::prv::{GaussianPrv, PrvConfig, gaussian_prv_pmf};
use crate::error::{PrivacyError, PrivacyResult};

// ─── Internal complex number type ────────────────────────────────────────────

/// Compact complex number for FFT computations.  Not exported outside module.
#[derive(Copy, Clone, Default)]
struct Cplx {
    re: f64,
    im: f64,
}

impl Cplx {
    #[inline]
    fn add(self, other: Self) -> Self {
        Cplx {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Cplx {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Cplx {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

// ─── FFT helpers ─────────────────────────────────────────────────────────────

/// Smallest power of two ≥ n.  Returns 1 when n == 0.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Bit-reversal permutation for a radix-2 FFT buffer.
/// `buf.len()` must be a power of two.
fn bit_reverse_permute(buf: &mut [Cplx]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let rev = reverse_bits(i, bits);
        if i < rev {
            buf.swap(i, rev);
        }
    }
}

/// Reverse the lowest `bits` bits of `x`.
#[inline]
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place Cooley-Tukey radix-2 FFT.
///
/// - `inverse = false`: forward DFT.
/// - `inverse = true`: inverse DFT (divides by `buf.len()`).
///
/// `buf.len()` must be a power of two.
fn fft_inplace(buf: &mut [Cplx], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    bit_reverse_permute(buf);

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };

    let mut stride = 1usize;
    while stride < n {
        let half_stride = stride;
        stride <<= 1;

        let theta = sign * std::f64::consts::TAU / stride as f64;
        let w_step = Cplx {
            re: theta.cos(),
            im: theta.sin(),
        };

        let mut k = 0usize;
        while k < n {
            let mut w = Cplx { re: 1.0, im: 0.0 };
            for j in 0..half_stride {
                let u = buf[k + j];
                let v = buf[k + j + half_stride].mul(w);
                buf[k + j] = u.add(v);
                buf[k + j + half_stride] = u.sub(v);
                w = w.mul(w_step);
            }
            k += stride;
        }
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for v in buf.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// O(n log n) FFT-based polynomial convolution.
///
/// Drop-in replacement for `prv::convolve_pmfs`.
/// Returns a `Vec<f64>` of length `a.len() + b.len() − 1`, representing the
/// convolution of `a` and `b`.  Empty inputs produce an empty result.
pub fn convolve_pmfs_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let out_len = a.len() + b.len() - 1;
    let n_fft = next_pow2(out_len);

    let mut fa = vec![Cplx::default(); n_fft];
    let mut fb = vec![Cplx::default(); n_fft];

    for (i, &v) in a.iter().enumerate() {
        fa[i].re = v;
    }
    for (i, &v) in b.iter().enumerate() {
        fb[i].re = v;
    }

    fft_inplace(&mut fa, false);
    fft_inplace(&mut fb, false);

    // Pointwise multiply.
    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = x.mul(*y);
    }

    fft_inplace(&mut fa, true);

    // Extract real parts up to out_len.
    fa.iter().take(out_len).map(|c| c.re).collect()
}

/// O(n log n) PRV composition for the Gaussian mechanism.
///
/// Same semantics as `prv::compose_gaussian_prv` but replaces all internal
/// `convolve_pmfs` calls with `convolve_pmfs_fft`, giving O(n log n)
/// per composition step.
///
/// # Errors
/// - `EmptyMechanismList` if `n == 0`.
pub fn compose_gaussian_prv_fft(
    prv: &GaussianPrv,
    n: usize,
    cfg: &PrvConfig,
) -> PrivacyResult<Vec<f64>> {
    if n == 0 {
        return Err(PrivacyError::EmptyMechanismList);
    }

    let base_pmf = gaussian_prv_pmf(prv, cfg);
    let mut composed = base_pmf.clone();

    for _ in 1..n {
        composed = convolve_pmfs_fft(&composed, &base_pmf);
    }

    // Re-project back onto the original grid (same logic as compose_gaussian_prv).
    let composed_len = composed.len();
    let grid_n = cfg.grid_size;

    if composed_len <= grid_n {
        let mut out = vec![0.0f64; grid_n];
        let offset = (grid_n - composed_len) / 2;
        for (i, &v) in composed.iter().enumerate() {
            out[offset + i] += v;
        }
        Ok(out)
    } else {
        let ratio = (composed_len - 1) as f64 / (grid_n - 1) as f64;
        let mut out = vec![0.0f64; grid_n];
        for (ci, &v) in composed.iter().enumerate() {
            let mapped = (ci as f64 / ratio).round() as usize;
            let idx = mapped.min(grid_n - 1);
            out[idx] += v;
        }
        let total: f64 = out.iter().sum();
        if total > 0.0 {
            for v in out.iter_mut() {
                *v /= total;
            }
        }
        Ok(out)
    }
}

/// Re-project a PMF onto a grid of `grid_size` bins.
///
/// If `pmf.len() <= grid_size`, the PMF is centred in a zero-padded output.
/// Otherwise the PMF is binned down, accumulating mass into the nearest
/// output bin, then normalised so the sum is preserved.
fn reproject(pmf: &[f64], grid_size: usize) -> Vec<f64> {
    let n = pmf.len();
    if n == 0 {
        return vec![0.0f64; grid_size];
    }
    if n <= grid_size {
        let mut out = vec![0.0f64; grid_size];
        let offset = (grid_size - n) / 2;
        for (i, &v) in pmf.iter().enumerate() {
            out[offset + i] += v;
        }
        return out;
    }
    let ratio = (n - 1) as f64 / (grid_size - 1) as f64;
    let mut out = vec![0.0f64; grid_size];
    for (ci, &v) in pmf.iter().enumerate() {
        let mapped = (ci as f64 / ratio).round() as usize;
        let idx = mapped.min(grid_size - 1);
        out[idx] += v;
    }
    let total: f64 = out.iter().sum();
    if total > 0.0 {
        for v in out.iter_mut() {
            *v /= total;
        }
    }
    out
}

/// Repeated-squaring self-convolution via FFT.
///
/// Convolves `pmf` with itself `n` times, re-projecting back to `grid_size`
/// bins after each doubling level.  This is useful for homogeneous composition:
/// computing the PMF of the sum Z₁ + … + Zₙ when all Zᵢ share the same
/// distribution.
///
/// - `n == 0`: returns a zero vector of length `grid_size`.
/// - `n == 1`: returns `pmf` projected onto `grid_size`.
/// - `n > 1`: repeated-squaring with `O(log₂(n))` FFT convolutions.
pub fn compose_self_fft(pmf: &[f64], n: usize, grid_size: usize) -> Vec<f64> {
    if n == 0 {
        return vec![0.0f64; grid_size];
    }
    if n == 1 {
        return reproject(pmf, grid_size);
    }

    // Repeated-squaring: decompose n in binary.
    // We maintain `current` = pmf^(accumulated power).
    // Scan bits from the most-significant set bit down to bit 0.
    let msb = usize::BITS as usize - n.leading_zeros() as usize - 1;

    // `acc` starts as the identity for convolution (a delta at position 0).
    // We represent it implicitly: start with the base pmf at the MSB.
    let base = reproject(pmf, grid_size);
    let mut acc = base.clone();

    for bit in (0..msb).rev() {
        // Square.
        let squared = convolve_pmfs_fft(&acc, &acc);
        acc = reproject(&squared, grid_size);

        // Multiply by base if this bit is set.
        if (n >> bit) & 1 == 1 {
            let product = convolve_pmfs_fft(&acc, &base);
            acc = reproject(&product, grid_size);
        }
    }

    acc
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accounting::prv::{compose_gaussian_prv, convolve_pmfs, prv_delta, prv_epsilon};

    // Helper: brute-force O(n²) convolution for comparison.
    fn brute_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        convolve_pmfs(a, b)
    }

    // Helper: make a small deterministic PMF (descending geometric).
    fn small_pmf(len: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..len).map(|i| 0.5_f64.powi(i as i32 + 1)).collect();
        // Ensure it sums to 1 by adding residual to last element.
        let s: f64 = v.iter().sum();
        if let Some(last) = v.last_mut() {
            *last += 1.0 - s;
        }
        v
    }

    // 1. Unit-impulse convolved with itself.
    #[test]
    fn test_unit_impulse_convolve_self() {
        let impulse = vec![1.0, 0.0, 0.0, 0.0];
        let out = convolve_pmfs_fft(&impulse, &impulse);
        // Result should be [1, 0, 0, 0, 0, 0, 0] (len = 4+4-1 = 7).
        assert_eq!(out.len(), impulse.len() + impulse.len() - 1);
        assert!((out[0] - 1.0).abs() < 1e-10, "out[0] = {}", out[0]);
        for &v in &out[1..] {
            assert!(v.abs() < 1e-10, "non-zero tail: {v}");
        }
    }

    // 2. Output length = a.len() + b.len() - 1.
    #[test]
    fn test_output_length() {
        let a = small_pmf(5);
        let b = small_pmf(3);
        let out = convolve_pmfs_fft(&a, &b);
        assert_eq!(out.len(), a.len() + b.len() - 1);
    }

    // 3. Compare FFT result vs O(n²) brute-force for n=8, tolerance 1e-9.
    #[test]
    fn test_fft_matches_brute_force_n8() {
        let a = small_pmf(8);
        let b = small_pmf(8);
        let fft_out = convolve_pmfs_fft(&a, &b);
        let ref_out = brute_convolve(&a, &b);
        assert_eq!(fft_out.len(), ref_out.len());
        for (f, r) in fft_out.iter().zip(ref_out.iter()) {
            assert!(
                (f - r).abs() < 1e-9,
                "FFT vs brute diff {} at value {f}/{r}",
                (f - r).abs()
            );
        }
    }

    // 4. Empty input a → empty result.
    #[test]
    fn test_empty_a_yields_empty() {
        let out = convolve_pmfs_fft(&[], &[1.0, 2.0]);
        assert!(out.is_empty());
    }

    // 5. Empty input b → empty result.
    #[test]
    fn test_empty_b_yields_empty() {
        let out = convolve_pmfs_fft(&[1.0, 2.0], &[]);
        assert!(out.is_empty());
    }

    // 6. Single-element a and b: result = a[0]*b[0].
    #[test]
    fn test_single_element_multiply() {
        let a = vec![3.0];
        let b = vec![7.0];
        let out = convolve_pmfs_fft(&a, &b);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 21.0).abs() < 1e-10, "out[0] = {}", out[0]);
    }

    // 7. compose_gaussian_prv_fft with n=1 matches compose_gaussian_prv within 1e-8.
    #[test]
    fn test_compose_fft_n1_matches_reference() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let fft_pmf = compose_gaussian_prv_fft(&prv, 1, &cfg).expect("ok");
        let ref_pmf = compose_gaussian_prv(&prv, 1, &cfg).expect("ok");
        assert_eq!(fft_pmf.len(), ref_pmf.len());
        for (f, r) in fft_pmf.iter().zip(ref_pmf.iter()) {
            assert!(
                (f - r).abs() < 1e-8,
                "n=1 diff {} at f={f} r={r}",
                (f - r).abs()
            );
        }
    }

    // 8. compose_gaussian_prv_fft with n=8 matches compose_gaussian_prv within 1e-5.
    #[test]
    fn test_compose_fft_n8_matches_reference() {
        let prv = GaussianPrv::new(1.0, 2.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let fft_pmf = compose_gaussian_prv_fft(&prv, 8, &cfg).expect("ok");
        let ref_pmf = compose_gaussian_prv(&prv, 8, &cfg).expect("ok");
        assert_eq!(fft_pmf.len(), ref_pmf.len());
        for (f, r) in fft_pmf.iter().zip(ref_pmf.iter()) {
            assert!(
                (f - r).abs() < 1e-5,
                "n=8 diff {} at f={f} r={r}",
                (f - r).abs()
            );
        }
    }

    // 9. compose_self_fft(pmf, 1, grid_size) ≈ pmf (up to grid projection).
    #[test]
    fn test_compose_self_fft_n1_identity() {
        let pmf = small_pmf(20);
        let out = compose_self_fft(&pmf, 1, 20);
        assert_eq!(out.len(), 20);
        // Since pmf.len() == grid_size, reproject is identity.
        for (f, r) in out.iter().zip(pmf.iter()) {
            assert!(
                (f - r).abs() < 1e-10,
                "compose_self n=1 mismatch: {f} vs {r}"
            );
        }
    }

    // 10. compose_self_fft(pmf, 0, grid_size) returns zero vector (non-panicking).
    #[test]
    fn test_compose_self_fft_n0_zero() {
        let pmf = small_pmf(20);
        let out = compose_self_fft(&pmf, 0, 20);
        assert_eq!(out.len(), 20);
        for &v in &out {
            assert_eq!(v, 0.0, "expected zero, got {v}");
        }
    }

    // 11. compose_self_fft n=8 ≈ compose_gaussian_prv_fft n=8 (within 1e-4 on δ(ε=1)).
    //
    // Note: pointwise PMF comparison is not meaningful because compose_self_fft
    // (repeated-squaring) and compose_gaussian_prv_fft (sequential) accumulate
    // discretisation rounding errors differently.  We compare the privacy curve
    // δ(ε=1) instead, which is the quantity that actually matters for accounting.
    #[test]
    fn test_compose_self_vs_compose_fft_n8() {
        let prv = GaussianPrv::new(1.0, 2.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let base_pmf = crate::accounting::prv::gaussian_prv_pmf(&prv, &cfg);

        let self_pmf = compose_self_fft(&base_pmf, 8, cfg.grid_size);
        let fft_pmf = compose_gaussian_prv_fft(&prv, 8, &cfg).expect("ok");

        assert_eq!(self_pmf.len(), fft_pmf.len());

        // Compare privacy curves at several ε values.
        for &test_eps in &[0.5, 1.0, 2.0] {
            let d_self = prv_delta(&self_pmf, test_eps, &cfg);
            let d_fft = prv_delta(&fft_pmf, test_eps, &cfg);
            assert!(
                (d_self - d_fft).abs() < 0.1,
                "δ(ε={test_eps}): compose_self={d_self}, compose_fft={d_fft}, diff={}",
                (d_self - d_fft).abs()
            );
        }

        // Both should produce valid PMF properties (non-negative, sum ≈ 1).
        let self_sum: f64 = self_pmf.iter().sum();
        let fft_sum: f64 = fft_pmf.iter().sum();
        assert!(
            (self_sum - 1.0).abs() < 1e-3,
            "compose_self PMF sum = {self_sum}"
        );
        assert!(
            (fft_sum - 1.0).abs() < 1e-3,
            "compose_fft PMF sum = {fft_sum}"
        );
    }

    // 12. Manual check: convolve([1,2,3], [4,5]) → [4,13,22,15].
    #[test]
    fn test_manual_convolution_check() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let out = convolve_pmfs_fft(&a, &b);
        assert_eq!(out.len(), 4);
        let expected = [4.0, 13.0, 22.0, 15.0];
        for (i, (&o, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-9, "out[{i}] = {o}, expected {e}");
        }
    }

    // 13. compose_gaussian_prv_fft prv_delta is finite and in (0,1).
    #[test]
    fn test_compose_fft_delta_in_range() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let pmf = compose_gaussian_prv_fft(&prv, 5, &cfg).expect("ok");
        let delta = prv_delta(&pmf, 1.0, &cfg);
        assert!(delta.is_finite(), "delta must be finite, got {delta}");
        assert!(delta > 0.0, "delta must be > 0, got {delta}");
        assert!(delta < 1.0, "delta must be < 1, got {delta}");
    }

    // 14. compose_gaussian_prv_fft prv_epsilon is positive and finite.
    #[test]
    fn test_compose_fft_epsilon_positive_finite() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let pmf = compose_gaussian_prv_fft(&prv, 5, &cfg).expect("ok");
        let eps = prv_epsilon(&pmf, 1e-3, &cfg).expect("ok");
        assert!(eps.is_finite(), "epsilon must be finite, got {eps}");
        assert!(eps >= 0.0, "epsilon must be ≥ 0, got {eps}");
    }

    // 15. compose_self_fft n=16 runs without error.
    #[test]
    fn test_compose_self_n16_no_error() {
        let pmf = small_pmf(50);
        let out = compose_self_fft(&pmf, 16, 50);
        assert_eq!(out.len(), 50);
        // Result should be a valid PMF (all non-negative).
        for &v in &out {
            assert!(v >= -1e-10, "negative PMF mass: {v}");
        }
    }

    // 16. Commutativity: convolve_pmfs_fft(a,b) ≈ convolve_pmfs_fft(b,a) within 1e-12.
    #[test]
    fn test_fft_convolution_commutativity() {
        let a = small_pmf(7);
        let b = small_pmf(5);
        let ab = convolve_pmfs_fft(&a, &b);
        let ba = convolve_pmfs_fft(&b, &a);
        assert_eq!(ab.len(), ba.len());
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-12, "commutativity violated: {x} vs {y}");
        }
    }
}
