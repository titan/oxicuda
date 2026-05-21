//! HRR with FFT-accelerated binding.
//!
//! Reference:
//! - T. A. Plate, "Holographic Reduced Representations," *IEEE Transactions
//!   on Neural Networks*, vol. 6, no. 3 (1995).
//! - The convolution theorem: circular convolution of two length-`D`
//!   real-valued vectors becomes elementwise complex multiplication in the
//!   Fourier domain. Hence
//!
//!   ```text
//!   conv(x, y) = IFFT( FFT(x) ⊙ FFT(y) )
//!   ```
//!
//!   reducing complexity from `O(D²)` (naive [`crate::ops::binding::circular_convolution`])
//!   to `O(D log D)`.
//!
//! Circular **correlation** (HRR unbinding) uses the conjugate of `FFT(x)`:
//!
//! ```text
//! corr(x, y) = IFFT( conj(FFT(x)) ⊙ FFT(y) )
//! ```
//!
//! All public entry points validate that inputs are non-empty, equal in
//! length, and a power of two (required by the radix-2 Cooley-Tukey FFT
//! used here). Padding callers may zero-pad to the next power of two before
//! calling.

use crate::error::{HdcError, HdcResult};

// ── FFT helpers ───────────────────────────────────────────────────────────────

/// Returns `true` iff `n >= 1` is an exact power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// `log2(n)` for a power-of-two `n >= 1`. Caller must guarantee the
/// power-of-two invariant.
fn log2_pow2(mut n: usize) -> usize {
    let mut k = 0usize;
    while n > 1 {
        n >>= 1;
        k += 1;
    }
    k
}

/// In-place bit reversal permutation of two parallel f32 buffers of equal
/// length `n` (a power of two). Used by the iterative Cooley-Tukey FFT.
fn bit_reverse_in_place(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

// ── Public FFT API ────────────────────────────────────────────────────────────

/// Stateless namespace for FFT-accelerated HRR primitives.
pub struct HrrFft;

/// Configuration for an FFT-binding HRR operation.
///
/// Currently a transparent wrapper around the chosen FFT length. The length
/// must be a power of two so that the radix-2 Cooley-Tukey decomposition
/// applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HrrFftConfig {
    /// FFT length (must be a power of two).
    pub dim: usize,
}

impl HrrFftConfig {
    /// Validate that `dim >= 1` and is a power of two.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::DimensionMismatch`] if `dim` is not a power of two.
    pub fn validate(&self) -> HdcResult<()> {
        if self.dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if !is_power_of_two(self.dim) {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim.next_power_of_two(),
                got: self.dim,
            });
        }
        Ok(())
    }
}

impl HrrFft {
    /// Radix-2 iterative Cooley-Tukey FFT in place on split real/imaginary
    /// buffers. If `inverse == true`, performs the inverse FFT with the
    /// standard `1/n` scaling.
    ///
    /// The transform direction is selected by the sign of the twiddle
    /// factors:
    ///
    /// ```text
    /// forward:  e^{-i·2π·k/n}
    /// inverse:  e^{+i·2π·k/n}
    /// ```
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if either buffer is empty.
    /// - [`HdcError::DimensionMismatch`] if `re.len() != im.len()` or the
    ///   length is not a power of two.
    pub fn fft(re: &mut [f32], im: &mut [f32], inverse: bool) -> HdcResult<()> {
        if re.is_empty() || im.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if re.len() != im.len() {
            return Err(HdcError::DimensionMismatch {
                expected: re.len(),
                got: im.len(),
            });
        }
        let n = re.len();
        if !is_power_of_two(n) {
            return Err(HdcError::DimensionMismatch {
                expected: n.next_power_of_two(),
                got: n,
            });
        }
        bit_reverse_in_place(re, im, n);
        let log_n = log2_pow2(n);
        let sign: f64 = if inverse { 1.0 } else { -1.0 };

        let mut s = 1usize;
        while s <= log_n {
            let m = 1usize << s;
            let half = m >> 1;
            let angle = sign * std::f64::consts::TAU / m as f64;
            // Principal twiddle e^{i·angle}.
            let wm_re = angle.cos() as f32;
            let wm_im = angle.sin() as f32;
            let mut k = 0usize;
            while k < n {
                // Rolling twiddle w^j.
                let mut w_re = 1f32;
                let mut w_im = 0f32;
                for j in 0..half {
                    let t_re = w_re * re[k + j + half] - w_im * im[k + j + half];
                    let t_im = w_re * im[k + j + half] + w_im * re[k + j + half];
                    let u_re = re[k + j];
                    let u_im = im[k + j];
                    re[k + j] = u_re + t_re;
                    im[k + j] = u_im + t_im;
                    re[k + j + half] = u_re - t_re;
                    im[k + j + half] = u_im - t_im;
                    // w := w * wm
                    let new_w_re = w_re * wm_re - w_im * wm_im;
                    let new_w_im = w_re * wm_im + w_im * wm_re;
                    w_re = new_w_re;
                    w_im = new_w_im;
                }
                k += m;
            }
            s += 1;
        }
        if inverse {
            let inv = 1f32 / n as f32;
            for x in re.iter_mut() {
                *x *= inv;
            }
            for x in im.iter_mut() {
                *x *= inv;
            }
        }
        Ok(())
    }

    /// `O(D log D)` circular convolution via the convolution theorem.
    ///
    /// Equivalent (up to floating-point rounding) to the naive `O(D²)`
    /// [`HrrFft::naive_circular_convolve`] and to
    /// [`crate::ops::binding::circular_convolution`].
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if either input is empty.
    /// - [`HdcError::DimensionMismatch`] if the inputs differ in length or
    ///   the common length is not a power of two.
    pub fn fft_circular_convolve(x: &[f32], y: &[f32]) -> HdcResult<Vec<f32>> {
        Self::validate_pair(x, y)?;
        let n = x.len();
        let mut xr: Vec<f32> = x.to_vec();
        let mut xi: Vec<f32> = vec![0f32; n];
        let mut yr: Vec<f32> = y.to_vec();
        let mut yi: Vec<f32> = vec![0f32; n];
        Self::fft(&mut xr, &mut xi, false)?;
        Self::fft(&mut yr, &mut yi, false)?;
        // Elementwise (X)(Y) in place into (xr, xi).
        for k in 0..n {
            let a_re = xr[k];
            let a_im = xi[k];
            let b_re = yr[k];
            let b_im = yi[k];
            xr[k] = a_re * b_re - a_im * b_im;
            xi[k] = a_re * b_im + a_im * b_re;
        }
        Self::fft(&mut xr, &mut xi, true)?;
        Ok(xr)
    }

    /// `O(D log D)` circular correlation (HRR unbinding) via the conjugate
    /// FFT of `x`:
    ///
    /// ```text
    /// corr(x, y) = IFFT( conj(FFT(x)) ⊙ FFT(y) )
    /// ```
    ///
    /// This matches the naive correlation
    /// [`crate::ops::binding::circular_correlation`] up to floating-point
    /// rounding.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if either input is empty.
    /// - [`HdcError::DimensionMismatch`] if the inputs differ in length or
    ///   the common length is not a power of two.
    pub fn fft_circular_correlate(x: &[f32], y: &[f32]) -> HdcResult<Vec<f32>> {
        Self::validate_pair(x, y)?;
        let n = x.len();
        let mut xr: Vec<f32> = x.to_vec();
        let mut xi: Vec<f32> = vec![0f32; n];
        let mut yr: Vec<f32> = y.to_vec();
        let mut yi: Vec<f32> = vec![0f32; n];
        Self::fft(&mut xr, &mut xi, false)?;
        Self::fft(&mut yr, &mut yi, false)?;
        // conj(X) * Y = (xr - i xi)(yr + i yi)
        //             = (xr*yr + xi*yi) + i*(xr*yi - xi*yr)
        for k in 0..n {
            let a_re = xr[k];
            let a_im = xi[k];
            let b_re = yr[k];
            let b_im = yi[k];
            xr[k] = a_re * b_re + a_im * b_im;
            xi[k] = a_re * b_im - a_im * b_re;
        }
        Self::fft(&mut xr, &mut xi, true)?;
        Ok(xr)
    }

    /// Naive `O(D²)` circular convolution kept here as the correctness
    /// reference for the FFT path and for self-contained tests.
    ///
    /// `c[k] = Σ_{j=0..n-1} x[j] · y[(k − j + n) mod n]`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if either input is empty.
    /// - [`HdcError::DimensionMismatch`] if the inputs differ in length.
    pub fn naive_circular_convolve(x: &[f32], y: &[f32]) -> HdcResult<Vec<f32>> {
        if x.is_empty() || y.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if x.len() != y.len() {
            return Err(HdcError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        let n = x.len();
        let mut c = vec![0f32; n];
        for (k, slot) in c.iter_mut().enumerate() {
            let mut acc = 0f64;
            for (j, &xj) in x.iter().enumerate() {
                let bk = (k + n - j) % n;
                acc += (xj as f64) * (y[bk] as f64);
            }
            *slot = acc as f32;
        }
        Ok(c)
    }

    /// Reject pairs that are empty, mismatched, or non-power-of-two.
    fn validate_pair(x: &[f32], y: &[f32]) -> HdcResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if x.len() != y.len() {
            return Err(HdcError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        if !is_power_of_two(x.len()) {
            return Err(HdcError::DimensionMismatch {
                expected: x.len().next_power_of_two(),
                got: x.len(),
            });
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0xFEED_F1F1_DEAD_C0DE)
    }

    /// Sample a length-`n` Gaussian vector via Box-Muller and L2-normalize.
    fn normalized_random(n: usize, r: &mut LcgRng) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            let (a, b) = r.normal_pair_f32();
            v.push(a);
            if v.len() < n {
                v.push(b);
            }
        }
        let norm: f64 = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x = ((*x as f64) / norm) as f32;
            }
        }
        v
    }

    /// Max absolute difference between two equal-length f32 slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    // ── log2 / power-of-two helpers ────────────────────────────────────────

    #[test]
    fn power_of_two_predicate() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(48));
    }

    // ── HrrFftConfig ───────────────────────────────────────────────────────

    #[test]
    fn config_accepts_power_of_two() {
        for &n in &[1usize, 2, 4, 8, 16, 64, 1024] {
            HrrFftConfig { dim: n }.validate().expect("valid");
        }
    }

    #[test]
    fn config_rejects_zero_dim() {
        let res = HrrFftConfig { dim: 0 }.validate();
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn config_rejects_non_power_of_two() {
        let res = HrrFftConfig { dim: 6 }.validate();
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── FFT round-trip ─────────────────────────────────────────────────────

    #[test]
    fn fft_then_ifft_round_trip() {
        let mut r = rng();
        for &n in &[1usize, 2, 4, 8, 16, 64, 128] {
            let x = normalized_random(n, &mut r);
            let mut re = x.clone();
            let mut im = vec![0f32; n];
            HrrFft::fft(&mut re, &mut im, false).expect("forward");
            HrrFft::fft(&mut re, &mut im, true).expect("inverse");
            let diff = max_abs_diff(&x, &re);
            assert!(diff < 1e-4, "n={n} round-trip diff {diff}");
            let im_diff = im.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
            assert!(im_diff < 1e-4, "n={n} residual imaginary {im_diff}");
        }
    }

    #[test]
    fn fft_of_delta_has_constant_magnitude() {
        let n = 16usize;
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        re[0] = 1.0;
        HrrFft::fft(&mut re, &mut im, false).expect("fft");
        for k in 0..n {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "k={k} mag={mag}");
        }
    }

    // ── FFT vs naive convolution ───────────────────────────────────────────

    #[test]
    fn fft_conv_matches_naive_random() {
        let mut r = rng();
        for &n in &[2usize, 4, 8, 16, 64, 128, 256] {
            let x = normalized_random(n, &mut r);
            let y = normalized_random(n, &mut r);
            let c_fft = HrrFft::fft_circular_convolve(&x, &y).expect("fft conv");
            let c_naive = HrrFft::naive_circular_convolve(&x, &y).expect("naive conv");
            let diff = max_abs_diff(&c_fft, &c_naive);
            assert!(diff < 1e-4, "n={n} diff {diff}");
        }
    }

    #[test]
    fn fft_conv_matches_existing_naive_in_ops() {
        // Cross-check against the in-tree O(n²) implementation
        // crate::ops::binding::circular_convolution.
        let mut r = rng();
        for &n in &[4usize, 8, 16, 32, 64] {
            let x = normalized_random(n, &mut r);
            let y = normalized_random(n, &mut r);
            let c_fft = HrrFft::fft_circular_convolve(&x, &y).expect("fft conv");
            let c_ref = crate::ops::binding::circular_convolution(&x, &y).expect("ref conv");
            let diff = max_abs_diff(&c_fft, &c_ref);
            assert!(diff < 1e-4, "n={n} diff {diff}");
        }
    }

    #[test]
    fn fft_conv_with_delta_identity() {
        // δ = [1, 0, 0, ...] is the identity element of circular convolution.
        let mut r = rng();
        let n = 32usize;
        let x = normalized_random(n, &mut r);
        let mut delta = vec![0f32; n];
        delta[0] = 1.0;
        let c = HrrFft::fft_circular_convolve(&x, &delta).expect("conv");
        let diff = max_abs_diff(&c, &x);
        assert!(diff < 1e-4, "conv(x, δ) ≠ x; diff {diff}");
    }

    #[test]
    fn fft_conv_is_commutative() {
        let mut r = rng();
        let n = 64usize;
        let x = normalized_random(n, &mut r);
        let y = normalized_random(n, &mut r);
        let xy = HrrFft::fft_circular_convolve(&x, &y).expect("xy");
        let yx = HrrFft::fft_circular_convolve(&y, &x).expect("yx");
        let diff = max_abs_diff(&xy, &yx);
        assert!(diff < 1e-4, "commutativity diff {diff}");
    }

    #[test]
    fn fft_conv_distributes_over_addition() {
        // Linearity: conv(x, y + z) = conv(x, y) + conv(x, z).
        let mut r = rng();
        let n = 64usize;
        let x = normalized_random(n, &mut r);
        let y = normalized_random(n, &mut r);
        let z = normalized_random(n, &mut r);
        let y_plus_z: Vec<f32> = y.iter().zip(z.iter()).map(|(&a, &b)| a + b).collect();
        let lhs = HrrFft::fft_circular_convolve(&x, &y_plus_z).expect("lhs");
        let xy = HrrFft::fft_circular_convolve(&x, &y).expect("xy");
        let xz = HrrFft::fft_circular_convolve(&x, &z).expect("xz");
        let rhs: Vec<f32> = xy.iter().zip(xz.iter()).map(|(&a, &b)| a + b).collect();
        let diff = max_abs_diff(&lhs, &rhs);
        assert!(diff < 1e-4, "linearity diff {diff}");
    }

    #[test]
    fn fft_conv_length_equals_input_length() {
        let n = 16usize;
        let x = vec![1.0f32; n];
        let y = vec![1.0f32; n];
        let c = HrrFft::fft_circular_convolve(&x, &y).expect("conv");
        assert_eq!(c.len(), n);
    }

    // ── Correlation ────────────────────────────────────────────────────────

    #[test]
    fn fft_corr_self_peaks_at_zero_lag() {
        let mut r = rng();
        let n = 64usize;
        let x = normalized_random(n, &mut r);
        let corr = HrrFft::fft_circular_correlate(&x, &x).expect("corr");
        let mut max_idx = 0usize;
        let mut max_val = corr[0];
        for (i, &v) in corr.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        assert_eq!(max_idx, 0, "self-correlation peaks at lag {max_idx}");
    }

    #[test]
    fn fft_correlate_matches_naive_via_flip() {
        // Cross-check against the naive correlation in crate::ops::binding.
        let mut r = rng();
        for &n in &[4usize, 8, 16, 32, 64] {
            let x = normalized_random(n, &mut r);
            let y = normalized_random(n, &mut r);
            let c_fft = HrrFft::fft_circular_correlate(&x, &y).expect("fft corr");
            let c_ref = crate::ops::binding::circular_correlation(&x, &y).expect("ref corr");
            let diff = max_abs_diff(&c_fft, &c_ref);
            assert!(diff < 1e-4, "n={n} corr diff {diff}");
        }
    }

    #[test]
    fn fft_unbind_round_trip_normalized() {
        // Standard HRR approximate-inverse: starting from `bound = conv(x, y)`,
        // we unbind with the cue `y` via correlation `corr(y, bound)`. The
        // returned vector is `x` plus diminishing noise — the canonical
        // retrieval metric is cosine similarity, which is close to 1 for
        // moderate D. A tight elementwise bound is not achievable because
        // HRR retrieval is *approximate* by construction.
        let mut r = rng();
        let n = 1024usize;
        let x = normalized_random(n, &mut r);
        let y = normalized_random(n, &mut r);
        let bound = HrrFft::fft_circular_convolve(&x, &y).expect("conv");
        let retrieved = HrrFft::fft_circular_correlate(&y, &bound).expect("corr");
        // Cosine similarity between retrieved and x.
        let dot: f64 = retrieved
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let norm_r: f64 = retrieved
            .iter()
            .map(|&a| (a as f64) * (a as f64))
            .sum::<f64>()
            .sqrt();
        let norm_x: f64 = x
            .iter()
            .map(|&a| (a as f64) * (a as f64))
            .sum::<f64>()
            .sqrt();
        let cosine = dot / (norm_r * norm_x);
        assert!(cosine > 0.5, "HRR unbind cosine {cosine} too low");
    }

    // ── Single-element vectors ─────────────────────────────────────────────

    #[test]
    fn fft_conv_single_element_vectors() {
        // n = 1: circular convolution reduces to scalar multiplication.
        let x = vec![3.0f32];
        let y = vec![5.0f32];
        let c = HrrFft::fft_circular_convolve(&x, &y).expect("conv");
        assert_eq!(c.len(), 1);
        assert!((c[0] - 15.0).abs() < 1e-5, "c[0]={}", c[0]);
        let corr = HrrFft::fft_circular_correlate(&x, &y).expect("corr");
        assert!((corr[0] - 15.0).abs() < 1e-5, "corr[0]={}", corr[0]);
    }

    // ── Errors ─────────────────────────────────────────────────────────────

    #[test]
    fn err_fft_empty_input() {
        let mut re: Vec<f32> = Vec::new();
        let mut im: Vec<f32> = Vec::new();
        let res = HrrFft::fft(&mut re, &mut im, false);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn err_fft_re_im_length_mismatch() {
        let mut re = vec![0f32; 4];
        let mut im = vec![0f32; 8];
        let res = HrrFft::fft(&mut re, &mut im, false);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_fft_non_power_of_two_length() {
        let mut re = vec![0f32; 6];
        let mut im = vec![0f32; 6];
        let res = HrrFft::fft(&mut re, &mut im, false);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_conv_empty_input() {
        let x: Vec<f32> = Vec::new();
        let y: Vec<f32> = Vec::new();
        let res = HrrFft::fft_circular_convolve(&x, &y);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn err_conv_length_mismatch() {
        let x = vec![0f32; 4];
        let y = vec![0f32; 8];
        let res = HrrFft::fft_circular_convolve(&x, &y);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_conv_non_power_of_two_length() {
        let x = vec![0f32; 6];
        let y = vec![0f32; 6];
        let res = HrrFft::fft_circular_convolve(&x, &y);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_corr_non_power_of_two_length() {
        let x = vec![0f32; 10];
        let y = vec![0f32; 10];
        let res = HrrFft::fft_circular_correlate(&x, &y);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_naive_conv_empty_and_mismatch() {
        let x: Vec<f32> = Vec::new();
        let y: Vec<f32> = Vec::new();
        assert!(matches!(
            HrrFft::naive_circular_convolve(&x, &y),
            Err(HdcError::EmptyInput)
        ));
        let x = vec![0f32; 4];
        let y = vec![0f32; 8];
        assert!(matches!(
            HrrFft::naive_circular_convolve(&x, &y),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn deterministic_repeated_calls() {
        let mut r = rng();
        let n = 64usize;
        let x = normalized_random(n, &mut r);
        let y = normalized_random(n, &mut r);
        let a = HrrFft::fft_circular_convolve(&x, &y).expect("a");
        let b = HrrFft::fft_circular_convolve(&x, &y).expect("b");
        assert_eq!(a, b);
        let c1 = HrrFft::fft_circular_correlate(&x, &y).expect("c1");
        let c2 = HrrFft::fft_circular_correlate(&x, &y).expect("c2");
        assert_eq!(c1, c2);
    }
}
