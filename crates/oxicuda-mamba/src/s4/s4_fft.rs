//! FFT-based S4 long convolution (`O(L log L)`).
//!
//! S4 (Gu et al. 2022, "Efficiently Modeling Long Sequences with Structured
//! State Spaces") computes the sequence output as a length-`L` causal
//! convolution with a structured SSM kernel.  The reference
//! [`crate::s4::s4_layer::naive_conv1d`] performs this directly in `O(L²)`.
//!
//! For long sequences this is prohibitive, so the same convolution is computed
//! in `O(L log L)` via the convolution theorem:
//!
//! ```text
//! conv(x, k) = IFFT( FFT(x) ⊙ FFT(k) )
//! ```
//!
//! after zero-padding both signals to a common length `m ≥ |x| + |k| − 1`
//! that is a power of two (so the radix-2 Cooley-Tukey FFT applies and the
//! circular convolution coincides with the linear convolution).
//!
//! ## Module contents
//!
//! - [`fft`] — in-place iterative radix-2 Cooley-Tukey FFT / IFFT on split
//!   real / imaginary buffers.
//! - [`fft_conv1d`] — full **linear** convolution of `x` with `kernel`
//!   (length `|x| + |k| − 1`); numerically equal to `naive_conv1d` would be
//!   if extended to the full support.
//! - [`s4_fft_conv`] — **causal** SSM convolution: the linear convolution
//!   truncated to the first `|u|` entries, exactly matching
//!   [`crate::s4::s4_layer::naive_conv1d`] for the SSM use-case.

use crate::error::{MambaError, MambaResult};

// ─── helpers ───────────────────────────────────────────────────────────────────

/// Return the smallest power of two `≥ n` (with `next_pow2(0) == 1`).
#[inline]
fn next_pow2(n: usize) -> usize {
    let mut m = 1_usize;
    while m < n {
        // `usize` cannot overflow here for any realistic convolution length;
        // saturating guards against pathological inputs without panicking.
        m = m.saturating_mul(2);
    }
    m
}

/// `true` iff `n` is a non-zero power of two.
#[inline]
fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

// ─── fft ───────────────────────────────────────────────────────────────────────

/// In-place iterative radix-2 Cooley-Tukey FFT on split complex data.
///
/// `re` and `im` hold the real and imaginary parts of the same complex signal
/// and **must** have equal length, which must be a non-zero power of two.
/// When `inverse` is `true` the inverse transform is computed and the result
/// is scaled by `1/n`.
///
/// The algorithm performs a bit-reversal permutation followed by `log₂ n`
/// stages of butterfly combinations; twiddle factors are evaluated with
/// `cos`/`sin`.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`] — if `re` is empty.
/// * [`MambaError::ShapeMismatch`] — if `re.len() != im.len()`.
/// * [`MambaError::InvalidChunkSize`] — if the length is not a power of two.
pub fn fft(re: &mut [f32], im: &mut [f32], inverse: bool) -> MambaResult<()> {
    let n = re.len();
    if n == 0 {
        return Err(MambaError::EmptyInput("fft input"));
    }
    if im.len() != n {
        return Err(MambaError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![im.len()],
        });
    }
    if !is_pow2(n) {
        return Err(MambaError::InvalidChunkSize(n));
    }

    // ── Bit-reversal permutation ─────────────────────────────────────────────
    // Reorder samples so that the in-place butterflies access contiguous pairs.
    let mut j = 0_usize;
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

    // ── Butterfly stages ─────────────────────────────────────────────────────
    // Sign convention: forward transform uses e^{-2πi k / len}; the inverse
    // flips the sign of the imaginary twiddle component.
    let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f32::consts::PI / (len as f32);
        let w_re = theta.cos();
        let w_im = theta.sin();
        let mut start = 0_usize;
        while start < n {
            // Running twiddle factor w^0, w^1, … updated multiplicatively.
            let mut cur_re = 1.0_f32;
            let mut cur_im = 0.0_f32;
            for k in 0..half {
                let a = start + k;
                let b = start + k + half;
                let t_re = cur_re * re[b] - cur_im * im[b];
                let t_im = cur_re * im[b] + cur_im * re[b];
                let u_re = re[a];
                let u_im = im[a];
                re[a] = u_re + t_re;
                im[a] = u_im + t_im;
                re[b] = u_re - t_re;
                im[b] = u_im - t_im;
                // Advance the twiddle: cur *= w.
                let next_re = cur_re * w_re - cur_im * w_im;
                let next_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
                cur_im = next_im;
            }
            start += len;
        }
        len <<= 1;
    }

    // ── Inverse normalization ────────────────────────────────────────────────
    if inverse {
        let inv_n = 1.0_f32 / (n as f32);
        for x in re.iter_mut() {
            *x *= inv_n;
        }
        for x in im.iter_mut() {
            *x *= inv_n;
        }
    }

    Ok(())
}

// ─── fft_conv1d ──────────────────────────────────────────────────────────────────

/// Full **linear** convolution of `x` with `kernel` via zero-padded FFT.
///
/// The output has length `x.len() + kernel.len() − 1`.  Both inputs are
/// zero-padded into complex buffers of length `m = next_pow2(|x| + |k| − 1)`,
/// transformed, multiplied point-wise, inverse-transformed, and the real part
/// is returned (truncated to the linear-convolution support).
///
/// This is the FFT analogue of repeatedly applying
/// [`crate::s4::s4_layer::naive_conv1d`] over the full support: for matching
/// inputs the two agree to floating-point round-off.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`] — if `x` or `kernel` is empty.
/// * Propagates [`fft`] errors.
pub fn fft_conv1d(x: &[f32], kernel: &[f32]) -> MambaResult<Vec<f32>> {
    if x.is_empty() {
        return Err(MambaError::EmptyInput("fft_conv1d x"));
    }
    if kernel.is_empty() {
        return Err(MambaError::EmptyInput("fft_conv1d kernel"));
    }

    let out_len = x.len() + kernel.len() - 1;
    let m = next_pow2(out_len);

    // Zero-padded complex buffers.
    let mut x_re = vec![0.0_f32; m];
    let mut x_im = vec![0.0_f32; m];
    let mut k_re = vec![0.0_f32; m];
    let mut k_im = vec![0.0_f32; m];
    x_re[..x.len()].copy_from_slice(x);
    k_re[..kernel.len()].copy_from_slice(kernel);

    // Forward transforms.
    fft(&mut x_re, &mut x_im, false)?;
    fft(&mut k_re, &mut k_im, false)?;

    // Point-wise complex multiplication: X ⊙ K.
    for i in 0..m {
        let pr = x_re[i] * k_re[i] - x_im[i] * k_im[i];
        let pi = x_re[i] * k_im[i] + x_im[i] * k_re[i];
        x_re[i] = pr;
        x_im[i] = pi;
    }

    // Inverse transform → time domain; take the real part.
    fft(&mut x_re, &mut x_im, true)?;

    Ok(x_re[..out_len].to_vec())
}

// ─── s4_fft_conv ─────────────────────────────────────────────────────────────────

/// Causal SSM convolution `y[t] = Σ_{j≤t} kernel[j] · u[t − j]`, length `|u|`.
///
/// Computes the full linear convolution via [`fft_conv1d`] and truncates it to
/// the first `u.len()` entries, which is precisely the causal output produced
/// by [`crate::s4::s4_layer::naive_conv1d`] — but in `O(L log L)`.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`] — if `u` or `kernel` is empty.
/// * Propagates [`fft`] / [`fft_conv1d`] errors.
pub fn s4_fft_conv(u: &[f32], kernel: &[f32]) -> MambaResult<Vec<f32>> {
    if u.is_empty() {
        return Err(MambaError::EmptyInput("s4_fft_conv u"));
    }
    if kernel.is_empty() {
        return Err(MambaError::EmptyInput("s4_fft_conv kernel"));
    }
    let full = fft_conv1d(u, kernel)?;
    // Causal truncation to the input length L.
    Ok(full[..u.len()].to_vec())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s4::s4_layer::naive_conv1d;

    const EPS: f32 = 1e-4;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= eps)
    }

    // ── fft round-trip ─────────────────────────────────────────────────────────

    /// FFT followed by IFFT recovers the original real signal (±1e-4).
    #[test]
    fn fft_ifft_round_trip() {
        let orig = [1.0_f32, -2.0, 3.5, 0.25, -1.0, 4.0, 0.0, -0.5];
        let mut re = orig.to_vec();
        let mut im = vec![0.0_f32; orig.len()];
        fft(&mut re, &mut im, false).expect("forward");
        fft(&mut re, &mut im, true).expect("inverse");
        for (i, (&r, &o)) in re.iter().zip(orig.iter()).enumerate() {
            assert!((r - o).abs() < EPS, "re[{i}]={r} expected {o}");
            assert!(im[i].abs() < EPS, "im[{i}]={} expected 0", im[i]);
        }
    }

    /// Round-trip works for a longer power-of-two length.
    #[test]
    fn fft_ifft_round_trip_len16() {
        let orig: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut re = orig.clone();
        let mut im = vec![0.0_f32; 16];
        fft(&mut re, &mut im, false).expect("forward");
        fft(&mut re, &mut im, true).expect("inverse");
        assert!(approx_eq(&re, &orig, EPS), "round-trip mismatch");
    }

    /// FFT of a delta [1,0,0,0] has all-equal (unit) magnitude across bins.
    #[test]
    fn fft_delta_flat_magnitude() {
        let mut re = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f32; 4];
        fft(&mut re, &mut im, false).expect("forward");
        for i in 0..4 {
            let mag = (re[i] * re[i] + im[i] * im[i]).sqrt();
            assert!((mag - 1.0).abs() < EPS, "bin {i} magnitude {mag} != 1");
        }
    }

    /// A shifted delta has flat magnitude too (linear-phase property).
    #[test]
    fn fft_shifted_delta_flat_magnitude() {
        let mut re = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut im = vec![0.0_f32; 8];
        fft(&mut re, &mut im, false).expect("forward");
        for i in 0..8 {
            let mag = (re[i] * re[i] + im[i] * im[i]).sqrt();
            assert!((mag - 1.0).abs() < EPS, "bin {i} magnitude {mag} != 1");
        }
    }

    // ── fft_conv1d == naive_conv1d (the correctness anchor) ─────────────────────

    /// fft_conv1d matches naive_conv1d over its first L entries (small case).
    #[test]
    fn fft_conv_matches_naive_small() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let kernel = [1.0_f32, 0.5];
        let full = fft_conv1d(&x, &kernel).expect("fft conv");
        let naive = naive_conv1d(&x, &kernel);
        // naive_conv1d returns length L; compare the overlapping causal prefix.
        for t in 0..x.len() {
            assert!(
                (full[t] - naive[t]).abs() < EPS,
                "t={t}: fft={} naive={}",
                full[t],
                naive[t]
            );
        }
    }

    /// fft_conv1d matches naive_conv1d for a longer 3-tap kernel.
    #[test]
    fn fft_conv_matches_naive_three_tap() {
        let x = [0.5_f32, -1.0, 2.0, 0.0, 3.5, -2.5, 1.0];
        let kernel = [1.0_f32, -0.5, 0.25];
        let full = fft_conv1d(&x, &kernel).expect("fft conv");
        let naive = naive_conv1d(&x, &kernel);
        for t in 0..x.len() {
            assert!(
                (full[t] - naive[t]).abs() < EPS,
                "t={t}: fft={} naive={}",
                full[t],
                naive[t]
            );
        }
    }

    /// fft_conv1d matches naive over multiple deterministic length pairs.
    #[test]
    fn fft_conv_matches_naive_many_lengths() {
        let xs: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            &[2.5, 0.0, -3.0, 4.0, 1.0, -2.0, 0.5, 0.25, -0.75],
        ];
        let kernels: &[&[f32]] = &[
            &[1.0],
            &[1.0, 0.5],
            &[0.3, -0.2, 0.1, 0.05],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for x in xs {
            for k in kernels {
                let full = fft_conv1d(x, k).expect("fft conv");
                let naive = naive_conv1d(x, k);
                for t in 0..x.len() {
                    assert!(
                        (full[t] - naive[t]).abs() < EPS,
                        "x.len={} k.len={} t={t}: fft={} naive={}",
                        x.len(),
                        k.len(),
                        full[t],
                        naive[t]
                    );
                }
            }
        }
    }

    /// fft_conv1d output length == x.len() + kernel.len() - 1.
    #[test]
    fn fft_conv_output_length() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = [1.0_f32, 2.0, 3.0];
        let full = fft_conv1d(&x, &kernel).expect("fft conv");
        assert_eq!(full.len(), x.len() + kernel.len() - 1);
    }

    /// Full linear conv matches a hand-computed reference (tail included).
    #[test]
    fn fft_conv_full_support_reference() {
        // x = [1,2,3], k = [1,1] → full = [1, 3, 5, 3]
        let x = [1.0_f32, 2.0, 3.0];
        let kernel = [1.0_f32, 1.0];
        let full = fft_conv1d(&x, &kernel).expect("fft conv");
        let expected = [1.0_f32, 3.0, 5.0, 3.0];
        assert!(approx_eq(&full, &expected, EPS), "got {full:?}");
    }

    // ── s4_fft_conv ────────────────────────────────────────────────────────────

    /// s4_fft_conv output length == u.len() (causal truncation).
    #[test]
    fn s4_fft_conv_output_length() {
        let u = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let kernel = [1.0_f32, 0.5, 0.25];
        let y = s4_fft_conv(&u, &kernel).expect("s4 fft conv");
        assert_eq!(y.len(), u.len());
    }

    /// s4_fft_conv with identity kernel [1] returns the input unchanged.
    #[test]
    fn s4_fft_conv_identity_kernel() {
        let u = [3.0_f32, -1.0, 2.5, 0.0, 4.0];
        let kernel = [1.0_f32];
        let y = s4_fft_conv(&u, &kernel).expect("s4 fft conv");
        assert!(
            approx_eq(&y, &u, EPS),
            "identity kernel changed input: {y:?}"
        );
    }

    /// s4_fft_conv with a zero kernel returns zeros.
    #[test]
    fn s4_fft_conv_zero_kernel() {
        let u = [1.0_f32, 2.0, 3.0, 4.0];
        let kernel = [0.0_f32, 0.0, 0.0];
        let y = s4_fft_conv(&u, &kernel).expect("s4 fft conv");
        assert!(
            y.iter().all(|&v| v.abs() < EPS),
            "expected zeros, got {y:?}"
        );
    }

    /// s4_fft_conv equals the causal naive_conv1d exactly (±1e-4).
    #[test]
    fn s4_fft_conv_matches_naive() {
        let u = [1.0_f32, -2.0, 3.0, 0.5, -1.5, 2.0, 4.0, -3.0];
        let kernel = [0.7_f32, -0.3, 0.2, 0.1];
        let y = s4_fft_conv(&u, &kernel).expect("s4 fft conv");
        let naive = naive_conv1d(&u, &kernel);
        assert!(approx_eq(&y, &naive, EPS), "fft={y:?} naive={naive:?}");
    }

    // ── algebraic properties ───────────────────────────────────────────────────

    /// Convolution is linear: conv(x,k1)+conv(x,k2) == conv(x,k1+k2).
    #[test]
    fn fft_conv_linearity() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let k1 = [1.0_f32, 0.5, 0.25];
        let k2 = [0.1_f32, -0.2, 0.3];
        let sum_k: Vec<f32> = k1.iter().zip(k2.iter()).map(|(a, b)| a + b).collect();
        let c1 = fft_conv1d(&x, &k1).expect("c1");
        let c2 = fft_conv1d(&x, &k2).expect("c2");
        let c_sum = fft_conv1d(&x, &sum_k).expect("c_sum");
        let lhs: Vec<f32> = c1.iter().zip(c2.iter()).map(|(a, b)| a + b).collect();
        assert!(approx_eq(&lhs, &c_sum, EPS), "linearity violated");
    }

    /// Convolution is commutative: fft_conv1d(x,k) == fft_conv1d(k,x).
    #[test]
    fn fft_conv_commutativity() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let k = [0.5_f32, -1.0, 2.0];
        let a = fft_conv1d(&x, &k).expect("a");
        let b = fft_conv1d(&k, &x).expect("b");
        assert!(
            approx_eq(&a, &b, EPS),
            "commutativity violated: {a:?} {b:?}"
        );
    }

    // ── single-element / non-pow2 lengths ──────────────────────────────────────

    /// Single-element inputs convolve to their product.
    #[test]
    fn fft_conv_single_elements() {
        let x = [3.0_f32];
        let k = [4.0_f32];
        let full = fft_conv1d(&x, &k).expect("fft conv");
        assert_eq!(full.len(), 1);
        assert!((full[0] - 12.0).abs() < EPS, "got {}", full[0]);
    }

    /// Non-power-of-two combined length (3 + 3 - 1 = 5) still works via padding.
    #[test]
    fn fft_conv_non_pow2_length() {
        // out_len = 5, padded to 8 internally; result must still match naive.
        let x = [1.0_f32, 2.0, 3.0];
        let k = [4.0_f32, 5.0, 6.0];
        let full = fft_conv1d(&x, &k).expect("fft conv");
        // full = [4, 13, 28, 27, 18]
        let expected = [4.0_f32, 13.0, 28.0, 27.0, 18.0];
        assert!(approx_eq(&full, &expected, EPS), "got {full:?}");
    }

    /// next_pow2 / is_pow2 helper behaviour.
    #[test]
    fn pow2_helpers() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(16), 16);
        assert_eq!(next_pow2(17), 32);
        assert!(is_pow2(1) && is_pow2(2) && is_pow2(1024));
        assert!(!is_pow2(0) && !is_pow2(3) && !is_pow2(6));
    }

    // ── determinism ────────────────────────────────────────────────────────────

    /// Repeated calls are deterministic (no hidden state).
    #[test]
    fn fft_conv_deterministic() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let k = [0.2_f32, 0.3, 0.5];
        let a = fft_conv1d(&x, &k).expect("a");
        let b = fft_conv1d(&x, &k).expect("b");
        assert_eq!(a, b, "fft_conv1d must be deterministic");
    }

    // ── error paths ─────────────────────────────────────────────────────────────

    /// fft on a non-power-of-two length errors.
    #[test]
    fn fft_err_non_pow2() {
        let mut re = vec![1.0_f32, 2.0, 3.0];
        let mut im = vec![0.0_f32; 3];
        let err = fft(&mut re, &mut im, false).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidChunkSize(3)));
    }

    /// fft on an empty buffer errors.
    #[test]
    fn fft_err_empty() {
        let mut re: Vec<f32> = vec![];
        let mut im: Vec<f32> = vec![];
        let err = fft(&mut re, &mut im, false).expect_err("should fail");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    /// fft with mismatched re/im lengths errors.
    #[test]
    fn fft_err_shape_mismatch() {
        let mut re = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut im = vec![0.0_f32; 2];
        let err = fft(&mut re, &mut im, false).expect_err("should fail");
        assert!(matches!(err, MambaError::ShapeMismatch { .. }));
    }

    /// fft_conv1d with an empty x errors.
    #[test]
    fn fft_conv_err_empty_x() {
        let x: Vec<f32> = vec![];
        let k = vec![1.0_f32];
        let err = fft_conv1d(&x, &k).expect_err("should fail");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    /// fft_conv1d with an empty kernel errors.
    #[test]
    fn fft_conv_err_empty_kernel() {
        let x = vec![1.0_f32, 2.0];
        let k: Vec<f32> = vec![];
        let err = fft_conv1d(&x, &k).expect_err("should fail");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    /// s4_fft_conv with empty inputs errors.
    #[test]
    fn s4_fft_conv_err_empty() {
        let u: Vec<f32> = vec![];
        let k = vec![1.0_f32];
        assert!(matches!(
            s4_fft_conv(&u, &k).expect_err("empty u"),
            MambaError::EmptyInput(_)
        ));
        let u2 = vec![1.0_f32];
        let k2: Vec<f32> = vec![];
        assert!(matches!(
            s4_fft_conv(&u2, &k2).expect_err("empty kernel"),
            MambaError::EmptyInput(_)
        ));
    }
}
