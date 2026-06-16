//! Number Theoretic Transform (NTT) over Z_p.
//!
//! Implements Cooley-Tukey FFT using modular arithmetic over the prime field
//! Z_{NTT_MOD}.  The chosen prime `p = 998_244_353 = 119 × 2^23 + 1` is
//! NTT-friendly: it supports power-of-two transform sizes up to 2^23, and its
//! primitive root is 3.
//!
//! # References
//! - Cooley & Tukey (1965) — iterative FFT algorithm.
//! - Crandall & Pomerance (2005) — modular arithmetic for polynomial
//!   multiplication.

use crate::error::{FftError, FftResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NTT prime modulus: `p = 119 × 2^23 + 1 = 998_244_353`.
///
/// This prime is chosen because `p - 1 = 119 × 2^23`, which means the
/// multiplicative group Z_p* contains a subgroup of order 2^k for any k ≤ 23.
/// This is the fundamental requirement for an NTT of power-of-2 length.
pub const NTT_MOD: u64 = 998_244_353;

/// Primitive root of `NTT_MOD`.
///
/// `g = 3` is a primitive root of Z_{998_244_353}^*: its multiplicative order
/// is exactly `p - 1 = 998_244_352`.
pub const NTT_PRIMITIVE_ROOT: u64 = 3;

// ---------------------------------------------------------------------------
// Modular arithmetic helpers
// ---------------------------------------------------------------------------

/// Fast modular exponentiation: computes `base^exp mod modulus`.
///
/// Uses binary (right-to-left) exponentiation in O(log exp) multiplications.
/// Intermediate products use `u128` to avoid 64-bit overflow.
///
/// # Panics
///
/// Does not panic. Returns 1 when `exp == 0` (by convention, `a^0 = 1` for
/// all `a`, including `a = 0`).
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}

/// Modular inverse of `a` modulo a prime `modulus`.
///
/// Uses Fermat's little theorem: `a^{-1} ≡ a^{p-2} (mod p)` when `p` is
/// prime.  Only valid for `a != 0 mod p`.
///
/// # Examples
///
/// ```
/// use oxicuda_fft::transforms::ntt::{mod_inv, NTT_MOD};
/// let inv3 = mod_inv(3, NTT_MOD);
/// assert_eq!((3_u128 * inv3 as u128) as u64 % NTT_MOD, 1);
/// ```
pub fn mod_inv(a: u64, modulus: u64) -> u64 {
    mod_pow(a, modulus - 2, modulus)
}

// ---------------------------------------------------------------------------
// Core NTT
// ---------------------------------------------------------------------------

/// In-place iterative Cooley-Tukey NTT over Z_{`NTT_MOD`}.
///
/// `a.len()` must be a power of 2 and at most 2^23.
///
/// When `invert = false`, computes the forward NTT:
/// ```text
/// A[k] = Σ_{j=0}^{n-1}  a[j] · ω^{jk}   where ω is a primitive n-th root of unity mod p
/// ```
/// When `invert = true`, computes the inverse NTT (INTT) and multiplies the
/// result by `n^{-1} mod p` so that INTT(NTT(a)) == a.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] when `a.len()` is not a power of two, is
/// zero, or exceeds 2^23.
pub fn ntt(a: &mut [u64], invert: bool) -> FftResult<()> {
    let n = a.len();
    if n == 0 {
        return Err(FftError::InvalidSize(
            "NTT input must not be empty".to_string(),
        ));
    }
    if !n.is_power_of_two() {
        return Err(FftError::InvalidSize(format!(
            "NTT length must be a power of 2, got {n}"
        )));
    }
    // Maximum NTT length for NTT_MOD is 2^23.
    if n > (1 << 23) {
        return Err(FftError::InvalidSize(format!(
            "NTT length {n} exceeds maximum 2^23 for NTT_MOD = {NTT_MOD}"
        )));
    }

    // ---- bit-reversal permutation ----------------------------------------
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_rev(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }

    // ---- Cooley-Tukey butterfly layers -----------------------------------
    // For each power-of-2 length `len` from 2 to n:
    //   primitive `len`-th root of unity = g^{(p-1)/len}  (or its inverse)
    let mut len = 2_usize;
    while len <= n {
        // ω = primitive len-th root of unity in Z_p
        let half_len = len >> 1;
        let w: u64 = if invert {
            // For INTT use ω^{-1} = g^{(p-1) - (p-1)/len}
            let fwd = mod_pow(NTT_PRIMITIVE_ROOT, (NTT_MOD - 1) / len as u64, NTT_MOD);
            mod_inv(fwd, NTT_MOD)
        } else {
            mod_pow(NTT_PRIMITIVE_ROOT, (NTT_MOD - 1) / len as u64, NTT_MOD)
        };

        let mut chunk_start = 0;
        while chunk_start < n {
            let mut wn: u64 = 1; // twiddle factor ω^j
            for j in 0..half_len {
                let u = a[chunk_start + j];
                let v =
                    ((a[chunk_start + j + half_len] as u128 * wn as u128) % NTT_MOD as u128) as u64;
                a[chunk_start + j] = (u + v) % NTT_MOD;
                a[chunk_start + j + half_len] = (u + NTT_MOD - v) % NTT_MOD;
                wn = ((wn as u128 * w as u128) % NTT_MOD as u128) as u64;
            }
            chunk_start += len;
        }
        len <<= 1;
    }

    // ---- INTT scaling: multiply by n^{-1} mod p -------------------------
    if invert {
        let n_inv = mod_inv(n as u64, NTT_MOD);
        for x in a.iter_mut() {
            *x = ((*x as u128 * n_inv as u128) % NTT_MOD as u128) as u64;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Polynomial multiplication helpers
// ---------------------------------------------------------------------------

/// Multiply two polynomials modulo `NTT_MOD` using NTT convolution.
///
/// Given coefficient vectors `a` and `b`, returns the coefficient vector of
/// their product (as a polynomial over Z_{`NTT_MOD`}).  The result has length
/// `a.len() + b.len() - 1`.  Both inputs and output are in Z_{`NTT_MOD`}.
///
/// # Errors
///
/// Propagates errors from [`ntt`] (invalid sizes after zero-padding).
pub fn ntt_multiply(a: &[u64], b: &[u64]) -> FftResult<Vec<u64>> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    let result_len = a.len() + b.len() - 1;
    // Next power of 2 large enough for the result
    let padded = next_pow2(result_len);

    let mut fa: Vec<u64> = a.to_vec();
    fa.resize(padded, 0);
    let mut fb: Vec<u64> = b.to_vec();
    fb.resize(padded, 0);

    ntt(&mut fa, false)?;
    ntt(&mut fb, false)?;

    // Point-wise multiplication mod p
    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = ((*x as u128 * *y as u128) % NTT_MOD as u128) as u64;
    }

    ntt(&mut fa, true)?;
    fa.truncate(result_len);
    Ok(fa)
}

/// Integer polynomial multiplication using NTT convolution.
///
/// Multiplies two polynomials with `i64` coefficients.  Internally reduces
/// everything to Z_{`NTT_MOD`}, performs the NTT convolution, then lifts the
/// result back to signed integers.  Valid as long as coefficient magnitudes in
/// the product stay below `NTT_MOD / 2 ≈ 499M`.
///
/// # Errors
///
/// Propagates errors from [`ntt_multiply`].
pub fn ntt_convolve(a: &[i64], b: &[i64]) -> FftResult<Vec<i64>> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    let p = NTT_MOD as i64;
    let au: Vec<u64> = a.iter().map(|&x| ((x % p + p) as u64) % NTT_MOD).collect();
    let bu: Vec<u64> = b.iter().map(|&x| ((x % p + p) as u64) % NTT_MOD).collect();

    let cu = ntt_multiply(&au, &bu)?;

    // Lift back: values > p/2 are negative in symmetric representation
    let half = NTT_MOD / 2;
    Ok(cu
        .into_iter()
        .map(|x| {
            if x > half {
                x as i64 - NTT_MOD as i64
            } else {
                x as i64
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// Bit-reverse an index of `bits` bits.
#[inline]
fn bit_rev(mut x: usize, bits: usize) -> usize {
    let mut result = 0_usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Smallest power of two >= `n`.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: perform a naive DFT over Z_p for cross-checking.
    fn naive_ntt(a: &[u64]) -> Vec<u64> {
        let n = a.len() as u64;
        let w = mod_pow(NTT_PRIMITIVE_ROOT, (NTT_MOD - 1) / n, NTT_MOD);
        (0..a.len())
            .map(|k| {
                a.iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        let exp = (k as u64 * j as u64) % (n);
                        ((x as u128 * mod_pow(w, exp, NTT_MOD) as u128) % NTT_MOD as u128) as u64
                    })
                    .fold(0_u64, |acc, v| (acc + v) % NTT_MOD)
            })
            .collect()
    }

    #[test]
    fn mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, NTT_MOD), 1024);
        assert_eq!(mod_pow(0, 5, 7), 0);
        assert_eq!(mod_pow(5, 0, 7), 1);
        assert_eq!(mod_pow(3, NTT_MOD - 2, NTT_MOD), mod_inv(3, NTT_MOD));
    }

    #[test]
    fn mod_inv_basic() {
        for a in [2_u64, 3, 7, 998244352] {
            let inv = mod_inv(a, NTT_MOD);
            let check = ((a as u128 * inv as u128) % NTT_MOD as u128) as u64;
            assert_eq!(check, 1, "a={a} inv={inv}");
        }
    }

    #[test]
    fn ntt_unit_impulse() {
        // NTT of [1, 0, 0, ..., 0] should be [1, 1, 1, ..., 1]
        let mut a = vec![0u64; 8];
        a[0] = 1;
        ntt(&mut a, false).expect("ntt");
        assert!(a.iter().all(|&x| x == 1), "expected all ones: {a:?}");
    }

    #[test]
    fn ntt_and_intt_roundtrip() {
        let original = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut a = original.clone();
        ntt(&mut a, false).expect("forward ntt");
        ntt(&mut a, true).expect("inverse ntt");
        assert_eq!(a, original, "INTT(NTT(a)) should equal a");
    }

    #[test]
    fn ntt_length_not_pow2_returns_error() {
        let mut a = vec![1u64; 6];
        let result = ntt(&mut a, false);
        assert!(result.is_err(), "NTT of non-power-of-2 length should fail");
    }

    #[test]
    fn ntt_empty_returns_error() {
        let mut a: Vec<u64> = vec![];
        let result = ntt(&mut a, false);
        assert!(result.is_err(), "NTT of empty slice should fail");
    }

    #[test]
    fn ntt_matches_naive_dft() {
        let a = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let expected = naive_ntt(&a);
        let mut got = a.clone();
        ntt(&mut got, false).expect("ntt");
        assert_eq!(got, expected, "NTT should match naive DFT");
    }

    #[test]
    fn ntt_multiply_simple() {
        // [1,1] * [1,1] = [1,2,1]
        let a = vec![1u64, 1];
        let b = vec![1u64, 1];
        let result = ntt_multiply(&a, &b).expect("ntt_multiply");
        assert_eq!(result, vec![1, 2, 1]);
    }

    #[test]
    fn ntt_multiply_length() {
        let a = vec![1u64; 4];
        let b = vec![1u64; 6];
        let result = ntt_multiply(&a, &b).expect("ntt_multiply");
        assert_eq!(
            result.len(),
            a.len() + b.len() - 1,
            "output len == a.len() + b.len() - 1"
        );
    }

    #[test]
    fn ntt_multiply_associative() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5];
        let c = vec![2u64, 1];
        let ab = ntt_multiply(&a, &b).expect("ab");
        let abc = ntt_multiply(&ab, &c).expect("abc");
        let bc = ntt_multiply(&b, &c).expect("bc");
        let a_bc = ntt_multiply(&a, &bc).expect("a_bc");
        assert_eq!(abc, a_bc, "(a*b)*c == a*(b*c)");
    }

    #[test]
    fn ntt_convolve_integers() {
        // [1,2,3] * [4,5] = [4, 13, 22, 15]
        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5];
        let result = ntt_convolve(&a, &b).expect("ntt_convolve");
        assert_eq!(result, vec![4, 13, 22, 15]);
    }

    #[test]
    fn ntt_convolve_empty_a_returns_empty() {
        let a: Vec<i64> = vec![];
        let b = vec![1i64, 2, 3];
        let result = ntt_convolve(&a, &b).expect("ntt_convolve");
        assert!(result.is_empty(), "empty * anything = empty");
    }

    #[test]
    fn ntt_is_linear() {
        // NTT(a+b) == NTT(a) + NTT(b) (mod p)
        let a = vec![1u64, 2, 3, 4];
        let b = vec![5u64, 6, 7, 8];
        let apb: Vec<u64> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x + y) % NTT_MOD)
            .collect();

        let mut ntt_apb = apb.clone();
        ntt(&mut ntt_apb, false).expect("ntt");

        let mut ntt_a = a.clone();
        ntt(&mut ntt_a, false).expect("ntt");
        let mut ntt_b = b.clone();
        ntt(&mut ntt_b, false).expect("ntt");

        let sum: Vec<u64> = ntt_a
            .iter()
            .zip(ntt_b.iter())
            .map(|(&x, &y)| (x + y) % NTT_MOD)
            .collect();
        assert_eq!(ntt_apb, sum, "NTT is linear");
    }

    #[test]
    fn ntt_multiply_identity() {
        // a * [1] == a
        let a = vec![3u64, 7, 11, 5];
        let one = vec![1u64];
        let result = ntt_multiply(&a, &one).expect("ntt_multiply");
        assert_eq!(result, a, "a * [1] should equal a");
    }
}
