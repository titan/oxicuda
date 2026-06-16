//! 4-wise independent hash family and the derived ±1 "tug-of-war" sign family.
//!
//! A family `H` is *`j`-wise independent* if, for any `j` distinct keys and any `j` target
//! values, a uniformly drawn `h ∈ H` maps the keys to those targets with probability
//! `(1/range)^j`. Carter-Wegman tells us that a degree-`(j-1)` polynomial over a prime field,
//! with coefficients drawn uniformly and independently, is exactly `j`-wise independent.
//!
//! Here we build a **degree-3** polynomial over the Mersenne prime field `p = 2^61 − 1`:
//!
//! ```text
//! h(x) = (a3·x³ + a2·x² + a1·x + a0) mod p,   a3 ∈ [1, p-1],  a2,a1,a0 ∈ [0, p-1]
//! ```
//!
//! evaluated by Horner's rule. Drawing the four coefficients independently makes the family
//! **4-wise independent** (Wegman & Carter 1981; see also Thorup & Zhang 2004). The induced
//! sign function
//!
//! ```text
//! s(x) = +1 if (h(x) & 1) == 0 else −1
//! ```
//!
//! inherits 4-wise independence of its bits. That is *precisely* the property the AMS
//! second-moment estimator needs: with 4-wise independent signs, `E[s_i·s_j] = 0` for `i ≠ j`
//! and, crucially, the **fourth-moment** cross terms `E[s_i s_j s_k s_l]` vanish unless the
//! indices pair up, which is what bounds `Var(X²) ≤ 2·F2²` and makes the median-of-means
//! estimator concentrate. A merely 2-universal (degree-1) sign family does **not** kill those
//! fourth-order terms and yields an estimator with unbounded relative variance, so the
//! distinction is real and load-bearing.

use crate::handle::LcgRng;
use crate::hash::twouniv::PRIME_MERSENNE_61;

/// Number of coefficients in a degree-3 polynomial (`a0..a3`).
const DEGREE3_COEFFS: usize = 4;

/// A single 4-wise independent hash function over the field `GF(2^61 − 1)`.
///
/// `coeffs[0]` is the constant term `a0`; `coeffs[3]` is the leading coefficient `a3`.
#[derive(Debug, Clone)]
pub struct FourWiseHash {
    coeffs: [u64; DEGREE3_COEFFS],
}

/// Reduce a 128-bit value modulo the Mersenne prime `p = 2^61 − 1`.
///
/// Uses the identity `2^61 ≡ 1 (mod p)`: fold the high bits down repeatedly. A single product
/// of two residues `< p` is `< 2^122`, so two folds suffice to land in `[0, p)`.
#[inline]
fn mod_mersenne61(mut value: u128) -> u64 {
    let p = PRIME_MERSENNE_61 as u128;
    // First fold: split into low 61 bits + the rest.
    value = (value & p) + (value >> 61);
    // The result can still be slightly above p (up to ~2^62), fold once more.
    value = (value & p) + (value >> 61);
    let mut r = value;
    if r >= p {
        r -= p;
    }
    (r as u64) & PRIME_MERSENNE_61
}

/// Modular multiplication of two residues in `[0, p)` over `GF(2^61 − 1)`.
#[inline]
fn mul_mod(a: u64, b: u64) -> u64 {
    mod_mersenne61((a as u128) * (b as u128))
}

/// Modular addition of two residues in `[0, p)` over `GF(2^61 − 1)`.
#[inline]
fn add_mod(a: u64, b: u64) -> u64 {
    let p = PRIME_MERSENNE_61;
    let s = a + b; // a, b < 2^61 ⇒ sum < 2^62, no u64 overflow.
    if s >= p { s - p } else { s }
}

impl FourWiseHash {
    /// Draw a random 4-wise independent hash: `a3 ∈ [1, p-1]`, `a2,a1,a0 ∈ [0, p-1]`.
    ///
    /// Requiring the leading coefficient to be non-zero keeps the polynomial genuinely degree-3
    /// (a degenerate `a3 = 0` would drop independence to 3-wise).
    #[must_use]
    pub fn new(rng: &mut LcgRng) -> Self {
        let a0 = rng.next_u64() % PRIME_MERSENNE_61;
        let a1 = rng.next_u64() % PRIME_MERSENNE_61;
        let a2 = rng.next_u64() % PRIME_MERSENNE_61;
        let a3 = 1 + rng.next_u64() % (PRIME_MERSENNE_61 - 1);
        Self {
            coeffs: [a0, a1, a2, a3],
        }
    }

    /// Construct from explicit coefficients `[a0, a1, a2, a3]` (each reduced mod `p`; `a3`
    /// forced non-zero). Useful for reproducible tests.
    #[must_use]
    pub fn with_coeffs(coeffs: [u64; DEGREE3_COEFFS]) -> Self {
        let a0 = coeffs[0] % PRIME_MERSENNE_61;
        let a1 = coeffs[1] % PRIME_MERSENNE_61;
        let a2 = coeffs[2] % PRIME_MERSENNE_61;
        let a3 = (coeffs[3] % PRIME_MERSENNE_61).max(1);
        Self {
            coeffs: [a0, a1, a2, a3],
        }
    }

    /// Evaluate `h(x) = (a3·x³ + a2·x² + a1·x + a0) mod p` via Horner's rule.
    #[must_use]
    pub fn hash(&self, x: u64) -> u64 {
        let xr = x % PRIME_MERSENNE_61;
        // Horner: ((a3·x + a2)·x + a1)·x + a0.
        let mut acc = self.coeffs[3];
        acc = add_mod(mul_mod(acc, xr), self.coeffs[2]);
        acc = add_mod(mul_mod(acc, xr), self.coeffs[1]);
        acc = add_mod(mul_mod(acc, xr), self.coeffs[0]);
        acc
    }

    /// The tug-of-war sign of `x`: `+1.0` if the low bit of `h(x)` is 0, else `−1.0`.
    #[must_use]
    pub fn sign(&self, x: u64) -> f64 {
        if self.hash(x) & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// Build `count` independent 4-wise hashes from a single RNG stream.
    #[must_use]
    pub fn many(rng: &mut LcgRng, count: usize) -> Vec<FourWiseHash> {
        (0..count).map(|_| FourWiseHash::new(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourwise_in_field() {
        let mut rng = LcgRng::new(7);
        let h = FourWiseHash::new(&mut rng);
        for x in 0..2000u64 {
            assert!(h.hash(x) < PRIME_MERSENNE_61, "hash out of field");
        }
    }

    #[test]
    fn fourwise_deterministic() {
        let mut rng = LcgRng::new(11);
        let h = FourWiseHash::new(&mut rng);
        assert_eq!(h.hash(123_456_789), h.hash(123_456_789));
    }

    #[test]
    fn fourwise_horner_matches_direct() {
        // Cross-check Horner against the explicit a3·x³ + a2·x² + a1·x + a0 evaluation.
        let h = FourWiseHash::with_coeffs([5, 7, 11, 13]);
        for x in [0u64, 1, 2, 3, 1000, 999_983] {
            let xr = x % PRIME_MERSENNE_61;
            let x2 = mul_mod(xr, xr);
            let x3 = mul_mod(x2, xr);
            let direct = add_mod(
                add_mod(add_mod(mul_mod(13, x3), mul_mod(11, x2)), mul_mod(7, xr)),
                5,
            );
            assert_eq!(h.hash(x), direct, "Horner mismatch at x={x}");
        }
    }

    #[test]
    fn fourwise_sign_is_pm_one() {
        let mut rng = LcgRng::new(3);
        let h = FourWiseHash::new(&mut rng);
        for x in 0..500u64 {
            let s = h.sign(x);
            assert!(s == 1.0 || s == -1.0);
        }
    }

    #[test]
    fn fourwise_sign_mean_near_zero() {
        // E[s] ≈ 0 over the family: average a single hash's signs over many keys.
        let mut rng = LcgRng::new(2024);
        let h = FourWiseHash::new(&mut rng);
        let n = 20_000u64;
        let sum: f64 = (0..n).map(|x| h.sign(x)).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "E[s] = {mean} not near 0");
    }

    #[test]
    fn fourwise_pairwise_uncorrelated() {
        // E[s_i · s_j] ≈ 0 for i ≠ j, averaged over many independent hashes.
        let mut rng = LcgRng::new(55);
        let trials = 4000usize;
        let (i, j) = (17u64, 42u64);
        let mut acc = 0.0f64;
        for _ in 0..trials {
            let h = FourWiseHash::new(&mut rng);
            acc += h.sign(i) * h.sign(j);
        }
        let corr = acc / trials as f64;
        assert!(corr.abs() < 0.05, "E[s_i·s_j] = {corr} not near 0");
    }

    #[test]
    fn fourwise_fourth_moment_pairing() {
        // 4-wise signature: E[s_a s_b s_c s_d] ≈ 0 when the four indices are distinct.
        // (A 2-universal family would give a non-vanishing, structured value here.)
        let mut rng = LcgRng::new(9001);
        let trials = 6000usize;
        let (a, b, c, d) = (3u64, 8u64, 21u64, 55u64);
        let mut acc = 0.0f64;
        for _ in 0..trials {
            let h = FourWiseHash::new(&mut rng);
            acc += h.sign(a) * h.sign(b) * h.sign(c) * h.sign(d);
        }
        let m4 = acc / trials as f64;
        assert!(m4.abs() < 0.06, "E[s_a s_b s_c s_d] = {m4} not near 0");
    }
}
