//! Shamir secret sharing over a prime field.
//!
//! Shamir, "How to Share a Secret", Communications of the ACM 1979.
//!
//! Uses the Mersenne prime p = 2^61 − 1 for efficient modular arithmetic.
//! Provides t-of-n threshold secret sharing with Lagrange interpolation.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Mersenne prime p = 2^61 − 1 for GF(p) arithmetic.
pub const PRIME: u64 = 2_305_843_009_213_693_951; // 2^61 - 1

/// Configuration for Shamir secret sharing.
#[derive(Debug, Clone, Copy)]
pub struct ShamirConfig {
    /// Minimum number of shares needed to reconstruct (t).
    pub threshold: usize,
    /// Total number of shares to generate (n).
    pub n_parties: usize,
}

impl ShamirConfig {
    /// Create a validated Shamir configuration.
    ///
    /// # Errors
    /// Returns `ThresholdTooLarge` if `threshold > n_parties` or
    /// `InsufficientClients` if `n_parties == 0`.
    pub fn new(threshold: usize, n_parties: usize) -> FedResult<Self> {
        if n_parties == 0 {
            return Err(FedError::InsufficientClients { min: 1, got: 0 });
        }
        if threshold == 0 || threshold > n_parties {
            return Err(FedError::ThresholdTooLarge {
                threshold,
                parties: n_parties,
            });
        }
        Ok(Self {
            threshold,
            n_parties,
        })
    }
}

/// Modular addition in GF(PRIME).
#[inline]
fn mod_add(a: u64, b: u64) -> u64 {
    let (sum, overflow) = a.overflowing_add(b);
    if overflow || sum >= PRIME {
        sum.wrapping_sub(PRIME)
    } else {
        sum
    }
}

/// Modular multiplication in GF(PRIME) using u128 to avoid overflow.
#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    // Reduce mod 2^61 - 1:
    // prod = hi * 2^61 + lo
    // prod mod (2^61-1) = (hi + lo) mod (2^61-1)
    let hi = (prod >> 61) as u64;
    let lo = (prod & PRIME as u128) as u64;
    let sum = hi + lo;
    if sum >= PRIME { sum - PRIME } else { sum }
}

/// Modular exponentiation in GF(PRIME) via square-and-multiply.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        exp >>= 1;
        base = (base as u128 * base as u128 % modulus as u128) as u64;
    }
    result
}

/// Modular inverse in GF(PRIME) via Fermat's little theorem.
/// `a^{-1} = a^{p-2} mod p`.
fn mod_inv(a: u64) -> FedResult<u64> {
    if a == 0 {
        return Err(FedError::ReconstructionFailed);
    }
    Ok(mod_pow(a, PRIME - 2, PRIME))
}

/// Evaluate a polynomial over GF(PRIME) at point x using Horner's method.
/// `poly[0]` is the constant term (the secret), `poly[1..t]` are random coefficients.
fn poly_eval(poly: &[u64], x: u64) -> u64 {
    let mut result = 0u64;
    for &coef in poly.iter().rev() {
        result = mod_add(mod_mul(result, x), coef);
    }
    result
}

/// Share a single scalar secret into n shares using Shamir's scheme.
///
/// Returns a vector of `(party_index, share_value)` pairs where
/// party_index is 1-indexed (1..=n_parties).
///
/// # Errors
/// Returns errors from `ShamirConfig::new` or `InvalidShareCount`.
pub fn share_scalar(
    secret: u64,
    config: &ShamirConfig,
    rng: &mut LcgRng,
) -> FedResult<Vec<(usize, u64)>> {
    let t = config.threshold;
    let n = config.n_parties;

    // Build degree-(t-1) polynomial: poly[0] = secret, poly[1..t-1] = random
    let mut poly = vec![0u64; t];
    poly[0] = secret % PRIME;
    for coef in poly.iter_mut().skip(1) {
        // Generate a non-zero random coefficient in [1, PRIME-1]
        let raw = rng.next_u64();
        *coef = (raw % (PRIME - 1)) + 1;
    }

    // Evaluate polynomial at points x = 1, 2, ..., n
    let shares: Vec<(usize, u64)> = (1..=n).map(|x| (x, poly_eval(&poly, x as u64))).collect();

    Ok(shares)
}

/// Reconstruct a secret from any `threshold` shares using Lagrange interpolation.
///
/// # Errors
/// Returns `InvalidShareCount` if fewer than `threshold` shares are provided,
/// or `ReconstructionFailed` if interpolation fails (e.g., duplicate x-values).
pub fn reconstruct_scalar(shares: &[(usize, u64)], threshold: usize) -> FedResult<u64> {
    if shares.len() < threshold {
        return Err(FedError::InvalidShareCount {
            min: threshold,
            got: shares.len(),
        });
    }

    // Use exactly `threshold` shares
    let used = &shares[..threshold];

    // Lagrange interpolation at x = 0:
    // f(0) = Σ_i y_i * Π_{j≠i} (0 - x_j) / (x_i - x_j)
    let mut secret = 0u64;
    for (i, &(xi, yi)) in used.iter().enumerate() {
        let xi_field = xi as u64;
        let mut num = 1u64;
        let mut den = 1u64;
        for (j, &(xj, _)) in used.iter().enumerate() {
            if i == j {
                continue;
            }
            let xj_field = xj as u64;
            // num *= (0 - xj) = PRIME - xj (mod PRIME)
            num = mod_mul(num, PRIME - xj_field);
            // den *= (xi - xj)
            let diff = if xi_field >= xj_field {
                xi_field - xj_field
            } else {
                PRIME - (xj_field - xi_field)
            };
            den = mod_mul(den, diff);
        }

        let den_inv = mod_inv(den)?;
        let lagrange_coef = mod_mul(num, den_inv);
        let term = mod_mul(yi, lagrange_coef);
        secret = mod_add(secret, term);
    }

    Ok(secret)
}

/// Scale for converting f32 values to u64 for Shamir sharing.
const GRADIENT_SCALE: f64 = 1_000_000.0;

/// Share a vector of f32 gradient values using Shamir secret sharing.
///
/// Each element is scaled by `1e6` and rounded to a u64 for exact GF(p) arithmetic.
///
/// Returns a `Vec<Vec<(usize, u64)>>` — outer index over elements, inner over parties.
///
/// # Errors
/// Returns errors if any element is non-finite.
pub fn share_gradient(
    grad: &[f32],
    config: &ShamirConfig,
    rng: &mut LcgRng,
) -> FedResult<Vec<Vec<(usize, u64)>>> {
    grad.iter()
        .map(|&v| {
            if !v.is_finite() {
                return Err(FedError::Internal(
                    "non-finite gradient element cannot be shared".into(),
                ));
            }
            // Scale and map negative values to GF(PRIME): use offset arithmetic
            let scaled = (v as f64 * GRADIENT_SCALE).round();
            // Map to GF(PRIME) with offset to handle negatives
            let int_val = if scaled >= 0.0 {
                (scaled as u64) % PRIME
            } else {
                let abs_scaled = (-scaled) as u64 % PRIME;
                if abs_scaled == 0 {
                    0
                } else {
                    PRIME - abs_scaled
                }
            };
            share_scalar(int_val, config, rng)
        })
        .collect()
}

/// Reconstruct a gradient vector from Shamir shares.
///
/// The inverse of `share_gradient`: unscales by `1e-6` and converts GF(p)
/// values back to signed f32. Values > PRIME/2 are treated as negative.
///
/// # Errors
/// Returns errors from `reconstruct_scalar` for each element.
pub fn reconstruct_gradient(
    shares_per_elem: &[Vec<(usize, u64)>],
    threshold: usize,
) -> FedResult<Vec<f32>> {
    shares_per_elem
        .iter()
        .map(|shares| {
            let secret = reconstruct_scalar(shares, threshold)?;
            // Convert back: values > PRIME/2 are negative
            let signed: f64 = if secret <= PRIME / 2 {
                secret as f64
            } else {
                -((PRIME - secret) as f64)
            };
            Ok((signed / GRADIENT_SCALE) as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shamir_config_valid() {
        let cfg = ShamirConfig::new(2, 5).expect("test invariant: valid shamir config");
        assert_eq!(cfg.threshold, 2);
        assert_eq!(cfg.n_parties, 5);
    }

    #[test]
    fn shamir_config_threshold_too_large() {
        assert!(matches!(
            ShamirConfig::new(6, 5),
            Err(FedError::ThresholdTooLarge { .. })
        ));
    }

    #[test]
    fn shamir_share_reconstruct_scalar_basic() {
        let config = ShamirConfig::new(2, 3).expect("test invariant: valid config");
        let mut rng = LcgRng::new(42);
        let secret = 12345u64;
        let shares =
            share_scalar(secret, &config, &mut rng).expect("test invariant: valid share_scalar");
        assert_eq!(shares.len(), 3);
        // Reconstruct from first 2 shares
        let reconstructed =
            reconstruct_scalar(&shares[..2], 2).expect("test invariant: valid reconstruct");
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn shamir_reconstruct_from_different_subsets() {
        let config = ShamirConfig::new(2, 4).expect("test invariant: valid config");
        let mut rng = LcgRng::new(7);
        let secret = 99999u64;
        let shares =
            share_scalar(secret, &config, &mut rng).expect("test invariant: valid share_scalar");
        // Test reconstruction from different pairs
        let r1 = reconstruct_scalar(&[shares[0], shares[1]], 2)
            .expect("test invariant: reconstruct pair 01");
        let r2 = reconstruct_scalar(&[shares[0], shares[2]], 2)
            .expect("test invariant: reconstruct pair 02");
        let r3 = reconstruct_scalar(&[shares[1], shares[3]], 2)
            .expect("test invariant: reconstruct pair 13");
        assert_eq!(r1, secret);
        assert_eq!(r2, secret);
        assert_eq!(r3, secret);
    }

    #[test]
    fn shamir_reconstruct_insufficient_shares() {
        assert!(matches!(
            reconstruct_scalar(&[(1, 100)], 2),
            Err(FedError::InvalidShareCount { .. })
        ));
    }

    #[test]
    fn shamir_gradient_roundtrip() {
        let config = ShamirConfig::new(2, 3).expect("test invariant: valid config");
        let mut rng = LcgRng::new(99);
        let grad = vec![0.5f32, -0.3, 1.2, -0.7];
        let shares =
            share_gradient(&grad, &config, &mut rng).expect("test invariant: valid share_gradient");
        let reconstructed =
            reconstruct_gradient(&shares, 2).expect("test invariant: valid reconstruct_gradient");
        for (orig, rec) in grad.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - rec).abs() < 1e-3,
                "gradient mismatch: orig={orig}, rec={rec}"
            );
        }
    }

    #[test]
    fn mod_mul_correctness() {
        // 3 * 4 = 12
        assert_eq!(mod_mul(3, 4), 12);
        // Large values
        let p = PRIME;
        assert_eq!(mod_mul(p - 1, p - 1), 1); // (-1) * (-1) = 1 mod p
    }

    #[test]
    fn mod_inv_inverse() {
        let a = 12345u64;
        let inv = mod_inv(a).expect("test invariant: valid mod inv");
        assert_eq!(mod_mul(a, inv), 1);
    }
}
