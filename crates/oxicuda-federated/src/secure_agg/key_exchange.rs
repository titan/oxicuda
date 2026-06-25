//! Diffie-Hellman key agreement for secure aggregation.
//!
//! Closes the protocol gap in [`crate::secure_agg::masking`]: the Bonawitz
//! pairwise-masking scheme requires every pair of clients `(i, j)` to hold a
//! *shared secret seed* that both can compute but no third party can. Until
//! now those seeds were caller-supplied. This module derives them from public
//! key exchange so that, after a single round of broadcasting public keys, any
//! pair can compute its shared mask seed locally.
//!
//! # Protocol
//!
//! Finite-field Diffie-Hellman over the same Mersenne prime field
//! `GF(p)`, `p = 2^61 − 1`, that [`crate::secure_agg::shamir`] uses.
//!
//! 1. The group is the multiplicative group of `GF(p)`, of order `p − 1`.
//!    `p − 1 = 2^61 − 2 = 2 · (2^60 − 1) = 2 · 3^2 · 5^2 · 7 · 11 · 13 · 31 ·
//!    41 · 61 · 151 · 331 · 1321`. A small generator `g = 37` is a primitive
//!    root modulo `p` (verified against every prime factor of `p − 1`).
//! 2. Each client `i` draws a private exponent `a_i ∈ [2, p − 2]` and
//!    publishes `A_i = g^{a_i} mod p`.
//! 3. The pair `(i, j)` shares `s_{ij} = A_j^{a_i} = A_i^{a_j} = g^{a_i a_j}`.
//! 4. The 61-bit field element `s_{ij}` is mixed into a 64-bit seed for
//!    [`crate::secure_agg::masking::generate_mask`].
//!
//! Finite-field DH is *not* itself a cryptographically hardened primitive at
//! this modulus size (61 bits is far below production parameters); it is a
//! deterministic, pure-Rust, allocation-free stand-in that exercises the full
//! key-agreement *plumbing* — keypair generation, public broadcast, symmetric
//! shared-secret derivation, and seed mixing — so the masking layer no longer
//! needs caller-supplied seeds. The arithmetic (`mod_pow`, primitive-root
//! checks, symmetric agreement) is exact.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Prime modulus for the DH group: the Mersenne prime `p = 2^61 − 1`.
///
/// Matches [`crate::secure_agg::shamir::PRIME`] so the two protocols share a
/// field; re-declared here to keep the module self-contained.
pub const DH_PRIME: u64 = 2_305_843_009_213_693_951; // 2^61 - 1

/// Generator (primitive root) of the multiplicative group `GF(DH_PRIME)*`.
///
/// `g = 37` has multiplicative order exactly `DH_PRIME − 1` (checked against
/// all prime factors of `DH_PRIME − 1` in the unit tests).
pub const DH_GENERATOR: u64 = 37;

/// Prime factors of `DH_PRIME − 1 = 2^61 − 2`.
///
/// `2^61 − 2 = 2 · 3² · 5² · 7 · 11 · 13 · 31 · 41 · 61 · 151 · 331 · 1321`.
/// Used to certify that [`DH_GENERATOR`] is a primitive root: `g` generates
/// the whole group iff `g^{(p−1)/q} ≠ 1` for every prime factor `q`.
const PRIME_FACTORS_OF_P_MINUS_1: [u64; 12] = [2, 3, 5, 7, 11, 13, 31, 41, 61, 151, 331, 1321];

/// Modular exponentiation `base^exp mod modulus` via square-and-multiply.
///
/// Uses `u128` intermediates so the 61-bit operands never overflow.
#[inline]
#[must_use]
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }
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

/// Test whether `g` is a primitive root modulo [`DH_PRIME`].
///
/// `g` generates the full group of order `p − 1` iff for every prime factor
/// `q` of `p − 1`, `g^{(p−1)/q} ≢ 1 (mod p)`.
#[must_use]
pub fn is_primitive_root(g: u64) -> bool {
    let g = g % DH_PRIME;
    if g == 0 {
        return false;
    }
    let order = DH_PRIME - 1;
    PRIME_FACTORS_OF_P_MINUS_1
        .iter()
        .all(|&q| mod_pow(g, order / q, DH_PRIME) != 1)
}

/// A Diffie-Hellman keypair for one client.
///
/// The `public` component is broadcast; the `private` exponent must never
/// leave the client. Constructed via [`DhKeyPair::generate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DhKeyPair {
    /// Secret exponent `a ∈ [2, p − 2]`. Keep private.
    private: u64,
    /// Public value `g^a mod p`. Broadcast to all peers.
    public: u64,
}

impl DhKeyPair {
    /// Draw a fresh keypair from `rng`.
    ///
    /// The private exponent is sampled uniformly in `[2, p − 2]` (excluding the
    /// trivial `0` and `1` exponents and the order-2 boundary), and the public
    /// value is `g^a mod p`.
    #[must_use]
    pub fn generate(rng: &mut LcgRng) -> Self {
        // Uniform private exponent in [2, p-2].
        let span = DH_PRIME - 3; // candidates: 0 .. span-1 → mapped to 2 .. p-2
        let private = 2 + (rng.next_u64() % span);
        let public = mod_pow(DH_GENERATOR, private, DH_PRIME);
        Self { private, public }
    }

    /// Construct a keypair from an explicit private exponent (testing / replay).
    ///
    /// # Errors
    /// Returns [`FedError::Internal`] if `private` is outside `[2, p − 2]`.
    pub fn from_private(private: u64) -> FedResult<Self> {
        if !(2..=DH_PRIME - 2).contains(&private) {
            return Err(FedError::Internal(format!(
                "DH private exponent {private} out of range [2, {}]",
                DH_PRIME - 2
            )));
        }
        let public = mod_pow(DH_GENERATOR, private, DH_PRIME);
        Ok(Self { private, public })
    }

    /// The public value `g^a mod p` to broadcast.
    #[must_use]
    #[inline]
    pub fn public(&self) -> u64 {
        self.public
    }

    /// Derive the raw shared field element with a peer's public value.
    ///
    /// Computes `peer_public^a mod p`. Both sides obtain the same value because
    /// `(g^b)^a = (g^a)^b = g^{ab}`.
    ///
    /// # Errors
    /// Returns [`FedError::Internal`] if `peer_public` is `0`, `1`, or
    /// `≥ p` (a degenerate or out-of-field public key that would collapse the
    /// shared secret to a constant).
    pub fn shared_field_element(&self, peer_public: u64) -> FedResult<u64> {
        if peer_public <= 1 || peer_public >= DH_PRIME {
            return Err(FedError::Internal(format!(
                "invalid peer public value {peer_public}"
            )));
        }
        Ok(mod_pow(peer_public, self.private, DH_PRIME))
    }

    /// Derive a 64-bit mask seed shared with the peer at `peer_public`.
    ///
    /// The 61-bit shared field element is diffused across the full 64-bit width
    /// with an xorshift/multiply finaliser (SplitMix64-style) so that the top
    /// bits — which a raw `GF(2^61−1)` element leaves zero — carry entropy
    /// before the value is handed to
    /// [`crate::secure_agg::masking::generate_mask`].
    ///
    /// The result is symmetric: clients `i` and `j` compute the identical seed.
    ///
    /// # Errors
    /// Propagates [`DhKeyPair::shared_field_element`] errors.
    pub fn shared_seed(&self, peer_public: u64) -> FedResult<u64> {
        let element = self.shared_field_element(peer_public)?;
        Ok(mix_seed(element))
    }
}

/// SplitMix64 finaliser: diffuse a field element into a full-width 64-bit seed.
#[inline]
#[must_use]
fn mix_seed(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Build the full pairwise shared-seed table for a cohort of `n` clients.
///
/// `public_keys[i]` is client `i`'s broadcast public value. The returned `n×n`
/// matrix `seeds` has `seeds[i][j] = seeds[j][i]` equal to the seed shared by
/// clients `i` and `j` (the diagonal is `0`). Every client can reproduce its
/// own row locally from its private key, but the server can build the whole
/// table from public broadcasts alone — exactly what
/// [`crate::secure_agg::masking::apply_pairwise_masks`] consumes.
///
/// # Errors
/// Returns [`FedError::DimensionMismatch`] if `private_keys` and `public_keys`
/// disagree in length, or [`FedError::Internal`] if any peer public key is
/// degenerate.
pub fn pairwise_seed_matrix(
    private_keys: &[DhKeyPair],
    public_keys: &[u64],
) -> FedResult<Vec<Vec<u64>>> {
    if private_keys.len() != public_keys.len() {
        return Err(FedError::DimensionMismatch {
            expected: private_keys.len(),
            got: public_keys.len(),
        });
    }
    let n = private_keys.len();
    let mut seeds = vec![vec![0u64; n]; n];
    for (i, kp) in private_keys.iter().enumerate() {
        for (j, &pk_j) in public_keys.iter().enumerate() {
            if i == j {
                continue;
            }
            seeds[i][j] = kp.shared_seed(pk_j)?;
        }
    }
    Ok(seeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_primitive_root() {
        assert!(
            is_primitive_root(DH_GENERATOR),
            "g=37 must be a primitive root mod 2^61-1"
        );
    }

    #[test]
    fn non_primitive_root_rejected() {
        // 1 generates only {1}; a perfect square cannot be a primitive root of
        // a group of even order. g^2 for the real generator is a quadratic
        // residue, hence not a primitive root.
        assert!(!is_primitive_root(1));
        let square = mod_pow(DH_GENERATOR, 2, DH_PRIME);
        assert!(!is_primitive_root(square));
    }

    #[test]
    fn mod_pow_matches_repeated_multiply() {
        // 3^13 mod 1000 = 1594323 mod 1000 = 323.
        assert_eq!(mod_pow(3, 13, 1000), 323);
        assert_eq!(mod_pow(2, 10, 1_000_000), 1024);
        // Fermat: g^(p-1) ≡ 1.
        assert_eq!(mod_pow(DH_GENERATOR, DH_PRIME - 1, DH_PRIME), 1);
    }

    #[test]
    fn key_agreement_is_symmetric() {
        let mut rng = LcgRng::new(2024);
        let alice = DhKeyPair::generate(&mut rng);
        let bob = DhKeyPair::generate(&mut rng);

        let ab = alice
            .shared_field_element(bob.public())
            .expect("alice derives shared element");
        let ba = bob
            .shared_field_element(alice.public())
            .expect("bob derives shared element");
        assert_eq!(ab, ba, "DH shared field element must be symmetric");

        let seed_ab = alice.shared_seed(bob.public()).expect("alice derives seed");
        let seed_ba = bob.shared_seed(alice.public()).expect("bob derives seed");
        assert_eq!(seed_ab, seed_ba, "derived mask seed must be symmetric");
    }

    #[test]
    fn distinct_pairs_have_distinct_seeds() {
        let mut rng = LcgRng::new(7);
        let a = DhKeyPair::generate(&mut rng);
        let b = DhKeyPair::generate(&mut rng);
        let c = DhKeyPair::generate(&mut rng);
        let ab = a.shared_seed(b.public()).expect("ab");
        let ac = a.shared_seed(c.public()).expect("ac");
        let bc = b.shared_seed(c.public()).expect("bc");
        assert_ne!(ab, ac);
        assert_ne!(ab, bc);
        assert_ne!(ac, bc);
    }

    #[test]
    fn from_private_reproduces_public() {
        let kp = DhKeyPair::from_private(123_456_789).expect("valid private");
        let expected = mod_pow(DH_GENERATOR, 123_456_789, DH_PRIME);
        assert_eq!(kp.public(), expected);
    }

    #[test]
    fn from_private_rejects_out_of_range() {
        assert!(matches!(
            DhKeyPair::from_private(1),
            Err(FedError::Internal(_))
        ));
        assert!(matches!(
            DhKeyPair::from_private(DH_PRIME - 1),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn degenerate_peer_public_rejected() {
        let mut rng = LcgRng::new(11);
        let kp = DhKeyPair::generate(&mut rng);
        assert!(matches!(
            kp.shared_field_element(0),
            Err(FedError::Internal(_))
        ));
        assert!(matches!(
            kp.shared_field_element(1),
            Err(FedError::Internal(_))
        ));
        assert!(matches!(
            kp.shared_field_element(DH_PRIME),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn seed_matrix_is_symmetric_with_zero_diagonal() {
        let mut rng = LcgRng::new(555);
        let n = 5;
        let keys: Vec<DhKeyPair> = (0..n).map(|_| DhKeyPair::generate(&mut rng)).collect();
        let publics: Vec<u64> = keys.iter().map(DhKeyPair::public).collect();
        let seeds = pairwise_seed_matrix(&keys, &publics).expect("seed matrix");
        for (i, row) in seeds.iter().enumerate() {
            assert_eq!(row[i], 0, "diagonal must be zero");
            for (j, &s_ij) in row.iter().enumerate().skip(i + 1) {
                assert_eq!(s_ij, seeds[j][i], "seed matrix must be symmetric");
                assert_ne!(s_ij, 0, "off-diagonal seed must be derived");
            }
        }
    }

    #[test]
    fn seed_matrix_dimension_mismatch() {
        let mut rng = LcgRng::new(1);
        let keys: Vec<DhKeyPair> = (0..3).map(|_| DhKeyPair::generate(&mut rng)).collect();
        let publics = vec![keys[0].public(), keys[1].public()]; // wrong length
        assert!(matches!(
            pairwise_seed_matrix(&keys, &publics),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dh_seeds_drive_cancelling_pairwise_masks() {
        // End-to-end: derive seeds via DH, feed them to the masking layer, and
        // confirm the masks still cancel when the server sums masked updates.
        use crate::secure_agg::masking::apply_pairwise_masks;

        let mut rng = LcgRng::new(909);
        let n = 3;
        let keys: Vec<DhKeyPair> = (0..n).map(|_| DhKeyPair::generate(&mut rng)).collect();
        let publics: Vec<u64> = keys.iter().map(DhKeyPair::public).collect();
        let seeds = pairwise_seed_matrix(&keys, &publics).expect("seed matrix");

        let grads: Vec<Vec<u32>> = vec![
            vec![10, 20, 30, 40],
            vec![1, 2, 3, 4],
            vec![100, 200, 300, 400],
        ];

        // Each client masks using its own row of the symmetric seed matrix.
        let masked: Vec<Vec<u32>> = (0..n)
            .map(|id| {
                apply_pairwise_masks(&grads[id], id, n, &seeds[id]).expect("masking succeeds")
            })
            .collect();

        // Server sums masked updates; pairwise masks (which are symmetric:
        // seeds[i][j] == seeds[j][i]) cancel because i adds and j subtracts.
        let len = grads[0].len();
        for k in 0..len {
            let mut sum_masked = 0u32;
            let mut sum_plain = 0u32;
            for c in 0..n {
                sum_masked = sum_masked.wrapping_add(masked[c][k]);
                sum_plain = sum_plain.wrapping_add(grads[c][k]);
            }
            assert_eq!(
                sum_masked, sum_plain,
                "DH-seeded pairwise masks must cancel at index {k}"
            );
        }
    }
}
