//! Pairwise random masking for secure aggregation.
//!
//! Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
//! Machine Learning", CCS 2017.
//!
//! Each pair of clients (i, j) with i < j shares a PRG seed that generates
//! a random mask: client i adds the mask, client j subtracts it. The masks
//! cancel when the server sums all masked updates.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Generate a deterministic pairwise mask between clients `i` and `j`.
///
/// Uses an LCG seeded with a combination of client indices and a shared
/// secret to produce a reproducible mask vector. Client `i` (i < j) adds
/// the mask; client `j` subtracts it.
///
/// # Arguments
/// - `i` — first client index
/// - `j` — second client index
/// - `shared_seed` — shared secret between clients i and j
/// - `n_elems` — number of parameters to mask
///
/// # Returns
/// Mask vector of u32 values (arithmetic is modulo 2^32).
pub fn generate_mask(i: usize, j: usize, shared_seed: u64, n_elems: usize) -> Vec<u32> {
    // Derive a deterministic seed from the pair + shared secret
    let pair_seed = shared_seed
        .wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add((j as u64).wrapping_mul(0x517cc1b727220a95));
    let mut rng = LcgRng::new(pair_seed);
    (0..n_elems).map(|_| rng.next_u32()).collect()
}

/// Apply an additive mask to a gradient vector (modular u32 arithmetic).
///
/// Interprets `gradient` elements as raw u32 bit patterns and performs
/// `out[i] = (gradient_bits[i] + mask[i]) mod 2^32`.
///
/// # Errors
/// Returns `DimensionMismatch` if `gradient` and `mask` have different lengths.
pub fn apply_mask(gradient: &[f32], mask: &[u32]) -> FedResult<Vec<u32>> {
    if gradient.len() != mask.len() {
        return Err(FedError::DimensionMismatch {
            expected: gradient.len(),
            got: mask.len(),
        });
    }
    Ok(gradient
        .iter()
        .zip(mask.iter())
        .map(|(&g, &m)| g.to_bits().wrapping_add(m))
        .collect())
}

/// Remove a mask from a masked gradient (subtract the mask modulo 2^32).
///
/// `unmasked[i] = f32::from_bits((masked[i] - mask[i]) mod 2^32)`
///
/// # Errors
/// Returns `DimensionMismatch` if `masked` and `mask` have different lengths.
pub fn unmask(masked: &[u32], mask: &[u32]) -> FedResult<Vec<f32>> {
    if masked.len() != mask.len() {
        return Err(FedError::DimensionMismatch {
            expected: masked.len(),
            got: mask.len(),
        });
    }
    Ok(masked
        .iter()
        .zip(mask.iter())
        .map(|(&m, &k)| f32::from_bits(m.wrapping_sub(k)))
        .collect())
}

/// Apply all pairwise masks for client `client_id` in an n-party protocol.
///
/// For each other client `j`:
/// - If `client_id < j`: add the pairwise mask (client id is the "adder")
/// - If `client_id > j`: subtract the pairwise mask (client is the "subtractor")
///
/// This ensures all pairwise masks cancel when the server sums all masked updates.
///
/// # Arguments
/// - `gradient` — client's model update as u32 bits
/// - `client_id` — this client's 0-indexed identifier
/// - `n_parties` — total number of parties
/// - `shared_seeds` — shared secret seeds, one per pair; `shared_seeds[j]` is the
///   seed shared between this client and party `j`
///
/// # Errors
/// Returns `DimensionMismatch` or `InsufficientClients` on invalid input.
pub fn apply_pairwise_masks(
    gradient: &[u32],
    client_id: usize,
    n_parties: usize,
    shared_seeds: &[u64],
) -> FedResult<Vec<u32>> {
    if shared_seeds.len() != n_parties {
        return Err(FedError::DimensionMismatch {
            expected: n_parties,
            got: shared_seeds.len(),
        });
    }
    let n = gradient.len();
    let mut masked = gradient.to_vec();
    for (other, &seed) in shared_seeds.iter().enumerate() {
        if other == client_id {
            continue;
        }
        let (i, j) = if client_id < other {
            (client_id, other)
        } else {
            (other, client_id)
        };
        let mask = generate_mask(i, j, seed, n);
        for (m, &k) in masked.iter_mut().zip(mask.iter()) {
            if client_id < other {
                *m = m.wrapping_add(k); // adder
            } else {
                *m = m.wrapping_sub(k); // subtractor
            }
        }
    }
    Ok(masked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_mask_deterministic() {
        let m1 = generate_mask(0, 1, 12345, 10);
        let m2 = generate_mask(0, 1, 12345, 10);
        assert_eq!(m1, m2, "mask generation must be deterministic");
    }

    #[test]
    fn generate_mask_different_pairs() {
        let m01 = generate_mask(0, 1, 42, 10);
        let m02 = generate_mask(0, 2, 42, 10);
        assert_ne!(m01, m02, "different pairs should produce different masks");
    }

    #[test]
    fn apply_unmask_roundtrip() {
        let grad = vec![1.0f32, -2.5, 3.7, 0.0];
        let mask = vec![11111u32, 22222, 33333, 44444];
        let masked = apply_mask(&grad, &mask).expect("test invariant: valid apply_mask");
        let unmasked = unmask(&masked, &mask).expect("test invariant: valid unmask");
        // The roundtrip should be exact for finite values representable in f32
        for (orig, rec) in grad.iter().zip(unmasked.iter()) {
            // Due to bit manipulation, only exactly representable values round-trip
            // We check that the bits are the same
            assert_eq!(orig.to_bits(), rec.to_bits(), "roundtrip mismatch");
        }
    }

    #[test]
    fn apply_mask_dimension_mismatch() {
        let grad = vec![1.0f32, 2.0];
        let mask = vec![1u32, 2, 3];
        assert!(matches!(
            apply_mask(&grad, &mask),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn pairwise_masks_cancel_on_sum() {
        // Two parties, their pairwise masks should cancel
        let n = 4;
        let shared_seeds = vec![999u64; 2]; // symmetric seed
        let grad0 = vec![1u32, 2, 3, 4];
        let grad1 = vec![5u32, 6, 7, 8];

        let masked0 =
            apply_pairwise_masks(&grad0, 0, 2, &shared_seeds).expect("test invariant: masked0");
        let masked1 =
            apply_pairwise_masks(&grad1, 1, 2, &shared_seeds).expect("test invariant: masked1");

        // Sum should equal grad0 + grad1 (masks cancel)
        for i in 0..n {
            let sum_masked = masked0[i].wrapping_add(masked1[i]);
            let sum_plain = grad0[i].wrapping_add(grad1[i]);
            assert_eq!(
                sum_masked, sum_plain,
                "pairwise masks should cancel on sum at index {i}"
            );
        }
    }
}
