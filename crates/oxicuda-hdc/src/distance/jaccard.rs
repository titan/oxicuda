//! Jaccard similarity for binary hypervectors.
//!
//! Converts ±1 encoding to 0/1 (present = (v == 1)), then computes J(A,B) = |A∩B| / |A∪B|.

use crate::error::{HdcError, HdcResult};

/// Jaccard similarity for binary HVs.
///
/// Treats 1 as set membership, -1 as absence.
/// J(A,B) = |A∩B| / |A∪B|
pub fn jaccard_binary(a: &[i8], b: &[i8]) -> HdcResult<f64> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let mut intersection = 0usize;
    let mut union_count = 0usize;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let a_pos = ai == 1;
        let b_pos = bi == 1;
        if a_pos && b_pos {
            intersection += 1;
        }
        if a_pos || b_pos {
            union_count += 1;
        }
    }
    if union_count == 0 {
        // Both HVs are all-negative: define J = 1 (identical empty sets)
        return Ok(1.0);
    }
    Ok(intersection as f64 / union_count as f64)
}

/// MinHash-style similarity estimate for binary HVs.
///
/// For ±1 binary HVs, minihash_similarity ≈ Jaccard of the +1 positions.
/// Implemented as: count positions where both are +1 (min(a,b)=1) vs max.
/// For Bernoulli symmetric vectors, this gives an unbiased estimate of Jaccard.
pub fn minihash_similarity(a: &[i8], b: &[i8]) -> HdcResult<f64> {
    // For ±1 encoding, the MinHash similarity equals the Jaccard of the +1-position sets.
    jaccard_binary(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_self_one() {
        let a: Vec<i8> = vec![1, -1, 1, 1, -1];
        let j = jaccard_binary(&a, &a).expect("jaccard");
        assert!((j - 1.0).abs() < 1e-9, "j={j}");
    }

    #[test]
    fn jaccard_disjoint_zero() {
        let a: Vec<i8> = vec![1, 1, -1, -1];
        let b: Vec<i8> = vec![-1, -1, 1, 1];
        let j = jaccard_binary(&a, &b).expect("jaccard");
        assert!((j).abs() < 1e-9, "j={j}");
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a: Vec<i8> = vec![1, 1, -1, -1];
        let b: Vec<i8> = vec![1, -1, 1, -1];
        let j = jaccard_binary(&a, &b).expect("jaccard");
        // Intersection = {pos 0}, Union = {pos 0, 1, 2} → J = 1/3
        assert!((j - 1.0 / 3.0).abs() < 1e-9, "j={j}");
    }
}
