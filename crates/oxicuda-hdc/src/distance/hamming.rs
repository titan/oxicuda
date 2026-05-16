//! Hamming distance for binary hypervectors (±1 encoding).

use crate::error::{HdcError, HdcResult};

/// Hamming distance fraction for binary HVs (±1 encoding).
///
/// For ±1 encoding: Hamming_frac = (D - dot(a,b)) / (2D) = 0.5 - dot(a,b)/(2D).
/// Returns a value in [0, 1].
pub fn hamming_frac(a: &[i8], b: &[i8]) -> HdcResult<f64> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = a.len();
    let count = hamming_count(a, b)?;
    Ok(count as f64 / dim as f64)
}

/// Number of positions that differ between two binary HVs.
/// For ±1 encoding: `a[i] != b[i]` ⟺ `a[i]*b[i] == -1`.
pub fn hamming_count(a: &[i8], b: &[i8]) -> HdcResult<usize> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let count: usize = a.iter().zip(b.iter()).filter(|&(ai, bi)| ai != bi).count();
    Ok(count)
}

/// Significance threshold for Hamming similarity under random binary HV baseline.
///
/// Two random binary HVs have Hamming_frac ≈ 0.5 (mean) with std ≈ 0.5/sqrt(D).
/// Returns the frac threshold beyond which deviation is significant at n_sigma standard deviations:
/// threshold = 0.5 - n_sigma * 0.5 / sqrt(D)
/// (values below this threshold are considered "similar").
pub fn hamming_similarity_threshold(dim: usize, n_sigma: f64) -> f64 {
    if dim == 0 {
        return 0.5;
    }
    0.5 - n_sigma * 0.5 / (dim as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_self_zero() {
        let a: Vec<i8> = vec![1, -1, 1, -1, 1, -1];
        assert_eq!(hamming_count(&a, &a).expect("count"), 0);
        let frac = hamming_frac(&a, &a).expect("frac");
        assert!((frac).abs() < 1e-9);
    }

    #[test]
    fn hamming_opposite_one() {
        let a: Vec<i8> = vec![1, 1, 1, -1, -1];
        let b: Vec<i8> = vec![-1, -1, -1, 1, 1];
        let frac = hamming_frac(&a, &b).expect("frac");
        assert!((frac - 1.0).abs() < 1e-9);
    }
}
