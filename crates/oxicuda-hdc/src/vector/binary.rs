//! Binary hypervectors with values in {-1, +1} (BSV/BSC model).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Generate a random binary HV of dimension D (±1).
pub fn random_binary(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut v = vec![0i8; dim];
    rng.fill_binary(&mut v);
    Ok(v)
}

/// Check that all values are in {-1, +1}.
pub fn validate_binary(hv: &[i8]) -> HdcResult<()> {
    for &v in hv {
        if v != 1 && v != -1 {
            return Err(HdcError::InvalidBinaryValue(v));
        }
    }
    Ok(())
}

/// Dot product (inner product) of two binary HVs — returns i64 sum.
pub fn binary_dot(a: &[i8], b: &[i8]) -> HdcResult<i64> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut sum = 0i64;
    for i in 0..a.len() {
        sum += (a[i] as i64) * (b[i] as i64);
    }
    Ok(sum)
}

/// Convert binary HV (±1) to bipolar count: sum of elements.
pub fn bipolar_count(hv: &[i8]) -> i64 {
    hv.iter().map(|&v| v as i64).sum()
}

/// Threshold-and-binarize a `Vec<i32>` (accumulated bundle) to ±1.
/// Tie-breaking (sum=0): random flip via rng.
pub fn threshold_binary(acc: &[i32], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
    if acc.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let result: Vec<i8> = acc
        .iter()
        .map(|&v| {
            if v > 0 {
                1i8
            } else if v < 0 {
                -1i8
            } else if rng.next_bool() {
                1i8
            } else {
                -1i8
            }
        })
        .collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn random_binary_all_valid() {
        let mut rng = LcgRng::new(1);
        let hv = random_binary(1000, &mut rng).expect("random_binary failed");
        assert_eq!(hv.len(), 1000);
        validate_binary(&hv).expect("validate failed");
    }

    #[test]
    fn binary_dot_self_equals_dim() {
        let mut rng = LcgRng::new(2);
        let hv = random_binary(512, &mut rng).expect("random_binary failed");
        let dot = binary_dot(&hv, &hv).expect("dot failed");
        assert_eq!(dot, 512);
    }

    #[test]
    fn threshold_binary_positive_accumulator() {
        let mut rng = LcgRng::new(3);
        let acc = vec![5i32; 100];
        let hv = threshold_binary(&acc, &mut rng).expect("threshold failed");
        assert!(hv.iter().all(|&v| v == 1));
    }
}
