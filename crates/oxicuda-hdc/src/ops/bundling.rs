//! Bundling operations: majority vote (binary), superposition (integer/complex), weighted bundling.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::complex;

/// Bundle K binary HVs by majority vote.
/// Ties broken randomly (passed-in rng).
pub fn bundle_binary(hvs: &[Vec<i8>], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hvs[0].len();
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    for (idx, hv) in hvs.iter().enumerate().skip(1) {
        if hv.len() != dim {
            return Err(HdcError::DimensionMismatch {
                expected: dim,
                got: hvs[idx].len(),
            });
        }
    }
    let mut acc = vec![0i32; dim];
    for hv in hvs {
        for (a, &v) in acc.iter_mut().zip(hv.iter()) {
            *a += v as i32;
        }
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

/// Bundle K integer HVs by element-wise sum.
pub fn bundle_integer(hvs: &[Vec<i32>]) -> HdcResult<Vec<i32>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hvs[0].len();
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    for (idx, hv) in hvs.iter().enumerate().skip(1) {
        if hv.len() != dim {
            return Err(HdcError::DimensionMismatch {
                expected: dim,
                got: hvs[idx].len(),
            });
        }
    }
    let mut result = vec![0i32; dim];
    for hv in hvs {
        for (r, &v) in result.iter_mut().zip(hv.iter()) {
            *r += v;
        }
    }
    Ok(result)
}

/// Bundle K complex HVs by element-wise complex sum + normalize each to unit circle.
pub fn bundle_complex(hvs: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    complex::complex_bundle(hvs)
}

/// Weighted binary bundle: each HV is weighted by a f32 score.
/// The weighted accumulator is thresholded to ±1.
pub fn weighted_bundle_binary(
    hvs: &[Vec<i8>],
    weights: &[f32],
    rng: &mut LcgRng,
) -> HdcResult<Vec<i8>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if hvs.len() != weights.len() {
        return Err(HdcError::DimensionMismatch {
            expected: hvs.len(),
            got: weights.len(),
        });
    }
    let dim = hvs[0].len();
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    for (idx, hv) in hvs.iter().enumerate().skip(1) {
        if hv.len() != dim {
            return Err(HdcError::DimensionMismatch {
                expected: dim,
                got: hvs[idx].len(),
            });
        }
    }
    let mut acc = vec![0f32; dim];
    for (hv, &w) in hvs.iter().zip(weights.iter()) {
        for (a, &v) in acc.iter_mut().zip(hv.iter()) {
            *a += w * (v as f32);
        }
    }
    let result: Vec<i8> = acc
        .iter()
        .map(|&v| {
            if v > 0.0 {
                1i8
            } else if v < 0.0 {
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
    fn bundle_two_identical_hvs() {
        let hv: Vec<i8> = vec![1, -1, 1, -1];
        let hvs = vec![hv.clone(), hv.clone()];
        let mut rng = LcgRng::new(30);
        let result = bundle_binary(&hvs, &mut rng).expect("bundle");
        // 2 identical → majority = original
        assert_eq!(result, hv);
    }

    #[test]
    fn bundle_integer_sums() {
        let a = vec![1, -1, 0, 2];
        let b = vec![0, 1, -1, -1];
        let bundled = bundle_integer(&[a, b]).expect("bundle");
        assert_eq!(bundled, vec![1, 0, -1, 1]);
    }
}
