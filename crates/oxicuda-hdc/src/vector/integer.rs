//! Integer hypervectors Z^D (MAP model) with values typically in {-1, 0, +1}.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Generate random integer HV with components sampled uniformly from {-1, 0, +1}.
pub fn random_integer(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<i32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut v = vec![0i32; dim];
    for x in v.iter_mut() {
        *x = ((rng.next_u64() as i64).rem_euclid(3) - 1) as i32;
    }
    Ok(v)
}

/// Element-wise multiply two integer HVs (MAP binding).
pub fn integer_bind(a: &[i32], b: &[i32]) -> HdcResult<Vec<i32>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let result: Vec<i32> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect();
    Ok(result)
}

/// Element-wise add (superposition / bundling).
pub fn integer_bundle(hvs: &[Vec<i32>]) -> HdcResult<Vec<i32>> {
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

/// Normalize to ±1 via sign (0 → random via rng).
pub fn integer_to_binary(hv: &[i32], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let result: Vec<i8> = hv
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

/// L2 norm of an integer HV.
pub fn integer_norm(hv: &[i32]) -> f64 {
    let sum_sq: i64 = hv.iter().map(|&v| (v as i64) * (v as i64)).sum();
    (sum_sq as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn random_integer_range() {
        let mut rng = LcgRng::new(10);
        let hv = random_integer(1000, &mut rng).expect("random_integer failed");
        assert_eq!(hv.len(), 1000);
        assert!(hv.iter().all(|&v| v == -1 || v == 0 || v == 1));
    }

    #[test]
    fn integer_bind_self() {
        let mut rng = LcgRng::new(11);
        let hv = random_integer(100, &mut rng).expect("failed");
        let bound = integer_bind(&hv, &hv).expect("bind failed");
        // v*v is 0 or 1 for values in {-1,0,+1}
        assert!(bound.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn integer_norm_all_ones() {
        let hv: Vec<i32> = vec![1; 100];
        let n = integer_norm(&hv);
        assert!((n - 10.0_f64).abs() < 1e-9);
    }
}
