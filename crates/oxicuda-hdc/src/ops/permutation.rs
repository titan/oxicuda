//! Permutation operations: cyclic shift, random permutation for sequential encoding.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Cyclic shift left by k positions (default: k=1 for sequential encoding).
/// `out[i] = hv[(i + k) % dim]`
pub fn cyclic_shift(hv: &[i8], k: usize) -> HdcResult<Vec<i8>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let k_eff = k % dim;
    let mut out = vec![0i8; dim];
    for i in 0..dim {
        out[i] = hv[(i + k_eff) % dim];
    }
    Ok(out)
}

/// Cyclic shift left for i32 HVs.
pub fn cyclic_shift_i32(hv: &[i32], k: usize) -> HdcResult<Vec<i32>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let k_eff = k % dim;
    let mut out = vec![0i32; dim];
    for i in 0..dim {
        out[i] = hv[(i + k_eff) % dim];
    }
    Ok(out)
}

/// Cyclic shift left for f32 HVs (used for complex interleaved or real HVs).
pub fn cyclic_shift_f32(hv: &[f32], k: usize) -> HdcResult<Vec<f32>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let k_eff = k % dim;
    let mut out = vec![0f32; dim];
    for i in 0..dim {
        out[i] = hv[(i + k_eff) % dim];
    }
    Ok(out)
}

/// Cyclic shift right by k positions.
/// `out[i] = hv[(i + dim - k) % dim]`
pub fn cyclic_shift_right(hv: &[i8], k: usize) -> HdcResult<Vec<i8>> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let k_eff = k % dim;
    cyclic_shift(hv, dim - k_eff)
}

/// Apply a permutation to a binary HV.
/// `perm[i]` is the source index for output position i.
pub fn random_permute(hv: &[i8], perm: &[usize]) -> HdcResult<Vec<i8>> {
    if hv.len() != perm.len() {
        return Err(HdcError::PermutationLengthMismatch {
            perm_len: perm.len(),
            dim: hv.len(),
        });
    }
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let mut out = vec![0i8; dim];
    for (i, &src) in perm.iter().enumerate() {
        if src >= dim {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: src,
                max: dim,
            });
        }
        out[i] = hv[src];
    }
    Ok(out)
}

/// Generate a random permutation of [0..dim] using Fisher-Yates shuffle with LcgRng.
pub fn random_permutation(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<usize>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut perm: Vec<usize> = (0..dim).collect();
    for i in (1..dim).rev() {
        let j = rng.next_usize(i + 1);
        perm.swap(i, j);
    }
    Ok(perm)
}

/// Apply the inverse of a permutation to a binary HV.
/// If `perm[i] = j`, inverse maps j back to i.
pub fn inverse_permute(hv: &[i8], perm: &[usize]) -> HdcResult<Vec<i8>> {
    if hv.len() != perm.len() {
        return Err(HdcError::PermutationLengthMismatch {
            perm_len: perm.len(),
            dim: hv.len(),
        });
    }
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let mut inv_perm = vec![0usize; dim];
    for (i, &p) in perm.iter().enumerate() {
        if p >= dim {
            return Err(HdcError::FeatureIndexOutOfRange { feat: p, max: dim });
        }
        inv_perm[p] = i;
    }
    let mut out = vec![0i8; dim];
    for i in 0..dim {
        out[i] = hv[inv_perm[i]];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn cyclic_shift_left_then_right_recovers() {
        let hv: Vec<i8> = vec![1, -1, 1, 1, -1, 1, -1, 1];
        let shifted = cyclic_shift(&hv, 3).expect("shift");
        let recovered = cyclic_shift_right(&shifted, 3).expect("shift_right");
        assert_eq!(recovered, hv);
    }

    #[test]
    fn permutation_and_inverse_roundtrip() {
        let mut rng = LcgRng::new(40);
        let hv: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let perm = random_permutation(8, &mut rng).expect("perm");
        let permuted = random_permute(&hv, &perm).expect("permute");
        let recovered = inverse_permute(&permuted, &perm).expect("inv_permute");
        assert_eq!(recovered, hv);
    }
}
