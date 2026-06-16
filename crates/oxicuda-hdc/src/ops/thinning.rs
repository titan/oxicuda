//! Context-Dependent Thinning (CDT) for binary ±1 hypervectors.
//!
//! CDT (Rachkovskij & Kussul, 2001) sparsifies a superposition (bundle) while
//! preserving its similarity structure and keeping the density of active bits
//! roughly constant. The classic *additive* CDT subsamples the active bits of an
//! input HV `z` in a content-dependent way:
//!
//! ```text
//! <z> = z ∧ ( ρ_1(z) ∨ ρ_2(z) ∨ … ∨ ρ_K(z) )
//! ```
//!
//! where `ρ_k` are permutations, `∧` is bitwise AND of active bits and `∨` is
//! bitwise OR. Because the result is `z` ANDed with something, every active bit
//! of `<z>` was active in `z` (it is a *subset*), so density never increases.
//!
//! # Representation
//! The crate stores binary HVs as `Vec<i8>` in `{-1, +1}`, where `+1` denotes an
//! **active** bit and `-1` an inactive bit. AND yields `+1` only when both
//! inputs are `+1`; OR yields `+1` when either input is `+1`. (The equivalent
//! `{0, 1}` reading: `+1 → 1`, `-1 → 0`.)
//!
//! # Permutation family
//! This implementation uses the `K` distinct cyclic shifts `1..=K` as the
//! permutation family (via [`crate::ops::permutation::cyclic_shift`]). This is a
//! standard deterministic CDT variant ("cyclic-shift CDT"): it needs no stored
//! permutation tables and is reproducible. An explicit-permutation entry point,
//! [`additive_thinning_with_perms`], is also provided.

use crate::error::{HdcError, HdcResult};
use crate::ops::permutation::cyclic_shift;
use crate::vector::binary::validate_binary;

/// Bitwise AND of two ±1 HVs: result is `+1` iff both inputs are `+1`.
///
/// # Errors
/// Returns [`HdcError::DimensionMismatch`] if lengths differ and
/// [`HdcError::EmptyInput`] if empty.
pub fn and_binary(a: &[i8], b: &[i8]) -> HdcResult<Vec<i8>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| if ai == 1 && bi == 1 { 1i8 } else { -1i8 })
        .collect())
}

/// Bitwise OR of two ±1 HVs: result is `+1` iff either input is `+1`.
///
/// # Errors
/// Returns [`HdcError::DimensionMismatch`] if lengths differ and
/// [`HdcError::EmptyInput`] if empty.
pub fn or_binary(a: &[i8], b: &[i8]) -> HdcResult<Vec<i8>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| if ai == 1 || bi == 1 { 1i8 } else { -1i8 })
        .collect())
}

/// Density of a binary ±1 HV: the fraction of `+1` (active) components.
#[must_use]
pub fn density(hv: &[i8]) -> f32 {
    if hv.is_empty() {
        return 0.0;
    }
    let active = hv.iter().filter(|&&v| v == 1).count();
    active as f32 / hv.len() as f32
}

/// Additive context-dependent thinning using an explicit permutation family.
///
/// Computes `z ∧ (ρ_1(z) ∨ … ∨ ρ_m(z))` where each `ρ_k` is given by `perms[k]`
/// (`out[i] = z[perms[k][i]]`). With no permutations the result is `z` unchanged.
///
/// # Errors
/// Returns [`HdcError::EmptyInput`] if `z` is empty,
/// [`HdcError::InvalidBinaryValue`] if `z` has a value outside `{-1, +1}`,
/// [`HdcError::PermutationLengthMismatch`] if any permutation length differs from
/// `z.len()`, and [`HdcError::FeatureIndexOutOfRange`] if a permutation entry
/// indexes outside `z`.
pub fn additive_thinning_with_perms(z: &[i8], perms: &[Vec<usize>]) -> HdcResult<Vec<i8>> {
    if z.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    validate_binary(z)?;
    if perms.is_empty() {
        return Ok(z.to_vec());
    }
    let dim = z.len();
    // OR-accumulate the permuted copies, starting from "all inactive".
    let mut or_acc = vec![-1i8; dim];
    for perm in perms {
        if perm.len() != dim {
            return Err(HdcError::PermutationLengthMismatch {
                perm_len: perm.len(),
                dim,
            });
        }
        for (i, &src) in perm.iter().enumerate() {
            if src >= dim {
                return Err(HdcError::FeatureIndexOutOfRange {
                    feat: src,
                    max: dim,
                });
            }
            if z[src] == 1 {
                or_acc[i] = 1;
            }
        }
    }
    and_binary(z, &or_acc)
}

/// Cyclic-shift context-dependent thinning of `z` with `k` shifts.
///
/// Uses the deterministic permutation family of cyclic shifts `1..=k` and
/// computes `z ∧ (shift_1(z) ∨ … ∨ shift_k(z))`. The output is a subset of the
/// active bits of `z`, so `density(<z>) <= density(z)`.
///
/// `k == 0` is treated as the identity and returns `z` unchanged (there is no
/// permutation to subsample with).
///
/// # Errors
/// Returns [`HdcError::EmptyInput`] if `z` is empty and
/// [`HdcError::InvalidBinaryValue`] if `z` has a value outside `{-1, +1}`.
pub fn context_dependent_thinning(z: &[i8], k: usize) -> HdcResult<Vec<i8>> {
    if z.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    validate_binary(z)?;
    if k == 0 {
        return Ok(z.to_vec());
    }
    let dim = z.len();
    let mut or_acc = vec![-1i8; dim];
    for shift in 1..=k {
        let permuted = cyclic_shift(z, shift)?;
        or_acc = or_binary(&or_acc, &permuted)?;
    }
    and_binary(z, &or_acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::cosine_binary;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    #[test]
    fn density_all_active_and_inactive() {
        let all_pos: Vec<i8> = vec![1i8; 16];
        let all_neg: Vec<i8> = vec![-1i8; 16];
        assert!((density(&all_pos) - 1.0).abs() < 1e-6);
        assert!((density(&all_neg) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn and_binary_correct() {
        let a: Vec<i8> = vec![1, 1, -1, -1];
        let b: Vec<i8> = vec![1, -1, 1, -1];
        let r = and_binary(&a, &b).expect("and");
        assert_eq!(r, vec![1, -1, -1, -1]);
    }

    #[test]
    fn or_binary_correct() {
        let a: Vec<i8> = vec![1, 1, -1, -1];
        let b: Vec<i8> = vec![1, -1, 1, -1];
        let r = or_binary(&a, &b).expect("or");
        assert_eq!(r, vec![1, 1, 1, -1]);
    }

    #[test]
    fn thinning_all_inactive_stays_inactive() {
        let z: Vec<i8> = vec![-1i8; 32];
        let t = context_dependent_thinning(&z, 3).expect("thin");
        assert!(t.iter().all(|&v| v == -1));
    }

    #[test]
    fn thinning_result_is_subset_of_active_bits() {
        let mut rng = LcgRng::new(101);
        let z = random_binary(256, &mut rng).expect("rand");
        let t = context_dependent_thinning(&z, 4).expect("thin");
        for i in 0..z.len() {
            if t[i] == 1 {
                assert_eq!(z[i], 1, "output bit {i} active but input inactive");
            }
        }
    }

    #[test]
    fn thinning_reduces_or_preserves_density() {
        let mut rng = LcgRng::new(202);
        let z = random_binary(512, &mut rng).expect("rand");
        let t = context_dependent_thinning(&z, 5).expect("thin");
        assert!(
            density(&t) <= density(&z) + 1e-6,
            "density increased: {} > {}",
            density(&t),
            density(&z)
        );
    }

    #[test]
    fn k_zero_returns_unchanged() {
        let mut rng = LcgRng::new(303);
        let z = random_binary(64, &mut rng).expect("rand");
        let t = context_dependent_thinning(&z, 0).expect("thin");
        assert_eq!(t, z);
    }

    #[test]
    fn empty_input_rejected() {
        let z: Vec<i8> = Vec::new();
        let err = context_dependent_thinning(&z, 3);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
        let err2 = additive_thinning_with_perms(&z, &[]);
        assert!(matches!(err2, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn dimension_mismatch_in_and_or_rejected() {
        let a: Vec<i8> = vec![1, -1, 1];
        let b: Vec<i8> = vec![1, -1];
        assert!(matches!(
            and_binary(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            or_binary(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn explicit_perm_variant_rejects_wrong_perm_length() {
        let z: Vec<i8> = vec![1, -1, 1, -1];
        let bad_perm = vec![vec![0usize, 1, 2]]; // length 3 != 4
        let err = additive_thinning_with_perms(&z, &bad_perm);
        assert!(matches!(
            err,
            Err(HdcError::PermutationLengthMismatch {
                perm_len: 3,
                dim: 4
            })
        ));
    }

    #[test]
    fn similarity_preserved_between_overlapping_inputs() {
        let mut rng = LcgRng::new(404);
        let dim = 1024;
        let base = random_binary(dim, &mut rng).expect("base");
        // Two inputs that overlap heavily with `base` (flip a few bits each).
        let mut x = base.clone();
        let mut y = base.clone();
        for i in 0..20 {
            x[i] = -x[i];
            y[dim - 1 - i] = -y[dim - 1 - i];
        }
        let unrelated = random_binary(dim, &mut rng).expect("unrelated");
        let tx = context_dependent_thinning(&x, 4).expect("tx");
        let ty = context_dependent_thinning(&y, 4).expect("ty");
        let tu = context_dependent_thinning(&unrelated, 4).expect("tu");
        let sim_xy = cosine_binary(&tx, &ty).expect("xy");
        let sim_xu = cosine_binary(&tx, &tu).expect("xu");
        assert!(
            sim_xy > sim_xu,
            "thinned overlap sim {sim_xy} not greater than unrelated sim {sim_xu}"
        );
    }

    #[test]
    fn determinism_same_input_same_output() {
        let mut rng = LcgRng::new(505);
        let z = random_binary(256, &mut rng).expect("rand");
        let a = context_dependent_thinning(&z, 4).expect("a");
        let b = context_dependent_thinning(&z, 4).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_binary_value_rejected() {
        let z: Vec<i8> = vec![1, 0, -1, 1];
        let err = context_dependent_thinning(&z, 2);
        assert!(matches!(err, Err(HdcError::InvalidBinaryValue(0))));
    }
}
