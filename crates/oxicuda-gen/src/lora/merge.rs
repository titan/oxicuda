//! LoRA weight merging and composition utilities.
//!
//! Provides functions to merge LoRA adapters into base weights,
//! unmerge them, verify roundtrip accuracy, scale adapters, and
//! compose multiple adapters.

use crate::error::{GenError, GenResult};
use crate::lora::adapter::LoraLinear;

// ─── merge_lora ───────────────────────────────────────────────────────────────

/// Merge a LoRA adapter into the base weight matrix.
///
/// `W_merged = W_0 + (α/r) * B @ A`
///
/// # Arguments
/// - `base`: The base weight `W_0` of shape `[out_features × in_features]` (flat).
/// - `lora`: The LoRA adapter.
///
/// # Errors
/// - `DimensionMismatch` if `base.len() != lora.out_features() * lora.in_features()`
/// - `EmptyInput` if `base` is empty
pub fn merge_lora(base: &[f32], lora: &LoraLinear) -> GenResult<Vec<f32>> {
    if base.is_empty() {
        return Err(GenError::EmptyInput("base weight is empty"));
    }
    let expected = lora.out_features() * lora.in_features();
    if base.len() != expected {
        return Err(GenError::DimensionMismatch {
            expected,
            got: base.len(),
        });
    }
    let delta = lora.delta_weight(); // [out × in]
    let scale = lora.scaling();
    let merged = base
        .iter()
        .zip(&delta)
        .map(|(&w, &d)| w + scale * d)
        .collect();
    Ok(merged)
}

// ─── unmerge_lora ─────────────────────────────────────────────────────────────

/// Unmerge a LoRA adapter from a merged weight matrix.
///
/// `W_0 = W_merged - (α/r) * B @ A`
///
/// # Errors
/// - `DimensionMismatch` if shapes don't match
/// - `EmptyInput` if inputs are empty
pub fn unmerge_lora(merged: &[f32], lora: &LoraLinear) -> GenResult<Vec<f32>> {
    if merged.is_empty() {
        return Err(GenError::EmptyInput("merged weight is empty"));
    }
    let expected = lora.out_features() * lora.in_features();
    if merged.len() != expected {
        return Err(GenError::DimensionMismatch {
            expected,
            got: merged.len(),
        });
    }
    let delta = lora.delta_weight();
    let scale = lora.scaling();
    let unmerged = merged
        .iter()
        .zip(&delta)
        .map(|(&w, &d)| w - scale * d)
        .collect();
    Ok(unmerged)
}

// ─── verify_merge_roundtrip ───────────────────────────────────────────────────

/// Verify the merge/unmerge roundtrip.
///
/// Checks that `||W_0 - unmerge(merge(W_0, L), L)||_∞ < tol`.
///
/// # Returns
/// `true` if the max absolute difference is below `tol`.
///
/// # Errors
/// - Propagates errors from `merge_lora` / `unmerge_lora`
pub fn verify_merge_roundtrip(base: &[f32], lora: &LoraLinear, tol: f32) -> GenResult<bool> {
    let merged = merge_lora(base, lora)?;
    let recovered = unmerge_lora(&merged, lora)?;
    let max_err = base
        .iter()
        .zip(&recovered)
        .map(|(&w, &r)| (w - r).abs())
        .fold(0.0_f32, f32::max);
    Ok(max_err < tol)
}

// ─── scale_adapter ────────────────────────────────────────────────────────────

/// Create a scaled copy of a LoRA adapter.
///
/// Produces a new `LoraLinear` where `B_new = scale * B` (and `A` unchanged).
/// This is equivalent to changing the effective scaling to `(α/r) * scale`.
///
/// # Arguments
/// - `lora`: The original adapter.
/// - `scale`: The multiplicative scale factor.
///
/// # Errors
/// - `DimensionMismatch` if `lora`'s own matrices are inconsistent with its
///   `rank` / `in_features` / `out_features` (only reachable if the adapter
///   was mutated into an invalid shape via [`LoraLinear::matrix_b_mut`]).
pub fn scale_adapter(lora: &LoraLinear, scale: f32) -> GenResult<LoraLinear> {
    let new_b: Vec<f32> = lora.matrix_b().iter().map(|&v| v * scale).collect();
    LoraLinear::from_matrices(
        lora.in_features(),
        lora.out_features(),
        lora.rank(),
        lora.scaling() * lora.rank() as f32, // recover alpha = scaling * rank
        lora.matrix_a().to_vec(),
        new_b,
    )
}

// ─── compose_adapters ─────────────────────────────────────────────────────────

/// Compose two LoRA adapters by adding their B matrices.
///
/// Approximate composition: `B_composed = B_1 + B_2` (exact when `A_1 = A_2`).
///
/// The result uses `A_1` and `B_1 + B_2`, and inherits the scaling of `lora1`.
///
/// # Errors
/// - `DimensionMismatch` if the adapters have different shapes
pub fn compose_adapters(lora1: &LoraLinear, lora2: &LoraLinear) -> GenResult<LoraLinear> {
    if lora1.in_features() != lora2.in_features() {
        return Err(GenError::DimensionMismatch {
            expected: lora1.in_features(),
            got: lora2.in_features(),
        });
    }
    if lora1.out_features() != lora2.out_features() {
        return Err(GenError::DimensionMismatch {
            expected: lora1.out_features(),
            got: lora2.out_features(),
        });
    }
    if lora1.rank() != lora2.rank() {
        return Err(GenError::DimensionMismatch {
            expected: lora1.rank(),
            got: lora2.rank(),
        });
    }
    // B_composed = B_1 + B_2
    let b_composed: Vec<f32> = lora1
        .matrix_b()
        .iter()
        .zip(lora2.matrix_b())
        .map(|(&a, &b)| a + b)
        .collect();
    LoraLinear::from_matrices(
        lora1.in_features(),
        lora1.out_features(),
        lora1.rank(),
        lora1.scaling() * lora1.rank() as f32,
        lora1.matrix_a().to_vec(),
        b_composed,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::lora::adapter::LoraConfig;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    fn make_lora(in_f: usize, out_f: usize, rank: usize) -> LoraLinear {
        let config = LoraConfig::new(rank, rank as f32).expect("new should succeed");
        let mut rng = make_rng();
        LoraLinear::new(in_f, out_f, &config, &mut rng).expect("new should succeed")
    }

    #[test]
    fn merge_unmerge_roundtrip() {
        let lora = make_lora(8, 16, 4);
        let base: Vec<f32> = (0..16 * 8).map(|i| i as f32 * 0.01).collect();
        let ok = verify_merge_roundtrip(&base, &lora, 1e-4)
            .expect("verify_merge_roundtrip should succeed");
        assert!(ok, "merge/unmerge roundtrip failed");
    }

    #[test]
    fn zero_b_merge_gives_identity() {
        let lora = make_lora(4, 8, 2); // B=0 after new()
        let base: Vec<f32> = (0..8 * 4).map(|i| i as f32).collect();
        let merged = merge_lora(&base, &lora).expect("merge_lora should succeed");
        for (&m, &b) in merged.iter().zip(&base) {
            assert!((m - b).abs() < 1e-5, "B=0 merge should not change base");
        }
    }

    #[test]
    fn zero_b_unmerge_gives_identity() {
        let lora = make_lora(4, 8, 2);
        let base: Vec<f32> = (0..8 * 4).map(|i| i as f32).collect();
        let unmerged = unmerge_lora(&base, &lora).expect("unmerge_lora should succeed");
        for (&u, &b) in unmerged.iter().zip(&base) {
            assert!((u - b).abs() < 1e-5, "B=0 unmerge should not change weight");
        }
    }

    #[test]
    fn scale_factor_applied() {
        // Create lora with nonzero B, then scale
        let config = LoraConfig::new(2, 2.0).expect("new should succeed");
        let mut rng = make_rng();
        let mut lora = LoraLinear::new(4, 8, &config, &mut rng).expect("new should succeed");
        // Set B to all-ones
        for v in lora.matrix_b_mut() {
            *v = 1.0;
        }
        let scaled =
            scale_adapter(&lora, 2.0).expect("scale_adapter should succeed for a valid adapter");
        // B in scaled should be 2.0
        for &v in scaled.matrix_b() {
            assert!((v - 2.0).abs() < 1e-5, "scaled B should be 2.0: {v}");
        }
    }

    #[test]
    fn compose_adapters_shape() {
        let lora1 = make_lora(8, 16, 4);
        let lora2 = make_lora(8, 16, 4);
        let composed = compose_adapters(&lora1, &lora2).expect("compose_adapters should succeed");
        assert_eq!(composed.in_features(), 8);
        assert_eq!(composed.out_features(), 16);
        assert_eq!(composed.rank(), 4);
    }

    #[test]
    fn compose_b_is_sum() {
        let config = LoraConfig::new(2, 2.0).expect("new should succeed");
        let mut rng = make_rng();
        let mut lora1 = LoraLinear::new(4, 4, &config, &mut rng).expect("new should succeed");
        let mut lora2 = LoraLinear::new(4, 4, &config, &mut rng).expect("new should succeed");
        // Set B1 = 1, B2 = 2
        for v in lora1.matrix_b_mut() {
            *v = 1.0;
        }
        for v in lora2.matrix_b_mut() {
            *v = 2.0;
        }
        let composed = compose_adapters(&lora1, &lora2).expect("compose_adapters should succeed");
        for &v in composed.matrix_b() {
            assert!((v - 3.0).abs() < 1e-5, "composed B should be 3.0: {v}");
        }
    }

    #[test]
    fn compose_dimension_mismatch() {
        let lora1 = make_lora(8, 16, 4);
        let lora2 = make_lora(4, 16, 4); // different in_features
        assert!(matches!(
            compose_adapters(&lora1, &lora2),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn merge_dimension_mismatch() {
        let lora = make_lora(8, 16, 4);
        let bad_base = vec![0.0_f32; 8 * 8]; // wrong shape
        assert!(matches!(
            merge_lora(&bad_base, &lora),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn merge_with_nonzero_b() {
        let config = LoraConfig::new(2, 2.0).expect("new should succeed");
        let mut rng = make_rng();
        let mut lora = LoraLinear::new(4, 4, &config, &mut rng).expect("new should succeed");
        // Set B and A to identity-like values
        for v in lora.matrix_b_mut() {
            *v = 0.01;
        }
        let base = vec![0.0_f32; 4 * 4];
        let merged = merge_lora(&base, &lora).expect("merge_lora should succeed");
        // With nonzero A and B, merged should differ from base
        let diff: f32 = merged
            .iter()
            .zip(&base)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff > 0.0, "merged should differ from base with nonzero B");
    }

    #[test]
    fn roundtrip_with_nonzero_b() {
        let config = LoraConfig::new(4, 4.0).expect("new should succeed");
        let mut rng = make_rng();
        let mut lora = LoraLinear::new(8, 16, &config, &mut rng).expect("new should succeed");
        for v in lora.matrix_b_mut() {
            *v = 0.01;
        }
        let base: Vec<f32> = (0..16 * 8).map(|i| (i as f32) * 0.001).collect();
        let ok = verify_merge_roundtrip(&base, &lora, 1e-4)
            .expect("verify_merge_roundtrip should succeed");
        assert!(ok, "roundtrip with nonzero B failed");
    }
}
