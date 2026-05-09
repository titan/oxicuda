//! Predictive uncertainty decomposition: entropy, mutual information (BALD),
//! and the aleatoric / epistemic split.
//!
//! For a classifier with posterior `p(θ|D)` and inputs `x*`, the predictive
//! distribution and its decomposition are
//! ```text
//!   p̄(y|x*) = E_θ[p(y|x*,θ)]                    // mean predictive
//!   H[p̄] = aleatoric + epistemic
//!   epistemic = I[y, θ | x*, D]
//!             = H[p̄] − E_θ[H[p(y|x*,θ)]]      // BALD score (Houlsby 2011)
//!   aleatoric = E_θ[H[p(y|x*,θ)]]              // mean per-sample entropy
//! ```
//!
//! All routines accept `[M × K]` row-major Monte-Carlo samples (e.g. from
//! [`super::deep_ensemble::DeepEnsemble`] or [`super::mc_dropout`]).

use crate::error::{BayesError, BayesResult};

/// Stable per-sample entropy: `−Σ_k p_k log p_k`. NaNs are treated as zero
/// contributions; `log` is floored at `1e-30`.
fn row_entropy(row: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in row {
        if p > 1e-30 {
            h -= p * p.ln();
        }
    }
    h
}

/// Per-sample mean predictive distribution `p̄(y) = (1/M) Σ_m p_m(y)`.
fn mean_predictive(samples: &[f32], n_classes: usize, n_members: usize) -> Vec<f32> {
    let mut p = vec![0.0_f64; n_classes];
    for m in 0..n_members {
        let row = &samples[m * n_classes..(m + 1) * n_classes];
        for (acc, &v) in p.iter_mut().zip(row.iter()) {
            *acc += v as f64;
        }
    }
    let inv_m = 1.0_f64 / n_members as f64;
    p.iter().map(|v| (*v * inv_m) as f32).collect()
}

fn check_shape(samples: &[f32], n_classes: usize, n_members: usize) -> BayesResult<()> {
    if samples.is_empty() || n_classes == 0 || n_members == 0 {
        return Err(BayesError::EmptyInputs);
    }
    if samples.len() != n_classes * n_members {
        return Err(BayesError::DimensionMismatch {
            expected: n_classes * n_members,
            got: samples.len(),
        });
    }
    Ok(())
}

/// Total predictive entropy `H[p̄]` for a single example given `M` Monte-Carlo
/// probability samples in row-major `[M × K]` layout.
///
/// # Errors
/// - [`BayesError::EmptyInputs`] for empty or zero-shape input.
/// - [`BayesError::DimensionMismatch`] for shape mismatches.
pub fn predictive_entropy(samples: &[f32], n_classes: usize, n_members: usize) -> BayesResult<f32> {
    check_shape(samples, n_classes, n_members)?;
    Ok(row_entropy(&mean_predictive(samples, n_classes, n_members)))
}

/// Aleatoric entropy `E_m[H[p_m]]` — average per-sample entropy.
///
/// # Errors
/// See [`predictive_entropy`].
pub fn aleatoric_entropy(samples: &[f32], n_classes: usize, n_members: usize) -> BayesResult<f32> {
    check_shape(samples, n_classes, n_members)?;
    let inv_m = 1.0_f32 / n_members as f32;
    let mut sum = 0.0_f32;
    for m in 0..n_members {
        sum += row_entropy(&samples[m * n_classes..(m + 1) * n_classes]);
    }
    Ok(sum * inv_m)
}

/// Mutual information / BALD score `I[y, θ] = H[p̄] − E_m[H[p_m]]`.
///
/// Always non-negative for valid probability distributions.
///
/// # Errors
/// See [`predictive_entropy`].
pub fn mutual_information(samples: &[f32], n_classes: usize, n_members: usize) -> BayesResult<f32> {
    let total = predictive_entropy(samples, n_classes, n_members)?;
    let aleatoric = aleatoric_entropy(samples, n_classes, n_members)?;
    Ok((total - aleatoric).max(0.0))
}

/// Convenience alias for [`mutual_information`].
///
/// # Errors
/// See [`predictive_entropy`].
pub fn epistemic_entropy(samples: &[f32], n_classes: usize, n_members: usize) -> BayesResult<f32> {
    mutual_information(samples, n_classes, n_members)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_two_classes_log2() {
        // Single member; uniform distribution gives ln(2)
        let s = vec![0.5_f32, 0.5];
        let h = predictive_entropy(&s, 2, 1).unwrap();
        assert!((h - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn entropy_certain_zero() {
        let s = vec![1.0_f32, 0.0];
        let h = predictive_entropy(&s, 2, 1).unwrap();
        assert!(h.abs() < 1e-5);
    }

    #[test]
    fn aleatoric_average_of_member_entropies() {
        // member1 = uniform → H = ln 2; member2 = certain → H = 0; mean = ln(2)/2
        let s = vec![0.5_f32, 0.5, 1.0, 0.0];
        let h = aleatoric_entropy(&s, 2, 2).unwrap();
        let expected = std::f32::consts::LN_2 / 2.0;
        assert!((h - expected).abs() < 1e-5);
    }

    #[test]
    fn bald_zero_when_members_agree() {
        let s = vec![0.7_f32, 0.3, 0.7, 0.3];
        let mi = mutual_information(&s, 2, 2).unwrap();
        assert!(mi.abs() < 1e-5);
    }

    #[test]
    fn bald_positive_when_members_disagree() {
        // member1 says (1, 0), member2 says (0, 1). Mean = (0.5, 0.5).
        // H[p̄] = ln 2; aleatoric = 0; MI = ln 2.
        let s = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mi = mutual_information(&s, 2, 2).unwrap();
        assert!((mi - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn epistemic_alias_matches_mutual_information() {
        let s = vec![1.0_f32, 0.0, 0.0, 1.0];
        let a = mutual_information(&s, 2, 2).unwrap();
        let b = epistemic_entropy(&s, 2, 2).unwrap();
        assert!((a - b).abs() < 1e-7);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let s = vec![0.5_f32, 0.5];
        let r = predictive_entropy(&s, 3, 1);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_zero_classes() {
        let s = vec![0.5_f32];
        assert!(predictive_entropy(&s, 0, 1).is_err());
    }

    #[test]
    fn rejects_zero_members() {
        let s = vec![0.5_f32, 0.5];
        assert!(predictive_entropy(&s, 2, 0).is_err());
    }

    #[test]
    fn rejects_empty() {
        let s: Vec<f32> = vec![];
        assert!(predictive_entropy(&s, 0, 0).is_err());
    }
}
