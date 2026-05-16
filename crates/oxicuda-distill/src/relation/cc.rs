//! CC — Correlation Congruence (Peng et al. 2019) — Gram matrix alignment.

use crate::error::{DistillError, DistillResult};

/// Compute the d×d Gram matrix G = Fᵀ F where F ∈ ℝⁿˣᵈ.
///
/// Returns a flat row-major `d × d` matrix: `G[i,j] = Σ_k F[k,i] · F[k,j]`.
#[must_use]
pub fn gram_matrix(feats: &[Vec<f32>]) -> Vec<f32> {
    let n = feats.len();
    let d = feats.first().map(|r| r.len()).unwrap_or(0);
    let mut g = vec![0.0_f32; d * d];
    for feat_row in feats.iter().take(n) {
        for i in 0..d {
            for j in 0..d {
                let fi = if i < feat_row.len() { feat_row[i] } else { 0.0 };
                let fj = if j < feat_row.len() { feat_row[j] } else { 0.0 };
                g[i * d + j] += fi * fj;
            }
        }
    }
    g
}

/// Frobenius squared distance: Σ (a_i − b_i)².
#[must_use]
pub fn frobenius_norm_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum()
}

/// Correlation Congruence loss: ‖G_s/n − G_t/n‖_F².
pub fn cc_loss(s_feats: &[Vec<f32>], t_feats: &[Vec<f32>]) -> DistillResult<f32> {
    if s_feats.is_empty() || t_feats.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_feats.len() != t_feats.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_feats.len(),
            got: t_feats.len(),
        });
    }
    let n = s_feats.len() as f32;
    let g_s: Vec<f32> = gram_matrix(s_feats).into_iter().map(|v| v / n).collect();
    let g_t: Vec<f32> = gram_matrix(t_feats).into_iter().map(|v| v / n).collect();
    Ok(frobenius_norm_sq(&g_s, &g_t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_symmetric() {
        let feats: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let g = gram_matrix(&feats);
        let d = 2;
        for i in 0..d {
            for j in 0..d {
                assert!((g[i * d + j] - g[j * d + i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn cc_loss_identical_is_zero() {
        let feats: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let loss = cc_loss(&feats, &feats).unwrap();
        assert!(loss < 1e-5);
    }
}
