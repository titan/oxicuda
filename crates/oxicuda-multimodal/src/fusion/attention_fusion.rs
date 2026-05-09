//! Attention-gated modality fusion.
//!
//! Learns a soft attention distribution over modality embeddings:
//! `out = sum_m alpha_m * modalities[m]`
//! where `alpha = softmax(W * [m_0; m_1; ...; m_M] + b)`.

use crate::error::{MmResult, MultiModalError};

// ─── AttentionFusion ─────────────────────────────────────────────────────────

/// Attention-gated combination of multiple modality embeddings.
///
/// Each modality provides a `[d_model]` vector. A learned linear layer maps
/// each modality embedding to a scalar score, then a softmax is applied to
/// produce attention weights.
#[derive(Debug, Clone)]
pub struct AttentionFusion {
    /// Number of modalities.
    pub n_modalities: usize,
    /// Feature dimension of each modality.
    pub d_model: usize,
    /// Attention weight matrix: `[d_model × n_modalities]`.
    pub w_attn: Vec<f32>,
    /// Attention bias: `[n_modalities]`.
    pub b_attn: Vec<f32>,
}

impl AttentionFusion {
    /// Create with zero weights (equal attention after softmax).
    pub fn zeros(n_modalities: usize, d_model: usize) -> MmResult<Self> {
        if n_modalities < 2 {
            return Err(MultiModalError::InvalidModalityCount { n: n_modalities });
        }
        if d_model == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(Self {
            n_modalities,
            d_model,
            w_attn: vec![0.0_f32; d_model * n_modalities],
            b_attn: vec![0.0_f32; n_modalities],
        })
    }

    /// Compute attention weights for a single set of modality embeddings.
    ///
    /// `modalities`: slice of `n_modalities` references, each `[d_model]`.
    ///
    /// Returns `(weights [n_modalities], fused [d_model])`.
    pub fn forward(&self, modalities: &[&[f32]]) -> MmResult<(Vec<f32>, Vec<f32>)> {
        if modalities.len() != self.n_modalities {
            return Err(MultiModalError::InvalidModalityCount {
                n: modalities.len(),
            });
        }
        for (m, modal) in modalities.iter().enumerate() {
            if modal.len() != self.d_model {
                return Err(MultiModalError::DimensionMismatch {
                    expected: self.d_model,
                    got: modal.len(),
                });
            }
            // Validate finiteness
            if modal.iter().any(|v| !v.is_finite()) {
                return Err(MultiModalError::NanEncountered {
                    location: "modality input",
                });
            }
            let _ = m;
        }

        // Compute attention logits: score[m] = sum_d W[d,m] * modalities[m][d] + b[m]
        let mut logits = self.b_attn.clone();
        for m in 0..self.n_modalities {
            for d in 0..self.d_model {
                logits[m] += modalities[m][d] * self.w_attn[d * self.n_modalities + m];
            }
        }

        // Softmax over modalities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        let mut weights: Vec<f32> = logits
            .iter()
            .map(|&l| {
                let e = (l - max_logit).exp();
                exp_sum += e;
                e
            })
            .collect();
        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 1.0 };
        for w in weights.iter_mut() {
            *w *= inv_sum;
        }

        // Weighted sum of modalities
        let mut fused = vec![0.0_f32; self.d_model];
        for m in 0..self.n_modalities {
            let alpha = weights[m];
            for d in 0..self.d_model {
                fused[d] += alpha * modalities[m][d];
            }
        }

        Ok((weights, fused))
    }

    /// Batched forward: list of modality matrices, each `[batch × d_model]`.
    ///
    /// Returns `(weights [batch × n_modalities], fused [batch × d_model])`.
    pub fn forward_batch(
        &self,
        modalities: &[Vec<f32>],
        batch: usize,
    ) -> MmResult<(Vec<f32>, Vec<f32>)> {
        if modalities.len() != self.n_modalities {
            return Err(MultiModalError::InvalidModalityCount {
                n: modalities.len(),
            });
        }
        for modal in modalities {
            if modal.len() != batch * self.d_model {
                return Err(MultiModalError::DimensionMismatch {
                    expected: batch * self.d_model,
                    got: modal.len(),
                });
            }
        }

        let mut all_weights = vec![0.0_f32; batch * self.n_modalities];
        let mut all_fused = vec![0.0_f32; batch * self.d_model];

        for bi in 0..batch {
            let modal_refs: Vec<&[f32]> = modalities
                .iter()
                .map(|m| &m[bi * self.d_model..(bi + 1) * self.d_model])
                .collect();

            let (w, f) = self.forward(&modal_refs)?;
            all_weights[bi * self.n_modalities..(bi + 1) * self.n_modalities].copy_from_slice(&w);
            all_fused[bi * self.d_model..(bi + 1) * self.d_model].copy_from_slice(&f);
        }

        Ok((all_weights, all_fused))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_fusion_weights_sum_to_one() {
        let af = AttentionFusion::zeros(3, 8).unwrap();
        let m0 = vec![1.0_f32; 8];
        let m1 = vec![2.0_f32; 8];
        let m2 = vec![0.5_f32; 8];
        let (weights, _fused) = af.forward(&[&m0, &m1, &m2]).unwrap();
        let s: f32 = weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "weights sum = {s}");
    }

    #[test]
    fn attention_fusion_fused_shape() {
        let af = AttentionFusion::zeros(2, 8).unwrap();
        let m0 = vec![0.0_f32; 8];
        let m1 = vec![1.0_f32; 8];
        let (_w, fused) = af.forward(&[&m0, &m1]).unwrap();
        assert_eq!(fused.len(), 8);
    }

    #[test]
    fn attention_fusion_zero_weights_uniform() {
        // With zero attention weights: all logits equal → uniform weights
        let af = AttentionFusion::zeros(4, 8).unwrap();
        let modals: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 8]).collect();
        let refs: Vec<&[f32]> = modals.iter().map(|v| v.as_slice()).collect();
        let (weights, _) = af.forward(&refs).unwrap();
        for &w in &weights {
            assert!((w - 0.25).abs() < 1e-6, "expected 0.25, got {w}");
        }
    }

    #[test]
    fn attention_fusion_weighted_sum_correct() {
        // With uniform weights, fused = mean of modalities
        let af = AttentionFusion::zeros(2, 4).unwrap();
        let m0 = vec![2.0_f32; 4];
        let m1 = vec![4.0_f32; 4];
        let (_w, fused) = af.forward(&[&m0, &m1]).unwrap();
        // Expected: 0.5 * 2 + 0.5 * 4 = 3
        for &v in &fused {
            assert!((v - 3.0).abs() < 1e-6, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn attention_fusion_invalid_modality_count() {
        let err = AttentionFusion::zeros(1, 8).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidModalityCount { .. }));
    }

    #[test]
    fn attention_fusion_invalid_feature_dim() {
        let err = AttentionFusion::zeros(2, 0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn attention_fusion_batch_correct() {
        let af = AttentionFusion::zeros(2, 4).unwrap();
        let m0 = vec![1.0_f32; 2 * 4];
        let m1 = vec![3.0_f32; 2 * 4];
        let (w, f) = af.forward_batch(&[m0, m1], 2).unwrap();
        assert_eq!(w.len(), 2 * 2);
        assert_eq!(f.len(), 2 * 4);
        // All batch weights sum to 1
        for bi in 0..2 {
            let s: f32 = w[bi * 2..(bi + 1) * 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn attention_fusion_finite_output() {
        let mut af = AttentionFusion::zeros(3, 8).unwrap();
        for (i, w) in af.w_attn.iter_mut().enumerate() {
            *w = (i as f32 * 0.1).sin();
        }
        let refs: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..8).map(|d| (i * d) as f32 * 0.1).collect())
            .collect();
        let refs_slices: Vec<&[f32]> = refs.iter().map(|v| v.as_slice()).collect();
        let (w, f) = af.forward(&refs_slices).unwrap();
        assert!(w.iter().all(|v| v.is_finite()));
        assert!(f.iter().all(|v| v.is_finite()));
    }
}
