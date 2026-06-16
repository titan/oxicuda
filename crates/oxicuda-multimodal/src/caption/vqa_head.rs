//! VQA multi-class classification head.
//!
//! Takes a fused image-question representation and produces logits over
//! a fixed answer vocabulary.

use crate::error::{MmResult, MultiModalError};

// ─── VqaHead ─────────────────────────────────────────────────────────────────

/// Two-layer MLP for VQA answer classification.
///
/// Architecture: `linear(d_in, d_hidden) → ReLU → linear(d_hidden, n_answers)`
#[derive(Debug, Clone)]
pub struct VqaHead {
    /// W1: `[d_in × d_hidden]`.
    pub w1: Vec<f32>,
    /// b1: `[d_hidden]`.
    pub b1: Vec<f32>,
    /// W2: `[d_hidden × n_answers]`.
    pub w2: Vec<f32>,
    /// b2: `[n_answers]`.
    pub b2: Vec<f32>,
    pub d_in: usize,
    pub d_hidden: usize,
    pub n_answers: usize,
}

impl VqaHead {
    /// Create with zero weights.
    pub fn zeros(d_in: usize, d_hidden: usize, n_answers: usize) -> MmResult<Self> {
        if d_in == 0 || d_hidden == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if n_answers == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(Self {
            w1: vec![0.0_f32; d_in * d_hidden],
            b1: vec![0.0_f32; d_hidden],
            w2: vec![0.0_f32; d_hidden * n_answers],
            b2: vec![0.0_f32; n_answers],
            d_in,
            d_hidden,
            n_answers,
        })
    }

    /// Forward on a single fused representation `[d_in]` → logits `[n_answers]`.
    pub fn forward(&self, fused_repr: &[f32]) -> MmResult<Vec<f32>> {
        if fused_repr.len() != self.d_in {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_in,
                got: fused_repr.len(),
            });
        }

        // Hidden: h = ReLU(W1 · x + b1)  [d_hidden]
        let mut h = vec![0.0_f32; self.d_hidden];
        for j in 0..self.d_hidden {
            let mut acc = self.b1[j];
            for i in 0..self.d_in {
                acc += fused_repr[i] * self.w1[i * self.d_hidden + j];
            }
            h[j] = acc.max(0.0); // ReLU
        }

        // Output logits: out = W2 · h + b2  [n_answers]
        let mut out = self.b2.clone();
        for a in 0..self.n_answers {
            for j in 0..self.d_hidden {
                out[a] += h[j] * self.w2[j * self.n_answers + a];
            }
        }
        Ok(out)
    }

    /// Batched forward: `x [batch × d_in]` → `logits [batch × n_answers]`.
    pub fn forward_batch(&self, x: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if x.len() != batch * self.d_in {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_in,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch * self.n_answers);
        for bi in 0..batch {
            let logits = self.forward(&x[bi * self.d_in..(bi + 1) * self.d_in])?;
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }
}

// ─── VQA loss ─────────────────────────────────────────────────────────────────

/// Cross-entropy loss for VQA.
///
/// `loss = -log(softmax(logits)[target])`.
pub fn vqa_loss(logits: &[f32], target: usize) -> MmResult<f32> {
    if logits.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    if target >= logits.len() {
        return Err(MultiModalError::TokenOutOfRange {
            token_id: target as u32,
            vocab_size: logits.len(),
        });
    }

    // Numerically stable: subtract max before exp
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();
    let log_sum = max_logit + sum_exp.ln();
    let loss = log_sum - logits[target];

    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "vqa_loss",
        });
    }
    Ok(loss)
}

/// Compute softmax of logits.
pub fn softmax(logits: &[f32]) -> MmResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    Ok(exps.iter().map(|&e| e * inv_sum).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vqa_head_output_shape() {
        let head = VqaHead::zeros(16, 32, 100).expect("zeros should succeed");
        let x = vec![0.5_f32; 16];
        let logits = head.forward(&x).expect("forward should succeed");
        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn vqa_head_batched_shape() {
        let head = VqaHead::zeros(8, 16, 50).expect("zeros should succeed");
        let x = vec![0.3_f32; 4 * 8];
        let logits = head
            .forward_batch(&x, 4)
            .expect("forward_batch should succeed");
        assert_eq!(logits.len(), 4 * 50);
    }

    #[test]
    fn vqa_head_zero_weights_bias_output() {
        let head = VqaHead::zeros(8, 4, 10).expect("zeros should succeed");
        let x = vec![1.0_f32; 8];
        let logits = head.forward(&x).expect("forward should succeed");
        // All logits should be 0 (bias=0, weights=0)
        for &v in &logits {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn vqa_softmax_sums_to_one() {
        let head = VqaHead::zeros(4, 8, 5).expect("zeros should succeed");
        let mut head_nonzero = head.clone();
        // Set non-zero weights
        for (i, w) in head_nonzero.w1.iter_mut().enumerate() {
            *w = (i as f32 * 0.1).sin();
        }
        for (i, w) in head_nonzero.w2.iter_mut().enumerate() {
            *w = (i as f32 * 0.13).cos();
        }
        let x = vec![0.5_f32; 4];
        let logits = head_nonzero.forward(&x).expect("forward should succeed");
        let probs = softmax(&logits).expect("softmax should succeed");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
    }

    #[test]
    fn vqa_loss_correct_answer_low_loss() {
        // If logit for correct answer is very high, loss ≈ 0
        let mut logits = vec![0.0_f32; 10];
        logits[3] = 100.0;
        let loss = vqa_loss(&logits, 3).expect("vqa_loss should succeed");
        assert!(loss < 0.01, "loss should be near zero: {loss}");
    }

    #[test]
    fn vqa_loss_nonnegative() {
        let logits: Vec<f32> = (0..5).map(|i| i as f32 * 0.5).collect();
        let loss = vqa_loss(&logits, 2).expect("vqa_loss should succeed");
        assert!(loss >= 0.0, "cross-entropy loss should be >= 0: {loss}");
    }

    #[test]
    fn vqa_loss_target_out_of_range() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let err = vqa_loss(&logits, 5).unwrap_err();
        assert!(matches!(err, MultiModalError::TokenOutOfRange { .. }));
    }

    #[test]
    fn vqa_loss_empty_logits() {
        let err = vqa_loss(&[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn softmax_output_shape_and_sum() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits).expect("softmax should succeed");
        assert_eq!(probs.len(), 4);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_max_gets_highest_prob() {
        let logits = vec![0.0_f32, 0.0, 10.0, 0.0];
        let probs = softmax(&logits).expect("softmax should succeed");
        assert!(
            probs[2] > 0.99,
            "max logit should get highest prob: {}",
            probs[2]
        );
    }

    #[test]
    fn vqa_head_invalid_params() {
        let err = VqaHead::zeros(0, 16, 10).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
        let err2 = VqaHead::zeros(8, 16, 0).unwrap_err();
        assert!(matches!(err2, MultiModalError::InvalidFeatureDim));
    }
}
