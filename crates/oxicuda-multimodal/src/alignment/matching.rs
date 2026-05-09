//! Image-Text Matching (ITM) binary classification head.
//!
//! A two-layer MLP classifier that takes a fused image-text representation
//! and produces a binary match/no-match logit.

use crate::error::{MmResult, MultiModalError};

// ─── ITM head ─────────────────────────────────────────────────────────────────

/// 2-layer MLP classification head for Image-Text Matching.
///
/// Architecture: `linear(d_in, d_hidden) → ReLU → linear(d_hidden, 1)`
#[derive(Debug, Clone)]
pub struct ItmHead {
    /// W1: `[d_in × d_hidden]`
    pub w1: Vec<f32>,
    /// b1: `[d_hidden]`
    pub b1: Vec<f32>,
    /// W2: `[d_hidden × 1]` (single logit output)
    pub w2: Vec<f32>,
    /// b2: scalar bias for the logit
    pub b2: f32,
    pub d_in: usize,
    pub d_hidden: usize,
}

impl ItmHead {
    /// Create with zero weights.
    #[must_use]
    pub fn zeros(d_in: usize, d_hidden: usize) -> Self {
        Self {
            w1: vec![0.0_f32; d_in * d_hidden],
            b1: vec![0.0_f32; d_hidden],
            w2: vec![0.0_f32; d_hidden],
            b2: 0.0,
            d_in,
            d_hidden,
        }
    }

    /// Forward: `x [d_in]` → `logit (scalar)`.
    pub fn forward_single(&self, x: &[f32]) -> MmResult<f32> {
        if x.len() != self.d_in {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_in,
                got: x.len(),
            });
        }
        // Hidden: h = ReLU(W1 · x + b1)
        let mut h = vec![0.0_f32; self.d_hidden];
        for j in 0..self.d_hidden {
            let mut acc = self.b1[j];
            for i in 0..self.d_in {
                acc += x[i] * self.w1[i * self.d_hidden + j];
            }
            h[j] = acc.max(0.0); // ReLU
        }
        // Output logit
        let mut logit = self.b2;
        for j in 0..self.d_hidden {
            logit += h[j] * self.w2[j];
        }
        Ok(logit)
    }

    /// Batched forward: `x [batch × d_in]` → `logits [batch]`.
    pub fn forward(&self, x: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if x.len() != batch * self.d_in {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_in,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch);
        for bi in 0..batch {
            out.push(self.forward_single(&x[bi * self.d_in..(bi + 1) * self.d_in])?);
        }
        Ok(out)
    }
}

// ─── ITM loss ────────────────────────────────────────────────────────────────

/// Sigmoid binary cross-entropy loss for ITM.
///
/// `loss = -mean_i( labels[i] * log(σ(logits[i])) + (1-labels[i]) * log(1-σ(logits[i])) )`
///
/// Uses numerically stable log-sigmoid formulation:
/// `log(σ(x)) = -log(1 + exp(-x))` for positive values.
pub fn itm_loss(logits: &[f32], labels: &[f32]) -> MmResult<f32> {
    if logits.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    if logits.len() != labels.len() {
        return Err(MultiModalError::DimensionMismatch {
            expected: logits.len(),
            got: labels.len(),
        });
    }

    let n = logits.len();
    let mut total = 0.0_f32;
    for (&x, &y) in logits.iter().zip(labels.iter()) {
        // Numerically stable BCE:
        // loss = max(x, 0) - x*y + log(1 + exp(-|x|))
        let bce = x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln();
        total += bce;
    }
    let loss = total / n as f32;
    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "itm_loss",
        });
    }
    Ok(loss)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn itm_head_output_scalar() {
        let head = ItmHead::zeros(8, 16);
        let x = vec![1.0_f32; 8];
        let logit = head.forward_single(&x).unwrap();
        assert!(logit.is_finite());
    }

    #[test]
    fn itm_head_batched_shape() {
        let head = ItmHead::zeros(8, 16);
        let x = vec![0.5_f32; 4 * 8];
        let logits = head.forward(&x, 4).unwrap();
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn itm_head_zero_weights_zero_logit() {
        let head = ItmHead::zeros(8, 4);
        let x = vec![1.0_f32; 8];
        let logit = head.forward_single(&x).unwrap();
        assert!((logit - 0.0).abs() < 1e-6);
    }

    #[test]
    fn itm_head_dimension_error() {
        let head = ItmHead::zeros(8, 4);
        let x = vec![0.0_f32; 7]; // wrong size
        let err = head.forward_single(&x).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn itm_loss_perfect_positive() {
        // Very large positive logit for label=1 → near-zero loss
        let logits = vec![100.0_f32, 100.0];
        let labels = vec![1.0_f32, 1.0];
        let loss = itm_loss(&logits, &labels).unwrap();
        assert!(loss < 0.01, "loss should be near zero: {loss}");
    }

    #[test]
    fn itm_loss_perfect_negative() {
        // Very large negative logit for label=0 → near-zero loss
        let logits = vec![-100.0_f32, -100.0];
        let labels = vec![0.0_f32, 0.0];
        let loss = itm_loss(&logits, &labels).unwrap();
        assert!(loss < 0.01, "loss should be near zero: {loss}");
    }

    #[test]
    fn itm_loss_nonnegative() {
        let logits = vec![0.1_f32, -0.2, 0.5, -1.0];
        let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
        let loss = itm_loss(&logits, &labels).unwrap();
        assert!(loss >= 0.0, "BCE loss must be >= 0, got {loss}");
    }

    #[test]
    fn itm_loss_random_labels_finite() {
        let logits: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        let labels: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let loss = itm_loss(&logits, &labels).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn itm_loss_empty_input() {
        let err = itm_loss(&[], &[]).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn itm_loss_length_mismatch() {
        let err = itm_loss(&[1.0_f32, 2.0], &[0.0_f32]).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }
}
