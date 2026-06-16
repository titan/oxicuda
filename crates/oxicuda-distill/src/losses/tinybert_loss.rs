//! TinyBERT distillation losses — Jiao et al., 2019.
//!
//! Provides MSE-based losses for aligning hidden states and attention matrices
//! between a large teacher BERT model and a compact student TinyBERT model.

use crate::error::{DistillError, DistillResult};

/// Configuration for [`TinyBertLoss`].
pub struct TinyBertLossConfig {
    /// Number of transformer layers to align.
    pub n_layers: usize,
    /// Hidden-state dimension `d_model`.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

/// TinyBERT hidden-state and attention-matrix distillation losses.
pub struct TinyBertLoss {
    config: TinyBertLossConfig,
}

impl TinyBertLoss {
    /// Construct a [`TinyBertLoss`] from `config`.
    ///
    /// # Errors
    /// Returns [`DistillError::InvalidConfig`] if any dimension is zero.
    pub fn new(config: TinyBertLossConfig) -> DistillResult<Self> {
        if config.n_layers == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_layers must be > 0".to_string(),
            });
        }
        if config.d_model == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "d_model must be > 0".to_string(),
            });
        }
        if config.n_heads == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_heads must be > 0".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Hidden-state distillation loss (MSE averaged over all elements).
    ///
    /// Both `student_hidden` and `teacher_hidden` must have length
    /// `n_layers × seq_len × d_model`.
    ///
    /// # Errors
    /// [`DistillError::DimensionMismatch`] when a slice has unexpected length.
    pub fn hidden_loss(
        &self,
        student_hidden: &[f32],
        teacher_hidden: &[f32],
        seq_len: usize,
    ) -> DistillResult<f32> {
        let expected = self.config.n_layers * seq_len * self.config.d_model;
        if student_hidden.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: student_hidden.len(),
            });
        }
        if teacher_hidden.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: teacher_hidden.len(),
            });
        }

        let mse = student_hidden
            .iter()
            .zip(teacher_hidden.iter())
            .map(|(&s, &t)| {
                let d = s - t;
                d * d
            })
            .sum::<f32>()
            / expected as f32;

        Ok(mse)
    }

    /// Attention-matrix distillation loss (MSE averaged over all elements).
    ///
    /// Both `student_attn` and `teacher_attn` must have length
    /// `n_layers × n_heads × seq_len × seq_len`.
    ///
    /// # Errors
    /// [`DistillError::DimensionMismatch`] when a slice has unexpected length.
    pub fn attention_loss(
        &self,
        student_attn: &[f32],
        teacher_attn: &[f32],
        seq_len: usize,
    ) -> DistillResult<f32> {
        let expected = self.config.n_layers * self.config.n_heads * seq_len * seq_len;
        if student_attn.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: student_attn.len(),
            });
        }
        if teacher_attn.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: teacher_attn.len(),
            });
        }

        let mse = student_attn
            .iter()
            .zip(teacher_attn.iter())
            .map(|(&s, &t)| {
                let d = s - t;
                d * d
            })
            .sum::<f32>()
            / expected as f32;

        Ok(mse)
    }

    /// Combined loss: `hidden_loss + attention_loss`.
    ///
    /// # Errors
    /// Propagates any [`DistillError`] from the component loss functions.
    pub fn total_loss(
        &self,
        student_h: &[f32],
        teacher_h: &[f32],
        student_a: &[f32],
        teacher_a: &[f32],
        seq_len: usize,
    ) -> DistillResult<f32> {
        let hl = self.hidden_loss(student_h, teacher_h, seq_len)?;
        let al = self.attention_loss(student_a, teacher_a, seq_len)?;
        Ok(hl + al)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_loss(n_layers: usize, d_model: usize, n_heads: usize) -> TinyBertLoss {
        TinyBertLoss::new(TinyBertLossConfig {
            n_layers,
            d_model,
            n_heads,
        })
        .expect("valid config")
    }

    // ── Test 1 ──────────────────────────────────────────────────────────
    /// `hidden_loss` must return a finite value.
    #[test]
    fn hidden_loss_finite() {
        let loss = make_loss(3, 8, 2);
        let seq = 4_usize;
        let len = 3 * seq * 8;
        let s: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
        let t: Vec<f32> = (0..len).map(|i| i as f32 * 0.09).collect();
        let v = loss.hidden_loss(&s, &t, seq).expect("ok");
        assert!(v.is_finite(), "hidden_loss is not finite: {v}");
    }

    // ── Test 2 ──────────────────────────────────────────────────────────
    /// `hidden_loss` must be non-negative (it is an MSE).
    #[test]
    fn hidden_loss_nonneg() {
        let loss = make_loss(2, 4, 2);
        let seq = 3_usize;
        let len = 2 * seq * 4;
        let s: Vec<f32> = (0..len).map(|i| (i % 5) as f32 - 2.0).collect();
        let t: Vec<f32> = (0..len).map(|i| (i % 7) as f32 - 3.0).collect();
        let v = loss.hidden_loss(&s, &t, seq).expect("ok");
        assert!(v >= 0.0, "hidden_loss must be non-negative, got {v}");
    }

    // ── Test 3 ──────────────────────────────────────────────────────────
    /// `attention_loss` must return a finite value.
    #[test]
    fn attn_loss_finite() {
        let loss = make_loss(2, 4, 2);
        let seq = 4_usize;
        let len = 2 * 2 * seq * seq;
        let s: Vec<f32> = (0..len).map(|i| i as f32 * 0.05).collect();
        let t: Vec<f32> = (0..len).map(|i| i as f32 * 0.04).collect();
        let v = loss.attention_loss(&s, &t, seq).expect("ok");
        assert!(v.is_finite(), "attn_loss is not finite: {v}");
    }

    // ── Test 4 ──────────────────────────────────────────────────────────
    /// `attention_loss` must be non-negative.
    #[test]
    fn attn_loss_nonneg() {
        let loss = make_loss(2, 4, 2);
        let seq = 3_usize;
        let len = 2 * 2 * seq * seq;
        let s: Vec<f32> = (0..len).map(|i| (i % 5) as f32).collect();
        let t: Vec<f32> = (0..len).map(|_| 0.5).collect();
        let v = loss.attention_loss(&s, &t, seq).expect("ok");
        assert!(v >= 0.0, "attn_loss must be non-negative, got {v}");
    }

    // ── Test 5 ──────────────────────────────────────────────────────────
    /// Identical student and teacher must yield zero for both component losses.
    #[test]
    fn loss_zero_identical() {
        let loss = make_loss(3, 8, 4);
        let seq = 5_usize;
        let h_len = 3 * seq * 8;
        let a_len = 3 * 4 * seq * seq;
        let h: Vec<f32> = (0..h_len).map(|i| i as f32 * 0.3).collect();
        let a: Vec<f32> = (0..a_len).map(|i| (i % 11) as f32 * 0.1).collect();
        let hl = loss.hidden_loss(&h, &h, seq).expect("ok");
        let al = loss.attention_loss(&a, &a, seq).expect("ok");
        assert!(hl < 1e-6, "identical hidden must give 0 loss, got {hl}");
        assert!(al < 1e-6, "identical attn must give 0 loss, got {al}");
    }

    // ── Test 6 ──────────────────────────────────────────────────────────
    /// `total_loss` must return a finite value.
    #[test]
    fn total_loss_finite() {
        let loss = make_loss(2, 4, 2);
        let seq = 4_usize;
        let h_len = 2 * seq * 4;
        let a_len = 2 * 2 * seq * seq;
        let sh: Vec<f32> = (0..h_len).map(|i| i as f32).collect();
        let th: Vec<f32> = (0..h_len).map(|i| i as f32 * 0.9).collect();
        let sa: Vec<f32> = (0..a_len).map(|i| i as f32 * 0.1).collect();
        let ta: Vec<f32> = (0..a_len).map(|i| i as f32 * 0.08).collect();
        let v = loss.total_loss(&sh, &th, &sa, &ta, seq).expect("ok");
        assert!(v.is_finite(), "total_loss is not finite: {v}");
    }

    // ── Test 7 ──────────────────────────────────────────────────────────
    /// A slice with wrong length must return [`DistillError::DimensionMismatch`].
    #[test]
    fn n_layers_mismatch() {
        let loss = make_loss(2, 4, 2);
        let seq = 3_usize;
        let correct_len = 2 * seq * 4;
        let wrong: Vec<f32> = vec![0.0; correct_len - 1];
        let right: Vec<f32> = vec![0.0; correct_len];
        let result = loss.hidden_loss(&wrong, &right, seq);
        assert!(
            result.is_err(),
            "wrong-length student slice must produce DimensionMismatch"
        );
    }

    // ── Test 8 ──────────────────────────────────────────────────────────
    /// `seq_len = 1` must work correctly.
    #[test]
    fn seq_len_1_ok() {
        let loss = make_loss(1, 4, 2);
        let seq = 1_usize;
        let h_len = seq * 4;
        let a_len = 2 * seq * seq;
        let sh = vec![1.0_f32; h_len];
        let th = vec![2.0_f32; h_len];
        let sa = vec![0.5_f32; a_len];
        let ta = vec![0.3_f32; a_len];
        let v = loss.total_loss(&sh, &th, &sa, &ta, seq).expect("ok");
        assert!(v.is_finite() && v >= 0.0);
    }

    // ── Test 9 ──────────────────────────────────────────────────────────
    /// Different student and teacher inputs must yield a positive loss.
    #[test]
    fn different_inputs_differ() {
        let loss = make_loss(2, 4, 2);
        let seq = 3_usize;
        let h_len = 2 * seq * 4;
        let a_len = 2 * 2 * seq * seq;
        let sh: Vec<f32> = (0..h_len).map(|i| i as f32).collect();
        let th: Vec<f32> = vec![100.0; h_len];
        let sa: Vec<f32> = (0..a_len).map(|i| i as f32 * 0.1).collect();
        let ta: Vec<f32> = vec![0.0; a_len];
        let v = loss.total_loss(&sh, &th, &sa, &ta, seq).expect("ok");
        assert!(v > 0.0, "different inputs must give positive loss, got {v}");
    }

    // ── Test 10 ─────────────────────────────────────────────────────────
    /// `n_layers = 0` must be rejected by the constructor.
    #[test]
    fn n_layers_0_error() {
        let result = TinyBertLoss::new(TinyBertLossConfig {
            n_layers: 0,
            d_model: 8,
            n_heads: 2,
        });
        assert!(result.is_err(), "n_layers=0 must produce an error");
    }
}
