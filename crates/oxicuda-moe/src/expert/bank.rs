//! ExpertBank: collection of N experts with unified dispatch interface.

use crate::error::{MoeError, MoeResult};
use crate::expert::ffn::{ExpertActivation, ExpertFfn, SwiGluExpert};
use crate::handle::LcgRng;

/// A collection of standard FFN experts.
#[derive(Debug, Clone)]
pub struct ExpertBank {
    experts: Vec<ExpertFfn>,
    /// Number of experts.
    pub n_experts: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// FFN hidden dimension.
    pub ffn_dim: usize,
}

impl ExpertBank {
    /// Create a bank of `n_experts` FFN experts.
    pub fn new(
        n_experts: usize,
        input_dim: usize,
        ffn_dim: usize,
        act: ExpertActivation,
        rng: &mut LcgRng,
    ) -> MoeResult<Self> {
        if n_experts == 0 {
            return Err(MoeError::InvalidExpertCount { n_experts });
        }
        if input_dim == 0 {
            return Err(MoeError::InvalidInputDim { dim: input_dim });
        }
        if ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: ffn_dim });
        }
        let experts: Vec<ExpertFfn> = (0..n_experts)
            .map(|_| ExpertFfn::new(input_dim, ffn_dim, act, rng))
            .collect();
        Ok(Self {
            experts,
            n_experts,
            input_dim,
            ffn_dim,
        })
    }

    /// Build a bank from a pre-constructed set of experts (e.g. produced by
    /// sparse upcycling from a dense checkpoint).
    ///
    /// All experts must share identical `input_dim` and `ffn_dim`.
    ///
    /// # Errors
    /// Returns [`MoeError::InvalidExpertCount`] for an empty list and
    /// [`MoeError::DimensionMismatch`] when the experts' dimensions disagree.
    pub fn from_experts(experts: Vec<ExpertFfn>) -> MoeResult<Self> {
        let n_experts = experts.len();
        if n_experts == 0 {
            return Err(MoeError::InvalidExpertCount { n_experts });
        }
        let input_dim = experts[0].input_dim;
        let ffn_dim = experts[0].ffn_dim;
        for e in &experts {
            if e.input_dim != input_dim {
                return Err(MoeError::DimensionMismatch {
                    expected: input_dim,
                    got: e.input_dim,
                });
            }
            if e.ffn_dim != ffn_dim {
                return Err(MoeError::DimensionMismatch {
                    expected: ffn_dim,
                    got: e.ffn_dim,
                });
            }
        }
        Ok(Self {
            experts,
            n_experts,
            input_dim,
            ffn_dim,
        })
    }

    /// Immutable view of the contained experts.
    #[must_use]
    pub fn experts(&self) -> &[ExpertFfn] {
        &self.experts
    }

    /// Mutable view of the contained experts (e.g. for in-place upcycling or
    /// merging of expert weights).
    pub fn experts_mut(&mut self) -> &mut [ExpertFfn] {
        &mut self.experts
    }

    /// Process a batch of tokens through a single expert.
    ///
    /// # Arguments
    /// * `expert_idx` — index of the expert to use
    /// * `tokens` — input tokens, shape `[n_tokens * input_dim]`
    /// * `n_tokens` — number of tokens
    pub fn forward_expert(
        &self,
        expert_idx: usize,
        tokens: &[f32],
        n_tokens: usize,
    ) -> MoeResult<Vec<f32>> {
        if expert_idx >= self.n_experts {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: expert_idx,
                n_experts: self.n_experts,
            });
        }
        self.experts[expert_idx].forward_batch(tokens, n_tokens)
    }

    /// CPU sequential dispatch: process all tokens through their assigned experts.
    ///
    /// # Arguments
    /// * `x` — input tokens, shape `[n_tokens * input_dim]`
    /// * `expert_assignments` — expert index per token (usize::MAX = dropped)
    /// * `n_tokens` — number of tokens
    /// * `scores` — gate scores per token (used for weighted combination), shape `[n_tokens]`
    pub fn forward_dispatched(
        &self,
        x: &[f32],
        expert_assignments: &[usize],
        n_tokens: usize,
        scores: &[f32],
    ) -> MoeResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected_x = n_tokens * self.input_dim;
        if x.len() != expected_x {
            return Err(MoeError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }
        if expert_assignments.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: expert_assignments.len(),
            });
        }
        if scores.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: scores.len(),
            });
        }

        let mut output = vec![0.0_f32; n_tokens * self.input_dim];

        for (tok, (&assignment, &score)) in expert_assignments.iter().zip(scores.iter()).enumerate()
        {
            // Skip dropped tokens
            if assignment == usize::MAX {
                continue;
            }
            if assignment >= self.n_experts {
                return Err(MoeError::ExpertIndexOutOfRange {
                    idx: assignment,
                    n_experts: self.n_experts,
                });
            }
            let x_tok = &x[tok * self.input_dim..(tok + 1) * self.input_dim];
            let expert_out = self.experts[assignment].forward(x_tok)?;
            let out_slice = &mut output[tok * self.input_dim..(tok + 1) * self.input_dim];
            for (out_val, exp_val) in out_slice.iter_mut().zip(expert_out.iter()) {
                *out_val += score * exp_val;
            }
        }

        Ok(output)
    }
}

/// A collection of SwiGLU experts (Mixtral-style MoE).
pub struct SwiGluBank {
    experts: Vec<SwiGluExpert>,
    /// Number of experts.
    pub n_experts: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// FFN hidden dimension.
    pub ffn_dim: usize,
}

impl SwiGluBank {
    /// Create a bank of `n_experts` SwiGLU experts.
    pub fn new(
        n_experts: usize,
        input_dim: usize,
        ffn_dim: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<Self> {
        if n_experts == 0 {
            return Err(MoeError::InvalidExpertCount { n_experts });
        }
        if input_dim == 0 {
            return Err(MoeError::InvalidInputDim { dim: input_dim });
        }
        if ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: ffn_dim });
        }
        let experts: Vec<SwiGluExpert> = (0..n_experts)
            .map(|_| SwiGluExpert::new(input_dim, ffn_dim, rng))
            .collect();
        Ok(Self {
            experts,
            n_experts,
            input_dim,
            ffn_dim,
        })
    }

    /// CPU sequential dispatch for SwiGLU experts.
    pub fn forward_dispatched(
        &self,
        x: &[f32],
        expert_assignments: &[usize],
        n_tokens: usize,
        scores: &[f32],
    ) -> MoeResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected_x = n_tokens * self.input_dim;
        if x.len() != expected_x {
            return Err(MoeError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }
        if expert_assignments.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: expert_assignments.len(),
            });
        }
        if scores.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: scores.len(),
            });
        }

        let mut output = vec![0.0_f32; n_tokens * self.input_dim];

        for (tok, (&assignment, &score)) in expert_assignments.iter().zip(scores.iter()).enumerate()
        {
            if assignment == usize::MAX {
                continue;
            }
            if assignment >= self.n_experts {
                return Err(MoeError::ExpertIndexOutOfRange {
                    idx: assignment,
                    n_experts: self.n_experts,
                });
            }
            let x_tok = &x[tok * self.input_dim..(tok + 1) * self.input_dim];
            let expert_out = self.experts[assignment].forward(x_tok)?;
            let out_slice = &mut output[tok * self.input_dim..(tok + 1) * self.input_dim];
            for (out_val, exp_val) in out_slice.iter_mut().zip(expert_out.iter()) {
                *out_val += score * exp_val;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn bank_new_and_forward() {
        let mut rng = LcgRng::new(0);
        let bank = ExpertBank::new(4, 8, 32, ExpertActivation::Gelu, &mut rng)
            .expect("new should succeed");
        let x = vec![0.5_f32; 8];
        let out = bank
            .forward_expert(0, &x, 1)
            .expect("forward_expert should succeed");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn bank_dispatch_skips_overflow() {
        let mut rng = LcgRng::new(1);
        let bank = ExpertBank::new(2, 4, 16, ExpertActivation::Relu, &mut rng)
            .expect("new should succeed");
        let x = vec![1.0_f32; 3 * 4];
        let assignments = [0_usize, usize::MAX, 1];
        let scores = [1.0_f32, 0.0, 0.5];
        let out = bank
            .forward_dispatched(&x, &assignments, 3, &scores)
            .expect("value should be present");
        assert_eq!(out.len(), 3 * 4);
        // Dropped token should produce zero output
        let dropped_row = &out[4..8];
        assert!(dropped_row.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn swiglu_bank_dispatch() {
        let mut rng = LcgRng::new(5);
        let bank = SwiGluBank::new(4, 8, 32, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 4 * 8];
        let assignments = [0_usize, 1, 2, 3];
        let scores = [0.8_f32, 0.6, 0.7, 0.9];
        let out = bank
            .forward_dispatched(&x, &assignments, 4, &scores)
            .expect("value should be present");
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
