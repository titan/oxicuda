//! Mixed operation: softmax-weighted sum of candidate operations.
//!
//! In DARTS, each edge between nodes is parameterised by a `MixedOp` which
//! maintains continuous architecture parameters `α` (logits) and computes a
//! weighted combination of all candidate ops during the search phase.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::primitives::{OpKind, OpWeights};

// ─── MixedOp ─────────────────────────────────────────────────────────────────

/// Differentiable mixed operation for DARTS.
///
/// Holds raw architecture logits `α` (one per op) and applies softmax to
/// obtain mixing weights `w`. The forward pass returns `Σ_k w_k * op_k(x)`.
#[derive(Debug, Clone)]
pub struct MixedOp {
    /// Raw logits before softmax: `α[k]` for each candidate op.
    pub arch_params: Vec<f32>,
    /// Which operation kinds are in the mixture (in the same order as `arch_params`).
    pub op_kinds: Vec<OpKind>,
}

impl MixedOp {
    /// Create a new `MixedOp` with architecture params initialised from `N(0, 0.01)`.
    #[must_use]
    pub fn new(op_kinds: Vec<OpKind>, rng: &mut LcgRng) -> Self {
        let n = op_kinds.len();
        let mut arch_params = vec![0.0_f32; n];
        rng.fill_normal(&mut arch_params);
        arch_params.iter_mut().for_each(|v| *v *= 0.01);
        Self {
            arch_params,
            op_kinds,
        }
    }

    /// Number of candidate operations.
    #[must_use]
    pub fn n_ops(&self) -> usize {
        self.op_kinds.len()
    }

    /// Return softmax of `arch_params` as mixing weights.
    ///
    /// Uses the numerically stable `exp(x - max) / Σ exp(x - max)` formulation.
    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        softmax(&self.arch_params)
    }

    /// CPU reference forward: `Σ_k w_k * op_k(input)`.
    ///
    /// # Arguments
    /// * `input` — feature map `[in_ch * H * W]`
    /// * `in_ch`, `h`, `w` — spatial dimensions of `input`
    /// * `out_ch` — number of output channels
    /// * `op_weights` — per-op weight tensors (one per op in `op_kinds`)
    pub fn forward_cpu(
        &self,
        input: &[f32],
        in_ch: usize,
        h: usize,
        w: usize,
        out_ch: usize,
        op_weights: &[OpWeights],
    ) -> NasResult<Vec<f32>> {
        let n = self.op_kinds.len();
        if op_weights.len() != n {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: op_weights.len(),
            });
        }
        if n == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        let ws = self.weights();
        let out_size = out_ch * h * w;
        let mut result = vec![0.0_f32; out_size];

        for (k, (&kind, wk)) in self.op_kinds.iter().zip(ws.iter()).enumerate() {
            let op_out = kind.forward_cpu(input, in_ch, h, w, out_ch, &op_weights[k])?;
            if op_out.len() != out_size {
                return Err(NasError::DimensionMismatch {
                    expected: out_size,
                    got: op_out.len(),
                });
            }
            for (r, &o) in result.iter_mut().zip(op_out.iter()) {
                *r += wk * o;
            }
        }
        Ok(result)
    }

    /// Compute gradient of arch params via softmax Jacobian (diagonal approximation).
    ///
    /// `grad_alpha[k] = w_k * (1 - w_k) * <output_grad, op_k_output>`
    ///
    /// # Arguments
    /// * `output_grad` — upstream gradient `[out_ch * H * W]`
    /// * `op_outputs` — each op's forward output (slice of `[out_ch * H * W]`)
    #[must_use]
    pub fn arch_gradient(&self, output_grad: &[f32], op_outputs: &[Vec<f32>]) -> Vec<f32> {
        let ws = self.weights();
        let n = self.op_kinds.len();
        let mut grad = vec![0.0_f32; n];
        for k in 0..n {
            let wk = ws[k];
            // dot product <output_grad, op_k_output>
            let dot: f32 = if k < op_outputs.len() {
                output_grad
                    .iter()
                    .zip(op_outputs[k].iter())
                    .map(|(g, o)| g * o)
                    .sum()
            } else {
                0.0
            };
            grad[k] = wk * (1.0 - wk) * dot;
        }
        grad
    }
}

// ─── softmax helper ──────────────────────────────────────────────────────────

/// Numerically stable softmax: `exp(x_i - max) / Σ exp(x_j - max)`.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = softmax(&logits);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum = {s}");
    }

    #[test]
    fn mixed_op_weights_sum_to_one() {
        let mut rng = LcgRng::new(42);
        let op = MixedOp::new(OpKind::all().to_vec(), &mut rng);
        let w = op.weights();
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum = {s}");
    }

    #[test]
    fn arch_gradient_shape() {
        let mut rng = LcgRng::new(1);
        let op = MixedOp::new(OpKind::all().to_vec(), &mut rng);
        let grad = vec![1.0_f32; 4 * 4 * 4];
        let op_outs: Vec<Vec<f32>> = (0..8).map(|_| vec![0.5_f32; 4 * 4 * 4]).collect();
        let g = op.arch_gradient(&grad, &op_outs);
        assert_eq!(g.len(), 8);
    }
}
