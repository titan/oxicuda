//! DARTS Mixed Operation — differentiable architecture with mixture of operations.
//!
//! Liu, Hanxiao et al. (2018) "DARTS: Differentiable Architecture Search".
//! <https://arxiv.org/abs/1806.09055>
//!
//! Each edge in the DARTS supernet cell is parameterised by a `DartsMixedOp`,
//! which holds raw architecture logits (one per candidate operation) and
//! per-operation weight matrices.  During the forward pass it computes a
//! softmax-weighted sum of all operation outputs:
//!
//! ```text
//! output = Σ_i softmax(arch_weights)[i] * op_i(x)
//! ```
//!
//! Architecture weights are updated via [`DartsMixedOp::update_arch_weights`]
//! which applies a gradient step (SGD-style, no momentum).

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── DartsConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`DartsMixedOp`] edge.
#[derive(Debug, Clone)]
pub struct DartsConfig {
    /// Number of candidate operations on this edge.
    pub n_ops: usize,
    /// Feature dimension: each op maps `[n_tokens × d_model]` → `[n_tokens × d_model]`.
    pub d_model: usize,
}

// ─── DartsMixedOp ────────────────────────────────────────────────────────────

/// DARTS differentiable mixed operation.
///
/// # Parameterisation
///
/// * `arch_weights` — raw logits `α ∈ ℝ^{n_ops}` (architecture parameters).
/// * `op_weights`   — per-operation linear map `[d_model × d_model]` row-major.
/// * `op_biases`    — per-operation bias `[d_model]`.
///
/// The mixing probabilities are `softmax(arch_weights)`.
pub struct DartsMixedOp {
    /// Raw architecture logits: `[n_ops]`.
    arch_weights: Vec<f32>,
    /// Per-op weight matrices: `[n_ops][d_model × d_model]` row-major.
    op_weights: Vec<Vec<f32>>,
    /// Per-op biases: `[n_ops][d_model]`.
    op_biases: Vec<Vec<f32>>,
    /// Block configuration.
    config: DartsConfig,
}

impl DartsMixedOp {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new `DartsMixedOp` with small-normal initialised architecture
    /// logits and Xavier-uniform initialised operation weights.
    ///
    /// # Errors
    ///
    /// * [`NasError::InvalidNumOps`] — if `config.n_ops == 0`.
    /// * [`NasError::DimensionMismatch`] — if `config.d_model == 0`.
    pub fn new(config: DartsConfig, rng: &mut LcgRng) -> NasResult<Self> {
        if config.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        if config.d_model == 0 {
            return Err(NasError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let n = config.n_ops;
        let d = config.d_model;
        let limit = (6.0_f32 / (d + d) as f32).sqrt(); // Xavier-uniform for square

        // Architecture logits: N(0, 0.01).
        let mut arch_weights = vec![0.0_f32; n];
        for v in arch_weights.iter_mut() {
            let (z, _) = rng.next_normal_pair();
            *v = z * 0.01;
        }

        // Per-op weight matrices and biases.
        let op_weights: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..d * d)
                    .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
                    .collect()
            })
            .collect();

        let op_biases: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0_f32; d]).collect();

        Ok(Self {
            arch_weights,
            op_weights,
            op_biases,
            config,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return the softmax probabilities over operations.
    ///
    /// Uses numerically stable `exp(x - max(x)) / Σ exp(x - max(x))`.
    #[must_use]
    pub fn arch_probs(&self) -> Vec<f32> {
        softmax(&self.arch_weights)
    }

    /// Return the index of the operation with the highest architecture weight.
    #[must_use]
    pub fn selected_op(&self) -> usize {
        self.arch_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Compute `output = Σ_i softmax(α)[i] * op_i(x)`.
    ///
    /// Each operation is a linear map: `op_i(x) = x @ W_i^T + b_i`.
    ///
    /// # Arguments
    ///
    /// * `x`       — flat `[n_tokens × d_model]` input.
    /// * `n_tokens` — number of input tokens.
    ///
    /// # Errors
    ///
    /// * [`NasError::DimensionMismatch`] — if `x.len() != n_tokens * d_model`.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> NasResult<Vec<f32>> {
        let d = self.config.d_model;

        if x.len() != n_tokens * d {
            return Err(NasError::DimensionMismatch {
                expected: n_tokens * d,
                got: x.len(),
            });
        }

        let probs = self.arch_probs();
        let mut output = vec![0.0_f32; n_tokens * d];

        for (k, &prob) in probs.iter().enumerate() {
            if prob < 1e-8 {
                continue; // Skip negligible contributions.
            }
            // op_k(x) = x @ W_k^T + b_k
            let op_out = linear(x, &self.op_weights[k], &self.op_biases[k], n_tokens, d)?;
            for (out_val, op_val) in output.iter_mut().zip(op_out.iter()) {
                *out_val += prob * op_val;
            }
        }

        Ok(output)
    }

    // ── Architecture weight update ─────────────────────────────────────────────

    /// Apply a gradient step to the architecture weights.
    ///
    /// `α ← α - lr * grad`
    ///
    /// # Errors
    ///
    /// * [`NasError::DimensionMismatch`] — if `grad.len() != n_ops`.
    /// * [`NasError::NanInArchParams`] — if any updated weight is NaN.
    pub fn update_arch_weights(&mut self, grad: &[f32], lr: f32) -> NasResult<()> {
        let n = self.config.n_ops;
        if grad.len() != n {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: grad.len(),
            });
        }
        for (w, g) in self.arch_weights.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
            if !w.is_finite() {
                return Err(NasError::NanInArchParams);
            }
        }
        Ok(())
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Numerically stable softmax.
fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exp.iter().map(|&e| e / sum).collect()
}

/// Dense linear: `y = x @ W^T + b` where `x` is `[m × k]` and `W` is `[n × k]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize) -> NasResult<Vec<f32>> {
    let n = b.len();
    if x.len() != m * k {
        return Err(NasError::DimensionMismatch {
            expected: m * k,
            got: x.len(),
        });
    }
    if w.len() != n * k {
        return Err(NasError::DimensionMismatch {
            expected: n * k,
            got: w.len(),
        });
    }
    let mut y = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = b[j];
            for l in 0..k {
                acc += x[i * k + l] * w[j * k + l];
            }
            y[i * n + j] = acc;
        }
    }
    Ok(y)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> DartsConfig {
        DartsConfig {
            n_ops: 3,
            d_model: 4,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    // ── 1. Output shape ───────────────────────────────────────────────────────
    #[test]
    fn output_shape() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        let n_tokens = 5;
        let x: Vec<f32> = (0..n_tokens * d).map(|i| i as f32 * 0.1).collect();
        let out = op.forward(&x, n_tokens).expect("darts: forward");
        assert_eq!(out.len(), n_tokens * d);
    }

    // ── 2. Output finite ──────────────────────────────────────────────────────
    #[test]
    fn output_finite() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        let n_tokens = 8;
        let x: Vec<f32> = (0..n_tokens * d)
            .map(|i| (i as f32 - 16.0) * 0.05)
            .collect();
        let out = op.forward(&x, n_tokens).expect("darts: forward");
        assert!(out.iter().all(|v| v.is_finite()), "output non-finite");
    }

    // ── 3. arch_probs sum to 1 ────────────────────────────────────────────────
    #[test]
    fn arch_probs_sum_to_1() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        let probs = op.arch_probs();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
    }

    // ── 4. selected_op in range ───────────────────────────────────────────────
    #[test]
    fn selected_op_in_range() {
        let cfg = tiny_config();
        let n_ops = cfg.n_ops;
        let mut rng = make_rng();
        let op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        let sel = op.selected_op();
        assert!(sel < n_ops, "selected_op={sel} >= n_ops={n_ops}");
    }

    // ── 5. update changes weights ─────────────────────────────────────────────
    #[test]
    fn update_changes_weights() {
        let cfg = tiny_config();
        let n_ops = cfg.n_ops;
        let mut rng = make_rng();
        let mut op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        let before = op.arch_weights.clone();
        let grad = vec![1.0_f32; n_ops];
        op.update_arch_weights(&grad, 0.1).expect("update");
        let changed = op
            .arch_weights
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "weights unchanged after update");
    }

    // ── 6. n_ops == 1 trivial case ────────────────────────────────────────────
    #[test]
    fn n_ops_1_trivial() {
        let cfg = DartsConfig {
            n_ops: 1,
            d_model: 4,
        };
        let d = cfg.d_model;
        let mut rng = make_rng();
        let op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new n_ops=1");
        let probs = op.arch_probs();
        assert_eq!(probs.len(), 1);
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "single op should have prob 1"
        );
        let x = vec![1.0_f32; d];
        let out = op.forward(&x, 1).expect("forward n_ops=1");
        assert_eq!(out.len(), d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 7. n_ops == 0 → error ─────────────────────────────────────────────────
    #[test]
    fn n_ops_0_error() {
        let cfg = DartsConfig {
            n_ops: 0,
            d_model: 4,
        };
        let mut rng = make_rng();
        let result = DartsMixedOp::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for n_ops=0");
    }

    // ── 8. different arch → different output ──────────────────────────────────
    #[test]
    fn different_arch_different_output() {
        let d = 4;
        let n_ops = 3;
        let mut rng = make_rng();

        let cfg1 = DartsConfig { n_ops, d_model: d };
        let mut op1 = DartsMixedOp::new(cfg1, &mut rng).expect("op1");

        let cfg2 = DartsConfig { n_ops, d_model: d };
        let mut op2 = DartsMixedOp::new(cfg2, &mut rng).expect("op2");

        // Force different arch weights.
        op1.arch_weights = vec![10.0, -10.0, 0.0]; // strongly selects op 0
        op2.arch_weights = vec![-10.0, 10.0, 0.0]; // strongly selects op 1

        let x = vec![1.0_f32; d];
        let out1 = op1.forward(&x, 1).expect("op1 forward");
        let out2 = op2.forward(&x, 1).expect("op2 forward");
        let diff: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "different arch should yield different outputs, diff={diff}"
        );
    }

    // ── 9. forward is a weighted sum ──────────────────────────────────────────
    #[test]
    fn forward_weighted_sum() {
        // With n_ops=2 and equal arch weights, output should be the average of
        // op_0(x) and op_1(x).
        let d = 4;
        let n_tokens = 1;
        let mut rng = make_rng();
        let cfg = DartsConfig {
            n_ops: 2,
            d_model: d,
        };
        let mut op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");

        // Force equal arch weights so softmax gives [0.5, 0.5].
        op.arch_weights = vec![0.0_f32; 2];

        let x = vec![1.0_f32; d];
        let out = op.forward(&x, n_tokens).expect("forward");

        // Compute each op independently.
        let out0 = linear(&x, &op.op_weights[0], &op.op_biases[0], n_tokens, d).expect("op0");
        let out1 = linear(&x, &op.op_weights[1], &op.op_biases[1], n_tokens, d).expect("op1");
        let expected: Vec<f32> = out0
            .iter()
            .zip(out1.iter())
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect();

        let max_err: f32 = out
            .iter()
            .zip(expected.iter())
            .map(|(o, e)| (o - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-5, "weighted sum mismatch, max_err={max_err}");
    }

    // ── 10. update gradient wrong length → error ──────────────────────────────
    #[test]
    fn update_wrong_grad_len_error() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let mut op = DartsMixedOp::new(cfg, &mut rng).expect("darts: new");
        // Pass gradient of wrong length.
        let grad = vec![1.0_f32; 10];
        let result = op.update_arch_weights(&grad, 0.01);
        assert!(
            result.is_err(),
            "expected DimensionMismatch for wrong grad len"
        );
    }

    // ── 11. d_model == 0 → error ──────────────────────────────────────────────
    #[test]
    fn d_model_0_error() {
        let cfg = DartsConfig {
            n_ops: 3,
            d_model: 0,
        };
        let mut rng = make_rng();
        let result = DartsMixedOp::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for d_model=0");
    }
}
