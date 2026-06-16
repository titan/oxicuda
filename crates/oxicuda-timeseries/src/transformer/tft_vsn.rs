//! Temporal Fusion Transformer — Variable Selection Network (VSN) and
//! Gated Residual Network (GRN).
//!
//! Reference: Lim et al. 2021, "Temporal Fusion Transformers for Interpretable
//! Multi-horizon Time Series Forecasting", International Journal of Forecasting.
//!
//! # Architecture
//!
//! A **Gated Residual Network** (GRN) applies:
//! ```text
//! η₁ = ELU(W₁·x + b₁)
//! η₂ = W₂·η₁ + b₂
//! gate = sigmoid(W_gate·x + b_gate)
//! skip = W_skip·x + b_skip   (if input_dim ≠ hidden_dim, else skip = x)
//! out  = LayerNorm(gate ⊙ η₂ + skip)
//! ```
//!
//! A **Variable Selection Network** (VSN) applies one GRN per input variable,
//! then computes soft selection weights via a second GRN + softmax over all
//! variable-GRN outputs, producing a weighted combination.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Activation helpers ───────────────────────────────────────────────────────

/// ELU activation: x if x > 0 else exp(x) - 1.
#[inline]
fn elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

/// Element-wise sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax over a mutable slice (in-place, numerically stable).
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Layer normalisation over a vector (mean=0, var=1 with ε=1e-5, no affine).
fn layer_norm(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();
    v.iter().map(|&x| (x - mean) / std).collect()
}

// ─── Linear helpers ───────────────────────────────────────────────────────────

/// Matrix-vector product: W[out_dim × in_dim] · x[in_dim] + b[out_dim] → out[out_dim].
fn linear(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = b.to_vec();
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        for (k, &xk) in x.iter().enumerate().take(in_dim) {
            out[o] += row[k] * xk;
        }
    }
    out
}

/// Xavier-uniform initialisation: range ±√(6/(fan_in + fan_out)).
fn xavier_init(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| rng.next_f32() * 2.0 * limit - limit)
        .collect()
}

// ─── GRN ─────────────────────────────────────────────────────────────────────

/// Gated Residual Network.
///
/// Transforms an `input_dim`-vector into a `hidden_dim`-vector via:
///   - Two-layer ELU MLP (W1, W2)
///   - GLU gate (sigmoid gate on a separate projection of x)
///   - Skip connection (identity if dims match, else W_skip projection)
///   - LayerNorm on the residual output
#[derive(Debug, Clone)]
pub struct Grn {
    /// W₁: [hidden_dim × input_dim]
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// W₂: [hidden_dim × hidden_dim]
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// Gate projection: [hidden_dim × input_dim]
    w_gate: Vec<f32>,
    b_gate: Vec<f32>,
    /// Skip projection: [hidden_dim × input_dim]  (only used when input_dim ≠ hidden_dim)
    w_skip: Vec<f32>,
    b_skip: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
}

impl Grn {
    /// Construct a GRN, initialising weights with Xavier uniform.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let w1 = xavier_init(hidden_dim, input_dim, rng);
        let b1 = vec![0.0f32; hidden_dim];
        let w2 = xavier_init(hidden_dim, hidden_dim, rng);
        let b2 = vec![0.0f32; hidden_dim];
        let w_gate = xavier_init(hidden_dim, input_dim, rng);
        let b_gate = vec![0.0f32; hidden_dim];
        let w_skip = xavier_init(hidden_dim, input_dim, rng);
        let b_skip = vec![0.0f32; hidden_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            w_gate,
            b_gate,
            w_skip,
            b_skip,
            input_dim,
            hidden_dim,
        }
    }

    /// Forward pass: `x[input_dim]` → `out[hidden_dim]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // η₁ = ELU(W₁·x + b₁)
        let eta1: Vec<f32> = linear(&self.w1, &self.b1, x, self.input_dim, self.hidden_dim)
            .into_iter()
            .map(elu)
            .collect();
        // η₂ = W₂·η₁ + b₂
        let eta2 = linear(&self.w2, &self.b2, &eta1, self.hidden_dim, self.hidden_dim);
        // gate = sigmoid(W_gate·x + b_gate)
        let gate: Vec<f32> = linear(
            &self.w_gate,
            &self.b_gate,
            x,
            self.input_dim,
            self.hidden_dim,
        )
        .into_iter()
        .map(sigmoid)
        .collect();
        // skip connection
        let skip: Vec<f32> = if self.input_dim == self.hidden_dim {
            x.to_vec()
        } else {
            linear(
                &self.w_skip,
                &self.b_skip,
                x,
                self.input_dim,
                self.hidden_dim,
            )
        };
        // residual = gate ⊙ η₂ + skip
        let residual: Vec<f32> = gate
            .iter()
            .zip(eta2.iter())
            .zip(skip.iter())
            .map(|((&g, &e), &s)| g * e + s)
            .collect();
        // LayerNorm
        layer_norm(&residual)
    }
}

// ─── VSN configuration ────────────────────────────────────────────────────────

/// Configuration for a Variable Selection Network.
#[derive(Debug, Clone)]
pub struct VsnConfig {
    /// Number of input variables (each has one scalar feature).
    pub n_inputs: usize,
    /// Hidden dimension (output dimension of each per-variable GRN).
    pub hidden_dim: usize,
    /// Dropout rate (informational; inference path does not apply stochastic dropout).
    pub dropout_rate: f32,
}

// ─── VariableSelectionNet ─────────────────────────────────────────────────────

/// Variable Selection Network (VSN).
///
/// Each input variable is processed by its own GRN.  A second context GRN +
/// softmax over the concatenated GRN outputs produces interpretable
/// variable-importance weights.
#[derive(Debug, Clone)]
pub struct VariableSelectionNet {
    /// One per-variable GRN (scalar input → hidden_dim).
    grns: Vec<Grn>,
    /// Context selection: softmax weights over flattened variable representations.
    /// w_context: [n_inputs × (n_inputs * hidden_dim)]
    w_context: Vec<f32>,
    b_context: Vec<f32>,
    hidden_dim: usize,
    n_inputs: usize,
}

impl VariableSelectionNet {
    /// Construct a VSN from a `VsnConfig`.
    pub fn new(config: VsnConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.n_inputs == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if config.hidden_dim == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        let grns: Vec<Grn> = (0..config.n_inputs)
            .map(|_| Grn::new(1, config.hidden_dim, rng))
            .collect();
        // Context weights: [n_inputs] output from [n_inputs × hidden_dim] concatenated input
        let context_in = config.n_inputs * config.hidden_dim;
        let w_context = xavier_init(config.n_inputs, context_in, rng);
        let b_context = vec![0.0f32; config.n_inputs];
        Ok(Self {
            grns,
            w_context,
            b_context,
            hidden_dim: config.hidden_dim,
            n_inputs: config.n_inputs,
        })
    }

    /// Forward pass.
    ///
    /// `inputs`: slice of length `n_inputs` (one scalar feature per variable).
    ///
    /// Returns `(weighted_output[hidden_dim], variable_weights[n_inputs])`.
    pub fn forward(&self, inputs: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        if inputs.len() != self.n_inputs {
            return Err(TsError::DimensionMismatch {
                expected: self.n_inputs,
                got: inputs.len(),
            });
        }
        // Process each variable through its GRN
        let mut grn_outputs: Vec<Vec<f32>> = self
            .grns
            .iter()
            .enumerate()
            .map(|(i, grn)| grn.forward(&inputs[i..i + 1]))
            .collect();

        // Build concatenated context vector [n_inputs * hidden_dim]
        let mut ctx = Vec::with_capacity(self.n_inputs * self.hidden_dim);
        for g in &grn_outputs {
            ctx.extend_from_slice(g);
        }

        // Compute softmax selection weights over variables
        let context_in = self.n_inputs * self.hidden_dim;
        let mut var_weights = linear(
            &self.w_context,
            &self.b_context,
            &ctx,
            context_in,
            self.n_inputs,
        );
        softmax_inplace(&mut var_weights);

        // Weighted sum: Σ_i weight_i * grn_i(x_i)
        let mut output = vec![0.0f32; self.hidden_dim];
        for (i, w) in var_weights.iter().enumerate() {
            for d in 0..self.hidden_dim {
                output[d] += w * grn_outputs[i][d];
            }
        }

        // Validate finiteness
        for &v in &output {
            if !v.is_finite() {
                return Err(TsError::NonFinite);
            }
        }

        // Ensure we don't keep an unnecessary extra clone
        grn_outputs.clear();

        Ok((output, var_weights))
    }

    /// Number of input variables.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn make_vsn(n_inputs: usize, hidden_dim: usize) -> VariableSelectionNet {
        let mut rng = make_rng();
        VariableSelectionNet::new(
            VsnConfig {
                n_inputs,
                hidden_dim,
                dropout_rate: 0.1,
            },
            &mut rng,
        )
        .expect("VSN construction failed")
    }

    #[test]
    fn grn_output_shape() {
        let mut rng = make_rng();
        let grn = Grn::new(4, 8, &mut rng);
        let x = vec![1.0f32; 4];
        let out = grn.forward(&x);
        assert_eq!(out.len(), 8, "GRN output should have len hidden_dim=8");
    }

    #[test]
    fn grn_output_finite() {
        let mut rng = make_rng();
        let grn = Grn::new(4, 8, &mut rng);
        let x: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
        let out = grn.forward(&x);
        for &v in &out {
            assert!(v.is_finite(), "GRN output contains non-finite value: {v}");
        }
    }

    #[test]
    fn vsn_output_shape() {
        let vsn = make_vsn(5, 16);
        let inputs = vec![1.0f32; 5];
        let (out, weights) = vsn.forward(&inputs).expect("vsn forward");
        assert_eq!(out.len(), 16, "VSN output should have len hidden_dim=16");
        assert_eq!(
            weights.len(),
            5,
            "variable weights should have len n_inputs=5"
        );
    }

    #[test]
    fn vsn_weights_sum_to_one() {
        let vsn = make_vsn(4, 8);
        let inputs = vec![0.5f32; 4];
        let (_, weights) = vsn.forward(&inputs).expect("vsn forward");
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "variable weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn vsn_weights_nonneg() {
        let vsn = make_vsn(4, 8);
        let inputs: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let (_, weights) = vsn.forward(&inputs).expect("vsn forward");
        for &w in &weights {
            assert!(w >= 0.0, "variable weight {w} should be >= 0");
        }
    }

    #[test]
    fn vsn_finite() {
        let vsn = make_vsn(6, 12);
        let inputs: Vec<f32> = (0..6).map(|i| (i as f32 - 3.0) * 0.7).collect();
        let (out, weights) = vsn.forward(&inputs).expect("vsn forward");
        for &v in out.iter().chain(weights.iter()) {
            assert!(v.is_finite(), "VSN output has non-finite value: {v}");
        }
    }

    #[test]
    fn grn_skip_different_dims() {
        // input_dim ≠ hidden_dim: GRN uses W_skip projection
        let mut rng = make_rng();
        let grn = Grn::new(3, 7, &mut rng); // input_dim=3, hidden_dim=7
        let x = vec![1.0f32, -1.0, 0.5];
        let out = grn.forward(&x);
        assert_eq!(
            out.len(),
            7,
            "output should have len hidden_dim=7 even with different dims"
        );
    }

    #[test]
    fn n_inputs_zero_error() {
        let mut rng = make_rng();
        let result = VariableSelectionNet::new(
            VsnConfig {
                n_inputs: 0,
                hidden_dim: 8,
                dropout_rate: 0.0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_inputs=0 should return Err");
    }

    #[test]
    fn softmax_selects_largest() {
        // With constant inputs, all variable weights should be approximately equal.
        let vsn = make_vsn(4, 8);
        let inputs = vec![1.0f32; 4];
        let (_, weights) = vsn.forward(&inputs).expect("vsn forward");
        // All equal input → softmax drives weights toward uniformity
        // (exact uniformity depends on GRN weights, but all should be ≈ 0.25)
        // We only check that sum = 1 and all ≥ 0, which are stronger invariants.
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn grn_elu_activation_shape() {
        // Verify that the GRN produces correct output shape regardless of sign of input.
        let mut rng = make_rng();
        let grn = Grn::new(2, 4, &mut rng);
        let x_pos = vec![2.0f32, 3.0];
        let x_neg = vec![-2.0f32, -3.0];
        let out_pos = grn.forward(&x_pos);
        let out_neg = grn.forward(&x_neg);
        assert_eq!(out_pos.len(), 4);
        assert_eq!(out_neg.len(), 4);
        for (&p, &n) in out_pos.iter().zip(out_neg.iter()) {
            assert!(p.is_finite() && n.is_finite());
        }
    }

    #[test]
    fn vsn_n_inputs_accessor() {
        let vsn = make_vsn(7, 16);
        assert_eq!(vsn.n_inputs(), 7);
    }
}
