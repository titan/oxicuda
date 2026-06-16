//! Expert FFN: standard transformer FFN with multiple activation options.
//!
//! Each expert has independent weights; supports GELU, SiLU, and ReLU activations,
//! as well as the SwiGLU variant used in Mixtral/LLaMA-style MoE models.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// GELU coefficient: sqrt(2/π) ≈ 0.7978845608
const GELU_COEFF: f32 = 0.797_884_6_f32;
/// GELU cubic coefficient
const GELU_CUBIC: f32 = 0.044_715_f32;

/// Compute GELU approximation (OpenAI variant).
///
/// `gelu(x) = x * 0.5 * (1 + tanh(GELU_COEFF * (x + GELU_CUBIC * x^3)))`
#[inline]
fn gelu_approx(val: f32) -> f32 {
    val * 0.5 * (1.0 + (GELU_COEFF * (val + GELU_CUBIC * val * val * val)).tanh())
}

/// Compute SiLU (Swish-1): `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline]
fn silu(val: f32) -> f32 {
    val / (1.0 + (-val).exp())
}

/// Expert activation function.
#[derive(Debug, Clone, Copy)]
pub enum ExpertActivation {
    /// GELU activation (Gaussian Error Linear Unit).
    Gelu,
    /// SiLU / Swish-1 activation.
    Silu,
    /// ReLU activation.
    Relu,
}

impl ExpertActivation {
    /// Apply the activation to a single value.
    #[inline]
    #[must_use]
    pub fn apply(self, val: f32) -> f32 {
        match self {
            Self::Gelu => gelu_approx(val),
            Self::Silu => silu(val),
            Self::Relu => val.max(0.0),
        }
    }
}

/// Expert FFN with standard two-layer structure.
///
/// Forward: `h = act(W1·x + b1)`, then `y = W2·h + b2`.
#[derive(Debug, Clone)]
pub struct ExpertFfn {
    /// First layer weights, shape `[ffn_dim * input_dim]`.
    pub w1: Vec<f32>,
    /// First layer biases, shape `[ffn_dim]`.
    pub b1: Vec<f32>,
    /// Second layer weights, shape `[input_dim * ffn_dim]`.
    pub w2: Vec<f32>,
    /// Second layer biases, shape `[input_dim]`.
    pub b2: Vec<f32>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// FFN hidden dimension.
    pub ffn_dim: usize,
    /// Activation function.
    pub activation: ExpertActivation,
}

impl ExpertFfn {
    /// Create a new expert FFN with Xavier initialization.
    #[must_use]
    pub fn new(input_dim: usize, ffn_dim: usize, act: ExpertActivation, rng: &mut LcgRng) -> Self {
        // Xavier uniform: std = sqrt(2 / (fan_in + fan_out))
        let std_w1 = (2.0 / (input_dim + ffn_dim) as f32).sqrt();
        let std_w2 = (2.0 / (ffn_dim + input_dim) as f32).sqrt();

        let mut w1 = vec![0.0_f32; ffn_dim * input_dim];
        let mut w2 = vec![0.0_f32; input_dim * ffn_dim];
        rng.fill_normal_scaled(&mut w1, std_w1);
        rng.fill_normal_scaled(&mut w2, std_w2);

        Self {
            w1,
            b1: vec![0.0_f32; ffn_dim],
            w2,
            b2: vec![0.0_f32; input_dim],
            input_dim,
            ffn_dim,
            activation: act,
        }
    }

    /// Single-token forward pass.
    ///
    /// Input: `x` of length `input_dim`.
    /// Output: vector of length `input_dim`.
    pub fn forward(&self, x: &[f32]) -> MoeResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(MoeError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        // h = act(W1·x + b1)
        let mut hidden = vec![0.0_f32; self.ffn_dim];
        for (hi, (w1_row, &bi)) in hidden
            .iter_mut()
            .zip(self.w1.chunks(self.input_dim).zip(self.b1.iter()))
        {
            let pre_act: f32 = w1_row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            *hi = self.activation.apply(pre_act + bi);
        }

        // y = W2·h + b2
        let mut output = vec![0.0_f32; self.input_dim];
        for (oi, (w2_row, &bi)) in output
            .iter_mut()
            .zip(self.w2.chunks(self.ffn_dim).zip(self.b2.iter()))
        {
            let dot: f32 = w2_row
                .iter()
                .zip(hidden.iter())
                .map(|(&wi, &hi)| wi * hi)
                .sum();
            *oi = dot + bi;
        }

        Ok(output)
    }

    /// Batch forward pass.
    ///
    /// Input: `x` of shape `[batch_size * input_dim]`.
    /// Output: vector of shape `[batch_size * input_dim]`.
    pub fn forward_batch(&self, x: &[f32], batch_size: usize) -> MoeResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = batch_size * self.input_dim;
        if x.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut output = vec![0.0_f32; batch_size * self.input_dim];
        for sample in 0..batch_size {
            let x_row = &x[sample * self.input_dim..(sample + 1) * self.input_dim];
            let y = self.forward(x_row)?;
            output[sample * self.input_dim..(sample + 1) * self.input_dim].copy_from_slice(&y);
        }
        Ok(output)
    }

    /// Return the total parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// SwiGLU expert variant as used in Mixtral/LLaMA-style MoE.
///
/// Forward: `h = SiLU(W1·x) ⊙ (W3·x)`, then `output = W2·h`.
#[derive(Debug, Clone)]
pub struct SwiGluExpert {
    /// Gate projection weights, shape `[ffn_dim * input_dim]`.
    pub w1: Vec<f32>,
    /// Value projection weights, shape `[ffn_dim * input_dim]`.
    pub w3: Vec<f32>,
    /// Output projection weights, shape `[input_dim * ffn_dim]`.
    pub w2: Vec<f32>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// FFN hidden dimension.
    pub ffn_dim: usize,
}

impl SwiGluExpert {
    /// Create a new SwiGLU expert with Xavier initialization.
    #[must_use]
    pub fn new(input_dim: usize, ffn_dim: usize, rng: &mut LcgRng) -> Self {
        let std_in = (2.0 / (input_dim + ffn_dim) as f32).sqrt();
        let std_out = (2.0 / (ffn_dim + input_dim) as f32).sqrt();

        let mut w1 = vec![0.0_f32; ffn_dim * input_dim];
        let mut w3 = vec![0.0_f32; ffn_dim * input_dim];
        let mut w2 = vec![0.0_f32; input_dim * ffn_dim];
        rng.fill_normal_scaled(&mut w1, std_in);
        rng.fill_normal_scaled(&mut w3, std_in);
        rng.fill_normal_scaled(&mut w2, std_out);

        Self {
            w1,
            w3,
            w2,
            input_dim,
            ffn_dim,
        }
    }

    /// Single-token forward pass.
    ///
    /// Input: `x` of length `input_dim`.
    /// Output: vector of length `input_dim`.
    pub fn forward(&self, x: &[f32]) -> MoeResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(MoeError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        // gate = SiLU(W1 · x), value = W3 · x; h = gate ⊙ value
        let mut hidden = vec![0.0_f32; self.ffn_dim];
        for (hi, (w1_row, w3_row)) in hidden.iter_mut().zip(
            self.w1
                .chunks(self.input_dim)
                .zip(self.w3.chunks(self.input_dim)),
        ) {
            let gate: f32 = w1_row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            let value: f32 = w3_row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            *hi = silu(gate) * value;
        }

        // output = W2 · h
        let mut output = vec![0.0_f32; self.input_dim];
        for (oi, w2_row) in output.iter_mut().zip(self.w2.chunks(self.ffn_dim)) {
            *oi = w2_row
                .iter()
                .zip(hidden.iter())
                .map(|(&wi, &hi)| wi * hi)
                .sum();
        }

        Ok(output)
    }

    /// Batch forward pass.
    ///
    /// Input: `x` of shape `[batch_size * input_dim]`.
    /// Output: vector of shape `[batch_size * input_dim]`.
    pub fn forward_batch(&self, x: &[f32], batch_size: usize) -> MoeResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = batch_size * self.input_dim;
        if x.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut output = vec![0.0_f32; batch_size * self.input_dim];
        for sample in 0..batch_size {
            let x_row = &x[sample * self.input_dim..(sample + 1) * self.input_dim];
            let y = self.forward(x_row)?;
            output[sample * self.input_dim..(sample + 1) * self.input_dim].copy_from_slice(&y);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn ffn_forward_shape() {
        let mut rng = LcgRng::new(42);
        let ffn = ExpertFfn::new(16, 64, ExpertActivation::Gelu, &mut rng);
        let x = vec![0.5_f32; 16];
        let out = ffn.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn ffn_batch_forward_shape() {
        let mut rng = LcgRng::new(42);
        let ffn = ExpertFfn::new(16, 64, ExpertActivation::Silu, &mut rng);
        let x = vec![0.5_f32; 4 * 16];
        let out = ffn
            .forward_batch(&x, 4)
            .expect("forward_batch should succeed");
        assert_eq!(out.len(), 4 * 16);
    }

    #[test]
    fn ffn_all_finite() {
        let mut rng = LcgRng::new(13);
        let ffn = ExpertFfn::new(8, 32, ExpertActivation::Relu, &mut rng);
        let x = vec![1.0_f32; 8];
        let out = ffn.forward(&x).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn swiglu_forward_shape() {
        let mut rng = LcgRng::new(7);
        let exp = SwiGluExpert::new(16, 64, &mut rng);
        let x = vec![0.3_f32; 16];
        let out = exp.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn swiglu_all_finite() {
        let mut rng = LcgRng::new(99);
        let exp = SwiGluExpert::new(8, 32, &mut rng);
        let x = vec![1.0_f32; 8];
        let out = exp.forward(&x).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gelu_approx_zero_is_zero() {
        assert!(gelu_approx(0.0).abs() < 1e-6);
    }

    #[test]
    fn silu_zero_is_zero() {
        assert!(silu(0.0).abs() < 1e-6);
    }
}
