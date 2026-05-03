//! Node feature update functions for message passing layers.

use crate::error::{GnnError, GnnResult};

// ─── Activation functions ─────────────────────────────────────────────────────

/// Rectified Linear Unit: `max(0, x)`.
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Leaky ReLU: `x if x >= 0, slope * x otherwise`.
pub fn leaky_relu(x: &[f32], slope: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v >= 0.0 { v } else { slope * v })
        .collect()
}

/// Exponential Linear Unit: `x if x > 0, alpha*(exp(x)-1) otherwise`.
pub fn elu(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect()
}

/// Parametric ReLU: `x if x >= 0, weight * x otherwise`.
pub fn prelu(x: &[f32], weight: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v >= 0.0 { v } else { weight * v })
        .collect()
}

// ─── Linear gemm helper ───────────────────────────────────────────────────────

/// Row-major matrix-vector multiply: `out[i] = Σ_j weight[i*in_dim + j] * x[j] + bias[i]`.
///
/// `weight`: `[out_dim × in_dim]`, `x`: `[in_dim]`, `bias`: `[out_dim]`.
/// Returns `[out_dim]`.
fn linear(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> GnnResult<Vec<f32>> {
    if weight.len() != out_dim * in_dim {
        return Err(GnnError::WeightShapeMismatch {
            r: out_dim,
            c: in_dim,
            d: x.len(),
        });
    }
    if bias.len() != out_dim {
        return Err(GnnError::DimensionMismatch {
            expected: out_dim,
            got: bias.len(),
        });
    }
    if x.len() != in_dim {
        return Err(GnnError::DimensionMismatch {
            expected: in_dim,
            got: x.len(),
        });
    }
    let mut out = bias.to_vec();
    for i in 0..out_dim {
        for j in 0..in_dim {
            out[i] += weight[i * in_dim + j] * x[j];
        }
    }
    Ok(out)
}

// ─── LinearUpdate ────────────────────────────────────────────────────────────

/// Simple linear update: `h_new = W * concat(h_old, msg) + b`.
pub struct LinearUpdate {
    h_dim: usize,
    msg_dim: usize,
    out_dim: usize,
}

impl LinearUpdate {
    /// Construct with the given hidden, message, and output dimensions.
    pub fn new(h_dim: usize, msg_dim: usize, out_dim: usize) -> Self {
        Self {
            h_dim,
            msg_dim,
            out_dim,
        }
    }

    /// Apply: concatenate `h` and `msg`, multiply by weight matrix `W`, add `bias`.
    ///
    /// - `h`: `[h_dim]`
    /// - `msg`: `[msg_dim]`
    /// - `weight`: `[out_dim × (h_dim + msg_dim)]`
    /// - `bias`: `[out_dim]`
    pub fn apply(
        &self,
        h: &[f32],
        msg: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> GnnResult<Vec<f32>> {
        if h.len() != self.h_dim {
            return Err(GnnError::DimensionMismatch {
                expected: self.h_dim,
                got: h.len(),
            });
        }
        if msg.len() != self.msg_dim {
            return Err(GnnError::DimensionMismatch {
                expected: self.msg_dim,
                got: msg.len(),
            });
        }
        let in_dim = self.h_dim + self.msg_dim;
        let mut concat = Vec::with_capacity(in_dim);
        concat.extend_from_slice(h);
        concat.extend_from_slice(msg);
        linear(&concat, weight, bias, in_dim, self.out_dim)
    }
}

// ─── MlpUpdate ───────────────────────────────────────────────────────────────

/// Two-layer MLP update with ReLU activation between layers.
pub struct MlpUpdate {
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
}

impl MlpUpdate {
    /// Construct with the given input, hidden, and output dimensions.
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            hidden_dim,
            out_dim,
        }
    }

    /// Apply the two-layer MLP:
    ///
    /// `h1 = ReLU(W1 * x + b1)`
    /// `out = W2 * h1 + b2`
    ///
    /// - `x`: `[in_dim]`
    /// - `w1`: `[hidden_dim × in_dim]`
    /// - `b1`: `[hidden_dim]`
    /// - `w2`: `[out_dim × hidden_dim]`
    /// - `b2`: `[out_dim]`
    pub fn apply(
        &self,
        x: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
    ) -> GnnResult<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(GnnError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let h1 = linear(x, w1, b1, self.in_dim, self.hidden_dim)?;
        let h1_act = relu(&h1);
        linear(&h1_act, w2, b2, self.hidden_dim, self.out_dim)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_positive_unchanged() {
        let out = relu(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn relu_negative_zeroed() {
        let out = relu(&[-1.0, -2.0, 0.0]);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn leaky_relu_negative_slope() {
        let out = leaky_relu(&[-2.0, 1.0], 0.1);
        assert!((out[0] - (-0.2)).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn elu_positive_unchanged() {
        let out = elu(&[1.0, 2.0], 1.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn elu_negative_exponential() {
        let out = elu(&[-1.0], 1.0);
        let expected = (-1.0_f32).exp() - 1.0;
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn prelu_negative() {
        let out = prelu(&[-3.0, 4.0], 0.25);
        assert!((out[0] - (-0.75)).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn linear_update_apply_correct() {
        let upd = LinearUpdate::new(2, 2, 2);
        let h = vec![1.0_f32, 0.0];
        let msg = vec![0.0_f32, 1.0];
        // W = I_4→2: first row [1,0,0,0], second row [0,1,0,0]
        // Actually W of shape [2×4]: output = [h[0]*1+h[1]*0+msg[0]*0+msg[1]*0, ...]
        let w = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0_f32, 0.0];
        let out = upd
            .apply(&h, &msg, &w, &b)
            .expect("test invariant: value must be valid");
        // concat = [1, 0, 0, 1]
        // out[0] = 1*1+0*0+0*0+1*0 = 1
        // out[1] = 1*0+0*1+0*0+1*0 = 0
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mlp_update_apply_shape() {
        let mlp = MlpUpdate::new(4, 8, 2);
        let x = vec![1.0_f32; 4];
        let w1 = vec![0.1_f32; 8 * 4];
        let b1 = vec![0.0_f32; 8];
        let w2 = vec![0.1_f32; 2 * 8];
        let b2 = vec![0.0_f32; 2];
        let out = mlp
            .apply(&x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn mlp_update_zero_weights_zero_output() {
        let mlp = MlpUpdate::new(3, 4, 2);
        let x = vec![1.0_f32; 3];
        let w1 = vec![0.0_f32; 4 * 3];
        let b1 = vec![0.0_f32; 4];
        let w2 = vec![0.0_f32; 2 * 4];
        let b2 = vec![0.0_f32; 2];
        let out = mlp
            .apply(&x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn linear_update_dimension_mismatch() {
        let upd = LinearUpdate::new(2, 2, 2);
        let err = upd.apply(&[1.0], &[1.0, 2.0], &[1.0; 4], &[0.0; 2]);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }
}
