use crate::handle::LcgRng;

/// Houlsby-style bottleneck adapter with LayerNorm.
///
/// Architecture: `LayerNorm(x) → Linear_down → GELU → Linear_up + x`
///
/// `down_w`: `[bottleneck_dim × in_dim]` (He-initialised).
/// `up_w`: `[in_dim × bottleneck_dim]` (zero-initialised for near-identity init).
/// `ln_w`, `ln_b`: `[in_dim]` LayerNorm parameters (ones, zeros).
#[derive(Debug, Clone)]
pub struct HoulsbyAdapter {
    /// Input/output dimension.
    pub in_dim: usize,
    /// Bottleneck (hidden) dimension.
    pub bottleneck_dim: usize,
    /// Down-projection weight, shape `[bottleneck_dim × in_dim]`.
    pub down_w: Vec<f32>,
    /// Down-projection bias, shape `[bottleneck_dim]`.
    pub down_b: Vec<f32>,
    /// Up-projection weight, shape `[in_dim × bottleneck_dim]`.
    pub up_w: Vec<f32>,
    /// Up-projection bias, shape `[in_dim]`.
    pub up_b: Vec<f32>,
    /// LayerNorm gain, shape `[in_dim]`.
    pub ln_w: Vec<f32>,
    /// LayerNorm bias, shape `[in_dim]`.
    pub ln_b: Vec<f32>,
}

impl HoulsbyAdapter {
    /// Construct a `HoulsbyAdapter`.
    ///
    /// `down_w` uses He init (N(0, sqrt(2 / in_dim))). `up_w` is zero-initialised
    /// so the adapter starts as the identity. LayerNorm parameters: `ln_w = ones`, `ln_b = zeros`.
    #[must_use]
    pub fn new(in_dim: usize, bottleneck_dim: usize, rng: &mut LcgRng) -> Self {
        let he_std = (2.0_f32 / in_dim as f32).sqrt();
        let mut down_w = vec![0.0_f32; bottleneck_dim * in_dim];
        rng.fill_normal(&mut down_w);
        for v in down_w.iter_mut() {
            *v *= he_std;
        }
        let down_b = vec![0.0_f32; bottleneck_dim];
        // Zero-init up_w for near-identity residual start
        let up_w = vec![0.0_f32; in_dim * bottleneck_dim];
        let up_b = vec![0.0_f32; in_dim];
        let ln_w = vec![1.0_f32; in_dim];
        let ln_b = vec![0.0_f32; in_dim];
        Self {
            in_dim,
            bottleneck_dim,
            down_w,
            down_b,
            up_w,
            up_b,
            ln_w,
            ln_b,
        }
    }

    /// Apply the Houlsby adapter to an input of shape `[seq_len × in_dim]`.
    ///
    /// Returns a vector of length `seq_len * in_dim`.
    #[must_use]
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * self.in_dim);
        for t in 0..seq_len {
            let token = &x[t * self.in_dim..(t + 1) * self.in_dim];
            // LayerNorm
            let normed = layer_norm(token, &self.ln_w, &self.ln_b, self.in_dim);
            // Down projection + bias + GELU
            let hidden: Vec<f32> = (0..self.bottleneck_dim)
                .map(|i| {
                    let acc = self.down_b[i]
                        + self.down_w[i * self.in_dim..(i + 1) * self.in_dim]
                            .iter()
                            .zip(normed.iter())
                            .map(|(w, n)| w * n)
                            .sum::<f32>();
                    gelu(acc)
                })
                .collect();
            // Up projection + bias + residual
            for (i, &xi) in token.iter().enumerate() {
                let acc = self.up_b[i]
                    + hidden
                        .iter()
                        .enumerate()
                        .map(|(j, &h)| self.up_w[i * self.bottleneck_dim + j] * h)
                        .sum::<f32>();
                out.push(acc + xi);
            }
        }
        out
    }
}

/// Per-token layer normalisation with learnable gain `w` and bias `b`.
///
/// Uses `eps = 1e-5`. `dim` is the feature dimension.
pub(crate) fn layer_norm(x: &[f32], w: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mean = x.iter().copied().sum::<f32>() / dim as f32;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
    let inv_std = (var + 1e-5_f32).sqrt().recip();
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) * inv_std * w[i] + b[i])
        .collect()
}

/// GELU activation: `0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))`.
pub(crate) fn gelu(x: f32) -> f32 {
    const C0: f32 = 0.797_884_56; // sqrt(2/π)
    const C1: f32 = 0.044_715;
    let inner = C0 * (x + C1 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}
