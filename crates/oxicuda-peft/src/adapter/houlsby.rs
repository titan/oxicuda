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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a deterministic input vector in `[-1, 1)` of the given length.
    fn make_input(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 1: output length == seq_len * in_dim (single token)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_output_shape_single_token() {
        let mut rng = LcgRng::new(1);
        let adapter = HoulsbyAdapter::new(8, 4, &mut rng);
        let x = make_input(8, 2);
        let out = adapter.forward(&x, 1);
        assert_eq!(
            out.len(),
            8,
            "output length must equal in_dim for seq_len=1"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 2: output length == seq_len * in_dim (multiple tokens)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_output_shape_multi_token() {
        let mut rng = LcgRng::new(3);
        let adapter = HoulsbyAdapter::new(16, 4, &mut rng);
        let x = make_input(16 * 5, 4);
        let out = adapter.forward(&x, 5);
        assert_eq!(out.len(), 80, "output length must be seq_len * in_dim");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 3: with up_w=0 and up_b=0 (the init state), forward(x) == x exactly.
    //
    // Proof: acc_i = up_b[i] + Σ_j up_w[i*bn+j] * hidden[j] = 0 + 0 = 0
    //        out[i] = acc_i + x[i] = x[i]
    // This holds regardless of the LayerNorm / GELU path because those
    // intermediates are only consumed by the up-projection which is zero.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_zero_up_w_is_identity() {
        let mut rng = LcgRng::new(7);
        let adapter = HoulsbyAdapter::new(8, 3, &mut rng);
        assert!(
            adapter.up_w.iter().all(|&v| v == 0.0),
            "up_w must be zero-initialised"
        );
        assert!(
            adapter.up_b.iter().all(|&v| v == 0.0),
            "up_b must be zero-initialised"
        );
        let x = make_input(8, 9);
        let out = adapter.forward(&x, 1);
        for (i, (&got, &expected)) in out.iter().zip(x.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "forward(x)[{i}] must equal x[{i}] when up_w=0"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 4: same seed → identical forward output (determinism)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_determinism_fixed_seed() {
        let x = make_input(16, 99);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let adapter_a = HoulsbyAdapter::new(16, 4, &mut rng_a);
        let adapter_b = HoulsbyAdapter::new(16, 4, &mut rng_b);
        let out_a = adapter_a.forward(&x, 1);
        let out_b = adapter_b.forward(&x, 1);
        assert_eq!(
            out_a, out_b,
            "same seed must yield identical forward outputs"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 5: all output values are finite for random input
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_finite_outputs() {
        let mut rng = LcgRng::new(13);
        let adapter = HoulsbyAdapter::new(8, 4, &mut rng);
        let x = make_input(8 * 4, 17);
        let out = adapter.forward(&x, 4);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 6: setting up_w to non-zero makes the adapter branch active,
    // so forward(x) != x.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_nonzero_up_w_breaks_identity() {
        let mut rng = LcgRng::new(21);
        let mut adapter = HoulsbyAdapter::new(8, 4, &mut rng);
        let x = make_input(8, 23);
        // Activate the up-projection so the adapter contributes to the output.
        for v in adapter.up_w.iter_mut() {
            *v = 1.0;
        }
        let out = adapter.forward(&x, 1);
        assert_ne!(out, x, "non-zero up_w must produce output different from x");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 7: bottleneck_dim < in_dim → the adapter compresses (sanity check
    // that the constructor accepts the configuration and sizes are correct).
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn houlsby_bottleneck_weight_sizes() {
        let mut rng = LcgRng::new(55);
        let in_dim = 12usize;
        let bottleneck_dim = 3usize;
        let adapter = HoulsbyAdapter::new(in_dim, bottleneck_dim, &mut rng);
        assert_eq!(adapter.down_w.len(), bottleneck_dim * in_dim);
        assert_eq!(adapter.up_w.len(), in_dim * bottleneck_dim);
        assert_eq!(adapter.down_b.len(), bottleneck_dim);
        assert_eq!(adapter.up_b.len(), in_dim);
        assert_eq!(adapter.ln_w.len(), in_dim);
        assert_eq!(adapter.ln_b.len(), in_dim);
        assert!(
            bottleneck_dim < in_dim,
            "bottleneck_dim must be strictly less than in_dim for compression"
        );
    }
}
