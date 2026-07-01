use crate::adapter::houlsby::gelu;
use crate::handle::LcgRng;

/// Pfeiffer-style bottleneck adapter (no LayerNorm, zero-init up projection).
///
/// Architecture: `x → Linear_down → GELU → Linear_up + x`
///
/// Simpler than the Houlsby adapter: no LayerNorm and the skip-init (zero up_w)
/// ensures the adapter starts as the identity function.
#[derive(Debug, Clone)]
pub struct PfeifferAdapter {
    /// Input/output dimension.
    pub in_dim: usize,
    /// Bottleneck dimension.
    pub bottleneck_dim: usize,
    /// Down-projection weight, shape `[bottleneck_dim × in_dim]`.
    pub down_w: Vec<f32>,
    /// Down-projection bias, shape `[bottleneck_dim]`.
    pub down_b: Vec<f32>,
    /// Up-projection weight, shape `[in_dim × bottleneck_dim]` (zero-initialised).
    pub up_w: Vec<f32>,
    /// Up-projection bias, shape `[in_dim]`.
    pub up_b: Vec<f32>,
}

impl PfeifferAdapter {
    /// Construct a `PfeifferAdapter`.
    ///
    /// `down_w` is He-initialised. `up_w` is zero-initialised (skip-init).
    #[must_use]
    pub fn new(in_dim: usize, bottleneck_dim: usize, rng: &mut LcgRng) -> Self {
        let he_std = (2.0_f32 / in_dim as f32).sqrt();
        let mut down_w = vec![0.0_f32; bottleneck_dim * in_dim];
        rng.fill_normal(&mut down_w);
        for v in down_w.iter_mut() {
            *v *= he_std;
        }
        let down_b = vec![0.0_f32; bottleneck_dim];
        let up_w = vec![0.0_f32; in_dim * bottleneck_dim];
        let up_b = vec![0.0_f32; in_dim];
        Self {
            in_dim,
            bottleneck_dim,
            down_w,
            down_b,
            up_w,
            up_b,
        }
    }

    /// Apply the Pfeiffer adapter to an input of shape `[seq_len × in_dim]`.
    ///
    /// Returns a vector of length `seq_len * in_dim`.
    #[must_use]
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * self.in_dim);
        for t in 0..seq_len {
            let token = &x[t * self.in_dim..(t + 1) * self.in_dim];
            // Down projection + bias + GELU
            let hidden: Vec<f32> = (0..self.bottleneck_dim)
                .map(|i| {
                    let acc = self.down_b[i]
                        + self.down_w[i * self.in_dim..(i + 1) * self.in_dim]
                            .iter()
                            .zip(token.iter())
                            .map(|(w, xi)| w * xi)
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
    // Test 1: output length == in_dim for a single token
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_output_shape_single_token() {
        let mut rng = LcgRng::new(1);
        let adapter = PfeifferAdapter::new(8, 4, &mut rng);
        let x = make_input(8, 2);
        let out = adapter.forward(&x, 1);
        assert_eq!(
            out.len(),
            8,
            "output length must equal in_dim for seq_len=1"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 2: output length == seq_len * in_dim for a multi-token sequence
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_output_shape_multi_token() {
        let mut rng = LcgRng::new(3);
        let adapter = PfeifferAdapter::new(16, 4, &mut rng);
        let x = make_input(16 * 3, 4);
        let out = adapter.forward(&x, 3);
        assert_eq!(out.len(), 48, "output length must be seq_len * in_dim");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 3: with up_w=0 and up_b=0 (the init state), forward(x) == x exactly.
    //
    // Proof: acc_i = 0 + Σ_j 0 * hidden[j] = 0; out[i] = 0 + x[i] = x[i].
    // No LayerNorm in Pfeiffer, so there is no mean-shift side-effect.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_zero_up_w_is_identity() {
        let mut rng = LcgRng::new(7);
        let adapter = PfeifferAdapter::new(8, 3, &mut rng);
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
                "forward(x)[{i}] must equal x[{i}] when up_w=0 and up_b=0"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 4: same RNG seed → byte-identical forward output (determinism)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_determinism_fixed_seed() {
        let x = make_input(16, 99);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let adapter_a = PfeifferAdapter::new(16, 4, &mut rng_a);
        let adapter_b = PfeifferAdapter::new(16, 4, &mut rng_b);
        let out_a = adapter_a.forward(&x, 1);
        let out_b = adapter_b.forward(&x, 1);
        assert_eq!(
            out_a, out_b,
            "same seed must yield identical forward outputs"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 5: all output values are finite for a random multi-token input
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_finite_outputs() {
        let mut rng = LcgRng::new(13);
        let adapter = PfeifferAdapter::new(8, 4, &mut rng);
        let x = make_input(8 * 4, 17);
        let out = adapter.forward(&x, 4);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 6: weight buffer sizes match the expected shapes
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_weight_buffer_sizes() {
        let mut rng = LcgRng::new(5);
        let in_dim = 12usize;
        let bottleneck_dim = 3usize;
        let adapter = PfeifferAdapter::new(in_dim, bottleneck_dim, &mut rng);
        assert_eq!(adapter.down_w.len(), bottleneck_dim * in_dim);
        assert_eq!(adapter.up_w.len(), in_dim * bottleneck_dim);
        assert_eq!(adapter.down_b.len(), bottleneck_dim);
        assert_eq!(adapter.up_b.len(), in_dim);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 7: non-zero up_w activates the adapter branch → output differs from x
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pfeiffer_nonzero_up_w_breaks_identity() {
        let mut rng = LcgRng::new(21);
        let mut adapter = PfeifferAdapter::new(8, 4, &mut rng);
        let x = make_input(8, 23);
        for v in adapter.up_w.iter_mut() {
            *v = 1.0;
        }
        let out = adapter.forward(&x, 1);
        assert_ne!(out, x, "non-zero up_w must produce output different from x");
    }
}
