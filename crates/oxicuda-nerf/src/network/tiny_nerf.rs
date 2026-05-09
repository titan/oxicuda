//! Compact 4-layer NeRF for tests and fast experiments.
//!
//! Architecture: input → FC → ReLU → FC → ReLU → FC → ReLU → FC → (sigma, RGB)
//! Last layer: no activation for sigma (then ReLU); Sigmoid for RGB.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

/// Tiny NeRF MLP: 3 hidden layers + output layer.
#[derive(Debug, Clone)]
pub struct TinyNerf {
    /// Weight-bias pairs for 4 layers.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Input dimensionality.
    input_dim: usize,
}

impl TinyNerf {
    /// Create a new TinyNerf.
    ///
    /// 3 hidden layers of `hidden_dim` followed by an output of 4 (sigma + RGB).
    #[must_use]
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / input_dim as f32).sqrt();
        let mut init = |fan_in: usize, fan_out: usize| -> (Vec<f32>, Vec<f32>) {
            let s = (2.0_f32 / fan_in as f32).sqrt();
            let mut w = vec![0.0_f32; fan_out * fan_in];
            for v in w.iter_mut() {
                let (a, _) = rng.next_normal_pair();
                *v = a * s;
            }
            (w, vec![0.0_f32; fan_out])
        };
        let _ = scale; // scale computed but only used via init closure

        let layer0 = init(input_dim, hidden_dim);
        let layer1 = init(hidden_dim, hidden_dim);
        let layer2 = init(hidden_dim, hidden_dim);
        let layer3 = init(hidden_dim, 4); // sigma + RGB

        Self {
            layers: vec![layer0, layer1, layer2, layer3],
            input_dim,
        }
    }

    /// Forward pass: returns `(sigma: f32, rgb: [f32; 3])`.
    ///
    /// sigma = ReLU(output\[0\]), rgb = Sigmoid(output\[1..4\]).
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `x.len() != input_dim`.
    pub fn forward(&self, x: &[f32]) -> NerfResult<(f32, [f32; 3])> {
        if x.len() != self.input_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        let h = self.layers[0].1.len(); // hidden_dim

        // Hidden layers 0–2 with ReLU
        let a0 = fc_relu(x, &self.layers[0].0, &self.layers[0].1, h);
        let a1 = fc_relu(&a0, &self.layers[1].0, &self.layers[1].1, h);
        let a2 = fc_relu(&a1, &self.layers[2].0, &self.layers[2].1, h);

        // Output layer: 4 units, no intermediate activation
        let out = fc_linear(&a2, &self.layers[3].0, &self.layers[3].1, 4);

        let sigma = out[0].max(0.0);
        let rgb = [sigmoid(out[1]), sigmoid(out[2]), sigmoid(out[3])];

        Ok((sigma, rgb))
    }
}

// ─── Activation utilities ────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn fc_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = (wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi)
            .max(0.0);
    }
    out
}

fn fc_linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_nerf_forward_shape() {
        let mut rng = LcgRng::new(77);
        let net = TinyNerf::new(10, 32, &mut rng);
        let x = vec![0.1_f32; 10];
        let (sigma, rgb) = net.forward(&x).unwrap();
        assert!(sigma >= 0.0);
        assert!(rgb.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn tiny_nerf_wrong_input() {
        let mut rng = LcgRng::new(88);
        let net = TinyNerf::new(10, 32, &mut rng);
        assert!(net.forward(&[0.0_f32; 5]).is_err());
    }
}
