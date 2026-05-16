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
