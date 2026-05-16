use crate::adapter::houlsby::gelu;
use crate::handle::LcgRng;

/// Parallel adapter: runs a bottleneck FFN in parallel with the main FFN, then sums.
///
/// Architecture: `out = ffn_out + (x → Linear_down → GELU → Linear_up)`
///
/// Unlike sequential adapters, the parallel adapter branch runs independently from the
/// main FFN sub-layer and its output is summed with the FFN output.
#[derive(Debug, Clone)]
pub struct ParallelAdapter {
    /// Input dimension.
    pub in_dim: usize,
    /// Bottleneck dimension.
    pub bottleneck_dim: usize,
    /// Down-projection weight, shape `[bottleneck_dim × in_dim]`.
    pub down_w: Vec<f32>,
    /// Down-projection bias, shape `[bottleneck_dim]`.
    pub down_b: Vec<f32>,
    /// Up-projection weight, shape `[in_dim × bottleneck_dim]`.
    pub up_w: Vec<f32>,
    /// Up-projection bias, shape `[in_dim]`.
    pub up_b: Vec<f32>,
}

impl ParallelAdapter {
    /// Construct a `ParallelAdapter`.
    ///
    /// `down_w` He-initialised; `up_w` zero-initialised so the adapter initially contributes nothing.
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

    /// Compute `ffn_out + adapter(x)` where `adapter(x) = Linear_up(GELU(Linear_down(x)))`.
    ///
    /// Both `ffn_out` and `x` must have length `seq_len * in_dim`.
    /// Returns a vector of length `seq_len * in_dim`.
    #[must_use]
    pub fn forward_parallel(&self, ffn_out: &[f32], x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * self.in_dim);
        for t in 0..seq_len {
            let token = &x[t * self.in_dim..(t + 1) * self.in_dim];
            let ffn_token = &ffn_out[t * self.in_dim..(t + 1) * self.in_dim];
            // Down → GELU
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
            // Up + sum with ffn_out
            for (i, &ffn_val) in ffn_token.iter().enumerate() {
                let acc = self.up_b[i]
                    + hidden
                        .iter()
                        .enumerate()
                        .map(|(j, &h)| self.up_w[i * self.bottleneck_dim + j] * h)
                        .sum::<f32>();
                out.push(ffn_val + acc);
            }
        }
        out
    }
}
