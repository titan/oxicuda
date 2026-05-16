//! Fully-connected spiking linear layer: `current = W·x + b` followed by LIF dynamics.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Spiking fully-connected layer wrapping a Linear projection and per-neuron LIF state.
#[derive(Debug, Clone)]
pub struct SpikingLinear {
    /// Weight matrix flattened as `[out_dim, in_dim]` row-major.
    pub w: Vec<f32>,
    /// Bias vector of length `out_dim`.
    pub b: Vec<f32>,
    /// LIF dynamics parameters used by [`SpikingLinear::forward_step`].
    pub lif_cfg: LifConfig,
    /// Persistent membrane state across timesteps.
    pub state: LifState,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl SpikingLinear {
    /// Allocate a new layer with Kaiming-normal weight initialisation.
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, lif_cfg: LifConfig, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / in_dim.max(1) as f32).sqrt();
        let mut w = vec![0.0_f32; out_dim * in_dim];
        rng.fill_normal(&mut w);
        for v in &mut w {
            *v *= scale;
        }
        let b = vec![0.0_f32; out_dim];
        let state = LifState::new(out_dim);
        Self {
            w,
            b,
            lif_cfg,
            state,
            in_dim,
            out_dim,
        }
    }

    /// One timestep of forward computation. Writes spike outputs to `out`.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) -> SnnResult<()> {
        if x.len() != self.in_dim {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        if out.len() != self.out_dim {
            return Err(SnnError::BadShape {
                expected: self.out_dim,
                got: out.len(),
            });
        }
        let mut current = vec![0.0_f32; self.out_dim];
        for (i, c_i) in current.iter_mut().enumerate() {
            let mut acc = self.b[i];
            let row_off = i * self.in_dim;
            for (j, &xj) in x.iter().enumerate() {
                acc += self.w[row_off + j] * xj;
            }
            *c_i = acc;
        }
        lif_step(&mut self.state, &current, &self.lif_cfg, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape() {
        let mut rng = LcgRng::new(1);
        let mut layer = SpikingLinear::new(8, 4, LifConfig::default(), &mut rng);
        let x = vec![0.5_f32; 8];
        let mut out = vec![0.0_f32; 4];
        layer.forward_step(&x, &mut out).expect("forward ok");
        assert_eq!(out.len(), 4);
        for &s in &out {
            assert!(s == 0.0 || s == 1.0);
        }
    }

    #[test]
    fn zero_w_with_bias_above_threshold_spikes() {
        let mut rng = LcgRng::new(2);
        let mut layer = SpikingLinear::new(8, 4, LifConfig::default(), &mut rng);
        // Zero out weights, raise bias above threshold.
        for w in &mut layer.w {
            *w = 0.0;
        }
        for b in &mut layer.b {
            *b = 2.0;
        }
        let x = vec![0.0_f32; 8];
        let mut out = vec![0.0_f32; 4];
        layer.forward_step(&x, &mut out).expect("forward ok");
        for &s in &out {
            assert_eq!(s, 1.0);
        }
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut rng = LcgRng::new(3);
        let mut layer = SpikingLinear::new(8, 4, LifConfig::default(), &mut rng);
        let x = vec![0.5_f32; 7];
        let mut out = vec![0.0_f32; 4];
        assert!(layer.forward_step(&x, &mut out).is_err());
    }
}
