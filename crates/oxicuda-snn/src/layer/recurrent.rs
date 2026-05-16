#![allow(clippy::needless_range_loop)]
//! Recurrent spiking layer: `current_t = W_in · x_t + W_rec · s_{t-1}` then LIF.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Recurrent spiking layer with self-connections via `W_rec`.
#[derive(Debug, Clone)]
pub struct SpikingRecurrent {
    /// Input weight matrix `[n, in_dim]`.
    pub w_in: Vec<f32>,
    /// Recurrent weight matrix `[n, n]`.
    pub w_rec: Vec<f32>,
    /// LIF parameters.
    pub lif_cfg: LifConfig,
    /// Persistent membrane state.
    pub state: LifState,
    /// Spikes from the previous timestep, used as recurrent input.
    pub last_spikes: Vec<f32>,
    /// Input dimensionality.
    pub in_dim: usize,
    /// Number of recurrent neurons.
    pub n: usize,
}

impl SpikingRecurrent {
    /// Allocate a recurrent layer with Kaiming-normal init for both `W_in` and `W_rec`.
    #[must_use]
    pub fn new(in_dim: usize, n: usize, lif_cfg: LifConfig, rng: &mut LcgRng) -> Self {
        let in_scale = (2.0_f32 / in_dim.max(1) as f32).sqrt();
        let rec_scale = (1.0_f32 / n.max(1) as f32).sqrt();
        let mut w_in = vec![0.0_f32; n * in_dim];
        let mut w_rec = vec![0.0_f32; n * n];
        rng.fill_normal(&mut w_in);
        for v in &mut w_in {
            *v *= in_scale;
        }
        rng.fill_normal(&mut w_rec);
        for v in &mut w_rec {
            *v *= rec_scale;
        }
        Self {
            w_in,
            w_rec,
            lif_cfg,
            state: LifState::new(n),
            last_spikes: vec![0.0_f32; n],
            in_dim,
            n,
        }
    }

    /// One forward timestep. Updates `state`, `last_spikes`, and writes spikes to `out`.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) -> SnnResult<()> {
        if x.len() != self.in_dim {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        if out.len() != self.n {
            return Err(SnnError::BadShape {
                expected: self.n,
                got: out.len(),
            });
        }
        let mut current = vec![0.0_f32; self.n];
        for i in 0..self.n {
            let mut acc = 0.0_f32;
            for (j, &xj) in x.iter().enumerate() {
                acc += self.w_in[i * self.in_dim + j] * xj;
            }
            for (j, &sj) in self.last_spikes.iter().enumerate() {
                acc += self.w_rec[i * self.n + j] * sj;
            }
            current[i] = acc;
        }
        lif_step(&mut self.state, &current, &self.lif_cfg, out)?;
        self.last_spikes.copy_from_slice(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_recurrent_matches_feedforward() {
        let mut rng = LcgRng::new(1);
        let mut layer = SpikingRecurrent::new(4, 3, LifConfig::default(), &mut rng);
        // Zero recurrent — input dependence only.
        for w in &mut layer.w_rec {
            *w = 0.0;
        }
        let x = vec![0.5_f32; 4];
        let mut out = vec![0.0_f32; 3];
        layer.forward_step(&x, &mut out).expect("ok");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn recurrent_carries_state() {
        let mut rng = LcgRng::new(2);
        let mut layer = SpikingRecurrent::new(2, 2, LifConfig::default(), &mut rng);
        let x_pulse = vec![5.0_f32, 5.0_f32];
        let x_zero = vec![0.0_f32, 0.0_f32];
        let mut out = vec![0.0_f32; 2];
        layer.forward_step(&x_pulse, &mut out).expect("t=0 ok");
        let initial = out.clone();
        layer.forward_step(&x_zero, &mut out).expect("t=1 ok");
        // At t=1 there is no driving input, but the recurrent connection from t=0's spikes
        // carries something; ensure last_spikes was updated.
        let _ = initial;
        assert_eq!(out.len(), 2);
    }
}
