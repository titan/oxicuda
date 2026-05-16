#![allow(clippy::needless_range_loop)]
//! Liquid State Machine (Maass et al. 2002) — a sparse random recurrent reservoir
//! of spiking neurons whose high-dimensional dynamic projection is read out by a
//! simple linear regressor.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Configuration of the LSM reservoir.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Number of recurrent neurons.
    pub n_neurons: usize,
    /// Probability of any (i, j) recurrent edge being non-zero.
    pub density: f32,
    /// Target spectral radius of `W_rec` after rescaling.
    pub spectral_radius: f32,
    /// Scale of `W_in` after Kaiming initialisation.
    pub w_in_scale: f32,
    /// LCG seed for reproducible reservoirs.
    pub seed: u64,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            n_neurons: 200,
            density: 0.1,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed: 42,
        }
    }
}

/// Liquid State Machine state container.
#[derive(Debug, Clone)]
pub struct Lsm {
    /// Recurrent weight matrix flattened `[n, n]`.
    pub w_rec: Vec<f32>,
    /// Input weight matrix `[n, in_dim]`.
    pub w_in: Vec<f32>,
    /// Persistent membrane state.
    pub state: LifState,
    /// Last spike pattern for recurrent feedback.
    pub last_spikes: Vec<f32>,
    /// Number of recurrent neurons.
    pub n: usize,
    /// Input dimensionality.
    pub in_dim: usize,
}

/// Estimate the spectral radius `ρ(W)` of an `n × n` matrix via power iteration.
#[must_use]
pub fn power_iteration_spectral_radius(w: &[f32], n: usize, n_iter: usize) -> f32 {
    if n == 0 || w.len() != n * n {
        return 0.0;
    }
    let mut v = vec![1.0_f32 / (n as f32).sqrt(); n];
    let mut lambda = 0.0_f32;
    for _ in 0..n_iter {
        let mut next = vec![0.0_f32; n];
        for i in 0..n {
            let mut acc = 0.0_f32;
            for j in 0..n {
                acc += w[i * n + j] * v[j];
            }
            next[i] = acc;
        }
        let norm = next.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-30 {
            return 0.0;
        }
        for x in &mut next {
            *x /= norm;
        }
        // Rayleigh quotient: λ = vᵀ W v
        let mut wv = vec![0.0_f32; n];
        for i in 0..n {
            let mut acc = 0.0_f32;
            for j in 0..n {
                acc += w[i * n + j] * next[j];
            }
            wv[i] = acc;
        }
        lambda = next
            .iter()
            .zip(wv.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .abs();
        v = next;
    }
    lambda
}

impl Lsm {
    /// Build a new reservoir according to `cfg` and the LIF dynamics in `lif_cfg`.
    pub fn new(in_dim: usize, cfg: &LsmConfig, _lif_cfg: &LifConfig) -> SnnResult<Self> {
        if cfg.n_neurons == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if !(0.0..=1.0).contains(&cfg.density) || cfg.density.is_nan() {
            return Err(SnnError::OutOfRange {
                name: "density".to_string(),
                val: cfg.density,
            });
        }
        let mut rng = LcgRng::new(cfg.seed);
        let n = cfg.n_neurons;

        // W_rec: sparse, each entry independently active with probability density.
        let mut w_rec = vec![0.0_f32; n * n];
        let weight_scale = (1.0_f32 / (cfg.density.max(1e-6) * n as f32)).sqrt();
        for w in &mut w_rec {
            if rng.next_f32() < cfg.density {
                *w = rng.next_normal() * weight_scale;
            }
        }

        // Rescale to target spectral radius.
        let rho = power_iteration_spectral_radius(&w_rec, n, 30);
        if rho > 1e-12 {
            let scale = cfg.spectral_radius / rho;
            for w in &mut w_rec {
                *w *= scale;
            }
        }

        // W_in: dense Kaiming-normal scaled by `w_in_scale`.
        let in_scale = (2.0_f32 / in_dim.max(1) as f32).sqrt() * cfg.w_in_scale;
        let mut w_in = vec![0.0_f32; n * in_dim];
        rng.fill_normal(&mut w_in);
        for v in &mut w_in {
            *v *= in_scale;
        }

        Ok(Self {
            w_rec,
            w_in,
            state: LifState::new(n),
            last_spikes: vec![0.0_f32; n],
            n,
            in_dim,
        })
    }

    /// One forward timestep on input `x`, writing spikes to `out`.
    pub fn forward_step(
        &mut self,
        x: &[f32],
        lif_cfg: &LifConfig,
        out: &mut [f32],
    ) -> SnnResult<()> {
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
        lif_step(&mut self.state, &current, lif_cfg, out)?;
        self.last_spikes.copy_from_slice(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_radius_close_to_target() {
        let cfg = LsmConfig {
            n_neurons: 50,
            density: 0.2,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed: 7,
        };
        let lif = LifConfig::default();
        let lsm = Lsm::new(4, &cfg, &lif).expect("ok");
        let rho = power_iteration_spectral_radius(&lsm.w_rec, lsm.n, 50);
        assert!(
            (rho - 0.9).abs() < 0.20,
            "spectral radius {rho} not close to target 0.9"
        );
    }

    #[test]
    fn density_correct() {
        let cfg = LsmConfig {
            n_neurons: 80,
            density: 0.25,
            spectral_radius: 0.5,
            w_in_scale: 1.0,
            seed: 11,
        };
        let lif = LifConfig::default();
        let lsm = Lsm::new(8, &cfg, &lif).expect("ok");
        let nonzeros = lsm.w_rec.iter().filter(|&&x| x != 0.0).count();
        let total = (lsm.n * lsm.n) as f32;
        let observed = nonzeros as f32 / total;
        assert!(
            (observed - 0.25).abs() < 0.07,
            "density {observed} not close to target 0.25"
        );
    }

    #[test]
    fn forward_shape() {
        let cfg = LsmConfig::default();
        let lif = LifConfig::default();
        let mut lsm = Lsm::new(4, &cfg, &lif).expect("ok");
        let x = vec![0.5_f32; 4];
        let mut out = vec![0.0_f32; cfg.n_neurons];
        lsm.forward_step(&x, &lif, &mut out).expect("ok");
        assert_eq!(out.len(), cfg.n_neurons);
    }
}
