#![allow(clippy::needless_range_loop)]
//! STDP-driven Self-Organising Map (Kohonen 1982) realised through spiking
//! competition and a Hebbian STDP-like update.
//!
//! A 2-D grid of `grid_w × grid_h` output units each holds an incoming weight
//! vector of length `in_dim`. A feed-forward input pattern is presented; the
//! unit whose weight vector best matches the input — i.e. the most strongly
//! driven / first-to-spike unit under winner-take-all competition — becomes the
//! *best matching unit* (BMU). An STDP-like Hebbian update then moves the BMU's
//! incoming weights — and those of its grid neighbours, weighted by a Gaussian
//! neighbourhood kernel over grid distance — toward the input pattern:
//!
//! ```text
//! bmu       = argmin_u ‖input − w_u‖²                   (winner-take-all)
//! h_u(t)    = exp(−d(u, bmu)² / (2 · σ(t)²))            (neighbourhood kernel)
//! σ(t)      = σ0 · exp(−t / anneal_tau)                 (radius annealing)
//! η(t)      = lr0 · exp(−t / anneal_tau)                (rate annealing)
//! Δw_u,k    = η(t) · h_u(t) · (input_k − w_u,k)         (Hebbian pull)
//! ```
//!
//! The BMU is the minimum-Euclidean-distance unit; for L2-normalised inputs and
//! weights this is exactly the maximum dot-product (membrane drive) unit, so the
//! distance criterion *is* the spiking winner-take-all competition. The
//! multiplicative `(input − w)` update is the standard Kohonen rule and the
//! fixed-point of STDP-style Hebbian learning under input normalisation:
//! co-active input/unit pairs grow the weight toward the input while the decay
//! term `−w` provides competition, so the winner's receptive field migrates onto
//! the presented cluster. Both the neighbourhood radius and the learning rate
//! anneal toward zero with iteration, freezing the map.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Configuration for the STDP-driven self-organising map.
#[derive(Debug, Clone, Copy)]
pub struct StdpSomConfig {
    /// Grid width (number of columns).
    pub grid_w: usize,
    /// Grid height (number of rows).
    pub grid_h: usize,
    /// Input / weight dimensionality per unit.
    pub in_dim: usize,
    /// Initial neighbourhood radius `σ0` (> 0).
    pub sigma0: f32,
    /// Initial learning rate `lr0` (> 0).
    pub lr0: f32,
    /// Annealing time constant for `σ` and `lr` (> 0).
    pub anneal_tau: f32,
}

impl Default for StdpSomConfig {
    fn default() -> Self {
        Self {
            grid_w: 4,
            grid_h: 4,
            in_dim: 2,
            sigma0: 2.0,
            lr0: 0.5,
            anneal_tau: 100.0,
        }
    }
}

impl StdpSomConfig {
    /// Construct and validate a SOM configuration.
    ///
    /// # Errors
    /// Returns [`SnnError::BadDim`] for any zero dimension and
    /// [`SnnError::OutOfRange`] for non-positive / non-finite `sigma0`, `lr0`,
    /// or `anneal_tau`.
    pub fn new(
        grid_w: usize,
        grid_h: usize,
        in_dim: usize,
        sigma0: f32,
        lr0: f32,
        anneal_tau: f32,
    ) -> SnnResult<Self> {
        let cfg = Self {
            grid_w,
            grid_h,
            in_dim,
            sigma0,
            lr0,
            anneal_tau,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration fields.
    ///
    /// # Errors
    /// See [`StdpSomConfig::new`].
    pub fn validate(&self) -> SnnResult<()> {
        if self.grid_w == 0 {
            return Err(SnnError::BadDim { got: self.grid_w });
        }
        if self.grid_h == 0 {
            return Err(SnnError::BadDim { got: self.grid_h });
        }
        if self.in_dim == 0 {
            return Err(SnnError::BadDim { got: self.in_dim });
        }
        if self.sigma0 <= 0.0 || !self.sigma0.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "sigma0".into(),
                val: self.sigma0,
            });
        }
        if self.lr0 <= 0.0 || !self.lr0.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "lr0".into(),
                val: self.lr0,
            });
        }
        if self.anneal_tau <= 0.0 || !self.anneal_tau.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "anneal_tau".into(),
                val: self.anneal_tau,
            });
        }
        Ok(())
    }

    /// Total number of grid units (`grid_w · grid_h`).
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.grid_w * self.grid_h
    }
}

/// An STDP-driven self-organising map.
#[derive(Debug, Clone)]
pub struct StdpSom {
    /// Configuration.
    pub cfg: StdpSomConfig,
    /// Weight matrix `[n_units × in_dim]` row-major (`weights[u*in_dim + k]`).
    pub weights: Vec<f32>,
}

impl StdpSom {
    /// Create a new SOM with weights initialised uniformly in `[0, 1)` from the
    /// supplied RNG.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid.
    pub fn new(cfg: StdpSomConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        cfg.validate()?;
        let n = cfg.n_units() * cfg.in_dim;
        let mut weights = vec![0.0_f32; n];
        for w in weights.iter_mut() {
            *w = rng.next_f32();
        }
        Ok(Self { cfg, weights })
    }

    /// Current learning rate `η(t) = lr0 · exp(−t / anneal_tau)`.
    #[must_use]
    pub fn learning_rate(&self, iter: usize) -> f32 {
        self.cfg.lr0 * (-(iter as f32) / self.cfg.anneal_tau).exp()
    }

    /// Current neighbourhood radius `σ(t) = σ0 · exp(−t / anneal_tau)`.
    #[must_use]
    pub fn neighbourhood_sigma(&self, iter: usize) -> f32 {
        self.cfg.sigma0 * (-(iter as f32) / self.cfg.anneal_tau).exp()
    }

    /// Map a flat unit index to its `(col, row)` grid coordinate.
    #[must_use]
    fn coord(&self, unit: usize) -> (usize, usize) {
        (unit % self.cfg.grid_w, unit / self.cfg.grid_w)
    }

    /// Squared Euclidean distance between an input and unit `u`'s weight vector.
    #[must_use]
    fn sq_dist(&self, unit: usize, input: &[f32]) -> f32 {
        let off = unit * self.cfg.in_dim;
        let mut acc = 0.0_f32;
        for k in 0..self.cfg.in_dim {
            let d = self.weights[off + k] - input[k];
            acc += d * d;
        }
        acc
    }

    /// Return the index of the best matching unit: the one whose weight vector
    /// is closest to `input` (minimum squared Euclidean distance), i.e. the
    /// winner-take-all spiking competition under normalised drive.
    ///
    /// # Errors
    /// Returns [`SnnError::BadShape`] if `input.len() != in_dim`.
    pub fn winner(&self, input: &[f32]) -> SnnResult<usize> {
        if input.len() != self.cfg.in_dim {
            return Err(SnnError::BadShape {
                expected: self.cfg.in_dim,
                got: input.len(),
            });
        }
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for u in 0..self.cfg.n_units() {
            let d = self.sq_dist(u, input);
            if d < best_dist {
                best_dist = d;
                best = u;
            }
        }
        Ok(best)
    }

    /// Run one competitive STDP-SOM update for `input` at iteration `iter`.
    ///
    /// Selects the BMU, then pulls the BMU and its grid neighbourhood toward the
    /// input with annealed rate and radius. Returns the winning unit index.
    ///
    /// # Errors
    /// Returns [`SnnError::BadShape`] if `input.len() != in_dim`.
    pub fn train_step(&mut self, input: &[f32], iter: usize) -> SnnResult<usize> {
        if input.len() != self.cfg.in_dim {
            return Err(SnnError::BadShape {
                expected: self.cfg.in_dim,
                got: input.len(),
            });
        }
        let bmu = self.winner(input)?;
        let (bx, by) = self.coord(bmu);
        let sigma = self.neighbourhood_sigma(iter).max(1e-6);
        let lr = self.learning_rate(iter);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let in_dim = self.cfg.in_dim;
        for u in 0..self.cfg.n_units() {
            let (ux, uy) = self.coord(u);
            let dx = ux as f32 - bx as f32;
            let dy = uy as f32 - by as f32;
            let grid_sq = dx * dx + dy * dy;
            let h = (-grid_sq / two_sigma_sq).exp();
            let scale = lr * h;
            if scale <= f32::EPSILON {
                continue;
            }
            let off = u * in_dim;
            for k in 0..in_dim {
                let w = &mut self.weights[off + k];
                *w += scale * (input[k] - *w);
            }
        }
        Ok(bmu)
    }

    /// Mean quantisation error: average Euclidean distance of each input to its
    /// best matching unit's weight vector.
    ///
    /// `inputs` is `[n_samples × in_dim]` row-major.
    ///
    /// # Errors
    /// Returns [`SnnError::EmptyInput`] if `inputs` is empty and
    /// [`SnnError::BadShape`] if its length is not a multiple of `in_dim`.
    pub fn quantization_error(&self, inputs: &[f32]) -> SnnResult<f32> {
        if inputs.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let in_dim = self.cfg.in_dim;
        if !inputs.len().is_multiple_of(in_dim) {
            return Err(SnnError::BadShape {
                expected: in_dim,
                got: inputs.len(),
            });
        }
        let n_samples = inputs.len() / in_dim;
        let mut total = 0.0_f32;
        for s in 0..n_samples {
            let sample = &inputs[s * in_dim..(s + 1) * in_dim];
            let bmu = self.winner(sample)?;
            total += self.sq_dist(bmu, sample).sqrt();
        }
        Ok(total / n_samples as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> StdpSomConfig {
        StdpSomConfig {
            grid_w: 4,
            grid_h: 4,
            in_dim: 2,
            sigma0: 2.0,
            lr0: 0.5,
            anneal_tau: 200.0,
        }
    }

    /// Three separated 2-D clusters laid out far apart in the unit square.
    fn clusters() -> [[f32; 2]; 3] {
        [[0.05, 0.05], [0.95, 0.05], [0.5, 0.95]]
    }

    #[test]
    fn new_initialises_in_unit_range() {
        let mut rng = LcgRng::new(42);
        let som = StdpSom::new(cfg(), &mut rng).expect("ok");
        assert_eq!(som.weights.len(), 16 * 2);
        for &w in &som.weights {
            assert!((0.0..1.0).contains(&w));
        }
    }

    #[test]
    fn training_reduces_quantization_error_and_separates_clusters() {
        let mut rng = LcgRng::new(7);
        let mut som = StdpSom::new(cfg(), &mut rng).expect("ok");
        let cl = clusters();
        // Flatten all cluster centroids as the evaluation set.
        let eval: Vec<f32> = cl.iter().flat_map(|c| c.iter().copied()).collect();
        let qe_start = som.quantization_error(&eval).expect("ok");

        // Train by cycling through the clusters with small jitter.
        for iter in 0..600 {
            let which = rng.next_usize(cl.len());
            let jitter_x = (rng.next_f32() - 0.5) * 0.02;
            let jitter_y = (rng.next_f32() - 0.5) * 0.02;
            let sample = [cl[which][0] + jitter_x, cl[which][1] + jitter_y];
            som.train_step(&sample, iter).expect("ok");
        }

        let qe_end = som.quantization_error(&eval).expect("ok");
        assert!(
            qe_end < qe_start,
            "quantization error should decrease: {qe_start} → {qe_end}"
        );

        // Distinct clusters should map to distinct BMUs after self-organisation.
        let b0 = som.winner(&cl[0]).expect("ok");
        let b1 = som.winner(&cl[1]).expect("ok");
        let b2 = som.winner(&cl[2]).expect("ok");
        assert!(
            b0 != b1 && b1 != b2 && b0 != b2,
            "separated clusters mapped to the same BMU: {b0}, {b1}, {b2}"
        );
    }

    #[test]
    fn neighbourhood_shrinks_with_iteration() {
        let mut rng = LcgRng::new(1);
        let som = StdpSom::new(cfg(), &mut rng).expect("ok");
        let s_early = som.neighbourhood_sigma(0);
        let s_mid = som.neighbourhood_sigma(200);
        let s_late = som.neighbourhood_sigma(2000);
        assert!(s_early > s_mid, "{s_early} !> {s_mid}");
        assert!(s_mid > s_late, "{s_mid} !> {s_late}");
        assert!(s_late < 0.05, "radius should be ~frozen late: {s_late}");
        // Learning rate anneals likewise.
        assert!(som.learning_rate(0) > som.learning_rate(2000));
    }

    #[test]
    fn winner_rejects_bad_input_len() {
        let mut rng = LcgRng::new(3);
        let som = StdpSom::new(cfg(), &mut rng).expect("ok");
        let err = som.winner(&[0.1, 0.2, 0.3]);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn quantization_error_rejects_ragged_input() {
        let mut rng = LcgRng::new(5);
        let som = StdpSom::new(cfg(), &mut rng).expect("ok");
        // 3 floats with in_dim=2 is not a whole number of samples.
        let err = som.quantization_error(&[0.1, 0.2, 0.3]);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
        let empty = som.quantization_error(&[]);
        assert!(matches!(empty, Err(SnnError::EmptyInput)));
    }

    #[test]
    fn config_rejects_zero_dim() {
        let err = StdpSomConfig::new(0, 4, 2, 1.0, 0.5, 100.0);
        assert!(matches!(err, Err(SnnError::BadDim { .. })));
        let err2 = StdpSomConfig::new(4, 4, 2, 0.0, 0.5, 100.0);
        assert!(matches!(err2, Err(SnnError::OutOfRange { .. })));
    }
}
