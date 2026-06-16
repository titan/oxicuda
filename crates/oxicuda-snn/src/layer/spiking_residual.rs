//! Spiking residual (skip-connection) layer — a spiking analogue of a ResNet block.
//!
//! A residual block routes its input through a learned transform *and* an
//! identity short-cut, summing the two before the non-linearity:
//!
//! ```text
//! v_{t+1} = decay · v_t + W · x_t + α · x_t        (membrane integration)
//! s_{t+1} = 1 if v_{t+1} ≥ v_th else 0             (surrogate-ready spike)
//! v_{t+1} ← reset(v_{t+1}, s_{t+1})                (hard / soft reset)
//! ```
//!
//! Here `W · x_t` is the transform branch and `α · x_t` is the identity skip
//! (`α = skip_scale`, default `1`). Because the skip is added directly into the
//! membrane drive, the residual path is *explicit* in [`crate::layer::spiking_residual::SpikingResidual::forward_step`]
//! rather than hidden inside a sub-layer, which keeps the dynamics inspectable
//! and surrogate-gradient ready. Spiking ResNets (e.g. SEW-ResNet, MS-ResNet)
//! rely on exactly this additive short-cut to train very deep SNNs.
//!
//! The identity skip requires the input and output to share a dimension `dim`,
//! so the transform matrix `W` is square (`[dim × dim]`, row-major).

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::ResetMode;

/// Configuration for a [`SpikingResidual`] layer.
#[derive(Debug, Clone, Copy)]
pub struct SpikingResidualConfig {
    /// Shared input/output dimension; must be `> 0`.
    pub dim: usize,
    /// Membrane leak factor `β = exp(−dt / τ_m)`; must lie in `[0, 1]`.
    pub decay: f32,
    /// Spike threshold; must be finite.
    pub v_th: f32,
    /// Scale `α` applied to the identity skip term; `0` disables the residual.
    pub skip_scale: f32,
    /// Reset behaviour applied after a spike.
    pub reset: ResetMode,
}

impl Default for SpikingResidualConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            decay: 0.9,
            v_th: 1.0,
            skip_scale: 1.0,
            reset: ResetMode::Hard,
        }
    }
}

/// Spiking residual layer with an explicit additive identity skip connection.
#[derive(Debug, Clone)]
pub struct SpikingResidual {
    /// Transform weight matrix `W`, row-major `[dim × dim]`.
    pub w: Vec<f32>,
    /// Persistent membrane potential, length `dim`.
    pub v: Vec<f32>,
    /// Shared input/output dimension.
    pub dim: usize,
    /// Membrane leak factor.
    pub decay: f32,
    /// Spike threshold.
    pub v_th: f32,
    /// Identity-skip scale `α`.
    pub skip_scale: f32,
    /// Reset behaviour after a spike.
    pub reset: ResetMode,
}

/// Validate the shared parts of a [`SpikingResidualConfig`].
fn validate_cfg(cfg: &SpikingResidualConfig) -> SnnResult<()> {
    if cfg.dim == 0 {
        return Err(SnnError::BadDim { got: cfg.dim });
    }
    if !cfg.v_th.is_finite() {
        return Err(SnnError::BadThreshold { v_th: cfg.v_th });
    }
    if !cfg.decay.is_finite() || !(0.0..=1.0).contains(&cfg.decay) {
        return Err(SnnError::OutOfRange {
            name: "decay".to_string(),
            val: cfg.decay,
        });
    }
    if !cfg.skip_scale.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "skip_scale".to_string(),
            val: cfg.skip_scale,
        });
    }
    Ok(())
}

impl SpikingResidual {
    /// Allocate a layer with Kaiming-normal transform weights and zero membrane.
    ///
    /// Returns [`SnnError::BadDim`] if `dim == 0`, [`SnnError::BadThreshold`] if
    /// `v_th` is non-finite, and [`SnnError::OutOfRange`] if `decay ∉ [0, 1]` or
    /// `skip_scale` is non-finite.
    pub fn new(cfg: SpikingResidualConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        let scale = (2.0_f32 / cfg.dim as f32).sqrt();
        let mut w = vec![0.0_f32; cfg.dim * cfg.dim];
        rng.fill_normal(&mut w);
        for v in &mut w {
            *v *= scale;
        }
        Ok(Self {
            w,
            v: vec![0.0_f32; cfg.dim],
            dim: cfg.dim,
            decay: cfg.decay,
            v_th: cfg.v_th,
            skip_scale: cfg.skip_scale,
            reset: cfg.reset,
        })
    }

    /// Build a layer from an explicit transform matrix `w` of length `dim²`.
    ///
    /// Useful for tests and for loading pre-trained weights. Returns
    /// [`SnnError::BadShape`] if `w.len() != dim · dim`, plus the same config
    /// validation as [`Self::new`].
    pub fn from_weights(w: Vec<f32>, cfg: SpikingResidualConfig) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        if w.len() != cfg.dim * cfg.dim {
            return Err(SnnError::BadShape {
                expected: cfg.dim * cfg.dim,
                got: w.len(),
            });
        }
        Ok(Self {
            w,
            v: vec![0.0_f32; cfg.dim],
            dim: cfg.dim,
            decay: cfg.decay,
            v_th: cfg.v_th,
            skip_scale: cfg.skip_scale,
            reset: cfg.reset,
        })
    }

    /// Reset the persistent membrane potential to zero.
    pub fn reset_state(&mut self) {
        for v in &mut self.v {
            *v = 0.0;
        }
    }

    /// One timestep of residual forward computation, writing spikes to `out`.
    ///
    /// The membrane drive is `W · x + skip_scale · x`, making the residual
    /// short-cut an explicit additive term. Returns [`SnnError::BadShape`] if
    /// either slice length differs from `dim`.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) -> SnnResult<()> {
        if x.len() != self.dim {
            return Err(SnnError::BadShape {
                expected: self.dim,
                got: x.len(),
            });
        }
        if out.len() != self.dim {
            return Err(SnnError::BadShape {
                expected: self.dim,
                got: out.len(),
            });
        }
        let dim = self.dim;
        let decay = self.decay;
        let v_th = self.v_th;
        let skip_scale = self.skip_scale;
        let reset = self.reset;
        for (i, ((v_i, out_i), &x_i)) in self
            .v
            .iter_mut()
            .zip(out.iter_mut())
            .zip(x.iter())
            .enumerate()
        {
            let row = &self.w[i * dim..(i + 1) * dim];
            let transform: f32 = row
                .iter()
                .zip(x.iter())
                .map(|(&w_ij, &x_j)| w_ij * x_j)
                .sum();
            let v_new = decay * *v_i + transform + skip_scale * x_i;
            let spike = if v_new >= v_th { 1.0_f32 } else { 0.0_f32 };
            *v_i = match reset {
                ResetMode::Hard => (1.0 - spike) * v_new,
                ResetMode::Soft => v_new - spike * v_th,
            };
            *out_i = spike;
        }
        Ok(())
    }

    /// Run a whole sequence given a flat `[seq_len · dim]` input, returning the
    /// flat `[seq_len · dim]` spike output. The membrane state persists across
    /// the sequence (it is *not* reset between steps).
    ///
    /// Returns [`SnnError::EmptyInput`] for an empty slice and
    /// [`SnnError::BadShape`] if the length is not a multiple of `dim`.
    pub fn forward_seq(&mut self, input: &[f32]) -> SnnResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if !input.len().is_multiple_of(self.dim) {
            return Err(SnnError::BadShape {
                expected: self.dim,
                got: input.len(),
            });
        }
        let seq_len = input.len() / self.dim;
        let mut out = vec![0.0_f32; input.len()];
        for t in 0..seq_len {
            let x = &input[t * self.dim..(t + 1) * self.dim];
            // Split the output borrow from the (immutable) input borrow.
            let mut step = vec![0.0_f32; self.dim];
            self.forward_step(x, &mut step)?;
            out[t * self.dim..(t + 1) * self.dim].copy_from_slice(&step);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg(dim: usize) -> SpikingResidualConfig {
        SpikingResidualConfig {
            dim,
            decay: 0.0,
            v_th: 0.5,
            skip_scale: 1.0,
            reset: ResetMode::Hard,
        }
    }

    #[test]
    fn output_shapes_correct() {
        let mut rng = LcgRng::new(1);
        let mut layer = SpikingResidual::new(base_cfg(4), &mut rng).expect("ctor");
        let seq_len = 5;
        let input = vec![0.3_f32; seq_len * 4];
        let out = layer.forward_seq(&input).expect("seq");
        assert_eq!(out.len(), seq_len * 4);
    }

    #[test]
    fn spikes_are_binary() {
        let mut rng = LcgRng::new(2);
        let mut layer = SpikingResidual::new(base_cfg(6), &mut rng).expect("ctor");
        let input = vec![0.8_f32; 4 * 6];
        let out = layer.forward_seq(&input).expect("seq");
        for &s in &out {
            assert!(s == 0.0 || s == 1.0, "non-binary spike {s}");
        }
    }

    #[test]
    fn skip_changes_output() {
        // Diagonal transform of 0.1 keeps W·x below threshold; the identity skip
        // pushes the membrane over it, so the residual must alter the spikes.
        let dim = 3;
        let mut w = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 0.1;
        }
        let cfg = base_cfg(dim); // decay 0, v_th 0.5, skip_scale 1
        let mut with_skip = SpikingResidual::from_weights(w.clone(), cfg).expect("ctor");
        let mut no_skip = with_skip.clone();
        no_skip.skip_scale = 0.0;

        let x = vec![1.0_f32; dim];
        let mut out_with = vec![0.0_f32; dim];
        let mut out_without = vec![0.0_f32; dim];
        with_skip.forward_step(&x, &mut out_with).expect("step");
        no_skip.forward_step(&x, &mut out_without).expect("step");

        assert_eq!(out_with, vec![1.0_f32; dim], "skip should drive spikes");
        assert_eq!(
            out_without,
            vec![0.0_f32; dim],
            "no-skip stays sub-threshold"
        );
        assert_ne!(out_with, out_without, "skip term did not change output");
    }

    #[test]
    fn zero_input_zero_skip_no_spikes() {
        let dim = 4;
        let mut rng = LcgRng::new(3);
        let mut cfg = base_cfg(dim);
        cfg.skip_scale = 0.0;
        let mut layer = SpikingResidual::new(cfg, &mut rng).expect("ctor");
        let x = vec![0.0_f32; dim];
        let mut out = vec![0.0_f32; dim];
        layer.forward_step(&x, &mut out).expect("step");
        assert!(
            out.iter().all(|&s| s == 0.0),
            "spurious spikes on zero input"
        );
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut rng = LcgRng::new(4);
        let mut layer = SpikingResidual::new(base_cfg(4), &mut rng).expect("ctor");
        let mut out = vec![0.0_f32; 4];
        assert!(matches!(
            layer.forward_step(&[0.0; 3], &mut out),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            layer.forward_seq(&[0.0; 7]),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(layer.forward_seq(&[]), Err(SnnError::EmptyInput)));
    }

    #[test]
    fn bad_threshold_and_decay_are_errors() {
        let mut rng = LcgRng::new(5);
        let mut bad_th = base_cfg(3);
        bad_th.v_th = f32::NAN;
        assert!(matches!(
            SpikingResidual::new(bad_th, &mut rng),
            Err(SnnError::BadThreshold { .. })
        ));

        let mut bad_decay = base_cfg(3);
        bad_decay.decay = 1.5;
        assert!(matches!(
            SpikingResidual::new(bad_decay, &mut rng),
            Err(SnnError::OutOfRange { .. })
        ));

        let mut zero_dim = base_cfg(0);
        zero_dim.dim = 0;
        assert!(matches!(
            SpikingResidual::new(zero_dim, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
    }
}
