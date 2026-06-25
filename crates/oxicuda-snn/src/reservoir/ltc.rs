#![allow(clippy::needless_range_loop)]
//! Liquid Time-Constant Network (LTC) — Hasani, Lechner, Amini, Rus & Grosu
//! 2021, "Liquid Time-constant Networks" (AAAI 2021).
//!
//! An LTC is a continuous-time recurrent network whose per-neuron time constant
//! is **input-dependent**. The hidden state `x ∈ ℝ^H` obeys the ODE
//!
//! ```text
//! dx/dt = −[ 1/τ + f(x, I, t, θ) ] · x  +  f(x, I, t, θ) · A
//! ```
//!
//! where `A ∈ ℝ^H` is a learnable reversal / bias vector, `τ ∈ ℝ^H` a per-neuron
//! base time constant, and `f` a bounded nonlinearity (here the logistic
//! sigmoid) of the synaptic drive:
//!
//! ```text
//! f = σ( W_in · I + W_rec · x + b )         σ(z) = 1 / (1 + e^{−z}) ∈ (0, 1)
//! ```
//!
//! Because `f` multiplies `x` inside the decay term, the **effective time
//! constant** `τ_eff = 1 / (1/τ + f)` shrinks as `f` grows — strong drive makes
//! the neuron respond faster. The network is integrated with the paper's fused
//! (semi-implicit / stable) Euler update, which keeps `x` bounded for any step
//! size:
//!
//! ```text
//! x(t + Δt) = ( x(t) + Δt · f · A ) / ( 1 + Δt · (1/τ + f) )
//! ```
//!
//! Each [`crate::reservoir::ltc::LtcCell::step`] applies `n_unfold` such fused substeps per input
//! sample. This network is **non-spiking** (analog / rate valued) but is
//! reservoir-adjacent: with `W_rec` fixed it behaves as a trainable-readout
//! liquid.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Configuration of an [`LtcCell`].
#[derive(Debug, Clone, Copy)]
pub struct LtcConfig {
    /// Input dimensionality. Must be `> 0`.
    pub in_dim: usize,
    /// Number of hidden neurons `H`. Must be `> 0`.
    pub hidden: usize,
    /// Integration step `Δt` of a single fused-Euler substep. Must be `> 0`.
    pub dt: f32,
    /// Base time constant `τ` shared by all neurons at initialisation.
    /// Must be `> 0`.
    pub tau_base: f32,
    /// Number of fused-Euler substeps applied per [`crate::reservoir::ltc::LtcCell::step`]. Must be `> 0`.
    pub n_unfold: usize,
}

impl Default for LtcConfig {
    fn default() -> Self {
        Self {
            in_dim: 1,
            hidden: 32,
            dt: 0.1,
            tau_base: 1.0,
            n_unfold: 6,
        }
    }
}

impl LtcConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadDim`] if `in_dim` or `hidden` is zero.
    /// * [`SnnError::BadDt`] if `dt` is non-positive or non-finite.
    /// * [`SnnError::BadTau`] if `tau_base` is non-positive or non-finite.
    /// * [`SnnError::BadTimesteps`] if `n_unfold` is zero.
    pub fn validate(&self) -> SnnResult<()> {
        if self.in_dim == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if self.hidden == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SnnError::BadDt { dt: self.dt });
        }
        if self.tau_base <= 0.0 || !self.tau_base.is_finite() {
            return Err(SnnError::BadTau { tau: self.tau_base });
        }
        if self.n_unfold == 0 {
            return Err(SnnError::BadTimesteps { got: 0 });
        }
        Ok(())
    }
}

/// Numerically-stable logistic sigmoid `σ(z) = 1 / (1 + e^{−z})`.
///
/// Evaluated in the branch that avoids overflow of `exp` for large `|z|`.
#[must_use]
#[inline]
pub fn stable_sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Liquid Time-Constant recurrent cell.
#[derive(Debug, Clone)]
pub struct LtcCell {
    /// Input weights `W_in`, row-major `[hidden × in_dim]`.
    pub w_in: Vec<f32>,
    /// Recurrent weights `W_rec`, row-major `[hidden × hidden]`.
    pub w_rec: Vec<f32>,
    /// Synaptic bias `b`, length `hidden`.
    pub b: Vec<f32>,
    /// Reversal / bias vector `A`, length `hidden`.
    pub a: Vec<f32>,
    /// Per-neuron base time constant `τ`, length `hidden` (all `> 0`).
    pub tau: Vec<f32>,
    /// Integration step `Δt`.
    dt: f32,
    /// Substeps per [`step`](Self::step).
    n_unfold: usize,
    /// Input dimensionality.
    in_dim: usize,
    /// Hidden dimensionality `H`.
    hidden: usize,
}

impl LtcCell {
    /// Build an LTC cell, randomly initialising `W_in`, `W_rec`, `b` and `A`
    /// from `rng`. The base time constant `τ` is set to `cfg.tau_base` for every
    /// neuron, and `A` is initialised small (near zero) so that, absent input,
    /// the state relaxes toward the origin.
    ///
    /// Weights use a fan-in scaling `1/√in_dim` (input) and `1/√hidden`
    /// (recurrent) to keep the synaptic drive in a well-conditioned range.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`LtcConfig::validate`].
    pub fn new(cfg: LtcConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        cfg.validate()?;
        let h = cfg.hidden;
        let d = cfg.in_dim;

        let in_scale = 1.0_f32 / (d as f32).sqrt();
        let rec_scale = 1.0_f32 / (h as f32).sqrt();

        let mut w_in = vec![0.0_f32; h * d];
        rng.fill_normal(&mut w_in);
        for v in &mut w_in {
            *v *= in_scale;
        }

        let mut w_rec = vec![0.0_f32; h * h];
        rng.fill_normal(&mut w_rec);
        for v in &mut w_rec {
            *v *= rec_scale;
        }

        // Small synaptic bias.
        let mut b = vec![0.0_f32; h];
        for v in &mut b {
            *v = 0.1 * rng.next_normal();
        }

        // Reversal vector A initialised near zero (so zero-input → decay to 0).
        let mut a = vec![0.0_f32; h];
        for v in &mut a {
            *v = 0.01 * rng.next_normal();
        }

        let tau = vec![cfg.tau_base; h];

        Ok(Self {
            w_in,
            w_rec,
            b,
            a,
            tau,
            dt: cfg.dt,
            n_unfold: cfg.n_unfold,
            in_dim: d,
            hidden: h,
        })
    }

    /// Input dimensionality.
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Hidden dimensionality `H`.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// A fresh zero hidden state of length `hidden`.
    #[must_use]
    pub fn zero_state(&self) -> Vec<f32> {
        vec![0.0_f32; self.hidden]
    }

    /// Advance `x_state` in place by `n_unfold` fused-Euler substeps under the
    /// constant input `input`.
    ///
    /// Each substep computes, for every neuron `i`,
    ///
    /// ```text
    /// z_i = b_i + Σ_j W_in[i,j]·I_j + Σ_k W_rec[i,k]·x_k
    /// f_i = σ(z_i)                                  ∈ (0, 1)
    /// x_i ← ( x_i + Δt·f_i·A_i ) / ( 1 + Δt·(1/τ_i + f_i) )
    /// ```
    ///
    /// The denominator `1 + Δt·(1/τ_i + f_i) > 1` strictly increases with `f_i`,
    /// shortening the effective time constant under stronger drive.
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadShape`] if `x_state.len() != hidden` or
    ///   `input.len() != in_dim`.
    pub fn step(&self, x_state: &mut [f32], input: &[f32]) -> SnnResult<()> {
        if x_state.len() != self.hidden {
            return Err(SnnError::BadShape {
                expected: self.hidden,
                got: x_state.len(),
            });
        }
        if input.len() != self.in_dim {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let h = self.hidden;
        let d = self.in_dim;
        let dt = self.dt;

        // Precompute the input contribution to z (constant across substeps).
        let mut input_drive = vec![0.0_f32; h];
        for i in 0..h {
            let mut acc = self.b[i];
            for j in 0..d {
                acc += self.w_in[i * d + j] * input[j];
            }
            input_drive[i] = acc;
        }

        let mut next = vec![0.0_f32; h];
        for _ in 0..self.n_unfold {
            for i in 0..h {
                // Recurrent contribution evaluated at the current state.
                let mut z = input_drive[i];
                let row = &self.w_rec[i * h..(i + 1) * h];
                for (k, &w_ik) in row.iter().enumerate() {
                    z += w_ik * x_state[k];
                }
                let f = stable_sigmoid(z);
                let inv_tau = 1.0 / self.tau[i];
                let numer = x_state[i] + dt * f * self.a[i];
                let denom = 1.0 + dt * (inv_tau + f);
                next[i] = numer / denom;
            }
            x_state.copy_from_slice(&next);
        }
        Ok(())
    }

    /// Run the cell over a sequence and return the hidden trajectory.
    ///
    /// `inputs` is row-major `[seq_len × in_dim]`. The hidden state starts at
    /// zero. The returned trajectory is row-major `[seq_len × hidden]` — the
    /// state *after* processing each timestep.
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadTimesteps`] if `seq_len == 0`.
    /// * [`SnnError::BadShape`] if `inputs.len() != seq_len * in_dim`.
    pub fn forward_seq(&self, inputs: &[f32], seq_len: usize) -> SnnResult<Vec<f32>> {
        if seq_len == 0 {
            return Err(SnnError::BadTimesteps { got: 0 });
        }
        let d = self.in_dim;
        if inputs.len() != seq_len * d {
            return Err(SnnError::BadShape {
                expected: seq_len * d,
                got: inputs.len(),
            });
        }
        let h = self.hidden;
        let mut traj = Vec::with_capacity(seq_len * h);
        let mut x = self.zero_state();
        for t in 0..seq_len {
            let u = &inputs[t * d..(t + 1) * d];
            self.step(&mut x, u)?;
            traj.extend_from_slice(&x);
        }
        Ok(traj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell(in_dim: usize, hidden: usize, seed: u64) -> LtcCell {
        let cfg = LtcConfig {
            in_dim,
            hidden,
            dt: 0.1,
            tau_base: 1.0,
            n_unfold: 6,
        };
        let mut rng = LcgRng::new(seed);
        LtcCell::new(cfg, &mut rng).expect("ctor")
    }

    #[test]
    fn sigmoid_is_stable_and_correct() {
        assert!((stable_sigmoid(0.0) - 0.5).abs() < 1e-6);
        // Symmetry σ(−z) = 1 − σ(z).
        assert!((stable_sigmoid(-3.0) - (1.0 - stable_sigmoid(3.0))).abs() < 1e-6);
        // No overflow / NaN at extremes.
        assert!(stable_sigmoid(1000.0).is_finite());
        assert!(stable_sigmoid(-1000.0).is_finite());
        assert!((stable_sigmoid(1000.0) - 1.0).abs() < 1e-6);
        assert!(stable_sigmoid(-1000.0).abs() < 1e-6);
    }

    #[test]
    fn new_matrix_shapes() {
        let cell = make_cell(3, 8, 1);
        assert_eq!(cell.w_in.len(), 8 * 3);
        assert_eq!(cell.w_rec.len(), 8 * 8);
        assert_eq!(cell.b.len(), 8);
        assert_eq!(cell.a.len(), 8);
        assert_eq!(cell.tau.len(), 8);
        assert!(cell.tau.iter().all(|&t| t > 0.0));
    }

    #[test]
    fn state_stays_bounded_for_bounded_input() {
        let cell = make_cell(4, 16, 7);
        let mut x = cell.zero_state();
        let input = vec![1.0_f32; 4]; // bounded drive
        for _ in 0..500 {
            cell.step(&mut x, &input).expect("step");
        }
        // With f ∈ (0,1), A small and the fused update contracting, the state
        // remains comfortably bounded.
        for (i, &v) in x.iter().enumerate() {
            assert!(v.is_finite(), "x[{i}] not finite");
            assert!(v.abs() < 10.0, "x[{i}]={v} blew up");
        }
    }

    #[test]
    fn zero_input_relaxes_toward_zero() {
        // A is near zero, so with no drive the fused update contracts x → 0.
        let cell = make_cell(2, 12, 3);
        let mut x = vec![1.0_f32; 12];
        let zero_in = vec![0.0_f32; 2];
        let initial_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for _ in 0..400 {
            cell.step(&mut x, &zero_in).expect("step");
        }
        let final_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            final_norm < 0.1 * initial_norm,
            "state failed to relax: |x0|={initial_norm}, |x|={final_norm}"
        );
    }

    #[test]
    fn larger_drive_shortens_effective_time_constant() {
        // The fused denominator 1 + dt·(1/τ + f) grows monotonically with f,
        // and f = σ(z) grows with z. Verify the denominator is larger under a
        // strong positive drive than a weak one, hence τ_eff is shorter.
        let dt = 0.1_f32;
        let tau = 1.0_f32;
        let inv_tau = 1.0 / tau;
        let f_weak = stable_sigmoid(0.0); // 0.5
        let f_strong = stable_sigmoid(6.0); // ≈ 0.9975
        let denom_weak = 1.0 + dt * (inv_tau + f_weak);
        let denom_strong = 1.0 + dt * (inv_tau + f_strong);
        assert!(
            denom_strong > denom_weak,
            "denominator must grow with f: weak={denom_weak}, strong={denom_strong}"
        );
        // Effective time constant τ_eff = 1/(1/τ + f) is correspondingly shorter.
        let tau_eff_weak = 1.0 / (inv_tau + f_weak);
        let tau_eff_strong = 1.0 / (inv_tau + f_strong);
        assert!(
            tau_eff_strong < tau_eff_weak,
            "τ_eff must shrink with f: weak={tau_eff_weak}, strong={tau_eff_strong}"
        );
    }

    #[test]
    fn strong_input_gives_faster_response_than_weak() {
        // Per-neuron fixed point of the fused update with constant drive `z` is
        // `x* = A·f / (1/τ + f)`, monotonically increasing in `f` for `A > 0`.
        // A larger input therefore raises `f = σ(z)` and pulls `x` further from
        // its zero start in a single step. To make the sign of `z` unambiguous
        // (the random `W_in`/`b` could otherwise flip it), we pin the weights so
        // that `z = input` exactly: `W_in = +1`, `W_rec = 0`, `b = 0`, `A = 1`.
        let mut cell = make_cell(1, 8, 21);
        for w in &mut cell.w_in {
            *w = 1.0;
        }
        for w in &mut cell.w_rec {
            *w = 0.0;
        }
        for b in &mut cell.b {
            *b = 0.0;
        }
        for a in &mut cell.a {
            *a = 1.0; // positive reversal so larger f pulls x up faster
        }
        let mut x_strong = cell.zero_state();
        let mut x_weak = cell.zero_state();
        cell.step(&mut x_strong, &[5.0]).expect("strong"); // f = σ(5) ≈ 0.993
        cell.step(&mut x_weak, &[0.0]).expect("weak"); // f = σ(0) = 0.5
        let move_strong: f32 = x_strong.iter().map(|v| v.abs()).sum();
        let move_weak: f32 = x_weak.iter().map(|v| v.abs()).sum();
        assert!(
            move_strong > move_weak,
            "strong drive should move state more: strong={move_strong}, weak={move_weak}"
        );
    }

    #[test]
    fn forward_seq_shape() {
        let cell = make_cell(3, 10, 5);
        let seq_len = 20;
        let inputs = vec![0.3_f32; seq_len * 3];
        let traj = cell.forward_seq(&inputs, seq_len).expect("seq");
        assert_eq!(traj.len(), seq_len * 10);
        assert!(traj.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_given_seed() {
        let cell_a = make_cell(2, 16, 1234);
        let cell_b = make_cell(2, 16, 1234);
        let inputs = vec![0.7_f32; 8 * 2];
        let ta = cell_a.forward_seq(&inputs, 8).expect("a");
        let tb = cell_b.forward_seq(&inputs, 8).expect("b");
        assert_eq!(ta.len(), tb.len());
        for (x, y) in ta.iter().zip(tb.iter()) {
            assert!((x - y).abs() < 1e-9, "non-deterministic LTC output");
        }
    }

    #[test]
    fn step_rejects_bad_shapes() {
        let cell = make_cell(3, 8, 9);
        let mut wrong_state = vec![0.0_f32; 7];
        assert!(matches!(
            cell.step(&mut wrong_state, &[0.0; 3]),
            Err(SnnError::BadShape { .. })
        ));
        let mut good_state = cell.zero_state();
        assert!(matches!(
            cell.step(&mut good_state, &[0.0; 2]),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn forward_seq_rejects_bad_input() {
        let cell = make_cell(2, 4, 11);
        assert!(matches!(
            cell.forward_seq(&[], 0),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            cell.forward_seq(&[0.0; 5], 3),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn invalid_config_is_error() {
        let mut rng = LcgRng::new(1);
        let bad_dim = LtcConfig {
            in_dim: 0,
            ..LtcConfig::default()
        };
        assert!(matches!(
            LtcCell::new(bad_dim, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
        let bad_dt = LtcConfig {
            dt: 0.0,
            ..LtcConfig::default()
        };
        assert!(matches!(
            LtcCell::new(bad_dt, &mut rng),
            Err(SnnError::BadDt { .. })
        ));
        let bad_tau = LtcConfig {
            tau_base: -1.0,
            ..LtcConfig::default()
        };
        assert!(matches!(
            LtcCell::new(bad_tau, &mut rng),
            Err(SnnError::BadTau { .. })
        ));
        let bad_unfold = LtcConfig {
            n_unfold: 0,
            ..LtcConfig::default()
        };
        assert!(matches!(
            LtcCell::new(bad_unfold, &mut rng),
            Err(SnnError::BadTimesteps { .. })
        ));
    }
}
