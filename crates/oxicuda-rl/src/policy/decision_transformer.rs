//! # Decision Transformer (DT)
//!
//! Chen et al. (2021), "Decision Transformer: Reinforcement Learning via
//! Sequence Modeling".
//!
//! The Decision Transformer re-frames RL as a conditional sequence modelling
//! problem.  Given a context of past (return-to-go, state, action) triples, it
//! predicts the next action that is consistent with the desired return.
//!
//! ## Architecture
//!
//! ```text
//! (R̂₀, s₀, a₀, R̂₁, s₁, a₁, …, R̂_{K-1}, s_{K-1}, a_{K-1})
//!                                           ↓
//!                               Token embeddings + positional
//!                                           ↓
//!                               N × [Linear + Tanh + Residual]
//!                                           ↓
//!                        last state token → action head → â
//! ```
//!
//! The self-attention step is replaced with a simplified
//! linear + tanh + residual block so that the full implementation stays
//! dependency-free (no external attention kernels required).

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

// ─── Box-Muller helpers ──────────────────────────────────────────────────────

/// Produce a pair of independent standard-normal samples via the Box-Muller
/// transform.
///
/// Both inputs are expected to be uniform in `(0, 1)`.  The caller is
/// responsible for clamping to avoid degenerate logarithms.
#[inline]
fn box_muller(rng: &mut LcgRng) -> (f32, f32) {
    let u1 = (rng.next_f32() + 1e-10_f32).min(1.0 - 1e-10_f32);
    let u2 = rng.next_f32();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Fill a `Vec<f32>` of length `n` with i.i.d. `N(0, scale²)` samples.
fn init_weights(n: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0_f32; n];
    let mut i = 0;
    while i + 1 < n {
        let (a, b) = box_muller(rng);
        v[i] = a * scale;
        v[i + 1] = b * scale;
        i += 2;
    }
    if i < n {
        let (a, _) = box_muller(rng);
        v[i] = a * scale;
    }
    v
}

// ─── DtConfig ────────────────────────────────────────────────────────────────

/// Hyperparameters for the [`DecisionTransformer`].
#[derive(Debug, Clone)]
pub struct DtConfig {
    /// Observation (state) dimension.
    pub state_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// `K`: number of past timesteps used as context.
    pub context_len: usize,
    /// Model (embedding) dimension `d_model`.
    pub d_model: usize,
    /// Number of attention heads.  Must divide `d_model`.
    pub n_heads: usize,
    /// Number of transformer layers.  Must be ≥ 1.
    pub n_layers: usize,
    /// Maximum episode length for positional (timestep) embeddings.
    pub max_ep_len: usize,
}

// ─── DecisionTransformer ─────────────────────────────────────────────────────

/// A Decision Transformer policy model.
///
/// Weights are randomly initialised in [`DecisionTransformer::new`] and can be
/// updated externally (e.g. from a GPU training step).  The forward pass is
/// carried out entirely in `f32` on the CPU.
#[derive(Debug, Clone)]
pub struct DecisionTransformer {
    /// State embedding matrix `[d_model × state_dim]` (row-major).
    state_emb: Vec<f32>,
    /// Action embedding matrix `[d_model × action_dim]` (row-major).
    action_emb: Vec<f32>,
    /// Return-to-go embedding matrix `[d_model × 1]` (row-major).
    return_emb: Vec<f32>,
    /// Timestep (positional) embedding table `[max_ep_len × d_model]`
    /// (row-major).
    timestep_emb: Vec<f32>,
    /// Per-layer linear weight matrices `[n_layers][d_model × d_model]`.
    layer_weights: Vec<Vec<f32>>,
    /// Per-layer bias vectors `[n_layers][d_model]`.
    layer_biases: Vec<Vec<f32>>,
    /// Action output head weight matrix `[action_dim × d_model]` (row-major).
    action_head: Vec<f32>,
    /// Action output head bias `[action_dim]`.
    action_head_b: Vec<f32>,
    /// Configuration snapshot.
    config: DtConfig,
}

impl DecisionTransformer {
    /// Create a new [`DecisionTransformer`] with randomly initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] when:
    /// - `n_heads` is 0.
    /// - `d_model` is not divisible by `n_heads`.
    /// - Any of `state_dim`, `action_dim`, `context_len`, `max_ep_len`,
    ///   `n_layers` is 0.
    pub fn new(config: DtConfig, rng: &mut LcgRng) -> RlResult<Self> {
        // ── Validation ───────────────────────────────────────────────────────
        if config.n_heads == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_heads".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.d_model % config.n_heads != 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "d_model".into(),
                msg: format!(
                    "must be divisible by n_heads (d_model={}, n_heads={})",
                    config.d_model, config.n_heads
                ),
            });
        }
        if config.state_dim == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "state_dim".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.action_dim == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "action_dim".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.context_len == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "context_len".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.max_ep_len == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "max_ep_len".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.n_layers == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_layers".into(),
                msg: "must be >= 1".into(),
            });
        }

        let scale = 1.0 / (config.d_model as f32).sqrt();

        // ── Embedding matrices ───────────────────────────────────────────────
        let state_emb = init_weights(config.d_model * config.state_dim, scale, rng);
        let action_emb = init_weights(config.d_model * config.action_dim, scale, rng);
        let return_emb = init_weights(config.d_model, scale, rng); // d_model × 1
        let timestep_emb = init_weights(config.max_ep_len * config.d_model, scale, rng);

        // ── Transformer layer weights ────────────────────────────────────────
        let mut layer_weights = Vec::with_capacity(config.n_layers);
        let mut layer_biases = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layer_weights.push(init_weights(config.d_model * config.d_model, scale, rng));
            layer_biases.push(vec![0.0_f32; config.d_model]);
        }

        // ── Action head ──────────────────────────────────────────────────────
        let action_head = init_weights(config.action_dim * config.d_model, scale, rng);
        let action_head_b = vec![0.0_f32; config.action_dim];

        Ok(Self {
            state_emb,
            action_emb,
            return_emb,
            timestep_emb,
            layer_weights,
            layer_biases,
            action_head,
            action_head_b,
            config,
        })
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Multiply a row-major weight matrix `W [out_dim × in_dim]` by a column
    /// vector `x [in_dim]` and return the result `y [out_dim]`.
    fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        debug_assert_eq!(w.len(), out_dim * in_dim);
        debug_assert_eq!(x.len(), in_dim);
        let mut y = vec![0.0_f32; out_dim];
        for i in 0..out_dim {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            let mut acc = 0.0_f32;
            for j in 0..in_dim {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        y
    }

    /// Predict the next action from a context window of past experience.
    ///
    /// # Arguments
    ///
    /// * `returns_to_go` — `[K]` desired return for each context position.
    /// * `states`        — `[K × state_dim]` flattened state history (row-major).
    /// * `actions`       — `[K × action_dim]` flattened action history (row-major).
    /// * `timesteps`     — `[K]` absolute timestep indices; each must be
    ///   `< max_ep_len`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] when any input length is wrong.
    /// * [`RlError::InvalidHyperparameter`] when a timestep index is
    ///   `>= max_ep_len`.
    pub fn predict_action(
        &self,
        returns_to_go: &[f32],
        states: &[f32],
        actions: &[f32],
        timesteps: &[usize],
    ) -> RlResult<Vec<f32>> {
        let k = self.config.context_len;
        let d = self.config.d_model;

        // ── Input validation ─────────────────────────────────────────────────
        if returns_to_go.len() != k {
            return Err(RlError::DimensionMismatch {
                expected: k,
                got: returns_to_go.len(),
            });
        }
        if states.len() != k * self.config.state_dim {
            return Err(RlError::DimensionMismatch {
                expected: k * self.config.state_dim,
                got: states.len(),
            });
        }
        if actions.len() != k * self.config.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: k * self.config.action_dim,
                got: actions.len(),
            });
        }
        if timesteps.len() != k {
            return Err(RlError::DimensionMismatch {
                expected: k,
                got: timesteps.len(),
            });
        }
        for &ts in timesteps {
            if ts >= self.config.max_ep_len {
                return Err(RlError::InvalidHyperparameter {
                    name: "timestep".into(),
                    msg: format!("timestep {ts} >= max_ep_len {}", self.config.max_ep_len),
                });
            }
        }

        // ── Build token sequence ─────────────────────────────────────────────
        // Sequence layout: [r_tok_0, s_tok_0, a_tok_0, r_tok_1, …]
        // Total: 3*K tokens, each of size d_model.
        let seq_len = 3 * k;
        let mut seq = vec![0.0_f32; seq_len * d];

        for t in 0..k {
            // Return-to-go embedding: return_emb is [d × 1], scalar input
            let r_val = returns_to_go[t];
            let r_emb: Vec<f32> = self.return_emb.iter().map(|&w| w * r_val).collect();

            // State embedding
            let s_t = &states[t * self.config.state_dim..(t + 1) * self.config.state_dim];
            let s_emb = Self::matvec(&self.state_emb, s_t, d, self.config.state_dim);

            // Action embedding
            let a_t = &actions[t * self.config.action_dim..(t + 1) * self.config.action_dim];
            let a_emb = Self::matvec(&self.action_emb, a_t, d, self.config.action_dim);

            // Positional embedding for this timestep
            let ts = timesteps[t];
            let pos_emb = &self.timestep_emb[ts * d..(ts + 1) * d];

            // Add positional embeddings and write into sequence buffer
            let r_idx = (3 * t) * d;
            let s_idx = (3 * t + 1) * d;
            let a_idx = (3 * t + 2) * d;
            for f in 0..d {
                seq[r_idx + f] = r_emb[f] + pos_emb[f];
                seq[s_idx + f] = s_emb[f] + pos_emb[f];
                seq[a_idx + f] = a_emb[f] + pos_emb[f];
            }
        }

        // ── Transformer layers ───────────────────────────────────────────────
        for l in 0..self.config.n_layers {
            let w = &self.layer_weights[l];
            let b = &self.layer_biases[l];
            // Process each token independently (simplified: no cross-token attention)
            for tok in 0..seq_len {
                let start = tok * d;
                let token_in: Vec<f32> = seq[start..start + d].to_vec();

                // Linear transform + bias
                let lin = Self::matvec(w, &token_in, d, d);
                let mut new_tok = vec![0.0_f32; d];
                for f in 0..d {
                    // Tanh nonlinearity
                    new_tok[f] = (lin[f] + b[f]).tanh();
                }
                // Residual connection: new = new_tok + token_in
                for f in 0..d {
                    seq[start + f] = new_tok[f] + token_in[f];
                }
            }
        }

        // ── Extract last state token ─────────────────────────────────────────
        // State tokens are at positions 3*t+1; last state token: 3*(K-1)+1
        let last_state_idx = (3 * (k - 1) + 1) * d;
        let last_state_tok = &seq[last_state_idx..last_state_idx + d];

        // ── Action head ──────────────────────────────────────────────────────
        let action_dim = self.config.action_dim;
        let mut action = Self::matvec(&self.action_head, last_state_tok, action_dim, d);
        for (a, &b) in action.iter_mut().zip(self.action_head_b.iter()) {
            *a += b;
        }

        Ok(action)
    }

    /// Return a reference to the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &DtConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DtConfig {
        DtConfig {
            state_dim: 4,
            action_dim: 2,
            context_len: 3,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            max_ep_len: 100,
        }
    }

    fn make_dt(seed: u64) -> DecisionTransformer {
        let mut rng = LcgRng::new(seed);
        DecisionTransformer::new(make_config(), &mut rng)
            .expect("valid DtConfig should construct DecisionTransformer")
    }

    fn make_inputs(cfg: &DtConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
        let k = cfg.context_len;
        let returns_to_go = vec![1.0_f32; k];
        let states = vec![0.1_f32; k * cfg.state_dim];
        let actions = vec![0.0_f32; k * cfg.action_dim];
        let timesteps: Vec<usize> = (0..k).collect();
        (returns_to_go, states, actions, timesteps)
    }

    #[test]
    fn output_shape() {
        let dt = make_dt(1);
        let cfg = make_config();
        let (rtg, s, a, ts) = make_inputs(&cfg);
        let out = dt
            .predict_action(&rtg, &s, &a, &ts)
            .expect("valid inputs should produce action");
        assert_eq!(
            out.len(),
            cfg.action_dim,
            "output length must equal action_dim"
        );
    }

    #[test]
    fn output_finite() {
        let dt = make_dt(2);
        let cfg = make_config();
        let (rtg, s, a, ts) = make_inputs(&cfg);
        let out = dt
            .predict_action(&rtg, &s, &a, &ts)
            .expect("valid inputs should produce finite action");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn context_len_1_works() {
        let mut rng = LcgRng::new(3);
        let cfg = DtConfig {
            state_dim: 2,
            action_dim: 1,
            context_len: 1,
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            max_ep_len: 50,
        };
        let dt = DecisionTransformer::new(cfg.clone(), &mut rng)
            .expect("context_len=1 config should construct");
        let out = dt
            .predict_action(&[0.5], &[0.1, 0.2], &[0.0], &[0])
            .expect("context_len=1 should predict");
        assert_eq!(out.len(), cfg.action_dim);
        assert!(out[0].is_finite());
    }

    #[test]
    fn different_states_different_actions() {
        let dt = make_dt(42);
        let cfg = make_config();
        let k = cfg.context_len;
        let rtg = vec![1.0_f32; k];
        let ts: Vec<usize> = (0..k).collect();
        let actions = vec![0.0_f32; k * cfg.action_dim];

        let states_a = vec![0.0_f32; k * cfg.state_dim];
        let states_b = vec![1.0_f32; k * cfg.state_dim];

        let out_a = dt
            .predict_action(&rtg, &states_a, &actions, &ts)
            .expect("states_a should predict");
        let out_b = dt
            .predict_action(&rtg, &states_b, &actions, &ts)
            .expect("states_b should predict");

        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "different states should produce different actions");
    }

    #[test]
    fn timestep_out_of_range_error() {
        let dt = make_dt(5);
        let cfg = make_config();
        let k = cfg.context_len;
        let rtg = vec![0.0_f32; k];
        let s = vec![0.0_f32; k * cfg.state_dim];
        let a = vec![0.0_f32; k * cfg.action_dim];
        // timesteps[0] = max_ep_len → out of range
        let mut ts: Vec<usize> = (0..k).collect();
        ts[0] = cfg.max_ep_len;
        let result = dt.predict_action(&rtg, &s, &a, &ts);
        assert!(result.is_err(), "out-of-range timestep should error");
    }

    #[test]
    fn returns_len_mismatch_error() {
        let dt = make_dt(6);
        let cfg = make_config();
        let k = cfg.context_len;
        let rtg = vec![0.0_f32; k + 1]; // wrong length
        let s = vec![0.0_f32; k * cfg.state_dim];
        let a = vec![0.0_f32; k * cfg.action_dim];
        let ts: Vec<usize> = (0..k).collect();
        assert!(dt.predict_action(&rtg, &s, &a, &ts).is_err());
    }

    #[test]
    fn states_len_mismatch_error() {
        let dt = make_dt(7);
        let cfg = make_config();
        let k = cfg.context_len;
        let rtg = vec![0.0_f32; k];
        let s = vec![0.0_f32; k * cfg.state_dim + 1]; // wrong length
        let a = vec![0.0_f32; k * cfg.action_dim];
        let ts: Vec<usize> = (0..k).collect();
        assert!(dt.predict_action(&rtg, &s, &a, &ts).is_err());
    }

    #[test]
    fn n_heads_zero_error() {
        let mut rng = LcgRng::new(8);
        let cfg = DtConfig {
            state_dim: 4,
            action_dim: 2,
            context_len: 3,
            d_model: 8,
            n_heads: 0,
            n_layers: 1,
            max_ep_len: 100,
        };
        assert!(
            DecisionTransformer::new(cfg, &mut rng).is_err(),
            "n_heads=0 should return error"
        );
    }

    #[test]
    fn d_model_not_divisible_by_n_heads_error() {
        let mut rng = LcgRng::new(9);
        let cfg = DtConfig {
            state_dim: 4,
            action_dim: 2,
            context_len: 3,
            d_model: 5,
            n_heads: 3,
            n_layers: 1,
            max_ep_len: 100,
        };
        assert!(
            DecisionTransformer::new(cfg, &mut rng).is_err(),
            "d_model=5 not divisible by n_heads=3 should error"
        );
    }

    #[test]
    fn action_finite_with_extreme_returns() {
        let dt = make_dt(10);
        let cfg = make_config();
        let k = cfg.context_len;
        // Alternating very large and very small returns-to-go
        let rtg: Vec<f32> = (0..k)
            .map(|i| if i % 2 == 0 { 1e10_f32 } else { -1e10_f32 })
            .collect();
        let s = vec![0.0_f32; k * cfg.state_dim];
        let a = vec![0.0_f32; k * cfg.action_dim];
        let ts: Vec<usize> = (0..k).collect();
        let out = dt
            .predict_action(&rtg, &s, &a, &ts)
            .expect("extreme returns should not panic");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "output[{i}]={v} should be finite even with extreme returns"
            );
        }
    }
}
