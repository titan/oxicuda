//! # Plan2Explore — Planning to Explore via Self-Supervised World Models.
//!
//! Sekar, Rhinehart, Wang, Goyal, Fearing, Pathak (2020),
//! "Planning to Explore via Self-Supervised World Models", ICML 2020,
//! <https://arxiv.org/abs/2005.05960>.
//!
//! Plan2Explore drives *task-agnostic* exploration by training an **ensemble**
//! of one-step latent forward models, each predicting the next
//! state/feature `ŷ_k(s, a)` from the current `(state, action)` pair. The
//! **disagreement** (prediction variance) across the ensemble is used as an
//! intrinsic reward: transitions the models cannot yet agree on are *novel* and
//! worth exploring, whereas well-modelled transitions yield low disagreement.
//!
//! ```text
//! ȳ(s,a)        = (1/K) Σ_k ŷ_k(s,a)
//! disagree(s,a) = (1/F) Σ_d  (1/K) Σ_k ( ŷ_k(s,a)[d] − ȳ(s,a)[d] )²
//! r^i(s,a)      = reward_scale · disagree(s,a)                      (≥ 0)
//! ```
//!
//! Each ensemble member is a tiny one-hidden-layer MLP
//! `(state ⊕ action) → tanh → feature` trained by full-batch gradient descent
//! on observed transitions. Distinct random initialisation per member produces
//! the initial disagreement; training on shared targets collapses the
//! disagreement on *seen* data while it stays high off-distribution — exactly
//! the latent-disagreement exploration signal of Sekar et al. (2020).

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

// ─── RNG helpers ───────────────────────────────────────────────────────────────

/// Uniform sample in `[0, 1)`.
///
/// NB: [`LcgRng::next_f32`] only spans `[0, 0.5)` in this crate, so we rescale
/// the raw 31-bit integer ourselves (`next_u32 ∈ [0, 2³¹)`).
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
}

/// One standard-normal variate via the Box–Muller transform.
#[inline]
fn sample_standard_normal(rng: &mut LcgRng) -> f32 {
    let u1 = unit_uniform(rng).max(1e-7_f32);
    let u2 = unit_uniform(rng);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

// ─── ForwardModel (single ensemble member) ──────────────────────────────────────

/// A single one-hidden-layer `tanh` MLP forward model `x ↦ ŷ`.
#[derive(Debug, Clone)]
struct ForwardModel {
    /// Input dimensionality `state_dim + action_dim`.
    in_dim: usize,
    /// Hidden-layer width.
    hidden: usize,
    /// First-layer weights `[hidden × in_dim]`, row-major.
    w1: Vec<f32>,
    /// First-layer bias `[hidden]`.
    b1: Vec<f32>,
    /// Second-layer weights `[out_dim × hidden]`, row-major.
    w2: Vec<f32>,
    /// Second-layer bias `[out_dim]`.
    b2: Vec<f32>,
}

impl ForwardModel {
    /// He-style random initialisation drawn from `rng`.
    fn new(in_dim: usize, hidden: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let s1 = (1.0_f32 / in_dim as f32).sqrt();
        let s2 = (1.0_f32 / hidden as f32).sqrt();
        let w1 = (0..hidden * in_dim)
            .map(|_| sample_standard_normal(rng) * s1)
            .collect();
        let w2 = (0..out_dim * hidden)
            .map(|_| sample_standard_normal(rng) * s2)
            .collect();
        Self {
            in_dim,
            hidden,
            w1,
            b1: vec![0.0_f32; hidden],
            w2,
            b2: vec![0.0_f32; out_dim],
        }
    }

    /// Forward pass, returning the hidden activations `h` and the output `ŷ`.
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let h: Vec<f32> = self
            .w1
            .chunks_exact(self.in_dim)
            .zip(&self.b1)
            .map(|(row, &b)| (b + dot(row, x)).tanh())
            .collect();
        let y: Vec<f32> = self
            .w2
            .chunks_exact(self.hidden)
            .zip(&self.b2)
            .map(|(row, &b)| b + dot(row, &h))
            .collect();
        (h, y)
    }

    /// Forward pass returning only the output `ŷ`.
    fn output(&self, x: &[f32]) -> Vec<f32> {
        self.forward(x).1
    }

    /// One full-batch gradient-descent epoch over `(inputs, targets)`.
    ///
    /// Returns the mean `½‖ŷ − y‖²` loss across the batch (pre-update).
    fn train_epoch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], lr: f32) -> f32 {
        let mut gw1 = vec![0.0_f32; self.w1.len()];
        let mut gb1 = vec![0.0_f32; self.b1.len()];
        let mut gw2 = vec![0.0_f32; self.w2.len()];
        let mut gb2 = vec![0.0_f32; self.b2.len()];
        let mut total_loss = 0.0_f32;

        for (x, t) in inputs.iter().zip(targets) {
            let (h, y) = self.forward(x);
            // Output-error δ = ŷ − y.
            let delta: Vec<f32> = y.iter().zip(t).map(|(&yi, &ti)| yi - ti).collect();
            total_loss += 0.5_f32 * delta.iter().map(|&d| d * d).sum::<f32>();

            // ∂L/∂W2 = δ ⊗ h ; ∂L/∂b2 = δ.
            gw2.chunks_exact_mut(self.hidden)
                .zip(&delta)
                .for_each(|(grow, &d)| {
                    grow.iter_mut().zip(&h).for_each(|(g, &hj)| *g += d * hj);
                });
            gb2.iter_mut().zip(&delta).for_each(|(g, &d)| *g += d);

            // ∂L/∂h = W2ᵀ δ.
            let mut dh = vec![0.0_f32; self.hidden];
            self.w2
                .chunks_exact(self.hidden)
                .zip(&delta)
                .for_each(|(row, &d)| {
                    dh.iter_mut().zip(row).for_each(|(acc, &w)| *acc += w * d);
                });

            // ∂L/∂z1 = ∂L/∂h ⊙ (1 − h²).
            let dz1: Vec<f32> = dh
                .iter()
                .zip(&h)
                .map(|(&g, &hj)| g * (1.0_f32 - hj * hj))
                .collect();

            // ∂L/∂W1 = dz1 ⊗ x ; ∂L/∂b1 = dz1.
            gw1.chunks_exact_mut(self.in_dim)
                .zip(&dz1)
                .for_each(|(grow, &d)| {
                    grow.iter_mut().zip(x).for_each(|(g, &xi)| *g += d * xi);
                });
            gb1.iter_mut().zip(&dz1).for_each(|(g, &d)| *g += d);
        }

        let inv = 1.0_f32 / inputs.len() as f32;
        let step = lr * inv;
        self.w1
            .iter_mut()
            .zip(&gw1)
            .for_each(|(w, &g)| *w -= step * g);
        self.b1
            .iter_mut()
            .zip(&gb1)
            .for_each(|(w, &g)| *w -= step * g);
        self.w2
            .iter_mut()
            .zip(&gw2)
            .for_each(|(w, &g)| *w -= step * g);
        self.b2
            .iter_mut()
            .zip(&gb2)
            .for_each(|(w, &g)| *w -= step * g);

        total_loss * inv
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for a [`Plan2Explore`] ensemble.
#[derive(Debug, Clone, Copy)]
pub struct Plan2ExploreConfig {
    /// State dimensionality `S`.
    pub state_dim: usize,
    /// Action dimensionality `A`.
    pub action_dim: usize,
    /// Predicted next-state/feature dimensionality `F`.
    pub feature_dim: usize,
    /// Hidden-layer width of each ensemble member.
    pub hidden_dim: usize,
    /// Number of ensemble members `K` (must be ≥ 2 for a meaningful variance).
    pub ensemble_size: usize,
    /// Gradient-descent learning rate.
    pub learning_rate: f32,
    /// Multiplicative scale applied to the disagreement intrinsic reward.
    pub reward_scale: f32,
}

impl Default for Plan2ExploreConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 1,
            feature_dim: 4,
            hidden_dim: 32,
            ensemble_size: 5,
            learning_rate: 0.05,
            reward_scale: 1.0,
        }
    }
}

// ─── Plan2Explore ───────────────────────────────────────────────────────────────

/// Ensemble of one-step latent world models with disagreement-based intrinsic
/// reward (Sekar et al. 2020).
#[derive(Debug, Clone)]
pub struct Plan2Explore {
    /// Model configuration.
    config: Plan2ExploreConfig,
    /// The `K` ensemble members.
    members: Vec<ForwardModel>,
}

impl Plan2Explore {
    /// Build a new ensemble with distinct random initialisation per member.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if any dimension is zero or
    /// `ensemble_size < 2`.
    pub fn new(config: Plan2ExploreConfig, rng: &mut LcgRng) -> RlResult<Self> {
        for (name, v) in [
            ("state_dim", config.state_dim),
            ("action_dim", config.action_dim),
            ("feature_dim", config.feature_dim),
            ("hidden_dim", config.hidden_dim),
        ] {
            if v == 0 {
                return Err(RlError::InvalidHyperparameter {
                    name: name.into(),
                    msg: "must be > 0".into(),
                });
            }
        }
        if config.ensemble_size < 2 {
            return Err(RlError::InvalidHyperparameter {
                name: "ensemble_size".into(),
                msg: "must be >= 2 for a meaningful disagreement".into(),
            });
        }

        let in_dim = config.state_dim + config.action_dim;
        let members = (0..config.ensemble_size)
            .map(|_| ForwardModel::new(in_dim, config.hidden_dim, config.feature_dim, rng))
            .collect();
        Ok(Self { config, members })
    }

    /// Number of ensemble members `K`.
    #[must_use]
    #[inline]
    pub fn ensemble_size(&self) -> usize {
        self.members.len()
    }

    /// Predicted feature dimensionality `F`.
    #[must_use]
    #[inline]
    pub fn feature_dim(&self) -> usize {
        self.config.feature_dim
    }

    /// Validate that `state` and `action` have the configured dimensionalities.
    fn check(&self, state: &[f32], action: &[f32]) -> RlResult<()> {
        if state.len() != self.config.state_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.config.state_dim,
                got: state.len(),
            });
        }
        if action.len() != self.config.action_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.config.action_dim,
                got: action.len(),
            });
        }
        Ok(())
    }

    /// Concatenate `state ⊕ action` into a single input vector.
    fn make_input(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        let mut x = Vec::with_capacity(self.config.state_dim + self.config.action_dim);
        x.extend_from_slice(state);
        x.extend_from_slice(action);
        x
    }

    /// Per-member predictions `ŷ_k(s, a)` — a `K`-element vector of
    /// `feature_dim`-length predictions.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a `state`/`action` shape error.
    pub fn predict(&self, state: &[f32], action: &[f32]) -> RlResult<Vec<Vec<f32>>> {
        self.check(state, action)?;
        let x = self.make_input(state, action);
        Ok(self.members.iter().map(|m| m.output(&x)).collect())
    }

    /// Ensemble mean prediction `ȳ(s, a)` of length `feature_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a `state`/`action` shape error.
    pub fn ensemble_mean(&self, state: &[f32], action: &[f32]) -> RlResult<Vec<f32>> {
        let preds = self.predict(state, action)?;
        Ok(Self::mean_of(&preds, self.config.feature_dim))
    }

    /// Mean of per-member predictions.
    fn mean_of(preds: &[Vec<f32>], feature_dim: usize) -> Vec<f32> {
        let mut mean = vec![0.0_f32; feature_dim];
        for p in preds {
            mean.iter_mut().zip(p).for_each(|(m, &v)| *m += v);
        }
        let inv_k = 1.0_f32 / preds.len() as f32;
        mean.iter_mut().for_each(|m| *m *= inv_k);
        mean
    }

    /// Ensemble **disagreement**: the mean (over feature dims) of the
    /// per-dimension prediction variance across the `K` members. Always ≥ 0.
    ///
    /// This is the raw Plan2Explore latent-disagreement exploration signal.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a `state`/`action` shape error.
    pub fn disagreement(&self, state: &[f32], action: &[f32]) -> RlResult<f32> {
        let preds = self.predict(state, action)?;
        let mean = Self::mean_of(&preds, self.config.feature_dim);
        let mut var_sum = 0.0_f32;
        for p in &preds {
            for (&v, &m) in p.iter().zip(&mean) {
                let diff = v - m;
                var_sum += diff * diff;
            }
        }
        let inv_k = 1.0_f32 / preds.len() as f32;
        Ok(var_sum * inv_k / self.config.feature_dim as f32)
    }

    /// Intrinsic exploration reward `reward_scale · disagreement(s, a)` (≥ 0).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a `state`/`action` shape error.
    pub fn intrinsic_reward(&self, state: &[f32], action: &[f32]) -> RlResult<f32> {
        Ok(self.config.reward_scale * self.disagreement(state, action)?)
    }

    /// Batched intrinsic rewards for `[B × state_dim]` states and
    /// `[B × action_dim]` actions. Returns a length-`B` vector.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if the flat slices are not exact
    /// multiples of their dimensionality or imply a different batch size.
    pub fn intrinsic_reward_batch(&self, states: &[f32], actions: &[f32]) -> RlResult<Vec<f32>> {
        let b = self.batch_size(states.len(), self.config.state_dim)?;
        let ba = self.batch_size(actions.len(), self.config.action_dim)?;
        if b != ba {
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: ba,
            });
        }
        states
            .chunks_exact(self.config.state_dim)
            .zip(actions.chunks_exact(self.config.action_dim))
            .map(|(s, a)| self.intrinsic_reward(s, a))
            .collect()
    }

    /// Mean prediction error `(1/(K·B·F)) Σ ‖ŷ_k − y‖²` over the batch and the
    /// ensemble.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on inconsistent batch shapes.
    pub fn prediction_mse(
        &self,
        states: &[f32],
        actions: &[f32],
        next_features: &[f32],
    ) -> RlResult<f32> {
        let b = self.batch_size(states.len(), self.config.state_dim)?;
        if actions.len() != b * self.config.action_dim
            || next_features.len() != b * self.config.feature_dim
        {
            return Err(RlError::DimensionMismatch {
                expected: b * self.config.action_dim,
                got: actions.len(),
            });
        }
        let mut total = 0.0_f32;
        let it = states
            .chunks_exact(self.config.state_dim)
            .zip(actions.chunks_exact(self.config.action_dim))
            .zip(next_features.chunks_exact(self.config.feature_dim));
        for ((s, a), t) in it {
            let preds = self.predict(s, a)?;
            for p in &preds {
                total += p
                    .iter()
                    .zip(t)
                    .map(|(&yi, &ti)| (yi - ti).powi(2))
                    .sum::<f32>();
            }
        }
        let denom = (self.members.len() * b * self.config.feature_dim) as f32;
        Ok(total / denom)
    }

    /// Train every ensemble member on the observed transitions for `epochs`
    /// full-batch gradient-descent epochs.
    ///
    /// Inputs are flat slices: `states` `[B × state_dim]`, `actions`
    /// `[B × action_dim]`, `next_features` `[B × feature_dim]`. Returns the
    /// post-training [`Plan2Explore::prediction_mse`].
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on inconsistent batch shapes or an
    /// empty batch.
    pub fn train(
        &mut self,
        states: &[f32],
        actions: &[f32],
        next_features: &[f32],
        epochs: usize,
    ) -> RlResult<f32> {
        let b = self.batch_size(states.len(), self.config.state_dim)?;
        if b == 0 {
            return Err(RlError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if actions.len() != b * self.config.action_dim
            || next_features.len() != b * self.config.feature_dim
        {
            return Err(RlError::DimensionMismatch {
                expected: b * self.config.feature_dim,
                got: next_features.len(),
            });
        }

        let inputs: Vec<Vec<f32>> = states
            .chunks_exact(self.config.state_dim)
            .zip(actions.chunks_exact(self.config.action_dim))
            .map(|(s, a)| {
                let mut x = Vec::with_capacity(self.config.state_dim + self.config.action_dim);
                x.extend_from_slice(s);
                x.extend_from_slice(a);
                x
            })
            .collect();
        let targets: Vec<Vec<f32>> = next_features
            .chunks_exact(self.config.feature_dim)
            .map(<[f32]>::to_vec)
            .collect();

        let lr = self.config.learning_rate;
        for member in &mut self.members {
            for _ in 0..epochs {
                member.train_epoch(&inputs, &targets, lr);
            }
        }
        self.prediction_mse(states, actions, next_features)
    }

    /// Infer the batch size from a flat slice length and per-row dimension.
    fn batch_size(&self, len: usize, dim: usize) -> RlResult<usize> {
        let _ = self;
        if len % dim != 0 {
            return Err(RlError::DimensionMismatch {
                expected: dim,
                got: len,
            });
        }
        Ok(len / dim)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Plan2ExploreConfig {
        Plan2ExploreConfig {
            state_dim: 3,
            action_dim: 1,
            feature_dim: 2,
            hidden_dim: 16,
            ensemble_size: 5,
            learning_rate: 0.05,
            reward_scale: 1.0,
        }
    }

    // Four transitions concentrated near the origin (the "seen" region).
    fn seen_data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let states = vec![
            0.10, 0.00, -0.10, //
            0.00, 0.10, 0.00, //
            -0.10, 0.00, 0.10, //
            0.05, -0.05, 0.00,
        ];
        let actions = vec![0.00, 0.10, -0.10, 0.05];
        let targets = vec![
            0.20, -0.10, //
            0.10, 0.00, //
            -0.10, 0.20, //
            0.00, 0.10,
        ];
        (states, actions, targets)
    }

    // States/actions far from the seen region (the "novel" region).
    fn unseen_states() -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![
            (vec![3.0, -2.5, 2.0], vec![1.5]),
            (vec![-3.0, 2.0, -2.5], vec![-1.5]),
            (vec![2.5, 2.5, 2.5], vec![2.0]),
            (vec![-2.5, -2.5, 1.5], vec![-2.0]),
        ]
    }

    #[test]
    fn new_ok_and_sizes() {
        let mut rng = LcgRng::new(1);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("valid config");
        assert_eq!(p2e.ensemble_size(), 5);
        assert_eq!(p2e.feature_dim(), 2);
    }

    #[test]
    fn predict_shapes() {
        let mut rng = LcgRng::new(2);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let preds = p2e.predict(&[0.1, 0.2, 0.3], &[0.4]).expect("ok");
        assert_eq!(preds.len(), 5, "one prediction per member");
        for p in &preds {
            assert_eq!(p.len(), 2, "each prediction has feature_dim entries");
        }
    }

    #[test]
    fn ensemble_mean_shape() {
        let mut rng = LcgRng::new(3);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let mean = p2e.ensemble_mean(&[0.1, 0.2, 0.3], &[0.4]).expect("ok");
        assert_eq!(mean.len(), 2);
    }

    #[test]
    fn disagreement_nonneg_and_finite() {
        let mut rng = LcgRng::new(4);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let d = p2e.disagreement(&[0.5, -0.5, 0.0], &[1.0]).expect("ok");
        assert!(d >= 0.0, "disagreement must be non-negative, got {d}");
        assert!(d.is_finite(), "disagreement must be finite");
    }

    #[test]
    fn intrinsic_reward_nonneg_and_scaled() {
        let mut cfg = tiny_config();
        cfg.reward_scale = 2.0;
        let mut rng = LcgRng::new(5);
        let p2e = Plan2Explore::new(cfg, &mut rng).expect("ok");
        let state = [0.3, 0.1, -0.2];
        let action = [0.7];
        let d = p2e.disagreement(&state, &action).expect("ok");
        let r = p2e.intrinsic_reward(&state, &action).expect("ok");
        assert!(r >= 0.0, "intrinsic reward must be non-negative");
        assert!(
            (r - 2.0 * d).abs() < 1e-5,
            "reward should equal scale·disagreement"
        );
    }

    #[test]
    fn intrinsic_reward_batch_shape() {
        let mut rng = LcgRng::new(6);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let states = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let actions = vec![0.0, 1.0];
        let r = p2e.intrinsic_reward_batch(&states, &actions).expect("ok");
        assert_eq!(r.len(), 2);
        for &ri in &r {
            assert!(ri >= 0.0 && ri.is_finite());
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let a = Plan2Explore::new(tiny_config(), &mut rng_a).expect("ok");
        let b = Plan2Explore::new(tiny_config(), &mut rng_b).expect("ok");
        let da = a.disagreement(&[0.2, 0.3, 0.4], &[0.5]).expect("ok");
        let db = b.disagreement(&[0.2, 0.3, 0.4], &[0.5]).expect("ok");
        assert!((da - db).abs() < 1e-9, "same seed must be deterministic");
    }

    #[test]
    fn training_reduces_prediction_error() {
        let mut rng = LcgRng::new(7);
        let mut p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let (s, a, t) = seen_data();
        let mse_before = p2e.prediction_mse(&s, &a, &t).expect("ok");
        let mse_after = p2e.train(&s, &a, &t, 500).expect("ok");
        assert!(
            mse_after < mse_before,
            "training should reduce error: before={mse_before}, after={mse_after}"
        );
        assert!(mse_after.is_finite());
    }

    #[test]
    fn training_reduces_disagreement_on_seen() {
        let mut rng = LcgRng::new(8);
        let mut p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let (s, a, t) = seen_data();
        let seen_state = [0.10, 0.00, -0.10];
        let seen_action = [0.00];
        let before = p2e.disagreement(&seen_state, &seen_action).expect("ok");
        p2e.train(&s, &a, &t, 500).expect("ok");
        let after = p2e.disagreement(&seen_state, &seen_action).expect("ok");
        assert!(
            after < before,
            "disagreement on a trained point should drop: before={before}, after={after}"
        );
    }

    #[test]
    fn novel_disagreement_exceeds_seen() {
        let mut rng = LcgRng::new(9);
        let mut p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        let (s, a, t) = seen_data();
        p2e.train(&s, &a, &t, 600).expect("ok");

        // Mean disagreement on the seen set.
        let mut seen_sum = 0.0_f32;
        let n_seen = a.len();
        for (sc, ac) in s.chunks_exact(3).zip(a.chunks_exact(1)) {
            seen_sum += p2e.disagreement(sc, ac).expect("ok");
        }
        let seen_mean = seen_sum / n_seen as f32;

        // Mean disagreement on far-away novel points.
        let novel = unseen_states();
        let mut novel_sum = 0.0_f32;
        for (sc, ac) in &novel {
            novel_sum += p2e.disagreement(sc, ac).expect("ok");
        }
        let novel_mean = novel_sum / novel.len() as f32;

        assert!(
            novel_mean > seen_mean,
            "novel disagreement {novel_mean} should exceed seen {seen_mean}"
        );
    }

    #[test]
    fn err_zero_state_dim() {
        let mut cfg = tiny_config();
        cfg.state_dim = 0;
        let mut rng = LcgRng::new(10);
        assert!(matches!(
            Plan2Explore::new(cfg, &mut rng),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_singleton_ensemble() {
        let mut cfg = tiny_config();
        cfg.ensemble_size = 1;
        let mut rng = LcgRng::new(11);
        assert!(matches!(
            Plan2Explore::new(cfg, &mut rng),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_predict_dim_mismatch() {
        let mut rng = LcgRng::new(12);
        let p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        assert!(matches!(
            p2e.predict(&[0.1, 0.2], &[0.3]),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_train_empty_batch() {
        let mut rng = LcgRng::new(13);
        let mut p2e = Plan2Explore::new(tiny_config(), &mut rng).expect("ok");
        assert!(p2e.train(&[], &[], &[], 10).is_err());
    }
}
