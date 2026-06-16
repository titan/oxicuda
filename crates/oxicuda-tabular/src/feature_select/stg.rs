//! STG: Feature selection using Stochastic Gates.
//!
//! Reference: Yamada, Lindenbaum, Negahban & Kluger (2020), "Feature Selection
//! using Stochastic Gates", ICML 2020.
//!
//! # Idea
//!
//! Each input feature `d` is multiplied by a continuous, Gaussian-relaxed
//! Bernoulli gate
//!
//! ```text
//! z_d = clamp(μ_d + σ · ε_d, 0, 1),   ε_d ~ N(0, 1)
//! ```
//!
//! during training, and `z_d = clamp(μ_d, 0, 1)` at evaluation time (`ε = 0`).
//! The gate location parameters `μ_d` are learnable; `σ` is a fixed relaxation
//! scale.  Sparsity is encouraged by an `L0`-surrogate regulariser equal to the
//! sum of per-feature *open* probabilities
//!
//! ```text
//! R(μ) = Σ_d Φ(μ_d / σ),
//! ```
//!
//! where `Φ` is the standard normal CDF.  Minimising `R` drives unhelpful gates
//! toward zero, performing differentiable feature selection.  The gated features
//! feed a small predictor MLP; the learned gate values double as feature
//! importances.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::nn::{Mlp, log_softmax};
use crate::preprocess::quantile_feat::std_normal_cdf;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an [`StgModel`].
#[derive(Debug, Clone)]
pub struct StgConfig {
    /// Number of input features (one gate per feature).
    pub n_features: usize,
    /// Hidden width of the predictor MLP.
    pub hidden_dim: usize,
    /// Number of hidden layers in the predictor MLP.
    pub n_layers: usize,
    /// Output dimension (1 for regression, `n_classes` for classification).
    pub output_dim: usize,
    /// Gate relaxation scale `σ` (must be positive; the paper uses `0.5`).
    pub sigma: f32,
    /// Weight `λ` of the `L0`-surrogate regulariser in the total loss.
    pub lambda: f32,
}

// ─── StgModel ─────────────────────────────────────────────────────────────────

/// A stochastic-gate feature selector with an attached predictor MLP.
#[derive(Debug, Clone)]
pub struct StgModel {
    /// Per-feature gate location parameters `μ`, length `n_features`.
    mu: Vec<f32>,
    /// Gate relaxation scale `σ`.
    sigma: f32,
    /// Regulariser weight `λ`.
    lambda: f32,
    /// Predictor MLP `n_features → … → output_dim`.
    predictor: Mlp,
    /// Number of input features.
    n_features: usize,
    /// Predictor output dimension.
    output_dim: usize,
}

impl StgModel {
    /// Construct a new STG model with gates initialised at `μ_d = 0.5` and a
    /// randomly-initialised predictor.
    ///
    /// # Errors
    /// - [`TabularError::InvalidFeatureCount`] if `n_features == 0` or
    ///   `output_dim == 0`.
    /// - [`TabularError::InvalidParameter`] if `hidden_dim == 0` or
    ///   `sigma <= 0`.
    pub fn new(config: StgConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if config.output_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if config.hidden_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "hidden_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if config.sigma <= 0.0 || !config.sigma.is_finite() {
            return Err(TabularError::InvalidParameter {
                name: "sigma".into(),
                msg: "must be a positive, finite value".into(),
            });
        }

        let mut dims = Vec::with_capacity(config.n_layers + 2);
        dims.push(config.n_features);
        for _ in 0..config.n_layers {
            dims.push(config.hidden_dim);
        }
        dims.push(config.output_dim);
        let predictor = Mlp::new(&dims, rng)?;

        Ok(Self {
            mu: vec![0.5_f32; config.n_features],
            sigma: config.sigma,
            lambda: config.lambda,
            predictor,
            n_features: config.n_features,
            output_dim: config.output_dim,
        })
    }

    /// Number of input features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Predictor output dimension.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Overwrite the gate location parameters `μ`.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `mu.len() != n_features`.
    pub fn set_gate_location(&mut self, mu: &[f32]) -> TabularResult<()> {
        if mu.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: mu.len(),
            });
        }
        self.mu.copy_from_slice(mu);
        Ok(())
    }

    /// Sample a stochastic gate vector `z_d = clamp(μ_d + σ·ε_d, 0, 1)`.
    ///
    /// Each entry lies in `[0, 1]`.
    pub fn sample_gates(&self, rng: &mut LcgRng) -> Vec<f32> {
        let mut eps = vec![0.0_f32; self.n_features];
        rng.fill_normal(&mut eps);
        self.mu
            .iter()
            .zip(eps.iter())
            .map(|(&m, &e)| (m + self.sigma * e).clamp(0.0, 1.0))
            .collect()
    }

    /// Deterministic evaluation gates `z_d = clamp(μ_d, 0, 1)` (`ε = 0`).
    #[must_use]
    pub fn eval_gates(&self) -> Vec<f32> {
        self.mu.iter().map(|&m| m.clamp(0.0, 1.0)).collect()
    }

    /// Per-feature open probability `Φ(μ_d / σ)`, each in `[0, 1]`.
    #[must_use]
    pub fn gate_open_prob(&self) -> Vec<f32> {
        self.mu
            .iter()
            .map(|&m| std_normal_cdf(m / self.sigma))
            .collect()
    }

    /// Learned feature importances — the deterministic evaluation gates.
    #[must_use]
    pub fn importances(&self) -> Vec<f32> {
        self.eval_gates()
    }

    /// `L0`-surrogate regulariser `R(μ) = Σ_d Φ(μ_d / σ)` (always `≥ 0`).
    #[must_use]
    pub fn regularization(&self) -> f32 {
        self.gate_open_prob().iter().sum()
    }

    /// Indices of features whose evaluation gate exceeds `threshold`.
    #[must_use]
    pub fn selected_features(&self, threshold: f32) -> Vec<usize> {
        self.eval_gates()
            .iter()
            .enumerate()
            .filter_map(|(d, &g)| if g > threshold { Some(d) } else { None })
            .collect()
    }

    /// Apply a gate vector to an input row, returning the masked features.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn apply_gates(&self, x: &[f32], gates: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.n_features || gates.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        Ok(x.iter().zip(gates.iter()).map(|(&xi, &g)| xi * g).collect())
    }

    /// Deterministic forward pass: mask with evaluation gates, then predict.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn forward_eval(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        let gates = self.eval_gates();
        let masked = self.apply_gates(x, &gates)?;
        Ok(self.predictor.forward(&masked))
    }

    /// Stochastic forward pass: mask with sampled gates, then predict.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn forward_train(&self, x: &[f32], rng: &mut LcgRng) -> TabularResult<Vec<f32>> {
        let gates = self.sample_gates(rng);
        let masked = self.apply_gates(x, &gates)?;
        Ok(self.predictor.forward(&masked))
    }

    /// Total training loss for a *classification* row: cross-entropy of the
    /// deterministic prediction against `label`, plus `λ · R(μ)`.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    /// - [`TabularError::LabelOutOfRange`] if `label >= output_dim`.
    pub fn classification_loss(&self, x: &[f32], label: usize) -> TabularResult<f32> {
        if label >= self.output_dim {
            return Err(TabularError::LabelOutOfRange {
                label,
                n_classes: self.output_dim,
            });
        }
        let logits = self.forward_eval(x)?;
        let lsm = log_softmax(&logits);
        let ce = -lsm[label];
        Ok(ce + self.lambda * self.regularization())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> StgConfig {
        StgConfig {
            n_features: 6,
            hidden_dim: 12,
            n_layers: 2,
            output_dim: 3,
            sigma: 0.5,
            lambda: 0.1,
        }
    }

    fn make_model() -> StgModel {
        let mut rng = LcgRng::new(42);
        StgModel::new(small_cfg(), &mut rng).expect("value should be present")
    }

    // ── 1. eval gates in [0, 1] ──────────────────────────────────────────────
    #[test]
    fn eval_gates_in_unit_interval() {
        let m = make_model();
        assert!(m.eval_gates().iter().all(|&g| (0.0..=1.0).contains(&g)));
    }

    // ── 2. sampled gates in [0, 1] ───────────────────────────────────────────
    #[test]
    fn sampled_gates_in_unit_interval() {
        let m = make_model();
        let mut rng = LcgRng::new(7);
        for _ in 0..50 {
            let g = m.sample_gates(&mut rng);
            assert!(g.iter().all(|&v| (0.0..=1.0).contains(&v)), "{g:?}");
        }
    }

    // ── 3. regulariser ≥ 0 ───────────────────────────────────────────────────
    #[test]
    fn regularization_non_negative() {
        let m = make_model();
        assert!(m.regularization() >= 0.0);
    }

    // ── 4. regulariser equals sum of open probabilities ──────────────────────
    #[test]
    fn regularization_is_sum_of_open_probs() {
        let m = make_model();
        let sum: f32 = m.gate_open_prob().iter().sum();
        assert!((m.regularization() - sum).abs() < 1e-6);
    }

    // ── 5. high μ → gate ≈ 1 ─────────────────────────────────────────────────
    #[test]
    fn high_mu_opens_gate() {
        let mut m = make_model();
        m.set_gate_location(&[5.0_f32; 6])
            .expect("set_gate_location should succeed");
        for &g in &m.eval_gates() {
            assert!((g - 1.0).abs() < 1e-6, "gate should be ~1, got {g}");
        }
        // open probability ≈ 1 as well
        assert!(m.gate_open_prob().iter().all(|&p| p > 0.999));
    }

    // ── 6. low μ → gate ≈ 0 ──────────────────────────────────────────────────
    #[test]
    fn low_mu_closes_gate() {
        let mut m = make_model();
        m.set_gate_location(&[-5.0_f32; 6])
            .expect("set_gate_location should succeed");
        for &g in &m.eval_gates() {
            assert!(g.abs() < 1e-6, "gate should be ~0, got {g}");
        }
        assert!(m.gate_open_prob().iter().all(|&p| p < 1e-3));
    }

    // ── 7. eval is deterministic ─────────────────────────────────────────────
    #[test]
    fn eval_is_deterministic() {
        let m = make_model();
        let g1 = m.eval_gates();
        let g2 = m.eval_gates();
        assert_eq!(g1, g2);
        let x = vec![0.3_f32; 6];
        assert_eq!(
            m.forward_eval(&x).expect("forward_eval should succeed"),
            m.forward_eval(&x).expect("forward_eval should succeed")
        );
    }

    // ── 8. sampled gates vary with the RNG ───────────────────────────────────
    #[test]
    fn sampled_gates_are_stochastic() {
        let m = make_model();
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(2);
        assert_ne!(m.sample_gates(&mut r1), m.sample_gates(&mut r2));
    }

    // ── 9. importances equal eval gates and lie in [0, 1] ────────────────────
    #[test]
    fn importances_match_eval_gates() {
        let m = make_model();
        assert_eq!(m.importances(), m.eval_gates());
        assert!(m.importances().iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    // ── 10. forward shape ────────────────────────────────────────────────────
    #[test]
    fn forward_shape() {
        let m = make_model();
        let x = vec![0.5_f32; 6];
        assert_eq!(
            m.forward_eval(&x)
                .expect("forward_eval should succeed")
                .len(),
            3
        );
        let mut rng = LcgRng::new(3);
        assert_eq!(
            m.forward_train(&x, &mut rng)
                .expect("forward_train should succeed")
                .len(),
            3
        );
    }

    // ── 11. apply_gates masks features ───────────────────────────────────────
    #[test]
    fn apply_gates_masks() {
        let m = make_model();
        let x = vec![2.0_f32; 6];
        let gates = vec![1.0_f32, 0.0, 0.5, 1.0, 0.0, 0.25];
        let out = m
            .apply_gates(&x, &gates)
            .expect("apply_gates should succeed");
        assert_eq!(out, vec![2.0, 0.0, 1.0, 2.0, 0.0, 0.5]);
    }

    // ── 12. selected_features respects threshold ─────────────────────────────
    #[test]
    fn selected_features_threshold() {
        let mut m = make_model();
        m.set_gate_location(&[0.9, 0.1, 0.8, 0.05, 1.0, 0.2])
            .expect("value should be present");
        let sel = m.selected_features(0.5);
        assert_eq!(sel, vec![0, 2, 4]);
    }

    // ── 13. classification loss finite ───────────────────────────────────────
    #[test]
    fn classification_loss_finite() {
        let m = make_model();
        let x = vec![0.4_f32; 6];
        let loss = m
            .classification_loss(&x, 1)
            .expect("classification_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss = {loss}");
    }

    // ── 14. classification loss rejects bad label ────────────────────────────
    #[test]
    fn classification_loss_bad_label() {
        let m = make_model();
        let x = vec![0.4_f32; 6];
        assert!(matches!(
            m.classification_loss(&x, 9),
            Err(TabularError::LabelOutOfRange { .. })
        ));
    }

    // ── 15. constructor validation ───────────────────────────────────────────
    #[test]
    fn new_rejects_bad_config() {
        let mut rng = LcgRng::new(1);
        let mut cfg = small_cfg();
        cfg.n_features = 0;
        assert!(StgModel::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.output_dim = 0;
        assert!(StgModel::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.sigma = 0.0;
        assert!(StgModel::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.hidden_dim = 0;
        assert!(StgModel::new(cfg, &mut rng).is_err());
    }

    // ── 16. wrong input length errors ────────────────────────────────────────
    #[test]
    fn forward_wrong_len_errs() {
        let m = make_model();
        assert!(m.forward_eval(&[0.0; 5]).is_err());
    }
}
