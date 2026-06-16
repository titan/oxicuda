//! Meta-SGD learner with per-parameter learned learning rates (Li et al. 2017).
//!
//! Unlike [`crate::maml::meta_sgd::MetaSgd`] which wraps a generic closure-based
//! interface, `MetaSgdLearner` owns explicit parameter and log-learning-rate vectors
//! and implements a concrete linear classifier for episodic cross-entropy.
//!
//! Key idea: rather than using a single scalar learning rate, Meta-SGD maintains
//! a per-parameter vector `α` (stored as `log_α`) that is jointly meta-optimized
//! alongside the model parameters `θ`.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Configuration for [`MetaSgdLearner`].
#[derive(Debug, Clone)]
pub struct MetaSgdLearnerConfig {
    /// Total number of model parameters (must be > 0).
    pub n_params: usize,
    /// Number of inner-loop gradient steps during adaptation (>= 1).
    pub n_inner_steps: usize,
    /// Outer-loop (meta) learning rate (> 0).
    pub meta_lr: f32,
    /// Initial value for every per-parameter learning rate α_i (> 0).
    pub lr_init: f32,
    /// Number of output classes for the linear classifier (>= 2).
    pub n_classes: usize,
    /// Feature dimensionality per example.
    pub feat_dim: usize,
}

/// Meta-SGD learner: jointly learns model parameters `θ` and per-parameter
/// learning rates `α` (stored as `log(α)` for unconstrained optimization).
///
/// The model implements a simple linear classifier:
/// logits = X · W^T  where W is reshaped from `params` as `[n_classes × feat_dim]`.
#[derive(Debug, Clone)]
pub struct MetaSgdLearner {
    /// Model parameters θ, length = `n_params`.
    pub params: Vec<f32>,
    /// Log per-parameter learning rates, length = `n_params`.
    pub log_alphas: Vec<f32>,
    /// Learner configuration.
    config: MetaSgdLearnerConfig,
}

// Central finite differences for gradient estimation.
fn fd_grads<F>(eval: F, params: &[f32], eps: f32) -> MetaResult<Vec<f32>>
where
    F: Fn(&[f32]) -> MetaResult<f32>,
{
    let n = params.len();
    let mut grad = vec![0.0_f32; n];
    let mut p = params.to_vec();

    for i in 0..n {
        let orig = p[i];
        p[i] = orig + eps;
        let f_plus = eval(&p)?;
        p[i] = orig - eps;
        let f_minus = eval(&p)?;
        p[i] = orig;
        if !f_plus.is_finite() || !f_minus.is_finite() {
            return Err(MetaError::NanEncountered {
                context: format!("fd_grads: non-finite f at param index {i}"),
            });
        }
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }

    Ok(grad)
}

impl MetaSgdLearner {
    /// Create a new `MetaSgdLearner`.
    ///
    /// Parameters are initialized with small Gaussian-like noise via [`LcgRng`].
    /// `log_alphas` are initialized to `ln(lr_init)` for every parameter.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::InvalidEpisodeConfig`] if `n_params == 0`, `n_inner_steps < 1`,
    /// `n_classes < 2`, `feat_dim == 0`, or `lr_init <= 0`.
    /// Returns [`MetaError::InvalidLr`] if `meta_lr <= 0`.
    pub fn new(config: MetaSgdLearnerConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.n_params == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_params must be > 0".into(),
            });
        }
        if config.n_inner_steps < 1 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_inner_steps must be >= 1".into(),
            });
        }
        if config.meta_lr <= 0.0 || !config.meta_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: config.meta_lr });
        }
        if config.n_classes < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_classes,
            });
        }
        if config.feat_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: 0 });
        }
        if config.lr_init <= 0.0 || !config.lr_init.is_finite() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "lr_init must be > 0".into(),
            });
        }

        // Small random initialization via LcgRng: uniform in [-0.05, 0.05]
        let params: Vec<f32> = (0..config.n_params)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.05)
            .collect();
        let log_alpha_val = config.lr_init.ln();
        let log_alphas = vec![log_alpha_val; config.n_params];

        Ok(Self {
            params,
            log_alphas,
            config,
        })
    }

    /// Return per-parameter learning rates as `exp(log_alpha)`.
    ///
    /// All values are guaranteed positive.
    pub fn alphas(&self) -> Vec<f32> {
        self.log_alphas.iter().map(|&la| la.exp()).collect()
    }

    /// Evaluate cross-entropy loss of a linear classifier with given `params`.
    ///
    /// The classifier interprets the first `n_classes * feat_dim` entries of `params`
    /// as the weight matrix W (row-major: `[n_classes × feat_dim]`).
    ///
    /// `x` has shape `[n_examples * feat_dim]`.
    /// `y` has shape `[n_examples]`, values in `[0, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] on size mismatches.
    /// Returns [`MetaError::NanEncountered`] for non-finite values.
    pub fn eval_loss(&self, params: &[f32], x: &[f32], y: &[u32]) -> MetaResult<f32> {
        let n_c = self.config.n_classes;
        let d = self.config.feat_dim;
        let w_size = n_c * d;

        if params.len() < w_size {
            return Err(MetaError::DimensionMismatch {
                expected: w_size,
                got: params.len(),
            });
        }
        if x.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        if !x.len().is_multiple_of(d) {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: x.len() % d,
            });
        }
        let n_examples = x.len() / d;
        if y.len() != n_examples {
            return Err(MetaError::DimensionMismatch {
                expected: n_examples,
                got: y.len(),
            });
        }

        let w = &params[..w_size];
        let mut total_loss = 0.0_f32;

        for i in 0..n_examples {
            let xi = &x[i * d..(i + 1) * d];
            // Compute logits = W · xi
            let mut logits = vec![0.0_f32; n_c];
            for c in 0..n_c {
                for j in 0..d {
                    logits[c] += w[c * d + j] * xi[j];
                }
            }

            // Numerically-stable softmax cross-entropy
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&z| (z - max_l).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            if !sum_exp.is_finite() || sum_exp == 0.0 {
                return Err(MetaError::NanEncountered {
                    context: "eval_loss: sum_exp is zero or non-finite".into(),
                });
            }

            let label = y[i] as usize;
            if label >= n_c {
                return Err(MetaError::InvalidEpisodeConfig {
                    msg: format!("label {label} >= n_classes {n_c}"),
                });
            }
            let log_prob = (exps[label] / sum_exp).ln();
            if !log_prob.is_finite() {
                return Err(MetaError::NanEncountered {
                    context: "eval_loss: log_prob is non-finite".into(),
                });
            }
            total_loss -= log_prob;
        }

        let loss = total_loss / n_examples as f32;
        if !loss.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "eval_loss: final loss non-finite".into(),
            });
        }
        Ok(loss)
    }

    /// Inner-loop adaptation: `n_inner_steps` of per-parameter SGD.
    ///
    /// Returns adapted parameters θ' (same length as `self.params`), using finite
    /// differences to estimate gradients at each step.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError`] on shape mismatch or non-finite values.
    pub fn adapt(&self, support_x: &[f32], support_y: &[u32]) -> MetaResult<Vec<f32>> {
        let alphas = self.alphas();
        let mut theta = self.params.clone();
        let eps = 1e-4_f32;

        for _ in 0..self.config.n_inner_steps {
            let theta_snap = theta.clone();
            let eval = |p: &[f32]| self.eval_loss(p, support_x, support_y);
            let grads = fd_grads(eval, &theta_snap, eps)?;
            for j in 0..theta.len() {
                theta[j] -= alphas[j] * grads[j];
            }
        }

        Ok(theta)
    }

    /// Outer-loop (meta) update of both `params` (θ) and `log_alphas`.
    ///
    /// 1. Inner-adapt on `support` to get θ'.
    /// 2. Estimate meta-gradient for θ via FD at θ' on `query`.
    /// 3. Estimate meta-gradient for log_α via chain rule: `∂L_q/∂log_α_j ≈ -α_j · g_support_j · g_query_j`.
    /// 4. Update θ and log_α by `-meta_lr * gradient`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError`] on any numerical failure.
    pub fn meta_update(
        &mut self,
        support_x: &[f32],
        support_y: &[u32],
        query_x: &[f32],
        query_y: &[u32],
    ) -> MetaResult<()> {
        let eps = 1e-4_f32;

        // Inner adapt
        let theta_prime = self.adapt(support_x, support_y)?;

        // Meta-gradient for θ: FD at θ' on query loss
        let eval_query = |p: &[f32]| self.eval_loss(p, query_x, query_y);
        let g_query = fd_grads(eval_query, &theta_prime, eps)?;

        // Meta-gradient for log_α: chain-rule approximation
        // ∂L_q(θ')/∂log_α_j = ∂L_q/∂θ'_j * ∂θ'_j/∂log_α_j
        // ∂θ'_j/∂log_α_j ≈ -α_j * g_support_j  (from one-step approximation)
        let eval_support = |p: &[f32]| self.eval_loss(p, support_x, support_y);
        let g_support = fd_grads(eval_support, &self.params, eps)?;
        let alphas = self.alphas();

        let meta_lr = self.config.meta_lr;

        // Update θ
        for (p, &gq) in self.params.iter_mut().zip(g_query.iter()) {
            *p -= meta_lr * gq;
        }

        // Update log_α via chain rule
        for (j, la) in self.log_alphas.iter_mut().enumerate() {
            let d_log_alpha = -alphas[j] * g_support[j] * g_query[j];
            *la -= meta_lr * d_log_alpha;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_learner(n_classes: usize, feat_dim: usize) -> MetaSgdLearner {
        let n_params = n_classes * feat_dim;
        let cfg = MetaSgdLearnerConfig {
            n_params,
            n_inner_steps: 2,
            meta_lr: 0.01,
            lr_init: 0.05,
            n_classes,
            feat_dim,
        };
        MetaSgdLearner::new(cfg, &mut LcgRng::new(42)).expect("value should be present")
    }

    fn make_data(n_classes: usize, feat_dim: usize) -> (Vec<f32>, Vec<u32>) {
        let n = n_classes;
        let mut x = vec![0.0_f32; n * feat_dim];
        let mut y = vec![0u32; n];
        for c in 0..n_classes {
            x[c * feat_dim + (c % feat_dim)] = 1.0;
            y[c] = c as u32;
        }
        (x, y)
    }

    #[test]
    fn new_creates_correct_shapes() {
        let learner = make_learner(3, 4);
        assert_eq!(learner.params.len(), 12);
        assert_eq!(learner.log_alphas.len(), 12);
    }

    #[test]
    fn alphas_positive() {
        let learner = make_learner(3, 4);
        for &a in &learner.alphas() {
            assert!(a > 0.0, "all alphas must be positive");
        }
    }

    #[test]
    fn adapt_changes_params() {
        let learner = make_learner(3, 4);
        let (x, y) = make_data(3, 4);
        let adapted = learner.adapt(&x, &y).expect("adapt should succeed");
        let changed = adapted
            .iter()
            .zip(learner.params.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-9);
        assert!(changed, "adapt must change at least one parameter");
    }

    #[test]
    fn adapt_returns_correct_length() {
        let learner = make_learner(3, 4);
        let (x, y) = make_data(3, 4);
        let adapted = learner.adapt(&x, &y).expect("adapt should succeed");
        assert_eq!(adapted.len(), learner.params.len());
    }

    #[test]
    fn meta_update_changes_params() {
        let mut learner = make_learner(3, 4);
        let params_before = learner.params.clone();
        let (sx, sy) = make_data(3, 4);
        let (qx, qy) = make_data(3, 4);
        learner
            .meta_update(&sx, &sy, &qx, &qy)
            .expect("meta_update should succeed");
        let changed = learner
            .params
            .iter()
            .zip(params_before.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-9);
        assert!(changed, "meta_update must change params");
    }

    #[test]
    fn meta_update_changes_alphas() {
        let mut learner = make_learner(3, 4);
        let log_alphas_before = learner.log_alphas.clone();
        let (sx, sy) = make_data(3, 4);
        let (qx, qy) = make_data(3, 4);
        learner
            .meta_update(&sx, &sy, &qx, &qy)
            .expect("meta_update should succeed");
        let changed = learner
            .log_alphas
            .iter()
            .zip(log_alphas_before.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-9);
        assert!(changed, "meta_update must change log_alphas");
    }

    #[test]
    fn eval_loss_finite() {
        let learner = make_learner(3, 4);
        let (x, y) = make_data(3, 4);
        let loss = learner
            .eval_loss(&learner.params.clone(), &x, &y)
            .expect("value should be present");
        assert!(loss.is_finite(), "eval_loss must be finite, got {loss}");
    }

    #[test]
    fn n_params_0_error() {
        let cfg = MetaSgdLearnerConfig {
            n_params: 0,
            n_inner_steps: 1,
            meta_lr: 0.01,
            lr_init: 0.01,
            n_classes: 2,
            feat_dim: 4,
        };
        let result = MetaSgdLearner::new(cfg, &mut LcgRng::new(1));
        assert!(result.is_err(), "n_params=0 must return Err");
    }

    #[test]
    fn n_classes_1_error() {
        let cfg = MetaSgdLearnerConfig {
            n_params: 4,
            n_inner_steps: 1,
            meta_lr: 0.01,
            lr_init: 0.01,
            n_classes: 1,
            feat_dim: 4,
        };
        let result = MetaSgdLearner::new(cfg, &mut LcgRng::new(1));
        assert!(
            matches!(result, Err(MetaError::InvalidNWay { .. })),
            "n_classes=1 must return InvalidNWay"
        );
    }
}
