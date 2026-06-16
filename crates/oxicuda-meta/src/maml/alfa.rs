//! ALFA — Adaptive Learning of hyperparameters for Fast Adaptation
//! (Baik et al., NeurIPS 2020: "Meta-Learning with Adaptive Hyperparameters").
//!
//! ALFA generalises the MAML inner loop so that the per-step update uses
//! **learned, per-layer** hyper-parameters instead of a single global learning
//! rate.  For each parameter group (layer) `l` the inner update is
//!
//! ```text
//!   θ_l ← θ_l − α_l ⊙ ∇_θ L(θ) + β_l ⊙ θ_l
//! ```
//!
//! where `α_l` is a learned per-layer learning rate and `β_l` is a learned
//! per-layer weight-modulation (a signed weight-decay) term.  Both are
//! meta-parameters, optimised in the outer loop alongside the parameter
//! initialisation `θ₀`.
//!
//! Two regimes are supported:
//!
//! * **MAML-LR (minimum):** a plain learnable per-layer `α_l` (and `β_l`),
//!   used when [`AlfaConfig::use_generator`] is `false`;
//! * **ALFA (stretch):** the per-layer hyper-parameters are *conditioned on the
//!   current gradient/weight statistics* through a tiny shared generator
//!   network, recovering ALFA's grad-conditioned adaptive behaviour when
//!   [`AlfaConfig::use_generator`] is `true`.
//!
//! Setting every `α_l` to the same scalar with `β_l = 0` and the generator
//! disabled recovers vanilla MAML exactly — both compute
//! `θ ← θ − lr · fd_gradient(L, θ)`.
//!
//! The inner-loop loss reuses the existing MAML linear-classifier closure
//! (`crate::maml::maml::task_loss_at_params`) and finite-difference gradient,
//! so ALFA slots directly into the crate's MAML machinery.

use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::handle::LcgRng;
use crate::maml::maml::task_loss_at_params;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters controlling ALFA meta-training.
#[derive(Debug, Clone)]
pub struct AlfaConfig {
    /// Number of inner-loop adaptation steps.
    pub inner_steps: usize,
    /// Outer-loop (meta) learning rate.
    pub meta_lr: f32,
    /// Finite-difference step used for both inner and meta gradients.
    pub fd_eps: f32,
    /// Initial per-layer learning rate `α_l`.
    pub init_lr: f32,
    /// Initial per-layer weight-modulation `β_l` (`0` ⇒ vanilla MAML update).
    pub init_wd: f32,
    /// Upper clamp for the per-layer learning rate (kept in `[0, clip_lr]`).
    pub clip_lr: f32,
    /// Enable the ALFA grad-conditioned generator (`false` ⇒ MAML-LR).
    pub use_generator: bool,
}

impl Default for AlfaConfig {
    fn default() -> Self {
        Self {
            inner_steps: 3,
            meta_lr: 0.01,
            fd_eps: 1e-4,
            init_lr: 0.05,
            init_wd: 0.0,
            clip_lr: 1.0,
            use_generator: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grad-conditioned generator
// ─────────────────────────────────────────────────────────────────────────────

/// A tiny shared generator mapping per-layer statistics
/// `[mean|∇L_l|, mean|θ_l|, 1]` to a pair of modulations `(Δα_l, Δβ_l)`.
///
/// The modulations are squashed with `tanh` and scaled by [`Self::scale`], so a
/// zero-initialised generator is the identity (no modulation) and ALFA reduces
/// to the plain per-layer scheme until the generator is meta-trained.
#[derive(Debug, Clone)]
struct AlfaGenerator {
    /// Two output rows (Δα, Δβ) × three inputs (grad stat, weight stat, bias).
    weights: Vec<f32>,
    scale: f32,
}

impl AlfaGenerator {
    fn zeros() -> Self {
        Self {
            weights: vec![0.0_f32; 6],
            scale: 0.1,
        }
    }

    /// `(Δα, Δβ)` from a layer's `[grad_stat, weight_stat]` features.
    fn modulation(&self, grad_stat: f32, weight_stat: f32, weights: &[f32]) -> (f32, f32) {
        let f = [grad_stat, weight_stat, 1.0_f32];
        let d_alpha: f32 = weights[0] * f[0] + weights[1] * f[1] + weights[2] * f[2];
        let d_beta: f32 = weights[3] * f[0] + weights[4] * f[1] + weights[5] * f[2];
        (d_alpha.tanh() * self.scale, d_beta.tanh() * self.scale)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ALFA learner
// ─────────────────────────────────────────────────────────────────────────────

/// An ALFA meta-learner over a flat parameter vector partitioned into layers.
pub struct Alfa {
    /// Meta-learned parameter initialisation `θ₀`.
    pub base_params: Vec<f32>,
    /// Learned per-layer learning rates `α_l`.
    pub per_layer_lr: Vec<f32>,
    /// Learned per-layer weight-modulation `β_l`.
    pub per_layer_wd: Vec<f32>,
    /// Sizes of each parameter group; `Σ layer_sizes == base_params.len()`.
    layer_sizes: Vec<usize>,
    generator: AlfaGenerator,
    config: AlfaConfig,
}

impl Alfa {
    /// Construct an ALFA learner over a parameter vector partitioned by
    /// `layer_sizes` (one `α_l`/`β_l` pair per group).
    ///
    /// # Errors
    /// `InvalidEpisodeConfig` on an empty/zero-sized partition or invalid
    /// config, `DimensionMismatch` if the partition does not cover
    /// `base_params`, and `InvalidLr` for a non-positive `meta_lr`.
    pub fn new(
        base_params: Vec<f32>,
        layer_sizes: Vec<usize>,
        config: AlfaConfig,
    ) -> MetaResult<Self> {
        if layer_sizes.is_empty() || layer_sizes.contains(&0) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "layer_sizes must be non-empty and all > 0".into(),
            });
        }
        let total: usize = layer_sizes.iter().sum();
        if total != base_params.len() {
            return Err(MetaError::DimensionMismatch {
                expected: total,
                got: base_params.len(),
            });
        }
        Self::validate_config(&config)?;

        let n_groups = layer_sizes.len();
        Ok(Self {
            base_params,
            per_layer_lr: vec![config.init_lr; n_groups],
            per_layer_wd: vec![config.init_wd; n_groups],
            layer_sizes,
            generator: AlfaGenerator::zeros(),
            config,
        })
    }

    /// Convenience constructor for a single linear classification head:
    /// parameters are `[W (n_classes × feat_dim), b (n_classes)]`, grouped into
    /// the weight group and the bias group.  Weights are Xavier-initialised,
    /// biases zero.
    ///
    /// # Errors
    /// `InvalidNWay` if `n_classes < 2`, `InvalidFeatDim` if `feat_dim == 0`,
    /// or any error from [`Self::new`].
    pub fn for_linear_head(
        n_classes: usize,
        feat_dim: usize,
        config: AlfaConfig,
        rng: &mut LcgRng,
    ) -> MetaResult<Self> {
        if n_classes < 2 {
            return Err(MetaError::InvalidNWay { n_way: n_classes });
        }
        if feat_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: feat_dim });
        }
        let n_w = n_classes * feat_dim;
        let n_b = n_classes;
        let limit = (6.0_f32 / (feat_dim + n_classes) as f32).sqrt();
        let mut base = vec![0.0_f32; n_w + n_b];
        for v in base[..n_w].iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Self::new(base, vec![n_w, n_b], config)
    }

    fn validate_config(config: &AlfaConfig) -> MetaResult<()> {
        if config.inner_steps == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "inner_steps must be > 0".into(),
            });
        }
        if config.fd_eps <= 0.0 || !config.fd_eps.is_finite() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "fd_eps must be > 0".into(),
            });
        }
        if config.meta_lr <= 0.0 || !config.meta_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: config.meta_lr });
        }
        if config.clip_lr <= 0.0 || !config.clip_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: config.clip_lr });
        }
        Ok(())
    }

    /// Number of parameter groups (layers).
    pub fn n_groups(&self) -> usize {
        self.layer_sizes.len()
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &AlfaConfig {
        &self.config
    }

    /// Effective per-layer `(α_l, β_l)` for the current step, optionally
    /// modulated by the grad-conditioned generator.
    fn effective_hyperparams(
        &self,
        theta: &[f32],
        grad: &[f32],
        per_layer_lr: &[f32],
        per_layer_wd: &[f32],
        gen_weights: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = self.layer_sizes.len();
        let mut eff_lr = per_layer_lr.to_vec();
        let mut eff_wd = per_layer_wd.to_vec();
        if !self.config.use_generator {
            for a in eff_lr.iter_mut() {
                *a = a.clamp(0.0, self.config.clip_lr);
            }
            return (eff_lr, eff_wd);
        }

        let mut offset = 0;
        for g in 0..n {
            let size = self.layer_sizes[g];
            let grad_g = &grad[offset..offset + size];
            let theta_g = &theta[offset..offset + size];
            let inv = 1.0_f32 / size as f32;
            let grad_stat = grad_g.iter().map(|v| v.abs()).sum::<f32>() * inv;
            let weight_stat = theta_g.iter().map(|v| v.abs()).sum::<f32>() * inv;
            let (d_alpha, d_beta) = self
                .generator
                .modulation(grad_stat, weight_stat, gen_weights);
            eff_lr[g] = (per_layer_lr[g] + d_alpha).clamp(0.0, self.config.clip_lr);
            eff_wd[g] = per_layer_wd[g] + d_beta;
            offset += size;
        }
        (eff_lr, eff_wd)
    }

    /// Core inner loop over an arbitrary scalar loss closure.
    fn adapt_core<F>(
        &self,
        base: &[f32],
        per_layer_lr: &[f32],
        per_layer_wd: &[f32],
        gen_weights: &[f32],
        loss_fn: &F,
    ) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let mut theta = base.to_vec();
        for _ in 0..self.config.inner_steps {
            let grad = fd_gradient(&theta, loss_fn, self.config.fd_eps);
            let (eff_lr, eff_wd) =
                self.effective_hyperparams(&theta, &grad, per_layer_lr, per_layer_wd, gen_weights);
            let mut offset = 0;
            for (g, &size) in self.layer_sizes.iter().enumerate() {
                let a = eff_lr[g];
                let b = eff_wd[g];
                let theta_g = &mut theta[offset..offset + size];
                let grad_g = &grad[offset..offset + size];
                for (t, &gd) in theta_g.iter_mut().zip(grad_g.iter()) {
                    *t = *t - a * gd + b * *t;
                }
                offset += size;
            }
        }
        theta
    }

    /// Adapt the base parameters on a support set (linear classifier head),
    /// returning the task-adapted parameters.
    ///
    /// # Errors
    /// `DimensionMismatch` if `base_params.len() != n_classes·feat_dim +
    /// n_classes`.
    pub fn inner_adapt(
        &self,
        support_x: &[f32],
        support_y: &[u32],
        n_classes: usize,
        feat_dim: usize,
    ) -> MetaResult<Vec<f32>> {
        self.check_linear_dims(n_classes, feat_dim)?;
        let support_loss =
            |p: &[f32]| task_loss_at_params(p, support_x, support_y, n_classes, feat_dim);
        Ok(self.adapt_core(
            &self.base_params,
            &self.per_layer_lr,
            &self.per_layer_wd,
            &self.generator.weights,
            &support_loss,
        ))
    }

    /// One ALFA meta-step on a (support, query) task split.
    ///
    /// Adapts on the support set, then takes a finite-difference gradient of the
    /// post-adaptation query loss with respect to *all* meta-parameters
    /// (`θ₀`, `α`, `β`, and the generator when enabled), and applies the update.
    /// Returns the query loss measured *before* the update.
    ///
    /// # Errors
    /// `DimensionMismatch` on a parameter-shape mismatch.
    pub fn meta_step(
        &mut self,
        support_x: &[f32],
        support_y: &[u32],
        query_x: &[f32],
        query_y: &[u32],
        n_classes: usize,
        feat_dim: usize,
    ) -> MetaResult<f32> {
        self.check_linear_dims(n_classes, feat_dim)?;

        let support_loss =
            |p: &[f32]| task_loss_at_params(p, support_x, support_y, n_classes, feat_dim);
        let query_loss = |p: &[f32]| task_loss_at_params(p, query_x, query_y, n_classes, feat_dim);

        let n_base = self.base_params.len();
        let n_lr = self.per_layer_lr.len();
        let n_wd = self.per_layer_wd.len();
        let use_gen = self.config.use_generator;

        // Pack all meta-parameters into a single vector for finite differencing.
        let mut meta = self.base_params.clone();
        meta.extend_from_slice(&self.per_layer_lr);
        meta.extend_from_slice(&self.per_layer_wd);
        if use_gen {
            meta.extend_from_slice(&self.generator.weights);
        }

        let empty: [f32; 0] = [];
        let eval = |m: &[f32]| {
            let base = &m[..n_base];
            let lr = &m[n_base..n_base + n_lr];
            let wd = &m[n_base + n_lr..n_base + n_lr + n_wd];
            let gen_w: &[f32] = if use_gen {
                &m[n_base + n_lr + n_wd..]
            } else {
                &empty
            };
            let adapted = self.adapt_core(base, lr, wd, gen_w, &support_loss);
            query_loss(&adapted)
        };

        let loss_before = eval(&meta);
        let grad = fd_gradient(&meta, &eval, self.config.fd_eps);
        let mut updated: Vec<f32> = meta
            .iter()
            .zip(grad.iter())
            .map(|(&p, &g)| p - self.config.meta_lr * g)
            .collect();
        // Keep the per-layer learning rates non-negative and bounded.
        for a in updated[n_base..n_base + n_lr].iter_mut() {
            *a = a.clamp(0.0, self.config.clip_lr);
        }

        self.base_params.copy_from_slice(&updated[..n_base]);
        self.per_layer_lr
            .copy_from_slice(&updated[n_base..n_base + n_lr]);
        self.per_layer_wd
            .copy_from_slice(&updated[n_base + n_lr..n_base + n_lr + n_wd]);
        if use_gen {
            self.generator
                .weights
                .copy_from_slice(&updated[n_base + n_lr + n_wd..]);
        }

        Ok(loss_before)
    }

    fn check_linear_dims(&self, n_classes: usize, feat_dim: usize) -> MetaResult<()> {
        let expected = n_classes * feat_dim + n_classes;
        if self.base_params.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: self.base_params.len(),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maml::maml::{MamlConfig, maml_adapt};

    const N_CLASSES: usize = 2;
    const FEAT_DIM: usize = 4;

    fn task() -> (Vec<f32>, Vec<u32>) {
        // Two separable classes in 4-D.
        let x = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.9, 0.1, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.9, 0.1, //
        ];
        let y = vec![0_u32, 0, 1, 1];
        (x, y)
    }

    fn support_loss(params: &[f32], x: &[f32], y: &[u32]) -> f32 {
        task_loss_at_params(params, x, y, N_CLASSES, FEAT_DIM)
    }

    #[test]
    fn new_rejects_mismatched_partition() {
        let cfg = AlfaConfig::default();
        let r = Alfa::new(vec![0.0; 5], vec![2, 2], cfg);
        assert!(matches!(r, Err(MetaError::DimensionMismatch { .. })));
    }

    #[test]
    fn new_rejects_empty_partition() {
        let cfg = AlfaConfig::default();
        let r = Alfa::new(vec![], vec![], cfg);
        assert!(matches!(r, Err(MetaError::InvalidEpisodeConfig { .. })));
    }

    #[test]
    fn for_linear_head_shapes() {
        let mut rng = LcgRng::new(1);
        let alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, AlfaConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(alfa.base_params.len(), N_CLASSES * FEAT_DIM + N_CLASSES);
        assert_eq!(alfa.n_groups(), 2);
        assert_eq!(alfa.per_layer_lr.len(), 2);
        assert_eq!(alfa.per_layer_wd.len(), 2);
    }

    #[test]
    fn inner_adapt_reduces_support_loss() {
        let mut rng = LcgRng::new(7);
        let alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, AlfaConfig::default(), &mut rng)
            .expect("value should be present");
        let (x, y) = task();
        let before = support_loss(&alfa.base_params, &x, &y);
        let adapted = alfa
            .inner_adapt(&x, &y, N_CLASSES, FEAT_DIM)
            .expect("inner_adapt should succeed");
        let after = support_loss(&adapted, &x, &y);
        assert!(
            after < before,
            "adaptation must reduce support loss: {after} !< {before}"
        );
        assert_eq!(adapted.len(), alfa.base_params.len());
    }

    #[test]
    fn per_layer_lr_changes_adapted_params() {
        let mut rng = LcgRng::new(7);
        let mut alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, AlfaConfig::default(), &mut rng)
            .expect("value should be present");
        let (x, y) = task();
        let baseline = alfa
            .inner_adapt(&x, &y, N_CLASSES, FEAT_DIM)
            .expect("inner_adapt should succeed");
        // Bump the weight-group learning rate; the adapted params must differ.
        alfa.per_layer_lr[0] *= 3.0;
        let bumped = alfa
            .inner_adapt(&x, &y, N_CLASSES, FEAT_DIM)
            .expect("inner_adapt should succeed");
        assert_ne!(baseline, bumped);
    }

    #[test]
    fn reduces_to_vanilla_maml_with_scalar_lr() {
        // α_l = const scalar, β_l = 0, generator off  ⇒  identical to maml_adapt.
        let lr = 0.05_f32;
        let steps = 3;
        let mut rng = LcgRng::new(123);
        let cfg = AlfaConfig {
            inner_steps: steps,
            meta_lr: 0.01,
            fd_eps: 1e-4,
            init_lr: lr,
            init_wd: 0.0,
            clip_lr: 1.0,
            use_generator: false,
        };
        let alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, cfg, &mut rng)
            .expect("for_linear_head should succeed");
        let (x, y) = task();

        let alfa_adapted = alfa
            .inner_adapt(&x, &y, N_CLASSES, FEAT_DIM)
            .expect("inner_adapt should succeed");
        let maml_cfg = MamlConfig {
            inner_lr: lr,
            n_inner_steps: steps,
        };
        let maml_adapted = maml_adapt(&alfa.base_params, &x, &y, N_CLASSES, FEAT_DIM, &maml_cfg)
            .expect("maml_adapt should succeed");

        for (a, m) in alfa_adapted.iter().zip(maml_adapted.iter()) {
            assert!(
                (a - m).abs() < 1e-6,
                "ALFA(scalar α, β=0) must match MAML: {a} vs {m}"
            );
        }
    }

    #[test]
    fn meta_step_finite_and_non_increasing() {
        let mut rng = LcgRng::new(2024);
        let mut alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, AlfaConfig::default(), &mut rng)
            .expect("value should be present");
        let (x, y) = task();
        let first = alfa
            .meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM)
            .expect("meta_step should succeed");
        assert!(first.is_finite());
        let mut last = first;
        for _ in 0..8 {
            last = alfa
                .meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM)
                .expect("meta_step should succeed");
            assert!(last.is_finite());
        }
        assert!(
            last <= first + 1e-4,
            "meta-loss should hold or decrease: {last} vs {first}"
        );
    }

    #[test]
    fn meta_step_with_generator_is_finite() {
        let cfg = AlfaConfig {
            use_generator: true,
            ..AlfaConfig::default()
        };
        let mut rng = LcgRng::new(11);
        let mut alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, cfg, &mut rng)
            .expect("for_linear_head should succeed");
        let (x, y) = task();
        let before = support_loss(&alfa.base_params, &x, &y);
        let adapted = alfa
            .inner_adapt(&x, &y, N_CLASSES, FEAT_DIM)
            .expect("inner_adapt should succeed");
        let after = support_loss(&adapted, &x, &y);
        assert!(after < before, "generator path must still adapt");
        let loss = alfa
            .meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM)
            .expect("meta_step should succeed");
        assert!(loss.is_finite());
    }

    #[test]
    fn meta_step_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(1);
        let mut alfa = Alfa::for_linear_head(N_CLASSES, FEAT_DIM, AlfaConfig::default(), &mut rng)
            .expect("value should be present");
        let (x, y) = task();
        // Wrong feat_dim ⇒ base_params no longer matches the linear-head shape.
        let r = alfa.meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM + 1);
        assert!(matches!(r, Err(MetaError::DimensionMismatch { .. })));
    }

    #[test]
    fn deterministic_under_seed() {
        let (x, y) = task();
        let mut a = Alfa::for_linear_head(
            N_CLASSES,
            FEAT_DIM,
            AlfaConfig::default(),
            &mut LcgRng::new(5),
        )
        .expect("value should be present");
        let mut b = Alfa::for_linear_head(
            N_CLASSES,
            FEAT_DIM,
            AlfaConfig::default(),
            &mut LcgRng::new(5),
        )
        .expect("value should be present");
        let la = a
            .meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM)
            .expect("meta_step should succeed");
        let lb = b
            .meta_step(&x, &y, &x, &y, N_CLASSES, FEAT_DIM)
            .expect("meta_step should succeed");
        assert_eq!(la, lb);
        assert_eq!(a.base_params, b.base_params);
        assert_eq!(a.per_layer_lr, b.per_layer_lr);
    }
}
