//! Conv-4 backbone ↔ MAML integration.
//!
//! The base [`crate::maml::maml`] driver adapts a *bare linear classifier*
//! (`n_classes · feat_dim + n_classes` weights) in its inner loop by treating
//! the model as a flat `Vec<f32>` and taking finite-difference SGD steps on the
//! support-set loss.  The MLP backbone already plugs into that pattern through
//! its [`crate::network::backbone::MlpBackbone::to_params`] /
//! `from_params` flatten/unflatten contract.
//!
//! This module supplies the same contract for the convolutional
//! [`Conv4Backbone`]: it bundles the four-block convnet *feature extractor*
//! with a linear classification head, exposes the whole thing as one flat
//! parameter vector ([`Conv4MamlModel::to_params`] / [`Conv4MamlModel::from_params`]),
//! provides the forward closure ([`Conv4MamlModel::forward_logits`]) and the flat
//! support-set loss ([`Conv4MamlModel::task_loss_at_params`]) the inner loop needs,
//! and finally implements the MAML inner-loop adaptation
//! ([`Conv4MamlModel::inner_adapt`]) so a `Conv4Backbone` can be meta-trained by
//! exactly the same finite-difference second-order machinery the bare-linear
//! `maml_adapt` uses.
//!
//! # Parameter layout
//!
//! The flat vector is `[ backbone.to_params() | head_weights | head_biases ]`,
//! where the head maps the `feat_dim = backbone.output_dim()`-vector to
//! `n_classes` logits with a row-major `[n_classes × feat_dim]` weight matrix
//! followed by an `n_classes` bias vector.  This is the natural concatenation of
//! the backbone's own layout and the bare-linear classifier layout the base
//! `maml` module already understands, so the head slice alone is bit-compatible
//! with [`crate::maml::maml::maml_adapt`].
//!
//! # Cost
//!
//! Finite-difference gradients perturb every parameter twice per inner step, so
//! the cost is `2 · n_params · n_support` backbone forwards per step.  That is
//! deliberately exact (no hand-derived conv backprop) and is intended for the
//! small synthetic episodes used to *validate* the integration — the same
//! trade-off the bare-linear `maml_adapt` makes.

use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::{cross_entropy_loss, inner_sgd_step};
use crate::handle::LcgRng;
use crate::network::conv4_backbone::{Conv4Backbone, Conv4Config};

/// Inner-loop configuration for [`Conv4MamlModel::inner_adapt`].
///
/// Mirrors [`crate::maml::maml::MamlConfig`] (inner learning rate + number of
/// SGD steps) and adds the finite-difference perturbation `eps` used to estimate
/// the support-set gradient (the bare-linear driver hard-codes `1e-4`; the
/// convnet exposes it because the magnitude of conv activations makes the choice
/// matter a little more).
#[derive(Debug, Clone)]
pub struct Conv4MamlConfig {
    /// Inner-loop SGD learning rate.
    pub inner_lr: f32,
    /// Number of inner-loop SGD steps.
    pub n_inner_steps: usize,
    /// Central finite-difference step used to estimate the support-set gradient.
    pub fd_eps: f32,
}

impl Default for Conv4MamlConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            n_inner_steps: 1,
            fd_eps: 1e-3,
        }
    }
}

/// A Conv-4 feature extractor with a linear classification head, presented as a
/// single flat parameter vector for the MAML inner loop.
///
/// The model owns the backbone (which holds its own conv + BN parameters) and
/// the head weight/bias buffers.  All MAML-facing operations route through the
/// flat parameter vector so the meta-learner never needs to know the convnet's
/// internal structure.
pub struct Conv4MamlModel {
    backbone: Conv4Backbone,
    /// Head weights `[n_classes × feat_dim]` row-major.
    head_w: Vec<f32>,
    /// Head biases (length `n_classes`).
    head_b: Vec<f32>,
    n_classes: usize,
    feat_dim: usize,
}

impl Conv4MamlModel {
    /// Construct the model: a freshly-initialised Conv-4 backbone for `cfg`
    /// together with a Xavier-initialised linear head onto `n_classes`.
    ///
    /// # Errors
    /// * [`MetaError::InvalidNWay`] if `n_classes < 2`.
    /// * any [`Conv4Backbone::new`] construction error.
    pub fn new(cfg: Conv4Config, n_classes: usize, rng: &mut LcgRng) -> MetaResult<Self> {
        if n_classes < 2 {
            return Err(MetaError::InvalidNWay { n_way: n_classes });
        }
        let backbone = Conv4Backbone::new(cfg, rng)?;
        let feat_dim = backbone.output_dim();
        let limit = (6.0_f32 / (feat_dim + n_classes) as f32).sqrt();
        let mut head_w = vec![0.0_f32; n_classes * feat_dim];
        for v in head_w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Ok(Self {
            backbone,
            head_w,
            head_b: vec![0.0_f32; n_classes],
            n_classes,
            feat_dim,
        })
    }

    /// Number of output classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Flattened backbone feature dimensionality (= the head's input width).
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }

    /// Read-only access to the underlying backbone.
    pub fn backbone(&self) -> &Conv4Backbone {
        &self.backbone
    }

    /// Number of trainable parameters: backbone parameters plus the linear head.
    pub fn n_params(&self) -> usize {
        self.backbone.n_params() + self.head_w.len() + self.head_b.len()
    }

    /// Number of parameters belonging to the linear head only
    /// (`n_classes · feat_dim + n_classes`).
    pub fn head_param_count(&self) -> usize {
        self.head_w.len() + self.head_b.len()
    }

    /// Flatten the model into `[ backbone | head_w | head_b ]`.
    pub fn to_params(&self) -> Vec<f32> {
        let mut out = self.backbone.to_params();
        out.reserve(self.head_w.len() + self.head_b.len());
        out.extend_from_slice(&self.head_w);
        out.extend_from_slice(&self.head_b);
        out
    }

    /// Overwrite the model from a flat `n_params()`-length slice laid out as
    /// `[ backbone | head_w | head_b ]`.
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `params.len() != n_params()`.
    pub fn from_params(&mut self, params: &[f32]) -> MetaResult<()> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: params.len(),
            });
        }
        let bb_n = self.backbone.n_params();
        self.backbone.from_params(&params[..bb_n])?;
        let hw = self.head_w.len();
        self.head_w.copy_from_slice(&params[bb_n..bb_n + hw]);
        self.head_b.copy_from_slice(&params[bb_n + hw..]);
        Ok(())
    }

    /// Linear-head logits for a single already-extracted feature vector.
    fn head_forward(head_w: &[f32], head_b: &[f32], feat: &[f32], n_classes: usize) -> Vec<f32> {
        let feat_dim = feat.len();
        let mut logits = vec![0.0_f32; n_classes];
        for (c, logit) in logits.iter_mut().enumerate() {
            let row = &head_w[c * feat_dim..(c + 1) * feat_dim];
            *logit = row
                .iter()
                .zip(feat.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f32>()
                + head_b[c];
        }
        logits
    }

    /// Forward a single `(in_channels × input_h × input_w)` image through the
    /// current backbone + head, returning the `n_classes` logits.
    ///
    /// # Errors
    /// Propagates [`Conv4Backbone::forward`] errors (e.g. wrong input length).
    pub fn forward_logits(&self, image: &[f32]) -> MetaResult<Vec<f32>> {
        let feat = self.backbone.forward(image)?;
        Ok(Self::head_forward(
            &self.head_w,
            &self.head_b,
            &feat,
            self.n_classes,
        ))
    }

    /// Mean cross-entropy of a *flat* parameter vector on a labelled support set
    /// of `(in_channels × input_h × input_w)` images.
    ///
    /// This is the closure the inner loop differentiates: it rebuilds the
    /// backbone + head from `params`, forwards every support image, and returns
    /// the cross-entropy.  Construction is validated up front; per-image forward
    /// failures (which here only stem from a programmer dimension bug) fall back
    /// to [`f32::MAX`] so the finite-difference driver still produces a finite
    /// gradient, exactly as the bare-linear `task_loss_at_params` does.
    ///
    /// # Errors
    /// * [`MetaError::DimensionMismatch`] if `params.len() != n_params()` or the
    ///   support buffer length is not `n_support · image_len`.
    /// * [`MetaError::EmptySupport`] for an empty support set.
    pub fn task_loss_at_params(
        &self,
        params: &[f32],
        support_images: &[f32],
        support_y: &[u32],
    ) -> MetaResult<f32> {
        let n_support = support_y.len();
        if n_support == 0 {
            return Err(MetaError::EmptySupport);
        }
        let cfg = self.backbone.config();
        let image_len = cfg.in_channels * cfg.input_h * cfg.input_w;
        if support_images.len() != n_support * image_len {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * image_len,
                got: support_images.len(),
            });
        }
        let n_params = self.n_params();
        if params.len() != n_params {
            return Err(MetaError::DimensionMismatch {
                expected: n_params,
                got: params.len(),
            });
        }

        // Rebuild backbone + head from the flat vector.  Cloning the backbone is
        // cheap relative to the convolution work and keeps `self` immutable so
        // the closure can be called repeatedly with different `params`.
        let bb_n = self.backbone.n_params();
        let hw = self.head_w.len();
        let mut bb = self.backbone.clone();
        bb.from_params(&params[..bb_n])?;
        let head_w = &params[bb_n..bb_n + hw];
        let head_b = &params[bb_n + hw..];

        let mut logits = vec![0.0_f32; n_support * self.n_classes];
        for (s, image) in support_images.chunks(image_len).enumerate() {
            let feat = match bb.forward(image) {
                Ok(f) => f,
                Err(_) => return Ok(f32::MAX),
            };
            let row = Self::head_forward(head_w, head_b, &feat, self.n_classes);
            logits[s * self.n_classes..(s + 1) * self.n_classes].copy_from_slice(&row);
        }
        Ok(cross_entropy_loss(&logits, support_y, self.n_classes).unwrap_or(f32::MAX))
    }

    /// MAML inner-loop adaptation: starting from the current parameters, take
    /// `cfg.n_inner_steps` finite-difference SGD steps that minimise the
    /// support-set cross-entropy, returning the adapted flat parameter vector.
    ///
    /// The model itself is left unchanged — the caller decides whether to load
    /// the adapted vector back via [`Self::from_params`] (inner adaptation) or to
    /// keep it only to compute an outer meta-gradient.
    ///
    /// # Errors
    /// * [`MetaError::InvalidLr`] if `cfg.inner_lr <= 0` or non-finite.
    /// * [`MetaError::InvalidEpisodeConfig`] if `cfg.fd_eps <= 0` or non-finite.
    /// * any error from [`Self::task_loss_at_params`].
    pub fn inner_adapt(
        &self,
        support_images: &[f32],
        support_y: &[u32],
        cfg: &Conv4MamlConfig,
    ) -> MetaResult<Vec<f32>> {
        if cfg.inner_lr <= 0.0 || !cfg.inner_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
        }
        if cfg.fd_eps <= 0.0 || !cfg.fd_eps.is_finite() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("fd_eps must be > 0 and finite, got {}", cfg.fd_eps),
            });
        }
        // Validate shapes once (and surface a real error rather than the
        // f32::MAX fallback) before the inner loop starts.
        let start = self.to_params();
        let _ = self.task_loss_at_params(&start, support_images, support_y)?;

        let mut adapted = start;
        for _ in 0..cfg.n_inner_steps {
            let loss_fn = |p: &[f32]| {
                self.task_loss_at_params(p, support_images, support_y)
                    .unwrap_or(f32::MAX)
            };
            let grad = fd_gradient(&adapted, &loss_fn, cfg.fd_eps);
            adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
        }
        Ok(adapted)
    }

    /// Convenience wrapper that evaluates the support-set loss at the *current*
    /// parameters.
    ///
    /// # Errors
    /// Propagates [`Self::task_loss_at_params`].
    pub fn support_loss(&self, support_images: &[f32], support_y: &[u32]) -> MetaResult<f32> {
        self.task_loss_at_params(&self.to_params(), support_images, support_y)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Conv4Config {
        Conv4Config {
            in_channels: 1,
            width: 2,
            input_h: 16,
            input_w: 16,
        }
    }

    fn make_model(n_classes: usize) -> Conv4MamlModel {
        let mut rng = LcgRng::new(2026);
        Conv4MamlModel::new(tiny_cfg(), n_classes, &mut rng).expect("valid conv-maml model")
    }

    /// Build a tiny `n_classes`-way 1-shot synthetic few-shot task on 1×16×16
    /// images.  Class `c` is a flat image whose constant pixel value is a
    /// class-specific level, so the classes are linearly separable in pixel
    /// space and hence in the convolutional feature space.
    fn synthetic_task(n_classes: usize) -> (Vec<f32>, Vec<u32>) {
        let img_len = 16 * 16;
        let mut images = Vec::with_capacity(n_classes * img_len);
        let mut labels = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let level = 0.2 + 0.5 * c as f32;
            images.extend(std::iter::repeat_n(level, img_len));
            labels.push(c as u32);
        }
        (images, labels)
    }

    #[test]
    fn new_rejects_single_class() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Conv4MamlModel::new(tiny_cfg(), 1, &mut rng),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn n_params_is_backbone_plus_head() {
        let model = make_model(3);
        let feat_dim = model.feat_dim();
        let expected = model.backbone().n_params() + 3 * feat_dim + 3;
        assert_eq!(model.n_params(), expected);
        assert_eq!(model.to_params().len(), model.n_params());
    }

    #[test]
    fn head_slice_is_bare_linear_compatible() {
        // The head portion of the flat vector must be exactly the bare-linear
        // classifier layout the base maml module understands:
        // n_classes · feat_dim weights followed by n_classes biases.
        let model = make_model(2);
        assert_eq!(
            model.head_param_count(),
            2 * model.feat_dim() + 2,
            "head layout must match the bare-linear maml classifier"
        );
    }

    #[test]
    fn to_from_params_round_trips_exactly() {
        let mut model = make_model(3);
        let original = model.to_params();
        let perturbed: Vec<f32> = original
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (i as f32) * 0.0005 - 0.3)
            .collect();
        model.from_params(&perturbed).expect("from_params ok");
        assert_eq!(
            model.to_params(),
            perturbed,
            "flatten/unflatten must round-trip exactly"
        );
    }

    #[test]
    fn from_params_wrong_length_errs() {
        let mut model = make_model(2);
        assert!(matches!(
            model.from_params(&[0.0_f32; 5]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_logits_shape() {
        let model = make_model(4);
        let img = vec![0.1_f32; 16 * 16];
        let logits = model.forward_logits(&img).expect("forward ok");
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn task_loss_matches_forward_logits_loss() {
        // task_loss_at_params at the current parameters must equal the
        // cross-entropy assembled from forward_logits, proving the flat-vector
        // path reproduces the structured forward pass.
        let model = make_model(3);
        let (images, labels) = synthetic_task(3);
        let n = labels.len();
        let mut logits = Vec::with_capacity(n * 3);
        for s in 0..n {
            let img = &images[s * 256..(s + 1) * 256];
            logits.extend_from_slice(&model.forward_logits(img).expect("forward"));
        }
        let direct = cross_entropy_loss(&logits, &labels, 3).expect("ce");
        let via_params = model
            .task_loss_at_params(&model.to_params(), &images, &labels)
            .expect("task loss");
        assert!(
            (direct - via_params).abs() < 1e-5,
            "flat-vector loss {via_params} must match structured loss {direct}"
        );
    }

    #[test]
    fn task_loss_empty_support_errs() {
        let model = make_model(2);
        assert!(matches!(
            model.task_loss_at_params(&model.to_params(), &[], &[]),
            Err(MetaError::EmptySupport)
        ));
    }

    #[test]
    fn task_loss_wrong_support_len_errs() {
        let model = make_model(2);
        let (_images, labels) = synthetic_task(2);
        let bad = vec![0.0_f32; 10];
        assert!(matches!(
            model.task_loss_at_params(&model.to_params(), &bad, &labels),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn inner_adapt_round_trips_through_from_params() {
        // The adapted vector returned by inner_adapt must be loadable, and the
        // backbone n_params must be unchanged — proving the layout is stable
        // under adaptation.
        let model = make_model(2);
        let (images, labels) = synthetic_task(2);
        let cfg = Conv4MamlConfig {
            inner_lr: 0.05,
            n_inner_steps: 1,
            fd_eps: 1e-3,
        };
        let adapted = model.inner_adapt(&images, &labels, &cfg).expect("adapt");
        assert_eq!(adapted.len(), model.n_params());
        let mut loaded = make_model(2);
        loaded.from_params(&adapted).expect("load adapted");
    }

    #[test]
    fn inner_adapt_reduces_support_loss() {
        // The defining MAML property: a few inner-loop steps on the support set
        // reduce the support-set loss for the convolutional backbone, exactly as
        // they do for the bare-linear classifier.
        let model = make_model(3);
        let (images, labels) = synthetic_task(3);
        let loss_before = model.support_loss(&images, &labels).expect("loss before");

        let cfg = Conv4MamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 3,
            fd_eps: 1e-3,
        };
        let adapted = model.inner_adapt(&images, &labels, &cfg).expect("adapt");
        let loss_after = model
            .task_loss_at_params(&adapted, &images, &labels)
            .expect("loss after");

        assert!(
            loss_after < loss_before,
            "MAML inner loop must reduce the Conv4 support loss: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn inner_adapt_changes_backbone_and_head() {
        // Adaptation must move parameters in *both* the backbone and the head
        // (otherwise the convnet would not actually be meta-trained).
        let model = make_model(2);
        let (images, labels) = synthetic_task(2);
        let cfg = Conv4MamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 2,
            fd_eps: 1e-3,
        };
        let before = model.to_params();
        let after = model.inner_adapt(&images, &labels, &cfg).expect("adapt");
        let bb_n = model.backbone().n_params();
        let backbone_moved = before[..bb_n]
            .iter()
            .zip(after[..bb_n].iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        let head_moved = before[bb_n..]
            .iter()
            .zip(after[bb_n..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(backbone_moved, "inner loop must update backbone params");
        assert!(head_moved, "inner loop must update head params");
    }

    #[test]
    fn inner_adapt_rejects_bad_lr() {
        let model = make_model(2);
        let (images, labels) = synthetic_task(2);
        let cfg = Conv4MamlConfig {
            inner_lr: 0.0,
            n_inner_steps: 1,
            fd_eps: 1e-3,
        };
        assert!(matches!(
            model.inner_adapt(&images, &labels, &cfg),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    #[test]
    fn inner_adapt_rejects_bad_fd_eps() {
        let model = make_model(2);
        let (images, labels) = synthetic_task(2);
        let cfg = Conv4MamlConfig {
            inner_lr: 0.05,
            n_inner_steps: 1,
            fd_eps: 0.0,
        };
        assert!(matches!(
            model.inner_adapt(&images, &labels, &cfg),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn inner_adapt_deterministic_under_seed() {
        let (images, labels) = synthetic_task(2);
        let cfg = Conv4MamlConfig::default();
        let a = make_model(2)
            .inner_adapt(&images, &labels, &cfg)
            .expect("a");
        let b = make_model(2)
            .inner_adapt(&images, &labels, &cfg)
            .expect("b");
        assert_eq!(a, b, "same seed + same task must adapt identically");
    }
}
