//! MEMO — Memory-Efficient Expandable MOdel (Zhou 2022).
//!
//! Reference: Zhou, D.-W., Wang, Q.-W., Ye, H.-J. & Zhan, D.-C. (2022). "A Model
//! or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning."
//! *International Conference on Learning Representations* (ICLR 2023, arXiv
//! 2205.13218).
//!
//! # Overview
//!
//! MEMO decouples a deep network into a **generalized block** (the shallow
//! layers, shared by every task) and **specialized blocks** (the deep layers,
//! one expanded copy per task). The key empirical insight is that the shallow
//! layers are highly transferable across tasks, so only the deep specialized
//! blocks need to be replicated — yielding far cheaper expansion than methods
//! that copy the *entire* backbone (e.g. DER).
//!
//! ```text
//!   x ──► generalized block g(·) ──► shared feature  s = g(x)
//!                                     │
//!              ┌──────────────────────┼──────────────────────┐
//!              ▼                       ▼                       ▼
//!         spec block φ₀          spec block φ₁     …     spec block φ_{K-1}
//!              │                       │                       │
//!              └────── concat features [φ₀(s) ‖ φ₁(s) ‖ … ] ───┘
//!                                     │
//!                                     ▼
//!                          unified classifier  W·feat + b  →  logits
//! ```
//!
//! Each block is a single ReLU layer. The generalized block is trained on the
//! first task and then **frozen**; every subsequent task appends a new
//! specialized block (the `expansion_rate` scales its width relative to the
//! shared feature dimension). The unified classifier grows to span all
//! registered tasks' concatenated specialized features.
//!
//! This module exposes the *architecture and forward path* (FP64) plus
//! expansion bookkeeping; weight learning is left to the caller / an external
//! optimiser, matching the other architecture primitives in this crate.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

fn xavier_init(buf: &mut [f64], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = xavier_scale(fan_in, fan_out);
    for v in buf.iter_mut() {
        let u = rng.next_f32() as f64 * 2.0 - 1.0;
        *v = u * scale;
    }
}

/// Linear forward `out = W x + b`, `W` row-major `[out_dim × in_dim]`.
fn linear_forward(w: &[f64], b: &[f64], x: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for i in 0..out_dim {
        let mut s = b[i];
        let row = i * in_dim;
        for j in 0..in_dim {
            s += w[row + j] * x[j];
        }
        out[i] = s;
    }
    out
}

// ─── configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`Memo`].
#[derive(Debug, Clone)]
pub struct MemoConfig {
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// Output width of the shared generalized block (`s = g(x)` dimension).
    pub shared_dim: usize,
    /// Specialized-block width as a fraction of `shared_dim`
    /// (`spec_dim = round(expansion_rate · shared_dim)`, floored at 1).
    pub expansion_rate: f64,
}

impl Default for MemoConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            shared_dim: 32,
            expansion_rate: 0.5,
        }
    }
}

impl MemoConfig {
    fn validate(&self) -> ContinualResult<()> {
        if self.input_dim == 0 || self.shared_dim == 0 {
            return Err(ContinualError::EmptyInput);
        }
        if !self.expansion_rate.is_finite() || self.expansion_rate <= 0.0 {
            return Err(ContinualError::InvalidThreshold {
                threshold: self.expansion_rate as f32,
            });
        }
        Ok(())
    }

    /// Resolved specialized-block width.
    fn spec_dim(&self) -> usize {
        ((self.expansion_rate * self.shared_dim as f64).round() as usize).max(1)
    }
}

// ─── specialized block ──────────────────────────────────────────────────────────

/// A single task-specific specialized block: one ReLU layer mapping the shared
/// feature `s` (dim `shared_dim`) to a `spec_dim`-wide representation.
#[derive(Debug, Clone)]
pub struct SpecializedBlock {
    /// Weight matrix, shape `[spec_dim × shared_dim]`, row-major.
    pub weights: Vec<f64>,
    /// Bias, length `spec_dim`.
    pub bias: Vec<f64>,
    /// Task identifier this block was created for.
    pub task_id: usize,
}

// ─── model state ────────────────────────────────────────────────────────────────

/// MEMO expandable model state.
#[derive(Debug, Clone)]
pub struct Memo {
    /// Generalized (shared) block weight, shape `[shared_dim × input_dim]`.
    generalized_w: Vec<f64>,
    /// Generalized block bias, length `shared_dim`.
    generalized_b: Vec<f64>,
    /// One specialized block per registered task, in insertion order.
    specialized: Vec<SpecializedBlock>,
    /// Unified classifier weight, shape `[n_outputs × total_spec_dim]`,
    /// row-major. Grows when blocks/outputs are added.
    classifier_w: Vec<f64>,
    /// Unified classifier bias, length `n_outputs`.
    classifier_b: Vec<f64>,
    /// Number of classifier outputs (logits).
    n_outputs: usize,
    config: MemoConfig,
}

impl Memo {
    /// Construct a MEMO model with a single specialized block (task 0) and a
    /// classifier producing `n_outputs_init` logits.
    ///
    /// # Errors
    /// - Propagates `MemoConfig::validate`.
    /// - [`ContinualError::EmptyInput`] if `n_outputs_init == 0`.
    pub fn new(
        config: MemoConfig,
        n_outputs_init: usize,
        rng: &mut LcgRng,
    ) -> ContinualResult<Self> {
        config.validate()?;
        if n_outputs_init == 0 {
            return Err(ContinualError::EmptyInput);
        }
        let shared = config.shared_dim;
        let spec = config.spec_dim();

        let mut generalized_w = vec![0.0_f64; shared * config.input_dim];
        xavier_init(&mut generalized_w, config.input_dim, shared, rng);
        let mut generalized_b = vec![0.0_f64; shared];
        xavier_init(&mut generalized_b, config.input_dim, shared, rng);

        let mut block_w = vec![0.0_f64; spec * shared];
        xavier_init(&mut block_w, shared, spec, rng);
        let mut block_b = vec![0.0_f64; spec];
        xavier_init(&mut block_b, shared, spec, rng);
        let block0 = SpecializedBlock {
            weights: block_w,
            bias: block_b,
            task_id: 0,
        };

        // Classifier over the single block's features.
        let mut classifier_w = vec![0.0_f64; n_outputs_init * spec];
        xavier_init(&mut classifier_w, spec, n_outputs_init, rng);
        let classifier_b = vec![0.0_f64; n_outputs_init];

        Ok(Self {
            generalized_w,
            generalized_b,
            specialized: vec![block0],
            classifier_w,
            classifier_b,
            n_outputs: n_outputs_init,
            config,
        })
    }

    /// Number of registered tasks (= number of specialized blocks).
    #[must_use]
    pub fn n_tasks(&self) -> usize {
        self.specialized.len()
    }

    /// Number of classifier outputs.
    #[must_use]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Total dimensionality of the concatenated specialized features.
    #[must_use]
    pub fn total_feature_dim(&self) -> usize {
        self.specialized.len() * self.config.spec_dim()
    }

    /// Apply the (frozen after task 0) generalized block: `s = relu(W x + b)`.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn shared_feature(&self, x: &[f64]) -> ContinualResult<Vec<f64>> {
        if x.len() != self.config.input_dim {
            return Err(ContinualError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }
        let pre = linear_forward(
            &self.generalized_w,
            &self.generalized_b,
            x,
            self.config.input_dim,
            self.config.shared_dim,
        );
        Ok(pre.into_iter().map(relu).collect())
    }

    /// Concatenated specialized features `[φ₀(s) ‖ φ₁(s) ‖ … ]` for input `x`.
    /// Length is [`Self::total_feature_dim`].
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn features(&self, x: &[f64]) -> ContinualResult<Vec<f64>> {
        let s = self.shared_feature(x)?;
        let shared = self.config.shared_dim;
        let spec = self.config.spec_dim();
        let mut feat = Vec::with_capacity(self.total_feature_dim());
        for block in &self.specialized {
            let pre = linear_forward(&block.weights, &block.bias, &s, shared, spec);
            feat.extend(pre.into_iter().map(relu));
        }
        Ok(feat)
    }

    /// Forward to logits: `W · features(x) + b`. Output length =
    /// [`Self::n_outputs`].
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn forward(&self, x: &[f64]) -> ContinualResult<Vec<f64>> {
        let feat = self.features(x)?;
        Ok(linear_forward(
            &self.classifier_w,
            &self.classifier_b,
            &feat,
            feat.len(),
            self.n_outputs,
        ))
    }

    /// Expand the model for a new task: append a freshly initialised
    /// specialized block and grow the classifier by `new_outputs` logits.
    ///
    /// The classifier is rebuilt so its input width matches the *new* total
    /// feature dimension; previously-learned classifier rows are preserved for
    /// the feature columns that already existed, and the new feature columns /
    /// new output rows are Xavier-initialised. The generalized block is left
    /// untouched (frozen).
    ///
    /// # Errors
    /// - [`ContinualError::EmptyInput`] if `new_outputs == 0`.
    pub fn expand(&mut self, new_outputs: usize, rng: &mut LcgRng) -> ContinualResult<()> {
        if new_outputs == 0 {
            return Err(ContinualError::EmptyInput);
        }
        let shared = self.config.shared_dim;
        let spec = self.config.spec_dim();
        let task_id = self.specialized.len();

        // New specialized block.
        let mut block_w = vec![0.0_f64; spec * shared];
        xavier_init(&mut block_w, shared, spec, rng);
        let mut block_b = vec![0.0_f64; spec];
        xavier_init(&mut block_b, shared, spec, rng);

        let old_feat_dim = self.total_feature_dim();
        let old_outputs = self.n_outputs;

        self.specialized.push(SpecializedBlock {
            weights: block_w,
            bias: block_b,
            task_id,
        });

        let new_feat_dim = self.total_feature_dim();
        let new_outputs_total = old_outputs + new_outputs;

        // Rebuild classifier `[new_outputs_total × new_feat_dim]`, copying the
        // old block of weights into the top-left corner.
        let mut new_w = vec![0.0_f64; new_outputs_total * new_feat_dim];
        // Xavier-init the whole thing first, then overwrite the preserved part.
        xavier_init(&mut new_w, new_feat_dim, new_outputs_total, rng);
        for o in 0..old_outputs {
            for c in 0..old_feat_dim {
                new_w[o * new_feat_dim + c] = self.classifier_w[o * old_feat_dim + c];
            }
        }
        let mut new_b = vec![0.0_f64; new_outputs_total];
        new_b[..old_outputs].copy_from_slice(&self.classifier_b[..old_outputs]);
        // (new output biases stay at 0.0)

        self.classifier_w = new_w;
        self.classifier_b = new_b;
        self.n_outputs = new_outputs_total;
        Ok(())
    }

    /// Argmax class prediction for input `x`.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn predict(&self, x: &[f64]) -> ContinualResult<usize> {
        let logits = self.forward(x)?;
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best)
    }

    /// Borrow the specialized block for `task` (insertion order).
    ///
    /// # Errors
    /// - [`ContinualError::TaskIndexOutOfRange`] if `task >= n_tasks`.
    pub fn specialized_block(&self, task: usize) -> ContinualResult<&SpecializedBlock> {
        self.specialized
            .get(task)
            .ok_or(ContinualError::TaskIndexOutOfRange {
                index: task,
                n_tasks: self.specialized.len(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MemoConfig {
        MemoConfig {
            input_dim: 6,
            shared_dim: 8,
            expansion_rate: 0.5, // spec_dim = round(0.5*8) = 4
        }
    }

    // -------------------- construction / validation ------------------------

    #[test]
    fn new_ok() {
        let mut rng = LcgRng::new(1);
        let m = Memo::new(cfg(), 3, &mut rng).expect("Memo should construct with valid config");
        assert_eq!(m.n_tasks(), 1);
        assert_eq!(m.n_outputs(), 3);
        assert_eq!(m.total_feature_dim(), 4); // one block × spec_dim 4
    }

    #[test]
    fn new_input_dim_0_error() {
        let mut rng = LcgRng::new(1);
        let c = MemoConfig {
            input_dim: 0,
            ..cfg()
        };
        assert!(matches!(
            Memo::new(c, 2, &mut rng),
            Err(ContinualError::EmptyInput)
        ));
    }

    #[test]
    fn new_bad_expansion_error() {
        let mut rng = LcgRng::new(1);
        let c = MemoConfig {
            expansion_rate: 0.0,
            ..cfg()
        };
        assert!(matches!(
            Memo::new(c, 2, &mut rng),
            Err(ContinualError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn new_zero_outputs_error() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Memo::new(cfg(), 0, &mut rng),
            Err(ContinualError::EmptyInput)
        ));
    }

    // -------------------- forward shapes -----------------------------------

    #[test]
    fn shared_feature_shape_and_relu_nonneg() {
        let mut rng = LcgRng::new(5);
        let m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        let s = m
            .shared_feature(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
            .expect("shared feature extraction should succeed on valid input");
        assert_eq!(s.len(), 8);
        assert!(s.iter().all(|&v| v >= 0.0), "ReLU output must be ≥ 0");
    }

    #[test]
    fn features_shape() {
        let mut rng = LcgRng::new(5);
        let m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        let f = m
            .features(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("feature extraction should succeed on valid input");
        assert_eq!(f.len(), m.total_feature_dim());
        assert_eq!(f.len(), 4);
    }

    #[test]
    fn forward_logits_shape() {
        let mut rng = LcgRng::new(5);
        let m = Memo::new(cfg(), 3, &mut rng).expect("Memo should construct with valid config");
        let logits = m
            .forward(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("Memo forward pass should succeed on valid input");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_dim_mismatch_error() {
        let mut rng = LcgRng::new(5);
        let m = Memo::new(cfg(), 3, &mut rng).expect("Memo should construct with valid config");
        let r = m.forward(&[0.1, 0.2]); // wrong input dim
        assert!(matches!(r, Err(ContinualError::DimensionMismatch { .. })));
    }

    // -------------------- expansion ----------------------------------------

    #[test]
    fn expand_grows_blocks_and_outputs() {
        let mut rng = LcgRng::new(9);
        let mut m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        assert_eq!(m.n_tasks(), 1);
        assert_eq!(m.total_feature_dim(), 4);
        m.expand(2, &mut rng)
            .expect("Memo expansion should succeed");
        assert_eq!(m.n_tasks(), 2);
        assert_eq!(m.n_outputs(), 4); // 2 + 2
        assert_eq!(m.total_feature_dim(), 8); // 2 blocks × 4
    }

    #[test]
    fn expand_forward_still_valid() {
        let mut rng = LcgRng::new(9);
        let mut m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        m.expand(3, &mut rng)
            .expect("Memo expansion should succeed");
        let logits = m
            .forward(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("Memo forward pass should succeed on valid input");
        assert_eq!(logits.len(), 5); // 2 + 3
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn expand_preserves_old_classifier_rows() {
        // The preserved top-left classifier weights should survive expansion.
        let mut rng = LcgRng::new(13);
        let mut m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        let old_w00 = m.classifier_w[0]; // output 0, feature 0
        let old_feat_dim = m.total_feature_dim();
        let old_w_last = m.classifier_w[old_feat_dim - 1]; // output 0, last feat
        m.expand(1, &mut rng)
            .expect("Memo expansion should succeed");
        let new_feat_dim = m.total_feature_dim();
        assert!((m.classifier_w[0] - old_w00).abs() < 1e-12);
        // The last *old* feature column for output 0 maps to index old_feat_dim-1
        // in the new row of width new_feat_dim.
        assert!((m.classifier_w[old_feat_dim - 1] - old_w_last).abs() < 1e-12);
        assert!(new_feat_dim > old_feat_dim);
    }

    #[test]
    fn expand_zero_outputs_error() {
        let mut rng = LcgRng::new(9);
        let mut m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        assert!(matches!(
            m.expand(0, &mut rng),
            Err(ContinualError::EmptyInput)
        ));
    }

    // -------------------- predict / accessors ------------------------------

    #[test]
    fn predict_in_range() {
        let mut rng = LcgRng::new(21);
        let m = Memo::new(cfg(), 4, &mut rng).expect("Memo should construct with valid config");
        let c = m
            .predict(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("Memo prediction should succeed on valid input");
        assert!(c < 4);
    }

    #[test]
    fn specialized_block_accessor() {
        let mut rng = LcgRng::new(21);
        let mut m = Memo::new(cfg(), 2, &mut rng).expect("Memo should construct with valid config");
        m.expand(1, &mut rng)
            .expect("Memo expansion should succeed");
        assert_eq!(
            m.specialized_block(0)
                .expect("specialized block should exist for valid index")
                .task_id,
            0
        );
        assert_eq!(
            m.specialized_block(1)
                .expect("specialized block should exist for valid index")
                .task_id,
            1
        );
        assert!(matches!(
            m.specialized_block(5),
            Err(ContinualError::TaskIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn deterministic_construction() {
        let mut a = LcgRng::new(777);
        let mut b = LcgRng::new(777);
        let ma = Memo::new(cfg(), 3, &mut a).expect("Memo should construct with valid config");
        let mb = Memo::new(cfg(), 3, &mut b).expect("Memo should construct with valid config");
        let x = [0.2, -0.1, 0.4, 0.3, 0.0, 0.5];
        assert_eq!(
            ma.forward(&x)
                .expect("Memo forward pass should succeed on valid input"),
            mb.forward(&x)
                .expect("Memo forward pass should succeed on valid input")
        );
    }

    #[test]
    fn config_default_valid() {
        assert!(MemoConfig::default().validate().is_ok());
    }
}
