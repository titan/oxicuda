//! Data2Vec struct — student/teacher encoder pair with EMA update.
//!
//! This module provides a struct-based wrapper that manages the full Data2Vec
//! training loop: a student encoder, an EMA teacher state, and the loss
//! computation via the functional API in [`crate::masked::data2vec`].
//!
//! ## Architecture
//! Both student and teacher are `n_layers`-deep encoders where each layer is a
//! single linear transform followed by ReLU, operating token-by-token over a
//! sequence of shape `[n_patches × d_model]`.
//!
//! Teacher weights are stored flat in [`Data2VecState::teacher_params`] in the
//! same layout as the student parameters:
//! ```text
//! [ layer_0_w (d_model × d_model) | layer_0_b (d_model) |
//!   layer_1_w (d_model × d_model) | layer_1_b (d_model) | ... ]
//! ```
//!
//! ## EMA update
//! ```text
//! teacher ← ema_decay · teacher + (1 − ema_decay) · student
//! ```
//!
//! Reference: "data2vec: A General Framework for Self-supervised Learning in
//! Speech, Vision and Language", Baevski et al., ICML 2022.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;
use crate::masked::data2vec::{Data2VecConfig, Data2VecState, data2vec_loss};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the struct-based [`Data2VecModel`].
#[derive(Debug, Clone)]
pub struct Data2VecModelConfig {
    /// Token / patch embedding dimension.
    pub d_model: usize,
    /// Number of encoder layers (must be >= 1).
    pub n_layers: usize,
    /// EMA decay coefficient for the teacher update (e.g. 0.999).
    pub ema_decay: f32,
    /// Fraction of tokens to mask during the student forward pass.
    pub mask_ratio: f32,
    /// Number of top teacher layer outputs to average for the target
    /// (passed through to [`Data2VecConfig::top_k_average`]).
    pub k_top_layers: usize,
}

impl Default for Data2VecModelConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            ema_decay: 0.999,
            mask_ratio: 0.65,
            k_top_layers: 1,
        }
    }
}

// ─── Data2VecModel ────────────────────────────────────────────────────────────

/// Struct-based Data2Vec model that owns student encoder layers and a teacher
/// EMA state.
///
/// Student weights per layer are stored as:
/// - `student_w[l]` — `[d_model × d_model]` weight matrix (row-major).
/// - `student_b[l]` — `[d_model]` bias vector.
///
/// Teacher weights are stored flat in [`Data2VecState::teacher_params`] in the
/// same sequential layout (w0, b0, w1, b1, …).
#[derive(Debug, Clone)]
pub struct Data2VecModel {
    /// Per-layer student weight matrices `n_layers × [d_model × d_model]`.
    student_w: Vec<Vec<f32>>,
    /// Per-layer student bias vectors `n_layers × [d_model]`.
    student_b: Vec<Vec<f32>>,
    /// EMA teacher state (flat parameter vector + step counter).
    teacher_state: Data2VecState,
    /// Configuration used to create this model.
    config: Data2VecModelConfig,
}

impl Data2VecModel {
    /// Create a new [`Data2VecModel`] with Kaiming-initialised student layers and
    /// a teacher state cloned from the initial student parameters.
    ///
    /// # Errors
    /// - [`SslError::InvalidParameter`] when `d_model == 0`.
    /// - [`SslError::InvalidParameter`] when `n_layers == 0`.
    pub fn new(config: Data2VecModelConfig, rng: &mut LcgRng) -> SslResult<Self> {
        if config.d_model == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_model".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.n_layers == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_layers".into(),
                reason: "must be >= 1".into(),
            });
        }

        let d = config.d_model;
        let mut student_w = Vec::with_capacity(config.n_layers);
        let mut student_b = Vec::with_capacity(config.n_layers);

        for _ in 0..config.n_layers {
            let w = kaiming_init(d, d, rng);
            let b = vec![0.0_f32; d];
            student_w.push(w);
            student_b.push(b);
        }

        // Flatten student params into teacher initial state.
        let flat_params = flatten_params(&student_w, &student_b);
        let teacher_state = Data2VecState::new(&flat_params);

        Ok(Self {
            student_w,
            student_b,
            teacher_state,
            config,
        })
    }

    /// Encode a sequence of patch embeddings with the **student** encoder.
    ///
    /// Each layer applies: `token ← ReLU(W · token + b)` independently per token.
    ///
    /// # Arguments
    /// * `x`        — flat `[n_patches × d_model]` input (row-major).
    /// * `n_patches` — number of tokens in the sequence.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] when `x.len() != n_patches * d_model`.
    pub fn encode_student(&self, x: &[f32], n_patches: usize) -> SslResult<Vec<f32>> {
        let d = self.config.d_model;
        let expected = n_patches * d;
        if x.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        apply_encoder_layers(
            x,
            n_patches,
            d,
            &self.student_w,
            &self.student_b,
            self.config.n_layers,
        )
    }

    /// Encode a sequence of patch embeddings with the **teacher** encoder.
    ///
    /// Uses the teacher weight matrices stored in [`Data2VecState::teacher_params`].
    ///
    /// # Arguments
    /// * `x`        — flat `[n_patches × d_model]` input (row-major).
    /// * `n_patches` — number of tokens in the sequence.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] when `x.len() != n_patches * d_model`.
    pub fn encode_teacher(&self, x: &[f32], n_patches: usize) -> SslResult<Vec<f32>> {
        let d = self.config.d_model;
        let expected = n_patches * d;
        if x.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let (teacher_w, teacher_b) =
            unflatten_params(self.teacher_state.teacher(), d, self.config.n_layers)?;
        apply_encoder_layers(
            x,
            n_patches,
            d,
            &teacher_w,
            &teacher_b,
            self.config.n_layers,
        )
    }

    /// Compute the Data2Vec loss for a masked input.
    ///
    /// 1. Encodes with the student encoder → `student_repr [n_patches × d_model]`.
    /// 2. Encodes with the teacher encoder → `teacher_repr [n_patches × d_model]`.
    /// 3. Computes Huber loss at masked positions via [`data2vec_loss`].
    ///
    /// # Arguments
    /// * `x`        — flat `[n_patches × d_model]` input.
    /// * `mask`     — `[n_patches]` boolean; `true` = masked position.
    /// * `n_patches` — number of tokens.
    ///
    /// # Errors
    /// Propagates dimension and config errors.
    pub fn loss(&self, x: &[f32], mask: &[bool], n_patches: usize) -> SslResult<f32> {
        let d = self.config.d_model;
        let student_repr = self.encode_student(x, n_patches)?;
        let teacher_repr = self.encode_teacher(x, n_patches)?;
        let d2v_config = Data2VecConfig {
            mask_ratio: self.config.mask_ratio,
            momentum: self.config.ema_decay,
            top_k_average: self.config.k_top_layers,
            ..Data2VecConfig::default()
        };
        let result = data2vec_loss(
            &student_repr,
            &teacher_repr,
            mask,
            n_patches,
            d,
            &d2v_config,
        )?;
        Ok(result.loss)
    }

    /// Apply the EMA update: `teacher ← ema_decay · teacher + (1 − ema_decay) · student`.
    ///
    /// # Errors
    /// - [`SslError::InvalidMomentum`] when `ema_decay` is not in `[0, 1]`.
    /// - [`SslError::DimensionMismatch`] when param shapes mismatch (should not
    ///   occur in normal usage).
    pub fn ema_update(&mut self) -> SslResult<()> {
        let flat_student = flatten_params(&self.student_w, &self.student_b);
        self.teacher_state
            .update_teacher(&flat_student, self.config.ema_decay)
    }

    /// Return the token/patch embedding dimension.
    #[inline]
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Kaiming (He) normal weight init: `scale = sqrt(2 / fan_in)`.
fn kaiming_init(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (2.0_f32 / in_dim as f32).sqrt();
    let mut w = vec![0.0_f32; out_dim * in_dim];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

/// Row-major matrix-vector multiply with optional ReLU.
///
/// `out[i] = max(0, b[i] + Σ_j w[i·in_dim + j] * x[j])` if `relu`, else without max.
fn linear_relu(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        let row = i * in_dim;
        for j in 0..in_dim {
            acc += w[row + j] * x[j];
        }
        out[i] = acc.max(0.0);
    }
    out
}

/// Flatten student weights and biases into a single contiguous `Vec<f32>`.
///
/// Layout: `[w0_flat, b0, w1_flat, b1, …]`
fn flatten_params(ws: &[Vec<f32>], bs: &[Vec<f32>]) -> Vec<f32> {
    let total: usize =
        ws.iter().map(|w| w.len()).sum::<usize>() + bs.iter().map(|b| b.len()).sum::<usize>();
    let mut flat = Vec::with_capacity(total);
    for (w, b) in ws.iter().zip(bs.iter()) {
        flat.extend_from_slice(w);
        flat.extend_from_slice(b);
    }
    flat
}

/// Per-layer weight and bias vectors `(weights, biases)`.
type LayerParams = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Inverse of [`flatten_params`]: reconstruct per-layer weight/bias vectors.
///
/// # Errors
/// [`SslError::DimensionMismatch`] when the flat slice is shorter than expected.
fn unflatten_params(flat: &[f32], d_model: usize, n_layers: usize) -> SslResult<LayerParams> {
    let w_size = d_model * d_model;
    let b_size = d_model;
    let layer_size = w_size + b_size;
    let expected = n_layers * layer_size;
    if flat.len() < expected {
        return Err(SslError::DimensionMismatch {
            expected,
            got: flat.len(),
        });
    }
    let mut ws = Vec::with_capacity(n_layers);
    let mut bs = Vec::with_capacity(n_layers);
    let mut offset = 0;
    for _ in 0..n_layers {
        ws.push(flat[offset..offset + w_size].to_vec());
        offset += w_size;
        bs.push(flat[offset..offset + b_size].to_vec());
        offset += b_size;
    }
    Ok((ws, bs))
}

/// Apply `n_layers` linear+ReLU layers to every token in `x [n_patches × d_model]`.
fn apply_encoder_layers(
    x: &[f32],
    n_patches: usize,
    d_model: usize,
    ws: &[Vec<f32>],
    bs: &[Vec<f32>],
    n_layers: usize,
) -> SslResult<Vec<f32>> {
    // Work token-by-token; keep results in a flat buffer.
    let mut current = x.to_vec();
    for l in 0..n_layers {
        let w = &ws[l];
        let b = &bs[l];
        let mut next = Vec::with_capacity(n_patches * d_model);
        for t in 0..n_patches {
            let start = t * d_model;
            let token = &current[start..start + d_model];
            next.extend_from_slice(&linear_relu(w, b, token, d_model, d_model));
        }
        current = next;
    }
    Ok(current)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::masked::data2vec::data2vec_mask;

    fn make_model(seed: u64) -> Data2VecModel {
        let mut rng = LcgRng::new(seed);
        Data2VecModel::new(Data2VecModelConfig::default(), &mut rng)
            .expect("value should be present")
    }

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn make_mask(n_patches: usize, mask_ratio: f32, seed: u64) -> Vec<bool> {
        let mut rng = LcgRng::new(seed);
        data2vec_mask(n_patches, mask_ratio, &mut rng).expect("data2vec_mask should succeed")
    }

    #[test]
    fn encode_student_shape() {
        let m = make_model(1);
        let n_patches = 8;
        let d = m.d_model();
        let x = random_vec(n_patches * d, 2);
        let out = m
            .encode_student(&x, n_patches)
            .expect("encode_student should succeed");
        assert_eq!(
            out.len(),
            n_patches * d,
            "student output must have len == n_patches * d_model"
        );
    }

    #[test]
    fn encode_teacher_shape() {
        let m = make_model(3);
        let n_patches = 8;
        let d = m.d_model();
        let x = random_vec(n_patches * d, 4);
        let out = m
            .encode_teacher(&x, n_patches)
            .expect("encode_teacher should succeed");
        assert_eq!(
            out.len(),
            n_patches * d,
            "teacher output must have len == n_patches * d_model"
        );
    }

    #[test]
    fn loss_finite() {
        let m = make_model(5);
        let n_patches = 8;
        let d = m.d_model();
        let x = random_vec(n_patches * d, 6);
        let mask = make_mask(n_patches, 0.5, 7);
        let l = m.loss(&x, &mask, n_patches).expect("loss should succeed");
        assert!(l.is_finite(), "loss must be finite, got {l}");
    }

    #[test]
    fn loss_nonneg() {
        let m = make_model(8);
        let n_patches = 8;
        let d = m.d_model();
        let x = random_vec(n_patches * d, 9);
        let mask = make_mask(n_patches, 0.5, 10);
        let l = m.loss(&x, &mask, n_patches).expect("loss should succeed");
        assert!(l >= 0.0, "Huber loss must be >= 0, got {l}");
    }

    #[test]
    fn ema_update_changes_teacher() {
        let mut m = make_model(11);
        let teacher_before = m.teacher_state.teacher_params.clone();
        // The student is already different from the teacher snapshot if we
        // modify student weights slightly.
        for v in m.student_w[0].iter_mut() {
            *v += 1.0;
        }
        m.ema_update().expect("ema_update should succeed");
        let teacher_after = &m.teacher_state.teacher_params;
        let diff: f32 = teacher_before
            .iter()
            .zip(teacher_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-8,
            "teacher must change after ema_update when student differs, diff={diff}"
        );
    }

    #[test]
    fn ema_update_preserves_student() {
        let mut m = make_model(12);
        let student_w_before: Vec<Vec<f32>> = m.student_w.clone();
        let student_b_before: Vec<Vec<f32>> = m.student_b.clone();
        m.ema_update().expect("ema_update should succeed");
        assert_eq!(
            m.student_w, student_w_before,
            "student weights must not change during ema_update"
        );
        assert_eq!(
            m.student_b, student_b_before,
            "student biases must not change during ema_update"
        );
    }

    #[test]
    fn d_model_0_error() {
        let mut rng = LcgRng::new(13);
        let result = Data2VecModel::new(
            Data2VecModelConfig {
                d_model: 0,
                ..Data2VecModelConfig::default()
            },
            &mut rng,
        );
        assert!(result.is_err(), "d_model=0 must return Err");
    }

    #[test]
    fn n_layers_1_works() {
        let mut rng = LcgRng::new(14);
        let m = Data2VecModel::new(
            Data2VecModelConfig {
                n_layers: 1,
                ..Data2VecModelConfig::default()
            },
            &mut rng,
        )
        .expect("value should be present");
        let n_patches = 4;
        let x = random_vec(n_patches * m.d_model(), 15);
        let out = m
            .encode_student(&x, n_patches)
            .expect("encode_student should succeed");
        assert_eq!(out.len(), n_patches * m.d_model());
    }

    #[test]
    fn different_x_different_encode() {
        let m = make_model(16);
        let n_patches = 4;
        let d = m.d_model();
        let x1 = random_vec(n_patches * d, 17);
        let x2 = random_vec(n_patches * d, 18);
        let e1 = m
            .encode_student(&x1, n_patches)
            .expect("encode_student should succeed");
        let e2 = m
            .encode_student(&x2, n_patches)
            .expect("encode_student should succeed");
        let diff: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "different inputs must produce different encodings, diff={diff}"
        );
    }

    #[test]
    fn n_layers_0_error() {
        let mut rng = LcgRng::new(19);
        let result = Data2VecModel::new(
            Data2VecModelConfig {
                n_layers: 0,
                ..Data2VecModelConfig::default()
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_layers=0 must return Err");
    }

    #[test]
    fn d_model_accessor() {
        let m = make_model(20);
        assert_eq!(m.d_model(), 64);
    }
}
