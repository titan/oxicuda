//! TinyBERT multi-stage knowledge distillation (Jiao et al. 2020 EMNLP).
//!
//! Distills a large BERT teacher into a smaller student via four distillation signals:
//! (1) embedding transfer, (2) attention matrix MSE, (3) hidden state MSE with learned
//! projection, and (4) prediction (logit) distillation.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Kaiming-uniform init: U(-√(6/fan_in), +√(6/fan_in)).
fn kaiming_uniform(n: usize, fan_in: usize, rng: &mut LcgRng) -> Vec<f32> {
    let bound = if fan_in == 0 {
        1.0_f32
    } else {
        (6.0_f32 / fan_in as f32).sqrt()
    };
    (0..n)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// Row-wise softmax applied in-place to a matrix stored flat `[rows × cols]`.
fn row_softmax_inplace(mat: &mut [f32], cols: usize) {
    if cols == 0 {
        return;
    }
    let rows = mat.len() / cols;
    for r in 0..rows {
        let row = &mut mat[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let sum_safe = sum.max(1e-30);
        for v in row.iter_mut() {
            *v /= sum_safe;
        }
    }
}

/// Two-layer MLP with ReLU for projection.
///
/// W1: `hidden × in_dim`, b1: `hidden`, W2: `out_dim × hidden`, b2: `out_dim`.
/// `h = relu(W1 @ x + b1)`, `out = W2 @ h + b2`.
pub fn mlp2_forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Vec<f32> {
    // Hidden layer: h[i] = relu(sum_j w1[i*in_dim+j]*x[j] + b1[i])
    let mut h = vec![0.0_f32; hidden];
    for i in 0..hidden {
        let mut acc = b1[i];
        for j in 0..in_dim {
            acc += w1[i * in_dim + j] * x[j];
        }
        h[i] = acc.max(0.0); // ReLU
    }
    // Output layer: out[i] = sum_j w2[i*hidden+j]*h[j] + b2[i]
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b2[i];
        for j in 0..hidden {
            acc += w2[i * hidden + j] * h[j];
        }
        out[i] = acc;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types and functions
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for TinyBERT-style multi-stage distillation.
#[derive(Debug, Clone)]
pub struct TinyBertConfig {
    /// Number of student transformer layers (L).
    pub n_student_layers: usize,
    /// Number of teacher transformer layers (M ≥ L).
    pub n_teacher_layers: usize,
    /// Student hidden dimension d_s.
    pub d_student: usize,
    /// Teacher hidden dimension d_t.
    pub d_teacher: usize,
    /// KD temperature for prediction distillation (default 1.0).
    pub temperature: f32,
}

/// Learned linear projection W: `d_student × d_teacher` for hidden/embedding transfer.
///
/// W is stored flat in row-major order indexed as `W[d_s, d_t]`.
#[derive(Debug, Clone)]
pub struct TinyBertProjection {
    /// Flat weight matrix of shape `d_student × d_teacher`.
    pub w: Vec<f32>,
    /// Student hidden dimension.
    pub d_student: usize,
    /// Teacher hidden dimension.
    pub d_teacher: usize,
}

impl TinyBertProjection {
    /// Create a new projection matrix with Kaiming-uniform initialization.
    pub fn new(d_student: usize, d_teacher: usize, rng: &mut LcgRng) -> DistillResult<Self> {
        if d_student == 0 || d_teacher == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "TinyBertProjection: d_student and d_teacher must be > 0".into(),
            });
        }
        let w = kaiming_uniform(d_student * d_teacher, d_student, rng);
        Ok(Self {
            w,
            d_student,
            d_teacher,
        })
    }

    /// Project student hidden states: `out = h_S @ W` (h_S is `[seq_len × d_student]`).
    ///
    /// Returns `[seq_len × d_teacher]`.
    pub fn project(&self, h_student: &[f32], seq_len: usize) -> DistillResult<Vec<f32>> {
        let expected = seq_len * self.d_student;
        if h_student.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if h_student.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: h_student.len(),
            });
        }
        // out[s, dt] = sum_{ds} h_S[s, ds] * W[ds, dt]
        let mut out = vec![0.0_f32; seq_len * self.d_teacher];
        for s in 0..seq_len {
            for dt in 0..self.d_teacher {
                let mut acc = 0.0_f32;
                for ds in 0..self.d_student {
                    acc += h_student[s * self.d_student + ds] * self.w[ds * self.d_teacher + dt];
                }
                out[s * self.d_teacher + dt] = acc;
            }
        }
        Ok(out)
    }
}

/// Attention transfer MSE loss.
///
/// Both `t_attn` and `s_attn` must be `[seq_len × seq_len]` flat row-major matrices.
/// If `use_softmax` is true, row-wise softmax is applied before computing the MSE.
pub fn attention_mse(
    t_attn: &[f32],
    s_attn: &[f32],
    seq_len: usize,
    use_softmax: bool,
) -> DistillResult<f32> {
    if t_attn.is_empty() || s_attn.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let expected = seq_len * seq_len;
    if t_attn.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: t_attn.len(),
        });
    }
    if s_attn.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: s_attn.len(),
        });
    }

    let (t_proc, s_proc): (Vec<f32>, Vec<f32>) = if use_softmax {
        let mut t_copy = t_attn.to_vec();
        let mut s_copy = s_attn.to_vec();
        row_softmax_inplace(&mut t_copy, seq_len);
        row_softmax_inplace(&mut s_copy, seq_len);
        (t_copy, s_copy)
    } else {
        (t_attn.to_vec(), s_attn.to_vec())
    };

    let n = expected as f32;
    let mse: f32 = t_proc
        .iter()
        .zip(s_proc.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n;
    Ok(mse)
}

/// Hidden state MSE loss after projection: `MSE(proj(h_S), h_T)`.
///
/// `h_T` is `[seq_len × d_teacher]`, `h_S` is `[seq_len × d_student]`.
pub fn hidden_mse(
    t_hidden: &[f32],
    s_hidden: &[f32],
    seq_len: usize,
    proj: &TinyBertProjection,
) -> DistillResult<f32> {
    if t_hidden.is_empty() || s_hidden.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let expected_t = seq_len * proj.d_teacher;
    if t_hidden.len() != expected_t {
        return Err(DistillError::DimensionMismatch {
            expected: expected_t,
            got: t_hidden.len(),
        });
    }
    let projected = proj.project(s_hidden, seq_len)?;
    if projected.len() != t_hidden.len() {
        return Err(DistillError::DimensionMismatch {
            expected: projected.len(),
            got: t_hidden.len(),
        });
    }
    let n = projected.len() as f32;
    let mse: f32 = projected
        .iter()
        .zip(t_hidden.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n;
    Ok(mse)
}

/// Embedding MSE loss after projection (structurally identical to [`hidden_mse`]).
///
/// `t_emb` is `[seq_len × d_teacher]`, `s_emb` is `[seq_len × d_student]`.
pub fn embedding_mse(
    t_emb: &[f32],
    s_emb: &[f32],
    seq_len: usize,
    proj: &TinyBertProjection,
) -> DistillResult<f32> {
    hidden_mse(t_emb, s_emb, seq_len, proj)
}

/// Prediction distillation loss (Hinton KD) at temperature `T`.
///
/// `loss = T² · KL(teacher_soft ‖ student_soft)` where `soft = softmax(logits / T)`.
pub fn prediction_loss(t_logits: &[f32], s_logits: &[f32], temperature: f32) -> DistillResult<f32> {
    if t_logits.is_empty() || s_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if t_logits.len() != s_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: t_logits.len(),
            got: s_logits.len(),
        });
    }
    let t_safe = temperature.max(1e-12);
    let p_t = softmax_with_temp(t_logits, t_safe);
    let p_s = softmax_with_temp(s_logits, t_safe);
    Ok(t_safe * t_safe * kl_divergence(&p_t, &p_s))
}

// ─────────────────────────────────────────────────────────────────────────────
// General distillation loss aggregator
// ─────────────────────────────────────────────────────────────────────────────

/// Combined TinyBERT general-distillation loss (no prediction stage):
///
/// `L = embedding_mse + (1/L) * Σ_l (attention_mse_l + hidden_mse_l)`
///
/// Layer mapping: evenly spaced teacher layers are selected for each student layer.
/// `teacher_layer[i] = round(i * (n_teacher-1) / (n_student-1))` for `i` in `0..n_student`.
pub struct TinyBertGeneralLoss {
    /// Projection for the embedding layer.
    pub embed_proj: TinyBertProjection,
    /// One projection per student layer.
    pub hidden_projs: Vec<TinyBertProjection>,
}

impl TinyBertGeneralLoss {
    /// Create projections for embedding + all student layers.
    pub fn new(cfg: &TinyBertConfig, rng: &mut LcgRng) -> DistillResult<Self> {
        if cfg.n_student_layers == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_student_layers must be > 0".into(),
            });
        }
        if cfg.n_teacher_layers < cfg.n_student_layers {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "n_teacher_layers ({}) must be >= n_student_layers ({})",
                    cfg.n_teacher_layers, cfg.n_student_layers
                ),
            });
        }
        let embed_proj = TinyBertProjection::new(cfg.d_student, cfg.d_teacher, rng)?;
        let mut hidden_projs = Vec::with_capacity(cfg.n_student_layers);
        for _ in 0..cfg.n_student_layers {
            hidden_projs.push(TinyBertProjection::new(cfg.d_student, cfg.d_teacher, rng)?);
        }
        Ok(Self {
            embed_proj,
            hidden_projs,
        })
    }

    /// Map student layer index `i` → teacher layer index using evenly-spaced selection.
    fn teacher_idx(i: usize, n_student: usize, n_teacher: usize) -> usize {
        if n_student <= 1 {
            return 0;
        }
        let mapped = (i as f32 * (n_teacher - 1) as f32 / (n_student - 1) as f32).round() as usize;
        mapped.min(n_teacher - 1)
    }

    /// Compute the total general-stage TinyBERT loss.
    ///
    /// # Arguments
    /// * `t_emb` — `[seq_len × d_teacher]`
    /// * `s_emb` — `[seq_len × d_student]`
    /// * `t_attns` — `[n_teacher_layers]` each `[seq_len × seq_len]`
    /// * `s_attns` — `[n_student_layers]` each `[seq_len × seq_len]`
    /// * `t_hiddens` — `[n_teacher_layers]` each `[seq_len × d_teacher]`
    /// * `s_hiddens` — `[n_student_layers]` each `[seq_len × d_student]`
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &self,
        t_emb: &[f32],
        s_emb: &[f32],
        t_attns: &[Vec<f32>],
        s_attns: &[Vec<f32>],
        t_hiddens: &[Vec<f32>],
        s_hiddens: &[Vec<f32>],
        seq_len: usize,
        cfg: &TinyBertConfig,
    ) -> DistillResult<f32> {
        if s_attns.len() != cfg.n_student_layers {
            return Err(DistillError::DimensionMismatch {
                expected: cfg.n_student_layers,
                got: s_attns.len(),
            });
        }
        if t_attns.len() < cfg.n_teacher_layers {
            return Err(DistillError::DimensionMismatch {
                expected: cfg.n_teacher_layers,
                got: t_attns.len(),
            });
        }
        if s_hiddens.len() != cfg.n_student_layers {
            return Err(DistillError::DimensionMismatch {
                expected: cfg.n_student_layers,
                got: s_hiddens.len(),
            });
        }
        if t_hiddens.len() < cfg.n_teacher_layers {
            return Err(DistillError::DimensionMismatch {
                expected: cfg.n_teacher_layers,
                got: t_hiddens.len(),
            });
        }

        // Embedding loss
        let emb_loss = embedding_mse(t_emb, s_emb, seq_len, &self.embed_proj)?;

        // Per-layer losses
        let n_s = cfg.n_student_layers;
        let n_t = cfg.n_teacher_layers;
        let mut layer_loss_sum = 0.0_f32;
        for i in 0..n_s {
            let t_idx = Self::teacher_idx(i, n_s, n_t);
            let attn_l = attention_mse(&t_attns[t_idx], &s_attns[i], seq_len, true)?;
            let hid_l = hidden_mse(
                &t_hiddens[t_idx],
                &s_hiddens[i],
                seq_len,
                &self.hidden_projs[i],
            )?;
            layer_loss_sum += attn_l + hid_l;
        }

        Ok(emb_loss + layer_loss_sum / n_s as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── 1. attention_mse_identical ──────────────────────────────────────────

    #[test]
    fn attention_mse_identical() {
        let attn: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect(); // 3×3
        let loss = attention_mse(&attn, &attn, 3, false).expect("attention_mse should succeed");
        assert!(loss.abs() < 1e-10, "identical attns → MSE ≈ 0, got {loss}");
    }

    // ── 2. attention_mse_nonneg ─────────────────────────────────────────────

    #[test]
    fn attention_mse_nonneg() {
        let t: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let s: Vec<f32> = vec![0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0];
        let loss = attention_mse(&t, &s, 3, false).expect("attention_mse should succeed");
        assert!(loss >= 0.0, "MSE must be non-negative");
    }

    // ── 3. attention_mse_with_softmax ───────────────────────────────────────

    #[test]
    fn attention_mse_with_softmax() {
        let t: Vec<f32> = (0..16).map(|i| i as f32).collect(); // 4×4
        let s: Vec<f32> = (0..16).map(|i| (15 - i) as f32).collect();
        let result = attention_mse(&t, &s, 4, true);
        assert!(result.is_ok(), "use_softmax=true should not error");
        assert!(result.expect("result should be present").is_finite());
    }

    // ── 4. attention_mse_no_softmax ─────────────────────────────────────────

    #[test]
    fn attention_mse_no_softmax() {
        let t: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
        let s: Vec<f32> = (0..16).map(|i| i as f32 * 0.04).collect();
        let result = attention_mse(&t, &s, 4, false);
        assert!(result.is_ok(), "use_softmax=false should not error");
        assert!(result.expect("result should be present").is_finite());
    }

    // ── 5. hidden_mse_identical (W=I when d_s==d_t special case via random W) ─

    #[test]
    fn hidden_mse_identical() {
        // Build a projection with identity-like W (d_s == d_t == 4, W = I)
        let d = 4usize;
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0; // identity
        }
        let proj = TinyBertProjection {
            w,
            d_student: d,
            d_teacher: d,
        };
        let h: Vec<f32> = (0..8).map(|i| i as f32).collect(); // seq_len=2, d=4
        let loss = hidden_mse(&h, &h, 2, &proj).expect("hidden_mse should succeed");
        assert!(loss.abs() < 1e-10, "identity proj + same input → MSE ≈ 0");
    }

    // ── 6. hidden_mse_nonneg ────────────────────────────────────────────────

    #[test]
    fn hidden_mse_nonneg() {
        let mut rng = make_rng();
        let proj = TinyBertProjection::new(8, 16, &mut rng).expect("new should succeed");
        let t: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // seq_len=2, d_t=16
        let s: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect(); // seq_len=2, d_s=8
        let loss = hidden_mse(&t, &s, 2, &proj).expect("hidden_mse should succeed");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    // ── 7. hidden_mse_shape_check ───────────────────────────────────────────

    #[test]
    fn hidden_mse_shape_check() {
        let mut rng = make_rng();
        let proj = TinyBertProjection::new(8, 16, &mut rng).expect("new should succeed");
        // seq_len=4, d_s=8, d_t=16
        let t: Vec<f32> = vec![0.0_f32; 4 * 16];
        let s: Vec<f32> = vec![0.0_f32; 4 * 8];
        let result = hidden_mse(&t, &s, 4, &proj);
        assert!(result.is_ok(), "seq_len=4, d_s=8, d_t=16 should work");
        assert!(result.expect("result should be present").is_finite());
    }

    // ── 8. embedding_mse_runs ───────────────────────────────────────────────

    #[test]
    fn embedding_mse_runs() {
        let mut rng = make_rng();
        let proj = TinyBertProjection::new(4, 8, &mut rng).expect("new should succeed");
        let t = vec![1.0_f32; 16]; // seq_len=2, d_t=8
        let s = vec![0.5_f32; 8]; // seq_len=2, d_s=4
        let result = embedding_mse(&t, &s, 2, &proj);
        assert!(result.is_ok());
        assert!(result.expect("result should be present").is_finite());
    }

    // ── 9. prediction_loss_identical ────────────────────────────────────────

    #[test]
    fn prediction_loss_identical() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = prediction_loss(&logits, &logits, 1.0).expect("prediction_loss should succeed");
        assert!(
            loss.abs() < 1e-5,
            "identical logits → pred loss ≈ 0, got {loss}"
        );
    }

    // ── 10. prediction_loss_nonneg ───────────────────────────────────────────

    #[test]
    fn prediction_loss_nonneg() {
        let t = vec![2.0_f32, 1.0, 0.0];
        let s = vec![0.5_f32, 1.5, 1.0];
        let loss = prediction_loss(&t, &s, 2.0).expect("prediction_loss should succeed");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    // ── 11. prediction_loss_temperature_scale ───────────────────────────────

    #[test]
    fn prediction_loss_temperature_scale() {
        let t = vec![3.0_f32, 1.0, 0.0];
        let s = vec![1.0_f32, 2.0, 0.5];
        let loss_t1 = prediction_loss(&t, &s, 1.0).expect("prediction_loss should succeed");
        let loss_t2 = prediction_loss(&t, &s, 2.0).expect("prediction_loss should succeed");
        // Losses at different temperatures must differ
        assert!(
            (loss_t1 - loss_t2).abs() > 1e-6,
            "T=1 loss ({loss_t1}) should differ from T=2 loss ({loss_t2})"
        );
    }

    // ── 12. projection_output_length ────────────────────────────────────────

    #[test]
    fn projection_output_length() {
        let mut rng = make_rng();
        let d_s = 6usize;
        let d_t = 12usize;
        let seq_len = 5usize;
        let proj = TinyBertProjection::new(d_s, d_t, &mut rng).expect("new should succeed");
        let h_s: Vec<f32> = (0..seq_len * d_s).map(|i| i as f32 * 0.01).collect();
        let out = proj.project(&h_s, seq_len).expect("project should succeed");
        assert_eq!(out.len(), seq_len * d_t);
    }

    // ── 13. general_loss_positive ────────────────────────────────────────────

    #[test]
    fn general_loss_positive() {
        let mut rng = make_rng();
        let cfg = TinyBertConfig {
            n_student_layers: 3,
            n_teacher_layers: 6,
            d_student: 4,
            d_teacher: 8,
            temperature: 1.0,
        };
        let agg = TinyBertGeneralLoss::new(&cfg, &mut rng).expect("new should succeed");
        let seq_len = 4usize;

        let t_emb = vec![0.1_f32; seq_len * cfg.d_teacher];
        let s_emb = vec![0.2_f32; seq_len * cfg.d_student];
        let t_attns: Vec<Vec<f32>> = (0..cfg.n_teacher_layers)
            .map(|_| vec![0.1_f32; seq_len * seq_len])
            .collect();
        let s_attns: Vec<Vec<f32>> = (0..cfg.n_student_layers)
            .map(|_| vec![0.2_f32; seq_len * seq_len])
            .collect();
        let t_hiddens: Vec<Vec<f32>> = (0..cfg.n_teacher_layers)
            .map(|_| vec![0.1_f32; seq_len * cfg.d_teacher])
            .collect();
        let s_hiddens: Vec<Vec<f32>> = (0..cfg.n_student_layers)
            .map(|_| vec![0.2_f32; seq_len * cfg.d_student])
            .collect();

        let loss = agg
            .compute(
                &t_emb, &s_emb, &t_attns, &s_attns, &t_hiddens, &s_hiddens, seq_len, &cfg,
            )
            .expect("value should be present");
        assert!(loss.is_finite(), "general loss must be finite");
        assert!(loss >= 0.0, "general loss must be non-negative");
    }

    // ── 14. general_loss_embedding_only ─────────────────────────────────────

    #[test]
    fn general_loss_embedding_only() {
        let mut rng = make_rng();
        let cfg = TinyBertConfig {
            n_student_layers: 1,
            n_teacher_layers: 1,
            d_student: 4,
            d_teacher: 8,
            temperature: 1.0,
        };
        let agg = TinyBertGeneralLoss::new(&cfg, &mut rng).expect("new should succeed");
        let seq_len = 2usize;
        let t_emb = vec![0.5_f32; seq_len * cfg.d_teacher];
        let s_emb = vec![0.3_f32; seq_len * cfg.d_student];
        let t_attns = vec![vec![0.1_f32; seq_len * seq_len]];
        let s_attns = vec![vec![0.2_f32; seq_len * seq_len]];
        let t_hiddens = vec![vec![0.1_f32; seq_len * cfg.d_teacher]];
        let s_hiddens = vec![vec![0.2_f32; seq_len * cfg.d_student]];
        let result = agg.compute(
            &t_emb, &s_emb, &t_attns, &s_attns, &t_hiddens, &s_hiddens, seq_len, &cfg,
        );
        assert!(result.is_ok(), "n_s=n_t=1 should work: {:?}", result.err());
    }

    // ── 15. attention_mse_dim_mismatch ───────────────────────────────────────

    #[test]
    fn attention_mse_dim_mismatch() {
        let t = vec![0.0_f32; 9]; // expects 3×3=9
        let s = vec![0.0_f32; 16]; // wrong: 4×4
        let result = attention_mse(&t, &s, 3, false);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "mismatched sizes should yield DimensionMismatch"
        );
    }

    // ── 16. hidden_mse_dim_mismatch ──────────────────────────────────────────

    #[test]
    fn hidden_mse_dim_mismatch() {
        let mut rng = make_rng();
        let proj = TinyBertProjection::new(4, 8, &mut rng).expect("new should succeed");
        let t = vec![0.0_f32; 8]; // seq_len=1, d_t=8
        let s = vec![0.0_f32; 5]; // wrong: 5 instead of 4
        let result = hidden_mse(&t, &s, 1, &proj);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "wrong s_hidden length should yield DimensionMismatch"
        );
    }

    // ── 17. prediction_loss_dim_mismatch ────────────────────────────────────

    #[test]
    fn prediction_loss_dim_mismatch() {
        let t = vec![1.0_f32, 2.0, 3.0];
        let s = vec![1.0_f32, 2.0]; // different length
        let result = prediction_loss(&t, &s, 1.0);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "different logit lengths should yield DimensionMismatch"
        );
    }

    // ── 18. empty_input_err ──────────────────────────────────────────────────

    #[test]
    fn empty_input_err() {
        let result = attention_mse(&[], &[0.0_f32; 9], 3, false);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty t_attn should yield EmptyInput"
        );
    }
}
