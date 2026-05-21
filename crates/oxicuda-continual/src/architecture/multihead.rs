//! Multi-Head Class-Incremental Classifier.
//!
//! One shared backbone + a dedicated linear output head per task. When a new
//! task arrives a fresh Xavier-initialised head is appended; old heads are
//! frozen. Backpropagation flows through the backbone and only the current
//! task's head, giving gradient isolation across tasks.
//!
//! Task-aware prediction: forward the correct head.
//! Task-agnostic prediction: forward all heads, pick maximum-softmax-confidence
//! (task_id, class_id) pair.
//!
//! Architecture
//! ```text
//! x (input_dim)
//!   → [W1, b1] → ReLU → hidden_dim
//!   → [W2, b2] → ReLU → rep_dim  (= hidden_dim / 2)
//!   → [head_k_W, head_k_b] → logits (n_classes_k)
//! ```

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the multi-head class-incremental classifier.
#[derive(Debug, Clone)]
pub struct MultiHeadConfig {
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// Backbone hidden layer width.
    pub hidden_dim: usize,
    /// Initial capacity hint for the heads vector (pre-allocation only).
    pub heads_init_capacity: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
}

impl Default for MultiHeadConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            heads_init_capacity: 4,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

// ─── Task head ───────────────────────────────────────────────────────────────

/// Linear output head for a single task.
#[derive(Debug, Clone)]
pub struct TaskHead {
    /// Weight matrix, shape `[n_classes × rep_dim]`, row-major.
    pub weights: Vec<f64>,
    /// Bias vector, length `n_classes`.
    pub bias: Vec<f64>,
    /// Number of output classes for this task.
    pub n_classes: usize,
    /// Opaque task identifier supplied by the caller.
    pub task_id: usize,
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Multi-head model state.
#[derive(Debug, Clone)]
pub struct MultiHeadState {
    // Backbone layer 1: [hidden_dim × input_dim]
    /// Backbone first-layer weight matrix, row-major.
    pub backbone_w1: Vec<f64>,
    /// Backbone first-layer bias, length `hidden_dim`.
    pub backbone_b1: Vec<f64>,
    // Backbone layer 2: [rep_dim × hidden_dim]
    /// Backbone second-layer weight matrix, row-major.
    pub backbone_w2: Vec<f64>,
    /// Backbone second-layer bias, length `rep_dim`.
    pub backbone_b2: Vec<f64>,
    /// One head per registered task.
    pub heads: Vec<TaskHead>,
    /// Number of registered tasks.
    pub n_tasks: usize,
    /// Representation dimensionality (`hidden_dim / 2`).
    pub rep_dim: usize,
    // Cached config fields.
    pub(crate) input_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) lr: f64,
    pub(crate) n_epochs: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Xavier uniform scale: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Fill a mutable slice with Xavier-uniform values in `[-scale, +scale]`.
fn xavier_init(buf: &mut [f64], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = xavier_scale(fan_in, fan_out);
    for v in buf.iter_mut() {
        let u = rng.next_f32() as f64 * 2.0 - 1.0;
        *v = u * scale;
    }
}

/// ReLU activation.
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Linear forward: `out = W * x + b`, W row-major `[out_dim × in_dim]`.
fn linear_forward(w: &[f64], b: &[f64], x: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for i in 0..out_dim {
        let mut s = b[i];
        for j in 0..in_dim {
            s += w[i * in_dim + j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Backbone forward pass: returns `(h1, rep)`.
///
/// - `h1`: post-ReLU hidden layer, length `hidden_dim`
/// - `rep`: post-ReLU second layer, length `rep_dim`
fn backbone_forward(state: &MultiHeadState, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let h1_pre = linear_forward(
        &state.backbone_w1,
        &state.backbone_b1,
        x,
        state.input_dim,
        state.hidden_dim,
    );
    let h1: Vec<f64> = h1_pre.iter().map(|&v| relu(v)).collect();
    let rep_pre = linear_forward(
        &state.backbone_w2,
        &state.backbone_b2,
        &h1,
        state.hidden_dim,
        state.rep_dim,
    );
    let rep: Vec<f64> = rep_pre.iter().map(|&v| relu(v)).collect();
    (h1, rep)
}

/// Softmax over a logit slice; returns probabilities.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let n = logits.len();
        return vec![1.0 / n as f64; n];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Cross-entropy loss for a single sample.
fn cross_entropy_single(logits: &[f64], y: usize) -> f64 {
    let probs = softmax(logits);
    let p = probs.get(y).copied().unwrap_or(1e-12).max(1e-12);
    -p.ln()
}

/// Maximum softmax confidence for a logit vector.
fn max_confidence(logits: &[f64]) -> f64 {
    let probs = softmax(logits);
    probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Find the task head by task_id, returning its index in `state.heads`.
fn find_head_idx(state: &MultiHeadState, task_id: usize) -> ContinualResult<usize> {
    state.heads.iter().position(|h| h.task_id == task_id).ok_or(
        ContinualError::TaskIndexOutOfRange {
            index: task_id,
            n_tasks: state.n_tasks,
        },
    )
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Initialise a new multi-head state (no heads yet).
pub fn multihead_new(cfg: &MultiHeadConfig, seed: u64) -> ContinualResult<MultiHeadState> {
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if cfg.lr <= 0.0 || !cfg.lr.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "multihead_new: lr must be positive and finite",
        });
    }
    if cfg.n_epochs == 0 {
        return Err(ContinualError::NanEncountered {
            location: "multihead_new: n_epochs must be >= 1",
        });
    }

    let rep_dim = (cfg.hidden_dim / 2).max(1);
    let mut rng = LcgRng::new(seed);

    let mut backbone_w1 = vec![0.0_f64; cfg.hidden_dim * cfg.input_dim];
    xavier_init(&mut backbone_w1, cfg.input_dim, cfg.hidden_dim, &mut rng);
    let backbone_b1 = vec![0.0_f64; cfg.hidden_dim];

    let mut backbone_w2 = vec![0.0_f64; rep_dim * cfg.hidden_dim];
    xavier_init(&mut backbone_w2, cfg.hidden_dim, rep_dim, &mut rng);
    let backbone_b2 = vec![0.0_f64; rep_dim];

    Ok(MultiHeadState {
        backbone_w1,
        backbone_b1,
        backbone_w2,
        backbone_b2,
        heads: Vec::with_capacity(cfg.heads_init_capacity),
        n_tasks: 0,
        rep_dim,
        input_dim: cfg.input_dim,
        hidden_dim: cfg.hidden_dim,
        lr: cfg.lr,
        n_epochs: cfg.n_epochs,
    })
}

/// Append a new Xavier-initialised output head for task `task_id`.
pub fn multihead_add_task(
    state: &mut MultiHeadState,
    n_classes: usize,
    task_id: usize,
    seed: u64,
) -> ContinualResult<()> {
    if n_classes == 0 {
        return Err(ContinualError::EmptyInput);
    }

    let mut rng = LcgRng::new(seed);
    let mut weights = vec![0.0_f64; n_classes * state.rep_dim];
    xavier_init(&mut weights, state.rep_dim, n_classes, &mut rng);
    let bias = vec![0.0_f64; n_classes];

    state.heads.push(TaskHead {
        weights,
        bias,
        n_classes,
        task_id,
    });
    state.n_tasks += 1;
    Ok(())
}

/// Train on task `task_id`.
///
/// Gradient flows through both the backbone and the current task's head.
/// All other heads are frozen (their parameters never touched).
pub fn multihead_fit_task(
    state: &mut MultiHeadState,
    x: &[f64],
    y: &[usize],
    n: usize,
    task_id: usize,
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if n == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if x.len() != n * state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: n * state.input_dim,
            got: x.len(),
        });
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }

    let head_idx = find_head_idx(state, task_id)?;

    let mut indices: Vec<usize> = (0..n).collect();
    let lr = state.lr;
    let n_epochs = state.n_epochs;
    let in_dim = state.input_dim;
    let hidden_dim = state.hidden_dim;
    let rep_dim = state.rep_dim;

    let mut final_loss = 0.0_f64;

    for epoch in 0..n_epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0_f64;

        for &s in &indices {
            let x_s = &x[s * in_dim..(s + 1) * in_dim];
            let y_s = y[s];

            // ── Forward ──────────────────────────────────────────────────────
            // Backbone layer 1
            let h1_pre: Vec<f64> = {
                let w = &state.backbone_w1;
                let b = &state.backbone_b1;
                (0..hidden_dim)
                    .map(|i| b[i] + (0..in_dim).map(|j| w[i * in_dim + j] * x_s[j]).sum::<f64>())
                    .collect()
            };
            let h1: Vec<f64> = h1_pre.iter().map(|&v| relu(v)).collect();

            // Backbone layer 2
            let rep_pre: Vec<f64> = {
                let w = &state.backbone_w2;
                let b = &state.backbone_b2;
                (0..rep_dim)
                    .map(|i| {
                        b[i] + (0..hidden_dim)
                            .map(|j| w[i * hidden_dim + j] * h1[j])
                            .sum::<f64>()
                    })
                    .collect()
            };
            let rep: Vec<f64> = rep_pre.iter().map(|&v| relu(v)).collect();

            // Current task head
            let n_cls = state.heads[head_idx].n_classes;
            let logits: Vec<f64> = {
                let w = &state.heads[head_idx].weights;
                let b = &state.heads[head_idx].bias;
                (0..n_cls)
                    .map(|i| {
                        b[i] + (0..rep_dim)
                            .map(|j| w[i * rep_dim + j] * rep[j])
                            .sum::<f64>()
                    })
                    .collect()
            };

            epoch_loss += cross_entropy_single(&logits, y_s);

            // ── Backward ─────────────────────────────────────────────────────
            let probs = softmax(&logits);
            let mut d_logits = probs;
            d_logits[y_s] -= 1.0;

            // d_rep = head_W^T * d_logits  (must use PRE-UPDATE head weights)
            let d_rep: Vec<f64> = (0..rep_dim)
                .map(|j| {
                    let w = &state.heads[head_idx].weights;
                    d_logits
                        .iter()
                        .enumerate()
                        .map(|(i, &dl)| w[i * rep_dim + j] * dl)
                        .sum()
                })
                .collect();

            // Head gradients (update AFTER d_rep is computed from pre-update weights)
            {
                let head = &mut state.heads[head_idx];
                for (i, (&dl, hb)) in d_logits.iter().zip(head.bias.iter_mut()).enumerate() {
                    *hb -= lr * dl;
                    for (hw, &rv) in head.weights[i * rep_dim..(i + 1) * rep_dim]
                        .iter_mut()
                        .zip(rep.iter())
                    {
                        *hw -= lr * dl * rv;
                    }
                }
            }

            // Through ReLU of layer 2
            let d_rep_pre: Vec<f64> = rep_pre
                .iter()
                .zip(d_rep.iter())
                .map(|(&r, &dr)| if r > 0.0 { dr } else { 0.0 })
                .collect();

            // Backbone layer 2 gradients
            for (i, (&drp, bb2)) in d_rep_pre
                .iter()
                .zip(state.backbone_b2.iter_mut())
                .enumerate()
            {
                *bb2 -= lr * drp;
                for (bw2, &hv) in state.backbone_w2[i * hidden_dim..(i + 1) * hidden_dim]
                    .iter_mut()
                    .zip(h1.iter())
                {
                    *bw2 -= lr * drp * hv;
                }
            }

            // d_h1 = backbone_w2^T * d_rep_pre
            let d_h1: Vec<f64> = (0..hidden_dim)
                .map(|j| {
                    d_rep_pre
                        .iter()
                        .enumerate()
                        .map(|(i, &dr)| state.backbone_w2[i * hidden_dim + j] * dr)
                        .sum()
                })
                .collect();

            // Through ReLU of layer 1
            let d_h1_pre: Vec<f64> = h1_pre
                .iter()
                .zip(d_h1.iter())
                .map(|(&h, &dh)| if h > 0.0 { dh } else { 0.0 })
                .collect();

            // Backbone layer 1 gradients
            for (i, (&dhp, bb1)) in d_h1_pre
                .iter()
                .zip(state.backbone_b1.iter_mut())
                .enumerate()
            {
                *bb1 -= lr * dhp;
                for (bw1, &xv) in state.backbone_w1[i * in_dim..(i + 1) * in_dim]
                    .iter_mut()
                    .zip(x_s.iter())
                {
                    *bw1 -= lr * dhp * xv;
                }
            }
        }

        final_loss = epoch_loss / n as f64;
        let _ = epoch; // suppress unused warning
    }

    if !final_loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "multihead_fit_task",
        });
    }
    Ok(final_loss)
}

/// Task-aware prediction: use the head for `task_id`.
pub fn multihead_predict(
    state: &MultiHeadState,
    x: &[f64],
    task_id: usize,
) -> ContinualResult<usize> {
    if x.len() != state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: state.input_dim,
            got: x.len(),
        });
    }

    let head_idx = find_head_idx(state, task_id)?;
    let (_, rep) = backbone_forward(state, x);
    let head = &state.heads[head_idx];
    let logits = linear_forward(
        &head.weights,
        &head.bias,
        &rep,
        state.rep_dim,
        head.n_classes,
    );

    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(pred)
}

/// Task-agnostic prediction: forward all heads, return `(task_id, class_id)` with
/// maximum softmax confidence.
pub fn multihead_predict_unknown_task(
    state: &MultiHeadState,
    x: &[f64],
) -> ContinualResult<(usize, usize)> {
    if state.heads.is_empty() {
        return Err(ContinualError::NoTasksInStream);
    }
    if x.len() != state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: state.input_dim,
            got: x.len(),
        });
    }

    let (_, rep) = backbone_forward(state, x);

    let mut best_conf = f64::NEG_INFINITY;
    let mut best_task_id = 0;
    let mut best_class = 0;

    for head in &state.heads {
        let logits = linear_forward(
            &head.weights,
            &head.bias,
            &rep,
            state.rep_dim,
            head.n_classes,
        );
        let conf = max_confidence(&logits);
        if conf > best_conf {
            best_conf = conf;
            best_task_id = head.task_id;
            let probs = softmax(&logits);
            best_class = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    Ok((best_task_id, best_class))
}

/// Return the total number of registered tasks.
pub fn multihead_n_tasks(state: &MultiHeadState) -> usize {
    state.n_tasks
}

/// Return the number of classes for the given `task_id`.
pub fn multihead_n_classes_for_task(
    state: &MultiHeadState,
    task_id: usize,
) -> ContinualResult<usize> {
    let head_idx = find_head_idx(state, task_id)?;
    Ok(state.heads[head_idx].n_classes)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> MultiHeadConfig {
        MultiHeadConfig {
            input_dim: 4,
            hidden_dim: 8,
            heads_init_capacity: 4,
            lr: 0.05,
            n_epochs: 3,
        }
    }

    #[test]
    fn multihead_new_correct_dims() {
        let cfg = make_cfg();
        let state = multihead_new(&cfg, 1).unwrap();
        assert_eq!(state.input_dim, 4);
        assert_eq!(state.hidden_dim, 8);
        assert_eq!(state.rep_dim, 4); // hidden_dim / 2
        assert_eq!(state.n_tasks, 0);
        assert!(state.heads.is_empty());
        assert_eq!(state.backbone_w1.len(), 8 * 4);
        assert_eq!(state.backbone_w2.len(), 4 * 8);
    }

    #[test]
    fn multihead_add_task_appends_head() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 2).unwrap();
        multihead_add_task(&mut state, 3, 0, 10).unwrap();
        multihead_add_task(&mut state, 5, 1, 11).unwrap();
        assert_eq!(multihead_n_tasks(&state), 2);
        assert_eq!(multihead_n_classes_for_task(&state, 0).unwrap(), 3);
        assert_eq!(multihead_n_classes_for_task(&state, 1).unwrap(), 5);
    }

    #[test]
    fn multihead_predict_task_aware_valid_class() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 3).unwrap();
        multihead_add_task(&mut state, 4, 0, 20).unwrap();
        let x = vec![0.1_f64; 4];
        let pred = multihead_predict(&state, &x, 0).unwrap();
        assert!(pred < 4);
    }

    #[test]
    fn multihead_predict_unknown_task_returns_valid() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 4).unwrap();
        multihead_add_task(&mut state, 3, 10, 30).unwrap();
        multihead_add_task(&mut state, 2, 20, 31).unwrap();
        let x = vec![0.5_f64; 4];
        let (tid, cid) = multihead_predict_unknown_task(&state, &x).unwrap();
        assert!(tid == 10 || tid == 20);
        let n_cls = multihead_n_classes_for_task(&state, tid).unwrap();
        assert!(cid < n_cls);
    }

    #[test]
    fn multihead_fit_task_returns_finite_loss() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 5).unwrap();
        multihead_add_task(&mut state, 3, 0, 40).unwrap();
        let x: Vec<f64> = (0..4 * 6).map(|i| i as f64 * 0.05).collect();
        let y: Vec<usize> = vec![0, 1, 2, 0, 1, 2];
        let mut rng = LcgRng::new(50);
        let loss = multihead_fit_task(&mut state, &x, &y, 6, 0, &mut rng).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn multihead_fit_task_weights_change() {
        // Force backbone weights positive so all ReLU units fire and gradients
        // propagate through the full network.
        let cfg = MultiHeadConfig {
            input_dim: 4,
            hidden_dim: 8,
            heads_init_capacity: 4,
            lr: 0.5,
            n_epochs: 5,
        };
        let mut state = multihead_new(&cfg, 6).unwrap();
        // Force positive backbone weights so all hidden units fire (no dying ReLU).
        state.backbone_w1.iter_mut().for_each(|w| *w = 0.1);
        state.backbone_w2.iter_mut().for_each(|w| *w = 0.1);
        multihead_add_task(&mut state, 3, 0, 50).unwrap();
        let head_before = state.heads[0].weights.clone();
        let x: Vec<f64> = (0..4 * 5).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let y: Vec<usize> = vec![0, 1, 2, 0, 1];
        let mut rng = LcgRng::new(60);
        let backbone_b2_before = state.backbone_b2.clone();
        multihead_fit_task(&mut state, &x, &y, 5, 0, &mut rng).unwrap();
        // Check backbone layer-2 weights (closer to loss, easier to update)
        let backbone_changed = backbone_b2_before
            .iter()
            .zip(state.backbone_b2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        let head_changed = head_before
            .iter()
            .zip(state.heads[0].weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(backbone_changed, "backbone layer-2 bias should update");
        assert!(head_changed, "head weights should update");
    }

    #[test]
    fn multihead_other_heads_frozen_during_training() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 7).unwrap();
        multihead_add_task(&mut state, 2, 0, 60).unwrap();
        multihead_add_task(&mut state, 3, 1, 61).unwrap();
        let head1_before = state.heads[1].weights.clone();
        // Train on task 0
        let x: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.1).collect();
        let y = vec![0_usize, 1, 0, 1];
        let mut rng = LcgRng::new(70);
        multihead_fit_task(&mut state, &x, &y, 4, 0, &mut rng).unwrap();
        // Task 1 head must not have changed
        assert_eq!(state.heads[1].weights, head1_before, "task 1 head frozen");
    }

    #[test]
    fn multihead_n_classes_for_unknown_task_errors() {
        let cfg = make_cfg();
        let state = multihead_new(&cfg, 8).unwrap();
        let res = multihead_n_classes_for_task(&state, 99);
        assert!(res.is_err());
    }

    #[test]
    fn multihead_predict_on_empty_errors() {
        let cfg = make_cfg();
        let state = multihead_new(&cfg, 9).unwrap();
        let x = vec![0.0_f64; 4];
        let res = multihead_predict_unknown_task(&state, &x);
        assert!(res.is_err());
    }

    #[test]
    fn multihead_fit_task_wrong_n_errors() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 10).unwrap();
        multihead_add_task(&mut state, 3, 0, 70).unwrap();
        let x = vec![0.0_f64; 4 * 3];
        let y = vec![0_usize, 1]; // n=3 but y has 2
        let mut rng = LcgRng::new(80);
        let res = multihead_fit_task(&mut state, &x, &y, 3, 0, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn multihead_add_zero_classes_errors() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 11).unwrap();
        let res = multihead_add_task(&mut state, 0, 0, 90);
        assert!(res.is_err());
    }

    #[test]
    fn multihead_backbone_weights_xavier_range() {
        let cfg = make_cfg();
        let state = multihead_new(&cfg, 12).unwrap();
        let scale = xavier_scale(cfg.input_dim, cfg.hidden_dim);
        for &w in &state.backbone_w1 {
            assert!(
                w.abs() <= scale + 1e-9,
                "backbone_w1 out of xavier range: {w}"
            );
        }
    }

    #[test]
    fn multihead_predict_dim_mismatch_errors() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 13).unwrap();
        multihead_add_task(&mut state, 3, 0, 100).unwrap();
        let x = vec![0.0_f64; 10]; // wrong
        let res = multihead_predict(&state, &x, 0);
        assert!(res.is_err());
    }

    #[test]
    fn multihead_multiple_tasks_sequential_training() {
        let cfg = make_cfg();
        let mut state = multihead_new(&cfg, 14).unwrap();
        multihead_add_task(&mut state, 2, 0, 200).unwrap();
        multihead_add_task(&mut state, 3, 1, 201).unwrap();

        let mut rng = LcgRng::new(202);

        // Train task 0
        let x0: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.05).collect();
        let y0 = vec![0_usize, 1, 0, 1];
        let loss0 = multihead_fit_task(&mut state, &x0, &y0, 4, 0, &mut rng).unwrap();

        // Train task 1
        let x1: Vec<f64> = (0..4 * 6).map(|i| (i as f64 + 1.0) * 0.03).collect();
        let y1 = vec![0_usize, 1, 2, 0, 1, 2];
        let loss1 = multihead_fit_task(&mut state, &x1, &y1, 6, 1, &mut rng).unwrap();

        assert!(loss0.is_finite() && loss0 >= 0.0);
        assert!(loss1.is_finite() && loss1 >= 0.0);
        assert_eq!(multihead_n_tasks(&state), 2);
    }
}
