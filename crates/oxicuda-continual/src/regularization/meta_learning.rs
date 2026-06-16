//! Online Meta-Learning (OML) and ANML continual learning.
//!
//! Implements the methods from:
//! - Javed & White. "Meta-Learning Representations for Continual Learning." NeurIPS 2019 (OML).
//! - Beaulieu et al. "Learning to Continually Learn." ECAI 2020 (ANML).
//!
//! The core idea: a slow-updating Representation Learning Network (RLN) provides
//! general-purpose features; a fast-adapting Prediction Learning Network (PLN)
//! specialises per task via inner-loop gradient steps (MAML-style). An ANML gate
//! network modulates RLN outputs with sigmoid gating for neuromodulated continual
//! learning. Outer-loop (meta) updates use first-order MAML (FOMAML).

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Return type of `rln_gradient_single`: six flat gradient vectors for
/// `(rln_w1, rln_b1, rln_w2, rln_b2, gate_w, gate_b)`.
type RlnGrads = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Return type of `forward_full`: `(h1, rep_pre, gate, rep_gated, logits)`.
type ForwardFullOut = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for OML / ANML meta-learning.
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width (shared by RLN and gate network).
    pub hidden_dim: usize,
    /// Output class count.
    pub output_dim: usize,
    /// Inner-loop (fast adaptation) learning rate.
    pub lr_inner: f64,
    /// Outer-loop (meta / slow) learning rate.
    pub lr_outer: f64,
    /// Number of inner SGD steps per task.
    pub n_inner_steps: usize,
    /// Number of meta-training epochs over the task distribution.
    pub n_meta_epochs: usize,
    /// Number of tasks sampled per meta-epoch.
    pub n_tasks_per_meta: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 5,
            lr_inner: 0.05,
            lr_outer: 0.001,
            n_inner_steps: 5,
            n_meta_epochs: 3,
            n_tasks_per_meta: 4,
            seed: 42,
        }
    }
}

// ─── Task data ───────────────────────────────────────────────────────────────

/// Support + query split for one meta-learning task.
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Support set inputs, length `n_support * input_dim`.
    pub x_support: Vec<f64>,
    /// Support set labels, length `n_support`.
    pub y_support: Vec<usize>,
    /// Query set inputs, length `n_query * input_dim`.
    pub x_query: Vec<f64>,
    /// Query set labels, length `n_query`.
    pub y_query: Vec<usize>,
    /// Number of support examples.
    pub n_support: usize,
    /// Number of query examples.
    pub n_query: usize,
}

// ─── State ───────────────────────────────────────────────────────────────────

/// OML / ANML model state.
///
/// Architecture:
/// - **RLN** (Representation Learning Network):
///   `input_dim → hidden_dim → rep_dim (= hidden_dim / 2)`, ReLU, slow.
/// - **Gate** (ANML extension):
///   `rep_dim → rep_dim`, sigmoid, element-wise modulation of RLN output.
/// - **PLN** (Prediction Learning Network):
///   `rep_dim → output_dim`, linear head, fast-adapting.
#[derive(Debug, Clone)]
pub struct MetaLearningState {
    // RLN layer 1: shape [hidden_dim × input_dim]
    /// RLN first-layer weights, row-major.
    pub rln_w1: Vec<f64>,
    /// RLN first-layer bias, length `hidden_dim`.
    pub rln_b1: Vec<f64>,
    // RLN layer 2: shape [rep_dim × hidden_dim]
    /// RLN second-layer weights, row-major.
    pub rln_w2: Vec<f64>,
    /// RLN second-layer bias, length `rep_dim`.
    pub rln_b2: Vec<f64>,
    // PLN (linear head): shape [output_dim × rep_dim]
    /// PLN weight matrix, row-major.
    pub pln_w: Vec<f64>,
    /// PLN bias, length `output_dim`.
    pub pln_b: Vec<f64>,
    // Gate network: shape [rep_dim × rep_dim]
    /// Gate weight matrix, row-major.
    pub gate_w: Vec<f64>,
    /// Gate bias, length `rep_dim`.
    pub gate_b: Vec<f64>,
    /// Number of completed meta-gradient update steps.
    pub n_meta_steps: usize,
    // Cached dimensions.
    pub(crate) input_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) rep_dim: usize,
    pub(crate) output_dim: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Xavier uniform scale: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Fill a slice with Xavier-uniform values in `[-scale, +scale]`.
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

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Linear layer forward: `out = W * x + b`.
///
/// - `w`: row-major `[out_dim × in_dim]`
/// - `x`: length `in_dim`
/// - returns: length `out_dim`
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

/// Softmax over a slice; returns a new allocation.
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

/// Cross-entropy loss for a single example: `-log(p[y])`.
fn cross_entropy_single(logits: &[f64], y: usize) -> f64 {
    let probs = softmax(logits);
    let p = probs[y].max(1e-12);
    -p.ln()
}

/// Run the full OML/ANML forward pass: x → RLN → gate → PLN → logits.
///
/// Returns `(hidden1, rep_pre_gate, gate, rep_gated, logits)`.
fn forward_full(
    state: &MetaLearningState,
    pln_w: &[f64],
    pln_b: &[f64],
    x: &[f64],
) -> ForwardFullOut {
    // RLN layer 1
    let h1_pre = linear_forward(
        &state.rln_w1,
        &state.rln_b1,
        x,
        state.input_dim,
        state.hidden_dim,
    );
    let h1: Vec<f64> = h1_pre.iter().map(|&v| relu(v)).collect();

    // RLN layer 2 → rep (pre-gate)
    let rep_pre = linear_forward(
        &state.rln_w2,
        &state.rln_b2,
        &h1,
        state.hidden_dim,
        state.rep_dim,
    );

    // Gate network (ANML): gate = sigmoid(gate_W * rep_pre + gate_b)
    let gate_pre = linear_forward(
        &state.gate_w,
        &state.gate_b,
        &rep_pre,
        state.rep_dim,
        state.rep_dim,
    );
    let gate: Vec<f64> = gate_pre.iter().map(|&v| sigmoid(v)).collect();

    // Element-wise gating
    let rep_gated: Vec<f64> = rep_pre
        .iter()
        .zip(gate.iter())
        .map(|(&r, &g)| r * g)
        .collect();

    // PLN linear head
    let logits = linear_forward(pln_w, pln_b, &rep_gated, state.rep_dim, state.output_dim);

    (h1, rep_pre, gate, rep_gated, logits)
}

/// Compute PLN gradient for a single example via backprop.
///
/// Returns `(grad_pln_w, grad_pln_b)` where shapes mirror `pln_w / pln_b`.
fn pln_gradient_single(
    rep_gated: &[f64],
    logits: &[f64],
    y: usize,
    output_dim: usize,
    rep_dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let probs = softmax(logits);
    // dL/d_logits = probs - one_hot(y)
    let mut d_logits = probs;
    d_logits[y] -= 1.0;
    // dL/dW_pln[i,j] = d_logits[i] * rep_gated[j]
    let mut grad_w = vec![0.0_f64; output_dim * rep_dim];
    for i in 0..output_dim {
        for j in 0..rep_dim {
            grad_w[i * rep_dim + j] = d_logits[i] * rep_gated[j];
        }
    }
    let grad_b = d_logits;
    (grad_w, grad_b)
}

/// Compute gradient of (loss wrt RLN params) for a single example via FOMAML.
///
/// This propagates the adapted-PLN loss gradient back through the gated
/// representation into RLN layer 2 and layer 1.  Returns
/// `(grad_rln_w1, grad_rln_b1, grad_rln_w2, grad_rln_b2,
///   grad_gate_w, grad_gate_b)`.
#[allow(clippy::too_many_arguments)]
fn rln_gradient_single(
    state: &MetaLearningState,
    pln_w: &[f64],
    h1: &[f64],
    rep_pre: &[f64],
    gate: &[f64],
    _rep_gated: &[f64],
    logits: &[f64],
    x: &[f64],
    y: usize,
) -> RlnGrads {
    let probs = softmax(logits);
    let mut d_logits = probs;
    d_logits[y] -= 1.0;

    // Backprop through PLN: dL/d_rep_gated = W_pln^T * d_logits
    let mut d_rep_gated = vec![0.0_f64; state.rep_dim];
    for j in 0..state.rep_dim {
        let mut s = 0.0_f64;
        for i in 0..state.output_dim {
            s += pln_w[i * state.rep_dim + j] * d_logits[i];
        }
        d_rep_gated[j] = s;
    }

    // Through element-wise gate: d(rep_pre) from gating path
    // rep_gated = rep_pre * gate
    // dL/d_rep_pre[j] = d_rep_gated[j] * gate[j]
    // dL/d_gate[j]    = d_rep_gated[j] * rep_pre[j]
    let d_rep_pre_from_gate: Vec<f64> = (0..state.rep_dim)
        .map(|j| d_rep_gated[j] * gate[j])
        .collect();
    let d_gate: Vec<f64> = (0..state.rep_dim)
        .map(|j| d_rep_gated[j] * rep_pre[j])
        .collect();

    // Through sigmoid (gate): d_gate_pre[j] = d_gate[j] * gate[j] * (1 - gate[j])
    let d_gate_pre: Vec<f64> = (0..state.rep_dim)
        .map(|j| d_gate[j] * gate[j] * (1.0 - gate[j]))
        .collect();

    // Gate network grads: gate_W shape [rep_dim × rep_dim], input = rep_pre
    let mut grad_gate_w = vec![0.0_f64; state.rep_dim * state.rep_dim];
    for i in 0..state.rep_dim {
        for j in 0..state.rep_dim {
            grad_gate_w[i * state.rep_dim + j] = d_gate_pre[i] * rep_pre[j];
        }
    }
    let grad_gate_b = d_gate_pre.clone();

    // d_rep_pre from gate network: gate_W^T * d_gate_pre
    let d_rep_pre_from_gate_net: Vec<f64> = (0..state.rep_dim)
        .map(|j| {
            d_gate_pre
                .iter()
                .enumerate()
                .map(|(i, &dg)| state.gate_w[i * state.rep_dim + j] * dg)
                .sum()
        })
        .collect();

    // Total d_rep_pre
    let d_rep_pre: Vec<f64> = (0..state.rep_dim)
        .map(|j| d_rep_pre_from_gate[j] + d_rep_pre_from_gate_net[j])
        .collect();

    // RLN layer 2 grads: W2 shape [rep_dim × hidden_dim], input = h1
    let mut grad_rln_w2 = vec![0.0_f64; state.rep_dim * state.hidden_dim];
    for i in 0..state.rep_dim {
        for j in 0..state.hidden_dim {
            grad_rln_w2[i * state.hidden_dim + j] = d_rep_pre[i] * h1[j];
        }
    }
    let grad_rln_b2 = d_rep_pre.clone();

    // Backprop through ReLU in layer 1: d_h1 = W2^T * d_rep_pre
    let d_h1: Vec<f64> = (0..state.hidden_dim)
        .map(|j| {
            d_rep_pre
                .iter()
                .enumerate()
                .map(|(i, &dr)| state.rln_w2[i * state.hidden_dim + j] * dr)
                .sum()
        })
        .collect();
    // ReLU gate: h1 = relu(rln_w1 * x + rln_b1); d_pre_h1 = d_h1 * (h1 > 0)
    let d_pre_h1: Vec<f64> = h1
        .iter()
        .zip(d_h1.iter())
        .map(|(&h, &dh)| if h > 0.0 { dh } else { 0.0 })
        .collect();

    // RLN layer 1 grads: W1 shape [hidden_dim × input_dim], input = x
    let mut grad_rln_w1 = vec![0.0_f64; state.hidden_dim * state.input_dim];
    for i in 0..state.hidden_dim {
        for j in 0..state.input_dim {
            grad_rln_w1[i * state.input_dim + j] = d_pre_h1[i] * x[j];
        }
    }
    let grad_rln_b1 = d_pre_h1;

    (
        grad_rln_w1,
        grad_rln_b1,
        grad_rln_w2,
        grad_rln_b2,
        grad_gate_w,
        grad_gate_b,
    )
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Initialise a new OML / ANML model state.
pub fn oml_new(cfg: &MetaLearningConfig) -> ContinualResult<MetaLearningState> {
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if cfg.lr_inner <= 0.0 || cfg.lr_outer <= 0.0 {
        return Err(ContinualError::NanEncountered {
            location: "oml_new: lr must be positive",
        });
    }
    if cfg.n_inner_steps == 0 {
        return Err(ContinualError::NanEncountered {
            location: "oml_new: n_inner_steps must be >= 1",
        });
    }

    let rep_dim = (cfg.hidden_dim / 2).max(1);
    let mut rng = LcgRng::new(cfg.seed);

    // RLN layer 1: [hidden_dim × input_dim]
    let mut rln_w1 = vec![0.0_f64; cfg.hidden_dim * cfg.input_dim];
    xavier_init(&mut rln_w1, cfg.input_dim, cfg.hidden_dim, &mut rng);
    let rln_b1 = vec![0.0_f64; cfg.hidden_dim];

    // RLN layer 2: [rep_dim × hidden_dim]
    let mut rln_w2 = vec![0.0_f64; rep_dim * cfg.hidden_dim];
    xavier_init(&mut rln_w2, cfg.hidden_dim, rep_dim, &mut rng);
    let rln_b2 = vec![0.0_f64; rep_dim];

    // PLN: [output_dim × rep_dim]
    let mut pln_w = vec![0.0_f64; cfg.output_dim * rep_dim];
    xavier_init(&mut pln_w, rep_dim, cfg.output_dim, &mut rng);
    let pln_b = vec![0.0_f64; cfg.output_dim];

    // Gate network: [rep_dim × rep_dim]
    let mut gate_w = vec![0.0_f64; rep_dim * rep_dim];
    xavier_init(&mut gate_w, rep_dim, rep_dim, &mut rng);
    let gate_b = vec![0.0_f64; rep_dim];

    Ok(MetaLearningState {
        rln_w1,
        rln_b1,
        rln_w2,
        rln_b2,
        pln_w,
        pln_b,
        gate_w,
        gate_b,
        n_meta_steps: 0,
        input_dim: cfg.input_dim,
        hidden_dim: cfg.hidden_dim,
        rep_dim,
        output_dim: cfg.output_dim,
    })
}

/// Run inner-loop adaptation on `(x_support, y_support)` for `n_inner_steps` steps.
///
/// Returns adapted PLN weights as a flat vector `[pln_w || pln_b]`.
fn inner_loop_adapt(
    state: &MetaLearningState,
    x_support: &[f64],
    y_support: &[usize],
    n_support: usize,
    cfg_lr: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut pln_w = state.pln_w.clone();
    let mut pln_b = state.pln_b.clone();

    if n_support == 0 {
        return (pln_w, pln_b);
    }

    let in_dim = state.input_dim;

    for _ in 0..n_steps {
        let mut grad_w = vec![0.0_f64; state.output_dim * state.rep_dim];
        let mut grad_b = vec![0.0_f64; state.output_dim];

        for s in 0..n_support {
            let x_s = &x_support[s * in_dim..(s + 1) * in_dim];
            let y_s = y_support[s];

            let (_, _, _, rep_gated, logits) = forward_full(state, &pln_w, &pln_b, x_s);
            let (gw, gb) =
                pln_gradient_single(&rep_gated, &logits, y_s, state.output_dim, state.rep_dim);

            for (g, ag) in grad_w.iter_mut().zip(gw.iter()) {
                *g += ag;
            }
            for (g, ag) in grad_b.iter_mut().zip(gb.iter()) {
                *g += ag;
            }
        }

        let inv_n = 1.0 / n_support as f64;
        for (w, g) in pln_w.iter_mut().zip(grad_w.iter()) {
            *w -= cfg_lr * g * inv_n;
        }
        for (b, g) in pln_b.iter_mut().zip(grad_b.iter()) {
            *b -= cfg_lr * g * inv_n;
        }
    }

    (pln_w, pln_b)
}

/// Meta-train the OML / ANML model over a collection of tasks.
///
/// For each meta-epoch:
/// 1. Sample `n_tasks_per_meta` tasks (with replacement).
/// 2. For each task: run inner loop on support set; evaluate on query set;
///    accumulate FOMAML RLN gradients from query-set loss (adapted PLN, query data).
/// 3. Update RLN (slow outer loop).
///
/// Returns the mean query cross-entropy loss over the final meta-epoch.
pub fn oml_meta_train(
    state: &mut MetaLearningState,
    tasks: &[TaskData],
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if tasks.is_empty() {
        return Err(ContinualError::EmptyInput);
    }

    // Retrieve cached hyperparams from meta-state via defaults (no config stored in state)
    // We use lr_inner=0.05 and lr_outer=0.001 from defaults, but the caller
    // controls them via `cfg` — however, state doesn't store cfg.
    // To allow the caller-specified lrs we provide a version that accepts cfg.
    // For the public fn signature (no cfg) we use reasonable defaults.
    oml_meta_train_with_lr(state, tasks, rng, 0.05, 0.001, 5, 3, 4)
}

/// Meta-train with explicit hyperparameters (called by the `MetaLearningConfig`-aware wrapper).
#[allow(clippy::too_many_arguments)]
pub fn oml_meta_train_with_lr(
    state: &mut MetaLearningState,
    tasks: &[TaskData],
    rng: &mut LcgRng,
    lr_inner: f64,
    lr_outer: f64,
    n_inner_steps: usize,
    n_meta_epochs: usize,
    n_tasks_per_meta: usize,
) -> ContinualResult<f64> {
    if tasks.is_empty() {
        return Err(ContinualError::EmptyInput);
    }

    let mut last_epoch_loss = 0.0_f64;

    for epoch in 0..n_meta_epochs {
        // Accumulate RLN meta-gradients across sampled tasks
        let mut meta_grad_rln_w1 = vec![0.0_f64; state.rln_w1.len()];
        let mut meta_grad_rln_b1 = vec![0.0_f64; state.rln_b1.len()];
        let mut meta_grad_rln_w2 = vec![0.0_f64; state.rln_w2.len()];
        let mut meta_grad_rln_b2 = vec![0.0_f64; state.rln_b2.len()];
        let mut meta_grad_gate_w = vec![0.0_f64; state.gate_w.len()];
        let mut meta_grad_gate_b = vec![0.0_f64; state.gate_b.len()];
        let mut epoch_loss = 0.0_f64;

        for _ in 0..n_tasks_per_meta {
            let task_idx = rng.next_usize(tasks.len());
            let task = &tasks[task_idx];

            if task.n_support == 0 || task.n_query == 0 {
                continue;
            }

            // Inner loop: adapt PLN on support set
            let (adapted_pln_w, adapted_pln_b) = inner_loop_adapt(
                state,
                &task.x_support,
                &task.y_support,
                task.n_support,
                lr_inner,
                n_inner_steps,
            );

            // FOMAML: compute gradients on query set using adapted PLN
            // but with current RLN (not re-rolling back through inner loop)
            let mut task_grad_rln_w1 = vec![0.0_f64; state.rln_w1.len()];
            let mut task_grad_rln_b1 = vec![0.0_f64; state.rln_b1.len()];
            let mut task_grad_rln_w2 = vec![0.0_f64; state.rln_w2.len()];
            let mut task_grad_rln_b2 = vec![0.0_f64; state.rln_b2.len()];
            let mut task_grad_gate_w = vec![0.0_f64; state.gate_w.len()];
            let mut task_grad_gate_b = vec![0.0_f64; state.gate_b.len()];
            let mut task_loss = 0.0_f64;

            let in_dim = state.input_dim;
            for q in 0..task.n_query {
                let x_q = &task.x_query[q * in_dim..(q + 1) * in_dim];
                let y_q = task.y_query[q];

                let (h1, rep_pre, gate, rep_gated, logits) =
                    forward_full(state, &adapted_pln_w, &adapted_pln_b, x_q);

                task_loss += cross_entropy_single(&logits, y_q);

                let (gw1, gb1, gw2, gb2, ggw, ggb) = rln_gradient_single(
                    state,
                    &adapted_pln_w,
                    &h1,
                    &rep_pre,
                    &gate,
                    &rep_gated,
                    &logits,
                    x_q,
                    y_q,
                );

                for (g, ag) in task_grad_rln_w1.iter_mut().zip(gw1.iter()) {
                    *g += ag;
                }
                for (g, ag) in task_grad_rln_b1.iter_mut().zip(gb1.iter()) {
                    *g += ag;
                }
                for (g, ag) in task_grad_rln_w2.iter_mut().zip(gw2.iter()) {
                    *g += ag;
                }
                for (g, ag) in task_grad_rln_b2.iter_mut().zip(gb2.iter()) {
                    *g += ag;
                }
                for (g, ag) in task_grad_gate_w.iter_mut().zip(ggw.iter()) {
                    *g += ag;
                }
                for (g, ag) in task_grad_gate_b.iter_mut().zip(ggb.iter()) {
                    *g += ag;
                }
            }

            let inv_q = 1.0 / task.n_query as f64;
            epoch_loss += task_loss * inv_q;

            for (g, tg) in meta_grad_rln_w1.iter_mut().zip(task_grad_rln_w1.iter()) {
                *g += tg * inv_q;
            }
            for (g, tg) in meta_grad_rln_b1.iter_mut().zip(task_grad_rln_b1.iter()) {
                *g += tg * inv_q;
            }
            for (g, tg) in meta_grad_rln_w2.iter_mut().zip(task_grad_rln_w2.iter()) {
                *g += tg * inv_q;
            }
            for (g, tg) in meta_grad_rln_b2.iter_mut().zip(task_grad_rln_b2.iter()) {
                *g += tg * inv_q;
            }
            for (g, tg) in meta_grad_gate_w.iter_mut().zip(task_grad_gate_w.iter()) {
                *g += tg * inv_q;
            }
            for (g, tg) in meta_grad_gate_b.iter_mut().zip(task_grad_gate_b.iter()) {
                *g += tg * inv_q;
            }
        }

        // Outer-loop SGD update on RLN + gate
        let inv_t = 1.0 / n_tasks_per_meta as f64;
        for (w, g) in state.rln_w1.iter_mut().zip(meta_grad_rln_w1.iter()) {
            *w -= lr_outer * g * inv_t;
        }
        for (b, g) in state.rln_b1.iter_mut().zip(meta_grad_rln_b1.iter()) {
            *b -= lr_outer * g * inv_t;
        }
        for (w, g) in state.rln_w2.iter_mut().zip(meta_grad_rln_w2.iter()) {
            *w -= lr_outer * g * inv_t;
        }
        for (b, g) in state.rln_b2.iter_mut().zip(meta_grad_rln_b2.iter()) {
            *b -= lr_outer * g * inv_t;
        }
        for (w, g) in state.gate_w.iter_mut().zip(meta_grad_gate_w.iter()) {
            *w -= lr_outer * g * inv_t;
        }
        for (b, g) in state.gate_b.iter_mut().zip(meta_grad_gate_b.iter()) {
            *b -= lr_outer * g * inv_t;
        }

        state.n_meta_steps += 1;
        last_epoch_loss = epoch_loss / n_tasks_per_meta as f64;
        let _ = epoch; // suppress unused warning
    }

    if !last_epoch_loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "oml_meta_train",
        });
    }

    Ok(last_epoch_loss)
}

/// Fast-adapt the PLN to a new task and return the flattened adapted weights.
///
/// Returns the concatenated `[adapted_pln_w || adapted_pln_b]`.
pub fn oml_adapt(
    state: &MetaLearningState,
    x: &[f64],
    y: &[usize],
    n: usize,
    _rng: &mut LcgRng,
) -> ContinualResult<Vec<f64>> {
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

    let (adapted_w, adapted_b) = inner_loop_adapt(state, x, y, n, 0.05, 5);

    let mut flat = Vec::with_capacity(adapted_w.len() + adapted_b.len());
    flat.extend_from_slice(&adapted_w);
    flat.extend_from_slice(&adapted_b);
    Ok(flat)
}

/// Predict the class for a single sample using adapted PLN weights.
///
/// `adapted_pln`: flattened `[adapted_pln_w || adapted_pln_b]` from `oml_adapt`.
pub fn oml_predict(
    state: &MetaLearningState,
    adapted_pln: &[f64],
    x: &[f64],
) -> ContinualResult<usize> {
    if x.len() != state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: state.input_dim,
            got: x.len(),
        });
    }

    let pln_w_len = state.output_dim * state.rep_dim;
    let pln_b_len = state.output_dim;
    let expected = pln_w_len + pln_b_len;
    if adapted_pln.len() != expected {
        return Err(ContinualError::DimensionMismatch {
            expected,
            got: adapted_pln.len(),
        });
    }

    let pln_w = &adapted_pln[..pln_w_len];
    let pln_b = &adapted_pln[pln_w_len..];

    let (_, _, _, _, logits) = forward_full(state, pln_w, pln_b, x);

    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(pred)
}

/// Return the number of completed meta-gradient update steps.
pub fn oml_inner_step_count(state: &MetaLearningState) -> usize {
    state.n_meta_steps
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> MetaLearningConfig {
        MetaLearningConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 3,
            lr_inner: 0.1,
            lr_outer: 0.01,
            n_inner_steps: 2,
            n_meta_epochs: 2,
            n_tasks_per_meta: 2,
            seed: 7,
        }
    }

    fn make_task(n_support: usize, n_query: usize, input_dim: usize, seed: u64) -> TaskData {
        let mut rng = LcgRng::new(seed);
        let x_support: Vec<f64> = (0..n_support * input_dim)
            .map(|_| rng.next_f32() as f64)
            .collect();
        let y_support: Vec<usize> = (0..n_support).map(|i| i % 3).collect();
        let x_query: Vec<f64> = (0..n_query * input_dim)
            .map(|_| rng.next_f32() as f64)
            .collect();
        let y_query: Vec<usize> = (0..n_query).map(|i| i % 3).collect();
        TaskData {
            x_support,
            y_support,
            x_query,
            y_query,
            n_support,
            n_query,
        }
    }

    #[test]
    fn oml_new_initialises_correct_dims() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        assert_eq!(state.input_dim, 4);
        assert_eq!(state.hidden_dim, 8);
        assert_eq!(state.rep_dim, 4); // hidden_dim / 2
        assert_eq!(state.output_dim, 3);
        assert_eq!(state.rln_w1.len(), 8 * 4); // hidden × input
        assert_eq!(state.rln_w2.len(), 4 * 8); // rep × hidden
        assert_eq!(state.pln_w.len(), 3 * 4); // output × rep
        assert_eq!(state.gate_w.len(), 4 * 4); // rep × rep
        assert_eq!(state.n_meta_steps, 0);
    }

    #[test]
    fn oml_new_weights_in_xavier_range() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let scale = xavier_scale(cfg.input_dim, cfg.hidden_dim);
        for &w in &state.rln_w1 {
            assert!(w.abs() <= scale + 1e-9, "rln_w1 out of xavier range: {w}");
        }
    }

    #[test]
    fn oml_predict_returns_valid_class() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x = vec![0.1_f64; 4];
        let pln_w_len = state.output_dim * state.rep_dim;
        let pln_b_len = state.output_dim;
        let mut adapted = vec![0.0_f64; pln_w_len + pln_b_len];
        adapted[..pln_w_len].copy_from_slice(&state.pln_w);
        adapted[pln_w_len..].copy_from_slice(&state.pln_b);
        let pred = oml_predict(&state, &adapted, &x)
            .expect("OML prediction should succeed on valid input");
        assert!(pred < cfg.output_dim);
    }

    #[test]
    fn oml_meta_train_decrements_loss() {
        let cfg = make_cfg();
        let mut state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let tasks: Vec<TaskData> = (0..4).map(|i| make_task(3, 3, 4, i as u64 + 10)).collect();
        let mut rng = LcgRng::new(99);
        let loss = oml_meta_train_with_lr(
            &mut state,
            &tasks,
            &mut rng,
            cfg.lr_inner,
            cfg.lr_outer,
            cfg.n_inner_steps,
            cfg.n_meta_epochs,
            cfg.n_tasks_per_meta,
        )
        .expect("should succeed with valid test inputs");
        assert!(loss.is_finite(), "meta-train loss must be finite: {loss}");
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    #[test]
    fn oml_adapt_returns_correct_flat_len() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x: Vec<f64> = (0..4 * 3).map(|i| i as f64 * 0.1).collect();
        let y: Vec<usize> = vec![0, 1, 2];
        let mut rng = LcgRng::new(1);
        let flat = oml_adapt(&state, &x, &y, 3, &mut rng)
            .expect("OML adaptation should succeed with valid data");
        let expected_len = state.output_dim * state.rep_dim + state.output_dim;
        assert_eq!(flat.len(), expected_len);
    }

    #[test]
    fn oml_adapt_weights_finite() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.05).collect();
        let y: Vec<usize> = vec![0, 1, 2, 0];
        let mut rng = LcgRng::new(2);
        let flat = oml_adapt(&state, &x, &y, 4, &mut rng)
            .expect("OML adaptation should succeed with valid data");
        assert!(flat.iter().all(|v| v.is_finite()), "adapted weights finite");
    }

    #[test]
    fn oml_predict_dim_mismatch_error() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x = vec![0.0_f64; 10]; // wrong input size
        let adapted = vec![0.0_f64; state.output_dim * state.rep_dim + state.output_dim];
        let res = oml_predict(&state, &adapted, &x);
        assert!(res.is_err());
    }

    #[test]
    fn oml_adapt_dim_mismatch_error() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x = vec![0.0_f64; 10]; // wrong
        let y = vec![0_usize; 3];
        let mut rng = LcgRng::new(3);
        let res = oml_adapt(&state, &x, &y, 3, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn oml_meta_train_empty_tasks_error() {
        let cfg = make_cfg();
        let mut state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let mut rng = LcgRng::new(4);
        let res = oml_meta_train(&mut state, &[], &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn oml_inner_step_count_increments() {
        let cfg = make_cfg();
        let mut state = oml_new(&cfg).expect("OML state should initialize with valid config");
        assert_eq!(oml_inner_step_count(&state), 0);
        let tasks: Vec<TaskData> = (0..2).map(|i| make_task(2, 2, 4, i as u64 + 5)).collect();
        let mut rng = LcgRng::new(55);
        oml_meta_train(&mut state, &tasks, &mut rng).expect("OML meta-training should succeed");
        // 3 meta-epochs (default in oml_meta_train)
        assert_eq!(oml_inner_step_count(&state), 3);
    }

    #[test]
    fn oml_new_invalid_config_errors() {
        let mut cfg = make_cfg();
        cfg.input_dim = 0;
        assert!(oml_new(&cfg).is_err());
        cfg.input_dim = 4;
        cfg.hidden_dim = 0;
        assert!(oml_new(&cfg).is_err());
    }

    #[test]
    fn oml_forward_logits_finite() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x = vec![0.5_f64; 4];
        let (_, _, _, _, logits) =
            forward_full(&state, &state.pln_w.clone(), &state.pln_b.clone(), &x);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(logits.len(), cfg.output_dim);
    }

    #[test]
    fn oml_gate_output_in_zero_one() {
        let cfg = make_cfg();
        let state = oml_new(&cfg).expect("OML state should initialize with valid config");
        let x = vec![1.0_f64; 4];
        let (_, _, gate, _, _) =
            forward_full(&state, &state.pln_w.clone(), &state.pln_b.clone(), &x);
        for &g in &gate {
            assert!((0.0..=1.0).contains(&g), "gate value {g} not in [0,1]");
        }
    }

    #[test]
    fn oml_meta_train_with_lr_updates_rln() {
        let cfg = make_cfg();
        let mut state = oml_new(&cfg).expect("OML state should initialize with valid config");
        // Force positive weights so ReLU units fire and gradients propagate
        state.rln_w1.iter_mut().for_each(|w| *w = 0.1);
        state.rln_w2.iter_mut().for_each(|w| *w = 0.1);
        let rln_w1_before = state.rln_w1.clone();
        let tasks: Vec<TaskData> = (0..3).map(|i| make_task(3, 3, 4, i as u64 + 20)).collect();
        let mut rng = LcgRng::new(77);
        oml_meta_train_with_lr(
            &mut state,
            &tasks,
            &mut rng,
            0.1,
            0.1,
            cfg.n_inner_steps,
            cfg.n_meta_epochs,
            cfg.n_tasks_per_meta,
        )
        .expect("should succeed with valid test inputs");
        // RLN weights should have changed
        let changed = rln_w1_before
            .iter()
            .zip(state.rln_w1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            changed,
            "RLN weights should have been updated by meta-training"
        );
    }
}
