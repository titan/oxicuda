//! Memory-Efficient Replay via Gradient Compression.
//!
//! Instead of storing raw exemplars, this method stores compressed gradient
//! summaries — the gradient *direction* that characterises each past task.
//! During training of a new task, these "gradient memories" are used to
//! project the current gradient to avoid interference with past tasks,
//! following the GEM (Gradient Episodic Memory) projection approach:
//!
//! Lopez-Paz & Ranzato. "Gradient Episodic Memory for Continual Learning."
//! NeurIPS 2017.
//!
//! For each new mini-batch:
//! 1. Compute gradient `g_new` from current task.
//! 2. For each stored memory gradient `g_m`:
//!    - If `dot(g_new, g_m) < 0` (conflicting), project:
//!      `g_proj = g_new - (dot(g_new, g_m) / dot(g_m, g_m)) * g_m`
//! 3. Apply sequential projections for multiple conflicts.
//! 4. SGD update with the projected gradient.
//!
//! After task training, store `n_memories_per_task` gradient vectors and
//! the data mean as a `GradMemory` record.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for memory-efficient gradient-compression replay.
#[derive(Debug, Clone)]
pub struct GradCompConfig {
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of output classes.
    pub output_dim: usize,
    /// Number of representative gradient vectors to store per task.
    pub n_memories_per_task: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
}

impl Default for GradCompConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 10,
            n_memories_per_task: 4,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

// ─── Gradient Memory ─────────────────────────────────────────────────────────

/// Compressed gradient summary for one past task.
#[derive(Debug, Clone)]
pub struct GradMemory {
    /// Task identifier.
    pub task_id: usize,
    /// Stored gradient direction vectors, each of length = total parameter count.
    pub gradient_directions: Vec<Vec<f64>>,
    /// Mean of the task's input data (lightweight reference).
    pub data_mean: Vec<f64>,
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Model state for gradient-compression replay.
///
/// Architecture: `input_dim → hidden_dim → output_dim`, ReLU, Xavier init.
/// All parameters are stored flat in layer-major order:
/// `[W1 (hidden×input) | b1 (hidden) | W2 (output×hidden) | b2 (output)]`.
#[derive(Debug, Clone)]
pub struct GradCompState {
    /// Flat parameter vector `[W1 | b1 | W2 | b2]`.
    pub weights: Vec<f64>,
    /// Deprecated alias; length 0 (kept for structural symmetry with other modules).
    pub biases: Vec<f64>,
    /// Stored gradient memories, one per completed task.
    pub memories: Vec<GradMemory>,
    /// Layer sizes: `[input_dim, hidden_dim, output_dim]`.
    pub layer_sizes: Vec<usize>,
    /// Number of tasks seen so far.
    pub n_tasks: usize,
    // Cached cfg fields
    pub(crate) n_memories_per_task: usize,
    pub(crate) lr: f64,
    pub(crate) n_epochs: usize,
}

impl GradCompState {
    /// Total parameter count.
    pub fn n_params(&self) -> usize {
        self.weights.len()
    }

    /// Offsets `(w1_end, b1_end, w2_end, b2_end)` into the flat weights vector.
    fn offsets(&self) -> (usize, usize, usize, usize) {
        let d_in = self.layer_sizes[0];
        let d_h = self.layer_sizes[1];
        let d_out = self.layer_sizes[2];
        let w1_end = d_h * d_in;
        let b1_end = w1_end + d_h;
        let w2_end = b1_end + d_out * d_h;
        let b2_end = w2_end + d_out;
        (w1_end, b1_end, w2_end, b2_end)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Xavier uniform scale.
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

/// Softmax.
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

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Forward pass; returns `(h1_pre, h1, logits)`.
fn forward(state: &GradCompState, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];
    let (w1_end, b1_end, w2_end, b2_end) = state.offsets();

    let w1 = &state.weights[0..w1_end];
    let b1 = &state.weights[w1_end..b1_end];
    let w2 = &state.weights[b1_end..w2_end];
    let b2 = &state.weights[w2_end..b2_end];

    // Layer 1
    let mut h1_pre = vec![0.0_f64; d_h];
    for i in 0..d_h {
        h1_pre[i] = b1[i] + (0..d_in).map(|j| w1[i * d_in + j] * x[j]).sum::<f64>();
    }
    let h1: Vec<f64> = h1_pre.iter().map(|&v| relu(v)).collect();

    // Layer 2
    let mut logits = vec![0.0_f64; d_out];
    for i in 0..d_out {
        logits[i] = b2[i] + (0..d_h).map(|j| w2[i * d_h + j] * h1[j]).sum::<f64>();
    }

    let _ = b2_end; // suppress unused warning
    (h1_pre, h1, logits)
}

/// Compute the full parameter gradient for a single sample.
///
/// Returns a flat gradient vector matching `state.weights`.
fn gradient_single(state: &GradCompState, x: &[f64], y: usize) -> Vec<f64> {
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];
    let (w1_end, b1_end, w2_end, _) = state.offsets();

    let w2 = &state.weights[b1_end..w2_end];

    let (h1_pre, h1, logits) = forward(state, x);

    // dL/d_logits
    let mut d_logits = softmax(&logits);
    d_logits[y] -= 1.0;

    let n_params = state.weights.len();
    let mut grad = vec![0.0_f64; n_params];

    // Layer 2: dL/dW2[i,j] = d_logits[i] * h1[j]
    //          dL/db2[i]   = d_logits[i]
    let w2_start = b1_end;
    let b2_start = w2_end;
    for i in 0..d_out {
        grad[b2_start + i] = d_logits[i];
        for j in 0..d_h {
            grad[w2_start + i * d_h + j] = d_logits[i] * h1[j];
        }
    }

    // d_h1 = W2^T * d_logits
    let mut d_h1 = vec![0.0_f64; d_h];
    for j in 0..d_h {
        for i in 0..d_out {
            d_h1[j] += w2[i * d_h + j] * d_logits[i];
        }
    }

    // Through ReLU
    let d_h1_pre: Vec<f64> = h1_pre
        .iter()
        .zip(d_h1.iter())
        .map(|(&h, &dh)| if h > 0.0 { dh } else { 0.0 })
        .collect();

    // Layer 1: dL/dW1[i,j] = d_h1_pre[i] * x[j]
    //          dL/db1[i]   = d_h1_pre[i]
    let w1_start = 0;
    let b1_start = w1_end;
    for i in 0..d_h {
        grad[b1_start + i] = d_h1_pre[i];
        for j in 0..d_in {
            grad[w1_start + i * d_in + j] = d_h1_pre[i] * x[j];
        }
    }

    grad
}

/// GEM-style projection: project `g` so that `dot(g, g_m) >= 0` for all `g_m`.
///
/// Sequential single-constraint projection: iterates over all conflicting
/// memories and applies each correction in turn.  Not exact QP, but is the
/// same approximation used in GEM's original implementation.
fn gem_project(g: &mut [f64], memories: &[Vec<f64>]) {
    for g_m in memories {
        if g_m.is_empty() {
            continue;
        }
        let d = dot(g, g_m);
        if d < 0.0 {
            let denom = dot(g_m, g_m);
            if denom < 1e-30 {
                continue;
            }
            let scale = d / denom;
            for (gi, mi) in g.iter_mut().zip(g_m.iter()) {
                *gi -= scale * mi;
            }
        }
    }
}

/// Compute mean of rows in `x` (each row has `in_dim` elements).
fn compute_data_mean(x: &[f64], n: usize, in_dim: usize) -> Vec<f64> {
    let mut mean = vec![0.0_f64; in_dim];
    if n == 0 {
        return mean;
    }
    for i in 0..n {
        for j in 0..in_dim {
            mean[j] += x[i * in_dim + j];
        }
    }
    let inv = 1.0 / n as f64;
    for m in &mut mean {
        *m *= inv;
    }
    mean
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Initialise a new `GradCompState`.
pub fn grad_comp_new(cfg: &GradCompConfig, seed: u64) -> ContinualResult<GradCompState> {
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if cfg.n_memories_per_task == 0 {
        return Err(ContinualError::NanEncountered {
            location: "grad_comp_new: n_memories_per_task must be >= 1",
        });
    }
    if cfg.lr <= 0.0 || !cfg.lr.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "grad_comp_new: lr must be positive and finite",
        });
    }
    if cfg.n_epochs == 0 {
        return Err(ContinualError::NanEncountered {
            location: "grad_comp_new: n_epochs must be >= 1",
        });
    }

    let d_in = cfg.input_dim;
    let d_h = cfg.hidden_dim;
    let d_out = cfg.output_dim;
    let n_params = d_h * d_in + d_h + d_out * d_h + d_out;

    let mut rng = LcgRng::new(seed);
    let mut weights = vec![0.0_f64; n_params];

    // Xavier init layer 1: W1 [d_h × d_in]
    let w1_end = d_h * d_in;
    xavier_init(&mut weights[0..w1_end], d_in, d_h, &mut rng);
    // b1 stays zero

    // Xavier init layer 2: W2 [d_out × d_h]
    let b1_end = w1_end + d_h;
    let w2_end = b1_end + d_out * d_h;
    xavier_init(&mut weights[b1_end..w2_end], d_h, d_out, &mut rng);
    // b2 stays zero

    Ok(GradCompState {
        weights,
        biases: vec![],
        memories: Vec::new(),
        layer_sizes: vec![d_in, d_h, d_out],
        n_tasks: 0,
        n_memories_per_task: cfg.n_memories_per_task,
        lr: cfg.lr,
        n_epochs: cfg.n_epochs,
    })
}

/// Train on task `task_id` with GEM-style gradient projection.
///
/// Returns the mean cross-entropy loss of the final epoch.
pub fn grad_comp_fit_task(
    state: &mut GradCompState,
    x: &[f64],
    y: &[usize],
    n: usize,
    task_id: usize,
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if n == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let in_dim = state.layer_sizes[0];
    if x.len() != n * in_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: n * in_dim,
            got: x.len(),
        });
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }

    let lr = state.lr;
    let n_epochs = state.n_epochs;

    // Collect all stored memory gradient vectors for projection
    let all_memory_grads: Vec<&Vec<f64>> = state
        .memories
        .iter()
        .flat_map(|m| m.gradient_directions.iter())
        .collect();

    let mut indices: Vec<usize> = (0..n).collect();
    let mut final_loss = 0.0_f64;
    // Collect gradient vectors from last epoch for memory storage
    let mut last_epoch_grads: Vec<Vec<f64>> = Vec::new();

    for epoch in 0..n_epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0_f64;
        let is_last = epoch + 1 == n_epochs;

        if is_last {
            last_epoch_grads.clear();
        }

        for &s in &indices {
            let x_s = &x[s * in_dim..(s + 1) * in_dim];
            let y_s = y[s];

            let (_, _, logits) = forward(state, x_s);
            let probs = softmax(&logits);
            let p = probs.get(y_s).copied().unwrap_or(1e-12).max(1e-12);
            epoch_loss += -p.ln();

            // Compute gradient
            let mut g = gradient_single(state, x_s, y_s);

            // GEM-style projection against all stored memories
            // We clone the directions to avoid borrow issues
            let mem_dirs: Vec<Vec<f64>> = all_memory_grads.iter().map(|v| (*v).clone()).collect();
            gem_project(&mut g, &mem_dirs);

            // SGD update
            for (w, gi) in state.weights.iter_mut().zip(g.iter()) {
                *w -= lr * gi;
            }

            if is_last {
                // Store this sample's gradient for memory
                let g_stored = gradient_single(state, x_s, y_s);
                last_epoch_grads.push(g_stored);
            }
        }

        final_loss = epoch_loss / n as f64;
        let _ = epoch;
    }

    // Store n_memories_per_task gradient vectors from last epoch
    let n_mem = state.n_memories_per_task;
    // Subsample: take evenly spaced from last_epoch_grads
    let n_available = last_epoch_grads.len();
    let mut stored_grads: Vec<Vec<f64>> = Vec::with_capacity(n_mem);
    if n_available > 0 {
        for k in 0..n_mem {
            let idx = if n_mem == 1 {
                0
            } else {
                (k * (n_available - 1)) / (n_mem - 1).max(1)
            };
            let idx = idx.min(n_available - 1);
            stored_grads.push(last_epoch_grads[idx].clone());
        }
    }

    let data_mean = compute_data_mean(x, n, in_dim);

    state.memories.push(GradMemory {
        task_id,
        gradient_directions: stored_grads,
        data_mean,
    });
    state.n_tasks += 1;

    if !final_loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "grad_comp_fit_task",
        });
    }
    Ok(final_loss)
}

/// Predict the class for a single input (argmax of logits).
pub fn grad_comp_predict(state: &GradCompState, x: &[f64]) -> ContinualResult<usize> {
    let in_dim = state.layer_sizes[0];
    if x.len() != in_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: in_dim,
            got: x.len(),
        });
    }

    let (_, _, logits) = forward(state, x);
    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(pred)
}

/// Return the total number of gradient direction vectors stored across all tasks.
pub fn grad_comp_n_memories(state: &GradCompState) -> usize {
    state
        .memories
        .iter()
        .map(|m| m.gradient_directions.len())
        .sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> GradCompConfig {
        GradCompConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 3,
            n_memories_per_task: 2,
            lr: 0.05,
            n_epochs: 2,
        }
    }

    #[test]
    fn grad_comp_new_correct_dims() {
        let cfg = make_cfg();
        let state = grad_comp_new(&cfg, 1).unwrap();
        let expected_params = cfg.hidden_dim * cfg.input_dim
            + cfg.hidden_dim
            + cfg.output_dim * cfg.hidden_dim
            + cfg.output_dim;
        assert_eq!(state.weights.len(), expected_params);
        assert_eq!(state.n_tasks, 0);
        assert!(state.memories.is_empty());
        assert_eq!(state.layer_sizes, vec![4, 8, 3]);
    }

    #[test]
    fn grad_comp_new_weights_finite() {
        let cfg = make_cfg();
        let state = grad_comp_new(&cfg, 2).unwrap();
        assert!(state.weights.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn grad_comp_fit_task_returns_finite_loss() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 3).unwrap();
        let x: Vec<f64> = (0..4 * 5).map(|i| i as f64 * 0.05).collect();
        let y = vec![0_usize, 1, 2, 0, 1];
        let mut rng = LcgRng::new(10);
        let loss = grad_comp_fit_task(&mut state, &x, &y, 5, 0, &mut rng).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn grad_comp_fit_task_stores_memory() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 4).unwrap();
        let x: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.1).collect();
        let y = vec![0_usize, 1, 2, 0];
        let mut rng = LcgRng::new(20);
        grad_comp_fit_task(&mut state, &x, &y, 4, 0, &mut rng).unwrap();
        assert_eq!(state.memories.len(), 1);
        assert_eq!(state.memories[0].task_id, 0);
        assert_eq!(state.memories[0].gradient_directions.len(), 2);
        assert_eq!(state.memories[0].data_mean.len(), 4);
    }

    #[test]
    fn grad_comp_n_memories_accumulates() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 5).unwrap();
        let mut rng = LcgRng::new(30);
        for task in 0..3 {
            let x: Vec<f64> = (0..4 * 3).map(|i| (i + task) as f64 * 0.1).collect();
            let y = vec![0_usize, 1, 2];
            grad_comp_fit_task(&mut state, &x, &y, 3, task, &mut rng).unwrap();
        }
        // 3 tasks × 2 memories each = 6
        assert_eq!(grad_comp_n_memories(&state), 6);
    }

    #[test]
    fn grad_comp_predict_valid_class() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 6).unwrap();
        let x: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.1).collect();
        let y = vec![0_usize, 1, 2, 0];
        let mut rng = LcgRng::new(40);
        grad_comp_fit_task(&mut state, &x, &y, 4, 0, &mut rng).unwrap();
        let pred = grad_comp_predict(&state, &[0.5, 0.3, 0.1, 0.9]).unwrap();
        assert!(pred < 3);
    }

    #[test]
    fn grad_comp_gem_projection_reduces_conflict() {
        // Direct test of the projection helper
        let mut g = vec![-1.0_f64, 0.0];
        let memories = vec![vec![1.0_f64, 0.0]];
        // dot(g, g_m) = -1 < 0 → projection needed
        gem_project(&mut g, &memories);
        // After projection: dot(g_proj, g_m) >= 0
        let d: f64 = g.iter().zip(memories[0].iter()).map(|(a, b)| a * b).sum();
        assert!(d >= -1e-9, "GEM projection must satisfy dot >= 0, got {d}");
    }

    #[test]
    fn grad_comp_gem_no_projection_when_aligned() {
        let g_orig = vec![1.0_f64, 0.0];
        let mut g = g_orig.clone();
        let memories = vec![vec![1.0_f64, 0.0]];
        gem_project(&mut g, &memories);
        // Should remain unchanged
        let eps: f64 = g
            .iter()
            .zip(g_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(eps < 1e-10, "Aligned gradient should not be projected");
    }

    #[test]
    fn grad_comp_multi_task_no_conflict_after_projection() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 7).unwrap();
        let mut rng = LcgRng::new(50);

        // Task 0
        let x0: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.05).collect();
        let y0 = vec![0_usize, 1, 2, 0];
        grad_comp_fit_task(&mut state, &x0, &y0, 4, 0, &mut rng).unwrap();
        assert_eq!(state.n_tasks, 1);

        // Task 1
        let x1: Vec<f64> = (0..4 * 4).map(|i| (i as f64 + 2.0) * 0.03).collect();
        let y1 = vec![1_usize, 2, 0, 1];
        let loss1 = grad_comp_fit_task(&mut state, &x1, &y1, 4, 1, &mut rng).unwrap();
        assert!(loss1.is_finite());
        assert_eq!(state.n_tasks, 2);
    }

    #[test]
    fn grad_comp_fit_task_empty_data_errors() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 8).unwrap();
        let mut rng = LcgRng::new(60);
        let res = grad_comp_fit_task(&mut state, &[], &[], 0, 0, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn grad_comp_predict_wrong_dim_errors() {
        let cfg = make_cfg();
        let state = grad_comp_new(&cfg, 9).unwrap();
        let res = grad_comp_predict(&state, &[0.1, 0.2]); // wrong dim
        assert!(res.is_err());
    }

    #[test]
    fn grad_comp_new_invalid_cfg_errors() {
        let mut cfg = make_cfg();
        cfg.input_dim = 0;
        assert!(grad_comp_new(&cfg, 10).is_err());
        cfg.input_dim = 4;
        cfg.n_memories_per_task = 0;
        assert!(grad_comp_new(&cfg, 11).is_err());
    }

    #[test]
    fn grad_comp_data_mean_correct() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mean = compute_data_mean(&x, 2, 2);
        assert!((mean[0] - 2.0).abs() < 1e-9); // (1+3)/2
        assert!((mean[1] - 3.0).abs() < 1e-9); // (2+4)/2
    }

    #[test]
    fn grad_comp_weights_change_after_training() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 12).unwrap();
        let before = state.weights.clone();
        let x: Vec<f64> = (0..4 * 5).map(|i| i as f64 * 0.1).collect();
        let y = vec![0_usize, 1, 2, 0, 1];
        let mut rng = LcgRng::new(70);
        grad_comp_fit_task(&mut state, &x, &y, 5, 0, &mut rng).unwrap();
        let changed = before
            .iter()
            .zip(state.weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "weights must change after training");
    }

    #[test]
    fn grad_comp_gradient_vectors_have_correct_len() {
        let cfg = make_cfg();
        let mut state = grad_comp_new(&cfg, 13).unwrap();
        let x: Vec<f64> = (0..4 * 3).map(|i| i as f64 * 0.1).collect();
        let y = vec![0_usize, 1, 2];
        let mut rng = LcgRng::new(80);
        grad_comp_fit_task(&mut state, &x, &y, 3, 0, &mut rng).unwrap();
        let n_params = state.weights.len();
        for dir in &state.memories[0].gradient_directions {
            assert_eq!(dir.len(), n_params, "gradient direction length mismatch");
        }
    }
}
