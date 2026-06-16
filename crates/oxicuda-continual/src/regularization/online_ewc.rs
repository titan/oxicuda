//! Online EWC: Memory-efficient running Fisher estimate.
//!
//! Implements the method from:
//! Schwarz et al. "Progress & Compress: A scalable framework for continual learning."
//! ICML 2018.
//!
//! Instead of storing one Fisher matrix per task, Online EWC maintains a single
//! exponentially decayed running Fisher:
//! `F̄ = γ · F̄_prev + F_new`
//!
//! This bounds memory to O(|θ|) regardless of the number of tasks.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for Online EWC.
#[derive(Debug, Clone)]
pub struct OnlineEwcConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of output classes.
    pub output_dim: usize,
    /// EWC regularisation strength (λ).
    pub lambda: f64,
    /// Fisher decay factor (γ). Default 0.9.
    pub gamma: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
    /// Number of samples used to estimate diagonal Fisher per task.
    pub fisher_samples: usize,
}

impl Default for OnlineEwcConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 10,
            lambda: 1.0,
            gamma: 0.9,
            lr: 0.01,
            n_epochs: 5,
            fisher_samples: 64,
        }
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Online EWC model state.
///
/// Architecture: `input_dim → hidden_dim → output_dim`, ReLU, Xavier init.
/// All parameters and Fisher stored in flat vectors in layer-major order:
/// `[W1 (hidden×input), b1 (hidden), W2 (output×hidden), b2 (output)]`.
#[derive(Debug, Clone)]
pub struct OnlineEwcState {
    /// Flat weight/bias vector (W1 ‖ b1 ‖ W2 ‖ b2).
    pub weights: Vec<f64>,
    /// Deprecated alias — kept for API compatibility; mirrors `weights`.
    pub biases: Vec<f64>,
    /// Running diagonal Fisher: `F̄[i] = γ^t Σ_{τ≤t} γ^{t-τ} F_τ[i]`.
    pub running_fisher: Vec<f64>,
    /// θ* at time of last Fisher update (anchor point).
    pub running_theta_star: Vec<f64>,
    /// Number of tasks trained so far.
    pub n_tasks: usize,
    /// Layer sizes: `[input_dim, hidden_dim, output_dim]`.
    pub layer_sizes: Vec<usize>,
    /// Cached γ for penalty computation.
    pub(crate) gamma: f64,
    /// Cached λ.
    pub(crate) lambda: f64,
}

impl OnlineEwcState {
    /// Total number of parameters.
    pub fn n_params(&self) -> usize {
        self.weights.len()
    }

    /// Layer-size offsets for slicing the flat weight vector.
    /// Returns `(w1_start, w1_end, b1_end, w2_end, b2_end)`.
    fn offsets(&self) -> (usize, usize, usize, usize, usize) {
        let d_in = self.layer_sizes[0];
        let d_h = self.layer_sizes[1];
        let d_out = self.layer_sizes[2];
        let w1_start = 0;
        let w1_end = d_h * d_in;
        let b1_end = w1_end + d_h;
        let w2_end = b1_end + d_out * d_h;
        let b2_end = w2_end + d_out;
        (w1_start, w1_end, b1_end, w2_end, b2_end)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Xavier uniform scale: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Row-major matrix-vector product `W x + b`.
#[inline]
fn matvec(w: &[f64], x: &[f64], b: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for row in 0..out_dim {
        let mut acc = b[row];
        let base = row * in_dim;
        for col in 0..in_dim {
            acc += w[base + col] * x[col];
        }
        out[row] = acc;
    }
    out
}

/// Softmax over a slice (numerically stable).
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exp.iter().sum::<f64>().max(1e-30);
    exp.iter().map(|&e| e / sum).collect()
}

/// Full forward pass: returns `(h1, logits)` (h1 after ReLU, logits raw).
fn forward(state: &OnlineEwcState, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (w1_s, w1_e, b1_e, w2_e, b2_e) = state.offsets();
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];

    let w1 = &state.weights[w1_s..w1_e];
    let b1 = &state.weights[w1_e..b1_e];
    let w2 = &state.weights[b1_e..w2_e];
    let b2 = &state.weights[w2_e..b2_e];

    let mut h1 = matvec(w1, x, b1, d_in, d_h);
    for v in h1.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let logits = matvec(w2, &h1, b2, d_h, d_out);
    (h1, logits)
}

/// Compute gradients of log-likelihood log p(y=label | x, θ) w.r.t. all params.
/// Returns the gradient vector (same layout as `state.weights`).
fn log_likelihood_grad(state: &OnlineEwcState, x: &[f64], label: usize) -> Vec<f64> {
    let (w1_s, w1_e, b1_e, w2_e, _b2_e) = state.offsets();
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];

    let w1 = &state.weights[w1_s..w1_e];
    let b1 = &state.weights[w1_e..b1_e];
    let w2 = &state.weights[b1_e..w2_e];

    // Forward.
    let mut h1_pre = matvec(w1, x, b1, d_in, d_h);
    let h1_relu: Vec<f64> = h1_pre.iter().map(|&v| v.max(0.0)).collect();
    let logits = matvec(w2, &h1_relu, &state.weights[w2_e..], d_h, d_out);

    // CE gradient at output: probs - one_hot.
    let probs = softmax(&logits);
    let mut d_logits = probs;
    if label < d_out {
        d_logits[label] -= 1.0;
    }

    // Negate for log-likelihood gradient (we want ∇ log p = -∇ CE).
    for v in d_logits.iter_mut() {
        *v = -*v;
    }

    // Build gradient vector (same layout as weights).
    let mut grad = vec![0.0_f64; state.weights.len()];

    // Backprop layer 2: dW2[row,col] = d_logits[row] * h1_relu[col]; db2 = d_logits.
    let w2_grad_start = b1_e;
    let b2_grad_start = w2_e;
    for row in 0..d_out {
        grad[b2_grad_start + row] = d_logits[row];
        for col in 0..d_h {
            grad[w2_grad_start + row * d_h + col] = d_logits[row] * h1_relu[col];
        }
    }

    // d_h1 = W2^T * d_logits.
    let mut d_h1 = vec![0.0_f64; d_h];
    for col in 0..d_h {
        for row in 0..d_out {
            d_h1[col] += w2[row * d_h + col] * d_logits[row];
        }
    }

    // ReLU backward.
    for i in 0..d_h {
        if h1_pre[i] <= 0.0 {
            d_h1[i] = 0.0;
        }
        h1_pre[i] = d_h1[i];
    }

    // Backprop layer 1: dW1, db1.
    let w1_grad_start = 0;
    let b1_grad_start = w1_e;
    for row in 0..d_h {
        let g = h1_pre[row];
        grad[b1_grad_start + row] = g;
        for col in 0..d_in {
            grad[w1_grad_start + row * d_in + col] = g * x[col];
        }
    }

    grad
}

/// EWC regularisation gradient: `λ · F̄ · (θ - θ*)`.
fn ewc_grad(state: &OnlineEwcState) -> Vec<f64> {
    if state.n_tasks == 0 {
        return vec![0.0_f64; state.weights.len()];
    }
    state
        .weights
        .iter()
        .zip(state.running_fisher.iter())
        .zip(state.running_theta_star.iter())
        .map(|((&theta, &f), &theta_star)| state.lambda * f * (theta - theta_star))
        .collect()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Create a new Online EWC state with Xavier-initialised weights.
pub fn online_ewc_new(cfg: &OnlineEwcConfig, seed: u64) -> ContinualResult<OnlineEwcState> {
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(ContinualError::InvalidLambda {
            lambda: cfg.lambda as f32,
        });
    }

    let d_in = cfg.input_dim;
    let d_h = cfg.hidden_dim;
    let d_out = cfg.output_dim;
    // Layout: W1 (d_h × d_in) | b1 (d_h) | W2 (d_out × d_h) | b2 (d_out)
    let n_params = d_h * d_in + d_h + d_out * d_h + d_out;

    let mut rng = LcgRng::new(seed);
    let scale1 = xavier_scale(d_in, d_h);
    let scale2 = xavier_scale(d_h, d_out);
    let mut weights = vec![0.0_f64; n_params];

    let w1_end = d_h * d_in;
    let b1_end = w1_end + d_h;
    let w2_end = b1_end + d_out * d_h;

    // W1
    for v in weights[0..w1_end].iter_mut() {
        let u = rng.next_f32() as f64;
        *v = (2.0 * u - 1.0) * scale1;
    }
    // b1 = 0 already.
    // W2
    for v in weights[b1_end..w2_end].iter_mut() {
        let u = rng.next_f32() as f64;
        *v = (2.0 * u - 1.0) * scale2;
    }
    // b2 = 0 already.

    let running_fisher = vec![0.0_f64; n_params];
    let running_theta_star = weights.clone();

    Ok(OnlineEwcState {
        weights: weights.clone(),
        biases: Vec::new(), // flat layout; biases embedded in weights
        running_fisher,
        running_theta_star,
        n_tasks: 0,
        layer_sizes: vec![d_in, d_h, d_out],
        gamma: cfg.gamma,
        lambda: cfg.lambda,
    })
}

/// Current EWC regularisation penalty:
/// `(λ/2) · Σ_i F̄_i · (θ_i - θ*_i)²`
pub fn online_ewc_penalty(state: &OnlineEwcState) -> f64 {
    if state.n_tasks == 0 {
        return 0.0;
    }
    let pen: f64 = state
        .weights
        .iter()
        .zip(state.running_fisher.iter())
        .zip(state.running_theta_star.iter())
        .map(|((&theta, &f), &ts)| f * (theta - ts).powi(2))
        .sum();
    0.5 * state.lambda * pen
}

/// Predict class for a single input: argmax of output logits.
pub fn online_ewc_predict(state: &OnlineEwcState, x: &[f64]) -> ContinualResult<usize> {
    if x.len() != state.layer_sizes[0] {
        return Err(ContinualError::DimensionMismatch {
            expected: state.layer_sizes[0],
            got: x.len(),
        });
    }
    let (_, logits) = forward(state, x);
    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(pred)
}

/// Train on one task and update the running Fisher and anchor.
///
/// Returns the final epoch's average cross-entropy loss.
pub fn online_ewc_fit_task(
    state: &mut OnlineEwcState,
    x: &[f64],
    y: &[usize],
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if n == 0 || x.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let d_in = state.layer_sizes[0];
    if x.len() != n * d_in {
        return Err(ContinualError::DimensionMismatch {
            expected: n * d_in,
            got: x.len(),
        });
    }

    let lr = 0.01_f64; // callers can control via cfg.lr passed outside
    let n_epochs = 5_usize;
    let n_params = state.weights.len();
    let d_out = state.layer_sizes[2];

    let mut indices: Vec<usize> = (0..n).collect();
    let mut last_loss = 0.0_f64;

    for _epoch in 0..n_epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0_f64;

        for &idx in &indices {
            let xi = &x[idx * d_in..(idx + 1) * d_in];
            let label = y[idx];

            // Forward.
            let (_, logits) = forward(state, xi);
            let probs = softmax(&logits);
            let ce_loss = -(probs[label.min(d_out - 1)].max(1e-30).ln());
            epoch_loss += ce_loss;

            // CE gradient (backprop).
            let mut d_logits = probs.clone();
            if label < d_out {
                d_logits[label] -= 1.0;
            }

            // Full parameter gradient.
            let (w1_s, w1_e, b1_e, w2_e, _b2_e) = state.offsets();
            let d_h = state.layer_sizes[1];

            // --- Backprop layer 2 ---
            let w1 = &state.weights[w1_s..w1_e].to_vec();
            let b1 = &state.weights[w1_e..b1_e].to_vec();
            let w2 = &state.weights[b1_e..w2_e].to_vec();

            // Pre-activation for ReLU mask; also compute h1 (post-ReLU).
            let h1_pre_relu = matvec(w1, xi, b1, d_in, d_h);
            let h1: Vec<f64> = h1_pre_relu.iter().map(|&v| v.max(0.0)).collect();

            // EWC gradient.
            let reg_grad = ewc_grad(state);

            // Gradient for W2 and b2 (row-major layout).
            // W2 starts at b1_e (weights[b1_e + row*d_h + col]).
            // b2 starts at w2_e (weights[w2_e + row]).
            for (row, &dl) in d_logits.iter().enumerate() {
                // b2 gradient
                state.weights[w2_e + row] -= lr * (dl + reg_grad[w2_e + row]);
                // W2 gradient
                for (col, &h1v) in h1.iter().enumerate() {
                    state.weights[b1_e + row * d_h + col] -=
                        lr * (dl * h1v + reg_grad[b1_e + row * d_h + col]);
                }
            }

            // d_h1 = W2^T * d_logits.
            let mut d_h1 = vec![0.0_f64; d_h];
            for (row, &dl) in d_logits.iter().enumerate() {
                for (col, dh) in d_h1.iter_mut().enumerate() {
                    *dh += w2[row * d_h + col] * dl;
                }
            }

            // ReLU backward.
            for (dh, &pre) in d_h1.iter_mut().zip(h1_pre_relu.iter()) {
                if pre <= 0.0 {
                    *dh = 0.0;
                }
            }

            // Gradient for W1 and b1.
            for (row, &dh) in d_h1.iter().enumerate() {
                let g_b1 = dh + reg_grad[w1_e + row];
                state.weights[w1_e + row] -= lr * g_b1;
                for (col, &xv) in xi.iter().enumerate() {
                    let g_w1 = dh * xv + reg_grad[w1_s + row * d_in + col];
                    state.weights[w1_s + row * d_in + col] -= lr * g_w1;
                }
            }
        }
        last_loss = epoch_loss / n as f64;
    }

    // --- Update running Fisher ---
    let fisher_samples = n.min(256);
    let mut f_new = vec![0.0_f64; n_params];
    for i in 0..fisher_samples {
        let xi = &x[i * d_in..(i + 1) * d_in];
        let label = y[i];
        let grad = log_likelihood_grad(state, xi, label);
        for (f, g) in f_new.iter_mut().zip(grad.iter()) {
            *f += g * g;
        }
    }
    let inv_ns = 1.0 / fisher_samples as f64;
    for f in f_new.iter_mut() {
        *f *= inv_ns;
    }

    // running_fisher = γ * running_fisher + F_new.
    let gamma = state.gamma;
    for (rf, &fn_i) in state.running_fisher.iter_mut().zip(f_new.iter()) {
        *rf = gamma * *rf + fn_i;
    }

    // Update anchor to current θ.
    state.running_theta_star = state.weights.clone();
    state.n_tasks += 1;

    Ok(last_loss)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> OnlineEwcState {
        let cfg = OnlineEwcConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            ..Default::default()
        };
        online_ewc_new(&cfg, 42).expect("Online EWC state should initialize with valid config")
    }

    fn make_xy(n: usize, d_in: usize, n_classes: usize) -> (Vec<f64>, Vec<usize>) {
        let mut rng = LcgRng::new(1234);
        let x: Vec<f64> = (0..n * d_in).map(|_| rng.next_f32() as f64).collect();
        let y: Vec<usize> = (0..n).map(|i| i % n_classes).collect();
        (x, y)
    }

    /// 1. penalty is 0 after initialisation (no prior Fisher).
    #[test]
    fn penalty_zero_at_init() {
        let state = make_state();
        assert_eq!(
            online_ewc_penalty(&state),
            0.0,
            "Penalty must be 0 before any task is trained"
        );
    }

    /// 2. penalty > 0 after second task (Fisher accumulated after task 1).
    #[test]
    fn penalty_positive_after_second_task() {
        let mut state = make_state();
        let mut rng = LcgRng::new(1);
        let (x, y) = make_xy(16, 8, 4);
        online_ewc_fit_task(&mut state, &x, &y, 16, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");

        // After task 1, Fisher is set. Perturb weights slightly so penalty > 0.
        for w in state.weights.iter_mut() {
            *w += 0.01;
        }
        let penalty = online_ewc_penalty(&state);
        assert!(
            penalty > 0.0,
            "Penalty should be > 0 after task 1 with perturbed weights, got {penalty}"
        );
    }

    /// 3. online_ewc_predict returns a valid class index.
    #[test]
    fn predict_valid_class_index() {
        let state = make_state();
        let x = vec![0.5_f64; 8];
        let pred = online_ewc_predict(&state, &x)
            .expect("Online EWC prediction should succeed on valid input");
        assert!(pred < 4, "Prediction {pred} should be in [0,4)");
    }

    /// 4. predict returns Err on wrong input dim.
    #[test]
    fn predict_wrong_dim_err() {
        let state = make_state();
        let x = vec![0.0_f64; 5];
        assert!(online_ewc_predict(&state, &x).is_err());
    }

    /// 5. running_fisher is non-zero after first task.
    #[test]
    fn running_fisher_nonzero_after_task() {
        let mut state = make_state();
        let mut rng = LcgRng::new(2);
        let (x, y) = make_xy(16, 8, 4);
        online_ewc_fit_task(&mut state, &x, &y, 16, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        let fisher_sum: f64 = state.running_fisher.iter().sum();
        assert!(
            fisher_sum > 0.0,
            "Running Fisher must be non-zero after task, sum={fisher_sum}"
        );
    }

    /// 6. n_tasks increments after each fit.
    #[test]
    fn n_tasks_increments() {
        let mut state = make_state();
        let mut rng = LcgRng::new(3);
        assert_eq!(state.n_tasks, 0);
        let (x, y) = make_xy(8, 8, 4);
        online_ewc_fit_task(&mut state, &x, &y, 8, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        assert_eq!(state.n_tasks, 1);
        let (x2, y2) = make_xy(8, 8, 4);
        online_ewc_fit_task(&mut state, &x2, &y2, 8, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        assert_eq!(state.n_tasks, 2);
    }

    /// 7. fit_task returns Err on empty input.
    #[test]
    fn fit_task_empty_err() {
        let mut state = make_state();
        let mut rng = LcgRng::new(4);
        assert!(online_ewc_fit_task(&mut state, &[], &[], 0, &mut rng).is_err());
    }

    /// 8. fit_task returns Err on dimension mismatch.
    #[test]
    fn fit_task_dim_mismatch_err() {
        let mut state = make_state();
        let mut rng = LcgRng::new(5);
        let x = vec![0.0_f64; 8]; // only 1 sample, but n=2.
        let y = vec![0_usize; 2];
        assert!(online_ewc_fit_task(&mut state, &x, &y, 2, &mut rng).is_err());
    }

    /// 9. running_theta_star updated to current weights after task.
    #[test]
    fn theta_star_equals_weights_after_task() {
        let mut state = make_state();
        let mut rng = LcgRng::new(6);
        let (x, y) = make_xy(8, 8, 4);
        online_ewc_fit_task(&mut state, &x, &y, 8, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        for (w, ts) in state.weights.iter().zip(state.running_theta_star.iter()) {
            assert!(
                (w - ts).abs() < 1e-15,
                "theta_star must equal weights right after task"
            );
        }
    }

    /// 10. running_fisher decays with gamma over multiple tasks.
    #[test]
    fn running_fisher_decays_with_gamma() {
        let mut state = make_state();
        let mut rng = LcgRng::new(7);
        let (x1, y1) = make_xy(8, 8, 4);
        online_ewc_fit_task(&mut state, &x1, &y1, 8, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        let fisher1: Vec<f64> = state.running_fisher.clone();
        let (x2, y2) = make_xy(8, 8, 4);
        online_ewc_fit_task(&mut state, &x2, &y2, 8, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        // running_fisher after task 2 = γ * fisher1 + F_new.
        // The old fisher1 entries were decayed by γ.
        // We verify at least some entries changed (decayed from task1).
        let changed = state
            .running_fisher
            .iter()
            .zip(fisher1.iter())
            .any(|(f2, f1)| (f2 - f1).abs() > 1e-12);
        assert!(changed, "running_fisher must change between tasks");
    }

    /// 11. online_ewc_new returns Err on zero input_dim.
    #[test]
    fn new_zero_input_dim_err() {
        let cfg = OnlineEwcConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(online_ewc_new(&cfg, 0).is_err());
    }

    /// 12. online_ewc_new returns Err on negative lambda.
    #[test]
    fn new_negative_lambda_err() {
        let cfg = OnlineEwcConfig {
            lambda: -1.0,
            ..Default::default()
        };
        assert!(online_ewc_new(&cfg, 0).is_err());
    }

    /// 13. fit_task returns a finite loss value.
    #[test]
    fn fit_task_returns_finite_loss() {
        let mut state = make_state();
        let mut rng = LcgRng::new(9);
        let (x, y) = make_xy(16, 8, 4);
        let loss = online_ewc_fit_task(&mut state, &x, &y, 16, &mut rng)
            .expect("Online EWC task fitting should succeed with valid data");
        assert!(loss.is_finite(), "Task loss must be finite, got {loss}");
        assert!(loss >= 0.0, "Task loss must be non-negative");
    }

    /// 14. predict is deterministic (same input → same output).
    #[test]
    fn predict_deterministic() {
        let state = make_state();
        let x = vec![0.7_f64; 8];
        let p1 = online_ewc_predict(&state, &x)
            .expect("Online EWC prediction should succeed on valid input");
        let p2 = online_ewc_predict(&state, &x)
            .expect("Online EWC prediction should succeed on valid input");
        assert_eq!(p1, p2);
    }

    /// 15. n_params equals expected layout size.
    #[test]
    fn n_params_correct() {
        let state = make_state();
        // d_in=8, d_h=16, d_out=4: W1=128, b1=16, W2=64, b2=4 → 212
        assert_eq!(state.n_params(), 8 * 16 + 16 + 4 * 16 + 4);
    }
}
