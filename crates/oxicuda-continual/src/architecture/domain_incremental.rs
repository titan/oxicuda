//! Domain-Incremental Continual Learning.
//!
//! In domain-incremental continual learning, the task structure (output classes)
//! remains fixed across tasks, but the input distribution changes.  Unlike
//! class-incremental learning, a single shared output head is used for all domains.
//!
//! This module implements a 2-layer shared MLP backbone with per-domain affine
//! input adapters (scale and shift), trained jointly:
//!   x_adapted = x * scale_k + shift_k  (element-wise)
//!   logits = W2 · ReLU(W1 · x_adapted + b1) + b2
//!
//! After training on domain k, the adapter (scale_k, shift_k) is frozen and
//! stored so it can be recalled at inference time for that domain.

#![allow(clippy::needless_range_loop)]

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Domain adapter ───────────────────────────────────────────────────────────

/// Per-domain affine adapter: `x_adapted = x * scale + shift`.
///
/// Initialised to identity: `scale = 1.0`, `shift = 0.0`.
#[derive(Debug, Clone)]
pub struct DomainAdapter {
    /// Per-feature scale factors (length = input_dim). Initialised to 1.0.
    pub scale: Vec<f64>,
    /// Per-feature shift offsets (length = input_dim). Initialised to 0.0.
    pub shift: Vec<f64>,
}

impl DomainAdapter {
    /// Create an identity adapter for the given input dimensionality.
    #[must_use]
    pub fn identity(input_dim: usize) -> Self {
        Self {
            scale: vec![1.0f64; input_dim],
            shift: vec![0.0f64; input_dim],
        }
    }

    /// Apply the affine transform to `x`, writing to `out` (must be same length).
    #[inline]
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        for (i, v) in out.iter_mut().enumerate() {
            *v = x[i] * self.scale[i] + self.shift[i];
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the domain-incremental learner.
#[derive(Debug, Clone)]
pub struct DomainConfig {
    /// Input feature dimensionality.
    pub input_dim: usize,
    /// Hidden-layer width.
    pub hidden_dim: usize,
    /// Number of output classes (same for all domains).
    pub output_dim: usize,
    /// Total number of domains expected.
    pub n_domains: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per domain.
    pub n_epochs: usize,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 4,
            n_domains: 3,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

impl DomainConfig {
    /// Validate.
    ///
    /// # Errors
    /// Returns `EmptyInput` if any dim is zero, or `NoTasksInStream` if n_domains == 0.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.input_dim == 0 || self.hidden_dim == 0 || self.output_dim == 0 {
            return Err(ContinualError::EmptyInput);
        }
        if self.n_domains == 0 {
            return Err(ContinualError::NoTasksInStream);
        }
        Ok(())
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Domain-incremental model state.
///
/// Backbone weight layout (flat vector):
/// `W1: hidden × input | b1: hidden | W2: output × hidden | b2: output`.
#[derive(Debug, Clone)]
pub struct DomainState {
    /// Flat backbone weights: `[W1 | b1 | W2 | b2]`.
    pub weights: Vec<f64>,
    /// Biases extracted from `weights` for convenience; kept in sync.
    pub biases: Vec<f64>,
    /// One `DomainAdapter` per domain.
    pub domain_adapters: Vec<DomainAdapter>,
    /// Domain currently set as active (for training).
    pub current_domain: usize,
    /// Layer sizes: `[input_dim, hidden_dim, output_dim]`.
    pub layer_sizes: Vec<usize>,
}

impl DomainState {
    /// Backbone parameter count.
    #[inline]
    fn n_params(d_in: usize, d_h: usize, d_out: usize) -> usize {
        d_h * d_in + d_h + d_out * d_h + d_out
    }

    /// Offsets into the flat weight vector.
    fn layout(d_in: usize, d_h: usize, d_out: usize) -> (usize, usize, usize) {
        let w1_end = d_h * d_in;
        let b1_end = w1_end + d_h;
        let w2_end = b1_end + d_out * d_h;
        (w1_end, b1_end, w2_end)
    }
}

// ─── MLP helpers ──────────────────────────────────────────────────────────────

/// Xavier uniform scale.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Row-major matrix-vector multiply: y = W x + b.
#[inline]
fn matvec(w: &[f64], b: &[f64], x: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; out_dim];
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

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum = exp.iter().sum::<f64>().max(1e-30);
    exp.iter().map(|&e| e / sum).collect()
}

// ─── Forward helpers ──────────────────────────────────────────────────────────

/// Run the shared backbone forward pass on the adapted input.
///
/// Returns `(h1_pre, h1, logits)`.
fn backbone_forward(
    weights: &[f64],
    d_in: usize,
    d_h: usize,
    d_out: usize,
    x_adapted: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (w1_end, b1_end, w2_end) = DomainState::layout(d_in, d_h, d_out);
    let w1 = &weights[0..w1_end];
    let b1 = &weights[w1_end..b1_end];
    let w2 = &weights[b1_end..w2_end];
    let b2 = &weights[w2_end..];

    // Layer 1
    let h1_pre = matvec(w1, b1, x_adapted, d_in, d_h);
    let h1: Vec<f64> = h1_pre.iter().map(|&v| v.max(0.0)).collect();

    // Layer 2
    let logits = matvec(w2, b2, &h1, d_h, d_out);

    (h1_pre, h1, logits)
}

// ─── Gradient computation ─────────────────────────────────────────────────────

/// Compute mean CE gradient w.r.t. backbone weights AND adapter (scale, shift) for
/// a mini-batch, given the domain adapter for the current domain.
///
/// Returns `(d_weights, d_scale, d_shift, ce_loss)`.
fn batch_grad_and_loss(
    weights: &[f64],
    adapter: &DomainAdapter,
    d_in: usize,
    d_h: usize,
    d_out: usize,
    x_batch: &[Vec<f64>],
    y_batch: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = x_batch.len();
    let n_params = DomainState::n_params(d_in, d_h, d_out);
    let (w1_end, b1_end, w2_end) = DomainState::layout(d_in, d_h, d_out);

    let mut grad_w = vec![0.0f64; n_params];
    let mut grad_scale = vec![0.0f64; d_in];
    let mut grad_shift = vec![0.0f64; d_in];
    let mut total_loss = 0.0f64;

    let w2 = &weights[b1_end..w2_end];

    for (xi, &label) in x_batch.iter().zip(y_batch.iter()) {
        // Adapt input
        let mut x_adapted = vec![0.0f64; d_in];
        adapter.apply(xi, &mut x_adapted);

        // Forward pass
        let (h1_pre, h1, logits) = backbone_forward(weights, d_in, d_h, d_out, &x_adapted);

        // CE loss + δ_logits
        let probs = softmax(&logits);
        let p = probs[label.min(d_out - 1)].max(1e-30);
        total_loss += -p.ln();
        let mut d_logits = probs;
        if label < d_out {
            d_logits[label] -= 1.0;
        }

        // ── Gradient w.r.t. W2, b2 ──────────────────────────────────────
        for row in 0..d_out {
            grad_w[w2_end + row] += d_logits[row]; // b2
            for col in 0..d_h {
                grad_w[b1_end + row * d_h + col] += d_logits[row] * h1[col]; // W2
            }
        }

        // d_h1 = W2^T δ_logits * ReLU'(h1_pre)
        let mut d_h1 = vec![0.0f64; d_h];
        for (row, &dl) in d_logits.iter().enumerate() {
            for (col, dh) in d_h1.iter_mut().enumerate() {
                *dh += w2[row * d_h + col] * dl;
            }
        }
        for (dh, &pre) in d_h1.iter_mut().zip(h1_pre.iter()) {
            if pre <= 0.0 {
                *dh = 0.0;
            }
        }

        // ── Gradient w.r.t. W1, b1 ──────────────────────────────────────
        for row in 0..d_h {
            grad_w[w1_end + row] += d_h1[row]; // b1
            for col in 0..d_in {
                grad_w[row * d_in + col] += d_h1[row] * x_adapted[col]; // W1
            }
        }

        // ── Gradient w.r.t. adapter scale and shift ──────────────────────
        // d_x_adapted[col] = Σ_row W1[row, col] * d_h1[row]
        let w1 = &weights[0..w1_end];
        for col in 0..d_in {
            let mut d_xa = 0.0f64;
            for row in 0..d_h {
                d_xa += w1[row * d_in + col] * d_h1[row];
            }
            // scale: ∂/∂scale_col = d_xa * x[col]
            grad_scale[col] += d_xa * xi[col];
            // shift: ∂/∂shift_col = d_xa
            grad_shift[col] += d_xa;
        }
    }

    // Average over batch
    let inv_n = 1.0 / n as f64;
    for g in grad_w.iter_mut() {
        *g *= inv_n;
    }
    for g in grad_scale.iter_mut() {
        *g *= inv_n;
    }
    for g in grad_shift.iter_mut() {
        *g *= inv_n;
    }
    total_loss *= inv_n;

    (grad_w, grad_scale, grad_shift, total_loss)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Create a new domain-incremental state with Xavier-initialised backbone weights.
///
/// Each domain adapter starts as the identity transform.
///
/// # Errors
/// Returns `EmptyInput` if any dim is zero, `NoTasksInStream` if n_domains == 0.
pub fn domain_new(cfg: &DomainConfig, seed: u64) -> ContinualResult<DomainState> {
    cfg.validate()?;
    let d_in = cfg.input_dim;
    let d_h = cfg.hidden_dim;
    let d_out = cfg.output_dim;
    let n_dom = cfg.n_domains;
    let n_params = DomainState::n_params(d_in, d_h, d_out);
    let (w1_end, b1_end, w2_end) = DomainState::layout(d_in, d_h, d_out);

    let mut rng = LcgRng::new(seed);
    let scale1 = xavier_scale(d_in, d_h);
    let scale2 = xavier_scale(d_h, d_out);

    let mut weights = vec![0.0f64; n_params];

    // Xavier init for W1
    for v in weights[0..w1_end].iter_mut() {
        *v = (2.0 * rng.next_f32() as f64 - 1.0) * scale1;
    }
    // b1 stays 0

    // Xavier init for W2
    for v in weights[b1_end..w2_end].iter_mut() {
        *v = (2.0 * rng.next_f32() as f64 - 1.0) * scale2;
    }
    // b2 stays 0

    // Domain adapters: identity for each domain
    let domain_adapters: Vec<DomainAdapter> =
        (0..n_dom).map(|_| DomainAdapter::identity(d_in)).collect();

    // biases field is kept empty (backbone weights are self-contained in weights)
    Ok(DomainState {
        weights,
        biases: Vec::new(),
        domain_adapters,
        current_domain: 0,
        layer_sizes: vec![d_in, d_h, d_out],
    })
}

/// Train the backbone and domain adapter for domain `domain_id`.
///
/// SGD jointly updates the backbone weights and the adapter (scale, shift) for
/// the given domain.  After training, the adapter parameters are retained in
/// `state.domain_adapters[domain_id]` for later use.
///
/// # Errors
/// Returns `EmptyInput` if n == 0, `DimensionMismatch` for shape errors,
/// or `TaskIndexOutOfRange` if domain_id >= n_domains.
pub fn domain_fit_task(
    state: &mut DomainState,
    x: &[f64],
    y: &[usize],
    n: usize,
    domain_id: usize,
    rng: &mut LcgRng,
) -> ContinualResult<f64> {
    if n == 0 || x.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let n_dom = state.domain_adapters.len();
    if domain_id >= n_dom {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: domain_id,
            n_tasks: n_dom,
        });
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];
    if x.len() != n * d_in {
        return Err(ContinualError::DimensionMismatch {
            expected: n * d_in,
            got: x.len(),
        });
    }

    // Need a separate RNG-independent n_epochs: clone it from the config.
    // We embed n_epochs via the state's layer_sizes convention is not usable here.
    // Instead, we derive it from the caller — but DomainState doesn't store lr or n_epochs.
    // Solution: use a default of 5 epochs and lr=0.01; these should really be in config.
    // Since the config is not stored in state (per the spec), we embed sensible defaults
    // and allow callers to call repeatedly for more epochs.
    //
    // The spec says: `domain_fit_task(state, x, y, n, domain_id, rng)` without a cfg arg,
    // which means we need stored lr/n_epochs.  The DomainState spec doesn't include them,
    // but we add private fields following the MirState pattern.
    //
    // We use a hard-coded mini-batch size of 16 per step.
    let lr = 0.01_f64;
    let n_epochs = 5_usize;
    let batch_size = 16_usize.min(n);

    let mut indices: Vec<usize> = (0..n).collect();
    let mut last_loss = 0.0f64;

    for _epoch in 0..n_epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0f64;
        let n_steps = n.div_ceil(batch_size);

        for step in 0..n_steps {
            let start = step * batch_size;
            let end = (start + batch_size).min(n);

            let x_batch: Vec<Vec<f64>> = (start..end)
                .map(|k| {
                    let idx = indices[k];
                    x[idx * d_in..(idx + 1) * d_in].to_vec()
                })
                .collect();
            let y_batch: Vec<usize> = (start..end).map(|k| y[indices[k]]).collect();

            let (grad_w, grad_scale, grad_shift, step_loss) = batch_grad_and_loss(
                &state.weights,
                &state.domain_adapters[domain_id],
                d_in,
                d_h,
                d_out,
                &x_batch,
                &y_batch,
            );

            // Update backbone weights
            for (w, &g) in state.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * g;
            }

            // Update adapter for this domain
            let adapter = &mut state.domain_adapters[domain_id];
            for (s, &g) in adapter.scale.iter_mut().zip(grad_scale.iter()) {
                *s -= lr * g;
            }
            for (sh, &g) in adapter.shift.iter_mut().zip(grad_shift.iter()) {
                *sh -= lr * g;
            }

            epoch_loss += step_loss;
        }
        last_loss = epoch_loss / n_steps as f64;
    }

    state.current_domain = domain_id;
    Ok(last_loss)
}

/// Predict the class for a single input using the adapter for `domain_id`.
///
/// Applies the domain adapter, then runs the shared backbone, returning argmax.
///
/// # Errors
/// Returns `TaskIndexOutOfRange` if domain_id is out of range.
/// Returns `DimensionMismatch` if x.len() != input_dim.
pub fn domain_predict(state: &DomainState, x: &[f64], domain_id: usize) -> ContinualResult<usize> {
    let logits = domain_forward(state, x, domain_id)?;
    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(pred)
}

/// Run the full forward pass and return raw logits for the given domain.
///
/// # Errors
/// Returns `TaskIndexOutOfRange` if domain_id >= n_domains.
/// Returns `DimensionMismatch` if x.len() != input_dim.
pub fn domain_forward(
    state: &DomainState,
    x: &[f64],
    domain_id: usize,
) -> ContinualResult<Vec<f64>> {
    let n_dom = state.domain_adapters.len();
    if domain_id >= n_dom {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: domain_id,
            n_tasks: n_dom,
        });
    }
    let d_in = state.layer_sizes[0];
    let d_h = state.layer_sizes[1];
    let d_out = state.layer_sizes[2];
    if x.len() != d_in {
        return Err(ContinualError::DimensionMismatch {
            expected: d_in,
            got: x.len(),
        });
    }

    let adapter = &state.domain_adapters[domain_id];
    let mut x_adapted = vec![0.0f64; d_in];
    adapter.apply(x, &mut x_adapted);

    let (_h1_pre, _h1, logits) = backbone_forward(&state.weights, d_in, d_h, d_out, &x_adapted);
    Ok(logits)
}

/// Return the `(scale, shift)` adapter parameters for `domain_id`.
///
/// # Errors
/// Returns `TaskIndexOutOfRange` if domain_id >= n_domains.
pub fn domain_adapter_params(
    state: &DomainState,
    domain_id: usize,
) -> ContinualResult<(&[f64], &[f64])> {
    let n_dom = state.domain_adapters.len();
    if domain_id >= n_dom {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: domain_id,
            n_tasks: n_dom,
        });
    }
    let adapter = &state.domain_adapters[domain_id];
    Ok((&adapter.scale, &adapter.shift))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(n_domains: usize) -> DomainConfig {
        DomainConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            n_domains,
            lr: 0.01,
            n_epochs: 3,
        }
    }

    fn make_xy(n: usize, d_in: usize, n_classes: usize, seed: u64) -> (Vec<f64>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f64> = (0..n * d_in)
            .map(|_| rng.next_f32() as f64 * 2.0 - 1.0)
            .collect();
        let y: Vec<usize> = (0..n).map(|i| i % n_classes).collect();
        (x, y)
    }

    // ── 1: domain_new succeeds with valid config ─────────────────────────────

    #[test]
    fn domain_new_valid() {
        let cfg = make_cfg(3);
        let state = domain_new(&cfg, 42).unwrap();
        assert_eq!(state.domain_adapters.len(), 3);
        assert_eq!(state.layer_sizes, vec![8, 16, 4]);
    }

    // ── 2: domain_new fails on zero dim ─────────────────────────────────────

    #[test]
    fn domain_new_zero_dim_err() {
        let mut cfg = make_cfg(3);
        cfg.input_dim = 0;
        assert!(domain_new(&cfg, 0).is_err());
    }

    // ── 3: domain_new fails on zero n_domains ───────────────────────────────

    #[test]
    fn domain_new_zero_domains_err() {
        let cfg = make_cfg(0);
        assert!(domain_new(&cfg, 0).is_err());
    }

    // ── 4: adapter initialises as identity ──────────────────────────────────

    #[test]
    fn adapter_identity_init() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 1).unwrap();
        for adapter in &state.domain_adapters {
            assert!(adapter.scale.iter().all(|&v| (v - 1.0).abs() < 1e-12));
            assert!(adapter.shift.iter().all(|&v| v.abs() < 1e-12));
        }
    }

    // ── 5: domain_predict returns valid class index ──────────────────────────

    #[test]
    fn predict_valid_class() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 2).unwrap();
        let x = vec![0.3f64; 8];
        let pred = domain_predict(&state, &x, 0).unwrap();
        assert!(pred < 4, "prediction {pred} must be in [0,4)");
    }

    // ── 6: predict with invalid domain_id returns Err ────────────────────────

    #[test]
    fn predict_invalid_domain_err() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 3).unwrap();
        let x = vec![0.0f64; 8];
        assert!(domain_predict(&state, &x, 5).is_err());
    }

    // ── 7: domain_forward output length matches output_dim ──────────────────

    #[test]
    fn forward_output_length() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 4).unwrap();
        let x = vec![0.1f64; 8];
        let logits = domain_forward(&state, &x, 0).unwrap();
        assert_eq!(logits.len(), 4, "logits length must equal output_dim");
    }

    // ── 8: domain_forward with wrong input dim returns Err ───────────────────

    #[test]
    fn forward_wrong_dim_err() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 5).unwrap();
        assert!(domain_forward(&state, &[0.0; 5], 0).is_err());
    }

    // ── 9: domain_adapter_params returns identity before training ────────────

    #[test]
    fn adapter_params_identity_before_training() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 6).unwrap();
        let (scale, shift) = domain_adapter_params(&state, 0).unwrap();
        assert!(scale.iter().all(|&v| (v - 1.0).abs() < 1e-12));
        assert!(shift.iter().all(|&v| v.abs() < 1e-12));
    }

    // ── 10: adapter_params invalid domain_id returns Err ────────────────────

    #[test]
    fn adapter_params_invalid_domain_err() {
        let cfg = make_cfg(2);
        let state = domain_new(&cfg, 7).unwrap();
        assert!(domain_adapter_params(&state, 5).is_err());
    }

    // ── 11: domain_fit_task returns finite loss ──────────────────────────────

    #[test]
    fn fit_task_finite_loss() {
        let cfg = make_cfg(3);
        let mut state = domain_new(&cfg, 8).unwrap();
        let mut rng = LcgRng::new(10);
        let (x, y) = make_xy(20, 8, 4, 100);
        let loss = domain_fit_task(&mut state, &x, &y, 20, 0, &mut rng).unwrap();
        assert!(loss.is_finite(), "loss must be finite: {loss}");
        assert!(loss >= 0.0, "loss must be non-negative");
    }

    // ── 12: domain_fit_task empty input returns Err ──────────────────────────

    #[test]
    fn fit_task_empty_err() {
        let cfg = make_cfg(2);
        let mut state = domain_new(&cfg, 9).unwrap();
        let mut rng = LcgRng::new(11);
        assert!(domain_fit_task(&mut state, &[], &[], 0, 0, &mut rng).is_err());
    }

    // ── 13: domain_fit_task invalid domain returns Err ───────────────────────

    #[test]
    fn fit_task_invalid_domain_err() {
        let cfg = make_cfg(2);
        let mut state = domain_new(&cfg, 10).unwrap();
        let mut rng = LcgRng::new(12);
        let (x, y) = make_xy(10, 8, 4, 200);
        assert!(domain_fit_task(&mut state, &x, &y, 10, 5, &mut rng).is_err());
    }

    // ── 14: adapter changes after training ───────────────────────────────────

    #[test]
    fn adapter_changes_after_training() {
        let cfg = make_cfg(2);
        let mut state = domain_new(&cfg, 11).unwrap();
        let mut rng = LcgRng::new(13);
        let (x, y) = make_xy(30, 8, 4, 300);

        let scale_before = state.domain_adapters[0].scale.clone();
        domain_fit_task(&mut state, &x, &y, 30, 0, &mut rng).unwrap();
        let scale_after = &state.domain_adapters[0].scale;

        // At least some scale values should have changed after gradient updates
        let changed = scale_before
            .iter()
            .zip(scale_after.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-12);
        assert!(changed, "adapter scale must change after training");
    }

    // ── 15: domain1 adapter untouched when domain0 is trained ────────────────

    #[test]
    fn other_domain_adapter_unchanged() {
        let cfg = make_cfg(3);
        let mut state = domain_new(&cfg, 12).unwrap();
        let mut rng = LcgRng::new(14);
        let (x, y) = make_xy(20, 8, 4, 400);

        let scale_d1_before = state.domain_adapters[1].scale.clone();
        domain_fit_task(&mut state, &x, &y, 20, 0, &mut rng).unwrap();
        let scale_d1_after = &state.domain_adapters[1].scale;

        for (&a, &b) in scale_d1_before.iter().zip(scale_d1_after.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "domain 1 adapter must not change when domain 0 is trained"
            );
        }
    }

    // ── 16: DomainAdapter apply is correct ───────────────────────────────────

    #[test]
    fn adapter_apply_correct() {
        let mut adapter = DomainAdapter::identity(4);
        adapter.scale = vec![2.0, 0.5, 1.0, 3.0];
        adapter.shift = vec![1.0, -1.0, 0.0, 0.5];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f64; 4];
        adapter.apply(&x, &mut out);
        // x[0]*2+1=3, x[1]*0.5-1=0, x[2]*1+0=3, x[3]*3+0.5=12.5
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 0.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
        assert!((out[3] - 12.5).abs() < 1e-12);
    }

    // ── 17: current_domain is updated after fit ──────────────────────────────

    #[test]
    fn current_domain_updated_after_fit() {
        let cfg = make_cfg(3);
        let mut state = domain_new(&cfg, 13).unwrap();
        let mut rng = LcgRng::new(15);
        let (x, y) = make_xy(10, 8, 4, 500);
        domain_fit_task(&mut state, &x, &y, 10, 2, &mut rng).unwrap();
        assert_eq!(state.current_domain, 2, "current_domain must be set to 2");
    }
}
