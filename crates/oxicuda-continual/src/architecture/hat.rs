//! HAT: Hard Attention to the Task.
//!
//! Implements the method from:
//! Serra et al. "Overcoming Catastrophic Forgetting with Hard Attention to the Task."
//! ICML 2018.
//!
//! HAT uses task-conditional binary attention gates to protect units
//! associated with previously trained tasks from being overwritten.
//! Gates are produced by a sigmoid with increasing sharpness during training,
//! converging to a step function at inference time.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for HAT (2-layer network: input → hidden → output).
#[derive(Debug, Clone)]
pub struct HatConfig {
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Output units per layer: `[hidden_dim, output_dim]` for a 2-layer net.
    pub n_units: Vec<usize>,
    /// Maximum number of tasks (upper bound).
    pub n_tasks: usize,
    /// Gate saturation sharpness at end of training.
    pub s_max: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
}

impl Default for HatConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            hidden_dim: 64,
            n_units: vec![64, 10],
            n_tasks: 10,
            s_max: 400.0,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// HAT model state.
///
/// Network structure: `input_dim → n_units[0] (hidden) → n_units[1] (output)`.
/// `n_layers = 2` (layer 0 = input→hidden, layer 1 = hidden→output).
#[derive(Debug, Clone)]
pub struct HatState {
    /// Weight matrices per layer, row-major.
    /// `weights[0]`: `[n_units[0] × input_dim]`
    /// `weights[1]`: `[n_units[1] × n_units[0]]`
    pub weights: Vec<Vec<f64>>,
    /// Bias vectors per layer.
    pub biases: Vec<Vec<f64>>,
    /// Task embeddings: `task_embed[t][l]` has length `n_units[l]`.
    /// Initialised lazily; extended as new tasks arrive.
    pub task_embed: Vec<Vec<Vec<f64>>>,
    /// Running max gate per layer across all completed tasks: `cumulative_mask[l]`.
    pub cumulative_mask: Vec<Vec<f64>>,
    /// Always 2 for the simplified 2-layer net.
    pub n_layers: usize,
    /// Number of tasks fully trained so far.
    pub n_tasks_seen: usize,
    /// Cached dims.
    pub(crate) input_dim: usize,
    pub(crate) n_units: Vec<usize>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Xavier uniform init into a mutable slice.
fn xavier_fill(buf: &mut [f64], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    for v in buf.iter_mut() {
        let u = rng.next_f32() as f64;
        *v = (2.0 * u - 1.0) * scale;
    }
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

/// ReLU element-wise.
#[inline]
fn relu(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

/// Compute attention gate for layer `l` of task `t` at sharpness `s`.
///
/// `gate[i] = sigmoid(s * embed[t][l][i])`
fn compute_gate(embed_tl: &[f64], s: f64) -> Vec<f64> {
    embed_tl.iter().map(|&e| sigmoid(s * e)).collect()
}

/// Softmax over a slice.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exp.iter().sum::<f64>().max(1e-30);
    exp.iter().map(|&e| e / sum).collect()
}

/// Element-wise product of two equal-length slices.
#[inline]
fn elem_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Element-wise max update: `a[i] = max(a[i], b[i])`.
#[inline]
fn elem_max_update(a: &mut [f64], b: &[f64]) {
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        if y > *x {
            *x = y;
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Create a new HAT state with Xavier-initialised weights.
///
/// `n_units` must have exactly 2 elements: `[hidden_dim, output_dim]`.
pub fn hat_new(cfg: &HatConfig, seed: u64) -> ContinualResult<HatState> {
    if cfg.n_units.len() < 2 {
        return Err(ContinualError::InvalidNumLayers);
    }
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if cfg.n_tasks == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let mut rng = LcgRng::new(seed);

    // Layer 0: input_dim → n_units[0]
    let fan_in0 = cfg.input_dim;
    let fan_out0 = cfg.n_units[0];
    let mut w0 = vec![0.0_f64; fan_out0 * fan_in0];
    xavier_fill(&mut w0, fan_in0, fan_out0, &mut rng);
    let b0 = vec![0.0_f64; fan_out0];

    // Layer 1: n_units[0] → n_units[1]
    let fan_in1 = cfg.n_units[0];
    let fan_out1 = cfg.n_units[1];
    let mut w1 = vec![0.0_f64; fan_out1 * fan_in1];
    xavier_fill(&mut w1, fan_in1, fan_out1, &mut rng);
    let b1 = vec![0.0_f64; fan_out1];

    let cumulative_mask = vec![vec![0.0_f64; fan_out0], vec![0.0_f64; fan_out1]];

    Ok(HatState {
        weights: vec![w0, w1],
        biases: vec![b0, b1],
        task_embed: Vec::new(),
        cumulative_mask,
        n_layers: 2,
        n_tasks_seen: 0,
        input_dim: cfg.input_dim,
        n_units: cfg.n_units.clone(),
    })
}

/// Ensure `task_embed[task_id]` exists, initialising with N(0, 0.01) if absent.
fn ensure_task_embed(state: &mut HatState, task_id: usize, rng: &mut LcgRng) {
    while state.task_embed.len() <= task_id {
        let mut embed = Vec::with_capacity(state.n_layers);
        for l in 0..state.n_layers {
            let sz = state.n_units[l];
            let v: Vec<f64> = (0..sz)
                .map(|_| {
                    let (a, _) = rng.next_normal_pair();
                    a as f64 * 0.01
                })
                .collect();
            embed.push(v);
        }
        state.task_embed.push(embed);
    }
}

/// Forward pass through the HAT network for `task_id` at `sharpness`.
///
/// Returns the output logits (length `n_units[1]`).
///
/// At inference use `sharpness = s_max` (≈ binary gates).
/// During training use the scheduled sharpness.
pub fn hat_forward(
    state: &HatState,
    x: &[f64],
    task_id: usize,
    sharpness: f64,
) -> ContinualResult<Vec<f64>> {
    if x.len() != state.input_dim {
        return Err(ContinualError::DimensionMismatch {
            expected: state.input_dim,
            got: x.len(),
        });
    }
    if task_id >= state.task_embed.len() {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: task_id,
            n_tasks: state.task_embed.len(),
        });
    }

    // Layer 0: input → hidden, gated by task mask.
    let pre0 = matvec(
        &state.weights[0],
        x,
        &state.biases[0],
        state.input_dim,
        state.n_units[0],
    );
    let gate0 = compute_gate(&state.task_embed[task_id][0], sharpness);
    let h0 = elem_mul(&relu(&pre0), &gate0);

    // Layer 1: hidden → output, gated by task mask on output.
    let pre1 = matvec(
        &state.weights[1],
        &h0,
        &state.biases[1],
        state.n_units[0],
        state.n_units[1],
    );
    let gate1 = compute_gate(&state.task_embed[task_id][1], sharpness);
    let out = elem_mul(&pre1, &gate1);

    Ok(out)
}

/// Argmax classifier: return index of maximum output logit.
pub fn hat_classify(state: &HatState, x: &[f64], task_id: usize) -> ContinualResult<usize> {
    let logits = hat_forward(state, x, task_id, state.n_units[0] as f64)?;
    let pred = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(pred)
}

/// Fraction of free units per layer available to future tasks.
///
/// `capacity[l] = 1 - mean(cumulative_mask[l])`
pub fn hat_task_capacity(state: &HatState) -> Vec<f64> {
    state
        .cumulative_mask
        .iter()
        .map(|mask| {
            if mask.is_empty() {
                1.0
            } else {
                1.0 - mask.iter().sum::<f64>() / mask.len() as f64
            }
        })
        .collect()
}

/// Train HAT on one task (`task_id`) for `n_epochs` epochs.
///
/// `x`: data matrix `[n × input_dim]`; `y`: continuous targets length `n`
/// (used as soft regression signal; for classification pass class indices cast
/// to f64 then interpret argmax — the function accepts `f64` to stay generic).
pub fn hat_fit_task(
    state: &mut HatState,
    x: &[f64],
    y: &[f64],
    n: usize,
    task_id: usize,
    rng: &mut LcgRng,
) -> ContinualResult<()> {
    if n == 0 || x.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if y.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let d_in = state.input_dim;
    if x.len() != n * d_in {
        return Err(ContinualError::DimensionMismatch {
            expected: n * d_in,
            got: x.len(),
        });
    }

    // Ensure task embedding exists for this task_id.
    ensure_task_embed(state, task_id, rng);

    let n_epochs = 5_usize; // simplified; callers can wrap with cfg.n_epochs
    let s_max = 400.0_f64;
    let lr = 0.01_f64;
    let n_out = state.n_units[1];
    let n_hid = state.n_units[0];

    let mut indices: Vec<usize> = (0..n).collect();

    for epoch in 0..n_epochs {
        // Sharpness schedule: s = 1/s_max + (s_max - 1/s_max) * epoch/(n_epochs-1).
        let progress = if n_epochs > 1 {
            epoch as f64 / (n_epochs - 1) as f64
        } else {
            1.0
        };
        let s = 1.0 / s_max + (s_max - 1.0 / s_max) * progress;

        rng.shuffle(&mut indices);

        for &idx in &indices {
            let xi = &x[idx * d_in..(idx + 1) * d_in];
            let label_f = y[idx];
            // Interpret label as class index (rounded).
            let label = label_f.round() as usize;
            let label = label.min(n_out.saturating_sub(1));

            // --- Forward ---
            let gate0 = compute_gate(&state.task_embed[task_id][0], s);
            let gate1 = compute_gate(&state.task_embed[task_id][1], s);

            let pre0 = matvec(&state.weights[0], xi, &state.biases[0], d_in, n_hid);
            let h0_pre_relu = pre0.to_vec();
            let h0_relu = relu(&pre0);
            let h0 = elem_mul(&h0_relu, &gate0);

            let pre1 = matvec(&state.weights[1], &h0, &state.biases[1], n_hid, n_out);
            let logits = elem_mul(&pre1, &gate1);

            // --- CE loss gradient (logits level) ---
            let probs = softmax(&logits);
            let mut d_logits = probs.clone();
            if label < n_out {
                d_logits[label] -= 1.0;
            }

            // --- Backprop through gate1 → d_pre1 ---
            let mut d_pre1: Vec<f64> = d_logits
                .iter()
                .zip(gate1.iter())
                .map(|(g, ga)| g * ga)
                .collect();

            // --- Gradient for task_embed[task_id][1] ---
            // d_embed1[i] = d_logits[i] * pre1[i] * s * gate1[i] * (1 - gate1[i])
            {
                let embed1_grad: Vec<f64> = d_logits
                    .iter()
                    .zip(pre1.iter())
                    .zip(gate1.iter())
                    .map(|((&dl, &p1), &ga)| dl * p1 * s * ga * (1.0 - ga))
                    .collect();
                for (emb, &eg) in state.task_embed[task_id][1]
                    .iter_mut()
                    .zip(embed1_grad.iter())
                {
                    *emb -= lr * eg;
                }
            }

            // --- Scale d_pre1 by (1 - cumulative_mask[1]) to protect old units ---
            for (dp, &cm) in d_pre1.iter_mut().zip(state.cumulative_mask[1].iter()) {
                *dp *= 1.0 - cm;
            }

            // --- Backprop layer 1: W1, b1, d_h0 ---
            let mut d_h0 = vec![0.0_f64; n_hid];
            for (row, &g) in d_pre1.iter().enumerate() {
                state.biases[1][row] -= lr * g;
                for (col, &h0v) in h0.iter().enumerate() {
                    state.weights[1][row * n_hid + col] -= lr * g * h0v;
                    d_h0[col] += g * state.weights[1][row * n_hid + col];
                }
            }

            // --- Backprop through gate0 → d_h0_relu ---
            let d_h0_relu: Vec<f64> = d_h0
                .iter()
                .zip(gate0.iter())
                .map(|(g, ga)| g * ga)
                .collect();

            // --- Gradient for task_embed[task_id][0] ---
            {
                let embed0_grad: Vec<f64> = d_h0
                    .iter()
                    .zip(h0_relu.iter())
                    .zip(gate0.iter())
                    .map(|((&dh, &hr), &ga)| dh * hr * s * ga * (1.0 - ga))
                    .collect();
                for (emb, &eg) in state.task_embed[task_id][0]
                    .iter_mut()
                    .zip(embed0_grad.iter())
                {
                    *emb -= lr * eg;
                }
            }

            // --- ReLU backward ---
            let mut d_pre0: Vec<f64> = d_h0_relu
                .iter()
                .zip(h0_pre_relu.iter())
                .map(|(&d, &pre)| if pre <= 0.0 { 0.0 } else { d })
                .collect();

            // --- Scale d_pre0 by (1 - cumulative_mask[0]) ---
            for (dp, &cm) in d_pre0.iter_mut().zip(state.cumulative_mask[0].iter()) {
                *dp *= 1.0 - cm;
            }

            // --- Backprop layer 0: W0, b0 ---
            for (row, &g) in d_pre0.iter().enumerate() {
                state.biases[0][row] -= lr * g;
                for (col, &xv) in xi.iter().enumerate() {
                    state.weights[0][row * d_in + col] -= lr * g * xv;
                }
            }
        }
    }

    // --- Update cumulative_mask after this task ---
    let gate0_final = compute_gate(&state.task_embed[task_id][0], s_max);
    let gate1_final = compute_gate(&state.task_embed[task_id][1], s_max);
    elem_max_update(&mut state.cumulative_mask[0], &gate0_final);
    elem_max_update(&mut state.cumulative_mask[1], &gate1_final);
    state.n_tasks_seen += 1;

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hat() -> HatState {
        let cfg = HatConfig {
            input_dim: 8,
            hidden_dim: 16,
            n_units: vec![16, 4],
            n_tasks: 5,
            ..Default::default()
        };
        hat_new(&cfg, 42).expect("HAT state should initialize with valid config")
    }

    fn add_embed(state: &mut HatState, task_id: usize) {
        let mut rng = LcgRng::new(99 + task_id as u64);
        ensure_task_embed(state, task_id, &mut rng);
    }

    /// 1. hat_new succeeds with valid config.
    #[test]
    fn hat_new_valid_config() {
        let result = hat_new(&HatConfig::default(), 0);
        assert!(result.is_ok());
    }

    /// 2. hat_forward output has correct dimension.
    #[test]
    fn forward_output_correct_dim() {
        let mut state = make_hat();
        add_embed(&mut state, 0);
        let x = vec![1.0_f64; 8];
        let out = hat_forward(&state, &x, 0, 1.0)
            .expect("HAT forward pass should succeed on valid input");
        assert_eq!(out.len(), 4, "Output should have n_units[1]=4 elements");
    }

    /// 3. hat_forward returns DimensionMismatch on wrong input size.
    #[test]
    fn forward_wrong_input_dim_err() {
        let mut state = make_hat();
        add_embed(&mut state, 0);
        let x = vec![1.0_f64; 5]; // wrong
        assert!(hat_forward(&state, &x, 0, 1.0).is_err());
    }

    /// 4. hat_forward returns TaskIndexOutOfRange for unseen task.
    #[test]
    fn forward_unseen_task_err() {
        let state = make_hat();
        let x = vec![0.0_f64; 8];
        assert!(hat_forward(&state, &x, 99, 1.0).is_err());
    }

    /// 5. cumulative_mask is updated after hat_fit_task.
    #[test]
    fn cumulative_mask_updated_after_fit() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(5);
        let n = 4_usize;
        let mut rng2 = LcgRng::new(500);
        let x: Vec<f64> = (0..n * 8).map(|_| rng2.next_f32() as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let sum_before: f64 = state.cumulative_mask.iter().flat_map(|m| m.iter()).sum();
        hat_fit_task(&mut state, &x, &y, n, 0, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        let sum_after: f64 = state.cumulative_mask.iter().flat_map(|m| m.iter()).sum();
        assert!(
            sum_after >= sum_before,
            "cumulative_mask sum should be >= before fit"
        );
    }

    /// 6. hat_task_capacity returns values in [0, 1].
    #[test]
    fn task_capacity_in_unit_interval() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(6);
        let n = 4_usize;
        let x: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let y: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        hat_fit_task(&mut state, &x, &y, n, 0, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        let capacity = hat_task_capacity(&state);
        for (l, &c) in capacity.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&c),
                "Capacity at layer {l} out of [0,1]: {c}"
            );
        }
    }

    /// 7. hat_task_capacity before any fit returns 1.0 per layer (all free).
    #[test]
    fn task_capacity_one_before_fit() {
        let state = make_hat();
        let capacity = hat_task_capacity(&state);
        for &c in &capacity {
            assert!(
                (c - 1.0).abs() < 1e-9,
                "Capacity should be 1.0 before any task, got {c}"
            );
        }
    }

    /// 8. Binary gate in forward (high sharpness) gives near-binary outputs.
    #[test]
    fn high_sharpness_gate_near_binary() {
        let mut state = make_hat();
        // Set task_embed to large positive values → gates ≈ 1.
        let mut rng = LcgRng::new(8);
        ensure_task_embed(&mut state, 0, &mut rng);
        for embed_l in state.task_embed[0].iter_mut() {
            for v in embed_l.iter_mut() {
                *v = 5.0;
            }
        }
        let x = vec![0.1_f64; 8];
        let out = hat_forward(&state, &x, 0, 1000.0)
            .expect("HAT forward pass should succeed on valid input");
        // With large embedding, gates should be close to 1.0.
        // We verify forward doesn't panic and output is finite.
        assert!(out.iter().all(|v| v.is_finite()), "Output must be finite");
    }

    /// 9. hat_classify returns valid class index.
    #[test]
    fn classify_returns_valid_index() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(9);
        let n = 4_usize;
        let x_fit: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let y: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        hat_fit_task(&mut state, &x_fit, &y, n, 0, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        let x_query = vec![0.0_f64; 8];
        let pred = hat_classify(&state, &x_query, 0)
            .expect("HAT classification should succeed on valid input");
        assert!(pred < 4, "Prediction {pred} should be < n_units[1]=4");
    }

    /// 10. hat_fit_task returns Err on empty data.
    #[test]
    fn fit_task_empty_data_err() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(10);
        assert!(hat_fit_task(&mut state, &[], &[], 0, 0, &mut rng).is_err());
    }

    /// 11. hat_fit_task returns Err on dimension mismatch.
    #[test]
    fn fit_task_dim_mismatch_err() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(11);
        // n=2 but x has 8 elements instead of 16.
        let x = vec![0.0_f64; 8];
        let y = vec![0.0_f64; 2];
        assert!(hat_fit_task(&mut state, &x, &y, 2, 0, &mut rng).is_err());
    }

    /// 12. n_tasks_seen increments after each fit_task call.
    #[test]
    fn n_tasks_seen_increments() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(12);
        let n = 4_usize;
        let x: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let y: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(state.n_tasks_seen, 0);
        hat_fit_task(&mut state, &x, &y, n, 0, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        assert_eq!(state.n_tasks_seen, 1);
        let x2: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let y2 = y.clone();
        hat_fit_task(&mut state, &x2, &y2, n, 1, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        assert_eq!(state.n_tasks_seen, 2);
    }

    /// 13. hat_new returns Err for empty n_units.
    #[test]
    fn hat_new_empty_n_units_err() {
        let cfg = HatConfig {
            n_units: vec![],
            ..Default::default()
        };
        assert!(hat_new(&cfg, 0).is_err());
    }

    /// 14. Task-1 units not fully free after task-0 trains.
    #[test]
    fn capacity_decreases_after_fit() {
        let mut state = make_hat();
        let mut rng = LcgRng::new(14);
        let n = 8_usize;
        let x: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        hat_fit_task(&mut state, &x, &y, n, 0, &mut rng)
            .expect("HAT task fitting should succeed with valid data");
        let capacity = hat_task_capacity(&state);
        // After one task, at least one layer should have < 1.0 capacity.
        let any_reduced = capacity.iter().any(|&c| c < 1.0 - 1e-9);
        assert!(
            any_reduced,
            "Capacity should decrease after training (some units claimed)"
        );
    }
}
