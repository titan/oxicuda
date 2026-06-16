//! Multi-gate Mixture-of-Experts (MMoE) routing.
//!
//! Implements the routing mechanism from:
//! Ma et al. "Modeling Task Relationships in Multi-task Learning with
//! Multi-gate Mixture-of-Experts." KDD 2018.
//!
//! A shared pool of `n_experts` experts is shared across `n_tasks` tasks, but
//! **each task has its own gating network** `g_t(x) = softmax(W_t · x)`. The
//! per-task gate decides how that task mixes the shared experts, letting tasks
//! with different relationships emphasise different experts while still sharing
//! representation capacity.
//!
//! This module provides the gating computation. Given an input `x ∈ R^{d}` it
//! returns one routing-weight vector per task, each a probability distribution
//! over the `n_experts` shared experts. An optional [`MultiGateRouter::combine`]
//! mixes pre-computed expert outputs into one mixed output per task.
//!
//! # RNG note
//! The gate weights are initialised with a **uniform** draw built from
//! `LcgRng::next_u32`. The crate's `LcgRng::next_f32` / `next_normal_pair`
//! are unsuitable for an unbiased `[0, 1)` uniform (the generator only exposes
//! the high 31 bits of state, so `next_f32` spans roughly `[0, 0.5)`), so we
//! map `next_u32()` ourselves via `next_u32() as f32 / 4_294_967_296.0`, which
//! is a correct `[0, 1)` uniform given the 31-bit output range. Weights are
//! then centred to `[-scale, scale)` so the initial gates are near-uniform.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Configuration for multi-gate (MMoE) routing.
#[derive(Debug, Clone)]
pub struct MultiGateConfig {
    /// Input feature dimension `d`. Must be `> 0`.
    pub input_dim: usize,
    /// Number of shared experts. Must be `> 0`.
    pub n_experts: usize,
    /// Number of tasks (independent gates). Must be `> 0`.
    pub n_tasks: usize,
}

/// Multi-gate router: one linear gate per task over a shared expert pool.
#[derive(Debug, Clone)]
pub struct MultiGateRouter {
    /// Gate weights, row-major with shape `[n_tasks * n_experts * input_dim]`.
    ///
    /// The block for task `t`, expert `e` starts at
    /// `(t * n_experts + e) * input_dim`.
    pub gate_weights: Vec<f32>,
    /// Gate biases, shape `[n_tasks * n_experts]` (one per task/expert logit).
    pub gate_bias: Vec<f32>,
    /// Routing configuration.
    pub config: MultiGateConfig,
}

/// Draw a uniform `f32` in `[0, 1)` from the crate's `LcgRng`.
///
/// `LcgRng::next_u32` only returns the high 31 bits of state (range
/// `[0, 2^31)`), so dividing by `2^31` gives a correct half-open uniform. This
/// sidesteps the biased `LcgRng::next_f32` (which spans only `[0, 0.5)`).
#[inline]
fn uniform_unit(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0
}

/// Numerically stable softmax over a slice of logits.
///
/// Subtracts the max before exponentiating to avoid overflow at large logits.
#[must_use]
fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&val| (val - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / (sum + 1e-12)).collect()
}

impl MultiGateRouter {
    /// Create a new multi-gate router with uniform-initialised gate weights.
    ///
    /// Weights are drawn from `[-init_scale, init_scale)` with
    /// `init_scale = 1 / sqrt(input_dim)`, a standard fan-in scaling that keeps
    /// initial logits small so every task gate starts close to uniform. Biases
    /// are initialised to zero.
    ///
    /// # Errors
    /// * [`MoeError::InvalidInputDim`] if `input_dim == 0`.
    /// * [`MoeError::InvalidExpertCount`] if `n_experts == 0`.
    /// * [`MoeError::Internal`] if `n_tasks == 0`.
    pub fn new(config: MultiGateConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if config.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: config.input_dim,
            });
        }
        if config.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: config.n_experts,
            });
        }
        if config.n_tasks == 0 {
            return Err(MoeError::Internal {
                msg: "n_tasks must be > 0".to_string(),
            });
        }

        let init_scale = 1.0_f32 / (config.input_dim as f32).sqrt();
        let weight_count = config.n_tasks * config.n_experts * config.input_dim;
        let mut gate_weights = vec![0.0_f32; weight_count];
        for w in &mut gate_weights {
            // Map [0,1) → [-init_scale, init_scale).
            *w = (uniform_unit(rng) * 2.0 - 1.0) * init_scale;
        }
        let gate_bias = vec![0.0_f32; config.n_tasks * config.n_experts];

        Ok(Self {
            gate_weights,
            gate_bias,
            config,
        })
    }

    /// Compute per-task routing weights for a single input `x`.
    ///
    /// Returns `n_tasks` vectors, each of length `n_experts`, where row `t` is
    /// `softmax(W_t · x + b_t)` — a probability distribution over the shared
    /// experts for task `t`.
    ///
    /// # Errors
    /// * [`MoeError::DimensionMismatch`] if `x.len() != input_dim`.
    /// * [`MoeError::NanEncountered`] if any gate value is NaN.
    pub fn forward(&self, x: &[f32]) -> MoeResult<Vec<Vec<f32>>> {
        let cfg = &self.config;
        if x.len() != cfg.input_dim {
            return Err(MoeError::DimensionMismatch {
                expected: cfg.input_dim,
                got: x.len(),
            });
        }

        let mut gates: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_tasks);
        for task in 0..cfg.n_tasks {
            // Compute logits for this task: one per expert.
            let mut logits = vec![0.0_f32; cfg.n_experts];
            for (exp_idx, logit) in logits.iter_mut().enumerate() {
                let base = (task * cfg.n_experts + exp_idx) * cfg.input_dim;
                // `new` sized `gate_weights` to exactly cover every (task,expert)
                // block, so this range is always valid; fall back to a 0 dot if
                // a malformed router somehow has a short buffer.
                let dot = match self.gate_weights.get(base..base + cfg.input_dim) {
                    Some(w_row) => x
                        .iter()
                        .zip(w_row.iter())
                        .map(|(&xi, &wi)| xi * wi)
                        .sum::<f32>(),
                    None => 0.0,
                };
                let bias = self
                    .gate_bias
                    .get(task * cfg.n_experts + exp_idx)
                    .copied()
                    .unwrap_or(0.0);
                *logit = dot + bias;
            }

            let gate = stable_softmax(&logits);
            if gate.iter().any(|v| v.is_nan()) {
                return Err(MoeError::NanEncountered {
                    context: "multi_gate softmax".to_string(),
                });
            }
            gates.push(gate);
        }

        Ok(gates)
    }

    /// Combine pre-computed expert outputs into one mixed output per task.
    ///
    /// `expert_outputs` is row-major `[n_experts * output_dim]`: the output of
    /// each shared expert for this input. The result is `n_tasks` vectors of
    /// length `output_dim`, where row `t` is the gate-weighted sum
    /// `sum_e g_t[e] * expert_outputs[e]`.
    ///
    /// # Errors
    /// * [`MoeError::DimensionMismatch`] if `x.len() != input_dim`, if
    ///   `output_dim == 0`, or if `expert_outputs.len() != n_experts * output_dim`.
    pub fn combine(
        &self,
        x: &[f32],
        expert_outputs: &[f32],
        output_dim: usize,
    ) -> MoeResult<Vec<Vec<f32>>> {
        let cfg = &self.config;
        if output_dim == 0 {
            return Err(MoeError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let expected = cfg.n_experts * output_dim;
        if expert_outputs.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: expert_outputs.len(),
            });
        }

        let gates = self.forward(x)?;
        let mut mixed: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_tasks);
        for gate in &gates {
            let mut out = vec![0.0_f32; output_dim];
            for (exp_idx, &weight) in gate.iter().enumerate() {
                let base = exp_idx * output_dim;
                // `expert_outputs.len()` was validated above, so this slice is
                // always in range.
                if let Some(expert_slice) = expert_outputs.get(base..base + output_dim) {
                    for (o, &e) in out.iter_mut().zip(expert_slice.iter()) {
                        *o += weight * e;
                    }
                }
            }
            mixed.push(out);
        }
        Ok(mixed)
    }

    /// Number of trainable parameters (gate weights + biases).
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.gate_weights.len() + self.gate_bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(input_dim: usize, n_experts: usize, n_tasks: usize, seed: u64) -> MultiGateRouter {
        let mut rng = LcgRng::new(seed);
        MultiGateRouter::new(
            MultiGateConfig {
                input_dim,
                n_experts,
                n_tasks,
            },
            &mut rng,
        )
        .expect("valid multi-gate config")
    }

    // --- Validation / construction ---

    #[test]
    fn new_zero_input_dim_errors() {
        let mut rng = LcgRng::new(0);
        let err = MultiGateRouter::new(
            MultiGateConfig {
                input_dim: 0,
                n_experts: 4,
                n_tasks: 2,
            },
            &mut rng,
        );
        assert!(matches!(err, Err(MoeError::InvalidInputDim { .. })));
    }

    #[test]
    fn new_zero_experts_errors() {
        let mut rng = LcgRng::new(0);
        let err = MultiGateRouter::new(
            MultiGateConfig {
                input_dim: 8,
                n_experts: 0,
                n_tasks: 2,
            },
            &mut rng,
        );
        assert!(matches!(err, Err(MoeError::InvalidExpertCount { .. })));
    }

    #[test]
    fn new_zero_tasks_errors() {
        let mut rng = LcgRng::new(0);
        let err = MultiGateRouter::new(
            MultiGateConfig {
                input_dim: 8,
                n_experts: 4,
                n_tasks: 0,
            },
            &mut rng,
        );
        assert!(matches!(err, Err(MoeError::Internal { .. })));
    }

    #[test]
    fn param_count_matches_dims() {
        let r = router(8, 4, 3, 1);
        assert_eq!(r.param_count(), 3 * 4 * 8 + 3 * 4);
    }

    // --- Gate distribution properties ---

    #[test]
    fn each_task_gate_sums_to_one() {
        let r = router(16, 6, 4, 42);
        let x = vec![0.3_f32; 16];
        let gates = r.forward(&x).expect("forward ok");
        for (task, gate) in gates.iter().enumerate() {
            let sum: f32 = gate.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "task {task} gate sums to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn all_gate_values_nonneg_and_le_one() {
        let r = router(12, 5, 3, 99);
        let x = vec![0.7_f32; 12];
        let gates = r.forward(&x).expect("forward ok");
        for gate in &gates {
            for &g in gate {
                assert!(
                    (0.0..=1.0 + 1e-6).contains(&g),
                    "gate value {g} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn output_shape_is_tasks_by_experts() {
        let n_tasks = 5_usize;
        let n_experts = 7_usize;
        let r = router(10, n_experts, n_tasks, 3);
        let x = vec![0.1_f32; 10];
        let gates = r.forward(&x).expect("forward ok");
        assert_eq!(gates.len(), n_tasks);
        for gate in &gates {
            assert_eq!(gate.len(), n_experts);
        }
    }

    #[test]
    fn single_task_reduces_to_one_softmax_gate() {
        let r = router(16, 8, 1, 7);
        let x = vec![0.5_f32; 16];
        let gates = r.forward(&x).expect("forward ok");
        assert_eq!(gates.len(), 1);
        let sum: f32 = gates[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Recompute the single gate directly from weights to confirm it is a
        // plain softmax(W_0 · x).
        let cfg = &r.config;
        let mut logits = vec![0.0_f32; cfg.n_experts];
        for (e, logit) in logits.iter_mut().enumerate() {
            let base = e * cfg.input_dim;
            let w = &r.gate_weights[base..base + cfg.input_dim];
            *logit = x.iter().zip(w).map(|(&xi, &wi)| xi * wi).sum::<f32>();
        }
        let expected = stable_softmax(&logits);
        for (got, exp) in gates[0].iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn uniform_weights_give_uniform_gates() {
        // Force all gate weights and biases to zero → every logit is 0 →
        // softmax is exactly 1/n_experts.
        let mut r = router(8, 5, 3, 11);
        for w in &mut r.gate_weights {
            *w = 0.0;
        }
        for b in &mut r.gate_bias {
            *b = 0.0;
        }
        let x = vec![1.234_f32; 8];
        let gates = r.forward(&x).expect("forward ok");
        let expected = 1.0_f32 / 5.0;
        for gate in &gates {
            for &g in gate {
                assert!(
                    (g - expected).abs() < 1e-6,
                    "gate {g} != uniform {expected}"
                );
            }
        }
    }

    #[test]
    fn distinct_task_weights_give_distinct_gates() {
        // With random per-task weights, two different tasks should generally
        // produce different gate distributions for a non-trivial input.
        let r = router(32, 8, 4, 2024);
        let mut x = vec![0.0_f32; 32];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.05 - 0.8;
        }
        let gates = r.forward(&x).expect("forward ok");
        let mut any_distinct = false;
        for t in 1..gates.len() {
            if gates[t]
                .iter()
                .zip(gates[0].iter())
                .any(|(a, b)| (a - b).abs() > 1e-4)
            {
                any_distinct = true;
            }
        }
        assert!(any_distinct, "all task gates were identical");
    }

    #[test]
    fn task_gates_differ_when_weights_set_apart() {
        // Deterministic check independent of RNG: hand-set two tasks' weights so
        // task 0 prefers expert 0 and task 1 prefers expert 1.
        let mut r = router(4, 2, 2, 5);
        for w in &mut r.gate_weights {
            *w = 0.0;
        }
        for b in &mut r.gate_bias {
            *b = 0.0;
        }
        // Bias task 0 toward expert 0, task 1 toward expert 1.
        // Layout is [task * n_experts + expert] with n_experts == 2.
        r.gate_bias[0] = 5.0; // task 0, expert 0
        r.gate_bias[3] = 5.0; // task 1, expert 1
        let x = vec![0.0_f32; 4];
        let gates = r.forward(&x).expect("forward ok");
        assert!(gates[0][0] > gates[0][1], "task 0 should favour expert 0");
        assert!(gates[1][1] > gates[1][0], "task 1 should favour expert 1");
    }

    // --- Numerical stability ---

    #[test]
    fn stable_softmax_no_nan_at_large_logits() {
        let gate = stable_softmax(&[1.0e30_f32, 2.0e30, 3.0e30, -1.0e30]);
        assert!(gate.iter().all(|v| v.is_finite()));
        let sum: f32 = gate.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn forward_no_nan_with_large_inputs() {
        let mut r = router(4, 3, 2, 8);
        // Large weights + large input → large logits, must stay finite.
        for w in &mut r.gate_weights {
            *w = 1.0e3;
        }
        let x = vec![1.0e6_f32; 4];
        let gates = r.forward(&x).expect("forward ok");
        for gate in &gates {
            assert!(gate.iter().all(|v| v.is_finite()));
            let sum: f32 = gate.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    // --- Determinism ---

    #[test]
    fn deterministic_given_seed() {
        let a = router(16, 4, 3, 1234);
        let b = router(16, 4, 3, 1234);
        assert_eq!(a.gate_weights, b.gate_weights);
        let x = vec![0.42_f32; 16];
        assert_eq!(
            a.forward(&x).expect("forward should succeed"),
            b.forward(&x).expect("forward should succeed")
        );
    }

    #[test]
    fn different_seed_changes_weights() {
        let a = router(16, 4, 3, 1);
        let b = router(16, 4, 3, 2);
        assert_ne!(a.gate_weights, b.gate_weights);
    }

    #[test]
    fn weights_within_init_range() {
        let input_dim = 64_usize;
        let r = router(input_dim, 8, 4, 77);
        let scale = 1.0_f32 / (input_dim as f32).sqrt();
        for &w in &r.gate_weights {
            assert!(
                w.abs() <= scale + 1e-6,
                "weight {w} exceeds init scale {scale}"
            );
        }
    }

    // --- forward error path ---

    #[test]
    fn forward_wrong_input_len_errors() {
        let r = router(8, 4, 2, 0);
        let x = vec![0.0_f32; 7];
        assert!(matches!(
            r.forward(&x),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    // --- combine ---

    #[test]
    fn combine_output_shape_and_finite() {
        let n_tasks = 3_usize;
        let n_experts = 4_usize;
        let output_dim = 5_usize;
        let r = router(8, n_experts, n_tasks, 21);
        let x = vec![0.3_f32; 8];
        let expert_outputs = vec![0.5_f32; n_experts * output_dim];
        let mixed = r
            .combine(&x, &expert_outputs, output_dim)
            .expect("combine ok");
        assert_eq!(mixed.len(), n_tasks);
        for row in &mixed {
            assert_eq!(row.len(), output_dim);
            assert!(row.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn combine_identical_experts_returns_that_value() {
        // If every expert outputs the same constant vector, the gate-weighted
        // mix (gates sum to 1) must reproduce that constant exactly.
        let n_experts = 6_usize;
        let output_dim = 3_usize;
        let r = router(8, n_experts, 2, 9);
        let x = vec![0.1_f32; 8];
        let mut expert_outputs = vec![0.0_f32; n_experts * output_dim];
        for e in 0..n_experts {
            for d in 0..output_dim {
                expert_outputs[e * output_dim + d] = (d as f32) + 1.0; // [1,2,3]
            }
        }
        let mixed = r
            .combine(&x, &expert_outputs, output_dim)
            .expect("combine ok");
        for row in &mixed {
            for (d, &val) in row.iter().enumerate() {
                assert!((val - ((d as f32) + 1.0)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn combine_wrong_output_len_errors() {
        let r = router(8, 4, 2, 0);
        let x = vec![0.0_f32; 8];
        let bad = vec![0.0_f32; 4 * 5 - 1];
        assert!(matches!(
            r.combine(&x, &bad, 5),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn combine_zero_output_dim_errors() {
        let r = router(8, 4, 2, 0);
        let x = vec![0.0_f32; 8];
        assert!(matches!(
            r.combine(&x, &[], 0),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    // --- RNG helper sanity ---

    #[test]
    fn uniform_unit_in_unit_range() {
        let mut rng = LcgRng::new(123);
        for _ in 0..2000 {
            let u = uniform_unit(&mut rng);
            assert!((0.0..1.0).contains(&u), "uniform_unit out of [0,1): {u}");
        }
    }

    #[test]
    fn uniform_unit_spans_above_half() {
        // Guards against the broken next_f32 (which never exceeds ~0.5):
        // a correct [0,1) uniform must produce values well above 0.5.
        let mut rng = LcgRng::new(987);
        let mut max_seen = 0.0_f32;
        for _ in 0..5000 {
            max_seen = max_seen.max(uniform_unit(&mut rng));
        }
        assert!(
            max_seen > 0.9,
            "uniform never exceeded 0.9 (max {max_seen})"
        );
    }
}
