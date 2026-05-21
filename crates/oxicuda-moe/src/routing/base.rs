//! BASE routing: balanced assignment via Sinkhorn iterations.
//!
//! Implements the routing from:
//! Lewis et al. "BASE Layers: Simplifying Training of Large, Sparse Models."
//! ICML 2021.
//!
//! Each token is assigned to exactly one expert via a doubly-stochastic
//! assignment matrix C ∈ R^{T×E} computed by alternating Sinkhorn normalisation
//! (column-then-row) applied to the initial row-wise softmax of gate logits.
//! This guarantees perfect load balance (each expert receives ≈ T/E tokens).

use crate::error::{MoeError, MoeResult};

/// Configuration for BASE routing.
#[derive(Debug, Clone)]
pub struct BaseConfig {
    /// Number of experts E.
    pub n_experts: usize,
    /// Input (token) feature dimension.
    pub input_dim: usize,
    /// Number of Sinkhorn iterations (default 3; paper uses ≥3 for convergence).
    pub n_iter: usize,
    /// Epsilon for numerical stability in normalization (default 1e-7).
    pub eps: f32,
    /// Temperature for initial softmax (default 1.0).
    pub temperature: f32,
}

impl Default for BaseConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            input_dim: 256,
            n_iter: 3,
            eps: 1e-7,
            temperature: 1.0,
        }
    }
}

/// Result of BASE routing.
#[derive(Debug, Clone)]
pub struct BaseResult {
    /// Soft assignment matrix `S ∈ [0,1]^{T×E}` after Sinkhorn, row-major.
    ///
    /// Row i sums to approximately 1.0 (token i's attention weights over experts).
    pub assignment: Vec<f32>,
    /// Hard assignment: `expert_assignments[t]` = argmax expert for token t.
    pub expert_assignments: Vec<usize>,
    /// Routing scores (same as soft assignment), useful for weighted combination.
    pub scores: Vec<f32>,
    /// Number of Sinkhorn iterations applied.
    pub n_iter: usize,
}

/// BASE router implementing balanced Sinkhorn assignment.
pub struct BaseRouter {
    /// Routing configuration.
    pub config: BaseConfig,
}

impl BaseRouter {
    /// Create a new BASE router.
    ///
    /// Returns `Err` when `n_experts == 0` or `input_dim == 0`.
    /// `n_iter == 0` is valid (produces raw row-softmax with no Sinkhorn).
    pub fn new(config: BaseConfig) -> MoeResult<Self> {
        if config.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: config.n_experts,
            });
        }
        if config.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: config.input_dim,
            });
        }
        Ok(Self { config })
    }

    /// Compute BASE routing for a batch of tokens.
    ///
    /// # Arguments
    /// - `logits`: raw router logits `[T × E]`, row-major.
    ///   (Pre-computed: typically `tokens @ router_weight^T`.)
    /// - `n_tokens`: T, number of tokens.
    ///
    /// # Returns
    /// `BaseResult` with soft + hard assignments.
    pub fn route(&self, logits: &[f32], n_tokens: usize) -> MoeResult<BaseResult> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = n_tokens * cfg.n_experts;
        if logits.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: logits.len(),
            });
        }

        // Step 1: row-wise softmax with temperature scaling.
        let mut s = row_softmax(logits, n_tokens, cfg.n_experts, cfg.temperature);

        // Step 2: Sinkhorn iterations (alternating column-then-row normalisation).
        self.sinkhorn_iterations(&mut s, n_tokens)?;

        // Step 3: Hard assignment — argmax per token row.
        let expert_assignments: Vec<usize> = (0..n_tokens)
            .map(|t| {
                let row = &s[t * cfg.n_experts..(t + 1) * cfg.n_experts];
                // Find argmax; ties broken by lower index.
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        let scores = s.clone();

        Ok(BaseResult {
            assignment: s,
            expert_assignments,
            scores,
            n_iter: cfg.n_iter,
        })
    }

    /// Run Sinkhorn iterations on a soft assignment matrix `[T × E]`.
    ///
    /// Modifies in place: alternating column-then-row normalisation for each
    /// iteration, which drives the matrix towards doubly stochastic.
    pub fn sinkhorn_iterations(&self, s: &mut [f32], n_tokens: usize) -> MoeResult<()> {
        let cfg = &self.config;
        let n_e = cfg.n_experts;
        let eps = cfg.eps;

        for _ in 0..cfg.n_iter {
            // --- Column normalisation ---
            // Collect column sums.
            let mut col_sums = vec![0.0_f32; n_e];
            for t in 0..n_tokens {
                for e in 0..n_e {
                    col_sums[e] += s[t * n_e + e];
                }
            }
            // Divide each entry by its column sum.
            for t in 0..n_tokens {
                for e in 0..n_e {
                    s[t * n_e + e] /= col_sums[e] + eps;
                }
            }

            // --- Row normalisation ---
            for t in 0..n_tokens {
                let row_start = t * n_e;
                let row_sum: f32 = s[row_start..row_start + n_e].iter().sum();
                let denom = row_sum + eps;
                for e in 0..n_e {
                    s[row_start + e] /= denom;
                }
            }
        }

        Ok(())
    }
}

/// Numerically stable row-wise softmax on a `[rows × cols]` matrix (row-major).
///
/// Applies max-shift per row before computing `exp`, then divides by the row sum.
/// The `temperature` parameter scales logits before exponentiation:
/// `p[i,j] = exp((logit[i,j] - max_i) / temperature) / Σ_j exp(…)`.
#[must_use]
pub fn row_softmax(logits: &[f32], rows: usize, cols: usize, temperature: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    let safe_temp = if temperature == 0.0 {
        1e-12
    } else {
        temperature
    };

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_slice = &logits[start..end];

        // Max-shift for numerical stability.
        let max_val = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut exp_sum = 0.0_f32;
        for (out_val, &logit) in out[start..end].iter_mut().zip(row_slice.iter()) {
            let e = ((logit - max_val) / safe_temp).exp();
            *out_val = e;
            exp_sum += e;
        }
        let denom = exp_sum + 1e-7;
        for out_val in out[start..end].iter_mut() {
            *out_val /= denom;
        }
    }

    out
}

/// Compute the Sinkhorn convergence deviation.
///
/// Returns `max_t |Σ_e S[t,e] - 1| + max_e |Σ_t S[t,e] - T/E|`.
///
/// A perfectly doubly-stochastic matrix (after scaling) yields 0.0.
#[must_use]
pub fn sinkhorn_convergence(s: &[f32], n_tokens: usize, n_experts: usize) -> f32 {
    if n_tokens == 0 || n_experts == 0 {
        return 0.0;
    }
    let expected_col_sum = n_tokens as f32 / n_experts as f32;

    // Compute row sums and find maximum deviation from 1.0.
    let mut max_row_dev = 0.0_f32;
    for t in 0..n_tokens {
        let row_sum: f32 = s[t * n_experts..(t + 1) * n_experts].iter().sum();
        let dev = (row_sum - 1.0).abs();
        if dev > max_row_dev {
            max_row_dev = dev;
        }
    }

    // Compute column sums and find maximum deviation from T/E.
    let mut col_sums = vec![0.0_f32; n_experts];
    for t in 0..n_tokens {
        for e in 0..n_experts {
            col_sums[e] += s[t * n_experts + e];
        }
    }
    let mut max_col_dev = 0.0_f32;
    for &cs in &col_sums {
        let dev = (cs - expected_col_sum).abs();
        if dev > max_col_dev {
            max_col_dev = dev;
        }
    }

    max_row_dev + max_col_dev
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // row_softmax
    // -----------------------------------------------------------------------

    #[test]
    fn row_softmax_single_row_sums_to_one() {
        let logits = [1.0_f32, 2.0, 3.0];
        let out = row_softmax(&logits, 1, 3, 1.0);
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");
    }

    #[test]
    fn row_softmax_uniform_input_near_uniform_output() {
        let logits = [0.0_f32, 0.0, 0.0];
        let out = row_softmax(&logits, 1, 3, 1.0);
        for &v in &out {
            assert!((v - 1.0 / 3.0).abs() < 1e-5, "expected ~1/3, got {v}");
        }
    }

    #[test]
    fn row_softmax_two_row_both_sum_to_one() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0];
        let out = row_softmax(&logits, 2, 2, 1.0);
        let s0: f32 = out[0..2].iter().sum();
        let s1: f32 = out[2..4].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5, "row0 sum={s0}");
        assert!((s1 - 1.0).abs() < 1e-5, "row1 sum={s1}");
    }

    #[test]
    fn row_softmax_large_value_numerically_stable() {
        let logits = [100.0_f32, 0.0];
        let out = row_softmax(&logits, 1, 2, 1.0);
        // exp(100) / (exp(100) + exp(0)) ≈ 1
        assert!(
            out[0] > 0.999,
            "expected ≈1 for dominant logit, got {}",
            out[0]
        );
        assert!(
            out[1] < 0.001,
            "expected ≈0 for dominated logit, got {}",
            out[1]
        );
    }

    #[test]
    fn row_softmax_all_values_non_negative() {
        let logits = [-5.0_f32, 0.0, 5.0, -1.0, 3.0, 2.0];
        let out = row_softmax(&logits, 2, 3, 1.0);
        for &v in &out {
            assert!(v >= 0.0, "negative probability: {v}");
        }
    }

    #[test]
    fn row_softmax_temperature_affects_sharpness() {
        // High temperature → flatter distribution
        let logits = [0.0_f32, 1.0];
        let hot = row_softmax(&logits, 1, 2, 10.0);
        let cold = row_softmax(&logits, 1, 2, 0.1);
        // Cold should be sharper: dominant entry should be larger
        assert!(cold[1] > hot[1], "cold={}, hot={}", cold[1], hot[1]);
    }

    // -----------------------------------------------------------------------
    // sinkhorn_iterations / sinkhorn_convergence
    // -----------------------------------------------------------------------

    #[test]
    fn sinkhorn_iterations_rows_sum_to_one_after_many_iters() {
        let cfg = BaseConfig {
            n_experts: 4,
            input_dim: 8,
            n_iter: 10,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let n_tokens = 8;
        let logits: Vec<f32> = (0..n_tokens * 4).map(|i| (i as f32) * 0.1).collect();
        let mut s = row_softmax(&logits, n_tokens, 4, 1.0);
        router.sinkhorn_iterations(&mut s, n_tokens).unwrap();
        for t in 0..n_tokens {
            let row_sum: f32 = s[t * 4..t * 4 + 4].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "token {t} row sum = {row_sum}"
            );
        }
    }

    #[test]
    fn sinkhorn_iterations_cols_sum_near_tokens_over_experts() {
        let n_tokens = 8_usize;
        let n_experts = 4_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 8,
            n_iter: 10,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits: Vec<f32> = (0..n_tokens * n_experts)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let mut s = row_softmax(&logits, n_tokens, n_experts, 1.0);
        router.sinkhorn_iterations(&mut s, n_tokens).unwrap();
        let expected = n_tokens as f32 / n_experts as f32; // = 2.0
        let mut col_sums = vec![0.0_f32; n_experts];
        for t in 0..n_tokens {
            for e in 0..n_experts {
                col_sums[e] += s[t * n_experts + e];
            }
        }
        for (e, &cs) in col_sums.iter().enumerate() {
            assert!(
                (cs - expected).abs() < 1e-3,
                "expert {e} col sum = {cs}, expected {expected}"
            );
        }
    }

    #[test]
    fn sinkhorn_convergence_perfect_doubly_stochastic_gives_zero() {
        // For T=2, E=2 a perfectly balanced assignment is [[0.5,0.5],[0.5,0.5]],
        // which has row sums = 1 and col sums = T/E = 1.
        let s = [0.5_f32, 0.5, 0.5, 0.5];
        let dev = sinkhorn_convergence(&s, 2, 2);
        assert!(dev < 1e-5, "convergence deviation = {dev}");
    }

    #[test]
    fn sinkhorn_convergence_imbalanced_gives_positive() {
        // All tokens to expert 0 → col 0 sum = 2, col 1 sum = 0
        let s = [1.0_f32, 0.0, 1.0, 0.0];
        let dev = sinkhorn_convergence(&s, 2, 2);
        assert!(dev > 0.5, "expected positive deviation, got {dev}");
    }

    // -----------------------------------------------------------------------
    // BaseRouter::new
    // -----------------------------------------------------------------------

    #[test]
    fn new_with_zero_experts_returns_err() {
        let cfg = BaseConfig {
            n_experts: 0,
            input_dim: 8,
            ..BaseConfig::default()
        };
        assert!(BaseRouter::new(cfg).is_err());
    }

    #[test]
    fn new_with_zero_input_dim_returns_err() {
        let cfg = BaseConfig {
            n_experts: 4,
            input_dim: 0,
            ..BaseConfig::default()
        };
        assert!(BaseRouter::new(cfg).is_err());
    }

    #[test]
    fn new_with_zero_iters_is_valid() {
        let cfg = BaseConfig {
            n_experts: 4,
            input_dim: 8,
            n_iter: 0,
            ..BaseConfig::default()
        };
        assert!(BaseRouter::new(cfg).is_ok());
    }

    // -----------------------------------------------------------------------
    // BaseRouter::route
    // -----------------------------------------------------------------------

    #[test]
    fn route_wrong_logits_length_returns_err() {
        let cfg = BaseConfig {
            n_experts: 4,
            input_dim: 8,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        // Correct would be 3*4=12; give 10 instead.
        let logits = vec![0.0_f32; 10];
        assert!(router.route(&logits, 3).is_err());
    }

    #[test]
    fn route_zero_tokens_returns_err() {
        let cfg = BaseConfig {
            n_experts: 4,
            input_dim: 8,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits = vec![0.0_f32; 0];
        assert!(router.route(&logits, 0).is_err());
    }

    #[test]
    fn route_expert_assignments_length_equals_n_tokens() {
        let n_tokens = 6_usize;
        let n_experts = 3_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 8,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits = vec![1.0_f32; n_tokens * n_experts];
        let result = router.route(&logits, n_tokens).unwrap();
        assert_eq!(result.expert_assignments.len(), n_tokens);
    }

    #[test]
    fn route_all_expert_assignments_in_range() {
        let n_tokens = 8_usize;
        let n_experts = 4_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 8,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits: Vec<f32> = (0..n_tokens * n_experts)
            .map(|i| (i as f32) * 0.3)
            .collect();
        let result = router.route(&logits, n_tokens).unwrap();
        for &a in &result.expert_assignments {
            assert!(a < n_experts, "assignment {a} >= n_experts {n_experts}");
        }
    }

    #[test]
    fn route_soft_assignment_non_negative_and_rows_sum_to_one() {
        let n_tokens = 5_usize;
        let n_experts = 4_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 8,
            n_iter: 3,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits: Vec<f32> = (0..n_tokens * n_experts)
            .map(|i| (i as f32) * 0.2 - 1.0)
            .collect();
        let result = router.route(&logits, n_tokens).unwrap();
        for &v in &result.assignment {
            assert!(v >= 0.0, "negative assignment value: {v}");
        }
        for t in 0..n_tokens {
            let row_sum: f32 = result.assignment[t * n_experts..(t + 1) * n_experts]
                .iter()
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "token {t} row sum = {row_sum}"
            );
        }
    }

    #[test]
    fn route_with_zero_iters_equals_raw_softmax() {
        let n_tokens = 4_usize;
        let n_experts = 3_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 8,
            n_iter: 0,
            temperature: 1.0,
            eps: 1e-7,
        };
        let router = BaseRouter::new(cfg).unwrap();
        let logits = vec![
            0.5_f32, 1.0, 1.5, 2.0, 0.0, 1.0, 0.3, 0.6, 0.9, 0.1, 0.4, 0.8,
        ];
        let result = router.route(&logits, n_tokens).unwrap();
        let expected = row_softmax(&logits, n_tokens, n_experts, 1.0);
        for (a, e) in result.assignment.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "zero-iter mismatch: {a} vs {e}");
        }
    }

    #[test]
    fn route_more_iters_gives_more_balanced_assignment() {
        let n_tokens = 8_usize;
        let n_experts = 4_usize;
        // Build deliberately skewed logits: token 0 strongly prefers expert 0,
        // making zero-iter softmax unbalanced.
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for t in 0..n_tokens {
            logits[t * n_experts] = 5.0; // all prefer expert 0
        }

        let cfg_few = BaseConfig {
            n_experts,
            input_dim: 8,
            n_iter: 0,
            ..BaseConfig::default()
        };
        let cfg_many = BaseConfig {
            n_experts,
            input_dim: 8,
            n_iter: 20,
            ..BaseConfig::default()
        };

        let r_few = BaseRouter::new(cfg_few)
            .unwrap()
            .route(&logits, n_tokens)
            .unwrap();
        let r_many = BaseRouter::new(cfg_many)
            .unwrap()
            .route(&logits, n_tokens)
            .unwrap();

        // Convergence deviation should be lower with more iterations.
        let dev_few = sinkhorn_convergence(&r_few.assignment, n_tokens, n_experts);
        let dev_many = sinkhorn_convergence(&r_many.assignment, n_tokens, n_experts);
        assert!(
            dev_many < dev_few,
            "more iters should improve balance: dev_few={dev_few}, dev_many={dev_many}"
        );
    }

    #[test]
    fn route_4tokens_2experts_each_expert_gets_roughly_2_tokens() {
        // Use logits that break symmetry: alternating preference for expert 0 vs 1.
        // Tokens 0,2 prefer expert 0; tokens 1,3 prefer expert 1.
        // After Sinkhorn, each expert should be assigned exactly 2 tokens.
        let n_tokens = 4_usize;
        let n_experts = 2_usize;
        let cfg = BaseConfig {
            n_experts,
            input_dim: 4,
            n_iter: 20,
            ..BaseConfig::default()
        };
        let router = BaseRouter::new(cfg).unwrap();
        // Row 0: [2.0, 0.0], Row 1: [0.0, 2.0], Row 2: [2.0, 0.0], Row 3: [0.0, 2.0]
        let logits = vec![2.0_f32, 0.0, 0.0, 2.0, 2.0_f32, 0.0, 0.0, 2.0];
        let result = router.route(&logits, n_tokens).unwrap();
        let mut counts = vec![0_usize; n_experts];
        for &a in &result.expert_assignments {
            counts[a] += 1;
        }
        // Each expert should receive exactly 2 tokens.
        for (e, &c) in counts.iter().enumerate() {
            assert!(c == 2, "expert {e} received {c} tokens, expected 2");
        }
    }

    #[test]
    fn default_config_fields() {
        let cfg = BaseConfig::default();
        assert_eq!(cfg.n_experts, 8);
        assert_eq!(cfg.n_iter, 3);
    }
}
