//! LoRAMoE: a mixture of low-rank LoRA adapters as experts (Sheng et al. 2024).
//!
//! Implements the adapter-mixture layer from:
//! Sheng et al. "LoRAMoE: Alleviating World Knowledge Forgetting in Large
//! Language Models via MoE-Style Plugin." ACL 2024.
//!
//! A frozen base projection `W_0 ∈ R^{d×d}` is augmented by `n_experts` low-rank
//! LoRA adapters. Adapter `e` factorises its weight delta as
//!
//! ```text
//! ΔW_e = (α / r) · B_e · A_e ,   A_e ∈ R^{r×d},  B_e ∈ R^{d×r}
//! ```
//!
//! so `rank(ΔW_e) ≤ r ≪ d`. A linear gating network selects the `top_k` adapters
//! per token and combines their deltas on top of the frozen base output:
//!
//! ```text
//! y = W_0 · x + Σ_{e ∈ top-k} g_e · (α / r) · B_e · (A_e · x)
//! ```
//!
//! As in standard LoRA, every `B_e` is initialised to zero, so a freshly built
//! layer reproduces the frozen base output exactly until the adapters are trained.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::moe::matvec;
use crate::routing::top_k::{TopKConfig, TopKRouter};

/// A single low-rank LoRA adapter expert.
#[derive(Debug, Clone)]
pub struct LoraExpert {
    /// Down-projection `A`, row-major `[rank · d_model]`.
    pub a: Vec<f32>,
    /// Up-projection `B`, row-major `[d_model · rank]`.
    pub b: Vec<f32>,
    /// LoRA bottleneck rank `r`.
    pub rank: usize,
    /// Model dimension `d`.
    pub d_model: usize,
    /// Effective scaling `α / r` applied to the delta.
    pub scale: f32,
}

impl LoraExpert {
    /// Create an adapter with the standard LoRA initialisation: `A ~ N(0, 1/d)`
    /// and `B = 0` (so the initial delta is exactly zero).
    ///
    /// # Errors
    /// Returns [`MoeError::InvalidInputDim`] for `d_model == 0` and
    /// [`MoeError::InvalidHiddenDim`] when `rank == 0` or `rank > d_model`.
    pub fn new(d_model: usize, rank: usize, alpha: f32, rng: &mut LcgRng) -> MoeResult<Self> {
        if d_model == 0 {
            return Err(MoeError::InvalidInputDim { dim: d_model });
        }
        if rank == 0 || rank > d_model {
            return Err(MoeError::InvalidHiddenDim { dim: rank });
        }
        let mut a = vec![0.0_f32; rank * d_model];
        // Kaiming-style init for A keeps the pre-delta activations well scaled.
        rng.fill_normal_scaled(&mut a, (1.0 / d_model as f32).sqrt());
        Ok(Self {
            a,
            b: vec![0.0_f32; d_model * rank],
            rank,
            d_model,
            scale: alpha / rank as f32,
        })
    }

    /// Compute the scaled low-rank delta `(α / r) · B · (A · x)` for one token.
    ///
    /// # Errors
    /// Returns [`MoeError::DimensionMismatch`] when `x.len() != d_model`.
    pub fn delta(&self, x: &[f32]) -> MoeResult<Vec<f32>> {
        if x.len() != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: x.len(),
            });
        }
        // h = A · x   (length rank)
        let hidden = matvec(&self.a, x, self.d_model)?;
        // out = scale · (B · h)   (length d_model)
        let mut out = matvec(&self.b, &hidden, self.rank)?;
        for v in &mut out {
            *v *= self.scale;
        }
        Ok(out)
    }
}

/// Configuration for a [`LoraMoe`] layer.
#[derive(Debug, Clone)]
pub struct LoraMoeConfig {
    /// Model dimension `d` (square base projection `W_0 ∈ R^{d×d}`).
    pub d_model: usize,
    /// Number of LoRA adapter experts.
    pub n_experts: usize,
    /// LoRA bottleneck rank `r` (`1 ≤ r ≤ d_model`).
    pub rank: usize,
    /// Adapters activated per token (`1 ≤ top_k ≤ n_experts`).
    pub top_k: usize,
    /// LoRA scaling numerator `α` (effective scale `α / r`).
    pub alpha: f32,
}

impl Default for LoraMoeConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_experts: 8,
            rank: 8,
            top_k: 2,
            alpha: 16.0,
        }
    }
}

/// Output of a [`LoraMoe`] forward pass.
#[derive(Debug)]
pub struct LoraMoeOutput {
    /// Output hidden states, shape `[n_tokens · d_model]`.
    pub hidden: Vec<f32>,
    /// Selected adapter indices per token, shape `[n_tokens · top_k]`.
    pub gate_indices: Vec<usize>,
    /// Renormalised gate weights aligned with `gate_indices`,
    /// shape `[n_tokens · top_k]`; each token's `top_k` weights sum to `1`.
    pub gate_weights: Vec<f32>,
}

/// Mixture of LoRA adapters over a frozen base projection.
pub struct LoraMoe {
    router: TopKRouter,
    experts: Vec<LoraExpert>,
    /// Frozen base weight `W_0`, row-major `[d_model · d_model]`.
    pub base: Vec<f32>,
    /// Model dimension.
    pub d_model: usize,
    /// Number of adapter experts.
    pub n_experts: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Adapters activated per token.
    pub top_k: usize,
}

impl LoraMoe {
    /// Build a new LoRAMoE layer. The base weight is random, every adapter starts
    /// at zero delta (`B = 0`).
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero `d_model` / `n_experts`, a `rank` outside
    /// `1 ..= d_model`, or a `top_k` outside `1 ..= n_experts`.
    pub fn new(cfg: LoraMoeConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.d_model == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.d_model });
        }
        if cfg.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_experts,
            });
        }
        if cfg.rank == 0 || cfg.rank > cfg.d_model {
            return Err(MoeError::InvalidHiddenDim { dim: cfg.rank });
        }
        if cfg.top_k == 0 || cfg.top_k > cfg.n_experts {
            return Err(MoeError::InvalidTopK {
                k: cfg.top_k,
                n_experts: cfg.n_experts,
            });
        }

        let router = TopKRouter::new(
            TopKConfig {
                k: cfg.top_k,
                n_experts: cfg.n_experts,
                input_dim: cfg.d_model,
                noise_std: 0.0,
            },
            rng,
        )?;

        let mut experts = Vec::with_capacity(cfg.n_experts);
        for _ in 0..cfg.n_experts {
            experts.push(LoraExpert::new(cfg.d_model, cfg.rank, cfg.alpha, rng)?);
        }

        let mut base = vec![0.0_f32; cfg.d_model * cfg.d_model];
        rng.fill_normal_scaled(&mut base, (1.0 / cfg.d_model as f32).sqrt());

        Ok(Self {
            router,
            experts,
            base,
            d_model: cfg.d_model,
            n_experts: cfg.n_experts,
            rank: cfg.rank,
            top_k: cfg.top_k,
        })
    }

    /// Apply the frozen base projection to a single token: `W_0 · x`.
    ///
    /// # Errors
    /// Propagates `matvec` errors (e.g. a `d_model` mismatch).
    pub fn base_forward(&self, x: &[f32]) -> MoeResult<Vec<f32>> {
        matvec(&self.base, x, self.d_model)
    }

    /// Run the LoRAMoE forward pass.
    ///
    /// # Arguments
    /// * `x` — input activations, row-major `[n_tokens · d_model]`.
    /// * `n_tokens` — number of tokens.
    /// * `d_model` — feature dimension, validated against the layer's `d_model`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input, a `d_model` mismatch, or a token
    /// buffer that is not `n_tokens · d_model` long.
    pub fn forward(&self, x: &[f32], n_tokens: usize, d_model: usize) -> MoeResult<LoraMoeOutput> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        if d_model != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: d_model,
            });
        }
        let expected = n_tokens * self.d_model;
        if x.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let routing = self.router.route(x, n_tokens)?;
        let k = self.top_k;

        let mut gate_weights = vec![0.0_f32; n_tokens * k];
        let mut hidden = vec![0.0_f32; n_tokens * self.d_model];

        for tok in 0..n_tokens {
            let x_tok = &x[tok * self.d_model..(tok + 1) * self.d_model];

            // Renormalise the token's top-k gate scores to sum to 1 (idempotent
            // for k > 1, and maps a lone top-1 score to exactly 1).
            let raw = &routing.scores[tok * k..(tok + 1) * k];
            let denom: f32 = raw.iter().sum::<f32>() + 1e-12;

            let out_slice = &mut hidden[tok * self.d_model..(tok + 1) * self.d_model];
            // Frozen base contribution.
            let base_out = matvec(&self.base, x_tok, self.d_model)?;
            out_slice.copy_from_slice(&base_out);

            // Gated low-rank adapter deltas.
            for slot in 0..k {
                let expert_idx = routing.indices[tok * k + slot];
                if expert_idx >= self.n_experts {
                    return Err(MoeError::ExpertIndexOutOfRange {
                        idx: expert_idx,
                        n_experts: self.n_experts,
                    });
                }
                let g = raw[slot] / denom;
                gate_weights[tok * k + slot] = g;
                let delta = self.experts[expert_idx].delta(x_tok)?;
                for (acc, &d) in out_slice.iter_mut().zip(delta.iter()) {
                    *acc += g * d;
                }
            }
        }

        Ok(LoraMoeOutput {
            hidden,
            gate_indices: routing.indices,
            gate_weights,
        })
    }

    /// Total parameter count (frozen base + gate + all adapters).
    #[must_use]
    pub fn param_count(&self) -> usize {
        let base = self.base.len();
        let gate = self.router.param_count();
        let adapters = self.n_experts * (2 * self.rank * self.d_model);
        base + gate + adapters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Numeric matrix rank via Gaussian elimination with partial pivoting.
    fn matrix_rank(m: &[f32], rows: usize, cols: usize, tol: f32) -> usize {
        let mut a = m.to_vec();
        let mut rank = 0;
        let mut pivot_col = 0;
        for _ in 0..rows {
            if pivot_col >= cols {
                break;
            }
            // Find the pivot row with the largest magnitude in pivot_col.
            let mut best_row = rank;
            let mut best_val = 0.0_f32;
            for r in rank..rows {
                let v = a[r * cols + pivot_col].abs();
                if v > best_val {
                    best_val = v;
                    best_row = r;
                }
            }
            if best_val <= tol {
                pivot_col += 1;
                continue;
            }
            // Swap the pivot row up to row `rank`.
            for c in 0..cols {
                a.swap(rank * cols + c, best_row * cols + c);
            }
            let pivot = a[rank * cols + pivot_col];
            for r in 0..rows {
                if r != rank {
                    let factor = a[r * cols + pivot_col] / pivot;
                    for c in 0..cols {
                        a[r * cols + c] -= factor * a[rank * cols + c];
                    }
                }
            }
            rank += 1;
            pivot_col += 1;
        }
        rank
    }

    #[test]
    fn matrix_rank_sanity() {
        // Full-rank 2x2 identity.
        assert_eq!(matrix_rank(&[1.0, 0.0, 0.0, 1.0], 2, 2, 1e-5), 2);
        // Rank-1 3x3 (every row is a multiple of [1,2,3]).
        let m = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, -1.0, -2.0, -3.0];
        assert_eq!(matrix_rank(&m, 3, 3, 1e-5), 1);
    }

    /// (a) Output equals the frozen base output when all adapters are zero (B = 0 init).
    #[test]
    fn zero_adapters_equal_base() {
        let mut rng = LcgRng::new(11);
        let cfg = LoraMoeConfig {
            d_model: 6,
            n_experts: 4,
            rank: 2,
            top_k: 2,
            alpha: 8.0,
        };
        let layer = LoraMoe::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 4;
        let x: Vec<f32> = (0..n_tokens * 6).map(|i| (i as f32 * 0.17).sin()).collect();
        let out = layer
            .forward(&x, n_tokens, 6)
            .expect("forward should succeed");
        for tok in 0..n_tokens {
            let base = layer
                .base_forward(&x[tok * 6..(tok + 1) * 6])
                .expect("value should be present");
            let got = &out.hidden[tok * 6..(tok + 1) * 6];
            for (g, b) in got.iter().zip(base.iter()) {
                assert!((g - b).abs() < 1e-6, "got {g} != base {b}");
            }
        }
    }

    /// (b) A trained adapter contributes a delta of rank <= r.
    #[test]
    fn adapter_delta_is_low_rank() {
        let mut rng = LcgRng::new(21);
        let d = 6;
        let r = 2;
        let mut expert = LoraExpert::new(d, r, 4.0, &mut rng).expect("new should succeed");
        // Train B to non-zero so the delta is active but still rank <= r.
        rng.fill_normal_scaled(&mut expert.b, 0.5);

        // Assemble the full delta matrix ΔW (d x d): column j = delta(e_j).
        let mut delta_w = vec![0.0_f32; d * d];
        for j in 0..d {
            let mut e_j = vec![0.0_f32; d];
            e_j[j] = 1.0;
            let col = expert.delta(&e_j).expect("delta should succeed");
            for (i, &v) in col.iter().enumerate() {
                delta_w[i * d + j] = v;
            }
        }
        let rank = matrix_rank(&delta_w, d, d, 1e-4);
        assert!(rank <= r, "delta rank {rank} exceeds r {r}");
        // And the delta is genuinely non-trivial (non-zero) here.
        assert!(rank >= 1, "expected an active (non-zero) delta");
    }

    /// (c) Gate weights sum to 1 per token (top-k renormalised).
    #[test]
    fn gate_weights_sum_to_one() {
        let mut rng = LcgRng::new(31);
        let cfg = LoraMoeConfig {
            d_model: 8,
            n_experts: 5,
            rank: 2,
            top_k: 3,
            alpha: 8.0,
        };
        let layer = LoraMoe::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 6;
        let x: Vec<f32> = (0..n_tokens * 8).map(|i| (i as f32 * 0.09).cos()).collect();
        let out = layer
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        for tok in 0..n_tokens {
            let w = &out.gate_weights[tok * 3..tok * 3 + 3];
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "token {tok} gate sum {sum}");
            for &gi in &out.gate_indices[tok * 3..tok * 3 + 3] {
                assert!(gi < 5);
            }
        }
    }

    /// (d) rank <= d_model is enforced; violations error.
    #[test]
    fn rank_validation_errors() {
        let mut rng = LcgRng::new(41);
        // rank > d_model.
        assert!(matches!(
            LoraExpert::new(4, 5, 8.0, &mut rng),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
        // rank == 0.
        assert!(matches!(
            LoraExpert::new(4, 0, 8.0, &mut rng),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
        // Same enforcement at the layer level.
        let bad = LoraMoeConfig {
            d_model: 4,
            n_experts: 3,
            rank: 9,
            top_k: 2,
            alpha: 8.0,
        };
        assert!(matches!(
            LoraMoe::new(bad, &mut rng),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
    }

    /// (e) Outputs are finite, and shape/dim mismatches error.
    #[test]
    fn finite_outputs_and_shape_errors() {
        let mut rng = LcgRng::new(51);
        let cfg = LoraMoeConfig {
            d_model: 8,
            n_experts: 4,
            rank: 3,
            top_k: 2,
            alpha: 8.0,
        };
        let layer = LoraMoe::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 5;
        let x = vec![0.3_f32; n_tokens * 8];
        let out = layer
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        assert_eq!(out.hidden.len(), n_tokens * 8);
        assert!(out.hidden.iter().all(|v| v.is_finite()));

        assert!(matches!(
            layer.forward(&x, n_tokens, 9),
            Err(MoeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.forward(&[0.0_f32; 3], n_tokens, 8),
            Err(MoeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.forward(&[], 0, 8),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn param_count_positive() {
        let mut rng = LcgRng::new(61);
        let layer =
            LoraMoe::new(LoraMoeConfig::default(), &mut rng).expect("value should be present");
        assert!(layer.param_count() > 0);
    }
}
