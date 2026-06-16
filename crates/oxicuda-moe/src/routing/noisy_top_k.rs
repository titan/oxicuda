//! Noisy top-k gating (Shazeer et al. 2017).
//!
//! Implements the gating network from:
//! Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated
//! Mixture-of-Experts Layer." ICLR 2017.
//!
//! The gate adds *tunable* Gaussian noise to the clean router logits **before**
//! the top-k selection:
//!
//! ```text
//! H[t,e] = (x · W_g)[e] + StandardNormal() · softplus( (x · W_noise)[e] )
//! G[t,·] = softmax( keep_top_k(H[t,·], k) )
//! ```
//!
//! Crucially the noise standard deviation `softplus(x · W_noise)` is itself a
//! learned, per-expert, input-dependent quantity — distinct from the fixed
//! `N(0, σ²)` jitter in [`crate::routing::top_k`]. The noise (a) encourages
//! exploration so under-used experts can win the top-k, and (b) is the basis of
//! Shazeer's differentiable *load* importance for the load-balancing loss.
//!
//! Non-selected experts receive logit `-∞` prior to the softmax, so the returned
//! gate weights are sparse (exactly `k` non-zeros per token) and sum to `1`.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::routing::top_k::topk;

/// Configuration for noisy top-k gating.
#[derive(Debug, Clone)]
pub struct NoisyTopKConfig {
    /// Number of experts `E`.
    pub n_experts: usize,
    /// Input feature dimension `d`.
    pub input_dim: usize,
    /// Number of experts selected per token `k` (`1 ≤ k ≤ E`).
    pub k: usize,
    /// If `false`, noise is disabled (evaluation mode → deterministic gate).
    pub noisy: bool,
}

/// Output of noisy top-k gating.
#[derive(Debug, Clone)]
pub struct NoisyTopKResult {
    /// Sparse gate weights `[T×E]` (row-major); exactly `k` non-zeros per token,
    /// each row summing to `1`.
    pub gates: Vec<f32>,
    /// Selected expert indices `[T×k]` (descending noisy-logit order).
    pub indices: Vec<usize>,
    /// Top-k gate scores `[T×k]` aligned with `indices`.
    pub top_scores: Vec<f32>,
}

/// Numerically-stable `softplus(x) = ln(1 + eˣ)`.
#[must_use]
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Noisy top-k router with clean and noise gate projections.
pub struct NoisyTopKRouter {
    config: NoisyTopKConfig,
    /// Clean gate weight `W_g ∈ R^{E×d}` (row-major).
    w_gate: Vec<f32>,
    /// Noise gate weight `W_noise ∈ R^{E×d}` (row-major).
    w_noise: Vec<f32>,
}

impl NoisyTopKRouter {
    /// Create a router with Xavier-style random gate weights.
    ///
    /// # Errors
    /// Returns [`MoeError`] when `n_experts == 0`, `input_dim == 0`, or
    /// `k` is `0` / greater than `n_experts`.
    pub fn new(config: NoisyTopKConfig, rng: &mut LcgRng) -> MoeResult<Self> {
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
        if config.k == 0 || config.k > config.n_experts {
            return Err(MoeError::InvalidTopK {
                k: config.k,
                n_experts: config.n_experts,
            });
        }
        let n = config.n_experts * config.input_dim;
        let scale = (1.0 / config.input_dim as f32).sqrt();
        let mut w_gate = vec![0.0_f32; n];
        let mut w_noise = vec![0.0_f32; n];
        rng.fill_normal(&mut w_gate);
        rng.fill_normal(&mut w_noise);
        for v in w_gate.iter_mut() {
            *v *= scale;
        }
        for v in w_noise.iter_mut() {
            *v *= scale;
        }
        Ok(Self {
            config,
            w_gate,
            w_noise,
        })
    }

    /// Construct from explicit gate / noise weights (for deterministic tests).
    ///
    /// # Errors
    /// Returns [`MoeError`] on invalid config or a weight-length mismatch.
    pub fn with_weights(
        config: NoisyTopKConfig,
        w_gate: Vec<f32>,
        w_noise: Vec<f32>,
    ) -> MoeResult<Self> {
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
        if config.k == 0 || config.k > config.n_experts {
            return Err(MoeError::InvalidTopK {
                k: config.k,
                n_experts: config.n_experts,
            });
        }
        let expected = config.n_experts * config.input_dim;
        if w_gate.len() != expected || w_noise.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: w_gate.len().min(w_noise.len()),
            });
        }
        Ok(Self {
            config,
            w_gate,
            w_noise,
        })
    }

    /// Total parameter count (`2·E·d`).
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.w_gate.len() + self.w_noise.len()
    }

    /// Route a batch of tokens `x ∈ R^{T×d}` (row-major).
    ///
    /// When `config.noisy` is `true`, standard-normal noise scaled by the learned
    /// per-expert `softplus` standard deviation is added before top-k selection.
    ///
    /// # Errors
    /// Returns [`MoeError`] for an empty input or a `x`/`T·d` length mismatch.
    pub fn route(
        &self,
        x: &[f32],
        n_tokens: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<NoisyTopKResult> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = n_tokens * cfg.input_dim;
        if x.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let n_e = cfg.n_experts;
        let d = cfg.input_dim;
        let k = cfg.k;
        let mut gates = vec![0.0_f32; n_tokens * n_e];
        let mut indices = vec![0_usize; n_tokens * k];
        let mut top_scores = vec![0.0_f32; n_tokens * k];

        for t in 0..n_tokens {
            let x_row = &x[t * d..(t + 1) * d];
            // Clean logits and noise-std logits.
            let mut noisy_logits = vec![0.0_f32; n_e];
            for (e, slot) in noisy_logits.iter_mut().enumerate() {
                let wg = &self.w_gate[e * d..(e + 1) * d];
                let wn = &self.w_noise[e * d..(e + 1) * d];
                let mut clean = 0.0_f32;
                let mut raw_std = 0.0_f32;
                for i in 0..d {
                    clean += x_row[i] * wg[i];
                    raw_std += x_row[i] * wn[i];
                }
                let mut h = clean;
                if cfg.noisy {
                    let (z, _) = rng.next_normal_pair();
                    h += z * softplus(raw_std);
                }
                *slot = h;
            }

            // Top-k over the noisy logits.
            let (topk_vals, topk_idx) = topk(&noisy_logits, k)?;

            // Softmax over the k selected logits (non-selected → 0 in `gates`).
            let max_v = topk_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0_f32;
            let mut exps = vec![0.0_f32; k];
            for j in 0..k {
                let e = (topk_vals[j] - max_v).exp();
                exps[j] = e;
                sum_exp += e;
            }
            let denom = sum_exp + 1e-9;
            for j in 0..k {
                let g = exps[j] / denom;
                let e = topk_idx[j];
                gates[t * n_e + e] = g;
                indices[t * k + j] = e;
                top_scores[t * k + j] = g;
            }
        }

        if gates.iter().any(|v| !v.is_finite()) {
            return Err(MoeError::NanEncountered {
                context: "noisy_top_k".to_string(),
            });
        }

        Ok(NoisyTopKResult {
            gates,
            indices,
            top_scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_experts: usize, input_dim: usize, k: usize, noisy: bool) -> NoisyTopKConfig {
        NoisyTopKConfig {
            n_experts,
            input_dim,
            k,
            noisy,
        }
    }

    #[test]
    fn softplus_is_positive_and_monotone() {
        assert!(softplus(-50.0) >= 0.0);
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(3.0) > softplus(1.0));
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-5);
        assert!(softplus(100.0).is_finite());
    }

    #[test]
    fn new_zero_experts_errors() {
        let mut rng = LcgRng::new(1);
        assert!(NoisyTopKRouter::new(cfg(0, 8, 1, true), &mut rng).is_err());
    }

    #[test]
    fn new_zero_input_dim_errors() {
        let mut rng = LcgRng::new(1);
        assert!(NoisyTopKRouter::new(cfg(4, 0, 1, true), &mut rng).is_err());
    }

    #[test]
    fn new_invalid_k_errors() {
        let mut rng = LcgRng::new(1);
        assert!(NoisyTopKRouter::new(cfg(4, 8, 0, true), &mut rng).is_err());
        assert!(NoisyTopKRouter::new(cfg(4, 8, 5, true), &mut rng).is_err());
    }

    #[test]
    fn param_count_is_two_e_d() {
        let mut rng = LcgRng::new(2);
        let router =
            NoisyTopKRouter::new(cfg(8, 16, 2, true), &mut rng).expect("value should be present");
        assert_eq!(router.param_count(), 2 * 8 * 16);
    }

    #[test]
    fn gates_rows_sum_to_one() {
        let mut rng = LcgRng::new(3);
        let router =
            NoisyTopKRouter::new(cfg(8, 16, 2, true), &mut rng).expect("value should be present");
        let n_tokens = 12;
        let x: Vec<f32> = (0..n_tokens * 16)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        for t in 0..n_tokens {
            let row_sum: f32 = res.gates[t * 8..(t + 1) * 8].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "token {t} sum {row_sum}");
        }
    }

    #[test]
    fn exactly_k_nonzero_gates_per_token() {
        let mut rng = LcgRng::new(4);
        let k = 3;
        let router =
            NoisyTopKRouter::new(cfg(8, 16, k, true), &mut rng).expect("value should be present");
        let n_tokens = 10;
        let x: Vec<f32> = (0..n_tokens * 16)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        for t in 0..n_tokens {
            let nz = res.gates[t * 8..(t + 1) * 8]
                .iter()
                .filter(|&&v| v > 0.0)
                .count();
            assert_eq!(nz, k, "token {t} has {nz} non-zeros, expected {k}");
        }
    }

    #[test]
    fn indices_in_range() {
        let mut rng = LcgRng::new(5);
        let router =
            NoisyTopKRouter::new(cfg(6, 12, 2, true), &mut rng).expect("value should be present");
        let n_tokens = 8;
        let x: Vec<f32> = (0..n_tokens * 12)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        assert!(res.indices.iter().all(|&e| e < 6));
        assert_eq!(res.indices.len(), n_tokens * 2);
    }

    #[test]
    fn gates_finite() {
        let mut rng = LcgRng::new(6);
        let router =
            NoisyTopKRouter::new(cfg(8, 16, 2, true), &mut rng).expect("value should be present");
        let n_tokens = 16;
        let x = vec![5.0_f32; n_tokens * 16]; // large magnitude
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        assert!(res.gates.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_input_errors() {
        let mut rng = LcgRng::new(7);
        let router =
            NoisyTopKRouter::new(cfg(4, 8, 1, true), &mut rng).expect("value should be present");
        let x: Vec<f32> = vec![];
        assert!(matches!(
            router.route(&x, 0, &mut rng),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn input_length_mismatch_errors() {
        let mut rng = LcgRng::new(8);
        let router =
            NoisyTopKRouter::new(cfg(4, 8, 1, true), &mut rng).expect("value should be present");
        let x = vec![0.0_f32; 3 * 7]; // wrong: should be 3*8
        assert!(matches!(
            router.route(&x, 3, &mut rng),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn noiseless_deterministic_and_matches_clean_argmax() {
        // With noisy=false, identical W_noise is irrelevant; result is repeatable.
        let n_e = 4;
        let d = 3;
        // Diagonal-ish gate: expert e prefers input dim e.
        let mut w_gate = vec![0.0_f32; n_e * d];
        for e in 0..d.min(n_e) {
            w_gate[e * d + e] = 1.0;
        }
        let w_noise = vec![0.0_f32; n_e * d];
        let router = NoisyTopKRouter::with_weights(cfg(n_e, d, 1, false), w_gate, w_noise)
            .expect("value should be present");
        // Token strongly aligned with expert 2.
        let x = vec![0.1_f32, 0.1, 5.0];
        let mut rng_a = LcgRng::new(0);
        let mut rng_b = LcgRng::new(123);
        let a = router
            .route(&x, 1, &mut rng_a)
            .expect("route should succeed");
        let b = router
            .route(&x, 1, &mut rng_b)
            .expect("route should succeed");
        assert_eq!(a.indices, b.indices, "noiseless must be RNG-independent");
        assert_eq!(a.indices[0], 2, "should pick expert 2");
    }

    #[test]
    fn noisy_can_change_selection_vs_noiseless() {
        // Construct a near-tie between experts with large noise std so noise
        // sometimes flips the winner.
        let n_e = 2;
        let d = 1;
        let w_gate = vec![0.01_f32, 0.0]; // expert 0 marginally preferred
        let w_noise = vec![3.0_f32, 3.0]; // large, equal noise std
        let router = NoisyTopKRouter::with_weights(cfg(n_e, d, 1, true), w_gate, w_noise)
            .expect("value should be present");
        let x = vec![1.0_f32];
        let mut rng = LcgRng::new(42);
        let mut saw_expert_1 = false;
        for _ in 0..200 {
            let r = router.route(&x, 1, &mut rng).expect("route should succeed");
            if r.indices[0] == 1 {
                saw_expert_1 = true;
                break;
            }
        }
        assert!(saw_expert_1, "noise should occasionally flip the winner");
    }

    #[test]
    fn with_weights_length_mismatch_errors() {
        let w_gate = vec![0.0_f32; 4 * 8];
        let w_noise = vec![0.0_f32; 10]; // wrong
        assert!(NoisyTopKRouter::with_weights(cfg(4, 8, 1, true), w_gate, w_noise).is_err());
    }

    #[test]
    fn top_scores_match_gate_entries() {
        let mut rng = LcgRng::new(11);
        let router =
            NoisyTopKRouter::new(cfg(6, 10, 2, true), &mut rng).expect("value should be present");
        let n_tokens = 5;
        let x: Vec<f32> = (0..n_tokens * 10)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        for t in 0..n_tokens {
            for j in 0..2 {
                let e = res.indices[t * 2 + j];
                let g = res.gates[t * 6 + e];
                assert!((g - res.top_scores[t * 2 + j]).abs() < 1e-6);
            }
        }
    }
}
