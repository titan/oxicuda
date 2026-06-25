//! Conditional computation routing: skip the expert entirely for some tokens.
//!
//! Implements the capacity-based conditional-computation mechanism popularised
//! by:
//! Raposo et al. "Mixture-of-Depths: Dynamically allocating compute in
//! transformer-based language models." 2024.
//!
//! A single scalar **router** assigns each token a weight
//! `w_t = σ(g · x_t)`. Only the tokens with the largest weights — up to a
//! capacity `C = ⌈T · capacity_factor⌉` — are *processed* by the expert /
//! sub-layer; the remaining tokens **bypass computation entirely** and are
//! copied through unchanged (a residual identity path). Each processed token's
//! contribution is additionally scaled by its router weight so the router stays
//! on the gradient path:
//!
//! ```text
//! y_t = x_t + w_t · f(x_t)   if token t is selected
//! y_t = x_t                  otherwise (computation skipped)
//! ```
//!
//! Selecting the *top-C* tokens (rather than a fixed per-token threshold) makes
//! the amount of compute deterministic and statically schedulable — the key
//! property that distinguishes conditional computation from plain early-exit.
//! With `capacity_factor ≥ 1` every token is processed and the layer reduces to
//! a standard gated residual block.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Numerically stable logistic sigmoid `σ(z) = 1 / (1 + e^{-z})`.
#[inline]
#[must_use]
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Configuration for [`ConditionalRouter`].
#[derive(Debug, Clone)]
pub struct ConditionalConfig {
    /// Token feature dimension.
    pub input_dim: usize,
    /// Fraction of tokens to process. `C = ⌈T · capacity_factor⌉`, clamped to
    /// `[1, T]`. Must be finite and `> 0`.
    pub capacity_factor: f32,
}

impl ConditionalConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero `input_dim` or a non-positive /
    /// non-finite `capacity_factor`.
    pub fn validate(&self) -> MoeResult<()> {
        if self.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: self.input_dim,
            });
        }
        if !self.capacity_factor.is_finite() || self.capacity_factor <= 0.0 {
            return Err(MoeError::InvalidCapacityFactor {
                factor: self.capacity_factor,
            });
        }
        Ok(())
    }
}

/// Outcome of a conditional routing decision.
#[derive(Debug, Clone)]
pub struct ConditionalRouting {
    /// Per-token router weight `w_t = σ(g · x_t)`, shape `[n_tokens]`.
    pub weights: Vec<f32>,
    /// `true` for the tokens selected for computation, shape `[n_tokens]`.
    pub processed: Vec<bool>,
    /// Indices of the processed tokens, ascending, length `n_processed`.
    pub processed_indices: Vec<usize>,
    /// Number of processed tokens (`= processed_indices.len()`).
    pub capacity: usize,
}

/// A scalar router deciding which tokens receive expert computation.
#[derive(Debug, Clone)]
pub struct ConditionalRouter {
    /// Router projection `g`, shape `[input_dim]`.
    pub gate: Vec<f32>,
    /// Configuration.
    pub config: ConditionalConfig,
}

impl ConditionalRouter {
    /// Create a router with a randomly initialised gate (`N(0, 0.01²)`).
    ///
    /// # Errors
    /// Propagates [`ConditionalConfig::validate`].
    pub fn new(cfg: ConditionalConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        cfg.validate()?;
        let mut gate = vec![0.0_f32; cfg.input_dim];
        rng.fill_normal_scaled(&mut gate, 0.01);
        Ok(Self { gate, config: cfg })
    }

    /// Capacity (number of processed tokens) for `n_tokens`.
    #[must_use]
    pub fn capacity(&self, n_tokens: usize) -> usize {
        let raw = (n_tokens as f32 * self.config.capacity_factor).ceil() as usize;
        raw.clamp(1, n_tokens.max(1))
    }

    /// Decide which tokens to process for input `x` (shape `[n_tokens·d]`).
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input, a shape mismatch, or a non-finite
    /// router weight.
    pub fn route(&self, x: &[f32], n_tokens: usize) -> MoeResult<ConditionalRouting> {
        let d = self.config.input_dim;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = n_tokens * d;
        if x.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut weights = vec![0.0_f32; n_tokens];
        for (tok, w) in weights.iter_mut().enumerate() {
            let row = &x[tok * d..(tok + 1) * d];
            let logit: f32 = row
                .iter()
                .zip(self.gate.iter())
                .map(|(&xi, &gi)| xi * gi)
                .sum();
            let sw = sigmoid(logit);
            if !sw.is_finite() {
                return Err(MoeError::NanEncountered {
                    context: "conditional router weight".to_string(),
                });
            }
            *w = sw;
        }

        let capacity = self.capacity(n_tokens);

        // Select the top-`capacity` tokens by router weight. Build an index
        // permutation sorted by weight descending; ties broken by token order
        // (stable) so the result is deterministic.
        let mut order: Vec<usize> = (0..n_tokens).collect();
        order.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut processed = vec![false; n_tokens];
        for &tok in order.iter().take(capacity) {
            processed[tok] = true;
        }
        let processed_indices: Vec<usize> = (0..n_tokens).filter(|&t| processed[t]).collect();

        Ok(ConditionalRouting {
            weights,
            processed,
            processed_indices,
            capacity,
        })
    }

    /// Gather the processed tokens into a dense `[n_processed·d]` buffer ready
    /// for an expert forward pass.
    ///
    /// # Errors
    /// Returns [`MoeError`] on a shape mismatch.
    pub fn gather(
        &self,
        x: &[f32],
        routing: &ConditionalRouting,
        n_tokens: usize,
    ) -> MoeResult<Vec<f32>> {
        let d = self.config.input_dim;
        if x.len() != n_tokens * d {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens * d,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; routing.processed_indices.len() * d];
        for (slot, &tok) in routing.processed_indices.iter().enumerate() {
            out[slot * d..(slot + 1) * d].copy_from_slice(&x[tok * d..(tok + 1) * d]);
        }
        Ok(out)
    }

    /// Combine expert outputs back with skipped tokens via the residual path:
    /// `y_t = x_t + w_t · f(x_t)` for processed tokens, `y_t = x_t` otherwise.
    ///
    /// `expert_out` is the dense `[n_processed·d]` output for the gathered
    /// tokens (in `routing.processed_indices` order).
    ///
    /// # Errors
    /// Returns [`MoeError`] on a shape mismatch.
    pub fn combine(
        &self,
        x: &[f32],
        expert_out: &[f32],
        routing: &ConditionalRouting,
        n_tokens: usize,
    ) -> MoeResult<Vec<f32>> {
        let d = self.config.input_dim;
        if x.len() != n_tokens * d {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens * d,
                got: x.len(),
            });
        }
        let n_processed = routing.processed_indices.len();
        if expert_out.len() != n_processed * d {
            return Err(MoeError::DimensionMismatch {
                expected: n_processed * d,
                got: expert_out.len(),
            });
        }

        // Start from the residual (every token keeps its input).
        let mut out = x.to_vec();
        for (slot, &tok) in routing.processed_indices.iter().enumerate() {
            let w = routing.weights[tok];
            let f = &expert_out[slot * d..(slot + 1) * d];
            let dst = &mut out[tok * d..(tok + 1) * d];
            for (o, &fi) in dst.iter_mut().zip(f.iter()) {
                *o += w * fi;
            }
        }
        Ok(out)
    }

    /// Fraction of tokens whose computation is skipped (`1 - C/T`).
    #[must_use]
    pub fn skip_fraction(&self, n_tokens: usize) -> f32 {
        if n_tokens == 0 {
            return 0.0;
        }
        1.0 - self.capacity(n_tokens) as f32 / n_tokens as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_monotone_and_bounded() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(50.0) > 0.99 && sigmoid(50.0) <= 1.0);
        assert!(sigmoid(-50.0) < 0.01 && sigmoid(-50.0) >= 0.0);
        assert!(sigmoid(1.0) > sigmoid(-1.0));
    }

    #[test]
    fn capacity_clamped_and_rounded() {
        let mut rng = LcgRng::new(1);
        let cfg = ConditionalConfig {
            input_dim: 4,
            capacity_factor: 0.5,
        };
        let r = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        // ⌈10·0.5⌉ = 5
        assert_eq!(r.capacity(10), 5);
        // ⌈3·0.5⌉ = 2
        assert_eq!(r.capacity(3), 2);
    }

    #[test]
    fn exactly_capacity_tokens_processed() {
        let mut rng = LcgRng::new(2);
        let cfg = ConditionalConfig {
            input_dim: 6,
            capacity_factor: 0.5,
        };
        let r = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 8;
        let mut x = vec![0.0_f32; n_tokens * 6];
        rng.fill_normal_scaled(&mut x, 1.0);
        let routing = r.route(&x, n_tokens).expect("route should succeed");
        let n_proc = routing.processed.iter().filter(|&&p| p).count();
        assert_eq!(n_proc, routing.capacity);
        assert_eq!(n_proc, 4);
        assert_eq!(routing.processed_indices.len(), 4);
    }

    #[test]
    fn skipped_tokens_pass_through_unchanged() {
        let mut rng = LcgRng::new(3);
        let d = 4;
        let cfg = ConditionalConfig {
            input_dim: d,
            capacity_factor: 0.5,
        };
        let r = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 6;
        let mut x = vec![0.0_f32; n_tokens * d];
        rng.fill_normal_scaled(&mut x, 0.7);
        let routing = r.route(&x, n_tokens).expect("route should succeed");

        // A dummy "expert" that returns all-ones; processed tokens must change,
        // skipped tokens must be bit-identical to the input.
        let gathered = r
            .gather(&x, &routing, n_tokens)
            .expect("gather should succeed");
        let expert_out = vec![1.0_f32; gathered.len()];
        let y = r
            .combine(&x, &expert_out, &routing, n_tokens)
            .expect("combine should succeed");

        for tok in 0..n_tokens {
            let xi = &x[tok * d..(tok + 1) * d];
            let yi = &y[tok * d..(tok + 1) * d];
            if routing.processed[tok] {
                // y = x + w·1, w>0 ⇒ strictly greater than x.
                let w = routing.weights[tok];
                for (a, b) in xi.iter().zip(yi.iter()) {
                    assert!((b - (a + w)).abs() < 1e-5);
                }
            } else {
                for (a, b) in xi.iter().zip(yi.iter()) {
                    assert!((a - b).abs() < 1e-7, "skipped token {tok} changed");
                }
            }
        }
    }

    #[test]
    fn full_capacity_processes_all() {
        let mut rng = LcgRng::new(4);
        let cfg = ConditionalConfig {
            input_dim: 4,
            capacity_factor: 1.0,
        };
        let r = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 7;
        let mut x = vec![0.0_f32; n_tokens * 4];
        rng.fill_normal_scaled(&mut x, 1.0);
        let routing = r.route(&x, n_tokens).expect("route should succeed");
        assert!(routing.processed.iter().all(|&p| p));
        assert!((r.skip_fraction(n_tokens)).abs() < 1e-6);
    }

    #[test]
    fn highest_weight_tokens_are_selected() {
        // Construct an input where the gate is a known direction so the token
        // with the largest projection is guaranteed selected at capacity 1.
        let cfg = ConditionalConfig {
            input_dim: 2,
            capacity_factor: 0.25, // ⌈4·0.25⌉ = 1
        };
        let mut rng = LcgRng::new(5);
        let mut router = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        router.gate = vec![1.0, 0.0]; // weight = σ(x[0])
        let n_tokens = 4;
        // token 2 has the largest first coordinate.
        let x = vec![
            0.0, 9.0, // tok 0
            1.0, 9.0, // tok 1
            5.0, 9.0, // tok 2  <- max
            -2.0, 9.0, // tok 3
        ];
        let routing = router.route(&x, n_tokens).expect("route should succeed");
        assert_eq!(routing.capacity, 1);
        assert_eq!(routing.processed_indices, vec![2]);
    }

    #[test]
    fn route_shape_mismatch_errors() {
        let mut rng = LcgRng::new(6);
        let cfg = ConditionalConfig {
            input_dim: 4,
            capacity_factor: 0.5,
        };
        let r = ConditionalRouter::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.0_f32; 9]; // not a multiple of 4
        assert!(matches!(
            r.route(&x, 2),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }
}
