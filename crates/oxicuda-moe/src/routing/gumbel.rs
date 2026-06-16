//! Gumbel-softmax stochastic routing for differentiable MoE gating.
//!
//! Hard top-k routing (`argmax`) has zero gradient with respect to the router
//! logits, so the router can only be trained through the auxiliary
//! load-balancing loss. **Gumbel-softmax** routing (Jang et al. 2017;
//! Maddison et al. 2017) replaces the deterministic `argmax` with a
//! *stochastic* relaxation that is differentiable end-to-end:
//!
//! 1. Draw i.i.d. Gumbel noise `g_i = −log(−log U_i)`, `U_i ~ Uniform(0,1)`.
//! 2. Form perturbed logits `(logit_i + g_i) / τ` and take their softmax — the
//!    **Gumbel-softmax** sample, a point on the simplex whose sharpness is set
//!    by the temperature `τ` (small `τ` → near one-hot, large `τ` → uniform).
//! 3. The **Gumbel-max** trick: `argmax_i(logit_i + g_i)` is an exact sample
//!    from `softmax(logit)`, used for the hard forward selection while the soft
//!    weights provide the straight-through gradient.
//!
//! This module provides:
//!
//! * [`gumbel_softmax`] — the soft simplex sample (a `[T×E]` distribution).
//! * [`GumbelRouter`] — a router that, per token, draws the Gumbel-softmax,
//!   selects the top-`k` experts by the perturbed logits (stochastic discrete
//!   routing), and returns straight-through (hard-forward / soft-backward)
//!   combine weights renormalised over the chosen experts.
//!
//! At evaluation time noise can be disabled (`noisy = false`), collapsing the
//! sampler to ordinary temperature-scaled top-k.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Configuration for the Gumbel-softmax router.
#[derive(Debug, Clone)]
pub struct GumbelConfig {
    /// Number of experts `E`.
    pub n_experts: usize,
    /// Router input dimension `d`.
    pub input_dim: usize,
    /// Number of experts selected per token (`1 ≤ k ≤ E`).
    pub k: usize,
    /// Softmax temperature `τ > 0` (anneal toward 0 during training).
    pub temperature: f32,
    /// Whether to add Gumbel noise (training) or route deterministically (eval).
    pub noisy: bool,
}

/// Result of Gumbel routing for a batch of tokens.
#[derive(Debug, Clone)]
pub struct GumbelRouteResult {
    /// Selected expert indices, shape `[n_tokens * k]` (row-major).
    pub indices: Vec<usize>,
    /// Combine weights for the selected experts, shape `[n_tokens * k]`,
    /// renormalised to sum to 1 per token (straight-through forward values).
    pub weights: Vec<f32>,
    /// Full Gumbel-softmax distribution, shape `[n_tokens * n_experts]`.
    pub soft: Vec<f32>,
}

/// Sample one standard Gumbel(0,1) variate via the inverse-CDF method.
#[inline]
fn sample_gumbel(rng: &mut LcgRng) -> f32 {
    let u = rng.next_f32().clamp(1e-9, 1.0 - 1e-9);
    -(-u.ln()).ln()
}

/// Numerically stable softmax over a row, with temperature scaling.
fn softmax_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| ((l - max) * inv_t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Compute the Gumbel-softmax distribution for one logit row.
///
/// With `noisy = false` this is just the temperature-scaled softmax.
fn gumbel_softmax_row(logits: &[f32], temperature: f32, noisy: bool, rng: &mut LcgRng) -> Vec<f32> {
    if noisy {
        let perturbed: Vec<f32> = logits.iter().map(|&l| l + sample_gumbel(rng)).collect();
        softmax_temp(&perturbed, temperature)
    } else {
        softmax_temp(logits, temperature)
    }
}

/// Compute the full Gumbel-softmax distribution for a batch of logit rows.
///
/// # Arguments
/// * `logits` — router logits, shape `[n_tokens * n_experts]`.
/// * `n_tokens`, `n_experts` — batch and expert counts.
/// * `temperature` — softmax temperature `τ > 0`.
/// * `noisy` — add Gumbel noise (`true`) or plain softmax (`false`).
///
/// # Errors
/// Returns [`MoeError::EmptyInput`] for zero tokens,
/// [`MoeError::InvalidExpertCount`] for zero experts,
/// [`MoeError::DimensionMismatch`] on a logits-length error, and
/// [`MoeError::Internal`] for a non-positive / non-finite temperature.
pub fn gumbel_softmax(
    logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
    temperature: f32,
    noisy: bool,
    rng: &mut LcgRng,
) -> MoeResult<Vec<f32>> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    if logits.len() != n_tokens * n_experts {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens * n_experts,
            got: logits.len(),
        });
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(MoeError::Internal {
            msg: format!("temperature must be positive and finite, got {temperature}"),
        });
    }
    let mut out = vec![0.0_f32; n_tokens * n_experts];
    for tok in 0..n_tokens {
        let row = &logits[tok * n_experts..(tok + 1) * n_experts];
        let dist = gumbel_softmax_row(row, temperature, noisy, rng);
        out[tok * n_experts..(tok + 1) * n_experts].copy_from_slice(&dist);
    }
    Ok(out)
}

/// Gumbel-softmax stochastic router with a learned linear gate.
pub struct GumbelRouter {
    /// Router weight matrix `[input_dim × n_experts]` (row-major).
    pub weight: Vec<f32>,
    /// Configuration.
    pub config: GumbelConfig,
}

impl GumbelRouter {
    /// Create a router with Xavier-initialised gate weights.
    ///
    /// # Errors
    /// Returns [`MoeError::InvalidExpertCount`] / [`MoeError::InvalidInputDim`]
    /// / [`MoeError::InvalidTopK`] / [`MoeError::Internal`] for invalid config.
    pub fn new(config: GumbelConfig, rng: &mut LcgRng) -> MoeResult<Self> {
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
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(MoeError::Internal {
                msg: format!(
                    "temperature must be positive and finite, got {}",
                    config.temperature
                ),
            });
        }
        let fan = (config.input_dim + config.n_experts) as f32;
        let scale = (6.0 / fan).sqrt();
        let mut weight = vec![0.0_f32; config.input_dim * config.n_experts];
        for w in &mut weight {
            *w = (rng.next_f32() * 2.0 - 1.0) * scale;
        }
        Ok(Self { weight, config })
    }

    /// Project tokens `x` `[n_tokens × input_dim]` to logits
    /// `[n_tokens × n_experts]`.
    fn logits(&self, x: &[f32], n_tokens: usize) -> MoeResult<Vec<f32>> {
        let d = self.config.input_dim;
        let e = self.config.n_experts;
        if x.len() != n_tokens * d {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens * d,
                got: x.len(),
            });
        }
        let mut logits = vec![0.0_f32; n_tokens * e];
        for tok in 0..n_tokens {
            let xrow = &x[tok * d..(tok + 1) * d];
            for j in 0..e {
                let mut acc = 0.0_f32;
                for (i, &xi) in xrow.iter().enumerate() {
                    acc += xi * self.weight[i * e + j];
                }
                logits[tok * e + j] = acc;
            }
        }
        Ok(logits)
    }

    /// Route a batch of tokens, returning stochastic top-`k` selections with
    /// straight-through combine weights.
    ///
    /// # Errors
    /// Propagates projection / sampling errors.
    pub fn route(
        &self,
        x: &[f32],
        n_tokens: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<GumbelRouteResult> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let e = self.config.n_experts;
        let k = self.config.k;
        let logits = self.logits(x, n_tokens)?;

        let mut indices = vec![0usize; n_tokens * k];
        let mut weights = vec![0.0_f32; n_tokens * k];
        let mut soft = vec![0.0_f32; n_tokens * e];

        for tok in 0..n_tokens {
            let row = &logits[tok * e..(tok + 1) * e];
            // Perturbed logits drive the *discrete* selection (Gumbel-max), and
            // their softmax is the differentiable relaxation.
            let perturbed: Vec<f32> = if self.config.noisy {
                row.iter().map(|&l| l + sample_gumbel(rng)).collect()
            } else {
                row.to_vec()
            };
            let dist = softmax_temp(&perturbed, self.config.temperature);
            soft[tok * e..(tok + 1) * e].copy_from_slice(&dist);

            // Top-k by perturbed logit (descending).
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| {
                perturbed[b]
                    .partial_cmp(&perturbed[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let chosen = &order[..k];

            // Straight-through combine weights: take the soft probabilities of
            // the chosen experts and renormalise so they sum to 1 per token.
            let mut wsum = 0.0_f32;
            for (slot, &exp) in chosen.iter().enumerate() {
                indices[tok * k + slot] = exp;
                weights[tok * k + slot] = dist[exp];
                wsum += dist[exp];
            }
            if wsum > 1e-12 {
                for slot in 0..k {
                    weights[tok * k + slot] /= wsum;
                }
            } else {
                // Degenerate row: fall back to uniform over the chosen experts.
                let uniform = 1.0 / k as f32;
                for slot in 0..k {
                    weights[tok * k + slot] = uniform;
                }
            }
        }

        Ok(GumbelRouteResult {
            indices,
            weights,
            soft,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg(n_experts: usize, k: usize) -> GumbelConfig {
        GumbelConfig {
            n_experts,
            input_dim: 16,
            k,
            temperature: 1.0,
            noisy: true,
        }
    }

    #[test]
    fn gumbel_softmax_rows_sum_to_one() {
        let mut rng = LcgRng::new(1);
        let n_tokens = 8;
        let n_experts = 6;
        let logits = vec![0.5_f32; n_tokens * n_experts];
        let dist = gumbel_softmax(&logits, n_tokens, n_experts, 1.0, true, &mut rng)
            .expect("gumbel_softmax should succeed");
        for tok in 0..n_tokens {
            let s: f32 = dist[tok * n_experts..(tok + 1) * n_experts].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "token {tok} sum {s}");
        }
    }

    #[test]
    fn gumbel_softmax_values_in_unit_range() {
        let mut rng = LcgRng::new(2);
        let mut logits = vec![0.0_f32; 4 * 5];
        rng.fill_normal(&mut logits);
        let dist = gumbel_softmax(&logits, 4, 5, 0.5, true, &mut rng)
            .expect("gumbel_softmax should succeed");
        assert!(dist.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn gumbel_softmax_low_temp_sharper() {
        // Lower temperature concentrates mass on the max logit.
        let mut rng = LcgRng::new(3);
        let logits = vec![0.0_f32, 3.0, 0.0, 0.0]; // expert 1 dominates
        // Disable noise to isolate the temperature effect.
        let hot = gumbel_softmax(&logits, 1, 4, 2.0, false, &mut rng)
            .expect("gumbel_softmax should succeed");
        let cold = gumbel_softmax(&logits, 1, 4, 0.2, false, &mut rng)
            .expect("gumbel_softmax should succeed");
        assert!(
            cold[1] > hot[1],
            "cold max prob {} should exceed hot {}",
            cold[1],
            hot[1]
        );
    }

    #[test]
    fn gumbel_softmax_no_noise_is_softmax() {
        let mut rng = LcgRng::new(4);
        // Without noise the result is deterministic and matches plain softmax.
        let logits = vec![1.0_f32, 2.0, 0.0];
        let d1 = gumbel_softmax(&logits, 1, 3, 1.0, false, &mut rng)
            .expect("gumbel_softmax should succeed");
        let mut rng2 = LcgRng::new(999);
        let d2 = gumbel_softmax(&logits, 1, 3, 1.0, false, &mut rng2)
            .expect("gumbel_softmax should succeed");
        // Different RNGs, same output (no noise consumed).
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn gumbel_softmax_empty_errors() {
        let mut rng = LcgRng::new(5);
        assert!(matches!(
            gumbel_softmax(&[], 0, 4, 1.0, true, &mut rng),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn gumbel_softmax_bad_temp_errors() {
        let mut rng = LcgRng::new(6);
        let logits = vec![0.0_f32; 4];
        assert!(matches!(
            gumbel_softmax(&logits, 1, 4, 0.0, true, &mut rng),
            Err(MoeError::Internal { .. })
        ));
    }

    #[test]
    fn gumbel_softmax_dim_mismatch_errors() {
        let mut rng = LcgRng::new(7);
        let logits = vec![0.0_f32; 10]; // not 2*4
        assert!(matches!(
            gumbel_softmax(&logits, 2, 4, 1.0, true, &mut rng),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn router_new_validates_config() {
        let mut rng = LcgRng::new(8);
        // k > n_experts.
        let bad = GumbelConfig {
            k: 5,
            ..base_cfg(4, 5)
        };
        assert!(matches!(
            GumbelRouter::new(bad, &mut rng),
            Err(MoeError::InvalidTopK { .. })
        ));
        // zero experts.
        assert!(matches!(
            GumbelRouter::new(base_cfg(0, 1), &mut rng),
            Err(MoeError::InvalidExpertCount { .. })
        ));
        // bad temperature.
        let bad_t = GumbelConfig {
            temperature: -1.0,
            ..base_cfg(4, 1)
        };
        assert!(matches!(
            GumbelRouter::new(bad_t, &mut rng),
            Err(MoeError::Internal { .. })
        ));
    }

    #[test]
    fn router_indices_valid_and_distinct() {
        let mut rng = LcgRng::new(9);
        let router = GumbelRouter::new(base_cfg(8, 2), &mut rng).expect("value should be present");
        let n_tokens = 16;
        let x = vec![0.3_f32; n_tokens * 16];
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        for tok in 0..n_tokens {
            let a = res.indices[tok * 2];
            let b = res.indices[tok * 2 + 1];
            assert!(a < 8 && b < 8, "index out of range");
            assert_ne!(a, b, "top-2 experts must be distinct");
        }
    }

    #[test]
    fn router_weights_sum_to_one() {
        let mut rng = LcgRng::new(10);
        let router = GumbelRouter::new(base_cfg(6, 2), &mut rng).expect("value should be present");
        let n_tokens = 10;
        let mut x = vec![0.0_f32; n_tokens * 16];
        rng.fill_normal(&mut x);
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        for tok in 0..n_tokens {
            let s: f32 = res.weights[tok * 2..tok * 2 + 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "token {tok} weight sum {s}");
        }
    }

    #[test]
    fn router_soft_distribution_valid() {
        let mut rng = LcgRng::new(11);
        let router = GumbelRouter::new(base_cfg(5, 1), &mut rng).expect("value should be present");
        let n_tokens = 8;
        let x = vec![0.5_f32; n_tokens * 16];
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        assert_eq!(res.soft.len(), n_tokens * 5);
        for tok in 0..n_tokens {
            let s: f32 = res.soft[tok * 5..(tok + 1) * 5].iter().sum();
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn router_top1_weight_is_one() {
        let mut rng = LcgRng::new(12);
        let router = GumbelRouter::new(base_cfg(4, 1), &mut rng).expect("value should be present");
        let n_tokens = 6;
        let x = vec![0.2_f32; n_tokens * 16];
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        // With k=1 the single chosen weight renormalises to 1.
        for tok in 0..n_tokens {
            assert!((res.weights[tok] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn router_deterministic_without_noise() {
        let cfg = GumbelConfig {
            noisy: false,
            ..base_cfg(6, 2)
        };
        let mut rng_build = LcgRng::new(13);
        let router = GumbelRouter::new(cfg, &mut rng_build).expect("new should succeed");
        let n_tokens = 8;
        let x = vec![0.4_f32; n_tokens * 16];
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(2);
        let a = router
            .route(&x, n_tokens, &mut r1)
            .expect("route should succeed");
        let b = router
            .route(&x, n_tokens, &mut r2)
            .expect("route should succeed");
        // Noise disabled → identical routing regardless of RNG state.
        assert_eq!(a.indices, b.indices);
        for (wa, wb) in a.weights.iter().zip(b.weights.iter()) {
            assert!((wa - wb).abs() < 1e-6);
        }
    }

    #[test]
    fn router_noise_changes_routing() {
        // With noise on, two different RNG streams should (very likely) differ
        // in at least one token's selection for near-uniform logits.
        let mut rng_build = LcgRng::new(14);
        let router =
            GumbelRouter::new(base_cfg(8, 1), &mut rng_build).expect("value should be present");
        let n_tokens = 32;
        let x = vec![0.01_f32; n_tokens * 16]; // near-uniform logits
        let mut r1 = LcgRng::new(100);
        let mut r2 = LcgRng::new(200);
        let a = router
            .route(&x, n_tokens, &mut r1)
            .expect("route should succeed");
        let b = router
            .route(&x, n_tokens, &mut r2)
            .expect("route should succeed");
        assert_ne!(a.indices, b.indices, "Gumbel noise should change routing");
    }

    #[test]
    fn router_output_finite() {
        let mut rng = LcgRng::new(15);
        let router = GumbelRouter::new(base_cfg(8, 3), &mut rng).expect("value should be present");
        let n_tokens = 12;
        let mut x = vec![0.0_f32; n_tokens * 16];
        rng.fill_normal(&mut x);
        let res = router
            .route(&x, n_tokens, &mut rng)
            .expect("route should succeed");
        assert!(res.weights.iter().all(|v| v.is_finite()));
        assert!(res.soft.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn router_x_size_mismatch_errors() {
        let mut rng = LcgRng::new(16);
        let router = GumbelRouter::new(base_cfg(4, 1), &mut rng).expect("value should be present");
        let x = vec![0.0_f32; 10]; // not n_tokens*16
        assert!(matches!(
            router.route(&x, 4, &mut rng),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }
}
