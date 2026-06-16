//! ST-MoE: Stable and Transferable Mixture-of-Experts routing (Zoph et al. 2022).
//!
//! Implements the sparse-expert layer from:
//! Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer & Fedus,
//! "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
//! arXiv:2202.08906, 2022.
//!
//! # Routing
//!
//! Each token `x_t` is scored against `n_experts` experts by a linear gate
//! `g = softmax(W_g · x)`.  The **top-2** experts are selected and their
//! softmax masses are **renormalised** to sum to one — these renormalised
//! masses become the *combine weights*:
//!
//! ```text
//! p        = softmax(W_g · x)                       // over all experts
//! (w, e)   = top2(p);   w_j ← w_j / (w_0 + w_1)     // selected + renormalised
//! y_t      = Σ_{j∈{0,1}} w_j · Expert_{e_j}(x_t)    // gate-weighted combine
//! ```
//!
//! # Stability and load-balancing losses
//!
//! ST-MoE carries two auxiliary losses that are the heart of the paper:
//!
//! * **Router z-loss** — the key stability term that keeps the router logits
//!   small (and hence the softmax well-conditioned):
//!
//!   ```text
//!   L_z = (1/B) Σ_b (logsumexp_e router_logits_{b,e})²
//!   ```
//!
//!   Because it penalises `logsumexp²`, scaling *all* logits by a constant
//!   `c > 1` multiplies the loss by approximately `c²` (a quadratic penalty),
//!   which is exactly what discourages the router from inflating its logits.
//!
//! * **Load-balancing loss** — the Switch-Transformer differentiable proxy for
//!   uniform expert utilisation:
//!
//!   ```text
//!   L_aux = n_experts · Σ_i f_i · P_i
//!   ```
//!
//!   with `f_i` the fraction of routing slots dispatched to expert `i` and
//!   `P_i = (1/T) Σ_t softmax(logits_t)[i]` the mean router probability for
//!   expert `i`.  It equals `1` at perfectly uniform assignment and grows toward
//!   `n_experts` as routing collapses.
//!
//! # Expert capacity
//!
//! Each expert has a finite buffer of
//! `capacity = max(1, ceil(top_k · T / n_experts · capacity_factor))` slots.
//! Tokens are placed in priority order (highest combine weight first per slot);
//! a `(token, slot)` pair that arrives at a full expert *overflows* — its
//! combine weight is **zeroed**, so the token simply receives a smaller (or, if
//! both of its experts overflow, a zero) contribution and the dropped mass is
//! counted in [`StMoeOutput::n_dropped`].

use crate::error::{MoeError, MoeResult};
use crate::expert::ffn::{ExpertActivation, ExpertFfn};
use crate::handle::LcgRng;
use crate::routing::top_k::{stable_softmax, topk};

/// Numerically stable `logsumexp` over a slice of logits.
///
/// `logsumexp(x) = m + log(Σ_e exp(x_e − m))` with `m = max_e x_e`.
#[must_use]
pub fn logsumexp(logits: &[f32]) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return max_val;
    }
    let sum_exp: f32 = logits.iter().map(|&lg| (lg - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Router z-loss `L_z = (1/B) Σ_b (logsumexp_e logits_{b,e})²`.
///
/// This is the ST-MoE stability term: it is the mean over the batch of the
/// squared log-partition function of the router, so it is always non-negative
/// and grows quadratically with the magnitude of the logits.
///
/// # Arguments
/// * `router_logits` — raw gate logits, shape `[n_tokens · n_experts]`.
///
/// # Errors
/// Returns [`MoeError`] on empty input, zero experts, or a logits length that is
/// not `n_tokens · n_experts`.
pub fn st_router_z_loss(
    router_logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected = n_tokens * n_experts;
    if router_logits.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: router_logits.len(),
        });
    }

    let mut acc = 0.0_f32;
    for tok in 0..n_tokens {
        let row = &router_logits[tok * n_experts..(tok + 1) * n_experts];
        let lse = logsumexp(row);
        acc += lse * lse;
    }
    let loss = acc / n_tokens as f32;
    if !loss.is_finite() {
        return Err(MoeError::NanEncountered {
            context: "st_router_z_loss".to_string(),
        });
    }
    Ok(loss)
}

/// Switch-style load-balancing auxiliary loss generalised to top-k routing.
///
/// `L = n_experts · Σ_i f_i · P_i`, where `f_i` is the fraction of routing slots
/// assigned to expert `i` and `P_i` is its mean router softmax probability.  The
/// loss equals `1` when both `f` and `P` are uniform and grows toward
/// `n_experts` as routing collapses, so it is minimised by balanced routing.
///
/// # Arguments
/// * `router_logits` — raw gate logits, shape `[n_tokens · n_experts]`.
/// * `selected_indices` — selected expert indices, shape `[n_tokens · top_k]`.
///
/// # Errors
/// Returns [`MoeError`] on empty input, zero experts, a logits-length mismatch, a
/// `selected_indices` length not divisible by `n_tokens`, or an out-of-range
/// expert index.
pub fn st_load_balance_loss(
    router_logits: &[f32],
    selected_indices: &[usize],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected_logits = n_tokens * n_experts;
    if router_logits.len() != expected_logits {
        return Err(MoeError::DimensionMismatch {
            expected: expected_logits,
            got: router_logits.len(),
        });
    }
    if selected_indices.is_empty() || !selected_indices.len().is_multiple_of(n_tokens) {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: selected_indices.len(),
        });
    }

    // f_i: fraction of routing slots dispatched to each expert.
    let mut slot_counts = vec![0_usize; n_experts];
    for &idx in selected_indices {
        if idx >= n_experts {
            return Err(MoeError::ExpertIndexOutOfRange { idx, n_experts });
        }
        slot_counts[idx] += 1;
    }
    let total_slots = selected_indices.len() as f32;
    let fraction: Vec<f32> = slot_counts
        .iter()
        .map(|&c| c as f32 / total_slots)
        .collect();

    // P_i: mean router softmax probability for each expert.
    let mut prob_sum = vec![0.0_f32; n_experts];
    for tok in 0..n_tokens {
        let probs = stable_softmax(&router_logits[tok * n_experts..(tok + 1) * n_experts]);
        for (acc, &p) in prob_sum.iter_mut().zip(probs.iter()) {
            *acc += p;
        }
    }
    let token_count = n_tokens as f32;
    let mean_prob: Vec<f32> = prob_sum.iter().map(|&s| s / token_count).collect();

    let loss = n_experts as f32
        * fraction
            .iter()
            .zip(mean_prob.iter())
            .map(|(&f, &p)| f * p)
            .sum::<f32>();
    if !loss.is_finite() {
        return Err(MoeError::NanEncountered {
            context: "st_load_balance_loss".to_string(),
        });
    }
    Ok(loss)
}

/// Configuration for an [`StMoeLayer`].
#[derive(Debug, Clone)]
pub struct StMoeConfig {
    /// Model (input/output) dimension `d_model`.
    pub d_model: usize,
    /// Hidden dimension of every expert FFN.
    pub ffn_dim: usize,
    /// Number of experts in the pool.
    pub n_experts: usize,
    /// Experts activated per token; ST-MoE uses `2`. Must satisfy `1 ≤ top_k ≤ n_experts`.
    pub top_k: usize,
    /// Expert-buffer capacity factor (`> 0`); `1.25` in the paper.
    pub capacity_factor: f32,
    /// Coefficient applied to the load-balancing auxiliary loss (`0.01` in the paper).
    pub load_balance_coef: f32,
    /// Coefficient applied to the router z-loss (`0.001` in the paper).
    pub router_z_loss_coef: f32,
    /// Expert FFN activation.
    pub activation: ExpertActivation,
}

impl Default for StMoeConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            ffn_dim: 1024,
            n_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_coef: 0.01,
            router_z_loss_coef: 0.001,
            activation: ExpertActivation::Gelu,
        }
    }
}

/// Routing decisions produced by an ST-MoE forward pass.
#[derive(Debug, Clone)]
pub struct StMoeRouting {
    /// Selected expert indices per token, shape `[n_tokens · top_k]`.
    pub expert_indices: Vec<usize>,
    /// Renormalised top-k combine weights, shape `[n_tokens · top_k]`; a weight is
    /// **zeroed** when its `(token, slot)` pair overflowed its expert's capacity.
    pub combine_weights: Vec<f32>,
    /// Raw gate logits before softmax, shape `[n_tokens · n_experts]`.
    pub router_logits: Vec<f32>,
}

/// Output of an [`StMoeLayer`] forward pass.
#[derive(Debug, Clone)]
pub struct StMoeOutput {
    /// Output hidden states, shape `[n_tokens · d_model]`.
    pub hidden: Vec<f32>,
    /// Router z-loss (scalar, *unweighted* by `router_z_loss_coef`).
    pub router_z_loss: f32,
    /// Load-balancing auxiliary loss (scalar, *unweighted* by `load_balance_coef`).
    pub load_balance_loss: f32,
    /// Combined auxiliary loss
    /// `load_balance_coef · load_balance_loss + router_z_loss_coef · router_z_loss`.
    pub aux_loss: f32,
    /// Per-expert token capacity used for this batch.
    pub capacity: usize,
    /// Number of `(token, slot)` pairs dropped due to capacity overflow.
    pub n_dropped: usize,
    /// Routing decisions for inspection / downstream losses.
    pub routing: StMoeRouting,
}

/// ST-MoE sparse layer: top-2 routing, expert capacity, z-loss + load-balance loss.
pub struct StMoeLayer {
    experts: Vec<ExpertFfn>,
    /// Gate weight matrix, shape `[n_experts · d_model]` (row-major per expert).
    pub gate_weights: Vec<f32>,
    /// Number of experts in the pool.
    pub n_experts: usize,
    /// Experts activated per token.
    pub top_k: usize,
    /// Model (input/output) dimension.
    pub d_model: usize,
    /// Expert FFN hidden dimension.
    pub ffn_dim: usize,
    /// Expert-buffer capacity factor.
    pub capacity_factor: f32,
    /// Load-balancing loss coefficient.
    pub load_balance_coef: f32,
    /// Router z-loss coefficient.
    pub router_z_loss_coef: f32,
}

impl StMoeLayer {
    /// Build a new ST-MoE layer with a randomly initialised gate and experts.
    ///
    /// The gate is initialised `N(0, 0.01)` (small logits, as ST-MoE prefers) and
    /// every expert FFN with Xavier initialisation.
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero `d_model` / `ffn_dim` / `n_experts`, a
    /// `top_k` outside `1 ..= n_experts`, or a non-positive `capacity_factor`.
    pub fn new(cfg: StMoeConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.d_model == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.d_model });
        }
        if cfg.ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: cfg.ffn_dim });
        }
        if cfg.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_experts,
            });
        }
        if cfg.top_k == 0 || cfg.top_k > cfg.n_experts {
            return Err(MoeError::InvalidTopK {
                k: cfg.top_k,
                n_experts: cfg.n_experts,
            });
        }
        if !cfg.capacity_factor.is_finite() || cfg.capacity_factor <= 0.0 {
            return Err(MoeError::InvalidCapacityFactor {
                factor: cfg.capacity_factor,
            });
        }

        let mut gate_weights = vec![0.0_f32; cfg.n_experts * cfg.d_model];
        rng.fill_normal_scaled(&mut gate_weights, 0.01);
        let experts: Vec<ExpertFfn> = (0..cfg.n_experts)
            .map(|_| ExpertFfn::new(cfg.d_model, cfg.ffn_dim, cfg.activation, rng))
            .collect();

        Ok(Self {
            experts,
            gate_weights,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            d_model: cfg.d_model,
            ffn_dim: cfg.ffn_dim,
            capacity_factor: cfg.capacity_factor,
            load_balance_coef: cfg.load_balance_coef,
            router_z_loss_coef: cfg.router_z_loss_coef,
        })
    }

    /// Compute raw gate logits `W_g · x`, shape `[n_tokens · n_experts]`.
    fn gate_logits(&self, tokens: &[f32], n_tokens: usize) -> Vec<f32> {
        let d = self.d_model;
        let mut logits = vec![0.0_f32; n_tokens * self.n_experts];
        for tok in 0..n_tokens {
            let x_row = &tokens[tok * d..(tok + 1) * d];
            for exp_idx in 0..self.n_experts {
                let w_row = &self.gate_weights[exp_idx * d..(exp_idx + 1) * d];
                let dot: f32 = x_row
                    .iter()
                    .zip(w_row.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum();
                logits[tok * self.n_experts + exp_idx] = dot;
            }
        }
        logits
    }

    /// Per-expert capacity for a batch of `n_tokens` tokens.
    ///
    /// `capacity = max(1, ceil(top_k · T / n_experts · capacity_factor))`.
    #[must_use]
    pub fn capacity(&self, n_tokens: usize) -> usize {
        let slots = (self.top_k * n_tokens) as f32;
        let raw = (slots / self.n_experts as f32 * self.capacity_factor).ceil() as usize;
        raw.max(1)
    }

    /// Run the full ST-MoE forward pass.
    ///
    /// # Arguments
    /// * `tokens` — input activations, row-major `[n_tokens · d_model]`.
    /// * `n_tokens` — number of tokens `T`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input or a `tokens` length not equal to
    /// `n_tokens · d_model`, and propagates expert-FFN errors.
    pub fn forward(&self, tokens: &[f32], n_tokens: usize) -> MoeResult<StMoeOutput> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = n_tokens * self.d_model;
        if tokens.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: tokens.len(),
            });
        }

        let k = self.top_k;
        let n_e = self.n_experts;
        let router_logits = self.gate_logits(tokens, n_tokens);

        // Top-k selection + renormalised combine weights per token.
        let mut expert_indices = vec![0_usize; n_tokens * k];
        let mut combine_weights = vec![0.0_f32; n_tokens * k];
        for tok in 0..n_tokens {
            let probs = stable_softmax(&router_logits[tok * n_e..(tok + 1) * n_e]);
            let (top_vals, top_idx) = topk(&probs, k)?;
            let mass: f32 = top_vals.iter().sum();
            let denom = if mass > 1e-12 { mass } else { 1.0 };
            for slot in 0..k {
                expert_indices[tok * k + slot] = top_idx[slot];
                combine_weights[tok * k + slot] = top_vals[slot] / denom;
            }
        }

        // Auxiliary losses are computed from the *pre-capacity* routing, matching
        // the ST-MoE training objective (the losses shape the router, not the
        // dropped tokens).
        let z = st_router_z_loss(&router_logits, n_tokens, n_e)?;
        let lb = st_load_balance_loss(&router_logits, &expert_indices, n_tokens, n_e)?;
        let aux = self.load_balance_coef * lb + self.router_z_loss_coef * z;

        // Expert capacity: place routing slots in descending combine-weight order
        // so the highest-priority tokens win the buffer; overflowed slots have
        // their combine weight zeroed (token dropping).
        let capacity = self.capacity(n_tokens);
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(n_tokens * k);
        for tok in 0..n_tokens {
            for slot in 0..k {
                order.push((tok, slot));
            }
        }
        order.sort_unstable_by(|&(ta, sa), &(tb, sb)| {
            let wa = combine_weights[ta * k + sa];
            let wb = combine_weights[tb * k + sb];
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut expert_load = vec![0_usize; n_e];
        let mut n_dropped = 0_usize;
        for &(tok, slot) in &order {
            let exp_idx = expert_indices[tok * k + slot];
            if expert_load[exp_idx] < capacity {
                expert_load[exp_idx] += 1;
            } else {
                combine_weights[tok * k + slot] = 0.0;
                n_dropped += 1;
            }
        }

        // Combine: y_t = Σ_slot w · Expert_e(x_t); experts are evaluated only for
        // surviving (non-zero-weight) slots, with results cached per token to avoid
        // recomputation when both slots hit the same expert (cannot happen here as
        // top-k indices are distinct, but the guard keeps it correct in general).
        let d = self.d_model;
        let mut hidden = vec![0.0_f32; n_tokens * d];
        for tok in 0..n_tokens {
            let x_tok = &tokens[tok * d..(tok + 1) * d];
            let out_slice = &mut hidden[tok * d..(tok + 1) * d];
            for slot in 0..k {
                let weight = combine_weights[tok * k + slot];
                if weight == 0.0 {
                    continue;
                }
                let exp_idx = expert_indices[tok * k + slot];
                if exp_idx >= n_e {
                    return Err(MoeError::ExpertIndexOutOfRange {
                        idx: exp_idx,
                        n_experts: n_e,
                    });
                }
                let expert_out = self.experts[exp_idx].forward(x_tok)?;
                for (acc, &val) in out_slice.iter_mut().zip(expert_out.iter()) {
                    *acc += weight * val;
                }
            }
        }

        Ok(StMoeOutput {
            hidden,
            router_z_loss: z,
            load_balance_loss: lb,
            aux_loss: aux,
            capacity,
            n_dropped,
            routing: StMoeRouting {
                expert_indices,
                combine_weights,
                router_logits,
            },
        })
    }

    /// Total trainable parameter count (gate + all expert FFNs).
    #[must_use]
    pub fn param_count(&self) -> usize {
        let gate = self.gate_weights.len();
        let experts: usize = self.experts.iter().map(ExpertFfn::param_count).sum();
        gate + experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ----------------------------------------------------------------------
    // logsumexp
    // ----------------------------------------------------------------------

    #[test]
    fn logsumexp_matches_naive() {
        let logits = [0.5_f32, -1.0, 2.0, 0.3];
        let naive = logits.iter().map(|&x| x.exp()).sum::<f32>().ln();
        assert!((logsumexp(&logits) - naive).abs() < 1e-5);
    }

    #[test]
    fn logsumexp_uniform_zero_is_log_n() {
        let logits = [0.0_f32; 4];
        assert!((logsumexp(&logits) - 4.0_f32.ln()).abs() < 1e-6);
    }

    // ----------------------------------------------------------------------
    // (a) z-loss == mean of (logsumexp(logits))² over the batch
    // ----------------------------------------------------------------------

    #[test]
    fn z_loss_equals_mean_squared_logsumexp() {
        let n_tokens = 5_usize;
        let n_experts = 4_usize;
        let mut rng = LcgRng::new(123);
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        rng.fill_normal_scaled(&mut logits, 1.3);

        let got = st_router_z_loss(&logits, n_tokens, n_experts)
            .expect("st_router_z_loss should succeed");

        let mut expected = 0.0_f32;
        for tok in 0..n_tokens {
            let lse = logsumexp(&logits[tok * n_experts..(tok + 1) * n_experts]);
            expected += lse * lse;
        }
        expected /= n_tokens as f32;

        assert!(
            (got - expected).abs() < 1e-4,
            "z_loss={got}, expected mean(lse²)={expected}"
        );
    }

    // ----------------------------------------------------------------------
    // (b) scaling all logits by c>1 increases z-loss by ≈ c² (quadratic penalty)
    // ----------------------------------------------------------------------

    #[test]
    fn z_loss_is_quadratic_in_logit_scale() {
        // Use a row whose logsumexp is dominated by a single large entry so that
        // logsumexp(c·logits) ≈ c·logsumexp(logits) and the squared penalty
        // scales as c². A spiked logit vector makes this near-exact.
        let n_tokens = 3_usize;
        let n_experts = 6_usize;
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for tok in 0..n_tokens {
            logits[tok * n_experts] = 8.0; // one dominant logit per token
        }
        let base = st_router_z_loss(&logits, n_tokens, n_experts)
            .expect("st_router_z_loss should succeed");

        let c = 2.0_f32;
        let scaled: Vec<f32> = logits.iter().map(|&x| c * x).collect();
        let scaled_loss = st_router_z_loss(&scaled, n_tokens, n_experts)
            .expect("st_router_z_loss should succeed");

        let ratio = scaled_loss / base;
        assert!(
            (ratio - c * c).abs() < 0.05,
            "scaling by c={c} gave ratio={ratio}, expected ≈{}",
            c * c
        );
    }

    #[test]
    fn z_loss_strictly_increases_with_scale() {
        let n_tokens = 4_usize;
        let n_experts = 5_usize;
        let mut rng = LcgRng::new(9);
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        rng.fill_normal_scaled(&mut logits, 1.0);
        // Shift so logsumexp is positive and scaling is monotone in the penalty.
        for v in logits.iter_mut() {
            *v += 3.0;
        }
        let base = st_router_z_loss(&logits, n_tokens, n_experts)
            .expect("st_router_z_loss should succeed");
        let scaled: Vec<f32> = logits.iter().map(|&x| 1.5 * x).collect();
        let scaled_loss = st_router_z_loss(&scaled, n_tokens, n_experts)
            .expect("st_router_z_loss should succeed");
        assert!(scaled_loss > base, "scaled={scaled_loss} !> base={base}");
    }

    // ----------------------------------------------------------------------
    // (c) top-2 routing selects the two highest-affinity experts and the combine
    //     weights are the renormalised top-2 softmax.
    // ----------------------------------------------------------------------

    #[test]
    fn top2_selects_two_highest_and_renormalises() {
        let mut rng = LcgRng::new(0);
        let cfg = StMoeConfig {
            d_model: 4,
            ffn_dim: 8,
            n_experts: 5,
            top_k: 2,
            capacity_factor: 100.0, // no dropping in this test
            ..StMoeConfig::default()
        };
        let layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 6;
        let mut x = vec![0.0_f32; n_tokens * 4];
        let mut xr = LcgRng::new(77);
        xr.fill_normal_scaled(&mut x, 1.0);

        let out = layer.forward(&x, n_tokens).expect("forward should succeed");
        let n_e = 5;
        let k = 2;
        for tok in 0..n_tokens {
            let probs = stable_softmax(&out.routing.router_logits[tok * n_e..(tok + 1) * n_e]);
            // Ground-truth top-2 by affinity.
            let mut idx: Vec<usize> = (0..n_e).collect();
            idx.sort_unstable_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .expect("partial_cmp should succeed")
            });
            let want0 = idx[0];
            let want1 = idx[1];
            let got0 = out.routing.expert_indices[tok * k];
            let got1 = out.routing.expert_indices[tok * k + 1];
            assert_eq!(got0, want0, "token {tok} top-1 expert");
            assert_eq!(got1, want1, "token {tok} top-2 expert");

            // Renormalised combine weights.
            let mass = probs[want0] + probs[want1];
            let w0 = probs[want0] / mass;
            let w1 = probs[want1] / mass;
            assert!((out.routing.combine_weights[tok * k] - w0).abs() < 1e-5);
            assert!((out.routing.combine_weights[tok * k + 1] - w1).abs() < 1e-5);
            // Combine weights sum to one.
            let wsum =
                out.routing.combine_weights[tok * k] + out.routing.combine_weights[tok * k + 1];
            assert!((wsum - 1.0).abs() < 1e-5, "token {tok} weights sum={wsum}");
        }
    }

    // ----------------------------------------------------------------------
    // (d) load-balancing aux loss is minimised (→ 1.0) at uniform assignment and
    //     larger under imbalance.
    // ----------------------------------------------------------------------

    #[test]
    fn load_balance_uniform_is_one() {
        let n_tokens = 8;
        let n_experts = 4;
        let logits = vec![0.0_f32; n_tokens * n_experts]; // uniform softmax → P_i = 1/4
        // Round-robin top-1 slots → f_i = 1/4 each.
        let selected: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let loss = st_load_balance_loss(&logits, &selected, n_tokens, n_experts)
            .expect("st_load_balance_loss should succeed");
        assert!((loss - 1.0).abs() < 1e-4, "uniform loss {loss} != 1");
    }

    #[test]
    fn load_balance_imbalanced_exceeds_uniform() {
        let n_tokens = 8;
        let n_experts = 4;

        let uniform_logits = vec![0.0_f32; n_tokens * n_experts];
        let uniform_sel: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let uniform = st_load_balance_loss(&uniform_logits, &uniform_sel, n_tokens, n_experts)
            .expect("st_load_balance_loss should succeed");

        // Collapse: strong bias + every slot on expert 0.
        let mut collapsed_logits = vec![0.0_f32; n_tokens * n_experts];
        for tok in 0..n_tokens {
            collapsed_logits[tok * n_experts] = 20.0;
        }
        let collapsed_sel = vec![0_usize; n_tokens];
        let collapsed =
            st_load_balance_loss(&collapsed_logits, &collapsed_sel, n_tokens, n_experts)
                .expect("st_load_balance_loss should succeed");

        assert!((uniform - 1.0).abs() < 1e-4, "uniform={uniform}");
        assert!(
            collapsed > uniform,
            "collapsed={collapsed} !> uniform={uniform}"
        );
        assert!(
            collapsed > 3.5,
            "collapsed {collapsed} should approach n_experts"
        );
    }

    // ----------------------------------------------------------------------
    // (e) expert capacity / token dropping handled.
    // ----------------------------------------------------------------------

    #[test]
    fn capacity_formula_matches() {
        let mut rng = LcgRng::new(1);
        let cfg = StMoeConfig {
            d_model: 4,
            ffn_dim: 8,
            n_experts: 4,
            top_k: 2,
            capacity_factor: 1.0,
            ..StMoeConfig::default()
        };
        let layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        // slots = top_k·T = 2·8 = 16; per expert = 16/4 = 4; cap_factor 1.0 → 4.
        assert_eq!(layer.capacity(8), 4);
    }

    #[test]
    fn overflow_tokens_are_dropped_and_zero_weighted() {
        // Force all tokens' top-1 onto expert 0 via a dominant gate column, with a
        // tight capacity so the buffer overflows.
        let mut rng = LcgRng::new(2);
        let cfg = StMoeConfig {
            d_model: 3,
            ffn_dim: 6,
            n_experts: 4,
            top_k: 2,
            capacity_factor: 0.25, // tight → guaranteed overflow
            ..StMoeConfig::default()
        };
        let mut layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        // Hand-craft the gate so expert 0 dominates for every token: row 0 large +.
        let d = 3;
        for w in layer.gate_weights[0..d].iter_mut() {
            *w = 5.0;
        }
        let n_tokens = 16;
        let x = vec![1.0_f32; n_tokens * d];
        let out = layer.forward(&x, n_tokens).expect("forward should succeed");

        assert!(out.n_dropped > 0, "expected dropped slots, got 0");

        // No expert exceeds capacity among surviving (non-zero-weight) slots.
        let k = 2;
        let mut load = [0_usize; 4];
        for tok in 0..n_tokens {
            for slot in 0..k {
                if out.routing.combine_weights[tok * k + slot] != 0.0 {
                    load[out.routing.expert_indices[tok * k + slot]] += 1;
                }
            }
        }
        for (e, &l) in load.iter().enumerate() {
            assert!(
                l <= out.capacity,
                "expert {e} load {l} > capacity {}",
                out.capacity
            );
        }
        // Every output value remains finite even with dropping.
        assert!(out.hidden.iter().all(|v| v.is_finite()));
    }

    // ----------------------------------------------------------------------
    // (f) output shape == input shape; deterministic given fixed weights.
    // ----------------------------------------------------------------------

    #[test]
    fn output_shape_equals_input_shape() {
        let mut rng = LcgRng::new(3);
        let cfg = StMoeConfig {
            d_model: 16,
            ffn_dim: 32,
            n_experts: 6,
            top_k: 2,
            ..StMoeConfig::default()
        };
        let layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 10;
        let x = vec![0.25_f32; n_tokens * 16];
        let out = layer.forward(&x, n_tokens).expect("forward should succeed");
        assert_eq!(out.hidden.len(), n_tokens * 16);
        assert!(out.aux_loss.is_finite());
    }

    #[test]
    fn forward_is_deterministic() {
        let cfg = StMoeConfig {
            d_model: 8,
            ffn_dim: 16,
            n_experts: 5,
            top_k: 2,
            ..StMoeConfig::default()
        };
        let mut rng_a = LcgRng::new(2024);
        let layer_a = StMoeLayer::new(cfg.clone(), &mut rng_a).expect("value should be present");
        let mut rng_b = LcgRng::new(2024);
        let layer_b = StMoeLayer::new(cfg, &mut rng_b).expect("new should succeed");

        let n_tokens = 7;
        let mut x = vec![0.0_f32; n_tokens * 8];
        let mut xr = LcgRng::new(555);
        xr.fill_normal_scaled(&mut x, 1.0);

        let out_a = layer_a
            .forward(&x, n_tokens)
            .expect("forward should succeed");
        let out_b = layer_b
            .forward(&x, n_tokens)
            .expect("forward should succeed");
        assert_eq!(out_a.hidden, out_b.hidden);
        assert_eq!(out_a.routing.expert_indices, out_b.routing.expert_indices);
        assert!((out_a.aux_loss - out_b.aux_loss).abs() < 1e-9);
    }

    #[test]
    fn aux_loss_combines_both_terms() {
        let mut rng = LcgRng::new(4);
        let cfg = StMoeConfig {
            d_model: 8,
            ffn_dim: 16,
            n_experts: 4,
            top_k: 2,
            load_balance_coef: 0.01,
            router_z_loss_coef: 0.001,
            ..StMoeConfig::default()
        };
        let layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 6;
        let mut x = vec![0.0_f32; n_tokens * 8];
        let mut xr = LcgRng::new(8);
        xr.fill_normal_scaled(&mut x, 1.0);
        let out = layer.forward(&x, n_tokens).expect("forward should succeed");
        let expected = 0.01 * out.load_balance_loss + 0.001 * out.router_z_loss;
        assert!((out.aux_loss - expected).abs() < 1e-6);
    }

    // ----------------------------------------------------------------------
    // Error paths.
    // ----------------------------------------------------------------------

    #[test]
    fn new_rejects_bad_top_k() {
        let mut rng = LcgRng::new(0);
        let cfg = StMoeConfig {
            n_experts: 4,
            top_k: 8,
            ..StMoeConfig::default()
        };
        assert!(StMoeLayer::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn forward_rejects_wrong_length() {
        let mut rng = LcgRng::new(0);
        let cfg = StMoeConfig {
            d_model: 4,
            ffn_dim: 8,
            n_experts: 4,
            top_k: 2,
            ..StMoeConfig::default()
        };
        let layer = StMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        assert!(layer.forward(&[0.0_f32; 5], 3).is_err());
    }

    #[test]
    fn z_loss_rejects_empty() {
        assert!(st_router_z_loss(&[], 0, 4).is_err());
    }
}
