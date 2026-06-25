//! Layer-conditional routing: share (or not) the router across a stack of MoE
//! layers.
//!
//! In a deep MoE transformer every block contains its own routing decision. Two
//! design choices govern how those decisions relate across depth:
//!
//! * **Per-layer routers** (`RouterSharing::PerLayer`) — each of the `n_layers`
//!   blocks owns an independent gate `W_g^{(ℓ)}`. Maximum routing flexibility;
//!   the default for Switch / GShard.
//! * **Shared router** (`RouterSharing::Shared`) — a *single* gate `W_g` is
//!   reused at every layer, so a token tends to be routed to the *same* expert
//!   index throughout the stack. This "sticky" routing (used by recurrent /
//!   weight-tied MoE and several parameter-efficient designs) cuts router
//!   parameters by `n_layers×` and improves expert-cache locality during
//!   serving because a token's expert is predictable across depth.
//!
//! This module owns the per-layer gate matrices, performs the per-layer top-k
//! routing, and exposes a **cross-layer agreement** metric — the fraction of
//! tokens whose top-1 expert is identical between two layers — which is `1.0` by
//! construction for a shared router given the same input, and typically lower
//! for per-layer routers.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::routing::top_k::{stable_softmax, topk};

/// Whether the router is shared across layers or independent per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSharing {
    /// One gate matrix reused at every layer.
    Shared,
    /// An independent gate matrix per layer.
    PerLayer,
}

/// Configuration for a [`LayerConditionalRouter`].
#[derive(Debug, Clone)]
pub struct LayerConditionalConfig {
    /// Number of stacked MoE layers (`> 0`).
    pub n_layers: usize,
    /// Number of experts per layer (`> 0`).
    pub n_experts: usize,
    /// Token feature dimension (`> 0`).
    pub input_dim: usize,
    /// Experts selected per token per layer (`1 ≤ top_k ≤ n_experts`).
    pub top_k: usize,
    /// Router sharing mode.
    pub sharing: RouterSharing,
}

impl LayerConditionalConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`MoeError`] for any zero dimension or an invalid `top_k`.
    pub fn validate(&self) -> MoeResult<()> {
        if self.n_layers == 0 {
            return Err(MoeError::Internal {
                msg: "n_layers must be > 0".to_string(),
            });
        }
        if self.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: self.n_experts,
            });
        }
        if self.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: self.input_dim,
            });
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(MoeError::InvalidTopK {
                k: self.top_k,
                n_experts: self.n_experts,
            });
        }
        Ok(())
    }
}

/// Per-layer routing decision.
#[derive(Debug, Clone)]
pub struct LayerRouteResult {
    /// Selected expert indices, shape `[n_tokens · top_k]`.
    pub indices: Vec<usize>,
    /// Renormalised top-k gate scores (each token sums to `1`),
    /// shape `[n_tokens · top_k]`.
    pub scores: Vec<f32>,
    /// Raw gate logits, shape `[n_tokens · n_experts]`.
    pub logits: Vec<f32>,
}

/// A router for a stack of MoE layers with configurable cross-layer sharing.
#[derive(Debug, Clone)]
pub struct LayerConditionalRouter {
    /// Gate matrices. With `Shared` there is exactly one entry, reused for every
    /// layer; with `PerLayer` there are `n_layers` entries. Each is row-major
    /// `[n_experts · input_dim]`.
    gates: Vec<Vec<f32>>,
    /// Configuration.
    pub config: LayerConditionalConfig,
}

impl LayerConditionalRouter {
    /// Build a router with randomly initialised gate(s) (`N(0, 0.01²)`).
    ///
    /// # Errors
    /// Propagates [`LayerConditionalConfig::validate`].
    pub fn new(cfg: LayerConditionalConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        cfg.validate()?;
        let n_gates = match cfg.sharing {
            RouterSharing::Shared => 1,
            RouterSharing::PerLayer => cfg.n_layers,
        };
        let weight_count = cfg.n_experts * cfg.input_dim;
        let gates: Vec<Vec<f32>> = (0..n_gates)
            .map(|_| {
                let mut g = vec![0.0_f32; weight_count];
                rng.fill_normal_scaled(&mut g, 0.01);
                g
            })
            .collect();
        Ok(Self { gates, config: cfg })
    }

    /// The gate matrix used for `layer_idx`.
    fn gate_for(&self, layer_idx: usize) -> &[f32] {
        match self.config.sharing {
            RouterSharing::Shared => &self.gates[0],
            RouterSharing::PerLayer => &self.gates[layer_idx],
        }
    }

    /// Route `n_tokens` tokens through layer `layer_idx`.
    ///
    /// # Errors
    /// Returns [`MoeError`] for an out-of-range `layer_idx`, empty input, or a
    /// shape mismatch.
    pub fn route_layer(
        &self,
        layer_idx: usize,
        x: &[f32],
        n_tokens: usize,
    ) -> MoeResult<LayerRouteResult> {
        let cfg = &self.config;
        if layer_idx >= cfg.n_layers {
            return Err(MoeError::Internal {
                msg: format!("layer index {layer_idx} >= n_layers {}", cfg.n_layers),
            });
        }
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let d = cfg.input_dim;
        if x.len() != n_tokens * d {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens * d,
                got: x.len(),
            });
        }

        let gate = self.gate_for(layer_idx);
        let mut logits = vec![0.0_f32; n_tokens * cfg.n_experts];
        for tok in 0..n_tokens {
            let row = &x[tok * d..(tok + 1) * d];
            for e in 0..cfg.n_experts {
                let w = &gate[e * d..(e + 1) * d];
                logits[tok * cfg.n_experts + e] =
                    w.iter().zip(row.iter()).map(|(&wi, &xi)| wi * xi).sum();
            }
        }

        let mut indices = vec![0_usize; n_tokens * cfg.top_k];
        let mut scores = vec![0.0_f32; n_tokens * cfg.top_k];
        for tok in 0..n_tokens {
            let probs = stable_softmax(&logits[tok * cfg.n_experts..(tok + 1) * cfg.n_experts]);
            let (top_vals, top_idx) = topk(&probs, cfg.top_k)?;
            let denom: f32 = top_vals.iter().sum::<f32>().max(1e-12);
            for slot in 0..cfg.top_k {
                scores[tok * cfg.top_k + slot] = top_vals[slot] / denom;
                indices[tok * cfg.top_k + slot] = top_idx[slot];
            }
        }

        if scores.iter().any(|v| !v.is_finite()) {
            return Err(MoeError::NanEncountered {
                context: "layer-conditional router scores".to_string(),
            });
        }

        Ok(LayerRouteResult {
            indices,
            scores,
            logits,
        })
    }

    /// Route the same `n_tokens` tokens through every layer, returning one
    /// [`LayerRouteResult`] per layer.
    ///
    /// # Errors
    /// Propagates [`Self::route_layer`].
    pub fn route_all(&self, x: &[f32], n_tokens: usize) -> MoeResult<Vec<LayerRouteResult>> {
        (0..self.config.n_layers)
            .map(|l| self.route_layer(l, x, n_tokens))
            .collect()
    }

    /// Fraction of tokens whose **top-1** expert agrees between two layers for
    /// the given input.
    ///
    /// For a [`RouterSharing::Shared`] router this is exactly `1.0`; for a
    /// per-layer router it reflects how correlated the layers' routing is.
    ///
    /// # Errors
    /// Propagates [`Self::route_layer`].
    pub fn cross_layer_agreement(
        &self,
        layer_a: usize,
        layer_b: usize,
        x: &[f32],
        n_tokens: usize,
    ) -> MoeResult<f32> {
        let ra = self.route_layer(layer_a, x, n_tokens)?;
        let rb = self.route_layer(layer_b, x, n_tokens)?;
        let k = self.config.top_k;
        let mut matches = 0_usize;
        for tok in 0..n_tokens {
            if ra.indices[tok * k] == rb.indices[tok * k] {
                matches += 1;
            }
        }
        Ok(matches as f32 / n_tokens as f32)
    }

    /// Total number of distinct gate matrices stored (`1` shared, else
    /// `n_layers`).
    #[must_use]
    pub fn n_gate_matrices(&self) -> usize {
        self.gates.len()
    }

    /// Total router parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.gates.iter().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(sharing: RouterSharing) -> LayerConditionalConfig {
        LayerConditionalConfig {
            n_layers: 4,
            n_experts: 6,
            input_dim: 8,
            top_k: 2,
            sharing,
        }
    }

    #[test]
    fn shared_router_stores_one_gate() {
        let mut rng = LcgRng::new(1);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::Shared), &mut rng)
            .expect("new should succeed");
        assert_eq!(r.n_gate_matrices(), 1);
        assert_eq!(r.param_count(), 6 * 8);
    }

    #[test]
    fn per_layer_router_stores_n_layers_gates() {
        let mut rng = LcgRng::new(2);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::PerLayer), &mut rng)
            .expect("new should succeed");
        assert_eq!(r.n_gate_matrices(), 4);
        assert_eq!(r.param_count(), 4 * 6 * 8);
    }

    #[test]
    fn shared_router_has_perfect_agreement() {
        let mut rng = LcgRng::new(3);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::Shared), &mut rng)
            .expect("new should succeed");
        let n_tokens = 12;
        let mut x = vec![0.0_f32; n_tokens * 8];
        rng.fill_normal_scaled(&mut x, 1.0);
        // Every pair of layers must route identically.
        for a in 0..4 {
            for b in 0..4 {
                let agree = r
                    .cross_layer_agreement(a, b, &x, n_tokens)
                    .expect("agreement should succeed");
                assert!((agree - 1.0).abs() < 1e-6, "shared layers {a},{b} disagree");
            }
        }
    }

    #[test]
    fn shared_router_routes_identically_across_layers() {
        let mut rng = LcgRng::new(4);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::Shared), &mut rng)
            .expect("new should succeed");
        let n_tokens = 5;
        let mut x = vec![0.0_f32; n_tokens * 8];
        rng.fill_normal_scaled(&mut x, 0.6);
        let all = r.route_all(&x, n_tokens).expect("route_all should succeed");
        // Indices and scores must be bit-identical across layers.
        for l in 1..4 {
            assert_eq!(all[l].indices, all[0].indices);
            for (a, b) in all[l].scores.iter().zip(all[0].scores.iter()) {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn per_layer_router_can_differ_across_layers() {
        let mut rng = LcgRng::new(5);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::PerLayer), &mut rng)
            .expect("new should succeed");
        let n_tokens = 64;
        let mut x = vec![0.0_f32; n_tokens * 8];
        rng.fill_normal_scaled(&mut x, 1.0);
        // Independent gates over many tokens: agreement should be < 1 (the
        // routers are not tied), confirming per-layer independence.
        let agree = r
            .cross_layer_agreement(0, 1, &x, n_tokens)
            .expect("agreement should succeed");
        assert!(
            agree < 1.0,
            "independent per-layer routers unexpectedly agreed on every token"
        );
    }

    #[test]
    fn scores_renormalise_to_one() {
        let mut rng = LcgRng::new(6);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::PerLayer), &mut rng)
            .expect("new should succeed");
        let n_tokens = 7;
        let mut x = vec![0.0_f32; n_tokens * 8];
        rng.fill_normal_scaled(&mut x, 0.9);
        let res = r
            .route_layer(2, &x, n_tokens)
            .expect("route should succeed");
        for tok in 0..n_tokens {
            let s: f32 = res.scores[tok * 2..tok * 2 + 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "token {tok} scores sum to {s}");
            for slot in 0..2 {
                assert!(res.indices[tok * 2 + slot] < 6);
            }
        }
    }

    #[test]
    fn out_of_range_layer_errors() {
        let mut rng = LcgRng::new(7);
        let r = LayerConditionalRouter::new(cfg(RouterSharing::PerLayer), &mut rng)
            .expect("new should succeed");
        let x = vec![0.1_f32; 8];
        assert!(r.route_layer(9, &x, 1).is_err());
    }
}
