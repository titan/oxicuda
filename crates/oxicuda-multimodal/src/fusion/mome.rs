//! Mixture-of-Modality-Experts (MoME) feed-forward router.
//!
//! VLMo (Bao et al. 2022, "VLMo: Unified Vision-Language Pre-Training with
//! Mixture-of-Modality-Experts") replaces the single shared FFN of a transformer
//! block with a small pool of modality-specialised FFN experts — a *vision*
//! expert (V-FFN), a *language* expert (L-FFN), and a *fusion* expert (VL-FFN) —
//! while sharing the self-attention across modalities. Each token is routed to
//! exactly one expert according to its modality tag (vision / text / fused).
//!
//! This module implements that hard, modality-conditioned routing for CPU
//! inference. Unlike a learned soft-gating MoE, the routing decision in VLMo is
//! deterministic and given by the token's modality, so the router here takes an
//! explicit per-token [`Modality`] tag rather than computing gate logits. The
//! optional [`MoMeRouter::forward_soft`] path additionally exposes a learned
//! per-token soft gate over the experts (Top-1 by gate logit) for the
//! Mixture-of-Modality-Experts ablation that *does* learn the routing.

use crate::error::{MmResult, MultiModalError};

/// Modality tag selecting which expert a token is routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Routed to the vision expert (index 0).
    Vision,
    /// Routed to the language expert (index 1).
    Text,
    /// Routed to the fusion (vision-language) expert (index 2).
    Fusion,
}

impl Modality {
    /// Expert index this modality maps to.
    #[must_use]
    pub fn expert_index(self) -> usize {
        match self {
            Modality::Vision => 0,
            Modality::Text => 1,
            Modality::Fusion => 2,
        }
    }
}

/// A single two-layer position-wise FFN expert: `W2·GELU(W1·x + b1) + b2`.
#[derive(Debug, Clone)]
pub struct FfnExpert {
    /// `W1`: `[d_model × d_hidden]` row-major.
    pub w1: Vec<f32>,
    /// `b1`: `[d_hidden]`.
    pub b1: Vec<f32>,
    /// `W2`: `[d_hidden × d_model]` row-major.
    pub w2: Vec<f32>,
    /// `b2`: `[d_model]`.
    pub b2: Vec<f32>,
}

impl FfnExpert {
    /// All-zero expert of the given shape.
    #[must_use]
    pub fn zeros(d_model: usize, d_hidden: usize) -> Self {
        Self {
            w1: vec![0.0_f32; d_model * d_hidden],
            b1: vec![0.0_f32; d_hidden],
            w2: vec![0.0_f32; d_hidden * d_model],
            b2: vec![0.0_f32; d_model],
        }
    }

    /// Forward a single `[d_model]` token through the expert.
    fn forward_token(&self, x: &[f32], d_model: usize, d_hidden: usize) -> Vec<f32> {
        let mut hidden = vec![0.0_f32; d_hidden];
        for (h, slot) in hidden.iter_mut().enumerate() {
            let mut acc = self.b1[h];
            for i in 0..d_model {
                acc += x[i] * self.w1[i * d_hidden + h];
            }
            *slot = gelu(acc);
        }
        let mut out = vec![0.0_f32; d_model];
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc = self.b2[o];
            for h in 0..d_hidden {
                acc += hidden[h] * self.w2[h * d_model + o];
            }
            *slot = acc;
        }
        out
    }
}

/// Tanh-approximation GELU (matches the rest of the crate's FFNs).
#[inline]
fn gelu(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

/// Configuration for the MoME router.
#[derive(Debug, Clone)]
pub struct MoMeConfig {
    /// Token / model dimension.
    pub d_model: usize,
    /// Per-expert hidden dimension.
    pub d_hidden: usize,
    /// Number of experts (VLMo uses 3: vision, language, fusion).
    pub n_experts: usize,
}

impl MoMeConfig {
    /// VLMo-style 3-expert preset.
    #[must_use]
    pub fn vlmo(d_model: usize, d_hidden: usize) -> Self {
        Self {
            d_model,
            d_hidden,
            n_experts: 3,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`MultiModalError::InvalidFeatureDim`] when `d_model == 0` or
    ///   `d_hidden == 0`.
    /// - [`MultiModalError::InvalidModalityCount`] when fewer than 1 expert.
    pub fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 || self.d_hidden == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.n_experts == 0 {
            return Err(MultiModalError::InvalidModalityCount { n: self.n_experts });
        }
        Ok(())
    }
}

/// Mixture-of-Modality-Experts router over a pool of [`FfnExpert`]s.
#[derive(Debug, Clone)]
pub struct MoMeRouter {
    /// The expert pool (length `cfg.n_experts`).
    pub experts: Vec<FfnExpert>,
    /// Per-expert gate vector `[n_experts × d_model]` used by
    /// [`MoMeRouter::forward_soft`]. Unused by the hard-routing path.
    pub gate: Vec<f32>,
    cfg: MoMeConfig,
}

impl MoMeRouter {
    /// Build a router with all-zero experts and gate.
    ///
    /// # Errors
    /// Propagates [`MoMeConfig::validate`].
    pub fn zeros(cfg: MoMeConfig) -> MmResult<Self> {
        cfg.validate()?;
        let experts = (0..cfg.n_experts)
            .map(|_| FfnExpert::zeros(cfg.d_model, cfg.d_hidden))
            .collect();
        let gate = vec![0.0_f32; cfg.n_experts * cfg.d_model];
        Ok(Self { experts, gate, cfg })
    }

    /// Hard modality routing: token `t` is processed solely by
    /// `experts[modalities[t].expert_index()]`.
    ///
    /// `tokens`: `[seq × d_model]` row-major. `modalities`: length `seq`.
    /// Returns the routed output `[seq × d_model]`.
    ///
    /// # Errors
    /// - [`MultiModalError::DimensionMismatch`] when `tokens.len() != seq*d_model`.
    /// - [`MultiModalError::MismatchedSeqLens`] when `modalities.len() != seq`.
    /// - [`MultiModalError::InvalidModalityCount`] when a modality maps to an
    ///   expert index outside the pool.
    pub fn forward_hard(
        &self,
        tokens: &[f32],
        modalities: &[Modality],
        seq: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        if tokens.len() != seq * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq * d,
                got: tokens.len(),
            });
        }
        if modalities.len() != seq {
            return Err(MultiModalError::MismatchedSeqLens {
                q_len: seq,
                kv_len: modalities.len(),
            });
        }
        let mut out = vec![0.0_f32; seq * d];
        for s in 0..seq {
            let e = modalities[s].expert_index();
            if e >= self.experts.len() {
                return Err(MultiModalError::InvalidModalityCount {
                    n: self.experts.len(),
                });
            }
            let x = &tokens[s * d..(s + 1) * d];
            let y = self.experts[e].forward_token(x, d, self.cfg.d_hidden);
            out[s * d..(s + 1) * d].copy_from_slice(&y);
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered {
                location: "mome_forward_hard",
            });
        }
        Ok(out)
    }

    /// Learned Top-1 soft routing: for each token compute gate logits
    /// `gate · x` per expert, pick the arg-max expert, and scale that expert's
    /// output by the softmax gate probability (the standard Switch-Transformer
    /// Top-1 weighting). Returns `(output [seq × d_model], chosen_expert [seq])`.
    ///
    /// # Errors
    /// Same shape errors as [`MoMeRouter::forward_hard`].
    pub fn forward_soft(&self, tokens: &[f32], seq: usize) -> MmResult<(Vec<f32>, Vec<usize>)> {
        let d = self.cfg.d_model;
        let n_e = self.cfg.n_experts;
        if tokens.len() != seq * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq * d,
                got: tokens.len(),
            });
        }
        let mut out = vec![0.0_f32; seq * d];
        let mut chosen = vec![0_usize; seq];
        for s in 0..seq {
            let x = &tokens[s * d..(s + 1) * d];
            // Gate logits per expert.
            let mut logits = vec![0.0_f32; n_e];
            for (e, slot) in logits.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for i in 0..d {
                    acc += x[i] * self.gate[e * d + i];
                }
                *slot = acc;
            }
            // Stable softmax over the gate logits.
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for l in logits.iter_mut() {
                *l = (*l - max_l).exp();
                sum += *l;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            // Top-1 expert = arg-max gate probability.
            let mut best_e = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for (e, &le) in logits.iter().enumerate() {
                let p = le * inv_sum;
                if p > best_p {
                    best_p = p;
                    best_e = e;
                }
            }
            chosen[s] = best_e;
            let y = self.experts[best_e].forward_token(x, d, self.cfg.d_hidden);
            for (o, &yo) in y.iter().enumerate() {
                out[s * d + o] = best_p * yo;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered {
                location: "mome_forward_soft",
            });
        }
        Ok((out, chosen))
    }

    /// Mutable access to an expert (for test weight injection / loading).
    pub fn expert_mut(&mut self, idx: usize) -> Option<&mut FfnExpert> {
        self.experts.get_mut(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn identity_expert(d_model: usize, d_hidden: usize) -> FfnExpert {
        // Not a true identity (GELU is nonlinear), but a deterministic
        // structured expert: W1 = I on the first min(d,h) channels.
        let mut e = FfnExpert::zeros(d_model, d_hidden);
        for k in 0..d_model.min(d_hidden) {
            e.w1[k * d_hidden + k] = 1.0;
            e.w2[k * d_model + k] = 1.0;
        }
        e
    }

    #[test]
    fn modality_expert_indices() {
        assert_eq!(Modality::Vision.expert_index(), 0);
        assert_eq!(Modality::Text.expert_index(), 1);
        assert_eq!(Modality::Fusion.expert_index(), 2);
    }

    #[test]
    fn zero_router_outputs_zero() {
        let cfg = MoMeConfig::vlmo(4, 8);
        let r = MoMeRouter::zeros(cfg).expect("router");
        let tokens = vec![0.5_f32; 3 * 4];
        let mods = [Modality::Vision, Modality::Text, Modality::Fusion];
        let out = r.forward_hard(&tokens, &mods, 3).expect("forward");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn routes_each_token_to_its_modality_expert() {
        // Build three experts that emit distinguishable constant biases, so the
        // output of a token reveals which expert processed it.
        let d = 4;
        let h = 4;
        let cfg = MoMeConfig::vlmo(d, h);
        let mut r = MoMeRouter::zeros(cfg).expect("router");
        for e in 0..3 {
            let bias = (e as f32 + 1.0) * 10.0;
            r.expert_mut(e).expect("expert").b2 = vec![bias; d];
        }
        let tokens = vec![0.0_f32; 3 * d];
        let mods = [Modality::Fusion, Modality::Vision, Modality::Text];
        let out = r.forward_hard(&tokens, &mods, 3).expect("forward");
        // token0 → fusion (expert 2, bias 30), token1 → vision (expert 0, bias 10),
        // token2 → text (expert 1, bias 20).
        assert!((out[0] - 30.0).abs() < 1e-4);
        assert!((out[d] - 10.0).abs() < 1e-4);
        assert!((out[2 * d] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn hard_routing_matches_manual_expert_call() {
        let d = 6;
        let h = 8;
        let cfg = MoMeConfig::vlmo(d, h);
        let mut r = MoMeRouter::zeros(cfg).expect("router");
        let mut rng = LcgRng::new(5);
        for e in 0..3 {
            let expert = r.expert_mut(e).expect("expert");
            rng.fill_normal(&mut expert.w1);
            rng.fill_normal(&mut expert.w2);
        }
        let mut tokens = vec![0.0_f32; 2 * d];
        rng.fill_normal(&mut tokens);
        let mods = [Modality::Text, Modality::Vision];
        let out = r.forward_hard(&tokens, &mods, 2).expect("forward");

        // Manually route token 0 through the language expert (index 1).
        let manual0 = r.experts[1].forward_token(&tokens[0..d], d, h);
        for i in 0..d {
            assert!((out[i] - manual0[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn soft_routing_picks_argmax_gate() {
        // Gate row for expert 1 aligned with the input → expert 1 wins.
        let d = 4;
        let h = 4;
        let cfg = MoMeConfig::vlmo(d, h);
        let mut r = MoMeRouter::zeros(cfg).expect("router");
        // gate[expert 1] = strong positive along channel 0.
        r.gate[d] = 10.0;
        let tokens = vec![1.0, 0.0, 0.0, 0.0]; // one token aligned with channel 0
        let (_out, chosen) = r.forward_soft(&tokens, 1).expect("forward");
        assert_eq!(chosen[0], 1, "expert 1 should be selected");
    }

    #[test]
    fn soft_routing_scales_by_gate_prob() {
        // With a structured expert and a dominant gate, the output equals the
        // gate probability times the expert output.
        let d = 4;
        let h = 4;
        let cfg = MoMeConfig::vlmo(d, h);
        let mut r = MoMeRouter::zeros(cfg).expect("router");
        *r.expert_mut(0).expect("e0") = identity_expert(d, h);
        // Make expert 0's gate dominate strongly so its prob ≈ 1.
        r.gate[0] = 50.0;
        let tokens = vec![0.5, 0.0, 0.0, 0.0];
        let (out, chosen) = r.forward_soft(&tokens, 1).expect("forward");
        assert_eq!(chosen[0], 0);
        // gate prob ≈ 1 → out ≈ expert0(token).
        let expert_out = r.experts[0].forward_token(&tokens, d, h);
        for i in 0..d {
            assert!(
                (out[i] - expert_out[i]).abs() < 1e-3,
                "out[{i}]={} expert={}",
                out[i],
                expert_out[i]
            );
        }
    }

    #[test]
    fn wrong_modality_count_errors() {
        let cfg = MoMeConfig::vlmo(4, 8);
        let r = MoMeRouter::zeros(cfg).expect("router");
        let tokens = vec![0.0_f32; 3 * 4];
        let mods = [Modality::Vision, Modality::Text]; // only 2 for seq 3
        assert!(matches!(
            r.forward_hard(&tokens, &mods, 3),
            Err(MultiModalError::MismatchedSeqLens { .. })
        ));
    }

    #[test]
    fn token_shape_mismatch_errors() {
        let cfg = MoMeConfig::vlmo(4, 8);
        let r = MoMeRouter::zeros(cfg).expect("router");
        let tokens = vec![0.0_f32; 2 * 4]; // seq says 3
        let mods = [Modality::Vision, Modality::Text, Modality::Fusion];
        assert!(matches!(
            r.forward_hard(&tokens, &mods, 3),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_config_errors() {
        let cfg = MoMeConfig {
            d_model: 0,
            d_hidden: 4,
            n_experts: 3,
        };
        assert!(matches!(
            MoMeRouter::zeros(cfg),
            Err(MultiModalError::InvalidFeatureDim)
        ));
    }

    #[test]
    fn deterministic_forward() {
        let d = 4;
        let h = 6;
        let cfg = MoMeConfig::vlmo(d, h);
        let mut r = MoMeRouter::zeros(cfg).expect("router");
        let mut rng = LcgRng::new(13);
        for e in 0..3 {
            rng.fill_normal(&mut r.expert_mut(e).expect("e").w1);
        }
        let tokens = vec![0.3_f32; 4 * d];
        let mods = [
            Modality::Vision,
            Modality::Text,
            Modality::Fusion,
            Modality::Vision,
        ];
        let a = r.forward_hard(&tokens, &mods, 4).expect("a");
        let b = r.forward_hard(&tokens, &mods, 4).expect("b");
        assert_eq!(a, b);
    }
}
