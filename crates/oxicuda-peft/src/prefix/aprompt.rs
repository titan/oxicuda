//! APrompt — Attention Prompt Tuning for Efficient Adaptation of PLMs.
//!
//! Reference: Wang Q, Mao Y, Wang J, Yu H, Li S, Wang S, Feng F, Huang L, Quan X,
//! Xu Z, Liu D (2023) "APrompt: Attention Prompt Tuning for Efficient Adaptation of
//! Pre-trained Language Models", *EMNLP 2023*: 9147–9160.
//! <https://aclanthology.org/2023.emnlp-main.567/>
//!
//! ## Design
//!
//! Rather than prepending soft tokens to the *input* sequence (prompt tuning) or to
//! the per-layer key/value cache as raw vectors (prefix tuning), APrompt injects a
//! small set of learnable prompt vectors *into the attention operation itself* as
//! additional key/value entries. Concretely, for a self-attention layer the original
//! keys `K ∈ ℝ^{n_kv × d_model}` and values `V ∈ ℝ^{n_kv × d_model}` are augmented to
//!
//! ```text
//! K' = [K ; prompt_keys]   ∈ ℝ^{(n_kv + n_prompt) × d_model}
//! V' = [V ; prompt_values] ∈ ℝ^{(n_kv + n_prompt) × d_model}
//! ```
//!
//! and every query attends over the augmented sets. Attention is multi-head with
//! `head_dim = d_model / n_heads` and the usual `1/√head_dim` scaling. The prompt
//! keys/values are the only learnable parameters — the base projections stay frozen.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an [`APrompt`] module.
#[derive(Debug, Clone)]
pub struct APromptConfig {
    /// Number of learnable prompt key/value pairs injected into the attention.
    pub n_prompt: usize,
    /// Model (embedding) dimension.
    pub d_model: usize,
    /// Number of attention heads; must divide `d_model` evenly.
    pub n_heads: usize,
}

// ---------------------------------------------------------------------------
// APrompt module
// ---------------------------------------------------------------------------

/// Attention-prompt module holding learnable prompt key/value pairs.
///
/// `prompt_keys` and `prompt_values` are each stored row-major as
/// `n_prompt × d_model`.
#[derive(Debug, Clone)]
pub struct APrompt {
    /// Prompt keys, flat row-major shape `n_prompt × d_model`.
    pub(crate) prompt_keys: Vec<f32>,
    /// Prompt values, flat row-major shape `n_prompt × d_model`.
    pub(crate) prompt_values: Vec<f32>,
    /// Module configuration.
    pub cfg: APromptConfig,
}

impl APrompt {
    /// Construct a new APrompt module with random initialization.
    ///
    /// Both prompt keys and prompt values are sampled from N(0, 0.02), mirroring
    /// the prefix/prompt-tuning init style used elsewhere in this crate.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if `n_prompt = 0`, `d_model = 0`,
    /// `n_heads = 0`, or [`PeftError::UnalignedDimension`] if `d_model % n_heads ≠ 0`.
    pub fn new(cfg: APromptConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        if cfg.n_prompt == 0 {
            return Err(PeftError::Internal {
                msg: "n_prompt must be >= 1".to_string(),
            });
        }
        if cfg.d_model == 0 {
            return Err(PeftError::Internal {
                msg: "d_model must be >= 1".to_string(),
            });
        }
        if cfg.n_heads == 0 {
            return Err(PeftError::Internal {
                msg: "n_heads must be >= 1".to_string(),
            });
        }
        if !cfg.d_model.is_multiple_of(cfg.n_heads) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.n_heads,
                in_dim: cfg.d_model,
            });
        }

        let size = cfg.n_prompt * cfg.d_model;
        let mut prompt_keys = vec![0.0_f32; size];
        rng.fill_normal(&mut prompt_keys);
        for v in prompt_keys.iter_mut() {
            *v *= 0.02;
        }
        let mut prompt_values = vec![0.0_f32; size];
        rng.fill_normal(&mut prompt_values);
        for v in prompt_values.iter_mut() {
            *v *= 0.02;
        }

        Ok(Self {
            prompt_keys,
            prompt_values,
            cfg,
        })
    }

    /// Multi-head attention over the prompt-augmented key/value sets.
    ///
    /// `queries` is `n_q × d_model`, `keys`/`values` are each `n_kv × d_model`
    /// (all flat row-major). For every head `h` and query `i` the logits are
    /// computed against `[keys ; prompt_keys]` (the `n_kv` originals followed by
    /// the `n_prompt` prompts), softmax-normalized with `1/√head_dim` scaling, and
    /// used to weight `[values ; prompt_values]`. The per-head outputs are
    /// concatenated back into `d_model`.
    ///
    /// Returns the attended output, flat row-major shape `n_q × d_model`.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if `n_q = 0` or `n_kv = 0`.
    /// - [`PeftError::DimensionMismatch`] if any of `queries`, `keys`, or `values`
    ///   does not match its declared `count × d_model` length.
    pub fn forward(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        n_q: usize,
        n_kv: usize,
    ) -> PeftResult<Vec<f32>> {
        if n_q == 0 {
            return Err(PeftError::EmptyInput);
        }
        if n_kv == 0 {
            return Err(PeftError::EmptyInput);
        }
        let d_model = self.cfg.d_model;
        if queries.len() != n_q * d_model {
            return Err(PeftError::DimensionMismatch {
                expected: n_q * d_model,
                got: queries.len(),
            });
        }
        if keys.len() != n_kv * d_model {
            return Err(PeftError::DimensionMismatch {
                expected: n_kv * d_model,
                got: keys.len(),
            });
        }
        if values.len() != n_kv * d_model {
            return Err(PeftError::DimensionMismatch {
                expected: n_kv * d_model,
                got: values.len(),
            });
        }

        let n_heads = self.cfg.n_heads;
        let n_prompt = self.cfg.n_prompt;
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_kv = n_kv + n_prompt;

        let mut out = vec![0.0_f32; n_q * d_model];

        // Returns the key slice (length head_dim) for augmented index `j`, head `h`.
        // Originals occupy [0, n_kv); prompts occupy [n_kv, total_kv).
        let key_slice = |j: usize, h: usize| -> &[f32] {
            let head_off = h * head_dim;
            if j < n_kv {
                let base = j * d_model + head_off;
                &keys[base..base + head_dim]
            } else {
                let p = j - n_kv;
                let base = p * d_model + head_off;
                &self.prompt_keys[base..base + head_dim]
            }
        };
        let value_slice = |j: usize, h: usize| -> &[f32] {
            let head_off = h * head_dim;
            if j < n_kv {
                let base = j * d_model + head_off;
                &values[base..base + head_dim]
            } else {
                let p = j - n_kv;
                let base = p * d_model + head_off;
                &self.prompt_values[base..base + head_dim]
            }
        };

        let mut logits = vec![0.0_f32; total_kv];
        for i in 0..n_q {
            for h in 0..n_heads {
                let head_off = h * head_dim;
                let q_base = i * d_model + head_off;
                let q_head = &queries[q_base..q_base + head_dim];

                // Logits over the augmented key set.
                for (j, slot) in logits.iter_mut().enumerate() {
                    let k_head = key_slice(j, h);
                    let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(&a, &b)| a * b).sum();
                    *slot = dot * scale;
                }

                // Numerically-stable softmax over the augmented key set.
                let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0_f32;
                for slot in logits.iter_mut() {
                    let e = (*slot - max_l).exp();
                    *slot = e;
                    denom += e;
                }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };

                // Weighted sum over the augmented value set.
                let out_base = i * d_model + head_off;
                let out_head = &mut out[out_base..out_base + head_dim];
                for (j, &w) in logits.iter().enumerate() {
                    let weight = w * inv;
                    if weight == 0.0 {
                        continue;
                    }
                    let v_head = value_slice(j, h);
                    for (o, &v) in out_head.iter_mut().zip(v_head.iter()) {
                        *o += weight * v;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Augmented key/value length seen by the attention: `n_kv + n_prompt`.
    #[inline]
    #[must_use]
    pub fn augmented_kv_len(&self, n_kv: usize) -> usize {
        n_kv + self.cfg.n_prompt
    }

    /// Total number of learnable parameters: `2 · n_prompt · d_model`.
    #[inline]
    #[must_use]
    pub fn n_params(&self) -> usize {
        2 * self.cfg.n_prompt * self.cfg.d_model
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_prompt: usize, d_model: usize, n_heads: usize) -> APromptConfig {
        APromptConfig {
            n_prompt,
            d_model,
            n_heads,
        }
    }

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| i as f32 * scale).collect()
    }

    // ── Test 1: forward output length ──────────────────────────────────────
    #[test]
    fn forward_output_length() {
        let mut rng = LcgRng::new(1);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let n_q = 3;
        let n_kv = 5;
        let q = ramp(n_q * 8, 0.01);
        let k = ramp(n_kv * 8, 0.02);
        let v = ramp(n_kv * 8, 0.03);
        let out = ap.forward(&q, &k, &v, n_q, n_kv).unwrap();
        assert_eq!(out.len(), n_q * 8);
    }

    // ── Test 2: augmented_kv_len ───────────────────────────────────────────
    #[test]
    fn augmented_kv_len_formula() {
        let mut rng = LcgRng::new(2);
        let ap = APrompt::new(cfg(4, 16, 4), &mut rng).unwrap();
        assert_eq!(ap.augmented_kv_len(10), 14);
        assert_eq!(ap.augmented_kv_len(0), 4);
    }

    // ── Test 3: n_params == 2 * n_prompt * d_model ─────────────────────────
    #[test]
    fn n_params_formula() {
        let mut rng = LcgRng::new(3);
        let ap = APrompt::new(cfg(5, 12, 3), &mut rng).unwrap();
        assert_eq!(ap.n_params(), 2 * 5 * 12);
    }

    // ── Test 4: zero prompt content → finite, shape-correct output ─────────
    #[test]
    fn zero_prompt_content_finite_output() {
        let mut rng = LcgRng::new(4);
        let mut ap = APrompt::new(cfg(3, 8, 2), &mut rng).unwrap();
        for v in ap.prompt_keys.iter_mut() {
            *v = 0.0;
        }
        for v in ap.prompt_values.iter_mut() {
            *v = 0.0;
        }
        let n_q = 2;
        let n_kv = 4;
        let q = ramp(n_q * 8, 0.05);
        let k = ramp(n_kv * 8, 0.07);
        let v = ramp(n_kv * 8, 0.09);
        let out = ap.forward(&q, &k, &v, n_q, n_kv).unwrap();
        assert_eq!(out.len(), n_q * 8);
        for &o in &out {
            assert!(o.is_finite(), "output must be finite, got {o}");
        }
    }

    // ── Test 5: nonzero prompt_values change the output ────────────────────
    #[test]
    fn nonzero_prompt_values_change_output() {
        let mut rng = LcgRng::new(5);
        let base = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        // Clone with prompt_values zeroed.
        let mut zeroed = base.clone();
        for v in zeroed.prompt_values.iter_mut() {
            *v = 0.0;
        }
        // Clone with prompt_values large.
        let mut large = base.clone();
        for v in large.prompt_values.iter_mut() {
            *v = 3.0;
        }
        let n_q = 2;
        let n_kv = 3;
        let q = ramp(n_q * 8, 0.1);
        let k = ramp(n_kv * 8, 0.05);
        let vv = ramp(n_kv * 8, 0.02);
        let out_zero = zeroed.forward(&q, &k, &vv, n_q, n_kv).unwrap();
        let out_large = large.forward(&q, &k, &vv, n_q, n_kv).unwrap();
        let diff: f32 = out_zero
            .iter()
            .zip(out_large.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "prompt_values must affect output, diff={diff}");
    }

    // ── Test 6: changing original keys changes output ──────────────────────
    #[test]
    fn changing_keys_changes_output() {
        let mut rng = LcgRng::new(6);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let n_q = 2;
        let n_kv = 3;
        let q = ramp(n_q * 8, 0.1);
        let k1 = ramp(n_kv * 8, 0.05);
        let mut k2 = k1.clone();
        // Perturb one key entry strongly.
        k2[0] += 5.0;
        let v = ramp(n_kv * 8, 0.02);
        let out1 = ap.forward(&q, &k1, &v, n_q, n_kv).unwrap();
        let out2 = ap.forward(&q, &k2, &v, n_q, n_kv).unwrap();
        let diff: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "changing keys must affect output, diff={diff}");
    }

    // ── Test 7: deterministic given seed ───────────────────────────────────
    #[test]
    fn deterministic_same_seed() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let a1 = APrompt::new(cfg(3, 8, 2), &mut r1).unwrap();
        let a2 = APrompt::new(cfg(3, 8, 2), &mut r2).unwrap();
        assert_eq!(a1.prompt_keys, a2.prompt_keys);
        assert_eq!(a1.prompt_values, a2.prompt_values);
        let q = ramp(2 * 8, 0.03);
        let k = ramp(3 * 8, 0.04);
        let v = ramp(3 * 8, 0.05);
        assert_eq!(
            a1.forward(&q, &k, &v, 2, 3).unwrap(),
            a2.forward(&q, &k, &v, 2, 3).unwrap()
        );
    }

    // ── Test 8: single head works ──────────────────────────────────────────
    #[test]
    fn single_head_works() {
        let mut rng = LcgRng::new(8);
        let ap = APrompt::new(cfg(2, 6, 1), &mut rng).unwrap();
        let n_q = 2;
        let n_kv = 3;
        let q = ramp(n_q * 6, 0.1);
        let k = ramp(n_kv * 6, 0.05);
        let v = ramp(n_kv * 6, 0.02);
        let out = ap.forward(&q, &k, &v, n_q, n_kv).unwrap();
        assert_eq!(out.len(), n_q * 6);
        for &o in &out {
            assert!(o.is_finite());
        }
    }

    // ── Test 9: d_model % n_heads != 0 → Err ───────────────────────────────
    #[test]
    fn unaligned_heads_errors() {
        let mut rng = LcgRng::new(9);
        let res = APrompt::new(cfg(2, 10, 3), &mut rng); // 10 % 3 != 0
        assert!(matches!(res, Err(PeftError::UnalignedDimension { .. })));
    }

    // ── Test 10: queries length mismatch → Err ─────────────────────────────
    #[test]
    fn queries_length_mismatch_errors() {
        let mut rng = LcgRng::new(10);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let bad_q = ramp(2 * 8 - 1, 0.1); // wrong length for n_q=2
        let k = ramp(3 * 8, 0.05);
        let v = ramp(3 * 8, 0.02);
        let res = ap.forward(&bad_q, &k, &v, 2, 3);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    // ── Test 11: keys length mismatch → Err ────────────────────────────────
    #[test]
    fn keys_length_mismatch_errors() {
        let mut rng = LcgRng::new(11);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let q = ramp(2 * 8, 0.1);
        let bad_k = ramp(3 * 8 + 2, 0.05);
        let v = ramp(3 * 8, 0.02);
        let res = ap.forward(&q, &bad_k, &v, 2, 3);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    // ── Test 12: values length mismatch → Err ──────────────────────────────
    #[test]
    fn values_length_mismatch_errors() {
        let mut rng = LcgRng::new(12);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let q = ramp(2 * 8, 0.1);
        let k = ramp(3 * 8, 0.05);
        let bad_v = ramp(3 * 8 - 3, 0.02);
        let res = ap.forward(&q, &k, &bad_v, 2, 3);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    // ── Test 13: n_q = 1, n_kv = 1 works ───────────────────────────────────
    #[test]
    fn single_query_single_kv_works() {
        let mut rng = LcgRng::new(13);
        let ap = APrompt::new(cfg(2, 4, 2), &mut rng).unwrap();
        let q = ramp(4, 0.1);
        let k = ramp(4, 0.05);
        let v = ramp(4, 0.02);
        let out = ap.forward(&q, &k, &v, 1, 1).unwrap();
        assert_eq!(out.len(), 4);
        for &o in &out {
            assert!(o.is_finite());
        }
    }

    // ── Test 14: output is a convex combination → bounded by aug V min/max ──
    #[test]
    fn output_bounded_by_augmented_values() {
        let mut rng = LcgRng::new(14);
        let d_model = 6;
        let n_heads = 2;
        let head_dim = d_model / n_heads;
        let n_prompt = 2;
        let ap = APrompt::new(cfg(n_prompt, d_model, n_heads), &mut rng).unwrap();
        let n_q = 3;
        let n_kv = 4;
        let q = ramp(n_q * d_model, 0.07);
        let k = ramp(n_kv * d_model, 0.05);
        let v = ramp(n_kv * d_model, 0.11);
        let out = ap.forward(&q, &k, &v, n_q, n_kv).unwrap();

        let total_kv = n_kv + n_prompt;
        // For each head and per-head dim, the output is a convex combination of
        // the augmented values' entries at that (head, dim), so it must lie within
        // [min, max] over the augmented value rows.
        for h in 0..n_heads {
            for d in 0..head_dim {
                let head_off = h * head_dim;
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for j in 0..total_kv {
                    let val = if j < n_kv {
                        v[j * d_model + head_off + d]
                    } else {
                        let p = j - n_kv;
                        ap.prompt_values[p * d_model + head_off + d]
                    };
                    lo = lo.min(val);
                    hi = hi.max(val);
                }
                for i in 0..n_q {
                    let o = out[i * d_model + head_off + d];
                    assert!(
                        o >= lo - 1e-4 && o <= hi + 1e-4,
                        "output {o} outside convex hull [{lo}, {hi}] at head {h} dim {d}"
                    );
                }
            }
        }
    }

    // ── Test 15: err — n_prompt = 0 ────────────────────────────────────────
    #[test]
    fn err_n_prompt_zero() {
        let mut rng = LcgRng::new(15);
        let res = APrompt::new(cfg(0, 8, 2), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 16: err — d_model = 0 ─────────────────────────────────────────
    #[test]
    fn err_d_model_zero() {
        let mut rng = LcgRng::new(16);
        let res = APrompt::new(cfg(2, 0, 1), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 17: err — n_heads = 0 ─────────────────────────────────────────
    #[test]
    fn err_n_heads_zero() {
        let mut rng = LcgRng::new(17);
        let res = APrompt::new(cfg(2, 8, 0), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 18: err — n_q = 0 ─────────────────────────────────────────────
    #[test]
    fn err_n_q_zero() {
        let mut rng = LcgRng::new(18);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let res = ap.forward(&[], &ramp(3 * 8, 0.05), &ramp(3 * 8, 0.02), 0, 3);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    // ── Test 19: err — n_kv = 0 ────────────────────────────────────────────
    #[test]
    fn err_n_kv_zero() {
        let mut rng = LcgRng::new(19);
        let ap = APrompt::new(cfg(2, 8, 2), &mut rng).unwrap();
        let res = ap.forward(&ramp(2 * 8, 0.1), &[], &[], 2, 0);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    // ── Test 20: n_prompt large relative to n_kv works ─────────────────────
    #[test]
    fn large_n_prompt_works() {
        let mut rng = LcgRng::new(20);
        let ap = APrompt::new(cfg(32, 8, 2), &mut rng).unwrap(); // 32 prompts, 1 kv
        let q = ramp(2 * 8, 0.1);
        let k = ramp(8, 0.05);
        let v = ramp(8, 0.02);
        let out = ap.forward(&q, &k, &v, 2, 1).unwrap();
        assert_eq!(out.len(), 2 * 8);
        for &o in &out {
            assert!(o.is_finite());
        }
        assert_eq!(ap.augmented_kv_len(1), 33);
    }

    // ── Test 21: softmax weights effectively normalized (uniform V → that V) ─
    #[test]
    fn uniform_values_reproduced() {
        // If every augmented value row equals a constant per (head,dim), the
        // convex-combination output must equal that constant exactly, proving the
        // attention weights sum to 1.
        let mut rng = LcgRng::new(21);
        let d_model = 4;
        let ap0 = APrompt::new(cfg(3, d_model, 2), &mut rng).unwrap();
        let mut ap = ap0;
        // Set prompt values to the constant 2.5 everywhere.
        for v in ap.prompt_values.iter_mut() {
            *v = 2.5;
        }
        let n_q = 2;
        let n_kv = 3;
        let q = ramp(n_q * d_model, 0.3);
        let k = ramp(n_kv * d_model, 0.1);
        // Original values also constant 2.5.
        let v = vec![2.5_f32; n_kv * d_model];
        let out = ap.forward(&q, &k, &v, n_q, n_kv).unwrap();
        for &o in &out {
            assert!(
                (o - 2.5).abs() < 1e-4,
                "convex combination of constant 2.5 must be 2.5, got {o}"
            );
        }
    }
}
