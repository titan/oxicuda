//! N-gram statistical language model with smoothing and perplexity evaluation.
//!
//! This module provides a classic count-based n-gram language model over a
//! sequence of integer token ids (the same `u32` ids produced by the
//! tokenizers in [`crate::tokenizer`]).  It is a self-contained CPU reference
//! useful for:
//!
//! * **Intrinsic LM evaluation** — cross-entropy and perplexity of a held-out
//!   token sequence under a trained n-gram model (the standard way to score a
//!   tokenizer + corpus combination).
//! * **Lightweight drafting / scoring** — a cheap auxiliary model whose
//!   `log_prob` can rank candidate continuations without a neural forward pass.
//!
//! # Model
//!
//! For order `n`, the model estimates
//!
//! ```text
//!   P(w_i | w_{i-n+1} … w_{i-1})
//! ```
//!
//! Two smoothing schemes are supported:
//!
//! * [`Smoothing::AddK`] — additive (Laplace / Lidstone) smoothing with a
//!   pseudo-count `k`:
//!   `P(w | ctx) = (count(ctx,w) + k) / (count(ctx) + k·V)`.
//! * [`Smoothing::Interpolated`] — Jelinek–Mercer linear interpolation that
//!   mixes the `n`-gram, `(n-1)`-gram, …, unigram and a uniform `1/V` floor
//!   with fixed weights `lambdas`, guaranteeing every probability is strictly
//!   positive (so perplexity is always finite).
//!
//! Both schemes return well-defined probabilities for **unseen contexts**: an
//! unseen context falls back through the lower-order distributions (or, for
//! add-k, becomes the uniform `1/V`).
//!
//! # Padding
//!
//! Sentences are padded with `n-1` beginning-of-sequence markers and a single
//! end-of-sequence marker (the standard `<s> … </s>` convention) so that the
//! model can score sentence-initial tokens and assign probability mass to
//! sentence termination.  The BOS/EOS ids are part of the configured
//! vocabulary size.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};

// ─── Smoothing ───────────────────────────────────────────────────────────────

/// Smoothing strategy for an [`NgramModel`].
#[derive(Debug, Clone, PartialEq)]
pub enum Smoothing {
    /// Additive smoothing with pseudo-count `k` (`k = 1.0` is Laplace).
    AddK {
        /// Pseudo-count added to every (context, word) numerator.
        k: f64,
    },
    /// Jelinek–Mercer interpolation over orders `n, n-1, …, 1` and a uniform
    /// floor.  `lambdas` has length `n + 1`: `lambdas[0]` weights the
    /// highest-order distribution, …, `lambdas[n-1]` the unigram, and
    /// `lambdas[n]` the uniform `1/V`.  The weights must be non-negative and
    /// sum to `1`.
    Interpolated {
        /// Mixing weights, highest order first, uniform floor last.
        lambdas: Vec<f64>,
    },
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for an [`NgramModel`].
#[derive(Debug, Clone)]
pub struct NgramConfig {
    /// Model order (`n` in "n-gram"): 1 = unigram, 2 = bigram, ….
    pub order: usize,
    /// Vocabulary size (must exceed every token id, BOS and EOS included).
    pub vocab_size: usize,
    /// Beginning-of-sequence marker id (used for left padding).
    pub bos_id: u32,
    /// End-of-sequence marker id (appended to every sentence).
    pub eos_id: u32,
    /// Smoothing scheme.
    pub smoothing: Smoothing,
}

impl NgramConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// [`LmError::InvalidConfig`] if `order == 0`, `vocab_size == 0`, the
    /// BOS/EOS ids are out of range, or the smoothing parameters are invalid
    /// (negative `k`, wrong-length or non-normalised `lambdas`).
    pub fn validate(&self) -> LmResult<()> {
        if self.order == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n-gram order must be >= 1".into(),
            });
        }
        if self.vocab_size == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n-gram vocab_size must be >= 1".into(),
            });
        }
        if self.bos_id as usize >= self.vocab_size || self.eos_id as usize >= self.vocab_size {
            return Err(LmError::InvalidConfig {
                msg: "BOS/EOS id must be within vocab_size".into(),
            });
        }
        match &self.smoothing {
            Smoothing::AddK { k } => {
                if !k.is_finite() || *k <= 0.0 {
                    return Err(LmError::InvalidConfig {
                        msg: "add-k smoothing requires k > 0".into(),
                    });
                }
            }
            Smoothing::Interpolated { lambdas } => {
                if lambdas.len() != self.order + 1 {
                    return Err(LmError::InvalidConfig {
                        msg: format!(
                            "interpolation needs {} lambdas (order {} + uniform), got {}",
                            self.order + 1,
                            self.order,
                            lambdas.len()
                        ),
                    });
                }
                let mut sum = 0.0;
                for &l in lambdas {
                    if !l.is_finite() || l < 0.0 {
                        return Err(LmError::InvalidConfig {
                            msg: "interpolation lambdas must be finite and non-negative".into(),
                        });
                    }
                    sum += l;
                }
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(LmError::InvalidConfig {
                        msg: format!("interpolation lambdas must sum to 1 (got {sum})"),
                    });
                }
            }
        }
        Ok(())
    }
}

// ─── NgramModel ──────────────────────────────────────────────────────────────

/// A trained count-based n-gram language model.
///
/// Build with [`NgramModel::new`], feed training sentences with
/// [`NgramModel::train`], then query [`NgramModel::log_prob`] /
/// [`NgramModel::perplexity`].
#[derive(Debug, Clone)]
pub struct NgramModel {
    config: NgramConfig,
    /// `order_counts[m]` holds counts of `(m+1)`-grams keyed by the full gram
    /// (context tokens followed by the predicted token).  `order_counts[0]`
    /// are unigram counts (key length 1).
    order_counts: Vec<HashMap<Vec<u32>, f64>>,
    /// `context_counts[m]` holds the total count of each `m`-gram context for
    /// the `(m+1)`-order distribution.  `context_counts[0]` is the single
    /// empty-context total (total unigram tokens).
    context_counts: Vec<HashMap<Vec<u32>, f64>>,
}

impl NgramModel {
    // ── Constructor ──────────────────────────────────────────────────────

    /// Create an untrained model with the given configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`NgramConfig::validate`].
    pub fn new(config: NgramConfig) -> LmResult<Self> {
        config.validate()?;
        let order = config.order;
        Ok(Self {
            config,
            order_counts: (0..order).map(|_| HashMap::new()).collect(),
            context_counts: (0..order).map(|_| HashMap::new()).collect(),
        })
    }

    /// Model order.
    pub fn order(&self) -> usize {
        self.config.order
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    // ── Training ─────────────────────────────────────────────────────────

    /// Accumulate counts from a single sentence (a slice of token ids).
    ///
    /// The sentence is padded with `order-1` BOS markers and one trailing EOS
    /// marker before counting.  Call repeatedly to train on a corpus.
    ///
    /// # Errors
    ///
    /// [`LmError::OutOfVocab`] if any token id is `>= vocab_size`.
    pub fn train(&mut self, sentence: &[u32]) -> LmResult<()> {
        for &t in sentence {
            if t as usize >= self.config.vocab_size {
                return Err(LmError::OutOfVocab { token: t });
            }
        }
        let padded = self.pad(sentence);
        let order = self.config.order;

        // For every gram order m+1 (1..=order), slide a window over `padded`.
        for m in 0..order {
            let gram_len = m + 1;
            if padded.len() < gram_len {
                continue;
            }
            for w in padded.windows(gram_len) {
                let gram = w.to_vec();
                *self.order_counts[m].entry(gram).or_insert(0.0) += 1.0;
                let ctx = w[..gram_len - 1].to_vec();
                *self.context_counts[m].entry(ctx).or_insert(0.0) += 1.0;
            }
        }
        Ok(())
    }

    /// Train on an entire corpus of sentences in one call.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::train`] for the first offending sentence.
    pub fn train_corpus(&mut self, corpus: &[Vec<u32>]) -> LmResult<()> {
        for sentence in corpus {
            self.train(sentence)?;
        }
        Ok(())
    }

    // ── Probability ──────────────────────────────────────────────────────

    /// Probability of `token` given a context (most-recent token last).
    ///
    /// The context is truncated to the model's order (only the last
    /// `order - 1` tokens matter).  The returned probability is strictly
    /// positive for every in-range token under both smoothing schemes.
    ///
    /// # Errors
    ///
    /// [`LmError::OutOfVocab`] if `token` is out of range.
    pub fn prob(&self, context: &[u32], token: u32) -> LmResult<f64> {
        if token as usize >= self.config.vocab_size {
            return Err(LmError::OutOfVocab { token });
        }
        Ok(match &self.config.smoothing {
            Smoothing::AddK { k } => self.prob_add_k(context, token, *k),
            Smoothing::Interpolated { lambdas } => self.prob_interpolated(context, token, lambdas),
        })
    }

    /// Natural-log probability of `token` given `context`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::prob`].
    pub fn log_prob(&self, context: &[u32], token: u32) -> LmResult<f64> {
        Ok(self.prob(context, token)?.ln())
    }

    /// Total natural-log probability of a full sentence under the model.
    ///
    /// The sentence is padded exactly as during training; the log-probabilities
    /// of every (real token + EOS) prediction are summed.  BOS predictions are
    /// not scored (they are pure context), matching standard perplexity
    /// accounting.
    ///
    /// Returns `(total_log_prob, n_predicted_tokens)`.
    ///
    /// # Errors
    ///
    /// [`LmError::OutOfVocab`] if any token id is out of range.
    pub fn sentence_log_prob(&self, sentence: &[u32]) -> LmResult<(f64, usize)> {
        for &t in sentence {
            if t as usize >= self.config.vocab_size {
                return Err(LmError::OutOfVocab { token: t });
            }
        }
        let padded = self.pad(sentence);
        let order = self.config.order;
        // The first `order-1` positions are BOS padding context; predictions
        // start at index `order-1` (the first real token) through the final
        // EOS at the end.
        let mut total = 0.0;
        let mut count = 0usize;
        let start = order - 1;
        for i in start..padded.len() {
            let ctx_lo = i.saturating_sub(order - 1);
            let context = &padded[ctx_lo..i];
            let token = padded[i];
            total += self.log_prob(context, token)?;
            count += 1;
        }
        Ok((total, count))
    }

    /// Cross-entropy (in **nats per token**) of a held-out corpus.
    ///
    /// # Errors
    ///
    /// [`LmError::EmptyInput`] if the corpus contributes no predicted tokens;
    /// otherwise propagates [`Self::sentence_log_prob`].
    pub fn cross_entropy(&self, corpus: &[Vec<u32>]) -> LmResult<f64> {
        let mut total = 0.0;
        let mut count = 0usize;
        for sentence in corpus {
            let (lp, n) = self.sentence_log_prob(sentence)?;
            total += lp;
            count += n;
        }
        if count == 0 {
            return Err(LmError::EmptyInput {
                context: "NgramModel::cross_entropy corpus",
            });
        }
        // Cross-entropy = - (1/N) Σ log P.
        Ok(-total / count as f64)
    }

    /// Perplexity of a held-out corpus: `exp(cross_entropy)`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::cross_entropy`].
    pub fn perplexity(&self, corpus: &[Vec<u32>]) -> LmResult<f64> {
        Ok(self.cross_entropy(corpus)?.exp())
    }

    // ── Private: padding ─────────────────────────────────────────────────

    /// Pad a sentence with `order-1` BOS markers and a trailing EOS marker.
    fn pad(&self, sentence: &[u32]) -> Vec<u32> {
        let pad = self.config.order.saturating_sub(1);
        let mut out = Vec::with_capacity(pad + sentence.len() + 1);
        for _ in 0..pad {
            out.push(self.config.bos_id);
        }
        out.extend_from_slice(sentence);
        out.push(self.config.eos_id);
        out
    }

    // ── Private: smoothed probabilities ──────────────────────────────────

    /// Add-k probability for the *full* order using the last `order-1` context
    /// tokens.  Falls back to uniform `1/V` for an unseen context.
    fn prob_add_k(&self, context: &[u32], token: u32, k: f64) -> f64 {
        let order = self.config.order;
        let want = order - 1;
        let ctx: Vec<u32> = if context.len() > want {
            context[context.len() - want..].to_vec()
        } else {
            context.to_vec()
        };
        // If the context is shorter than required (e.g. user passed a partial
        // context), back off to the matching lower order so the estimate is
        // still well-defined.
        let m = ctx.len().min(order - 1); // gram order index = m  (gram_len = m+1)
        let mut gram = ctx.clone();
        gram.push(token);
        let num = self.order_counts[m].get(&gram).copied().unwrap_or(0.0) + k;
        let denom = self.context_counts[m].get(&ctx).copied().unwrap_or(0.0)
            + k * self.config.vocab_size as f64;
        // denom is always > 0 because k > 0 and vocab_size >= 1.
        num / denom
    }

    /// Maximum-likelihood probability of `token` given an exact `m`-gram
    /// context for the `(m+1)`-order distribution; `None` if the context was
    /// never observed.
    fn ml_prob(&self, ctx: &[u32], token: u32, m: usize) -> Option<f64> {
        let ctx_total = self.context_counts[m].get(ctx).copied().unwrap_or(0.0);
        if ctx_total == 0.0 {
            return None;
        }
        let mut gram = ctx.to_vec();
        gram.push(token);
        let c = self.order_counts[m].get(&gram).copied().unwrap_or(0.0);
        Some(c / ctx_total)
    }

    /// Jelinek–Mercer interpolation across all orders plus a uniform floor.
    fn prob_interpolated(&self, context: &[u32], token: u32, lambdas: &[f64]) -> f64 {
        let order = self.config.order;
        let want = order - 1;
        let ctx_full: Vec<u32> = if context.len() > want {
            context[context.len() - want..].to_vec()
        } else {
            context.to_vec()
        };

        let mut p = 0.0;
        // Highest order first: lambdas[0] ↔ gram order index (order-1).
        // For gram order index m, the context is the last m tokens of ctx_full.
        for (li, m) in (0..order).rev().enumerate() {
            let lambda = lambdas[li];
            if lambda == 0.0 {
                continue;
            }
            // m tokens of context (m may exceed available context length when
            // scoring sentence-initial tokens — in that case there simply is
            // no count and this order contributes 0 mass for this prediction).
            let ctx = if ctx_full.len() >= m {
                &ctx_full[ctx_full.len() - m..]
            } else {
                continue;
            };
            if let Some(ml) = self.ml_prob(ctx, token, m) {
                p += lambda * ml;
            }
        }
        // Uniform floor guarantees strict positivity.
        p += lambdas[order] * (1.0 / self.config.vocab_size as f64);
        p
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bigram_addk() -> NgramModel {
        let cfg = NgramConfig {
            order: 2,
            vocab_size: 6, // ids 0..5; 4 = BOS, 5 = EOS
            bos_id: 4,
            eos_id: 5,
            smoothing: Smoothing::AddK { k: 1.0 },
        };
        NgramModel::new(cfg).expect("bigram config should be valid")
    }

    #[test]
    fn config_validation_rejects_bad() {
        let bad_order = NgramConfig {
            order: 0,
            vocab_size: 4,
            bos_id: 2,
            eos_id: 3,
            smoothing: Smoothing::AddK { k: 1.0 },
        };
        assert!(bad_order.validate().is_err());

        let bad_k = NgramConfig {
            order: 2,
            vocab_size: 4,
            bos_id: 2,
            eos_id: 3,
            smoothing: Smoothing::AddK { k: -1.0 },
        };
        assert!(bad_k.validate().is_err());

        let bad_lambdas = NgramConfig {
            order: 2,
            vocab_size: 4,
            bos_id: 2,
            eos_id: 3,
            smoothing: Smoothing::Interpolated {
                lambdas: vec![0.5, 0.4], // wrong length (need 3) and sum != 1
            },
        };
        assert!(bad_lambdas.validate().is_err());
    }

    #[test]
    fn probabilities_form_a_distribution_add_k() {
        // After training, P(· | ctx) must sum to 1 over the whole vocab.
        let mut m = bigram_addk();
        m.train(&[0, 1, 2]).expect("train ok");
        m.train(&[0, 1, 3]).expect("train ok");
        let ctx = [1u32];
        let total: f64 = (0..m.vocab_size() as u32)
            .map(|tok| m.prob(&ctx, tok).expect("prob ok"))
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "sum={total}");
    }

    #[test]
    fn probabilities_form_a_distribution_interpolated() {
        let cfg = NgramConfig {
            order: 3,
            vocab_size: 6,
            bos_id: 4,
            eos_id: 5,
            smoothing: Smoothing::Interpolated {
                lambdas: vec![0.5, 0.3, 0.15, 0.05],
            },
        };
        let mut m = NgramModel::new(cfg).expect("trigram config should be valid");
        m.train(&[0, 1, 2, 3]).expect("train ok");
        m.train(&[0, 1, 2, 0]).expect("train ok");
        let ctx = [0u32, 1, 2];
        let total: f64 = (0..m.vocab_size() as u32)
            .map(|tok| m.prob(&ctx, tok).expect("prob ok"))
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "sum={total}");
    }

    #[test]
    fn add_k_known_value() {
        // Train so context [1] is followed once by 2 and once by 3.
        // count([1])=2, count([1,2])=1. With k=1, V=6:
        // P(2|1) = (1+1)/(2+6) = 2/8 = 0.25.
        let mut m = bigram_addk();
        m.train(&[0, 1, 2]).expect("train ok"); // grams: (1,2) etc.
        m.train(&[0, 1, 3]).expect("train ok"); // grams: (1,3)
        let p = m.prob(&[1], 2).expect("prob ok");
        assert!((p - 0.25).abs() < 1e-9, "p={p}");
        // Unseen continuation: P(0|1) = (0+1)/(2+6) = 0.125.
        let p0 = m.prob(&[1], 0).expect("prob ok");
        assert!((p0 - 0.125).abs() < 1e-9, "p0={p0}");
    }

    #[test]
    fn unseen_context_backs_off_to_uniform_add_k() {
        // Context [3] never seen → add-k denom = 0 + k·V = 6, num = 0 + 1.
        let mut m = bigram_addk();
        m.train(&[0, 1, 2]).expect("train ok");
        let p = m.prob(&[3], 0).expect("prob ok");
        assert!((p - 1.0 / 6.0).abs() < 1e-9, "p={p}");
    }

    #[test]
    fn out_of_vocab_token_errors() {
        let m = bigram_addk();
        assert!(matches!(
            m.prob(&[0], 99),
            Err(LmError::OutOfVocab { token: 99 })
        ));
    }

    #[test]
    fn train_out_of_vocab_errors() {
        let mut m = bigram_addk();
        assert!(matches!(
            m.train(&[0, 99]),
            Err(LmError::OutOfVocab { token: 99 })
        ));
    }

    #[test]
    fn perplexity_lower_on_seen_than_unseen() {
        // A model trained on a repeated pattern should assign lower perplexity
        // to that pattern than to a random one.
        let mut m = bigram_addk();
        let train: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        m.train_corpus(&train).expect("train corpus ok");

        let seen = vec![vec![0u32, 1, 2]];
        let unseen = vec![vec![3u32, 3, 3]];
        let ppl_seen = m.perplexity(&seen).expect("ppl seen");
        let ppl_unseen = m.perplexity(&unseen).expect("ppl unseen");
        assert!(
            ppl_seen < ppl_unseen,
            "expected seen ppl {ppl_seen} < unseen ppl {ppl_unseen}"
        );
        assert!(ppl_seen.is_finite() && ppl_seen > 0.0);
    }

    #[test]
    fn perplexity_is_finite_for_all_inputs() {
        // Smoothing must keep perplexity finite even on never-before-seen text.
        let mut m = bigram_addk();
        m.train(&[0, 1]).expect("train ok");
        let ppl = m.perplexity(&[vec![3, 3, 3]]).expect("ppl ok");
        assert!(ppl.is_finite(), "ppl={ppl}");
    }

    #[test]
    fn cross_entropy_empty_corpus_errors() {
        let m = bigram_addk();
        assert!(matches!(
            m.cross_entropy(&[]),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn interpolated_perplexity_beats_unigram_floor_on_structured_data() {
        // On strongly structured data, a trigram interpolation should achieve
        // a lower perplexity than a pure-uniform model would (V tokens → ppl V).
        let cfg = NgramConfig {
            order: 3,
            vocab_size: 6,
            bos_id: 4,
            eos_id: 5,
            smoothing: Smoothing::Interpolated {
                lambdas: vec![0.7, 0.2, 0.08, 0.02],
            },
        };
        let mut m = NgramModel::new(cfg).expect("trigram config should be valid");
        let corpus: Vec<Vec<u32>> = vec![vec![0, 1, 2, 0, 1, 2]; 8];
        m.train_corpus(&corpus).expect("train ok");
        let ppl = m.perplexity(&[vec![0, 1, 2, 0, 1, 2]]).expect("ppl ok");
        // Uniform model over V=6 symbols would give ppl=6; the structured model
        // must do meaningfully better.
        assert!(ppl < 6.0, "interpolated ppl {ppl} should beat uniform 6");
    }

    #[test]
    fn unigram_model_works() {
        // order=1 model: context is always empty, pure unigram distribution.
        let cfg = NgramConfig {
            order: 1,
            vocab_size: 4,
            bos_id: 2,
            eos_id: 3,
            smoothing: Smoothing::AddK { k: 1.0 },
        };
        let mut m = NgramModel::new(cfg).expect("unigram config should be valid");
        m.train(&[0, 0, 1]).expect("train ok"); // tokens 0,0,1 + EOS(3)
        // count(0)=2, total unigrams = 0,0,1,EOS = 4. P(0)=(2+1)/(4+4)=3/8.
        let p0 = m.prob(&[], 0).expect("prob ok");
        assert!((p0 - 3.0 / 8.0).abs() < 1e-9, "p0={p0}");
        let total: f64 = (0..4u32).map(|t| m.prob(&[], t).expect("prob")).sum();
        assert!((total - 1.0).abs() < 1e-9, "sum={total}");
    }
}
