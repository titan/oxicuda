//! Joint CTC + attention decoding (Watanabe 2017).
//!
//! The hybrid CTC/attention architecture (Watanabe et al., 2017) combines the
//! complementary strengths of a frame-synchronous **CTC** branch and a
//! label-synchronous **attention decoder** branch that share an encoder. Both
//! branches score a hypothesis `y`, and the joint score interpolates the two
//! log-probabilities with a weight `λ ∈ [0, 1]`:
//!
//! ```text
//!   score(y) = λ · log p_ctc(y | x)  +  (1 − λ) · log p_att(y | x)
//! ```
//!
//! `λ = 1` recovers pure CTC; `λ = 0` recovers pure attention. This module
//! provides two faithful, deterministic pieces:
//!
//! - [`JointCtcAttention::joint_score`] — the joint score of a *given*
//!   hypothesis, using the exact CTC marginal log-likelihood (reusing the
//!   crate's [`ctc_forward_log`] forward algorithm) for the CTC term and the
//!   summed per-step attention log-probabilities for the attention term.
//! - [`JointCtcAttention::decode`] — a frame-synchronous greedy *one-best*
//!   joint decode: at each frame the per-symbol joint log-probability
//!   `λ·ctc + (1−λ)·attn` is maximised, and the resulting frame path is reduced
//!   by the standard CTC collapse (merge consecutive duplicates, drop blank).
//!
//! ## References
//! - Watanabe, S. et al. (2017). "Hybrid CTC/Attention Architecture for
//!   End-to-End Speech Recognition." *IEEE J. Sel. Topics Signal Process.*
//!   11(8), 1240–1253.

use crate::ctc::forward::ctc_forward_log;
use crate::error::{AudioError, AudioResult};

// ─── Public type ─────────────────────────────────────────────────────────────

/// Joint CTC + attention scorer / decoder.
#[derive(Debug, Clone, Copy)]
pub struct JointCtcAttention {
    lambda: f32,
    blank: usize,
}

impl JointCtcAttention {
    /// Build a joint CTC/attention decoder.
    ///
    /// `lambda` is the CTC interpolation weight `λ ∈ [0, 1]` (`1` = pure CTC,
    /// `0` = pure attention). `blank` is the CTC blank index; it is validated
    /// against the vocabulary size at scoring / decoding time.
    ///
    /// # Errors
    /// - [`AudioError::Internal`] if `lambda` is not a finite value in `[0, 1]`.
    pub fn new(lambda: f32, blank: usize) -> AudioResult<Self> {
        if !(lambda.is_finite() && (0.0..=1.0).contains(&lambda)) {
            return Err(AudioError::Internal(format!(
                "joint_ctc_attention: lambda must be a finite value in [0, 1], got {lambda}"
            )));
        }
        Ok(Self { lambda, blank })
    }

    /// The CTC interpolation weight `λ`.
    #[must_use]
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// The CTC blank index.
    #[must_use]
    pub fn blank(&self) -> usize {
        self.blank
    }

    /// Joint score of a given hypothesis `hyp`.
    ///
    /// The CTC term is the exact marginal log-likelihood `log p_ctc(hyp | x)`
    /// from [`ctc_forward_log`] over the `[t, vocab]` CTC log-probabilities. The
    /// attention term is `Σ_u attn_logprobs[u, hyp[u]]` over the `[attn_steps,
    /// vocab]` attention-decoder log-probabilities (one row per output step).
    ///
    /// Returns `λ · ctc_logprob + (1 − λ) · attn_logprob`.
    ///
    /// # Errors
    /// - [`AudioError::InvalidVocabSize`] if `vocab == 0`.
    /// - [`AudioError::BlankOutOfRange`] if `blank ≥ vocab`.
    /// - [`AudioError::ShapeMismatch`] if `ctc_logprobs.len() != t·vocab`,
    ///   `attn_logprobs.len() != attn_steps·vocab`, a hypothesis token is
    ///   `≥ vocab`, or `hyp.len() > attn_steps` (too few attention rows).
    /// - Any error propagated by [`ctc_forward_log`].
    pub fn joint_score(
        &self,
        ctc_logprobs: &[f32],
        t: usize,
        vocab: usize,
        attn_logprobs: &[f32],
        attn_steps: usize,
        hyp: &[usize],
    ) -> AudioResult<f32> {
        if vocab == 0 {
            return Err(AudioError::InvalidVocabSize(vocab));
        }
        if self.blank >= vocab {
            return Err(AudioError::BlankOutOfRange {
                blank: self.blank,
                vocab,
            });
        }
        if attn_logprobs.len() != attn_steps * vocab {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "attn_logprobs length {} does not match attn_steps({}) * vocab({})",
                    attn_logprobs.len(),
                    attn_steps,
                    vocab
                ),
            });
        }
        if hyp.len() > attn_steps {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "hypothesis length {} exceeds attn_steps {}",
                    hyp.len(),
                    attn_steps
                ),
            });
        }
        for (u, &tok) in hyp.iter().enumerate() {
            if tok >= vocab {
                return Err(AudioError::ShapeMismatch {
                    msg: format!("hyp[{u}] = {tok} is out of range for vocab {vocab}"),
                });
            }
        }

        // CTC marginal log-likelihood (also validates ctc_logprobs / t / vocab).
        let ctc_score = ctc_forward_log(ctc_logprobs, t, vocab, hyp, self.blank)?;

        // Attention: sum of per-step log-probabilities for the hypothesis tokens.
        let mut attn_score = 0.0_f32;
        for (u, &tok) in hyp.iter().enumerate() {
            attn_score += attn_logprobs[u * vocab + tok];
        }

        Ok(self.lambda * ctc_score + (1.0 - self.lambda) * attn_score)
    }

    /// Frame-synchronous greedy *one-best* joint decode.
    ///
    /// Both `ctc_logprobs` and `attn_logprobs` are row-major `[t, vocab]`. At
    /// each frame the joint per-symbol log-probability
    /// `λ·ctc[t,v] + (1−λ)·attn[t,v]` is maximised; the resulting frame-level
    /// path is reduced by CTC collapse (merge consecutive duplicates, drop
    /// blank) to yield the decoded label sequence.
    ///
    /// # Errors
    /// - [`AudioError::InvalidVocabSize`] if `vocab == 0`.
    /// - [`AudioError::InvalidSequenceLength`] if `t == 0`.
    /// - [`AudioError::BlankOutOfRange`] if `blank ≥ vocab`.
    /// - [`AudioError::ShapeMismatch`] if either stream's length is not
    ///   `t · vocab`.
    pub fn decode(
        &self,
        ctc_logprobs: &[f32],
        attn_logprobs: &[f32],
        t: usize,
        vocab: usize,
    ) -> AudioResult<Vec<usize>> {
        if vocab == 0 {
            return Err(AudioError::InvalidVocabSize(vocab));
        }
        if t == 0 {
            return Err(AudioError::InvalidSequenceLength(t));
        }
        if self.blank >= vocab {
            return Err(AudioError::BlankOutOfRange {
                blank: self.blank,
                vocab,
            });
        }
        if ctc_logprobs.len() != t * vocab {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "ctc_logprobs length {} does not match t({}) * vocab({})",
                    ctc_logprobs.len(),
                    t,
                    vocab
                ),
            });
        }
        if attn_logprobs.len() != t * vocab {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "attn_logprobs length {} does not match t({}) * vocab({})",
                    attn_logprobs.len(),
                    t,
                    vocab
                ),
            });
        }

        let mut path: Vec<usize> = Vec::with_capacity(t);
        for ts in 0..t {
            let ctc_row = &ctc_logprobs[ts * vocab..(ts + 1) * vocab];
            let attn_row = &attn_logprobs[ts * vocab..(ts + 1) * vocab];
            let mut best_v = 0_usize;
            let mut best = f32::NEG_INFINITY;
            for (v, (&c, &a)) in ctc_row.iter().zip(attn_row.iter()).enumerate() {
                let joint = self.lambda * c + (1.0 - self.lambda) * a;
                if joint > best {
                    best = joint;
                    best_v = v;
                }
            }
            path.push(best_v);
        }

        Ok(collapse_ctc(&path, self.blank))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Standard CTC collapse: merge consecutive duplicate symbols, then drop blanks.
fn collapse_ctc(path: &[usize], blank: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut prev: Option<usize> = None;
    for &p in path {
        if Some(p) != prev && p != blank {
            out.push(p);
        }
        prev = Some(p);
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[t, vocab]` log-prob matrix where frame `ts` peaks strongly at
    /// `argmax_path[ts]`. The rows are valid log-softmax values.
    fn peaked(t: usize, vocab: usize, argmax_path: &[usize]) -> Vec<f32> {
        let mut lp = vec![0.0_f32; t * vocab];
        for ts in 0..t {
            let peak = argmax_path[ts % argmax_path.len()];
            let row = &mut lp[ts * vocab..(ts + 1) * vocab];
            for (i, v) in row.iter_mut().enumerate() {
                *v = if i == peak { 12.0 } else { 0.0 };
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            for v in row.iter_mut() {
                *v = (*v - max) - sum.ln();
            }
        }
        lp
    }

    /// Reference greedy CTC decode: per-frame argmax then CTC collapse.
    fn greedy_ctc(ctc: &[f32], t: usize, vocab: usize, blank: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(t);
        for ts in 0..t {
            let row = &ctc[ts * vocab..(ts + 1) * vocab];
            let mut best = 0;
            let mut bv = f32::NEG_INFINITY;
            for (i, &x) in row.iter().enumerate() {
                if x > bv {
                    bv = x;
                    best = i;
                }
            }
            path.push(best);
        }
        collapse_ctc(&path, blank)
    }

    #[test]
    fn lambda_one_depends_only_on_ctc() {
        // blank=0, vocab=4. CTC path: 1,1,0,2,2  → collapse → [1,2].
        let t = 5;
        let vocab = 4;
        let blank = 0;
        let ctc = peaked(t, vocab, &[1, 1, 0, 2, 2]);
        // Two very different attention streams must not change the λ=1 result.
        let attn_a = peaked(t, vocab, &[3, 3, 3, 3, 3]);
        let attn_b = peaked(t, vocab, &[2, 1, 0, 3, 1]);
        let jca = JointCtcAttention::new(1.0, blank).expect("new");
        let out_a = jca.decode(&ctc, &attn_a, t, vocab).expect("decode");
        let out_b = jca.decode(&ctc, &attn_b, t, vocab).expect("decode");
        let reference = greedy_ctc(&ctc, t, vocab, blank);
        assert_eq!(out_a, reference, "λ=1 must match CTC-only decode");
        assert_eq!(out_a, out_b, "λ=1 must ignore the attention stream");
        assert_eq!(out_a, vec![1, 2]);
    }

    #[test]
    fn lambda_zero_depends_only_on_attention() {
        // λ=0 → result determined solely by the attention stream.
        let t = 5;
        let vocab = 4;
        let blank = 0;
        let attn = peaked(t, vocab, &[2, 0, 3, 3, 1]); // collapse → [2,3,1]
        let ctc_a = peaked(t, vocab, &[1, 1, 1, 1, 1]);
        let ctc_b = peaked(t, vocab, &[3, 2, 1, 0, 2]);
        let jca = JointCtcAttention::new(0.0, blank).expect("new");
        let out_a = jca.decode(&ctc_a, &attn, t, vocab).expect("decode");
        let out_b = jca.decode(&ctc_b, &attn, t, vocab).expect("decode");
        assert_eq!(out_a, out_b, "λ=0 must ignore the CTC stream");
        assert_eq!(out_a, vec![2, 3, 1]);
    }

    #[test]
    fn agreeing_streams_decode_to_shared_sequence() {
        let t = 6;
        let vocab = 5;
        let blank = 0;
        let path = [1, 0, 2, 2, 0, 3]; // collapse → [1,2,3]
        let ctc = peaked(t, vocab, &path);
        let attn = peaked(t, vocab, &path);
        for &lam in &[0.0_f32, 0.3, 0.5, 0.7, 1.0] {
            let jca = JointCtcAttention::new(lam, blank).expect("new");
            let out = jca.decode(&ctc, &attn, t, vocab).expect("decode");
            assert_eq!(out, vec![1, 2, 3], "λ={lam} should recover the shared seq");
        }
    }

    #[test]
    fn invalid_lambda_errors() {
        assert!(matches!(
            JointCtcAttention::new(-0.1, 0).unwrap_err(),
            AudioError::Internal(_)
        ));
        assert!(matches!(
            JointCtcAttention::new(1.5, 0).unwrap_err(),
            AudioError::Internal(_)
        ));
        assert!(matches!(
            JointCtcAttention::new(f32::NAN, 0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn blank_out_of_range_errors() {
        let t = 3;
        let vocab = 3;
        let ctc = peaked(t, vocab, &[1, 2, 1]);
        let attn = peaked(t, vocab, &[1, 2, 1]);
        let jca = JointCtcAttention::new(0.5, 5).expect("new"); // blank=5 ≥ vocab=3
        assert!(matches!(
            jca.decode(&ctc, &attn, t, vocab).unwrap_err(),
            AudioError::BlankOutOfRange { blank: 5, vocab: 3 }
        ));
    }

    #[test]
    fn shape_mismatch_errors() {
        let t = 4;
        let vocab = 3;
        let ctc = peaked(t, vocab, &[1, 2, 1, 2]);
        let attn = vec![0.0_f32; (t - 1) * vocab]; // wrong length
        let jca = JointCtcAttention::new(0.5, 0).expect("new");
        assert!(matches!(
            jca.decode(&ctc, &attn, t, vocab).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn empty_sequence_errors() {
        let jca = JointCtcAttention::new(0.5, 0).expect("new");
        assert!(matches!(
            jca.decode(&[], &[], 0, 3).unwrap_err(),
            AudioError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn joint_score_lambda_extremes() {
        // λ=1 → joint_score == CTC forward score; λ=0 → attention sum.
        let t = 4;
        let vocab = 4;
        let blank = 0;
        let ctc = peaked(t, vocab, &[1, 0, 2, 0]);
        let hyp = [1_usize, 2];
        // Attention rows: one per hypothesis token.
        let attn = peaked(hyp.len(), vocab, &[1, 2]);

        let ctc_only = JointCtcAttention::new(1.0, blank).expect("new");
        let s_ctc = ctc_only
            .joint_score(&ctc, t, vocab, &attn, hyp.len(), &hyp)
            .expect("score");
        let expected_ctc = ctc_forward_log(&ctc, t, vocab, &hyp, blank).expect("fwd");
        assert!(
            (s_ctc - expected_ctc).abs() < 1e-5,
            "{s_ctc} vs {expected_ctc}"
        );

        let attn_only = JointCtcAttention::new(0.0, blank).expect("new");
        let s_attn = attn_only
            .joint_score(&ctc, t, vocab, &attn, hyp.len(), &hyp)
            .expect("score");
        let expected_attn = attn[1] + attn[vocab + 2];
        assert!(
            (s_attn - expected_attn).abs() < 1e-5,
            "{s_attn} vs {expected_attn}"
        );
    }

    #[test]
    fn joint_score_interpolates() {
        let t = 3;
        let vocab = 4;
        let blank = 0;
        let ctc = peaked(t, vocab, &[1, 2, 0]);
        let hyp = [1_usize, 2];
        let attn = peaked(hyp.len(), vocab, &[1, 2]);
        let ctc_term = ctc_forward_log(&ctc, t, vocab, &hyp, blank).expect("fwd");
        let attn_term = attn[1] + attn[vocab + 2];
        let lam = 0.4_f32;
        let jca = JointCtcAttention::new(lam, blank).expect("new");
        let s = jca
            .joint_score(&ctc, t, vocab, &attn, hyp.len(), &hyp)
            .expect("score");
        let expected = lam * ctc_term + (1.0 - lam) * attn_term;
        assert!((s - expected).abs() < 1e-5, "{s} vs {expected}");
    }

    #[test]
    fn joint_score_too_few_attn_rows_errors() {
        let t = 3;
        let vocab = 4;
        let ctc = peaked(t, vocab, &[1, 2, 0]);
        let hyp = [1_usize, 2, 3];
        let attn = peaked(2, vocab, &[1, 2]); // only 2 rows for 3-token hyp
        let jca = JointCtcAttention::new(0.5, 0).expect("new");
        assert!(matches!(
            jca.joint_score(&ctc, t, vocab, &attn, 2, &hyp).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn collapse_blank_only_is_empty() {
        assert!(collapse_ctc(&[0, 0, 0], 0).is_empty());
    }

    #[test]
    fn collapse_merges_repeats_and_keeps_blank_separated() {
        // 1 1 _ 1 2 2 → [1, 1, 2]  (blank separates the repeated 1's).
        assert_eq!(collapse_ctc(&[1, 1, 0, 1, 2, 2], 0), vec![1, 1, 2]);
    }
}
