//! Beam-search lattice rescoring with a shallow-fusion language model.
//!
//! This module **re-ranks an existing set of ASR hypotheses** (an n-best list
//! or a decoded lattice) using an external language model via *shallow fusion*.
//! It is deliberately **distinct from CTC prefix beam search** in
//! [`crate::ctc::beam_search`]: that module *produces* hypotheses from a
//! per-frame acoustic posterior matrix, whereas this module *consumes* an
//! already-decoded hypothesis set (token sequences carrying acoustic
//! log-probabilities) and rescores it with a separately-provided LM.
//!
//! ## Shallow fusion
//!
//! Each hypothesis receives a combined score
//!
//! ```text
//! total = acoustic_score
//!       + lm_weight            · lm_score(tokens)
//!       + word_insertion_penalty · len(tokens)
//! ```
//!
//! where `lm_score(tokens)` is the LM log-probability of the token sequence,
//! supplied by a caller-provided closure `Fn(&[usize]) -> f32`. A positive
//! `word_insertion_penalty` rewards longer sequences (counteracting the LM's
//! natural bias toward shorter strings); a zero `lm_weight` recovers a pure
//! acoustic ranking (plus the length term).
//!
//! ## Operations
//!
//! - [`LatticeRescorer::rescore`] scores every hypothesis and returns them
//!   sorted by descending total score.
//! - [`LatticeRescorer::best`] returns the single highest-scoring hypothesis.
//! - [`LatticeRescorer::beam_expand`] performs a left-to-right prefix-beam
//!   search over a lattice given as per-step `(token, acoustic_logprob)`
//!   candidates, applying the LM incrementally and pruning to `beam_size`.
//!
//! All operations are deterministic for a fixed scorer closure: ties are broken
//! by the lexicographically smallest token sequence, so equal-scoring
//! hypotheses always appear in a stable, reproducible order.

use std::cmp::Ordering;

use crate::error::{AudioError, AudioResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`LatticeRescorer`].
#[derive(Debug, Clone, PartialEq)]
pub struct RescoreConfig {
    /// Shallow-fusion weight applied to the LM log-probability.
    pub lm_weight: f32,
    /// Per-token insertion bonus/penalty added to the total score.
    pub word_insertion_penalty: f32,
    /// Maximum number of prefixes retained during [`LatticeRescorer::beam_expand`].
    pub beam_size: usize,
}

impl RescoreConfig {
    /// A neutral default: LM weight `0.5`, no insertion penalty, beam size `8`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            lm_weight: 0.5,
            word_insertion_penalty: 0.0,
            beam_size: 8,
        }
    }
}

// ─── Hypothesis types ─────────────────────────────────────────────────────────

/// An input hypothesis: a token sequence with its acoustic log-probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Decoded token indices.
    pub tokens: Vec<usize>,
    /// Acoustic model log-probability of this sequence.
    pub acoustic_score: f32,
}

impl Hypothesis {
    /// Convenience constructor.
    #[must_use]
    pub fn new(tokens: Vec<usize>, acoustic_score: f32) -> Self {
        Self {
            tokens,
            acoustic_score,
        }
    }
}

/// A rescored hypothesis: the original tokens and acoustic score, the LM score,
/// and the fused total.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredHypothesis {
    /// Decoded token indices.
    pub tokens: Vec<usize>,
    /// Acoustic model log-probability (carried through unchanged).
    pub acoustic_score: f32,
    /// LM log-probability returned by the scorer closure.
    pub lm_score: f32,
    /// Shallow-fusion total: `acoustic + lm_weight·lm + wip·len`.
    pub total_score: f32,
}

// ─── Comparison ───────────────────────────────────────────────────────────────

/// Order two scored hypotheses for a descending sort with a stable tie-break.
///
/// Primary key: descending `total_score`. Tie-break: ascending lexicographic
/// token order, so equal-scoring hypotheses come out deterministically. Any
/// non-comparable (NaN) total sinks below comparable ones.
fn cmp_scored(a: &ScoredHypothesis, b: &ScoredHypothesis) -> Ordering {
    match b.total_score.partial_cmp(&a.total_score) {
        Some(Ordering::Equal) | None => a.tokens.cmp(&b.tokens),
        Some(other) => other,
    }
}

// ─── Rescorer ─────────────────────────────────────────────────────────────────

/// Shallow-fusion lattice / n-best rescorer.
#[derive(Debug, Clone)]
pub struct LatticeRescorer {
    cfg: RescoreConfig,
}

impl LatticeRescorer {
    /// Construct a new rescorer from configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidBeamWidth`] if `beam_size == 0`.
    /// - [`AudioError::NonFinite`] if `lm_weight` is not finite.
    pub fn new(cfg: RescoreConfig) -> AudioResult<Self> {
        if cfg.beam_size == 0 {
            return Err(AudioError::InvalidBeamWidth(cfg.beam_size));
        }
        if !cfg.lm_weight.is_finite() {
            return Err(AudioError::NonFinite {
                msg: "lm_weight must be finite".to_string(),
            });
        }
        Ok(Self { cfg })
    }

    /// Borrow the configuration this rescorer was built with.
    #[must_use]
    pub fn config(&self) -> &RescoreConfig {
        &self.cfg
    }

    /// Compute the shallow-fusion total for a single (acoustic, lm, len) triple.
    #[inline]
    fn fuse(&self, acoustic: f32, lm: f32, len: usize) -> f32 {
        acoustic + self.cfg.lm_weight * lm + self.cfg.word_insertion_penalty * len as f32
    }

    /// Score one hypothesis with the supplied LM scorer.
    fn score_one<L>(&self, hyp: &Hypothesis, lm_scorer: &L) -> ScoredHypothesis
    where
        L: Fn(&[usize]) -> f32,
    {
        let lm_score = lm_scorer(&hyp.tokens);
        let total_score = self.fuse(hyp.acoustic_score, lm_score, hyp.tokens.len());
        ScoredHypothesis {
            tokens: hyp.tokens.clone(),
            acoustic_score: hyp.acoustic_score,
            lm_score,
            total_score,
        }
    }

    /// Rescore an n-best / lattice hypothesis set via shallow fusion.
    ///
    /// Returns one [`ScoredHypothesis`] per input, sorted by descending
    /// `total_score` (ties broken lexicographically by token sequence).
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `hypotheses` is empty.
    pub fn rescore<L>(
        &self,
        hypotheses: &[Hypothesis],
        lm_scorer: L,
    ) -> AudioResult<Vec<ScoredHypothesis>>
    where
        L: Fn(&[usize]) -> f32,
    {
        if hypotheses.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "hypotheses is empty".to_string(),
            });
        }
        let mut scored: Vec<ScoredHypothesis> = hypotheses
            .iter()
            .map(|h| self.score_one(h, &lm_scorer))
            .collect();
        scored.sort_by(cmp_scored);
        Ok(scored)
    }

    /// Return the single best hypothesis after rescoring (max `total_score`).
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `hypotheses` is empty.
    pub fn best<L>(&self, hypotheses: &[Hypothesis], lm_scorer: L) -> AudioResult<ScoredHypothesis>
    where
        L: Fn(&[usize]) -> f32,
    {
        if hypotheses.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "hypotheses is empty".to_string(),
            });
        }
        let mut best: Option<ScoredHypothesis> = None;
        for hyp in hypotheses {
            let cand = self.score_one(hyp, &lm_scorer);
            best = Some(match best {
                None => cand,
                // `cmp_scored` orders best-first, so keep `current` only when it
                // already sorts before `cand`.
                Some(current) => {
                    if cmp_scored(&current, &cand) == Ordering::Less {
                        current
                    } else {
                        cand
                    }
                }
            });
        }
        // `best` is always `Some` because `hypotheses` is non-empty.
        best.ok_or_else(|| AudioError::Internal("empty hypothesis set after scoring".to_string()))
    }

    /// Prefix-beam expansion over a lattice given as per-step candidates.
    ///
    /// `arcs[t]` lists the `(token, acoustic_logprob)` choices available at step
    /// `t`. The search runs left-to-right: every surviving prefix is extended by
    /// every candidate token, the acoustic score is accumulated, the LM score of
    /// the *extended* prefix is recomputed via `lm_scorer`, and the beam is
    /// pruned to the top `beam_size` prefixes by current total score (stable
    /// lexicographic tie-break). The returned beam is sorted descending.
    ///
    /// Every returned hypothesis has exactly `arcs.len()` tokens.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `arcs` is empty or any step is empty.
    pub fn beam_expand<L>(
        &self,
        arcs: &[Vec<(usize, f32)>],
        lm_scorer: L,
    ) -> AudioResult<Vec<ScoredHypothesis>>
    where
        L: Fn(&[usize]) -> f32,
    {
        if arcs.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "arcs is empty".to_string(),
            });
        }
        for (step, candidates) in arcs.iter().enumerate() {
            if candidates.is_empty() {
                return Err(AudioError::EmptyInput {
                    msg: format!("arc step {step} has no candidates"),
                });
            }
        }

        // Beam entries carry the running acoustic sum alongside the scored view.
        struct BeamEntry {
            tokens: Vec<usize>,
            acoustic: f32,
            lm: f32,
            total: f32,
        }

        // Seed with a single empty prefix (zero acoustic, LM of the empty seq).
        let seed_lm = lm_scorer(&[]);
        let mut beam: Vec<BeamEntry> = vec![BeamEntry {
            tokens: Vec::new(),
            acoustic: 0.0,
            lm: seed_lm,
            total: self.fuse(0.0, seed_lm, 0),
        }];

        for candidates in arcs {
            let mut next: Vec<BeamEntry> = Vec::with_capacity(beam.len() * candidates.len());
            for entry in &beam {
                for &(token, acoustic_logprob) in candidates {
                    let mut tokens = Vec::with_capacity(entry.tokens.len() + 1);
                    tokens.extend_from_slice(&entry.tokens);
                    tokens.push(token);
                    let acoustic = entry.acoustic + acoustic_logprob;
                    let lm = lm_scorer(&tokens);
                    let total = self.fuse(acoustic, lm, tokens.len());
                    next.push(BeamEntry {
                        tokens,
                        acoustic,
                        lm,
                        total,
                    });
                }
            }

            // Prune to top `beam_size` by total (descending), lexicographic tie-break.
            next.sort_by(|a, b| match b.total.partial_cmp(&a.total) {
                Some(Ordering::Equal) | None => a.tokens.cmp(&b.tokens),
                Some(other) => other,
            });
            next.truncate(self.cfg.beam_size);
            beam = next;
        }

        let mut scored: Vec<ScoredHypothesis> = beam
            .into_iter()
            .map(|e| ScoredHypothesis {
                tokens: e.tokens,
                acoustic_score: e.acoustic,
                lm_score: e.lm,
                total_score: e.total,
            })
            .collect();
        scored.sort_by(cmp_scored);
        Ok(scored)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rescorer(lm_weight: f32, wip: f32, beam_size: usize) -> LatticeRescorer {
        LatticeRescorer::new(RescoreConfig {
            lm_weight,
            word_insertion_penalty: wip,
            beam_size,
        })
        .expect("valid config")
    }

    fn sample_hyps() -> Vec<Hypothesis> {
        vec![
            Hypothesis::new(vec![1, 2], -3.0),
            Hypothesis::new(vec![1, 3, 4], -2.0),
            Hypothesis::new(vec![5], -5.0),
        ]
    }

    /// LM that prefers a specific target sequence (high log-prob) and disfavours
    /// everything else.
    fn favouring_lm(target: Vec<usize>) -> impl Fn(&[usize]) -> f32 {
        move |tokens: &[usize]| {
            if tokens == target.as_slice() {
                0.0
            } else {
                -100.0
            }
        }
    }

    // ── construction / validation ────────────────────────────────────────────

    #[test]
    fn new_valid_ok() {
        assert!(LatticeRescorer::new(RescoreConfig::tiny()).is_ok());
    }

    #[test]
    fn new_beam_size_zero_err() {
        let mut cfg = RescoreConfig::tiny();
        cfg.beam_size = 0;
        assert_eq!(
            LatticeRescorer::new(cfg).unwrap_err(),
            AudioError::InvalidBeamWidth(0)
        );
    }

    #[test]
    fn new_non_finite_lm_weight_err() {
        let mut cfg = RescoreConfig::tiny();
        cfg.lm_weight = f32::NAN;
        assert!(matches!(
            LatticeRescorer::new(cfg).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
        let mut cfg2 = RescoreConfig::tiny();
        cfg2.lm_weight = f32::INFINITY;
        assert!(matches!(
            LatticeRescorer::new(cfg2).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
    }

    // ── rescore ────────────────────────────────────────────────────────────────

    #[test]
    fn rescore_same_count_and_sorted() {
        let r = rescorer(0.5, 0.0, 8);
        let hyps = sample_hyps();
        let scored = r.rescore(&hyps, |_| -1.0).expect("ok");
        assert_eq!(scored.len(), hyps.len());
        for pair in scored.windows(2) {
            assert!(
                pair[0].total_score >= pair[1].total_score,
                "not sorted descending: {} < {}",
                pair[0].total_score,
                pair[1].total_score
            );
        }
    }

    #[test]
    fn rescore_total_formula_hand_check() {
        // lm_weight = 2.0, wip = 0.5. Hypothesis [1,2] with acoustic -3.0 and a
        // constant LM closure returning -1.5:
        // total = -3.0 + 2.0*(-1.5) + 0.5*2 = -3.0 - 3.0 + 1.0 = -5.0.
        let r = rescorer(2.0, 0.5, 8);
        let hyps = vec![Hypothesis::new(vec![1, 2], -3.0)];
        let scored = r.rescore(&hyps, |_| -1.5).expect("ok");
        assert!((scored[0].lm_score - (-1.5)).abs() < 1e-6);
        assert!(
            (scored[0].total_score - (-5.0)).abs() < 1e-5,
            "total = {}",
            scored[0].total_score
        );
    }

    #[test]
    fn rescore_lm_weight_zero_ranks_by_acoustic_plus_wip() {
        // With lm_weight = 0 the LM cannot affect ranking; only acoustic + wip*len.
        let r = rescorer(0.0, 0.0, 8);
        let hyps = sample_hyps();
        // A wild LM closure — must be ignored for ranking with lm_weight = 0.
        let scored = r
            .rescore(&hyps, |t| if t == [5] { 1000.0 } else { -1000.0 })
            .expect("ok");
        // Best acoustic is [1,3,4] at -2.0; it must rank first.
        assert_eq!(scored[0].tokens, vec![1, 3, 4]);
        for s in &scored {
            assert!((s.total_score - s.acoustic_score).abs() < 1e-6);
        }
    }

    #[test]
    fn rescore_single_hypothesis_scored_as_is() {
        let r = rescorer(1.0, 0.0, 4);
        let hyps = vec![Hypothesis::new(vec![7, 8, 9], -1.25)];
        let scored = r.rescore(&hyps, |_| -0.5).expect("ok");
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].tokens, vec![7, 8, 9]);
        assert!((scored[0].acoustic_score - (-1.25)).abs() < 1e-6);
        assert!((scored[0].total_score - (-1.75)).abs() < 1e-5);
    }

    #[test]
    fn rescore_empty_err() {
        let r = rescorer(0.5, 0.0, 8);
        assert!(matches!(
            r.rescore(&[], |_| 0.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn rescore_strong_lm_overrides_acoustic() {
        // [5] has the *worst* acoustic (-5.0). A strong LM that favours [5]
        // with a large lm_weight should still push it to the top.
        let r = rescorer(10.0, 0.0, 8);
        let hyps = sample_hyps();
        let scored = r.rescore(&hyps, favouring_lm(vec![5])).expect("ok");
        assert_eq!(
            scored[0].tokens,
            vec![5],
            "strong LM should override acoustic ranking"
        );
    }

    #[test]
    fn rescore_wip_favours_longer_sequences() {
        // Equal acoustic, equal LM; a positive WIP must rank the longer one first.
        let r = rescorer(1.0, 5.0, 8);
        let hyps = vec![
            Hypothesis::new(vec![1], -2.0),
            Hypothesis::new(vec![1, 2, 3], -2.0),
        ];
        let scored = r.rescore(&hyps, |_| 0.0).expect("ok");
        assert_eq!(
            scored[0].tokens,
            vec![1, 2, 3],
            "positive WIP should favour the longer hypothesis"
        );
    }

    #[test]
    fn rescore_deterministic() {
        let r = rescorer(0.7, 0.1, 8);
        let hyps = sample_hyps();
        let a = r.rescore(&hyps, |t| -(t.len() as f32)).expect("ok");
        let b = r.rescore(&hyps, |t| -(t.len() as f32)).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn rescore_tie_break_stable_lexicographic() {
        // Construct an exact tie in total_score (lm_weight = 0, wip = 0, equal
        // acoustic). The lexicographically smaller token sequence must come first.
        let r = rescorer(0.0, 0.0, 8);
        let hyps = vec![
            Hypothesis::new(vec![2, 0], -1.0),
            Hypothesis::new(vec![1, 9], -1.0),
            Hypothesis::new(vec![1, 5], -1.0),
        ];
        let scored = r.rescore(&hyps, |_| 0.0).expect("ok");
        let order: Vec<&Vec<usize>> = scored.iter().map(|s| &s.tokens).collect();
        assert_eq!(order, vec![&vec![1, 5], &vec![1, 9], &vec![2, 0]]);
    }

    // ── best ─────────────────────────────────────────────────────────────────

    #[test]
    fn best_matches_max_total() {
        let r = rescorer(1.0, 0.0, 8);
        let hyps = sample_hyps();
        let scored = r.rescore(&hyps, |t| -(t.len() as f32)).expect("ok");
        let best = r.best(&hyps, |t| -(t.len() as f32)).expect("ok");
        assert_eq!(best, scored[0]);
    }

    #[test]
    fn best_empty_err() {
        let r = rescorer(0.5, 0.0, 8);
        assert!(matches!(
            r.best(&[], |_| 0.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn best_strong_lm_picks_favoured() {
        let r = rescorer(10.0, 0.0, 8);
        let hyps = sample_hyps();
        let best = r.best(&hyps, favouring_lm(vec![5])).expect("ok");
        assert_eq!(best.tokens, vec![5]);
    }

    // ── beam_expand ────────────────────────────────────────────────────────────

    #[test]
    fn beam_expand_respects_beam_size() {
        let r = rescorer(0.0, 0.0, 2);
        let arcs = vec![
            vec![(1, -0.1), (2, -0.2), (3, -0.3)],
            vec![(4, -0.1), (5, -0.2), (6, -0.3)],
        ];
        let out = r.beam_expand(&arcs, |_| 0.0).expect("ok");
        assert!(out.len() <= 2, "beam should be capped at beam_size");
    }

    #[test]
    fn beam_expand_length_matches_arcs() {
        let r = rescorer(0.0, 0.0, 4);
        let arcs = vec![
            vec![(1, -0.1), (2, -0.5)],
            vec![(3, -0.2), (4, -0.4)],
            vec![(5, -0.3), (6, -0.6)],
        ];
        let out = r.beam_expand(&arcs, |_| 0.0).expect("ok");
        assert!(!out.is_empty());
        for hyp in &out {
            assert_eq!(
                hyp.tokens.len(),
                arcs.len(),
                "each hypothesis must span all arc steps"
            );
        }
    }

    #[test]
    fn beam_expand_greedy_when_beam_one() {
        // beam_size = 1, lm_weight = 0 → pure greedy: pick the max-acoustic token
        // at each step. Steps prefer tokens 1 and 4 respectively.
        let r = rescorer(0.0, 0.0, 1);
        let arcs = vec![vec![(1, -0.1), (2, -0.9)], vec![(4, -0.05), (5, -0.8)]];
        let out = r.beam_expand(&arcs, |_| 0.0).expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tokens, vec![1, 4]);
        // Greedy acoustic sum = -0.1 + -0.05 = -0.15.
        assert!((out[0].acoustic_score - (-0.15)).abs() < 1e-5);
    }

    #[test]
    fn beam_expand_lm_steers_path() {
        // Acoustic alone would prefer token 1 then 4. A strong LM favouring the
        // full sequence [2, 5] should override the greedy acoustic choice when
        // the beam is wide enough to retain the lower-acoustic prefix.
        let r = rescorer(50.0, 0.0, 4);
        let arcs = vec![vec![(1, -0.1), (2, -0.9)], vec![(4, -0.1), (5, -0.9)]];
        let out = r.beam_expand(&arcs, favouring_lm(vec![2, 5])).expect("ok");
        assert_eq!(out[0].tokens, vec![2, 5], "LM should steer the best path");
    }

    #[test]
    fn beam_expand_sorted_descending() {
        let r = rescorer(1.0, 0.0, 8);
        let arcs = vec![
            vec![(1, -0.1), (2, -0.2), (3, -0.3)],
            vec![(4, -0.1), (5, -0.2)],
        ];
        let out = r
            .beam_expand(&arcs, |t| -(t.iter().sum::<usize>() as f32))
            .expect("ok");
        for pair in out.windows(2) {
            assert!(
                pair[0].total_score >= pair[1].total_score,
                "beam output not sorted descending"
            );
        }
    }

    #[test]
    fn beam_expand_total_formula_consistent() {
        // Verify the reported total equals acoustic + lm_weight*lm + wip*len for
        // each returned hypothesis under a known closure.
        let r = rescorer(2.0, 0.5, 8);
        let arcs = vec![vec![(1, -0.3), (2, -0.7)], vec![(3, -0.2), (4, -0.6)]];
        let lm = |t: &[usize]| -(t.len() as f32);
        let out = r.beam_expand(&arcs, lm).expect("ok");
        for hyp in &out {
            let expected = hyp.acoustic_score + 2.0 * hyp.lm_score + 0.5 * hyp.tokens.len() as f32;
            assert!(
                (hyp.total_score - expected).abs() < 1e-5,
                "total {} != expected {}",
                hyp.total_score,
                expected
            );
            // LM score must equal the closure evaluated on the final tokens.
            assert!((hyp.lm_score - lm(&hyp.tokens)).abs() < 1e-6);
        }
    }

    #[test]
    fn beam_expand_deterministic() {
        let r = rescorer(0.5, 0.0, 3);
        let arcs = vec![
            vec![(1, -0.1), (2, -0.2), (3, -0.15)],
            vec![(4, -0.1), (5, -0.3)],
        ];
        let lm = |t: &[usize]| -(t.iter().map(|&x| x as f32).sum::<f32>());
        let a = r.beam_expand(&arcs, lm).expect("ok");
        let b = r.beam_expand(&arcs, lm).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn beam_expand_empty_arcs_err() {
        let r = rescorer(0.5, 0.0, 8);
        let arcs: Vec<Vec<(usize, f32)>> = Vec::new();
        assert!(matches!(
            r.beam_expand(&arcs, |_| 0.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn beam_expand_empty_step_err() {
        let r = rescorer(0.5, 0.0, 8);
        let arcs = vec![vec![(1, -0.1)], Vec::new(), vec![(2, -0.2)]];
        assert!(matches!(
            r.beam_expand(&arcs, |_| 0.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn beam_expand_single_step_single_candidate() {
        let r = rescorer(1.0, 0.0, 4);
        let arcs = vec![vec![(42, -0.5)]];
        let out = r.beam_expand(&arcs, |_| -1.0).expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tokens, vec![42]);
        // total = -0.5 + 1.0*(-1.0) + 0 = -1.5.
        assert!((out[0].total_score - (-1.5)).abs() < 1e-5);
    }

    #[test]
    fn beam_expand_tie_break_stable() {
        // Two equal-acoustic candidates with a neutral LM at a single step:
        // the lexicographically smaller token must come first.
        let r = rescorer(0.0, 0.0, 8);
        let arcs = vec![vec![(9, -1.0), (3, -1.0), (7, -1.0)]];
        let out = r.beam_expand(&arcs, |_| 0.0).expect("ok");
        let order: Vec<usize> = out.iter().map(|h| h.tokens[0]).collect();
        assert_eq!(order, vec![3, 7, 9]);
    }
}
