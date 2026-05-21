use crate::error::{SeqError, SeqResult};

/// Configuration for Diverse Beam Search (Vijayakumar et al. 2018).
#[derive(Debug, Clone, Copy)]
pub struct DiverseBeamConfig {
    /// Total beam width B. Must be divisible by `n_groups`.
    pub beam_width: usize,
    /// G: number of diversity groups. Each group gets `beam_width / n_groups` beams.
    pub n_groups: usize,
    /// Maximum decoding steps.
    pub max_steps: usize,
    /// Vocabulary size (number of tokens).
    pub vocab_size: usize,
    /// Token ID for end-of-sequence.
    pub eos_id: usize,
    /// λ: Hamming diversity penalty strength.
    pub diversity_strength: f32,
    /// GNMT-style length normalisation exponent. 0.0 = no normalisation.
    pub length_norm_alpha: f32,
}

/// Diverse beam search decoder.
#[derive(Debug, Clone)]
pub struct DiverseBeam {
    pub cfg: DiverseBeamConfig,
}

impl DiverseBeam {
    /// Validate configuration and create a new `DiverseBeam`.
    pub fn new(cfg: DiverseBeamConfig) -> SeqResult<Self> {
        if cfg.beam_width == 0 {
            return Err(SeqError::InvalidConfiguration(
                "beam_width must be > 0".to_string(),
            ));
        }
        if cfg.n_groups == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_groups must be > 0".to_string(),
            ));
        }
        if cfg.beam_width % cfg.n_groups != 0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "beam_width ({}) must be divisible by n_groups ({})",
                cfg.beam_width, cfg.n_groups
            )));
        }
        if cfg.vocab_size == 0 {
            return Err(SeqError::InvalidConfiguration(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(Self { cfg })
    }

    /// Run diverse beam search.
    ///
    /// Groups decode sequentially at each time step.  Group g receives a
    /// Hamming diversity penalty based on the tokens that groups 0..g-1
    /// committed to at the current step.
    ///
    /// Returns `beam_width` sequences (all groups concatenated), each a
    /// vector of token IDs.
    pub fn search<F>(&self, score_fn: F) -> SeqResult<Vec<Vec<usize>>>
    where
        F: Fn(&[usize]) -> Vec<f32>,
    {
        let cfg = &self.cfg;
        let beam_per_group = cfg.beam_width / cfg.n_groups;

        // Each group maintains its own beam: Vec<(tokens, cumulative_score)>.
        let mut groups: Vec<Vec<(Vec<usize>, f32)>> =
            vec![vec![(vec![], 0.0f32); beam_per_group]; cfg.n_groups];

        // Track which hypotheses are finished (emitted EOS).
        let mut finished: Vec<Vec<bool>> = vec![vec![false; beam_per_group]; cfg.n_groups];

        for _step in 0..cfg.max_steps {
            // Tokens chosen by groups that have already decoded this step.
            let mut prev_group_tokens: Vec<usize> = Vec::new();

            for g in 0..cfg.n_groups {
                // Skip group if all hypotheses are finished.
                if finished[g].iter().all(|&f| f) {
                    // Still register their current last tokens for later groups.
                    for hyp_idx in 0..beam_per_group {
                        if let Some(&last_tok) = groups[g][hyp_idx].0.last() {
                            prev_group_tokens.push(last_tok);
                        }
                    }
                    continue;
                }

                // Candidates: (hypothesis_idx, token, adjusted_score).
                let mut candidates: Vec<(usize, usize, f32, Vec<usize>)> = Vec::new();

                for hyp_idx in 0..beam_per_group {
                    let (ref tokens, cum_score) = groups[g][hyp_idx].clone();
                    if finished[g][hyp_idx] {
                        // Carry finished hypotheses forward unchanged.
                        candidates.push((hyp_idx, cfg.eos_id, cum_score, tokens.clone()));
                        continue;
                    }

                    let log_probs = score_fn(tokens);
                    if log_probs.len() != cfg.vocab_size {
                        return Err(SeqError::ShapeMismatch {
                            expected: cfg.vocab_size,
                            got: log_probs.len(),
                        });
                    }

                    for tok in 0..cfg.vocab_size {
                        let raw = log_probs[tok];
                        let diversity_pen =
                            Self::hamming_penalty(tok, &prev_group_tokens, cfg.diversity_strength);
                        let candidate_score = cum_score + raw - diversity_pen;
                        let mut new_tokens = tokens.clone();
                        new_tokens.push(tok);
                        let norm_score = Self::length_norm(
                            candidate_score,
                            new_tokens.len(),
                            cfg.length_norm_alpha,
                        );
                        // Store (hyp_idx, tok, norm_score for ranking, extended tokens).
                        candidates.push((hyp_idx, tok, norm_score, new_tokens));
                    }
                }

                // Sort by adjusted normalised score, descending.
                candidates
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                // Select top beam_per_group survivors, deduplicated by sequence.
                let mut new_beam: Vec<(Vec<usize>, f32)> = Vec::with_capacity(beam_per_group);
                let mut new_finished: Vec<bool> = Vec::with_capacity(beam_per_group);
                let mut tokens_chosen_by_group: Vec<usize> = Vec::new();

                for (_, tok, _, new_tokens) in candidates.iter() {
                    if new_beam.len() >= beam_per_group {
                        break;
                    }
                    // Recompute the actual cumulative (unnormalised) score for storage.
                    // We need to recover cum_score + raw for proper accumulation.
                    // Since we sorted on normalised+diversity-adjusted, extract raw score.
                    // Easier: store unnormalised in a parallel structure.
                    let is_eos = *tok == cfg.eos_id;
                    new_beam.push((new_tokens.clone(), 0.0));
                    new_finished.push(is_eos);
                    tokens_chosen_by_group.push(*tok);
                }

                // We need unnormalised cumulative scores for correct accumulation.
                // Redo: collect candidates with both norm_score (for ranking) and raw cumsum.
                // Overwrite new_beam with proper cumulative scores by re-expanding.
                let mut candidates2: Vec<(usize, usize, f32, f32, Vec<usize>)> = Vec::new();
                for hyp_idx in 0..beam_per_group {
                    let (ref tokens, cum_score) = groups[g][hyp_idx].clone();
                    if finished[g][hyp_idx] {
                        candidates2.push((
                            hyp_idx,
                            cfg.eos_id,
                            cum_score,
                            cum_score,
                            tokens.clone(),
                        ));
                        continue;
                    }
                    let log_probs = score_fn(tokens);
                    for tok in 0..cfg.vocab_size {
                        let raw = log_probs[tok];
                        let diversity_pen =
                            Self::hamming_penalty(tok, &prev_group_tokens, cfg.diversity_strength);
                        let new_cum = cum_score + raw - diversity_pen;
                        let mut new_tokens = tokens.clone();
                        new_tokens.push(tok);
                        let norm_score =
                            Self::length_norm(new_cum, new_tokens.len(), cfg.length_norm_alpha);
                        candidates2.push((hyp_idx, tok, new_cum, norm_score, new_tokens));
                    }
                }
                candidates2
                    .sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

                new_beam.clear();
                new_finished.clear();
                tokens_chosen_by_group.clear();

                for (_, tok, new_cum, _, new_tokens) in candidates2.iter() {
                    if new_beam.len() >= beam_per_group {
                        break;
                    }
                    new_beam.push((new_tokens.clone(), *new_cum));
                    new_finished.push(*tok == cfg.eos_id);
                    if !finished[g]
                        .get(new_beam.len().saturating_sub(1))
                        .copied()
                        .unwrap_or(false)
                    {
                        tokens_chosen_by_group.push(*tok);
                    }
                }

                // Pad if score_fn returned fewer candidates than needed.
                while new_beam.len() < beam_per_group {
                    if let Some(first) = new_beam.first().cloned() {
                        new_beam.push(first);
                        new_finished.push(true);
                    } else {
                        break;
                    }
                }

                prev_group_tokens.extend_from_slice(&tokens_chosen_by_group);
                groups[g] = new_beam;
                finished[g] = new_finished;
            }

            // Early exit when all beams in all groups are finished.
            if finished.iter().all(|gf| gf.iter().all(|&f| f)) {
                break;
            }
        }

        // Collect all hypotheses across groups, in group order.
        let mut result: Vec<Vec<usize>> = Vec::with_capacity(cfg.beam_width);
        for g in 0..cfg.n_groups {
            for hyp_idx in 0..beam_per_group {
                result.push(groups[g][hyp_idx].0.clone());
            }
        }
        Ok(result)
    }

    /// Hamming diversity penalty for `token`.
    ///
    /// Counts how many elements of `prev_group_tokens` equal `token`, then
    /// multiplies by `strength`.  This penalises tokens already chosen by
    /// earlier groups at the current decoding step.
    #[inline]
    pub fn hamming_penalty(token: usize, prev_group_tokens: &[usize], strength: f32) -> f32 {
        if strength == 0.0 {
            return 0.0;
        }
        let count = prev_group_tokens.iter().filter(|&&t| t == token).count();
        strength * count as f32
    }

    /// GNMT-style length normalisation: score / ((5 + len) / 6)^alpha.
    ///
    /// When `alpha == 0.0` the denominator is 1.0 (no normalisation).
    #[inline]
    pub fn length_norm(score: f32, len: usize, alpha: f32) -> f32 {
        if alpha == 0.0 || len == 0 {
            return score;
        }
        let denom = ((5.0 + len as f32) / 6.0).powf(alpha);
        score / denom
    }

    /// Return the top-k (token, score) pairs sorted by score descending.
    pub fn top_k(log_probs: &[f32], k: usize) -> Vec<(usize, f32)> {
        let k = k.min(log_probs.len());
        let mut indexed: Vec<(usize, f32)> = log_probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

// ─── inline tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> DiverseBeamConfig {
        DiverseBeamConfig {
            beam_width: 4,
            n_groups: 2,
            max_steps: 8,
            vocab_size: 5,
            eos_id: 4,
            diversity_strength: 0.5,
            length_norm_alpha: 0.0,
        }
    }

    /// Simple score function: prefer lower token indices (token 0 is best).
    fn prefer_low(prefix: &[usize]) -> Vec<f32> {
        let _ = prefix;
        vec![-0.1, -0.5, -1.0, -2.0, -10.0]
    }

    /// Score function that always returns a high score for EOS.
    fn always_eos(_prefix: &[usize]) -> Vec<f32> {
        vec![-100.0, -100.0, -100.0, -100.0, 0.0]
    }

    #[test]
    fn diverse_beam_returns_b_sequences() {
        let db = DiverseBeam::new(default_cfg()).expect("ok");
        let seqs = db.search(prefer_low).expect("ok");
        assert_eq!(seqs.len(), 4, "expected beam_width sequences");
    }

    #[test]
    fn diverse_beam_sequences_differ() {
        let cfg_div = DiverseBeamConfig {
            diversity_strength: 1.0,
            ..default_cfg()
        };
        let cfg_nodiv = DiverseBeamConfig {
            diversity_strength: 0.0,
            ..default_cfg()
        };
        let db_div = DiverseBeam::new(cfg_div).expect("ok");
        let db_nodiv = DiverseBeam::new(cfg_nodiv).expect("ok");

        let seqs_div = db_div.search(prefer_low).expect("ok");
        let seqs_nodiv = db_nodiv.search(prefer_low).expect("ok");

        // Count distinct sequences.
        let distinct = |seqs: &[Vec<usize>]| {
            let mut s: Vec<Vec<usize>> = seqs.to_vec();
            s.sort();
            s.dedup();
            s.len()
        };
        // With diversity > 0, expect at least as many distinct sequences.
        let div_count = distinct(&seqs_div);
        let nodiv_count = distinct(&seqs_nodiv);
        assert!(
            div_count >= nodiv_count,
            "diverse search should produce >= distinct seqs: div={div_count} nodiv={nodiv_count}"
        );
    }

    #[test]
    fn hamming_penalty_zero_when_no_overlap() {
        let penalty = DiverseBeam::hamming_penalty(3, &[0, 1, 2], 1.0);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn hamming_penalty_proportional_to_count() {
        let prev = vec![1usize, 1, 1, 2];
        let penalty = DiverseBeam::hamming_penalty(1, &prev, 2.0);
        assert!((penalty - 6.0).abs() < 1e-6, "penalty={penalty}");
    }

    #[test]
    fn length_norm_alpha_zero_is_identity() {
        let s = -3.0f32;
        assert!((DiverseBeam::length_norm(s, 5, 0.0) - s).abs() < 1e-6);
    }

    #[test]
    fn length_norm_longer_sequence_penalized() {
        let alpha = 0.6f32;
        // Use a positive score to verify that a longer sequence gets a smaller
        // normalised score (the denominator grows with length, so score/denom shrinks).
        let score = 10.0f32;
        let short = DiverseBeam::length_norm(score, 3, alpha);
        let long = DiverseBeam::length_norm(score, 10, alpha);
        assert!(
            long < short,
            "longer seq should have lower normalised score: short={short} long={long}"
        );
    }

    #[test]
    fn top_k_returns_k_items() {
        let probs = vec![-1.0f32, -0.5, -2.0, -0.1, -3.0];
        let top = DiverseBeam::top_k(&probs, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn top_k_sorted_desc() {
        let probs = vec![-1.0f32, -0.5, -2.0, -0.1, -3.0];
        let top = DiverseBeam::top_k(&probs, 4);
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted desc: {:?}", top);
        }
    }

    #[test]
    fn top_k_selects_highest_scores() {
        let probs = vec![-3.0f32, -0.1, -2.0, -5.0];
        let top = DiverseBeam::top_k(&probs, 1);
        assert_eq!(top[0].0, 1, "token 1 has max log-prob");
    }

    #[test]
    fn diverse_beam_empty_sequences_on_immediate_eos() {
        // EOS (token 4) always gets best score → all sequences should be [4].
        let cfg = DiverseBeamConfig {
            beam_width: 2,
            n_groups: 1,
            max_steps: 5,
            vocab_size: 5,
            eos_id: 4,
            diversity_strength: 0.0,
            length_norm_alpha: 0.0,
        };
        let db = DiverseBeam::new(cfg).expect("ok");
        let seqs = db.search(always_eos).expect("ok");
        assert_eq!(seqs.len(), 2);
        for s in &seqs {
            assert!(!s.is_empty());
            assert_eq!(*s.last().expect("non-empty"), 4);
        }
    }

    #[test]
    fn diverse_beam_respects_max_steps() {
        let cfg = DiverseBeamConfig {
            beam_width: 2,
            n_groups: 1,
            max_steps: 3,
            vocab_size: 3,
            eos_id: 99,
            diversity_strength: 0.0,
            length_norm_alpha: 0.0,
        };
        let db = DiverseBeam::new(cfg).expect("ok");
        let score_no_eos = |_: &[usize]| vec![-1.0f32, -2.0, -3.0];
        let seqs = db.search(score_no_eos).expect("ok");
        for s in &seqs {
            assert!(
                s.len() <= 3,
                "sequence longer than max_steps: len={}",
                s.len()
            );
        }
    }

    #[test]
    fn new_err_beam_not_divisible() {
        let mut cfg = default_cfg();
        cfg.beam_width = 5;
        cfg.n_groups = 2;
        assert!(matches!(
            DiverseBeam::new(cfg),
            Err(SeqError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn new_err_zero_groups() {
        let mut cfg = default_cfg();
        cfg.n_groups = 0;
        assert!(matches!(
            DiverseBeam::new(cfg),
            Err(SeqError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn new_err_zero_beam() {
        let mut cfg = default_cfg();
        cfg.beam_width = 0;
        assert!(matches!(
            DiverseBeam::new(cfg),
            Err(SeqError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn new_err_zero_vocab() {
        let mut cfg = default_cfg();
        cfg.vocab_size = 0;
        assert!(matches!(
            DiverseBeam::new(cfg),
            Err(SeqError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn n_groups_1_matches_standard_beam_top_choice() {
        // With n_groups=1 and no diversity, the top beam should greedily prefer
        // the highest-probability token at each step.
        let cfg = DiverseBeamConfig {
            beam_width: 2,
            n_groups: 1,
            max_steps: 4,
            vocab_size: 3,
            eos_id: 99,
            diversity_strength: 0.0,
            length_norm_alpha: 0.0,
        };
        let db = DiverseBeam::new(cfg).expect("ok");
        let score_fn = |_: &[usize]| vec![-0.1f32, -1.0, -5.0];
        let seqs = db.search(score_fn).expect("ok");
        // Top beam should be all token-0.
        assert_eq!(seqs[0], vec![0, 0, 0, 0]);
    }

    #[test]
    fn diverse_beam_single_token_vocab() {
        let cfg = DiverseBeamConfig {
            beam_width: 2,
            n_groups: 1,
            max_steps: 3,
            vocab_size: 1,
            eos_id: 0,
            diversity_strength: 0.0,
            length_norm_alpha: 0.0,
        };
        let db = DiverseBeam::new(cfg).expect("ok");
        let score_fn = |_: &[usize]| vec![0.0f32];
        let seqs = db.search(score_fn).expect("ok");
        assert_eq!(seqs.len(), 2);
        for s in &seqs {
            assert!(!s.is_empty(), "sequence must not be empty");
        }
    }

    #[test]
    fn diverse_beam_eos_in_group0_still_returns_full() {
        // Group 0 finishes immediately; group 1 should still decode normally.
        let cfg = DiverseBeamConfig {
            beam_width: 4,
            n_groups: 2,
            max_steps: 5,
            vocab_size: 5,
            eos_id: 4,
            diversity_strength: 0.3,
            length_norm_alpha: 0.0,
        };
        let db = DiverseBeam::new(cfg).expect("ok");

        // Group 0 will pick EOS; group 1 will pick token 0.
        let call_count = std::cell::Cell::new(0u32);
        let seqs = db
            .search(|prefix| {
                call_count.set(call_count.get() + 1);
                if prefix.is_empty() || prefix.last().copied() == Some(4) {
                    vec![
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                        0.0,
                    ]
                } else {
                    vec![-0.1, -0.5, -1.0, -2.0, -10.0]
                }
            })
            .expect("ok");

        assert_eq!(seqs.len(), 4, "must return beam_width sequences");
    }
}
