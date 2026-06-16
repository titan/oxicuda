//! CTC decoding: best-path (greedy) and prefix-beam search (Graves 2006; Hannun 2014).
//!
//! Given per-frame log-probabilities `[T, C]` over a blank-augmented alphabet, a
//! CTC decoder produces the most probable *label* sequence after applying the
//! CTC collapse `B` (merge repeats, then drop blanks).
//!
//! Two strategies are provided:
//!
//! * [`ctc_greedy_decode`] — best-path decoding: take the arg-max symbol at each
//!   frame and collapse. Fast (`O(T·C)`) but only a lower bound on the true
//!   sequence probability because it ignores alignment multiplicity.
//!
//! * [`ctc_prefix_beam_search`] — prefix-beam search: maintain a beam of label
//!   prefixes, tracking, for each prefix, the probability that it ends in a
//!   blank (`p_b`) versus a non-blank (`p_nb`). This correctly sums the
//!   probabilities of distinct alignments that collapse to the same prefix and
//!   recovers higher-probability sequences than greedy decoding.
//!
//! All probabilities are accumulated in **log-space**.

use crate::error::{SeqError, SeqResult};
use std::collections::HashMap;

/// Numerically-stable `log(exp(a) + exp(b))`.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Validate the emission tensor shape and blank index.
fn validate(log_probs: &[f64], t_len: usize, n_symbols: usize, blank: usize) -> SeqResult<()> {
    if t_len == 0 || n_symbols == 0 {
        return Err(SeqError::EmptyInput);
    }
    if log_probs.len() != t_len * n_symbols {
        return Err(SeqError::ShapeMismatch {
            expected: t_len * n_symbols,
            got: log_probs.len(),
        });
    }
    if blank >= n_symbols {
        return Err(SeqError::IndexOutOfBounds {
            index: blank,
            len: n_symbols,
        });
    }
    Ok(())
}

/// Best-path (greedy) CTC decode: arg-max per frame followed by CTC collapse.
///
/// Returns the decoded label sequence (blanks removed, repeats merged).
pub fn ctc_greedy_decode(
    log_probs: &[f64],
    t_len: usize,
    n_symbols: usize,
    blank: usize,
) -> SeqResult<Vec<usize>> {
    validate(log_probs, t_len, n_symbols, blank)?;
    let mut raw = Vec::with_capacity(t_len);
    for ti in 0..t_len {
        let row = &log_probs[ti * n_symbols..ti * n_symbols + n_symbols];
        let mut best = 0usize;
        let mut best_val = row[0];
        for (c, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return Err(SeqError::NumericalInstability(
                    "NaN in CTC log-probs".into(),
                ));
            }
            if v > best_val {
                best_val = v;
                best = c;
            }
        }
        raw.push(best);
    }
    // Collapse: merge consecutive duplicates, then remove blanks.
    let mut out = Vec::new();
    let mut prev = usize::MAX;
    for &sym in &raw {
        if sym != prev && sym != blank {
            out.push(sym);
        }
        prev = sym;
    }
    Ok(out)
}

/// Per-prefix log-probabilities split by trailing-blank vs trailing-non-blank.
#[derive(Clone, Copy)]
struct PrefixProb {
    /// log P(prefix, last frame emitted blank).
    p_blank: f64,
    /// log P(prefix, last frame emitted a non-blank).
    p_non_blank: f64,
}

impl PrefixProb {
    #[inline]
    fn total(&self) -> f64 {
        log_add_exp(self.p_blank, self.p_non_blank)
    }
}

/// A scored CTC decoding hypothesis returned by [`ctc_prefix_beam_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct CtcHypothesis {
    /// The decoded label sequence.
    pub labels: Vec<usize>,
    /// Total log-probability of the prefix (summed over alignments).
    pub log_prob: f64,
}

/// Prefix-beam-search CTC decoding (Graves 2006; Hannun 2014).
///
/// * `beam_width` — maximum number of prefixes kept after each frame (`≥ 1`).
///
/// Returns the surviving hypotheses sorted by descending total log-probability;
/// the first element is the most probable decoding.
pub fn ctc_prefix_beam_search(
    log_probs: &[f64],
    t_len: usize,
    n_symbols: usize,
    blank: usize,
    beam_width: usize,
) -> SeqResult<Vec<CtcHypothesis>> {
    validate(log_probs, t_len, n_symbols, blank)?;
    if beam_width == 0 {
        return Err(SeqError::InvalidParameter {
            name: "beam_width".into(),
            value: 0.0,
        });
    }
    for &v in log_probs {
        if v.is_nan() {
            return Err(SeqError::NumericalInstability(
                "NaN in CTC log-probs".into(),
            ));
        }
    }

    // beam maps prefix -> PrefixProb. The empty prefix starts with p_blank = 0.
    let mut beam: HashMap<Vec<usize>, PrefixProb> = HashMap::new();
    beam.insert(
        Vec::new(),
        PrefixProb {
            p_blank: 0.0,
            p_non_blank: f64::NEG_INFINITY,
        },
    );

    for ti in 0..t_len {
        let row = &log_probs[ti * n_symbols..ti * n_symbols + n_symbols];
        let mut next: HashMap<Vec<usize>, PrefixProb> = HashMap::new();

        for (prefix, prob) in &beam {
            // 1) Emit blank: prefix is unchanged, accumulates into p_blank.
            let entry = next.entry(prefix.clone()).or_insert(PrefixProb {
                p_blank: f64::NEG_INFINITY,
                p_non_blank: f64::NEG_INFINITY,
            });
            entry.p_blank = log_add_exp(entry.p_blank, prob.total() + row[blank]);

            // 2) Emit a non-blank symbol c.
            for c in 0..n_symbols {
                if c == blank {
                    continue;
                }
                let lp_c = row[c];
                let last = prefix.last().copied();
                if last == Some(c) {
                    // Repeat of the current last label.
                    // (a) extends to a NEW token only from a blank-terminated path.
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(c);
                    let e = next.entry(new_prefix).or_insert(PrefixProb {
                        p_blank: f64::NEG_INFINITY,
                        p_non_blank: f64::NEG_INFINITY,
                    });
                    e.p_non_blank = log_add_exp(e.p_non_blank, prob.p_blank + lp_c);
                    // (b) merges into the SAME prefix from a non-blank path.
                    let e_same = next.entry(prefix.clone()).or_insert(PrefixProb {
                        p_blank: f64::NEG_INFINITY,
                        p_non_blank: f64::NEG_INFINITY,
                    });
                    e_same.p_non_blank = log_add_exp(e_same.p_non_blank, prob.p_non_blank + lp_c);
                } else {
                    // Distinct from the last label: always extends the prefix,
                    // from either blank- or non-blank-terminated paths.
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(c);
                    let e = next.entry(new_prefix).or_insert(PrefixProb {
                        p_blank: f64::NEG_INFINITY,
                        p_non_blank: f64::NEG_INFINITY,
                    });
                    e.p_non_blank = log_add_exp(e.p_non_blank, prob.total() + lp_c);
                }
            }
        }

        // Prune to the top `beam_width` prefixes by total probability.
        let mut scored: Vec<(Vec<usize>, PrefixProb)> = next.into_iter().collect();
        scored.sort_by(|a, b| {
            b.1.total()
                .partial_cmp(&a.1.total())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(beam_width);
        beam = scored.into_iter().collect();
    }

    let mut hyps: Vec<CtcHypothesis> = beam
        .into_iter()
        .map(|(labels, prob)| CtcHypothesis {
            labels,
            log_prob: prob.total(),
        })
        .collect();
    hyps.sort_by(|a, b| {
        b.log_prob
            .partial_cmp(&a.log_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(hyps)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn to_log(probs: &[f64]) -> Vec<f64> {
        probs.iter().map(|&p| p.max(1e-30).ln()).collect()
    }

    #[test]
    fn greedy_collapses_repeats_and_blanks() {
        // argmax path per frame: [1, 1, 0(blank), 2] → collapse → [1, 2].
        let probs = vec![
            0.1, 0.8, 0.1, //
            0.1, 0.8, 0.1, //
            0.8, 0.1, 0.1, //
            0.1, 0.1, 0.8, //
        ];
        let lp = to_log(&probs);
        let out = ctc_greedy_decode(&lp, 4, 3, 0).expect("decode");
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn greedy_all_blank_is_empty() {
        let probs = vec![
            0.9, 0.05, 0.05, //
            0.9, 0.05, 0.05, //
        ];
        let lp = to_log(&probs);
        let out = ctc_greedy_decode(&lp, 2, 3, 0).expect("decode");
        assert!(out.is_empty());
    }

    #[test]
    fn greedy_repeat_without_blank_merges() {
        // [1,1] with no separating blank collapses to [1].
        let probs = vec![
            0.1, 0.8, 0.1, //
            0.1, 0.8, 0.1, //
        ];
        let lp = to_log(&probs);
        let out = ctc_greedy_decode(&lp, 2, 3, 0).expect("decode");
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn greedy_blank_at_last_index() {
        // blank = C-1 = 2; argmax path [0, 1, 2(blank)] → [0, 1].
        let probs = vec![
            0.8, 0.1, 0.1, //
            0.1, 0.8, 0.1, //
            0.1, 0.1, 0.8, //
        ];
        let lp = to_log(&probs);
        let out = ctc_greedy_decode(&lp, 3, 3, 2).expect("decode");
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn beam_returns_sorted_hypotheses() {
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
            0.4, 0.3, 0.3, //
        ];
        let lp = to_log(&probs);
        let hyps = ctc_prefix_beam_search(&lp, 4, 3, 0, 8).expect("beam");
        assert!(!hyps.is_empty());
        for w in hyps.windows(2) {
            assert!(w[0].log_prob >= w[1].log_prob - 1e-12);
        }
    }

    #[test]
    fn beam_top1_matches_greedy_for_peaked_input() {
        // Sharply peaked emissions → beam best == greedy best.
        let probs = vec![
            0.02, 0.96, 0.02, //
            0.96, 0.02, 0.02, //
            0.02, 0.02, 0.96, //
        ];
        let lp = to_log(&probs);
        let greedy = ctc_greedy_decode(&lp, 3, 3, 0).expect("greedy");
        let beam = ctc_prefix_beam_search(&lp, 3, 3, 0, 16).expect("beam");
        assert_eq!(beam[0].labels, greedy);
    }

    #[test]
    fn beam_total_probability_consistent_with_loss() {
        // For a peaked distribution where one sequence dominates, the beam's top
        // log-prob should be close to the negative CTC loss of that sequence.
        use crate::ctc::ctc_loss::ctc_loss;
        let probs = vec![
            0.02, 0.96, 0.02, //
            0.96, 0.02, 0.02, //
            0.02, 0.02, 0.96, //
        ];
        let lp = to_log(&probs);
        let beam = ctc_prefix_beam_search(&lp, 3, 3, 0, 32).expect("beam");
        let best = &beam[0];
        let loss = ctc_loss(&lp, 3, 3, &best.labels, 0).expect("loss");
        // beam log-prob ≤ −loss is not guaranteed exactly, but they should be
        // close for a strongly peaked distribution.
        assert!(
            (best.log_prob - (-loss)).abs() < 0.2,
            "beam={} loss={loss}",
            best.log_prob
        );
    }

    #[test]
    fn beam_width_one_is_valid() {
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.4, 0.3, 0.3, //
        ];
        let lp = to_log(&probs);
        let hyps = ctc_prefix_beam_search(&lp, 2, 3, 0, 1).expect("beam");
        assert_eq!(hyps.len(), 1);
    }

    #[test]
    fn beam_recovers_empty_when_blank_dominates() {
        let probs = vec![
            0.9, 0.05, 0.05, //
            0.9, 0.05, 0.05, //
        ];
        let lp = to_log(&probs);
        let hyps = ctc_prefix_beam_search(&lp, 2, 3, 0, 8).expect("beam");
        assert!(hyps[0].labels.is_empty());
    }

    #[test]
    fn greedy_shape_mismatch_errors() {
        let lp = vec![0.0; 5];
        assert!(ctc_greedy_decode(&lp, 2, 3, 0).is_err());
    }

    #[test]
    fn greedy_blank_out_of_range_errors() {
        let lp = to_log(&[0.5, 0.5, 0.5, 0.5]);
        assert!(ctc_greedy_decode(&lp, 2, 2, 9).is_err());
    }

    #[test]
    fn beam_zero_width_errors() {
        let lp = to_log(&[0.5, 0.5, 0.5, 0.5]);
        assert!(ctc_prefix_beam_search(&lp, 2, 2, 0, 0).is_err());
    }

    #[test]
    fn beam_nan_errors() {
        let lp = vec![f64::NAN, 0.0, 0.0, 0.0];
        assert!(ctc_prefix_beam_search(&lp, 2, 2, 0, 4).is_err());
    }

    #[test]
    fn greedy_nan_errors() {
        let lp = vec![0.0, f64::NAN, 0.0, 0.0, 0.0, 0.0];
        assert!(ctc_greedy_decode(&lp, 2, 3, 0).is_err());
    }
}
