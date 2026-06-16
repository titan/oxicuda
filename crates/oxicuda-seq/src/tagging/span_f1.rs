//! Span-based (entity-level) precision / recall / F1 — the `seqeval` metric.
//!
//! Token-level accuracy is a poor measure for sequence labelling: a model that
//! gets every `O` right but mangles entity boundaries can still score highly.
//! The standard NER / chunking metric (CoNLL-2000/2003, implemented by the
//! `seqeval` library) instead compares **entity spans**: a predicted span is a
//! true positive only when an identical span — same entity type **and** same
//! start/end boundaries — appears in the gold sequence.
//!
//! Given gold tags `y` and predicted tags `ŷ`, spans are extracted with
//! [`crate::tagging::bioes::extract_spans`] and matched exactly:
//!
//! ```text
//! precision = |gold ∩ pred| / |pred|
//! recall    = |gold ∩ pred| / |gold|
//! f1        = 2 P R / (P + R)
//! ```
//!
//! Per-type breakdowns and a micro-averaged total are provided.

use crate::error::{SeqError, SeqResult};
use crate::tagging::bioes::{Span, Tag, extract_spans};
use std::collections::BTreeMap;

// ─── Counts ──────────────────────────────────────────────────────────────────

/// Precision / recall / F1 with the raw TP/FP/FN counts behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct PrfScore {
    /// True positives (exactly-matched spans).
    pub tp: usize,
    /// False positives (predicted spans with no gold match).
    pub fp: usize,
    /// False negatives (gold spans with no predicted match).
    pub fn_: usize,
    /// Precision `TP / (TP + FP)` (0 when no predictions).
    pub precision: f64,
    /// Recall `TP / (TP + FN)` (0 when no gold spans).
    pub recall: f64,
    /// Harmonic-mean F1 (0 when `P + R == 0`).
    pub f1: f64,
}

impl PrfScore {
    /// Build a [`PrfScore`] from raw counts, computing P/R/F1.
    #[must_use]
    pub fn from_counts(tp: usize, fp: usize, fn_: usize) -> Self {
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            tp,
            fp,
            fn_,
            precision,
            recall,
            f1,
        }
    }
}

/// Full span-F1 report: a micro-averaged overall score plus per-type scores.
#[derive(Debug, Clone)]
pub struct SpanF1Report {
    /// Micro-averaged score over all entity types.
    pub overall: PrfScore,
    /// Per-entity-type scores, keyed by type name (sorted).
    pub per_type: BTreeMap<String, PrfScore>,
}

// ─── Span multiset matching ──────────────────────────────────────────────────

/// Count exactly-matched spans (true positives) between two span lists.
///
/// Spans are matched as a multiset: each gold span can satisfy at most one
/// predicted span.  Returns `(tp, n_pred, n_gold)`.
fn match_spans(gold: &[Span], pred: &[Span]) -> (usize, usize, usize) {
    // Multiset of gold spans by value.
    let mut remaining: BTreeMap<(String, usize, usize), usize> = BTreeMap::new();
    for g in gold {
        *remaining
            .entry((g.entity_type.clone(), g.start, g.end))
            .or_insert(0) += 1;
    }
    let mut tp = 0usize;
    for p in pred {
        let key = (p.entity_type.clone(), p.start, p.end);
        if let Some(cnt) = remaining.get_mut(&key) {
            if *cnt > 0 {
                *cnt -= 1;
                tp += 1;
            }
        }
    }
    (tp, pred.len(), gold.len())
}

/// Compute the micro-averaged span-F1 from pre-extracted gold/predicted spans.
#[must_use]
pub fn span_f1_from_spans(gold: &[Span], pred: &[Span]) -> SpanF1Report {
    let (tp, n_pred, n_gold) = match_spans(gold, pred);
    let overall = PrfScore::from_counts(tp, n_pred - tp, n_gold - tp);

    // Per-type.
    let mut types: BTreeMap<String, (Vec<Span>, Vec<Span>)> = BTreeMap::new();
    for g in gold {
        types
            .entry(g.entity_type.clone())
            .or_default()
            .0
            .push(g.clone());
    }
    for p in pred {
        types
            .entry(p.entity_type.clone())
            .or_default()
            .1
            .push(p.clone());
    }
    let mut per_type = BTreeMap::new();
    for (ty, (g, p)) in types {
        let (t, np, ng) = match_spans(&g, &p);
        per_type.insert(ty, PrfScore::from_counts(t, np - t, ng - t));
    }

    SpanF1Report { overall, per_type }
}

/// Compute span-F1 directly from gold and predicted **tag** sequences.
///
/// Both sequences must have equal length (one tag per token).
///
/// # Errors
///
/// * [`SeqError::LengthMismatch`] — if `gold.len() ≠ pred.len()`.
/// * [`SeqError::EmptyInput`]     — if the sequences are empty.
pub fn span_f1(gold: &[Tag], pred: &[Tag]) -> SeqResult<SpanF1Report> {
    if gold.len() != pred.len() {
        return Err(SeqError::LengthMismatch {
            a: gold.len(),
            b: pred.len(),
        });
    }
    if gold.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let gspans = extract_spans(gold);
    let pspans = extract_spans(pred);
    Ok(span_f1_from_spans(&gspans, &pspans))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagging::bioes::parse_tags;

    fn tags(strs: &[&str]) -> Vec<Tag> {
        parse_tags(strs).expect("parse")
    }

    #[test]
    fn prf_from_counts_basic() {
        let s = PrfScore::from_counts(3, 1, 1);
        assert_eq!(s.tp, 3);
        assert!((s.precision - 0.75).abs() < 1e-12);
        assert!((s.recall - 0.75).abs() < 1e-12);
        assert!((s.f1 - 0.75).abs() < 1e-12);
    }

    #[test]
    fn prf_zero_predictions() {
        let s = PrfScore::from_counts(0, 0, 4);
        assert_eq!(s.precision, 0.0);
        assert_eq!(s.recall, 0.0);
        assert_eq!(s.f1, 0.0);
    }

    #[test]
    fn perfect_match_is_one() {
        let g = tags(&["B-PER", "I-PER", "O", "S-LOC"]);
        let report = span_f1(&g, &g).expect("ok");
        assert!((report.overall.f1 - 1.0).abs() < 1e-12);
        assert!((report.overall.precision - 1.0).abs() < 1e-12);
        assert!((report.overall.recall - 1.0).abs() < 1e-12);
        assert_eq!(report.overall.tp, 2);
    }

    #[test]
    fn boundary_error_counts_as_miss() {
        // gold PER spans [0..1]; pred PER span [0..0] — different boundary.
        let g = tags(&["B-PER", "I-PER", "O"]);
        let p = tags(&["B-PER", "O", "O"]);
        let r = span_f1(&g, &p).expect("ok");
        assert_eq!(r.overall.tp, 0); // no exact match
        assert_eq!(r.overall.fp, 1); // predicted PER[0..0]
        assert_eq!(r.overall.fn_, 1); // gold PER[0..1] missed
        assert_eq!(r.overall.f1, 0.0);
    }

    #[test]
    fn type_error_counts_as_miss() {
        // Same boundary, wrong type.
        let g = tags(&["S-PER"]);
        let p = tags(&["S-LOC"]);
        let r = span_f1(&g, &p).expect("ok");
        assert_eq!(r.overall.tp, 0);
        assert_eq!(r.overall.fp, 1);
        assert_eq!(r.overall.fn_, 1);
    }

    #[test]
    fn partial_credit_micro_average() {
        // gold: PER[0..1], LOC[3]; pred: PER[0..1], LOC[4] (LOC boundary wrong)
        let g = tags(&["B-PER", "I-PER", "O", "S-LOC", "O"]);
        let p = tags(&["B-PER", "I-PER", "O", "O", "S-LOC"]);
        let r = span_f1(&g, &p).expect("ok");
        // 1 TP (PER), 1 FP (LOC@4), 1 FN (LOC@3).
        assert_eq!(r.overall.tp, 1);
        assert_eq!(r.overall.fp, 1);
        assert_eq!(r.overall.fn_, 1);
        assert!((r.overall.precision - 0.5).abs() < 1e-12);
        assert!((r.overall.recall - 0.5).abs() < 1e-12);
        assert!((r.overall.f1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn per_type_breakdown() {
        let g = tags(&["B-PER", "I-PER", "S-LOC", "S-ORG"]);
        let p = tags(&["B-PER", "I-PER", "S-LOC", "O"]);
        let r = span_f1(&g, &p).expect("ok");
        // PER perfect, LOC perfect, ORG missed.
        assert!((r.per_type["PER"].f1 - 1.0).abs() < 1e-12);
        assert!((r.per_type["LOC"].f1 - 1.0).abs() < 1e-12);
        assert_eq!(r.per_type["ORG"].tp, 0);
        assert_eq!(r.per_type["ORG"].fn_, 1);
    }

    #[test]
    fn length_mismatch_errors() {
        let g = tags(&["O", "O"]);
        let p = tags(&["O"]);
        assert!(matches!(
            span_f1(&g, &p),
            Err(SeqError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn empty_errors() {
        assert!(matches!(span_f1(&[], &[]), Err(SeqError::EmptyInput)));
    }

    #[test]
    fn all_outside_gives_zero_spans() {
        let g = tags(&["O", "O", "O"]);
        let p = tags(&["O", "O", "O"]);
        let r = span_f1(&g, &p).expect("ok");
        // No spans at all: P/R/F1 conventionally 0 (seqeval returns 0).
        assert_eq!(r.overall.tp, 0);
        assert_eq!(r.overall.fp, 0);
        assert_eq!(r.overall.fn_, 0);
        assert_eq!(r.overall.f1, 0.0);
    }

    #[test]
    fn precision_recall_can_differ() {
        // gold has 1 span, pred has 2 (one matching, one spurious).
        let g = tags(&["S-PER", "O", "O"]);
        let p = tags(&["S-PER", "S-LOC", "O"]);
        let r = span_f1(&g, &p).expect("ok");
        assert_eq!(r.overall.tp, 1);
        assert_eq!(r.overall.fp, 1);
        assert_eq!(r.overall.fn_, 0);
        assert!((r.overall.precision - 0.5).abs() < 1e-12);
        assert!((r.overall.recall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_predictions_only_one_tp() {
        // Two identical predicted spans, one gold: exactly one TP, one FP.
        let g = [Span {
            entity_type: "PER".into(),
            start: 0,
            end: 0,
        }];
        let p = [
            Span {
                entity_type: "PER".into(),
                start: 0,
                end: 0,
            },
            Span {
                entity_type: "PER".into(),
                start: 0,
                end: 0,
            },
        ];
        let r = span_f1_from_spans(&g, &p);
        assert_eq!(r.overall.tp, 1);
        assert_eq!(r.overall.fp, 1);
    }

    #[test]
    fn from_spans_matches_tag_path() {
        let g = tags(&["B-PER", "E-PER", "O"]);
        let p = tags(&["B-PER", "E-PER", "S-LOC"]);
        let via_tags = span_f1(&g, &p).expect("ok");
        let gs = extract_spans(&g);
        let ps = extract_spans(&p);
        let via_spans = span_f1_from_spans(&gs, &ps);
        assert_eq!(via_tags.overall, via_spans.overall);
    }
}
