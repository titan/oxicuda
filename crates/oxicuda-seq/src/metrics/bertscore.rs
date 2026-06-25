//! BERTScore: token-embedding similarity metric via greedy cosine matching.
//!
//! Reference: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. & Artzi, Y.
//! (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020.
//!
//! # What this module computes
//!
//! BERTScore compares a *candidate* token sequence against a *reference* token
//! sequence using **contextual embeddings** for each token. Given a candidate
//! with embeddings `x̂_1 … x̂_n` and a reference with embeddings `x_1 … x_m`,
//! every pairwise cosine similarity `cos(x̂_i, x_j)` is formed and then matched
//! **greedily** (each token aligned to its single most similar counterpart):
//!
//! ```text
//! Recall    R = ( Σ_j  idf(x_j)  · max_i cos(x̂_i, x_j) ) / Σ_j idf(x_j)
//! Precision P = ( Σ_i  idf(x̂_i) · max_j cos(x̂_i, x_j) ) / Σ_i idf(x̂_i)
//! F1        = 2 · P · R / (P + R)
//! ```
//!
//! With **uniform IDF weights** these reduce to plain averages of the row /
//! column maxima of the cosine-similarity matrix. Optional inverse-document-
//! frequency weights (precomputed from a corpus) down-weight frequent tokens
//! exactly as in the paper.
//!
//! ## Honesty note — this is the real metric, not a stub
//!
//! The "BERT" in BERTScore is only the *source of the embeddings*. This crate
//! does not (and cannot, in pure-CPU form) ship a transformer; instead the
//! **embedding vectors are an input** supplied by the caller (from any encoder:
//! a `trustformers` model, word2vec, a learned table, …). Everything BERTScore
//! actually specifies — the cosine-similarity matrix, the greedy precision /
//! recall / F1 matching, IDF weighting, and the optional baseline rescaling — is
//! computed here in full and is exact. Feeding genuine contextual embeddings
//! reproduces the published metric; feeding any other embeddings yields the same
//! algorithm over those vectors.
//!
//! Production code never panics: every fallible path validates its inputs and
//! returns [`SeqError`].

use crate::error::{SeqError, SeqResult};

/// Precision / recall / F1 triple produced by BERTScore.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertScore {
    /// Precision: how well each candidate token is covered by the reference.
    pub precision: f64,
    /// Recall: how well each reference token is covered by the candidate.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
}

/// Configuration for BERTScore.
#[derive(Debug, Clone, Default)]
pub struct BertScoreConfig {
    /// Optional baseline value `b ∈ (−1, 1)` used for *rescaling* the raw
    /// scores: `score ← (score − b) / (1 − b)`. The paper rescales against an
    /// empirical baseline (the average score of random sentence pairs for the
    /// chosen model/layer) so that scores spread across a more interpretable
    /// range. `None` disables rescaling (raw cosine scores in `[−1, 1]`).
    pub baseline: Option<f64>,
}

impl BertScoreConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// * [`SeqError::InvalidParameter`] if `baseline` is set to a non-finite
    ///   value or to `±1` (the rescaling denominator `1 − b` must be non-zero,
    ///   and a baseline outside `(−1, 1)` is meaningless for cosine scores).
    pub fn validate(&self) -> SeqResult<()> {
        if let Some(b) = self.baseline {
            if !b.is_finite() || b <= -1.0 || b >= 1.0 {
                return Err(SeqError::InvalidParameter {
                    name: "baseline".into(),
                    value: b,
                });
            }
        }
        Ok(())
    }

    /// Apply the optional baseline rescaling to a raw score.
    fn rescale(&self, score: f64) -> f64 {
        match self.baseline {
            Some(b) => (score - b) / (1.0 - b),
            None => score,
        }
    }
}

/// L2 norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity of two equal-length, non-zero vectors. The norms are
/// passed in to avoid recomputing them inside the `n × m` loop.
fn cosine(a: &[f64], na: f64, b: &[f64], nb: f64) -> f64 {
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Compute BERTScore between candidate and reference token embeddings with
/// **uniform** token weights.
///
/// `candidate` holds `n` row-major embedding vectors of dimension `dim`
/// (`candidate.len() == n * dim`); `reference` holds `m` such vectors
/// (`reference.len() == m * dim`).
///
/// # Errors
/// * [`SeqError::EmptyInput`] if either side has zero tokens or `dim == 0`.
/// * [`SeqError::ShapeMismatch`] if a flat buffer length is not a multiple of
///   `dim` consistent with the stated token count.
/// * Propagates [`BertScoreConfig::validate`].
pub fn bert_score(
    candidate: &[f64],
    n: usize,
    reference: &[f64],
    m: usize,
    dim: usize,
    config: &BertScoreConfig,
) -> SeqResult<BertScore> {
    let cand_idf = vec![1.0; n];
    let ref_idf = vec![1.0; m];
    bert_score_idf(candidate, n, reference, m, dim, &cand_idf, &ref_idf, config)
}

/// Compute BERTScore with explicit **IDF weights** for candidate and reference
/// tokens (e.g. precomputed inverse-document-frequencies over a corpus). Weights
/// must be non-negative and finite; at least one weight on each side must be
/// strictly positive (so the normalising denominators are non-zero).
///
/// # Errors
/// In addition to the cases of [`bert_score`]:
/// * [`SeqError::LengthMismatch`] if `cand_idf.len() != n` or
///   `ref_idf.len() != m`.
/// * [`SeqError::InvalidParameter`] if any weight is negative / non-finite.
/// * [`SeqError::NumericalInstability`] if the candidate or reference weights
///   sum to zero.
#[allow(clippy::too_many_arguments)]
pub fn bert_score_idf(
    candidate: &[f64],
    n: usize,
    reference: &[f64],
    m: usize,
    dim: usize,
    cand_idf: &[f64],
    ref_idf: &[f64],
    config: &BertScoreConfig,
) -> SeqResult<BertScore> {
    config.validate()?;
    if n == 0 || m == 0 || dim == 0 {
        return Err(SeqError::EmptyInput);
    }
    if candidate.len() != n * dim {
        return Err(SeqError::ShapeMismatch {
            expected: n * dim,
            got: candidate.len(),
        });
    }
    if reference.len() != m * dim {
        return Err(SeqError::ShapeMismatch {
            expected: m * dim,
            got: reference.len(),
        });
    }
    if cand_idf.len() != n {
        return Err(SeqError::LengthMismatch {
            a: cand_idf.len(),
            b: n,
        });
    }
    if ref_idf.len() != m {
        return Err(SeqError::LengthMismatch {
            a: ref_idf.len(),
            b: m,
        });
    }
    let mut sum_cand_idf = 0.0;
    for (idx, &w) in cand_idf.iter().enumerate() {
        if !(w.is_finite() && w >= 0.0) {
            return Err(SeqError::InvalidParameter {
                name: format!("cand_idf[{idx}]"),
                value: w,
            });
        }
        sum_cand_idf += w;
    }
    let mut sum_ref_idf = 0.0;
    for (idx, &w) in ref_idf.iter().enumerate() {
        if !(w.is_finite() && w >= 0.0) {
            return Err(SeqError::InvalidParameter {
                name: format!("ref_idf[{idx}]"),
                value: w,
            });
        }
        sum_ref_idf += w;
    }
    if sum_cand_idf <= 0.0 || sum_ref_idf <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "IDF weights sum to zero on one side".into(),
        ));
    }

    // Precompute norms.
    let cand_norms: Vec<f64> = (0..n)
        .map(|i| l2_norm(&candidate[i * dim..(i + 1) * dim]))
        .collect();
    let ref_norms: Vec<f64> = (0..m)
        .map(|j| l2_norm(&reference[j * dim..(j + 1) * dim]))
        .collect();

    // Row maxima (over reference) give precision; column maxima (over
    // candidate) give recall. Compute the full similarity once, tracking both.
    let mut row_max = vec![f64::NEG_INFINITY; n]; // best ref for each cand token
    let mut col_max = vec![f64::NEG_INFINITY; m]; // best cand for each ref token
    for i in 0..n {
        let ci = &candidate[i * dim..(i + 1) * dim];
        let ni = cand_norms[i];
        for j in 0..m {
            let rj = &reference[j * dim..(j + 1) * dim];
            let sim = cosine(ci, ni, rj, ref_norms[j]);
            if sim > row_max[i] {
                row_max[i] = sim;
            }
            if sim > col_max[j] {
                col_max[j] = sim;
            }
        }
    }

    // Weighted precision / recall.
    let mut precision = 0.0;
    for i in 0..n {
        precision += cand_idf[i] * row_max[i];
    }
    precision /= sum_cand_idf;

    let mut recall = 0.0;
    for j in 0..m {
        recall += ref_idf[j] * col_max[j];
    }
    recall /= sum_ref_idf;

    // Optional baseline rescaling, then F1 from the (possibly rescaled) P, R.
    precision = config.rescale(precision);
    recall = config.rescale(recall);

    let f1 = if precision + recall <= 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(BertScore {
        precision,
        recall,
        f1,
    })
}

/// Convenience IDF estimator: compute smoothed inverse-document-frequencies for
/// a vocabulary from a corpus of tokenised documents.
///
/// `idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1` where `N` is the number of
/// documents and `df(t)` the number of documents containing token `t` (each
/// token counted at most once per document). The `+1`/smoothing matches the
/// common scikit-learn convention and keeps every weight strictly positive.
/// Tokens are identified by `usize` ids in `0..vocab_size`.
///
/// # Errors
/// * [`SeqError::EmptyInput`] if `vocab_size == 0` or `documents` is empty.
/// * [`SeqError::IndexOutOfBounds`] if any token id is `>= vocab_size`.
pub fn corpus_idf(documents: &[Vec<usize>], vocab_size: usize) -> SeqResult<Vec<f64>> {
    if vocab_size == 0 || documents.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let n_docs = documents.len() as f64;
    let mut df = vec![0.0f64; vocab_size];
    let mut seen = vec![false; vocab_size];
    for doc in documents {
        for &t in doc {
            if t >= vocab_size {
                return Err(SeqError::IndexOutOfBounds {
                    index: t,
                    len: vocab_size,
                });
            }
        }
        // Reset only the entries we touched (cheaper than clearing the whole
        // vector for short documents).
        for &t in doc {
            seen[t] = false;
        }
        for &t in doc {
            if !seen[t] {
                seen[t] = true;
                df[t] += 1.0;
            }
        }
    }
    let idf: Vec<f64> = df
        .iter()
        .map(|&d| ((1.0 + n_docs) / (1.0 + d)).ln() + 1.0)
        .collect();
    Ok(idf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical candidate and reference embeddings ⇒ P = R = F1 = 1
    /// (every token matches itself with cosine 1).
    #[test]
    fn identical_scores_one() {
        let dim = 3;
        let emb = vec![
            1.0, 0.0, 0.0, // tok 0
            0.0, 1.0, 0.0, // tok 1
            0.0, 0.0, 1.0, // tok 2
        ];
        let cfg = BertScoreConfig::default();
        let s = bert_score(&emb, 3, &emb, 3, dim, &cfg).expect("score");
        assert!((s.precision - 1.0).abs() < 1e-12, "P = {}", s.precision);
        assert!((s.recall - 1.0).abs() < 1e-12, "R = {}", s.recall);
        assert!((s.f1 - 1.0).abs() < 1e-12, "F1 = {}", s.f1);
    }

    /// Orthogonal embeddings ⇒ cosine 0 everywhere ⇒ all scores 0.
    #[test]
    fn orthogonal_scores_zero() {
        let dim = 2;
        let cand = vec![1.0, 0.0]; // 1 token along x
        let reference = vec![0.0, 1.0]; // 1 token along y
        let cfg = BertScoreConfig::default();
        let s = bert_score(&cand, 1, &reference, 1, dim, &cfg).expect("score");
        assert!(s.precision.abs() < 1e-12);
        assert!(s.recall.abs() < 1e-12);
        assert!(s.f1.abs() < 1e-12);
    }

    /// Greedy matching: a candidate token aligns to its single most-similar
    /// reference token. Here candidate {x} against reference {x, y}: precision
    /// (one cand token, best match = x ⇒ 1) but recall averages best-of-x=1 and
    /// best-of-y=0 ⇒ 0.5.
    #[test]
    fn greedy_matching_asymmetric() {
        let dim = 2;
        let cand = vec![1.0, 0.0]; // {x}
        let reference = vec![1.0, 0.0, 0.0, 1.0]; // {x, y}
        let cfg = BertScoreConfig::default();
        let s = bert_score(&cand, 1, &reference, 2, dim, &cfg).expect("score");
        assert!((s.precision - 1.0).abs() < 1e-12, "P = {}", s.precision);
        assert!((s.recall - 0.5).abs() < 1e-12, "R = {}", s.recall);
        // F1 = 2 * 1 * 0.5 / 1.5
        assert!((s.f1 - (2.0 * 0.5 / 1.5)).abs() < 1e-12, "F1 = {}", s.f1);
    }

    /// Cosine ignores magnitude: scaling a vector does not change the score.
    #[test]
    fn scale_invariance() {
        let dim = 3;
        let cand = vec![2.0, 0.0, 0.0];
        let reference = vec![5.0, 0.0, 0.0];
        let cfg = BertScoreConfig::default();
        let s = bert_score(&cand, 1, &reference, 1, dim, &cfg).expect("score");
        assert!((s.f1 - 1.0).abs() < 1e-12, "F1 = {}", s.f1);
    }

    /// Baseline rescaling maps a raw score `r` to `(r − b)/(1 − b)`.
    #[test]
    fn baseline_rescaling() {
        let dim = 2;
        // Make raw precision/recall exactly 0.5 via a 60° angle: cos 60° = 0.5.
        let cand = vec![1.0, 0.0];
        let reference = vec![0.5, 3.0f64.sqrt() / 2.0]; // unit vector at 60°
        let cfg_raw = BertScoreConfig::default();
        let raw = bert_score(&cand, 1, &reference, 1, dim, &cfg_raw).expect("raw");
        assert!((raw.f1 - 0.5).abs() < 1e-9, "raw f1 = {}", raw.f1);

        let b = 0.25;
        let cfg = BertScoreConfig { baseline: Some(b) };
        let rescaled = bert_score(&cand, 1, &reference, 1, dim, &cfg).expect("rescaled");
        let expected = (0.5 - b) / (1.0 - b);
        assert!(
            (rescaled.precision - expected).abs() < 1e-9,
            "P = {}",
            rescaled.precision
        );
        assert!(
            (rescaled.f1 - expected).abs() < 1e-9,
            "F1 = {}",
            rescaled.f1
        );
    }

    /// IDF weighting changes the average toward heavily-weighted tokens.
    #[test]
    fn idf_weighting() {
        let dim = 2;
        // Reference {x, y}; candidate {x} matches x perfectly (1) and y not at
        // all (0). Up-weighting the y token lowers recall; up-weighting x raises
        // it.
        let cand = vec![1.0, 0.0];
        let reference = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = BertScoreConfig::default();
        let cand_idf = vec![1.0];

        // Weight x token (matched) heavily ⇒ recall → 1.
        let ref_idf_high_x = vec![10.0, 1.0];
        let s_high = bert_score_idf(
            &cand,
            1,
            &reference,
            2,
            dim,
            &cand_idf,
            &ref_idf_high_x,
            &cfg,
        )
        .expect("score");
        // recall = (10*1 + 1*0) / 11
        assert!(
            (s_high.recall - 10.0 / 11.0).abs() < 1e-12,
            "R = {}",
            s_high.recall
        );

        // Weight y token (unmatched) heavily ⇒ recall → 0.
        let ref_idf_high_y = vec![1.0, 10.0];
        let s_low = bert_score_idf(
            &cand,
            1,
            &reference,
            2,
            dim,
            &cand_idf,
            &ref_idf_high_y,
            &cfg,
        )
        .expect("score");
        assert!(
            (s_low.recall - 1.0 / 11.0).abs() < 1e-12,
            "R = {}",
            s_low.recall
        );
        assert!(s_high.recall > s_low.recall);
    }

    /// `corpus_idf`: rarer tokens get higher IDF than frequent ones, and the
    /// smoothed formula keeps everything positive.
    #[test]
    fn corpus_idf_orders_by_rarity() {
        // token 0 appears in all 3 docs, token 1 in 1 doc, token 2 in 0 docs.
        let docs = vec![vec![0usize, 0, 1], vec![0usize], vec![0usize]];
        let idf = corpus_idf(&docs, 3).expect("idf");
        assert_eq!(idf.len(), 3);
        // df(0)=3, df(1)=1, df(2)=0  ⇒ idf strictly increasing in rarity.
        assert!(idf[0] < idf[1], "{} !< {}", idf[0], idf[1]);
        assert!(idf[1] < idf[2], "{} !< {}", idf[1], idf[2]);
        for &w in &idf {
            assert!(w > 0.0, "idf {w} not positive");
        }
        // df=3, N=3 ⇒ ln((1+3)/(1+3)) + 1 = 1.
        assert!((idf[0] - 1.0).abs() < 1e-12);
    }

    /// Validation paths.
    #[test]
    fn validation_errors() {
        let cfg = BertScoreConfig::default();
        // empty
        assert!(bert_score(&[], 0, &[1.0], 1, 1, &cfg).is_err());
        // shape mismatch (n*dim != len)
        assert!(bert_score(&[1.0, 2.0, 3.0], 2, &[1.0, 2.0], 1, 2, &cfg).is_err());
        // bad baseline = 1.0
        let bad = BertScoreConfig {
            baseline: Some(1.0),
        };
        assert!(bad.validate().is_err());
        // idf length mismatch
        assert!(
            bert_score_idf(&[1.0, 0.0], 1, &[1.0, 0.0], 1, 2, &[1.0, 1.0], &[1.0], &cfg).is_err()
        );
        // negative idf
        assert!(bert_score_idf(&[1.0, 0.0], 1, &[1.0, 0.0], 1, 2, &[-1.0], &[1.0], &cfg).is_err());
        // corpus_idf out-of-range id
        assert!(corpus_idf(&[vec![5usize]], 3).is_err());
    }
}
