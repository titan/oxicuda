//! Two-stage re-ranking with exact distances.
//!
//! Approximate indexes (IVFPQ, HNSW-PQ, LSH, binary codes …) trade accuracy for
//! speed: they return a *candidate set* that is cheap to produce but ordered by
//! an **approximate** score (ADC distance, Hamming distance, hash agreement).
//! A re-ranking stage takes that candidate set, recomputes the **exact** metric
//! against the full-precision vectors, and returns the true top-`k`.  This is
//! the standard "coarse → fine" recipe used by FAISS (`IndexRefine`), ScaNN, and
//! DiskANN: stage 1 narrows millions of vectors to a few hundred candidates,
//! stage 2 exactly scores only those.
//!
//! The functions here are deliberately index-agnostic — they accept the
//! candidate ids produced by *any* first stage plus a corpus of full-precision
//! vectors, and re-rank by exact L2 or exact inner product.  Recall@k of the
//! two-stage pipeline is then bounded below by the fraction of true neighbours
//! that survived into the candidate set, which the helpers here let callers
//! measure directly.
use crate::error::{AnnError, AnnResult};

/// Exact metric used by the re-ranking stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankMetric {
    /// Squared Euclidean distance; smaller is better (ascending order).
    L2Squared,
    /// Negative inner product, so that "smaller is better" still holds and the
    /// returned scores sort ascending exactly like L2 (the caller may negate to
    /// recover the raw inner product).
    NegInnerProduct,
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
fn neg_ip(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Re-rank a candidate id list against full-precision vectors and return the
/// exact top-`k` as `(id, score)` ascending by `score` (smaller is better).
///
/// `candidates` are ids into the row-major `corpus` of shape `[n_corpus, dim]`.
/// Duplicate candidate ids are de-duplicated (an approximate stage may surface
/// the same id from several probed lists).
///
/// # Errors
/// - [`AnnError::EmptyInput`] when `query` is empty.
/// - [`AnnError::InvalidK`] when `k == 0`.
/// - [`AnnError::DimensionMismatch`] when `query.len() != dim` or the corpus
///   length is not a multiple of `dim`.
/// - [`AnnError::IdOutOfRange`] when a candidate id is ≥ `n_corpus`.
pub fn rerank_exact(
    query: &[f32],
    corpus: &[f32],
    dim: usize,
    candidates: &[usize],
    k: usize,
    metric: RerankMetric,
) -> AnnResult<Vec<(usize, f32)>> {
    if query.is_empty() {
        return Err(AnnError::EmptyInput);
    }
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim: 0 });
    }
    if k == 0 {
        return Err(AnnError::InvalidK {
            k,
            n: candidates.len(),
        });
    }
    if query.len() != dim {
        return Err(AnnError::DimensionMismatch {
            expected: dim,
            got: query.len(),
        });
    }
    if !corpus.len().is_multiple_of(dim) {
        return Err(AnnError::DimensionMismatch {
            expected: (corpus.len() / dim) * dim,
            got: corpus.len(),
        });
    }
    let n_corpus = corpus.len() / dim;

    // De-duplicate candidate ids while preserving first-seen order.
    let mut seen = vec![false; n_corpus];
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    for &id in candidates {
        if id >= n_corpus {
            return Err(AnnError::IdOutOfRange { id, n: n_corpus });
        }
        if seen[id] {
            continue;
        }
        seen[id] = true;
        let x = &corpus[id * dim..(id + 1) * dim];
        let score = match metric {
            RerankMetric::L2Squared => l2_sq(query, x),
            RerankMetric::NegInnerProduct => neg_ip(query, x),
        };
        scored.push((id, score));
    }

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

/// Re-rank candidates that already carry an approximate score, keeping only the
/// best `rerank_depth` by approximate score before paying for the exact metric.
///
/// This is the production pattern: the first stage may return *thousands* of
/// candidates ranked by a cheap proxy; re-scoring all of them exactly is
/// wasteful.  We first keep the `rerank_depth` most promising candidates (by the
/// supplied approximate score, ascending) and only then compute exact distances
/// on that shortlist.  Returns the exact top-`k`.
///
/// `scored_candidates` is a slice of `(id, approx_score)` where a **smaller**
/// approximate score means more promising.
///
/// # Errors
/// Same as [`rerank_exact`]; additionally [`AnnError::Internal`] when
/// `rerank_depth == 0`.
pub fn rerank_two_stage(
    query: &[f32],
    corpus: &[f32],
    dim: usize,
    scored_candidates: &[(usize, f32)],
    rerank_depth: usize,
    k: usize,
    metric: RerankMetric,
) -> AnnResult<Vec<(usize, f32)>> {
    if rerank_depth == 0 {
        return Err(AnnError::Internal {
            msg: "rerank: rerank_depth must be ≥ 1".to_string(),
        });
    }
    // Sort candidates by approximate score and keep the best `rerank_depth`.
    let mut shortlist: Vec<(usize, f32)> = scored_candidates.to_vec();
    shortlist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    shortlist.truncate(rerank_depth);

    let ids: Vec<usize> = shortlist.into_iter().map(|(id, _)| id).collect();
    rerank_exact(query, corpus, dim, &ids, k, metric)
}

/// Compute recall@k of a candidate set: the fraction of the true exact top-`k`
/// ids that are present anywhere in `candidates`.
///
/// This measures the **ceiling** of a two-stage pipeline — re-ranking can only
/// return neighbours that the first stage surfaced, so the pipeline's recall@k
/// equals this value when `rerank_depth ≥ |candidates|`.
///
/// `truth_topk` is the exact top-`k` id list (e.g. from a brute-force oracle).
///
/// # Errors
/// [`AnnError::InvalidK`] when `truth_topk` is empty.
pub fn candidate_recall(candidates: &[usize], truth_topk: &[usize]) -> AnnResult<f32> {
    if truth_topk.is_empty() {
        return Err(AnnError::InvalidK { k: 0, n: 0 });
    }
    let cand_set: std::collections::HashSet<usize> = candidates.iter().copied().collect();
    let hits = truth_topk.iter().filter(|id| cand_set.contains(id)).count();
    Ok(hits as f32 / truth_topk.len() as f32)
}

/// Brute-force exact top-`k` over a full corpus, used as the oracle for recall
/// measurement and as a stand-in first stage in tests.
///
/// Returns `(id, score)` ascending by `score` under the chosen metric.
///
/// # Errors
/// Same shape/validity errors as [`rerank_exact`].
pub fn exact_topk(
    query: &[f32],
    corpus: &[f32],
    dim: usize,
    k: usize,
    metric: RerankMetric,
) -> AnnResult<Vec<(usize, f32)>> {
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim: 0 });
    }
    if !corpus.len().is_multiple_of(dim) {
        return Err(AnnError::DimensionMismatch {
            expected: (corpus.len() / dim) * dim,
            got: corpus.len(),
        });
    }
    let n = corpus.len() / dim;
    let all: Vec<usize> = (0..n).collect();
    rerank_exact(query, corpus, dim, &all, k, metric)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_f32() - 0.5).collect()
    }

    fn brute_ids(query: &[f32], corpus: &[f32], dim: usize, k: usize) -> Vec<usize> {
        exact_topk(query, corpus, dim, k, RerankMetric::L2Squared)
            .expect("exact_topk with valid parameters")
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn rerank_returns_k() {
        let dim = 6;
        let n = 50;
        let corpus = rand_data(n, dim, 1);
        let q = rand_data(1, dim, 99);
        let cands: Vec<usize> = (0..20).collect();
        let res = rerank_exact(&q, &corpus, dim, &cands, 5, RerankMetric::L2Squared)
            .expect("rerank_exact with valid parameters");
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn rerank_ascending_l2() {
        let dim = 5;
        let n = 40;
        let corpus = rand_data(n, dim, 2);
        let q = rand_data(1, dim, 88);
        let cands: Vec<usize> = (0..n).collect();
        let res = rerank_exact(&q, &corpus, dim, &cands, 10, RerankMetric::L2Squared)
            .expect("rerank_exact with valid parameters");
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "L2 not ascending");
        }
    }

    #[test]
    fn rerank_matches_brute_when_all_candidates() {
        // If the candidate set is the whole corpus, re-rank == exact top-k.
        let dim = 8;
        let n = 60;
        let corpus = rand_data(n, dim, 3);
        let q = rand_data(1, dim, 77);
        let cands: Vec<usize> = (0..n).collect();
        let res = rerank_exact(&q, &corpus, dim, &cands, 5, RerankMetric::L2Squared)
            .expect("rerank_exact with valid parameters");
        let ids: Vec<usize> = res.iter().map(|(id, _)| *id).collect();
        let truth = brute_ids(&q, &corpus, dim, 5);
        assert_eq!(ids, truth);
    }

    #[test]
    fn rerank_finds_self_at_top() {
        let dim = 6;
        let n = 50;
        let corpus = rand_data(n, dim, 4);
        let q = corpus[10 * dim..11 * dim].to_vec();
        let cands: Vec<usize> = (0..n).collect();
        let res = rerank_exact(&q, &corpus, dim, &cands, 1, RerankMetric::L2Squared)
            .expect("rerank_exact with valid parameters");
        assert_eq!(res[0].0, 10);
        assert!(res[0].1 < 1e-6);
    }

    #[test]
    fn rerank_dedups_candidates() {
        let dim = 4;
        let n = 20;
        let corpus = rand_data(n, dim, 5);
        let q = rand_data(1, dim, 33);
        // Many duplicates of a few ids.
        let cands = vec![3usize, 3, 3, 7, 7, 1, 1, 1, 1];
        let res = rerank_exact(&q, &corpus, dim, &cands, 10, RerankMetric::L2Squared)
            .expect("rerank_exact with valid parameters");
        let mut ids: Vec<usize> = res.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3, 7], "duplicates not removed");
    }

    #[test]
    fn rerank_neg_ip_ordering() {
        let dim = 5;
        let n = 30;
        let corpus = rand_data(n, dim, 6);
        let q = rand_data(1, dim, 44);
        let cands: Vec<usize> = (0..n).collect();
        let res = rerank_exact(&q, &corpus, dim, &cands, n, RerankMetric::NegInnerProduct)
            .expect("rerank_exact with valid parameters");
        // Ascending neg-IP == descending IP; verify against direct computation.
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        // Top result has the largest inner product.
        let best_ip = -res[0].1;
        for i in 0..n {
            let x = &corpus[i * dim..(i + 1) * dim];
            let ip: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            assert!(ip <= best_ip + 1e-5, "found larger IP than top result");
        }
    }

    #[test]
    fn two_stage_shortlist_limits_exact_scoring() {
        // rerank_depth caps the shortlist; results must come only from it.
        let dim = 6;
        let n = 80;
        let corpus = rand_data(n, dim, 7);
        let q = rand_data(1, dim, 55);
        // Approximate scores: assign random-ish proxy = id (so shortlist = lowest ids).
        let scored: Vec<(usize, f32)> = (0..n).map(|i| (i, i as f32)).collect();
        let res = rerank_two_stage(&q, &corpus, dim, &scored, 10, 5, RerankMetric::L2Squared)
            .expect("rerank_two_stage with valid parameters");
        // All returned ids must be < 10 (the shortlist depth).
        for (id, _) in &res {
            assert!(*id < 10, "id {id} escaped the shortlist");
        }
    }

    #[test]
    fn two_stage_recovers_truth_with_deep_rerank() {
        // With a generous rerank_depth and a candidate set containing the truth,
        // the two-stage result must equal the exact top-k.
        let dim = 8;
        let n = 100;
        let corpus = rand_data(n, dim, 8);
        let q = rand_data(1, dim, 66);
        // Proxy score = approximate (corrupted) L2 so the truth survives the shortlist.
        let mut rng = LcgRng::new(123);
        let scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let x = &corpus[i * dim..(i + 1) * dim];
                let approx = l2_sq(&q, x) + (rng.next_f32() - 0.5) * 0.01;
                (i, approx)
            })
            .collect();
        let res = rerank_two_stage(&q, &corpus, dim, &scored, 50, 5, RerankMetric::L2Squared)
            .expect("rerank_two_stage with valid parameters");
        let ids: Vec<usize> = res.iter().map(|(id, _)| *id).collect();
        let truth = brute_ids(&q, &corpus, dim, 5);
        assert_eq!(
            ids, truth,
            "deep two-stage rerank should recover exact top-k"
        );
    }

    #[test]
    fn candidate_recall_full() {
        let truth = vec![1usize, 5, 9, 3];
        let cands = vec![9usize, 1, 7, 3, 5, 11];
        let r = candidate_recall(&cands, &truth).expect("truth is non-empty");
        assert!((r - 1.0).abs() < 1e-6, "expected full recall, got {r}");
    }

    #[test]
    fn candidate_recall_half() {
        let truth = vec![1usize, 2, 3, 4];
        let cands = vec![1usize, 2, 99];
        let r = candidate_recall(&cands, &truth).expect("truth is non-empty");
        assert!((r - 0.5).abs() < 1e-6, "expected 0.5, got {r}");
    }

    #[test]
    fn candidate_recall_zero() {
        let truth = vec![1usize, 2, 3];
        let cands = vec![10usize, 20];
        let r = candidate_recall(&cands, &truth).expect("truth is non-empty");
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn pipeline_recall_equals_candidate_recall() {
        // The end-to-end recall of a deep two-stage pipeline equals the
        // candidate set's recall ceiling.
        let dim = 6;
        let n = 120;
        let corpus = rand_data(n, dim, 9);
        let q = rand_data(1, dim, 77);
        let truth = brute_ids(&q, &corpus, dim, 5);

        // First stage returns only a subset (ids divisible by 2) plus the truth
        // partially — here we deliberately drop one true neighbour.
        let mut cand_ids: Vec<usize> = (0..n).step_by(2).collect();
        cand_ids.retain(|id| *id != truth[0]); // drop one true neighbour
        let ceiling = candidate_recall(&cand_ids, &truth).expect("truth is non-empty");

        let scored: Vec<(usize, f32)> = cand_ids.iter().map(|&i| (i, 0.0)).collect();
        let res = rerank_two_stage(
            &q,
            &corpus,
            dim,
            &scored,
            cand_ids.len(),
            5,
            RerankMetric::L2Squared,
        )
        .expect("rerank_two_stage with valid parameters");
        let got_ids: Vec<usize> = res.iter().map(|(id, _)| *id).collect();
        let pipeline_recall = candidate_recall(&got_ids, &truth).expect("truth is non-empty");
        assert!(
            (pipeline_recall - ceiling).abs() < 1e-6,
            "pipeline {pipeline_recall} != ceiling {ceiling}"
        );
    }

    #[test]
    fn exact_topk_basic() {
        let dim = 4;
        let n = 30;
        let corpus = rand_data(n, dim, 10);
        let q = rand_data(1, dim, 88);
        let res = exact_topk(&q, &corpus, dim, 5, RerankMetric::L2Squared)
            .expect("exact_topk with valid parameters");
        assert_eq!(res.len(), 5);
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn rerank_err_empty_query() {
        let corpus = rand_data(10, 4, 1);
        let err = rerank_exact(&[], &corpus, 4, &[0, 1], 2, RerankMetric::L2Squared).unwrap_err();
        assert!(matches!(err, AnnError::EmptyInput));
    }

    #[test]
    fn rerank_err_k_zero() {
        let corpus = rand_data(10, 4, 1);
        let q = rand_data(1, 4, 2);
        let err = rerank_exact(&q, &corpus, 4, &[0, 1], 0, RerankMetric::L2Squared).unwrap_err();
        assert!(matches!(err, AnnError::InvalidK { .. }));
    }

    #[test]
    fn rerank_err_id_out_of_range() {
        let corpus = rand_data(10, 4, 1);
        let q = rand_data(1, 4, 3);
        let err = rerank_exact(&q, &corpus, 4, &[0, 999], 2, RerankMetric::L2Squared).unwrap_err();
        assert!(matches!(err, AnnError::IdOutOfRange { .. }));
    }

    #[test]
    fn rerank_err_dim_mismatch() {
        let corpus = rand_data(10, 4, 1);
        let err =
            rerank_exact(&[1.0, 2.0], &corpus, 4, &[0], 1, RerankMetric::L2Squared).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }

    #[test]
    fn two_stage_err_zero_depth() {
        let corpus = rand_data(10, 4, 1);
        let q = rand_data(1, 4, 4);
        let scored = vec![(0usize, 0.1f32), (1, 0.2)];
        let err =
            rerank_two_stage(&q, &corpus, 4, &scored, 0, 2, RerankMetric::L2Squared).unwrap_err();
        assert!(matches!(err, AnnError::Internal { .. }));
    }
}
