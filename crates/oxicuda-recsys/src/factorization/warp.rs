//! WARP loss (Weighted Approximate-Rank Pairwise, Weston et al. 2011 AISTATS)
//! and LambdaRank learning signal (Burges et al. 2006 ICML) for learning-to-rank
//! in recommendation systems.
//!
//! WARP samples negatives until a violated triple is found and weights the loss
//! by the harmonic approximation of the positive item's rank, pushing the model
//! to correct high-rank violations more aggressively.
//!
//! LambdaRank computes per-item gradient signals by weighting pairwise violations
//! by the change in NDCG that would result from correcting each pair.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for WARP loss computation.
#[derive(Debug, Clone)]
pub struct WarpConfig {
    /// Total number of items.
    pub n_items: usize,
    /// Maximum number of negative samples before giving up on finding a violation.
    pub n_neg_samples: usize,
    /// Margin for the pairwise ranking loss. Default: 1.0.
    pub margin: f32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self {
            n_items: 0,
            n_neg_samples: 100,
            margin: 1.0,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result of a WARP loss computation over a batch of (user, pos_item) pairs.
#[derive(Debug, Clone)]
pub struct WarpResult {
    /// Average WARP loss over the batch.
    pub loss: f32,
    /// Number of violated triples found.
    pub n_violated: usize,
    /// Average approximate rank of the positive items (when violations occurred).
    pub avg_rank: f32,
}

// ── Main WARP functions ───────────────────────────────────────────────────────

/// Compute WARP loss for a batch of (user, pos_item) pairs.
///
/// `scores` is a matrix of shape `n_users × n_items` (row-major): `scores[u][i]`
/// is the model score for item `i` for user `u`.
///
/// `pos_items[u]` is the index of the positive item for user `u`.
///
/// For each user the algorithm samples negative items uniformly at random until
/// either a violation is found (`score[neg] > score[pos] - margin`) or the
/// negative sample budget `cfg.n_neg_samples` is exhausted.  When a violation is
/// found at trial `t` (1-indexed):
///
/// ```text
/// rank_approx   = floor((n_items - 1) / t)
/// warp_weight   = H_{rank_approx}  (harmonic number)
/// loss_u        = warp_weight * (margin - pos_score + neg_score)
/// ```
///
/// # Errors
///
/// Returns [`RecsysError::InvalidNumItems`] if `cfg.n_items == 0`,
/// [`RecsysError::ItemOutOfBounds`] if a `pos_item` index ≥ `cfg.n_items`,
/// and [`RecsysError::DimensionMismatch`] if `scores.len()` ≠ `n_users * n_items`.
pub fn warp_loss(
    scores: &[f32],
    pos_items: &[usize],
    cfg: &WarpConfig,
    rng: &mut LcgRng,
) -> RecsysResult<WarpResult> {
    let n_users = pos_items.len();
    let n_items = cfg.n_items;

    if n_items == 0 {
        return Err(RecsysError::InvalidNumItems { n: n_items });
    }

    // Zero-user case: return a neutral result immediately.
    if n_users == 0 {
        return Ok(WarpResult {
            loss: 0.0,
            n_violated: 0,
            avg_rank: 0.0,
        });
    }

    let expected = n_users * n_items;
    if scores.len() != expected {
        return Err(RecsysError::DimensionMismatch {
            expected,
            got: scores.len(),
        });
    }

    // Validate all pos_item indices upfront.
    for (u, &pos) in pos_items.iter().enumerate() {
        if pos >= n_items {
            return Err(RecsysError::ItemOutOfBounds {
                idx: pos,
                n: n_items,
            });
        }
        let _ = u;
    }

    let mut total_loss: f32 = 0.0;
    let mut n_violated: usize = 0;
    let mut rank_sum: f32 = 0.0;

    for u in 0..n_users {
        let pos_item = pos_items[u];
        let user_scores = &scores[u * n_items..(u + 1) * n_items];
        let pos_score = user_scores[pos_item];

        // Sample negatives until violation or budget exhausted.
        let mut violated = false;
        for trial in 1..=cfg.n_neg_samples {
            // Uniform random negative item ≠ pos_item.
            let neg_item = sample_neg_item(rng, n_items, pos_item);
            let neg_score = user_scores[neg_item];

            if neg_score > pos_score - cfg.margin {
                // Violation found.
                let rank_approx = (n_items - 1) / trial;
                let warp_weight = harmonic_number(rank_approx);
                let violation = cfg.margin - pos_score + neg_score;
                total_loss += warp_weight * violation.max(0.0);
                n_violated += 1;
                rank_sum += rank_approx as f32;
                violated = true;
                break;
            }
        }
        let _ = violated;
    }

    let avg_loss = total_loss / n_users as f32;
    let avg_rank = if n_violated > 0 {
        rank_sum / n_violated as f32
    } else {
        0.0
    };

    Ok(WarpResult {
        loss: avg_loss,
        n_violated,
        avg_rank,
    })
}

/// Compute the WARP gradient vectors for a single violated (pos, neg, weight) triple.
///
/// Returns `(pos_grad, neg_grad)` where each vector has length `n_items`.
/// `pos_grad[pos_item] = +warp_weight`, `neg_grad[neg_item] = -warp_weight`,
/// all other entries are zero.
pub fn warp_triple_gradient(
    pos_item: usize,
    neg_item: usize,
    warp_weight: f32,
    n_items: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut pos_grad = vec![0.0_f32; n_items];
    let mut neg_grad = vec![0.0_f32; n_items];
    if pos_item < n_items {
        pos_grad[pos_item] = warp_weight;
    }
    if neg_item < n_items {
        neg_grad[neg_item] = -warp_weight;
    }
    (pos_grad, neg_grad)
}

/// Harmonic number H_k = Σ_{i=1}^{k} 1/i.
///
/// H_0 = 0, H_1 = 1, H_2 = 1.5, etc.
///
/// The naive loop is efficient for the item-count scales used in typical
/// recommendation systems (k < 10^6).  For k = 0 we return 0.0 by convention.
pub fn harmonic_number(k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let mut h = 0.0_f32;
    for i in 1..=k {
        h += 1.0 / i as f32;
    }
    h
}

// ── LambdaRank ────────────────────────────────────────────────────────────────

/// Compute LambdaRank per-item gradient weights (Burges et al. 2006 ICML).
///
/// For each pair `(i, j)` where `relevance[i] > relevance[j]` but
/// `scores[i] < scores[j]` (a ranking violation):
///
/// ```text
/// ΔNDCG_ij = |1/log2(rank_i+2) - 1/log2(rank_j+2)| * |rel_i - rel_j| / IDCG
/// lambda_ij = ΔNDCG_ij * sigmoid(score_j - score_i)
/// ```
///
/// The net lambda for item `i` is:
/// ```text
/// lambda[i] = Σ_{j: violated above} lambda_ij  −  Σ_{j: violated below} lambda_ji
/// ```
///
/// # Errors
///
/// Returns [`RecsysError::EmptyInput`] when `n_items == 0`.
pub fn lambda_rank_weights(
    scores: &[f32],
    relevance: &[f32],
    n_items: usize,
) -> RecsysResult<Vec<f32>> {
    if n_items == 0 {
        return Err(RecsysError::EmptyInput);
    }
    if scores.len() != n_items || relevance.len() != n_items {
        return Err(RecsysError::DimensionMismatch {
            expected: n_items,
            got: scores.len().min(relevance.len()),
        });
    }

    // ── Sort items by descending score → rank[item] = 0-based position ────
    let mut order: Vec<usize> = (0..n_items).collect();
    order.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0usize; n_items];
    for (pos, &item) in order.iter().enumerate() {
        rank[item] = pos;
    }

    // ── IDCG: ideal DCG (items sorted by descending relevance) ────────────
    let mut sorted_rel: Vec<f32> = relevance.to_vec();
    sorted_rel.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f32 = sorted_rel
        .iter()
        .enumerate()
        .map(|(pos, &rel)| rel / (pos as f32 + 2.0).log2())
        .sum();

    let mut lambdas = vec![0.0_f32; n_items];

    if idcg <= 0.0 {
        // All relevances are zero: no gradient signal.
        return Ok(lambdas);
    }

    // ── Pairwise loop (O(n²) — typical recsys lists are short) ────────────
    for i in 0..n_items {
        for j in 0..n_items {
            if i == j {
                continue;
            }
            // Violation: i should rank above j but doesn't.
            if relevance[i] > relevance[j] && scores[i] < scores[j] {
                let rank_i = rank[i] as f32;
                let rank_j = rank[j] as f32;
                // |ΔNDCG| approximation: swap discount difference × |Δrel| / IDCG
                let disc_i = 1.0 / (rank_i + 2.0).log2();
                let disc_j = 1.0 / (rank_j + 2.0).log2();
                let delta_ndcg =
                    (disc_i - disc_j).abs() * (relevance[i] - relevance[j]).abs() / idcg;
                let lambda_ij = delta_ndcg * sigmoid(scores[j] - scores[i]);
                lambdas[i] += lambda_ij;
                lambdas[j] -= lambda_ij;
            }
        }
    }

    Ok(lambdas)
}

// ── NDCG ─────────────────────────────────────────────────────────────────────

/// Compute NDCG@k from an already-ranked relevance list.
///
/// `ranked_relevance[i]` is the relevance of the item at rank `i+1` (0-indexed).
///
/// Returns NDCG@k ∈ [0, 1]. Returns 0.0 if IDCG = 0 (all relevances are 0).
pub fn ndcg_at_k_from_ranked(ranked_relevance: &[f32], k: usize) -> f32 {
    if k == 0 || ranked_relevance.is_empty() {
        return 0.0;
    }
    let k = k.min(ranked_relevance.len());

    // DCG@k: discount by log2(rank + 2) where rank is 0-indexed
    let dcg: f32 = ranked_relevance[..k]
        .iter()
        .enumerate()
        .map(|(pos, &rel)| rel / (pos as f32 + 2.0).log2())
        .sum();

    // IDCG@k: ideal ordering (descending)
    let mut ideal = ranked_relevance[..k].to_vec();
    ideal.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f32 = ideal
        .iter()
        .enumerate()
        .map(|(pos, &rel)| rel / (pos as f32 + 2.0).log2())
        .sum();

    if idcg <= 0.0 { 0.0 } else { dcg / idcg }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Sigmoid function σ(x) = 1 / (1 + e^{-x}).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sample a uniformly random item index in [0, n_items) that is ≠ pos_item.
/// Retries until a different item is chosen (expected ≤ 2 draws for n_items ≥ 2).
#[inline]
fn sample_neg_item(rng: &mut LcgRng, n_items: usize, pos_item: usize) -> usize {
    loop {
        let candidate = rng.next_usize(n_items);
        if candidate != pos_item {
            return candidate;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn make_cfg(n_items: usize) -> WarpConfig {
        WarpConfig {
            n_items,
            n_neg_samples: 200,
            margin: 1.0,
        }
    }

    // ── WARP loss ──────────────────────────────────────────────────────────

    #[test]
    fn warp_loss_empty_batch() {
        let mut rng = make_rng();
        let cfg = make_cfg(5);
        let result = warp_loss(&[], &[], &cfg, &mut rng).expect("warp_loss should succeed");
        assert_eq!(result.n_violated, 0);
        assert_eq!(result.loss, 0.0);
    }

    #[test]
    fn warp_loss_no_violation() {
        // pos_score = 10.0, all negatives score = 0.0  →  no violation
        let mut rng = make_rng();
        let n_items = 5;
        let cfg = make_cfg(n_items);
        // 1 user, 5 items: item 0 scores 10, rest 0
        let scores = vec![10.0_f32, 0.0, 0.0, 0.0, 0.0];
        let pos_items = vec![0usize];
        let result =
            warp_loss(&scores, &pos_items, &cfg, &mut rng).expect("warp_loss should succeed");
        assert_eq!(result.n_violated, 0, "No violation expected");
        assert_eq!(result.loss, 0.0);
    }

    #[test]
    fn warp_loss_violation_occurs() {
        // pos_score = 0.0, neg_score = 2.0 → immediate violation on first draw
        let mut rng = make_rng();
        let n_items = 5;
        let cfg = make_cfg(n_items);
        // item 0 = pos with score 0, rest score 2
        let scores = vec![0.0_f32, 2.0, 2.0, 2.0, 2.0];
        let pos_items = vec![0usize];
        let result =
            warp_loss(&scores, &pos_items, &cfg, &mut rng).expect("warp_loss should succeed");
        assert!(result.n_violated >= 1, "Expected at least one violation");
    }

    #[test]
    fn warp_loss_correct_margin() {
        // Single user, single violation: the loss must equal warp_weight * violation
        let mut rng = make_rng();
        let n_items = 2;
        // item 0 = pos (score 0), item 1 = neg (score 2) → immediate violation
        let scores = vec![0.0_f32, 2.0];
        let pos_items = vec![0usize];
        let cfg = WarpConfig {
            n_items,
            n_neg_samples: 200,
            margin: 1.0,
        };
        let result =
            warp_loss(&scores, &pos_items, &cfg, &mut rng).expect("warp_loss should succeed");
        // rank_approx = (2-1)/1 = 1; H(1)=1.0; violation = 1-0+2=3
        let expected_loss = harmonic_number(1) * (1.0 - 0.0 + 2.0);
        assert!(
            (result.loss - expected_loss).abs() < 1e-5,
            "Expected loss {expected_loss}, got {}",
            result.loss
        );
    }

    // ── Harmonic number ────────────────────────────────────────────────────

    #[test]
    fn harmonic_number_k1_is_1() {
        assert!((harmonic_number(1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn harmonic_number_k2_is_1_5() {
        assert!((harmonic_number(2) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn harmonic_number_k0_is_0() {
        assert_eq!(harmonic_number(0), 0.0);
    }

    // ── WARP triple gradient ───────────────────────────────────────────────

    #[test]
    fn warp_triple_gradient_correct_size() {
        let n = 10;
        let (pg, ng) = warp_triple_gradient(2, 7, 1.5, n);
        assert_eq!(pg.len(), n);
        assert_eq!(ng.len(), n);
    }

    #[test]
    fn warp_triple_gradient_pos_positive() {
        let (pg, _) = warp_triple_gradient(3, 6, 2.0, 10);
        assert!((pg[3] - 2.0).abs() < 1e-7, "pos_grad[3] should be 2.0");
    }

    #[test]
    fn warp_triple_gradient_neg_negative() {
        let (_, ng) = warp_triple_gradient(3, 6, 2.0, 10);
        assert!((ng[6] + 2.0).abs() < 1e-7, "neg_grad[6] should be -2.0");
    }

    #[test]
    fn warp_triple_gradient_other_zero() {
        let (pg, ng) = warp_triple_gradient(1, 5, 1.0, 8);
        for (i, (&p, &n_val)) in pg.iter().zip(ng.iter()).enumerate() {
            if i != 1 {
                assert_eq!(p, 0.0, "pos_grad[{i}] should be 0");
            }
            if i != 5 {
                assert_eq!(n_val, 0.0, "neg_grad[{i}] should be 0");
            }
        }
    }

    #[test]
    fn warp_loss_deterministic_seed() {
        let n_items = 5;
        let cfg = make_cfg(n_items);
        let scores = vec![0.0_f32, 2.0, 0.5, 1.5, 0.3];
        let pos_items = vec![0usize];

        let mut rng1 = LcgRng::new(99);
        let mut rng2 = LcgRng::new(99);
        let r1 = warp_loss(&scores, &pos_items, &cfg, &mut rng1).expect("warp_loss should succeed");
        let r2 = warp_loss(&scores, &pos_items, &cfg, &mut rng2).expect("warp_loss should succeed");
        assert_eq!(r1.loss, r2.loss, "Same seed must produce same loss");
        assert_eq!(r1.n_violated, r2.n_violated);
    }

    #[test]
    fn warp_avg_rank_positive_when_violations() {
        let mut rng = make_rng();
        let n_items = 5;
        let cfg = make_cfg(n_items);
        let scores = vec![0.0_f32, 2.0, 2.0, 2.0, 2.0];
        let pos_items = vec![0usize];
        let result =
            warp_loss(&scores, &pos_items, &cfg, &mut rng).expect("warp_loss should succeed");
        if result.n_violated > 0 {
            assert!(result.avg_rank >= 0.0, "avg_rank should be ≥ 0");
        }
    }

    // ── LambdaRank ─────────────────────────────────────────────────────────

    #[test]
    fn lambda_rank_empty() {
        let result = lambda_rank_weights(&[], &[], 0);
        assert!(
            matches!(result, Err(RecsysError::EmptyInput)),
            "Expected EmptyInput error"
        );
    }

    #[test]
    fn lambda_rank_correct_direction() {
        // Item 0: high relevance (1.0) but low score (0.0)  → should get positive lambda
        // Item 1: low relevance (0.0) but high score (1.0)
        let scores = vec![0.0_f32, 1.0];
        let relevance = vec![1.0_f32, 0.0];
        let lambdas = lambda_rank_weights(&scores, &relevance, 2)
            .expect("lambda_rank_weights should succeed");
        assert!(
            lambdas[0] > 0.0,
            "High-relevance item at low rank should get positive lambda, got {}",
            lambdas[0]
        );
        assert!(
            lambdas[1] < 0.0,
            "Low-relevance item at high rank should get negative lambda, got {}",
            lambdas[1]
        );
    }

    #[test]
    fn lambda_rank_zero_for_perfect_ranking() {
        // Items already sorted by relevance: highest relevance at highest score.
        // No pairwise violations → all lambdas should be 0.
        let scores = vec![3.0_f32, 2.0, 1.0, 0.0];
        let relevance = vec![1.0_f32, 0.8, 0.5, 0.0];
        let lambdas = lambda_rank_weights(&scores, &relevance, 4)
            .expect("lambda_rank_weights should succeed");
        for (i, &l) in lambdas.iter().enumerate() {
            assert!(
                l.abs() < 1e-7,
                "lambda[{i}] = {l} should be 0 for perfect ranking"
            );
        }
    }

    // ── NDCG ───────────────────────────────────────────────────────────────

    #[test]
    fn ndcg_at_k_perfect_ranking() {
        // Ranked relevance already in ideal order at k=2
        let ranked = vec![1.0_f32, 1.0, 0.0, 0.0];
        let ndcg = ndcg_at_k_from_ranked(&ranked, 2);
        assert!(
            (ndcg - 1.0).abs() < 1e-5,
            "Perfect ranking should have NDCG=1.0, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_at_k_worst_ranking() {
        // Reverse relevance: high-relevance items at the bottom
        let ranked = vec![0.0_f32, 0.0, 1.0, 1.0];
        let ndcg = ndcg_at_k_from_ranked(&ranked, 4);
        assert!(
            ndcg < 1.0,
            "Reversed ranking should have NDCG < 1.0, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_at_k_all_zero_relevance() {
        let ranked = vec![0.0_f32, 0.0, 0.0];
        let ndcg = ndcg_at_k_from_ranked(&ranked, 3);
        assert_eq!(ndcg, 0.0, "All-zero relevance must give NDCG=0");
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn err_pos_item_out_of_bounds() {
        let mut rng = make_rng();
        let cfg = make_cfg(3);
        let scores = vec![1.0_f32, 2.0, 3.0];
        let pos_items = vec![5usize]; // out of bounds
        let result = warp_loss(&scores, &pos_items, &cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::ItemOutOfBounds { .. })),
            "Expected ItemOutOfBounds error"
        );
    }
}
