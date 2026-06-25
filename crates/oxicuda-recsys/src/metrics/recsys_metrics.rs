use std::collections::HashSet;

pub fn precision_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f32 / k as f32
}

pub fn recall_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f32 / relevant.len() as f32
}

pub fn ndcg_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }
    let dcg: f32 = recommended
        .iter()
        .take(k)
        .enumerate()
        .map(|(pos, id)| {
            if relevant.contains(id) {
                1.0 / (pos as f32 + 2.0).log2()
            } else {
                0.0
            }
        })
        .sum();

    let ideal_k = k.min(relevant.len());
    let idcg: f32 = (0..ideal_k)
        .map(|pos| 1.0 / (pos as f32 + 2.0).log2())
        .sum();

    if idcg < 1e-12 {
        return 0.0;
    }
    dcg / idcg
}

pub fn map_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum_prec = 0.0_f32;

    for (pos, id) in recommended.iter().take(k).enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum_prec += hits as f32 / (pos + 1) as f32;
        }
    }

    if hits == 0 {
        return 0.0;
    }
    sum_prec / relevant.len().min(k) as f32
}

pub fn mrr(recommended: &[usize], relevant: &HashSet<usize>) -> f32 {
    for (pos, id) in recommended.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (pos + 1) as f32;
        }
    }
    0.0
}

pub fn hit_rate_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    let hit = recommended.iter().take(k).any(|id| relevant.contains(id));
    if hit { 1.0 } else { 0.0 }
}

/// AUC via Wilcoxon-Mann-Whitney statistic.
pub fn auc_score(scores: &[(f32, bool)]) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = sorted.iter().filter(|&&(_, label)| label).count();
    let n_neg = sorted.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Assign ranks (1-based), handle ties by averaging
    let n = sorted.len();
    let mut ranks = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f32 / 2.0;
        for rank in ranks.iter_mut().skip(i).take(j - i) {
            *rank = avg_rank;
        }
        i = j;
    }

    let rank_sum_pos: f32 = sorted
        .iter()
        .zip(ranks.iter())
        .filter(|&(&(_, label), _)| label)
        .map(|(_, &r)| r)
        .sum();

    let u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) as f32 / 2.0;
    u_pos / (n_pos * n_neg) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_all_ties_is_half() {
        // Every score identical ⇒ no pair is ordered ⇒ AUC must be exactly 0.5
        // via the averaged-rank tie correction.
        let scores = vec![(1.0, true), (1.0, false), (1.0, true), (1.0, false)];
        let auc = auc_score(&scores);
        assert!(
            (auc - 0.5).abs() < 1e-6,
            "all-ties AUC must be 0.5, got {auc}"
        );
    }

    #[test]
    fn auc_partial_ties_pathological() {
        // Two positives and two negatives. Positives score {1, 0} and negatives
        // score {1, 0}: the two 1.0s tie (one pos, one neg) and the two 0.0s tie
        // (one pos, one neg). By symmetry AUC = 0.5 exactly.
        let scores = vec![(1.0, true), (0.0, true), (1.0, false), (0.0, false)];
        let auc = auc_score(&scores);
        assert!(
            (auc - 0.5).abs() < 1e-6,
            "symmetric tied AUC must be 0.5, got {auc}"
        );
    }

    #[test]
    fn auc_perfect_and_inverted() {
        // Perfect separation ⇒ 1.0; fully inverted ⇒ 0.0.
        let perfect = vec![(2.0, true), (1.5, true), (1.0, false), (0.5, false)];
        assert!((auc_score(&perfect) - 1.0).abs() < 1e-6);
        let inverted = vec![(0.5, true), (1.0, true), (1.5, false), (2.0, false)];
        assert!(auc_score(&inverted).abs() < 1e-6);
    }

    #[test]
    fn auc_one_tie_breaks_perfect() {
        // One pos and one neg tie at the top; the half-credit from that tied pair
        // pulls AUC just below the perfect 1.0.
        let scores = vec![(2.0, true), (2.0, false), (1.0, false)];
        let auc = auc_score(&scores);
        // n_pos=1, n_neg=2. Ranks: the two 2.0s share avg rank (2+3)/2=2.5, the
        // 1.0 gets rank 1. rank_sum_pos = 2.5 ⇒ U = 2.5 - 1 = 1.5 ⇒ 1.5/2 = 0.75.
        assert!((auc - 0.75).abs() < 1e-6, "expected 0.75, got {auc}");
    }

    #[test]
    fn auc_single_class_returns_half() {
        let only_pos = vec![(1.0, true), (2.0, true)];
        assert!((auc_score(&only_pos) - 0.5).abs() < 1e-6);
        let only_neg = vec![(1.0, false), (2.0, false)];
        assert!((auc_score(&only_neg) - 0.5).abs() < 1e-6);
    }

    /// Reference NDCG@k using the textbook gain `1/log2(rank+1)` (1-based rank)
    /// with binary relevance — the exact definition `sklearn.metrics.ndcg_score`
    /// uses for binary relevances.
    fn reference_ndcg(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f64 {
        if k == 0 || relevant.is_empty() {
            return 0.0;
        }
        let mut dcg = 0.0_f64;
        for (pos, id) in recommended.iter().take(k).enumerate() {
            if relevant.contains(id) {
                // 1-based rank = pos + 1 ⇒ discount = 1/log2(rank+1) = 1/log2(pos+2).
                dcg += 1.0 / ((pos as f64 + 2.0).log2());
            }
        }
        let ideal = k.min(relevant.len());
        let mut idcg = 0.0_f64;
        for pos in 0..ideal {
            idcg += 1.0 / ((pos as f64 + 2.0).log2());
        }
        if idcg < 1e-12 { 0.0 } else { dcg / idcg }
    }

    #[test]
    fn ndcg_idcg_matches_reference() {
        let relevant: HashSet<usize> = [1usize, 3, 5, 7].into_iter().collect();
        let cases: [(Vec<usize>, usize); 4] = [
            (vec![1, 2, 3, 4, 5, 6, 7], 5),
            (vec![0, 1, 2, 3, 5, 7], 6),
            (vec![7, 5, 3, 1], 4),
            (vec![2, 4, 6, 8, 1], 3),
        ];
        for (rec, k) in &cases {
            let got = ndcg_at_k(rec, &relevant, *k) as f64;
            let want = reference_ndcg(rec, &relevant, *k);
            assert!(
                (got - want).abs() < 1e-5,
                "NDCG@{k} for {rec:?}: got {got}, reference {want}"
            );
        }
    }

    #[test]
    fn ndcg_idcg_denominator_handles_few_relevant() {
        // Only one relevant item but k=10: IDCG uses a single 1/log2(2)=1 term,
        // so a top-1 hit must give NDCG exactly 1.0.
        let relevant: HashSet<usize> = [4usize].into_iter().collect();
        let rec = vec![4usize, 0, 1, 2, 3];
        assert!((ndcg_at_k(&rec, &relevant, 10) - 1.0).abs() < 1e-6);
        // A hit at rank 2 gives 1/log2(3) ≈ 0.6309.
        let rec2 = vec![9usize, 4, 1, 2];
        let v = ndcg_at_k(&rec2, &relevant, 10) as f64;
        assert!((v - 1.0 / 3.0_f64.log2()).abs() < 1e-5, "got {v}");
    }
}
