//! Beyond-accuracy recommendation metrics: diversity, coverage, and novelty.
//!
//! Accuracy metrics (NDCG, Recall, …) only measure how many relevant items a
//! ranking surfaces. Production recommenders are additionally judged on whether
//! the catalogue is well covered, whether recommendations are diverse, and
//! whether they surprise the user with long-tail (novel) items.
//!
//! All functions operate on item-id lists and (where needed) item feature
//! vectors or global popularity counts. They are pure and allocation-light.

use std::collections::HashSet;

/// Intra-list diversity (ILD): mean pairwise distance among the top-`k`
/// recommended items, where distance is `1 − cosine_similarity` between item
/// feature vectors.
///
/// `features[item * feat_dim .. ]` is the row for an item id. Returns a value in
/// `[0, 2]` (0 = all items identical direction, 2 = all anti-parallel). With
/// fewer than two items the diversity is defined as `0`.
///
/// # Errors
///
/// Returns `None` if `feat_dim == 0`, if any recommended id is out of range for
/// `features`, or if `features.len()` is not a multiple of `feat_dim`.
pub fn intra_list_diversity(
    recommended: &[usize],
    features: &[f32],
    feat_dim: usize,
    k: usize,
) -> Option<f32> {
    if feat_dim == 0 || features.len() / feat_dim * feat_dim != features.len() {
        return None;
    }
    let n_items = features.len() / feat_dim;
    let top: Vec<usize> = recommended.iter().take(k).copied().collect();
    if top.iter().any(|&i| i >= n_items) {
        return None;
    }
    if top.len() < 2 {
        return Some(0.0);
    }

    let cosine = |a: usize, b: usize| -> f32 {
        let va = &features[a * feat_dim..(a + 1) * feat_dim];
        let vb = &features[b * feat_dim..(b + 1) * feat_dim];
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for d in 0..feat_dim {
            dot += va[d] * vb[d];
            na += va[d] * va[d];
            nb += vb[d] * vb[d];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
        dot / denom
    };

    let mut sum = 0.0_f32;
    let mut pairs = 0usize;
    for a in 0..top.len() {
        for b in (a + 1)..top.len() {
            sum += 1.0 - cosine(top[a], top[b]);
            pairs += 1;
        }
    }
    Some(sum / pairs as f32)
}

/// Catalogue coverage: fraction of the `n_catalog` distinct items that appear in
/// at least one user's top-`k` recommendation list.
///
/// `rec_lists` is one ranked list per user. Returns a value in `[0, 1]`. Returns
/// `0.0` if `n_catalog == 0`.
pub fn catalog_coverage(rec_lists: &[Vec<usize>], n_catalog: usize, k: usize) -> f32 {
    if n_catalog == 0 {
        return 0.0;
    }
    let mut seen: HashSet<usize> = HashSet::new();
    for list in rec_lists {
        for &item in list.iter().take(k) {
            if item < n_catalog {
                seen.insert(item);
            }
        }
    }
    seen.len() as f32 / n_catalog as f32
}

/// Self-information novelty: the mean `-log2(p(item))` over the top-`k`
/// recommended items, where `p(item)` is the empirical interaction probability
/// derived from global `popularity` counts.
///
/// Rare (long-tail) items carry more self-information, so higher values indicate
/// more novel recommendations. Items absent from training (`popularity == 0`)
/// contribute the maximum information for the smallest representable probability.
///
/// # Errors
///
/// Returns `None` if `popularity` is empty, if its total is zero, or if any
/// recommended id is out of range.
pub fn novelty_self_information(
    recommended: &[usize],
    popularity: &[u64],
    k: usize,
) -> Option<f32> {
    if popularity.is_empty() {
        return None;
    }
    let total: u64 = popularity.iter().sum();
    if total == 0 {
        return None;
    }
    let top: Vec<usize> = recommended.iter().take(k).copied().collect();
    if top.iter().any(|&i| i >= popularity.len()) {
        return None;
    }
    if top.is_empty() {
        return Some(0.0);
    }
    let total_f = total as f32;
    let mut sum = 0.0_f32;
    for &item in &top {
        let count = popularity[item];
        // Floor probability at 1/total so unseen items get max information.
        let p = (count.max(1) as f32) / total_f;
        sum += -p.log2();
    }
    Some(sum / top.len() as f32)
}

/// Gini coefficient of the recommendation-frequency distribution across the
/// catalogue. `0` means perfectly even exposure (every item recommended equally
/// often), approaching `1` means a few items dominate exposure.
///
/// `rec_lists` is one ranked list per user; the top-`k` of each contributes to
/// per-item exposure counts over `n_catalog` items. Returns `0.0` if
/// `n_catalog == 0` or no items were recommended.
pub fn gini_index(rec_lists: &[Vec<usize>], n_catalog: usize, k: usize) -> f32 {
    if n_catalog == 0 {
        return 0.0;
    }
    let mut counts = vec![0u64; n_catalog];
    for list in rec_lists {
        for &item in list.iter().take(k) {
            if item < n_catalog {
                counts[item] += 1;
            }
        }
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    counts.sort_unstable();
    // Gini via the ordered formula:
    //   G = (Σ_j (2j - n - 1) x_j) / (n Σ x_j),  j = 1..n (1-based)
    let n = n_catalog as f64;
    let total_f = total as f64;
    let mut weighted = 0.0_f64;
    for (idx, &c) in counts.iter().enumerate() {
        let j = (idx + 1) as f64; // 1-based rank
        weighted += (2.0 * j - n - 1.0) * c as f64;
    }
    (weighted / (n * total_f)) as f32
}

/// Personalisation: the mean pairwise dissimilarity between users' top-`k`
/// recommendation lists, measured as `1 − |A ∩ B| / k`. `1.0` means every user
/// receives a unique list; `0.0` means all users get identical lists.
///
/// Returns `0.0` when there are fewer than two users or `k == 0`.
pub fn personalization(rec_lists: &[Vec<usize>], k: usize) -> f32 {
    if rec_lists.len() < 2 || k == 0 {
        return 0.0;
    }
    let top_sets: Vec<HashSet<usize>> = rec_lists
        .iter()
        .map(|list| list.iter().take(k).copied().collect())
        .collect();
    let mut sum = 0.0_f32;
    let mut pairs = 0usize;
    for a in 0..top_sets.len() {
        for b in (a + 1)..top_sets.len() {
            let inter = top_sets[a].intersection(&top_sets[b]).count();
            let denom = top_sets[a].len().max(top_sets[b].len()).max(1);
            sum += 1.0 - inter as f32 / denom as f32;
            pairs += 1;
        }
    }
    sum / pairs as f32
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ild_identical_items_zero() {
        // Two items with the same feature vector → cosine 1 → diversity 0.
        let features = vec![1.0_f32, 0.0, 1.0, 0.0];
        let d = intra_list_diversity(&[0, 1], &features, 2, 2).expect("ild");
        assert!(
            d.abs() < 1e-5,
            "identical items should have 0 diversity, got {d}"
        );
    }

    #[test]
    fn ild_orthogonal_items_one() {
        // Orthogonal vectors → cosine 0 → diversity 1.
        let features = vec![1.0_f32, 0.0, 0.0, 1.0];
        let d = intra_list_diversity(&[0, 1], &features, 2, 2).expect("ild");
        assert!(
            (d - 1.0).abs() < 1e-5,
            "orthogonal items should have diversity 1, got {d}"
        );
    }

    #[test]
    fn ild_antiparallel_items_two() {
        let features = vec![1.0_f32, 0.0, -1.0, 0.0];
        let d = intra_list_diversity(&[0, 1], &features, 2, 2).expect("ild");
        assert!(
            (d - 2.0).abs() < 1e-5,
            "anti-parallel should give 2, got {d}"
        );
    }

    #[test]
    fn ild_single_item_zero() {
        let features = vec![1.0_f32, 2.0, 3.0, 4.0];
        let d = intra_list_diversity(&[0], &features, 2, 5).expect("ild");
        assert!(d.abs() < 1e-7);
    }

    #[test]
    fn ild_out_of_range_none() {
        let features = vec![1.0_f32, 0.0];
        assert!(intra_list_diversity(&[0, 5], &features, 2, 2).is_none());
    }

    #[test]
    fn ild_bad_feat_dim_none() {
        let features = vec![1.0_f32, 0.0, 1.0];
        // feat_dim 2 does not divide len 3
        assert!(intra_list_diversity(&[0], &features, 2, 1).is_none());
        assert!(intra_list_diversity(&[0], &features, 0, 1).is_none());
    }

    #[test]
    fn coverage_full() {
        let lists = vec![vec![0usize, 1], vec![2, 3]];
        let c = catalog_coverage(&lists, 4, 2);
        assert!((c - 1.0).abs() < 1e-6, "all 4 items covered, got {c}");
    }

    #[test]
    fn coverage_half() {
        let lists = vec![vec![0usize, 1], vec![0, 1]];
        let c = catalog_coverage(&lists, 4, 2);
        assert!((c - 0.5).abs() < 1e-6, "2 of 4 items covered, got {c}");
    }

    #[test]
    fn coverage_respects_k() {
        let lists = vec![vec![0usize, 1, 2, 3]];
        let c = catalog_coverage(&lists, 4, 2); // only first 2 counted
        assert!((c - 0.5).abs() < 1e-6, "k truncation failed, got {c}");
    }

    #[test]
    fn coverage_empty_catalog_zero() {
        let lists = vec![vec![0usize]];
        assert_eq!(catalog_coverage(&lists, 0, 1), 0.0);
    }

    #[test]
    fn novelty_uniform_popularity() {
        // 4 items each with count 1 → p = 1/4 → -log2(1/4) = 2 for every item.
        let pop = vec![1u64, 1, 1, 1];
        let nov = novelty_self_information(&[0, 1], &pop, 2).expect("novelty");
        assert!((nov - 2.0).abs() < 1e-5, "expected 2 bits, got {nov}");
    }

    #[test]
    fn novelty_rare_item_higher() {
        // Popular item 0 (count 99) vs rare item 1 (count 1).
        let pop = vec![99u64, 1];
        let pop_nov = novelty_self_information(&[0], &pop, 1).expect("novelty");
        let rare_nov = novelty_self_information(&[1], &pop, 1).expect("novelty");
        assert!(
            rare_nov > pop_nov,
            "rare item must be more novel: {rare_nov} vs {pop_nov}"
        );
    }

    #[test]
    fn novelty_zero_total_none() {
        let pop = vec![0u64, 0];
        assert!(novelty_self_information(&[0], &pop, 1).is_none());
        assert!(novelty_self_information(&[0], &[], 1).is_none());
    }

    #[test]
    fn novelty_out_of_range_none() {
        let pop = vec![1u64, 2];
        assert!(novelty_self_information(&[5], &pop, 1).is_none());
    }

    #[test]
    fn gini_perfectly_even_zero() {
        // Every item recommended exactly once.
        let lists = vec![vec![0usize], vec![1], vec![2], vec![3]];
        let g = gini_index(&lists, 4, 1);
        assert!(g.abs() < 1e-5, "even exposure should give Gini 0, got {g}");
    }

    #[test]
    fn gini_concentrated_high() {
        // Only item 0 ever recommended across a catalogue of 4.
        let lists = vec![vec![0usize], vec![0], vec![0], vec![0]];
        let g = gini_index(&lists, 4, 1);
        assert!(
            g > 0.5,
            "concentrated exposure should give high Gini, got {g}"
        );
    }

    #[test]
    fn gini_empty_zero() {
        let lists: Vec<Vec<usize>> = vec![];
        assert_eq!(gini_index(&lists, 4, 1), 0.0);
        assert_eq!(gini_index(&[vec![0usize]], 0, 1), 0.0);
    }

    #[test]
    fn personalization_identical_lists_zero() {
        let lists = vec![vec![0usize, 1, 2], vec![0, 1, 2]];
        let p = personalization(&lists, 3);
        assert!(
            p.abs() < 1e-5,
            "identical lists → 0 personalization, got {p}"
        );
    }

    #[test]
    fn personalization_disjoint_lists_one() {
        let lists = vec![vec![0usize, 1, 2], vec![3, 4, 5]];
        let p = personalization(&lists, 3);
        assert!(
            (p - 1.0).abs() < 1e-5,
            "disjoint lists → 1 personalization, got {p}"
        );
    }

    #[test]
    fn personalization_single_user_zero() {
        let lists = vec![vec![0usize, 1]];
        assert_eq!(personalization(&lists, 2), 0.0);
        assert_eq!(personalization(&[vec![0usize], vec![1]], 0), 0.0);
    }
}
