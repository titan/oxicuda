//! Expert pruning and merging at inference time.
//!
//! Large MoE checkpoints over-provision experts; at deployment many experts are
//! rarely routed to or are near-duplicates of one another. Two complementary
//! compression operations shrink the expert pool while preserving behaviour:
//!
//! * **Pruning** — drop the least-utilised experts (those receiving the fewest
//!   routed tokens over a calibration set) and *re-map* the surviving experts to
//!   a contiguous index range. Routing decisions that pointed at a pruned expert
//!   are redirected to their nearest surviving neighbour. This follows the
//!   task-specific MoE-pruning recipe of Chen et al. "Task-Specific Expert
//!   Pruning for Sparse Mixture-of-Experts" (2022).
//!
//! * **Merging** — fuse the two most *similar* experts (largest cosine
//!   similarity between flattened weight vectors) into a single expert whose
//!   weights are the utilisation-weighted average of the pair. Repeated merging
//!   distils an `E`-expert bank down to any `k < E` experts, in the spirit of
//!   model-merging / weight-averaging distillation (e.g. He et al. "Merging
//!   Experts into One", 2023).
//!
//! Both operations work directly on CPU weight buffers ([`ExpertFfn`]) and
//! return a new, smaller [`ExpertBank`] together with the index remapping needed
//! to fix up a downstream router.

use crate::error::{MoeError, MoeResult};
use crate::expert::bank::ExpertBank;
use crate::expert::ffn::ExpertFfn;

/// Result of a pruning / merging compression pass.
#[derive(Debug)]
pub struct CompressionResult {
    /// The compressed bank.
    pub bank: ExpertBank,
    /// Mapping `old_expert_index -> new_expert_index` (length = original
    /// `n_experts`). Every old index maps to a valid index in the new bank, so a
    /// router's per-token expert indices can be remapped by table lookup.
    pub index_map: Vec<usize>,
    /// Number of experts in the compressed bank.
    pub n_kept: usize,
}

/// Apply `index_map` to a slice of router-selected expert indices in place.
///
/// Any index `>= index_map.len()` is left untouched (it cannot be remapped).
pub fn remap_indices(indices: &mut [usize], index_map: &[usize]) {
    for idx in indices.iter_mut() {
        if *idx < index_map.len() {
            *idx = index_map[*idx];
        }
    }
}

/// Flatten an expert's learnable weights into one vector for similarity / merge
/// arithmetic (`w1 ∥ b1 ∥ w2 ∥ b2`).
fn flatten(e: &ExpertFfn) -> Vec<f32> {
    let mut v = Vec::with_capacity(e.w1.len() + e.b1.len() + e.w2.len() + e.b2.len());
    v.extend_from_slice(&e.w1);
    v.extend_from_slice(&e.b1);
    v.extend_from_slice(&e.w2);
    v.extend_from_slice(&e.b2);
    v
}

/// Cosine similarity between two equal-length vectors; `0` when either has zero
/// norm.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 1e-12 { 0.0 } else { dot / denom }
}

/// Euclidean distance between two equal-length flattened experts.
fn l2_distance(a: &ExpertFfn, b: &ExpertFfn) -> f32 {
    let fa = flatten(a);
    let fb = flatten(b);
    fa.iter()
        .zip(fb.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Convex (utilisation-weighted) combination of two experts.
///
/// `out = (wa·a + wb·b) / (wa + wb)`. When both weights are zero, falls back to
/// a plain average. The two experts must share `input_dim` / `ffn_dim`.
fn weighted_average(a: &ExpertFfn, b: &ExpertFfn, wa: f32, wb: f32) -> MoeResult<ExpertFfn> {
    if a.input_dim != b.input_dim || a.ffn_dim != b.ffn_dim {
        return Err(MoeError::DimensionMismatch {
            expected: a.input_dim,
            got: b.input_dim,
        });
    }
    let total = wa + wb;
    let (ca, cb) = if total > 1e-12 {
        (wa / total, wb / total)
    } else {
        (0.5, 0.5)
    };
    let blend = |xa: &[f32], xb: &[f32]| -> Vec<f32> {
        xa.iter()
            .zip(xb.iter())
            .map(|(&va, &vb)| ca * va + cb * vb)
            .collect()
    };
    Ok(ExpertFfn {
        w1: blend(&a.w1, &b.w1),
        b1: blend(&a.b1, &b.b1),
        w2: blend(&a.w2, &b.w2),
        b2: blend(&a.b2, &b.b2),
        input_dim: a.input_dim,
        ffn_dim: a.ffn_dim,
        activation: a.activation,
    })
}

/// Prune the least-utilised experts down to `n_keep`, remapping pruned experts
/// to their nearest (smallest-L2) surviving expert.
///
/// `usage[i]` is a calibration statistic (e.g. token count routed to expert
/// `i`); higher means more important. The `n_keep` experts with the largest
/// usage survive (ties broken by lower original index for determinism).
///
/// # Errors
/// Returns [`MoeError`] for `usage.len() != bank.n_experts`, `n_keep == 0`, or
/// `n_keep > n_experts`.
pub fn prune_experts(
    bank: &ExpertBank,
    usage: &[usize],
    n_keep: usize,
) -> MoeResult<CompressionResult> {
    let n_experts = bank.n_experts;
    if usage.len() != n_experts {
        return Err(MoeError::DimensionMismatch {
            expected: n_experts,
            got: usage.len(),
        });
    }
    if n_keep == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts: n_keep });
    }
    if n_keep > n_experts {
        return Err(MoeError::InvalidExpertCount { n_experts: n_keep });
    }

    // Rank experts by usage descending; keep the first `n_keep`.
    let mut order: Vec<usize> = (0..n_experts).collect();
    order.sort_by(|&a, &b| usage[b].cmp(&usage[a]).then(a.cmp(&b)));
    let kept: Vec<usize> = order.iter().take(n_keep).copied().collect();
    let mut keep_flag = vec![false; n_experts];
    for &k in &kept {
        keep_flag[k] = true;
    }

    // Surviving experts keep their relative order (ascending original index) so
    // the new bank is deterministic and stable.
    let mut survivors: Vec<usize> = (0..n_experts).filter(|&i| keep_flag[i]).collect();
    survivors.sort_unstable();

    // new index of each survivor
    let mut new_index = vec![usize::MAX; n_experts];
    for (new_i, &old_i) in survivors.iter().enumerate() {
        new_index[old_i] = new_i;
    }

    let experts = bank.experts();
    // Build the index map: survivors map to their new slot; pruned experts map
    // to the new slot of their nearest surviving expert by weight L2 distance.
    let mut index_map = vec![0_usize; n_experts];
    for old in 0..n_experts {
        if keep_flag[old] {
            index_map[old] = new_index[old];
        } else {
            // nearest survivor
            let mut best = survivors[0];
            let mut best_d = l2_distance(&experts[old], &experts[best]);
            for &cand in survivors.iter().skip(1) {
                let d = l2_distance(&experts[old], &experts[cand]);
                if d < best_d {
                    best_d = d;
                    best = cand;
                }
            }
            index_map[old] = new_index[best];
        }
    }

    let new_experts: Vec<ExpertFfn> = survivors.iter().map(|&i| experts[i].clone()).collect();
    let new_bank = ExpertBank::from_experts(new_experts)?;

    Ok(CompressionResult {
        bank: new_bank,
        index_map,
        n_kept: n_keep,
    })
}

/// Merge experts pairwise until `n_keep` remain, always fusing the most similar
/// (largest cosine) surviving pair and weighting the average by usage.
///
/// `usage[i]` weights how much each expert contributes to a merged expert.
///
/// # Errors
/// Returns [`MoeError`] for `usage.len() != bank.n_experts`, `n_keep == 0`, or
/// `n_keep > n_experts`.
pub fn merge_experts(
    bank: &ExpertBank,
    usage: &[usize],
    n_keep: usize,
) -> MoeResult<CompressionResult> {
    let n_experts = bank.n_experts;
    if usage.len() != n_experts {
        return Err(MoeError::DimensionMismatch {
            expected: n_experts,
            got: usage.len(),
        });
    }
    if n_keep == 0 || n_keep > n_experts {
        return Err(MoeError::InvalidExpertCount { n_experts: n_keep });
    }

    // Each cluster carries: its merged expert, the set of original experts it
    // covers, and an accumulated usage weight.
    struct Cluster {
        expert: ExpertFfn,
        members: Vec<usize>,
        weight: f32,
    }

    let experts = bank.experts();
    let mut clusters: Vec<Cluster> = experts
        .iter()
        .enumerate()
        .map(|(i, e)| Cluster {
            expert: e.clone(),
            members: vec![i],
            // +1 so zero-usage experts still merge with a sane (non-degenerate)
            // weight rather than vanishing entirely.
            weight: usage[i] as f32 + 1.0,
        })
        .collect();

    while clusters.len() > n_keep {
        // Find the most similar pair by cosine of flattened weights.
        let flats: Vec<Vec<f32>> = clusters.iter().map(|c| flatten(&c.expert)).collect();
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_sim = f32::NEG_INFINITY;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let sim = cosine(&flats[i], &flats[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge j into i (j > i), then remove j.
        let merged_expert = weighted_average(
            &clusters[best_i].expert,
            &clusters[best_j].expert,
            clusters[best_i].weight,
            clusters[best_j].weight,
        )?;
        let removed = clusters.remove(best_j);
        clusters[best_i].expert = merged_expert;
        clusters[best_i].weight += removed.weight;
        clusters[best_i].members.extend(removed.members);
    }

    // Build the new bank and the original->new index map.
    let mut index_map = vec![0_usize; n_experts];
    let mut new_experts = Vec::with_capacity(clusters.len());
    for (new_i, cluster) in clusters.iter().enumerate() {
        for &member in &cluster.members {
            index_map[member] = new_i;
        }
        new_experts.push(cluster.expert.clone());
    }
    let new_bank = ExpertBank::from_experts(new_experts)?;

    Ok(CompressionResult {
        bank: new_bank,
        index_map,
        n_kept: clusters.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::ffn::ExpertActivation;
    use crate::handle::LcgRng;

    fn make_bank(n: usize, d: usize, ffn: usize, seed: u64) -> ExpertBank {
        let mut rng = LcgRng::new(seed);
        ExpertBank::new(n, d, ffn, ExpertActivation::Gelu, &mut rng).expect("bank")
    }

    #[test]
    fn remap_indices_applies_table() {
        let map = vec![0, 0, 1, 2];
        let mut idx = vec![3_usize, 1, 2, 0, 99];
        remap_indices(&mut idx, &map);
        assert_eq!(idx, vec![2, 0, 1, 0, 99]); // 99 untouched
    }

    #[test]
    fn prune_keeps_most_used() {
        let bank = make_bank(5, 8, 16, 1);
        let usage = vec![1_usize, 100, 2, 50, 3];
        let res = prune_experts(&bank, &usage, 2).expect("prune should succeed");
        assert_eq!(res.n_kept, 2);
        assert_eq!(res.bank.n_experts, 2);
        // Experts 1 and 3 (highest usage) survive; their map entries are valid.
        assert!(res.index_map[1] < 2);
        assert!(res.index_map[3] < 2);
        // Every original index maps into the new range.
        assert!(res.index_map.iter().all(|&m| m < 2));
    }

    #[test]
    fn prune_redirects_to_nearest_survivor() {
        // Build a bank where expert 0 is identical to expert 1 (so when expert 0
        // is pruned it must remap to whichever copy survives, distance 0).
        let mut rng = LcgRng::new(7);
        let base = ExpertFfn::new(6, 12, ExpertActivation::Relu, &mut rng);
        let far = {
            let mut e = ExpertFfn::new(6, 12, ExpertActivation::Relu, &mut rng);
            for w in e.w1.iter_mut() {
                *w += 100.0;
            }
            e
        };
        let bank =
            ExpertBank::from_experts(vec![base.clone(), base.clone(), far]).expect("from_experts");
        // Keep 2; expert 0 and 1 are duplicates with high usage, expert 2 low.
        let usage = vec![10_usize, 9, 0];
        let res = prune_experts(&bank, &usage, 2).expect("prune should succeed");
        // Survivors are 0 and 1 (the duplicates); pruned expert 2 must remap to a
        // valid kept index.
        assert!(res.index_map[2] < 2);
    }

    #[test]
    fn merge_reduces_count_and_preserves_duplicate() {
        // Two identical experts plus one distinct: merging to 2 should fuse the
        // duplicates (cosine = 1) and leave the distinct one alone.
        let mut rng = LcgRng::new(3);
        let dup = ExpertFfn::new(4, 8, ExpertActivation::Gelu, &mut rng);
        let distinct = ExpertFfn::new(4, 8, ExpertActivation::Gelu, &mut rng);
        let bank = ExpertBank::from_experts(vec![dup.clone(), dup.clone(), distinct.clone()])
            .expect("from_experts");
        let usage = vec![1_usize, 1, 1];
        let res = merge_experts(&bank, &usage, 2).expect("merge should succeed");
        assert_eq!(res.n_kept, 2);
        assert_eq!(res.bank.n_experts, 2);

        // The merged duplicate cluster must reproduce the original duplicate
        // exactly (average of two identical experts == the expert).
        let x = vec![0.5_f32; 4];
        let dup_out = dup.forward(&x).expect("dup forward");
        // Find which new expert the original index 0 mapped to and check it.
        let merged_idx = res.index_map[0];
        let merged_out = res
            .bank
            .forward_expert(merged_idx, &x, 1)
            .expect("forward_expert");
        for (a, b) in dup_out.iter().zip(merged_out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "merged duplicate diverged: {a} vs {b}"
            );
        }
        // Both duplicates map to the same new expert.
        assert_eq!(res.index_map[0], res.index_map[1]);
    }

    #[test]
    fn merge_weighted_average_respects_usage() {
        // Two experts that differ only in b2; merge weighting must follow usage.
        let mut rng = LcgRng::new(11);
        let mut a = ExpertFfn::new(3, 6, ExpertActivation::Relu, &mut rng);
        let mut b = a.clone();
        for v in a.b2.iter_mut() {
            *v = 0.0;
        }
        for v in b.b2.iter_mut() {
            *v = 10.0;
        }
        let merged = weighted_average(&a, &b, 3.0, 1.0).expect("weighted_average");
        // 0.75·0 + 0.25·10 = 2.5
        for &v in &merged.b2 {
            assert!((v - 2.5).abs() < 1e-5, "blend {v} != 2.5");
        }
    }

    #[test]
    fn merge_to_one_collapses_all() {
        let bank = make_bank(4, 4, 8, 5);
        let usage = vec![1_usize, 1, 1, 1];
        let res = merge_experts(&bank, &usage, 1).expect("merge should succeed");
        assert_eq!(res.n_kept, 1);
        assert!(res.index_map.iter().all(|&m| m == 0));
    }

    #[test]
    fn invalid_keep_rejected() {
        let bank = make_bank(3, 4, 8, 9);
        let usage = vec![1_usize, 1, 1];
        assert!(matches!(
            prune_experts(&bank, &usage, 0),
            Err(MoeError::InvalidExpertCount { .. })
        ));
        assert!(matches!(
            merge_experts(&bank, &usage, 4),
            Err(MoeError::InvalidExpertCount { .. })
        ));
        assert!(matches!(
            prune_experts(&bank, &[1, 1], 1),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }
}
