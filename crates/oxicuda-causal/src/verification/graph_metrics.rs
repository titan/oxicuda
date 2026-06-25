//! Structure-recovery metrics for comparing a learned graph against ground truth.
//!
//! Used by the NOTEARS recovery and PC orientation verification suites. All
//! inputs are edge lists; directed edges are ordered `(parent, child)`,
//! undirected skeleton edges are normalized to `(min, max)`.

use std::collections::HashSet;

fn norm_undirected(e: (usize, usize)) -> (usize, usize) {
    if e.0 <= e.1 { e } else { (e.1, e.0) }
}

/// Precision / recall / F1 of an *undirected skeleton* against the truth.
pub struct SkeletonScore {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

/// Score a learned skeleton (set of undirected pairs) vs the true skeleton.
#[must_use]
pub fn skeleton_score(learned: &[(usize, usize)], truth: &[(usize, usize)]) -> SkeletonScore {
    let l: HashSet<(usize, usize)> = learned.iter().copied().map(norm_undirected).collect();
    let t: HashSet<(usize, usize)> = truth.iter().copied().map(norm_undirected).collect();
    let tp = l.intersection(&t).count();
    let fp = l.len() - tp;
    let fn_ = t.len() - tp;
    let precision = if l.is_empty() {
        1.0
    } else {
        tp as f64 / l.len() as f64
    };
    let recall = if t.is_empty() {
        1.0
    } else {
        tp as f64 / t.len() as f64
    };
    let f1 = if precision + recall < 1e-12 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    SkeletonScore {
        precision,
        recall,
        f1,
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
    }
}

/// Structural Hamming Distance between two *directed* edge sets.
///
/// Counts edges present in exactly one graph. A reversed edge (`a→b` vs `b→a`)
/// contributes 2 (one missing, one extra), matching the common convention used
/// when comparing against ground-truth DAGs.
#[must_use]
pub fn structural_hamming_distance(learned: &[(usize, usize)], truth: &[(usize, usize)]) -> usize {
    let l: HashSet<(usize, usize)> = learned.iter().copied().collect();
    let t: HashSet<(usize, usize)> = truth.iter().copied().collect();
    let missing = t.difference(&l).count();
    let extra = l.difference(&t).count();
    missing + extra
}

/// Count correctly-oriented edges among those whose skeleton was recovered.
///
/// Returns `(correct_orientation, recovered_skeleton_edges)`. An edge counts as
/// correctly oriented only if the directed learned edge equals the directed true
/// edge; recovered-but-reversed edges count toward the denominator, not the
/// numerator.
#[must_use]
pub fn orientation_accuracy(
    learned_directed: &[(usize, usize)],
    truth_directed: &[(usize, usize)],
) -> (usize, usize) {
    let truth_dir: HashSet<(usize, usize)> = truth_directed.iter().copied().collect();
    let truth_skel: HashSet<(usize, usize)> = truth_directed
        .iter()
        .copied()
        .map(norm_undirected)
        .collect();
    let mut correct = 0usize;
    let mut recovered = 0usize;
    let mut seen_skel: HashSet<(usize, usize)> = HashSet::new();
    for &e in learned_directed {
        let key = norm_undirected(e);
        if truth_skel.contains(&key) && seen_skel.insert(key) {
            recovered += 1;
            if truth_dir.contains(&e) {
                correct += 1;
            }
        }
    }
    (correct, recovered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_skeleton() {
        let truth = vec![(0, 1), (1, 2), (2, 3)];
        let learned = vec![(1, 0), (2, 1), (3, 2)]; // same skeleton, any order
        let s = skeleton_score(&learned, &truth);
        assert!((s.precision - 1.0).abs() < 1e-12);
        assert!((s.recall - 1.0).abs() < 1e-12);
        assert!((s.f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn partial_skeleton() {
        let truth = vec![(0, 1), (1, 2), (2, 3)];
        let learned = vec![(0, 1), (1, 2), (0, 3)]; // 2 right, 1 wrong, miss 1
        let s = skeleton_score(&learned, &truth);
        assert_eq!(s.true_positives, 2);
        assert_eq!(s.false_positives, 1);
        assert_eq!(s.false_negatives, 1);
        assert!((s.precision - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.recall - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn shd_reversed_edge_costs_two() {
        let truth = vec![(0, 1)];
        let learned = vec![(1, 0)];
        assert_eq!(structural_hamming_distance(&learned, &truth), 2);
        // Identical graphs: distance 0.
        assert_eq!(structural_hamming_distance(&truth, &truth), 0);
        // One extra edge: distance 1.
        assert_eq!(structural_hamming_distance(&[(0, 1), (1, 2)], &truth), 1);
    }

    #[test]
    fn orientation_counts() {
        let truth = vec![(0, 2), (1, 2)]; // collider into 2
        // Recovered both edges; one oriented right, one reversed.
        let learned = vec![(0, 2), (2, 1)];
        let (correct, recovered) = orientation_accuracy(&learned, &truth);
        assert_eq!(recovered, 2);
        assert_eq!(correct, 1);
    }
}
