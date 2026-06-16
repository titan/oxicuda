//! HDC k-Nearest-Neighbour classifier (associative-recall style).
//!
//! Unlike the centroid [`crate::classifier::hd_classifier::HdClassifier`], which
//! collapses every class into a single prototype hypervector, this classifier
//! stores **all** training exemplars (each an HV together with its label) and
//! classifies a query by majority vote among the `k` most-similar stored
//! exemplars under cosine similarity. This is the associative-recall k-NN scheme
//! used in HDC (e.g. Imani et al.).
//!
//! Similarities are recomputed per query against the stored ±1 exemplars; no
//! floating-point distances are cached.

use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::vector::binary::validate_binary;

/// HDC k-Nearest-Neighbour classifier storing individual exemplars.
///
/// Classification computes cosine similarity from the query to every stored
/// exemplar, keeps the `k` highest, and returns the majority label among them.
/// Ties (equal vote counts) are broken toward the smallest label index, and
/// similarity ties are broken by exemplar insertion order, so results are fully
/// deterministic.
pub struct HdKnn {
    /// Dimension of stored hypervectors.
    dim: usize,
    /// Number of neighbours to consult per query.
    k: usize,
    /// Stored exemplar hypervectors (±1), one per training sample.
    exemplars: Vec<Vec<i8>>,
    /// Label for each stored exemplar (parallel to `exemplars`).
    labels: Vec<usize>,
}

impl HdKnn {
    /// Create a new k-NN classifier for `dim`-dimensional HVs consulting `k`
    /// neighbours.
    ///
    /// # Errors
    /// Returns [`HdcError::ZeroDimension`] if `dim == 0`, and
    /// [`HdcError::EmptyInput`] if `k == 0`.
    pub fn new(dim: usize, k: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if k == 0 {
            return Err(HdcError::EmptyInput);
        }
        Ok(Self {
            dim,
            k,
            exemplars: Vec::new(),
            labels: Vec::new(),
        })
    }

    /// Store a training exemplar `hv` with its `label`.
    ///
    /// # Errors
    /// Returns [`HdcError::DimensionMismatch`] if `hv.len() != dim`, and
    /// [`HdcError::InvalidBinaryValue`] if `hv` contains a value outside
    /// `{-1, +1}`.
    pub fn add(&mut self, hv: &[i8], label: usize) -> HdcResult<()> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        validate_binary(hv)?;
        self.exemplars.push(hv.to_vec());
        self.labels.push(label);
        Ok(())
    }

    /// Number of stored exemplars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Whether no exemplars have been stored yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// Configured neighbour count `k`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute, for every stored exemplar, the pair `(index, similarity)` of the
    /// query against that exemplar, sorted by descending similarity with
    /// insertion order as a stable tie-break.
    fn ranked(&self, query: &[i8]) -> HdcResult<Vec<(usize, f32)>> {
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.exemplars.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.exemplars.len());
        for (idx, ex) in self.exemplars.iter().enumerate() {
            let sim = cosine_binary(query, ex)?;
            scored.push((idx, sim));
        }
        // Descending similarity; equal similarities keep insertion (index) order.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        Ok(scored)
    }

    /// Majority label among the `k` nearest neighbours.
    ///
    /// Uses an effective neighbour count of `min(k, len())`. Vote ties are broken
    /// toward the smallest label index.
    ///
    /// # Errors
    /// Returns [`HdcError::EmptyItemMemory`] if no exemplars are stored, and
    /// [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn classify(&self, query: &[i8]) -> HdcResult<usize> {
        let scored = self.ranked(query)?;
        let eff_k = self.k.min(scored.len());
        Ok(Self::majority(
            scored.iter().take(eff_k).map(|&(idx, _)| self.labels[idx]),
        ))
    }

    /// Classify and also return the top-`k` `(label, similarity)` pairs sorted by
    /// descending similarity (insertion order as tie-break).
    ///
    /// # Errors
    /// Returns [`HdcError::EmptyItemMemory`] if no exemplars are stored, and
    /// [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn classify_with_scores(&self, query: &[i8]) -> HdcResult<(usize, Vec<(usize, f32)>)> {
        let scored = self.ranked(query)?;
        let eff_k = self.k.min(scored.len());
        let top: Vec<(usize, f32)> = scored
            .iter()
            .take(eff_k)
            .map(|&(idx, sim)| (self.labels[idx], sim))
            .collect();
        let label = Self::majority(top.iter().map(|&(label, _)| label));
        Ok((label, top))
    }

    /// Per-class vote counts among the `k` nearest neighbours, sorted by
    /// descending vote count with the smallest label index as tie-break.
    ///
    /// # Errors
    /// Returns [`HdcError::EmptyItemMemory`] if no exemplars are stored, and
    /// [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn class_votes(&self, query: &[i8]) -> HdcResult<Vec<(usize, usize)>> {
        let scored = self.ranked(query)?;
        let eff_k = self.k.min(scored.len());
        let mut votes: Vec<(usize, usize)> = Vec::new();
        for &(idx, _) in scored.iter().take(eff_k) {
            let label = self.labels[idx];
            if let Some(entry) = votes.iter_mut().find(|(l, _)| *l == label) {
                entry.1 += 1;
            } else {
                votes.push((label, 1));
            }
        }
        votes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(votes)
    }

    /// Majority vote over an iterator of labels; ties resolved to the smallest
    /// label index. The input order is irrelevant to the result.
    fn majority<I: Iterator<Item = usize>>(labels: I) -> usize {
        let mut counts: Vec<(usize, usize)> = Vec::new();
        for label in labels {
            if let Some(entry) = counts.iter_mut().find(|(l, _)| *l == label) {
                entry.1 += 1;
            } else {
                counts.push((label, 1));
            }
        }
        let mut best_label = 0usize;
        let mut best_count = 0usize;
        let mut seen = false;
        for &(label, count) in &counts {
            if !seen || count > best_count || (count == best_count && label < best_label) {
                best_label = label;
                best_count = count;
                seen = true;
            }
        }
        best_label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    #[test]
    fn new_rejects_zero_dim() {
        let err = HdKnn::new(0, 3);
        assert!(matches!(err, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn new_rejects_zero_k() {
        let err = HdKnn::new(64, 0);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn classify_empty_is_empty_item_memory() {
        let knn = HdKnn::new(64, 3).expect("new");
        let q: Vec<i8> = vec![1i8; 64];
        let err = knn.classify(&q);
        assert!(matches!(err, Err(HdcError::EmptyItemMemory)));
    }

    #[test]
    fn single_exemplar_classifies_to_its_label() {
        let mut knn = HdKnn::new(64, 3).expect("new");
        let hv: Vec<i8> = vec![1i8; 64];
        knn.add(&hv, 7).expect("add");
        assert_eq!(knn.classify(&hv).expect("classify"), 7);
        assert_eq!(knn.len(), 1);
        assert!(!knn.is_empty());
    }

    #[test]
    fn k1_returns_nearest_neighbour_label() {
        let mut knn = HdKnn::new(8, 1).expect("new");
        let a: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let b: Vec<i8> = vec![-1, -1, -1, -1, -1, -1, -1, -1];
        knn.add(&a, 0).expect("add a");
        knn.add(&b, 1).expect("add b");
        // Query closer to a.
        let q: Vec<i8> = vec![1, 1, 1, 1, 1, 1, -1, 1];
        assert_eq!(knn.classify(&q).expect("classify"), 0);
    }

    #[test]
    fn majority_vote_k3_two_classes() {
        let mut knn = HdKnn::new(8, 3).expect("new");
        // Three exemplars: two of class 0 (similar to query), one of class 1.
        let q: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let near0a: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, -1];
        let near0b: Vec<i8> = vec![1, 1, 1, 1, 1, 1, -1, 1];
        let far1: Vec<i8> = vec![-1, -1, -1, -1, -1, -1, -1, -1];
        knn.add(&near0a, 0).expect("add");
        knn.add(&near0b, 0).expect("add");
        knn.add(&far1, 1).expect("add");
        assert_eq!(knn.classify(&q).expect("classify"), 0);
    }

    #[test]
    fn exact_match_similarity_is_one() {
        let mut knn = HdKnn::new(64, 1).expect("new");
        let mut rng = LcgRng::new(11);
        let hv = random_binary(64, &mut rng).expect("rand");
        knn.add(&hv, 3).expect("add");
        let (label, scores) = knn.classify_with_scores(&hv).expect("scores");
        assert_eq!(label, 3);
        assert_eq!(scores.len(), 1);
        assert!((scores[0].1 - 1.0).abs() < 1e-6, "sim={}", scores[0].1);
    }

    #[test]
    fn add_dimension_mismatch_rejected() {
        let mut knn = HdKnn::new(64, 3).expect("new");
        let hv: Vec<i8> = vec![1i8; 32];
        let err = knn.add(&hv, 0);
        assert!(matches!(
            err,
            Err(HdcError::DimensionMismatch {
                expected: 64,
                got: 32
            })
        ));
    }

    #[test]
    fn classify_dimension_mismatch_rejected() {
        let mut knn = HdKnn::new(64, 3).expect("new");
        let hv: Vec<i8> = vec![1i8; 64];
        knn.add(&hv, 0).expect("add");
        let q: Vec<i8> = vec![1i8; 16];
        let err = knn.classify(&q);
        assert!(matches!(
            err,
            Err(HdcError::DimensionMismatch {
                expected: 64,
                got: 16
            })
        ));
    }

    #[test]
    fn effective_k_caps_at_len() {
        let mut knn = HdKnn::new(8, 100).expect("new");
        let a: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        knn.add(&a, 5).expect("add");
        // k far exceeds the single stored exemplar; still classifies fine.
        assert_eq!(knn.classify(&a).expect("classify"), 5);
        let (_, scores) = knn.classify_with_scores(&a).expect("scores");
        assert_eq!(scores.len(), 1);
    }

    #[test]
    fn two_random_clusters_classify_correctly() {
        let mut rng = LcgRng::new(2024);
        let dim = 512;
        let mut knn = HdKnn::new(dim, 3).expect("new");
        let center0 = random_binary(dim, &mut rng).expect("c0");
        let center1 = random_binary(dim, &mut rng).expect("c1");
        // Add noisy variants of each center (flip a few bits).
        for s in 0..5 {
            let mut e0 = center0.clone();
            let mut e1 = center1.clone();
            let flip = (s * 7) % dim;
            e0[flip] = -e0[flip];
            e1[flip] = -e1[flip];
            knn.add(&e0, 0).expect("add0");
            knn.add(&e1, 1).expect("add1");
        }
        assert_eq!(knn.classify(&center0).expect("c0"), 0);
        assert_eq!(knn.classify(&center1).expect("c1"), 1);
    }

    #[test]
    fn label_tie_broken_to_smallest_index() {
        // k=2 with one neighbour of label 2 and one of label 5, equal similarity
        // structure → tie resolves to the smaller label (2).
        let mut knn = HdKnn::new(4, 2).expect("new");
        let a: Vec<i8> = vec![1, 1, -1, -1];
        let b: Vec<i8> = vec![-1, -1, 1, 1];
        knn.add(&a, 5).expect("add a (label 5)");
        knn.add(&b, 2).expect("add b (label 2)");
        // Query equidistant: orthogonal-ish to both. Both sims equal (0.0).
        let q: Vec<i8> = vec![1, -1, 1, -1];
        let sa = cosine_binary(&q, &a).expect("sa");
        let sb = cosine_binary(&q, &b).expect("sb");
        assert!((sa - sb).abs() < 1e-6, "sa={sa} sb={sb}");
        assert_eq!(knn.classify(&q).expect("classify"), 2);
    }

    #[test]
    fn class_votes_reports_counts() {
        let mut knn = HdKnn::new(8, 3).expect("new");
        let q: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let near0a: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, -1];
        let near0b: Vec<i8> = vec![1, 1, 1, 1, 1, 1, -1, 1];
        let far1: Vec<i8> = vec![-1, -1, -1, -1, -1, -1, -1, -1];
        knn.add(&near0a, 0).expect("add");
        knn.add(&near0b, 0).expect("add");
        knn.add(&far1, 1).expect("add");
        let votes = knn.class_votes(&q).expect("votes");
        assert_eq!(votes[0], (0, 2));
        assert!(votes.contains(&(1, 1)));
    }
}
