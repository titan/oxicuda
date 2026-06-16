//! Inverted Multi-Index (IMI).
//!
//! Babenko & Lempitsky, "The Inverted Multi-Index", CVPR 2012 / TPAMI 2015.
//!
//! A plain IVF index partitions the space into `K` Voronoi cells with a single
//! coarse quantiser, so `K` is limited by the cost of one `K`-way distance scan
//! per query.  The inverted multi-index splits each `dim`-vector into **two
//! halves** and quantises each half independently with its own `K`-centroid
//! codebook.  The effective number of cells becomes `K²` (the Cartesian product
//! of the two codebooks) while the per-query coarse cost stays `O(K)` — the
//! distances to the two half-codebooks are computed once and *combined*.
//!
//! ## Cell enumeration: the multi-sequence algorithm
//!
//! To visit the product cells in increasing order of combined query distance
//! `d₁(i) + d₂(j)` without sorting all `K²` of them, IMI uses a best-first
//! "multi-sequence" merge:
//!   1. Sort each half's centroids by distance to the query → orders `r₁`, `r₂`.
//!   2. Start a min-heap with the pair `(r₁[0], r₂[0])`.
//!   3. Pop the smallest-sum pair `(a, b)`; emit cell `(r₁[a], r₂[b])`; push its
//!      successors `(a+1, b)` and `(a, b+1)` (de-duplicated).
//!
//! This yields the `multi_len` nearest product cells in `O(multi_len · log)`
//! time, which the search then concatenates into a candidate list and re-ranks
//! by exact L2.
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;
use std::collections::BinaryHeap;

/// Configuration for an [`InvertedMultiIndex`].
#[derive(Debug, Clone, Copy)]
pub struct ImiConfig {
    /// Full vector dimensionality. Must be ≥ 2 and even (split into two halves).
    pub dim: usize,
    /// Centroids per half-codebook `K`. Must be ≥ 1; the index has up to `K²`
    /// product cells.
    pub k: usize,
    /// k-means epochs used to train each half-codebook.
    pub n_iter: usize,
}

impl ImiConfig {
    fn validate(&self) -> AnnResult<()> {
        if self.dim < 2 {
            return Err(AnnError::InvalidVectorDim { dim: self.dim });
        }
        if !self.dim.is_multiple_of(2) {
            return Err(AnnError::Internal {
                msg: format!("imi: dim must be even, got {}", self.dim),
            });
        }
        if self.k == 0 {
            return Err(AnnError::InvalidK {
                k: 0,
                n: usize::MAX,
            });
        }
        Ok(())
    }
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Inverted Multi-Index over a corpus of `dim`-dimensional vectors.
///
/// Holds the two half-codebooks, the stored vectors, and the inverted lists
/// keyed by the product cell `(c₁, c₂)` flattened as `c₁ * k + c₂`.
#[derive(Debug, Clone)]
pub struct InvertedMultiIndex {
    /// First-half codebook, row-major `k × half`.
    codebook1: Vec<f32>,
    /// Second-half codebook, row-major `k × half`.
    codebook2: Vec<f32>,
    /// Inverted lists; `lists[c1 * k + c2]` holds the ids assigned to that cell.
    lists: Vec<Vec<usize>>,
    /// Stored vectors, row-major `[n, dim]`, indexed by id.
    vectors: Vec<f32>,
    dim: usize,
    half: usize,
    k: usize,
    n: usize,
}

impl InvertedMultiIndex {
    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Centroids per half-codebook `K`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of product cells `K²`.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.k * self.k
    }

    /// Train the two half-codebooks and add every vector to its product cell.
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] when `n == 0`.
    /// - [`AnnError::DimensionMismatch`] when `data.len() != n * dim`.
    /// - configuration errors from [`ImiConfig`] validation.
    pub fn train(
        data: &[f32],
        n: usize,
        cfg: &ImiConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<InvertedMultiIndex> {
        cfg.validate()?;
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let dim = cfg.dim;
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        let half = dim / 2;
        let k = cfg.k.min(n);

        // Split into two half-matrices and train one k-means each.
        let mut h1 = vec![0.0_f32; n * half];
        let mut h2 = vec![0.0_f32; n * half];
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            h1[i * half..(i + 1) * half].copy_from_slice(&row[..half]);
            h2[i * half..(i + 1) * half].copy_from_slice(&row[half..]);
        }
        let km1 = KMeans::fit(&h1, n, half, k, cfg.n_iter.max(1), rng)?;
        let km2 = KMeans::fit(&h2, n, half, k, cfg.n_iter.max(1), rng)?;

        // Pad codebooks up to the requested k (cycling learned centroids) so the
        // product-cell index space is always full size cfg.k².
        let mut codebook1 = vec![0.0_f32; cfg.k * half];
        let mut codebook2 = vec![0.0_f32; cfg.k * half];
        let c1 = km1.centroids();
        let c2 = km2.centroids();
        for c in 0..cfg.k {
            let src = (c % k) * half;
            codebook1[c * half..(c + 1) * half].copy_from_slice(&c1[src..src + half]);
            codebook2[c * half..(c + 1) * half].copy_from_slice(&c2[src..src + half]);
        }

        let mut index = InvertedMultiIndex {
            codebook1,
            codebook2,
            lists: vec![Vec::new(); cfg.k * cfg.k],
            vectors: Vec::with_capacity(n * dim),
            dim,
            half,
            k: cfg.k,
            n: 0,
        };
        for i in 0..n {
            index.add(&data[i * dim..(i + 1) * dim])?;
        }
        Ok(index)
    }

    /// Assign a vector to its product cell and append it to the inverted list.
    ///
    /// Returns the id given to the vector (a running counter).
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `v.len() != dim`.
    pub fn add(&mut self, v: &[f32]) -> AnnResult<usize> {
        if v.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }
        let id = self.n;
        let c1 = self.nearest_half(&v[..self.half], &self.codebook1);
        let c2 = self.nearest_half(&v[self.half..], &self.codebook2);
        self.lists[c1 * self.k + c2].push(id);
        self.vectors.extend_from_slice(v);
        self.n += 1;
        Ok(id)
    }

    /// Nearest centroid index within one half-codebook.
    fn nearest_half(&self, half_vec: &[f32], codebook: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let centroid = &codebook[c * self.half..(c + 1) * self.half];
            let d = l2_sq(half_vec, centroid);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Enumerate the `multi_len` nearest product cells for `query` in increasing
    /// order of combined half-distance, using the multi-sequence algorithm.
    ///
    /// Returns flattened cell ids `c1 * k + c2`.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `query.len() != dim`.
    pub fn multi_sequence(&self, query: &[f32], multi_len: usize) -> AnnResult<Vec<usize>> {
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // Half-distances to each codebook.
        let mut d1: Vec<(f32, usize)> = (0..self.k)
            .map(|c| {
                let centroid = &self.codebook1[c * self.half..(c + 1) * self.half];
                (l2_sq(&query[..self.half], centroid), c)
            })
            .collect();
        let mut d2: Vec<(f32, usize)> = (0..self.k)
            .map(|c| {
                let centroid = &self.codebook2[c * self.half..(c + 1) * self.half];
                (l2_sq(&query[self.half..], centroid), c)
            })
            .collect();
        d1.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        d2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Min-heap over (combined_distance, rank1, rank2). We invert the sum so
        // the std max-heap behaves as a min-heap.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut visited = vec![false; self.k * self.k];
        let push = |heap: &mut BinaryHeap<HeapEntry>, visited: &mut [bool], a: usize, b: usize| {
            if a < d1.len() && b < d2.len() && !visited[a * self.k + b] {
                visited[a * self.k + b] = true;
                heap.push(HeapEntry {
                    sum: d1[a].0 + d2[b].0,
                    a,
                    b,
                });
            }
        };
        push(&mut heap, &mut visited, 0, 0);

        let mut cells = Vec::with_capacity(multi_len);
        while cells.len() < multi_len {
            let Some(top) = heap.pop() else {
                break;
            };
            let cell = d1[top.a].1 * self.k + d2[top.b].1;
            cells.push(cell);
            push(&mut heap, &mut visited, top.a + 1, top.b);
            push(&mut heap, &mut visited, top.a, top.b + 1);
        }
        Ok(cells)
    }

    /// Search for the top-`k` nearest neighbours of `query`.
    ///
    /// Visits product cells in multi-sequence order until at least `k`
    /// candidates have been gathered (bounded by `max_cells`), then re-ranks the
    /// gathered candidates by exact squared L2.
    ///
    /// # Errors
    /// - [`AnnError::IndexEmpty`] when the index holds no vectors.
    /// - [`AnnError::InvalidK`] when `k == 0`.
    /// - [`AnnError::DimensionMismatch`] when `query.len() != dim`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        max_cells: usize,
    ) -> AnnResult<Vec<(usize, f32)>> {
        if self.n == 0 {
            return Err(AnnError::IndexEmpty);
        }
        if k == 0 {
            return Err(AnnError::InvalidK { k, n: self.n });
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let budget = max_cells.max(1).min(self.k * self.k);
        let cells = self.multi_sequence(query, budget)?;

        // Gather candidates from every budgeted cell. The cell budget (`max_cells`)
        // is the recall knob: more cells ⇒ more candidates ⇒ higher recall, at the
        // cost of exact-scoring a larger shortlist. We deliberately do not stop
        // early, so a full-cell budget yields an exhaustive (exact) search.
        let mut candidates: Vec<usize> = Vec::new();
        for &cell in &cells {
            candidates.extend_from_slice(&self.lists[cell]);
        }

        // Exact re-rank.
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&id| {
                let x = &self.vectors[id * self.dim..(id + 1) * self.dim];
                (id, l2_sq(query, x))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

/// Heap entry for the multi-sequence merge. Ordered so that the `BinaryHeap`
/// (a max-heap) pops the **smallest** combined distance first.
struct HeapEntry {
    sum: f32,
    a: usize,
    b: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sum == other.sum
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so the smaller sum is "greater" and pops first.
        other
            .sum
            .partial_cmp(&self.sum)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_normal_pair().0).collect()
    }

    fn cfg(dim: usize) -> ImiConfig {
        ImiConfig {
            dim,
            k: 4,
            n_iter: 15,
        }
    }

    fn brute_topk(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, l2_sq(q, &data[i * dim..(i + 1) * dim])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn imi_train_basic() {
        let dim = 8;
        let n = 100;
        let data = rand_data(n, dim, 1);
        let mut rng = LcgRng::new(2);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        assert_eq!(idx.len(), n);
        assert_eq!(idx.k(), 4);
        assert_eq!(idx.n_cells(), 16);
        assert!(!idx.is_empty());
    }

    #[test]
    fn imi_all_vectors_assigned() {
        let dim = 8;
        let n = 80;
        let data = rand_data(n, dim, 3);
        let mut rng = LcgRng::new(4);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let total: usize = idx.lists.iter().map(Vec::len).sum();
        assert_eq!(total, n, "every vector must land in exactly one cell");
    }

    #[test]
    fn imi_multi_sequence_increasing_distance() {
        // Cells must be enumerated in non-decreasing combined-distance order.
        let dim = 8;
        let n = 120;
        let data = rand_data(n, dim, 5);
        let mut rng = LcgRng::new(6);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 99);
        let cells = idx
            .multi_sequence(&q, 16)
            .expect("multi_sequence should succeed");

        // Recompute combined distance for each emitted cell and check ordering.
        let half = dim / 2;
        let mut prev = f32::NEG_INFINITY;
        for &cell in &cells {
            let c1 = cell / idx.k();
            let c2 = cell % idx.k();
            let d1 = l2_sq(&q[..half], &idx.codebook1[c1 * half..(c1 + 1) * half]);
            let d2 = l2_sq(&q[half..], &idx.codebook2[c2 * half..(c2 + 1) * half]);
            let sum = d1 + d2;
            assert!(sum >= prev - 1e-4, "cells not ordered: {sum} < {prev}");
            prev = sum;
        }
    }

    #[test]
    fn imi_multi_sequence_first_cell_is_nearest() {
        let dim = 8;
        let n = 100;
        let data = rand_data(n, dim, 7);
        let mut rng = LcgRng::new(8);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 33);
        let cells = idx
            .multi_sequence(&q, 1)
            .expect("multi_sequence should succeed");
        assert_eq!(cells.len(), 1);
        // The first cell must minimise d1+d2 over ALL k² cells.
        let half = dim / 2;
        let combined = |c1: usize, c2: usize| {
            l2_sq(&q[..half], &idx.codebook1[c1 * half..(c1 + 1) * half])
                + l2_sq(&q[half..], &idx.codebook2[c2 * half..(c2 + 1) * half])
        };
        let mut best = f32::INFINITY;
        for c1 in 0..idx.k() {
            for c2 in 0..idx.k() {
                best = best.min(combined(c1, c2));
            }
        }
        let got_c1 = cells[0] / idx.k();
        let got_c2 = cells[0] % idx.k();
        assert!((combined(got_c1, got_c2) - best).abs() < 1e-4);
    }

    #[test]
    fn imi_multi_sequence_no_duplicates() {
        let dim = 8;
        let n = 100;
        let data = rand_data(n, dim, 9);
        let mut rng = LcgRng::new(10);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 22);
        let cells = idx
            .multi_sequence(&q, 16)
            .expect("multi_sequence should succeed");
        let mut seen = std::collections::HashSet::new();
        for &c in &cells {
            assert!(seen.insert(c), "duplicate cell {c}");
        }
    }

    #[test]
    fn imi_multi_sequence_bounded_by_k2() {
        let dim = 8;
        let n = 60;
        let data = rand_data(n, dim, 11);
        let mut rng = LcgRng::new(12);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 44);
        // Ask for more cells than exist; should cap at k².
        let cells = idx
            .multi_sequence(&q, 1000)
            .expect("multi_sequence should succeed");
        assert!(cells.len() <= idx.n_cells());
    }

    #[test]
    fn imi_search_returns_k() {
        let dim = 8;
        let n = 200;
        let data = rand_data(n, dim, 13);
        let mut rng = LcgRng::new(14);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 55);
        let res = idx.search(&q, 5, 16).expect("search should succeed");
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn imi_search_finds_self() {
        let dim = 8;
        let n = 150;
        let data = rand_data(n, dim, 15);
        let mut rng = LcgRng::new(16);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        // Query with vector 42 itself, scanning all cells → must find it.
        let q = data[42 * dim..43 * dim].to_vec();
        let res = idx
            .search(&q, 1, idx.n_cells())
            .expect("search should succeed");
        assert_eq!(res[0].0, 42, "self not found");
        assert!(res[0].1 < 1e-5);
    }

    #[test]
    fn imi_search_sorted_ascending() {
        let dim = 8;
        let n = 120;
        let data = rand_data(n, dim, 17);
        let mut rng = LcgRng::new(18);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let q = rand_data(1, dim, 66);
        let res = idx
            .search(&q, 10, idx.n_cells())
            .expect("search should succeed");
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "not ascending");
        }
    }

    #[test]
    fn imi_full_scan_recall() {
        // Scanning ALL cells must achieve high recall vs brute force.
        let dim = 8;
        let n = 200;
        let data = rand_data(n, dim, 19);
        let mut rng = LcgRng::new(20);
        let larger = ImiConfig {
            dim,
            k: 8,
            n_iter: 20,
        };
        let idx = InvertedMultiIndex::train(&data, n, &larger, &mut rng)
            .expect("training should succeed");
        let n_q = 10;
        let queries = rand_data(n_q, dim, 321);
        let k = 5;
        let mut hits = 0usize;
        for qi in 0..n_q {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt: std::collections::HashSet<usize> =
                brute_topk(&data, n, dim, q, k).into_iter().collect();
            let res = idx
                .search(q, k, idx.n_cells())
                .expect("search should succeed");
            hits += res.iter().filter(|(id, _)| gt.contains(id)).count();
        }
        let recall = hits as f32 / (n_q * k) as f32;
        assert!(recall >= 0.9, "full-scan recall {recall} too low");
    }

    #[test]
    fn imi_more_cells_better_recall() {
        // Scanning more cells should not reduce recall.
        let dim = 8;
        let n = 300;
        let data = rand_data(n, dim, 21);
        let mut rng = LcgRng::new(22);
        let larger = ImiConfig {
            dim,
            k: 8,
            n_iter: 20,
        };
        let idx = InvertedMultiIndex::train(&data, n, &larger, &mut rng)
            .expect("training should succeed");
        let n_q = 12;
        let queries = rand_data(n_q, dim, 654);
        let k = 5;

        let recall_for = |cells: usize| -> f32 {
            let mut hits = 0usize;
            for qi in 0..n_q {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let gt: std::collections::HashSet<usize> =
                    brute_topk(&data, n, dim, q, k).into_iter().collect();
                let res = idx.search(q, k, cells).expect("search should succeed");
                hits += res.iter().filter(|(id, _)| gt.contains(id)).count();
            }
            hits as f32 / (n_q * k) as f32
        };

        let few = recall_for(4);
        let many = recall_for(idx.n_cells());
        assert!(
            many >= few - 1e-6,
            "more cells gave worse recall: {many} < {few}"
        );
    }

    #[test]
    fn imi_add_increments_len() {
        let dim = 4;
        let n = 40;
        let data = rand_data(n, dim, 23);
        let mut rng = LcgRng::new(24);
        let mut idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let id = idx.add(&[0.1, 0.2, 0.3, 0.4]).expect("add should succeed");
        assert_eq!(id, n);
        assert_eq!(idx.len(), n + 1);
    }

    #[test]
    fn imi_err_odd_dim() {
        let mut rng = LcgRng::new(25);
        let bad = ImiConfig {
            dim: 7,
            k: 4,
            n_iter: 5,
        };
        let data = rand_data(20, 7, 1);
        let err = InvertedMultiIndex::train(&data, 20, &bad, &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::Internal { .. }));
    }

    #[test]
    fn imi_err_empty() {
        let mut rng = LcgRng::new(26);
        let err = InvertedMultiIndex::train(&[], 0, &cfg(4), &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::EmptyInput));
    }

    #[test]
    fn imi_err_dim_mismatch() {
        let mut rng = LcgRng::new(27);
        let err = InvertedMultiIndex::train(&[1.0, 2.0, 3.0], 5, &cfg(4), &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }

    #[test]
    fn imi_err_search_k_zero() {
        let dim = 4;
        let n = 30;
        let data = rand_data(n, dim, 28);
        let mut rng = LcgRng::new(29);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let err = idx.search(&[0.1, 0.2, 0.3, 0.4], 0, 8).unwrap_err();
        assert!(matches!(err, AnnError::InvalidK { .. }));
    }

    #[test]
    fn imi_err_query_dim_mismatch() {
        let dim = 4;
        let n = 30;
        let data = rand_data(n, dim, 30);
        let mut rng = LcgRng::new(31);
        let idx = InvertedMultiIndex::train(&data, n, &cfg(dim), &mut rng)
            .expect("training should succeed");
        let err = idx.search(&[0.1, 0.2], 3, 8).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }
}
