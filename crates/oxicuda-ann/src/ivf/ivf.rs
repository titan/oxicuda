use super::train::train_coarse;
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::topk::heap::BoundedMaxHeap;

/// Inverted File Index for approximate nearest neighbor search.
pub struct IvfIndex {
    /// Coarse centroids `[n_lists, dim]`.
    coarse: Vec<f32>,
    pub n_lists: usize,
    /// Each posting list holds the original vector IDs belonging to that cluster.
    posting_lists: Vec<Vec<usize>>,
    /// Per-list vector storage: `vectors[list_id]` is flat row-major `[count, dim]`.
    ///
    /// Previously a single flat `Vec<f32>` was iterated with a global counter that
    /// assumed list-order traversal matched insertion order — that is only true when
    /// all vectors for list 0 are added before all vectors for list 1, etc.  When
    /// callers interleave `add` calls across lists the counter desynchronises and the
    /// search scores the wrong stored vector for each id.  Per-list storage removes
    /// the aliasing entirely.
    vectors: Vec<Vec<f32>>,
    pub dim: usize,
    fitted: bool,
}

impl IvfIndex {
    #[must_use]
    pub fn new(n_lists: usize, dim: usize) -> Self {
        Self {
            coarse: Vec::new(),
            n_lists,
            posting_lists: vec![Vec::new(); n_lists],
            vectors: vec![Vec::new(); n_lists],
            dim,
            fitted: false,
        }
    }

    /// Train the coarse quantizer using k-means.
    pub fn train(&mut self, data: &[f32], n: usize, rng: &mut LcgRng) -> AnnResult<()> {
        self.coarse = train_coarse(data, n, self.dim, self.n_lists, 50, rng)?;
        self.fitted = true;
        Ok(())
    }

    /// Add a vector with external `id` to the index.
    pub fn add(&mut self, v: &[f32], id: usize) {
        let list_id = self.assign_to_list(v);
        self.posting_lists[list_id].push(id);
        self.vectors[list_id].extend_from_slice(v);
    }

    fn assign_to_list(&self, v: &[f32]) -> usize {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for c in 0..self.n_lists {
            let center = &self.coarse[c * self.dim..(c + 1) * self.dim];
            let d: f32 = v
                .iter()
                .zip(center.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Nearest-centroid distances for probing.
    fn probe_order(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = (0..self.n_lists)
            .map(|c| {
                let center = &self.coarse[c * self.dim..(c + 1) * self.dim];
                let d: f32 = query
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (c, d)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(nprobe).map(|(c, _)| *c).collect()
    }

    /// Return the index of the nearest coarse centroid to `v` (used by IVFPQ).
    pub fn nearest_list(&self, v: &[f32]) -> usize {
        self.assign_to_list(v)
    }

    /// Return sorted list indices for probing (used by IVFPQ).
    pub fn probe_lists(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        self.probe_order(query, nprobe)
    }

    /// Search for `k` approximate nearest neighbors with `nprobe` lists probed.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> AnnResult<Vec<(usize, f32)>> {
        if !self.fitted {
            return Err(AnnError::NotFitted);
        }
        let total: usize = self.vectors.iter().map(|l| l.len() / self.dim).sum();
        if total == 0 {
            return Err(AnnError::IndexEmpty);
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if nprobe == 0 || nprobe > self.n_lists {
            return Err(AnnError::InvalidNumProbes {
                nprobe,
                nlist: self.n_lists,
            });
        }

        let actual_k = k.min(total);
        if actual_k == 0 {
            return Err(AnnError::InvalidK { k, n: total });
        }

        let mut heap = BoundedMaxHeap::new(actual_k);
        let probed = self.probe_order(query, nprobe);

        for &list_id in &probed {
            let list_vecs = &self.vectors[list_id];
            for (item_idx, &id) in self.posting_lists[list_id].iter().enumerate() {
                let vec_row = &list_vecs[item_idx * self.dim..(item_idx + 1) * self.dim];
                let d: f32 = query
                    .iter()
                    .zip(vec_row.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                heap.push(d, id);
            }
        }

        Ok(heap.into_sorted_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::IvfIndex;
    use crate::error::AnnError;
    use crate::handle::LcgRng;

    // ---------------------------------------------------------------------------
    // Error-path tests
    // ---------------------------------------------------------------------------

    #[test]
    fn not_fitted_returns_not_fitted_error() {
        let idx = IvfIndex::new(2, 3);
        let result = idx.search(&[0.0_f32; 3], 1, 1);
        assert!(
            matches!(result, Err(AnnError::NotFitted)),
            "expected NotFitted, got {result:?}"
        );
    }

    #[test]
    fn empty_after_train_returns_index_empty() {
        let mut rng = LcgRng::new(1);
        let train_data = [0.0_f32, 0.0, 100.0, 100.0];
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 2, &mut rng)
            .expect("train should succeed on two-point data");
        // No add() calls — index is fitted but empty.
        let result = idx.search(&[0.0_f32, 0.0], 1, 1);
        assert!(
            matches!(result, Err(AnnError::IndexEmpty)),
            "expected IndexEmpty before any add, got {result:?}"
        );
    }

    #[test]
    fn query_wrong_dimension_returns_dimension_mismatch() {
        let mut rng = LcgRng::new(2);
        let train_data = [0.0_f32, 0.0, 100.0, 100.0];
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 2, &mut rng).expect("train");
        idx.add(&[0.0_f32, 0.0], 0);
        let result = idx.search(&[0.0_f32, 0.0, 0.0], 1, 1);
        assert!(
            matches!(
                result,
                Err(AnnError::DimensionMismatch {
                    expected: 2,
                    got: 3
                })
            ),
            "expected DimensionMismatch{{expected:2,got:3}}, got {result:?}"
        );
    }

    #[test]
    fn nprobe_zero_returns_invalid_num_probes() {
        let mut rng = LcgRng::new(3);
        let train_data = [0.0_f32, 0.0, 100.0, 100.0];
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 2, &mut rng).expect("train");
        idx.add(&[0.0_f32, 0.0], 0);
        let result = idx.search(&[0.0_f32, 0.0], 1, 0);
        assert!(
            matches!(result, Err(AnnError::InvalidNumProbes { .. })),
            "expected InvalidNumProbes for nprobe=0, got {result:?}"
        );
    }

    #[test]
    fn nprobe_exceeds_n_lists_returns_invalid_num_probes() {
        let mut rng = LcgRng::new(4);
        let train_data = [0.0_f32, 0.0, 100.0, 100.0];
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 2, &mut rng).expect("train");
        idx.add(&[0.0_f32, 0.0], 0);
        // n_lists=2, nprobe=3 → invalid
        let result = idx.search(&[0.0_f32, 0.0], 1, 3);
        assert!(
            matches!(result, Err(AnnError::InvalidNumProbes { .. })),
            "expected InvalidNumProbes for nprobe=3 > n_lists=2, got {result:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // Assignment correctness
    // ---------------------------------------------------------------------------

    /// After training on two well-separated clusters, `nearest_list` must assign
    /// every cluster-A vector to one list and every cluster-B vector to the other.
    /// This verifies that the coarse quantizer correctly partitions vectors at
    /// assignment time (same logic used by `add`).
    #[test]
    fn nearest_list_consistent_across_same_cluster_vectors() {
        let mut rng = LcgRng::new(42);
        // 12 points: 6 near [0,0], 6 near [100,100]
        let mut train_data = Vec::with_capacity(24);
        for i in 0..6_u32 {
            let off = i as f32 * 0.01;
            train_data.extend_from_slice(&[off, off]);
        }
        for i in 0..6_u32 {
            let off = i as f32 * 0.01;
            train_data.extend_from_slice(&[100.0 + off, 100.0 + off]);
        }
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 12, &mut rng)
            .expect("train should succeed on two-cluster data");

        // All near-origin vectors must map to the same list.
        let list_a = idx.nearest_list(&[0.0_f32, 0.0]);
        for i in 0..6_u32 {
            let off = i as f32 * 0.01;
            assert_eq!(
                idx.nearest_list(&[off, off]),
                list_a,
                "cluster-A vector {i} assigned to wrong list"
            );
        }
        // All near-[100,100] vectors must map to the OTHER list.
        let list_b = idx.nearest_list(&[100.0_f32, 100.0]);
        assert_ne!(
            list_a, list_b,
            "two well-separated clusters must map to different lists"
        );
        for i in 0..6_u32 {
            let off = i as f32 * 0.01;
            assert_eq!(
                idx.nearest_list(&[100.0 + off, 100.0 + off]),
                list_b,
                "cluster-B vector {i} assigned to wrong list"
            );
        }
    }

    /// With nprobe = n_lists (probe every list), the IVF search must return the
    /// exact nearest neighbour — identical to brute-force L2.
    ///
    /// Vectors are added in INTERLEAVED cluster order (A, B, A, B …) to expose
    /// any per-list vs. insertion-order aliasing in the vector store.
    #[test]
    fn nprobe_all_finds_exact_nearest_neighbor() {
        let mut rng = LcgRng::new(7);
        let train_data = [
            0.0_f32, 0.0, // cluster A
            0.1_f32, 0.0, // cluster A
            10.0_f32, 10.0, // cluster B
            10.1_f32, 10.0, // cluster B
        ];
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 4, &mut rng).expect("train");

        // Interleaved: A, B, A, B
        idx.add(&[0.0_f32, 0.0], 0);
        idx.add(&[10.0_f32, 10.0], 1);
        idx.add(&[0.1_f32, 0.0], 2);
        idx.add(&[10.1_f32, 10.0], 3);

        // Query is closest to id=2 = [0.1, 0.0]
        let query = [0.11_f32, 0.0];
        let results = idx
            .search(&query, 1, 2)
            .expect("search with nprobe=n_lists should succeed");
        assert!(!results.is_empty(), "must return at least 1 result");

        // Brute-force: compute true nearest among the 4 added vectors.
        let vecs: [[f32; 2]; 4] = [[0.0, 0.0], [10.0, 10.0], [0.1, 0.0], [10.1, 10.0]];
        let bf_nearest = (0..4_usize)
            .min_by(|&a, &b| {
                let da: f32 = query
                    .iter()
                    .zip(vecs[a].iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum();
                let db: f32 = query
                    .iter()
                    .zip(vecs[b].iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("iterator is non-empty");

        assert_eq!(
            results[0].0, bf_nearest,
            "IVF with nprobe=n_lists must return exact nearest neighbor (got id={}, want id={})",
            results[0].0, bf_nearest
        );
    }

    // ---------------------------------------------------------------------------
    // Result-shape properties
    // ---------------------------------------------------------------------------

    #[test]
    fn search_results_sorted_ascending_by_distance() {
        let mut rng = LcgRng::new(13);
        // 10 evenly spaced points on the x-axis, dim=2
        let mut train_data = Vec::with_capacity(20);
        for i in 0..10_u32 {
            train_data.extend_from_slice(&[i as f32, 0.0_f32]);
        }
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 10, &mut rng).expect("train");
        for i in 0..10_u32 {
            idx.add(&[i as f32, 0.0_f32], i as usize);
        }
        let query = [4.5_f32, 0.0];
        let results = idx.search(&query, 6, 2).expect("search should succeed");
        assert!(!results.is_empty(), "must find at least one result");
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results not sorted ascending: d[i]={} d[i+1]={}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn search_returns_at_most_k_results() {
        let mut rng = LcgRng::new(17);
        let mut train_data = Vec::with_capacity(20);
        for i in 0..10_u32 {
            train_data.extend_from_slice(&[i as f32, 0.0_f32]);
        }
        let mut idx = IvfIndex::new(2, 2);
        idx.train(&train_data, 10, &mut rng).expect("train");
        for i in 0..10_u32 {
            idx.add(&[i as f32, 0.0_f32], i as usize);
        }
        let query = [5.0_f32, 0.0];
        for k in [1_usize, 3, 5, 10] {
            let results = idx.search(&query, k, 2).expect("search should succeed");
            assert!(
                results.len() <= k,
                "returned {} results for k={k}",
                results.len()
            );
        }
    }
}
