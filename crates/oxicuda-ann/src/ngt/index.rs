//! NGT / ANNG — Approximate Neighborhood Graph index.
//!
//! Implements the ANNG (Approximate Neighborhood Graph) variant of NGT
//! (Iwasaki & Miyazaki, *Neighborhood Graph and Tree*). The index is built
//! incrementally: each newly inserted node is connected bidirectionally to its
//! approximate `edge_count` nearest neighbors, where those neighbors are found
//! by running the very same approximate graph search over the partially-built
//! graph. Queries are answered with an ε-relaxed greedy best-first graph
//! search seeded from a small set of deterministically-chosen start nodes.
//!
//! # Distance
//!
//! All ordering uses the **squared** Euclidean (L2²) distance via
//! [`crate::distance::l2::l2_sq`]; this preserves nearest-neighbor ordering and
//! avoids square roots. [`NgtIndex::search`] therefore returns squared-L2
//! distances paired with node ids (the `(u32, f32)` convention shared with the
//! HNSW index), in ascending order.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{AnnError, AnnResult};

/// A totally-ordered wrapper around `f32` for use inside binary heaps. `NaN` is
/// treated as equal to itself and ordered as the largest value so it never
/// poisons the heap ordering.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(o) => o,
            // Push NaN to the "largest" end deterministically.
            None => match (self.0.is_nan(), other.0.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => std::cmp::Ordering::Equal,
            },
        }
    }
}

/// Configuration for an [`NgtIndex`].
#[derive(Debug, Clone)]
pub struct NgtConfig {
    /// Vector dimensionality. Must be `>= 1`.
    pub dim: usize,
    /// Number of neighbors (`k`) each node keeps in the ANNG. Must be `>= 1`.
    pub edge_count: usize,
    /// ε relaxation of the search bound; must be `>= 0`. A larger value lets the
    /// greedy search explore more of the frontier before terminating, trading
    /// speed for recall.
    pub search_epsilon: f32,
    /// Number of deterministic start seeds used to begin a search. Must be
    /// `>= 1`.
    pub seed_count: usize,
}

/// Incrementally-built Approximate Neighborhood Graph index.
pub struct NgtIndex {
    /// Flat `n × dim` row-major point storage.
    points: Vec<f32>,
    /// Per-node adjacency lists of neighbor ids (bidirectional, deduplicated).
    adjacency: Vec<Vec<u32>>,
    /// Index configuration.
    cfg: NgtConfig,
}

impl NgtIndex {
    /// Create an empty index from a validated configuration.
    ///
    /// # Errors
    /// - [`AnnError::InvalidVectorDim`] if `cfg.dim == 0`.
    /// - [`AnnError::Internal`] if `cfg.edge_count == 0`, `cfg.seed_count == 0`,
    ///   or `cfg.search_epsilon < 0` (or non-finite).
    pub fn new(cfg: NgtConfig) -> AnnResult<Self> {
        if cfg.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: cfg.dim });
        }
        if cfg.edge_count == 0 {
            return Err(AnnError::Internal {
                msg: "edge_count must be >= 1".to_string(),
            });
        }
        if cfg.seed_count == 0 {
            return Err(AnnError::Internal {
                msg: "seed_count must be >= 1".to_string(),
            });
        }
        if cfg.search_epsilon < 0.0 || !cfg.search_epsilon.is_finite() {
            return Err(AnnError::Internal {
                msg: format!(
                    "search_epsilon must be a finite value >= 0, got {}",
                    cfg.search_epsilon
                ),
            });
        }
        Ok(Self {
            points: Vec::new(),
            adjacency: Vec::new(),
            cfg,
        })
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// `true` when no points have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// Configured vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Borrow the stored vector for node `id`.
    #[inline]
    fn vector(&self, id: u32) -> &[f32] {
        let s = id as usize * self.cfg.dim;
        &self.points[s..s + self.cfg.dim]
    }

    /// Squared-L2 distance between `query` and node `id` (lengths already
    /// validated by callers).
    #[inline]
    fn dist_to(&self, query: &[f32], id: u32) -> f32 {
        let v = self.vector(id);
        query
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Insert vector `v`, connecting it bidirectionally to its approximate
    /// nearest neighbors, and return its newly-assigned id.
    ///
    /// The first inserted point becomes node `0` with an empty adjacency list.
    /// Every subsequent point first runs [`NgtIndex::search`] over the existing
    /// graph to obtain up to `edge_count` approximate neighbors, then adds a
    /// deduplicated edge between the new node and each of them in both
    /// directions.
    ///
    /// # Errors
    /// Returns [`AnnError::DimensionMismatch`] if `v.len() != dim`.
    pub fn insert(&mut self, v: &[f32]) -> AnnResult<u32> {
        if v.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: v.len(),
            });
        }

        // Empty graph: first node, no edges.
        if self.adjacency.is_empty() {
            self.points.extend_from_slice(v);
            self.adjacency.push(Vec::new());
            return Ok(0);
        }

        // Find approximate neighbors among the already-present nodes.
        let neighbors = self.search(v, self.cfg.edge_count)?;

        let new_id = self.adjacency.len() as u32;
        self.points.extend_from_slice(v);
        self.adjacency.push(Vec::new());

        for (nbr, _) in neighbors {
            if nbr == new_id {
                continue;
            }
            // new_id -> nbr (dedup).
            if !self.adjacency[new_id as usize].contains(&nbr) {
                self.adjacency[new_id as usize].push(nbr);
            }
            // nbr -> new_id (dedup).
            let nbr_idx = nbr as usize;
            if !self.adjacency[nbr_idx].contains(&new_id) {
                self.adjacency[nbr_idx].push(new_id);
            }
        }

        Ok(new_id)
    }

    /// Batch-insert `n` points (a flat `n × dim` row-major slice) in order.
    ///
    /// # Errors
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - Any error propagated from [`NgtIndex::insert`].
    pub fn build(&mut self, data: &[f32], n: usize) -> AnnResult<()> {
        if data.len() != n * self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * self.cfg.dim,
                got: data.len(),
            });
        }
        let dim = self.cfg.dim;
        for i in 0..n {
            self.insert(&data[i * dim..i * dim + dim])?;
        }
        Ok(())
    }

    /// Deterministically choose up to `seed_count` distinct start node ids.
    ///
    /// Seeds always include node `0` and the last-inserted node, then fill the
    /// remainder with evenly-spaced ids across the current id range. No
    /// randomness is involved, so a given graph state always yields identical
    /// seeds (and hence identical searches).
    fn seeds(&self) -> Vec<u32> {
        let n = self.adjacency.len();
        if n == 0 {
            return Vec::new();
        }
        let want = self.cfg.seed_count.min(n);
        let mut out: Vec<u32> = Vec::with_capacity(want);
        let push_unique = |out: &mut Vec<u32>, id: u32| {
            if out.len() < want && !out.contains(&id) {
                out.push(id);
            }
        };

        push_unique(&mut out, 0);
        push_unique(&mut out, (n - 1) as u32);

        // Evenly spaced ids across [0, n) to diversify the remaining seeds.
        if out.len() < want {
            // `want` is at least 1 here; guard the divisor.
            let step = (n.max(1)) as f32 / (want.max(1)) as f32;
            let mut t = 0.0_f32;
            while out.len() < want {
                let id = (t as usize).min(n - 1) as u32;
                push_unique(&mut out, id);
                t += step;
                // Safety net against pathological stalls: scan linearly if the
                // spacing failed to introduce a new id.
                if t >= n as f32 {
                    break;
                }
            }
            // Linear fill for any leftover slots.
            let mut id = 0u32;
            while out.len() < want && (id as usize) < n {
                push_unique(&mut out, id);
                id += 1;
            }
        }

        out
    }

    /// ε-relaxed greedy best-first graph search for the `k` approximate nearest
    /// neighbors of `query`, returned ascending by squared-L2 distance.
    ///
    /// The search maintains a visited set, a bounded result max-heap holding the
    /// `k` smallest distances seen, and a frontier min-heap of candidates to
    /// expand. It starts from the deterministic `seeds` (node `0`, the
    /// last-inserted node, then evenly-spaced ids). On each
    /// step it pops the nearest frontier node `c`; if `dist(c)` exceeds
    /// `(1 + ε)` times the current `k`-th best distance (treated as `+∞` while
    /// fewer than `k` results exist) the search terminates. Otherwise every
    /// unvisited neighbor of `c` is evaluated, offered to the result heap, and
    /// pushed onto the frontier.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Err(AnnError::InvalidK {
                k,
                n: self.adjacency.len(),
            });
        }
        if query.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        if self.adjacency.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.adjacency.len();
        let mut visited = vec![false; n];
        // Frontier: min-heap by distance (via Reverse).
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        // Result: max-heap by distance, bounded to k smallest.
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();

        let eps = self.cfg.search_epsilon;

        for seed in self.seeds() {
            let si = seed as usize;
            if si >= n || visited[si] {
                continue;
            }
            visited[si] = true;
            let d = self.dist_to(query, seed);
            frontier.push(Reverse((OrdF32(d), seed)));
            result.push((OrdF32(d), seed));
            if result.len() > k {
                result.pop();
            }
        }

        while let Some(Reverse((OrdF32(c_dist), c_id))) = frontier.pop() {
            // Current k-th best distance (worst kept result), or +inf if we do
            // not yet hold k results.
            let kth_best = if result.len() >= k {
                result.peek().map_or(f32::INFINITY, |(OrdF32(d), _)| *d)
            } else {
                f32::INFINITY
            };
            // ε-relaxed termination: stop once the closest frontier candidate is
            // farther than (1 + ε) * kth_best.
            if c_dist > (1.0 + eps) * kth_best {
                break;
            }

            // `c_id` is always a valid node id (only such ids enter the heaps).
            let neighbors = &self.adjacency[c_id as usize];
            for &nbr in neighbors {
                let ni = nbr as usize;
                if ni >= n || visited[ni] {
                    continue;
                }
                visited[ni] = true;
                let d = self.dist_to(query, nbr);
                frontier.push(Reverse((OrdF32(d), nbr)));
                // Offer to the bounded result heap: insert while we have fewer
                // than k, otherwise replace the current worst if `d` beats it.
                let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(w), _)| *w);
                if result.len() < k {
                    result.push((OrdF32(d), nbr));
                } else if d < worst {
                    result.pop();
                    result.push((OrdF32(d), nbr));
                }
            }
        }

        let mut out: Vec<(u32, f32)> = result.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::l2_sq;

    fn cfg(dim: usize) -> NgtConfig {
        NgtConfig {
            dim,
            edge_count: 4,
            search_epsilon: 0.1,
            seed_count: 2,
        }
    }

    fn idx(dim: usize) -> NgtIndex {
        NgtIndex::new(cfg(dim)).unwrap()
    }

    /// Six points in three clearly separated 2-D clusters.
    fn clustered() -> (Vec<f32>, usize) {
        // clusters around (0,0), (10,10), (-10,5)
        let data = vec![
            0.0, 0.0, // 0
            0.5, -0.3, // 1
            10.0, 10.0, // 2
            9.7, 10.2, // 3
            -10.0, 5.0, // 4
            -9.8, 4.6, // 5
        ];
        (data, 6)
    }

    #[test]
    fn new_empty_index() {
        let index = idx(3);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.dim(), 3);
    }

    #[test]
    fn insert_returns_sequential_ids() {
        let mut index = idx(2);
        assert_eq!(index.insert(&[0.0, 0.0]).unwrap(), 0);
        assert_eq!(index.insert(&[1.0, 1.0]).unwrap(), 1);
        assert_eq!(index.insert(&[2.0, 2.0]).unwrap(), 2);
        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn build_sets_len() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        assert_eq!(index.len(), n);
    }

    #[test]
    fn search_results_sorted_and_bounded() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        let res = index.search(&[0.1, 0.1], 3).unwrap();
        assert!(res.len() <= 3);
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "not ascending: {res:?}");
        }
    }

    #[test]
    fn search_k_larger_than_n_returns_n() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        let res = index.search(&[0.0, 0.0], 100).unwrap();
        assert_eq!(res.len(), n);
    }

    #[test]
    fn search_finds_exact_nearest_in_clusters() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        // Query exactly equal to stored point 3 -> nearest must be id 3, dist ~0.
        let res = index.search(&[9.7, 10.2], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 3, "results: {res:?}");
        assert!(res[0].1.abs() < 1e-5, "dist={}", res[0].1);
    }

    #[test]
    fn search_query_near_cluster_returns_member() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        // Near the (-10,5) cluster: nearest should be 4 or 5.
        let res = index.search(&[-9.9, 4.8], 1).unwrap();
        assert!(res[0].0 == 4 || res[0].0 == 5, "results: {res:?}");
    }

    #[test]
    fn single_point_index_returns_that_point() {
        let mut index = idx(2);
        index.insert(&[3.0, 4.0]).unwrap();
        let res = index.search(&[3.0, 4.0], 5).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-6);
    }

    #[test]
    fn duplicate_points_are_insertable() {
        let mut index = idx(2);
        let id0 = index.insert(&[1.0, 1.0]).unwrap();
        let id1 = index.insert(&[1.0, 1.0]).unwrap();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(index.len(), 2);
        let res = index.search(&[1.0, 1.0], 2).unwrap();
        assert_eq!(res.len(), 2);
        // Both duplicates sit at distance ~0.
        assert!(res.iter().all(|(_, d)| d.abs() < 1e-6));
    }

    #[test]
    fn larger_epsilon_keeps_or_improves_recall() {
        // Build the same graph twice with different epsilon and compare recall
        // of top-1 against brute-force ground truth over several queries.
        let (data, n) = clustered();
        let dim = 2;

        let build_with = |eps: f32| {
            let c = NgtConfig {
                dim,
                edge_count: 3,
                search_epsilon: eps,
                seed_count: 2,
            };
            let mut index = NgtIndex::new(c).unwrap();
            index.build(&data, n).unwrap();
            index
        };
        let small = build_with(0.0);
        let big = build_with(5.0);

        let queries = [[0.2, 0.1], [9.9, 9.8], [-9.9, 4.9], [5.0, 5.0]];
        let brute_top1 = |q: &[f32]| -> u32 {
            let mut best = (0u32, f32::INFINITY);
            for i in 0..n {
                let d = l2_sq(q, &data[i * dim..i * dim + dim]).unwrap();
                if d < best.1 {
                    best = (i as u32, d);
                }
            }
            best.0
        };

        let mut small_hits = 0usize;
        let mut big_hits = 0usize;
        for q in &queries {
            let gt = brute_top1(q);
            if small.search(q, 1).unwrap()[0].0 == gt {
                small_hits += 1;
            }
            if big.search(q, 1).unwrap()[0].0 == gt {
                big_hits += 1;
            }
        }
        assert!(
            big_hits >= small_hits,
            "big_hits={big_hits} small_hits={small_hits}"
        );
    }

    #[test]
    fn deterministic_build_and_search() {
        let (data, n) = clustered();
        let mut a = idx(2);
        let mut b = idx(2);
        a.build(&data, n).unwrap();
        b.build(&data, n).unwrap();
        for q in [[0.0_f32, 0.0], [10.0, 10.0], [-10.0, 5.0], [3.0, 3.0]] {
            let ra = a.search(&q, 3).unwrap();
            let rb = b.search(&q, 3).unwrap();
            assert_eq!(ra, rb, "query {q:?} produced different results");
        }
    }

    #[test]
    fn every_node_has_degree_at_least_one() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        for id in 0..n {
            assert!(
                !index.adjacency[id].is_empty(),
                "node {id} is isolated: {:?}",
                index.adjacency
            );
        }
    }

    #[test]
    fn search_is_self_consistent_for_all_stored_points() {
        let (data, n) = clustered();
        let dim = 2;
        let mut index = idx(dim);
        index.build(&data, n).unwrap();
        // Each stored point should retrieve itself as the top-1 (dist ~0).
        for i in 0..n {
            let q = &data[i * dim..i * dim + dim];
            let res = index.search(q, 1).unwrap();
            assert_eq!(res[0].0 as usize, i, "point {i} did not retrieve itself");
            assert!(res[0].1.abs() < 1e-5);
        }
    }

    #[test]
    fn err_insert_dim_mismatch() {
        let mut index = idx(3);
        assert!(matches!(
            index.insert(&[1.0, 2.0]),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_search_dim_mismatch() {
        let (data, n) = clustered();
        let mut index = idx(2);
        index.build(&data, n).unwrap();
        assert!(matches!(
            index.search(&[1.0, 2.0, 3.0], 1),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_search_k_zero() {
        let mut index = idx(2);
        index.insert(&[0.0, 0.0]).unwrap();
        assert!(matches!(
            index.search(&[0.0, 0.0], 0),
            Err(AnnError::InvalidK { k: 0, .. })
        ));
    }

    #[test]
    fn err_edge_count_zero() {
        let c = NgtConfig {
            dim: 2,
            edge_count: 0,
            search_epsilon: 0.0,
            seed_count: 1,
        };
        assert!(matches!(NgtIndex::new(c), Err(AnnError::Internal { .. })));
    }

    #[test]
    fn err_seed_count_zero() {
        let c = NgtConfig {
            dim: 2,
            edge_count: 1,
            search_epsilon: 0.0,
            seed_count: 0,
        };
        assert!(matches!(NgtIndex::new(c), Err(AnnError::Internal { .. })));
    }

    #[test]
    fn err_negative_epsilon() {
        let c = NgtConfig {
            dim: 2,
            edge_count: 1,
            search_epsilon: -0.5,
            seed_count: 1,
        };
        assert!(matches!(NgtIndex::new(c), Err(AnnError::Internal { .. })));
    }

    #[test]
    fn err_build_data_length_mismatch() {
        let mut index = idx(2);
        // 3 floats for n=2 (needs 4).
        assert!(matches!(
            index.build(&[0.0, 0.0, 1.0], 2),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_dim_zero_in_config() {
        let c = NgtConfig {
            dim: 0,
            edge_count: 1,
            search_epsilon: 0.0,
            seed_count: 1,
        };
        assert!(matches!(
            NgtIndex::new(c),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn len_and_is_empty_consistent() {
        let mut index = idx(2);
        assert!(index.is_empty());
        for i in 0..5 {
            index.insert(&[i as f32, 0.0]).unwrap();
            assert!(!index.is_empty());
            assert_eq!(index.len(), i + 1);
        }
    }

    #[test]
    fn search_on_empty_returns_empty() {
        // Empty graph + valid k -> empty result (not an error).
        let index = idx(2);
        let res = index.search(&[0.0, 0.0], 3).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn larger_dataset_recall_is_high() {
        // A modestly larger grid: every grid point should retrieve itself.
        let dim = 2;
        let mut data = Vec::new();
        let side = 6;
        for x in 0..side {
            for y in 0..side {
                data.push(x as f32);
                data.push(y as f32);
            }
        }
        let n = side * side;
        let mut index = NgtIndex::new(NgtConfig {
            dim,
            edge_count: 6,
            search_epsilon: 0.3,
            seed_count: 4,
        })
        .unwrap();
        index.build(&data, n).unwrap();

        let mut hits = 0usize;
        for i in 0..n {
            let q = &data[i * dim..i * dim + dim];
            let res = index.search(q, 1).unwrap();
            if res[0].0 as usize == i {
                hits += 1;
            }
        }
        let recall = hits as f32 / n as f32;
        assert!(recall >= 0.9, "self-recall too low: {recall}");
    }
}
