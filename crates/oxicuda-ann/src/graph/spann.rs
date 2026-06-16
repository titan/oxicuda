//! SPANN — memory-disk hybrid posting-list ANN index (in-memory core).
//!
//! Reference: Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu,
//! Zengzhong Li, Mao Yang, Jingdong Wang, *"SPANN: Highly-efficient Billion-scale
//! Approximate Nearest Neighbor Search"*, NeurIPS 2021.
//!
//! SPANN partitions the dataset into many fine-grained **posting lists**, one
//! per centroid, with roughly `√n` centroids. The two ideas we implement:
//!
//! * **Closure / boundary-point duplication.** Each vector is assigned not only
//!   to its nearest centroid but *also* to every other nearby centroid whose
//!   distance is within a relative slack of the nearest (`dist(x, c) ≤
//!   (1 + ε) · dist(x, c*)`), capped at `replica_count` assignments. Points near
//!   a Voronoi boundary therefore appear in several posting lists, which is what
//!   lets a query that lands in the "wrong" cell still recover boundary
//!   neighbours. This is the paper's central trick for closing the recall gap of
//!   plain inverted-file search.
//! * **Coarse centroid index + posting-list scan.** At query time we find the
//!   `n_probe` nearest centroids (brute force over the small centroid set — the
//!   "in-memory SPTAG" head index in the paper) and exhaustively re-rank the
//!   union of their posting lists against the *original* vectors.
//!
//! Centroids are produced by k-means (reusing
//! [`crate::kmeans::kmeans::KMeans`]). The boundary expansion uses **squared**
//! L2 with the slack applied on the squared scale, i.e. a replica is kept when
//! `dist²(x, c) ≤ (1 + ε)² · dist²(x, c*)`.
//!
//! Results are `(u32, f32)` = `(id, dist²)` ascending.

use std::collections::HashSet;

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// SPANN build / search configuration.
#[derive(Debug, Clone)]
pub struct SpannConfig {
    /// Number of centroids (posting lists). When `0`, defaults to `⌈√n⌉`.
    pub n_centroids: usize,
    /// k-means iteration budget for centroid training.
    pub kmeans_iters: usize,
    /// Boundary slack ε (`>= 0`): a vector replicates into centroid `c` when
    /// `dist²(x, c) ≤ (1 + ε)² · dist²(x, c*)` where `c*` is its nearest
    /// centroid.
    pub boundary_epsilon: f32,
    /// Maximum number of posting lists any single vector may join (`>= 1`).
    pub replica_count: usize,
    /// Default number of centroids probed per query (`>= 1`).
    pub n_probe: usize,
}

impl Default for SpannConfig {
    fn default() -> Self {
        Self {
            n_centroids: 0, // → √n
            kmeans_iters: 25,
            boundary_epsilon: 0.30,
            replica_count: 8,
            n_probe: 8,
        }
    }
}

/// In-memory SPANN index.
pub struct SpannIndex {
    /// Flat `n × dim` row-major original vectors.
    points: Vec<f32>,
    /// Flat `n_centroids × dim` row-major centroids (the head index).
    centroids: Vec<f32>,
    /// `posting[c]` = vector ids assigned (incl. boundary replicas) to centroid
    /// `c`.
    posting: Vec<Vec<u32>>,
    /// Number of centroids.
    n_centroids: usize,
    /// Vector dimensionality.
    dim: usize,
    /// Number of original vectors.
    n: usize,
    /// Cached configuration (with `n_centroids` resolved to the concrete value).
    cfg: SpannConfig,
}

impl SpannIndex {
    /// Number of indexed (original) vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when no vectors are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of centroids / posting lists.
    #[must_use]
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// Read-only access to centroid `c`'s posting list.
    #[must_use]
    pub fn posting_list(&self, c: usize) -> &[u32] {
        if c < self.posting.len() {
            &self.posting[c]
        } else {
            &[]
        }
    }

    /// Total number of (vector, centroid) assignments across all posting lists.
    /// Equals `n` when there is no replication and exceeds it when boundary
    /// points are duplicated.
    #[must_use]
    pub fn total_assignments(&self) -> usize {
        self.posting.iter().map(|p| p.len()).sum()
    }

    /// Flat centroid storage `[n_centroids, dim]`.
    #[must_use]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    fn point(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.points[s..s + self.dim]
    }

    fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }

    fn dist_to_centroid(v: &[f32], c: &[f32]) -> f32 {
        v.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    /// Build a SPANN index over `data` (row-major `n × dim`).
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] if `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - [`AnnError::Internal`] if `cfg.replica_count == 0`, `cfg.n_probe == 0`,
    ///   or `cfg.boundary_epsilon` is negative / non-finite.
    /// - Propagates [`KMeans::fit`] errors.
    pub fn build(
        data: &[f32],
        n: usize,
        dim: usize,
        cfg: SpannConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim });
        }
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if cfg.replica_count == 0 || cfg.n_probe == 0 {
            return Err(AnnError::Internal {
                msg: "replica_count and n_probe must both be >= 1".to_string(),
            });
        }
        if cfg.boundary_epsilon < 0.0 || !cfg.boundary_epsilon.is_finite() {
            return Err(AnnError::Internal {
                msg: format!(
                    "boundary_epsilon must be finite and >= 0, got {}",
                    cfg.boundary_epsilon
                ),
            });
        }

        // Resolve centroid count: default to ⌈√n⌉, clamp into [1, n].
        let mut n_centroids = if cfg.n_centroids == 0 {
            (n as f32).sqrt().ceil() as usize
        } else {
            cfg.n_centroids
        };
        n_centroids = n_centroids.clamp(1, n);

        // Train the head index (centroids) with k-means.
        let km = KMeans::fit(data, n, dim, n_centroids, cfg.kmeans_iters, rng)?;
        let centroids = km.centroids().to_vec();

        let mut resolved_cfg = cfg.clone();
        resolved_cfg.n_centroids = n_centroids;

        let mut index = Self {
            points: data.to_vec(),
            centroids,
            posting: vec![Vec::new(); n_centroids],
            n_centroids,
            dim,
            n,
            cfg: resolved_cfg,
        };

        index.assign_with_boundary();
        Ok(index)
    }

    /// Assign every vector to its nearest centroid plus all sufficiently-close
    /// boundary centroids (closure / duplication), capped at `replica_count`.
    fn assign_with_boundary(&mut self) {
        let eps = self.cfg.boundary_epsilon;
        // Slack on the squared scale: keep c when d²(x,c) <= (1+ε)² · d²(x,c*).
        let slack_sq = (1.0 + eps) * (1.0 + eps);
        let cap = self.cfg.replica_count.max(1);

        for id in 0..self.n as u32 {
            let v = self.point(id);
            // Distances to all centroids.
            let mut cd: Vec<(usize, f32)> = (0..self.n_centroids)
                .map(|c| (c, Self::dist_to_centroid(v, self.centroid(c))))
                .collect();
            cd.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let nearest_sq = cd[0].1;
            let threshold = slack_sq * nearest_sq;

            // Always assign to the nearest centroid; then duplicate into
            // boundary centroids within the slack, up to `cap` total.
            let mut assigned = 0usize;
            for &(c, d) in &cd {
                if assigned == 0 {
                    self.posting[c].push(id);
                    assigned += 1;
                    continue;
                }
                if assigned >= cap {
                    break;
                }
                // Within slack? (When nearest_sq == 0, threshold == 0, so only
                // exact-coincident centroids replicate — the right behaviour.)
                if d <= threshold {
                    self.posting[c].push(id);
                    assigned += 1;
                } else {
                    // Sorted ascending: once out of slack, all later are too.
                    break;
                }
            }
        }
    }

    /// The `n_probe` nearest centroids to `query`, ascending by distance.
    fn nearest_centroids(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        let mut cd: Vec<(usize, f32)> = (0..self.n_centroids)
            .map(|c| (c, Self::dist_to_centroid(query, self.centroid(c))))
            .collect();
        cd.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        cd.truncate(n_probe.min(self.n_centroids));
        cd.into_iter().map(|(c, _)| c).collect()
    }

    /// Approximate top-`k` search using the default `n_probe`.
    ///
    /// # Errors
    /// See [`Self::search_with_probe`].
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        self.search_with_probe(query, k, self.cfg.n_probe)
    }

    /// Approximate top-`k` search probing the `n_probe` nearest centroids and
    /// exhaustively re-ranking the union of their posting lists against the
    /// original vectors. Returns `(id, dist²)` ascending.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the index is empty.
    pub fn search_with_probe(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
    ) -> AnnResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Err(AnnError::InvalidK { k, n: self.n });
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.n == 0 {
            return Err(AnnError::IndexEmpty);
        }

        let probe = n_probe.max(1);
        let cells = self.nearest_centroids(query, probe);

        // Gather candidate ids from the probed posting lists (deduplicated,
        // since boundary points may appear in several).
        let mut seen: HashSet<u32> = HashSet::new();
        let mut cands: Vec<(u32, f32)> = Vec::new();
        for &c in &cells {
            for &id in &self.posting[c] {
                if seen.insert(id) {
                    let d = Self::dist_to_centroid(query, self.point(id));
                    cands.push((id, d));
                }
            }
        }

        cands.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        cands.truncate(k.min(cands.len()));
        Ok(cands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::l2_sq;

    fn clustered_data(rng: &mut LcgRng) -> (Vec<f32>, usize, usize) {
        // 9 well-separated 4-D clusters, ~16 each => n = 144 so √n = 12.
        let dim = 4;
        let mut centres = Vec::new();
        for gx in 0..3 {
            for gy in 0..3 {
                centres.push([gx as f32 * 30.0, gy as f32 * 30.0, 0.0, 0.0]);
            }
        }
        let mut data = Vec::new();
        for c in &centres {
            for _ in 0..16 {
                for &cx in c.iter().take(dim) {
                    data.push(cx + (rng.next_f32() - 0.5) * 5.0);
                }
            }
        }
        (data, centres.len() * 16, dim)
    }

    fn brute_topk(data: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                (
                    i,
                    l2_sq(query, &data[i * dim..(i + 1) * dim]).expect("value should be present"),
                )
            })
            .collect();
        d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        d.truncate(k);
        d.into_iter().map(|(i, _)| i).collect()
    }

    #[test]
    fn build_sets_metadata_and_sqrt_n_centroids() {
        let mut rng = LcgRng::new(1);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(idx.len(), n);
        assert_eq!(idx.dim(), dim);
        // Default n_centroids == ⌈√n⌉.
        let expected = (n as f32).sqrt().ceil() as usize;
        assert_eq!(
            idx.n_centroids(),
            expected,
            "expected ~√n = {expected} centroids, got {}",
            idx.n_centroids()
        );
    }

    // LOAD-BEARING: boundary duplication. With ε > 0, total assignments must
    // exceed n (some vectors land in multiple posting lists); with ε == 0 and
    // replica_count == 1 there is no duplication (exactly n assignments).
    #[test]
    fn boundary_duplication_increases_assignments() {
        let mut rng = LcgRng::new(7);
        let (data, n, dim) = clustered_data(&mut rng);

        // No duplication baseline.
        let cfg_none = SpannConfig {
            n_centroids: 16,
            kmeans_iters: 20,
            boundary_epsilon: 0.0,
            replica_count: 1,
            n_probe: 4,
        };
        let idx_none =
            SpannIndex::build(&data, n, dim, cfg_none, &mut rng).expect("build should succeed");
        assert_eq!(
            idx_none.total_assignments(),
            n,
            "ε=0, replica=1 must give exactly n assignments"
        );

        // Generous duplication.
        let cfg_dup = SpannConfig {
            n_centroids: 16,
            kmeans_iters: 20,
            boundary_epsilon: 0.8,
            replica_count: 6,
            n_probe: 4,
        };
        let idx_dup =
            SpannIndex::build(&data, n, dim, cfg_dup, &mut rng).expect("build should succeed");
        assert!(
            idx_dup.total_assignments() > n,
            "ε=0.8 should duplicate boundary points: total={} n={n}",
            idx_dup.total_assignments()
        );
    }

    // LOAD-BEARING: a constructed boundary point is duplicated into both
    // adjacent cells. Place two centroids and a point exactly on the midline.
    #[test]
    fn midline_point_replicates_into_both_cells() {
        // Two clusters along x; a point on the boundary should join both lists.
        let dim = 2;
        // Cluster A around x=0, cluster B around x=10. A midline point at x=5.
        let mut data: Vec<f32> = Vec::new();
        // 8 points near 0
        for i in 0..8 {
            data.push(0.0 + i as f32 * 0.01);
            data.push(0.0);
        }
        // 8 points near 10
        for i in 0..8 {
            data.push(10.0 + i as f32 * 0.01);
            data.push(0.0);
        }
        // 1 midline point at (5,0) — id 16.
        data.push(5.0);
        data.push(0.0);
        let n = 17;

        let cfg = SpannConfig {
            n_centroids: 2,
            kmeans_iters: 30,
            boundary_epsilon: 0.5, // (1.5)²=2.25 slack covers the symmetric midline
            replica_count: 2,
            n_probe: 2,
        };
        let mut rng = LcgRng::new(123);
        let idx = SpannIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");
        assert_eq!(idx.n_centroids(), 2);
        // The midline point (id 16) is equidistant to both centroids, so with
        // ε ≥ 0 the slack is satisfied for the 2nd centroid => it appears twice.
        let appearances = (0..idx.n_centroids())
            .filter(|&c| idx.posting_list(c).contains(&16))
            .count();
        assert_eq!(
            appearances, 2,
            "midline point must be duplicated into both cells"
        );
    }

    // LOAD-BEARING: recall@k vs brute force is high on clustered data with
    // enough probes.
    #[test]
    fn recall_high_with_enough_probes() {
        let mut rng = LcgRng::new(11);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = SpannConfig {
            n_centroids: 12,
            kmeans_iters: 30,
            boundary_epsilon: 0.5,
            replica_count: 8,
            n_probe: 12, // probe all centroids → effectively exhaustive
        };
        let idx = SpannIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");

        let k = 10;
        let n_queries = 25;
        let mut hits = 0usize;
        let mut q_rng = LcgRng::new(404);
        for _ in 0..n_queries {
            let base = (q_rng.next_u32() as usize) % n;
            let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
            for v in query.iter_mut() {
                *v += (q_rng.next_f32() - 0.5) * 1.0;
            }
            let gt: HashSet<usize> = brute_topk(&data, n, dim, &query, k).into_iter().collect();
            let got = idx
                .search_with_probe(&query, k, 12)
                .expect("search_with_probe should succeed");
            hits += got
                .iter()
                .filter(|&&(id, _)| gt.contains(&(id as usize)))
                .count();
        }
        let recall = hits as f32 / (n_queries * k) as f32;
        assert!(recall > 0.9, "SPANN recall@{k} = {recall:.3} <= 0.9");
    }

    // More probes never reduce recall (monotone).
    #[test]
    fn more_probes_do_not_reduce_recall() {
        let mut rng = LcgRng::new(13);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = SpannConfig {
            n_centroids: 12,
            kmeans_iters: 30,
            boundary_epsilon: 0.4,
            replica_count: 6,
            n_probe: 2,
        };
        let idx = SpannIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");
        let k = 10;
        let n_queries = 25;
        let measure = |probe: usize| -> f32 {
            let mut q_rng = LcgRng::new(505);
            let mut hits = 0usize;
            for _ in 0..n_queries {
                let base = (q_rng.next_u32() as usize) % n;
                let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
                for v in query.iter_mut() {
                    *v += (q_rng.next_f32() - 0.5) * 2.0;
                }
                let gt: HashSet<usize> = brute_topk(&data, n, dim, &query, k).into_iter().collect();
                let got = idx
                    .search_with_probe(&query, k, probe)
                    .expect("search_with_probe should succeed");
                hits += got
                    .iter()
                    .filter(|&&(id, _)| gt.contains(&(id as usize)))
                    .count();
            }
            hits as f32 / (n_queries * k) as f32
        };
        let r_low = measure(1);
        let r_high = measure(12);
        assert!(
            r_high >= r_low - 1e-6,
            "recall(probe=12)={r_high:.3} < recall(probe=1)={r_low:.3}"
        );
    }

    #[test]
    fn search_finds_exact_self() {
        let mut rng = LcgRng::new(17);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        for &probe_id in &[0usize, 50, 100, 143] {
            let q = &data[probe_id * dim..(probe_id + 1) * dim];
            let res = idx
                .search_with_probe(q, 1, idx.n_centroids())
                .expect("value should be present");
            assert!(!res.is_empty());
            let d = l2_sq(
                q,
                &data[res[0].0 as usize * dim..(res[0].0 as usize + 1) * dim],
            )
            .expect("value should be present");
            assert!(d < 1e-4, "probe_id={probe_id} found={} d={d}", res[0].0);
        }
    }

    #[test]
    fn every_point_in_at_least_one_list() {
        let mut rng = LcgRng::new(19);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        let mut seen = vec![false; n];
        for c in 0..idx.n_centroids() {
            for &id in idx.posting_list(c) {
                seen[id as usize] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "every vector must be assigned");
    }

    #[test]
    fn deterministic_build_same_seed() {
        let (data, n, dim) = {
            let mut r = LcgRng::new(31);
            clustered_data(&mut r)
        };
        let mut a_rng = LcgRng::new(31);
        let mut b_rng = LcgRng::new(31);
        let a = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut a_rng)
            .expect("value should be present");
        let b = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut b_rng)
            .expect("value should be present");
        assert_eq!(a.n_centroids(), b.n_centroids());
        for c in 0..a.n_centroids() {
            assert_eq!(a.posting_list(c), b.posting_list(c), "list {c}");
        }
    }

    #[test]
    fn single_point_index() {
        let mut rng = LcgRng::new(23);
        let data = vec![3.0_f32, 4.0];
        let idx = SpannIndex::build(&data, 1, 2, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.n_centroids(), 1);
        let res = idx
            .search(&[3.0_f32, 4.0], 1)
            .expect("search should succeed");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-6);
    }

    #[test]
    fn err_empty_and_dim_zero() {
        let mut rng = LcgRng::new(37);
        assert!(matches!(
            SpannIndex::build(&[], 0, 2, SpannConfig::default(), &mut rng),
            Err(AnnError::EmptyInput)
        ));
        assert!(matches!(
            SpannIndex::build(&[], 1, 0, SpannConfig::default(), &mut rng),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn err_data_length_mismatch() {
        let mut rng = LcgRng::new(41);
        assert!(matches!(
            SpannIndex::build(&[0.0_f32, 1.0, 2.0], 2, 2, SpannConfig::default(), &mut rng),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_zero_replica_or_probe() {
        let mut rng = LcgRng::new(43);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = SpannConfig {
            n_centroids: 8,
            kmeans_iters: 5,
            boundary_epsilon: 0.1,
            replica_count: 0,
            n_probe: 2,
        };
        assert!(matches!(
            SpannIndex::build(&data, n, dim, cfg, &mut rng),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_negative_epsilon() {
        let mut rng = LcgRng::new(47);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = SpannConfig {
            n_centroids: 8,
            kmeans_iters: 5,
            boundary_epsilon: -0.1,
            replica_count: 2,
            n_probe: 2,
        };
        assert!(matches!(
            SpannIndex::build(&data, n, dim, cfg, &mut rng),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_search_k_zero_and_wrong_dim() {
        let mut rng = LcgRng::new(53);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = SpannIndex::build(&data, n, dim, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        assert!(matches!(
            idx.search(&data[0..dim], 0),
            Err(AnnError::InvalidK { k: 0, .. })
        ));
        let bad = vec![0.0_f32; dim + 1];
        assert!(matches!(
            idx.search(&bad, 5),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn search_k_greater_than_candidates() {
        let mut rng = LcgRng::new(59);
        let data = vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let idx = SpannIndex::build(&data, 3, 2, SpannConfig::default(), &mut rng)
            .expect("value should be present");
        let res = idx
            .search_with_probe(&[0.0_f32, 0.0], 100, idx.n_centroids())
            .expect("value should be present");
        assert!(res.len() <= 3);
        assert!(!res.is_empty());
    }
}
