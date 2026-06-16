use super::train::train_coarse;
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::pq::adc::{adc_distance, build_adc_table};
use crate::pq::codebook::PqCodebook;
use crate::pq::encode::encode_vector;
use crate::pq::train::train_pq;
use crate::topk::heap::BoundedMaxHeap;

/// Configuration for an IVFADC index (IVF with residual Product Quantization).
#[derive(Debug, Clone)]
pub struct IvfAdcConfig {
    /// Number of coarse Voronoi lists.
    pub n_lists: usize,
    /// Number of PQ sub-spaces (dim % m == 0 required).
    pub m: usize,
    /// PQ centroids per sub-space (typically 256, must be ≤ data size).
    pub ksub: usize,
    /// Number of coarse lists probed at query time.
    pub n_probe: usize,
    /// Iterations for coarse k-means training.
    pub n_iter_coarse: usize,
    /// Iterations for residual PQ training.
    pub n_iter_pq: usize,
}

/// Inverted File Index with Asymmetric Distance Computation over residual PQ codes.
///
/// Vectors are assigned to a coarse centroid; the *residual* (x − centroid) is
/// encoded with PQ, enabling fast approximate distance via ADC tables.
pub struct IvfAdcIndex {
    coarse_centroids: Vec<f32>, // [n_lists * dim]
    pq: PqCodebook,
    posting_lists: Vec<Vec<u32>>,      // [n_lists][n_vecs_in_list]
    residual_codes: Vec<Vec<Vec<u8>>>, // [n_lists][n_vecs_in_list][m bytes]
    dim: usize,
    n_lists: usize,
    n_probe: usize,
}

impl IvfAdcIndex {
    /// Train the coarse quantizer and residual PQ codebook.
    ///
    /// After training, the index is empty; call [`add`](Self::add) to populate it.
    pub fn train(
        data: &[f32],
        n: usize,
        dim: usize,
        cfg: &IvfAdcConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if !dim.is_multiple_of(cfg.m) {
            return Err(AnnError::InvalidNumSubspaces { m: cfg.m, dim });
        }
        if cfg.n_probe > cfg.n_lists {
            return Err(AnnError::InvalidNumProbes {
                nprobe: cfg.n_probe,
                nlist: cfg.n_lists,
            });
        }

        // Step 1: train coarse quantizer.
        let coarse_centroids = train_coarse(data, n, dim, cfg.n_lists, cfg.n_iter_coarse, rng)?;

        // Step 2: assign each point to its nearest coarse centroid.
        let assignments = assign_to_coarse(data, n, dim, &coarse_centroids, cfg.n_lists);

        // Step 3: compute residuals.
        let mut residuals = vec![0.0_f32; n * dim];
        for i in 0..n {
            let list_j = assignments[i];
            let v = &data[i * dim..(i + 1) * dim];
            let c = &coarse_centroids[list_j * dim..(list_j + 1) * dim];
            let r = &mut residuals[i * dim..(i + 1) * dim];
            for d in 0..dim {
                r[d] = v[d] - c[d];
            }
        }

        // Step 4: train residual PQ on residuals.
        let pq = train_pq(&residuals, n, dim, cfg.m, cfg.ksub, cfg.n_iter_pq, rng)?;

        Ok(Self {
            coarse_centroids,
            pq,
            posting_lists: vec![Vec::new(); cfg.n_lists],
            residual_codes: vec![Vec::new(); cfg.n_lists],
            dim,
            n_lists: cfg.n_lists,
            n_probe: cfg.n_probe,
        })
    }

    /// Add a single vector with the given external id to the index.
    pub fn add(&mut self, v: &[f32], id: u32) -> AnnResult<()> {
        if v.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }

        let list_j = nearest_centroid_index(v, &self.coarse_centroids, self.n_lists, self.dim);

        // Compute residual.
        let mut residual = vec![0.0_f32; self.dim];
        let c = &self.coarse_centroids[list_j * self.dim..(list_j + 1) * self.dim];
        for d in 0..self.dim {
            residual[d] = v[d] - c[d];
        }

        let code = encode_vector(&residual, &self.pq);

        self.posting_lists[list_j].push(id);
        self.residual_codes[list_j].push(code);

        Ok(())
    }

    /// Search for the `k` approximate nearest neighbors of `query`.
    ///
    /// Probes `n_probe` coarse lists and uses ADC over the residual PQ codes.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let n_indexed = self.n_indexed();
        if n_indexed == 0 {
            // Return empty result set for an empty index.
            return Ok(Vec::new());
        }

        let actual_k = k.min(n_indexed);
        let mut heap = BoundedMaxHeap::new(actual_k);

        // Determine probe order: top-n_probe coarse centroids by L2².
        let probe_lists = coarse_probe_order(
            query,
            &self.coarse_centroids,
            self.n_lists,
            self.dim,
            self.n_probe,
        );

        for &j in &probe_lists {
            // Residual query for this coarse centroid.
            let c = &self.coarse_centroids[j * self.dim..(j + 1) * self.dim];
            let mut q_res = vec![0.0_f32; self.dim];
            for d in 0..self.dim {
                q_res[d] = query[d] - c[d];
            }

            let adc_table = build_adc_table(&q_res, &self.pq);

            for (k_idx, &id) in self.posting_lists[j].iter().enumerate() {
                let code = &self.residual_codes[j][k_idx];
                let dist = adc_distance(code, &adc_table, self.pq.m, self.pq.ksub);
                heap.push(dist, id as usize);
            }
        }

        let raw = heap.into_sorted_vec();
        Ok(raw
            .into_iter()
            .map(|(id, dist)| (id as u32, dist))
            .collect())
    }

    /// Total number of vectors currently indexed.
    pub fn n_indexed(&self) -> usize {
        self.posting_lists.iter().map(|v| v.len()).sum()
    }
}

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------

fn nearest_centroid_index(v: &[f32], centroids: &[f32], n_lists: usize, dim: usize) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for c in 0..n_lists {
        let center = &centroids[c * dim..(c + 1) * dim];
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

fn assign_to_coarse(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    n_lists: usize,
) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let v = &data[i * dim..(i + 1) * dim];
            nearest_centroid_index(v, centroids, n_lists, dim)
        })
        .collect()
}

fn coarse_probe_order(
    query: &[f32],
    centroids: &[f32],
    n_lists: usize,
    dim: usize,
    n_probe: usize,
) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = (0..n_lists)
        .map(|c| {
            let center = &centroids[c * dim..(c + 1) * dim];
            let d: f32 = query
                .iter()
                .zip(center.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (c, d)
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(n_probe).map(|(c, _)| *c).collect()
}

// --------------------------------------------------------------------------
// Unit tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn rand_vecs_normal(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_normal()).collect()
    }

    fn default_cfg() -> IvfAdcConfig {
        IvfAdcConfig {
            n_lists: 4,
            m: 2,
            ksub: 8,
            n_probe: 2,
            n_iter_coarse: 10,
            n_iter_pq: 10,
        }
    }

    // 1. Train returns correct structure — n_indexed == 0 after training.
    #[test]
    fn train_returns_correct_structure() {
        let mut rng = make_rng(1);
        let n = 400;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        assert_eq!(idx.n_indexed(), 0, "no vectors added yet");
        assert_eq!(idx.n_lists, 4);
        assert_eq!(idx.dim, dim);
    }

    // 2. Add increments count.
    #[test]
    fn add_increments_count() {
        let mut rng = make_rng(2);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..50 {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        assert_eq!(idx.n_indexed(), 50);
    }

    // 3. Search returns k results.
    #[test]
    fn search_returns_k_results() {
        let mut rng = make_rng(3);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        let results = idx.search(&data[0..dim], 5).expect("search should succeed");
        assert_eq!(results.len(), 5, "expected exactly 5 results");
    }

    // 4. Search top-1 finds the query point.
    #[test]
    fn search_top1_finds_query() {
        let mut rng = make_rng(4);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        // Full probe to maximise recall.
        let cfg = IvfAdcConfig {
            n_probe: 4,
            ..default_cfg()
        };
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        let results = idx.search(&data[0..dim], 5).expect("search should succeed");
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "query id=0 should appear in top-5; got {:?}",
            results
        );
    }

    // 5. Residuals smaller than raw vectors (mean squared norm).
    #[test]
    fn residuals_smaller_than_raw() {
        let mut rng = make_rng(5);
        let n = 300;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");

        let mean_raw_norm: f32 = data
            .chunks_exact(dim)
            .map(|v| v.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            / n as f32;

        let mean_res_norm: f32 = data
            .chunks_exact(dim)
            .map(|v| {
                let j = nearest_centroid_index(v, &idx.coarse_centroids, idx.n_lists, idx.dim);
                let c = &idx.coarse_centroids[j * idx.dim..(j + 1) * idx.dim];
                v.iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / n as f32;

        assert!(
            mean_res_norm < mean_raw_norm,
            "residual norm {mean_res_norm} should be less than raw norm {mean_raw_norm}"
        );
    }

    // 6. Full probe high recall — query multiple training points → recall ≥ 50%.
    #[test]
    fn full_probe_high_recall() {
        let mut rng = make_rng(6);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = IvfAdcConfig {
            n_probe: 4, // probe all lists
            ..default_cfg()
        };
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        // Query 20 training points, check that at least 50% appear in top-5.
        let n_queries = 20;
        let k = 5;
        let mut hits = 0usize;
        for qi in 0..n_queries {
            let results = idx
                .search(&data[qi * dim..(qi + 1) * dim], k)
                .expect("search should succeed");
            if results.iter().any(|(id, _)| *id == qi as u32) {
                hits += 1;
            }
        }
        let recall = hits as f32 / n_queries as f32;
        assert!(
            recall >= 0.5,
            "recall@5 = {recall:.2} < 0.50; IVFADC with full probe should find queries reasonably well"
        );
    }

    // 7. n_probe > n_lists → Err.
    #[test]
    fn nprobe_gt_nlists_error() {
        let mut rng = make_rng(7);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = IvfAdcConfig {
            n_probe: 10, // > n_lists=4
            ..default_cfg()
        };
        let result = IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng);
        assert!(result.is_err(), "expected Err for n_probe > n_lists");
    }

    // 8. m not dividing dim → Err(InvalidNumSubspaces).
    #[test]
    fn m_not_dividing_dim_error() {
        let mut rng = make_rng(8);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = IvfAdcConfig {
            m: 3, // 8 % 3 != 0
            ..default_cfg()
        };
        let result = IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng);
        assert!(
            matches!(result, Err(AnnError::InvalidNumSubspaces { .. })),
            "expected InvalidNumSubspaces, got {:?}",
            result.err()
        );
    }

    // 9. Search on empty index → Ok with 0 results.
    #[test]
    fn search_empty_index_returns_empty() {
        let mut rng = make_rng(9);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        let results = idx.search(&data[0..dim], 5).expect("search should succeed");
        assert!(results.is_empty(), "empty index should return 0 results");
    }

    // 10. IDs preserved — add with custom ids and find them.
    #[test]
    fn ids_preserved() {
        let mut rng = make_rng(10);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = IvfAdcConfig {
            n_probe: 4,
            ..default_cfg()
        };
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        // Add three specific vectors with custom ids.
        idx.add(&data[0..dim], 100).expect("add should succeed");
        idx.add(&data[dim..2 * dim], 200)
            .expect("add should succeed");
        idx.add(&data[2 * dim..3 * dim], 300)
            .expect("add should succeed");

        let results = idx.search(&data[0..dim], 3).expect("search should succeed");
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&100), "id=100 should appear; got {ids:?}");
    }

    // 11. ADC distance is finite and non-negative.
    #[test]
    fn adc_dist_finite_nonneg() {
        let mut rng = make_rng(11);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = IvfAdcConfig {
            n_probe: 4,
            ..default_cfg()
        };
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        let results = idx
            .search(&data[0..dim], 10)
            .expect("search should succeed");
        for (_, dist) in &results {
            assert!(dist.is_finite(), "dist must be finite, got {dist}");
            assert!(*dist >= 0.0, "dist must be non-negative, got {dist}");
        }
    }

    // 12. Posting list sizes sum to n after adding n vectors.
    #[test]
    fn posting_list_sizes_sum_to_n() {
        let mut rng = make_rng(12);
        let n = 150;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let cfg = default_cfg();
        let mut idx =
            IvfAdcIndex::train(&data, n, dim, &cfg, &mut rng).expect("training should succeed");
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i as u32)
                .expect("add should succeed");
        }
        let total: usize = idx.posting_lists.iter().map(|v| v.len()).sum();
        assert_eq!(total, n, "posting list sum should equal n={n}");
    }
}
