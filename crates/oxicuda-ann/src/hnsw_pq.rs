//! HNSW-PQ — compressed HNSW with Product-Quantized codes per node and ADC search.
//!
//! Stores PQ codes (instead of raw vectors) at every graph node. Distance from
//! a query to a node is estimated via the Asymmetric Distance Computation
//! (ADC) table: at query time we precompute, for each PQ subspace `s` and
//! centroid index `k`, the squared-L2 distance between the query subvector
//! `q_s` and the centroid `c_{s,k}`. The estimated distance from the query to
//! a stored point `i` is then the sum, over subspaces, of the table value at
//! `(s, code[i, s])`.
//!
//! Build strategy
//! --------------
//! 1. Train a PQ codebook on `data` (per-subspace k-means).
//! 2. Encode every input vector into a `pq_m`-byte PQ code.
//! 3. Reconstruct (decode) each vector by concatenating the chosen subspace
//!    centroids, and feed those reconstructed vectors into a standard
//!    [`HnswGraph`] build via [`hnsw_insert`].
//!
//! Because `‖q − reconstruct(code[i])‖² = Σ_s ‖q_s − c_{s, code[i,s]}‖²`,
//! distances computed over the reconstructed vectors inside the HNSW build
//! and search loops are exactly equal to ADC distances. This guarantees that
//! the graph is constructed *using ADC distances* (as the module contract
//! requires) while reusing the verified [`hnsw_insert`] / [`hnsw_search`]
//! routines without modification.

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::hnsw::graph::HnswGraph;
use crate::hnsw::insert::hnsw_insert;
use crate::hnsw::search::hnsw_search;
use crate::pq::codebook::PqCodebook;
use crate::pq::encode::{encode_batch, encode_vector};
use crate::pq::train::train_pq;

/// Configuration for an [`HnswPq`] index.
#[derive(Debug, Clone)]
pub struct HnswPqConfig {
    /// Vector dimensionality (must be `>= 1` and divisible by `pq_m`).
    pub dim: usize,
    /// HNSW maximum out-degree per node at layers `>= 1` (must be `>= 2`).
    pub m: usize,
    /// HNSW build-time candidate-list size (`ef_construction`, `>= 1`).
    pub ef_construction: usize,
    /// HNSW search-time candidate-list size (`ef`, `>= 1`).
    pub ef: usize,
    /// Number of PQ subspaces (`>= 1`, must divide `dim`).
    pub pq_m: usize,
    /// Centroids per PQ subspace (`2..=256` so codes fit in `u8`).
    pub pq_ksub: usize,
}

/// HNSW with PQ-compressed node representation and ADC-based scoring.
pub struct HnswPq {
    /// Underlying HNSW topology. Its node vectors are the *reconstructed*
    /// (decoded) PQ approximations, which preserves the build-time graph
    /// invariants while implicitly making node→node and query→node distances
    /// equal to ADC distances.
    graph: HnswGraph,
    /// Trained PQ codebook (`pq_m` subspaces × `pq_ksub` centroids × `dsub`).
    codebook: PqCodebook,
    /// Flat PQ codes (`n × pq_m`, row-major), one byte per subspace per point.
    pq_codes: Vec<u8>,
    /// Cached configuration.
    cfg: HnswPqConfig,
    /// Number of indexed points (== `pq_codes.len() / pq_m`).
    n: usize,
}

impl HnswPq {
    /// Create an empty index from a validated configuration.
    ///
    /// # Errors
    /// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
    /// - [`AnnError::InvalidNumSubspaces`] if `pq_m == 0` or `dim % pq_m != 0`.
    /// - [`AnnError::InvalidK`] if `pq_ksub < 2` or `pq_ksub > 256`.
    /// - [`AnnError::Internal`] for invalid `m`, `ef_construction`, `ef`.
    pub fn new(cfg: HnswPqConfig) -> AnnResult<Self> {
        if cfg.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: cfg.dim });
        }
        if cfg.pq_m == 0 || !cfg.dim.is_multiple_of(cfg.pq_m) {
            return Err(AnnError::InvalidNumSubspaces {
                m: cfg.pq_m,
                dim: cfg.dim,
            });
        }
        if cfg.pq_ksub < 2 || cfg.pq_ksub > 256 {
            return Err(AnnError::InvalidK {
                k: cfg.pq_ksub,
                n: 256,
            });
        }
        if cfg.m < 2 {
            return Err(AnnError::Internal {
                msg: format!("HNSW m must be >= 2, got {}", cfg.m),
            });
        }
        if cfg.ef_construction == 0 {
            return Err(AnnError::Internal {
                msg: "ef_construction must be >= 1".to_string(),
            });
        }
        if cfg.ef == 0 {
            return Err(AnnError::Internal {
                msg: "ef must be >= 1".to_string(),
            });
        }

        let dsub = cfg.dim / cfg.pq_m;
        let graph = HnswGraph::new(cfg.dim, cfg.m, cfg.ef_construction, cfg.ef);
        let codebook = PqCodebook::new(cfg.pq_m, cfg.pq_ksub, dsub);
        Ok(Self {
            graph,
            codebook,
            pq_codes: Vec::new(),
            cfg,
            n: 0,
        })
    }

    /// Train the PQ codebook on `data` (row-major `n × dim`), encode all
    /// vectors, then build the HNSW topology using ADC distances (achieved by
    /// inserting the reconstructed PQ approximations into the underlying
    /// [`HnswGraph`]).
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] if `n == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - Any error propagated from [`train_pq`].
    pub fn build(&mut self, data: &[f32], n: usize, rng: &mut LcgRng) -> AnnResult<()> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if data.len() != n * self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * self.cfg.dim,
                got: data.len(),
            });
        }

        // Train PQ codebook (per-subspace k-means).
        let n_epochs: usize = 25;
        self.codebook = train_pq(
            data,
            n,
            self.cfg.dim,
            self.cfg.pq_m,
            self.cfg.pq_ksub,
            n_epochs,
            rng,
        )?;

        // Encode every input vector into a `pq_m`-byte code.
        self.pq_codes = encode_batch(data, n, &self.codebook);

        // Reset the underlying HNSW (in case build() is called more than once).
        self.graph = HnswGraph::new(
            self.cfg.dim,
            self.cfg.m,
            self.cfg.ef_construction,
            self.cfg.ef,
        );

        // Reconstruct every point from its PQ code and feed it to hnsw_insert.
        // L2 over reconstructions == ADC distance, so the graph is built using
        // ADC distances as specified.
        let dim = self.cfg.dim;
        let mut reconstructed = vec![0.0_f32; dim];
        for i in 0..n {
            self.reconstruct_into(i, &mut reconstructed)?;
            hnsw_insert(&mut self.graph, &reconstructed, rng);
        }

        self.n = n;
        Ok(())
    }

    /// Build the per-query ADC table `[pq_m × pq_ksub]`: for each subspace `s`
    /// and centroid index `k`, the squared-L2 distance between the query
    /// subvector `q_s` and centroid `c_{s,k}`.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    pub fn adc_table(&self, query: &[f32]) -> AnnResult<Vec<f32>> {
        if query.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        let pq_m = self.cfg.pq_m;
        let ksub = self.cfg.pq_ksub;
        let dsub = self.codebook.dsub;
        let mut table = vec![0.0_f32; pq_m * ksub];
        for s in 0..pq_m {
            let q_sub = &query[s * dsub..(s + 1) * dsub];
            for k in 0..ksub {
                let centroid = self.codebook.centroid(s, k);
                let d: f32 = q_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                table[s * ksub + k] = d;
            }
        }
        Ok(table)
    }

    /// Estimate the squared-L2 distance from the query (whose ADC `table` was
    /// produced by [`HnswPq::adc_table`]) to stored node `node_id` by summing
    /// `table[s · ksub + code[node_id · pq_m + s]]` across all subspaces.
    ///
    /// # Errors
    /// - [`AnnError::IdOutOfRange`] if `node_id >= len()`.
    /// - [`AnnError::DimensionMismatch`] if `table.len() != pq_m * pq_ksub`.
    pub fn adc_distance(&self, table: &[f32], node_id: u32) -> AnnResult<f32> {
        let pq_m = self.cfg.pq_m;
        let ksub = self.cfg.pq_ksub;
        if table.len() != pq_m * ksub {
            return Err(AnnError::DimensionMismatch {
                expected: pq_m * ksub,
                got: table.len(),
            });
        }
        let idx = node_id as usize;
        if idx >= self.n {
            return Err(AnnError::IdOutOfRange { id: idx, n: self.n });
        }
        let row = &self.pq_codes[idx * pq_m..(idx + 1) * pq_m];
        let mut acc = 0.0_f32;
        for (s, &code) in row.iter().enumerate() {
            acc += table[s * ksub + code as usize];
        }
        Ok(acc)
    }

    /// Approximate top-`k` nearest neighbors of `query`, returned ascending by
    /// ADC distance. Uses the HNSW topology over the PQ-reconstructed nodes,
    /// where the graph's L2² is mathematically identical to ADC.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if no points have been indexed.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Err(AnnError::InvalidK { k, n: self.n });
        }
        if query.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        if self.n == 0 {
            return Err(AnnError::IndexEmpty);
        }

        // The graph stores PQ-reconstructed vectors, so its native L2² is ADC.
        // Cap k at n to match the documented "search k > n returns n" contract.
        let effective_k = k.min(self.n);
        hnsw_search(&self.graph, query, effective_k)
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when no points have been indexed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Read-only access to the raw PQ codes (`n × pq_m`, row-major).
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.pq_codes
    }

    /// Read-only access to the trained PQ codebook.
    #[must_use]
    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }

    /// Reconstruct point `idx` from its PQ code by concatenating the chosen
    /// centroid per subspace into `out`.
    fn reconstruct_into(&self, idx: usize, out: &mut [f32]) -> AnnResult<()> {
        if out.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: out.len(),
            });
        }
        let pq_m = self.cfg.pq_m;
        let dsub = self.codebook.dsub;
        let row = &self.pq_codes[idx * pq_m..(idx + 1) * pq_m];
        for (s, &code) in row.iter().enumerate() {
            let centroid = self.codebook.centroid(s, code as usize);
            out[s * dsub..(s + 1) * dsub].copy_from_slice(centroid);
        }
        Ok(())
    }

    /// Encode a fresh vector into a PQ code (mainly for testing / introspection).
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] if `v.len() != dim`.
    pub fn encode(&self, v: &[f32]) -> AnnResult<Vec<u8>> {
        if v.len() != self.cfg.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: v.len(),
            });
        }
        Ok(encode_vector(v, &self.codebook))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(dim: usize, pq_m: usize, pq_ksub: usize) -> HnswPqConfig {
        HnswPqConfig {
            dim,
            m: 8,
            ef_construction: 32,
            ef: 32,
            pq_m,
            pq_ksub,
        }
    }

    /// 6 well-separated 2-D clusters in 4-D (pad each cluster centre with zeros).
    fn well_separated_4d() -> (Vec<f32>, usize, usize) {
        let centers = [
            [0.0_f32, 0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0],
            [-50.0, 0.0, 0.0, 0.0],
            [0.0, -50.0, 0.0, 0.0],
            [25.0, 25.0, 0.0, 0.0],
        ];
        let mut data = Vec::new();
        for c in centers.iter() {
            data.extend_from_slice(c);
        }
        (data, centers.len(), 4)
    }

    #[test]
    fn build_sets_len_and_not_empty() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(1);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        assert_eq!(idx.len(), n);
        assert!(!idx.is_empty());
    }

    #[test]
    fn adc_table_dimensions_and_nonneg() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(2);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let query = vec![10.0_f32, 5.0, 0.0, 0.0];
        let table = idx
            .adc_table(&query)
            .expect("adc_table should succeed with valid query");
        assert_eq!(table.len(), 2 * 4);
        for &v in &table {
            assert!(v >= 0.0, "ADC table entry negative: {v}");
            assert!(v.is_finite(), "ADC table entry non-finite: {v}");
        }
    }

    #[test]
    fn adc_distance_matches_manual_sum_on_tiny() {
        // n=2 in 2D, pq_m=2, pq_ksub=2 -> code length 2.
        let cfg = HnswPqConfig {
            dim: 2,
            m: 2,
            ef_construction: 4,
            ef: 4,
            pq_m: 2,
            pq_ksub: 2,
        };
        let mut idx = HnswPq::new(cfg).expect("HnswPq::new should succeed with valid config");
        let data = vec![0.0_f32, 0.0, 10.0, 10.0];
        let mut rng = LcgRng::new(3);
        idx.build(&data, 2, &mut rng)
            .expect("build should succeed with valid data");
        // Query equal to point 0.
        let query = vec![0.0_f32, 0.0];
        let table = idx
            .adc_table(&query)
            .expect("adc_table should succeed with valid query");
        // For each node, the ADC distance computed via the public method must
        // equal a manual sum of `table[s * ksub + code[s]]`.
        let pq_m = 2;
        let ksub = 2;
        for node in 0..2u32 {
            let api_d = idx
                .adc_distance(&table, node)
                .expect("adc_distance should succeed for valid node");
            let codes = idx.codes();
            let row = &codes[node as usize * pq_m..(node as usize + 1) * pq_m];
            let mut manual = 0.0_f32;
            for (s, &c) in row.iter().enumerate() {
                manual += table[s * ksub + c as usize];
            }
            assert!(
                (api_d - manual).abs() < 1e-6,
                "node {node}: api={api_d} manual={manual}"
            );
        }
    }

    #[test]
    fn search_returns_sorted_ascending() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(4);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let query = vec![0.0_f32, 0.0, 0.0, 0.0];
        let res = idx.search(&query, 3).expect("search should return results");
        assert!(res.len() <= 3);
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "not ascending: {res:?}");
        }
    }

    #[test]
    fn search_query_equal_to_point_recovers_it() {
        // Well-separated clusters with pq_ksub >= 6 so the codebook covers each.
        let (data, n, dim) = well_separated_4d();
        let cfg = HnswPqConfig {
            dim,
            m: 8,
            ef_construction: 32,
            ef: 32,
            pq_m: 2,
            pq_ksub: 6,
        };
        let mut idx = HnswPq::new(cfg).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(5);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        for target in 0..n {
            let q = &data[target * dim..(target + 1) * dim];
            let res = idx.search(q, 1).expect("search should return results");
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].0 as usize, target, "target={target} res={res:?}");
            assert!(res[0].1 < 1e-3, "target={target} dist={}", res[0].1);
        }
    }

    #[test]
    fn search_k_greater_than_n_returns_n() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(6);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let query = vec![0.0_f32, 0.0, 0.0, 0.0];
        let res = idx
            .search(&query, 100)
            .expect("search should return results");
        assert_eq!(res.len(), n);
    }

    #[test]
    fn deterministic_build_and_search() {
        let (data, n, dim) = well_separated_4d();
        let mut a = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut b = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut ra = LcgRng::new(7);
        let mut rb = LcgRng::new(7);
        a.build(&data, n, &mut ra)
            .expect("build should succeed with valid data");
        b.build(&data, n, &mut rb)
            .expect("build should succeed with valid data");
        let query = vec![10.0_f32, 0.0, 0.0, 0.0];
        let resa = a.search(&query, 3).expect("search should return results");
        let resb = b.search(&query, 3).expect("search should return results");
        assert_eq!(resa, resb);
    }

    #[test]
    fn err_dim_not_multiple_of_pq_m() {
        let cfg = HnswPqConfig {
            dim: 5,
            m: 2,
            ef_construction: 2,
            ef: 2,
            pq_m: 2,
            pq_ksub: 4,
        };
        let r = HnswPq::new(cfg);
        assert!(matches!(r, Err(AnnError::InvalidNumSubspaces { .. })));
    }

    #[test]
    fn err_dim_zero() {
        let cfg = HnswPqConfig {
            dim: 0,
            m: 2,
            ef_construction: 2,
            ef: 2,
            pq_m: 1,
            pq_ksub: 4,
        };
        assert!(matches!(
            HnswPq::new(cfg),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn err_ksub_below_two_and_above_256() {
        let cfg_lo = HnswPqConfig {
            dim: 4,
            m: 2,
            ef_construction: 2,
            ef: 2,
            pq_m: 2,
            pq_ksub: 1,
        };
        assert!(matches!(
            HnswPq::new(cfg_lo),
            Err(AnnError::InvalidK { .. })
        ));
        let cfg_hi = HnswPqConfig {
            dim: 4,
            m: 2,
            ef_construction: 2,
            ef: 2,
            pq_m: 2,
            pq_ksub: 300,
        };
        assert!(matches!(
            HnswPq::new(cfg_hi),
            Err(AnnError::InvalidK { .. })
        ));
    }

    #[test]
    fn err_build_n_zero() {
        let mut idx =
            HnswPq::new(small_cfg(4, 2, 4)).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(8);
        let r = idx.build(&[], 0, &mut rng);
        assert!(matches!(r, Err(AnnError::EmptyInput)));
    }

    #[test]
    fn err_build_data_length_mismatch() {
        let mut idx =
            HnswPq::new(small_cfg(4, 2, 4)).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(9);
        let r = idx.build(&[0.0_f32, 1.0, 2.0], 2, &mut rng);
        assert!(matches!(r, Err(AnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_query_wrong_length() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(10);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let bad = vec![0.0_f32; 3];
        assert!(matches!(
            idx.search(&bad, 1),
            Err(AnnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            idx.adc_table(&bad),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_k_zero() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(11);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let q = vec![0.0_f32; dim];
        assert!(matches!(
            idx.search(&q, 0),
            Err(AnnError::InvalidK { k: 0, .. })
        ));
    }

    #[test]
    fn codebook_dimensions_consistent() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(12);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let cb = idx.codebook();
        assert_eq!(cb.m, 2);
        assert_eq!(cb.ksub, 4);
        assert_eq!(cb.dsub, dim / 2);
        assert_eq!(cb.centroids_raw().len(), 2 * 4 * (dim / 2));
    }

    #[test]
    fn pq_codes_are_in_range() {
        let (data, n, dim) = well_separated_4d();
        let cfg = HnswPqConfig {
            dim,
            m: 4,
            ef_construction: 16,
            ef: 16,
            pq_m: 2,
            pq_ksub: 5,
        };
        let mut idx = HnswPq::new(cfg).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(13);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let codes = idx.codes();
        assert_eq!(codes.len(), n * 2);
        assert!(codes.iter().all(|&c| (c as usize) < 5));
    }

    #[test]
    fn adc_table_positive_for_nontrivial_query() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(14);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        // A query far from every cluster centre should have a sum of minimum
        // table-row values that is strictly positive.
        let query = vec![1000.0_f32, 1000.0, 1000.0, 1000.0];
        let table = idx
            .adc_table(&query)
            .expect("adc_table should succeed with valid query");
        let pq_m = 2usize;
        let ksub = 4usize;
        let mut min_sum = 0.0_f32;
        for s in 0..pq_m {
            let row = &table[s * ksub..(s + 1) * ksub];
            let min = row
                .iter()
                .copied()
                .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
            min_sum += min;
        }
        assert!(min_sum > 0.0, "min_sum={min_sum}");
    }

    #[test]
    fn empty_index_search_errors() {
        let idx =
            HnswPq::new(small_cfg(4, 2, 4)).expect("HnswPq::new should succeed with valid config");
        let q = vec![0.0_f32; 4];
        let r = idx.search(&q, 1);
        assert!(matches!(r, Err(AnnError::IndexEmpty)));
    }

    #[test]
    fn adc_distance_id_out_of_range_errors() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(15);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let q = vec![0.0_f32; dim];
        let table = idx
            .adc_table(&q)
            .expect("adc_table should succeed with valid query");
        let r = idx.adc_distance(&table, n as u32);
        assert!(matches!(r, Err(AnnError::IdOutOfRange { .. })));
    }

    #[test]
    fn adc_distance_bad_table_length_errors() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(16);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let bogus = vec![0.0_f32; 3]; // not pq_m * ksub == 8
        let r = idx.adc_distance(&bogus, 0);
        assert!(matches!(r, Err(AnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn adc_distance_equals_graph_l2_on_reconstruction() {
        // Build the index, then verify our adc_distance() matches a manual
        // L2² between query and reconstructed point (since they are equal).
        let (data, n, dim) = well_separated_4d();
        let cfg = HnswPqConfig {
            dim,
            m: 4,
            ef_construction: 16,
            ef: 16,
            pq_m: 2,
            pq_ksub: 6,
        };
        let mut idx = HnswPq::new(cfg).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(17);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");

        let query = vec![3.0_f32, -2.0, 0.5, 1.5];
        let table = idx
            .adc_table(&query)
            .expect("adc_table should succeed with valid query");

        let mut recon = vec![0.0_f32; dim];
        for i in 0..n {
            idx.reconstruct_into(i, &mut recon)
                .expect("reconstruct_into should succeed for valid index");
            let manual: f32 = query
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let api = idx
                .adc_distance(&table, i as u32)
                .expect("adc_distance should succeed for valid node");
            assert!(
                (api - manual).abs() < 1e-4,
                "i={i} api={api} manual={manual}"
            );
        }
    }

    #[test]
    fn rebuild_overwrites_previous_state() {
        let (data, n, dim) = well_separated_4d();
        // Use pq_ksub=2 so the second (smaller) build (n=3, ksub=2) is valid.
        let cfg = HnswPqConfig {
            dim,
            m: 4,
            ef_construction: 16,
            ef: 16,
            pq_m: 2,
            pq_ksub: 2,
        };
        let mut idx = HnswPq::new(cfg).expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(18);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        assert_eq!(idx.len(), n);
        // Smaller subsequent build replaces state.
        let small = &data[..(3 * dim)];
        let mut rng2 = LcgRng::new(19);
        idx.build(small, 3, &mut rng2)
            .expect("rebuild should succeed with valid data");
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.codes().len(), 3 * 2);
    }

    #[test]
    fn encode_roundtrip_yields_valid_codes() {
        let (data, n, dim) = well_separated_4d();
        let mut idx = HnswPq::new(small_cfg(dim, 2, 4))
            .expect("HnswPq::new should succeed with valid config");
        let mut rng = LcgRng::new(20);
        idx.build(&data, n, &mut rng)
            .expect("build should succeed with valid data");
        let code = idx
            .encode(&data[0..dim])
            .expect("encode should succeed for valid vector");
        assert_eq!(code.len(), 2);
        assert!(code.iter().all(|&c| (c as usize) < 4));
    }
}
