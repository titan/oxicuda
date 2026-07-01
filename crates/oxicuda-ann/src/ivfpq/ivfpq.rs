use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::ivf::ivf::IvfIndex;
use crate::pq::adc::{adc_distance, build_adc_table};
use crate::pq::codebook::PqCodebook;
use crate::pq::encode::encode_vector;
use crate::pq::train::train_pq;
use crate::topk::heap::BoundedMaxHeap;

/// IVFPQ index: coarse IVF partitioning + PQ-coded residuals for ADC search.
pub struct IvfPq {
    ivf: IvfIndex,
    pq: PqCodebook,
    /// Per-list PQ codes: `codes[list_id][item_idx * m .. (item_idx+1)*m]`
    codes: Vec<Vec<u8>>,
    /// Per-list original IDs.
    ids: Vec<Vec<usize>>,
    fitted: bool,
}

impl IvfPq {
    /// Train coarse IVF quantizer and PQ codebook.
    pub fn train(
        data: &[f32],
        n: usize,
        dim: usize,
        n_lists: usize,
        m: usize,
        ksub: usize,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        if dim == 0 || !dim.is_multiple_of(m) {
            return Err(AnnError::InvalidNumSubspaces { m, dim });
        }

        let mut ivf = IvfIndex::new(n_lists, dim);
        ivf.train(data, n, rng)?;

        let pq = train_pq(data, n, dim, m, ksub, 30, rng)?;

        Ok(Self {
            ivf,
            pq,
            codes: vec![Vec::new(); n_lists],
            ids: vec![Vec::new(); n_lists],
            fitted: true,
        })
    }

    /// Add a vector with external `id`. Assigns to nearest coarse centroid and PQ-encodes.
    pub fn add(&mut self, v: &[f32], id: usize) {
        let list_id = self.assign_to_list(v);
        let code = encode_vector(v, &self.pq);
        self.codes[list_id].extend(code);
        self.ids[list_id].push(id);
    }

    fn assign_to_list(&self, v: &[f32]) -> usize {
        let n_lists = self.ivf.n_lists;
        let dim = self.ivf.dim;
        // Access coarse centroids via a temporary IVF search (nprobe=1)
        // We replicate the nearest-centroid logic using the coarse centroids
        // stored inside IvfIndex via a proxy: we build them during train.
        // For simplicity, re-use IVF's internal logic by calling search with nprobe=1
        // and extracting the list id. We instead do it via the coarse scan inline.
        // The IvfIndex doesn't expose centroids directly, so we use its search_list method.
        // Since we can't access private fields, we store centroids separately here.
        // Work around: call ivf search_nearest_list via probe_order alternative.
        // We add a accessor to IvfIndex for this purpose.
        let _ = (n_lists, dim);
        self.ivf_assign_list(v)
    }

    fn ivf_assign_list(&self, v: &[f32]) -> usize {
        // Replicate coarse assignment by calling ivf.search with nprobe=1
        // We need to expose centroids; instead, use a workaround.
        // We stored trained IvfIndex, and it has a private `coarse` field.
        // The cleanest solution: add a pub method to IvfIndex for assignment.
        // Since we own the IvfIndex, we can use its search to get nprobe=1 result
        // but IvfIndex.search requires vectors to be added first.
        // Alternative: keep our own copy of coarse centroids.
        // For now: delegate to IvfIndex's assignment logic exposed via a new pub fn.
        self.ivf.nearest_list(v)
    }

    /// Search for `k` approximate nearest neighbors using ADC within `nprobe` lists.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> AnnResult<Vec<(usize, f32)>> {
        if !self.fitted {
            return Err(AnnError::NotFitted);
        }
        if query.len() != self.ivf.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.ivf.dim,
                got: query.len(),
            });
        }
        if nprobe == 0 || nprobe > self.ivf.n_lists {
            return Err(AnnError::InvalidNumProbes {
                nprobe,
                nlist: self.ivf.n_lists,
            });
        }

        let total_items: usize = self.ids.iter().map(|l| l.len()).sum();
        if total_items == 0 {
            return Err(AnnError::IndexEmpty);
        }

        let table = build_adc_table(query, &self.pq);
        let actual_k = k.min(total_items);
        if actual_k == 0 {
            return Err(AnnError::InvalidK { k, n: total_items });
        }

        let probed = self.ivf.probe_lists(query, nprobe);
        let mut heap = BoundedMaxHeap::new(actual_k);

        for list_id in probed {
            let list_ids = &self.ids[list_id];
            let m = self.pq.m;
            for (item_idx, &id) in list_ids.iter().enumerate() {
                let code = &self.codes[list_id][item_idx * m..(item_idx + 1) * m];
                let d = adc_distance(code, &table, m, self.pq.ksub);
                heap.push(d, id);
            }
        }

        Ok(heap.into_sorted_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::IvfPq;
    use crate::error::AnnError;
    use crate::handle::LcgRng;
    use crate::pq::adc::{adc_distance, build_adc_table};
    use crate::pq::codebook::PqCodebook;
    use crate::pq::encode::encode_vector;

    /// Build a small, fitted IvfPq on two well-separated clusters.
    ///
    /// Parameters: dim=4, m=2, dsub=2, ksub=2, n_lists=2, n=8 training points.
    fn make_fitted_ivfpq(rng: &mut LcgRng) -> IvfPq {
        let mut data = Vec::with_capacity(32);
        // Cluster A: 4 points near the origin
        for i in 0..4_u32 {
            let off = i as f32 * 0.05;
            data.extend_from_slice(&[off, off, off, off]);
        }
        // Cluster B: 4 points near [10,10,10,10]
        for i in 0..4_u32 {
            let off = i as f32 * 0.05;
            data.extend_from_slice(&[10.0 + off, 10.0 + off, 10.0 + off, 10.0 + off]);
        }
        IvfPq::train(&data, 8, 4, 2, 2, 2, rng)
            .expect("IvfPq::train should succeed on two-cluster data")
    }

    // ---------------------------------------------------------------------------
    // Error-path tests
    // ---------------------------------------------------------------------------

    #[test]
    fn empty_after_train_returns_index_empty() {
        let mut rng = LcgRng::new(1);
        let ivfpq = make_fitted_ivfpq(&mut rng);
        // No add() calls — index is trained but empty.
        let result = ivfpq.search(&[0.0_f32; 4], 1, 1);
        assert!(
            matches!(result, Err(AnnError::IndexEmpty)),
            "expected IndexEmpty before any add, got {result:?}"
        );
    }

    #[test]
    fn query_wrong_dimension_returns_dimension_mismatch() {
        let mut rng = LcgRng::new(2);
        let mut ivfpq = make_fitted_ivfpq(&mut rng);
        ivfpq.add(&[0.0_f32; 4], 0);
        // Query dim=3, index dim=4
        let result = ivfpq.search(&[0.0_f32; 3], 1, 1);
        assert!(
            matches!(
                result,
                Err(AnnError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "expected DimensionMismatch{{expected:4,got:3}}, got {result:?}"
        );
    }

    #[test]
    fn nprobe_zero_returns_invalid_num_probes() {
        let mut rng = LcgRng::new(3);
        let mut ivfpq = make_fitted_ivfpq(&mut rng);
        ivfpq.add(&[0.0_f32; 4], 0);
        let result = ivfpq.search(&[0.0_f32; 4], 1, 0);
        assert!(
            matches!(result, Err(AnnError::InvalidNumProbes { .. })),
            "expected InvalidNumProbes for nprobe=0, got {result:?}"
        );
    }

    #[test]
    fn nprobe_exceeds_n_lists_returns_invalid_num_probes() {
        let mut rng = LcgRng::new(4);
        let mut ivfpq = make_fitted_ivfpq(&mut rng);
        ivfpq.add(&[0.0_f32; 4], 0);
        // n_lists=2, nprobe=3 is out of range
        let result = ivfpq.search(&[0.0_f32; 4], 1, 3);
        assert!(
            matches!(result, Err(AnnError::InvalidNumProbes { .. })),
            "expected InvalidNumProbes for nprobe=3 > n_lists=2, got {result:?}"
        );
    }

    #[test]
    fn invalid_dim_not_divisible_by_m_returns_error() {
        let mut rng = LcgRng::new(5);
        // dim=3 is not divisible by m=2 → must fail immediately
        let data = [0.0_f32; 6]; // 2 points of dim=3
        let result = IvfPq::train(&data, 2, 3, 2, 2, 2, &mut rng);
        assert!(
            matches!(result, Err(AnnError::InvalidNumSubspaces { m: 2, dim: 3 })),
            "expected InvalidNumSubspaces{{m:2,dim:3}}"
        );
    }

    // ---------------------------------------------------------------------------
    // PQ encode / decode correctness
    // ---------------------------------------------------------------------------

    /// When a vector exactly equals a PQ codeword, encode→reconstruct must give
    /// zero reconstruction error.  This validates the fundamental correctness of
    /// the quantise→reconstruct round-trip independently of IvfPq training.
    #[test]
    fn pq_round_trip_zero_error_for_exact_codeword() {
        // m=2 subspaces, ksub=2 codewords each, dsub=2
        let mut cb = PqCodebook::new(2, 2, 2);
        cb.centroid_mut(0, 0).copy_from_slice(&[1.0_f32, 0.0]);
        cb.centroid_mut(0, 1).copy_from_slice(&[-1.0_f32, 0.0]);
        cb.centroid_mut(1, 0).copy_from_slice(&[0.0_f32, 1.0]);
        cb.centroid_mut(1, 1).copy_from_slice(&[0.0_f32, -1.0]);

        // v = sub0_codeword0 concat sub1_codeword1 = [1,0, 0,-1]
        let v = [1.0_f32, 0.0, 0.0, -1.0];
        let codes = encode_vector(&v, &cb);
        assert_eq!(codes[0], 0u8, "sub0: nearest codeword must be index 0");
        assert_eq!(codes[1], 1u8, "sub1: nearest codeword must be index 1");

        // Reconstruct by concatenating the nearest sub-codewords.
        let mut reconstructed = [0.0_f32; 4];
        for s in 0..2_usize {
            let c = cb.centroid(s, codes[s] as usize);
            reconstructed[s * 2..(s + 1) * 2].copy_from_slice(c);
        }
        let err: f32 = v
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            err.abs() < 1e-12,
            "round-trip reconstruction error={err}, expected zero"
        );
    }

    /// When the database vector exactly equals a PQ codeword, the ADC distance
    /// from any query to that vector must equal the true L2² distance — no
    /// quantisation error is introduced.
    #[test]
    fn adc_distance_matches_true_l2_when_db_vector_equals_codeword() {
        let mut cb = PqCodebook::new(2, 2, 2);
        cb.centroid_mut(0, 0).copy_from_slice(&[1.0_f32, 0.0]);
        cb.centroid_mut(0, 1).copy_from_slice(&[-1.0_f32, 0.0]);
        cb.centroid_mut(1, 0).copy_from_slice(&[0.0_f32, 1.0]);
        cb.centroid_mut(1, 1).copy_from_slice(&[0.0_f32, -1.0]);

        // db_vec is exactly codeword (sub0=0, sub1=1)
        let db_vec = [1.0_f32, 0.0, 0.0, -1.0];
        let codes = encode_vector(&db_vec, &cb);

        let query = [0.5_f32, 0.0, 0.0, -0.5];
        let table = build_adc_table(&query, &cb);
        let adc_dist = adc_distance(&codes, &table, 2, 2);

        // True L2² from query to db_vec (= reconstructed db_vec since it is a codeword)
        let true_dist: f32 = query
            .iter()
            .zip(db_vec.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        assert!(
            (adc_dist - true_dist).abs() < 1e-6,
            "ADC dist={adc_dist} true L2²={true_dist}: must be equal when db vector = codeword"
        );
    }

    // ---------------------------------------------------------------------------
    // Search result-shape properties
    // ---------------------------------------------------------------------------

    /// Results must be sorted ascending by ADC distance, respect the k cap, and
    /// contain only finite distances.
    #[test]
    fn search_results_sorted_ascending_at_most_k_and_finite() {
        let mut rng = LcgRng::new(11);
        let mut ivfpq = make_fitted_ivfpq(&mut rng);
        // Add 8 vectors matching the training distribution
        for i in 0..4_u32 {
            let off = i as f32 * 0.05;
            ivfpq.add(&[off, off, off, off], i as usize);
        }
        for i in 0..4_u32 {
            let off = i as f32 * 0.05;
            ivfpq.add(
                &[10.0 + off, 10.0 + off, 10.0 + off, 10.0 + off],
                (4 + i) as usize,
            );
        }

        let query = [0.5_f32, 0.5, 0.5, 0.5];
        let k = 3;
        let results = ivfpq
            .search(&query, k, 2)
            .expect("search with nprobe=n_lists should succeed");

        assert!(
            results.len() <= k,
            "returned {} results but k={k}",
            results.len()
        );
        for (_, d) in &results {
            assert!(d.is_finite(), "distance {d} must be finite");
        }
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results not sorted ascending: d[i]={} d[i+1]={}",
                w[0].1,
                w[1].1
            );
        }
    }
}
