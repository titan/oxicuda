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
