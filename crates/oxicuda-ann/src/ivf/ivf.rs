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
    /// Stored vectors (row-major `[total, dim]`).
    vectors: Vec<f32>,
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
            vectors: Vec::new(),
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
        self.vectors.extend_from_slice(v);
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
        if self.vectors.is_empty() {
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

        let total = self.vectors.len() / self.dim;
        let actual_k = k.min(total);
        if actual_k == 0 {
            return Err(AnnError::InvalidK { k, n: total });
        }

        let mut heap = BoundedMaxHeap::new(actual_k);
        let probed = self.probe_order(query, nprobe);

        // Build reverse mapping: id -> row in self.vectors
        // (Simplified: we just store by insertion order)
        let mut global_pos = 0usize;
        for list_id in 0..self.n_lists {
            for &id in &self.posting_lists[list_id] {
                let vec_row = &self.vectors[global_pos * self.dim..(global_pos + 1) * self.dim];
                if probed.contains(&list_id) {
                    let d: f32 = query
                        .iter()
                        .zip(vec_row.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    heap.push(d, id);
                }
                global_pos += 1;
            }
        }

        Ok(heap.into_sorted_vec())
    }
}
