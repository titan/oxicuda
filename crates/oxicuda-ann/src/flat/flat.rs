use crate::error::{AnnError, AnnResult};
use crate::topk::heap::BoundedMaxHeap;

/// Brute-force flat index supporting L2 and inner-product search.
pub struct FlatIndex {
    vectors: Vec<f32>,
    dim: usize,
}

impl FlatIndex {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
        }
    }

    pub fn add(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.vectors.extend_from_slice(v);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len().checked_div(self.dim).unwrap_or(0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search by L2² distance. Returns `(id, dist_sq)` sorted ascending by distance.
    pub fn search_l2(&self, query: &[f32], k: usize) -> AnnResult<Vec<(usize, f32)>> {
        let n = self.len();
        if n == 0 {
            return Err(AnnError::IndexEmpty);
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || k > n {
            return Err(AnnError::InvalidK { k, n });
        }

        let mut heap = BoundedMaxHeap::new(k);
        for (i, chunk) in self.vectors.chunks_exact(self.dim).enumerate() {
            let d: f32 = query
                .iter()
                .zip(chunk.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            heap.push(d, i);
        }
        Ok(heap.into_sorted_vec())
    }

    /// Search by inner product (returns negated IP as distance so smaller = better).
    pub fn search_ip(&self, query: &[f32], k: usize) -> AnnResult<Vec<(usize, f32)>> {
        let n = self.len();
        if n == 0 {
            return Err(AnnError::IndexEmpty);
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || k > n {
            return Err(AnnError::InvalidK { k, n });
        }

        let mut heap = BoundedMaxHeap::new(k);
        for (i, chunk) in self.vectors.chunks_exact(self.dim).enumerate() {
            let ip: f32 = query.iter().zip(chunk.iter()).map(|(a, b)| a * b).sum();
            heap.push(-ip, i);
        }
        Ok(heap.into_sorted_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_distance_zero() {
        let mut idx = FlatIndex::new(3);
        idx.add(&[1.0, 2.0, 3.0]);
        idx.add(&[4.0, 5.0, 6.0]);
        let res = idx
            .search_l2(&[1.0, 2.0, 3.0], 1)
            .expect("flat L2 search should succeed");
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-6);
    }
}
