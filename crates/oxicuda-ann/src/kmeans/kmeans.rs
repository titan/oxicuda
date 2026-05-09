use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;

/// k-means model.
pub struct KMeans {
    centroids: Vec<f32>,
    pub k: usize,
    pub dim: usize,
}

impl KMeans {
    /// Access flat centroid storage, `[k, dim]` row-major.
    #[must_use]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// k-means++ initialisation: seed k centroids from data with D² weighting.
    pub fn kmeans_pp_init(
        data: &[f32],
        n: usize,
        dim: usize,
        k: usize,
        rng: &mut LcgRng,
    ) -> Vec<f32> {
        let mut centers: Vec<f32> = Vec::with_capacity(k * dim);

        // First center: uniformly random
        let first = rng.next_u32() as usize % n;
        centers.extend_from_slice(&data[first * dim..(first + 1) * dim]);

        let mut min_dists = vec![f32::INFINITY; n];

        for _ in 1..k {
            let last_center = &centers[centers.len() - dim..];
            let mut total = 0.0_f64;
            for (i, row) in data.chunks_exact(dim).enumerate() {
                let d: f32 = row
                    .iter()
                    .zip(last_center.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
                total += min_dists[i] as f64;
            }

            // Weighted random pick
            let threshold = rng.next_f32() as f64 * total;
            let mut cumsum = 0.0_f64;
            let mut chosen = n - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                cumsum += d as f64;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centers.extend_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
        }

        centers
    }

    /// Fit k-means on `n` rows of `dim`-dim data for `n_epochs` iterations.
    pub fn fit(
        data: &[f32],
        n: usize,
        dim: usize,
        k: usize,
        n_epochs: usize,
        rng: &mut LcgRng,
    ) -> AnnResult<KMeans> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if k == 0 || k > n {
            return Err(AnnError::InvalidK { k, n });
        }

        let mut centroids = Self::kmeans_pp_init(data, n, dim, k, rng);
        let mut assignments = vec![0usize; n];
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![0.0_f32; k * dim];

        for epoch in 0..n_epochs {
            // Assignment step
            let mut changed = 0usize;
            for (i, row) in data.chunks_exact(dim).enumerate() {
                let best = Self::nearest_centroid(row, &centroids, k, dim);
                if best != assignments[i] {
                    changed += 1;
                    assignments[i] = best;
                }
            }

            if epoch > 0 && changed == 0 {
                return Ok(KMeans { centroids, k, dim });
            }

            // Update step
            for v in new_centroids.iter_mut() {
                *v = 0.0;
            }
            for v in counts.iter_mut() {
                *v = 0;
            }
            for (i, row) in data.chunks_exact(dim).enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (d, &x) in row.iter().enumerate() {
                    new_centroids[c * dim + d] += x;
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..dim {
                        centroids[c * dim + d] = new_centroids[c * dim + d] * inv;
                    }
                }
                // If empty cluster: re-seed from a random point
                else {
                    let rand_i = rng.next_u32() as usize % n;
                    centroids[c * dim..(c + 1) * dim]
                        .copy_from_slice(&data[rand_i * dim..(rand_i + 1) * dim]);
                }
            }
        }

        Ok(KMeans { centroids, k, dim })
    }

    fn nearest_centroid(row: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
        let mut best_c = 0;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let center = &centroids[c * dim..(c + 1) * dim];
            let d: f32 = row
                .iter()
                .zip(center.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best_c = c;
            }
        }
        best_c
    }

    /// Assign each of `n` rows in `data` to the nearest centroid.
    pub fn predict(&self, data: &[f32], n: usize, dim: usize) -> AnnResult<Vec<usize>> {
        if dim != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        Ok(data
            .chunks_exact(dim)
            .map(|row| Self::nearest_centroid(row, &self.centroids, self.k, self.dim))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_1d() {
        let mut rng = LcgRng::new(42);
        let mut data = Vec::with_capacity(100);
        for _ in 0..50 {
            data.push(rng.next_f32() * 0.1);
        }
        for _ in 0..50 {
            data.push(10.0 + rng.next_f32() * 0.1);
        }
        let km = KMeans::fit(&data, 100, 1, 2, 100, &mut rng).unwrap();
        let c0 = km.centroids()[0];
        let c1 = km.centroids()[1];
        let low = c0.min(c1);
        let high = c0.max(c1);
        assert!(low < 1.0, "low centroid={low}");
        assert!(high > 9.0, "high centroid={high}");
    }
}
