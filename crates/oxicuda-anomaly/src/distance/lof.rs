//! Local Outlier Factor (Breunig et al. 2000).
//!
//! `LOF_k(x) = (1/k) · Σ_{o ∈ N_k(x)} [lrd_k(o) / lrd_k(x)]`
//!
//! where
//!
//! ```text
//! lrd_k(x)          = k / Σ_{o ∈ N_k(x)} reach_dist_k(x, o)
//! reach_dist_k(x,o) = max(kNN_dist_k(o), dist(x, o))
//! ```
//!
//! LOF ≈ 1.0 → normal; LOF >> 1.0 → strong outlier.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Euclidean distance ───────────────────────────────────────────────────────

/// Euclidean distance between two equal-length slices.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> AnomalyResult<f32> {
    if a.len() != b.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sq.sqrt())
}

// ─── k-NN brute force ────────────────────────────────────────────────────────

/// Find the `k` nearest neighbours of `query` in `data` (brute-force).
///
/// Returns `(indices [k], distances [k])` sorted by distance ascending.
pub fn knn_query(
    query: &[f32],
    data: &[f32],
    n_samples: usize,
    k: usize,
) -> AnomalyResult<(Vec<usize>, Vec<f32>)> {
    if k == 0 {
        return Err(AnomalyError::InvalidK { k: 0 });
    }
    if n_samples < k {
        return Err(AnomalyError::InsufficientSamples {
            need: k,
            got: n_samples,
        });
    }
    let n_feat = query.len();
    if data.len() != n_samples * n_feat {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * n_feat,
            got: data.len(),
        });
    }

    // Compute all distances
    let mut dists: Vec<(usize, f32)> = (0..n_samples)
        .map(|i| {
            let row = &data[i * n_feat..(i + 1) * n_feat];
            let d = row
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (i, d)
        })
        .collect();

    // Partial sort: find k smallest
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let indices: Vec<usize> = dists[..k].iter().map(|(i, _)| *i).collect();
    let distances: Vec<f32> = dists[..k].iter().map(|(_, d)| *d).collect();
    Ok((indices, distances))
}

// ─── Lof ─────────────────────────────────────────────────────────────────────

/// Local Outlier Factor detector.
pub struct Lof {
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    k: usize,
    /// k-NN distances for each training sample: `[n * k]`.
    knn_dists: Vec<f32>,
    /// k-NN indices for each training sample: `[n * k]`.
    knn_indices: Vec<usize>,
    /// Local reachability density for each training sample: `[n]`.
    lrd: Vec<f32>,
}

impl Lof {
    /// Create an unfitted LOF with neighbour count `k`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            k,
            knn_dists: Vec::new(),
            knn_indices: Vec::new(),
            lrd: Vec::new(),
        }
    }

    /// Fit: precompute k-NN, reach_dist, and lrd for all training points.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if self.k == 0 {
            return Err(AnomalyError::InvalidK { k: 0 });
        }
        if n_samples <= self.k {
            return Err(AnomalyError::InsufficientSamples {
                need: self.k + 1,
                got: n_samples,
            });
        }

        self.data = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;

        // Compute k-NN for every training point (exclude self)
        self.knn_dists = vec![0.0_f32; n_samples * self.k];
        self.knn_indices = vec![0_usize; n_samples * self.k];

        for i in 0..n_samples {
            let query = &data[i * n_features..(i + 1) * n_features];
            // Compute distances to all others
            let mut dists: Vec<(usize, f32)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let row = &data[j * n_features..(j + 1) * n_features];
                    let d = row
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (ki, &(idx, dist)) in dists.iter().take(self.k).enumerate() {
                self.knn_indices[i * self.k + ki] = idx;
                self.knn_dists[i * self.k + ki] = dist;
            }
        }

        // k-distance of training point i = knn_dists[i * k + (k-1)]
        // lrd_k(i) = k / Σ_{o ∈ N_k(i)} reach_dist_k(i, o)
        self.lrd = vec![0.0_f32; n_samples];
        for i in 0..n_samples {
            let query = &data[i * n_features..(i + 1) * n_features];
            let mut sum_reach = 0.0_f32;
            for ki in 0..self.k {
                let j = self.knn_indices[i * self.k + ki];
                let d_ij = self.knn_dists[i * self.k + ki];
                // k-distance of j = knn_dists[j * k + (k-1)]
                let kd_j = self.knn_dists[j * self.k + self.k - 1];
                // reach_dist(i, j) = max(k-dist(j), dist(i, j))
                let row_j = &data[j * n_features..(j + 1) * n_features];
                let dist_ij = row_j
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let _ = d_ij; // already computed above, use fresh dist_ij
                let reach = kd_j.max(dist_ij).max(0.0);
                sum_reach += reach;
            }
            self.lrd[i] = if sum_reach < 1e-12 {
                f32::INFINITY
            } else {
                self.k as f32 / sum_reach
            };
        }

        Ok(())
    }

    /// Compute LOF score for a new point `x`.
    ///
    /// LOF > 1 suggests an outlier; LOF ≈ 1 suggests normality.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_samples == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }

        let (nn_idx, nn_dist) = knn_query(x, &self.data, self.n_samples, self.k)?;

        // lrd of query x
        let mut sum_reach_x = 0.0_f32;
        for ki in 0..self.k {
            let j = nn_idx[ki];
            let kd_j = self.knn_dists[j * self.k + self.k - 1];
            let reach = kd_j.max(nn_dist[ki]).max(0.0);
            sum_reach_x += reach;
        }
        let lrd_x = if sum_reach_x < 1e-12 {
            f32::INFINITY
        } else {
            self.k as f32 / sum_reach_x
        };

        // If x is entirely inside cluster, LOF = 0
        if lrd_x.is_infinite() {
            return Ok(0.0);
        }

        // LOF = (1/k) Σ_{o} lrd(o) / lrd_x
        let lof: f32 = nn_idx.iter().map(|&j| self.lrd[j] / lrd_x).sum::<f32>() / self.k as f32;

        Ok(lof)
    }

    /// Batch LOF scoring; `x` is `[n * n_features]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lof_fit_score_basic() {
        let data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let mut lof = Lof::new(3);
        lof.fit(&data, 20, 1).unwrap();
        let s = lof.score(&[0.5_f32]).unwrap();
        assert!(s.is_finite(), "lof={s}");
    }

    #[test]
    fn lof_unfitted_error() {
        let lof = Lof::new(3);
        assert!(lof.score(&[0.0_f32]).is_err());
    }

    #[test]
    fn euclidean_distance_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let d = euclidean_distance(&v, &v).unwrap();
        assert!(d.abs() < 1e-6, "d={d}");
    }
}
