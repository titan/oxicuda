//! Feature normalisation: quantile, z-score, and min-max.

use crate::error::{TabularError, TabularResult};

// ─── QuantileNormalizer ───────────────────────────────────────────────────────

/// Empirical quantile normaliser: maps each feature value to its rank in `[0, 1]`.
pub struct QuantileNormalizer {
    sorted_vals: Vec<Vec<f32>>,
    n_features: usize,
}

impl QuantileNormalizer {
    /// Fit on training data `data [n_samples * n_features]`.
    pub fn fit(data: &[f32], n_samples: usize, n_features: usize) -> TabularResult<Self> {
        if data.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if n_samples == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        let mut sorted_vals = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let mut col: Vec<f32> = (0..n_samples).map(|s| data[s * n_features + f]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_vals.push(col);
        }
        Ok(Self {
            sorted_vals,
            n_features,
        })
    }

    /// Transform a single sample `x [n_features]` → quantile ranks in `[0, 1]`.
    pub fn transform(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let out = x
            .iter()
            .zip(self.sorted_vals.iter())
            .map(|(&xi, col)| {
                let n = col.len();
                // Binary search for lower-bound rank
                let rank = col.partition_point(|&v| v <= xi);
                (rank.min(n)) as f32 / n as f32
            })
            .collect();
        Ok(out)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> TabularResult<(Self, Vec<f32>)> {
        let normaliser = Self::fit(data, n_samples, n_features)?;
        let mut out = Vec::with_capacity(data.len());
        for s in 0..n_samples {
            let row = &data[s * n_features..(s + 1) * n_features];
            out.extend_from_slice(&normaliser.transform(row)?);
        }
        Ok((normaliser, out))
    }
}

// ─── StandardNormalizer ───────────────────────────────────────────────────────

/// Z-score normaliser: `(x - mean) / (std + ε)`.
pub struct StandardNormalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
    n_features: usize,
}

impl StandardNormalizer {
    /// Fit on training data.
    pub fn fit(data: &[f32], n_samples: usize, n_features: usize) -> TabularResult<Self> {
        if data.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if n_samples == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        let n = n_samples as f32;
        let mut mean = vec![0.0_f32; n_features];
        for s in 0..n_samples {
            for f in 0..n_features {
                mean[f] += data[s * n_features + f];
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut var = vec![0.0_f32; n_features];
        for s in 0..n_samples {
            for f in 0..n_features {
                let d = data[s * n_features + f] - mean[f];
                var[f] += d * d;
            }
        }
        let std: Vec<f32> = var.iter().map(|&v| (v / n).sqrt()).collect();

        Ok(Self {
            mean,
            std,
            n_features,
        })
    }

    /// Transform a single sample.
    pub fn transform(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let out = x
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(&xi, (&m, &s))| (xi - m) / (s + 1e-7))
            .collect();
        Ok(out)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> TabularResult<(Self, Vec<f32>)> {
        let norm = Self::fit(data, n_samples, n_features)?;
        let mut out = Vec::with_capacity(data.len());
        for s in 0..n_samples {
            let row = &data[s * n_features..(s + 1) * n_features];
            out.extend_from_slice(&norm.transform(row)?);
        }
        Ok((norm, out))
    }
}

// ─── MinMaxNormalizer ─────────────────────────────────────────────────────────

/// Min-max normaliser: maps each feature to `[0, 1]`.
pub struct MinMaxNormalizer {
    min: Vec<f32>,
    max: Vec<f32>,
    n_features: usize,
}

impl MinMaxNormalizer {
    /// Fit on training data.
    pub fn fit(data: &[f32], n_samples: usize, n_features: usize) -> TabularResult<Self> {
        if data.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if n_samples == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        let mut min = vec![f32::INFINITY; n_features];
        let mut max = vec![f32::NEG_INFINITY; n_features];

        for s in 0..n_samples {
            for f in 0..n_features {
                let v = data[s * n_features + f];
                if v < min[f] {
                    min[f] = v;
                }
                if v > max[f] {
                    max[f] = v;
                }
            }
        }
        Ok(Self {
            min,
            max,
            n_features,
        })
    }

    /// Transform a single sample.
    pub fn transform(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let out = x
            .iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .map(|(&xi, (&lo, &hi))| {
                let range = hi - lo;
                if range < 1e-10 {
                    0.5
                } else {
                    (xi - lo) / range
                }
            })
            .collect();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_range_in_01() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
        let (norm, out) =
            QuantileNormalizer::fit_transform(&data, 4, 2).expect("fit_transform should succeed");
        assert!(out.iter().all(|&v| (0.0_f32..=1.0).contains(&v)));
        // Transform training values should also be in range
        let row = &data[0..2];
        let t = norm.transform(row).expect("transform should succeed");
        assert!(t.iter().all(|&v| (0.0_f32..=1.0).contains(&v)));
    }

    #[test]
    fn standard_normalizer_zero_mean() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (norm, _) =
            StandardNormalizer::fit_transform(&data, 3, 2).expect("fit_transform should succeed");
        let t = norm
            .transform(&[3.0_f32, 4.0])
            .expect("transform should succeed");
        assert!(t[0].abs() < 1.0); // roughly centred
        let _ = t;
    }

    #[test]
    fn minmax_range() {
        let data = vec![0.0_f32, 0.0, 5.0, 10.0, 10.0, 10.0];
        let norm = MinMaxNormalizer::fit(&data, 3, 2).expect("fit should succeed");
        let t = norm
            .transform(&[5.0_f32, 5.0])
            .expect("transform should succeed");
        assert!((t[0] - 0.5).abs() < 1e-5);
        assert!((t[1] - 0.5).abs() < 1e-5);
    }
}
