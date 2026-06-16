//! Feature embedder: continuous normalization + categorical validation.

use crate::error::{TabularError, TabularResult};

/// Holds fit statistics for continuous features and category sizes for categorical.
pub struct FeatureEmbedder {
    pub cont_mean: Vec<f32>,
    pub cont_std: Vec<f32>,
    pub cat_sizes: Vec<usize>,
    pub n_cont: usize,
}

impl FeatureEmbedder {
    /// Construct a new `FeatureEmbedder`.
    pub fn new(n_cont: usize, cat_sizes: Vec<usize>) -> Self {
        Self {
            cont_mean: vec![0.0_f32; n_cont],
            cont_std: vec![1.0_f32; n_cont],
            cat_sizes,
            n_cont,
        }
    }

    /// Fit continuous feature statistics from a `[n_samples * n_cont]` data matrix.
    pub fn fit_cont(&mut self, data: &[f32], n_samples: usize) -> TabularResult<()> {
        if data.len() != n_samples * self.n_cont {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * self.n_cont,
                got: data.len(),
            });
        }
        if n_samples == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }

        let n = n_samples as f32;
        let mut mean = vec![0.0_f32; self.n_cont];
        for s in 0..n_samples {
            for f in 0..self.n_cont {
                mean[f] += data[s * self.n_cont + f];
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut var = vec![0.0_f32; self.n_cont];
        for s in 0..n_samples {
            for f in 0..self.n_cont {
                let d = data[s * self.n_cont + f] - mean[f];
                var[f] += d * d;
            }
        }

        self.cont_mean = mean;
        self.cont_std = var.iter().map(|&v| (v / n).sqrt().max(1e-7)).collect();
        Ok(())
    }

    /// Z-score normalise a `[n_cont]` continuous feature vector.
    pub fn normalize_cont(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.n_cont {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_cont,
                got: x.len(),
            });
        }
        let out = x
            .iter()
            .zip(self.cont_mean.iter().zip(self.cont_std.iter()))
            .map(|(&xi, (&m, &s))| (xi - m) / s)
            .collect();
        Ok(out)
    }

    /// Validate that each categorical index is within range.
    pub fn validate_cat(&self, x_cat: &[usize]) -> TabularResult<()> {
        if x_cat.len() != self.cat_sizes.len() {
            return Err(TabularError::DimensionMismatch {
                expected: self.cat_sizes.len(),
                got: x_cat.len(),
            });
        }
        for (i, (&val, &n)) in x_cat.iter().zip(self.cat_sizes.iter()).enumerate() {
            if val >= n {
                return Err(TabularError::CategoricalOutOfRange { feat: i, val, n });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_normalize_cont() {
        let mut emb = FeatureEmbedder::new(2, vec![3, 4]);
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        emb.fit_cont(&data, 3).expect("fit_cont should succeed");
        let out = emb
            .normalize_cont(&[3.0_f32, 4.0])
            .expect("normalize_cont should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn embedder_validate_cat_ok() {
        let emb = FeatureEmbedder::new(1, vec![5, 3]);
        assert!(emb.validate_cat(&[4, 2]).is_ok());
    }

    #[test]
    fn embedder_validate_cat_out_of_range() {
        let emb = FeatureEmbedder::new(1, vec![5, 3]);
        assert!(emb.validate_cat(&[5, 0]).is_err());
    }
}
