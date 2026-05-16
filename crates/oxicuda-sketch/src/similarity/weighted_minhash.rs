//! Weighted MinHash (Ioffe 2010): consistent weighted sampling for weighted Jaccard.
//!
//! Given weighted feature vector `w_i >= 0`, compute K independent samples each picking
//! `(feature_index, scale_value)` with probability proportional to `w_i`.
//! Jaccard_weighted(A, B) ≈ (1/K) * count(`sig_A[i]` == `sig_B[i]`).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Weighted MinHash signature.
#[derive(Debug, Clone)]
pub struct WeightedMinHash {
    pub k: usize,
    pub r: Vec<Vec<f64>>, // r[i][j] for feature j and hash i (gamma-distributed)
    pub c: Vec<Vec<f64>>, // c[i][j] log-uniform
    pub beta: Vec<Vec<f64>>, // beta[i][j] uniform [0,1)
    pub signature_kstar: Vec<usize>,
    pub signature_t: Vec<i64>,
    pub n_features: usize,
}

impl WeightedMinHash {
    /// Create a fresh weighted MinHash for a vector of `n_features` with `k` samples.
    pub fn new(k: usize, n_features: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if k == 0 || n_features == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(k, n_features)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut r = vec![vec![0.0; n_features]; k];
        let mut c = vec![vec![0.0; n_features]; k];
        let mut beta = vec![vec![0.0; n_features]; k];
        for i in 0..k {
            for j in 0..n_features {
                r[i][j] = -rng.next_f64().max(1.0e-12).ln() - rng.next_f64().max(1.0e-12).ln();
                c[i][j] = -rng.next_f64().max(1.0e-12).ln() - rng.next_f64().max(1.0e-12).ln();
                beta[i][j] = rng.next_f64();
            }
        }
        Ok(Self {
            k,
            r,
            c,
            beta,
            signature_kstar: vec![0usize; k],
            signature_t: vec![0i64; k],
            n_features,
        })
    }

    /// Compute the signature for a non-negative weight vector.
    pub fn signature(&mut self, weights: &[f64]) -> SketchResult<()> {
        if weights.len() != self.n_features {
            return Err(SketchError::DimensionMismatch {
                a: weights.len(),
                b: self.n_features,
            });
        }
        for i in 0..self.k {
            let mut best_a = f64::INFINITY;
            let mut best_kstar = 0usize;
            let mut best_t = 0i64;
            for (j, &w) in weights.iter().enumerate().take(self.n_features) {
                if w <= 0.0 {
                    continue;
                }
                let t = ((w.ln() / self.r[i][j]).floor() + self.beta[i][j]) as i64;
                let y = self.r[i][j] * ((t as f64) - self.beta[i][j]);
                let a = self.c[i][j] / (y + self.r[i][j]).exp().max(1.0e-300);
                if a < best_a {
                    best_a = a;
                    best_kstar = j;
                    best_t = t;
                }
            }
            self.signature_kstar[i] = best_kstar;
            self.signature_t[i] = best_t;
        }
        Ok(())
    }

    /// Estimate weighted Jaccard via signature equality.
    pub fn estimate_weighted_jaccard(&self, other: &WeightedMinHash) -> SketchResult<f64> {
        if self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.k,
                b: other.k,
            });
        }
        let mut matches = 0usize;
        for i in 0..self.k {
            if self.signature_kstar[i] == other.signature_kstar[i]
                && self.signature_t[i] == other.signature_t[i]
            {
                matches += 1;
            }
        }
        Ok(matches as f64 / self.k as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wmh_constructs() {
        let mut rng = LcgRng::new(7);
        let w = WeightedMinHash::new(8, 16, &mut rng).expect("ok");
        assert_eq!(w.k, 8);
    }

    #[test]
    fn wmh_identical_vectors_match() {
        let mut rng1 = LcgRng::new(7);
        let mut rng2 = LcgRng::new(7);
        let mut a = WeightedMinHash::new(64, 32, &mut rng1).expect("ok");
        let mut b = WeightedMinHash::new(64, 32, &mut rng2).expect("ok");
        let w = vec![1.0; 32];
        a.signature(&w).expect("ok");
        b.signature(&w).expect("ok");
        let j = a.estimate_weighted_jaccard(&b).expect("ok");
        assert!((j - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wmh_invalid_params() {
        let mut rng = LcgRng::new(7);
        assert!(WeightedMinHash::new(0, 4, &mut rng).is_err());
        assert!(WeightedMinHash::new(4, 0, &mut rng).is_err());
    }
}
