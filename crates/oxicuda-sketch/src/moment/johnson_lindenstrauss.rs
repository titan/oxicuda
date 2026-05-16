//! Johnson-Lindenstrauss (1984) random projection.
//!
//! Project `d`-dim vector into `k`-dim via `y = (1 / sqrt(k)) * G * x` where `G` is a
//! Gaussian (or Rademacher) random matrix. Preserves pairwise distances within `(1 ± eps)`
//! factor with high probability when `k = O(log(n) / eps^2)`.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Johnson-Lindenstrauss projection from `d`-dim → `k`-dim.
#[derive(Debug, Clone)]
pub struct JlProjection {
    pub d: usize,
    pub k: usize,
    pub g: Vec<f64>, // k * d row-major
}

impl JlProjection {
    /// Create with Gaussian random projection matrix.
    pub fn new_gaussian(d: usize, k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut g = vec![0.0; d * k];
        for v in g.iter_mut() {
            *v = rng.next_normal();
        }
        Ok(Self { d, k, g })
    }

    /// Rademacher (±1 with prob 0.5 each) JL projection.
    pub fn new_rademacher(d: usize, k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut g = vec![0.0; d * k];
        for v in g.iter_mut() {
            *v = if rng.next_bool() { 1.0 } else { -1.0 };
        }
        Ok(Self { d, k, g })
    }

    /// Project a vector `x` of length `d` to `k`-dim.
    pub fn project(&self, x: &[f64]) -> SketchResult<Vec<f64>> {
        if x.len() != self.d {
            return Err(SketchError::DimensionMismatch {
                a: x.len(),
                b: self.d,
            });
        }
        let scale = 1.0 / (self.k as f64).sqrt();
        let mut y = vec![0.0; self.k];
        for (i, y_i) in y.iter_mut().enumerate().take(self.k) {
            let mut acc = 0.0;
            for (j, &xj) in x.iter().enumerate().take(self.d) {
                acc += self.g[i * self.d + j] * xj;
            }
            *y_i = scale * acc;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jl_constructs_gaussian() {
        let mut rng = LcgRng::new(11);
        let j = JlProjection::new_gaussian(8, 4, &mut rng).expect("ok");
        assert_eq!(j.g.len(), 32);
    }

    #[test]
    fn jl_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(JlProjection::new_gaussian(0, 4, &mut rng).is_err());
        assert!(JlProjection::new_gaussian(4, 0, &mut rng).is_err());
    }

    #[test]
    fn jl_preserves_distances() {
        let mut rng = LcgRng::new(7);
        let d = 64;
        let k = 200;
        let j = JlProjection::new_rademacher(d, k, &mut rng).expect("ok");
        let mut errs = Vec::new();
        for _ in 0..40 {
            let x: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            let y: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            let diff: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
            let true_d2: f64 = diff.iter().map(|v| v * v).sum();
            let px = j.project(&x).expect("ok");
            let py = j.project(&y).expect("ok");
            let proj_d2: f64 = px.iter().zip(&py).map(|(a, b)| (a - b) * (a - b)).sum();
            errs.push((proj_d2 / true_d2 - 1.0).abs());
        }
        let mean: f64 = errs.iter().sum::<f64>() / errs.len() as f64;
        assert!(mean < 0.3, "JL mean rel-err = {mean}");
    }
}
