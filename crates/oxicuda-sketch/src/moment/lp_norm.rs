//! Lp norm estimation via stable random projections (Indyk 2000).
//!
//! For `p ∈ (0, 2]`, draw entries of projection matrix from p-stable distribution.
//! For `p = 2`: Gaussian.
//! For `p = 1`: Cauchy.
//! Estimate: median of |y_i| / median_constant.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Lp norm sketch with `k` projections.
#[derive(Debug, Clone)]
pub struct LpNormSketch {
    pub p: f64,
    pub k: usize,
    pub state: Vec<f64>,
    pub dim: usize,
    pub coeffs: Vec<f64>, // k * dim flat
}

impl LpNormSketch {
    /// New L1 sketch: use Cauchy entries via inverse-CDF tan(π(u - 0.5)).
    pub fn new_l1(dim: usize, k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if dim == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(dim,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut coeffs = vec![0.0; dim * k];
        for v in coeffs.iter_mut() {
            // Inverse CDF of standard Cauchy.
            let u = rng.next_f64();
            *v = (std::f64::consts::PI * (u - 0.5)).tan();
        }
        Ok(Self {
            p: 1.0,
            k,
            state: vec![0.0; k],
            dim,
            coeffs,
        })
    }

    /// New L2 sketch with Gaussian entries.
    pub fn new_l2(dim: usize, k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if dim == 0 || k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(dim,k)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut coeffs = vec![0.0; dim * k];
        for v in coeffs.iter_mut() {
            *v = rng.next_normal();
        }
        Ok(Self {
            p: 2.0,
            k,
            state: vec![0.0; k],
            dim,
            coeffs,
        })
    }

    /// Update with (index, value): `state[i]` += `coeffs[i][idx]` * value.
    pub fn update(&mut self, idx: usize, c: f64) {
        if idx >= self.dim {
            return;
        }
        for i in 0..self.k {
            self.state[i] += self.coeffs[i * self.dim + idx] * c;
        }
    }

    /// Estimate Lp norm via median of |y_i| / median_constant.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let mut abs: Vec<f64> = self.state.iter().map(|x| x.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = abs[abs.len() / 2];
        // Median normalisation constant: for L1 Cauchy, median(|cauchy|) = 1, so we divide by 1.
        // For L2 Gaussian, median(|N(0,1)|) ≈ 0.6745. So estimate = med / 0.6745.
        match self.p {
            2.0 => med / 0.6745,
            _ => med,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(LpNormSketch::new_l1(0, 4, &mut rng).is_err());
        assert!(LpNormSketch::new_l2(4, 0, &mut rng).is_err());
    }

    #[test]
    fn lp_l2_estimate_close() {
        let mut rng = LcgRng::new(11);
        let dim = 50;
        let k = 200;
        let mut s = LpNormSketch::new_l2(dim, k, &mut rng).expect("ok");
        for i in 0..dim {
            s.update(i, 1.0);
        }
        // True L2 = sqrt(50) ≈ 7.07.
        let est = s.estimate();
        assert!((est - 7.07).abs() < 2.0, "L2 estimate {est}");
    }
}
