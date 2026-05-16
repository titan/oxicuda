//! AMS-style L2-norm sketch (Alon, Matias, Szegedy 1996).
//!
//! Maintain `d * t` Rademacher hashes; for input pair (i, c) update `X[d][t] += c * s_{d,t}(i)`
//! where `s` is a {-1, +1} hash of `i`.
//! Estimate: median over `d` of mean of `X[d][t]^2` over `t`. Unbiased for L2^2.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// AMS L2 sketch with `d` median copies of `t` parallel estimators each.
#[derive(Debug, Clone)]
pub struct AmsL2Sketch {
    pub d: usize,
    pub t: usize,
    pub state: Vec<f64>,           // d * t
    pub hashes: Vec<TwoUniversal>, // d * t hashes mod 2
}

impl AmsL2Sketch {
    /// New AMS sketch with `d` median rows × `t` mean columns.
    pub fn new(d: usize, t: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || t == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,t)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let hashes = TwoUniversal::many(rng, d * t, 2);
        Ok(Self {
            d,
            t,
            state: vec![0.0; d * t],
            hashes,
        })
    }

    /// Update the sketch with `(i, c)` (interpreted as `x[i] += c`).
    pub fn update(&mut self, i: u64, c: f64) {
        for row in 0..self.d {
            for col in 0..self.t {
                let idx = row * self.t + col;
                let s = if self.hashes[idx].hash(i) == 0 {
                    -1.0
                } else {
                    1.0
                };
                self.state[idx] += s * c;
            }
        }
    }

    /// Estimate the L2^2 norm via median-of-means.
    #[must_use]
    pub fn estimate_l2_squared(&self) -> f64 {
        let mut means = Vec::with_capacity(self.d);
        for row in 0..self.d {
            let mut sum = 0.0;
            for col in 0..self.t {
                let v = self.state[row * self.t + col];
                sum += v * v;
            }
            means.push(sum / self.t as f64);
        }
        means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        means[self.d / 2]
    }

    /// Estimate L2 norm (square-root of L2^2 estimate).
    #[must_use]
    pub fn estimate_l2(&self) -> f64 {
        self.estimate_l2_squared().max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ams_constructs() {
        let mut rng = LcgRng::new(11);
        let s = AmsL2Sketch::new(5, 32, &mut rng).expect("ok");
        assert_eq!(s.state.len(), 160);
    }

    #[test]
    fn ams_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(AmsL2Sketch::new(0, 4, &mut rng).is_err());
        assert!(AmsL2Sketch::new(4, 0, &mut rng).is_err());
    }

    #[test]
    fn ams_l2_close_to_truth() {
        let mut rng = LcgRng::new(7);
        let mut s = AmsL2Sketch::new(7, 256, &mut rng).expect("ok");
        // Insert a vector with known L2^2 = sum c_i^2 = 100 * 1 = 100.
        for i in 0..100u64 {
            s.update(i, 1.0);
        }
        let est = s.estimate_l2_squared();
        let rel = (est - 100.0).abs() / 100.0;
        assert!(rel < 0.3, "AMS L2^2 rel-err = {rel}");
    }
}
