//! Count-Min Sketch (Cormode, Muthukrishnan 2005).
//!
//! A `d x w` table; for each row a 2-universal hash maps the key to a column.
//! `insert(x, c)`: for each row `T[row][h_row(x)]` += c.
//! `query(x)`: returns min over rows.
//!
//! Guarantee: with high probability `query(x) <= true_count(x) + eps * total`
//! where `w = ceil(e/eps)`, `d = ceil(ln(1/delta))`.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// Count-Min Sketch.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    pub d: usize,
    pub w: usize,
    pub table: Vec<u64>,
    pub hashes: Vec<TwoUniversal>,
}

impl CountMinSketch {
    /// Create a Count-Min Sketch with depth `d` and width `w`. RNG is used to draw hash coefs.
    pub fn new(d: usize, w: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || w == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,w)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let hashes = TwoUniversal::many(rng, d, w as u64);
        let table = vec![0u64; d * w];
        Ok(Self {
            d,
            w,
            table,
            hashes,
        })
    }

    /// Construct CMS with parameters chosen from `(eps, delta)`:
    /// `w = ceil(e / eps)`, `d = ceil(ln(1/delta))`.
    pub fn from_eps_delta(eps: f64, delta: f64, rng: &mut LcgRng) -> SketchResult<Self> {
        if !(0.0 < eps && eps < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "eps".to_string(),
                reason: "must be in (0,1)".to_string(),
            });
        }
        if !(0.0 < delta && delta < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "delta".to_string(),
                reason: "must be in (0,1)".to_string(),
            });
        }
        let w = (std::f64::consts::E / eps).ceil() as usize;
        let d = (1.0 / delta).ln().ceil() as usize;
        Self::new(d.max(1), w.max(1), rng)
    }

    /// Increment count of `x` by `c`.
    pub fn update(&mut self, x: u64, c: u64) {
        for row in 0..self.d {
            let col = self.hashes[row].hash(x) as usize;
            self.table[row * self.w + col] = self.table[row * self.w + col].saturating_add(c);
        }
    }

    /// Insert with count = 1.
    pub fn add(&mut self, x: u64) {
        self.update(x, 1);
    }

    /// Estimate the frequency of `x` (lower-bounded by true count, upper-bounded with high prob).
    #[must_use]
    pub fn query(&self, x: u64) -> u64 {
        let mut best = u64::MAX;
        for row in 0..self.d {
            let col = self.hashes[row].hash(x) as usize;
            let v = self.table[row * self.w + col];
            if v < best {
                best = v;
            }
        }
        best
    }

    /// Merge another CMS into this one. Both must share the same hashes (same dimensions).
    pub fn merge(&mut self, other: &CountMinSketch) -> SketchResult<()> {
        if self.d != other.d || self.w != other.w {
            return Err(SketchError::DimensionMismatch {
                a: self.d * self.w,
                b: other.d * other.w,
            });
        }
        for i in 0..self.table.len() {
            self.table[i] = self.table[i].saturating_add(other.table[i]);
        }
        Ok(())
    }

    /// Total of all counts in the sketch (note: counts `d*total` because each insertion writes to d rows).
    #[must_use]
    pub fn total(&self) -> u64 {
        let mut s = 0u64;
        for &v in &self.table {
            s = s.saturating_add(v);
        }
        s / self.d.max(1) as u64
    }

    /// Reset the sketch to empty.
    pub fn clear(&mut self) {
        for v in self.table.iter_mut() {
            *v = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cms_constructs() {
        let mut rng = LcgRng::new(11);
        let cms = CountMinSketch::new(4, 16, &mut rng).expect("ok");
        assert_eq!(cms.d, 4);
        assert_eq!(cms.w, 16);
    }

    #[test]
    fn cms_query_is_over_estimate() {
        let mut rng = LcgRng::new(11);
        let mut cms = CountMinSketch::new(5, 256, &mut rng).expect("ok");
        for i in 0..1000u64 {
            cms.add(i % 100);
        }
        for k in 0..100u64 {
            let q = cms.query(k);
            assert!(q >= 10, "CMS underestimated for key={k}: {q}");
        }
    }

    #[test]
    fn cms_unseen_estimate_small() {
        let mut rng = LcgRng::new(7);
        let mut cms = CountMinSketch::new(5, 1024, &mut rng).expect("ok");
        for i in 0..500u64 {
            cms.add(i);
        }
        let q = cms.query(99999);
        // Unseen item should have at most O(eps * total) over-estimate.
        assert!(q < 50, "unseen item estimate too large: {q}");
    }

    #[test]
    fn cms_from_eps_delta() {
        let mut rng = LcgRng::new(11);
        let cms = CountMinSketch::from_eps_delta(0.01, 0.01, &mut rng).expect("ok");
        assert!(cms.w >= 271); // ceil(e/0.01) = 272 (or 271)
        assert!(cms.d >= 4);
    }

    #[test]
    fn cms_merge_sums_counts() {
        // Share hash coefficients by drawing both CMS from the same RNG state.
        let mut rng = LcgRng::new(11);
        let mut cms1 = CountMinSketch::new(3, 64, &mut rng).expect("ok");
        let mut rng2 = LcgRng::new(11);
        let mut cms2 = CountMinSketch::new(3, 64, &mut rng2).expect("ok");
        cms1.add(7);
        cms1.add(7);
        cms2.add(7);
        cms1.merge(&cms2).expect("ok");
        assert!(cms1.query(7) >= 3);
    }
}
