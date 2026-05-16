//! Count Sketch (Charikar, Chen, Farach-Colton 2002).
//!
//! Per row r: position hash h_r(x) -> [0, w) and sign hash s_r(x) -> {-1, +1}.
//! update(x, c): `T[r][h_r(x)]` += s_r(x) * c.
//! query(x): median over rows of s_r(x) * `T[r][h_r(x)]`.
//! Provides unbiased estimate.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// Count Sketch.
#[derive(Debug, Clone)]
pub struct CountSketch {
    pub d: usize,
    pub w: usize,
    pub table: Vec<i64>,
    pub pos_hashes: Vec<TwoUniversal>,
    pub sign_hashes: Vec<TwoUniversal>,
}

impl CountSketch {
    /// New count sketch with `d` rows and `w` columns.
    pub fn new(d: usize, w: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || w == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,w)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let pos_hashes = TwoUniversal::many(rng, d, w as u64);
        let sign_hashes = TwoUniversal::many(rng, d, 2);
        Ok(Self {
            d,
            w,
            table: vec![0i64; d * w],
            pos_hashes,
            sign_hashes,
        })
    }

    /// Sign at row `r` for input `x` (returns +1 or -1).
    fn sign(&self, row: usize, x: u64) -> i64 {
        if self.sign_hashes[row].hash(x) == 0 {
            -1
        } else {
            1
        }
    }

    /// Update count of `x` by `c`.
    pub fn update(&mut self, x: u64, c: i64) {
        for row in 0..self.d {
            let col = self.pos_hashes[row].hash(x) as usize;
            let s = self.sign(row, x);
            self.table[row * self.w + col] += s * c;
        }
    }

    /// Insert with count = +1.
    pub fn add(&mut self, x: u64) {
        self.update(x, 1);
    }

    /// Query estimate using the median of `s * T[r][h(x)]` over rows.
    #[must_use]
    pub fn query(&self, x: u64) -> i64 {
        let mut samples: Vec<i64> = (0..self.d)
            .map(|row| {
                let col = self.pos_hashes[row].hash(x) as usize;
                self.sign(row, x) * self.table[row * self.w + col]
            })
            .collect();
        samples.sort_unstable();
        // Median (use lower middle for even-length, doesn't matter much here)
        samples[self.d / 2]
    }

    /// Merge another count sketch (must share dimensions).
    pub fn merge(&mut self, other: &CountSketch) -> SketchResult<()> {
        if self.d != other.d || self.w != other.w {
            return Err(SketchError::DimensionMismatch {
                a: self.table.len(),
                b: other.table.len(),
            });
        }
        for i in 0..self.table.len() {
            self.table[i] = self.table[i].saturating_add(other.table[i]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cs_constructs() {
        let mut rng = LcgRng::new(11);
        let cs = CountSketch::new(5, 64, &mut rng).expect("ok");
        assert_eq!(cs.table.len(), 320);
    }

    #[test]
    fn cs_estimate_close_to_true() {
        let mut rng = LcgRng::new(7);
        let mut cs = CountSketch::new(7, 1024, &mut rng).expect("ok");
        for _ in 0..100 {
            cs.add(42);
        }
        let q = cs.query(42);
        // Should be very close to 100 — Count Sketch is unbiased and has small variance with d=7, w=1024.
        assert!((q - 100).abs() < 20, "estimate = {q}");
    }

    #[test]
    fn cs_unseen_zero_centred() {
        let mut rng = LcgRng::new(7);
        let mut cs = CountSketch::new(7, 1024, &mut rng).expect("ok");
        for i in 0..500u64 {
            cs.add(i);
        }
        let q = cs.query(99999);
        assert!(q.abs() < 50, "unseen estimate = {q}");
    }
}
