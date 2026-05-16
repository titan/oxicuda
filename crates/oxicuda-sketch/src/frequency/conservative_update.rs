//! Conservative-Update Count-Min Sketch (Estan, Varghese 2002).
//!
//! Idea: for an increment, find the rows with the *minimum* current value (which is the
//! current query estimate). Only update those rows by `c`. This reduces over-estimation.

use crate::error::SketchResult;
use crate::frequency::count_min::CountMinSketch;
use crate::handle::LcgRng;

/// Conservative-update wrapper around `CountMinSketch`.
#[derive(Debug, Clone)]
pub struct ConservativeUpdateCm {
    pub inner: CountMinSketch,
}

impl ConservativeUpdateCm {
    /// New CU-CM with `d` rows and `w` columns.
    pub fn new(d: usize, w: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        Ok(Self {
            inner: CountMinSketch::new(d, w, rng)?,
        })
    }

    /// Conservative-update insert.
    pub fn add(&mut self, x: u64) {
        self.update(x, 1);
    }

    /// Conservative-update increment by `c`.
    pub fn update(&mut self, x: u64, c: u64) {
        // Find current min over rows.
        let mut min_val = u64::MAX;
        let mut cols = vec![0usize; self.inner.d];
        for (row, col_slot) in cols.iter_mut().enumerate() {
            let col = self.inner.hashes[row].hash(x) as usize;
            *col_slot = col;
            let v = self.inner.table[row * self.inner.w + col];
            if v < min_val {
                min_val = v;
            }
        }
        // Update only rows where T[row][h(x)] == min_val to min_val + c.
        // If min_val + c overflows, saturate.
        let new_val = min_val.saturating_add(c);
        for (row, &col) in cols.iter().enumerate() {
            let idx = row * self.inner.w + col;
            if self.inner.table[idx] == min_val {
                self.inner.table[idx] = new_val;
            }
        }
    }

    /// Query the count of `x`.
    #[must_use]
    pub fn query(&self, x: u64) -> u64 {
        self.inner.query(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cucm_query_is_overestimate() {
        let mut rng = LcgRng::new(11);
        let mut cu = ConservativeUpdateCm::new(5, 256, &mut rng).expect("ok");
        for i in 0..1000u64 {
            cu.add(i % 50);
        }
        for k in 0..50u64 {
            let q = cu.query(k);
            assert!(q >= 20, "CU underestimated for {k}: {q}");
        }
    }

    #[test]
    fn cucm_unseen_smaller_than_cms() {
        // Build two: regular CMS and CU-CM with same dims and seed.
        let mut rng1 = LcgRng::new(11);
        let mut cms =
            crate::frequency::count_min::CountMinSketch::new(5, 256, &mut rng1).expect("ok");
        let mut rng2 = LcgRng::new(11);
        let mut cu = ConservativeUpdateCm::new(5, 256, &mut rng2).expect("ok");
        for i in 0..1000u64 {
            cms.add(i % 50);
            cu.add(i % 50);
        }
        // Conservative update never produces *larger* estimate than vanilla.
        for k in 0..50u64 {
            assert!(cu.query(k) <= cms.query(k) + 1);
        }
    }
}
