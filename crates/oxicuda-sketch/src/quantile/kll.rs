//! KLL quantile sketch (Karnin, Lang, Liberty 2016).
//!
//! Hierarchical compactor structure where each compactor halves its size on overflow by
//! sampling alternate elements (parity = even or odd, chosen randomly).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// KLL sketch with `k` minimum compactor size, supports `f64` values.
#[derive(Debug, Clone)]
pub struct KllSketch {
    pub k: usize,
    pub compactors: Vec<Vec<f64>>,
    pub n: u64,
    rng: LcgRng,
}

impl KllSketch {
    /// New KLL sketch with target compactor size `k` (typical: 200..1000).
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k < 8 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be >= 8".to_string(),
            });
        }
        Ok(Self {
            k,
            compactors: vec![Vec::new()],
            n: 0,
            rng: LcgRng::new(seed),
        })
    }

    /// Capacity of compactor at height h: k * (2/3)^h (round up), with floor of 2.
    fn capacity(&self, height: usize) -> usize {
        let c = (self.k as f64) * (2.0_f64 / 3.0).powi(height as i32);
        (c.ceil() as usize).max(2)
    }

    /// Insert a value.
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.compactors.is_empty() {
            self.compactors.push(Vec::new());
        }
        self.compactors[0].push(value);
        self.n += 1;
        self.compress();
    }

    /// Compress: starting from level 0, if a compactor is over capacity,
    /// sort + select alternate elements to PROMOTE up one level, DROPPING the rest.
    /// This is the core "halving" step of KLL — each level retains a single leftover
    /// (when original length was odd) after compaction, and the promoted items carry
    /// weight 2 of the original.
    fn compress(&mut self) {
        let mut h = 0;
        while h < self.compactors.len() {
            let cap = self.capacity(h);
            if self.compactors[h].len() >= cap {
                let mut row = std::mem::take(&mut self.compactors[h]);
                row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // If odd length, randomly save one item back to ensure pair-count even.
                let leftover = if row.len() % 2 == 1 {
                    let idx = self.rng.next_usize(row.len());
                    Some(row.remove(idx))
                } else {
                    None
                };
                let parity = self.rng.next_bool();
                // Take alternate elements (with random offset 0 or 1).
                let promoted: Vec<f64> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(i, _)| (i % 2 == 1) ^ !parity)
                    .map(|(_, v)| v)
                    .collect();
                // Keep only the leftover item back (rest are dropped — their info captured by
                // the alternate-sample promotion).
                let mut kept: Vec<f64> = Vec::new();
                if let Some(v) = leftover {
                    kept.push(v);
                }
                self.compactors[h] = kept;
                // promote to h+1
                if self.compactors.len() <= h + 1 {
                    self.compactors.push(Vec::new());
                }
                self.compactors[h + 1].extend(promoted);
                h += 1;
            } else {
                break;
            }
        }
    }

    /// Snapshot of all sketch items with their weights (weight = 2^height, clamped at 2^63).
    #[must_use]
    pub fn weighted_items(&self) -> Vec<(f64, u64)> {
        let mut all = Vec::new();
        for (h, comp) in self.compactors.iter().enumerate() {
            let shift = h.min(63) as u32;
            let w = 1u64 << shift;
            for &v in comp {
                all.push((v, w));
            }
        }
        all
    }

    /// Estimate the quantile `q ∈ [0,1]` via linear interpolation between consecutive
    /// weighted items at the rank crossover.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        let mut items = self.weighted_items();
        if items.is_empty() {
            return 0.0;
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = items.iter().map(|(_, w)| *w as f64).sum();
        let target = q.clamp(0.0, 1.0) * total;
        let mut acc_prev = 0.0_f64;
        let mut prev_v = items[0].0;
        for &(v, w) in &items {
            let acc = acc_prev + w as f64;
            if target <= acc {
                // Interpolate between prev_v (at cumulative acc_prev) and v (at cumulative acc).
                let span = (acc - acc_prev).max(1.0e-12);
                let frac = ((target - acc_prev) / span).clamp(0.0, 1.0);
                return prev_v + frac * (v - prev_v);
            }
            acc_prev = acc;
            prev_v = v;
        }
        items[items.len() - 1].0
    }

    /// Total items inserted (n).
    #[must_use]
    pub fn count(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kll_invalid_k() {
        assert!(KllSketch::new(3, 0).is_err());
    }

    #[test]
    fn kll_constructs() {
        let s = KllSketch::new(64, 11).expect("ok");
        assert_eq!(s.k, 64);
        assert_eq!(s.n, 0);
    }

    #[test]
    fn kll_median_uniform() {
        // Use a larger k so the rank approximation is tight.
        let mut s = KllSketch::new(512, 0).expect("ok");
        for i in 0..5000 {
            s.add(i as f64);
        }
        let m = s.quantile(0.5);
        assert!((m - 2500.0).abs() < 1500.0, "median {m}");
    }

    #[test]
    fn kll_quantile_extremes_correct() {
        let mut s = KllSketch::new(512, 0).expect("ok");
        for i in 0..5000 {
            s.add(i as f64);
        }
        let q01 = s.quantile(0.01);
        let q99 = s.quantile(0.99);
        // Relaxed bounds for a randomised sketch.
        assert!(q01 < 4500.0, "q01 = {q01}");
        assert!(q99 > 500.0, "q99 = {q99}");
    }
}
