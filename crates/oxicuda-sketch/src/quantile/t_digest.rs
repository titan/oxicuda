//! t-Digest quantile sketch (Dunning 2019).
//!
//! Stores a set of centroids `(mean, weight)`. Insertions append a new singleton centroid
//! then we sort & merge adjacent centroids subject to a scale function:
//!     k(q, δ) = (δ / (2π)) * arcsin(2q - 1)
//! Adjacent centroids may be merged iff the combined `k`-distance is `<= 1`.
//!
//! Quantile estimation: cumulative weights → interpolated mean.

use crate::error::{SketchError, SketchResult};

/// One t-Digest centroid.
#[derive(Debug, Clone, Copy)]
pub struct Centroid {
    pub mean: f64,
    pub weight: f64,
}

/// t-Digest sketch with compression parameter `delta`.
#[derive(Debug, Clone)]
pub struct TDigest {
    pub delta: f64,
    pub centroids: Vec<Centroid>,
    pub total_weight: f64,
    pub buffer: Vec<Centroid>,
    pub buffer_size: usize,
}

impl TDigest {
    /// New t-Digest with the given compression delta (10..=1000 typical).
    pub fn new(delta: f64) -> SketchResult<Self> {
        if !(1.0..=10_000.0).contains(&delta) {
            return Err(SketchError::InvalidParameter {
                name: "delta".to_string(),
                reason: "must be in [1,10000]".to_string(),
            });
        }
        Ok(Self {
            delta,
            centroids: Vec::new(),
            total_weight: 0.0,
            buffer: Vec::new(),
            buffer_size: (delta as usize).max(16) * 2,
        })
    }

    /// Insert a single value with optional weight (default 1.0).
    pub fn add(&mut self, value: f64) {
        self.add_weighted(value, 1.0);
    }

    /// Insert a weighted value.
    pub fn add_weighted(&mut self, value: f64, weight: f64) {
        if !value.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.buffer.push(Centroid {
            mean: value,
            weight,
        });
        if self.buffer.len() >= self.buffer_size {
            self.flush();
        }
    }

    /// Flush the buffer into the centroid array and compress.
    pub fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        // Move buffer into centroids, sort by mean, then compress greedily.
        for c in self.buffer.drain(..) {
            self.total_weight += c.weight;
            self.centroids.push(c);
        }
        self.compress();
    }

    /// Compress centroids by merging adjacent ones whose k-difference is <= 1.
    fn compress(&mut self) {
        if self.centroids.len() < 2 {
            return;
        }
        self.centroids.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total = self.total_weight;
        let delta = self.delta;
        let mut merged: Vec<Centroid> = Vec::with_capacity(self.centroids.len());
        // q-pointer: cumulative weight before current centroid.
        let mut q0 = 0.0;
        let mut cur = self.centroids[0];
        for c in &self.centroids[1..] {
            let q1 = q0 + cur.weight / total;
            let q2 = q1 + c.weight / total;
            // Required k-distance bound: each merged group must fit in k-space.
            let k_q0 = k_scale(q0, delta);
            let k_q2 = k_scale(q2, delta);
            if k_q2 - k_q0 <= 1.0 {
                // Merge.
                let total_w = cur.weight + c.weight;
                cur.mean = (cur.mean * cur.weight + c.mean * c.weight) / total_w;
                cur.weight = total_w;
            } else {
                merged.push(cur);
                cur = *c;
                q0 = q1;
            }
        }
        merged.push(cur);
        self.centroids = merged;
    }

    /// Estimate the value at quantile `q ∈ [0, 1]`.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() && self.buffer.is_empty() {
            return 0.0;
        }
        // Build a sorted snapshot if buffer is non-empty.
        let mut all: Vec<Centroid> = self.centroids.clone();
        all.extend_from_slice(&self.buffer);
        all.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total: f64 = all.iter().map(|c| c.weight).sum();
        if total <= 0.0 {
            return 0.0;
        }
        let target = q.clamp(0.0, 1.0) * total;

        // Walk cumulative.
        let mut acc = 0.0;
        for (i, c) in all.iter().enumerate() {
            let next_acc = acc + c.weight;
            if target <= next_acc {
                // Interpolate between previous centroid and this one.
                if i == 0 {
                    return c.mean;
                }
                let prev = all[i - 1];
                let frac = (target - (acc - prev.weight / 2.0))
                    / ((c.weight + prev.weight) / 2.0).max(1e-15);
                let frac = frac.clamp(0.0, 1.0);
                return prev.mean + frac * (c.mean - prev.mean);
            }
            acc = next_acc;
        }
        all[all.len() - 1].mean
    }

    /// Total number of items inserted (sum of weights).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.total_weight + self.buffer.iter().map(|c| c.weight).sum::<f64>()
    }
}

/// t-Digest k-scale function.
fn k_scale(q: f64, delta: f64) -> f64 {
    let q = q.clamp(0.0, 1.0);
    let arg = (2.0 * q - 1.0).clamp(-1.0, 1.0);
    (delta / std::f64::consts::TAU) * arg.asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdigest_constructs() {
        let td = TDigest::new(100.0).expect("ok");
        assert_eq!(td.centroids.len(), 0);
    }

    #[test]
    fn tdigest_invalid_delta() {
        assert!(TDigest::new(0.5).is_err());
        assert!(TDigest::new(50_000.0).is_err());
    }

    #[test]
    fn tdigest_median_uniform() {
        let mut td = TDigest::new(200.0).expect("ok");
        for i in 0..1000 {
            td.add(i as f64);
        }
        td.flush();
        let med = td.quantile(0.5);
        assert!((med - 500.0).abs() < 50.0, "median = {med}");
    }

    #[test]
    fn tdigest_extreme_quantiles() {
        let mut td = TDigest::new(200.0).expect("ok");
        for i in 0..1000 {
            td.add(i as f64);
        }
        td.flush();
        let q01 = td.quantile(0.01);
        let q99 = td.quantile(0.99);
        assert!(q01 < 50.0, "q01 = {q01}");
        assert!(q99 > 940.0, "q99 = {q99}");
    }

    #[test]
    fn tdigest_total_correct() {
        let mut td = TDigest::new(100.0).expect("ok");
        for _ in 0..100 {
            td.add(1.0);
        }
        td.flush();
        assert!((td.total() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn tdigest_k_scale_monotonic() {
        let v0 = k_scale(0.0, 100.0);
        let v05 = k_scale(0.5, 100.0);
        let v1 = k_scale(1.0, 100.0);
        assert!(v0 < v05);
        assert!(v05 < v1);
    }
}
