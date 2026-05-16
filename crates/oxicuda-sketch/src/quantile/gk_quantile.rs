//! Greenwald-Khanna (2001) ε-approximate quantile algorithm.
//!
//! Maintains a list of tuples `(v_i, g_i, Δ_i)` where:
//! - `v_i` is the item value,
//! - `g_i = r_min(v_i) - r_min(v_{i-1})` is the gap,
//! - `Δ_i = r_max(v_i) - r_min(v_i)` bounds the uncertainty.
//!
//! For any quantile `q`, returns a value `v` with rank in `[qn - εn, qn + εn]`.

use crate::error::{SketchError, SketchResult};

/// One GK tuple.
#[derive(Debug, Clone, Copy)]
struct GkTuple {
    v: f64,
    g: u64,
    delta: u64,
}

/// Greenwald-Khanna quantile sketch.
#[derive(Debug, Clone)]
pub struct GreenwaldKhanna {
    pub epsilon: f64,
    tuples: Vec<GkTuple>,
    pub n: u64,
}

impl GreenwaldKhanna {
    /// Construct a GK sketch with target relative error `epsilon ∈ (0, 1)`.
    pub fn new(epsilon: f64) -> SketchResult<Self> {
        if !(0.0 < epsilon && epsilon < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "must be in (0, 1)".to_string(),
            });
        }
        Ok(Self {
            epsilon,
            tuples: Vec::new(),
            n: 0,
        })
    }

    /// Insert a value into the sketch.
    pub fn add(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        let pos = self.tuples.partition_point(|t| t.v < v);
        let delta = if pos == 0 || pos == self.tuples.len() {
            0u64
        } else {
            // f(r_i, n) = 2 * eps * n, the upper bound on r_max - r_min.
            (2.0 * self.epsilon * self.n as f64).floor() as u64
        };
        self.tuples.insert(pos, GkTuple { v, g: 1, delta });
        self.n += 1;
        // Compress every 1/(2*eps) insertions.
        let compress_interval = (1.0 / (2.0 * self.epsilon)).ceil() as u64;
        if self.n % compress_interval.max(1) == 0 {
            self.compress();
        }
    }

    /// Compress: merge adjacent tuples whose combined uncertainty stays bounded.
    fn compress(&mut self) {
        if self.tuples.len() < 3 {
            return;
        }
        let two_eps_n = (2.0 * self.epsilon * self.n as f64).floor() as u64;
        // Iterate from second-to-last back to second; try to merge i and i+1.
        let mut i = self.tuples.len() - 2;
        while i >= 1 {
            let merged_g = self.tuples[i].g + self.tuples[i + 1].g;
            if merged_g + self.tuples[i + 1].delta <= two_eps_n {
                self.tuples[i + 1].g = merged_g;
                self.tuples.remove(i);
                if i == 0 {
                    break;
                }
                i -= 1;
            } else {
                i -= 1;
            }
        }
    }

    /// Estimate the value at quantile `q ∈ [0, 1]`.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        if self.tuples.is_empty() {
            return 0.0;
        }
        let r = (q.clamp(0.0, 1.0) * self.n as f64).ceil() as u64;
        let eps_n = (self.epsilon * self.n as f64).floor() as u64;
        let mut r_min = 0u64;
        // Find the tuple t such that r_min(t) <= r <= r_max(t).
        for t in &self.tuples {
            r_min += t.g;
            let r_max = r_min + t.delta;
            // r is "within" this tuple if r in [r_min - eps_n, r_max + eps_n].
            let lower_ok = r_min <= r.saturating_add(eps_n);
            let upper_ok = r <= r_max.saturating_add(eps_n);
            if lower_ok && upper_ok {
                return t.v;
            }
        }
        // Fallback: last tuple.
        self.tuples[self.tuples.len() - 1].v
    }

    /// Number of tuples currently stored.
    #[must_use]
    pub fn size(&self) -> usize {
        self.tuples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gk_constructs() {
        let g = GreenwaldKhanna::new(0.01).expect("ok");
        assert_eq!(g.size(), 0);
    }

    #[test]
    fn gk_invalid_eps() {
        assert!(GreenwaldKhanna::new(0.0).is_err());
        assert!(GreenwaldKhanna::new(1.5).is_err());
    }

    #[test]
    fn gk_median_uniform() {
        let mut g = GreenwaldKhanna::new(0.05).expect("ok");
        for i in 0..1000 {
            g.add(i as f64);
        }
        let m = g.quantile(0.5);
        assert!((m - 500.0).abs() < 100.0, "median {m}");
    }

    #[test]
    fn gk_quantile_extremes() {
        let mut g = GreenwaldKhanna::new(0.02).expect("ok");
        for i in 0..1000 {
            g.add(i as f64);
        }
        let lo = g.quantile(0.05);
        let hi = g.quantile(0.95);
        assert!(lo < 200.0);
        assert!(hi > 800.0);
    }
}
