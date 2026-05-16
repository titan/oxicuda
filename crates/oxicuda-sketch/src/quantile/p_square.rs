//! P² (P-square) online single-quantile algorithm (Jain, Chlamtac 1985).
//!
//! Maintains 5 markers tracking a single target quantile `q` using piecewise parabolic
//! interpolation. Constant memory, no histogram needed.

use crate::error::{SketchError, SketchResult};

/// P-square estimator for a single quantile `q`.
#[derive(Debug, Clone)]
pub struct PSquare {
    pub q: f64,
    pub heights: [f64; 5],
    pub positions: [f64; 5],
    pub desired: [f64; 5],
    pub increments: [f64; 5],
    pub initialised: bool,
    pub samples: Vec<f64>,
}

impl PSquare {
    /// New P-square targeting quantile `q ∈ (0, 1)`.
    pub fn new(q: f64) -> SketchResult<Self> {
        if !(0.0 < q && q < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "q".to_string(),
                reason: "must be in (0, 1)".to_string(),
            });
        }
        Ok(Self {
            q,
            heights: [0.0; 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired: [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0],
            increments: [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0],
            initialised: false,
            samples: Vec::new(),
        })
    }

    /// Add a value.
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if !self.initialised {
            self.samples.push(x);
            if self.samples.len() == 5 {
                self.samples
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..5 {
                    self.heights[i] = self.samples[i];
                }
                self.initialised = true;
                self.samples.clear();
            }
            return;
        }

        // Find cell k such that heights[k] <= x < heights[k+1].
        let k = if x < self.heights[0] {
            self.heights[0] = x;
            0usize
        } else if x >= self.heights[4] {
            self.heights[4] = x;
            3usize
        } else {
            let mut idx = 0usize;
            for i in 0..4 {
                if x < self.heights[i + 1] {
                    idx = i;
                    break;
                }
            }
            idx
        };

        // Increment positions of markers k+1..4.
        for i in (k + 1)..5 {
            self.positions[i] += 1.0;
        }
        // Update desired positions for all markers.
        for i in 0..5 {
            self.desired[i] += self.increments[i];
        }

        // Adjust heights of markers 1..=3 if their desired position is far from current.
        for i in 1..4 {
            let d = self.desired[i] - self.positions[i];
            let above = self.positions[i + 1] - self.positions[i];
            let below = self.positions[i - 1] - self.positions[i];
            if (d >= 1.0 && above > 1.0) || (d <= -1.0 && below < -1.0) {
                let s = d.signum();
                let h_new = parabolic(self, i, s);
                let h_new = if self.heights[i - 1] < h_new && h_new < self.heights[i + 1] {
                    h_new
                } else {
                    linear(self, i, s)
                };
                self.heights[i] = h_new;
                self.positions[i] += s;
            }
        }
    }

    /// Current estimate of the quantile.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        if !self.initialised {
            if self.samples.is_empty() {
                return 0.0;
            }
            let mut s = self.samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((s.len() as f64) * self.q) as usize;
            return s[idx.min(s.len() - 1)];
        }
        self.heights[2]
    }
}

fn parabolic(ps: &PSquare, i: usize, s: f64) -> f64 {
    let a = (s / (ps.positions[i + 1] - ps.positions[i - 1])).abs();
    let t1 = (ps.positions[i] - ps.positions[i - 1] + s) * (ps.heights[i + 1] - ps.heights[i])
        / (ps.positions[i + 1] - ps.positions[i]);
    let t2 = (ps.positions[i + 1] - ps.positions[i] - s) * (ps.heights[i] - ps.heights[i - 1])
        / (ps.positions[i] - ps.positions[i - 1]);
    ps.heights[i] + a * s.signum() * (t1 + t2)
}

fn linear(ps: &PSquare, i: usize, s: f64) -> f64 {
    let j = if s >= 0.0 { i + 1 } else { i - 1 };
    ps.heights[i] + s * (ps.heights[j] - ps.heights[i]) / (ps.positions[j] - ps.positions[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psquare_constructs() {
        let p = PSquare::new(0.5).expect("ok");
        assert!(!p.initialised);
    }

    #[test]
    fn psquare_invalid_q() {
        assert!(PSquare::new(0.0).is_err());
        assert!(PSquare::new(1.0).is_err());
    }

    #[test]
    fn psquare_median_uniform() {
        let mut p = PSquare::new(0.5).expect("ok");
        for i in 0..1000 {
            p.add(i as f64);
        }
        let e = p.estimate();
        assert!((e - 500.0).abs() < 50.0, "median {e}");
    }

    #[test]
    fn psquare_p95_uniform() {
        let mut p = PSquare::new(0.95).expect("ok");
        for i in 0..1000 {
            p.add(i as f64);
        }
        let e = p.estimate();
        assert!((e - 950.0).abs() < 50.0, "p95 {e}");
    }
}
