//! Tug-of-war / AMS second-moment (F2) sketch (Alon, Matias, Szegedy 1996).
//!
//! Estimates `F2 = Σ_i f_i²`, the squared L2 norm of the frequency vector defined by a stream
//! of `(key, delta)` updates (`f_i = Σ deltas for key i`). The structure is a `d × t` table of
//! one-dimensional "tug-of-war" sketches
//!
//! ```text
//! X[r, c] = Σ_i s_{r,c}(i) · f_i,
//! ```
//!
//! where each `s_{r,c}` is a `±1` sign drawn from a **4-wise independent** family
//! ([`crate::hash::fourwise::FourWiseHash`]). Each `X[r,c]²` is an *unbiased* estimator of `F2`:
//!
//! ```text
//! E[X²] = Σ_i f_i² · E[s(i)²]  +  Σ_{i≠j} f_i f_j · E[s(i) s(j)] = F2 + 0,
//! ```
//!
//! using `E[s²] = 1` (pairwise/2-wise) and `E[s(i)s(j)] = 0` for `i ≠ j` (2-wise independence).
//! Its **variance** is controlled by the fourth moment:
//!
//! ```text
//! Var(X²) = E[X⁴] − F2² ≤ 2·F2²,
//! ```
//!
//! which requires `E[s_i s_j s_k s_l] = 0` whenever the four indices do not pair up — exactly the
//! guarantee 4-wise independence provides and that a merely 2-universal sign hash does **not**.
//! Averaging `t` independent columns shrinks the variance by `1/t`, and taking the **median**
//! over `d` rows boosts the success probability exponentially (median-of-means). With
//! `t = O(1/ε²)` and `d = O(log(1/δ))` the estimate is within `(1±ε)·F2` w.p. `1−δ`.
//!
//! All `d·t` sign hashes are seeded deterministically from a single `u64` seed, so two sketches
//! built with the same `(d, t, seed)` share identical signs and can be **linearly merged**
//! (`X_merged[r,c] = X_A[r,c] + X_B[r,c]`), which is correct because `X` is linear in `f`.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::fourwise::FourWiseHash;

/// Tug-of-war F2 (second-moment) sketch: `d` median rows × `t` mean columns of `±1` sketches.
#[derive(Debug, Clone)]
pub struct AmsF2Sketch {
    /// Number of median rows.
    pub d: usize,
    /// Number of mean columns per row.
    pub t: usize,
    /// Seed used to derive the sign hashes (retained so merges can verify compatibility).
    pub seed: u64,
    /// `d · t` accumulators in row-major order: `state[r*t + c] = X[r, c]`.
    state: Vec<f64>,
    /// `d · t` independent 4-wise sign hashes, row-major, parallel to `state`.
    signs: Vec<FourWiseHash>,
}

impl AmsF2Sketch {
    /// New F2 sketch with `d` median rows and `t` mean columns, sign hashes derived from `seed`.
    ///
    /// Returns [`SketchError::InvalidParameter`] if `d` or `t` is zero.
    pub fn new(d: usize, t: usize, seed: u64) -> SketchResult<Self> {
        if d == 0 || t == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,t)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut rng = LcgRng::new(seed);
        let signs = FourWiseHash::many(&mut rng, d * t);
        Ok(Self {
            d,
            t,
            seed,
            state: vec![0.0; d * t],
            signs,
        })
    }

    /// Apply `f[key] += delta`. Non-finite deltas are ignored.
    pub fn update(&mut self, key: u64, delta: f64) {
        if !delta.is_finite() {
            return;
        }
        for idx in 0..self.state.len() {
            self.state[idx] += self.signs[idx].sign(key) * delta;
        }
    }

    /// Estimate `F2 = Σ_i f_i²` by median-over-rows of the within-row mean of `X[r,c]²`.
    #[must_use]
    pub fn estimate_f2(&self) -> f64 {
        let mut row_means = Vec::with_capacity(self.d);
        for r in 0..self.d {
            let mut sum = 0.0;
            for c in 0..self.t {
                let v = self.state[r * self.t + c];
                sum += v * v;
            }
            row_means.push(sum / self.t as f64);
        }
        row_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median_of_sorted(&row_means)
    }

    /// Estimate the L2 norm `sqrt(F2)`.
    #[must_use]
    pub fn estimate_l2(&self) -> f64 {
        self.estimate_f2().max(0.0).sqrt()
    }

    /// Linearly merge `other` into `self`: `X[r,c] += X_other[r,c]`.
    ///
    /// Requires identical `(d, t, seed)` so the two share the same sign hashes (otherwise the
    /// column-wise add is meaningless). Returns [`SketchError::ShapeMismatch`] on a `(d, t)`
    /// mismatch and [`SketchError::InvalidParameter`] on a seed mismatch.
    pub fn merge(&mut self, other: &Self) -> SketchResult<()> {
        if self.d != other.d || self.t != other.t {
            return Err(SketchError::ShapeMismatch {
                expected: vec![self.d, self.t],
                got: vec![other.d, other.t],
            });
        }
        if self.seed != other.seed {
            return Err(SketchError::InvalidParameter {
                name: "seed".to_string(),
                reason: "merge requires identical seeds (shared sign hashes)".to_string(),
            });
        }
        for idx in 0..self.state.len() {
            self.state[idx] += other.state[idx];
        }
        Ok(())
    }

    /// Convenience: merge two F2 sketches into a fresh one. Both must share `(d, t, seed)`.
    pub fn merged(a: &Self, b: &Self) -> SketchResult<Self> {
        let mut out = a.clone();
        out.merge(b)?;
        Ok(out)
    }
}

/// Median of an already-sorted slice (lower-middle for even lengths). Returns `0.0` if empty.
#[inline]
fn median_of_sorted(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f2_invalid_params() {
        assert!(AmsF2Sketch::new(0, 4, 0).is_err());
        assert!(AmsF2Sketch::new(4, 0, 0).is_err());
    }

    #[test]
    fn f2_constructs() {
        let s = AmsF2Sketch::new(7, 64, 1).expect("ok");
        assert_eq!(s.state.len(), 7 * 64);
        assert_eq!(s.signs.len(), 7 * 64);
    }

    #[test]
    fn f2_single_item_squared() {
        // One key with total count c ⇒ F2 = c². Exact regardless of seed: X = ±c everywhere.
        let mut s = AmsF2Sketch::new(9, 64, 42).expect("ok");
        let c = 17.0;
        s.update(7, c);
        let est = s.estimate_f2();
        assert!((est - c * c).abs() < 1.0e-6, "single-item F2 = {est}");
    }

    #[test]
    fn f2_distinct_unit_items() {
        // K distinct items each count 1 ⇒ F2 = K. Tight with enough rows/cols + fixed seed.
        let k = 500u64;
        let mut s = AmsF2Sketch::new(15, 4096, 123_456).expect("ok");
        for i in 0..k {
            s.update(i, 1.0);
        }
        let est = s.estimate_f2();
        let rel = (est - k as f64).abs() / k as f64;
        assert!(rel < 0.10, "F2 rel-err = {rel} (est={est}, K={k})");
    }

    #[test]
    fn f2_deterministic_tight() {
        // Fixed seed, generous d/t ⇒ within a few percent of true Σ f_i².
        // Stream: key i appears (i+1) times for i in 0..50 ⇒ F2 = Σ (i+1)².
        let mut s = AmsF2Sketch::new(21, 8192, 2_718_281).expect("ok");
        let mut truth = 0.0f64;
        for i in 0..50u64 {
            let c = (i + 1) as f64;
            s.update(i, c);
            truth += c * c;
        }
        let est = s.estimate_f2();
        let rel = (est - truth).abs() / truth;
        assert!(rel < 0.05, "F2 rel-err = {rel} (est={est}, truth={truth})");
    }

    #[test]
    fn f2_negative_deltas_cancel() {
        // +c then -c ⇒ frequency 0 ⇒ F2 = 0 exactly.
        let mut s = AmsF2Sketch::new(5, 32, 1).expect("ok");
        s.update(3, 10.0);
        s.update(3, -10.0);
        assert!(s.estimate_f2().abs() < 1.0e-9);
    }

    #[test]
    fn f2_unbiasedness_over_seeds() {
        // Averaging estimate_f2() over many independent seeds converges to true F2,
        // and the many-seed average is much closer than a typical single seed.
        let truth = {
            let mut t = 0.0f64;
            for i in 0..40u64 {
                let c = ((i % 7) + 1) as f64;
                t += c * c;
            }
            t
        };
        let seeds = 200usize;
        let mut acc = 0.0f64;
        for s_idx in 0..seeds {
            // Small per-sketch budget so a SINGLE sketch is deliberately noisy; the
            // cross-seed AVERAGE is what must converge (unbiasedness).
            let mut sk = AmsF2Sketch::new(1, 16, 1000 + s_idx as u64).expect("ok");
            for i in 0..40u64 {
                let c = ((i % 7) + 1) as f64;
                sk.update(i, c);
            }
            acc += sk.estimate_f2();
        }
        let mean = acc / seeds as f64;
        let rel = (mean - truth).abs() / truth;
        assert!(
            rel < 0.05,
            "averaged F2 rel-err = {rel} (mean={mean}, truth={truth})"
        );
    }

    #[test]
    fn f2_merge_linearity() {
        // Split the stream; sketch each half with the SAME seed; merge ≈ whole.
        let seed = 555_777;
        let (d, t) = (17, 4096);
        let mut whole = AmsF2Sketch::new(d, t, seed).expect("ok");
        let mut half_a = AmsF2Sketch::new(d, t, seed).expect("ok");
        let mut half_b = AmsF2Sketch::new(d, t, seed).expect("ok");
        for i in 0..120u64 {
            let c = ((i % 5) + 1) as f64;
            whole.update(i, c);
            if i % 2 == 0 {
                half_a.update(i, c);
            } else {
                half_b.update(i, c);
            }
        }
        let merged = AmsF2Sketch::merged(&half_a, &half_b).expect("merge ok");
        let ew = whole.estimate_f2();
        let em = merged.estimate_f2();
        // Identical signs ⇒ merged state EQUALS whole state cell-for-cell ⇒ estimates equal.
        assert!(
            (ew - em).abs() / ew.max(1.0) < 1.0e-9,
            "merge≠whole: {ew} vs {em}"
        );
    }

    #[test]
    fn f2_merge_rejects_mismatch() {
        let a = AmsF2Sketch::new(4, 16, 1).expect("ok");
        let mut b_dim = AmsF2Sketch::new(5, 16, 1).expect("ok");
        assert!(b_dim.merge(&a).is_err(), "dim mismatch must error");
        let mut b_seed = AmsF2Sketch::new(4, 16, 2).expect("ok");
        assert!(b_seed.merge(&a).is_err(), "seed mismatch must error");
    }
}
