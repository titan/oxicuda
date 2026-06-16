//! Differentially-private Count-Min sketch (Laplace mechanism).
//!
//! The Count-Min sketch already provides a compact frequency summary; adding
//! calibrated noise makes its released counts **ε-differentially private**
//! (Dwork & Roth 2014; the construction follows the pan-private sketch of Mir,
//! Muthukrishnan, Nikolov & Wright 2011).
//!
//! # Privacy accounting
//!
//! A single record (one increment of weight 1) changes exactly one cell per
//! row, so the L1 sensitivity of the **whole table** is `d` (one per row). To
//! release the entire noisy table under ε-DP we therefore add independent
//! Laplace noise of scale `b = d / ε` to every cell. Equivalently, a *query*
//! reads one cell per row and takes the minimum/median; because the rows are
//! disjoint in their contribution to any single record, per-cell noise of scale
//! `1/ε_row` with `ε_row = ε/d` composes (by basic sequential composition
//! across the `d` rows) to `ε`-DP for the released query.
//!
//! # Estimation
//!
//! Laplace noise has mean zero, so the noisy per-row estimates are unbiased.
//! The classic Count-Min `min` estimator is no longer ideal once values can be
//! negative, so this sketch also offers the **median-of-rows** estimator, which
//! is robust to the symmetric Laplace perturbation and is the recommended
//! readout for the differentially-private regime.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// A differentially-private Count-Min sketch over `i64` counters.
#[derive(Debug, Clone)]
pub struct DpCountMin {
    /// Number of rows (hash functions).
    d: usize,
    /// Number of columns per row.
    w: usize,
    /// Privacy budget ε used to calibrate the Laplace noise.
    epsilon: f64,
    /// Laplace scale `b = d / ε` applied to each cell at finalisation.
    scale: f64,
    /// The integer count table (pre-noise).
    table: Vec<i64>,
    /// The noisy table (populated by [`DpCountMin::finalize`]).
    noisy: Vec<f64>,
    /// Whether the noisy table has been generated.
    finalized: bool,
    /// 2-universal hashes, one per row.
    hashes: Vec<TwoUniversal>,
    /// RNG used for Laplace sampling.
    rng: LcgRng,
}

impl DpCountMin {
    /// Create a DP Count-Min with depth `d`, width `w`, and privacy budget
    /// `epsilon > 0`. `rng` seeds the hash family; the noise RNG is derived from
    /// it deterministically so the whole structure is reproducible.
    pub fn new(d: usize, w: usize, epsilon: f64, rng: &mut LcgRng) -> SketchResult<Self> {
        if d == 0 || w == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,w)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        let hashes = TwoUniversal::many(rng, d, w as u64);
        // Derive an independent noise RNG seed from the hash RNG stream.
        let noise_seed = rng.next_u64();
        Ok(Self {
            d,
            w,
            epsilon,
            scale: d as f64 / epsilon,
            table: vec![0i64; d * w],
            noisy: vec![0.0; d * w],
            finalized: false,
            hashes,
            rng: LcgRng::new(noise_seed),
        })
    }

    /// Increment the count of `x` by `c` (may be negative for turnstile streams).
    pub fn update(&mut self, x: u64, c: i64) {
        self.finalized = false;
        for row in 0..self.d {
            let col = self.hashes[row].hash(x) as usize;
            let idx = row * self.w + col;
            self.table[idx] = self.table[idx].saturating_add(c);
        }
    }

    /// Insert `x` with count 1.
    pub fn add(&mut self, x: u64) {
        self.update(x, 1);
    }

    /// Draw a Laplace(0, `scale`) variate via inverse-CDF on a uniform sample.
    fn laplace(&mut self) -> f64 {
        // u ∈ (−0.5, 0.5]; Laplace inverse-CDF is −b·sgn(u)·ln(1 − 2|u|).
        let u = self.rng.next_f64() - 0.5;
        let s = if u < 0.0 { -1.0 } else { 1.0 };
        -self.scale * s * (1.0 - 2.0 * u.abs()).max(1.0e-300).ln()
    }

    /// Generate the noisy table by adding fresh Laplace noise to every cell.
    ///
    /// Must be called before any private query. Re-finalising redraws noise.
    pub fn finalize(&mut self) {
        for i in 0..self.table.len() {
            let noise = self.laplace();
            self.noisy[i] = self.table[i] as f64 + noise;
        }
        self.finalized = true;
    }

    /// Private frequency estimate of `x` using the **median-of-rows** estimator.
    ///
    /// Returns an error if [`DpCountMin::finalize`] has not been called since the
    /// last update.
    pub fn query_median(&self, x: u64) -> SketchResult<f64> {
        if !self.finalized {
            return Err(SketchError::InvalidParameter {
                name: "state".to_string(),
                reason: "call finalize() before querying".to_string(),
            });
        }
        let mut vals: Vec<f64> = (0..self.d)
            .map(|row| {
                let col = self.hashes[row].hash(x) as usize;
                self.noisy[row * self.w + col]
            })
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = self.d / 2;
        let med = if self.d % 2 == 1 {
            vals[mid]
        } else {
            0.5 * (vals[mid - 1] + vals[mid])
        };
        Ok(med)
    }

    /// Private frequency estimate of `x` using the **minimum-of-rows** estimator
    /// (the classic Count-Min readout; biased upward but reported for parity).
    pub fn query_min(&self, x: u64) -> SketchResult<f64> {
        if !self.finalized {
            return Err(SketchError::InvalidParameter {
                name: "state".to_string(),
                reason: "call finalize() before querying".to_string(),
            });
        }
        let mut best = f64::INFINITY;
        for row in 0..self.d {
            let col = self.hashes[row].hash(x) as usize;
            let v = self.noisy[row * self.w + col];
            if v < best {
                best = v;
            }
        }
        Ok(best)
    }

    /// The Laplace scale `b = d / ε` used for each cell.
    #[must_use]
    pub fn noise_scale(&self) -> f64 {
        self.scale
    }

    /// The configured privacy budget ε.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Depth (number of rows).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.d
    }

    /// Width (columns per row).
    #[must_use]
    pub fn width(&self) -> usize {
        self.w
    }

    /// Reset all counts and noise to empty.
    pub fn clear(&mut self) {
        for v in self.table.iter_mut() {
            *v = 0;
        }
        for v in self.noisy.iter_mut() {
            *v = 0.0;
        }
        self.finalized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_with_valid_params() {
        let mut rng = LcgRng::new(11);
        let cm = DpCountMin::new(5, 256, 1.0, &mut rng).expect("ok");
        assert_eq!(cm.depth(), 5);
        assert_eq!(cm.width(), 256);
        assert!((cm.epsilon() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_epsilon() {
        let mut rng = LcgRng::new(11);
        assert!(DpCountMin::new(5, 256, 0.0, &mut rng).is_err());
        assert!(DpCountMin::new(5, 256, -1.0, &mut rng).is_err());
    }

    #[test]
    fn rejects_zero_dims() {
        let mut rng = LcgRng::new(11);
        assert!(DpCountMin::new(0, 256, 1.0, &mut rng).is_err());
        assert!(DpCountMin::new(5, 0, 1.0, &mut rng).is_err());
    }

    #[test]
    fn noise_scale_is_d_over_epsilon() {
        let mut rng = LcgRng::new(11);
        let cm = DpCountMin::new(6, 64, 2.0, &mut rng).expect("ok");
        assert!((cm.noise_scale() - 3.0).abs() < 1e-12); // 6 / 2
    }

    #[test]
    fn query_requires_finalize() {
        let mut rng = LcgRng::new(11);
        let mut cm = DpCountMin::new(5, 256, 1.0, &mut rng).expect("ok");
        cm.add(42);
        assert!(cm.query_median(42).is_err());
        cm.finalize();
        assert!(cm.query_median(42).is_ok());
    }

    #[test]
    fn high_epsilon_estimate_close_to_truth() {
        // Large epsilon → tiny noise → accurate counts.
        let mut rng = LcgRng::new(7);
        let mut cm = DpCountMin::new(7, 2048, 50.0, &mut rng).expect("ok");
        for _ in 0..1000 {
            cm.add(123);
        }
        cm.finalize();
        let est = cm.query_median(123).expect("ok");
        assert!((est - 1000.0).abs() < 50.0, "estimate {est} far from 1000");
    }

    #[test]
    fn median_estimator_unbiased_over_repeats() {
        // Averaging the median estimate across many noise redraws should sit
        // near the true count because Laplace noise is mean-zero.
        let mut rng = LcgRng::new(99);
        let mut cm = DpCountMin::new(9, 4096, 1.0, &mut rng).expect("ok");
        let truth = 500.0;
        for _ in 0..500 {
            cm.add(77);
        }
        let mut acc = 0.0;
        let reps = 200;
        for _ in 0..reps {
            cm.finalize();
            acc += cm.query_median(77).expect("ok");
        }
        let mean = acc / reps as f64;
        // Within a few noise standard deviations of truth.
        assert!(
            (mean - truth).abs() < 30.0,
            "mean estimate {mean} vs {truth}"
        );
    }

    #[test]
    fn finalize_redraws_noise() {
        let mut rng = LcgRng::new(3);
        let mut cm = DpCountMin::new(5, 512, 0.5, &mut rng).expect("ok");
        cm.add(10);
        cm.finalize();
        let a = cm.query_min(10).expect("ok");
        cm.finalize();
        let b = cm.query_min(10).expect("ok");
        // Different noise draws → (almost surely) different estimates.
        assert!((a - b).abs() > 1e-12, "noise was not redrawn: {a} == {b}");
    }

    #[test]
    fn turnstile_decrement_supported() {
        let mut rng = LcgRng::new(5);
        let mut cm = DpCountMin::new(9, 4096, 80.0, &mut rng).expect("ok");
        for _ in 0..100 {
            cm.update(55, 1);
        }
        for _ in 0..40 {
            cm.update(55, -1);
        }
        cm.finalize();
        let est = cm.query_median(55).expect("ok");
        assert!((est - 60.0).abs() < 20.0, "net count estimate {est} vs 60");
    }

    #[test]
    fn clear_empties_sketch() {
        let mut rng = LcgRng::new(11);
        let mut cm = DpCountMin::new(5, 256, 10.0, &mut rng).expect("ok");
        for _ in 0..100 {
            cm.add(9);
        }
        cm.clear();
        cm.finalize();
        let est = cm.query_median(9).expect("ok");
        assert!(est.abs() < 10.0, "cleared estimate {est} not near 0");
    }

    #[test]
    fn laplace_noise_has_expected_spread() {
        // The empirical std of the per-cell noise should be near b·√2.
        let mut rng = LcgRng::new(123);
        let mut cm = DpCountMin::new(2, 50_000, 1.0, &mut rng).expect("ok");
        cm.finalize(); // empty table → noisy[i] is pure noise
        let n = cm.noisy.len() as f64;
        let mean: f64 = cm.noisy.iter().sum::<f64>() / n;
        let var: f64 = cm.noisy.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let expected = cm.noise_scale() * std::f64::consts::SQRT_2; // b·√2
        let rel = (std - expected).abs() / expected;
        assert!(
            rel < 0.15,
            "noise std {std} vs expected {expected} (rel {rel})"
        );
    }

    #[test]
    fn min_estimator_available() {
        let mut rng = LcgRng::new(11);
        let mut cm = DpCountMin::new(5, 1024, 30.0, &mut rng).expect("ok");
        for _ in 0..200 {
            cm.add(1);
        }
        cm.finalize();
        let est = cm.query_min(1).expect("ok");
        assert!(est.is_finite());
    }
}
