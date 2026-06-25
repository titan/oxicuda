//! Builder-style constructors for the most-used sketches.
//!
//! Ergonomic, fluent alternatives to the positional constructors, mirroring the
//! `Foo::builder().a(..).b(..).build()` idiom. They validate parameters at
//! `build()` time and translate accuracy targets (`ε`, `δ`, false-positive
//! rate, standard error) into concrete dimensions.
//!
//! ```ignore
//! let hll = HllBuilder::new().precision(14).seed(7).build()?;
//! let cm  = CmBuilder::new().epsilon(0.01).delta(0.001).seed(7).build()?;
//! ```

use crate::cardinality::hll::HyperLogLog;
use crate::error::{SketchError, SketchResult};
use crate::frequency::count_min::CountMinSketch;
use crate::handle::LcgRng;
use crate::membership::bloom::BloomFilter;

/// Fluent builder for [`HyperLogLog`].
///
/// Either set [`precision`](HllBuilder::precision) directly, or set a target
/// relative [`standard_error`](HllBuilder::standard_error) and let the builder
/// pick the smallest precision achieving it (`SE ≈ 1.04/√m`, `m = 2^p`).
#[derive(Debug, Clone)]
pub struct HllBuilder {
    precision: Option<u32>,
    target_se: Option<f64>,
    seed: u64,
}

impl Default for HllBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HllBuilder {
    /// A fresh builder (defaults: precision unset, seed 0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            precision: None,
            target_se: None,
            seed: 0,
        }
    }

    /// Set the HyperLogLog precision `p` directly (`m = 2^p`, `4 ≤ p ≤ 16`).
    #[must_use]
    pub fn precision(mut self, p: u32) -> Self {
        self.precision = Some(p);
        self
    }

    /// Set a target relative standard error; the smallest `p` with
    /// `1.04/√(2^p) ≤ target` is selected at `build()`.
    #[must_use]
    pub fn standard_error(mut self, target: f64) -> Self {
        self.target_se = Some(target);
        self
    }

    /// Set the hash seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the [`HyperLogLog`]. If both precision and standard-error targets
    /// are set, the explicit precision wins.
    pub fn build(self) -> SketchResult<HyperLogLog> {
        let p = if let Some(p) = self.precision {
            p
        } else if let Some(target) = self.target_se {
            if !(target.is_finite() && target > 0.0) {
                return Err(SketchError::InvalidParameter {
                    name: "standard_error".to_string(),
                    reason: "must be a finite positive fraction".to_string(),
                });
            }
            // 1.04/sqrt(2^p) <= target  ⇔  2^p >= (1.04/target)^2.
            let needed_m = (1.04 / target).powi(2);
            let mut p = 4u32;
            while (1u64 << p) < needed_m.ceil() as u64 && p < 16 {
                p += 1;
            }
            p
        } else {
            return Err(SketchError::InvalidParameter {
                name: "precision".to_string(),
                reason: "set precision(p) or standard_error(target) before build()".to_string(),
            });
        };
        HyperLogLog::new(p, self.seed)
    }
}

/// Fluent builder for [`CountMinSketch`].
///
/// Either set [`dims`](CmBuilder::dims) directly, or set accuracy targets
/// [`epsilon`](CmBuilder::epsilon) and [`delta`](CmBuilder::delta); the builder
/// then uses `w = ⌈e/ε⌉`, `d = ⌈ln(1/δ)⌉`.
#[derive(Debug, Clone)]
pub struct CmBuilder {
    dims: Option<(usize, usize)>,
    epsilon: Option<f64>,
    delta: Option<f64>,
    seed: u64,
}

impl Default for CmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CmBuilder {
    /// A fresh builder (defaults: dims unset, seed 0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            dims: None,
            epsilon: None,
            delta: None,
            seed: 0,
        }
    }

    /// Set the table dimensions `(d, w)` directly.
    #[must_use]
    pub fn dims(mut self, d: usize, w: usize) -> Self {
        self.dims = Some((d, w));
        self
    }

    /// Set the additive-error fraction `ε ∈ (0, 1)` (controls width `w`).
    #[must_use]
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = Some(eps);
        self
    }

    /// Set the failure probability `δ ∈ (0, 1)` (controls depth `d`).
    #[must_use]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
        self
    }

    /// Set the hash seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the [`CountMinSketch`]. Explicit `dims` take precedence over
    /// `(epsilon, delta)`.
    pub fn build(self) -> SketchResult<CountMinSketch> {
        let mut rng = LcgRng::new(self.seed);
        if let Some((d, w)) = self.dims {
            return CountMinSketch::new(d, w, &mut rng);
        }
        match (self.epsilon, self.delta) {
            (Some(eps), Some(delta)) => CountMinSketch::from_eps_delta(eps, delta, &mut rng),
            _ => Err(SketchError::InvalidParameter {
                name: "dims".to_string(),
                reason: "set dims(d, w) or both epsilon(..) and delta(..) before build()"
                    .to_string(),
            }),
        }
    }
}

/// Fluent builder for [`BloomFilter`].
///
/// Either set [`bits_hashes`](BloomBuilder::bits_hashes) directly, or set a
/// capacity and target [`false_positive`](BloomBuilder::false_positive) rate;
/// the builder picks `m = −n·ln(p)/(ln 2)²`, `k = (m/n)·ln 2`.
#[derive(Debug, Clone)]
pub struct BloomBuilder {
    bits_hashes: Option<(usize, usize)>,
    capacity: Option<usize>,
    fp_rate: Option<f64>,
    seed: u64,
}

impl Default for BloomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomBuilder {
    /// A fresh builder (defaults: unset, seed 0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bits_hashes: None,
            capacity: None,
            fp_rate: None,
            seed: 0,
        }
    }

    /// Set `m` bits and `k` hash functions directly.
    #[must_use]
    pub fn bits_hashes(mut self, m: usize, k: usize) -> Self {
        self.bits_hashes = Some((m, k));
        self
    }

    /// Set the expected item capacity `n`.
    #[must_use]
    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = Some(n);
        self
    }

    /// Set the target false-positive rate `p ∈ (0, 1)`.
    #[must_use]
    pub fn false_positive(mut self, p: f64) -> Self {
        self.fp_rate = Some(p);
        self
    }

    /// Set the base hash seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the [`BloomFilter`]. Explicit `bits_hashes` take precedence over
    /// `(capacity, false_positive)`.
    pub fn build(self) -> SketchResult<BloomFilter> {
        if let Some((m, k)) = self.bits_hashes {
            return BloomFilter::new(m, k, self.seed);
        }
        match (self.capacity, self.fp_rate) {
            (Some(n), Some(p)) => BloomFilter::with_expected_fp(n, p, self.seed),
            _ => Err(SketchError::InvalidParameter {
                name: "bits_hashes".to_string(),
                reason: "set bits_hashes(m, k) or both capacity(..) and false_positive(..)"
                    .to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hll_builder_precision() {
        let hll = HllBuilder::new().precision(14).seed(7).build().expect("ok");
        assert_eq!(hll.p, 14);
        assert_eq!(hll.seed, 7);
    }

    #[test]
    fn hll_builder_standard_error_picks_precision() {
        // SE target 0.01 ⇒ need 2^p >= (1.04/0.01)^2 = 10816 ⇒ p = 14 (16384).
        let hll = HllBuilder::new().standard_error(0.01).build().expect("ok");
        assert_eq!(hll.p, 14);
        let realised_se = 1.04 / (hll.m as f64).sqrt();
        assert!(realised_se <= 0.01, "SE {realised_se} exceeds target");
    }

    #[test]
    fn hll_builder_loose_se_small_precision() {
        // SE target 0.10 ⇒ need 2^p >= ~108 ⇒ p = 7 (128).
        let hll = HllBuilder::new().standard_error(0.10).build().expect("ok");
        assert!(hll.p <= 7, "precision {} larger than needed", hll.p);
        assert!(1.04 / (hll.m as f64).sqrt() <= 0.10);
    }

    #[test]
    fn hll_builder_errors() {
        assert!(HllBuilder::new().build().is_err());
        assert!(HllBuilder::new().standard_error(-1.0).build().is_err());
        assert!(HllBuilder::new().precision(99).build().is_err());
    }

    #[test]
    fn cm_builder_dims() {
        let cm = CmBuilder::new().dims(5, 256).seed(3).build().expect("ok");
        assert_eq!(cm.d, 5);
        assert_eq!(cm.w, 256);
    }

    #[test]
    fn cm_builder_eps_delta() {
        // w = ceil(e/0.01) = 272, d = ceil(ln(1000)) = 7.
        let cm = CmBuilder::new()
            .epsilon(0.01)
            .delta(0.001)
            .build()
            .expect("ok");
        assert_eq!(cm.w, (std::f64::consts::E / 0.01).ceil() as usize);
        assert_eq!(cm.d, (1.0f64 / 0.001).ln().ceil() as usize);
    }

    #[test]
    fn cm_builder_errors() {
        assert!(CmBuilder::new().build().is_err());
        assert!(CmBuilder::new().epsilon(0.01).build().is_err());
        assert!(CmBuilder::new().epsilon(2.0).delta(0.1).build().is_err());
    }

    #[test]
    fn cm_builder_deterministic_seed() {
        let a = CmBuilder::new().dims(4, 128).seed(42).build().expect("ok");
        let b = CmBuilder::new().dims(4, 128).seed(42).build().expect("ok");
        // Same seed ⇒ identical hash coefficients ⇒ identical queries.
        let mut a = a;
        let mut b = b;
        for i in 0..100u64 {
            a.update(i, 1);
            b.update(i, 1);
        }
        for i in 0..100u64 {
            assert_eq!(a.query(i), b.query(i));
        }
    }

    #[test]
    fn bloom_builder_dims_and_fp() {
        let bf = BloomBuilder::new()
            .bits_hashes(4096, 5)
            .seed(1)
            .build()
            .expect("ok");
        assert_eq!(bf.m, 4096);
        assert_eq!(bf.k, 5);

        let bf2 = BloomBuilder::new()
            .capacity(1000)
            .false_positive(0.01)
            .build()
            .expect("ok");
        assert!(bf2.m >= 1000);
        assert!(bf2.k >= 1);
    }

    #[test]
    fn bloom_builder_errors() {
        assert!(BloomBuilder::new().build().is_err());
        assert!(BloomBuilder::new().capacity(100).build().is_err());
        assert!(
            BloomBuilder::new()
                .capacity(100)
                .false_positive(1.5)
                .build()
                .is_err()
        );
    }
}
