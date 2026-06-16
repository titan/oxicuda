//! Continuous-value (level / thermometer) encoders for HDC.
//!
//! Encoding a real scalar `x ∈ [min, max]` into a binary `{±1}` hypervector so that the
//! HD similarity between two encodings is a *monotone decreasing* function of `|x₁ − x₂|`
//! is a core primitive for HDC regression and analog-feature classification
//! (Rahimi 2016; Widdows & Cohen 2015; Schlegel 2021 survey).
//!
//! Two construction schemes are provided, both producing `Vec<i8>` in `{−1, +1}` to match
//! the crate-wide binary hypervector representation:
//!
//! - **Level hypervectors (continuous interpolation).** A bank of `n_levels` HVs is built
//!   so that level `0` is a random seed and each successive level flips a fixed contiguous
//!   block of `⌈D / (n_levels − 1)⌉` components. Thus level `0` and level `n_levels − 1`
//!   are (nearly) antipodal, and the cosine similarity between level `i` and level `j`
//!   decreases linearly in `|i − j|`. A continuous value is encoded by linear interpolation
//!   between its two bracketing level HVs (majority of a weighted bundle), yielding a smooth,
//!   monotone similarity profile.
//!
//! - **Thermometer code.** A real value is quantised to `k ∈ {0, …, n_levels − 1}` and the
//!   HV is built by partitioning the `D` components into `n_levels` contiguous bands; bands
//!   `≤ k` are set to `+1`, the rest to `−1`. The Hamming distance between two thermometer
//!   codes is exactly proportional to the difference of their quantised levels.
//!
//! Both encoders reject out-of-range inputs via [`HdcError::FeatureIndexOutOfRange`] when
//! the value falls outside `[min, max]` after clamping is disabled.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary::random_binary;

/// Continuous-value encoder using a bank of monotone-similarity level hypervectors.
///
/// Level `0` is a random binary seed; each subsequent level flips the sign of a fixed
/// contiguous block of components relative to the previous level, so similarity decreases
/// monotonically with level distance.
pub struct LevelEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// Number of discrete levels (≥ 2).
    n_levels: usize,
    /// Lower bound of the encoded value range.
    min: f32,
    /// Upper bound of the encoded value range.
    max: f32,
    /// Pre-computed level hypervectors (`n_levels` entries, each length `dim`).
    level_hvs: Vec<Vec<i8>>,
}

impl LevelEncoder {
    /// Create a new level encoder.
    ///
    /// - `n_levels`: number of discrete reference levels (must be ≥ 2).
    /// - `dim`: hypervector dimension (must be ≥ 1).
    /// - `min`, `max`: inclusive value range; `min` must be strictly less than `max`.
    /// - `rng`: random number generator for the seed level.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::InvalidNgramOrder`] (reused for level-count) if `n_levels < 2`.
    /// - [`HdcError::InvalidProbability`] if `min >= max`.
    pub fn new(
        n_levels: usize,
        dim: usize,
        min: f32,
        max: f32,
        rng: &mut LcgRng,
    ) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_levels < 2 {
            return Err(HdcError::InvalidNgramOrder(n_levels));
        }
        if min >= max {
            return Err(HdcError::InvalidProbability(min as f64));
        }

        // Number of components flipped per level step. Spread the full D flips across the
        // (n_levels - 1) steps so level 0 and level n_levels-1 become (nearly) antipodal.
        let steps = n_levels - 1;
        let flip_per_step = dim.div_ceil(steps);

        let seed = random_binary(dim, rng)?;
        let mut level_hvs = Vec::with_capacity(n_levels);
        level_hvs.push(seed.clone());

        let mut current = seed;
        for step in 0..steps {
            let start = step * flip_per_step;
            let end = (start + flip_per_step).min(dim);
            for c in current.iter_mut().take(end).skip(start) {
                *c = -*c;
            }
            level_hvs.push(current.clone());
        }

        Ok(Self {
            dim,
            n_levels,
            min,
            max,
            level_hvs,
        })
    }

    /// Map a continuous value to a fractional level index in `[0, n_levels − 1]`.
    fn fractional_level(&self, value: f32) -> HdcResult<f32> {
        if value < self.min || value > self.max {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: 0,
                max: self.n_levels,
            });
        }
        let frac = (value - self.min) / (self.max - self.min);
        Ok(frac * (self.n_levels - 1) as f32)
    }

    /// Encode a continuous value into a binary hypervector by linear interpolation between
    /// the two bracketing level HVs.
    ///
    /// For a fractional level `f` with integer floor `lo` and ceil `hi`, the output takes
    /// component `c` from level `hi` with probability `f − lo` and from level `lo` otherwise.
    /// The threshold is deterministic per component (compared against a position-dependent
    /// fraction) so the encoding is fully reproducible for a fixed value.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `value` is outside `[min, max]`.
    pub fn encode(&self, value: f32) -> HdcResult<Vec<i8>> {
        let f = self.fractional_level(value)?;
        let lo = f.floor() as usize;
        let hi = (lo + 1).min(self.n_levels - 1);
        let weight_hi = f - lo as f32; // in [0, 1)

        // Deterministic interpolation: the first `round(weight_hi * dim)` components (by a
        // fixed stride pattern) are taken from the higher level, the rest from the lower.
        // Using a contiguous block keeps the similarity profile monotone and reproducible.
        let n_from_hi = (weight_hi * self.dim as f32).round() as usize;
        let lo_hv = &self.level_hvs[lo];
        let hi_hv = &self.level_hvs[hi];

        let mut out = vec![0i8; self.dim];
        for (i, slot) in out.iter_mut().enumerate() {
            // Spread the "from-hi" picks uniformly using a stride so that partial
            // interpolation stays between the two bracket levels in similarity.
            let from_hi = (i * n_from_hi) / self.dim != (i.wrapping_add(1) * n_from_hi) / self.dim;
            *slot = if from_hi { hi_hv[i] } else { lo_hv[i] };
        }
        Ok(out)
    }

    /// Return the pre-computed hypervector for an exact integer level index.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `level >= n_levels`.
    pub fn level_hv(&self, level: usize) -> HdcResult<&[i8]> {
        if level >= self.n_levels {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: level,
                max: self.n_levels,
            });
        }
        Ok(&self.level_hvs[level])
    }

    /// Number of discrete levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Hypervector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Encode a quantised value as a thermometer code in `{−1, +1}^D`.
///
/// The `D` components are partitioned into `n_levels` contiguous bands. Bands with index
/// `≤ level` are set to `+1`; the remaining bands are `−1`. The Hamming distance between two
/// thermometer codes is proportional to the difference of their levels, giving a monotone
/// similarity scale ideal for ordinal features.
///
/// # Errors
///
/// - [`HdcError::ZeroDimension`] if `dim == 0`.
/// - [`HdcError::InvalidNgramOrder`] if `n_levels < 1`.
/// - [`HdcError::FeatureIndexOutOfRange`] if `level >= n_levels`.
pub fn thermometer_encode(level: usize, n_levels: usize, dim: usize) -> HdcResult<Vec<i8>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    if n_levels == 0 {
        return Err(HdcError::InvalidNgramOrder(n_levels));
    }
    if level >= n_levels {
        return Err(HdcError::FeatureIndexOutOfRange {
            feat: level,
            max: n_levels,
        });
    }
    let band = dim.div_ceil(n_levels);
    // Number of leading +1 components: bands 0..=level.
    let n_active = ((level + 1) * band).min(dim);
    let mut out = vec![-1i8; dim];
    for slot in out.iter_mut().take(n_active) {
        *slot = 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::cosine_binary;
    use crate::distance::hamming::hamming_count;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0x1EEE_1123_4567_89AB)
    }

    #[test]
    fn level_encoder_construction() {
        let mut r = LcgRng::new(1);
        let enc = LevelEncoder::new(8, 1024, 0.0, 1.0, &mut r).expect("new");
        assert_eq!(enc.n_levels(), 8);
        assert_eq!(enc.dim(), 1024);
    }

    #[test]
    fn level_encoding_monotone_similarity() {
        // Similarity between level 0 and level i must decrease monotonically in i.
        let mut r = LcgRng::new(2);
        let enc = LevelEncoder::new(10, 2000, 0.0, 1.0, &mut r).expect("new");
        let base = enc.level_hv(0).expect("level 0").to_vec();
        let mut prev = f32::INFINITY;
        for i in 0..enc.n_levels() {
            let sim = cosine_binary(&base, enc.level_hv(i).expect("level")).expect("cosine");
            assert!(
                sim <= prev + 1e-6,
                "similarity not monotone at level {i}: {sim} > {prev}"
            );
            prev = sim;
        }
    }

    #[test]
    fn level_self_similarity_is_one() {
        let mut r = LcgRng::new(3);
        let enc = LevelEncoder::new(6, 1000, -1.0, 1.0, &mut r).expect("new");
        for i in 0..enc.n_levels() {
            let hv = enc.level_hv(i).expect("level");
            let sim = cosine_binary(hv, hv).expect("cosine");
            assert!((sim - 1.0).abs() < 1e-6, "level {i} self-sim = {sim}");
        }
    }

    #[test]
    fn level_extremes_nearly_antipodal() {
        // Level 0 vs the last level should be (close to) antipodal: cosine near -1.
        let mut r = LcgRng::new(4);
        let enc = LevelEncoder::new(5, 1000, 0.0, 10.0, &mut r).expect("new");
        let lo = enc.level_hv(0).expect("lo");
        let hi = enc.level_hv(enc.n_levels() - 1).expect("hi");
        let sim = cosine_binary(lo, hi).expect("cosine");
        assert!(sim < -0.9, "extremes not antipodal: {sim}");
    }

    #[test]
    fn encode_value_in_range_deterministic() {
        let mut r = LcgRng::new(5);
        let enc = LevelEncoder::new(8, 1024, 0.0, 1.0, &mut r).expect("new");
        let a = enc.encode(0.37).expect("encode");
        let b = enc.encode(0.37).expect("encode");
        assert_eq!(a, b, "encoding must be deterministic");
        assert_eq!(a.len(), 1024);
        for &v in &a {
            assert!(v == 1 || v == -1);
        }
    }

    #[test]
    fn encode_close_values_more_similar_than_far() {
        let mut r = LcgRng::new(6);
        let enc = LevelEncoder::new(16, 4000, 0.0, 1.0, &mut r).expect("new");
        let v_ref = enc.encode(0.5).expect("ref");
        let v_near = enc.encode(0.55).expect("near");
        let v_far = enc.encode(0.95).expect("far");
        let sim_near = cosine_binary(&v_ref, &v_near).expect("near sim");
        let sim_far = cosine_binary(&v_ref, &v_far).expect("far sim");
        assert!(
            sim_near > sim_far,
            "near={sim_near} should exceed far={sim_far}"
        );
    }

    #[test]
    fn encode_at_exact_level_matches_level_hv() {
        // value at min should equal level 0; value at max should equal last level.
        let mut r = LcgRng::new(7);
        let enc = LevelEncoder::new(8, 800, 2.0, 5.0, &mut r).expect("new");
        let at_min = enc.encode(2.0).expect("min");
        assert_eq!(at_min, enc.level_hv(0).expect("level 0"));
        let at_max = enc.encode(5.0).expect("max");
        assert_eq!(at_max, enc.level_hv(enc.n_levels() - 1).expect("last"));
    }

    #[test]
    fn encode_out_of_range_errors() {
        let mut r = LcgRng::new(8);
        let enc = LevelEncoder::new(4, 256, 0.0, 1.0, &mut r).expect("new");
        assert!(matches!(
            enc.encode(-0.1),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
        assert!(matches!(
            enc.encode(1.5),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn level_hv_out_of_range_errors() {
        let mut r = LcgRng::new(9);
        let enc = LevelEncoder::new(4, 256, 0.0, 1.0, &mut r).expect("new");
        assert!(matches!(
            enc.level_hv(4),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn new_rejects_bad_args() {
        let mut r = rng();
        assert!(matches!(
            LevelEncoder::new(2, 0, 0.0, 1.0, &mut r),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            LevelEncoder::new(1, 256, 0.0, 1.0, &mut r),
            Err(HdcError::InvalidNgramOrder(1))
        ));
        assert!(matches!(
            LevelEncoder::new(4, 256, 1.0, 1.0, &mut r),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn thermometer_monotone_hamming() {
        // Hamming distance from level 0 grows monotonically with level.
        let n_levels = 8;
        let dim = 800;
        let base = thermometer_encode(0, n_levels, dim).expect("base");
        let mut prev = 0usize;
        for level in 0..n_levels {
            let hv = thermometer_encode(level, n_levels, dim).expect("therm");
            let dist = hamming_count(&base, &hv).expect("hamming");
            assert!(
                dist >= prev,
                "non-monotone at level {level}: {dist} < {prev}"
            );
            prev = dist;
        }
    }

    #[test]
    fn thermometer_band_structure() {
        // level 0 has the first band active; the highest level is all +1.
        let hv0 = thermometer_encode(0, 4, 8).expect("therm0");
        assert_eq!(hv0, vec![1, 1, -1, -1, -1, -1, -1, -1]);
        let hv_top = thermometer_encode(3, 4, 8).expect("therm top");
        assert!(hv_top.iter().all(|&v| v == 1));
    }

    #[test]
    fn thermometer_rejects_bad_args() {
        assert!(matches!(
            thermometer_encode(0, 4, 0),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            thermometer_encode(0, 0, 8),
            Err(HdcError::InvalidNgramOrder(0))
        ));
        assert!(matches!(
            thermometer_encode(5, 4, 8),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }
}
