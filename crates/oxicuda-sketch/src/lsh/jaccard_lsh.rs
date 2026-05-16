//! Jaccard LSH (banded MinHash): r-row × b-band structure for approximate-NN queries.
//!
//! Given MinHash signatures of length `K = r * b`, partition into `b` bands each of `r` rows.
//! Two signatures collide in a band iff their `r` row-values match exactly.
//! Probability of at least one band collision: `1 - (1 - s^r)^b` for true similarity `s`.

use crate::error::{SketchError, SketchResult};
use crate::similarity::minhash::MinHash;

/// Jaccard LSH with `r` rows per band and `b` bands.
#[derive(Debug, Clone)]
pub struct JaccardLsh {
    pub r: usize,
    pub b: usize,
    pub k: usize,
}

impl JaccardLsh {
    /// Construct with rows/band `r` and bands `b`; total signature length k = r * b.
    pub fn new(r: usize, b: usize) -> SketchResult<Self> {
        if r == 0 || b == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(r, b)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { r, b, k: r * b })
    }

    /// Extract the b banded hash values from a MinHash signature.
    ///
    /// For each band, hash the `r` row values together using xxh3-min into a single u64.
    pub fn band_keys(&self, mh: &MinHash) -> SketchResult<Vec<u64>> {
        if mh.signature.len() != self.k {
            return Err(SketchError::DimensionMismatch {
                a: mh.signature.len(),
                b: self.k,
            });
        }
        let mut bands = Vec::with_capacity(self.b);
        for band in 0..self.b {
            // Pack rows into bytes and hash.
            let mut bytes: Vec<u8> = Vec::with_capacity(self.r * 8);
            for row in 0..self.r {
                let v = mh.signature[band * self.r + row];
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let h = crate::hash::xxh3_min::xxh3_64(&bytes, band as u64);
            bands.push(h);
        }
        Ok(bands)
    }

    /// Probability of a band-level collision for Jaccard similarity `s`.
    #[must_use]
    pub fn collision_probability(&self, s: f64) -> f64 {
        let pr = s.clamp(0.0, 1.0).powi(self.r as i32);
        1.0 - (1.0 - pr).powi(self.b as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn jlsh_constructs() {
        let lsh = JaccardLsh::new(4, 8).expect("ok");
        assert_eq!(lsh.k, 32);
    }

    #[test]
    fn jlsh_invalid_params() {
        assert!(JaccardLsh::new(0, 4).is_err());
        assert!(JaccardLsh::new(4, 0).is_err());
    }

    #[test]
    fn jlsh_collision_prob_monotonic() {
        let lsh = JaccardLsh::new(5, 10).expect("ok");
        assert!(lsh.collision_probability(0.9) > lsh.collision_probability(0.1));
    }

    #[test]
    fn jlsh_band_keys_identical_sets() {
        let rng = LcgRng::new(11);
        let lsh = JaccardLsh::new(4, 8).expect("ok");
        let mh1 = MinHash::from_set(&[1, 2, 3, 4, 5, 6], 32, &mut rng.clone()).expect("ok");
        let mh2 = MinHash::from_set(&[1, 2, 3, 4, 5, 6], 32, &mut rng.clone()).expect("ok");
        let b1 = lsh.band_keys(&mh1).expect("ok");
        let b2 = lsh.band_keys(&mh2).expect("ok");
        assert_eq!(b1, b2);
    }
}
