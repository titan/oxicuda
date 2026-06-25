//! p-stable LSH (E2LSH) for general `L_p` metrics — Datar, Immorlica, Indyk,
//! Mirrokni (SoCG 2004).
//!
//! A locality-sensitive hash family for the `L_p` distance based on `p`-stable
//! distributions. For each hash function we draw a random vector
//! `a ∈ ℝ^dim` whose entries are i.i.d. samples from a `p`-stable distribution
//! and a uniform offset `b ∈ [0, r)`, then map a point `x` to the integer
//!
//! ```text
//!     h_{a,b}(x) = ⌊ (a · x + b) / r ⌋ .
//! ```
//!
//! By the `p`-stability property, `a · x − a · y` is distributed as
//! `‖x − y‖_p · Z`, where `Z` follows the same `p`-stable distribution. Hence
//! the collision probability `p(c)` for two points at distance `c = ‖x − y‖_p`
//! is
//!
//! ```text
//!     p(c) = Pr[h(x) = h(y)]
//!          = ∫_0^r (1/c) · f_p(t/c) · (1 − t/r) dt ,
//! ```
//!
//! where `f_p` is the density of the absolute value of the `p`-stable variable.
//! This is monotonically decreasing in `c`, which is exactly the LSH property:
//! near points collide more often than far points.
//!
//! ## Supported exponents
//!
//! * `p = 2` (Euclidean): the 2-stable distribution is the standard Gaussian.
//!   `a` is drawn from `N(0, 1)^dim` (Box–Muller via [`LcgRng::next_normal`]).
//! * `p = 1` (Manhattan): the 1-stable distribution is the standard Cauchy.
//!   `a` is drawn from `Cauchy(0, 1)^dim` via inverse transform
//!   `tan(π (u − 1/2))`.
//!
//! Concatenating `k` such integer hashes yields one *compound* signature; using
//! `l` independent compound signatures (the bands) gives the usual
//! amplification `1 − (1 − P^k)^l` for the collision probability.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// The `L_p` exponent supported by [`PStableLsh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableNorm {
    /// `L_1` (Manhattan) — Cauchy-stable projections.
    L1,
    /// `L_2` (Euclidean) — Gaussian-stable projections.
    L2,
}

impl StableNorm {
    /// The numeric exponent `p`.
    #[must_use]
    pub fn exponent(self) -> f64 {
        match self {
            StableNorm::L1 => 1.0,
            StableNorm::L2 => 2.0,
        }
    }
}

/// p-stable LSH family.
///
/// Stores `k · l` projection vectors (each of length `dim`) and matching
/// offsets. A query point is reduced to `l` band keys, each the hash of `k`
/// quantised projections.
#[derive(Debug, Clone)]
pub struct PStableLsh {
    /// Exponent of the `L_p` metric this family is sensitive to.
    pub norm: StableNorm,
    /// Input dimensionality.
    pub dim: usize,
    /// Number of integer hashes concatenated per band (`k`).
    pub k: usize,
    /// Number of independent bands (`l`).
    pub l: usize,
    /// Quantisation width `r > 0`.
    pub r: f64,
    /// Projection vectors: `k · l` rows of length `dim`, row-major.
    proj: Vec<f64>,
    /// Per-hash offsets `b ∈ [0, r)`; length `k · l`.
    offsets: Vec<f64>,
}

impl PStableLsh {
    /// Construct a p-stable LSH family.
    ///
    /// * `norm` — the `L_p` metric (`L1` or `L2`).
    /// * `dim` — input dimension (`> 0`).
    /// * `k` — hashes concatenated per band (`> 0`).
    /// * `l` — number of bands (`> 0`).
    /// * `r` — quantisation width (`> 0`); larger `r` ⇒ higher collision
    ///   probability and coarser buckets.
    pub fn new(
        norm: StableNorm,
        dim: usize,
        k: usize,
        l: usize,
        r: f64,
        rng: &mut LcgRng,
    ) -> SketchResult<Self> {
        if dim == 0 || k == 0 || l == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(dim, k, l)".to_string(),
                reason: "must all be positive".to_string(),
            });
        }
        if !(r.is_finite() && r > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "r".to_string(),
                reason: "must be a finite positive width".to_string(),
            });
        }
        let n_hash = k * l;
        let mut proj = vec![0.0f64; n_hash * dim];
        for v in proj.iter_mut() {
            *v = match norm {
                StableNorm::L2 => rng.next_normal(),
                StableNorm::L1 => standard_cauchy(rng),
            };
        }
        let offsets: Vec<f64> = (0..n_hash).map(|_| rng.next_f64() * r).collect();
        Ok(Self {
            norm,
            dim,
            k,
            l,
            r,
            proj,
            offsets,
        })
    }

    /// Quantised projection of `x` under hash index `h` (`0 ≤ h < k·l`):
    /// `⌊ (a_h · x + b_h) / r ⌋`.
    fn hash_one(&self, h: usize, x: &[f64]) -> i64 {
        let base = h * self.dim;
        let mut dot = self.offsets[h];
        for (j, &xj) in x.iter().enumerate().take(self.dim) {
            dot += self.proj[base + j] * xj;
        }
        (dot / self.r).floor() as i64
    }

    /// Compute the full `k · l` integer hash vector for `x`.
    pub fn raw_hashes(&self, x: &[f64]) -> SketchResult<Vec<i64>> {
        if x.len() != self.dim {
            return Err(SketchError::DimensionMismatch {
                a: x.len(),
                b: self.dim,
            });
        }
        Ok((0..self.k * self.l).map(|h| self.hash_one(h, x)).collect())
    }

    /// Compute the `l` band keys for `x`: each band collapses its `k` integer
    /// hashes into a single `u64` bucket id via xxh3-min over their bytes.
    ///
    /// Two points collide in a band iff all `k` of that band's quantised
    /// projections agree, so identical band keys indicate an `L_p`-near pair
    /// with the usual LSH probability.
    pub fn band_keys(&self, x: &[f64]) -> SketchResult<Vec<u64>> {
        let raw = self.raw_hashes(x)?;
        let mut bands = Vec::with_capacity(self.l);
        for band in 0..self.l {
            let mut bytes = Vec::with_capacity(self.k * 8);
            for row in 0..self.k {
                let v = raw[band * self.k + row];
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bands.push(crate::hash::xxh3_min::xxh3_64(&bytes, band as u64));
        }
        Ok(bands)
    }

    /// Collision probability `p(c)` for a single integer hash at `L_p` distance
    /// `c` between two points, evaluated by deterministic numerical
    /// integration of the closed-form expression
    /// `∫_0^r (1/c) f_p(t/c) (1 − t/r) dt` over a fixed grid.
    ///
    /// At `c = 0` the points coincide and `p = 1`.
    #[must_use]
    pub fn collision_probability(&self, c: f64) -> f64 {
        if c <= 0.0 {
            return 1.0;
        }
        let steps = 2_000usize;
        let h = self.r / steps as f64;
        // Composite trapezoidal rule on g(t) = (1/c) f_p(t/c) (1 - t/r).
        let g = |t: f64| -> f64 {
            let z = t / c;
            let density = match self.norm {
                StableNorm::L2 => {
                    // Density of |N(0,1)| at z: 2 φ(z) = sqrt(2/π) e^{-z²/2}.
                    (2.0 / std::f64::consts::PI).sqrt() * (-0.5 * z * z).exp()
                }
                StableNorm::L1 => {
                    // Density of |Cauchy(0,1)| at z: 2/(π (1+z²)).
                    2.0 / (std::f64::consts::PI * (1.0 + z * z))
                }
            };
            (density / c) * (1.0 - t / self.r)
        };
        let mut acc = 0.5 * (g(0.0) + g(self.r));
        for i in 1..steps {
            acc += g(i as f64 * h);
        }
        (acc * h).clamp(0.0, 1.0)
    }

    /// Amplified collision probability across all `l` bands of `k` hashes for a
    /// point pair at `L_p` distance `c`: `1 − (1 − p(c)^k)^l`.
    #[must_use]
    pub fn amplified_probability(&self, c: f64) -> f64 {
        let p = self.collision_probability(c);
        let single_band = p.powi(self.k as i32);
        1.0 - (1.0 - single_band).powi(self.l as i32)
    }
}

/// Draw one sample from the standard Cauchy distribution (the 1-stable law)
/// via inverse-CDF `tan(π (u − 1/2))`, with `u ∈ [0, 1)` from the LCG.
fn standard_cauchy(rng: &mut LcgRng) -> f64 {
    // Avoid the exact endpoints where tan diverges.
    let u = rng.next_f64().clamp(1.0e-12, 1.0 - 1.0e-12);
    (std::f64::consts::PI * (u - 0.5)).tan()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn l1(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
    }

    #[test]
    fn pstable_constructs() {
        let mut rng = LcgRng::new(7);
        let lsh = PStableLsh::new(StableNorm::L2, 16, 4, 8, 1.0, &mut rng).expect("ok");
        assert_eq!(lsh.dim, 16);
        assert_eq!(lsh.k, 4);
        assert_eq!(lsh.l, 8);
    }

    #[test]
    fn pstable_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(PStableLsh::new(StableNorm::L2, 0, 4, 8, 1.0, &mut rng).is_err());
        assert!(PStableLsh::new(StableNorm::L2, 4, 0, 8, 1.0, &mut rng).is_err());
        assert!(PStableLsh::new(StableNorm::L2, 4, 4, 0, 1.0, &mut rng).is_err());
        assert!(PStableLsh::new(StableNorm::L2, 4, 4, 8, 0.0, &mut rng).is_err());
        assert!(PStableLsh::new(StableNorm::L2, 4, 4, 8, f64::NAN, &mut rng).is_err());
    }

    #[test]
    fn pstable_identical_point_same_keys() {
        let mut rng = LcgRng::new(11);
        let lsh = PStableLsh::new(StableNorm::L2, 32, 4, 8, 2.0, &mut rng).expect("ok");
        let x: Vec<f64> = (0..32).map(|i| (i as f64) - 16.0).collect();
        let a = lsh.band_keys(&x).expect("ok");
        let b = lsh.band_keys(&x).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn pstable_dimension_mismatch() {
        let mut rng = LcgRng::new(3);
        let lsh = PStableLsh::new(StableNorm::L2, 8, 2, 4, 1.0, &mut rng).expect("ok");
        assert!(lsh.band_keys(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn pstable_collision_prob_monotone_decreasing_l2() {
        let mut rng = LcgRng::new(5);
        let lsh = PStableLsh::new(StableNorm::L2, 8, 1, 1, 4.0, &mut rng).expect("ok");
        let p0 = lsh.collision_probability(0.0);
        let p_small = lsh.collision_probability(0.5);
        let p_mid = lsh.collision_probability(2.0);
        let p_far = lsh.collision_probability(8.0);
        assert!((p0 - 1.0).abs() < 1e-9);
        assert!(p_small > p_mid, "{p_small} !> {p_mid}");
        assert!(p_mid > p_far, "{p_mid} !> {p_far}");
        assert!((0.0..=1.0).contains(&p_small));
    }

    #[test]
    fn pstable_collision_prob_monotone_decreasing_l1() {
        let mut rng = LcgRng::new(9);
        let lsh = PStableLsh::new(StableNorm::L1, 8, 1, 1, 4.0, &mut rng).expect("ok");
        let p_small = lsh.collision_probability(0.5);
        let p_far = lsh.collision_probability(8.0);
        assert!(p_small > p_far, "{p_small} !> {p_far}");
    }

    #[test]
    fn pstable_near_points_collide_more_often_l2() {
        // Empirically: a near pair should collide in (at least) more bands than
        // a far pair, matching the monotone collision probability.
        let mut rng = LcgRng::new(2024);
        let dim = 24;
        let lsh = PStableLsh::new(StableNorm::L2, dim, 2, 64, 3.0, &mut rng).expect("ok");
        let base: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let near: Vec<f64> = base.iter().map(|v| v + 0.05).collect();
        let far: Vec<f64> = base.iter().map(|v| v + 5.0).collect();
        assert!(l2(&base, &near) < l2(&base, &far));

        let kb = lsh.band_keys(&base).expect("ok");
        let kn = lsh.band_keys(&near).expect("ok");
        let kf = lsh.band_keys(&far).expect("ok");
        let near_hits = kb.iter().zip(&kn).filter(|(a, b)| a == b).count();
        let far_hits = kb.iter().zip(&kf).filter(|(a, b)| a == b).count();
        assert!(
            near_hits > far_hits,
            "near band-collisions {near_hits} should exceed far {far_hits}"
        );
    }

    #[test]
    fn pstable_near_points_collide_more_often_l1() {
        let mut rng = LcgRng::new(4242);
        let dim = 24;
        let lsh = PStableLsh::new(StableNorm::L1, dim, 2, 64, 3.0, &mut rng).expect("ok");
        let base: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).cos()).collect();
        let near: Vec<f64> = base.iter().map(|v| v + 0.03).collect();
        let far: Vec<f64> = base.iter().map(|v| v + 4.0).collect();
        assert!(l1(&base, &near) < l1(&base, &far));

        let kb = lsh.band_keys(&base).expect("ok");
        let kn = lsh.band_keys(&near).expect("ok");
        let kf = lsh.band_keys(&far).expect("ok");
        let near_hits = kb.iter().zip(&kn).filter(|(a, b)| a == b).count();
        let far_hits = kb.iter().zip(&kf).filter(|(a, b)| a == b).count();
        assert!(
            near_hits > far_hits,
            "near band-collisions {near_hits} should exceed far {far_hits}"
        );
    }

    #[test]
    fn pstable_amplified_prob_in_unit_interval() {
        let mut rng = LcgRng::new(123);
        let lsh = PStableLsh::new(StableNorm::L2, 8, 4, 16, 2.0, &mut rng).expect("ok");
        for &c in &[0.0_f64, 0.5, 1.0, 4.0, 16.0] {
            let p = lsh.amplified_probability(c);
            assert!((0.0..=1.0).contains(&p), "amplified p={p} for c={c}");
        }
        // Amplified probability is also monotone decreasing in distance.
        assert!(lsh.amplified_probability(0.5) >= lsh.amplified_probability(8.0));
    }

    #[test]
    fn pstable_cauchy_sampler_is_heavy_tailed() {
        // The Cauchy median is 0; check the empirical median is near 0 even
        // though the mean is undefined / unstable.
        let mut rng = LcgRng::new(77);
        let mut samples: Vec<f64> = (0..4001).map(|_| standard_cauchy(&mut rng)).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median = samples[samples.len() / 2];
        assert!(median.abs() < 0.2, "Cauchy median {median} not near 0");
    }
}
