//! Calibration and diagnostics for the LSH families.
//!
//! These routines measure empirical properties of an LSH configuration so a
//! caller can verify the hash family behaves as theory predicts before relying
//! on it for recall:
//!
//! * [`bucket_size_distribution`] — hash a corpus with a [`RandomProjLsh`] and
//!   summarise how points spread across buckets (load balance). Skewed
//!   distributions (a few huge buckets) hurt query latency; near-uniform ones
//!   are the goal.
//! * [`empirical_collision_rate`] — fraction of point pairs that land in the
//!   same bucket, the quantity LSH theory parameterises by distance.
//! * [`projection_isotropy`] — for sign-random-projection LSH the random
//!   hyperplane normals must be (approximately) isotropic, i.e. the Gram matrix
//!   of normalised rows ≈ identity. Returns the maximum absolute off-diagonal
//!   correlation; small values certify isotropy at high `dim`.
//! * [`minhash_jaccard_bias`] — empirical bias of the MinHash Jaccard estimator
//!   for a known ground-truth similarity, used to check unbiasedness even for
//!   small sketches.
//!
//! Everything is deterministic given a seeded [`LcgRng`].

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::lsh::minhash::MinHash;
use crate::lsh::random_proj::RandomProjLsh;

/// Summary statistics of how a corpus distributes across LSH buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketStats {
    /// Number of distinct (non-empty) buckets observed.
    pub n_buckets: usize,
    /// Total number of hashed points.
    pub n_points: usize,
    /// Largest bucket occupancy.
    pub max_load: usize,
    /// Smallest non-empty bucket occupancy.
    pub min_load: usize,
    /// Mean occupancy over non-empty buckets.
    pub mean_load: f32,
    /// Population standard deviation of occupancy over non-empty buckets.
    pub std_load: f32,
    /// Load-balance factor `max_load / mean_load` (`1.0` = perfectly uniform).
    pub imbalance: f32,
}

/// Hash every vector in a row-major `[n × dim]` corpus and summarise bucket
/// occupancy. The full packed hash (`Vec<u32>`) is used as the bucket key.
///
/// # Errors
/// - [`AnnError::EmptyInput`] if `n == 0`.
/// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim` or the LSH
///   dimensionality disagrees with `dim`.
pub fn bucket_size_distribution(
    lsh: &RandomProjLsh,
    data: &[f32],
    n: usize,
    dim: usize,
) -> AnnResult<BucketStats> {
    if n == 0 {
        return Err(AnnError::EmptyInput);
    }
    if dim != lsh.dim {
        return Err(AnnError::DimensionMismatch {
            expected: lsh.dim,
            got: dim,
        });
    }
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }

    use std::collections::HashMap;
    let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
    for i in 0..n {
        let key = lsh.hash(&data[i * dim..(i + 1) * dim]);
        *counts.entry(key).or_insert(0) += 1;
    }

    let loads: Vec<usize> = counts.values().copied().collect();
    let n_buckets = loads.len();
    let max_load = loads.iter().copied().max().unwrap_or(0);
    let min_load = loads.iter().copied().min().unwrap_or(0);
    let mean_load = if n_buckets > 0 {
        n as f32 / n_buckets as f32
    } else {
        0.0
    };
    let var = if n_buckets > 0 {
        loads
            .iter()
            .map(|&l| {
                let d = l as f32 - mean_load;
                d * d
            })
            .sum::<f32>()
            / n_buckets as f32
    } else {
        0.0
    };
    let std_load = var.sqrt();
    let imbalance = if mean_load > 0.0 {
        max_load as f32 / mean_load
    } else {
        0.0
    };

    Ok(BucketStats {
        n_buckets,
        n_points: n,
        max_load,
        min_load,
        mean_load,
        std_load,
        imbalance,
    })
}

/// Empirical single-table collision rate: fraction of the `n*(n-1)/2` unordered
/// point pairs whose full LSH codes are identical.
///
/// For small `n` this is exact (all pairs are enumerated). For large corpora a
/// caller should subsample first; this routine enumerates pairs directly.
///
/// # Errors
/// - [`AnnError::EmptyInput`] if `n < 2`.
/// - [`AnnError::DimensionMismatch`] on shape / dimensionality disagreement.
pub fn empirical_collision_rate(
    lsh: &RandomProjLsh,
    data: &[f32],
    n: usize,
    dim: usize,
) -> AnnResult<f32> {
    if n < 2 {
        return Err(AnnError::EmptyInput);
    }
    if dim != lsh.dim {
        return Err(AnnError::DimensionMismatch {
            expected: lsh.dim,
            got: dim,
        });
    }
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }
    let codes: Vec<Vec<u32>> = (0..n)
        .map(|i| lsh.hash(&data[i * dim..(i + 1) * dim]))
        .collect();
    let mut collisions = 0usize;
    let mut pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            pairs += 1;
            if codes[i] == codes[j] {
                collisions += 1;
            }
        }
    }
    Ok(collisions as f32 / pairs as f32)
}

/// Maximum absolute off-diagonal correlation of the (row-normalised) random
/// projection hyperplanes — a scalar isotropy measure.
///
/// Generates `n_hashes` Gaussian hyperplane normals in `dim` dimensions with a
/// seeded RNG (matching [`RandomProjLsh::new`]'s sampling), normalises each to
/// unit length, and returns `max_{i≠j} |⟨ŵ_i, ŵ_j⟩|`. For isotropic Gaussian
/// rows this concentrates near `0` as `dim` grows (≈ `O(1/√dim)`), so a small
/// value certifies isotropy for `dim ≥ 128`.
///
/// # Errors
/// - [`AnnError::EmptyInput`] if `n_hashes < 2`.
/// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
pub fn projection_isotropy(n_hashes: usize, dim: usize, rng: &mut LcgRng) -> AnnResult<f32> {
    if n_hashes < 2 {
        return Err(AnnError::EmptyInput);
    }
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim });
    }
    // Sample and unit-normalise each hyperplane normal.
    let mut rows = vec![0.0_f32; n_hashes * dim];
    rng.fill_normal(&mut rows);
    for h in 0..n_hashes {
        let r = &mut rows[h * dim..(h + 1) * dim];
        let norm: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for x in r.iter_mut() {
                *x *= inv;
            }
        }
    }
    let mut max_off = 0.0_f32;
    for i in 0..n_hashes {
        let ri = &rows[i * dim..(i + 1) * dim];
        for j in (i + 1)..n_hashes {
            let rj = &rows[j * dim..(j + 1) * dim];
            let dot: f32 = ri.iter().zip(rj.iter()).map(|(a, b)| a * b).sum();
            if dot.abs() > max_off {
                max_off = dot.abs();
            }
        }
    }
    Ok(max_off)
}

/// Empirical bias of the MinHash Jaccard estimator against a known true
/// similarity, averaged over `n_trials` independent hash-family draws.
///
/// Two sets are constructed with a controlled overlap so the exact Jaccard
/// similarity is `true_jaccard = inter / (2*size - inter)` where
/// `inter = round(size * overlap_frac)`. For each trial a fresh [`MinHash`] is
/// drawn from `rng`, both sets are signed, and the estimate recorded. Returns
/// `(mean_estimate − true_jaccard)`; an unbiased estimator yields a value near
/// `0` even for small `n_hashes`.
///
/// # Errors
/// - [`AnnError::EmptyInput`] if `n_hashes == 0`, `set_size == 0`, or
///   `n_trials == 0`.
/// - [`AnnError::Internal`] if `overlap_frac` is not in `[0, 1]`.
pub fn minhash_jaccard_bias(
    n_hashes: usize,
    set_size: usize,
    overlap_frac: f32,
    n_trials: usize,
    rng: &mut LcgRng,
) -> AnnResult<f32> {
    if n_hashes == 0 || set_size == 0 || n_trials == 0 {
        return Err(AnnError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&overlap_frac) {
        return Err(AnnError::Internal {
            msg: format!("overlap_frac={overlap_frac} not in [0, 1]"),
        });
    }

    let inter = ((set_size as f32) * overlap_frac).round() as usize;
    let inter = inter.min(set_size);
    // set_a = {0 .. set_size}; set_b shares the first `inter` ids and adds
    // `set_size - inter` disjoint ids past set_a's range.
    let set_a: Vec<u32> = (0..set_size as u32).collect();
    let mut set_b: Vec<u32> = (0..inter as u32).collect();
    let extra = set_size - inter;
    for e in 0..extra as u32 {
        set_b.push(set_size as u32 + e);
    }
    let union = 2 * set_size - inter;
    let true_jaccard = inter as f32 / union as f32;

    let mut acc = 0.0_f32;
    for _ in 0..n_trials {
        let mh = MinHash::new(n_hashes, rng);
        let sa = mh.hash(&set_a);
        let sb = mh.hash(&set_b);
        acc += MinHash::jaccard_estimate(&sa, &sb);
    }
    let mean_estimate = acc / n_trials as f32;
    Ok(mean_estimate - true_jaccard)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vecs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut v);
        v
    }

    // ── bucket distribution ────────────────────────────────────────────────

    #[test]
    fn bucket_stats_basic_shape() {
        let mut rng = LcgRng::new(1);
        let dim = 16;
        let lsh = RandomProjLsh::new(8, dim, &mut rng);
        let n = 256;
        let data = rand_vecs(n, dim, 2);
        let stats = bucket_size_distribution(&lsh, &data, n, dim).expect("stats");
        assert_eq!(stats.n_points, n);
        assert!(stats.n_buckets >= 1 && stats.n_buckets <= n);
        assert!(stats.max_load >= stats.min_load);
        assert!(stats.mean_load > 0.0);
        assert!(stats.imbalance >= 1.0 - 1e-4);
        // sum of loads equals n (mean * n_buckets ≈ n)
        let recovered = stats.mean_load * stats.n_buckets as f32;
        assert!((recovered - n as f32).abs() < 1.0);
    }

    #[test]
    fn bucket_stats_more_hashes_more_buckets() {
        // More hyperplanes → finer partition → (weakly) more buckets.
        let mut rng = LcgRng::new(3);
        let dim = 16;
        let n = 400;
        let data = rand_vecs(n, dim, 4);
        let lsh_coarse = RandomProjLsh::new(2, dim, &mut rng);
        let lsh_fine = RandomProjLsh::new(12, dim, &mut rng);
        let coarse = bucket_size_distribution(&lsh_coarse, &data, n, dim).expect("coarse");
        let fine = bucket_size_distribution(&lsh_fine, &data, n, dim).expect("fine");
        assert!(
            fine.n_buckets >= coarse.n_buckets,
            "fine={} coarse={}",
            fine.n_buckets,
            coarse.n_buckets
        );
        // A 2-hyperplane code can produce at most 4 buckets.
        assert!(coarse.n_buckets <= 4);
    }

    #[test]
    fn bucket_stats_validates() {
        let mut rng = LcgRng::new(5);
        let lsh = RandomProjLsh::new(4, 8, &mut rng);
        assert!(bucket_size_distribution(&lsh, &[], 0, 8).is_err());
        assert!(bucket_size_distribution(&lsh, &[0.0; 8], 1, 4).is_err()); // dim mismatch
        assert!(bucket_size_distribution(&lsh, &[0.0; 7], 1, 8).is_err()); // len mismatch
    }

    // ── collision rate ─────────────────────────────────────────────────────

    #[test]
    fn collision_rate_in_unit_interval() {
        let mut rng = LcgRng::new(6);
        let dim = 32;
        let lsh = RandomProjLsh::new(16, dim, &mut rng);
        let n = 100;
        let data = rand_vecs(n, dim, 7);
        let rate = empirical_collision_rate(&lsh, &data, n, dim).expect("rate");
        assert!((0.0..=1.0).contains(&rate), "rate={rate}");
    }

    #[test]
    fn collision_rate_monotone_in_bits() {
        // More bits → fewer collisions (codes get more specific).
        let mut rng = LcgRng::new(8);
        let dim = 32;
        let n = 120;
        let data = rand_vecs(n, dim, 9);
        let lsh_few = RandomProjLsh::new(2, dim, &mut rng);
        let lsh_many = RandomProjLsh::new(20, dim, &mut rng);
        let r_few = empirical_collision_rate(&lsh_few, &data, n, dim).expect("few");
        let r_many = empirical_collision_rate(&lsh_many, &data, n, dim).expect("many");
        assert!(r_few >= r_many, "few={r_few} many={r_many}");
    }

    #[test]
    fn collision_rate_identical_points_all_collide() {
        let mut rng = LcgRng::new(10);
        let dim = 8;
        let lsh = RandomProjLsh::new(10, dim, &mut rng);
        let one = rand_vecs(1, dim, 11);
        let mut data = Vec::new();
        for _ in 0..5 {
            data.extend_from_slice(&one);
        }
        let rate = empirical_collision_rate(&lsh, &data, 5, dim).expect("rate");
        assert!((rate - 1.0).abs() < 1e-6, "rate={rate}");
    }

    #[test]
    fn collision_rate_validates() {
        let mut rng = LcgRng::new(12);
        let lsh = RandomProjLsh::new(4, 8, &mut rng);
        assert!(empirical_collision_rate(&lsh, &[0.0; 8], 1, 8).is_err());
    }

    // ── isotropy ───────────────────────────────────────────────────────────

    #[test]
    fn isotropy_small_at_high_dim() {
        // d=128 → off-diagonal correlations of unit Gaussian rows ≈ O(1/√d).
        let mut rng = LcgRng::new(13);
        let max_off = projection_isotropy(64, 128, &mut rng).expect("isotropy");
        assert!(
            max_off < 0.5,
            "max off-diagonal correlation {max_off} too large"
        );
    }

    #[test]
    fn isotropy_improves_with_dimension() {
        let mut rng_a = LcgRng::new(14);
        let mut rng_b = LcgRng::new(14);
        let low = projection_isotropy(48, 16, &mut rng_a).expect("low");
        let high = projection_isotropy(48, 256, &mut rng_b).expect("high");
        // Higher dimension → tighter concentration around 0 (with margin).
        assert!(high <= low + 0.05, "low={low} high={high}");
    }

    #[test]
    fn isotropy_validates() {
        let mut rng = LcgRng::new(15);
        assert!(projection_isotropy(1, 64, &mut rng).is_err());
        assert!(projection_isotropy(8, 0, &mut rng).is_err());
    }

    // ── MinHash unbiasedness ───────────────────────────────────────────────

    #[test]
    fn minhash_bias_small_for_small_sketch() {
        // Even with only 64 hashes, averaged over many trials the estimator is
        // close to unbiased.
        let mut rng = LcgRng::new(16);
        let bias = minhash_jaccard_bias(64, 50, 0.5, 200, &mut rng).expect("bias");
        assert!(
            bias.abs() < 0.05,
            "bias={bias} too large for 64-hash sketch"
        );
    }

    #[test]
    fn minhash_bias_shrinks_with_more_hashes() {
        // Variance (and hence empirical |bias| over a fixed trial budget) falls
        // as the sketch grows; check the larger sketch is no worse with margin.
        let mut rng_a = LcgRng::new(17);
        let mut rng_b = LcgRng::new(17);
        let small = minhash_jaccard_bias(16, 60, 0.4, 300, &mut rng_a)
            .expect("small")
            .abs();
        let large = minhash_jaccard_bias(256, 60, 0.4, 300, &mut rng_b)
            .expect("large")
            .abs();
        assert!(large <= small + 0.03, "small={small} large={large}");
    }

    #[test]
    fn minhash_bias_disjoint_sets_zero_similarity() {
        let mut rng = LcgRng::new(18);
        // overlap 0 → true jaccard 0; estimator should be ~0 too.
        let bias = minhash_jaccard_bias(128, 40, 0.0, 100, &mut rng).expect("bias");
        // mean_estimate - 0 == mean_estimate; must be small and non-negative-ish.
        assert!(bias.abs() < 0.05, "bias={bias}");
    }

    #[test]
    fn minhash_bias_validates() {
        let mut rng = LcgRng::new(19);
        assert!(minhash_jaccard_bias(0, 10, 0.5, 10, &mut rng).is_err());
        assert!(minhash_jaccard_bias(8, 0, 0.5, 10, &mut rng).is_err());
        assert!(minhash_jaccard_bias(8, 10, 0.5, 0, &mut rng).is_err());
        assert!(minhash_jaccard_bias(8, 10, 1.5, 10, &mut rng).is_err());
    }
}
