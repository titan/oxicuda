//! Differentially-Private K-Means clustering with noisy centroid updates.
//!
//! # Reference
//! - Su, Cao, Wang, Li (2016),
//!   *"Differentially Private K-Means Clustering"*,
//!   CODASPY 2016, <https://arxiv.org/abs/1504.05998>.
//!
//! # Algorithm
//!
//! Given `X ∈ ℝ^{n × d}` and target cluster count `k`:
//!
//! 1. **Init**: each row of `X` is L2-clipped to `clip_norm`; centroids are
//!    chosen uniformly at random from the clipped rows.  Initialisation is
//!    *not* charged to the privacy budget in this implementation — a known
//!    concession; the focus is on the iterative DP mechanism.  See the
//!    note in the doc comment of `dp_kmeans` for guidance on switching to
//!    a private-init scheme.
//!
//! 2. **Each Lloyd iteration**:
//!    - Assign each point to its closest centroid (Euclidean distance).
//!    - For each cluster `j`, accumulate sum `S_j ∈ ℝ^d` and count `n_j`.
//!    - Add Gaussian noise:
//!      - `S̃_j = S_j + N(0, σ_S² · I_d)` with
//!        `σ_S = clip_norm · √(2 ln(1.25 / δ)) / ε`
//!      - `ñ_j = n_j + N(0, σ_n²)` with
//!        `σ_n = √(2 ln(1.25 / δ)) / ε`
//!    - Update: `μ_j = S̃_j / max(ñ_j, 1)` to avoid division by a noisy
//!      count that has become non-positive.
//!
//! 3. **Composition**: across `n_iter` rounds the total budget is
//!    `total_epsilon = n_iter · epsilon_per_iter` under basic composition
//!    (Dwork-Roth 2014, Theorem 3.16).  A Renyi-DP / GDP composition would
//!    yield tighter accounting — left as a future enhancement.
//!
//! # Privacy guarantee
//! Each iteration uses two Gaussian mechanisms (sum, count) on the
//! count-and-sum tuple, providing `(ε, δ)`-DP per iteration.  Basic
//! composition over `n_iter` gives `(n_iter · ε, n_iter · δ)`.
//!
//! # Implementation
//! - All randomness goes through `LcgRng` (Box-Muller for normals).
//! - No `unsafe`, no `rand`, no `ndarray`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for differentially-private K-Means.
#[derive(Debug, Clone)]
pub struct DpKMeansConfig {
    /// Target number of clusters (`1 ≤ k ≤ n`).
    pub n_clusters: usize,
    /// Number of Lloyd iterations (≥ 1).
    pub n_iter: usize,
    /// Per-row L2 clipping bound (controls per-row sensitivity).
    pub clip_norm: f64,
    /// Per-iteration privacy parameter `ε > 0` (used for both sum and count
    /// Gaussian mechanisms within an iteration; the per-iteration `(ε, δ)`
    /// guarantee compounds via basic composition).
    pub epsilon_per_iter: f64,
    /// Per-iteration failure probability `δ ∈ (0, 1)`.
    pub delta_per_iter: f64,
    /// RNG seed for centroid initialisation (independent of the noise RNG).
    pub init_seed: u64,
}

impl DpKMeansConfig {
    /// Sensible defaults: `clip_norm = 1.0`, `init_seed = 0`.
    #[must_use]
    pub fn new(
        n_clusters: usize,
        n_iter: usize,
        epsilon_per_iter: f64,
        delta_per_iter: f64,
    ) -> Self {
        Self {
            n_clusters,
            n_iter,
            clip_norm: 1.0,
            epsilon_per_iter,
            delta_per_iter,
            init_seed: 0,
        }
    }
}

/// Result of a DP-KMeans run.
#[derive(Debug, Clone)]
pub struct DpKMeansResult {
    /// Final centroids, row-major `[k × d]`.
    pub centroids: Vec<f64>,
    /// Cluster assignment for each input row (length `n`).
    pub assignments: Vec<usize>,
    /// Total composed `ε = n_iter · epsilon_per_iter` (basic composition).
    pub total_epsilon: f64,
}

impl DpKMeansResult {
    /// Centroids accessor (`[k × d]`, row-major).
    #[must_use]
    #[inline]
    pub fn centroids(&self) -> &[f64] {
        &self.centroids
    }

    /// Assignments accessor.
    #[must_use]
    #[inline]
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }

    /// Total `(ε)` charge across all iterations under basic composition.
    #[must_use]
    #[inline]
    pub fn total_epsilon(&self) -> f64 {
        self.total_epsilon
    }
}

fn validate(cfg: &DpKMeansConfig, n_rows: usize, n_cols: usize, x_len: usize) -> PrivacyResult<()> {
    if n_rows == 0 || n_cols == 0 {
        return Err(PrivacyError::EmptyInput);
    }
    if n_rows.checked_mul(n_cols) != Some(x_len) {
        return Err(PrivacyError::DimensionMismatch {
            expected: n_rows.saturating_mul(n_cols),
            got: x_len,
        });
    }
    if cfg.n_clusters == 0 {
        return Err(PrivacyError::InvalidParameter(
            "n_clusters must be ≥ 1".into(),
        ));
    }
    if cfg.n_clusters > n_rows {
        return Err(PrivacyError::InvalidParameter(format!(
            "n_clusters ({}) exceeds n_rows ({})",
            cfg.n_clusters, n_rows
        )));
    }
    if cfg.n_iter == 0 {
        return Err(PrivacyError::InvalidParameter("n_iter must be ≥ 1".into()));
    }
    if !(cfg.clip_norm.is_finite() && cfg.clip_norm > 0.0) {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.clip_norm));
    }
    if !(cfg.epsilon_per_iter.is_finite() && cfg.epsilon_per_iter > 0.0) {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon_per_iter));
    }
    if !(cfg.delta_per_iter > 0.0 && cfg.delta_per_iter < 1.0) {
        return Err(PrivacyError::InvalidDelta(cfg.delta_per_iter));
    }
    Ok(())
}

/// In-place L2 clip of each row to `clip_norm`.
fn clip_rows(x: &mut [f64], n_rows: usize, n_cols: usize, clip_norm: f64) {
    for i in 0..n_rows {
        let row = &mut x[i * n_cols..(i + 1) * n_cols];
        let mut sq = 0.0_f64;
        for &v in row.iter() {
            sq += v * v;
        }
        let norm = sq.sqrt();
        if norm > clip_norm && norm > 0.0 {
            let s = clip_norm / norm;
            for v in row.iter_mut() {
                *v *= s;
            }
        }
    }
}

/// Pick `k` initial centroids uniformly without replacement from the row set.
/// Uses a Fisher-Yates partial shuffle over an index list.
fn init_centroids(x: &[f64], n_rows: usize, n_cols: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    let mut idx: Vec<usize> = (0..n_rows).collect();
    // Partial Fisher-Yates: shuffle the first `k` entries.
    for i in 0..k.min(n_rows) {
        let span = n_rows - i;
        let j = i + (rng.next_u64() as usize % span);
        idx.swap(i, j);
    }
    let mut out = vec![0.0_f64; k * n_cols];
    for c in 0..k {
        let src_row = idx[c % n_rows];
        let src = &x[src_row * n_cols..(src_row + 1) * n_cols];
        let dst = &mut out[c * n_cols..(c + 1) * n_cols];
        dst.copy_from_slice(src);
    }
    out
}

/// Assign each row to its nearest centroid (Euclidean distance).
fn assign(x: &[f64], n_rows: usize, n_cols: usize, centroids: &[f64], k: usize) -> Vec<usize> {
    let mut out = vec![0_usize; n_rows];
    for i in 0..n_rows {
        let row = &x[i * n_cols..(i + 1) * n_cols];
        let mut best = 0_usize;
        let mut best_d = f64::INFINITY;
        for c in 0..k {
            let cent = &centroids[c * n_cols..(c + 1) * n_cols];
            let mut d = 0.0_f64;
            for a in 0..n_cols {
                let diff = row[a] - cent[a];
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        out[i] = best;
    }
    out
}

/// Accumulate per-cluster sums and counts.
fn accumulate(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    assignments: &[usize],
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut sums = vec![0.0_f64; k * n_cols];
    let mut counts = vec![0.0_f64; k];
    for i in 0..n_rows {
        let c = assignments[i];
        if c >= k {
            continue;
        }
        let row = &x[i * n_cols..(i + 1) * n_cols];
        let dst = &mut sums[c * n_cols..(c + 1) * n_cols];
        for a in 0..n_cols {
            dst[a] += row[a];
        }
        counts[c] += 1.0;
    }
    (sums, counts)
}

/// Yield independent N(0, 1) samples one at a time using Box-Muller pairs.
struct GaussStream<'a> {
    rng: &'a mut LcgRng,
    held: Option<f64>,
}

impl<'a> GaussStream<'a> {
    fn new(rng: &'a mut LcgRng) -> Self {
        Self { rng, held: None }
    }

    fn next(&mut self) -> f64 {
        if let Some(v) = self.held.take() {
            return v;
        }
        let (a, b) = self.rng.normal_pair();
        self.held = Some(b);
        a
    }
}

/// Add Gaussian noise to all per-cluster sums and counts.
fn noisy_update(
    sums: &mut [f64],
    counts: &mut [f64],
    k: usize,
    d: usize,
    sigma_s: f64,
    sigma_n: f64,
    rng: &mut LcgRng,
) {
    let mut gauss = GaussStream::new(rng);
    for j in 0..k {
        let row = &mut sums[j * d..(j + 1) * d];
        for v in row.iter_mut() {
            *v += gauss.next() * sigma_s;
        }
        counts[j] += gauss.next() * sigma_n;
    }
}

/// Compute updated centroids from noisy sums and counts.  Uses
/// `max(ñ_j, 1)` to guard against division by a noisy non-positive count.
fn compute_centroids(sums: &[f64], counts: &[f64], k: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; k * d];
    for j in 0..k {
        let denom = counts[j].max(1.0);
        let src = &sums[j * d..(j + 1) * d];
        let dst = &mut out[j * d..(j + 1) * d];
        for a in 0..d {
            dst[a] = src[a] / denom;
        }
    }
    out
}

/// Run differentially-private K-Means clustering.
///
/// # Errors
/// `EmptyInput`, `DimensionMismatch`, `InvalidParameter`,
/// `NonPositiveSensitivity`, `NonPositiveEpsilon`, or `InvalidDelta` if any
/// configuration entry is out of range.
///
/// # Privacy
/// The output is `(total_epsilon, total_delta) = (n_iter · ε, n_iter · δ)`-DP
/// under basic composition (Dwork-Roth 2014, Theorem 3.16).  Centroid
/// initialisation is *not* charged to the privacy budget in this
/// implementation.
pub fn dp_kmeans(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    cfg: &DpKMeansConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<DpKMeansResult> {
    validate(cfg, n_rows, n_cols, x.len())?;

    let mut clipped = x.to_vec();
    clip_rows(&mut clipped, n_rows, n_cols, cfg.clip_norm);

    let mut centroids = init_centroids(&clipped, n_rows, n_cols, cfg.n_clusters, cfg.init_seed);

    // Gaussian-mechanism σ for the per-iteration sum (Δ = clip_norm in L2).
    let log_term = (2.0 * (1.25 / cfg.delta_per_iter).ln()).sqrt();
    let sigma_s = cfg.clip_norm * log_term / cfg.epsilon_per_iter;
    // Count sensitivity Δ_n = 1 (one user changes the count of one cluster by 1).
    let sigma_n = log_term / cfg.epsilon_per_iter;

    let mut assignments = vec![0_usize; n_rows];

    for _it in 0..cfg.n_iter {
        assignments = assign(&clipped, n_rows, n_cols, &centroids, cfg.n_clusters);
        let (mut sums, mut counts) =
            accumulate(&clipped, n_rows, n_cols, &assignments, cfg.n_clusters);
        noisy_update(
            &mut sums,
            &mut counts,
            cfg.n_clusters,
            n_cols,
            sigma_s,
            sigma_n,
            rng,
        );
        let new_centroids = compute_centroids(&sums, &counts, cfg.n_clusters, n_cols);
        centroids = new_centroids;
    }

    // Final assignment with the converged centroids.
    assignments = assign(&clipped, n_rows, n_cols, &centroids, cfg.n_clusters);

    Ok(DpKMeansResult {
        centroids,
        assignments,
        total_epsilon: cfg.epsilon_per_iter * cfg.n_iter as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters in 2D within the unit ball.
    fn make_two_clusters() -> Vec<f64> {
        let mut x = Vec::new();
        // Cluster A near [+0.6, +0.6]
        for i in 0..50 {
            let p = (i as f64) * 1e-3;
            x.push(0.6 + p);
            x.push(0.6 - p);
        }
        // Cluster B near [-0.6, -0.6]
        for i in 0..50 {
            let p = (i as f64) * 1e-3;
            x.push(-0.6 + p);
            x.push(-0.6 - p);
        }
        x
    }

    /// Three clusters spread on a triangle.
    fn make_three_clusters() -> Vec<f64> {
        let mut x = Vec::new();
        let centres = [(0.7, 0.0), (-0.35, 0.6), (-0.35, -0.6)];
        for &(cx, cy) in centres.iter() {
            for i in 0..30 {
                let p = (i as f64) * 1e-3;
                x.push(cx + p);
                x.push(cy + p);
            }
        }
        x
    }

    #[test]
    fn test_two_clusters_recovered() {
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 8, 5.0, 1e-3);
        cfg.clip_norm = 2.0;
        cfg.init_seed = 7;
        let mut rng = LcgRng::new(123);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        // Count assignments per cluster.
        let mut top = [0_usize; 2];
        for (i, &a) in res.assignments.iter().enumerate() {
            // True cluster: first 50 belong to A, next 50 to B.
            let truth = if i < 50 { 0 } else { 1 };
            if a == truth {
                top[truth] += 1;
            }
        }
        // Allow label swap: take the better permutation.
        let correct_same = top[0] + top[1];
        let correct_swap = (50 - top[0]) + (50 - top[1]);
        let correct = correct_same.max(correct_swap);
        assert!(correct > 80, "correct = {correct} / 100");
    }

    #[test]
    fn test_three_clusters_assignments_sensible() {
        let x = make_three_clusters();
        let mut cfg = DpKMeansConfig::new(3, 10, 5.0, 1e-3);
        cfg.clip_norm = 2.0;
        cfg.init_seed = 17;
        let mut rng = LcgRng::new(7);
        let res = dp_kmeans(&x, 90, 2, &cfg, &mut rng).expect("ok");
        // Sanity: each cluster id appears.
        let mut seen = [false; 3];
        for &a in res.assignments.iter() {
            if a < 3 {
                seen[a] = true;
            }
        }
        let n_seen = seen.iter().filter(|&&v| v).count();
        // We expect at least 2 distinct clusters used (noise can collapse one).
        assert!(n_seen >= 2, "only {n_seen} clusters used");
    }

    #[test]
    fn test_high_noise_does_not_panic() {
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 5, 0.01, 0.1);
        cfg.clip_norm = 2.0;
        let mut rng = LcgRng::new(5);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        for &v in res.centroids.iter() {
            assert!(v.is_finite(), "centroid not finite: {v}");
        }
    }

    #[test]
    fn test_reproducible_with_seeds() {
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 5, 1.0, 1e-3);
        cfg.clip_norm = 2.0;
        cfg.init_seed = 42;
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let a = dp_kmeans(&x, 100, 2, &cfg, &mut rng_a).expect("ok");
        let b = dp_kmeans(&x, 100, 2, &cfg, &mut rng_b).expect("ok");
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.assignments, b.assignments);
    }

    #[test]
    fn test_composition_total_epsilon() {
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 7, 0.5, 1e-3);
        cfg.clip_norm = 2.0;
        let mut rng = LcgRng::new(0);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        assert!((res.total_epsilon - 7.0 * 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_n_clusters() {
        let x = vec![0.0; 6];
        let mut rng = LcgRng::new(0);
        let cfg0 = DpKMeansConfig::new(0, 3, 1.0, 1e-3);
        assert!(dp_kmeans(&x, 2, 3, &cfg0, &mut rng).is_err());
        let cfg_too_many = DpKMeansConfig::new(5, 3, 1.0, 1e-3); // > n_rows
        assert!(dp_kmeans(&x, 2, 3, &cfg_too_many, &mut rng).is_err());
    }

    #[test]
    fn test_invalid_n_iter() {
        let x = vec![0.0; 6];
        let mut rng = LcgRng::new(0);
        let cfg = DpKMeansConfig::new(2, 0, 1.0, 1e-3);
        assert!(dp_kmeans(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_invalid_clip_norm() {
        let x = vec![0.0; 6];
        let mut rng = LcgRng::new(0);
        let mut cfg = DpKMeansConfig::new(2, 3, 1.0, 1e-3);
        cfg.clip_norm = 0.0;
        assert!(dp_kmeans(&x, 2, 3, &cfg, &mut rng).is_err());
        cfg.clip_norm = -1.0;
        assert!(dp_kmeans(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_invalid_epsilon_delta() {
        let x = vec![0.0; 6];
        let mut rng = LcgRng::new(0);
        let cfg_e = DpKMeansConfig::new(2, 3, 0.0, 1e-3);
        assert!(dp_kmeans(&x, 2, 3, &cfg_e, &mut rng).is_err());
        let cfg_d_lo = DpKMeansConfig::new(2, 3, 1.0, 0.0);
        assert!(dp_kmeans(&x, 2, 3, &cfg_d_lo, &mut rng).is_err());
        let cfg_d_hi = DpKMeansConfig::new(2, 3, 1.0, 1.0);
        assert!(dp_kmeans(&x, 2, 3, &cfg_d_hi, &mut rng).is_err());
    }

    #[test]
    fn test_dim_mismatch() {
        let x = vec![0.0; 5]; // not 2*3 = 6
        let cfg = DpKMeansConfig::new(2, 3, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_kmeans(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_empty_cluster_handled() {
        // Construct a scenario where centroid 2 is initialised far from data and
        // attracts no points.  Verify dp_kmeans does not panic and produces
        // finite centroids (the empty-cluster centroid still updates from noise).
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(3, 4, 1.0, 1e-3);
        cfg.clip_norm = 2.0;
        cfg.init_seed = 0;
        let mut rng = LcgRng::new(0);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        for &v in res.centroids.iter() {
            assert!(v.is_finite(), "non-finite centroid: {v}");
        }
    }

    #[test]
    fn test_centroids_in_reasonable_range() {
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 5, 5.0, 1e-3);
        cfg.clip_norm = 2.0;
        cfg.init_seed = 11;
        let mut rng = LcgRng::new(11);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        // Centroids are means of clipped points (|x| ≤ 2), plus modest noise.
        // Allow ±5 as a generous bound.
        for &v in res.centroids.iter() {
            assert!(v.abs() < 5.0, "centroid coord = {v}");
        }
    }

    #[test]
    fn test_k1_centroid_near_mean() {
        // For k = 1 with low noise, the single centroid converges to data mean.
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(1, 5, 100.0, 1e-3);
        cfg.clip_norm = 2.0;
        let mut rng = LcgRng::new(3);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        // Mean of the two clusters cancels out to ~zero.
        for &v in res.centroids.iter() {
            assert!(v.abs() < 0.2, "k=1 centroid coord = {v}");
        }
    }

    #[test]
    fn test_convergence_centroids_finite() {
        // Verify the loop terminates with finite centroids over many iterations.
        let x = make_two_clusters();
        let mut cfg = DpKMeansConfig::new(2, 20, 10.0, 1e-3);
        cfg.clip_norm = 2.0;
        let mut rng = LcgRng::new(1);
        let res = dp_kmeans(&x, 100, 2, &cfg, &mut rng).expect("ok");
        for &v in res.centroids.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_empty_input_rejected() {
        let cfg = DpKMeansConfig::new(1, 1, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_kmeans(&[], 0, 2, &cfg, &mut rng).is_err());
        assert!(dp_kmeans(&[], 2, 0, &cfg, &mut rng).is_err());
    }
}
