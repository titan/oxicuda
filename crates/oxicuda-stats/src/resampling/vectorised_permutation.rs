//! Vectorised permutation test: evaluate multiple test statistics simultaneously
//! reusing the same random permutations.
//!
//! Running separate permutation tests for k statistics costs k × n_perms × n
//! operations.  By computing all k statistics on every permuted dataset we pay
//! only n_perms × n + k × n_perms, which is much cheaper when n ≫ k.
//!
//! # Reference
//! Westfall, P. H. and Young, S. S. (1993). *Resampling-Based Multiple
//! Testing*.  Wiley, New York. (§2.2 – simultaneous permutation tests)

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Type alias for a boxed two-sample test statistic function.
pub type TwoSampleStatFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;

// ─── public types ─────────────────────────────────────────────────────────────

/// Configuration for the vectorised permutation test.
#[derive(Debug, Clone, Copy)]
pub struct VecPermConfig {
    /// Number of random permutations to generate.
    pub n_permutations: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for VecPermConfig {
    fn default() -> Self {
        Self {
            n_permutations: 1_000,
            seed: 0,
        }
    }
}

/// Result of a vectorised permutation test.
#[derive(Debug, Clone)]
pub struct VecPermResult {
    /// Observed test statistics (one per function supplied).
    pub statistics: Vec<f64>,
    /// Permutation p-values (two-sided: proportion of |perm stat| ≥ |observed stat|).
    pub p_values: Vec<f64>,
    /// Number of permutations actually performed.
    pub n_perms: usize,
}

// ─── permutation matrix ───────────────────────────────────────────────────────

/// Generate `n_perms` random permutations of indices `0..n`.
///
/// For small n (≤ 8), produces *exact* enumeration of all n! permutations
/// (up to `n_perms`).  For larger n, generates random Fisher-Yates shuffles.
///
/// Returns a `Vec<Vec<usize>>` of shape `[n_perms][n]`.
pub fn permutation_matrix(n: usize, n_perms: usize, rng: &mut LcgRng) -> Vec<Vec<usize>> {
    if n == 0 || n_perms == 0 {
        return Vec::new();
    }

    // Exact enumeration for tiny n (avoids duplicates when n! ≤ n_perms).
    let n_factorial: Option<usize> = (1..=n).try_fold(1usize, |acc, i| acc.checked_mul(i));
    let use_exact = n_factorial.is_some_and(|nf| nf <= n_perms && n <= 8);

    if use_exact {
        let all = all_permutations(n);
        // Cycle through all permutations until n_perms are filled.
        all.into_iter().cycle().take(n_perms).collect()
    } else {
        let base: Vec<usize> = (0..n).collect();
        let mut result = Vec::with_capacity(n_perms);
        for _ in 0..n_perms {
            let mut perm = base.clone();
            fisher_yates_shuffle(&mut perm, rng);
            result.push(perm);
        }
        result
    }
}

/// Fisher-Yates in-place shuffle.
#[inline]
fn fisher_yates_shuffle(perm: &mut [usize], rng: &mut LcgRng) {
    let n = perm.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        perm.swap(i, j);
    }
}

/// Generate all permutations of `0..n` via Heap's algorithm (n ≤ 8 only).
fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut perm: Vec<usize> = (0..n).collect();
    heap_permute(n, &mut perm, &mut out);
    out
}

fn heap_permute(k: usize, perm: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if k == 1 {
        out.push(perm.clone());
        return;
    }
    for i in 0..k {
        heap_permute(k - 1, perm, out);
        if k % 2 == 0 {
            perm.swap(i, k - 1);
        } else {
            perm.swap(0, k - 1);
        }
    }
}

// ─── batch statistics helper ──────────────────────────────────────────────────

/// Compute four common two-sample statistics on `(x, y)` in a single pass.
///
/// Returns `[mean_diff, variance_ratio, ks_stat, rank_sum_stat]`:
/// - `mean_diff` = mean(x) − mean(y)
/// - `variance_ratio` = var(x) / var(y)  (Inf if var(y) = 0)
/// - `ks_stat` = max |F_x(t) − F_y(t)|  (Kolmogorov-Smirnov statistic)
/// - `rank_sum_stat` = (Wilcoxon W − expected) / std (standardised rank sum)
///
/// # Panics
/// Does not panic; returns `[NaN, ...]` if inputs are empty.
pub fn batch_two_sample_stats(x: &[f64], y: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 {
        return vec![f64::NAN; 4];
    }

    // Mean difference.
    let mx: f64 = x.iter().sum::<f64>() / nx as f64;
    let my: f64 = y.iter().sum::<f64>() / ny as f64;
    let mean_diff = mx - my;

    // Variance ratio.
    let vx = sample_var(x, mx);
    let vy = sample_var(y, my);
    let var_ratio = if vy.abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        vx / vy
    };

    // Kolmogorov-Smirnov statistic via sorted merge.
    let ks = ks_stat(x, nx, y, ny);

    // Standardised Wilcoxon rank-sum statistic.
    let rank_sum = wilcoxon_rank_sum_standardised(x, nx, y, ny);

    vec![mean_diff, var_ratio, ks, rank_sum]
}

/// Sample variance (Bessel-corrected).
#[inline]
fn sample_var(v: &[f64], mean: f64) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Kolmogorov-Smirnov statistic D = max |F_x(t) − F_y(t)|.
fn ks_stat(x: &[f64], nx: usize, y: &[f64], ny: usize) -> f64 {
    let mut xs: Vec<f64> = x.to_vec();
    let mut ys: Vec<f64> = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut d = 0.0_f64;
    let mut xi = 0usize;
    let mut yi = 0usize;
    let mut fx = 0.0_f64;
    let mut fy = 0.0_f64;
    while xi < nx || yi < ny {
        let take_x = yi >= ny || (xi < nx && xs[xi] <= ys[yi]);
        if take_x {
            fx = (xi + 1) as f64 / nx as f64;
            xi += 1;
        } else {
            fy = (yi + 1) as f64 / ny as f64;
            yi += 1;
        }
        d = d.max((fx - fy).abs());
    }
    d
}

/// Standardised Wilcoxon rank-sum (Mann-Whitney U) statistic.
///
/// Returns (W − E[W]) / sqrt(Var[W]) where W is the sum of ranks for group x.
fn wilcoxon_rank_sum_standardised(x: &[f64], nx: usize, y: &[f64], ny: usize) -> f64 {
    let n = nx + ny;
    // Merge and rank.
    let mut combined: Vec<(f64, usize)> = x
        .iter()
        .map(|&v| (v, 0usize))
        .chain(y.iter().map(|&v| (v, 1usize)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Average ranks for ties.
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average rank
        for slot in ranks.iter_mut().take(j).skip(i) {
            *slot = avg_rank;
        }
        i = j;
    }

    let w: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, grp), _)| *grp == 0)
        .map(|(_, &r)| r)
        .sum();

    let e_w = nx as f64 * (n as f64 + 1.0) / 2.0;
    let var_w = nx as f64 * ny as f64 * (n as f64 + 1.0) / 12.0;
    if var_w.abs() < f64::EPSILON {
        return 0.0;
    }
    (w - e_w) / var_w.sqrt()
}

// ─── vectorised permutation test ─────────────────────────────────────────────

/// Vectorised permutation test: applies multiple test statistics to the same
/// set of random permutations simultaneously.
///
/// # Arguments
/// * `x` — first group (length `nx`).
/// * `y` — second group (length `ny`).
/// * `nx` — size of group x (must equal `x.len()`).
/// * `ny` — size of group y (must equal `y.len()`).
/// * `statistics` — slice of boxed functions `(&[f64], &[f64]) -> f64`.
/// * `cfg` — permutation configuration.
///
/// # Algorithm
/// 1. Compute observed statistics on `(x, y)`.
/// 2. Pre-generate `cfg.n_permutations` Fisher-Yates shuffles of `0..nx+ny`.
/// 3. For each permutation, split into groups by the shuffled indices.
/// 4. For each statistic s, count permutations where |perm_s| ≥ |obs_s|.
/// 5. p-value = (count + 1) / (n_perms + 1)  (adds 1 for the observed value).
///
/// # Errors
/// Returns an error if inputs are empty, sizes disagree, or `n_permutations = 0`.
pub fn vectorised_permutation_test(
    x: &[f64],
    y: &[f64],
    nx: usize,
    ny: usize,
    statistics: &[TwoSampleStatFn],
    cfg: &VecPermConfig,
) -> StatsResult<VecPermResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if x.len() != nx || y.len() != ny {
        return Err(StatsError::DimensionMismatch { a: x.len(), b: nx });
    }
    if statistics.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "statistics".into(),
            reason: "must supply at least one test statistic".into(),
        });
    }
    if cfg.n_permutations == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_permutations".into(),
            reason: "must be > 0".into(),
        });
    }

    let n = nx + ny;
    let k = statistics.len();

    // Observed statistics.
    let obs: Vec<f64> = statistics.iter().map(|f| f(x, y)).collect();

    // Combined data for permutation.
    let combined: Vec<f64> = x.iter().chain(y.iter()).copied().collect();

    let mut rng = LcgRng::new(cfg.seed);
    // Pre-generate all permutation index vectors.
    let perm_mat = permutation_matrix(n, cfg.n_permutations, &mut rng);
    let n_perms = perm_mat.len();

    // Count permutations with |stat| ≥ |observed stat| for each statistic.
    let mut counts = vec![0usize; k];

    let mut perm_x = vec![0.0_f64; nx];
    let mut perm_y = vec![0.0_f64; ny];

    for perm in &perm_mat {
        // Split combined into two groups according to permutation.
        for i in 0..nx {
            perm_x[i] = combined[perm[i]];
        }
        for i in 0..ny {
            perm_y[i] = combined[perm[nx + i]];
        }
        for (s_idx, stat_fn) in statistics.iter().enumerate() {
            let perm_val = stat_fn(&perm_x, &perm_y);
            if perm_val.abs() >= obs[s_idx].abs() - 1e-12 {
                counts[s_idx] += 1;
            }
        }
    }

    // Monte Carlo p-value with +1 continuity correction.
    let p_values: Vec<f64> = counts
        .iter()
        .map(|&c| (c + 1) as f64 / (n_perms + 1) as f64)
        .collect();

    Ok(VecPermResult {
        statistics: obs,
        p_values,
        n_perms,
    })
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn mean_diff_fn() -> TwoSampleStatFn {
        Box::new(|x: &[f64], y: &[f64]| {
            let mx = x.iter().sum::<f64>() / x.len() as f64;
            let my = y.iter().sum::<f64>() / y.len() as f64;
            mx - my
        })
    }

    fn abs_mean_diff_fn() -> TwoSampleStatFn {
        Box::new(|x: &[f64], y: &[f64]| {
            let mx = x.iter().sum::<f64>() / x.len() as f64;
            let my = y.iter().sum::<f64>() / y.len() as f64;
            (mx - my).abs()
        })
    }

    // ── 1. permutation_matrix shape ──────────────────────────────────────────
    #[test]
    fn perm_matrix_shape() {
        let mut rng = LcgRng::new(1);
        let mat = permutation_matrix(5, 20, &mut rng);
        assert_eq!(mat.len(), 20);
        for row in &mat {
            assert_eq!(row.len(), 5);
        }
    }

    // ── 2. permutation_matrix: each row is a valid permutation ───────────────
    #[test]
    fn perm_matrix_valid_permutations() {
        let mut rng = LcgRng::new(2);
        let mat = permutation_matrix(6, 50, &mut rng);
        for row in &mat {
            let mut sorted = row.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
        }
    }

    // ── 3. exact enumeration for small n ─────────────────────────────────────
    #[test]
    fn perm_matrix_exact_for_small_n() {
        let mut rng = LcgRng::new(3);
        // n=3: 6 permutations; request 6 → exact
        let mat = permutation_matrix(3, 6, &mut rng);
        assert_eq!(mat.len(), 6);
        // All 6 should be distinct (exact enumeration).
        let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
        for row in &mat {
            seen.insert(row.clone());
        }
        assert_eq!(
            seen.len(),
            6,
            "Expected 6 distinct permutations of 3 elements"
        );
    }

    // ── 4. batch_two_sample_stats returns 4 elements ─────────────────────────
    #[test]
    fn batch_stats_length() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let stats = batch_two_sample_stats(&x, &y);
        assert_eq!(stats.len(), 4);
    }

    // ── 5. batch_two_sample_stats: mean_diff correct ─────────────────────────
    #[test]
    fn batch_stats_mean_diff() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0]; // my=5, mx=2, diff=-3
        let stats = batch_two_sample_stats(&x, &y);
        assert!((stats[0] - (-3.0)).abs() < 1e-10, "mean_diff={}", stats[0]);
    }

    // ── 6. batch_two_sample_stats: KS stat in [0,1] ──────────────────────────
    #[test]
    fn batch_stats_ks_range() {
        let x = [1.0, 3.0, 5.0, 7.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let stats = batch_two_sample_stats(&x, &y);
        let ks = stats[2];
        assert!((0.0..=1.0).contains(&ks), "KS stat={ks} out of [0,1]");
    }

    // ── 7. vectorised test: p-value in [0,1] ─────────────────────────────────
    #[test]
    fn vec_perm_p_value_range() {
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let y: Vec<f64> = (11..=20).map(|v| v as f64).collect();
        let cfg = VecPermConfig {
            n_permutations: 200,
            seed: 5,
        };
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn()];
        let result = vectorised_permutation_test(&x, &y, 10, 10, &stats, &cfg).expect("ok");
        assert!(result.p_values[0] >= 0.0 && result.p_values[0] <= 1.0);
    }

    // ── 8. vectorised test: clear shift → small p ────────────────────────────
    #[test]
    fn vec_perm_clear_shift_small_p() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![100.0, 101.0, 102.0];
        let cfg = VecPermConfig {
            n_permutations: 500,
            seed: 7,
        };
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn()];
        let result = vectorised_permutation_test(&x, &y, 3, 3, &stats, &cfg).expect("ok");
        assert!(
            result.p_values[0] < 0.1,
            "p={} expected small",
            result.p_values[0]
        );
    }

    // ── 9. vectorised test: no shift → large p ───────────────────────────────
    #[test]
    fn vec_perm_no_shift_large_p() {
        // Identical groups → p-value should be large (non-significant)
        let x = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let y = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let cfg = VecPermConfig {
            n_permutations: 200,
            seed: 9,
        };
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn()];
        let result = vectorised_permutation_test(&x, &y, 5, 5, &stats, &cfg).expect("ok");
        // Mean diff is 0 for any permutation, so all |perm| ≥ |0|, p = 1.0
        assert!(result.p_values[0] > 0.5, "p={}", result.p_values[0]);
    }

    // ── 10. vectorised test: multiple statistics computed simultaneously ───────
    #[test]
    fn vec_perm_multiple_statistics() {
        let x: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let y: Vec<f64> = (9..=16).map(|v| v as f64).collect();
        let cfg = VecPermConfig {
            n_permutations: 300,
            seed: 13,
        };
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn(), abs_mean_diff_fn()];
        let result = vectorised_permutation_test(&x, &y, 8, 8, &stats, &cfg).expect("ok");
        assert_eq!(result.statistics.len(), 2);
        assert_eq!(result.p_values.len(), 2);
        // Both statistics measure the same shift, so p-values should be similar
        let p0 = result.p_values[0];
        let p1 = result.p_values[1];
        assert!(
            (p0 - p1).abs() < 0.3,
            "p0={p0} p1={p1} should be similar for equivalent statistics"
        );
    }

    // ── 11. vectorised test: n_perms reported correctly ──────────────────────
    #[test]
    fn vec_perm_n_perms_reported() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let cfg = VecPermConfig {
            n_permutations: 123,
            seed: 17,
        };
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn()];
        let result = vectorised_permutation_test(&x, &y, 3, 3, &stats, &cfg).expect("ok");
        assert_eq!(result.n_perms, 123);
    }

    // ── 12. empty input returns error ─────────────────────────────────────────
    #[test]
    fn vec_perm_empty_error() {
        let cfg = VecPermConfig::default();
        let stats: Vec<TwoSampleStatFn> = vec![mean_diff_fn()];
        let r = vectorised_permutation_test(&[], &[1.0], 0, 1, &stats, &cfg);
        assert!(r.is_err());
    }

    // ── 13. batch_two_sample_stats: empty input returns NaN vector ────────────
    #[test]
    fn batch_stats_empty_returns_nan() {
        let stats = batch_two_sample_stats(&[], &[1.0, 2.0]);
        assert!(stats.iter().all(|v| v.is_nan()));
    }

    // ── 14. wilcoxon rank sum: identical groups → stat near zero ─────────────
    #[test]
    fn rank_sum_identical_groups_near_zero() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let stats = batch_two_sample_stats(&x, &y);
        let rank_stat = stats[3];
        assert!(
            rank_stat.abs() < 1e-6,
            "rank_sum_stat={rank_stat} should be ≈0 for identical groups"
        );
    }

    // ── 15. permutation_matrix: n=0 or n_perms=0 returns empty ──────────────
    #[test]
    fn perm_matrix_empty_cases() {
        let mut rng = LcgRng::new(0);
        assert!(permutation_matrix(0, 10, &mut rng).is_empty());
        assert!(permutation_matrix(5, 0, &mut rng).is_empty());
    }
}
