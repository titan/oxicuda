//! PERMANOVA — Permutation-based Multivariate ANOVA using distance matrices.
//!
//! Implements the Anderson (2001) method: partitions a dissimilarity matrix into
//! among-group and within-group components and tests via permutation.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Configuration for PERMANOVA.
#[derive(Debug, Clone)]
pub struct PermanovaConfig {
    /// Number of permutations for the null distribution.
    pub n_permutations: usize,
    /// Random seed for the LCG generator.
    pub seed: u64,
}

impl Default for PermanovaConfig {
    fn default() -> Self {
        Self {
            n_permutations: 999,
            seed: 42,
        }
    }
}

/// Result of a PERMANOVA analysis.
#[derive(Debug, Clone)]
pub struct PermanovaResult {
    /// Observed pseudo-F statistic.
    pub f_statistic: f64,
    /// Permutation p-value: (count ≥ F_obs + 1) / (n_permutations + 1).
    pub p_value: f64,
    /// R² = SS_A / SS_T (proportion of variation explained by grouping).
    pub r_squared: f64,
    /// Numerator degrees of freedom (number of groups - 1).
    pub df_groups: usize,
    /// Denominator degrees of freedom (n - number of groups).
    pub df_residual: usize,
}

// ---------------------------------------------------------------------------
// Distance metric
// ---------------------------------------------------------------------------

/// Distance metric selector for constructing a distance matrix from raw data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistMetric {
    /// Euclidean distance: sqrt(Σ (x_i - y_i)²)
    Euclidean,
    /// Manhattan (L1) distance: Σ |x_i - y_i|
    Manhattan,
    /// Bray-Curtis dissimilarity: Σ|x_i - y_i| / (Σ x_i + Σ y_i)
    BrayCurtis,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Inline accessor for a row-major n×n matrix stored in a flat slice.
#[inline(always)]
fn mat_get(dist: &[f64], n: usize, i: usize, j: usize) -> f64 {
    dist[i * n + j]
}

/// Compute the PERMANOVA pseudo-F statistic for the given group assignment.
///
/// # Algorithm (Anderson 2001)
/// SS_T = (1/n) Σ_{i<j} d_{ij}²
/// SS_W = Σ_k (1/n_k) Σ_{i<j, both in k} d_{ij}²
/// SS_A = SS_T - SS_W
/// F    = (SS_A / (K-1)) / (SS_W / (n-K))
pub fn permanova_f_statistic(dist: &[f64], n: usize, groups: &[usize]) -> StatsResult<f64> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if dist.len() != n * n {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n * n],
            got: vec![dist.len()],
        });
    }
    if groups.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: groups.len(),
        });
    }

    // Unique groups
    let max_group = *groups.iter().max().unwrap_or(&0);
    let k_groups = max_group + 1; // number of groups

    // Count samples per group
    let mut n_k = vec![0usize; k_groups];
    for &g in groups {
        n_k[g] += 1;
    }
    let active_groups: Vec<usize> = (0..k_groups).filter(|&g| n_k[g] > 0).collect();
    let k = active_groups.len();
    if k < 2 {
        return Err(StatsError::InvalidParameter {
            name: "groups".into(),
            reason: "must have at least 2 distinct groups".into(),
        });
    }
    if n <= k {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: k + 1,
        });
    }

    // SS_T = (1/n) Σ_{i<j} d_ij^2
    let mut ss_t = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = mat_get(dist, n, i, j);
            ss_t += d * d;
        }
    }
    ss_t /= n as f64;

    // SS_W = Σ_k (1/n_k) Σ_{i<j in k} d_ij^2
    // Build group membership lists
    let mut group_members: Vec<Vec<usize>> = vec![Vec::new(); k_groups];
    for i in 0..n {
        group_members[groups[i]].push(i);
    }

    let mut ss_w = 0.0f64;
    for &g in &active_groups {
        let members = &group_members[g];
        let nk = members.len();
        if nk < 2 {
            continue;
        }
        let mut ss_wk = 0.0f64;
        for a in 0..nk {
            for b in (a + 1)..nk {
                let d = mat_get(dist, n, members[a], members[b]);
                ss_wk += d * d;
            }
        }
        ss_w += ss_wk / nk as f64;
    }

    let ss_a = ss_t - ss_w;
    let df_a = (k - 1) as f64;
    let df_w = (n - k) as f64;

    if ss_w < 0.0 || df_w <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "PERMANOVA: non-positive SS_W or df_residual".into(),
        ));
    }

    let ms_a = ss_a / df_a;
    let ms_w = ss_w / df_w;

    if ms_w <= 0.0 {
        // All within-group variation is zero → F is infinite / undefined
        return Ok(f64::INFINITY);
    }

    Ok(ms_a / ms_w)
}

/// Run a full PERMANOVA: compute observed F, permute group labels, return result.
///
/// # Arguments
/// * `dist`   — n×n flat row-major distance/dissimilarity matrix
/// * `n`      — number of samples
/// * `groups` — group assignment for each sample (0-based indices)
/// * `cfg`    — configuration
pub fn permanova(
    dist: &[f64],
    n: usize,
    groups: &[usize],
    cfg: &PermanovaConfig,
) -> StatsResult<PermanovaResult> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if cfg.n_permutations == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_permutations".into(),
            reason: "must be > 0".into(),
        });
    }

    // Validate all group labels and matrix size
    for (idx, &g) in groups.iter().enumerate() {
        // We allow any usize value for group; just validate it won't panic in mat_get
        let _ = (idx, g); // groups are validated inside f_statistic
    }

    let f_obs = permanova_f_statistic(dist, n, groups)?;

    // Derive K and df
    let max_group = *groups.iter().max().unwrap_or(&0);
    let k_groups = max_group + 1;
    let mut n_k = vec![0usize; k_groups];
    for &g in groups {
        n_k[g] += 1;
    }
    let k = n_k.iter().filter(|&&v| v > 0).count();
    let df_groups = k - 1;
    let df_residual = n - k;
    let r_squared = {
        // Recompute SS for R^2
        let mut ss_t = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = mat_get(dist, n, i, j);
                ss_t += d * d;
            }
        }
        ss_t /= n as f64;

        let mut group_members: Vec<Vec<usize>> = vec![Vec::new(); k_groups];
        for (i, &g) in groups.iter().enumerate().take(n) {
            group_members[g].push(i);
        }
        let mut ss_w = 0.0f64;
        for members in group_members.iter().take(k_groups) {
            let nk = members.len();
            if nk < 2 {
                continue;
            }
            let mut ss_wk = 0.0f64;
            for a in 0..nk {
                for b in (a + 1)..nk {
                    let d = mat_get(dist, n, members[a], members[b]);
                    ss_wk += d * d;
                }
            }
            ss_w += ss_wk / nk as f64;
        }
        if ss_t > 0.0 {
            (ss_t - ss_w) / ss_t
        } else {
            0.0
        }
    };

    // Permutation test: shuffle group labels repeatedly
    let mut rng = LcgRng::new(cfg.seed);
    let mut perm_groups = groups.to_vec();
    let mut count_geq = 0usize;

    for _ in 0..cfg.n_permutations {
        // Fisher-Yates shuffle of group labels
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            perm_groups.swap(i, j);
        }
        // Recompute F for this permutation — use a fast internal path
        let f_perm = permanova_f_statistic_raw(dist, n, &perm_groups, k_groups);
        if f_perm >= f_obs {
            count_geq += 1;
        }
    }

    let p_value = (count_geq + 1) as f64 / (cfg.n_permutations + 1) as f64;

    Ok(PermanovaResult {
        f_statistic: f_obs,
        p_value,
        r_squared,
        df_groups,
        df_residual,
    })
}

/// Fast, infallible internal F computation (for permutation loop; skips validation).
#[inline]
fn permanova_f_statistic_raw(dist: &[f64], n: usize, groups: &[usize], k_groups: usize) -> f64 {
    let mut n_k = vec![0usize; k_groups];
    for &g in groups {
        let g = g.min(k_groups - 1);
        n_k[g] += 1;
    }

    let mut ss_t = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i * n + j];
            ss_t += d * d;
        }
    }
    ss_t /= n as f64;

    // Build group members inline
    let mut group_members: Vec<Vec<usize>> = vec![Vec::new(); k_groups];
    for (i, &g) in groups.iter().enumerate().take(n) {
        let gi = g.min(k_groups - 1);
        group_members[gi].push(i);
    }

    let mut ss_w = 0.0f64;
    for members in group_members.iter() {
        let nk = members.len();
        if nk < 2 {
            continue;
        }
        let mut ss_wk = 0.0f64;
        for a in 0..nk {
            for b in (a + 1)..nk {
                let d = dist[members[a] * n + members[b]];
                ss_wk += d * d;
            }
        }
        ss_w += ss_wk / nk as f64;
    }

    let k = n_k.iter().filter(|&&v| v > 0).count();
    if k < 2 || n <= k {
        return 0.0;
    }
    let ss_a = ss_t - ss_w;
    let ms_a = ss_a / (k as f64 - 1.0);
    let ms_w = ss_w / (n as f64 - k as f64);
    if ms_w <= 0.0 {
        return f64::INFINITY;
    }
    ms_a / ms_w
}

// ---------------------------------------------------------------------------
// Convenience: build distance matrix from data
// ---------------------------------------------------------------------------

/// Build an n×n distance matrix from raw data (n rows × d columns, row-major).
///
/// # Arguments
/// * `data`   — flat row-major data matrix (n × d)
/// * `n`      — number of samples
/// * `d`      — number of features (dimensions)
/// * `metric` — distance metric to use
///
/// # Returns
/// Flat row-major n×n symmetric distance matrix.
pub fn distance_matrix_from_data(
    data: &[f64],
    n: usize,
    d: usize,
    metric: DistMetric,
) -> StatsResult<Vec<f64>> {
    if data.len() != n * d {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n * d],
            got: vec![data.len()],
        });
    }
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                mat[i * n + j] = 0.0;
                continue;
            }
            let xi = &data[i * d..(i + 1) * d];
            let xj = &data[j * d..(j + 1) * d];
            let dist_val = match metric {
                DistMetric::Euclidean => xi
                    .iter()
                    .zip(xj)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                DistMetric::Manhattan => xi.iter().zip(xj).map(|(a, b)| (a - b).abs()).sum::<f64>(),
                DistMetric::BrayCurtis => {
                    let num: f64 = xi.iter().zip(xj).map(|(a, b)| (a - b).abs()).sum();
                    let denom: f64 = xi.iter().zip(xj).map(|(a, b)| a + b).sum();
                    if denom == 0.0 { 0.0 } else { num / denom }
                }
            };
            mat[i * n + j] = dist_val;
        }
    }
    Ok(mat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple Euclidean distance matrix for small data.
    fn euclidean_dist_2d(pts: &[(f64, f64)]) -> Vec<f64> {
        let n = pts.len();
        let mut mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i].0 - pts[j].0;
                let dy = pts[i].1 - pts[j].1;
                mat[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        mat
    }

    // ---- permanova_f_statistic ----

    #[test]
    fn f_statistic_well_separated_groups() {
        // Two clearly separated clusters in 2D
        let pts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0), // group 0
            (10.0, 10.0),
            (11.0, 10.0),
            (10.0, 11.0), // group 1
        ];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 0, 1, 1, 1];
        let f =
            permanova_f_statistic(&dist, 6, &groups).expect("permanova_f_statistic should succeed");
        // Large F expected for well-separated groups
        assert!(f > 10.0, "F={f} should be large for well-separated groups");
    }

    #[test]
    fn f_statistic_identical_groups() {
        // All points at the same location → SS_W = 0, F undefined (inf)
        let pts = [(0.0, 0.0); 4];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 1, 1];
        let f =
            permanova_f_statistic(&dist, 4, &groups).expect("permanova_f_statistic should succeed");
        // SS_W = 0 → ms_w = 0 → F = infinity
        assert!(f.is_infinite() || f >= 0.0);
    }

    #[test]
    fn f_statistic_mixed_groups() {
        // Overlapping — F should be moderate/small
        let pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.5, 0.0)];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 1, 1];
        let f =
            permanova_f_statistic(&dist, 4, &groups).expect("permanova_f_statistic should succeed");
        assert!(f >= 0.0);
    }

    #[test]
    fn f_statistic_shape_mismatch_error() {
        let dist = vec![0.0; 8]; // Not 3×3 = 9
        let groups = [0, 0, 1];
        assert!(matches!(
            permanova_f_statistic(&dist, 3, &groups),
            Err(StatsError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn f_statistic_single_group_error() {
        let pts = [(0.0, 0.0), (1.0, 0.0)];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0]; // only one group
        assert!(matches!(
            permanova_f_statistic(&dist, 2, &groups),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    // ---- permanova (full test with p-value) ----

    #[test]
    fn permanova_well_separated_small_pvalue() {
        let pts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (20.0, 20.0),
            (21.0, 20.0),
            (20.0, 21.0),
        ];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 0, 1, 1, 1];
        let cfg = PermanovaConfig {
            n_permutations: 499,
            seed: 1234,
        };
        let r = permanova(&dist, 6, &groups, &cfg).expect("permanova should succeed");
        assert!(
            r.p_value <= 0.1,
            "p={} expected small for well-separated groups",
            r.p_value
        );
        assert!(r.r_squared > 0.8);
        assert_eq!(r.df_groups, 1);
        assert_eq!(r.df_residual, 4);
    }

    #[test]
    fn permanova_overlapping_large_pvalue() {
        // Groups are almost identical — p-value should be large
        let pts = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 1, 1];
        let cfg = PermanovaConfig {
            n_permutations: 199,
            seed: 99,
        };
        let r = permanova(&dist, 4, &groups, &cfg).expect("permanova should succeed");
        // p should be well above 0.01
        assert!(
            r.p_value > 0.01,
            "p={} expected large for overlapping groups",
            r.p_value
        );
    }

    #[test]
    fn permanova_r_squared_range() {
        let pts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (5.0, 5.0),
            (6.0, 5.0),
            (5.0, 6.0),
        ];
        let dist = euclidean_dist_2d(&pts);
        let groups = [0, 0, 0, 1, 1, 1];
        let cfg = PermanovaConfig {
            n_permutations: 99,
            seed: 7,
        };
        let r = permanova(&dist, 6, &groups, &cfg).expect("permanova should succeed");
        assert!(r.r_squared >= 0.0 && r.r_squared <= 1.0);
    }

    // ---- distance_matrix_from_data ----

    #[test]
    fn distance_euclidean_construction() {
        // Two 2D points: (0,0) and (3,4) → distance = 5
        let data = [0.0, 0.0, 3.0, 4.0];
        let mat = distance_matrix_from_data(&data, 2, 2, DistMetric::Euclidean)
            .expect("distance_matrix_from_data should succeed");
        // Row-major flatten of (row=0, col=1) in a 2x2 distance matrix → index 1.
        assert!((mat[1] - 5.0).abs() < 1e-12);
        assert_eq!(mat[0], 0.0);
        assert_eq!(mat[3], 0.0);
    }

    #[test]
    fn distance_manhattan_construction() {
        let data = [0.0, 0.0, 3.0, 4.0];
        let mat = distance_matrix_from_data(&data, 2, 2, DistMetric::Manhattan)
            .expect("distance_matrix_from_data should succeed");
        // Row-major flatten of (row=0, col=1) in a 2x2 distance matrix → index 1.
        assert!((mat[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn distance_bray_curtis_construction() {
        // (1,1) and (3,3): |1-3|+|1-3| = 4; (1+3)+(1+3) = 8 → BC = 0.5
        let data = [1.0, 1.0, 3.0, 3.0];
        let mat = distance_matrix_from_data(&data, 2, 2, DistMetric::BrayCurtis)
            .expect("distance_matrix_from_data should succeed");
        // Row-major flatten of (row=0, col=1) in a 2x2 distance matrix → index 1.
        assert!((mat[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn distance_shape_mismatch_error() {
        // 3 points × 2 dims = 6, but we provide 5
        let data = [0.0; 5];
        assert!(matches!(
            distance_matrix_from_data(&data, 3, 2, DistMetric::Euclidean),
            Err(StatsError::ShapeMismatch { .. })
        ));
    }
}
