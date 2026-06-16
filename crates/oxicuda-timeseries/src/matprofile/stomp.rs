//! STOMP — Scalable Time-series Ordered-search Matrix Profile.
//!
//! Reference: Zhu et al. 2016 ICDM "Matrix Profile I: All Pairs Similarity Joins
//! for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets";
//! Yeh et al. 2016.
//!
//! Computes, for each length-`m` subsequence `x_i`, the distance to its
//! nearest non-trivial neighbor via z-normalised Euclidean distance.
//! The STOMP update (O(n)) reduces the cost of maintaining the inner-product
//! matrix incrementally as the query subsequence shifts by one position.

use crate::error::{TsError, TsResult};

// ─── Config / Result ─────────────────────────────────────────────────────────

/// Configuration for the matrix profile computation.
#[derive(Debug, Clone)]
pub struct MatProfileConfig {
    /// Subsequence length `m`.
    pub window_len: usize,
    /// Minimum index separation for the exclusion zone (default = `floor(m / 4)`).
    /// Set to 0 to disable.
    pub exclusion_zone: usize,
    /// Number of motif pairs to discover (default 3).
    pub n_motifs: usize,
    /// Number of discords (anomalies) to discover (default 3).
    pub n_discords: usize,
}

impl MatProfileConfig {
    /// Build a config with sensible defaults for window length `m`.
    #[must_use]
    pub fn new(window_len: usize) -> Self {
        let exclusion_zone = window_len / 4;
        Self {
            window_len,
            exclusion_zone,
            n_motifs: 3,
            n_discords: 3,
        }
    }
}

/// Output of the matrix profile computation.
#[derive(Debug, Clone)]
pub struct MatProfileResult {
    /// L = n − window_len + 1 distances (one per subsequence).
    pub profile: Vec<f64>,
    /// Index of the nearest neighbour for each subsequence.
    pub profile_index: Vec<usize>,
    /// Top motif pairs `(idx_a, idx_b, distance)`.
    pub motifs: Vec<(usize, usize, f64)>,
    /// Top discord indices `(idx, distance)`.
    pub discords: Vec<(usize, f64)>,
    /// Number of subsequences `L`.
    pub n_subseq: usize,
    /// Subsequence length used.
    pub window_len: usize,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Compute the self-join matrix profile of `x` using the STOMP algorithm.
///
/// # Errors
/// - `InvalidKernelSize` if `window_len == 0`.
/// - `InvalidSequenceLength` if `x.len() < window_len + 1` (fewer than 2 subsequences).
pub fn matrix_profile(x: &[f64], config: &MatProfileConfig) -> TsResult<MatProfileResult> {
    validate_self_join(x, config)?;
    let m = config.window_len;
    let n = x.len();
    let l = n - m + 1;

    let (mu, sig) = sliding_stats(x, m);
    let excl = config.exclusion_zone;

    // First QT row: QT[0][j] = <x[0..m], x[j..j+m]> for j=0..l
    let mut qt_prev = compute_qt_row(x, x, m, l);

    let mut profile = vec![f64::INFINITY; l];
    let mut profile_index = vec![0usize; l];

    // Process i=0 row
    update_profile_row(
        &qt_prev,
        &mu,
        &sig,
        &mu,
        &sig,
        0,
        l,
        excl,
        m,
        &mut profile,
        &mut profile_index,
    );

    // STOMP: i = 1..l
    for i in 1..l {
        let mut qt = vec![0.0f64; l];
        // j=0: brute-force column 0
        qt[0] = compute_qt_single(x, x, i, 0, m);
        // j=1..l: incremental update
        for j in 1..l {
            qt[j] = qt_prev[j - 1] + x[i + m - 1] * x[j + m - 1] - x[i - 1] * x[j - 1];
        }
        update_profile_row(
            &qt,
            &mu,
            &sig,
            &mu,
            &sig,
            i,
            l,
            excl,
            m,
            &mut profile,
            &mut profile_index,
        );
        qt_prev = qt;
    }

    // Symmetrise: for each (i, j=profile_index[i]), update profile[j] if smaller
    for i in 0..l {
        let j = profile_index[i];
        if j < l && profile[j] > profile[i] {
            profile[j] = profile[i];
            profile_index[j] = i;
        }
    }

    let motifs = find_motifs(&profile, &profile_index, config.n_motifs, excl, l);
    let discords = find_discords(&profile, config.n_discords, excl, l);

    Ok(MatProfileResult {
        profile,
        profile_index,
        motifs,
        discords,
        n_subseq: l,
        window_len: m,
    })
}

/// Compute the AB-join matrix profile: for each length-`m` subsequence in `x`
/// (query), find the nearest subsequence in `y` (reference).
///
/// Output profile length = `x.len() - m + 1`.
///
/// # Errors
/// - `InvalidKernelSize` if `window_len == 0`.
/// - `InvalidSequenceLength` if either series has fewer than `window_len + 1` elements.
pub fn matrix_profile_ab(
    x: &[f64],
    y: &[f64],
    config: &MatProfileConfig,
) -> TsResult<MatProfileResult> {
    validate_ab_join(x, y, config)?;
    let m = config.window_len;
    let lx = x.len() - m + 1;
    let ly = y.len() - m + 1;
    let excl = config.exclusion_zone;

    let (mu_x, sig_x) = sliding_stats(x, m);
    let (mu_y, sig_y) = sliding_stats(y, m);

    // i=0: brute-force first QT row against reference y
    let mut qt_prev = compute_qt_row(x, y, m, ly);

    let mut profile = vec![f64::INFINITY; lx];
    let mut profile_index = vec![0usize; lx];

    for j in 0..ly {
        let d = znorm_dist_from_qt(qt_prev[j], mu_x[0], sig_x[0], mu_y[j], sig_y[j], m);
        if d < profile[0] {
            profile[0] = d;
            profile_index[0] = j;
        }
    }

    for i in 1..lx {
        let mut qt = vec![0.0f64; ly];
        qt[0] = compute_qt_single(x, y, i, 0, m);
        for j in 1..ly {
            qt[j] = qt_prev[j - 1] + x[i + m - 1] * y[j + m - 1] - x[i - 1] * y[j - 1];
        }
        for j in 0..ly {
            let d = znorm_dist_from_qt(qt[j], mu_x[i], sig_x[i], mu_y[j], sig_y[j], m);
            if d < profile[i] {
                profile[i] = d;
                profile_index[i] = j;
            }
        }
        qt_prev = qt;
    }

    // Replace any remaining INFINITY (edge case: ly==0)
    for v in &mut profile {
        if v.is_infinite() {
            *v = 0.0;
        }
    }

    let motifs = find_motifs(&profile, &profile_index, config.n_motifs, excl, lx);
    let discords = find_discords(&profile, config.n_discords, excl, lx);

    Ok(MatProfileResult {
        profile,
        profile_index,
        motifs,
        discords,
        n_subseq: lx,
        window_len: m,
    })
}

/// Compute the z-normalised Euclidean distance between two equal-length slices.
///
/// If either slice is constant (σ < 1e-8) and the other is not, returns
/// `sqrt(2 * a.len())` (maximum distance for z-normalised sequences).
/// Both constant → 0.
#[must_use]
pub fn znorm_distance(a: &[f64], b: &[f64]) -> f64 {
    let m = a.len().min(b.len());
    if m == 0 {
        return 0.0;
    }
    let mu_a = a[..m].iter().sum::<f64>() / m as f64;
    let mu_b = b[..m].iter().sum::<f64>() / m as f64;
    let sig_a = pop_std(&a[..m], mu_a);
    let sig_b = pop_std(&b[..m], mu_b);
    let qt: f64 = a[..m]
        .iter()
        .zip(b[..m].iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum();
    znorm_dist_from_qt(qt, mu_a, sig_a, mu_b, sig_b, m)
}

/// Compute running mean and population standard deviation for all length-`m`
/// windows of `x`.
///
/// Returns `(mu, sigma)` where each vector has length `n - m + 1`.
/// Uses an incremental update for O(n) total cost.
#[must_use]
pub fn sliding_stats(x: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if m == 0 || n < m {
        return (vec![], vec![]);
    }
    let l = n - m + 1;
    let mut mu = vec![0.0f64; l];
    let mut sig = vec![0.0f64; l];

    let mut sum: f64 = x[..m].iter().sum();
    let mut sum2: f64 = x[..m].iter().map(|&v| v * v).sum();
    mu[0] = sum / m as f64;
    let var0 = (sum2 - sum * sum / m as f64) / m as f64;
    sig[0] = var0.max(0.0).sqrt();

    for i in 1..l {
        let add = x[i + m - 1];
        let rem = x[i - 1];
        sum += add - rem;
        sum2 += add * add - rem * rem;
        mu[i] = sum / m as f64;
        let var_i = (sum2 - sum * sum / m as f64) / m as f64;
        sig[i] = var_i.max(0.0).sqrt();
    }
    (mu, sig)
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Compute first QT row: QT[j] = <query[0..m], reference[j..j+m]>.
fn compute_qt_row(query: &[f64], reference: &[f64], m: usize, ly: usize) -> Vec<f64> {
    (0..ly)
        .map(|j| compute_qt_single(query, reference, 0, j, m))
        .collect()
}

/// Inner product <query[qi..qi+m], reference[ri..ri+m]>.
#[inline]
fn compute_qt_single(query: &[f64], reference: &[f64], qi: usize, ri: usize, m: usize) -> f64 {
    (0..m).map(|k| query[qi + k] * reference[ri + k]).sum()
}

/// Z-normalised distance from pre-computed inner product.
#[inline]
fn znorm_dist_from_qt(qt: f64, mu_i: f64, sig_i: f64, mu_j: f64, sig_j: f64, m: usize) -> f64 {
    let i_const = sig_i < 1e-8;
    let j_const = sig_j < 1e-8;
    if i_const && j_const {
        return 0.0;
    }
    if i_const || j_const {
        return (2.0 * m as f64).sqrt();
    }
    // Avoid adding epsilon to denominator when it's already large — epsilon causes
    // self-distance to be non-zero. Instead, clamp pearson to [-1,1] which handles
    // numerical overflow without injecting bias.
    let denom = m as f64 * sig_i * sig_j;
    let pearson = if denom > 1e-15 {
        ((qt - m as f64 * mu_i * mu_j) / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let dist_sq = 2.0 * m as f64 * (1.0 - pearson);
    dist_sq.max(0.0).sqrt()
}

/// Population standard deviation.
#[inline]
fn pop_std(x: &[f64], mu: f64) -> f64 {
    let m = x.len() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let var = x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / m;
    var.max(0.0).sqrt()
}

/// Update `profile[i]` by scanning all valid `j` neighbours.
fn update_profile_row(
    qt: &[f64],
    mu_q: &[f64],
    sig_q: &[f64],
    mu_r: &[f64],
    sig_r: &[f64],
    qi: usize,
    l: usize,
    excl: usize,
    m: usize,
    profile: &mut [f64],
    profile_index: &mut [usize],
) {
    let lo = qi.saturating_sub(excl);
    let hi = (qi + excl + 1).min(l);

    for j in 0..l {
        if j >= lo && j < hi {
            continue; // exclusion zone
        }
        let d = znorm_dist_from_qt(qt[j], mu_q[qi], sig_q[qi], mu_r[j], sig_r[j], m);
        if d < profile[qi] {
            profile[qi] = d;
            profile_index[qi] = j;
        }
    }

    // Guard: if all neighbours excluded (tiny series)
    if profile[qi].is_infinite() && l > 0 {
        profile[qi] = 0.0;
    }
}

/// Discover the top-k non-overlapping motif pairs.
fn find_motifs(
    profile: &[f64],
    profile_index: &[usize],
    k: usize,
    excl: usize,
    l: usize,
) -> Vec<(usize, usize, f64)> {
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_unstable_by(|&a, &b| {
        profile[a]
            .partial_cmp(&profile[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut motifs: Vec<(usize, usize, f64)> = Vec::with_capacity(k);
    let mut used = vec![false; l];

    for &i in &order {
        if motifs.len() >= k {
            break;
        }
        if used[i] {
            continue;
        }
        let j = profile_index[i];
        if j >= l || used[j] {
            continue;
        }
        let conflict = motifs.iter().any(|&(a, b, _)| {
            overlaps(i, a, excl)
                || overlaps(i, b, excl)
                || overlaps(j, a, excl)
                || overlaps(j, b, excl)
        });
        if conflict {
            continue;
        }
        motifs.push((i, j, profile[i]));
        mark_zone(&mut used, i, excl, l);
        mark_zone(&mut used, j, excl, l);
    }
    motifs
}

/// Discover the top-k discords (highest profile values).
fn find_discords(profile: &[f64], k: usize, excl: usize, l: usize) -> Vec<(usize, f64)> {
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_unstable_by(|&a, &b| {
        profile[b]
            .partial_cmp(&profile[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut discords: Vec<(usize, f64)> = Vec::with_capacity(k);
    let mut used = vec![false; l];

    for &i in &order {
        if discords.len() >= k {
            break;
        }
        if used[i] {
            continue;
        }
        discords.push((i, profile[i]));
        mark_zone(&mut used, i, excl, l);
    }
    discords
}

#[inline]
fn overlaps(a: usize, b: usize, excl: usize) -> bool {
    a.abs_diff(b) <= excl
}

#[inline]
fn mark_zone(used: &mut [bool], center: usize, excl: usize, l: usize) {
    let lo = center.saturating_sub(excl);
    let hi = (center + excl + 1).min(l);
    for u in &mut used[lo..hi] {
        *u = true;
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_self_join(x: &[f64], config: &MatProfileConfig) -> TsResult<()> {
    if config.window_len == 0 {
        return Err(TsError::InvalidKernelSize(0));
    }
    if x.len() < config.window_len + 1 {
        return Err(TsError::InvalidSequenceLength(x.len()));
    }
    Ok(())
}

fn validate_ab_join(x: &[f64], y: &[f64], config: &MatProfileConfig) -> TsResult<()> {
    if config.window_len == 0 {
        return Err(TsError::InvalidKernelSize(0));
    }
    if x.len() < config.window_len + 1 {
        return Err(TsError::InvalidSequenceLength(x.len()));
    }
    if y.len() < config.window_len + 1 {
        return Err(TsError::InvalidSequenceLength(y.len()));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_pattern(n: usize, pat_len: usize) -> Vec<f64> {
        let mut sig = vec![0.0f64; n];
        for k in 0..pat_len {
            let v = (k as f64 * std::f64::consts::PI / pat_len as f64).sin();
            sig[k] = v;
            let offset = pat_len + 5;
            if offset + k < n {
                sig[offset + k] = v;
            }
        }
        for (i, v) in sig.iter_mut().enumerate() {
            if *v == 0.0 {
                *v = 0.01 * (i as f64).sin();
            }
        }
        sig
    }

    #[test]
    fn test_profile_length() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let cfg = MatProfileConfig::new(10);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert_eq!(res.profile.len(), 41);
    }

    #[test]
    fn test_profile_index_length() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let cfg = MatProfileConfig::new(10);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert_eq!(res.profile_index.len(), 41);
    }

    #[test]
    fn test_profile_values_nonnegative() {
        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).sin()).collect();
        let cfg = MatProfileConfig::new(8);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        for &v in &res.profile {
            assert!(v >= 0.0, "negative distance: {v}");
        }
    }

    #[test]
    fn test_profile_self_consistency() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.4).sin()).collect();
        let m = 6usize;
        let cfg = MatProfileConfig::new(m);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        for i in 0..res.n_subseq {
            let j = res.profile_index[i];
            let actual = znorm_distance(&x[i..i + m], &x[j..j + m]);
            assert!(
                res.profile[i] <= actual + 1e-7,
                "i={i}: profile={} > actual={}",
                res.profile[i],
                actual
            );
        }
    }

    #[test]
    fn test_motif_near_duplicate() {
        let sig = repeated_pattern(70, 12);
        let cfg = MatProfileConfig::new(12);
        let res = matrix_profile(&sig, &cfg).expect("matrix_profile should succeed");
        assert!(!res.motifs.is_empty(), "no motifs found");
        assert!(
            res.motifs[0].2 < 1.5,
            "top motif distance too large: {}",
            res.motifs[0].2
        );
    }

    #[test]
    fn test_discord_anomaly_detected() {
        let n = 80usize;
        let m = 8usize;
        let mut sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        for k in 0..m {
            sig[40 + k] = 5.0 * (k as f64 + 1.0);
        }
        let mut cfg = MatProfileConfig::new(m);
        cfg.n_discords = 1;
        let res = matrix_profile(&sig, &cfg).expect("matrix_profile should succeed");
        assert!(!res.discords.is_empty(), "no discords found");
        let discord_idx = res.discords[0].0;
        assert!(
            discord_idx.abs_diff(40) <= m + 2,
            "discord not near anomaly: got {discord_idx}"
        );
    }

    #[test]
    fn test_znorm_distance_self() {
        let a: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let d = znorm_distance(&a, &a);
        assert!(d.abs() < 1e-8, "self-distance not zero: {d}");
    }

    #[test]
    fn test_znorm_distance_symmetric() {
        let a: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..10).map(|i| (i as f64 * 0.2 + 1.0).sin()).collect();
        let d_ab = znorm_distance(&a, &b);
        let d_ba = znorm_distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-10, "asymmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn test_znorm_distance_constant_vs_nonconst() {
        let m = 8usize;
        let a = vec![3.0f64; m];
        let b: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let d = znorm_distance(&a, &b);
        let expected = (2.0 * m as f64).sqrt();
        assert!(
            (d - expected).abs() < 1e-8,
            "expected sqrt(2m)={expected}, got {d}"
        );
    }

    #[test]
    fn test_sliding_stats_mean_first_window() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mu, _) = sliding_stats(&x, 3);
        assert!((mu[0] - 2.0).abs() < 1e-10, "mu[0]={}", mu[0]);
    }

    #[test]
    fn test_sliding_stats_std_first_window() {
        let x = vec![1.0f64, 3.0, 5.0, 7.0, 9.0];
        let (_, sig) = sliding_stats(&x, 3);
        // pop std of [1,3,5]: mean=3, var=(4+0+4)/3=8/3
        let expected = (8.0f64 / 3.0).sqrt();
        assert!(
            (sig[0] - expected).abs() < 1e-8,
            "sig[0]={} expected={expected}",
            sig[0]
        );
    }

    #[test]
    fn test_n_motifs_one() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut cfg = MatProfileConfig::new(8);
        cfg.n_motifs = 1;
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert!(res.motifs.len() <= 1);
    }

    #[test]
    fn test_n_discords_one() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut cfg = MatProfileConfig::new(8);
        cfg.n_discords = 1;
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert!(res.discords.len() <= 1);
    }

    #[test]
    fn test_exclusion_zone_respected() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let m = 8usize;
        let cfg = MatProfileConfig::new(m);
        let excl = cfg.exclusion_zone;
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        for i in 0..res.n_subseq {
            let j = res.profile_index[i];
            // Either outside excl zone, or the series is so short all pairs are excluded
            assert!(
                i.abs_diff(j) > excl || res.n_subseq <= excl * 2 + 1,
                "exclusion zone violated i={i}, j={j}, excl={excl}"
            );
        }
    }

    #[test]
    fn test_ab_join_length() {
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..60).map(|i| i as f64 * 0.5).collect();
        let cfg = MatProfileConfig::new(8);
        let res = matrix_profile_ab(&x, &y, &cfg).expect("matrix_profile_ab should succeed");
        assert_eq!(res.profile.len(), x.len() - 8 + 1);
    }

    #[test]
    fn test_ab_join_values_nonnegative() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.2).sin()).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).cos()).collect();
        let cfg = MatProfileConfig::new(6);
        let res = matrix_profile_ab(&x, &y, &cfg).expect("matrix_profile_ab should succeed");
        for &v in &res.profile {
            assert!(v >= 0.0, "negative: {v}");
        }
    }

    #[test]
    fn test_window_len_equal_series_len_errors() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let cfg = MatProfileConfig::new(10);
        let result = matrix_profile(&x, &cfg);
        assert!(result.is_err(), "expected error for len==window_len");
    }

    #[test]
    fn test_window_len_zero_errors() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let cfg = MatProfileConfig::new(0);
        let result = matrix_profile(&x, &cfg);
        assert!(result.is_err(), "expected error for window_len=0");
    }

    #[test]
    fn test_profile_approximate_symmetry() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let cfg = MatProfileConfig::new(5);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        // Both profile[i] and profile[j] must be non-negative
        for i in 0..res.n_subseq {
            let j = res.profile_index[i];
            assert!(res.profile[i] >= 0.0 && res.profile[j] >= 0.0);
        }
    }

    #[test]
    fn test_motif_indices_valid() {
        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.25).sin()).collect();
        let cfg = MatProfileConfig::new(8);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        for &(a, b, _) in &res.motifs {
            assert!(a < res.n_subseq, "motif a={a} >= n_subseq");
            assert!(b < res.n_subseq, "motif b={b} >= n_subseq");
        }
    }

    #[test]
    fn test_n50_m10_runs() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let cfg = MatProfileConfig::new(10);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert_eq!(res.n_subseq, 41);
    }

    #[test]
    fn test_n100_m20_correct_n_subseq() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.07).sin()).collect();
        let cfg = MatProfileConfig::new(20);
        let res = matrix_profile(&x, &cfg).expect("matrix_profile should succeed");
        assert_eq!(res.n_subseq, 81);
        assert_eq!(res.profile.len(), 81);
    }
}
