//! Dynamic Time Warping distance and alignment.
//!
//! Sakoe & Chiba (1978) "Dynamic programming algorithm optimization for
//! spoken word recognition." IEEE Trans. ASSP 26(1):43-49.
//!
//! Petitjean, Ketterlin & Gançarski (2011) "A global averaging method for
//! dynamic time warping, with applications to clustering."
//! Pattern Recognition 44(3):678-693. (DBA)

use crate::error::{TsError, TsResult};

// ── Config & Result ──────────────────────────────────────────────────────────

/// Configuration for DTW computation.
#[derive(Debug, Clone)]
pub struct DtwConfig {
    /// Sakoe-Chiba band radius (None = unconstrained full DTW).
    pub band: Option<usize>,
    /// Normalize distance by warping path length.
    pub normalize: bool,
    /// Number of features per time step (1 for univariate).
    pub n_features: usize,
}

impl Default for DtwConfig {
    fn default() -> Self {
        Self {
            band: None,
            normalize: false,
            n_features: 1,
        }
    }
}

/// Result of a DTW alignment computation.
#[derive(Debug, Clone)]
pub struct DtwResult {
    /// Raw DTW distance.
    pub distance: f64,
    /// Warping path from (0,0) to (N-1,M-1). Each element is (i, j).
    pub path: Vec<(usize, usize)>,
    /// Number of steps in warping path.
    pub path_len: usize,
    /// `distance / path_len` if normalize=true, else same as `distance`.
    pub normalized_distance: f64,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Compute DTW distance and warping path between sequences `a` and `b`.
///
/// `a`: N × n_features row-major flat slice.
/// `b`: M × n_features row-major flat slice.
///
/// # Errors
///
/// - [`TsError::InvalidNumVariates`] when `n_features == 0`.
/// - [`TsError::EmptyInput`] when `n == 0` or `m == 0`.
/// - [`TsError::ShapeMismatch`] when slice lengths don't match.
pub fn dtw(a: &[f64], n: usize, b: &[f64], m: usize, config: &DtwConfig) -> TsResult<DtwResult> {
    validate_inputs(a, n, b, m, config)?;

    let cost = build_cost_matrix(a, n, b, m, config);
    let path = backtrack_path(&cost, n, m);
    let path_len = path.len();
    let distance = cost[(n - 1) * m + (m - 1)];
    let normalized_distance = if config.normalize && path_len > 0 {
        distance / path_len as f64
    } else {
        distance
    };

    Ok(DtwResult {
        distance,
        path,
        path_len,
        normalized_distance,
    })
}

/// Compute DTW distance only (no path tracking, same algorithm).
///
/// # Errors
///
/// Same as [`dtw`].
pub fn dtw_distance(a: &[f64], n: usize, b: &[f64], m: usize, config: &DtwConfig) -> TsResult<f64> {
    validate_inputs(a, n, b, m, config)?;
    let cost = build_cost_matrix(a, n, b, m, config);
    Ok(cost[(n - 1) * m + (m - 1)])
}

/// Compute the full N×M DTW cost matrix (useful for visualization).
///
/// Does not validate inputs; caller should ensure correctness.
#[must_use]
pub fn dtw_cost_matrix(a: &[f64], n: usize, b: &[f64], m: usize, config: &DtwConfig) -> Vec<f64> {
    if n == 0 || m == 0 || config.n_features == 0 {
        return Vec::new();
    }
    build_cost_matrix(a, n, b, m, config)
}

/// Compute symmetric K×K DTW distance matrix for K sequences of equal length L.
///
/// `sequences`: K × L × n_features row-major flat slice.
///
/// # Errors
///
/// - [`TsError::EmptyInput`] when `k == 0` or `l == 0`.
/// - [`TsError::ShapeMismatch`] when slice length doesn't match `k * l * n_features`.
pub fn dtw_distance_matrix(
    sequences: &[f64],
    k: usize,
    l: usize,
    config: &DtwConfig,
) -> TsResult<Vec<f64>> {
    if k == 0 || l == 0 {
        return Err(TsError::EmptyInput {
            msg: "DTW distance matrix: k and l must be > 0".to_owned(),
        });
    }
    if config.n_features == 0 {
        return Err(TsError::InvalidNumVariates(0));
    }
    let expected = k * l * config.n_features;
    if sequences.len() != expected {
        return Err(TsError::ShapeMismatch {
            msg: format!(
                "distance_matrix: expected {} elements (k={k} l={l} f={}), got {}",
                expected,
                config.n_features,
                sequences.len()
            ),
        });
    }

    let stride = l * config.n_features;
    let mut matrix = vec![0.0_f64; k * k];

    for i in 0..k {
        let ai = &sequences[i * stride..(i + 1) * stride];
        for j in (i + 1)..k {
            let bj = &sequences[j * stride..(j + 1) * stride];
            let d = build_cost_matrix(ai, l, bj, l, config)[(l - 1) * l + (l - 1)];
            matrix[i * k + j] = d;
            matrix[j * k + i] = d;
        }
    }
    Ok(matrix)
}

/// DTW Barycenter Averaging (Petitjean 2011).
///
/// Returns centroid of length `l * n_features`.
///
/// # Errors
///
/// - [`TsError::EmptyInput`] when `k == 0` or `l == 0`.
/// - [`TsError::ShapeMismatch`] when slice length doesn't match.
pub fn dtw_barycenter(
    sequences: &[f64],
    k: usize,
    l: usize,
    n_iter: usize,
    config: &DtwConfig,
) -> TsResult<Vec<f64>> {
    if k == 0 {
        return Err(TsError::EmptyInput {
            msg: "DBA: k must be > 0".to_owned(),
        });
    }
    if l == 0 {
        return Err(TsError::EmptyInput {
            msg: "DBA: l must be > 0".to_owned(),
        });
    }
    if config.n_features == 0 {
        return Err(TsError::InvalidNumVariates(0));
    }
    let f = config.n_features;
    let expected = k * l * f;
    if sequences.len() != expected {
        return Err(TsError::ShapeMismatch {
            msg: format!(
                "barycenter: expected {} elements, got {}",
                expected,
                sequences.len()
            ),
        });
    }

    let stride = l * f;

    // Initialize centroid as mean of all sequences
    let mut centroid = vec![0.0_f64; l * f];
    for seq_k in 0..k {
        let seq = &sequences[seq_k * stride..(seq_k + 1) * stride];
        for idx in 0..l * f {
            centroid[idx] += seq[idx];
        }
    }
    let inv_k = 1.0 / k as f64;
    for v in centroid.iter_mut() {
        *v *= inv_k;
    }

    for _ in 0..n_iter {
        // Accumulator: sum of assigned points per centroid position (flat: l * f)
        let mut acc: Vec<f64> = vec![0.0_f64; l * f];
        // Count of assigned points per position (for averaging)
        let mut cnt: Vec<usize> = vec![0_usize; l];

        for seq_k in 0..k {
            let seq = &sequences[seq_k * stride..(seq_k + 1) * stride];
            // Get warping path between seq and centroid
            let path = {
                let cost = build_cost_matrix(seq, l, &centroid, l, config);
                backtrack_path(&cost, l, l)
            };
            // For each (i, j) in path: assign seq[i] to centroid position j
            for &(i, j) in &path {
                for feat in 0..f {
                    acc[j * f + feat] += seq[i * f + feat];
                }
                cnt[j] += 1;
            }
        }

        // Update centroid
        for j in 0..l {
            if cnt[j] > 0 {
                let inv = 1.0 / cnt[j] as f64;
                for feat in 0..f {
                    centroid[j * f + feat] = acc[j * f + feat] * inv;
                }
            }
        }
    }

    Ok(centroid)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn validate_inputs(a: &[f64], n: usize, b: &[f64], m: usize, config: &DtwConfig) -> TsResult<()> {
    if config.n_features == 0 {
        return Err(TsError::InvalidNumVariates(0));
    }
    if n == 0 || m == 0 {
        return Err(TsError::EmptyInput {
            msg: "DTW sequences must be non-empty".to_owned(),
        });
    }
    let expected_a = n * config.n_features;
    if a.len() != expected_a {
        return Err(TsError::ShapeMismatch {
            msg: format!(
                "a: expected {} elements (n={n} f={}), got {}",
                expected_a,
                config.n_features,
                a.len()
            ),
        });
    }
    let expected_b = m * config.n_features;
    if b.len() != expected_b {
        return Err(TsError::ShapeMismatch {
            msg: format!(
                "b: expected {} elements (m={m} f={}), got {}",
                expected_b,
                config.n_features,
                b.len()
            ),
        });
    }
    Ok(())
}

/// Euclidean distance between time step `i` of `a` and step `j` of `b`.
#[inline]
fn point_dist(a: &[f64], i: usize, b: &[f64], j: usize, n_features: usize) -> f64 {
    let a_off = i * n_features;
    let b_off = j * n_features;
    let mut sq = 0.0_f64;
    for f in 0..n_features {
        let d = a[a_off + f] - b[b_off + f];
        sq += d * d;
    }
    sq.sqrt()
}

/// Build the N×M DTW accumulated cost matrix (row-major: D[i*m + j]).
fn build_cost_matrix(a: &[f64], n: usize, b: &[f64], m: usize, config: &DtwConfig) -> Vec<f64> {
    let f = config.n_features;
    let mut d = vec![f64::INFINITY; n * m];

    let in_band = |i: usize, j: usize| -> bool {
        match config.band {
            None => true,
            Some(r) => i.abs_diff(j) <= r,
        }
    };

    // Fill D[0,0]
    if in_band(0, 0) {
        d[0] = point_dist(a, 0, b, 0, f);
    }

    // First row
    for j in 1..m {
        if in_band(0, j) {
            let prev = d[j - 1];
            if prev.is_finite() {
                d[j] = point_dist(a, 0, b, j, f) + prev;
            }
        }
    }

    // First column
    for i in 1..n {
        if in_band(i, 0) {
            let prev = d[(i - 1) * m];
            if prev.is_finite() {
                d[i * m] = point_dist(a, i, b, 0, f) + prev;
            }
        }
    }

    // Interior
    for i in 1..n {
        for j in 1..m {
            if !in_band(i, j) {
                continue;
            }
            let local = point_dist(a, i, b, j, f);
            let prev = d[(i - 1) * m + (j - 1)]
                .min(d[(i - 1) * m + j])
                .min(d[i * m + (j - 1)]);
            if prev.is_finite() {
                d[i * m + j] = local + prev;
            }
        }
    }

    d
}

/// Backtrack through cost matrix from (n-1, m-1) to (0, 0) and return forward path.
fn backtrack_path(cost: &[f64], n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::with_capacity(n + m);
    let mut i = n - 1;
    let mut j = m - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            // Choose predecessor with minimum cost
            let diag = cost[(i - 1) * m + (j - 1)];
            let up = cost[(i - 1) * m + j];
            let left = cost[i * m + (j - 1)];

            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }

    path.reverse();
    path
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> DtwConfig {
        DtwConfig::default()
    }

    fn seq(vals: &[f64]) -> Vec<f64> {
        vals.to_vec()
    }

    #[test]
    fn dtw_self_distance_zero() {
        let a = seq(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let n = 5;
        let cfg = make_cfg();
        let d = dtw_distance(&a, n, &a, n, &cfg).expect("ok");
        assert!(d.abs() < 1e-10, "self-distance should be 0, got {d}");
    }

    #[test]
    fn dtw_symmetric() {
        let a = seq(&[1.0, 3.0, 2.0, 5.0]);
        let b = seq(&[2.0, 4.0, 1.0, 3.0, 2.0]);
        let cfg = make_cfg();
        let dab = dtw_distance(&a, 4, &b, 5, &cfg).expect("ok");
        let dba = dtw_distance(&b, 5, &a, 4, &cfg).expect("ok");
        assert!(
            (dab - dba).abs() < 1e-10,
            "DTW should be symmetric: {dab} vs {dba}"
        );
    }

    #[test]
    fn dtw_non_negative() {
        let a = seq(&[-1.0, 2.0, -3.0]);
        let b = seq(&[4.0, -5.0, 6.0]);
        let cfg = make_cfg();
        let d = dtw_distance(&a, 3, &b, 3, &cfg).expect("ok");
        assert!(d >= 0.0, "DTW distance must be non-negative, got {d}");
    }

    #[test]
    fn dtw_triangle_inequality() {
        let a = seq(&[0.0, 1.0, 2.0]);
        let b = seq(&[1.0, 2.0, 3.0]);
        let c = seq(&[0.0, 2.0, 4.0]);
        let cfg = make_cfg();
        let dab = dtw_distance(&a, 3, &b, 3, &cfg).expect("ok");
        let dbc = dtw_distance(&b, 3, &c, 3, &cfg).expect("ok");
        let dac = dtw_distance(&a, 3, &c, 3, &cfg).expect("ok");
        assert!(
            dac <= dab + dbc + 1e-10,
            "triangle inequality violated: {dac} > {dab} + {dbc}"
        );
    }

    #[test]
    fn dtw_path_starts_and_ends() {
        let a = seq(&[1.0, 2.0, 3.0]);
        let b = seq(&[1.0, 2.0, 3.0, 4.0]);
        let cfg = make_cfg();
        let res = dtw(&a, 3, &b, 4, &cfg).expect("ok");
        assert_eq!(res.path[0], (0, 0), "path must start at (0,0)");
        assert_eq!(
            *res.path.last().expect("non-empty"),
            (2, 3),
            "path must end at (N-1,M-1)"
        );
    }

    #[test]
    fn dtw_path_monotonic() {
        let a: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..7).map(|i| i as f64 * 0.7).collect();
        let cfg = make_cfg();
        let res = dtw(&a, 5, &b, 7, &cfg).expect("ok");
        for w in res.path.windows(2) {
            let (i0, j0) = w[0];
            let (i1, j1) = w[1];
            assert!(
                i1 >= i0 && j1 >= j0 && (i1 > i0 || j1 > j0),
                "path not monotonic at ({i0},{j0}) → ({i1},{j1})"
            );
        }
    }

    #[test]
    fn dtw_path_length_bounds() {
        let a: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let cfg = make_cfg();
        let res = dtw(&a, 4, &b, 6, &cfg).expect("ok");
        assert!(res.path_len >= 6, "path_len >= max(N,M)");
        assert!(res.path_len < 4 + 6, "path_len <= N+M-1");
    }

    #[test]
    fn dtw_normalized_correct() {
        let a: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..4).map(|i| i as f64 + 0.5).collect();
        let mut cfg = make_cfg();
        cfg.normalize = true;
        let res = dtw(&a, 4, &b, 4, &cfg).expect("ok");
        let expected = res.distance / res.path_len as f64;
        assert!(
            (res.normalized_distance - expected).abs() < 1e-10,
            "normalized={} expected={expected}",
            res.normalized_distance
        );
    }

    #[test]
    fn dtw_band_larger_or_equal_unconstrained() {
        let a: Vec<f64> = (0..5).map(|i| (i as f64 * 0.7).sin()).collect();
        let b: Vec<f64> = (0..5).map(|i| (i as f64 * 0.5 + 0.3).cos()).collect();
        let cfg_free = make_cfg();
        let d_free = dtw_distance(&a, 5, &b, 5, &cfg_free).expect("ok");
        let mut cfg_band = make_cfg();
        cfg_band.band = Some(1);
        let d_band = dtw_distance(&a, 5, &b, 5, &cfg_band).expect("ok");
        assert!(
            d_band >= d_free - 1e-10,
            "band should not decrease distance: band={d_band} free={d_free}"
        );
    }

    #[test]
    fn dtw_band_zero_diagonal_path() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![1.5, 2.5, 3.5];
        let mut cfg = make_cfg();
        cfg.band = Some(0);
        let res = dtw(&a, 3, &b, 3, &cfg).expect("ok");
        assert_eq!(res.path, vec![(0, 0), (1, 1), (2, 2)]);
        let expected_dist: f64 = 0.5 + 0.5 + 0.5;
        assert!((res.distance - expected_dist).abs() < 1e-10);
    }

    #[test]
    fn dtw_length_1_self() {
        let a = vec![5.0_f64];
        let cfg = make_cfg();
        let res = dtw(&a, 1, &a, 1, &cfg).expect("ok");
        assert!(res.distance.abs() < 1e-10);
        assert_eq!(res.path, vec![(0, 0)]);
    }

    #[test]
    fn dtw_different_lengths() {
        let a: Vec<f64> = vec![1.0, 2.0];
        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = make_cfg();
        dtw(&a, 2, &b, 5, &cfg).expect("different lengths should work");
    }

    #[test]
    fn dtw_bivariate() {
        let a: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let b: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0];
        let mut cfg = make_cfg();
        cfg.n_features = 2;
        let res = dtw(&a, 3, &b, 2, &cfg).expect("ok");
        assert!(res.distance >= 0.0);
    }

    #[test]
    fn dtw_distance_matrix_shape() {
        let k = 4;
        let l = 5;
        let seqs: Vec<f64> = (0..k * l).map(|i| i as f64).collect();
        let cfg = make_cfg();
        let mat = dtw_distance_matrix(&seqs, k, l, &cfg).expect("ok");
        assert_eq!(mat.len(), k * k);
    }

    #[test]
    fn dtw_distance_matrix_symmetric() {
        let k = 3;
        let l = 4;
        let seqs: Vec<f64> = (0..k * l).map(|i| (i as f64 * 0.3 + 1.0).sin()).collect();
        let cfg = make_cfg();
        let mat = dtw_distance_matrix(&seqs, k, l, &cfg).expect("ok");
        for i in 0..k {
            for j in 0..k {
                let diff = (mat[i * k + j] - mat[j * k + i]).abs();
                assert!(
                    diff < 1e-10,
                    "matrix not symmetric at ({i},{j}): diff={diff}"
                );
            }
        }
    }

    #[test]
    fn dtw_distance_matrix_diagonal_zero() {
        let k = 3;
        let l = 4;
        let seqs: Vec<f64> = (0..k * l).map(|i| i as f64).collect();
        let cfg = make_cfg();
        let mat = dtw_distance_matrix(&seqs, k, l, &cfg).expect("ok");
        for i in 0..k {
            assert!(
                mat[i * k + i].abs() < 1e-10,
                "diagonal[{i}] should be 0, got {}",
                mat[i * k + i]
            );
        }
    }

    #[test]
    fn dtw_barycenter_length() {
        let k = 3;
        let l = 5;
        let seqs: Vec<f64> = (0..k * l).map(|i| i as f64 * 0.1).collect();
        let cfg = make_cfg();
        let bary = dtw_barycenter(&seqs, k, l, 3, &cfg).expect("ok");
        assert_eq!(bary.len(), l);
    }

    #[test]
    fn dtw_barycenter_identical_sequences() {
        let k = 3;
        let l = 4;
        let seq: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let seqs: Vec<f64> = seq.iter().cloned().cycle().take(k * l).collect();
        let cfg = make_cfg();
        let bary = dtw_barycenter(&seqs, k, l, 5, &cfg).expect("ok");
        for (i, (&b, &s)) in bary.iter().zip(seq.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-8,
                "barycenter[{i}]={b} should equal seq[{i}]={s}"
            );
        }
    }

    #[test]
    fn dtw_barycenter_closer_than_arbitrary() {
        let k = 3;
        let l = 5;
        // Three sequences that differ, barycenter should be representative
        let s0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s1: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let s2: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let mut seqs = s0.clone();
        seqs.extend_from_slice(&s1);
        seqs.extend_from_slice(&s2);
        let cfg = make_cfg();
        let bary = dtw_barycenter(&seqs, k, l, 5, &cfg).expect("ok");

        let avg_dtw_to = |center: &[f64]| {
            let stride = l;
            let mut total = 0.0_f64;
            for ki in 0..k {
                let seq = &seqs[ki * stride..(ki + 1) * stride];
                let cost = build_cost_matrix(seq, l, center, l, &cfg);
                total += cost[(l - 1) * l + (l - 1)];
            }
            total / k as f64
        };

        let avg_bary = avg_dtw_to(&bary);
        let avg_first = avg_dtw_to(&s0);
        assert!(
            avg_bary <= avg_first + 1e-6,
            "barycenter avg DTW ({avg_bary}) should be ≤ first seq avg DTW ({avg_first})"
        );
    }

    #[test]
    fn dtw_err_empty_input() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0_f64];
        let cfg = make_cfg();
        assert!(matches!(
            dtw(&a, 0, &b, 1, &cfg).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn dtw_err_n_features_zero() {
        let a = vec![1.0_f64];
        let b = vec![1.0_f64];
        let mut cfg = make_cfg();
        cfg.n_features = 0;
        assert!(matches!(
            dtw(&a, 1, &b, 1, &cfg).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn dtw_err_slice_length_mismatch() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![1.0_f64, 2.0];
        let cfg = make_cfg();
        // n=2 but a has 3 elements → mismatch
        assert!(matches!(
            dtw(&a, 2, &b, 2, &cfg).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }
}
