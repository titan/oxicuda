//! Offline change-point detection.
//!
//! References:
//! - Killick, Fearnhead & Eckley (2012) "Optimal Detection of Changepoints With
//!   a Linear Computational Cost." JASA 107(500):1590-1598. (PELT)
//! - Scott & Knott (1974) / Sen & Srivastava (1975) — binary segmentation.
//! - Page (1954) "Continuous Inspection Schemes." Biometrika 41. (CUSUM)
//!
//! A change point is a time index where the statistical properties of a series
//! shift.  This module detects changes in the **mean** of a univariate series
//! using the segment cost
//!
//! ```text
//! C([a, b)) = Σ_{t=a}^{b-1} (y_t − μ_{[a,b)})²      (within-segment SSE),
//! ```
//!
//! which is the maximum-likelihood cost for a piecewise-constant Gaussian mean
//! with constant variance.  Three algorithms are provided:
//!
//! - **PELT** — exact dynamic program with pruning; minimises total segment cost
//!   plus a per-changepoint penalty `β`, running in near-linear time.
//! - **Binary segmentation** — greedy recursive splitting at the most
//!   significant change, fast and approximate.
//! - **CUSUM** — cumulative-sum control statistic for detecting a single shift,
//!   the classic streaming-style detector.
//!
//! All segment costs are computed in `O(1)` from prefix sums of `y` and `y²`.
use crate::error::{TsError, TsResult};

/// Prefix sums enabling `O(1)` segment-SSE queries.
struct PrefixStats {
    /// `csum[i] = Σ_{t<i} y_t`, length `n + 1`.
    csum: Vec<f64>,
    /// `csum2[i] = Σ_{t<i} y_t²`, length `n + 1`.
    csum2: Vec<f64>,
}

impl PrefixStats {
    fn new(y: &[f64]) -> Self {
        let n = y.len();
        let mut csum = vec![0.0_f64; n + 1];
        let mut csum2 = vec![0.0_f64; n + 1];
        for i in 0..n {
            csum[i + 1] = csum[i] + y[i];
            csum2[i + 1] = csum2[i] + y[i] * y[i];
        }
        Self { csum, csum2 }
    }

    /// Within-segment SSE of `y[a..b]` (half-open). `0` for empty segments.
    #[inline]
    fn segment_cost(&self, a: usize, b: usize) -> f64 {
        if b <= a {
            return 0.0;
        }
        let len = (b - a) as f64;
        let sum = self.csum[b] - self.csum[a];
        let sum2 = self.csum2[b] - self.csum2[a];
        // SSE = Σy² − (Σy)²/len.
        (sum2 - sum * sum / len).max(0.0)
    }
}

/// Configuration for [`pelt`].
#[derive(Debug, Clone, Copy)]
pub struct PeltConfig {
    /// Per-changepoint penalty `β`. Larger ⇒ fewer change points. A common
    /// default is the BIC penalty `β = (k+1)·ln(n)` with `k = 1` parameter.
    pub penalty: f64,
    /// Minimum segment length (≥ 1). Segments shorter than this are disallowed.
    pub min_size: usize,
}

impl PeltConfig {
    /// BIC-penalised config for a series of length `n` (penalty `2·ln n`).
    #[must_use]
    pub fn bic(n: usize) -> Self {
        let penalty = if n >= 2 { 2.0 * (n as f64).ln() } else { 1.0 };
        Self {
            penalty,
            min_size: 1,
        }
    }
}

/// Detect change points with the PELT exact dynamic program.
///
/// Returns the **interior** change-point indices (strictly between `0` and `n`),
/// sorted ascending; an empty result means no change.  The objective minimised
/// is `Σ_segments C(segment) + penalty · (#changepoints)`.
///
/// # Errors
/// - [`TsError::EmptyInput`] when `y` is empty.
/// - [`TsError::NonFinite`] when any value is non-finite.
/// - [`TsError::Internal`] when `min_size == 0` or `penalty` is non-finite.
pub fn pelt(y: &[f64], config: &PeltConfig) -> TsResult<Vec<usize>> {
    if y.is_empty() {
        return Err(TsError::EmptyInput {
            msg: "pelt: empty series".to_string(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TsError::NonFinite);
    }
    if config.min_size == 0 || !config.penalty.is_finite() {
        return Err(TsError::Internal(
            "pelt: min_size must be ≥ 1 and penalty finite".to_string(),
        ));
    }
    let n = y.len();
    let stats = PrefixStats::new(y);
    let min_size = config.min_size.max(1);
    let beta = config.penalty;

    // f[t] = minimal penalised cost of segmenting y[0..t].
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = -beta; // so the first segment incurs no penalty.
    // last_cp[t] = the last change point chosen in the optimal segmentation of y[0..t].
    let mut last_cp = vec![0usize; n + 1];
    // Candidate set R for pruning.
    let mut candidates: Vec<usize> = vec![0];

    for t in min_size..=n {
        let mut best = f64::INFINITY;
        let mut best_s = 0usize;
        for &s in &candidates {
            if t - s < min_size {
                continue;
            }
            let cost = f[s] + stats.segment_cost(s, t) + beta;
            if cost < best {
                best = cost;
                best_s = s;
            }
        }
        f[t] = best;
        last_cp[t] = best_s;

        // PELT pruning: drop candidates s for which
        // f[s] + C(s, t) + K ≥ f[t]  (with K = 0 for this cost) can never win.
        let threshold = f[t];
        candidates.retain(|&s| f[s] + stats.segment_cost(s, t) <= threshold);
        candidates.push(t);
    }

    // Backtrack the change points.
    let mut cps = Vec::new();
    let mut t = n;
    while t > 0 {
        let s = last_cp[t];
        if s > 0 {
            cps.push(s);
        }
        if s == t {
            break;
        }
        t = s;
    }
    cps.sort_unstable();
    cps.dedup();
    Ok(cps)
}

/// Configuration for [`binary_segmentation`].
#[derive(Debug, Clone, Copy)]
pub struct BinSegConfig {
    /// Minimum reduction in SSE required to accept a split.
    pub min_gain: f64,
    /// Minimum segment length (≥ 1).
    pub min_size: usize,
    /// Maximum number of change points to return (`0` = unlimited).
    pub max_changepoints: usize,
}

impl BinSegConfig {
    /// A config whose `min_gain` is a fraction of the total series SSE.
    #[must_use]
    pub fn new(min_gain: f64) -> Self {
        Self {
            min_gain,
            min_size: 1,
            max_changepoints: 0,
        }
    }
}

/// Detect change points by greedy binary segmentation.
///
/// Recursively finds the single split that most reduces within-segment SSE; if
/// the reduction exceeds `min_gain` the split is accepted and both halves are
/// recursed.  Returns interior change points sorted ascending.
///
/// # Errors
/// - [`TsError::EmptyInput`] when `y` is empty.
/// - [`TsError::NonFinite`] when any value is non-finite.
/// - [`TsError::Internal`] when `min_size == 0` or `min_gain` is non-finite.
pub fn binary_segmentation(y: &[f64], config: &BinSegConfig) -> TsResult<Vec<usize>> {
    if y.is_empty() {
        return Err(TsError::EmptyInput {
            msg: "binseg: empty series".to_string(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TsError::NonFinite);
    }
    if config.min_size == 0 || !config.min_gain.is_finite() {
        return Err(TsError::Internal(
            "binseg: min_size must be ≥ 1 and min_gain finite".to_string(),
        ));
    }
    let n = y.len();
    let stats = PrefixStats::new(y);
    let mut cps = Vec::new();

    // Stack of segments [a, b) to consider.
    let mut stack = vec![(0usize, n)];
    while let Some((a, b)) = stack.pop() {
        if config.max_changepoints != 0 && cps.len() >= config.max_changepoints {
            break;
        }
        if let Some((split, gain)) = best_split(&stats, a, b, config.min_size) {
            if gain > config.min_gain {
                cps.push(split);
                stack.push((a, split));
                stack.push((split, b));
            }
        }
    }
    cps.sort_unstable();
    cps.truncate(if config.max_changepoints == 0 {
        cps.len()
    } else {
        config.max_changepoints
    });
    Ok(cps)
}

/// Find the best split point in `[a, b)` and the SSE reduction it yields.
fn best_split(stats: &PrefixStats, a: usize, b: usize, min_size: usize) -> Option<(usize, f64)> {
    if b - a < 2 * min_size {
        return None;
    }
    let whole = stats.segment_cost(a, b);
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split = a + min_size;
    for split in (a + min_size)..=(b - min_size) {
        let gain = whole - stats.segment_cost(a, split) - stats.segment_cost(split, b);
        if gain > best_gain {
            best_gain = gain;
            best_split = split;
        }
    }
    if best_gain.is_finite() {
        Some((best_split, best_gain))
    } else {
        None
    }
}

/// Result of a [`cusum`] scan.
#[derive(Debug, Clone)]
pub struct CusumResult {
    /// Index of the maximum cumulative-sum deviation (the estimated change
    /// location), or `None` when no value crossed the threshold.
    pub changepoint: Option<usize>,
    /// The maximum absolute cumulative deviation `S_max`.
    pub max_statistic: f64,
    /// Full cumulative-deviation path `S_t = Σ_{i≤t} (y_i − ȳ)`, length `n`.
    pub cumulative: Vec<f64>,
}

/// Single-shift change detection via the CUSUM statistic.
///
/// Computes the running cumulative deviation from the global mean,
/// `S_t = Σ_{i=0}^{t} (y_i − ȳ)`, and reports the index of maximum `|S_t|` as the
/// most likely change location when `S_max` exceeds `threshold`.  This is the
/// canonical detector for a single mean shift.
///
/// # Errors
/// - [`TsError::EmptyInput`] when `y` is empty.
/// - [`TsError::NonFinite`] when any value is non-finite or `threshold < 0`.
pub fn cusum(y: &[f64], threshold: f64) -> TsResult<CusumResult> {
    if y.is_empty() {
        return Err(TsError::EmptyInput {
            msg: "cusum: empty series".to_string(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) || !(threshold >= 0.0 && threshold.is_finite()) {
        return Err(TsError::NonFinite);
    }
    let n = y.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let mut cumulative = vec![0.0_f64; n];
    let mut running = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut argmax = 0usize;
    for t in 0..n {
        running += y[t] - mean;
        cumulative[t] = running;
        if running.abs() > max_abs {
            max_abs = running.abs();
            argmax = t;
        }
    }
    let changepoint = if max_abs > threshold {
        Some(argmax)
    } else {
        None
    };
    Ok(CusumResult {
        changepoint,
        max_statistic: max_abs,
        cumulative,
    })
}

/// Segment means implied by a change-point set, as `(start, end, mean)` tuples.
///
/// Convenience for reconstructing the piecewise-constant signal from detected
/// change points.
///
/// # Errors
/// - [`TsError::EmptyInput`] when `y` is empty.
/// - [`TsError::Internal`] when change points are not strictly increasing within
///   `(0, n)`.
pub fn segment_means(y: &[f64], changepoints: &[usize]) -> TsResult<Vec<(usize, usize, f64)>> {
    if y.is_empty() {
        return Err(TsError::EmptyInput {
            msg: "segment_means: empty series".to_string(),
        });
    }
    let n = y.len();
    let mut bounds = Vec::with_capacity(changepoints.len() + 2);
    bounds.push(0usize);
    let mut prev = 0usize;
    for &cp in changepoints {
        if cp == 0 || cp >= n || cp <= prev {
            return Err(TsError::Internal(format!(
                "segment_means: invalid changepoint {cp} for n={n}"
            )));
        }
        bounds.push(cp);
        prev = cp;
    }
    bounds.push(n);

    let mut out = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mean = y[a..b].iter().sum::<f64>() / (b - a) as f64;
        out.push((a, b, mean));
    }
    Ok(out)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Two segments: `len0` samples at `m0`, then `len1` samples at `m1`.
    fn two_level(len0: usize, m0: f64, len1: usize, m1: f64) -> Vec<f64> {
        let mut v = vec![m0; len0];
        v.extend(std::iter::repeat_n(m1, len1));
        v
    }

    #[test]
    fn pelt_detects_single_shift() {
        let y = two_level(30, 0.0, 30, 10.0);
        let cfg = PeltConfig::bic(y.len());
        let cps = pelt(&y, &cfg).expect("pelt should succeed");
        assert_eq!(
            cps.len(),
            1,
            "expected exactly one change point, got {cps:?}"
        );
        assert_eq!(cps[0], 30, "change point at wrong index: {cps:?}");
    }

    #[test]
    fn pelt_no_change_constant() {
        let y = vec![5.0_f64; 50];
        let cfg = PeltConfig::bic(y.len());
        let cps = pelt(&y, &cfg).expect("pelt should succeed");
        assert!(
            cps.is_empty(),
            "constant series should have no change points"
        );
    }

    #[test]
    fn pelt_two_shifts() {
        let mut y = vec![0.0_f64; 30];
        y.extend(std::iter::repeat_n(10.0, 30));
        y.extend(std::iter::repeat_n(-5.0, 30));
        let cfg = PeltConfig::bic(y.len());
        let cps = pelt(&y, &cfg).expect("pelt should succeed");
        assert_eq!(cps.len(), 2, "expected two change points, got {cps:?}");
        assert_eq!(cps, vec![30, 60]);
    }

    #[test]
    fn pelt_higher_penalty_fewer_cps() {
        let mut y = vec![0.0_f64; 20];
        y.extend(std::iter::repeat_n(3.0, 20));
        y.extend(std::iter::repeat_n(6.0, 20));
        let low = PeltConfig {
            penalty: 5.0,
            min_size: 1,
        };
        let high = PeltConfig {
            penalty: 1e6,
            min_size: 1,
        };
        let cps_low = pelt(&y, &low).expect("pelt should succeed");
        let cps_high = pelt(&y, &high).expect("pelt should succeed");
        assert!(
            cps_high.len() <= cps_low.len(),
            "higher penalty gave more CPs"
        );
        assert!(cps_high.is_empty(), "huge penalty should suppress all CPs");
    }

    #[test]
    fn pelt_respects_min_size() {
        let y = two_level(2, 0.0, 40, 10.0);
        let cfg = PeltConfig {
            penalty: 5.0,
            min_size: 5,
        };
        let cps = pelt(&y, &cfg).expect("pelt should succeed");
        // Any detected change point must leave segments ≥ min_size.
        let n = y.len();
        for &cp in &cps {
            assert!(cp >= 5 && n - cp >= 5, "min_size violated at {cp}");
        }
    }

    #[test]
    fn pelt_changepoints_sorted() {
        let mut y = vec![0.0_f64; 25];
        y.extend(std::iter::repeat_n(5.0, 25));
        y.extend(std::iter::repeat_n(0.0, 25));
        let cps = pelt(&y, &PeltConfig::bic(y.len())).expect("value should be present");
        for w in cps.windows(2) {
            assert!(w[0] < w[1], "change points not sorted");
        }
    }

    #[test]
    fn binseg_detects_single_shift() {
        let y = two_level(30, 0.0, 30, 8.0);
        // Total SSE is large; require a meaningful gain.
        let cfg = BinSegConfig {
            min_gain: 50.0,
            min_size: 3,
            max_changepoints: 0,
        };
        let cps = binary_segmentation(&y, &cfg).expect("binary_segmentation should succeed");
        assert!(cps.contains(&30), "binseg missed the shift: {cps:?}");
    }

    #[test]
    fn binseg_no_change_constant() {
        let y = vec![2.0_f64; 40];
        let cfg = BinSegConfig::new(1.0);
        let cps = binary_segmentation(&y, &cfg).expect("binary_segmentation should succeed");
        assert!(cps.is_empty());
    }

    #[test]
    fn binseg_max_changepoints_cap() {
        let mut y = vec![0.0_f64; 20];
        for k in 1..6 {
            y.extend(std::iter::repeat_n(k as f64 * 5.0, 20));
        }
        let cfg = BinSegConfig {
            min_gain: 1.0,
            min_size: 3,
            max_changepoints: 2,
        };
        let cps = binary_segmentation(&y, &cfg).expect("binary_segmentation should succeed");
        assert!(cps.len() <= 2, "cap not honoured: {cps:?}");
    }

    #[test]
    fn binseg_two_shifts() {
        let mut y = vec![0.0_f64; 30];
        y.extend(std::iter::repeat_n(10.0, 30));
        y.extend(std::iter::repeat_n(20.0, 30));
        let cfg = BinSegConfig {
            min_gain: 50.0,
            min_size: 5,
            max_changepoints: 0,
        };
        let cps = binary_segmentation(&y, &cfg).expect("binary_segmentation should succeed");
        assert!(
            cps.contains(&30) && cps.contains(&60),
            "missed shifts: {cps:?}"
        );
    }

    #[test]
    fn binseg_sorted_output() {
        let mut y = vec![0.0_f64; 25];
        y.extend(std::iter::repeat_n(7.0, 25));
        y.extend(std::iter::repeat_n(14.0, 25));
        let cfg = BinSegConfig {
            min_gain: 20.0,
            min_size: 4,
            max_changepoints: 0,
        };
        let cps = binary_segmentation(&y, &cfg).expect("binary_segmentation should succeed");
        for w in cps.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn cusum_detects_shift_location() {
        let y = two_level(40, 0.0, 40, 5.0);
        let res = cusum(&y, 1.0).expect("cusum should succeed");
        assert!(res.changepoint.is_some());
        let cp = res.changepoint.expect("changepoint should be present");
        // CUSUM peaks near the true change at index 40 (within a few samples).
        assert!((cp as isize - 39).abs() <= 3, "CUSUM cp {cp} far from 40");
    }

    #[test]
    fn cusum_no_change_below_threshold() {
        let y = vec![3.0_f64; 50];
        let res = cusum(&y, 0.5).expect("cusum should succeed");
        assert!(res.changepoint.is_none(), "flat series flagged a change");
        assert!(res.max_statistic < 1e-9);
    }

    #[test]
    fn cusum_high_threshold_suppresses() {
        let y = two_level(30, 0.0, 30, 2.0);
        let res = cusum(&y, 1e6).expect("cusum should succeed");
        assert!(res.changepoint.is_none(), "huge threshold should suppress");
    }

    #[test]
    fn cusum_cumulative_length() {
        let y = two_level(20, 1.0, 20, 4.0);
        let res = cusum(&y, 0.0).expect("cusum should succeed");
        assert_eq!(res.cumulative.len(), 40);
        // Cumulative deviation always returns near 0 at the end (sums to ~0).
        assert!(res.cumulative.last().expect("last should succeed").abs() < 1e-6);
    }

    #[test]
    fn segment_means_reconstruction() {
        let y = two_level(20, 1.0, 20, 9.0);
        let segs = segment_means(&y, &[20]).expect("segment_means should succeed");
        assert_eq!(segs.len(), 2);
        assert!((segs[0].2 - 1.0).abs() < 1e-9);
        assert!((segs[1].2 - 9.0).abs() < 1e-9);
        assert_eq!(segs[0], (0, 20, 1.0));
    }

    #[test]
    fn segment_means_no_changepoints() {
        let y = vec![4.0_f64; 10];
        let segs = segment_means(&y, &[]).expect("segment_means should succeed");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0], (0, 10, 4.0));
    }

    #[test]
    fn pelt_err_empty() {
        let cfg = PeltConfig::bic(10);
        assert!(matches!(
            pelt(&[], &cfg).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn pelt_err_nonfinite() {
        let y = vec![1.0, f64::NAN, 3.0];
        let cfg = PeltConfig::bic(3);
        assert!(matches!(pelt(&y, &cfg).unwrap_err(), TsError::NonFinite));
    }

    #[test]
    fn pelt_err_zero_min_size() {
        let y = vec![1.0_f64; 10];
        let cfg = PeltConfig {
            penalty: 5.0,
            min_size: 0,
        };
        assert!(matches!(pelt(&y, &cfg).unwrap_err(), TsError::Internal(_)));
    }

    #[test]
    fn binseg_err_empty() {
        let cfg = BinSegConfig::new(1.0);
        assert!(matches!(
            binary_segmentation(&[], &cfg).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn cusum_err_empty() {
        assert!(matches!(
            cusum(&[], 1.0).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn cusum_err_negative_threshold() {
        let y = vec![1.0_f64; 10];
        assert!(matches!(cusum(&y, -1.0).unwrap_err(), TsError::NonFinite));
    }

    #[test]
    fn segment_means_err_bad_cp() {
        let y = vec![1.0_f64; 10];
        // Change point out of (0, n).
        assert!(matches!(
            segment_means(&y, &[10]).unwrap_err(),
            TsError::Internal(_)
        ));
    }
}
