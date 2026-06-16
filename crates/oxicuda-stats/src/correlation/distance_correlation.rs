//! Distance correlation (Székely, Rizzo & Bakirov 2007) and its bias-corrected
//! variant (Székely & Rizzo 2013).
//!
//! Distance correlation `dCor(X, Y) ∈ [0, 1]` measures *general* (not merely
//! linear) statistical dependence between two random vectors of arbitrary —
//! and possibly unequal — dimension. The population distance correlation is
//! zero **iff** `X` and `Y` are independent, a property Pearson correlation
//! lacks.
//!
//! # Construction
//! For samples `x₁…xₙ` and `y₁…yₙ`, form the pairwise Euclidean distance
//! matrices `a_{ij} = ‖xᵢ − xⱼ‖` and `b_{ij} = ‖yᵢ − yⱼ‖`, then *double-centre*
//! them:
//!
//! ```text
//! A_{ij} = a_{ij} − ā_{i·} − ā_{·j} + ā_{··}
//! ```
//!
//! The sample (squared) distance covariance and variances are
//!
//! ```text
//! dCov²(X,Y) = (1/n²) Σ A_{ij} B_{ij},   dVar²(X) = dCov²(X,X)
//! dCor²(X,Y) = dCov²(X,Y) / √(dVar²(X) dVar²(Y))
//! ```
//!
//! The bias-corrected estimator uses U-centring instead and is an unbiased
//! estimator of the squared population distance covariance, so it fluctuates
//! around zero under independence (and may be negative).
//!
//! # References
//! - Székely, G. J., Rizzo, M. L. & Bakirov, N. K. (2007). *Measuring and
//!   testing dependence by correlation of distances*. Ann. Statist. 35(6):
//!   2769-2794.
//! - Székely, G. J. & Rizzo, M. L. (2013). *The distance correlation t-test of
//!   independence in high dimension*. J. Multivar. Anal. 117: 193-213.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Result of a full distance-correlation computation.
#[derive(Debug, Clone, Copy)]
pub struct DistanceCorrelation {
    /// Distance correlation `dCor ∈ [0, 1]`.
    pub dcor: f64,
    /// Distance covariance `dCov = √(dCov²)`.
    pub dcov: f64,
    /// Distance standard deviation of `X`, `dVar(X) = √(dVar²(X))`.
    pub dvar_x: f64,
    /// Distance standard deviation of `Y`.
    pub dvar_y: f64,
    /// Number of observations.
    pub n: usize,
}

/// Result of a permutation-based independence test.
#[derive(Debug, Clone, Copy)]
pub struct DistanceTestResult {
    /// Observed sample squared distance covariance `dCov²(X, Y)`.
    pub statistic: f64,
    /// Permutation p-value `(#{permuted ≥ observed} + 1) / (n_perm + 1)`.
    pub p_value: f64,
    /// Number of permutations used.
    pub n_perm: usize,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_block(data: &[f64], n: usize, dim: usize, label: &str) -> StatsResult<()> {
    if dim == 0 {
        return Err(StatsError::InvalidParameter {
            name: label.to_owned(),
            reason: "dimension must be ≥ 1".to_owned(),
        });
    }
    if data.len() != n * dim {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n * dim],
            got: vec![data.len()],
        });
    }
    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Distance matrices and centring
// ---------------------------------------------------------------------------

/// Pairwise Euclidean distance matrix for a row-major `n × dim` block.
fn euclidean_distance_matrix(data: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..dim {
                let diff = data[i * dim + k] - data[j * dim + k];
                s += diff * diff;
            }
            let dist = s.sqrt();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Double-centred distance matrix (Székely et al. 2007).
fn double_center(a: &[f64], n: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = a[i * n + j];
            row_mean[i] += val;
            col_mean[j] += val;
            grand += val;
        }
    }
    for r in row_mean.iter_mut() {
        *r /= nf;
    }
    for c in col_mean.iter_mut() {
        *c /= nf;
    }
    grand /= nf * nf;
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = a[i * n + j] - row_mean[i] - col_mean[j] + grand;
        }
    }
    b
}

/// U-centred distance matrix for the bias-corrected estimator (requires n ≥ 3).
fn u_center(a: &[f64], n: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut row_sum = vec![0.0; n];
    let mut col_sum = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = a[i * n + j];
            row_sum[i] += val;
            col_sum[j] += val;
            grand += val;
        }
    }
    let inv_nm2 = 1.0 / (nf - 2.0);
    let inv_grand = 1.0 / ((nf - 1.0) * (nf - 2.0));
    let mut ac = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                ac[i * n + j] =
                    a[i * n + j] - row_sum[i] * inv_nm2 - col_sum[j] * inv_nm2 + grand * inv_grand;
            }
        }
    }
    ac
}

fn hadamard_mean(a: &[f64], b: &[f64], n: usize) -> f64 {
    let total: f64 = a.iter().zip(b.iter()).map(|(p, q)| p * q).sum();
    total / (n as f64 * n as f64)
}

fn u_inner(a: &[f64], b: &[f64], n: usize) -> f64 {
    let total: f64 = a.iter().zip(b.iter()).map(|(p, q)| p * q).sum();
    total / (n as f64 * (n as f64 - 3.0))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Full distance-correlation analysis for two row-major blocks
/// (`x` is `n × px`, `y` is `n × qy`).
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::InvalidParameter`] if `px == 0` or `qy == 0`.
/// - [`StatsError::ShapeMismatch`] if a block length disagrees with `n × dim`.
/// - [`StatsError::NonFiniteValue`] if any entry is non-finite.
pub fn distance_correlation_full(
    x: &[f64],
    px: usize,
    y: &[f64],
    qy: usize,
    n: usize,
) -> StatsResult<DistanceCorrelation> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    validate_block(x, n, px, "px")?;
    validate_block(y, n, qy, "qy")?;
    let dx = euclidean_distance_matrix(x, n, px);
    let dy = euclidean_distance_matrix(y, n, qy);
    let a = double_center(&dx, n);
    let b = double_center(&dy, n);
    let dcov2 = hadamard_mean(&a, &b, n);
    let dvarx2 = hadamard_mean(&a, &a, n);
    let dvary2 = hadamard_mean(&b, &b, n);
    let dcor = if dvarx2 > 0.0 && dvary2 > 0.0 {
        (dcov2 / (dvarx2 * dvary2).sqrt()).max(0.0).sqrt()
    } else {
        0.0
    };
    Ok(DistanceCorrelation {
        dcor,
        dcov: dcov2.max(0.0).sqrt(),
        dvar_x: dvarx2.max(0.0).sqrt(),
        dvar_y: dvary2.max(0.0).sqrt(),
        n,
    })
}

/// Distance correlation between two univariate samples.
///
/// # Errors
/// [`StatsError::DimensionMismatch`] if the samples differ in length, plus the
/// errors of [`distance_correlation_full`].
pub fn distance_correlation(x: &[f64], y: &[f64]) -> StatsResult<f64> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let n = x.len();
    Ok(distance_correlation_full(x, 1, y, 1, n)?.dcor)
}

/// Distance covariance between two univariate samples.
///
/// # Errors
/// As [`distance_correlation`].
pub fn distance_covariance(x: &[f64], y: &[f64]) -> StatsResult<f64> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let n = x.len();
    Ok(distance_correlation_full(x, 1, y, 1, n)?.dcov)
}

/// Bias-corrected distance correlation (Székely & Rizzo 2013).
///
/// Returns a value in `[-1, 1]` that is an unbiased-in-square estimator and
/// therefore fluctuates around zero under independence (it can be negative).
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 4`.
/// - Otherwise as [`distance_correlation_full`].
pub fn bias_corrected_distance_correlation(
    x: &[f64],
    px: usize,
    y: &[f64],
    qy: usize,
    n: usize,
) -> StatsResult<f64> {
    if n < 4 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 4 });
    }
    validate_block(x, n, px, "px")?;
    validate_block(y, n, qy, "qy")?;
    let dx = euclidean_distance_matrix(x, n, px);
    let dy = euclidean_distance_matrix(y, n, qy);
    let ac = u_center(&dx, n);
    let bc = u_center(&dy, n);
    let dcov_xy = u_inner(&ac, &bc, n);
    let dvar_x = u_inner(&ac, &ac, n);
    let dvar_y = u_inner(&bc, &bc, n);
    let prod = dvar_x * dvar_y;
    let bcdcor = if prod > 0.0 {
        (dcov_xy / prod.sqrt()).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    Ok(bcdcor)
}

/// Permutation test of independence based on the distance covariance.
///
/// The y-sample labels are randomly permuted `n_perm` times; the p-value is the
/// proportion of permuted statistics at least as large as the observed one
/// (with the usual `+1` correction).
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::InvalidParameter`] if `n_perm == 0`.
/// - Otherwise as [`distance_correlation_full`].
pub fn distance_covariance_test(
    x: &[f64],
    px: usize,
    y: &[f64],
    qy: usize,
    n: usize,
    n_perm: usize,
    rng: &mut LcgRng,
) -> StatsResult<DistanceTestResult> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if n_perm == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_perm".to_owned(),
            reason: "must be ≥ 1".to_owned(),
        });
    }
    validate_block(x, n, px, "px")?;
    validate_block(y, n, qy, "qy")?;
    let dx = euclidean_distance_matrix(x, n, px);
    let dy = euclidean_distance_matrix(y, n, qy);
    let a = double_center(&dx, n);
    let b = double_center(&dy, n);
    let observed = hadamard_mean(&a, &b, n);

    let mut perm: Vec<usize> = (0..n).collect();
    let mut dyp = vec![0.0; n * n];
    let mut count = 0usize;
    for _ in 0..n_perm {
        // Fisher-Yates shuffle of the sample labels.
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            perm.swap(i, j);
        }
        for i in 0..n {
            for j in 0..n {
                dyp[i * n + j] = dy[perm[i] * n + perm[j]];
            }
        }
        let bp = double_center(&dyp, n);
        let stat = hadamard_mean(&a, &bp, n);
        if stat >= observed {
            count += 1;
        }
    }
    let p_value = (count as f64 + 1.0) / (n_perm as f64 + 1.0);
    Ok(DistanceTestResult {
        statistic: observed,
        p_value,
        n_perm,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_relationship_gives_unit_correlation() {
        // dCor(X, a + bX) = 1 exactly.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let dcor = distance_correlation(&x, &y).expect("ok");
        assert!(
            (dcor - 1.0).abs() < 1e-9,
            "affine dCor should be 1, got {dcor}"
        );
    }

    #[test]
    fn identical_samples_give_unit_correlation() {
        let x = [3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0];
        let dcor = distance_correlation(&x, &x).expect("ok");
        assert!((dcor - 1.0).abs() < 1e-9, "dCor(X,X) = 1, got {dcor}");
    }

    #[test]
    fn correlation_in_unit_interval_and_symmetric() {
        let mut rng = LcgRng::new(11);
        let x: Vec<f64> = (0..40).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..40).map(|_| rng.next_f64()).collect();
        let dxy = distance_correlation(&x, &y).expect("ok");
        let dyx = distance_correlation(&y, &x).expect("ok");
        assert!(
            (0.0..=1.0).contains(&dxy),
            "dCor must be in [0,1], got {dxy}"
        );
        assert!((dxy - dyx).abs() < 1e-12, "dCor must be symmetric");
    }

    #[test]
    fn detects_nonlinear_dependence() {
        // y = x² on a symmetric grid: Pearson ≈ 0 but the dependence is real.
        let x: Vec<f64> = (0..15).map(|i| i as f64 - 7.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let dcor = distance_correlation(&x, &y).expect("ok");
        assert!(dcor > 0.2, "should detect quadratic dependence, got {dcor}");
    }

    #[test]
    fn dcov_and_dvar_non_negative() {
        let mut rng = LcgRng::new(5);
        let x: Vec<f64> = (0..30).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..30).map(|_| rng.next_f64()).collect();
        let r = distance_correlation_full(&x, 1, &y, 1, 30).expect("ok");
        assert!(r.dcov >= 0.0 && r.dvar_x >= 0.0 && r.dvar_y >= 0.0);
    }

    #[test]
    fn multivariate_block_runs() {
        // x: 20 × 2, y: 20 × 3.
        let mut rng = LcgRng::new(73);
        let x: Vec<f64> = (0..40).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..60).map(|_| rng.next_f64()).collect();
        let r = distance_correlation_full(&x, 2, &y, 3, 20).expect("ok");
        assert!((0.0..=1.0).contains(&r.dcor));
    }

    #[test]
    fn bias_corrected_affine_is_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let bc = bias_corrected_distance_correlation(&x, 1, &y, 1, 8).expect("ok");
        assert!(
            (bc - 1.0).abs() < 1e-9,
            "bias-corrected affine = 1, got {bc}"
        );
    }

    #[test]
    fn bias_corrected_near_zero_under_independence() {
        let mut rng = LcgRng::new(424242);
        let x: Vec<f64> = (0..80).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..80).map(|_| rng.next_f64()).collect();
        let bc = bias_corrected_distance_correlation(&x, 1, &y, 1, 80).expect("ok");
        assert!(
            bc.abs() < 0.3,
            "independent bias-corrected dCor ≈ 0, got {bc}"
        );
    }

    #[test]
    fn permutation_test_detects_dependence() {
        // Perfect dependence ⇒ small p-value.
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let mut rng = LcgRng::new(2718);
        let res = distance_covariance_test(&x, 1, &y, 1, 20, 99, &mut rng).expect("ok");
        assert!(
            res.p_value < 0.05,
            "p-value should be small, got {}",
            res.p_value
        );
        assert!(res.statistic > 0.0);
    }

    #[test]
    fn permutation_test_pvalue_in_range() {
        let mut rng = LcgRng::new(31415);
        let x: Vec<f64> = (0..25).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..25).map(|_| rng.next_f64()).collect();
        let res = distance_covariance_test(&x, 1, &y, 1, 25, 99, &mut rng).expect("ok");
        assert!(res.p_value > 0.0 && res.p_value <= 1.0);
    }

    #[test]
    fn shape_and_size_errors() {
        // length mismatch (univariate).
        assert!(matches!(
            distance_correlation(&[1.0, 2.0, 3.0], &[1.0, 2.0]),
            Err(StatsError::DimensionMismatch { .. })
        ));
        // n < 2.
        assert!(matches!(
            distance_correlation_full(&[1.0], 1, &[2.0], 1, 1),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
        // dim mismatch with n.
        assert!(matches!(
            distance_correlation_full(&[1.0, 2.0, 3.0], 2, &[1.0, 2.0], 1, 2),
            Err(StatsError::ShapeMismatch { .. })
        ));
        // bias-corrected needs n ≥ 4.
        assert!(matches!(
            bias_corrected_distance_correlation(&[1.0, 2.0, 3.0], 1, &[1.0, 2.0, 3.0], 1, 3),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
        // dim = 0.
        assert!(matches!(
            distance_correlation_full(&[], 0, &[], 0, 2),
            Err(StatsError::InvalidParameter { .. })
        ));
    }
}
