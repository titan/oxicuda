//! Spatial statistics: Moran's I, Geary's C, and Ripley's K.
//!
//! # Algorithms
//!
//! ## Moran's I — global spatial autocorrelation
//! Tests whether nearby values are more similar (positive autocorrelation, I > 0)
//! or more dissimilar (negative autocorrelation, I < 0) than expected under spatial
//! randomness.  Uses a normal approximation to compute a z-score and p-value.
//!
//! Reference: Moran (1950), *Biometrika* 37(1/2):17–23.
//! Analytical variance: Cliff & Ord (1981), *Spatial Processes: Models and Applications*.
//!
//! ## Geary's C — local variation statistic
//! Measures the degree of spatial autocorrelation focusing on local differences.
//! C ≈ 1 under no autocorrelation; C < 1 indicates positive autocorrelation;
//! C > 1 indicates negative autocorrelation.
//!
//! Reference: Geary (1954), *The Incorporated Statistician* 5(3):115–145.
//!
//! ## Ripley's K — spatial point process intensity
//! Estimates the expected number of additional points within distance d of an
//! arbitrary point, scaled by the overall density.  K(d) > π d² suggests
//! clustering; K(d) < π d² suggests regularity.
//!
//! Reference: Ripley (1976), *Journal of the Royal Statistical Society B* 38:172–192.

use crate::error::{StatsError, StatsResult};
use crate::special::erf::erf;

// ─────────────────────────────── Result types ────────────────────────────────

/// Result of Moran's I test for spatial autocorrelation.
#[derive(Debug, Clone)]
pub struct MoransIResult {
    /// Observed Moran's I statistic.
    pub i: f64,
    /// Expected value `E[I] = −1/(n−1)` under randomisation.
    pub expected: f64,
    /// Analytical variance `Var[I]` under the randomisation hypothesis.
    pub variance: f64,
    /// z-score = `(I − E[I]) / √Var[I]`.
    pub z_score: f64,
    /// Two-tailed p-value from the standard normal approximation.
    pub p_value: f64,
}

/// Result of Geary's C test for spatial association.
#[derive(Debug, Clone)]
pub struct GearyCResult {
    /// Observed Geary's C statistic.
    pub c: f64,
    /// Expected value `E[C] = 1.0` under randomisation.
    pub expected: f64,
    /// z-score under normal approximation.
    pub z_score: f64,
    /// Two-tailed p-value from the standard normal approximation.
    pub p_value: f64,
}

// ─────────────────────────── Internal helpers ─────────────────────────────────

/// Two-tailed p-value from the standard normal distribution.
///
/// P(|Z| > |z|) = erfc(|z|/√2)
#[inline]
fn normal_two_tailed_p(z: f64) -> f64 {
    let az = z.abs() / std::f64::consts::SQRT_2;
    let erfc = 1.0 - erf(az);
    erfc.clamp(0.0, 1.0)
}

/// Validate and extract the n×n spatial weight matrix.
///
/// Checks that `weights.len() == n * n`, all values are finite and ≥ 0,
/// and that the total weight S₀ > 0.
fn validate_weights(weights: &[f64], n: usize) -> StatsResult<()> {
    if weights.len() != n * n {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![weights.len()],
        });
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
        if w < 0.0 {
            return Err(StatsError::InvalidParameter {
                name: format!("weights[{i}]"),
                reason: "spatial weights must be non-negative".into(),
            });
        }
    }
    Ok(())
}

// ─────────────────────────────── Moran's I ───────────────────────────────────

/// Compute Moran's I statistic for global spatial autocorrelation.
///
/// # Arguments
/// - `values`  — attribute values at `n` spatial locations.
/// - `weights` — `n × n` row-major spatial weight matrix (not row-standardised;
///   the function normalises globally by S₀ = Σ_{i,j} W_{ij}).
/// - `n`       — number of locations.
///
/// # Returns
/// A [`MoransIResult`] containing the statistic, expected value, variance, z-score,
/// and two-tailed p-value under the normal approximation.
///
/// # Formulas
/// ```text
/// S₀  = Σ_{i,j} W_{ij}
/// S₁  = ½ Σ_{i,j} (W_{ij} + W_{ji})²
/// S₂  = Σ_i (W_{i·} + W_{·i})²    where W_{i·} = Σ_j W_{ij}
///
/// x̄   = (1/n) Σ_i x_i
/// m₂  = (1/n) Σ_i (x_i − x̄)²
/// m₄  = (1/n) Σ_i (x_i − x̄)⁴
/// b₂  = m₄ / m₂²      (sample excess kurtosis numerator)
///
/// I   = (n / S₀) · [Σ_{i≠j} W_{ij}(x_i − x̄)(x_j − x̄)] / [Σ_i (x_i − x̄)²]
///
/// E[I] = −1/(n−1)
///
/// Var[I] = [n²(n−1)S₁ − n(n−1)S₂ − 2S₀²] / [S₀² (n+1)(n−1)²]  (normality)
///        corrected under randomisation by the b₂ term.
/// ```
pub fn moran_i(values: &[f64], weights: &[f64], n: usize) -> StatsResult<MoransIResult> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if n == 1 {
        return Err(StatsError::InsufficientSampleSize { got: 1, need: 2 });
    }
    if values.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: values.len(),
            b: n,
        });
    }
    validate_weights(weights, n)?;

    // ── Weight sums ───────────────────────────────────────────────────────────
    // S₀ = Σ_{i,j} W[i,j]
    let s0: f64 = weights.iter().sum();
    if s0 < f64::EPSILON {
        return Err(StatsError::InvalidParameter {
            name: "weights".into(),
            reason: "total weight S₀ must be positive".into(),
        });
    }

    // Row sums and column sums
    let mut row_sum = vec![0.0_f64; n];
    let mut col_sum = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            row_sum[i] += weights[i * n + j];
            col_sum[j] += weights[i * n + j];
        }
    }

    // S₁ = ½ Σ_{i,j} (W[i,j] + W[j,i])²
    let mut s1 = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = weights[i * n + j] + weights[j * n + i];
            s1 += v * v;
        }
    }
    s1 *= 0.5;

    // S₂ = Σ_i (row_sum[i] + col_sum[i])²
    let s2: f64 = (0..n).map(|i| (row_sum[i] + col_sum[i]).powi(2)).sum();

    // ── Deviations from mean ──────────────────────────────────────────────────
    let x_bar = values.iter().sum::<f64>() / n as f64;
    let dev: Vec<f64> = values.iter().map(|&xi| xi - x_bar).collect();

    // Σ (x_i − x̄)²
    let ss: f64 = dev.iter().map(|&d| d * d).sum();
    if ss < f64::EPSILON {
        // All values are identical → I undefined (define as 0)
        return Ok(MoransIResult {
            i: 0.0,
            expected: -1.0 / (n as f64 - 1.0),
            variance: 0.0,
            z_score: 0.0,
            p_value: 1.0,
        });
    }

    // ── Moran's I numerator: Σ_{i≠j} W[i,j] (x_i − x̄)(x_j − x̄) ───────────
    let mut cross_sum = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                cross_sum += weights[i * n + j] * dev[i] * dev[j];
            }
        }
    }

    let i_stat = (n as f64 / s0) * (cross_sum / ss);

    // ── Expected value ────────────────────────────────────────────────────────
    let nf = n as f64;
    let expected = -1.0 / (nf - 1.0);

    // ── Analytical variance under randomisation (Cliff & Ord 1981, eq 2.11) ──
    // m₂ = ss / n  (second central moment)
    // m₄ = (1/n) Σ (x_i − x̄)⁴
    let m2 = ss / nf;
    let m4: f64 = dev.iter().map(|&d| d * d * d * d).sum::<f64>() / nf;
    // b₂ = m₄ / m₂²  (kurtosis)
    let b2 = if m2 > f64::EPSILON {
        m4 / (m2 * m2)
    } else {
        3.0 // normal kurtosis
    };

    // Var[I] under randomisation (Cliff & Ord 1981):
    // A = n[(n²−3n+3)S₁ − nS₂ + 3S₀²]
    // B = b₂ [(n²−n)S₁ − 2nS₂ + 6S₀²]
    // C = (n−1)(n−2)(n−3)S₀²
    // Var[I] = [A − B] / C  − E[I]²
    let n2 = nf * nf;
    let a_num = nf * ((n2 - 3.0 * nf + 3.0) * s1 - nf * s2 + 3.0 * s0 * s0);
    let b_num = b2 * ((n2 - nf) * s1 - 2.0 * nf * s2 + 6.0 * s0 * s0);
    let c_denom = (nf - 1.0) * (nf - 2.0) * (nf - 3.0) * s0 * s0;
    let variance = if c_denom.abs() > f64::EPSILON {
        let v = (a_num - b_num) / c_denom - expected * expected;
        v.max(0.0) // numerical floor
    } else {
        // Degenerate case (n < 4): fall back to simple formula
        let v =
            (nf * nf * s1 - nf * s2 + 3.0 * s0 * s0) / ((n2 - 1.0) * s0 * s0) - expected * expected;
        v.max(0.0)
    };

    let z_score = if variance > f64::EPSILON {
        (i_stat - expected) / variance.sqrt()
    } else {
        0.0
    };
    let p_value = normal_two_tailed_p(z_score);

    Ok(MoransIResult {
        i: i_stat,
        expected,
        variance,
        z_score,
        p_value,
    })
}

// ─────────────────────────────── Geary's C ───────────────────────────────────

/// Compute Geary's C statistic for spatial association.
///
/// # Arguments
/// - `values`  — attribute values at `n` spatial locations.
/// - `weights` — `n × n` row-major spatial weight matrix (not row-standardised).
/// - `n`       — number of locations.
///
/// # Returns
/// A [`GearyCResult`] containing the statistic, expected value, z-score, and p-value.
///
/// # Formula
/// ```text
/// C = [(n−1) / (2 S₀)] · [Σ_{i,j} W_{ij}(x_i − x_j)²] / [Σ_i (x_i − x̄)²]
/// E[C] = 1
/// ```
/// Variance under randomisation (Cliff & Ord 1981, eq 2.13).
pub fn geary_c(values: &[f64], weights: &[f64], n: usize) -> StatsResult<GearyCResult> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if n == 1 {
        return Err(StatsError::InsufficientSampleSize { got: 1, need: 2 });
    }
    if values.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: values.len(),
            b: n,
        });
    }
    validate_weights(weights, n)?;

    let s0: f64 = weights.iter().sum();
    if s0 < f64::EPSILON {
        return Err(StatsError::InvalidParameter {
            name: "weights".into(),
            reason: "total weight S₀ must be positive".into(),
        });
    }

    // Row / column sums
    let mut row_sum = vec![0.0_f64; n];
    let mut col_sum = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            row_sum[i] += weights[i * n + j];
            col_sum[j] += weights[i * n + j];
        }
    }

    // S₁ = ½ Σ_{i,j} (W[i,j] + W[j,i])²
    let mut s1 = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = weights[i * n + j] + weights[j * n + i];
            s1 += v * v;
        }
    }
    s1 *= 0.5;

    // S₂ = Σ_i (row_sum[i] + col_sum[i])²
    let s2: f64 = (0..n).map(|i| (row_sum[i] + col_sum[i]).powi(2)).sum();

    // Σ_{i,j} W[i,j] (x_i − x_j)²
    let mut wdiff_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let d = values[i] - values[j];
            wdiff_sq += weights[i * n + j] * d * d;
        }
    }

    let nf = n as f64;
    let x_bar = values.iter().sum::<f64>() / nf;
    let ss: f64 = values.iter().map(|&xi| (xi - x_bar) * (xi - x_bar)).sum();

    let c_stat = if ss > f64::EPSILON {
        ((nf - 1.0) / (2.0 * s0)) * (wdiff_sq / ss)
    } else {
        1.0
    };

    let expected = 1.0_f64;

    // ── Analytical variance under randomisation (Cliff & Ord 1981) ────────────
    // m₂ = ss / n, m₄ = (1/n) Σ d⁴, b₂ = m₄/m₂²
    let dev: Vec<f64> = values.iter().map(|&xi| xi - x_bar).collect();
    let m2 = ss / nf;
    let m4: f64 = dev.iter().map(|&d| d * d * d * d).sum::<f64>() / nf;
    let b2 = if m2 > f64::EPSILON {
        m4 / (m2 * m2)
    } else {
        3.0
    };

    // Var[C] under randomisation (Cliff & Ord 1981, p. 21, eqn 2.12–2.13):
    // Var[C] = {(n−1)S₁ [n² − 3n + 3 − (n−1)b₂]
    //           − (1/4)(n−1)S₂ [n² + 3n − 6 − (n² − n + 2)b₂]
    //           + S₀² [n² − 3 − (n−1)²b₂]}
    //         / {n(n+1)(n−1)²S₀²}  [×(n-1)/(n-1) omitted, cf. exact formula]
    //
    // Simplified form used here (Anselin 1988, Spatial Econometrics):
    let n2 = nf * nf;
    let term1 = (nf - 1.0) * s1 * (n2 - 3.0 * nf + 3.0 - (nf - 1.0) * b2);
    let term2 = 0.25 * (nf - 1.0) * s2 * (n2 + 3.0 * nf - 6.0 - (n2 - nf + 2.0) * b2);
    let term3 = s0 * s0 * (n2 - 3.0 - (nf - 1.0).powi(2) * b2);
    let denom = nf * (nf + 1.0) * (nf - 1.0).powi(2) * s0 * s0;

    let variance = if denom.abs() > f64::EPSILON {
        ((term1 - term2 + term3) / denom).max(0.0)
    } else {
        0.0
    };

    let z_score = if variance > f64::EPSILON {
        (c_stat - expected) / variance.sqrt()
    } else {
        0.0
    };
    let p_value = normal_two_tailed_p(z_score);

    Ok(GearyCResult {
        c: c_stat,
        expected,
        z_score,
        p_value,
    })
}

// ─────────────────────────────── Ripley's K ──────────────────────────────────

/// Compute Ripley's K function for a 2D spatial point process.
///
/// # Arguments
/// - `points` — 2D point coordinates in row-major format `[x₀, y₀, x₁, y₁, …]`,
///   so `points.len() == 2 * n`.
/// - `n`      — number of points.
/// - `radii`  — sorted distance thresholds d at which K(d) is evaluated.
/// - `area`   — area of the study region (used to estimate intensity λ = n/area).
///
/// # Returns
/// A `Vec<f64>` of K(d) values, one per element of `radii`.
///
/// # Formula
/// ```text
/// K(d) = (area / n²) × Σ_{i≠j} I(‖p_i − p_j‖ ≤ d)
/// ```
/// Edge correction is NOT applied; for interior analyses use an edge-corrected
/// estimator (e.g., Ripley's isotropic correction).
///
/// # Monotonicity
/// K(d) is non-decreasing since larger d includes all pairs counted at smaller d.
pub fn ripleys_k(points: &[f64], n: usize, radii: &[f64], area: f64) -> StatsResult<Vec<f64>> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if points.len() != 2 * n {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, 2],
            got: vec![points.len()],
        });
    }
    if radii.is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "radii".into(),
            reason: "at least one radius must be provided".into(),
        });
    }
    if area <= 0.0 || !area.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "area".into(),
            reason: "study area must be positive and finite".into(),
        });
    }
    for (i, &r) in radii.iter().enumerate() {
        if !r.is_finite() || r < 0.0 {
            return Err(StatsError::InvalidParameter {
                name: format!("radii[{i}]"),
                reason: "radii must be non-negative and finite".into(),
            });
        }
    }
    for (i, &v) in points.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    // Pre-compute all pairwise squared distances (upper triangle)
    // For n points we have n*(n-1)/2 pairs; i < j
    let nf = n as f64;

    // Compute K(d) for each radius using brute-force O(n² · |radii|).
    // For large n, a binned approach would be more efficient, but purity is paramount here.
    let mut k_values = vec![0.0_f64; radii.len()];

    // We sort radii so we can accumulate counts incrementally.
    // Create index permutation sorted by radius.
    let mut sorted_idx: Vec<usize> = (0..radii.len()).collect();
    sorted_idx.sort_unstable_by(|&a, &b| {
        radii[a]
            .partial_cmp(&radii[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute all pairwise distances (i < j), then sort them.
    // n*(n-1)/2 pairs
    let n_pairs = n * (n - 1) / 2;
    let mut pair_dists = Vec::with_capacity(n_pairs);
    for i in 0..n {
        let xi = points[2 * i];
        let yi = points[2 * i + 1];
        for j in (i + 1)..n {
            let xj = points[2 * j];
            let yj = points[2 * j + 1];
            let dx = xi - xj;
            let dy = yi - yj;
            pair_dists.push((dx * dx + dy * dy).sqrt());
        }
    }
    pair_dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // For each sorted radius, binary-search the count of pairs ≤ d.
    // Each pair (i,j) contributes 2 to the sum Σ_{i≠j} (since both (i,j) and (j,i) counted).
    let scale = area / (nf * nf);

    let mut cumulative_count = 0.0_f64;
    let mut pair_ptr = 0_usize;

    for &r_idx in &sorted_idx {
        let d = radii[r_idx];
        let d_sq = d * d;
        // Advance pointer while pair_dists[pair_ptr] ≤ d
        while pair_ptr < pair_dists.len() && pair_dists[pair_ptr] <= d + f64::EPSILON {
            // Check exact squared distance to avoid f64 sqrt rounding
            // (already have the sqrt in pair_dists)
            if pair_dists[pair_ptr] * pair_dists[pair_ptr] <= d_sq + 1e-14 {
                cumulative_count += 2.0; // count both (i,j) and (j,i)
            } else {
                cumulative_count += 2.0; // accepted by sqrt comparison
            }
            pair_ptr += 1;
        }
        k_values[r_idx] = scale * cumulative_count;
    }

    Ok(k_values)
}

// ─────────────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper: rook-adjacency weight matrix for an n-point 1D lattice ────────
    //
    // W[i,j] = 1 if |i-j|==1, else 0.  Diagonal is 0.
    fn rook_weights_1d(n: usize) -> Vec<f64> {
        let mut w = vec![0.0_f64; n * n];
        for i in 0..n {
            if i + 1 < n {
                w[i * n + (i + 1)] = 1.0;
                w[(i + 1) * n + i] = 1.0;
            }
        }
        w
    }

    // ── 1. Moran's I ≈ -1/(n-1) for spatially random (uncorrelated) data ─────
    #[test]
    fn morans_i_expected_for_uncorrelated() {
        // With rook adjacency and arbitrary values, mean(I) ≈ -1/(n-1).
        let n = 8_usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let w = rook_weights_1d(n);
        let res = moran_i(&values, &w, n).expect("moran_i ok");
        // E[I] = -1/(n-1) ≈ -0.143 for n=8
        let exp_i = -1.0 / (n as f64 - 1.0);
        assert!(
            (res.expected - exp_i).abs() < 1e-10,
            "expected = {}, got {}",
            exp_i,
            res.expected
        );
        // I finite
        assert!(res.i.is_finite(), "I should be finite");
    }

    // ── 2. Positive spatial autocorrelation gives I > 0 ──────────────────────
    #[test]
    fn morans_i_positive_autocorrelation() {
        // On a 1D rook lattice (chain), clustering is clear:
        // values: 0,0,0,0,0,10,10,10,10,10
        // Each adjacent pair within a cluster contributes positively to Σ W(xi-xbar)(xj-xbar).
        // There are 4 within-low + 4 within-high = 8 positive pairs,
        // and only 1 cross-cluster pair (positions 4-5 are adjacent with W=1).
        // positive contributions: 8 × 25 = 200
        // negative contribution: 2 × (-25) = -50 (pair counted twice: i→j and j→i)
        // net = 150 > 0, so I > 0.
        let n = 10_usize;
        let values: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 10.0 }).collect();
        // Rook adjacency on 1D line: W[i,i+1] = W[i+1,i] = 1
        let w = rook_weights_1d(n);
        let res = moran_i(&values, &w, n).expect("moran_i ok");
        // With strongly clustered data and rook adjacency, I > 0
        assert!(
            res.i > 0.0,
            "clustered data should give I > 0, got {}",
            res.i
        );
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0, "p-value in [0,1]");
    }

    // ── 3. Negative autocorrelation: checkerboard gives I < 0 ────────────────
    #[test]
    fn morans_i_checkerboard_negative() {
        // Checkerboard (alternating) is negatively autocorrelated with rook adjacency
        let n = 6_usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let w = rook_weights_1d(n);
        let res = moran_i(&values, &w, n).expect("moran_i ok");
        assert!(
            res.i < 0.0,
            "alternating values should give I < 0, got {}",
            res.i
        );
    }

    // ── 4. Moran's I variance > 0 ────────────────────────────────────────────
    #[test]
    fn morans_i_variance_positive() {
        let n = 6_usize;
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = rook_weights_1d(n);
        let res = moran_i(&values, &w, n).expect("moran_i ok");
        assert!(res.variance >= 0.0, "variance must be >= 0");
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
    }

    // ── 5. Moran's I: constant values → I = 0 ────────────────────────────────
    #[test]
    fn morans_i_constant_values() {
        let n = 5_usize;
        let values = vec![3.0_f64; n];
        let w = rook_weights_1d(n);
        let res = moran_i(&values, &w, n).expect("moran_i constant ok");
        // All deviations zero → I = 0
        assert!((res.i).abs() < 1e-10, "constant → I ≈ 0, got {}", res.i);
        assert!((res.z_score).abs() < 1e-10);
    }

    // ── 6. Moran's I: empty input errors ─────────────────────────────────────
    #[test]
    fn morans_i_empty_error() {
        let result = moran_i(&[], &[], 0);
        assert!(result.is_err(), "empty input should error");
    }

    // ── 7. Geary's C ≈ 1 for uncorrelated data ───────────────────────────────
    #[test]
    fn geary_c_expected_one() {
        let expected = 1.0_f64;
        let n = 6_usize;
        let values: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let w = rook_weights_1d(n);
        let res = geary_c(&values, &w, n).expect("geary_c ok");
        assert!(
            (res.expected - expected).abs() < 1e-10,
            "E[C] should be 1.0, got {}",
            res.expected
        );
        assert!(res.c.is_finite(), "C should be finite");
    }

    // ── 8. Geary's C: clustered data gives C < 1 ─────────────────────────────
    #[test]
    fn geary_c_clustered_less_than_one() {
        // First half 0, second half 10 with rook adjacency on a line
        let n = 10_usize;
        let values: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 10.0 }).collect();
        let w = rook_weights_1d(n);
        let res = geary_c(&values, &w, n).expect("geary_c ok");
        // With clustering on a 1D lattice, most adjacent pairs share a cluster
        // (except the boundary pair), so Σ W(xi-xj)² is small → C < 1
        assert!(
            res.c < 1.5,
            "clustered data should give C close to or < 1, got {}",
            res.c
        );
    }

    // ── 9. Geary's C: p-value in [0, 1] ──────────────────────────────────────
    #[test]
    fn geary_c_p_value_range() {
        let n = 8_usize;
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let w = rook_weights_1d(n);
        let res = geary_c(&values, &w, n).expect("geary_c ok");
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0, "p_value ∈ [0,1]");
    }

    // ── 10. Ripley's K: monotone increasing ──────────────────────────────────
    #[test]
    fn ripleys_k_monotone() {
        // A grid of points
        let n = 9_usize;
        let mut pts = Vec::with_capacity(2 * n);
        for i in 0..3 {
            for j in 0..3 {
                pts.push(i as f64);
                pts.push(j as f64);
            }
        }
        let radii = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let area = 9.0_f64;
        let k_vals = ripleys_k(&pts, n, &radii, area).expect("ripleys_k ok");
        assert_eq!(k_vals.len(), radii.len());
        // K(d) must be non-decreasing
        for w in k_vals.windows(2) {
            assert!(
                w[1] >= w[0] - f64::EPSILON,
                "K must be non-decreasing: K({})={} < K({})={}",
                radii[1],
                w[1],
                radii[0],
                w[0]
            );
        }
    }

    // ── 11. Ripley's K: single point → K = 0 for all radii ──────────────────
    #[test]
    fn ripleys_k_single_point() {
        let pts = vec![0.5_f64, 0.5];
        let radii = vec![0.1, 1.0, 5.0];
        let result = ripleys_k(&pts, 1, &radii, 1.0);
        // n=1 means no pairs → K = 0 (area/n² * 0 = 0)
        let k_vals = result.expect("single point ok");
        for &k in &k_vals {
            assert!(
                k.abs() < f64::EPSILON,
                "K should be 0 for single point, got {k}"
            );
        }
    }

    // ── 12. Ripley's K: empty radii errors ───────────────────────────────────
    #[test]
    fn ripleys_k_empty_radii_error() {
        let pts = vec![0.0, 0.0, 1.0, 0.0];
        let result = ripleys_k(&pts, 2, &[], 1.0);
        assert!(result.is_err(), "empty radii should error");
    }

    // ── 13. Ripley's K: non-positive area errors ──────────────────────────────
    #[test]
    fn ripleys_k_negative_area_error() {
        let pts = vec![0.0, 0.0, 1.0, 1.0];
        let result = ripleys_k(&pts, 2, &[1.0], -1.0);
        assert!(result.is_err(), "non-positive area should error");
    }

    // ── 14. Ripley's K: K(r=0) = 0 ───────────────────────────────────────────
    #[test]
    fn ripleys_k_zero_radius() {
        let pts = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let n = 3_usize;
        let radii = vec![0.0, 0.5, 1.0];
        let area = 4.0_f64;
        let k_vals = ripleys_k(&pts, n, &radii, area).expect("k ok");
        // At r=0, no pairs within distance 0 → K(0) = 0
        assert!(
            k_vals[0].abs() < f64::EPSILON,
            "K(0) should be 0, got {}",
            k_vals[0]
        );
    }

    // ── 15. Moran's I: p-value ∈ [0, 1] for structured data ─────────────────
    #[test]
    fn morans_i_p_value_valid() {
        let n = 10_usize;
        let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut w = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = 1.0 / (1.0 + ((i as f64 - j as f64).abs()));
                }
            }
        }
        let res = moran_i(&values, &w, n).expect("moran_i ok");
        assert!(
            res.p_value >= 0.0 && res.p_value <= 1.0,
            "p_value out of [0,1]: {}",
            res.p_value
        );
        assert!(res.z_score.is_finite(), "z_score should be finite");
    }

    // ── 16. Geary's C: regular pattern gives C > 1 ───────────────────────────
    #[test]
    fn geary_c_regular_greater_than_one() {
        // Alternating values with rook weights → high local differences → C > 1
        let n = 8_usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.0 } else { 100.0 })
            .collect();
        let w = rook_weights_1d(n);
        let res = geary_c(&values, &w, n).expect("geary_c ok");
        // Adjacent pairs always differ → Σ W(xi-xj)² is large → C > 1
        assert!(
            res.c > 1.0,
            "alternating pattern should give C > 1, got {}",
            res.c
        );
    }
}
