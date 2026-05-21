//! SSL feature-quality metrics — Wang & Isola 2020 and related diagnostics.
//!
//! Implements the four canonical representation-quality measures used to
//! diagnose self-supervised learning features without downstream labels:
//!
//! | Metric | Paper | Interpretation |
//! |--------|-------|----------------|
//! | [`uniformity_loss`]       | Wang & Isola 2020, Eq. 2 | Lower ⟹ more uniform on sphere |
//! | [`alignment_loss`]        | Wang & Isola 2020, Eq. 3 | Lower ⟹ better positive-pair alignment |
//! | [`effective_rank`]        | Roy & Vetterli 2007       | Higher ⟹ less dimensional collapse |
//! | [`collapse_score`]        | convenience composite     | Higher ⟹ more collapsed |
//! | [`pairwise_cosine_stats`] | —                         | Distribution of off-diagonal cosines |
//!
//! All functions accept **row-major** flat slices `z: &[f32]` with `n` rows
//! and `d` columns.  L2-normalisation of rows is performed internally so the
//! caller does not need to pre-normalise.

use crate::error::{SslError, SslResult};

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// L2-normalise each row of `z` in-place.  Rows with near-zero norm are mapped
/// to the zero vector (they contribute `cos = 0` to all pair computations).
fn l2_normalise_rows(z: &mut [f32], n: usize, d: usize) {
    for i in 0..n {
        let row = &mut z[i * d..(i + 1) * d];
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Uniformity ───────────────────────────────────────────────────────────────

/// **Uniformity loss** (Wang & Isola 2020, Eq. 2).
///
/// ```text
/// L_unif = log E_{i<j}[ exp(-2 ‖ẑ_i − ẑ_j‖²) ]
/// ```
///
/// Rows of `z` are L2-normalised internally.  For unit vectors the squared
/// Euclidean distance simplifies to `‖ẑ_i − ẑ_j‖² = 2(1 − ẑ_i·ẑ_j)`, so
/// the kernel becomes `exp(-4(1 − cos))`.
///
/// # Returns
/// A scalar in `(-∞, 0]`.  `0` means complete collapse (all pairs at distance
/// 0); more negative ⟹ more uniform.
///
/// # Errors
/// - [`SslError::EmptyInput`] — `n < 2` or `d == 0`.
/// - [`SslError::DimensionMismatch`] — `z.len() != n * d`.
pub fn uniformity_loss(z: &[f32], n: usize, d: usize) -> SslResult<f32> {
    if n < 2 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }

    // Work on a normalised copy to avoid mutating caller data.
    let mut z_hat = z.to_vec();
    l2_normalise_rows(&mut z_hat, n, d);

    // Accumulate kernel values over all distinct pairs (i < j).
    // For unit vectors: ||ẑ_i - ẑ_j||² = 2(1 - ẑ_i · ẑ_j).
    // Kernel: exp(-2 * dist²) = exp(-4*(1 - cos)).
    let mut kernel_sum = 0.0_f64;
    let num_pairs = n * (n - 1) / 2;
    for i in 0..n {
        let zi = &z_hat[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let zj = &z_hat[j * d..(j + 1) * d];
            let cos = dot(zi, zj);
            // Clamp cosine to [-1, 1] to guard against floating-point drift.
            let cos_clamped = cos.clamp(-1.0, 1.0);
            let dist_sq = 2.0_f64 * (1.0 - cos_clamped as f64);
            kernel_sum += (-2.0 * dist_sq).exp();
        }
    }

    let mean_kernel = kernel_sum / num_pairs as f64;
    Ok(mean_kernel.ln() as f32)
}

// ─── Alignment ────────────────────────────────────────────────────────────────

/// **Alignment loss** (Wang & Isola 2020, Eq. 3).
///
/// ```text
/// L_align = E_i[ ‖ẑ1_i − ẑ2_i‖^alpha ]
/// ```
///
/// `z1` and `z2` are `[N, D]` flat slices representing positive pairs.  Both
/// are L2-normalised row-wise internally.  For unit vectors:
/// `‖ẑ_i − ẑ_j‖² = 2(1 − cos)`, so for `alpha = 2` the result is
/// `mean(2 − 2·cos)`, matching the BYOL loss formulation.
///
/// # Parameters
/// - `alpha`: the exponent (Wang & Isola recommend 2.0).  Must be `> 0`.
///
/// # Returns
/// A non-negative scalar; `0` means perfect alignment.
///
/// # Errors
/// - [`SslError::EmptyInput`] — `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] — slice lengths mismatch `n * d`.
/// - [`SslError::InvalidParameter`] — `alpha <= 0` or not finite.
pub fn alignment_loss(z1: &[f32], z2: &[f32], n: usize, d: usize, alpha: f32) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z1.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z1.len(),
        });
    }
    if z2.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z2.len(),
        });
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(SslError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be finite and > 0".into(),
        });
    }

    let mut z1_hat = z1.to_vec();
    let mut z2_hat = z2.to_vec();
    l2_normalise_rows(&mut z1_hat, n, d);
    l2_normalise_rows(&mut z2_hat, n, d);

    let half_alpha = alpha / 2.0;
    let mut total = 0.0_f64;
    for i in 0..n {
        let a = &z1_hat[i * d..(i + 1) * d];
        let b = &z2_hat[i * d..(i + 1) * d];
        let cos = dot(a, b).clamp(-1.0, 1.0);
        // Squared distance between unit vectors: ||ẑ1 - ẑ2||² = 2(1 - cos)
        let dist_sq = 2.0_f64 * (1.0 - cos as f64);
        // Loss = dist_sq^(alpha/2) = (||ẑ||²)^(alpha/2)
        total += dist_sq.powf(half_alpha as f64);
    }

    Ok((total / n as f64) as f32)
}

// ─── Effective rank ───────────────────────────────────────────────────────────

/// **Effective rank** (Roy & Vetterli 2007).
///
/// Measures the intrinsic dimensionality of the feature distribution using the
/// Shannon entropy of the column-variance spectrum:
///
/// ```text
/// p_j = σ_j² / Σ_k σ_k²
/// eff_rank = exp( -Σ_j p_j log(p_j) )
/// ```
///
/// where `σ_j²` is the sample variance of column `j` of `Z` (after L2-normalising
/// each row). This approximates the spectrum-based effective rank without a full
/// eigendecomposition.
///
/// # Returns
/// A value in `[1.0, d as f32]`.  `d` ⟹ all dimensions equally active
/// (no collapse); `1` ⟹ complete rank-1 collapse.
///
/// # Errors
/// - [`SslError::EmptyInput`] — `n < 2` or `d == 0`.
/// - [`SslError::DimensionMismatch`] — `z.len() != n * d`.
/// - [`SslError::InvalidParameter`] — total variance is zero (all-zero features).
pub fn effective_rank(z: &[f32], n: usize, d: usize) -> SslResult<f32> {
    if n < 2 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }

    // L2-normalise rows so that magnitude differences don't dominate the
    // variance spectrum — we care about *direction* diversity.
    let mut z_hat = z.to_vec();
    l2_normalise_rows(&mut z_hat, n, d);

    // Compute column means.
    let mut col_mean = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            col_mean[j] += z_hat[i * d + j] as f64;
        }
    }
    for m in col_mean.iter_mut() {
        *m /= n as f64;
    }

    // Compute column variances σ_j² = (1/n) Σ_i (z_{ij} - μ_j)².
    let mut col_var = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            let diff = z_hat[i * d + j] as f64 - col_mean[j];
            col_var[j] += diff * diff;
        }
    }
    for v in col_var.iter_mut() {
        *v /= n as f64;
    }

    let total_var: f64 = col_var.iter().sum();
    if total_var < 1e-30 {
        return Err(SslError::InvalidParameter {
            name: "z".into(),
            reason: "total column variance is zero; features appear to be all-zero".into(),
        });
    }

    // Compute entropy H = -Σ p_j log(p_j) where p_j = σ_j² / Σ σ_k².
    let mut entropy = 0.0_f64;
    for &v in col_var.iter() {
        if v > 0.0 {
            let p = v / total_var;
            entropy -= p * p.ln();
        }
    }

    Ok(entropy.exp() as f32)
}

// ─── Collapse score ───────────────────────────────────────────────────────────

/// **Representation collapse severity** (composite diagnostic).
///
/// Combines the Wang & Isola uniformity and alignment losses into a single
/// scalar that rises when the representation collapses:
///
/// ```text
/// collapse_score = alignment_loss(z1, z2, alpha=2) − uniformity_loss(z)
/// ```
///
/// `uniformity_loss` is negative for well-spread features, so subtracting it
/// (i.e. adding its absolute value) makes `collapse_score` larger when
/// features are both poorly aligned *and* poorly spread.  Neither term is
/// published as a standalone metric; this is a diagnostic convenience.
///
/// # Returns
/// A real-valued scalar; larger ⟹ more collapsed.
///
/// # Errors
/// Propagates errors from [`uniformity_loss`] and [`alignment_loss`].
pub fn collapse_score(z1: &[f32], z2: &[f32], n: usize, d: usize) -> SslResult<f32> {
    // For uniformity we use z1 as the pool of features.
    let u = uniformity_loss(z1, n, d)?;
    let a = alignment_loss(z1, z2, n, d, 2.0)?;
    // u <= 0; a >= 0. collapse_score = a - u (both terms non-negative when
    // representation is good: a ≈ 0, u very negative → score near 0).
    Ok(a - u)
}

// ─── Pairwise cosine statistics ───────────────────────────────────────────────

/// **Pairwise cosine similarity distribution statistics**.
///
/// Returns `(mean, std, max)` of all `n*(n-1)/2` pairwise cosine similarities
/// between distinct L2-normalised rows.  This gives a distributional view of
/// how similar features are to each other — collapsed representations cluster
/// near `max ≈ mean ≈ 1`.
///
/// # Returns
/// `(mean, std, max)` of all off-diagonal cosines.
///
/// # Errors
/// - [`SslError::EmptyInput`] — `n < 2` or `d == 0`.
/// - [`SslError::DimensionMismatch`] — `z.len() != n * d`.
pub fn pairwise_cosine_stats(z: &[f32], n: usize, d: usize) -> SslResult<(f32, f32, f32)> {
    if n < 2 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }

    let mut z_hat = z.to_vec();
    l2_normalise_rows(&mut z_hat, n, d);

    let num_pairs = n * (n - 1) / 2;
    let mut cosines = Vec::with_capacity(num_pairs);
    let mut max_cos = f32::NEG_INFINITY;

    for i in 0..n {
        let zi = &z_hat[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let zj = &z_hat[j * d..(j + 1) * d];
            let c = dot(zi, zj).clamp(-1.0, 1.0);
            if c > max_cos {
                max_cos = c;
            }
            cosines.push(c);
        }
    }

    let mean = cosines.iter().map(|&c| c as f64).sum::<f64>() / num_pairs as f64;
    let var = cosines
        .iter()
        .map(|&c| {
            let diff = c as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / num_pairs as f64;
    let std = var.sqrt();

    Ok((mean as f32, std as f32, max_cos))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build the standard axis-aligned basis for R^d, returned as a flat
    /// row-major vec of shape [d, d].
    fn basis(d: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; d * d];
        for i in 0..d {
            v[i * d + i] = 1.0;
        }
        v
    }

    /// Repeat the same unit vector `n` times → perfectly collapsed batch.
    fn all_same(unit: &[f32], n: usize) -> Vec<f32> {
        let d = unit.len();
        let mut v = Vec::with_capacity(n * d);
        for _ in 0..n {
            v.extend_from_slice(unit);
        }
        v
    }

    // ── uniformity_loss ───────────────────────────────────────────────────────

    /// Four points at ±e1 and ±e2 in R^2: uniformity should be finite and < 0.
    #[test]
    fn uniformity_perfectly_uniform_sphere() {
        // ±e1 and ±e2: 4 points on the 2-d unit sphere.
        let z = vec![
            1.0_f32, 0.0, // e1
            -1.0, 0.0, // -e1
            0.0, 1.0, // e2
            0.0, -1.0, // -e2
        ];
        let l = uniformity_loss(&z, 4, 2).unwrap();
        // Cosines: e1·(-e1) = -1 (dist² = 4, kernel = exp(-8))
        //          e1·e2 = 0  (dist² = 2, kernel = exp(-4))
        // There are 6 pairs; result must be a finite negative number.
        assert!(l.is_finite(), "l = {l}");
        assert!(l < 0.0, "expected negative uniformity, got {l}");
    }

    /// All vectors the same → all pairwise distances = 0 → kernel = exp(0) = 1
    /// → mean = 1 → log(1) = 0 (maximum collapse value).
    #[test]
    fn uniformity_all_same_point_high() {
        let z = all_same(&[1.0_f32, 0.0, 0.0], 6);
        let l = uniformity_loss(&z, 6, 3).unwrap();
        assert!((l - 0.0).abs() < 1e-5, "expected ≈ 0, got {l}");
    }

    /// Uniformity with n=2 should succeed and be well-defined.
    #[test]
    fn uniformity_two_orthogonal_points() {
        let z = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = uniformity_loss(&z, 2, 2).unwrap();
        // cos = 0 → dist² = 2 → kernel = exp(-4) → log(exp(-4)) = -4
        assert!((l - (-4.0)).abs() < 1e-4, "expected -4, got {l}");
    }

    // ── alignment_loss ────────────────────────────────────────────────────────

    /// Identical positive pairs ⟹ distance = 0 ⟹ loss = 0.
    #[test]
    fn alignment_identical_pairs_zero_loss() {
        let z: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let l = alignment_loss(&z, &z, 4, 4, 2.0).unwrap();
        assert!(l.abs() < 1e-5, "expected 0, got {l}");
    }

    /// e1 paired with e2: cos = 0 → dist² = 2 → loss = 2^(alpha/2) = 2 (alpha=2).
    #[test]
    fn alignment_orthogonal_pairs_max_loss() {
        let z1 = vec![1.0_f32, 0.0];
        let z2 = vec![0.0_f32, 1.0];
        let l = alignment_loss(&z1, &z2, 1, 2, 2.0).unwrap();
        assert!((l - 2.0).abs() < 1e-5, "expected 2, got {l}");
    }

    /// Single positive pair (n=1) must succeed without empty-input error.
    #[test]
    fn alignment_n1_works() {
        let z1 = vec![1.0_f32, 0.0, 0.0];
        let z2 = vec![0.0_f32, 1.0, 0.0];
        let l = alignment_loss(&z1, &z2, 1, 3, 2.0).unwrap();
        assert!(l.is_finite() && l >= 0.0, "l = {l}");
    }

    /// alpha=1: loss = dist (not dist²), must still be non-negative and finite.
    #[test]
    fn alignment_alpha_one_finite() {
        let z1 = vec![1.0_f32, 0.0];
        let z2 = vec![0.0_f32, 1.0];
        let l = alignment_loss(&z1, &z2, 1, 2, 1.0).unwrap();
        // dist_sq = 2, dist = sqrt(2) → l = sqrt(2)
        assert!((l - 2.0_f32.sqrt()).abs() < 1e-4, "l = {l}");
    }

    /// Invalid alpha (non-positive) returns InvalidParameter error.
    #[test]
    fn alignment_invalid_alpha_returns_error() {
        let z1 = vec![1.0_f32, 0.0];
        let z2 = vec![0.0_f32, 1.0];
        let r = alignment_loss(&z1, &z2, 1, 2, 0.0);
        assert!(
            matches!(r, Err(SslError::InvalidParameter { .. })),
            "expected InvalidParameter, got {r:?}"
        );
    }

    // ── effective_rank ────────────────────────────────────────────────────────

    /// Orthonormal basis of R^d: each column has the same variance (1/d after
    /// normalisation) → eff_rank ≈ d.
    #[test]
    fn effective_rank_uniform_full_rank() {
        let d = 8_usize;
        // Use d rows of the identity: e1, …, ed — each direction equally active.
        let z = basis(d);
        let er = effective_rank(&z, d, d).unwrap();
        // Exact eff_rank = d when all variances are equal.
        assert!((er - d as f32).abs() < 0.5, "expected ≈ {d}, got {er}");
    }

    /// Features concentrated almost entirely in one direction should yield
    /// effective rank close to 1.  We construct rows of the form
    /// `[1, ε_i, 0, …, 0]` where `ε_i` is a tiny but varying second component.
    /// After L2-normalisation each row is near `e1` but has a slightly different
    /// angle, so column 1 has a tiny but non-zero variance while columns 2…d-1
    /// have zero variance. The entropy is dominated by p_0 ≈ 1 → eff_rank ≈ 1.
    ///
    /// We also verify that perfectly collapsed inputs (all-same direction after
    /// normalisation) trigger `InvalidParameter` rather than silently returning
    /// a meaningless value.
    #[test]
    fn effective_rank_rank1_collapsed() {
        let d = 8_usize;
        let n = 16_usize;

        // Scenario A: all rows normalise to exactly e1 (scale-only variation)
        // → total column variance = 0 → must return InvalidParameter.
        let mut z_collapsed = vec![0.0_f32; n * d];
        for i in 0..n {
            z_collapsed[i * d] = 1.0 + (i as f32) * 0.1; // only col-0 non-zero
        }
        assert!(
            matches!(
                effective_rank(&z_collapsed, n, d),
                Err(SslError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for completely collapsed input"
        );

        // Scenario B: rows lie near e1 but with a tiny, varying second component
        // → col-1 has non-zero variance; eff_rank should be close to 1.
        let mut z_near_rank1 = vec![0.0_f32; n * d];
        for i in 0..n {
            z_near_rank1[i * d] = 1.0;
            z_near_rank1[i * d + 1] = (i as f32) * 0.001; // tiny varying offset
        }
        let er = effective_rank(&z_near_rank1, n, d).unwrap();
        assert!(er < 2.0, "expected eff_rank ≈ 1, got {er}");
        assert!(er >= 1.0, "eff_rank must be >= 1, got {er}");
    }

    /// Effective rank must always lie in `[1, d]` for any valid input.
    #[test]
    fn effective_rank_in_range() {
        let d = 4_usize;
        let n = 8_usize;
        // Build a random-ish matrix using a deterministic LCG.
        let mut state: u64 = 0xdead_beef_cafe;
        let z: Vec<f32> = (0..n * d)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map to [-1, 1]
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let er = effective_rank(&z, n, d).unwrap();
        assert!(er >= 1.0 - 1e-4, "eff_rank below 1: {er}");
        assert!(er <= d as f32 + 1e-4, "eff_rank above d: {er}");
    }

    // ── pairwise_cosine_stats ─────────────────────────────────────────────────

    /// Orthogonal basis: all pairwise cosines = 0 → mean = 0, std = 0, max = 0.
    #[test]
    fn pairwise_cosine_orthogonal_basis() {
        let d = 4_usize;
        let z = basis(d);
        let (mean, std, max) = pairwise_cosine_stats(&z, d, d).unwrap();
        assert!(mean.abs() < 1e-5, "mean = {mean}");
        assert!(std.abs() < 1e-5, "std = {std}");
        assert!(max.abs() < 1e-5, "max = {max}");
    }

    /// All rows the same unit vector: all pairwise cosines = 1.
    #[test]
    fn pairwise_cosine_same_direction() {
        let n = 5_usize;
        let unit = [1.0_f32, 0.0, 0.0];
        let z = all_same(&unit, n);
        let (mean, std, max) = pairwise_cosine_stats(&z, n, 3).unwrap();
        assert!((mean - 1.0).abs() < 1e-5, "mean = {mean}");
        assert!(std.abs() < 1e-5, "std = {std}");
        assert!((max - 1.0).abs() < 1e-5, "max = {max}");
    }

    // ── error paths ───────────────────────────────────────────────────────────

    /// All metrics must return EmptyInput when n=0 or n=1 (where applicable).
    #[test]
    fn empty_input_returns_error() {
        // uniformity / pairwise_cosine require n >= 2
        assert!(
            matches!(uniformity_loss(&[], 0, 4), Err(SslError::EmptyInput)),
            "uniformity n=0"
        );
        assert!(
            matches!(
                uniformity_loss(&[1.0, 0.0, 0.0, 0.0], 1, 4),
                Err(SslError::EmptyInput)
            ),
            "uniformity n=1"
        );
        // alignment allows n=1 (single pair is valid)
        assert!(
            matches!(
                alignment_loss(&[], &[], 0, 4, 2.0),
                Err(SslError::EmptyInput)
            ),
            "alignment n=0"
        );
        // effective_rank requires n >= 2 (need variance)
        assert!(
            matches!(effective_rank(&[], 0, 4), Err(SslError::EmptyInput)),
            "eff_rank n=0"
        );
        assert!(
            matches!(pairwise_cosine_stats(&[], 0, 4), Err(SslError::EmptyInput)),
            "pairwise_cosine n=0"
        );
    }

    /// DimensionMismatch is returned when slice length != n*d.
    #[test]
    fn dimension_mismatch_returns_error() {
        // Supply n=2, d=3 but only 4 elements (should be 6).
        let z = vec![1.0_f32; 4];
        assert!(
            matches!(
                uniformity_loss(&z, 2, 3),
                Err(SslError::DimensionMismatch {
                    expected: 6,
                    got: 4
                })
            ),
            "uniformity mismatch"
        );
        assert!(
            matches!(
                effective_rank(&z, 2, 3),
                Err(SslError::DimensionMismatch {
                    expected: 6,
                    got: 4
                })
            ),
            "eff_rank mismatch"
        );
        assert!(
            matches!(
                pairwise_cosine_stats(&z, 2, 3),
                Err(SslError::DimensionMismatch {
                    expected: 6,
                    got: 4
                })
            ),
            "pairwise_cosine mismatch"
        );
        // alignment: z1 correct, z2 wrong length
        let z_ok = vec![1.0_f32; 6];
        assert!(
            matches!(
                alignment_loss(&z_ok, &z, 2, 3, 2.0),
                Err(SslError::DimensionMismatch {
                    expected: 6,
                    got: 4
                })
            ),
            "alignment z2 mismatch"
        );
    }

    // ── collapse_score ────────────────────────────────────────────────────────

    /// Perfect alignment + any spread: collapse_score is finite.
    #[test]
    fn collapse_score_identical_pairs_finite() {
        let d = 4_usize;
        let n = 4_usize;
        let z = basis(d);
        let score = collapse_score(&z, &z, n, d).unwrap();
        // Identical pairs → alignment = 0; uniformity = some negative value.
        // score = 0 - u = -u > 0.
        assert!(score.is_finite(), "score = {score}");
        assert!(
            score >= 0.0,
            "score should be >= 0 for identical pairs, got {score}"
        );
    }
}
