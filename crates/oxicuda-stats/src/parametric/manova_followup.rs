//! MANOVA follow-up analysis: descriptive discriminant analysis (DDA),
//! univariate follow-up ANOVAs (Bonferroni-adjusted), and Roy's largest root.
//!
//! When a one-way MANOVA (see [`super::manova`]) is significant, this module
//! interprets *where* and *how* the groups differ. The core engine solves the
//! generalized eigenproblem `W⁻¹B v = λ v` — equivalently the symmetric
//! eigenproblem `W^{-1/2} B W^{-1/2} u = λ u` via a Jacobi rotation sweep — to
//! recover canonical discriminant functions. From the eigenvalues we report
//! canonical correlations `ρ_i = sqrt(λ_i / (1 + λ_i))`, raw and standardized
//! discriminant coefficients, structure correlations (variable–function
//! correlations), and the proportion of discriminative variance each function
//! explains. Roy's largest root is the leading eigenvalue.
//!
//! `B` (between-groups SSCP) and `W` (within-groups / error SSCP) are the same
//! matrices used by the parametric MANOVA (`H` and `E` respectively); here we
//! recompute them from a flat row-major design so callers can pass labelled
//! observations directly.

use crate::distributions::f_dist::FDist;
use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{matrix_inverse_lu, matrix_mul};

/// Univariate one-way ANOVA result for a single dependent variable, including a
/// Bonferroni-adjusted significance flag computed across all `p` variables.
#[derive(Debug, Clone)]
pub struct UnivariateAnova {
    /// Zero-based index of the dependent variable.
    pub variable: usize,
    /// F statistic.
    pub f_statistic: f64,
    /// Between-groups degrees of freedom (`g - 1`).
    pub df_between: f64,
    /// Within-groups degrees of freedom (`n - g`).
    pub df_within: f64,
    /// Raw (unadjusted) p-value.
    pub p_value: f64,
    /// Bonferroni-adjusted p-value, `min(1, p_value * p)`.
    pub p_value_bonferroni: f64,
    /// Whether the variable is significant at `alpha` after Bonferroni
    /// correction across the `p` univariate tests.
    pub significant_bonferroni: bool,
}

/// Full result of a MANOVA follow-up battery.
#[derive(Debug, Clone)]
pub struct ManovaFollowup {
    /// Number of groups.
    pub g: usize,
    /// Number of dependent variables.
    pub p: usize,
    /// Total number of observations.
    pub n: usize,
    /// Number of (non-degenerate) canonical discriminant functions,
    /// `min(g - 1, p)`.
    pub n_functions: usize,
    /// Eigenvalues `λ_i` of `W⁻¹B`, sorted descending. Length `n_functions`.
    pub eigenvalues: Vec<f64>,
    /// Canonical correlations `ρ_i = sqrt(λ_i / (1 + λ_i))`. Length `n_functions`.
    pub canonical_correlations: Vec<f64>,
    /// Proportion of discriminative variance per function, `λ_i / Σ λ`.
    /// Length `n_functions`.
    pub variance_explained: Vec<f64>,
    /// Raw (unstandardized) discriminant coefficients, `n_functions × p`,
    /// row-major: row `i` is the loading vector of function `i`.
    pub raw_coefficients: Vec<f64>,
    /// Standardized discriminant coefficients, `n_functions × p`, row-major.
    /// Standardization uses the pooled within-group standard deviations.
    pub standardized_coefficients: Vec<f64>,
    /// Structure correlations (canonical loadings): correlation between each
    /// original variable and each discriminant function, `n_functions × p`,
    /// row-major.
    pub structure_correlations: Vec<f64>,
    /// Roy's largest root (the leading eigenvalue `λ_1`).
    pub roys_largest_root: f64,
    /// Per-variable univariate follow-up ANOVAs. Length `p`.
    pub univariate_anovas: Vec<UnivariateAnova>,
}

/// Run the MANOVA follow-up battery on a flat, row-major design.
///
/// # Arguments
/// * `data` — `n * p` values, row-major; row `i` is the `p`-dimensional
///   observation `i`.
/// * `n` — number of observations (rows).
/// * `p` — number of dependent variables (columns).
/// * `labels` — group label per observation, each in `0..g`. Length `n`.
/// * `g` — number of groups.
/// * `alpha` — family-wise significance level for the Bonferroni flag.
///
/// # Errors
/// Returns an error when fewer than two groups are supplied, the design is
/// empty or dimensionally inconsistent, a label is out of range, a group has
/// too few observations, the data contain non-finite values, or the within
/// SSCP matrix `W` is singular (cannot invert / factor).
pub fn manova_followup(
    data: &[f64],
    n: usize,
    p: usize,
    labels: &[usize],
    g: usize,
    alpha: f64,
) -> StatsResult<ManovaFollowup> {
    validate_inputs(data, n, p, labels, g, alpha)?;

    // Group sizes and per-group means; grand mean.
    let mut group_counts = vec![0usize; g];
    for &lbl in labels {
        group_counts[lbl] += 1;
    }
    for &count in &group_counts {
        // Need >= 2 per group so a within-group variance exists.
        if count < 2 {
            return Err(StatsError::InsufficientSampleSize {
                got: count,
                need: 2,
            });
        }
    }

    let mut group_means = vec![0.0f64; g * p];
    let mut grand_mean = vec![0.0f64; p];
    for i in 0..n {
        let lbl = labels[i];
        for j in 0..p {
            let v = data[i * p + j];
            group_means[lbl * p + j] += v;
            grand_mean[j] += v;
        }
    }
    for gm in grand_mean.iter_mut().take(p) {
        *gm /= n as f64;
    }
    for k in 0..g {
        let nk = group_counts[k] as f64;
        for j in 0..p {
            group_means[k * p + j] /= nk;
        }
    }

    // Between (B) and Within (W) SSCP matrices (p x p, symmetric).
    let (b_mat, w_mat) = between_within_sscp(data, n, p, labels, g, &group_means, &grand_mean);

    // Pooled within-group variances/covariances scaled to standard deviations.
    // The within SSCP divided by (n - g) is the pooled covariance estimate.
    let df_within = (n - g) as f64;
    let mut pooled_sd = vec![0.0f64; p];
    for j in 0..p {
        let var_j = w_mat[j * p + j] / df_within;
        if var_j <= 0.0 {
            return Err(StatsError::NumericalInstability(format!(
                "variable {j} has zero pooled within-group variance"
            )));
        }
        pooled_sd[j] = var_j.sqrt();
    }

    // Solve the generalized eigenproblem W^{-1} B v = lambda v via the
    // symmetric form S = W^{-1/2} B W^{-1/2}.
    let (eigenvalues_full, eigenvectors_full) = generalized_symmetric_eig(&w_mat, &b_mat, p)?;

    let n_functions = (g - 1).min(p);
    // Keep the leading `n_functions` eigenpairs (already sorted descending).
    let mut eigenvalues = Vec::with_capacity(n_functions);
    for &lam in eigenvalues_full.iter().take(n_functions) {
        // Clamp tiny negatives produced by round-off to zero.
        eigenvalues.push(if lam < 0.0 && lam > -1e-9 { 0.0 } else { lam });
    }

    // Raw discriminant coefficients: columns of `eigenvectors_full` are the
    // generalized eigenvectors v_i (loadings in the original variable space).
    // We store them row-major as `n_functions x p`.
    let mut raw_coefficients = vec![0.0f64; n_functions * p];
    for f in 0..n_functions {
        for j in 0..p {
            raw_coefficients[f * p + j] = eigenvectors_full[j * p + f];
        }
    }

    // Standardized coefficients: raw_{f,j} * pooled_sd_j (within-group
    // standardization, the conventional reporting convention).
    let mut standardized_coefficients = vec![0.0f64; n_functions * p];
    for f in 0..n_functions {
        for j in 0..p {
            standardized_coefficients[f * p + j] = raw_coefficients[f * p + j] * pooled_sd[j];
        }
    }

    // Canonical correlations and variance explained.
    let mut canonical_correlations = Vec::with_capacity(n_functions);
    for &lam in &eigenvalues {
        let l = lam.max(0.0);
        canonical_correlations.push((l / (1.0 + l)).sqrt());
    }
    let sum_lambda: f64 = eigenvalues.iter().map(|&l| l.max(0.0)).sum();
    let mut variance_explained = Vec::with_capacity(n_functions);
    for &lam in &eigenvalues {
        if sum_lambda > 0.0 {
            variance_explained.push(lam.max(0.0) / sum_lambda);
        } else {
            variance_explained.push(0.0);
        }
    }

    // Structure correlations: correlation between each original variable and
    // each canonical discriminant score. We compute discriminant scores per
    // observation, then Pearson-correlate against each centered variable.
    let structure_correlations =
        structure_correlations_matrix(data, n, p, n_functions, &raw_coefficients, &grand_mean);

    // Roy's largest root.
    let roys_largest_root = eigenvalues.first().copied().unwrap_or(0.0);

    // Univariate follow-up ANOVAs (one per dependent variable) with Bonferroni.
    let univariate_anovas = univariate_followups(data, n, p, labels, g, &group_counts, alpha)?;

    Ok(ManovaFollowup {
        g,
        p,
        n,
        n_functions,
        eigenvalues,
        canonical_correlations,
        variance_explained,
        raw_coefficients,
        standardized_coefficients,
        structure_correlations,
        roys_largest_root,
        univariate_anovas,
    })
}

/// Validate the flat-design inputs.
fn validate_inputs(
    data: &[f64],
    n: usize,
    p: usize,
    labels: &[usize],
    g: usize,
    alpha: f64,
) -> StatsResult<()> {
    if g < 2 {
        return Err(StatsError::InsufficientSampleSize { got: g, need: 2 });
    }
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "p".into(),
            reason: "must be > 0".into(),
        });
    }
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if data.len() != n * p {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![data.len()],
        });
    }
    if labels.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: labels.len(),
            b: n,
        });
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(StatsError::ProbabilityOutOfRange { value: alpha });
    }
    for (idx, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(idx));
        }
    }
    for &lbl in labels {
        if lbl >= g {
            return Err(StatsError::IndexOutOfBounds { index: lbl, len: g });
        }
    }
    // n must exceed g so df_within = n - g > 0.
    if n <= g {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: g + 1,
        });
    }
    Ok(())
}

/// Compute the between-groups (`B`) and within-groups (`W`) SSCP matrices.
///
/// `B = Σ_k n_k (m_k - m̄)(m_k - m̄)ᵀ` and
/// `W = Σ_k Σ_{i∈k} (x_i - m_k)(x_i - m_k)ᵀ`, both `p × p` and symmetric.
fn between_within_sscp(
    data: &[f64],
    n: usize,
    p: usize,
    labels: &[usize],
    g: usize,
    group_means: &[f64],
    grand_mean: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut b_mat = vec![0.0f64; p * p];
    let mut w_mat = vec![0.0f64; p * p];

    // Group sizes for B.
    let mut group_counts = vec![0usize; g];
    for &lbl in labels {
        group_counts[lbl] += 1;
    }
    for k in 0..g {
        let nk = group_counts[k] as f64;
        for i in 0..p {
            let di = group_means[k * p + i] - grand_mean[i];
            for j in 0..p {
                let dj = group_means[k * p + j] - grand_mean[j];
                b_mat[i * p + j] += nk * di * dj;
            }
        }
    }

    for r in 0..n {
        let lbl = labels[r];
        for i in 0..p {
            let di = data[r * p + i] - group_means[lbl * p + i];
            for j in 0..p {
                let dj = data[r * p + j] - group_means[lbl * p + j];
                w_mat[i * p + j] += di * dj;
            }
        }
    }
    (b_mat, w_mat)
}

/// Solve the generalized symmetric eigenproblem `W⁻¹B v = λ v` by reducing it
/// to the symmetric standard problem `S u = λ u` with
/// `S = W^{-1/2} B W^{-1/2}`, solving `S` with cyclic Jacobi rotations, and
/// mapping the eigenvectors back to the original space via `v = W^{-1/2} u`.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by descending eigenvalue.
/// `eigenvectors` is `p × p` column-major in the sense that column `c` (entries
/// `eigenvectors[row * p + c]`) is the generalized eigenvector `v_c`.
fn generalized_symmetric_eig(
    w_mat: &[f64],
    b_mat: &[f64],
    p: usize,
) -> StatsResult<(Vec<f64>, Vec<f64>)> {
    // W^{-1/2} via the symmetric inverse square root of W.
    let w_inv_half = symmetric_inverse_sqrt(w_mat, p)?;

    // S = W^{-1/2} B W^{-1/2}. Symmetrize against round-off.
    let tmp = matrix_mul(&w_inv_half, b_mat, p, p, p)?;
    let mut s = matrix_mul(&tmp, &w_inv_half, p, p, p)?;
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (s[i * p + j] + s[j * p + i]);
            s[i * p + j] = avg;
            s[j * p + i] = avg;
        }
    }

    let (mut evals, evecs_u) = jacobi_eigen(&s, p)?;

    // Map back: v = W^{-1/2} u (columns).
    let v_mat = matrix_mul(&w_inv_half, &evecs_u, p, p, p)?;

    // Sort descending by eigenvalue, permuting eigenvector columns.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sorted_vals = vec![0.0f64; p];
    let mut sorted_vecs = vec![0.0f64; p * p];
    for (new_c, &old_c) in order.iter().enumerate() {
        sorted_vals[new_c] = evals[old_c];
        for row in 0..p {
            sorted_vecs[row * p + new_c] = v_mat[row * p + old_c];
        }
    }
    evals = sorted_vals;
    Ok((evals, sorted_vecs))
}

/// Symmetric inverse square root `M^{-1/2}` of a symmetric positive-definite
/// matrix `M`, computed via its Jacobi eigendecomposition
/// `M = U diag(d) Uᵀ ⇒ M^{-1/2} = U diag(d^{-1/2}) Uᵀ`.
fn symmetric_inverse_sqrt(m: &[f64], p: usize) -> StatsResult<Vec<f64>> {
    // Symmetrize defensively.
    let mut sym = m.to_vec();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (sym[i * p + j] + sym[j * p + i]);
            sym[i * p + j] = avg;
            sym[j * p + i] = avg;
        }
    }
    let (evals, evecs) = jacobi_eigen(&sym, p)?;
    // d^{-1/2}; require positivity (W must be SPD for the DDA to be defined).
    let mut inv_sqrt_d = vec![0.0f64; p];
    for (i, &d) in evals.iter().enumerate() {
        if d <= 1e-12 {
            return Err(StatsError::SingularMatrix(
                "within SSCP matrix W is not positive definite".into(),
            ));
        }
        inv_sqrt_d[i] = 1.0 / d.sqrt();
    }
    // U diag(inv_sqrt_d) Uᵀ.
    // First form U * diag (scale columns of U).
    let mut ud = vec![0.0f64; p * p];
    for row in 0..p {
        for col in 0..p {
            ud[row * p + col] = evecs[row * p + col] * inv_sqrt_d[col];
        }
    }
    // Then (U diag) * Uᵀ.
    let mut out = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for k in 0..p {
                acc += ud[i * p + k] * evecs[j * p + k];
            }
            out[i * p + j] = acc;
        }
    }
    Ok(out)
}

/// Cyclic Jacobi eigensolver for a real symmetric `p × p` matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvector `c` is column `c`,
/// i.e. `eigenvectors[row * p + c]`. Eigenvalues are not pre-sorted.
fn jacobi_eigen(a_in: &[f64], p: usize) -> StatsResult<(Vec<f64>, Vec<f64>)> {
    if p == 0 {
        return Err(StatsError::EmptyInput);
    }
    let mut a = a_in.to_vec();
    // V starts as identity (eigenvectors accumulate here).
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    if p == 1 {
        return Ok((vec![a[0]], v));
    }

    let max_sweeps = 100usize;
    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0f64;
        for i in 0..p {
            for j in (i + 1)..p {
                off += a[i * p + j] * a[i * p + j];
            }
        }
        if off.sqrt() <= 1e-14 {
            break;
        }
        for q_idx in 0..p {
            for r_idx in (q_idx + 1)..p {
                let apq = a[q_idx * p + r_idx];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[q_idx * p + q_idx];
                let arr = a[r_idx * p + r_idx];
                // Rotation angle theta via the standard Jacobi formula.
                let phi = (arr - app) / (2.0 * apq);
                let t = phi.signum() / (phi.abs() + (phi * phi + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Apply rotation to rows/cols q_idx and r_idx of A.
                for k in 0..p {
                    let akq = a[k * p + q_idx];
                    let akr = a[k * p + r_idx];
                    a[k * p + q_idx] = c * akq - s * akr;
                    a[k * p + r_idx] = s * akq + c * akr;
                }
                for k in 0..p {
                    let aqk = a[q_idx * p + k];
                    let ark = a[r_idx * p + k];
                    a[q_idx * p + k] = c * aqk - s * ark;
                    a[r_idx * p + k] = s * aqk + c * ark;
                }
                // Accumulate eigenvectors.
                for k in 0..p {
                    let vkq = v[k * p + q_idx];
                    let vkr = v[k * p + r_idx];
                    v[k * p + q_idx] = c * vkq - s * vkr;
                    v[k * p + r_idx] = s * vkq + c * vkr;
                }
            }
        }
    }
    let mut evals = vec![0.0f64; p];
    for i in 0..p {
        evals[i] = a[i * p + i];
    }
    Ok((evals, v))
}

/// Structure correlations: Pearson correlation between each original variable
/// (centered at the grand mean) and each canonical discriminant score across
/// all observations. Returns an `n_functions × p` row-major matrix.
fn structure_correlations_matrix(
    data: &[f64],
    n: usize,
    p: usize,
    n_functions: usize,
    raw_coefficients: &[f64],
    grand_mean: &[f64],
) -> Vec<f64> {
    // Discriminant scores: score[i][f] = sum_j raw_{f,j} * (x_ij - mean_j).
    // (Centering the variables shifts scores by a constant, which does not
    // affect correlations, but keeps the arithmetic well-conditioned.)
    let mut scores = vec![0.0f64; n * n_functions];
    for i in 0..n {
        for f in 0..n_functions {
            let mut acc = 0.0;
            for j in 0..p {
                acc += raw_coefficients[f * p + j] * (data[i * p + j] - grand_mean[j]);
            }
            scores[i * n_functions + f] = acc;
        }
    }
    // Means and variances of scores.
    let mut score_mean = vec![0.0f64; n_functions];
    for i in 0..n {
        for f in 0..n_functions {
            score_mean[f] += scores[i * n_functions + f];
        }
    }
    for sm in score_mean.iter_mut().take(n_functions) {
        *sm /= n as f64;
    }
    let mut score_var = vec![0.0f64; n_functions];
    for i in 0..n {
        for f in 0..n_functions {
            let d = scores[i * n_functions + f] - score_mean[f];
            score_var[f] += d * d;
        }
    }
    // Variable variances (centered at grand mean -> use sample mean for var).
    let mut var_mean = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            var_mean[j] += data[i * p + j];
        }
    }
    for vm in var_mean.iter_mut().take(p) {
        *vm /= n as f64;
    }
    let mut var_var = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            let d = data[i * p + j] - var_mean[j];
            var_var[j] += d * d;
        }
    }

    let mut out = vec![0.0f64; n_functions * p];
    for f in 0..n_functions {
        for j in 0..p {
            let mut cov = 0.0;
            for i in 0..n {
                cov +=
                    (scores[i * n_functions + f] - score_mean[f]) * (data[i * p + j] - var_mean[j]);
            }
            let denom = (score_var[f] * var_var[j]).sqrt();
            out[f * p + j] = if denom > 1e-300 { cov / denom } else { 0.0 };
        }
    }
    out
}

/// One-way ANOVA for each dependent variable, with a Bonferroni-adjusted
/// significance flag across the `p` tests.
fn univariate_followups(
    data: &[f64],
    n: usize,
    p: usize,
    labels: &[usize],
    g: usize,
    group_counts: &[usize],
    alpha: f64,
) -> StatsResult<Vec<UnivariateAnova>> {
    let df_between = (g - 1) as f64;
    let df_within = (n - g) as f64;
    let mut out = Vec::with_capacity(p);
    for j in 0..p {
        // Grand mean and per-group means for variable j.
        let mut grand = 0.0f64;
        let mut group_sum = vec![0.0f64; g];
        for i in 0..n {
            let v = data[i * p + j];
            grand += v;
            group_sum[labels[i]] += v;
        }
        grand /= n as f64;
        let mut group_mean = vec![0.0f64; g];
        for k in 0..g {
            group_mean[k] = group_sum[k] / group_counts[k] as f64;
        }
        let mut ss_between = 0.0f64;
        for k in 0..g {
            ss_between += group_counts[k] as f64 * (group_mean[k] - grand).powi(2);
        }
        let mut ss_within = 0.0f64;
        for i in 0..n {
            let d = data[i * p + j] - group_mean[labels[i]];
            ss_within += d * d;
        }
        let ms_between = ss_between / df_between;
        let ms_within = ss_within / df_within;
        let (f_stat, p_value) = if ms_within <= 0.0 {
            // All within-group variance is zero: if there is any between-group
            // separation the effect is infinitely significant; otherwise none.
            if ss_between > 0.0 {
                (f64::INFINITY, 0.0)
            } else {
                (0.0, 1.0)
            }
        } else {
            let f = ms_between / ms_within;
            let fd = FDist::new(df_between, df_within)?;
            let pv = 1.0 - fd.cdf(f)?;
            (f, pv.clamp(0.0, 1.0))
        };
        let p_value_bonferroni = (p_value * p as f64).min(1.0);
        out.push(UnivariateAnova {
            variable: j,
            f_statistic: f_stat,
            df_between,
            df_within,
            p_value,
            p_value_bonferroni,
            significant_bonferroni: p_value_bonferroni < alpha,
        });
    }
    Ok(out)
}

/// Convenience: invert `W` (exposed for callers that want `W⁻¹B` explicitly).
///
/// Not used internally — the symmetric reduction is numerically preferable —
/// but provided so downstream code can verify the eigenproblem if desired.
///
/// # Errors
/// Returns an error if `W` is singular or shapes are inconsistent.
pub fn w_inv_b(w_mat: &[f64], b_mat: &[f64], p: usize) -> StatsResult<Vec<f64>> {
    let w_inv = matrix_inverse_lu(w_mat, p)?;
    matrix_mul(&w_inv, b_mat, p, p, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parametric::anova::one_way_anova;

    /// Three well-separated groups along axis 0, with axis 1 as pure (shared)
    /// noise structure. `seed`-free deterministic synthetic data.
    fn separated_data() -> (Vec<f64>, usize, usize, Vec<usize>, usize) {
        // p = 2, g = 3, 6 obs per group.
        let centers = [0.0, 10.0, 20.0]; // along variable 0
        let noise = [-0.3, -0.1, 0.05, 0.1, 0.2, 0.05]; // shared within-group jitter
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for (k, &c) in centers.iter().enumerate() {
            for (i, &z) in noise.iter().enumerate() {
                // Variable 0: strongly separated. Variable 1: same distribution
                // across groups (no separation) -> a pure-noise discriminator.
                data.push(c + z);
                data.push(2.0 + z * 0.5 + ((i as f64) * 0.01));
                labels.push(k);
            }
        }
        let n = labels.len();
        (data, n, 2, labels, 3)
    }

    #[test]
    fn rho_squared_identity_holds() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        for (i, &lam) in r.eigenvalues.iter().enumerate() {
            let rho = r.canonical_correlations[i];
            let expected = lam / (1.0 + lam);
            assert!(
                (rho * rho - expected).abs() < 1e-9,
                "rho^2 = {} but lambda/(1+lambda) = {} for function {i}",
                rho * rho,
                expected
            );
        }
    }

    #[test]
    fn first_function_captures_almost_all_variance() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        assert!(!r.variance_explained.is_empty());
        // With one dominant separation axis, the leading function explains
        // essentially all discriminative variance.
        assert!(
            r.variance_explained[0] > 0.99,
            "leading variance_explained = {}",
            r.variance_explained[0]
        );
        // Proportions sum to 1.
        let total: f64 = r.variance_explained.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "variance_explained sum = {total}"
        );
    }

    #[test]
    fn number_of_functions_is_min_gm1_p() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        assert_eq!(r.n_functions, (g - 1).min(p));
        assert_eq!(r.eigenvalues.len(), r.n_functions);
        assert_eq!(r.canonical_correlations.len(), r.n_functions);
        assert_eq!(r.standardized_coefficients.len(), r.n_functions * p);
        // Standardized coefficients must all be finite.
        for &c in &r.standardized_coefficients {
            assert!(c.is_finite(), "non-finite standardized coefficient");
        }
        for &c in &r.raw_coefficients {
            assert!(c.is_finite(), "non-finite raw coefficient");
        }
    }

    #[test]
    fn roys_root_is_leading_eigenvalue() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        assert!((r.roys_largest_root - r.eigenvalues[0]).abs() < 1e-12);
        // Sorted descending.
        for w in r.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-12, "eigenvalues not sorted descending");
        }
    }

    #[test]
    fn univariate_f_matches_independent_anova() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        // Cross-check each variable's F against an independent one-way ANOVA.
        for j in 0..p {
            // Build per-group slices for variable j.
            let mut groups: Vec<Vec<f64>> = vec![Vec::new(); g];
            for i in 0..n {
                groups[labels[i]].push(data[i * p + j]);
            }
            let refs: Vec<&[f64]> = groups.iter().map(|v| v.as_slice()).collect();
            let indep = one_way_anova(&refs).expect("anova ok");
            assert!(
                (r.univariate_anovas[j].f_statistic - indep.f_statistic).abs() < 1e-6,
                "variable {j}: followup F {} != independent F {}",
                r.univariate_anovas[j].f_statistic,
                indep.f_statistic
            );
        }
    }

    #[test]
    fn discriminating_variable_has_large_f_noise_small() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        // Variable 0 is the big separator; variable 1 barely separates.
        assert!(
            r.univariate_anovas[0].f_statistic > r.univariate_anovas[1].f_statistic,
            "expected variable 0 F {} > variable 1 F {}",
            r.univariate_anovas[0].f_statistic,
            r.univariate_anovas[1].f_statistic
        );
        // The strong separator should be Bonferroni-significant.
        assert!(r.univariate_anovas[0].significant_bonferroni);
        // Bonferroni-adjusted p is >= raw p.
        for ua in &r.univariate_anovas {
            assert!(ua.p_value_bonferroni >= ua.p_value - 1e-12);
        }
    }

    #[test]
    fn structure_correlations_in_range() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        assert_eq!(r.structure_correlations.len(), r.n_functions * p);
        for &c in &r.structure_correlations {
            assert!(c.is_finite() && c.abs() <= 1.0 + 1e-9, "structure corr {c}");
        }
        // The discriminating variable correlates strongly with function 0.
        assert!(
            r.structure_correlations[0].abs() > 0.9,
            "variable 0 structure corr with f0 = {}",
            r.structure_correlations[0]
        );
    }

    #[test]
    fn group_means_are_separated_on_first_function() {
        let (data, n, p, labels, g) = separated_data();
        let r = manova_followup(&data, n, p, &labels, g, 0.05).expect("ok");
        // Project each group's mean onto function 0 and verify the projections
        // are well separated (range >> within-group spread proxy).
        let mut grand = vec![0.0f64; p];
        for i in 0..n {
            for j in 0..p {
                grand[j] += data[i * p + j];
            }
        }
        for v in &mut grand {
            *v /= n as f64;
        }
        let mut group_proj = vec![Vec::<f64>::new(); g];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..p {
                s += r.raw_coefficients[j] * (data[i * p + j] - grand[j]);
            }
            group_proj[labels[i]].push(s);
        }
        let means: Vec<f64> = group_proj
            .iter()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect();
        let max_m = means.iter().cloned().fold(f64::MIN, f64::max);
        let min_m = means.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            (max_m - min_m).abs() > 1.0,
            "group mean projections not separated: {means:?}"
        );
    }

    #[test]
    fn rejects_fewer_than_two_groups() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![0usize, 0];
        assert!(manova_followup(&data, 2, 2, &labels, 1, 0.05).is_err());
    }

    #[test]
    fn rejects_label_out_of_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // label 2 is out of range for g = 2.
        let labels = vec![0usize, 1, 2];
        assert!(matches!(
            manova_followup(&data, 3, 2, &labels, 2, 0.05),
            Err(StatsError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        // data has 4 entries but n*p = 3*2 = 6.
        let labels = vec![0usize, 1, 0];
        assert!(matches!(
            manova_followup(&data, 3, 2, &labels, 2, 0.05),
            Err(StatsError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_group_with_too_few_observations() {
        // g = 2, but group 1 has only a single observation.
        let data = vec![
            1.0, 2.0, // group 0
            1.1, 2.1, // group 0
            1.2, 1.9, // group 0
            5.0, 9.0, // group 1 (only one)
        ];
        let labels = vec![0usize, 0, 0, 1];
        assert!(matches!(
            manova_followup(&data, 4, 2, &labels, 2, 0.05),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    #[test]
    fn rejects_singular_within_matrix() {
        // Make variable 1 an exact affine copy of variable 0 within every
        // observation -> W is rank-deficient (singular), so the symmetric
        // inverse square root must fail.
        let mut data = Vec::new();
        let mut labels = Vec::new();
        let centers = [0.0, 5.0];
        let jitter = [-0.2, -0.1, 0.1, 0.2];
        for (k, &c) in centers.iter().enumerate() {
            for &z in &jitter {
                let x = c + z;
                data.push(x);
                data.push(3.0 * x + 1.0); // perfectly collinear with variable 0
                labels.push(k);
            }
        }
        let n = labels.len();
        let res = manova_followup(&data, n, 2, &labels, 2, 0.05);
        assert!(
            matches!(res, Err(StatsError::SingularMatrix(_))),
            "expected SingularMatrix, got {res:?}"
        );
    }

    #[test]
    fn rejects_non_finite_value() {
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let labels = vec![0usize, 1, 0];
        assert!(matches!(
            manova_followup(&data, 3, 2, &labels, 2, 0.05),
            Err(StatsError::NonFiniteValue(_))
        ));
    }
}
