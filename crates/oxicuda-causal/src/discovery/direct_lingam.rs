//! DirectLiNGAM — A direct method for learning a linear non-Gaussian SEM.
//!
//! Reference: Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y.,
//! Washio, T., Hoyer, P. O., & Bollen, K. (2011). "DirectLiNGAM: A Direct Method
//! for Learning a Linear Non-Gaussian Structural Equation Model." *Journal of
//! Machine Learning Research*, 12, 1225-1248.
//!
//! # Algorithm
//!
//! Given observed data `X ∈ ℝ^{n × d}` produced by a linear non-Gaussian
//! structural equation model `x_i = Σ_j b_ij x_j + e_i` (with mutually
//! independent, non-Gaussian errors `e_i`), DirectLiNGAM recovers a causal
//! ordering `π` of the variables one at a time:
//!
//! 1. For each round `k = 0..d`:
//!    - For every variable `i` not yet ordered, regress every other
//!      not-yet-ordered variable `j` on `x_i`, obtaining residuals `e_j(i)`.
//!    - Score `i` by the **non-Gaussianity of those residuals**.  The
//!      most-exogenous candidate `i*` produces residuals that are *most
//!      non-Gaussian* (the LiNGAM identifiability theorem).
//!    - Append `i*` to the ordering; replace every other not-yet-ordered
//!      variable by its residual after regressing on `x_{i*}`.
//! 2. Once the ordering is fixed, the strictly lower-triangular matrix `B` of
//!    structural coefficients (in the permuted basis) is recovered by
//!    regressing variable at order-position `k` on its predecessors.
//!
//! # Non-Gaussianity measure
//!
//! Following the classic LiNGAM kurtosis-based ordering (Shimizu, 2006), we
//! score residual non-Gaussianity by the squared excess kurtosis.  For a
//! candidate `i` and residuals `{e_j(i)}_{j ≠ i}`, define
//!
//! ```text
//! T(i) = - Σ_{j ≠ i} (kurtosis(e_j(i)) - 3)^2
//! ```
//!
//! and pick `i* = argmin T(i)` (equivalently, the candidate whose induced
//! residuals are *most non-Gaussian* in the L² excess-kurtosis sense).  This
//! is equivalent up to lower-order terms to the original entropy/MI-based
//! score for many heavy-tailed and platykurtic non-Gaussian noises.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`direct_lingam`].
#[derive(Debug, Clone)]
pub struct DirectLingamConfig {
    /// Safety cap on the outer loop. Must satisfy `max_iter ≥ d`.
    pub max_iter: usize,
    /// Ridge regularisation on the per-candidate OLS step (avoids singular
    /// matrices when columns are collinear). Must be strictly positive.
    /// A typical value is `1e-8`.
    pub reg: f64,
}

impl Default for DirectLingamConfig {
    fn default() -> Self {
        Self {
            max_iter: 64,
            reg: 1e-8,
        }
    }
}

/// Result of running [`direct_lingam`].
#[derive(Debug, Clone)]
pub struct DirectLingamResult {
    /// Causal ordering of the `d` variables (length `d`).  Entry `ordering[k]`
    /// is the index (in the original data) of the variable placed at
    /// order-position `k`.
    pub ordering: Vec<usize>,
    /// Strictly-lower-triangular `d×d` matrix of structural coefficients in
    /// the **original variable basis**, stored row-major (length `d*d`).
    /// Entry `b[i*d + j]` is the structural coefficient on variable `j` in
    /// the equation for variable `i`.  Entries on or above the diagonal of
    /// the permuted ordering are zero by construction.
    pub b: Vec<f64>,
}

/// Run DirectLiNGAM on a row-major data matrix `x` of shape `(n, d)`.
///
/// # Parameters
/// - `x`: row-major `n × d` data matrix (`x.len() == n * d`).
/// - `n`: number of samples; must satisfy `n ≥ d + 1`.
/// - `d`: number of variables; must satisfy `d ≥ 2`.
/// - `cfg`: see [`DirectLingamConfig`].
///
/// # Errors
/// - [`CausalError::EmptyInput`] if `d == 0` or `x.is_empty()`.
/// - [`CausalError::DimensionMismatch`] if `x.len() != n * d`.
/// - [`CausalError::IncompatibleData`] if `d == 1`, `n < d + 1`,
///   `reg ≤ 0`, or `max_iter < d`.
pub fn direct_lingam(
    x: &[f64],
    n: usize,
    d: usize,
    cfg: &DirectLingamConfig,
) -> CausalResult<DirectLingamResult> {
    // ---- input validation ------------------------------------------------
    if d == 0 || x.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    if x.len() != n * d {
        return Err(CausalError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    if d < 2 || n < d + 1 || cfg.reg <= 0.0 || cfg.max_iter < d {
        return Err(CausalError::IncompatibleData);
    }

    // ---- working buffer (column-centred) --------------------------------
    let mut work = x.to_vec();
    centre_columns(&mut work, n, d);

    let mut ordering: Vec<usize> = Vec::with_capacity(d);
    let mut remaining: Vec<usize> = (0..d).collect();

    // ---- iterative most-exogenous selection -----------------------------
    while !remaining.is_empty() {
        let exo_pos = pick_most_exogenous(&work, &remaining, n, d, cfg.reg)?;
        let exo_var = remaining[exo_pos];
        ordering.push(exo_var);
        remaining.swap_remove(exo_pos);

        if remaining.is_empty() {
            break;
        }
        // Replace each remaining column by its residual after regressing on
        // the just-selected exogenous variable.  This is the LiNGAM
        // "subtract the explained part" recursion.
        residualise_columns(&mut work, exo_var, &remaining, n, d, cfg.reg);
    }

    // ---- recover B from the original (un-residualised) centred data -----
    let mut centred = x.to_vec();
    centre_columns(&mut centred, n, d);
    let b = recover_b_matrix(&centred, &ordering, n, d, cfg.reg)?;

    Ok(DirectLingamResult { ordering, b })
}

// =====================================================================
// helpers
// =====================================================================

/// Subtract column-wise mean from `x` in place.
fn centre_columns(x: &mut [f64], n: usize, d: usize) {
    for j in 0..d {
        let mut mu = 0.0_f64;
        for i in 0..n {
            mu += x[i * d + j];
        }
        mu /= n as f64;
        for i in 0..n {
            x[i * d + j] -= mu;
        }
    }
}

/// Sample excess kurtosis: `(n Σ(e - μ)^4) / (Σ(e - μ)^2)^2 - 3`.
///
/// For a centred sample this simplifies to `n Σe^4 / (Σe^2)^2 - 3`.
fn excess_kurtosis(e: &[f64]) -> f64 {
    let n = e.len();
    if n == 0 {
        return 0.0;
    }
    let mu = e.iter().sum::<f64>() / n as f64;
    let mut m2 = 0.0_f64;
    let mut m4 = 0.0_f64;
    for &v in e {
        let z = v - mu;
        let z2 = z * z;
        m2 += z2;
        m4 += z2 * z2;
    }
    if m2 <= 1e-30 {
        return 0.0;
    }
    let num = (n as f64) * m4;
    let denom = m2 * m2;
    num / denom - 3.0
}

/// Pick the index (within `remaining`) of the most-exogenous candidate.
///
/// Returns the **position inside the `remaining` slice**, not the variable
/// index in the original data.
fn pick_most_exogenous(
    x: &[f64],
    remaining: &[usize],
    n: usize,
    d: usize,
    reg: f64,
) -> CausalResult<usize> {
    if remaining.len() == 1 {
        return Ok(0);
    }
    let mut best_pos = 0_usize;
    let mut best_score = f64::INFINITY;
    for (pos, &cand) in remaining.iter().enumerate() {
        // T(cand) = - Σ_j (kurtosis(e_j(cand)) - 3)^2
        let mut score = 0.0_f64;
        for &other in remaining {
            if other == cand {
                continue;
            }
            let resid = ridge_residual(x, cand, other, n, d, reg);
            let ek = excess_kurtosis(&resid);
            score -= ek * ek;
        }
        if score < best_score {
            best_score = score;
            best_pos = pos;
        }
    }
    Ok(best_pos)
}

/// Compute the residual of `x_other` regressed on `x_cand` with ridge `reg`.
///
/// For univariate OLS through the origin (data already centred):
/// `β = Σ x_cand · x_other / (Σ x_cand² + reg)`,
/// residual `e_i = x_other_i - β · x_cand_i`.
fn ridge_residual(x: &[f64], cand: usize, other: usize, n: usize, d: usize, reg: f64) -> Vec<f64> {
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let xc = x[i * d + cand];
        let xo = x[i * d + other];
        sxx += xc * xc;
        sxy += xc * xo;
    }
    let beta = sxy / (sxx + reg);
    let mut e = vec![0.0_f64; n];
    for i in 0..n {
        e[i] = x[i * d + other] - beta * x[i * d + cand];
    }
    e
}

/// Replace, in place, each column listed in `remaining` by its residual after
/// regressing on column `exo_var`.
fn residualise_columns(
    x: &mut [f64],
    exo_var: usize,
    remaining: &[usize],
    n: usize,
    d: usize,
    reg: f64,
) {
    for &other in remaining {
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        for i in 0..n {
            let xc = x[i * d + exo_var];
            let xo = x[i * d + other];
            sxx += xc * xc;
            sxy += xc * xo;
        }
        let beta = sxy / (sxx + reg);
        for i in 0..n {
            x[i * d + other] -= beta * x[i * d + exo_var];
        }
    }
}

/// Solve a `d × d` symmetric positive-(semi-)definite system `A β = b` by
/// Gauss-Jordan with partial pivoting.
fn solve_dense(a: &[f64], b: &[f64], n: usize) -> CausalResult<Vec<f64>> {
    // Build augmented matrix [A | b].
    let cols = n + 1;
    let mut m = vec![0.0_f64; n * cols];
    for i in 0..n {
        for j in 0..n {
            m[i * cols + j] = a[i * n + j];
        }
        m[i * cols + n] = b[i];
    }
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..n {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        x[i] = m[i * cols + n];
    }
    Ok(x)
}

/// Recover the strictly lower-triangular structural matrix `B` (in the
/// *original* variable basis) given a causal ordering.
fn recover_b_matrix(
    centred: &[f64],
    ordering: &[usize],
    n: usize,
    d: usize,
    reg: f64,
) -> CausalResult<Vec<f64>> {
    let mut b = vec![0.0_f64; d * d];
    for (k, &var) in ordering.iter().enumerate() {
        if k == 0 {
            continue;
        }
        // Predictors = ordering[0..k].
        let preds: &[usize] = &ordering[..k];
        let dp = preds.len();
        // Build A = X_preds^T X_preds + reg I,  c = X_preds^T y.
        let mut a = vec![0.0_f64; dp * dp];
        let mut c = vec![0.0_f64; dp];
        for ii in 0..dp {
            for jj in 0..dp {
                let mut s = 0.0_f64;
                for row in 0..n {
                    s += centred[row * d + preds[ii]] * centred[row * d + preds[jj]];
                }
                a[ii * dp + jj] = s;
            }
            a[ii * dp + ii] += reg;
            let mut s = 0.0_f64;
            for row in 0..n {
                s += centred[row * d + preds[ii]] * centred[row * d + var];
            }
            c[ii] = s;
        }
        let beta = solve_dense(&a, &c, dp)?;
        for (jj, &pred) in preds.iter().enumerate() {
            b[var * d + pred] = beta[jj];
        }
    }
    Ok(b)
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng_uniform(rng: &mut LcgRng) -> f64 {
        // Map LcgRng::next_f32 ∈ [0,1) into (-1, 1).
        (rng.next_f32() as f64) * 2.0 - 1.0
    }

    fn make_chain_two(n: usize, beta: f64, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0.0_f64; n * 2];
        for i in 0..n {
            let u1 = rng_uniform(&mut rng);
            let u2 = rng_uniform(&mut rng);
            data[i * 2] = u1;
            data[i * 2 + 1] = beta * u1 + u2;
        }
        data
    }

    #[test]
    fn invalid_empty() {
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&[], 0, 0, &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_d_one() {
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&[1.0, 2.0, 3.0], 3, 1, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_n_too_small() {
        // n=2, d=2 → n < d+1 (=3).
        let cfg = DirectLingamConfig::default();
        let data = vec![0.0_f64; 4];
        let r = direct_lingam(&data, 2, 2, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_reg_zero() {
        let cfg = DirectLingamConfig {
            max_iter: 32,
            reg: 0.0,
        };
        let data = vec![0.0_f64; 30];
        let r = direct_lingam(&data, 10, 3, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_reg_negative() {
        let cfg = DirectLingamConfig {
            max_iter: 32,
            reg: -1.0,
        };
        let data = vec![0.0_f64; 30];
        let r = direct_lingam(&data, 10, 3, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_dim_mismatch() {
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&[1.0, 2.0, 3.0], 4, 2, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_max_iter_too_small() {
        let cfg = DirectLingamConfig {
            max_iter: 1, // need ≥ d = 3
            reg: 1e-8,
        };
        let data = vec![0.0_f64; 30];
        let r = direct_lingam(&data, 10, 3, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn recovers_two_variable_chain_ordering() {
        // x1 = u1, x2 = 0.5 x1 + u2, both noises uniform[-1,1].
        let n = 1024;
        let data = make_chain_two(n, 0.5, 1234);
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        assert_eq!(r.ordering, vec![0, 1]);
    }

    #[test]
    fn recovers_two_variable_chain_b_coeff() {
        // For d=2 and the canonical row-major layout, B[1,0] is at index 2.
        let n = 2048;
        let data = make_chain_two(n, 0.5, 9876);
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        let beta = r.b[2]; // B[1, 0]
        assert!(
            (beta - 0.5).abs() < 0.05,
            "expected ~0.5, got {beta}; ordering={:?}",
            r.ordering
        );
        // Diagonal & upper-triangular (in permuted basis) must be zero.
        assert_eq!(r.b[0], 0.0); // B[0, 0]
        assert_eq!(r.b[3], 0.0); // B[1, 1]
        assert_eq!(r.b[1], 0.0); // B[0, 1]
    }

    #[test]
    fn recovers_three_variable_chain() {
        // x1 = u1, x2 = 0.7 x1 + u2, x3 = 0.4 x2 + u3.
        let n = 1536;
        let mut rng = LcgRng::new(424242);
        let mut data = vec![0.0_f64; n * 3];
        for i in 0..n {
            let u1 = rng_uniform(&mut rng);
            let u2 = rng_uniform(&mut rng);
            let u3 = rng_uniform(&mut rng);
            data[i * 3] = u1;
            data[i * 3 + 1] = 0.7 * u1 + u2;
            data[i * 3 + 2] = 0.4 * data[i * 3 + 1] + u3;
        }
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, 3, &cfg).expect("direct_lingam should succeed");
        // Ordering: 0 must precede 1, which must precede 2.
        let pos: Vec<usize> = {
            let mut p = vec![0_usize; 3];
            for (k, &v) in r.ordering.iter().enumerate() {
                p[v] = k;
            }
            p
        };
        assert!(
            pos[0] < pos[1] && pos[1] < pos[2],
            "ordering = {:?}, positions = {:?}",
            r.ordering,
            pos
        );
    }

    #[test]
    fn permutation_invariance() {
        // Permute the columns of a chain and check the recovered DAG is the
        // same up to the relabelling.
        let n = 1024;
        let base = make_chain_two(n, 0.5, 24680);
        // Swap columns 0 and 1.
        let mut swapped = vec![0.0_f64; n * 2];
        for i in 0..n {
            swapped[i * 2] = base[i * 2 + 1];
            swapped[i * 2 + 1] = base[i * 2];
        }
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&swapped, n, 2, &cfg).expect("direct_lingam should succeed");
        // The exogenous variable is now column 1 (originally x1).
        assert_eq!(r.ordering, vec![1, 0]);
        let beta = r.b[1]; // B[0, 1]
        assert!(
            (beta - 0.5).abs() < 0.05,
            "expected ~0.5, got {beta}; ordering={:?}",
            r.ordering
        );
    }

    #[test]
    fn deterministic_with_seed() {
        let n = 256;
        let data = make_chain_two(n, 0.5, 31415);
        let cfg = DirectLingamConfig::default();
        let r1 = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        let r2 = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        assert_eq!(r1.ordering, r2.ordering);
        assert_eq!(r1.b, r2.b);
    }

    #[test]
    fn ridge_robustness_minimal_n() {
        // n = d + 1 = 3 — the smallest legal sample size.
        let n = 3_usize;
        let d = 2_usize;
        let data = vec![1.0_f64, 0.5, -0.5, 0.25, 0.7, -0.3];
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, d, &cfg);
        assert!(r.is_ok(), "ridge OLS must not blow up at n=d+1");
        let r = r.expect("r should be present");
        assert_eq!(r.ordering.len(), 2);
        assert_eq!(r.b.len(), 4);
        for v in &r.b {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn large_n_runs() {
        let n = 4096;
        let data = make_chain_two(n, 0.5, 55555);
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        assert_eq!(r.ordering, vec![0, 1]);
        let beta = r.b[2]; // B[1, 0]
        assert!((beta - 0.5).abs() < 0.03, "large-n β = {beta}");
    }

    #[test]
    fn excess_kurtosis_uniform_is_negative() {
        // Uniform[-1,1] has excess kurtosis = -1.2 in the limit.
        let mut rng = LcgRng::new(2024);
        let mut samples = vec![0.0_f64; 8192];
        for v in samples.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
        let ek = excess_kurtosis(&samples);
        assert!(
            ek < -0.5,
            "uniform should give negative excess kurtosis, got {ek}"
        );
    }

    #[test]
    fn excess_kurtosis_zero_on_zero_data() {
        let zeros = vec![0.0_f64; 16];
        let ek = excess_kurtosis(&zeros);
        assert!(ek.abs() < 1e-12);
    }

    #[test]
    fn b_matrix_strictly_lower_triangular_in_ordering() {
        // For any recovered ordering π, b[π[k], π[m]] must be 0 for all m ≥ k.
        let n = 1024;
        let data = make_chain_two(n, 0.5, 67890);
        let cfg = DirectLingamConfig::default();
        let r = direct_lingam(&data, n, 2, &cfg).expect("direct_lingam should succeed");
        let d = 2_usize;
        for k in 0..d {
            for m in k..d {
                let i = r.ordering[k];
                let j = r.ordering[m];
                let entry = r.b[i * d + j];
                assert_eq!(
                    entry, 0.0,
                    "B[{i},{j}] in ordering-position ({k},{m}) must be zero"
                );
            }
        }
    }

    #[test]
    fn config_default_values() {
        let cfg = DirectLingamConfig::default();
        assert!(cfg.max_iter >= 2);
        assert!(cfg.reg > 0.0);
    }
}
