//! R-Learner — Quasi-oracle estimation of heterogeneous treatment effects.
//!
//! Reference: Nie, X. & Wager, S. (2021). "Quasi-oracle estimation of
//! heterogeneous treatment effects." *Biometrika*, 108(2), 299-319.
//! See also Robinson, P. M. (1988). "Root-N-consistent semiparametric
//! regression." *Econometrica*, 56(4), 931-954.
//!
//! # Algorithm
//!
//! The R-Learner minimises the **residualised loss**
//!
//! ```text
//!   τ̂ = argmin_τ  Σ_i  [ (Y_i - g(X_i))  -  τ(X_i) · (T_i - m(X_i)) ]^2,
//! ```
//!
//! where the nuisance regressions `g(X) = E[Y | X]` and `m(X) = E[T | X]`
//! are estimated via **K-fold cross-fitting**: the predicted values
//! `ĝ(X_i)` and `m̂(X_i)` for each `i` are produced by models fitted on
//! the *complement* fold (i.e. on the data not containing `i`).  Cross-
//! fitting eliminates the over-fitting bias that would otherwise spoil
//! the orthogonality of the residualised loss (Chernozhukov et al. 2018).
//!
//! For a per-sample CATE we adopt the Robinson partial-out closed form
//!
//! ```text
//!   τ̂(x_i) = (T_i - m̂(x_i)) · (Y_i - ĝ(x_i))
//!            / ( (T_i - m̂(x_i))^2 + λ_τ ),
//! ```
//!
//! where `λ_τ > 0` is a small stabiliser preventing division by near-zero
//! treatment residuals.  The aggregate ATE is reported as the simple
//! arithmetic mean of the per-sample CATEs.
//!
//! # Nuisance regressions
//!
//! Both `g` and `m` are estimated by ridge regression with an intercept,
//!
//! ```text
//!   β̂ = (X̃^T X̃ + λ I)^{-1} X̃^T y,
//! ```
//!
//! where `X̃` is `X` augmented with a column of ones.  Independent ridge
//! parameters `ridge_g` and `ridge_m` are exposed via the configuration.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`r_learner`].
#[derive(Debug, Clone)]
pub struct RLearnerConfig {
    /// Number of cross-fitting folds. Must satisfy `n_folds ≥ 2`.
    pub n_folds: usize,
    /// Ridge penalty for the outcome regression `g(X) = E[Y|X]`. Must be
    /// strictly positive.
    pub ridge_g: f64,
    /// Ridge penalty for the treatment regression `m(X) = E[T|X]`. Must be
    /// strictly positive.
    pub ridge_m: f64,
    /// Stabiliser added to the denominator of the per-sample CATE formula.
    /// Must be strictly positive.  A typical value is `1e-3`.
    pub ridge_tau: f64,
}

impl Default for RLearnerConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            ridge_g: 1e-3,
            ridge_m: 1e-3,
            ridge_tau: 1e-6,
        }
    }
}

/// Result of running [`r_learner`].
#[derive(Debug, Clone)]
pub struct RLearnerResult {
    /// Conditional average treatment effect at each sample. Length `n`.
    pub cate: Vec<f64>,
    /// Aggregate ATE — arithmetic mean of [`Self::cate`].
    pub ate: f64,
    /// Cross-fitted outcome predictions `ĝ(X_i)`. Length `n`.
    pub g_hat: Vec<f64>,
    /// Cross-fitted treatment predictions `m̂(X_i)`. Length `n`.
    pub m_hat: Vec<f64>,
}

/// Fit the R-Learner on `(X, Y, T)` and return per-sample CATE estimates.
///
/// # Parameters
/// - `x`: row-major `n × d` covariate matrix (`x.len() == n * d`).
/// - `n`: number of samples.
/// - `d`: number of covariates.
/// - `y`: outcomes, length `n`.
/// - `t`: treatments, length `n` (can be binary `{0, 1}` or continuous).
/// - `cfg`: see [`RLearnerConfig`].
///
/// The folds are formed deterministically by `index % n_folds` — tests can
/// shuffle the data upstream if randomisation is desired.
///
/// # Errors
/// - [`CausalError::EmptyInput`] if any array is empty, `n == 0`, or `d == 0`.
/// - [`CausalError::DimensionMismatch`] if `x.len() != n * d`,
///   `y.len() != n`, or `t.len() != n`.
/// - [`CausalError::InvalidNumFolds`] if `n_folds < 2` or
///   `n < n_folds * (d + 1)`.
/// - [`CausalError::IncompatibleData`] if any ridge value is non-positive.
/// - [`CausalError::MatrixSingular`] if the augmented design matrix in any
///   fold is rank-deficient even after ridge regularisation.
pub fn r_learner(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    t: &[f64],
    cfg: &RLearnerConfig,
) -> CausalResult<RLearnerResult> {
    // ---- input validation -----------------------------------------------
    if n == 0 || d == 0 || x.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    if x.len() != n * d {
        return Err(CausalError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    if y.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if t.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len(),
        });
    }
    if cfg.n_folds < 2 {
        return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
    }
    if cfg.ridge_g <= 0.0 || cfg.ridge_m <= 0.0 || cfg.ridge_tau <= 0.0 {
        return Err(CausalError::IncompatibleData);
    }
    // We need at least d+1 training rows per fold for a unique ridge solve
    // (the augmented design has d+1 columns).  Equivalently
    // n - fold_size ≥ d + 1 with the tightest fold_size = ceil(n / K).
    if n < cfg.n_folds * (d + 1) {
        return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
    }

    let k = cfg.n_folds;
    let mut g_hat = vec![0.0_f64; n];
    let mut m_hat = vec![0.0_f64; n];

    // Fold assignment: fold_id[i] = i % k.  Deterministic, no rand.
    for fold in 0..k {
        // Indices belonging to this held-out fold.
        let test_idx: Vec<usize> = (0..n).filter(|i| i % k == fold).collect();
        let train_idx: Vec<usize> = (0..n).filter(|i| i % k != fold).collect();
        let n_train = train_idx.len();
        if n_train < d + 1 {
            return Err(CausalError::InvalidNumFolds { k });
        }

        // Build training matrices (row-major) with an intercept column.
        let dp1 = d + 1;
        let mut x_train = vec![0.0_f64; n_train * dp1];
        let mut y_train = vec![0.0_f64; n_train];
        let mut t_train = vec![0.0_f64; n_train];
        for (r, &orig) in train_idx.iter().enumerate() {
            for j in 0..d {
                x_train[r * dp1 + j] = x[orig * d + j];
            }
            x_train[r * dp1 + d] = 1.0;
            y_train[r] = y[orig];
            t_train[r] = t[orig];
        }

        let beta_g = ridge_solve(&x_train, &y_train, n_train, dp1, cfg.ridge_g)?;
        let beta_m = ridge_solve(&x_train, &t_train, n_train, dp1, cfg.ridge_m)?;

        for &i in &test_idx {
            let mut g = beta_g[d]; // intercept
            let mut m = beta_m[d];
            for j in 0..d {
                g += beta_g[j] * x[i * d + j];
                m += beta_m[j] * x[i * d + j];
            }
            g_hat[i] = g;
            m_hat[i] = m;
        }
    }

    // ---- per-sample Robinson partial-out CATE ---------------------------
    let mut cate = vec![0.0_f64; n];
    for i in 0..n {
        let y_res = y[i] - g_hat[i];
        let t_res = t[i] - m_hat[i];
        let denom = t_res * t_res + cfg.ridge_tau;
        cate[i] = (t_res * y_res) / denom;
    }
    let ate = cate.iter().sum::<f64>() / n as f64;

    Ok(RLearnerResult {
        cate,
        ate,
        g_hat,
        m_hat,
    })
}

// =====================================================================
// helpers
// =====================================================================

/// Solve `(X^T X + λ I) β = X^T y` via Gauss-Jordan with partial pivoting.
///
/// `x_mat` is row-major `(n, p)` (already augmented with an intercept column
/// if desired). Returns the coefficient vector of length `p`.
fn ridge_solve(
    x_mat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
) -> CausalResult<Vec<f64>> {
    // XtX (p, p) and Xty (p,)
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for row in 0..n {
        for i in 0..p {
            let xri = x_mat[row * p + i];
            for j in 0..p {
                xtx[i * p + j] += xri * x_mat[row * p + j];
            }
            xty[i] += xri * y[row];
        }
    }
    for i in 0..p {
        xtx[i * p + i] += lambda;
    }
    gauss_jordan_solve(&xtx, &xty, p)
}

/// Solve a `p × p` linear system `A β = b` by Gauss-Jordan with partial pivoting.
fn gauss_jordan_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = p + 1;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p] = b[i];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
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
        for r in 0..p {
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
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        x[i] = m[i * cols + p];
    }
    Ok(x)
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng_uniform(rng: &mut LcgRng) -> f64 {
        (rng.next_f32() as f64) * 2.0 - 1.0
    }

    /// Generate (X, T, Y) with a constant treatment effect of `tau` and
    /// linear outcome relationship `Y = α^T X + tau * T + noise`.
    fn make_constant_effect(
        n: usize,
        d: usize,
        tau: f64,
        binary_t: bool,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0_f64; n * d];
        for v in x.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
        let alpha: Vec<f64> = (0..d).map(|i| 0.5 + 0.1 * i as f64).collect();
        let mut t = vec![0.0_f64; n];
        for (i, item) in t.iter_mut().enumerate() {
            // Treatment depends weakly on X[0] so cross-fitting is meaningful.
            let lin = 0.3 * x[i * d];
            *item = if binary_t {
                if lin + 0.15 * rng_uniform(&mut rng) > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                lin + 0.2 * rng_uniform(&mut rng)
            };
        }
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..d {
                s += alpha[j] * x[i * d + j];
            }
            y[i] = s + tau * t[i] + 0.05 * rng_uniform(&mut rng);
        }
        (x, t, y)
    }

    #[test]
    fn invalid_empty_n_zero() {
        let cfg = RLearnerConfig::default();
        let r = r_learner(&[], 0, 2, &[], &[], &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_empty_d_zero() {
        let cfg = RLearnerConfig::default();
        let r = r_learner(&[1.0, 2.0], 2, 0, &[1.0, 2.0], &[1.0, 0.0], &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_n_less_than_folds() {
        let cfg = RLearnerConfig {
            n_folds: 10,
            ..RLearnerConfig::default()
        };
        let n = 5;
        let d = 2;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 11);
        let r = r_learner(&x, n, d, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::InvalidNumFolds { .. })));
    }

    #[test]
    fn invalid_dim_mismatch_x() {
        let cfg = RLearnerConfig::default();
        let r = r_learner(&[1.0, 2.0, 3.0], 4, 2, &[1.0; 4], &[0.0; 4], &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_dim_mismatch_y() {
        let cfg = RLearnerConfig::default();
        let x = vec![0.0_f64; 100];
        let t = vec![0.0_f64; 50];
        let y = vec![0.0_f64; 49];
        let r = r_learner(&x, 50, 2, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_dim_mismatch_t() {
        let cfg = RLearnerConfig::default();
        let x = vec![0.0_f64; 100];
        let t = vec![0.0_f64; 49];
        let y = vec![0.0_f64; 50];
        let r = r_learner(&x, 50, 2, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_n_folds_one() {
        let cfg = RLearnerConfig {
            n_folds: 1,
            ..RLearnerConfig::default()
        };
        let n = 100;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 23);
        let r = r_learner(&x, n, d, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::InvalidNumFolds { .. })));
    }

    #[test]
    fn invalid_ridge_zero() {
        let cfg = RLearnerConfig {
            ridge_g: 0.0,
            ..RLearnerConfig::default()
        };
        let n = 100;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 41);
        let r = r_learner(&x, n, d, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_ridge_negative_m() {
        let cfg = RLearnerConfig {
            ridge_m: -0.01,
            ..RLearnerConfig::default()
        };
        let n = 100;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 51);
        let r = r_learner(&x, n, d, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_ridge_tau_zero() {
        let cfg = RLearnerConfig {
            ridge_tau: 0.0,
            ..RLearnerConfig::default()
        };
        let n = 100;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 67);
        let r = r_learner(&x, n, d, &y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn recovers_constant_cate_binary_t() {
        // Y = α^T X + 1.5 * T + tiny noise, with binary T.
        let n = 600;
        let d = 3;
        let tau = 1.5_f64;
        let (x, t, y) = make_constant_effect(n, d, tau, true, 314_159);
        let cfg = RLearnerConfig::default();
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        assert_eq!(r.cate.len(), n);
        assert!(
            (r.ate - tau).abs() < 0.30,
            "binary T ATE far from tau: got {} expected ~{}",
            r.ate,
            tau
        );
        // Median CATE should also be close to tau.
        let mut sorted = r.cate.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        let median = sorted[n / 2];
        assert!((median - tau).abs() < 0.5, "median CATE = {median}");
    }

    #[test]
    fn recovers_constant_cate_continuous_t() {
        let n = 600;
        let d = 3;
        let tau = 1.5_f64;
        let (x, t, y) = make_constant_effect(n, d, tau, false, 271_828);
        let cfg = RLearnerConfig::default();
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        assert!(
            (r.ate - tau).abs() < 0.20,
            "continuous T ATE far from tau: got {} expected ~{}",
            r.ate,
            tau
        );
    }

    #[test]
    fn ate_equals_mean_cate() {
        let n = 200;
        let d = 2;
        let (x, t, y) = make_constant_effect(n, d, 0.8, true, 1001);
        let cfg = RLearnerConfig::default();
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        let mean: f64 = r.cate.iter().sum::<f64>() / n as f64;
        assert!((mean - r.ate).abs() < 1e-12);
    }

    #[test]
    fn deterministic() {
        let n = 200;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 7);
        let cfg = RLearnerConfig::default();
        let r1 = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        let r2 = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        assert_eq!(r1.cate, r2.cate);
        assert_eq!(r1.g_hat, r2.g_hat);
        assert_eq!(r1.m_hat, r2.m_hat);
        assert_eq!(r1.ate, r2.ate);
    }

    #[test]
    fn cross_fitting_uses_different_predictions() {
        // With K=2 folds, samples in fold 0 (even indices) are predicted by a
        // model trained on fold 1 (odd indices) — and vice versa.  If we
        // truly cross-fit, g_hat for an even index must NOT equal what
        // training the same ridge on the full dataset would predict at that
        // point. We test this by comparing K=2 cross-fitted predictions to
        // an "in-sample" full-dataset ridge fit.
        let n = 200;
        let d = 3;
        let (x, t, y) = make_constant_effect(n, d, 1.0, true, 99_991);
        let cfg = RLearnerConfig {
            n_folds: 2,
            ..RLearnerConfig::default()
        };
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");

        // Fit the same ridge on ALL the data and compare.
        let dp1 = d + 1;
        let mut x_aug = vec![0.0_f64; n * dp1];
        for i in 0..n {
            for j in 0..d {
                x_aug[i * dp1 + j] = x[i * d + j];
            }
            x_aug[i * dp1 + d] = 1.0;
        }
        let beta_full =
            ridge_solve(&x_aug, &y, n, dp1, cfg.ridge_g).expect("ridge_solve should succeed");
        let mut g_full = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = beta_full[d];
            for j in 0..d {
                s += beta_full[j] * x[i * d + j];
            }
            g_full[i] = s;
        }
        // Average squared difference must be > 0 — cross-fitting really uses
        // different models.
        let diff: f64 = r
            .g_hat
            .iter()
            .zip(g_full.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n as f64;
        assert!(
            diff > 1e-6,
            "cross-fitted g_hat ≈ full-fit g — no cross-fitting?"
        );
    }

    #[test]
    fn n_folds_two_vs_five() {
        let n = 500;
        let d = 3;
        let tau = 1.5_f64;
        let (x, t, y) = make_constant_effect(n, d, tau, true, 4242);
        let cfg2 = RLearnerConfig {
            n_folds: 2,
            ..RLearnerConfig::default()
        };
        let cfg5 = RLearnerConfig {
            n_folds: 5,
            ..RLearnerConfig::default()
        };
        let r2 = r_learner(&x, n, d, &y, &t, &cfg2).expect("r_learner should succeed");
        let r5 = r_learner(&x, n, d, &y, &t, &cfg5).expect("r_learner should succeed");
        // Both should land near tau within a generous tolerance.
        assert!(
            (r2.ate - tau).abs() < 0.3,
            "K=2 ATE = {} (tau = {tau})",
            r2.ate
        );
        assert!(
            (r5.ate - tau).abs() < 0.3,
            "K=5 ATE = {} (tau = {tau})",
            r5.ate
        );
        // The two predictions should be similar in magnitude but not
        // identical — they differ in the fold partition.
        let ate_diff = (r2.ate - r5.ate).abs();
        assert!(ate_diff < 0.5);
    }

    #[test]
    fn result_lengths_match_n() {
        let n = 100;
        let d = 2;
        let (x, t, y) = make_constant_effect(n, d, 0.7, true, 1212);
        let cfg = RLearnerConfig::default();
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        assert_eq!(r.cate.len(), n);
        assert_eq!(r.g_hat.len(), n);
        assert_eq!(r.m_hat.len(), n);
        for v in r.cate.iter().chain(r.g_hat.iter()).chain(r.m_hat.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn large_n_runs() {
        let n = 2000;
        let d = 4;
        let tau = 1.5_f64;
        let (x, t, y) = make_constant_effect(n, d, tau, true, 31415);
        let cfg = RLearnerConfig::default();
        let r = r_learner(&x, n, d, &y, &t, &cfg).expect("r_learner should succeed");
        assert!(
            (r.ate - tau).abs() < 0.10,
            "large-n ATE = {} (expected ~{tau})",
            r.ate
        );
    }

    #[test]
    fn config_defaults_are_sane() {
        let cfg = RLearnerConfig::default();
        assert!(cfg.n_folds >= 2);
        assert!(cfg.ridge_g > 0.0);
        assert!(cfg.ridge_m > 0.0);
        assert!(cfg.ridge_tau > 0.0);
    }
}
