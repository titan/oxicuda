//! Cross-fitted local centering (Robinson 1988 partial-linear model).
//!
//! Reference: Robinson, P. M. (1988). "Root-N-consistent semiparametric regression."
//! *Econometrica*, 56(4), 931-954.
//!
//! # Overview
//!
//! Implements **K-fold cross-fitted** estimation of the nuisance regressions
//! `E[Y|X]` and `E[A|X]` via ridge regression (with intercept), delivering
//! de-biased residuals suitable for downstream causal estimators.
//!
//! For each held-out fold *k*:
//! - Fit ridge on the training folds: `β̂_Y = (X̃^T X̃ + λ_Y I)^{-1} X̃^T Y`
//! - Predict on fold *k*: `Ŷ[k] = X̃[k] β̂_Y`, and analogously for A.
//!
//! The resulting residuals `Ỹ = Y − Ŷ`, `Ã = A − Â` satisfy the Robinson
//! orthogonality condition, enabling consistent estimation of the ATE via
//! `γ̂ = ⟨Ỹ, Ã⟩ / ‖Ã‖²`.
//!
//! The pseudo-CATE (per-sample treatment-effect proxy) is
//! `ψ_i = Ỹ_i · Ã_i / (Ã_i² + λ)` where `λ ≥ 0` stabilises near-zero residuals.

use crate::error::{CausalError, CausalResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`LocalCentering::fit`].
#[derive(Debug, Clone)]
pub struct LocalCenteringConfig {
    /// Number of cross-fitting folds. Must satisfy `n_folds ≥ 2`.
    pub n_folds: usize,
    /// Ridge penalty for the outcome regression `E[Y|X]`. Must be strictly positive.
    pub ridge_y: f64,
    /// Ridge penalty for the treatment regression `E[A|X]`. Must be strictly positive.
    pub ridge_a: f64,
}

impl Default for LocalCenteringConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            ridge_y: 1e-3,
            ridge_a: 1e-3,
        }
    }
}

/// Results from [`LocalCentering::fit`].
#[derive(Debug, Clone)]
pub struct LocalCenteringResult {
    /// `Y − Ê[Y|X]`, length `n`.
    pub y_residuals: Vec<f64>,
    /// `A − Ê[A|X]`, length `n`.
    pub a_residuals: Vec<f64>,
    /// Cross-fitted predictions `Ê[Y|X]`, length `n`.
    pub y_hat: Vec<f64>,
    /// Cross-fitted predictions `Ê[A|X]`, length `n`.
    pub a_hat: Vec<f64>,
    /// R² of the outcome regression, clamped to `[0, 1]`.
    pub y_r2: f64,
    /// R² of the treatment regression, clamped to `[0, 1]`.
    pub a_r2: f64,
}

/// Cross-fitted local centering following Robinson (1988).
pub struct LocalCentering;

impl LocalCentering {
    /// Estimate `E[Y|X]` and `E[A|X]` via K-fold cross-fitted ridge regression
    /// and return the residuals.
    ///
    /// # Parameters
    /// - `x`: row-major `n × p` covariate matrix.
    /// - `y`: outcome vector of length `n`.
    /// - `a`: treatment vector of length `n`.
    /// - `n`, `p`: number of samples and features.
    /// - `cfg`: cross-fitting configuration.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] for validation failures,
    /// [`CausalError::InvalidNumFolds`] if fold requirements are not met,
    /// and [`CausalError::MatrixSingular`] on rank-deficient systems.
    pub fn fit(
        x: &[f64],
        y: &[f64],
        a: &[f64],
        n: usize,
        p: usize,
        cfg: &LocalCenteringConfig,
    ) -> CausalResult<LocalCenteringResult> {
        // ── validation ───────────────────────────────────────────────────────
        if p == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "p must be ≥ 1".into(),
            });
        }
        if cfg.n_folds < 2 {
            return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
        }
        if n < cfg.n_folds {
            return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
        }
        if cfg.ridge_y <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("ridge_y must be > 0, got {}", cfg.ridge_y),
            });
        }
        if cfg.ridge_a <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("ridge_a must be > 0, got {}", cfg.ridge_a),
            });
        }
        if x.len() != n * p {
            return Err(CausalError::InvalidParameter {
                reason: format!("x.len()={} != n*p={}", x.len(), n * p),
            });
        }
        if y.len() != n {
            return Err(CausalError::InvalidParameter {
                reason: format!("y.len()={} != n={n}", y.len()),
            });
        }
        if a.len() != n {
            return Err(CausalError::InvalidParameter {
                reason: format!("a.len()={} != n={n}", a.len()),
            });
        }

        let k = cfg.n_folds;
        let dp1 = p + 1; // augmented feature dimension (with intercept).

        let mut y_hat = vec![0.0_f64; n];
        let mut a_hat = vec![0.0_f64; n];

        // ── K-fold cross-fitting ─────────────────────────────────────────────
        for fold in 0..k {
            let val_idx: Vec<usize> = (0..n).filter(|&i| i % k == fold).collect();
            let train_idx: Vec<usize> = (0..n).filter(|&i| i % k != fold).collect();
            let n_train = train_idx.len();

            // Build training design matrix with intercept (last column = 1).
            let mut x_train = vec![0.0_f64; n_train * dp1];
            let mut y_train = vec![0.0_f64; n_train];
            let mut a_train = vec![0.0_f64; n_train];

            for (r, &orig) in train_idx.iter().enumerate() {
                for j in 0..p {
                    x_train[r * dp1 + j] = x[orig * p + j];
                }
                x_train[r * dp1 + p] = 1.0; // intercept
                y_train[r] = y[orig];
                a_train[r] = a[orig];
            }

            let beta_y = ridge_solve(&x_train, &y_train, n_train, dp1, cfg.ridge_y)?;
            let beta_a = ridge_solve(&x_train, &a_train, n_train, dp1, cfg.ridge_a)?;

            // Predict on validation fold.
            for &i in &val_idx {
                let mut yp = beta_y[p]; // intercept coefficient
                let mut ap = beta_a[p];
                for j in 0..p {
                    yp += beta_y[j] * x[i * p + j];
                    ap += beta_a[j] * x[i * p + j];
                }
                y_hat[i] = yp;
                a_hat[i] = ap;
            }
        }

        // ── residuals ────────────────────────────────────────────────────────
        let y_residuals: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
        let a_residuals: Vec<f64> = (0..n).map(|i| a[i] - a_hat[i]).collect();

        // ── R² ───────────────────────────────────────────────────────────────
        let y_r2 = compute_r2(y, &y_residuals, n);
        let a_r2 = compute_r2(a, &a_residuals, n);

        Ok(LocalCenteringResult {
            y_residuals,
            a_residuals,
            y_hat,
            a_hat,
            y_r2,
            a_r2,
        })
    }

    /// Robinson (1988) ATE estimate: `γ̂ = Σ(Ỹ_i · Ã_i) / Σ(Ã_i²)`.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] if the denominator `Σ Ã²` is
    /// near zero (A residuals carry no variation).
    pub fn robinson_ate(y_res: &[f64], a_res: &[f64]) -> CausalResult<f64> {
        let num: f64 = y_res.iter().zip(a_res.iter()).map(|(yr, ar)| yr * ar).sum();
        let den: f64 = a_res.iter().map(|ar| ar * ar).sum();
        if den < 1e-12 {
            return Err(CausalError::InvalidParameter {
                reason: "near-zero A residuals".into(),
            });
        }
        Ok(num / den)
    }

    /// Per-sample pseudo-CATE (treatment-effect proxy):
    /// `ψ_i = Ỹ_i · Ã_i / (Ã_i² + λ)`.
    ///
    /// `lambda ≥ 0` stabilises division when A residuals are near zero.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if `y_res.len() != a_res.len()`.
    pub fn robinson_pseudo_cate(
        y_res: &[f64],
        a_res: &[f64],
        lambda: f64,
    ) -> CausalResult<Vec<f64>> {
        if y_res.len() != a_res.len() {
            return Err(CausalError::DimensionMismatch {
                expected: y_res.len(),
                got: a_res.len(),
            });
        }
        Ok(y_res
            .iter()
            .zip(a_res.iter())
            .map(|(yr, ar)| yr * ar / (ar * ar + lambda))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve `(X^T X + λ I) β = X^T y` via Cholesky decomposition on the Gram matrix.
/// Falls back to Gauss-Jordan with partial pivoting if Cholesky fails.
fn ridge_solve(
    x_mat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
) -> CausalResult<Vec<f64>> {
    // Build XtX (p×p) and Xty (p).
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
    // Add ridge regularisation.
    for i in 0..p {
        xtx[i * p + i] += lambda;
    }
    // Try Cholesky first; fall back to Gauss-Jordan.
    cholesky_solve(&xtx, &xty, p).or_else(|_| gauss_jordan_solve(&xtx, &xty, p))
}

/// Cholesky L L^T factorisation + forward/backward substitution for SPD systems.
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s: f64 = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s < 1e-18 {
                    return Err(CausalError::MatrixSingular);
                }
                l[i * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    let mut z = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * p + j] * z[j];
        }
        z[i] = s / l[i * p + i];
    }
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for j in (i + 1)..p {
            s -= l[j * p + i] * x[j];
        }
        x[i] = s / l[i * p + i];
    }
    Ok(x)
}

/// Gauss-Jordan elimination with partial pivoting — fallback solver.
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

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute R² = 1 − SS_res / SS_tot, clamped to [0, 1].
fn compute_r2(target: &[f64], residuals: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mean = target.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = target.iter().map(|&v| (v - mean).powi(2)).sum();
    if ss_tot < 1e-15 {
        return 1.0; // constant target → perfect fit (R²=1 by convention).
    }
    let ss_res: f64 = residuals.iter().map(|&r| r * r).sum();
    (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a tiny dataset: Y = tau * A + noise (small), X constant.
    fn make_simple(n: usize, tau: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f64> = vec![1.0_f64; n]; // single constant feature (p=1)
        let a: Vec<f64> = (0..n).map(|_| rng.next_normal() as f64).collect();
        let y: Vec<f64> = a
            .iter()
            .map(|&ai| tau * ai + 0.01 * rng.next_normal() as f64)
            .collect();
        (x, y, a)
    }

    /// Build a dataset: Y = noise, A = noise (uncorrelated).
    fn make_uncorrelated(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f64> = vec![1.0_f64; n];
        let a: Vec<f64> = (0..n).map(|_| rng.next_normal() as f64).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.next_normal() as f64).collect();
        (x, y, a)
    }

    #[test]
    fn pure_signal_ate() {
        let n = 20;
        let (x, y, a) = make_simple(n, 2.0, 1);
        let cfg = LocalCenteringConfig {
            n_folds: 2,
            ridge_y: 1e-3,
            ridge_a: 1e-3,
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        let ate = LocalCentering::robinson_ate(&res.y_residuals, &res.a_residuals)
            .expect("robinson_ate should succeed");
        assert!((ate - 2.0).abs() < 0.3, "ATE={ate} not within 0.3 of 2.0");
    }

    #[test]
    fn residual_reconstruction_y() {
        let n = 30;
        let (x, y, a) = make_simple(n, 1.5, 2);
        let cfg = LocalCenteringConfig::default();
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        for (i, (&yi, (&yh, &yr))) in y
            .iter()
            .zip(res.y_hat.iter().zip(res.y_residuals.iter()))
            .enumerate()
        {
            let reconstructed = yh + yr;
            assert!(
                (reconstructed - yi).abs() < 1e-10,
                "y[{i}]: hat+res={reconstructed} != y={yi}"
            );
        }
    }

    #[test]
    fn residual_reconstruction_a() {
        let n = 30;
        let (x, y, a) = make_simple(n, 1.5, 3);
        let cfg = LocalCenteringConfig::default();
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        for (i, (&ai, (&ah, &ar))) in a
            .iter()
            .zip(res.a_hat.iter().zip(res.a_residuals.iter()))
            .enumerate()
        {
            let reconstructed = ah + ar;
            assert!(
                (reconstructed - ai).abs() < 1e-10,
                "a[{i}]: hat+res={reconstructed} != a={ai}"
            );
        }
    }

    #[test]
    fn r2_in_range() {
        let n = 40;
        let (x, y, a) = make_simple(n, 1.0, 4);
        let cfg = LocalCenteringConfig::default();
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        assert!(
            (0.0..=1.0).contains(&res.y_r2),
            "y_r2={} out of [0,1]",
            res.y_r2
        );
        assert!(
            (0.0..=1.0).contains(&res.a_r2),
            "a_r2={} out of [0,1]",
            res.a_r2
        );
    }

    #[test]
    fn pseudo_cate_length() {
        let n = 20;
        let y_res: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let a_res: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let psi = LocalCentering::robinson_pseudo_cate(&y_res, &a_res, 1e-6)
            .expect("robinson_pseudo_cate should succeed");
        assert_eq!(psi.len(), n);
    }

    #[test]
    fn lambda_zero_pure() {
        // a_res = ±1 → denominator = a_res² + 0 = 1 → psi[i] = y_res[i] * a_res[i].
        let y_res = vec![0.5_f64, -0.3, 0.8];
        let a_res = vec![1.0_f64, -1.0, 1.0];
        let psi = LocalCentering::robinson_pseudo_cate(&y_res, &a_res, 0.0)
            .expect("robinson_pseudo_cate should succeed");
        for i in 0..3 {
            let expected = y_res[i] * a_res[i]; // / 1.0
            assert!(
                (psi[i] - expected).abs() < 1e-12,
                "psi[{i}]={} expected={expected}",
                psi[i]
            );
        }
    }

    #[test]
    fn n_folds_2() {
        let n = 10;
        let (x, y, a) = make_simple(n, 1.0, 5);
        let cfg = LocalCenteringConfig {
            n_folds: 2,
            ..LocalCenteringConfig::default()
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg);
        assert!(res.is_ok());
    }

    #[test]
    fn err_n_lt_n_folds() {
        let n = 3;
        let (x, y, a) = make_simple(n, 1.0, 6);
        let cfg = LocalCenteringConfig {
            n_folds: 5,
            ..LocalCenteringConfig::default()
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg);
        assert!(matches!(res, Err(CausalError::InvalidNumFolds { .. })));
    }

    #[test]
    fn err_n_folds_lt_2() {
        let n = 10;
        let (x, y, a) = make_simple(n, 1.0, 7);
        let cfg = LocalCenteringConfig {
            n_folds: 1,
            ..LocalCenteringConfig::default()
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg);
        assert!(matches!(res, Err(CausalError::InvalidNumFolds { .. })));
    }

    #[test]
    fn err_ridge_zero() {
        let n = 10;
        let (x, y, a) = make_simple(n, 1.0, 8);
        let cfg = LocalCenteringConfig {
            ridge_y: 0.0,
            ..LocalCenteringConfig::default()
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg);
        assert!(matches!(res, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn err_dim_mismatch() {
        let n = 10;
        let x = vec![1.0_f64; n];
        let y = vec![0.0_f64; n - 1]; // wrong length
        let a = vec![0.0_f64; n];
        let cfg = LocalCenteringConfig::default();
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg);
        assert!(matches!(res, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn err_p_zero() {
        let n = 10;
        let x: Vec<f64> = vec![];
        let y = vec![0.0_f64; n];
        let a = vec![0.0_f64; n];
        let cfg = LocalCenteringConfig::default();
        let res = LocalCentering::fit(&x, &y, &a, n, 0, &cfg);
        assert!(matches!(res, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn robinson_ate_zero() {
        // Y ⊥ A (independent noise) → ATE ≈ 0 with large tolerance ± 0.3.
        let n = 60;
        let (x, y, a) = make_uncorrelated(n, 9);
        let cfg = LocalCenteringConfig {
            n_folds: 2,
            ..LocalCenteringConfig::default()
        };
        let res = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        let ate = LocalCentering::robinson_ate(&res.y_residuals, &res.a_residuals)
            .expect("robinson_ate should succeed");
        assert!(
            ate.abs() < 0.5,
            "uncorrelated ATE={ate} should be near zero"
        );
    }

    #[test]
    fn ate_err_zero_a_res() {
        // All a_res = 0 → robinson_ate must return Err.
        let y_res = vec![1.0_f64, 2.0, 3.0];
        let a_res = vec![0.0_f64; 3];
        let r = LocalCentering::robinson_ate(&y_res, &a_res);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn deterministic() {
        let n = 20;
        let (x, y, a) = make_simple(n, 1.0, 10);
        let cfg = LocalCenteringConfig::default();
        let r1 = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        let r2 = LocalCentering::fit(&x, &y, &a, n, 1, &cfg).expect("fit should succeed");
        assert_eq!(r1.y_hat, r2.y_hat);
        assert_eq!(r1.a_hat, r2.a_hat);
        assert_eq!(r1.y_residuals, r2.y_residuals);
        assert_eq!(r1.a_residuals, r2.a_residuals);
    }

    #[test]
    fn default_config() {
        let cfg = LocalCenteringConfig::default();
        assert_eq!(cfg.n_folds, 5);
        assert!((cfg.ridge_y - 1e-3).abs() < 1e-12);
        assert!((cfg.ridge_a - 1e-3).abs() < 1e-12);
    }
}
