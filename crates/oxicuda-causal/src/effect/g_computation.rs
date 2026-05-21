//! G-computation — Parametric standardisation for causal effect estimation.
//!
//! Reference: Robins, J. M. (1986). "A new approach to causal inference in
//! mortality studies with a sustained exposure period — application to control
//! of the healthy worker survivor effect." *Mathematical Modelling*, 7(9-12),
//! 1393-1512.  See also Hernán, M. A. & Robins, J. M. (2020). *Causal
//! Inference: What If*. Boca Raton: Chapman & Hall/CRC, Chapter 13
//! ("Standardization and the parametric g-formula").
//!
//! # Algorithm
//!
//! G-computation (a.k.a. *parametric g-formula* or *standardisation*) proceeds
//! in two steps:
//!
//! 1. **Outcome model**.  A parametric regression `μ̂(T, X) = E[Y | T, X]` is
//!    fitted on the observed data using a *full-interaction* linear model so
//!    that the treatment effect is allowed to vary with the covariates:
//!
//!    ```text
//!      Z_i = [ 1,  T_i,  X_i,  T_i · X_i ]  ∈ R^{2d+2}
//!      μ̂(T_i, X_i) = β̂^T Z_i
//!    ```
//!
//!    where `β̂` is obtained by *ridge* OLS, `β̂ = (Z^T Z + λ I)^{-1} Z^T y`.
//!    The ridge penalty ensures a unique solution even when columns of `Z`
//!    are exactly collinear (e.g. all treated, all untreated, or near-zero
//!    covariate variance).
//!
//! 2. **Counterfactual standardisation**.  For each unit `i` we form two
//!    counterfactual predictions by *toggling* the treatment indicator while
//!    keeping `X_i` fixed,
//!
//!    ```text
//!      μ̂_1(x_i) = β̂^T [ 1, 1, x_i,  x_i ]
//!      μ̂_0(x_i) = β̂^T [ 1, 0, x_i,  0   ]
//!    ```
//!
//!    The **average treatment effect** is then
//!
//!    ```text
//!      ATE = (1/n) Σ_i ( μ̂_1(x_i) - μ̂_0(x_i) )
//!    ```
//!
//!    and the **average treatment effect on the treated** is
//!
//!    ```text
//!      ATT = (1/n_1) Σ_{i : T_i=1} ( Y_i - μ̂_0(x_i) ),     n_1 = #{i : T_i=1}.
//!    ```
//!
//! Crucially, **for ATT we use the observed `Y_i` for the treated**, not the
//! model prediction `μ̂_1(x_i)` — this is Robins' standard recipe and gives the
//! correct estimator when the outcome model is misspecified for the treated
//! arm but the unconfoundedness assumption (`Y(0) ⫫ T | X`) holds.
//!
//! Assumptions: (i) **conditional exchangeability** `Y(t) ⫫ T | X`,
//! (ii) **positivity** `0 < P(T=1 | X=x) < 1`, (iii) **consistency**
//! `Y = Y(T)`, and (iv) **correct specification** of the outcome model.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`g_computation`].
#[derive(Debug, Clone)]
pub struct GComputationConfig {
    /// Ridge penalty for the outcome model.  Must be strictly positive.
    /// A typical value is `1e-3`; increase if the design matrix is highly
    /// collinear or if `n` is small relative to `2d + 2`.
    pub ridge: f64,
}

impl Default for GComputationConfig {
    fn default() -> Self {
        Self { ridge: 1e-3 }
    }
}

/// Output of [`g_computation`].
#[derive(Debug, Clone)]
pub struct GComputationResult {
    /// Average treatment effect:
    /// `(1/n) Σ_i [μ̂_1(x_i) − μ̂_0(x_i)]`.
    pub ate: f64,
    /// Average treatment effect on the treated:
    /// `(1/n_1) Σ_{T_i=1} [Y_i − μ̂_0(x_i)]`.
    /// Returns `0.0` when there are no treated units (the value is well
    /// defined as a vacuous sum; callers can detect this by counting `T_i = 1`
    /// upstream).
    pub att: f64,
    /// Counterfactual outcome under treatment for each sample (length `n`).
    pub mu_1: Vec<f64>,
    /// Counterfactual outcome under no-treatment for each sample (length `n`).
    pub mu_0: Vec<f64>,
    /// Fitted coefficients for the full-interaction outcome model in the
    /// order `[ intercept, β_T, β_X(1..d), β_{T·X}(1..d) ]`.  Length is
    /// `2 · d + 2`.  Exposed for diagnostics — e.g. inspecting the
    /// treatment-by-covariate interaction terms `β_{T·X}` reveals effect
    /// heterogeneity along each covariate direction.
    pub coefficients: Vec<f64>,
}

/// Fit a parametric g-formula and return the standardised ATE and ATT.
///
/// # Parameters
/// - `x`: row-major `n × d` covariate matrix (length `n · d`).
/// - `n`: number of samples.  Must be `> 0`.
/// - `d`: number of covariates.  Must be `> 0`.
/// - `t`: binary treatment indicator, length `n`, each entry in `{0.0, 1.0}`.
/// - `y`: outcome vector, length `n`.
/// - `cfg`: see [`GComputationConfig`].
///
/// # Errors
/// - [`CausalError::EmptyInput`] if `n == 0`, `d == 0`, or any slice is empty.
/// - [`CausalError::DimensionMismatch`] if `x.len() != n · d`, `t.len() != n`,
///   or `y.len() != n`.
/// - [`CausalError::IncompatibleData`] if `cfg.ridge ≤ 0` or if any
///   `t[i] ∉ {0.0, 1.0}`.
/// - [`CausalError::MatrixSingular`] if the ridge-augmented normal equations
///   are rank-deficient (only possible for pathological inputs).
pub fn g_computation(
    x: &[f64],
    n: usize,
    d: usize,
    t: &[f64],
    y: &[f64],
    cfg: &GComputationConfig,
) -> CausalResult<GComputationResult> {
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
    if t.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len(),
        });
    }
    if y.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if cfg.ridge <= 0.0 {
        return Err(CausalError::IncompatibleData);
    }
    for &ti in t {
        if !(ti == 0.0 || ti == 1.0) {
            return Err(CausalError::IncompatibleData);
        }
    }

    // ---- design matrix Z = [1, T, X, T·X] -------------------------------
    let p = 2 * d + 2;
    let mut z = vec![0.0_f64; n * p];
    for i in 0..n {
        let row = i * p;
        z[row] = 1.0; // intercept
        z[row + 1] = t[i]; // treatment
        for j in 0..d {
            z[row + 2 + j] = x[i * d + j]; // X
            z[row + 2 + d + j] = t[i] * x[i * d + j]; // T · X
        }
    }

    // ---- ridge solve  β = (Z^T Z + λ I)^{-1} Z^T y ----------------------
    let beta = ridge_solve(&z, y, n, p, cfg.ridge)?;

    // ---- counterfactual predictions -------------------------------------
    let mut mu_1 = vec![0.0_f64; n];
    let mut mu_0 = vec![0.0_f64; n];
    for i in 0..n {
        // μ̂_1(x) = β_0 + β_T·1 + Σ_j β_{X,j} x_j + Σ_j β_{TX,j} · 1 · x_j
        let mut m1 = beta[0] + beta[1];
        let mut m0 = beta[0];
        for j in 0..d {
            let xij = x[i * d + j];
            m1 += beta[2 + j] * xij + beta[2 + d + j] * xij;
            m0 += beta[2 + j] * xij;
        }
        mu_1[i] = m1;
        mu_0[i] = m0;
    }

    // ---- ATE and ATT ----------------------------------------------------
    let ate: f64 = mu_1
        .iter()
        .zip(mu_0.iter())
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / n as f64;

    let mut att_sum = 0.0_f64;
    let mut n1 = 0_usize;
    for i in 0..n {
        if t[i] == 1.0 {
            att_sum += y[i] - mu_0[i];
            n1 += 1;
        }
    }
    let att = if n1 == 0 { 0.0 } else { att_sum / n1 as f64 };

    Ok(GComputationResult {
        ate,
        att,
        mu_1,
        mu_0,
        coefficients: beta,
    })
}

// =====================================================================
// helpers — ridge solve via Gauss-Jordan with partial pivoting
// =====================================================================

/// Solve `(Z^T Z + λ I) β = Z^T y` for `β`.  `z` is row-major `(n, p)`.
fn ridge_solve(z: &[f64], y: &[f64], n: usize, p: usize, lambda: f64) -> CausalResult<Vec<f64>> {
    let mut zty = vec![0.0_f64; p];
    let mut ztz = vec![0.0_f64; p * p];
    for row in 0..n {
        for i in 0..p {
            let zri = z[row * p + i];
            for j in 0..p {
                ztz[i * p + j] += zri * z[row * p + j];
            }
            zty[i] += zri * y[row];
        }
    }
    for i in 0..p {
        ztz[i * p + i] += lambda;
    }
    gauss_jordan_solve(&ztz, &zty, p)
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

    /// Generate a dataset where Y = α₀ + α₁^T X + τ · T + γ^T (T·X) + noise.
    /// `tau` is the *baseline* treatment effect at X = 0, and `gamma` controls
    /// effect heterogeneity along each covariate direction.
    fn make_linear_dataset(
        n: usize,
        d: usize,
        tau: f64,
        gamma_scale: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0_f64; n * d];
        for v in x.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
        let alpha: Vec<f64> = (0..d).map(|i| 0.5 + 0.1 * i as f64).collect();
        let gamma: Vec<f64> = (0..d)
            .map(|i| gamma_scale * (0.3 - 0.05 * i as f64))
            .collect();
        // Treatment assignment depends weakly on X[0].
        let mut t = vec![0.0_f64; n];
        for i in 0..n {
            let lin = 0.5 * x[i * d];
            t[i] = if lin + 0.15 * rng_uniform(&mut rng) > 0.0 {
                1.0
            } else {
                0.0
            };
        }
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.7_f64; // intercept α_0
            for j in 0..d {
                s += alpha[j] * x[i * d + j];
            }
            // treatment effect with heterogeneity
            let mut hetero = tau;
            for j in 0..d {
                hetero += gamma[j] * x[i * d + j];
            }
            y[i] = s + hetero * t[i] + 0.05 * rng_uniform(&mut rng);
        }
        (x, t, y)
    }

    // -------------------- input validation tests ---------------------------

    #[test]
    fn invalid_empty_n_zero() {
        let cfg = GComputationConfig::default();
        let r = g_computation(&[], 0, 2, &[], &[], &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_empty_d_zero() {
        let cfg = GComputationConfig::default();
        let r = g_computation(&[1.0, 2.0], 2, 0, &[1.0, 0.0], &[1.0, 2.0], &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_dim_mismatch_x() {
        let cfg = GComputationConfig::default();
        // x.len() = 3 but n*d = 4
        let r = g_computation(&[1.0, 2.0, 3.0], 4, 1, &[0.0; 4], &[1.0; 4], &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_dim_mismatch_t() {
        let cfg = GComputationConfig::default();
        let x = vec![0.0_f64; 100];
        let t = vec![0.0_f64; 49]; // wrong length
        let y = vec![0.0_f64; 50];
        let r = g_computation(&x, 50, 2, &t, &y, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_dim_mismatch_y() {
        let cfg = GComputationConfig::default();
        let x = vec![0.0_f64; 100];
        let t = vec![0.0_f64; 50];
        let y = vec![0.0_f64; 51]; // wrong length
        let r = g_computation(&x, 50, 2, &t, &y, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_ridge_zero() {
        let cfg = GComputationConfig { ridge: 0.0 };
        let (x, t, y) = make_linear_dataset(50, 2, 3.0, 0.0, 17);
        let r = g_computation(&x, 50, 2, &t, &y, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_ridge_negative() {
        let cfg = GComputationConfig { ridge: -0.01 };
        let (x, t, y) = make_linear_dataset(50, 2, 3.0, 0.0, 18);
        let r = g_computation(&x, 50, 2, &t, &y, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_non_binary_treatment() {
        let cfg = GComputationConfig::default();
        let (x, mut t, y) = make_linear_dataset(50, 2, 1.0, 0.0, 19);
        t[3] = 0.5; // not binary
        let r = g_computation(&x, 50, 2, &t, &y, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    // -------------------- correctness tests --------------------------------

    /// "Known constant ATE": Y = 2 + 3·T + 0.5·X + tiny noise → ATE ≈ 3.
    #[test]
    fn recovers_constant_ate_three() {
        let n = 800;
        let d = 1;
        // Build the dataset manually so the relationship is *exactly*
        // Y = 2 + 3·T + 0.5·X + tiny noise.
        let mut rng = LcgRng::new(2_718_281);
        let mut x = vec![0.0_f64; n * d];
        let mut t = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x[i] = rng_uniform(&mut rng);
            t[i] = if (i % 2) == 0 { 1.0 } else { 0.0 };
            y[i] = 2.0 + 3.0 * t[i] + 0.5 * x[i] + 0.01 * rng_uniform(&mut rng);
        }
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert!(
            (res.ate - 3.0).abs() < 0.05,
            "expected ATE ≈ 3.0, got {}",
            res.ate
        );
        // ATT should also be ≈ 3.0 (constant effect, no heterogeneity).
        assert!(
            (res.att - 3.0).abs() < 0.10,
            "expected ATT ≈ 3.0, got {}",
            res.att
        );
        // mu_1 - mu_0 should be ≈ 3.0 for every sample (constant effect).
        for i in 0..n {
            assert!(
                (res.mu_1[i] - res.mu_0[i] - 3.0).abs() < 0.10,
                "mu_1 - mu_0 = {} at i={}",
                res.mu_1[i] - res.mu_0[i],
                i
            );
        }
    }

    #[test]
    fn null_ate_when_y_independent_of_t() {
        // Y = 1 + 0.5·X, with no treatment effect at all.
        let n = 400;
        let d = 2;
        let mut rng = LcgRng::new(31_415);
        let mut x = vec![0.0_f64; n * d];
        let mut t = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..d {
                x[i * d + j] = rng_uniform(&mut rng);
            }
            t[i] = if rng.next_f32() > 0.5 { 1.0 } else { 0.0 };
            y[i] = 1.0 + 0.5 * x[i * d] - 0.3 * x[i * d + 1] + 0.02 * rng_uniform(&mut rng);
        }
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert!(
            res.ate.abs() < 0.10,
            "expected null ATE near 0, got {}",
            res.ate
        );
    }

    #[test]
    fn att_differs_from_ate_with_heterogeneity() {
        // Heterogeneous effect:  τ(x) = 1.0 + 2.0 · x[0]
        // Treatment is assigned more often when x[0] > 0 → ATT > ATE.
        let n = 1000;
        let d = 1;
        let mut rng = LcgRng::new(987_654);
        let mut x = vec![0.0_f64; n * d];
        let mut t = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x[i] = rng_uniform(&mut rng);
            // Strong treatment-X correlation.
            t[i] = if x[i] + 0.05 * rng_uniform(&mut rng) > 0.0 {
                1.0
            } else {
                0.0
            };
            let tau = 1.0 + 2.0 * x[i];
            y[i] = 0.5 + 0.3 * x[i] + tau * t[i] + 0.02 * rng_uniform(&mut rng);
        }
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        // ATE = E[1 + 2X] = 1.0 (X is U(-1,1)).
        // ATT averages τ over treated (mostly x > 0) ≈ 1 + 2·E[X|X>0] = 1 + 1 = 2.
        assert!(
            (res.ate - 1.0).abs() < 0.20,
            "ATE = {} expected ~1.0",
            res.ate
        );
        assert!(
            res.att > res.ate + 0.3,
            "expected ATT > ATE (heterogeneity), got ATE={} ATT={}",
            res.ate,
            res.att
        );
    }

    #[test]
    fn deterministic() {
        let (x, t, y) = make_linear_dataset(200, 3, 1.5, 0.3, 7);
        let cfg = GComputationConfig::default();
        let r1 = g_computation(&x, 200, 3, &t, &y, &cfg).unwrap();
        let r2 = g_computation(&x, 200, 3, &t, &y, &cfg).unwrap();
        assert_eq!(r1.ate, r2.ate);
        assert_eq!(r1.att, r2.att);
        assert_eq!(r1.mu_1, r2.mu_1);
        assert_eq!(r1.mu_0, r2.mu_0);
        assert_eq!(r1.coefficients, r2.coefficients);
    }

    #[test]
    fn large_n_runs() {
        let n = 5000;
        let d = 4;
        let (x, t, y) = make_linear_dataset(n, d, 1.5, 0.0, 4242);
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert!(
            (res.ate - 1.5).abs() < 0.10,
            "large-n ATE = {} (expected ~1.5)",
            res.ate
        );
        assert!(res.mu_1.iter().all(|v| v.is_finite()));
        assert!(res.mu_0.iter().all(|v| v.is_finite()));
        assert_eq!(res.coefficients.len(), 2 * d + 2);
    }

    #[test]
    fn d_equals_one() {
        let (x, t, y) = make_linear_dataset(300, 1, 2.0, 0.0, 1010);
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, 300, 1, &t, &y, &cfg).unwrap();
        assert_eq!(res.mu_1.len(), 300);
        assert_eq!(res.mu_0.len(), 300);
        assert_eq!(res.coefficients.len(), 4); // 2*1 + 2
        assert!((res.ate - 2.0).abs() < 0.15);
    }

    #[test]
    fn d_equals_five() {
        let n = 600;
        let d = 5;
        let (x, t, y) = make_linear_dataset(n, d, 1.0, 0.0, 5050);
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert_eq!(res.coefficients.len(), 12); // 2*5 + 2
        assert!(
            (res.ate - 1.0).abs() < 0.15,
            "d=5 ATE = {} (expected ~1.0)",
            res.ate
        );
    }

    #[test]
    fn result_field_lengths_match_n() {
        let n = 150;
        let d = 3;
        let (x, t, y) = make_linear_dataset(n, d, 0.8, 0.0, 1234);
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert_eq!(res.mu_1.len(), n);
        assert_eq!(res.mu_0.len(), n);
        assert_eq!(res.coefficients.len(), 2 * d + 2);
        for v in res
            .mu_1
            .iter()
            .chain(res.mu_0.iter())
            .chain(res.coefficients.iter())
        {
            assert!(v.is_finite(), "non-finite value in result");
        }
    }

    #[test]
    fn coefficients_recover_known_effect() {
        // Y = 1 + 5·T + 0.5·X (constant-effect, single covariate).  The
        // intercept should be ≈ 1, β_T ≈ 5, β_X ≈ 0.5, β_{T·X} ≈ 0.
        let n = 1000;
        let mut rng = LcgRng::new(112_233);
        let mut x = vec![0.0_f64; n];
        let mut t = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x[i] = rng_uniform(&mut rng);
            t[i] = if (i % 3) == 0 { 1.0 } else { 0.0 };
            y[i] = 1.0 + 5.0 * t[i] + 0.5 * x[i] + 0.005 * rng_uniform(&mut rng);
        }
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, 1, &t, &y, &cfg).unwrap();
        let beta = &res.coefficients;
        assert!((beta[0] - 1.0).abs() < 0.05, "β_0 = {}", beta[0]);
        assert!((beta[1] - 5.0).abs() < 0.05, "β_T = {}", beta[1]);
        assert!((beta[2] - 0.5).abs() < 0.10, "β_X = {}", beta[2]);
        assert!(beta[3].abs() < 0.10, "β_{{T·X}} = {}", beta[3]);
    }

    #[test]
    fn all_treated_att_equals_observed_minus_mu0() {
        // When every unit is treated, ATT = mean(Y) − mean(μ̂_0).  The
        // counterfactual estimate must still be finite (ridge handles
        // collinearity of [1, T=1] columns).
        let n = 200;
        let d = 2;
        let mut rng = LcgRng::new(77_777);
        let mut x = vec![0.0_f64; n * d];
        for v in x.iter_mut() {
            *v = rng_uniform(&mut rng);
        }
        let t = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            y[i] = 1.0 + 0.4 * x[i * d] + 0.05 * rng_uniform(&mut rng);
        }
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert!(res.att.is_finite());
        // Manual check.
        let expected: f64 = (0..n).map(|i| y[i] - res.mu_0[i]).sum::<f64>() / n as f64;
        assert!((res.att - expected).abs() < 1e-9);
    }

    #[test]
    fn no_treated_units_att_is_zero() {
        // With no treated units, ATT is reported as 0.0 (vacuous mean).
        let n = 80;
        let d = 2;
        let (x, _, y) = make_linear_dataset(n, d, 1.0, 0.0, 333);
        let t = vec![0.0_f64; n];
        let cfg = GComputationConfig::default();
        let res = g_computation(&x, n, d, &t, &y, &cfg).unwrap();
        assert_eq!(res.att, 0.0);
        assert!(res.ate.is_finite());
    }

    #[test]
    fn config_default_is_sane() {
        let cfg = GComputationConfig::default();
        assert!(cfg.ridge > 0.0);
    }
}
