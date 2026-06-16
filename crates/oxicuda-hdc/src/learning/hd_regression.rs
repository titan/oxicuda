//! Hyperdimensional regression via ridge least-squares on encoded hypervectors
//! (Hersche, Karunaratne, Benini & Rahimi, NeurIPS 2023 — "Regression in HD space").
//!
//! Given a training set of encoded hypervectors `{x_i ∈ {±1}^D}` with scalar targets
//! `{y_i ∈ ℝ}`, HD regression fits a real-valued *readout* hypervector `w ∈ ℝ^D` such that the
//! prediction `ŷ = ⟨w, x⟩` minimises the ridge-regularised squared error
//!
//! ```text
//! L(w) = Σ_i (⟨w, x_i⟩ − y_i)² + λ‖w‖² .
//! ```
//!
//! Targets are mean-centred first, so the constant offset is captured by an explicit bias term
//! and the readout only has to model variation about the mean.
//!
//! Rather than forming the `D × D` normal-equation matrix (prohibitive for large `D`), this
//! implementation uses the **dual (kernel) ridge solution**, which only requires solving an
//! `n × n` linear system in the number of training samples `n`:
//!
//! ```text
//! K       = X Xᵀ            (n × n Gram matrix of HD samples, K[i][j] = ⟨x_i, x_j⟩)
//! α       = (K + λ I)⁻¹ ỹ   (dual coefficients; ỹ = mean-centred targets)
//! w       = Xᵀ α            (primal readout, never materialised here)
//! ŷ(x)    = bias + ⟨w, x⟩ = bias + Σ_i α_i ⟨x_i, x⟩ .
//! ```
//!
//! The `n × n` system is solved by Gaussian elimination with partial pivoting (self-contained,
//! no external linear-algebra dependency). For `n ≤ D` this is exact ridge regression. Because
//! `‖x_i‖² = D`, the natural scale for `λ` is a small multiple of `D`.
//!
//! All hypervectors are the crate-standard binary `Vec<i8>` in `{−1, +1}`.

use crate::error::{HdcError, HdcResult};
use crate::vector::binary::{binary_dot, validate_binary};

/// Configuration for an [`HdRegressor`] fit.
#[derive(Debug, Clone)]
pub struct HdRegressionConfig {
    /// Hypervector dimension `D` (must be ≥ 1).
    pub dim: usize,
    /// Ridge regularisation strength `λ` (≥ 0). Larger values shrink the readout.
    pub ridge_lambda: f64,
}

impl Default for HdRegressionConfig {
    fn default() -> Self {
        Self {
            dim: 10_000,
            ridge_lambda: 1.0,
        }
    }
}

/// Hyperdimensional ridge regressor with a dual (kernel) closed-form solution.
pub struct HdRegressor {
    cfg: HdRegressionConfig,
    /// Stored training hypervectors (each length `dim`).
    train_hvs: Vec<Vec<i8>>,
    /// Dual coefficients `α` (one per training sample), set by [`fit`].
    ///
    /// [`fit`]: HdRegressor::fit
    alpha: Vec<f64>,
    /// Bias term (mean of training targets, subtracted before the solve and added back).
    bias: f64,
    /// Whether [`fit`] has been called successfully.
    fitted: bool,
}

impl HdRegressor {
    /// Create a new, unfitted regressor.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `cfg.dim == 0`.
    /// - [`HdcError::InvalidProbability`] (reused) if `cfg.ridge_lambda < 0`.
    pub fn new(cfg: HdRegressionConfig) -> HdcResult<Self> {
        if cfg.dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if cfg.ridge_lambda < 0.0 {
            return Err(HdcError::InvalidProbability(cfg.ridge_lambda));
        }
        Ok(Self {
            cfg,
            train_hvs: Vec::new(),
            alpha: Vec::new(),
            bias: 0.0,
            fitted: false,
        })
    }

    /// Fit the regressor on encoded training hypervectors and scalar targets.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if there are no samples.
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != targets.len()` or any HV has the
    ///   wrong dimension.
    /// - [`HdcError::InvalidBinaryValue`] if any HV component is not `±1`.
    /// - [`HdcError::DivisionByZero`] if the regularised Gram system is singular.
    pub fn fit(&mut self, hvs: &[Vec<i8>], targets: &[f64]) -> HdcResult<()> {
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if hvs.len() != targets.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: targets.len(),
            });
        }
        for hv in hvs {
            if hv.len() != self.cfg.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.cfg.dim,
                    got: hv.len(),
                });
            }
            validate_binary(hv)?;
        }

        let n = hvs.len();
        // Centre the targets so the model captures the constant offset via the bias term.
        self.bias = targets.iter().sum::<f64>() / n as f64;
        let centred: Vec<f64> = targets.iter().map(|&y| y - self.bias).collect();

        // Build the n × n Gram matrix K[i][j] = ⟨x_i, x_j⟩ (symmetric).
        let mut gram = vec![0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let dot = binary_dot(&hvs[i], &hvs[j])? as f64;
                gram[i * n + j] = dot;
                gram[j * n + i] = dot;
            }
        }

        // Right-hand side: the mean-centred targets (prediction scale is set directly by w).
        let mut rhs: Vec<f64> = centred;

        // Regularise the diagonal: (K + λ I).
        for d in 0..n {
            gram[d * n + d] += self.cfg.ridge_lambda;
        }

        // Solve (K + λ I) α = rhs.
        solve_linear_system(&mut gram, &mut rhs, n)?;

        self.alpha = rhs;
        self.train_hvs = hvs.to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Predict the scalar target for a query hypervector.
    ///
    /// `ŷ(x) = bias + Σ_i α_i ⟨x_i, x⟩`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::PrototypeNotBuilt`] (reused as "not fitted") if [`fit`] was not called.
    /// - [`HdcError::DimensionMismatch`] if `query` has the wrong dimension.
    ///
    /// [`fit`]: HdRegressor::fit
    pub fn predict(&self, query: &[i8]) -> HdcResult<f64> {
        if !self.fitted {
            return Err(HdcError::PrototypeNotBuilt);
        }
        if query.len() != self.cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        let mut acc = 0f64;
        for (xi, &ai) in self.train_hvs.iter().zip(self.alpha.iter()) {
            let dot = binary_dot(xi, query)? as f64;
            acc += ai * dot;
        }
        Ok(self.bias + acc)
    }

    /// Predict targets for a batch of query hypervectors.
    ///
    /// # Errors
    ///
    /// Same as [`predict`](HdRegressor::predict).
    pub fn predict_batch(&self, queries: &[Vec<i8>]) -> HdcResult<Vec<f64>> {
        queries.iter().map(|q| self.predict(q)).collect()
    }

    /// Coefficient of determination `R²` of the model on a labelled evaluation set.
    ///
    /// `R² = 1 − SS_res / SS_tot`. Returns `0` when the target variance is zero.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if there are no samples.
    /// - Propagates errors from [`predict`](HdRegressor::predict).
    pub fn r2_score(&self, hvs: &[Vec<i8>], targets: &[f64]) -> HdcResult<f64> {
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if hvs.len() != targets.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: targets.len(),
            });
        }
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut ss_res = 0f64;
        let mut ss_tot = 0f64;
        for (hv, &y) in hvs.iter().zip(targets.iter()) {
            let pred = self.predict(hv)?;
            ss_res += (y - pred).powi(2);
            ss_tot += (y - mean).powi(2);
        }
        if ss_tot < f64::EPSILON {
            return Ok(0.0);
        }
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Number of stored training samples.
    pub fn n_samples(&self) -> usize {
        self.train_hvs.len()
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Solve the dense linear system `A x = b` in place via Gaussian elimination with partial
/// pivoting. `a` is row-major `n × n`; on return `b` holds the solution `x`.
///
/// # Errors
///
/// - [`HdcError::DivisionByZero`] if the matrix is (numerically) singular.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> HdcResult<()> {
    for col in 0..n {
        // Partial pivot: find the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut pivot_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-12 {
            return Err(HdcError::DivisionByZero);
        }
        // Swap pivot row into place.
        if pivot_row != col {
            for c in 0..n {
                a.swap(col * n + c, pivot_row * n + c);
            }
            b.swap(col, pivot_row);
        }
        // Eliminate below the pivot.
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            if factor != 0.0 {
                for c in col..n {
                    a[row * n + c] -= factor * a[col * n + c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    for col in (0..n).rev() {
        let mut sum = b[col];
        for c in (col + 1)..n {
            sum -= a[col * n + c] * b[c];
        }
        b[col] = sum / a[col * n + col];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    fn make_cfg(dim: usize, lambda: f64) -> HdRegressionConfig {
        HdRegressionConfig {
            dim,
            ridge_lambda: lambda,
        }
    }

    #[test]
    fn config_default_is_valid() {
        let cfg = HdRegressionConfig::default();
        assert_eq!(cfg.dim, 10_000);
        assert!(cfg.ridge_lambda >= 0.0);
    }

    #[test]
    fn new_rejects_zero_dim() {
        assert!(matches!(
            HdRegressor::new(make_cfg(0, 1.0)),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn new_rejects_negative_lambda() {
        assert!(matches!(
            HdRegressor::new(make_cfg(64, -1.0)),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn predict_before_fit_errors() {
        let reg = HdRegressor::new(make_cfg(64, 1.0)).expect("new");
        let mut r = LcgRng::new(1);
        let q = random_binary(64, &mut r).expect("hv");
        assert!(matches!(reg.predict(&q), Err(HdcError::PrototypeNotBuilt)));
    }

    #[test]
    fn fit_empty_errors() {
        let mut reg = HdRegressor::new(make_cfg(64, 1.0)).expect("new");
        assert!(matches!(reg.fit(&[], &[]), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn fit_length_mismatch_errors() {
        let mut reg = HdRegressor::new(make_cfg(64, 1.0)).expect("new");
        let mut r = LcgRng::new(2);
        let hvs = vec![random_binary(64, &mut r).expect("hv")];
        let targets = vec![1.0, 2.0];
        assert!(matches!(
            reg.fit(&hvs, &targets),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_wrong_dim_errors() {
        let mut reg = HdRegressor::new(make_cfg(64, 1.0)).expect("new");
        let mut r = LcgRng::new(3);
        let hvs = vec![random_binary(32, &mut r).expect("hv")];
        let targets = vec![1.0];
        assert!(matches!(
            reg.fit(&hvs, &targets),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_recovers_training_targets_low_ridge() {
        // With small ridge and distinct random HVs, the model should fit training data well.
        let mut r = LcgRng::new(42);
        let dim = 2000;
        let n = 8;
        let hvs: Vec<Vec<i8>> = (0..n)
            .map(|_| random_binary(dim, &mut r).expect("hv"))
            .collect();
        let targets: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();

        let mut reg = HdRegressor::new(make_cfg(dim, 1e-3)).expect("new");
        reg.fit(&hvs, &targets).expect("fit");

        for (hv, &y) in hvs.iter().zip(targets.iter()) {
            let pred = reg.predict(hv).expect("predict");
            assert!(
                (pred - y).abs() < 0.1,
                "training fit poor: pred={pred} target={y}"
            );
        }
    }

    #[test]
    fn r2_on_training_is_high() {
        let mut r = LcgRng::new(7);
        let dim = 4000;
        let n = 10;
        let hvs: Vec<Vec<i8>> = (0..n)
            .map(|_| random_binary(dim, &mut r).expect("hv"))
            .collect();
        let targets: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let mut reg = HdRegressor::new(make_cfg(dim, 1e-2)).expect("new");
        reg.fit(&hvs, &targets).expect("fit");
        let r2 = reg.r2_score(&hvs, &targets).expect("r2");
        assert!(r2 > 0.9, "training R² too low: {r2}");
    }

    #[test]
    fn constant_target_predicts_constant() {
        // All targets equal → model should predict that constant via the bias term.
        let mut r = LcgRng::new(11);
        let dim = 1000;
        let n = 6;
        let hvs: Vec<Vec<i8>> = (0..n)
            .map(|_| random_binary(dim, &mut r).expect("hv"))
            .collect();
        let targets = vec![3.5; n];

        let mut reg = HdRegressor::new(make_cfg(dim, 1.0)).expect("new");
        reg.fit(&hvs, &targets).expect("fit");
        let q = random_binary(dim, &mut r).expect("query");
        let pred = reg.predict(&q).expect("predict");
        assert!((pred - 3.5).abs() < 1e-3, "pred={pred}");
    }

    #[test]
    fn predict_batch_matches_single() {
        let mut r = LcgRng::new(13);
        let dim = 800;
        let n = 5;
        let hvs: Vec<Vec<i8>> = (0..n)
            .map(|_| random_binary(dim, &mut r).expect("hv"))
            .collect();
        let targets: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut reg = HdRegressor::new(make_cfg(dim, 0.1)).expect("new");
        reg.fit(&hvs, &targets).expect("fit");

        let batch = reg.predict_batch(&hvs).expect("batch");
        assert_eq!(batch.len(), n);
        for (i, hv) in hvs.iter().enumerate() {
            let single = reg.predict(hv).expect("single");
            assert!((batch[i] - single).abs() < 1e-9);
        }
    }

    #[test]
    fn predict_wrong_dim_errors() {
        let mut r = LcgRng::new(17);
        let dim = 256;
        let hvs = vec![random_binary(dim, &mut r).expect("hv")];
        let targets = vec![1.0];
        let mut reg = HdRegressor::new(make_cfg(dim, 1.0)).expect("new");
        reg.fit(&hvs, &targets).expect("fit");
        let bad = random_binary(128, &mut r).expect("bad");
        assert!(matches!(
            reg.predict(&bad),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn n_samples_and_fitted_flags() {
        let mut r = LcgRng::new(19);
        let dim = 256;
        let n = 4;
        let hvs: Vec<Vec<i8>> = (0..n)
            .map(|_| random_binary(dim, &mut r).expect("hv"))
            .collect();
        let targets: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut reg = HdRegressor::new(make_cfg(dim, 1.0)).expect("new");
        assert!(!reg.is_fitted());
        assert_eq!(reg.n_samples(), 0);
        reg.fit(&hvs, &targets).expect("fit");
        assert!(reg.is_fitted());
        assert_eq!(reg.n_samples(), n);
    }

    #[test]
    fn solver_solves_2x2() {
        // [[2,1],[1,3]] x = [3,5] → x = [0.8, 1.4]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![3.0, 5.0];
        solve_linear_system(&mut a, &mut b, 2).expect("solve");
        assert!((b[0] - 0.8).abs() < 1e-9, "x0={}", b[0]);
        assert!((b[1] - 1.4).abs() < 1e-9, "x1={}", b[1]);
    }

    #[test]
    fn solver_detects_singular() {
        // Singular system: two identical rows.
        let mut a = vec![1.0, 2.0, 1.0, 2.0];
        let mut b = vec![1.0, 1.0];
        assert!(matches!(
            solve_linear_system(&mut a, &mut b, 2),
            Err(HdcError::DivisionByZero)
        ));
    }
}
