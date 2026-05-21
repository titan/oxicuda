//! Functional / linearised Laplace approximation
//! (Immer, Korzepa, Bauer 2021 -- "Improving predictions of Bayesian neural
//! nets via local linearization", AISTATS).
//!
//! The classical Laplace approximation builds a Gaussian posterior over the
//! weights of a deep network from the curvature of the negative log-posterior
//! at the MAP estimate `θ*`. When that Gaussian is naively pushed through the
//! non-linear network, the resulting predictive distribution can be highly
//! biased. Functional / linearised Laplace fixes that by *linearising* the
//! network around `θ*`:
//!
//! ```text
//! f(x; θ) ≈ f(x; θ*) + J(x) · (θ − θ*),     J(x) = ∂f(x; θ) / ∂θ|_{θ=θ*}
//! ```
//!
//! With a Gaussian weight posterior `θ ~ N(θ*, Σ)`, the predictive at a test
//! point `x` is then exactly Gaussian:
//!
//! ```text
//! f(x) ~ N( f(x; θ*),  J(x) Σ J(x)ᵀ ).
//! ```
//!
//! The posterior precision matrix is the *Generalized Gauss-Newton* (GGN)
//! constructed from per-sample Jacobians `Jₙ ∈ ℝ^{output_dim × n_params}` and
//! a Gaussian (`L2`) prior with precision `α`:
//!
//! ```text
//! H = α · I + Σₙ JₙᵀJₙ,            Σ = H⁻¹.
//! ```
//!
//! This module materialises `H` and `Σ` as dense matrices and computes the
//! predictive variance `diag(J Σ Jᵀ)` in closed form. It is intended for
//! the regime in which `n_params` is small enough to invert a dense matrix
//! (a few hundred parameters); for very large networks one should fall back
//! to a Kronecker-factored (KFAC) or diagonal approximation.

use crate::error::{BayesError, BayesResult};

/// Configuration of a [`FunctionalLaplace`] fitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionalLaplaceConfig {
    /// Number of parameters `P` of the linearised model (≥ 1).
    pub n_params: usize,
    /// Per-sample output dimensionality `O` (≥ 1).
    pub output_dim: usize,
    /// Gaussian prior precision `α > 0`, so that `p(θ) = N(0, (1/α) I)`.
    pub prior_precision: f32,
}

/// Functional / linearised Laplace posterior.
///
/// Stores the symmetric positive-definite GGN-based posterior precision
/// `H ∈ ℝ^{P×P}` and its inverse `Σ = H⁻¹`. Both are kept in row-major,
/// length `P²` `Vec<f32>`.
#[derive(Debug, Clone)]
pub struct FunctionalLaplace {
    /// Original configuration.
    cfg: FunctionalLaplaceConfig,
    /// Posterior precision `H` (row-major, `P×P`).
    posterior_precision: Vec<f32>,
    /// Posterior covariance `Σ = H⁻¹` (row-major, `P×P`).
    posterior_covariance: Vec<f32>,
}

impl FunctionalLaplace {
    /// Build a fresh posterior initialised to the prior, i.e.
    /// `H = α I`, `Σ = (1/α) I`.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] if `n_params == 0` or `output_dim == 0`.
    /// - [`BayesError::InvalidPriorVariance`] if `prior_precision` is not
    ///   strictly positive and finite.
    pub fn new(cfg: FunctionalLaplaceConfig) -> BayesResult<Self> {
        if cfg.n_params == 0 || cfg.output_dim == 0 {
            return Err(BayesError::EmptyInputs);
        }
        if !(cfg.prior_precision.is_finite() && cfg.prior_precision > 0.0) {
            return Err(BayesError::InvalidPriorVariance);
        }
        let p = cfg.n_params;
        let mut h = vec![0.0_f32; p * p];
        let mut s = vec![0.0_f32; p * p];
        let inv_alpha = 1.0_f32 / cfg.prior_precision;
        for i in 0..p {
            h[i * p + i] = cfg.prior_precision;
            s[i * p + i] = inv_alpha;
        }
        Ok(Self {
            cfg,
            posterior_precision: h,
            posterior_covariance: s,
        })
    }

    /// Number of parameters `P`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.cfg.n_params
    }

    /// Output dimensionality `O`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.cfg.output_dim
    }

    /// Prior precision `α`.
    #[must_use]
    pub fn prior_precision(&self) -> f32 {
        self.cfg.prior_precision
    }

    /// Posterior precision matrix `H` (row-major, `P×P`).
    #[must_use]
    pub fn posterior_precision(&self) -> &[f32] {
        &self.posterior_precision
    }

    /// Posterior covariance matrix `Σ = H⁻¹` (row-major, `P×P`).
    #[must_use]
    pub fn posterior_covariance(&self) -> &[f32] {
        &self.posterior_covariance
    }

    /// Accumulate the GGN contribution `Σₙ JₙᵀJₙ` from `n_samples` Jacobians
    /// and refresh the posterior covariance `Σ = H⁻¹`.
    ///
    /// `jacobians` is row-major of length `n_samples · output_dim · n_params`.
    /// Each block of `output_dim · n_params` floats is one per-sample Jacobian
    /// matrix `Jₙ ∈ ℝ^{O × P}` itself stored row-major.
    ///
    /// The current value of `H` is *not* reset; calling `fit` twice with two
    /// disjoint mini-batches is equivalent (up to round-off) to calling it
    /// once on the concatenation. This is the standard online behaviour of
    /// the GGN.
    ///
    /// # Errors
    /// - [`BayesError::InsufficientSamples`] if `n_samples == 0`.
    /// - [`BayesError::DimensionMismatch`] if the slice length does not match
    ///   `n_samples · output_dim · n_params`.
    /// - [`BayesError::NanEncountered`] if the Cholesky inversion encounters
    ///   a non-positive pivot (numerically singular `H`).
    pub fn fit(&mut self, jacobians: &[f32], n_samples: usize) -> BayesResult<()> {
        if n_samples == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        let o = self.cfg.output_dim;
        let p = self.cfg.n_params;
        let expected = n_samples
            .checked_mul(o)
            .and_then(|v| v.checked_mul(p))
            .ok_or_else(|| BayesError::Internal("jacobian length overflow".into()))?;
        if jacobians.len() != expected {
            return Err(BayesError::DimensionMismatch {
                expected,
                got: jacobians.len(),
            });
        }
        // H ← H + Σₙ JₙᵀJₙ. Only the per-sample contribution is symmetric, so
        // we update the full P×P block to preserve symmetry exactly.
        let block = o * p;
        for n in 0..n_samples {
            let jac = &jacobians[n * block..(n + 1) * block];
            for i in 0..p {
                for j in i..p {
                    let mut acc = 0.0_f32;
                    for k in 0..o {
                        acc += jac[k * p + i] * jac[k * p + j];
                    }
                    self.posterior_precision[i * p + j] += acc;
                    if j != i {
                        self.posterior_precision[j * p + i] += acc;
                    }
                }
            }
        }
        self.posterior_covariance = invert_spd(&self.posterior_precision, p)?;
        Ok(())
    }

    /// Predictive mean and per-output variance at a test point.
    ///
    /// `jacobian` is row-major `output_dim × n_params`; `map_output` is the
    /// network output `f(x; θ*)` at the MAP estimate. Because the model is
    /// *linearised* at the MAP, the predictive mean is exactly `map_output`.
    /// For each output index `o`, the variance is `J_o · Σ · J_oᵀ` where
    /// `J_o` is row `o` of the Jacobian.
    ///
    /// Returns `(mean, variance)`, each of length `output_dim`.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `jacobian.len() != output_dim *
    ///   n_params` or `map_output.len() != output_dim`.
    pub fn predict(
        &self,
        jacobian: &[f32],
        map_output: &[f32],
    ) -> BayesResult<(Vec<f32>, Vec<f32>)> {
        let o = self.cfg.output_dim;
        let p = self.cfg.n_params;
        if jacobian.len() != o * p {
            return Err(BayesError::DimensionMismatch {
                expected: o * p,
                got: jacobian.len(),
            });
        }
        if map_output.len() != o {
            return Err(BayesError::DimensionMismatch {
                expected: o,
                got: map_output.len(),
            });
        }
        let mean = map_output.to_vec();
        let mut variance = vec![0.0_f32; o];
        for row in 0..o {
            let j_row = &jacobian[row * p..(row + 1) * p];
            // var_o = j_row · Σ · j_rowᵀ
            let mut acc = 0.0_f32;
            for i in 0..p {
                let ji = j_row[i];
                if ji == 0.0 {
                    continue;
                }
                let sigma_row = &self.posterior_covariance[i * p..(i + 1) * p];
                let mut inner = 0.0_f32;
                for k in 0..p {
                    inner += sigma_row[k] * j_row[k];
                }
                acc += ji * inner;
            }
            // Σ is SPD (positive-definite) so acc ≥ 0; clamp tiny negatives
            // that arise from floating-point round-off.
            variance[row] = acc.max(0.0_f32);
        }
        Ok((mean, variance))
    }
}

// ─── Numerics ────────────────────────────────────────────────────────────────

/// Invert a symmetric positive-definite `n×n` matrix `a` (row-major) by an
/// in-place Cholesky factorisation followed by triangular back-substitution.
///
/// Returns a fresh `n×n` symmetric `Vec<f32>` containing `a⁻¹`.
///
/// # Errors
/// [`BayesError::NanEncountered`] is returned if a non-positive pivot is
/// encountered, signalling that `a` is not numerically positive-definite.
fn invert_spd(a: &[f32], n: usize) -> BayesResult<Vec<f32>> {
    let mut l = vec![0.0_f32; n * n];
    // Cholesky: a = L Lᵀ, L lower-triangular.
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(BayesError::NanEncountered {
                        location: "functional_laplace::invert_spd: non-positive Cholesky pivot",
                    });
                }
                l[i * n + i] = sum.sqrt();
            } else {
                let l_jj = l[j * n + j];
                if !(l_jj.is_finite() && l_jj > 0.0) {
                    return Err(BayesError::NanEncountered {
                        location: "functional_laplace::invert_spd: zero diagonal in Cholesky",
                    });
                }
                l[i * n + j] = sum / l_jj;
            }
        }
    }
    // Solve L Lᵀ x = e_j for each unit vector e_j to assemble A⁻¹ column by
    // column. Result is symmetric so we only fill the lower triangle then
    // mirror.
    let mut inv = vec![0.0_f32; n * n];
    let mut y = vec![0.0_f32; n];
    let mut x = vec![0.0_f32; n];
    for col in 0..n {
        // Forward solve: L y = e_col.
        for i in 0..n {
            let mut sum = if i == col { 1.0_f32 } else { 0.0_f32 };
            for k in 0..i {
                sum -= l[i * n + k] * y[k];
            }
            y[i] = sum / l[i * n + i];
        }
        // Backward solve: Lᵀ x = y.
        for ii in 0..n {
            let i = n - 1 - ii;
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[k * n + i] * x[k];
            }
            x[i] = sum / l[i * n + i];
        }
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    // Symmetrise to undo round-off.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5_f32 * (inv[i * n + j] + inv[j * n + i]);
            inv[i * n + j] = avg;
            inv[j * n + i] = avg;
        }
    }
    Ok(inv)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n_params: usize, output_dim: usize, prior_precision: f32) -> FunctionalLaplace {
        FunctionalLaplace::new(FunctionalLaplaceConfig {
            n_params,
            output_dim,
            prior_precision,
        })
        .expect("test invariant: FunctionalLaplace::new must succeed")
    }

    #[test]
    fn new_prior_only_covariance_is_isotropic() {
        let lap = make(3, 2, 4.0);
        let sigma = lap.posterior_covariance();
        let p = lap.n_params();
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { 1.0_f32 / 4.0 } else { 0.0 };
                assert!(
                    (sigma[i * p + j] - target).abs() < 1e-6,
                    "Σ[{i},{j}] = {} vs {}",
                    sigma[i * p + j],
                    target
                );
            }
        }
    }

    #[test]
    fn new_prior_only_precision_is_isotropic() {
        let lap = make(3, 2, 4.0);
        let h = lap.posterior_precision();
        let p = lap.n_params();
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { 4.0_f32 } else { 0.0 };
                assert!((h[i * p + j] - target).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn posterior_covariance_symmetric() {
        let mut lap = make(3, 2, 1.0);
        let jacobians = vec![
            1.0_f32, 2.0, 3.0, 0.5, -0.5, 1.5, // sample 0 (2 rows × 3 cols)
            0.1, -0.2, 0.3, 0.7, 0.4, -0.6, // sample 1
        ];
        lap.fit(&jacobians, 2)
            .expect("test invariant: fit must succeed");
        let sigma = lap.posterior_covariance();
        let p = lap.n_params();
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (sigma[i * p + j] - sigma[j * p + i]).abs() < 1e-5,
                    "asymmetry at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn predict_mean_equals_map_output_exactly() {
        let mut lap = make(2, 3, 1.0);
        let jacobians = vec![0.7_f32, 0.3, 0.1, 0.9, -0.2, 0.4];
        lap.fit(&jacobians, 1)
            .expect("test invariant: fit must succeed");
        let jac_test = vec![0.5_f32, -0.5, 1.0, 0.0, 0.3, 0.3];
        let map_output = vec![1.23_f32, -4.56, 7.89];
        let (mean, _var) = lap
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        assert_eq!(mean, map_output);
    }

    #[test]
    fn predict_variance_non_negative() {
        let mut lap = make(4, 2, 0.5);
        let jacobians: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        lap.fit(&jacobians, 1)
            .expect("test invariant: fit must succeed");
        let jac_test = vec![1.0_f32, -1.0, 0.5, -0.5, 0.25, 0.25, 0.0, 1.0];
        let map_output = vec![0.0_f32, 0.0];
        let (_mean, var) = lap
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        for &v in &var {
            assert!(v >= 0.0, "variance {v} must be non-negative");
        }
    }

    #[test]
    fn prior_only_predictive_variance_matches_closed_form() {
        // Σ = (1/α) I  →  var_o = (1/α) · ‖J_o‖².
        let alpha = 2.5_f32;
        let lap = make(3, 2, alpha);
        let jac_test = vec![1.0_f32, 2.0, 3.0, 0.5, -1.5, 2.5];
        let map_output = vec![0.1_f32, 0.2];
        let (_mean, var) = lap
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        let expected_0 = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / alpha;
        let expected_1 = (0.5 * 0.5 + 1.5 * 1.5 + 2.5 * 2.5) / alpha;
        assert!(
            (var[0] - expected_0).abs() < 1e-5,
            "v0={} vs {}",
            var[0],
            expected_0
        );
        assert!(
            (var[1] - expected_1).abs() < 1e-5,
            "v1={} vs {}",
            var[1],
            expected_1
        );
    }

    #[test]
    fn fit_data_shrinks_variance_below_prior() {
        let alpha = 1.0_f32;
        let mut lap = make(2, 1, alpha);
        let jac_test = vec![1.0_f32, 1.0];
        let map_output = vec![0.0_f32];
        let (_, var_prior) = lap
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        let jacobians = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        lap.fit(&jacobians, 3)
            .expect("test invariant: fit must succeed");
        let (_, var_post) = lap
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        assert!(
            var_post[0] < var_prior[0],
            "variance should shrink with data: prior={}, post={}",
            var_prior[0],
            var_post[0]
        );
    }

    #[test]
    fn fit_accumulates_additively_two_batches() {
        let cfg = FunctionalLaplaceConfig {
            n_params: 2,
            output_dim: 1,
            prior_precision: 1.0,
        };
        let batch1 = vec![1.0_f32, 0.0, 0.5, 0.5];
        let batch2 = vec![-1.0_f32, 1.0, 0.3, -0.3];
        let mut lap_a = FunctionalLaplace::new(cfg)
            .expect("test invariant: FunctionalLaplace::new must succeed");
        lap_a
            .fit(&batch1, 2)
            .expect("test invariant: fit must succeed");
        lap_a
            .fit(&batch2, 2)
            .expect("test invariant: fit must succeed");

        let mut combined = batch1.clone();
        combined.extend_from_slice(&batch2);
        let mut lap_b = FunctionalLaplace::new(cfg)
            .expect("test invariant: FunctionalLaplace::new must succeed");
        lap_b
            .fit(&combined, 4)
            .expect("test invariant: fit must succeed");

        let p = cfg.n_params;
        for i in 0..p {
            for j in 0..p {
                let a = lap_a.posterior_precision()[i * p + j];
                let b = lap_b.posterior_precision()[i * p + j];
                assert!((a - b).abs() < 1e-4, "H mismatch at ({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn posterior_covariance_is_psd() {
        let mut lap = make(3, 2, 0.5);
        let jacobians = vec![
            0.8_f32, -0.4, 0.6, 0.2, 0.7, -0.1, // sample 0
            -0.3, 0.5, 0.9, 0.4, -0.2, 0.3, // sample 1
            0.1, 0.2, -0.5, 0.8, 0.6, 0.0, // sample 2
        ];
        lap.fit(&jacobians, 3)
            .expect("test invariant: fit must succeed");
        let p = lap.n_params();
        let sigma = lap.posterior_covariance();
        // Test PSD via several random vectors v: vᵀΣv ≥ 0.
        let test_vectors: [[f32; 3]; 5] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 0.5],
            [0.7, 0.3, -0.9],
        ];
        for v in &test_vectors {
            let mut q = 0.0_f32;
            for i in 0..p {
                for j in 0..p {
                    q += v[i] * sigma[i * p + j] * v[j];
                }
            }
            assert!(q >= -1e-5, "vᵀΣv = {q} should be ≥ 0");
        }
    }

    #[test]
    fn higher_prior_precision_means_smaller_prior_variance() {
        let lap_low = make(2, 1, 0.5);
        let lap_high = make(2, 1, 5.0);
        let jac_test = vec![1.0_f32, 1.0];
        let map_output = vec![0.0_f32];
        let (_, v_low) = lap_low
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        let (_, v_high) = lap_high
            .predict(&jac_test, &map_output)
            .expect("test invariant: predict must succeed");
        assert!(
            v_high[0] < v_low[0],
            "high precision should yield smaller variance: high={}, low={}",
            v_high[0],
            v_low[0]
        );
    }

    #[test]
    fn deterministic() {
        let mut lap_a = make(3, 2, 1.5);
        let mut lap_b = make(3, 2, 1.5);
        let jacobians: Vec<f32> = (0..12).map(|i| (i as f32 - 5.5) * 0.2).collect();
        lap_a
            .fit(&jacobians, 2)
            .expect("test invariant: fit must succeed");
        lap_b
            .fit(&jacobians, 2)
            .expect("test invariant: fit must succeed");
        let p = lap_a.n_params();
        for i in 0..(p * p) {
            assert_eq!(
                lap_a.posterior_covariance()[i],
                lap_b.posterior_covariance()[i]
            );
            assert_eq!(
                lap_a.posterior_precision()[i],
                lap_b.posterior_precision()[i]
            );
        }
    }

    #[test]
    fn err_jacobians_wrong_length() {
        let mut lap = make(2, 2, 1.0);
        let jacobians = vec![0.0_f32; 5]; // expected 1·2·2 = 4 for one sample.
        let r = lap.fit(&jacobians, 1);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_predict_jacobian_wrong_length() {
        let lap = make(2, 2, 1.0);
        let bad_jac = vec![0.0_f32; 5]; // expected output_dim·n_params = 4.
        let map_output = vec![0.0_f32, 0.0];
        let r = lap.predict(&bad_jac, &map_output);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_predict_map_output_wrong_length() {
        let lap = make(2, 2, 1.0);
        let jac = vec![0.0_f32; 4];
        let bad_map = vec![0.0_f32; 3]; // expected output_dim = 2.
        let r = lap.predict(&jac, &bad_map);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_n_params_zero() {
        let r = FunctionalLaplace::new(FunctionalLaplaceConfig {
            n_params: 0,
            output_dim: 2,
            prior_precision: 1.0,
        });
        assert!(matches!(r, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn err_output_dim_zero() {
        let r = FunctionalLaplace::new(FunctionalLaplaceConfig {
            n_params: 2,
            output_dim: 0,
            prior_precision: 1.0,
        });
        assert!(matches!(r, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn err_prior_precision_non_positive() {
        for bad in [0.0_f32, -1.0, -0.001, f32::NAN, f32::INFINITY] {
            let r = FunctionalLaplace::new(FunctionalLaplaceConfig {
                n_params: 2,
                output_dim: 2,
                prior_precision: bad,
            });
            assert!(
                matches!(r, Err(BayesError::InvalidPriorVariance)),
                "expected InvalidPriorVariance for prior={bad}"
            );
        }
    }

    #[test]
    fn err_n_samples_zero() {
        let mut lap = make(2, 1, 1.0);
        let r = lap.fit(&[], 0);
        assert!(matches!(r, Err(BayesError::InsufficientSamples { .. })));
    }

    #[test]
    fn scalar_output_dim_one() {
        let mut lap = make(2, 1, 1.0);
        let jacobians = vec![1.0_f32, 0.0, 0.0, 1.0]; // two scalar Jacobians
        lap.fit(&jacobians, 2)
            .expect("test invariant: fit must succeed");
        let (mean, var) = lap
            .predict(&[1.0_f32, 0.0], &[2.5_f32])
            .expect("test invariant: predict must succeed");
        assert_eq!(mean.len(), 1);
        assert_eq!(var.len(), 1);
        // H = I + diag(1,1) = 2 I  →  Σ = 0.5 I  →  var = 0.5·1² = 0.5.
        assert!((var[0] - 0.5).abs() < 1e-5, "var={}", var[0]);
        assert_eq!(mean[0], 2.5);
    }

    #[test]
    fn identity_jacobian_sanity() {
        // J = I_p, single sample → H = α I + I = (α+1) I → Σ = (1/(α+1)) I.
        // For a unit test Jacobian (also identity), var_i = 1/(α+1).
        let alpha = 1.0_f32;
        let mut lap = make(3, 3, alpha);
        let identity = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        lap.fit(&identity, 1)
            .expect("test invariant: fit must succeed");
        let (_mean, var) = lap
            .predict(&identity, &[0.0_f32, 0.0, 0.0])
            .expect("test invariant: predict must succeed");
        let expected = 1.0_f32 / (alpha + 1.0);
        for &v in &var {
            assert!((v - expected).abs() < 1e-5, "var={v} vs {expected}");
        }
    }

    #[test]
    fn predict_off_diagonal_uses_full_covariance() {
        // Build a non-isotropic posterior: prior + a single sample with two
        // correlated rows. Verify that the predictive variance for a Jacobian
        // that mixes the two parameters depends on the off-diagonal of Σ.
        let mut lap = make(2, 1, 1.0);
        let jacobians = vec![1.0_f32, 1.0]; // 1 sample, 1 output, 2 params
        lap.fit(&jacobians, 1)
            .expect("test invariant: fit must succeed");
        // H = [[2,1],[1,2]] → Σ = 1/3 * [[2,-1],[-1,2]].
        let sigma = lap.posterior_covariance();
        assert!((sigma[0] - (2.0_f32 / 3.0)).abs() < 1e-4);
        assert!((sigma[1] - (-1.0_f32 / 3.0)).abs() < 1e-4);
        assert!((sigma[3] - (2.0_f32 / 3.0)).abs() < 1e-4);
        // var for J=[1,1]: 1·(2/3) + 1·(-1/3) + 1·(-1/3) + 1·(2/3) = 2/3.
        let (_, var) = lap
            .predict(&[1.0_f32, 1.0], &[0.0_f32])
            .expect("test invariant: predict must succeed");
        assert!((var[0] - 2.0_f32 / 3.0).abs() < 1e-4, "var={}", var[0]);
    }
}
