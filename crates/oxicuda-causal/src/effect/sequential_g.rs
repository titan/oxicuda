//! Sequential g-estimation of Structural Nested Mean Models (SNMM).
//!
//! Robins JM (1994) "Correcting for non-compliance in randomized trials using
//! structural nested mean models." *Communications in Statistics: Theory and
//! Methods* 23(8):2379–2412.
//!
//! # Problem
//!
//! For a sequence of time-varying treatments A₀, A₁, …, A_{K−1} and a final
//! outcome Y, the SNMM posits a blip function ψ(Aₖ; γ) that characterises
//! the direct effect of treatment at time k. The causal parameter γ satisfies
//! the g-estimating equation:
//!
//! ```text
//!   Σ_i Σ_k (A_{ik} − π_{ik}) · [Y_i − Σ_{j≥k} ψ(A_{ij}; γ)] = 0
//! ```
//!
//! where π_{ik} = E[A_{ik} | Ā_{k−1}, X] is the propensity score at time k.
//! The residual (A − π) acts as an instrument that is mean-zero conditional on
//! the history under the no-unmeasured-confounders assumption.
//!
//! # Blip functions
//!
//! - **ConstantAdditive**: ψ(Aₖ; γ) = γ · Aₖ. Yields a scalar γ̂ = num/den.
//! - **LinearModifier**: ψ(Aₖ, Vᵢ; γ) = (γ₀ + γ₁·Vᵢ) · Aₖ. Yields (γ₀, γ₁)
//!   via a 2×2 linear system.
//!
//! # Propensity model
//!
//! At each time k a logistic model is fitted by gradient descent. The design
//! matrix at k=0 is `[1, x₁, …, x_p]`; at k>0 it is `[1, A_{k−1}, x₁, …, x_p]`.

use crate::error::{CausalError, CausalResult};
use crate::handle::CausalHandle;

/// Blip-function specification for the SNMM.
#[derive(Clone, Debug)]
pub enum BlipFunction {
    /// ψ(Aₖ; γ) = γ · Aₖ — a single scalar causal effect.
    ConstantAdditive,
    /// ψ(Aₖ, Vᵢ; γ) = (γ₀ + γ₁·Vᵢ) · Aₖ — effect modified by covariate column `col`.
    LinearModifier { col: usize },
}

/// Configuration for the sequential g-estimator.
#[derive(Clone, Debug)]
pub struct SequentialGConfig {
    /// Blip-function type.
    pub blip_function: BlipFunction,
    /// Number of gradient-descent iterations for each propensity model.
    pub propensity_max_iter: usize,
    /// Ridge regularisation strength (L2) applied to non-intercept weights.
    pub propensity_ridge: f64,
    /// Gradient-descent step size.
    pub propensity_step_size: f64,
    /// Bootstrap replications for SE estimation; 0 skips bootstrap.
    pub bootstrap_reps: usize,
}

impl Default for SequentialGConfig {
    fn default() -> Self {
        Self {
            blip_function: BlipFunction::ConstantAdditive,
            propensity_max_iter: 200,
            propensity_ridge: 1e-4,
            propensity_step_size: 0.05,
            bootstrap_reps: 0,
        }
    }
}

/// Results returned by the sequential g-estimator.
pub struct SequentialGResult {
    /// Estimated causal parameter(s): `[γ]` for ConstantAdditive, `[γ₀, γ₁]` for LinearModifier.
    pub gamma: Vec<f64>,
    /// Bootstrap standard errors; all zeros when `bootstrap_reps = 0`.
    pub se_gamma: Vec<f64>,
    /// Blipped-down outcome Y(0): Y_i − Σ_k ψ(A_{ik}; γ̂).
    pub blipped_down_y: Vec<f64>,
    /// Propensity matrix in row-major order, shape (n × k_times).
    pub propensity_matrix: Vec<f64>,
}

/// Stateless namespace for sequential g-estimation.
pub struct SequentialGEstimator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Fit propensity scores for all time points.
///
/// Returns a flat Vec of length n * k_times in row-major order.
fn fit_propensity_all(
    a: &[f64],
    x: &[f64],
    n: usize,
    k_times: usize,
    p: usize,
    max_iter: usize,
    ridge: f64,
    step_size: f64,
) -> Vec<f64> {
    let mut pi_mat = vec![0.5_f64; n * k_times];

    for k in 0..k_times {
        // Design dimension: intercept + (A_{k-1} if k>0) + p covariates.
        let d_cols = if k == 0 { 1 + p } else { 2 + p };
        let mut alpha = vec![0.0_f64; d_cols];

        // Build design matrix rows for current time point k.
        // D_k[i] = [1, (A_{i,k-1} if k>0), x_{i,0}, ..., x_{i,p-1}]
        // We compute GD directly without materialising the full matrix.
        for _iter in 0..max_iter {
            let mut grad = vec![0.0_f64; d_cols];
            for i in 0..n {
                // Compute dot product with design row.
                let mut dot = alpha[0]; // intercept
                let mut col = 1_usize;
                if k > 0 {
                    dot += alpha[col] * a[i * k_times + (k - 1)];
                    col += 1;
                }
                for j in 0..p {
                    dot += alpha[col + j] * x[i * p + j];
                }
                let pred = sigmoid(dot);
                let a_ik = a[i * k_times + k];
                let err = (pred - a_ik) / n as f64;

                // Accumulate gradient.
                grad[0] += err; // intercept
                let mut col = 1_usize;
                if k > 0 {
                    grad[col] += err * a[i * k_times + (k - 1)];
                    col += 1;
                }
                for j in 0..p {
                    grad[col + j] += err * x[i * p + j];
                }
            }
            // Update weights with optional ridge (skip intercept at index 0).
            alpha[0] -= step_size * grad[0];
            let non_intercept_start = 1_usize;
            for j in non_intercept_start..d_cols {
                alpha[j] -= step_size * (grad[j] + ridge * alpha[j]);
            }
        }

        // Store predicted propensities.
        for i in 0..n {
            let mut dot = alpha[0];
            let mut col = 1_usize;
            if k > 0 {
                dot += alpha[col] * a[i * k_times + (k - 1)];
                col += 1;
            }
            for j in 0..p {
                dot += alpha[col + j] * x[i * p + j];
            }
            pi_mat[i * k_times + k] = sigmoid(dot);
        }
    }

    pi_mat
}

/// Core g-estimating equation solver — ConstantAdditive blip.
///
/// Returns γ as a single-element Vec.
fn solve_constant_additive(
    y: &[f64],
    a: &[f64],
    pi_mat: &[f64],
    n: usize,
    k_times: usize,
) -> CausalResult<Vec<f64>> {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for i in 0..n {
        // Cumulative treatment from time k onward: Σ_{j=k}^{k_times−1} A_{ij}.
        // We precompute suffix sums.
        let mut cum_a = vec![0.0_f64; k_times + 1];
        for j in (0..k_times).rev() {
            cum_a[j] = cum_a[j + 1] + a[i * k_times + j];
        }
        for k in 0..k_times {
            let r_ik = a[i * k_times + k] - pi_mat[i * k_times + k];
            num += r_ik * y[i];
            den += r_ik * cum_a[k];
        }
    }

    if den.abs() < 1e-10 {
        return Err(CausalError::InvalidParameter {
            reason: "near-zero denominator in g-estimating equation (ConstantAdditive)".to_string(),
        });
    }
    Ok(vec![num / den])
}

/// Core g-estimating equation solver — LinearModifier blip.
///
/// Returns [γ₀, γ₁].
fn solve_linear_modifier(
    y: &[f64],
    a: &[f64],
    x: &[f64],
    pi_mat: &[f64],
    n: usize,
    k_times: usize,
    p: usize,
    col: usize,
) -> CausalResult<Vec<f64>> {
    // Build the 2×2 moment system.
    let mut m00 = 0.0_f64;
    let mut m01 = 0.0_f64;
    let mut m11 = 0.0_f64;
    let mut b0 = 0.0_f64;
    let mut b1 = 0.0_f64;

    for i in 0..n {
        let v_i = x[i * p + col];
        let mut cum_a = vec![0.0_f64; k_times + 1];
        for j in (0..k_times).rev() {
            cum_a[j] = cum_a[j + 1] + a[i * k_times + j];
        }
        for k in 0..k_times {
            let r_ik = a[i * k_times + k] - pi_mat[i * k_times + k];
            let ca = cum_a[k];
            m00 += r_ik * ca;
            m01 += r_ik * v_i * ca;
            m11 += r_ik * v_i * v_i * ca;
            b0 += r_ik * y[i];
            b1 += r_ik * v_i * y[i];
        }
    }

    // The system is symmetric: m10 = m01.
    let det = m00 * m11 - m01 * m01;
    if det.abs() < 1e-10 {
        return Err(CausalError::InvalidParameter {
            reason: "near-zero determinant in 2×2 g-estimating system (LinearModifier)".to_string(),
        });
    }
    let gamma0 = (m11 * b0 - m01 * b1) / det;
    let gamma1 = (m00 * b1 - m01 * b0) / det;
    Ok(vec![gamma0, gamma1])
}

/// Compute the blipped-down outcome Y(0).
fn blip_down(
    y: &[f64],
    a: &[f64],
    x: &[f64],
    n: usize,
    k_times: usize,
    p: usize,
    gamma: &[f64],
    blip: &BlipFunction,
) -> Vec<f64> {
    let mut blipped = y.to_vec();
    match blip {
        BlipFunction::ConstantAdditive => {
            let g = gamma[0];
            for i in 0..n {
                let total_a: f64 = (0..k_times).map(|k| a[i * k_times + k]).sum();
                blipped[i] -= g * total_a;
            }
        }
        BlipFunction::LinearModifier { col } => {
            let g0 = gamma[0];
            let g1 = gamma[1];
            for i in 0..n {
                let v_i = x[i * p + col];
                let total_a: f64 = (0..k_times).map(|k| a[i * k_times + k]).sum();
                blipped[i] -= (g0 + g1 * v_i) * total_a;
            }
        }
    }
    blipped
}

/// Inner fitting routine without bootstrap — shared between main fit and bootstrap.
fn fit_inner(
    y: &[f64],
    a: &[f64],
    x: &[f64],
    n: usize,
    k_times: usize,
    p: usize,
    cfg: &SequentialGConfig,
) -> CausalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pi_mat = fit_propensity_all(
        a,
        x,
        n,
        k_times,
        p,
        cfg.propensity_max_iter,
        cfg.propensity_ridge,
        cfg.propensity_step_size,
    );

    let gamma = match &cfg.blip_function {
        BlipFunction::ConstantAdditive => solve_constant_additive(y, a, &pi_mat, n, k_times)?,
        BlipFunction::LinearModifier { col } => {
            solve_linear_modifier(y, a, x, &pi_mat, n, k_times, p, *col)?
        }
    };

    let blipped = blip_down(y, a, x, n, k_times, p, &gamma, &cfg.blip_function);
    Ok((gamma, pi_mat, blipped))
}

impl SequentialGEstimator {
    /// Fit the sequential g-estimator.
    ///
    /// # Parameters
    ///
    /// - `y` — outcome vector of length `n`.
    /// - `a` — treatment matrix (n × k_times) in row-major order.
    /// - `x` — baseline covariate matrix (n × p) in row-major order.
    /// - `n`, `k_times`, `p` — dimension declarations.
    /// - `cfg` — algorithm configuration.
    /// - `handle` — provides the PRNG for bootstrap sampling.
    ///
    /// # Errors
    ///
    /// Returns [`CausalError::InvalidParameter`] or [`CausalError::DimensionMismatch`]
    /// on any invalid input or near-zero g-estimating denominator.
    pub fn fit(
        y: &[f64],
        a: &[f64],
        x: &[f64],
        n: usize,
        k_times: usize,
        p: usize,
        cfg: &SequentialGConfig,
        handle: &mut CausalHandle,
    ) -> CausalResult<SequentialGResult> {
        // Validate dimensions.
        if n == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "n must be > 0".to_string(),
            });
        }
        if k_times == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "k_times must be > 0".to_string(),
            });
        }
        if p == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "p must be > 0".to_string(),
            });
        }
        if y.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }
        if a.len() != n * k_times {
            return Err(CausalError::DimensionMismatch {
                expected: n * k_times,
                got: a.len(),
            });
        }
        if x.len() != n * p {
            return Err(CausalError::DimensionMismatch {
                expected: n * p,
                got: x.len(),
            });
        }
        // Validate LinearModifier column index.
        if let BlipFunction::LinearModifier { col } = &cfg.blip_function
            && *col >= p
        {
            return Err(CausalError::InvalidParameter {
                reason: format!("LinearModifier col={col} is out of bounds for p={p}"),
            });
        }

        // Main fit.
        let (gamma, pi_mat, blipped_down_y) = fit_inner(y, a, x, n, k_times, p, cfg)?;

        // Bootstrap SE.
        let n_gamma = gamma.len();
        let se_gamma = if cfg.bootstrap_reps == 0 {
            vec![0.0_f64; n_gamma]
        } else {
            let reps = cfg.bootstrap_reps;
            let mut boot_gammas: Vec<Vec<f64>> = Vec::with_capacity(reps);

            for _ in 0..reps {
                // Sample n indices with replacement.
                let indices: Vec<usize> = (0..n)
                    .map(|_| (handle.rng.next_u64() % n as u64) as usize)
                    .collect();

                // Build resampled arrays.
                let mut y_b = vec![0.0_f64; n];
                let mut a_b = vec![0.0_f64; n * k_times];
                let mut x_b = vec![0.0_f64; n * p];
                for (new_i, &orig_i) in indices.iter().enumerate() {
                    y_b[new_i] = y[orig_i];
                    for k in 0..k_times {
                        a_b[new_i * k_times + k] = a[orig_i * k_times + k];
                    }
                    for j in 0..p {
                        x_b[new_i * p + j] = x[orig_i * p + j];
                    }
                }

                // Fit without bootstrap to avoid recursion.
                let no_boot_cfg = SequentialGConfig {
                    bootstrap_reps: 0,
                    blip_function: cfg.blip_function.clone(),
                    propensity_max_iter: cfg.propensity_max_iter,
                    propensity_ridge: cfg.propensity_ridge,
                    propensity_step_size: cfg.propensity_step_size,
                };
                // skip failed bootstrap rep (near-zero denominator, etc.)
                if let Ok((g_b, _, _)) = fit_inner(&y_b, &a_b, &x_b, n, k_times, p, &no_boot_cfg) {
                    boot_gammas.push(g_b);
                }
            }

            // Compute standard deviations.
            if boot_gammas.is_empty() {
                vec![0.0_f64; n_gamma]
            } else {
                let m = boot_gammas.len() as f64;
                (0..n_gamma)
                    .map(|j| {
                        let mean = boot_gammas.iter().map(|g| g[j]).sum::<f64>() / m;
                        let var = boot_gammas
                            .iter()
                            .map(|g| (g[j] - mean).powi(2))
                            .sum::<f64>()
                            / m.max(2.0 - 1.0); // sample std dev (m−1 denominator)
                        var.sqrt()
                    })
                    .collect()
            }
        };

        Ok(SequentialGResult {
            gamma,
            se_gamma,
            blipped_down_y,
            propensity_matrix: pi_mat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::CausalHandle;

    fn make_handle() -> CausalHandle {
        CausalHandle::new(80, 42)
    }

    // --- Error paths ---

    #[test]
    fn error_n_zero() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        assert!(SequentialGEstimator::fit(&[], &[], &[], 0, 1, 1, &cfg, &mut h).is_err());
    }

    #[test]
    fn error_k_zero() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        assert!(SequentialGEstimator::fit(&[1.0], &[], &[1.0], 1, 0, 1, &cfg, &mut h).is_err());
    }

    #[test]
    fn error_p_zero() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        assert!(SequentialGEstimator::fit(&[1.0], &[1.0], &[], 1, 1, 0, &cfg, &mut h).is_err());
    }

    #[test]
    fn error_y_dim_mismatch() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        // n=2 but y has 3 elements
        assert!(
            SequentialGEstimator::fit(
                &[1.0, 2.0, 3.0],
                &[0.0, 1.0],
                &[1.0, 1.0],
                2,
                1,
                1,
                &cfg,
                &mut h
            )
            .is_err()
        );
    }

    #[test]
    fn error_a_dim_mismatch() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        // n=2, k=2, so a should have 4 elements but we give 3
        assert!(
            SequentialGEstimator::fit(
                &[1.0, 2.0],
                &[0.0, 1.0, 0.0],
                &[1.0, 1.0],
                2,
                2,
                1,
                &cfg,
                &mut h
            )
            .is_err()
        );
    }

    #[test]
    fn error_x_dim_mismatch() {
        let cfg = SequentialGConfig::default();
        let mut h = make_handle();
        // n=2, p=2, so x should have 4 elements but we give 2
        assert!(
            SequentialGEstimator::fit(&[1.0, 2.0], &[0.0, 1.0], &[1.0, 1.0], 2, 1, 2, &cfg, &mut h)
                .is_err()
        );
    }

    #[test]
    fn error_linear_modifier_col_out_of_bounds() {
        let cfg = SequentialGConfig {
            blip_function: BlipFunction::LinearModifier { col: 5 },
            ..Default::default()
        };
        let mut h = make_handle();
        let y = vec![1.0, 2.0];
        let a = vec![0.0, 1.0];
        let x = vec![1.0, 1.0, 2.0, 2.0]; // n=2, p=2
        assert!(SequentialGEstimator::fit(&y, &a, &x, 2, 1, 2, &cfg, &mut h).is_err());
    }

    #[test]
    fn error_all_treated_near_zero_denominator() {
        // If A_i = 1 for all i and π ≈ 1, then r_{ik} ≈ 0 → denominator ≈ 0.
        let n = 10;
        let cfg = SequentialGConfig {
            propensity_max_iter: 500,
            ..Default::default()
        };
        let mut h = make_handle();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // All treated
        let a = vec![1.0_f64; n];
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        // This may or may not error depending on how well propensity fits;
        // we just ensure it doesn't panic.
        let _result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h);
    }

    // --- Correctness tests ---

    #[test]
    fn gamma_near_true_constant_additive() {
        // DGP: A_i ~ Bernoulli(0.5), Y_i = 0.5·A_i + ε, k=1.
        // g-estimate should recover γ ≈ 0.5.
        let n = 200;
        let mut h = CausalHandle::new(80, 1234);
        let mut a = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let x = vec![1.0_f64; n]; // constant covariate

        // Generate data deterministically via LcgRng.
        for i in 0..n {
            let u = h.rng.next_f32() as f64;
            a[i] = if u < 0.5 { 1.0 } else { 0.0 };
            let noise = h.rng.next_normal() as f64 * 0.5;
            y[i] = 0.5 * a[i] + noise;
        }

        let cfg = SequentialGConfig {
            propensity_max_iter: 300,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        let gamma = result.gamma[0];
        // Allow generous tolerance due to finite-sample noise.
        assert!((gamma - 0.5).abs() < 0.4, "gamma={gamma} not close to 0.5");
    }

    #[test]
    fn gamma_near_zero_null_dgp() {
        // DGP: Y_i = ε (no treatment effect).
        let n = 200;
        let mut h = CausalHandle::new(80, 9999);
        let mut a = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let x = vec![1.0_f64; n];

        for i in 0..n {
            let u = h.rng.next_f32() as f64;
            a[i] = if u < 0.5 { 1.0 } else { 0.0 };
            y[i] = h.rng.next_normal() as f64 * 0.5;
        }

        let cfg = SequentialGConfig {
            propensity_max_iter: 300,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        let gamma = result.gamma[0];
        assert!(
            gamma.abs() < 0.5,
            "gamma={gamma} should be near zero for null DGP"
        );
    }

    #[test]
    fn propensity_matrix_correct_shape() {
        let n = 20;
        let k = 3;
        let p = 2;
        let mut h = make_handle();
        let y = vec![0.0_f64; n];
        let a: Vec<f64> = (0..n * k)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let x = vec![1.0_f64; n * p];
        let cfg = SequentialGConfig::default();
        let result = SequentialGEstimator::fit(&y, &a, &x, n, k, p, &cfg, &mut h).unwrap();
        assert_eq!(result.propensity_matrix.len(), n * k);
    }

    #[test]
    fn blipped_down_y_correct_length() {
        let n = 30;
        let k = 2;
        let p = 1;
        let mut h = make_handle();
        let y = vec![1.0_f64; n];
        let a: Vec<f64> = (0..n * k)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let x = vec![0.5_f64; n];
        let cfg = SequentialGConfig::default();
        let result = SequentialGEstimator::fit(&y, &a, &x, n, k, p, &cfg, &mut h).unwrap();
        assert_eq!(result.blipped_down_y.len(), n);
    }

    #[test]
    fn constant_additive_single_time_point() {
        // k=1 is the simplest case — equivalent to IPW-Wald in balanced data.
        let n = 50;
        let mut h = CausalHandle::new(80, 7777);
        let x = vec![1.0_f64; n];
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let true_gamma = 1.0_f64;
        let y: Vec<f64> = (0..n)
            .map(|i| true_gamma * a[i] + 0.1 * h.rng.next_normal() as f64)
            .collect();
        let cfg = SequentialGConfig {
            propensity_max_iter: 400,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        assert_eq!(result.gamma.len(), 1);
        assert!(
            (result.gamma[0] - true_gamma).abs() < 0.5,
            "gamma={} not close enough to {}",
            result.gamma[0],
            true_gamma
        );
    }

    #[test]
    fn linear_modifier_returns_two_params() {
        let n = 30;
        let p = 2;
        let k = 1;
        let mut h = make_handle();
        let x: Vec<f64> = (0..n * p).map(|i| (i as f64) / (n * p) as f64).collect();
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let v = x[i * p];
                (1.5 + 0.5 * v) * a[i] + 0.1 * (i as f64).sin()
            })
            .collect();
        let cfg = SequentialGConfig {
            blip_function: BlipFunction::LinearModifier { col: 0 },
            propensity_max_iter: 300,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, k, p, &cfg, &mut h).unwrap();
        assert_eq!(result.gamma.len(), 2);
    }

    #[test]
    fn bootstrap_reps_produces_nonneg_se() {
        let n = 40;
        let mut h = make_handle();
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let x = vec![1.0_f64; n];
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * a[i] + 0.1 * (i as f64).sin())
            .collect();
        let cfg = SequentialGConfig {
            bootstrap_reps: 10,
            propensity_max_iter: 100,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        for &se in &result.se_gamma {
            assert!(se >= 0.0, "se must be non-negative, got {se}");
        }
    }

    #[test]
    fn zero_bootstrap_gives_zero_se() {
        let n = 20;
        let mut h = make_handle();
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let x = vec![1.0_f64; n];
        let y: Vec<f64> = (0..n).map(|i| 0.5 * a[i]).collect();
        let cfg = SequentialGConfig::default(); // bootstrap_reps = 0
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        for &se in &result.se_gamma {
            assert_eq!(se, 0.0);
        }
    }

    #[test]
    fn deterministic_same_seed_same_gamma() {
        let n = 40;
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let x = vec![1.0_f64; n];
        let y: Vec<f64> = (0..n).map(|i| 0.5 * a[i] + 0.1).collect();
        let cfg = SequentialGConfig {
            propensity_max_iter: 100,
            ..Default::default()
        };

        let mut h1 = CausalHandle::new(80, 42);
        let mut h2 = CausalHandle::new(80, 42);

        let r1 = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h1).unwrap();
        let r2 = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h2).unwrap();

        assert_eq!(r1.gamma, r2.gamma);
    }

    #[test]
    fn multi_time_point_fit_succeeds() {
        let n = 30;
        let k = 4;
        let p = 2;
        let mut h = make_handle();
        let a: Vec<f64> = (0..n * k)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let x: Vec<f64> = (0..n * p).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| 0.3 * (i as f64) * 0.1).collect();
        let cfg = SequentialGConfig {
            propensity_max_iter: 100,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, k, p, &cfg, &mut h).unwrap();
        assert_eq!(result.propensity_matrix.len(), n * k);
        assert_eq!(result.blipped_down_y.len(), n);
        assert_eq!(result.gamma.len(), 1);
    }

    #[test]
    fn blipped_down_equals_y_when_gamma_zero() {
        // When A is all zero, the g-estimating equation denominator is near zero.
        // Use mixed A so denominator is finite, but check structure.
        let n = 20;
        let mut h = make_handle();
        // Alternating treatment, zero outcome → γ should be near zero, blipped ≈ y.
        let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let x = vec![1.0_f64; n];
        let y = vec![0.0_f64; n];
        let cfg = SequentialGConfig {
            propensity_max_iter: 100,
            ..Default::default()
        };
        let result = SequentialGEstimator::fit(&y, &a, &x, n, 1, 1, &cfg, &mut h).unwrap();
        // With y=0, num=0, so gamma=0, blipped_down = y = 0.
        for &bd in &result.blipped_down_y {
            assert!(
                bd.abs() < 1e-10,
                "blipped_down_y should be 0 when y=0, got {bd}"
            );
        }
    }
}
