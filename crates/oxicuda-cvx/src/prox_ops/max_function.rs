//! Proximal operator of the max function `f(x) = max_i x_i`.
//!
//! The max function is the support function of the unit probability simplex
//! `Δ = {p : p ≥ 0, Σ p_i = 1}`, i.e. `max_i x_i = sup_{p ∈ Δ} ⟨p, x⟩`.  Its
//! convex conjugate is therefore the indicator `ι_Δ`, and the (extended) Moreau
//! decomposition gives a closed form in terms of the Euclidean projection onto
//! the simplex `P_Δ`:
//!
//! ```text
//!   prox_{λ·max}(v) = v − λ · P_Δ(v / λ),     λ > 0.
//! ```
//!
//! This reuses the crate's `project_simplex` operator.  For `λ = 0` the max term
//! vanishes and the prox is the identity.
//!
//! # Optimality
//!
//! `prox_{λ·max}(v) = argmin_x { ½‖x − v‖² + λ·max_i x_i }`.  The optimality
//! condition is `0 ∈ (x − v) + λ ∂max(x)` where `∂max(x)` is the convex hull of
//! `{e_i : i ∈ argmax x} ⊆ Δ`.  Writing `x = v − λ s` with `s = P_Δ(v/λ)`
//! recovers exactly that subgradient inclusion.

use crate::error::{CvxError, CvxResult};
use crate::projection::project_simplex;

/// Evaluate `f(x) = max_i x_i`.
///
/// Returns `Err(EmptyInput)` for an empty slice (the max is undefined).
pub fn max_value(x: &[f64]) -> CvxResult<f64> {
    if x.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let mut m = f64::NEG_INFINITY;
    for &v in x {
        if v > m {
            m = v;
        }
    }
    Ok(m)
}

/// Proximal operator of `λ · max(·)`.
///
/// Computes `prox_{λ·max}(v) = v − λ · P_Δ(v / λ)` for `λ > 0`, and the identity
/// for `λ = 0`.
///
/// # Parameters
/// * `v`      – input vector (length ≥ 1).
/// * `lambda` – non-negative scaling of the max function.
///
/// # Errors
/// * [`CvxError::EmptyInput`] if `v` is empty.
/// * [`CvxError::InvalidParameter`] if `lambda` is negative or non-finite.
pub fn prox_max(v: &[f64], lambda: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "prox_max requires lambda ≥ 0, got {lambda}"
        )));
    }
    if lambda == 0.0 {
        return Ok(v.to_vec());
    }
    // Scaled input v / λ projected onto the unit simplex.
    let scaled: Vec<f64> = v.iter().map(|vi| vi / lambda).collect();
    let proj = project_simplex(&scaled, 1.0)?;
    // prox = v − λ · P_Δ(v/λ).
    Ok(v.iter()
        .zip(proj.iter())
        .map(|(vi, pi)| vi - lambda * pi)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force the prox objective `½‖x−v‖² + λ·max(x)` along a few
    /// directions to confirm the analytic prox is a (local ⇒ global, convex)
    /// minimiser.
    fn prox_objective(x: &[f64], v: &[f64], lambda: f64) -> f64 {
        let half_sq: f64 = x
            .iter()
            .zip(v.iter())
            .map(|(xi, vi)| 0.5 * (xi - vi).powi(2))
            .sum();
        half_sq + lambda * x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn prox_max_analytic_two_d() {
        // v = [3, 1], λ = 1.  P_Δ([3,1]) = [1, 0] ⇒ prox = [3,1] − [1,0] = [2,1].
        let p = prox_max(&[3.0, 1.0], 1.0).expect("ok");
        assert!((p[0] - 2.0).abs() < 1e-12, "p0={}", p[0]);
        assert!((p[1] - 1.0).abs() < 1e-12, "p1={}", p[1]);
    }

    #[test]
    fn prox_max_optimality_subgradient() {
        // At the prox point, v − x must be a valid subgradient of max (∈ Δ,
        // supported on argmax x).
        let v = [3.0, 1.0];
        let lambda = 1.0;
        let x = prox_max(&v, lambda).expect("ok");
        let s: Vec<f64> = v
            .iter()
            .zip(x.iter())
            .map(|(vi, xi)| (vi - xi) / lambda)
            .collect();
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "subgradient sum {sum} ≠ 1");
        assert!(s.iter().all(|&si| si >= -1e-12), "subgradient not ≥ 0");
        // Support: x[0]=2 is the unique max, so s must be concentrated on idx 0.
        assert!((s[0] - 1.0).abs() < 1e-10 && s[1].abs() < 1e-10);
    }

    #[test]
    fn prox_max_beats_neighbours() {
        let v = [3.0, 1.0, -0.5];
        let lambda = 0.7;
        let x = prox_max(&v, lambda).expect("ok");
        let f_star = prox_objective(&x, &v, lambda);
        // Perturb each coordinate both directions; objective must not improve.
        for i in 0..x.len() {
            for &eps in &[0.05_f64, -0.05] {
                let mut y = x.clone();
                y[i] += eps;
                assert!(
                    prox_objective(&y, &v, lambda) >= f_star - 1e-9,
                    "perturbation improved objective at coord {i}"
                );
            }
        }
    }

    #[test]
    fn prox_max_already_in_simplex_shifts_to_origin() {
        // v in the simplex (sums to 1) ⇒ P_Δ(v) = v ⇒ prox = 0.
        let v = [0.2, 0.3, 0.5];
        let p = prox_max(&v, 1.0).expect("ok");
        for pi in p {
            assert!(pi.abs() < 1e-10);
        }
    }

    #[test]
    fn prox_max_lambda_zero_is_identity() {
        let v = [1.0, -2.0, 3.5];
        let p = prox_max(&v, 0.0).expect("ok");
        assert_eq!(p, v.to_vec());
    }

    #[test]
    fn prox_max_negative_lambda_errors() {
        assert!(matches!(
            prox_max(&[1.0], -0.5),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn prox_max_empty_errors() {
        assert!(matches!(prox_max(&[], 1.0), Err(CvxError::EmptyInput)));
        assert!(matches!(max_value(&[]), Err(CvxError::EmptyInput)));
    }

    #[test]
    fn max_value_basic() {
        assert!((max_value(&[1.0, 5.0, 3.0]).expect("ok") - 5.0).abs() < 1e-12);
    }
}
