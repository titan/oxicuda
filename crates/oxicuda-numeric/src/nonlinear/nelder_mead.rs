//! Nelder-Mead downhill-simplex minimiser (Nelder & Mead, 1965).
//!
//! A derivative-free direct-search method for an unconstrained scalar objective
//! `φ : ℝⁿ → ℝ`.  It maintains a simplex of `n + 1` vertices and at each
//! iteration reflects, expands, contracts, or shrinks the simplex away from the
//! worst vertex.  No gradient is required, which makes the method robust on
//! noisy, non-smooth, or black-box objectives where the BFGS/L-BFGS family
//! cannot be applied.
//!
//! Per-iteration logic (standard coefficients `α = 1, γ = 2, ρ = 1/2, σ = 1/2`):
//!
//! 1. Order vertices so `f₀ ≤ … ≤ fₙ`.
//! 2. Compute the centroid `x̄` of the best `n` vertices.
//! 3. **Reflect**: `x_r = x̄ + α (x̄ − xₙ)`.
//!    - If `f₀ ≤ f_r < f_{n−1}` accept `x_r`.
//! 4. **Expand**: if `f_r < f₀`, try `x_e = x̄ + γ (x_r − x̄)`; keep the better
//!    of `x_e` and `x_r`.
//! 5. **Contract**: if `f_r ≥ f_{n−1}`, try `x_c = x̄ + ρ (xₙ − x̄)` (inside) or
//!    `x̄ + ρ (x_r − x̄)` (outside); accept if it improves on `xₙ`/`x_r`.
//! 6. **Shrink**: otherwise pull every non-best vertex toward `x₀`.
//!
//! Convergence is declared when the simplex collapses, measured by the spread
//! of function values `max f − min f` and the geometric size of the simplex.

use crate::error::{NumericError, NumericResult};

/// Configuration for [`nelder_mead`].
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations (simplex updates).
    pub max_iter: usize,
    /// Convergence tolerance on the spread of vertex function values.
    pub ftol: f64,
    /// Convergence tolerance on the geometric size of the simplex.
    pub xtol: f64,
    /// Reflection coefficient `α > 0` (typically `1`).
    pub alpha: f64,
    /// Expansion coefficient `γ > 1` (typically `2`).
    pub gamma: f64,
    /// Contraction coefficient `0 < ρ < 1` (typically `1/2`).
    pub rho: f64,
    /// Shrink coefficient `0 < σ < 1` (typically `1/2`).
    pub sigma: f64,
    /// Initial step used to build the simplex around `x0` (non-zero coordinates).
    pub initial_step: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iter: 2000,
            ftol: 1.0e-10,
            xtol: 1.0e-10,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            initial_step: 0.05,
        }
    }
}

/// Result of a Nelder-Mead run.
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    /// Best vertex (minimiser estimate).
    pub x: Vec<f64>,
    /// Objective value at the best vertex.
    pub fx: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// Spread `max f − min f` over the final simplex.
    pub f_spread: f64,
}

/// Minimises `phi` from a starting point `x0` using the Nelder-Mead simplex.
///
/// The initial simplex is constructed by perturbing each coordinate of `x0`
/// independently by `cfg.initial_step` (scaled by the coordinate magnitude so
/// that zero coordinates still move).
///
/// # Errors
///
/// * [`NumericError::EmptyInput`] if `x0` is empty.
/// * [`NumericError::InvalidParameter`] for invalid coefficients, non-finite
///   `x0`, or a non-finite objective value at `x0`.
/// * [`NumericError::NotConverged`] if the simplex has not collapsed within the
///   tolerances after `max_iter` iterations.
pub fn nelder_mead<P>(phi: P, x0: &[f64], cfg: &NelderMeadConfig) -> NumericResult<NelderMeadResult>
where
    P: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    validate_config(cfg)?;
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(NumericError::InvalidParameter(
            "x0 has non-finite entries".into(),
        ));
    }

    let eval = |x: &[f64]| -> NumericResult<f64> {
        let f = phi(x);
        if f.is_finite() {
            Ok(f)
        } else {
            Err(NumericError::NumericalInstability(
                "objective became non-finite".into(),
            ))
        }
    };

    // Build the initial simplex: x0 plus n perturbed copies.
    let mut verts: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    verts.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        let step = cfg.initial_step * (1.0 + x0[i].abs());
        v[i] += step;
        verts.push(v);
    }
    let mut fvals: Vec<f64> = Vec::with_capacity(n + 1);
    for v in &verts {
        fvals.push(eval(v)?);
    }

    let mut iters = 0_usize;
    let mut order: Vec<usize> = (0..=n).collect();

    for it in 0..cfg.max_iter {
        iters = it;
        // Order vertices ascending by function value.
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let f_spread = fvals[worst] - fvals[best];
        let size = simplex_size(&verts, best, n);
        if f_spread <= cfg.ftol && size <= cfg.xtol {
            return Ok(NelderMeadResult {
                x: verts[best].clone(),
                fx: fvals[best],
                iters: it,
                f_spread,
            });
        }

        // Centroid of all but the worst vertex.
        let mut centroid = vec![0.0_f64; n];
        for (&idx, _) in order.iter().take(n).zip(0..n) {
            for (c, &xv) in centroid.iter_mut().zip(&verts[idx]) {
                *c += xv;
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection.
        let x_ref = combine(&centroid, &centroid, &verts[worst], cfg.alpha);
        let f_ref = eval(&x_ref)?;

        if f_ref < fvals[best] {
            // Expansion.
            let x_exp = combine(&centroid, &x_ref, &centroid, cfg.gamma);
            let f_exp = eval(&x_exp)?;
            if f_exp < f_ref {
                verts[worst] = x_exp;
                fvals[worst] = f_exp;
            } else {
                verts[worst] = x_ref;
                fvals[worst] = f_ref;
            }
        } else if f_ref < fvals[second_worst] {
            // Accept reflection.
            verts[worst] = x_ref;
            fvals[worst] = f_ref;
        } else {
            // Contraction.
            let accepted = if f_ref < fvals[worst] {
                // Outside contraction toward the reflected point.
                let x_oc = combine(&centroid, &x_ref, &centroid, cfg.rho);
                let f_oc = eval(&x_oc)?;
                if f_oc <= f_ref {
                    verts[worst] = x_oc;
                    fvals[worst] = f_oc;
                    true
                } else {
                    false
                }
            } else {
                // Inside contraction toward the worst vertex.
                let x_ic = combine(&centroid, &verts[worst], &centroid, cfg.rho);
                let f_ic = eval(&x_ic)?;
                if f_ic < fvals[worst] {
                    verts[worst] = x_ic;
                    fvals[worst] = f_ic;
                    true
                } else {
                    false
                }
            };
            if !accepted {
                // Shrink toward the best vertex.
                let xb = verts[best].clone();
                for &idx in order.iter().skip(1) {
                    for (xj, &bj) in verts[idx].iter_mut().zip(&xb) {
                        *xj = bj + cfg.sigma * (*xj - bj);
                    }
                    fvals[idx] = eval(&verts[idx])?;
                }
            }
        }
    }

    // Final ordering for the best vertex on exit.
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = order[0];
    let worst = order[n];
    let f_spread = fvals[worst] - fvals[best];
    let size = simplex_size(&verts, best, n);
    if f_spread <= cfg.ftol && size <= cfg.xtol {
        Ok(NelderMeadResult {
            x: verts[best].clone(),
            fx: fvals[best],
            iters,
            f_spread,
        })
    } else {
        Err(NumericError::NotConverged {
            iter: cfg.max_iter,
            residual: f_spread,
        })
    }
}

fn validate_config(cfg: &NelderMeadConfig) -> NumericResult<()> {
    if cfg.alpha <= 0.0 {
        return Err(NumericError::InvalidParameter(format!(
            "alpha must be positive, got {}",
            cfg.alpha
        )));
    }
    if cfg.gamma <= 1.0 {
        return Err(NumericError::InvalidParameter(format!(
            "gamma must exceed 1, got {}",
            cfg.gamma
        )));
    }
    if !(cfg.rho > 0.0 && cfg.rho < 1.0) {
        return Err(NumericError::InvalidParameter(format!(
            "rho must lie in (0,1), got {}",
            cfg.rho
        )));
    }
    if !(cfg.sigma > 0.0 && cfg.sigma < 1.0) {
        return Err(NumericError::InvalidParameter(format!(
            "sigma must lie in (0,1), got {}",
            cfg.sigma
        )));
    }
    if !(cfg.initial_step.is_finite() && cfg.initial_step > 0.0) {
        return Err(NumericError::InvalidParameter(format!(
            "initial_step must be positive finite, got {}",
            cfg.initial_step
        )));
    }
    if !(cfg.ftol >= 0.0 && cfg.xtol >= 0.0) {
        return Err(NumericError::InvalidParameter(
            "tolerances must be non-negative".into(),
        ));
    }
    Ok(())
}

/// Compute `base + coeff * (a − b)` componentwise.
fn combine(base: &[f64], a: &[f64], b: &[f64], coeff: f64) -> Vec<f64> {
    base.iter()
        .zip(a)
        .zip(b)
        .map(|((&bb, &av), &bv)| bb + coeff * (av - bv))
        .collect()
}

/// Largest L∞ distance from the best vertex to any other vertex.
fn simplex_size(verts: &[Vec<f64>], best: usize, n: usize) -> f64 {
    let mut size = 0.0_f64;
    for v in verts {
        let mut d = 0.0_f64;
        for k in 0..n {
            d = d.max((v[k] - verts[best][k]).abs());
        }
        size = size.max(d);
    }
    size
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NelderMeadConfig {
        NelderMeadConfig::default()
    }

    #[test]
    fn quadratic_bowl() {
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let r = nelder_mead(phi, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 3.0).abs() < 1e-4, "x={}", r.x[0]);
        assert!((r.x[1] + 1.0).abs() < 1e-4, "y={}", r.x[1]);
        assert!(r.fx < 1e-7);
    }

    #[test]
    fn rosenbrock() {
        let phi = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        let c = NelderMeadConfig {
            max_iter: 5000,
            ftol: 1e-12,
            xtol: 1e-10,
            ..cfg()
        };
        let r = nelder_mead(phi, &[-1.2, 1.0], &c).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x={}", r.x[0]);
        assert!((r.x[1] - 1.0).abs() < 1e-3, "y={}", r.x[1]);
    }

    #[test]
    fn one_dimensional() {
        // Minimise (x − 2)^4 + 1; derivative-free should still locate x = 2.
        let phi = |v: &[f64]| (v[0] - 2.0).powi(4) + 1.0;
        let r = nelder_mead(phi, &[0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 2.0).abs() < 1e-2, "x={}", r.x[0]);
        assert!((r.fx - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_smooth_objective() {
        // L1-style objective with a kink at the minimum: |x − 1| + |y + 2|.
        // Gradient methods struggle here; the simplex copes.
        let phi = |v: &[f64]| (v[0] - 1.0).abs() + (v[1] + 2.0).abs();
        let c = NelderMeadConfig {
            max_iter: 4000,
            ftol: 1e-9,
            xtol: 1e-9,
            ..cfg()
        };
        let r = nelder_mead(phi, &[5.0, 5.0], &c).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x={}", r.x[0]);
        assert!((r.x[1] + 2.0).abs() < 1e-3, "y={}", r.x[1]);
    }

    #[test]
    fn already_near_minimum() {
        let phi = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
        let r = nelder_mead(phi, &[0.0, 0.0], &cfg()).expect("ok");
        assert!(r.fx < 1e-6);
        assert!(r.x[0].abs() < 1e-3 && r.x[1].abs() < 1e-3);
    }

    #[test]
    fn three_dimensional() {
        let phi = |v: &[f64]| (v[0] - 1.0).powi(2) + (v[1] - 2.0).powi(2) + (v[2] - 3.0).powi(2);
        let r = nelder_mead(phi, &[0.0, 0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3);
        assert!((r.x[1] - 2.0).abs() < 1e-3);
        assert!((r.x[2] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn decreases_objective() {
        let phi = |v: &[f64]| (v[0] + 4.0).powi(2) + 7.0 * (v[1] - 6.0).powi(2);
        let start = [0.0, 0.0];
        let f0 = phi(&start);
        let r = nelder_mead(phi, &start, &cfg()).expect("ok");
        assert!(r.fx < f0);
    }

    #[test]
    fn spread_collapses() {
        let phi = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let r = nelder_mead(phi, &[1.0, 1.0, 1.0], &cfg()).expect("ok");
        assert!(r.f_spread <= cfg().ftol, "spread={}", r.f_spread);
    }

    #[test]
    fn output_finite_and_sized() {
        let phi = |v: &[f64]| v.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>();
        let r = nelder_mead(phi, &[0.0; 4], &cfg()).expect("ok");
        assert_eq!(r.x.len(), 4);
        for v in &r.x {
            assert!(v.is_finite());
        }
        assert!(r.fx.is_finite());
    }

    #[test]
    fn max_iter_bound() {
        let phi = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        let c = NelderMeadConfig {
            max_iter: 2,
            ftol: 1e-14,
            xtol: 1e-14,
            ..cfg()
        };
        let res = nelder_mead(phi, &[-3.0, 4.0], &c);
        assert!(res.is_err());
    }

    #[test]
    fn rejects_bad_input() {
        let phi = |v: &[f64]| v[0] * v[0];
        assert!(nelder_mead(phi, &[], &cfg()).is_err());
        let bad = NelderMeadConfig { rho: 1.5, ..cfg() };
        assert!(nelder_mead(|v: &[f64]| v[0] * v[0], &[1.0], &bad).is_err());
        let bad_step = NelderMeadConfig {
            initial_step: 0.0,
            ..cfg()
        };
        assert!(nelder_mead(|v: &[f64]| v[0] * v[0], &[1.0], &bad_step).is_err());
    }

    #[test]
    fn handles_scaled_coordinates() {
        // Minimum far from the origin where coordinate magnitudes differ widely.
        let phi = |v: &[f64]| (v[0] - 1000.0).powi(2) + (v[1] - 0.001).powi(2);
        let c = NelderMeadConfig {
            max_iter: 4000,
            ..cfg()
        };
        let r = nelder_mead(phi, &[0.0, 0.0], &c).expect("ok");
        assert!((r.x[0] - 1000.0).abs() < 1.0, "x={}", r.x[0]);
        assert!((r.x[1] - 0.001).abs() < 1e-2, "y={}", r.x[1]);
    }
}
