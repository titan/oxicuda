//! Tanh–sinh (double-exponential) quadrature.
//!
//! Takahasi & Mori 1974. The change of variables
//! `x = tanh(½ π sinh(t))` maps the finite interval `(−1, 1)` onto
//! `t ∈ (−∞, ∞)` and makes the integrand decay double-exponentially toward the
//! endpoints. Applying the trapezoidal rule in `t` then converges extremely
//! fast — even for integrands with integrable endpoint singularities (e.g.
//! `1/√(1 − x²)` or `ln x`), which defeat Gauss–Legendre.
//!
//! For a general interval `[a, b]` the affine map `x = ½(b + a) + ½(b − a) u`
//! with `u ∈ [−1, 1]` is applied first. The rule refines the step size `h` by
//! repeated halving (Romberg-style level doubling) until successive estimates
//! agree to the requested tolerance.

use crate::error::{NumericError, NumericResult};

/// Configuration for the tanh–sinh integrator.
#[derive(Debug, Clone)]
pub struct TanhSinhConfig {
    /// Absolute convergence tolerance on successive levels.
    pub tol: f64,
    /// Maximum number of level doublings (each halves the step `h`).
    pub max_levels: usize,
}

impl Default for TanhSinhConfig {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_levels: 12,
        }
    }
}

/// Result of a tanh–sinh integration.
#[derive(Debug, Clone)]
pub struct TanhSinhResult {
    /// Estimated value of the integral.
    pub value: f64,
    /// Estimated absolute error (difference between the last two levels).
    pub error: f64,
    /// Number of refinement levels actually used.
    pub levels: usize,
}

/// Integrate `f` over `[a, b]` with the tanh–sinh rule.
///
/// The integrand is evaluated strictly inside `(a, b)`, so integrable endpoint
/// singularities are handled without sampling the singular endpoints
/// themselves.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] if `a == b` or the bounds are
/// non-finite, or if `max_levels == 0`.
pub fn tanh_sinh_integrate<F>(
    f: F,
    a: f64,
    b: f64,
    cfg: &TanhSinhConfig,
) -> NumericResult<TanhSinhResult>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || (a - b).abs() == 0.0 {
        return Err(NumericError::InvalidParameter(
            "tanh_sinh: [a, b] must be finite with a != b".to_string(),
        ));
    }
    if cfg.max_levels == 0 {
        return Err(NumericError::InvalidParameter(
            "tanh_sinh: max_levels must be >= 1".to_string(),
        ));
    }

    let half_span = 0.5 * (b - a);
    let mid = 0.5 * (b + a);
    let half_pi = 0.5 * std::f64::consts::PI;

    // Abscissa / weight for parameter `t` of the DE transform on (-1, 1):
    //   u(t)  = tanh(½ π sinh t)
    //   w(t)  = ½ π cosh t / cosh²(½ π sinh t)
    // mapped to [a, b] via x = mid + half_span * u and scaled by half_span.
    let eval_node = |t: f64, acc: &mut f64| {
        let sinh_t = t.sinh();
        let cosh_t = t.cosh();
        let psh = half_pi * sinh_t;
        let cosh_psh = psh.cosh();
        let u = psh.tanh();
        // Guard against the endpoints where 1 - u² underflows.
        if u.abs() >= 1.0 {
            return;
        }
        let weight = half_pi * cosh_t / (cosh_psh * cosh_psh);
        let x = mid + half_span * u;
        let fx = f(x);
        if fx.is_finite() {
            *acc += weight * fx;
        }
    };

    // The trapezoidal sum truncates `t` where the weight underflows. A limit of
    // 4.5 places the outermost abscissa within ~1e-16 of the endpoints, deep
    // enough to resolve integrable endpoint singularities such as 1/√x.
    let t_limit = 4.5_f64;

    // Level 0: step h0 with the central node plus symmetric wings.
    let mut h = 1.0_f64;
    let mut sum = 0.0_f64;
    eval_node(0.0, &mut sum); // t = 0 (central node)
    {
        let mut k = 1usize;
        loop {
            let t = k as f64 * h;
            if t > t_limit {
                break;
            }
            eval_node(t, &mut sum);
            eval_node(-t, &mut sum);
            k += 1;
        }
    }
    let mut prev = sum * h * half_span;
    let mut value = prev;
    let mut error = f64::INFINITY;
    let mut used = 1usize;

    // Refine: halve h, add only the new midpoints (odd multiples of the new h).
    for level in 1..cfg.max_levels {
        h *= 0.5;
        let mut new_sum = 0.0_f64;
        let mut k = 1usize;
        loop {
            let t = k as f64 * h;
            if t > t_limit {
                break;
            }
            eval_node(t, &mut new_sum);
            eval_node(-t, &mut new_sum);
            k += 2; // odd multiples only — even ones came from the coarser grid
        }
        // Combined sum at the finer step: previous nodes contribute at the new h
        // because halving the step preserves all prior abscissae.
        sum += new_sum;
        value = sum * h * half_span;
        error = (value - prev).abs();
        used = level + 1;
        if error <= cfg.tol * value.abs().max(1.0) {
            break;
        }
        prev = value;
    }

    Ok(TanhSinhResult {
        value,
        error,
        levels: used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
        tanh_sinh_integrate(f, a, b, &TanhSinhConfig::default())
            .expect("value should be present")
            .value
    }

    #[test]
    fn constant() {
        // ∫_0^2 3 dx = 6.
        let v = integrate(|_x| 3.0, 0.0, 2.0);
        assert!((v - 6.0).abs() < 1e-10, "v={v}");
    }

    #[test]
    fn polynomial() {
        // ∫_0^1 x² dx = 1/3.
        let v = integrate(|x| x * x, 0.0, 1.0);
        assert!((v - 1.0 / 3.0).abs() < 1e-10, "v={v}");
    }

    #[test]
    fn exponential() {
        // ∫_0^1 eˣ dx = e − 1.
        let v = integrate(|x: f64| x.exp(), 0.0, 1.0);
        assert!((v - (std::f64::consts::E - 1.0)).abs() < 1e-10, "v={v}");
    }

    #[test]
    fn endpoint_singularity_inverse_sqrt() {
        // ∫_0^1 1/√x dx = 2 — an integrable endpoint singularity. DE quadrature
        // recovers ~8 digits at this node density.
        let v = integrate(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0);
        assert!((v - 2.0).abs() < 1e-7, "v={v}");
    }

    #[test]
    fn endpoint_singularity_log() {
        // ∫_0^1 ln x dx = -1 (logarithmic endpoint singularity).
        let v = integrate(|x: f64| x.ln(), 0.0, 1.0);
        assert!((v - (-1.0)).abs() < 1e-7, "v={v}");
    }

    #[test]
    fn arcsine_density() {
        // ∫_{-1}^{1} 1/√(1 − x²) dx = π (singular at both endpoints).
        let v = integrate(|x: f64| 1.0 / (1.0 - x * x).sqrt(), -1.0, 1.0);
        assert!((v - PI).abs() < 1e-7, "v={v}");
    }

    #[test]
    fn sine_over_interval() {
        // ∫_0^π sin x dx = 2.
        let v = integrate(|x: f64| x.sin(), 0.0, PI);
        assert!((v - 2.0).abs() < 1e-10, "v={v}");
    }

    #[test]
    fn reversed_limits_sign() {
        // ∫_b^a = −∫_a^b.
        let fwd = integrate(|x| x * x, 0.0, 1.0);
        let rev = integrate(|x| x * x, 1.0, 0.0);
        assert!((fwd + rev).abs() < 1e-10, "fwd={fwd} rev={rev}");
    }

    #[test]
    fn degenerate_interval_error() {
        let res = tanh_sinh_integrate(|x| x, 1.0, 1.0, &TanhSinhConfig::default());
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    #[test]
    fn max_levels_zero_error() {
        let cfg = TanhSinhConfig {
            tol: 1e-9,
            max_levels: 0,
        };
        let res = tanh_sinh_integrate(|x| x, 0.0, 1.0, &cfg);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    #[test]
    fn error_estimate_finite() {
        let r = tanh_sinh_integrate(|x: f64| x.cos(), 0.0, 1.0, &TanhSinhConfig::default())
            .expect("value should be present");
        assert!(r.error.is_finite());
        assert!(r.levels >= 1);
    }
}
