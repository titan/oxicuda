//! Plain Monte Carlo integration over a `d`-dimensional axis-aligned box.

use crate::error::{NumericError, NumericResult};
use crate::handle::LcgRng;

/// Estimate ∫_box `f(x) dx` with `n_samples` independent samples drawn uniformly on
/// `[lo[i], hi[i]]`. Returns the integral estimate and the standard error.
pub fn monte_carlo_integrate<F>(
    f: F,
    lo: &[f64],
    hi: &[f64],
    n_samples: usize,
    rng: &mut LcgRng,
) -> NumericResult<(f64, f64)>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    if lo.len() != hi.len() {
        return Err(NumericError::DimensionMismatch {
            a: lo.len(),
            b: hi.len(),
        });
    }
    if lo.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if n_samples == 0 {
        return Err(NumericError::InvalidParameter(
            "n_samples must be ≥ 1".into(),
        ));
    }
    let d = lo.len();
    let mut vol = 1.0_f64;
    for i in 0..d {
        if hi[i] <= lo[i] {
            return Err(NumericError::InvalidParameter(
                "hi must be > lo for each dim".into(),
            ));
        }
        vol *= hi[i] - lo[i];
    }
    let mut x = vec![0.0_f64; d];
    let mut sum = 0.0_f64;
    let mut sq = 0.0_f64;
    for _ in 0..n_samples {
        for i in 0..d {
            x[i] = rng.next_range(lo[i], hi[i]);
        }
        let fv = f(&x)?;
        sum += fv;
        sq += fv * fv;
    }
    let mean = sum / n_samples as f64;
    let variance = (sq / n_samples as f64 - mean * mean).max(0.0);
    let stderr = (variance / n_samples as f64).sqrt() * vol;
    Ok((vol * mean, stderr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mc_constant_integrand() {
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(1.0) };
        let mut rng = LcgRng::new(0);
        let (v, _e) =
            monte_carlo_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 1000, &mut rng).expect("ok");
        assert!((v - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn mc_circle_area() {
        // ∫_box 1[x²+y²<1] dx dy = π for box [-1,1]²
        let f = |x: &[f64]| -> NumericResult<f64> {
            Ok(if x[0] * x[0] + x[1] * x[1] < 1.0 {
                1.0
            } else {
                0.0
            })
        };
        let mut rng = LcgRng::new(123);
        let (v, _e) =
            monte_carlo_integrate(f, &[-1.0, -1.0], &[1.0, 1.0], 50_000, &mut rng).expect("ok");
        assert!((v - std::f64::consts::PI).abs() < 0.1);
    }
}
