//! Durand-Kerner (Weierstrass) method — simultaneous all-roots iteration.
//!
//! For monic `p(z) = z^n + c_{n-1} z^{n-1} + … + c_0`:
//! `z_i ← z_i - p(z_i) / Π_{j ≠ i} (z_i - z_j)`.

use crate::error::{NumericError, NumericResult};
use crate::root::aberth_all_roots::Complex64;

fn horner_c(coeffs: &[f64], z: Complex64) -> Complex64 {
    let n = coeffs.len();
    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mut acc = Complex64::new(coeffs[n - 1], 0.0);
    for i in (0..(n - 1)).rev() {
        acc = acc.mul(z).add(Complex64::new(coeffs[i], 0.0));
    }
    acc
}

/// Durand-Kerner all-roots solver. Polynomial indexed by power.
pub fn durand_kerner(coeffs: &[f64], tol: f64, max_iter: usize) -> NumericResult<Vec<Complex64>> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let an = coeffs[coeffs.len() - 1];
    if an.abs() < 1.0e-300 {
        return Err(NumericError::InvalidParameter(
            "leading coefficient is zero".into(),
        ));
    }
    let n = coeffs.len() - 1;
    if n == 0 {
        return Ok(vec![]);
    }
    // Normalize to monic
    let normalized: Vec<f64> = coeffs.iter().map(|&c| c / an).collect();
    // Initial guesses on a circle of radius 1 in the complex plane, slightly rotated.
    let mut z: Vec<Complex64> = (0..n)
        .map(|k| {
            let r = 1.0_f64;
            let theta = std::f64::consts::TAU * (k as f64) / (n as f64) + 0.4;
            Complex64::from_polar(r, theta)
        })
        .collect();
    for _ in 0..max_iter {
        let mut max_step = 0.0_f64;
        let mut new_z = z.clone();
        for i in 0..n {
            let mut denom = Complex64::new(1.0, 0.0);
            for (j, zj) in z.iter().enumerate() {
                if j == i {
                    continue;
                }
                let diff = z[i].sub(*zj);
                if diff.abs() < 1.0e-300 {
                    return Err(NumericError::NumericalInstability(
                        "two iterates collided in Durand-Kerner".into(),
                    ));
                }
                denom = denom.mul(diff);
            }
            let p_val = horner_c(&normalized, z[i]);
            let step = p_val.div(denom);
            new_z[i] = z[i].sub(step);
            let mag = step.abs();
            if mag > max_step {
                max_step = mag;
            }
        }
        z = new_z;
        if max_step < tol {
            return Ok(z);
        }
    }
    let resid: f64 = z.iter().map(|zi| horner_c(&normalized, *zi).abs()).sum();
    if resid < tol * 100.0 {
        Ok(z)
    } else {
        Err(NumericError::NotConverged {
            iter: max_iter,
            residual: resid,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dk_quadratic() {
        // x² - 1 = 0 → ±1
        let coeffs = vec![-1.0_f64, 0.0, 1.0];
        let roots = durand_kerner(&coeffs, 1.0e-10, 200).expect("ok");
        let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!((reals[0] + 1.0).abs() < 1.0e-6);
        assert!((reals[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn dk_cubic_three_real() {
        let coeffs = vec![-6.0_f64, 11.0, -6.0, 1.0];
        let roots = durand_kerner(&coeffs, 1.0e-10, 400).expect("ok");
        let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!((reals[0] - 1.0).abs() < 1.0e-4);
        assert!((reals[1] - 2.0).abs() < 1.0e-4);
        assert!((reals[2] - 3.0).abs() < 1.0e-4);
    }
}
