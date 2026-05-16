//! Poincaré ball model of hyperbolic space.
//!
//! The unit ball `B^d = {x in R^d : ||x|| < 1}` with Riemannian metric `g_x = (2 / (1 - ||x||^2))^2 * I_d`.
//! Distance: `d(u, v) = arcosh(1 + 2 ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))`.

use crate::error::{ManifoldError, ManifoldResult};

/// Hyperbolic distance in the Poincaré ball.
pub fn poincare_distance(u: &[f64], v: &[f64]) -> ManifoldResult<f64> {
    if u.len() != v.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: u.len(),
            b: v.len(),
        });
    }
    let u_n2 = u.iter().map(|x| x * x).sum::<f64>();
    let v_n2 = v.iter().map(|x| x * x).sum::<f64>();
    if u_n2 >= 1.0 - 1e-15 || v_n2 >= 1.0 - 1e-15 {
        return Err(ManifoldError::ManifoldConstraint(
            "poincare: point on or outside unit ball".into(),
        ));
    }
    let diff_n2 = u.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    let arg = 1.0 + 2.0 * diff_n2 / ((1.0 - u_n2) * (1.0 - v_n2));
    Ok(acosh(arg))
}

fn acosh(x: f64) -> f64 {
    (x + (x * x - 1.0).max(0.0).sqrt()).ln()
}

/// Möbius addition in the Poincaré ball:
/// `x oplus y = ((1 + 2 <x,y> + ||y||^2) x + (1 - ||x||^2) y) / (1 + 2<x,y> + ||x||^2 ||y||^2)`.
pub fn mobius_add(x: &[f64], y: &[f64]) -> ManifoldResult<Vec<f64>> {
    if x.len() != y.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let xn2: f64 = x.iter().map(|v| v * v).sum();
    let yn2: f64 = y.iter().map(|v| v * v).sum();
    let xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let denom = 1.0 + 2.0 * xy + xn2 * yn2;
    let denom = if denom.abs() < 1e-15 { 1e-15 } else { denom };
    let alpha = 1.0 + 2.0 * xy + yn2;
    let beta = 1.0 - xn2;
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() {
        out[i] = (alpha * x[i] + beta * y[i]) / denom;
    }
    Ok(out)
}

/// Project a point into the Poincaré ball with margin `epsilon`.
pub fn poincare_project(x: &[f64], epsilon: f64) -> Vec<f64> {
    let n2: f64 = x.iter().map(|v| v * v).sum();
    let n = n2.sqrt();
    let cap = 1.0 - epsilon;
    if n > cap {
        x.iter().map(|v| v * (cap / n)).collect()
    } else {
        x.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_to_self_zero() {
        let u = vec![0.1, 0.2];
        let d = poincare_distance(&u, &u).expect("ok");
        assert!(d.abs() < 1e-9);
    }

    #[test]
    fn distance_symmetric() {
        let u = vec![0.1, 0.2];
        let v = vec![-0.3, 0.05];
        let d1 = poincare_distance(&u, &v).expect("ok");
        let d2 = poincare_distance(&v, &u).expect("ok");
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn mobius_add_with_zero_identity() {
        let u = vec![0.2, 0.3];
        let z = vec![0.0; 2];
        let s = mobius_add(&u, &z).expect("ok");
        for (a, b) in u.iter().zip(&s) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn projection_inside_ball() {
        let x = vec![10.0, 0.0];
        let p = poincare_project(&x, 0.01);
        let n: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(n < 1.0);
    }
}
