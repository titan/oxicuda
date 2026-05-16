//! Local element matrices for a P1 (linear Lagrange) triangle.

use crate::error::{PdeError, PdeResult};

/// Local stiffness matrix `K_e` of size 3x3 for a P1 triangle with vertices
/// `(x0,y0)`, `(x1,y1)`, `(x2,y2)`.
///
/// Formula: `K_e = (1/(4*Area)) * B^T * B` where rows of B are `[b_i, c_i]`,
/// with `b_i = y_{i+1} - y_{i+2}`, `c_i = x_{i+2} - x_{i+1}` (cyclic).
pub fn p1_local_stiffness(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> PdeResult<[f64; 9]> {
    let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if area.abs() < 1.0e-15 {
        return Err(PdeError::SingularMatrix(format!(
            "degenerate triangle area={area}"
        )));
    }
    let b0 = y1 - y2;
    let b1 = y2 - y0;
    let b2 = y0 - y1;
    let c0 = x2 - x1;
    let c1 = x0 - x2;
    let c2 = x1 - x0;
    let b = [b0, b1, b2];
    let c = [c0, c1, c2];
    let inv = 1.0 / (4.0 * area);
    let mut k = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            k[i * 3 + j] = inv * (b[i] * b[j] + c[i] * c[j]);
        }
    }
    Ok(k)
}

/// Local mass matrix `M_e` of size 3x3 for a P1 triangle.
///
/// Formula: `M_e = (Area/12) * [[2,1,1],[1,2,1],[1,1,2]]`.
pub fn p1_local_mass(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> PdeResult<[f64; 9]> {
    let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if area.abs() < 1.0e-15 {
        return Err(PdeError::SingularMatrix(format!(
            "degenerate triangle area={area}"
        )));
    }
    let s = area / 12.0;
    Ok([2.0 * s, s, s, s, 2.0 * s, s, s, s, 2.0 * s])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_stiffness_reference_triangle() {
        // Reference triangle (0,0), (1,0), (0,1) -> known stiffness matrix:
        // K = 1/2 * [[2,-1,-1],[-1,1,0],[-1,0,1]]
        let k = p1_local_stiffness(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        let expected = [1.0, -0.5, -0.5, -0.5, 0.5, 0.0, -0.5, 0.0, 0.5];
        for i in 0..9 {
            assert!(
                (k[i] - expected[i]).abs() < 1.0e-12,
                "k[{i}]={} != {}",
                k[i],
                expected[i]
            );
        }
    }

    #[test]
    fn local_mass_reference_triangle() {
        // Reference triangle area = 1/2, so M = 1/24 * [[2,1,1],[1,2,1],[1,1,2]]
        let m = p1_local_mass(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        let one24 = 1.0 / 24.0;
        let expected = [
            2.0 * one24,
            one24,
            one24,
            one24,
            2.0 * one24,
            one24,
            one24,
            one24,
            2.0 * one24,
        ];
        for i in 0..9 {
            assert!((m[i] - expected[i]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn degenerate_triangle_errors() {
        let res = p1_local_stiffness(0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
        assert!(res.is_err());
    }

    #[test]
    fn stiffness_row_sums_zero() {
        // The P1 stiffness K_e has rows summing to zero (linear functions are in kernel).
        let k = p1_local_stiffness(0.1, 0.2, 1.5, 0.3, 0.4, 1.2).expect("ok");
        for i in 0..3 {
            let s: f64 = (0..3).map(|j| k[i * 3 + j]).sum();
            assert!(s.abs() < 1.0e-12);
        }
    }
}
