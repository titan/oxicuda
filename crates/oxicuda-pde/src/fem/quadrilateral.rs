//! Local element matrices for a Q1 (bilinear) quadrilateral.
//!
//! Standard isoparametric 4-node bilinear quadrilateral on the reference
//! square `[−1,1]²`.  Node ordering (counter-clockwise):
//!
//! - Node 0: (−1, −1)
//! - Node 1: ( 1, −1)
//! - Node 2: ( 1,  1)
//! - Node 3: (−1,  1)
//!
//! Bilinear shape functions on the reference square:
//!   N0(ξ,η) = ¼(1−ξ)(1−η)
//!   N1(ξ,η) = ¼(1+ξ)(1−η)
//!   N2(ξ,η) = ¼(1+ξ)(1+η)
//!   N3(ξ,η) = ¼(1−ξ)(1+η)
//!
//! Unlike the affine triangle, the bilinear map is generally *not* affine, so
//! the Jacobian varies over the element.  All integrals therefore use a
//! 2×2 Gauss–Legendre rule (4 points at `±1/√3`, weights `1`), which is exact
//! for the bilinear integrands of the mass/load and adequate for the
//! gradient products of a parallelogram and a good approximation otherwise.

use crate::error::{PdeError, PdeResult};

/// Number of DOFs per Q1 quadrilateral element.
pub const Q1_N_DOFS: usize = 4;

/// A 2×2 matrix stored as nested fixed-size arrays (row-major).
pub type Mat2 = [[f64; 2]; 2];

/// The result of [`q1_jacobian`]: the Jacobian matrix, its determinant, and
/// its inverse.
pub type JacobianResult = (Mat2, f64, Mat2);

// ── Shape functions ────────────────────────────────────────────────────────────

/// Evaluate all 4 Q1 bilinear shape functions at reference coordinates `(xi, eta)`.
///
/// Node order: `(-1,-1)`, `(1,-1)`, `(1,1)`, `(-1,1)`.
#[inline]
pub fn q1_shape_fn(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta), // N0
        0.25 * (1.0 + xi) * (1.0 - eta), // N1
        0.25 * (1.0 + xi) * (1.0 + eta), // N2
        0.25 * (1.0 - xi) * (1.0 + eta), // N3
    ]
}

/// Evaluate the reference-space gradients `[[∂Ni/∂ξ, ∂Ni/∂η]; 4]` at `(xi, eta)`.
#[inline]
pub fn q1_shape_grad(xi: f64, eta: f64) -> [[f64; 2]; 4] {
    [
        // N0 = ¼(1−ξ)(1−η)
        [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
        // N1 = ¼(1+ξ)(1−η)
        [0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
        // N2 = ¼(1+ξ)(1+η)
        [0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
        // N3 = ¼(1−ξ)(1+η)
        [-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
    ]
}

// ── 2×2 Gauss–Legendre quadrature ──────────────────────────────────────────────

/// Return the 2×2 Gauss–Legendre quadrature rule for the reference square
/// `[−1,1]²`.
///
/// Returns `(points[4], weights[4])` where each point is `[ξ, η]` at the four
/// combinations of `±1/√3`, all with weight `1`.  The weights sum to `4`, the
/// area of `[−1,1]²`.
pub fn gauss2x2() -> ([[f64; 2]; 4], [f64; 4]) {
    let g = 1.0 / 3.0_f64.sqrt();
    let points = [[-g, -g], [g, -g], [g, g], [-g, g]];
    let weights = [1.0, 1.0, 1.0, 1.0];
    (points, weights)
}

// ── Jacobian ──────────────────────────────────────────────────────────────────

/// Compute the bilinear Jacobian at reference point `(xi, eta)` for the
/// quadrilateral with physical node coordinates `nodes`.
///
/// Returns `(J[2][2], det_J, J_inv[2][2])` where
/// `J[a][b] = Σ_i (∂Ni/∂ξ_b) · x_i[a]`, i.e.
/// `J = [[∂x/∂ξ, ∂x/∂η], [∂y/∂ξ, ∂y/∂η]]`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` if `det_J ≤ 1e-15` (degenerate or inverted
/// element), since a valid, positively-oriented quadrilateral must have a
/// strictly positive Jacobian determinant at every Gauss point.
pub fn q1_jacobian(nodes: &[[f64; 2]; 4], xi: f64, eta: f64) -> PdeResult<JacobianResult> {
    let grad = q1_shape_grad(xi, eta);

    // J[a][b] = Σ_i grad[i][b] * nodes[i][a]
    let mut j = [[0.0_f64; 2]; 2];
    for i in 0..4 {
        let x = nodes[i][0];
        let y = nodes[i][1];
        let dxi = grad[i][0];
        let deta = grad[i][1];
        j[0][0] += dxi * x; // ∂x/∂ξ
        j[0][1] += deta * x; // ∂x/∂η
        j[1][0] += dxi * y; // ∂y/∂ξ
        j[1][1] += deta * y; // ∂y/∂η
    }

    let det_j = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if det_j <= 1.0e-15 {
        return Err(PdeError::SingularMatrix(format!(
            "degenerate or inverted Q1 quadrilateral: det_J={det_j}"
        )));
    }

    let inv_det = 1.0 / det_j;
    let j_inv = [
        [inv_det * j[1][1], inv_det * (-j[0][1])],
        [inv_det * (-j[1][0]), inv_det * j[0][0]],
    ];

    Ok((j, det_j, j_inv))
}

// ── Local stiffness matrix ─────────────────────────────────────────────────────

/// Compute the local 4×4 stiffness matrix (row-major, length 16) for `−∇·(∇u)`
/// via 2×2 Gauss quadrature.
///
/// `K[i*4+j] = Σ_q w_q · det_J(q) · (∇Ni · ∇Nj)` with physical gradients
/// obtained from the reference gradients by `∇N = J⁻ᵀ · ∇_ref N`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate/inverted elements.
pub fn q1_local_stiffness(nodes: &[[f64; 2]; 4]) -> PdeResult<[f64; 16]> {
    let (points, weights) = gauss2x2();
    let mut k = [0.0_f64; 16];

    for q in 0..4 {
        let xi = points[q][0];
        let eta = points[q][1];
        let (_, det_j, j_inv) = q1_jacobian(nodes, xi, eta)?;
        let grad_ref = q1_shape_grad(xi, eta);

        // Physical gradients: ∇N = J⁻ᵀ · ∇_ref N.
        // [dNi/dx, dNi/dy] = [[j_inv00, j_inv10],[j_inv01, j_inv11]] · [dNi/dξ, dNi/dη]
        let mut grad_phys = [[0.0_f64; 2]; 4];
        for i in 0..4 {
            grad_phys[i][0] = j_inv[0][0] * grad_ref[i][0] + j_inv[1][0] * grad_ref[i][1];
            grad_phys[i][1] = j_inv[0][1] * grad_ref[i][0] + j_inv[1][1] * grad_ref[i][1];
        }

        let wq = weights[q] * det_j;
        for i in 0..4 {
            for j in 0..4 {
                let dot = grad_phys[i][0] * grad_phys[j][0] + grad_phys[i][1] * grad_phys[j][1];
                k[i * 4 + j] += wq * dot;
            }
        }
    }

    Ok(k)
}

// ── Local mass matrix ─────────────────────────────────────────────────────────

/// Compute the local 4×4 mass matrix (row-major, length 16): `∫ NᵀN dΩ` via
/// 2×2 Gauss quadrature.
///
/// `M[i*4+j] = Σ_q w_q · det_J(q) · Ni(ξ_q, η_q) · Nj(ξ_q, η_q)`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate/inverted elements.
pub fn q1_local_mass(nodes: &[[f64; 2]; 4]) -> PdeResult<[f64; 16]> {
    let (points, weights) = gauss2x2();
    let mut m = [0.0_f64; 16];

    for q in 0..4 {
        let xi = points[q][0];
        let eta = points[q][1];
        let (_, det_j, _) = q1_jacobian(nodes, xi, eta)?;
        let n = q1_shape_fn(xi, eta);
        let wq = weights[q] * det_j;
        for i in 0..4 {
            for j in 0..4 {
                m[i * 4 + j] += wq * n[i] * n[j];
            }
        }
    }

    Ok(m)
}

// ── Local load vector ─────────────────────────────────────────────────────────

/// Compute the local 4-element load vector for a constant source `f`:
/// `∫ f·N dΩ` via 2×2 Gauss quadrature.
///
/// `b[i] = Σ_q w_q · det_J(q) · f · Ni(ξ_q, η_q)`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate/inverted elements.
pub fn q1_local_load(nodes: &[[f64; 2]; 4], f: f64) -> PdeResult<[f64; 4]> {
    let (points, weights) = gauss2x2();
    let mut b = [0.0_f64; 4];

    for q in 0..4 {
        let xi = points[q][0];
        let eta = points[q][1];
        let (_, det_j, _) = q1_jacobian(nodes, xi, eta)?;
        let n = q1_shape_fn(xi, eta);
        let wq = weights[q] * det_j * f;
        for i in 0..4 {
            b[i] += wq * n[i];
        }
    }

    Ok(b)
}

// ── Verification helpers ──────────────────────────────────────────────────────

/// Check the partition-of-unity property: `Σ_i N_i(ξ, η) = 1`.
///
/// Returns the computed sum; should be ≈ 1.0 for all `(ξ, η)`.
#[inline]
pub fn q1_partition_of_unity(xi: f64, eta: f64) -> f64 {
    q1_shape_fn(xi, eta).iter().sum()
}

/// Check whether a 4×4 (row-major) matrix is symmetric within tolerance `tol`.
///
/// Returns `true` iff `|A[i,j] − A[j,i]| ≤ tol` for all `i, j`.
pub fn q1_matrix_is_symmetric(a: &[f64; 16], tol: f64) -> bool {
    for i in 0..4 {
        for j in 0..4 {
            if (a[i * 4 + j] - a[j * 4 + i]).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-12;

    /// The reference quadrilateral as a physical element: the unit square
    /// `[0,1]²` with the standard CCW node ordering.
    fn unit_square() -> [[f64; 2]; 4] {
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    /// The four corners of the reference square `[−1,1]²` in node order.
    fn ref_corners() -> [[f64; 2]; 4] {
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    }

    // ── Partition of unity ─────────────────────────────────────────────────

    #[test]
    fn partition_of_unity_several_points() {
        for &(xi, eta) in &[
            (0.0, 0.0),
            (0.3, -0.7),
            (-0.5, 0.5),
            (1.0, 1.0),
            (-1.0, 0.25),
            (0.9, -0.1),
        ] {
            let s = q1_partition_of_unity(xi, eta);
            assert!(
                (s - 1.0).abs() < TOL,
                "partition of unity at ({xi},{eta}) = {s}"
            );
        }
    }

    #[test]
    fn shape_fn_kronecker_at_corners() {
        let corners = ref_corners();
        for (node, corner) in corners.iter().enumerate() {
            let n = q1_shape_fn(corner[0], corner[1]);
            for (k, &nk) in n.iter().enumerate() {
                let expected = if k == node { 1.0 } else { 0.0 };
                assert!(
                    (nk - expected).abs() < TOL,
                    "N{k} at corner {node} = {nk}, expected {expected}"
                );
            }
        }
    }

    // ── Shape gradients ────────────────────────────────────────────────────

    #[test]
    fn shape_grad_rows_sum_to_zero() {
        // Σ_i ∂Ni/∂ξ = 0 and Σ_i ∂Ni/∂η = 0 everywhere (constant reproduction).
        for &(xi, eta) in &[(0.0, 0.0), (0.4, -0.6), (-0.3, 0.8), (1.0, -1.0)] {
            let g = q1_shape_grad(xi, eta);
            let sum_dxi: f64 = (0..4).map(|i| g[i][0]).sum();
            let sum_deta: f64 = (0..4).map(|i| g[i][1]).sum();
            assert!(sum_dxi.abs() < TOL, "Σ ∂Ni/∂ξ = {sum_dxi} at ({xi},{eta})");
            assert!(
                sum_deta.abs() < TOL,
                "Σ ∂Ni/∂η = {sum_deta} at ({xi},{eta})"
            );
        }
    }

    #[test]
    fn shape_grad_reproduces_linear_field() {
        // Σ_i N_i x_i must reproduce the affine map; the derivative wrt ξ of
        // the x-coordinate of the unit square equals 1/2 (since ξ∈[−1,1]).
        let g = q1_shape_grad(0.2, -0.4);
        let dx_dxi: f64 = (0..4).map(|i| g[i][0] * unit_square()[i][0]).sum();
        assert!((dx_dxi - 0.5).abs() < TOL, "∂x/∂ξ = {dx_dxi}, expected 0.5");
    }

    // ── Gauss quadrature ───────────────────────────────────────────────────

    #[test]
    fn gauss2x2_weights_sum_to_four() {
        let (_, w) = gauss2x2();
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 4.0).abs() < 1.0e-14,
            "weights sum = {sum}, expected 4"
        );
    }

    #[test]
    fn gauss2x2_points_at_inv_sqrt3() {
        let (p, _) = gauss2x2();
        let g = 1.0 / 3.0_f64.sqrt();
        for point in &p {
            assert!((point[0].abs() - g).abs() < TOL);
            assert!((point[1].abs() - g).abs() < TOL);
        }
    }

    // ── Jacobian ───────────────────────────────────────────────────────────

    #[test]
    fn jacobian_unit_square_det_is_quarter() {
        // For the unit square [0,1]² mapped from [−1,1]², the Jacobian is
        // constant: J = diag(1/2, 1/2), det = 1/4.
        let nodes = unit_square();
        for &(xi, eta) in &[(0.0, 0.0), (0.5, -0.5), (-0.9, 0.9)] {
            let (j, det_j, _) = q1_jacobian(&nodes, xi, eta).expect("ok");
            assert!(
                (det_j - 0.25).abs() < TOL,
                "det_J = {det_j} at ({xi},{eta})"
            );
            assert!((j[0][0] - 0.5).abs() < TOL);
            assert!((j[1][1] - 0.5).abs() < TOL);
            assert!(j[0][1].abs() < TOL);
            assert!(j[1][0].abs() < TOL);
        }
    }

    #[test]
    fn jacobian_inverse_consistency() {
        // J · J⁻¹ = I.
        let nodes = [[0.0, 0.0], [2.0, 0.3], [2.4, 2.1], [0.1, 1.9]];
        let (j, _, j_inv) = q1_jacobian(&nodes, 0.2, -0.3).expect("ok");
        let mut prod = [[0.0_f64; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    prod[a][b] += j[a][c] * j_inv[c][b];
                }
            }
        }
        assert!((prod[0][0] - 1.0).abs() < TOL);
        assert!((prod[1][1] - 1.0).abs() < TOL);
        assert!(prod[0][1].abs() < TOL);
        assert!(prod[1][0].abs() < TOL);
    }

    #[test]
    fn jacobian_degenerate_collinear_errors() {
        // All four nodes collinear on the x-axis → zero-area element (the
        // y-derivative row of J vanishes identically), so det_J = 0 at every
        // reference point including the centre → Err.
        let nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let res = q1_jacobian(&nodes, 0.0, 0.0);
        assert!(res.is_err(), "collinear quad should give singular Jacobian");
        // Also degenerate at a Gauss point.
        let g = 1.0 / 3.0_f64.sqrt();
        assert!(q1_jacobian(&nodes, g, g).is_err());
    }

    #[test]
    fn jacobian_collapsed_to_segment_errors() {
        // Three collinear nodes plus a coincident node collapse the element to
        // a segment; det_J ≤ 0 at the Gauss points → the stiffness assembly
        // surfaces an Err.
        let nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]];
        assert!(q1_local_stiffness(&nodes).is_err());
    }

    #[test]
    fn jacobian_inverted_element_errors() {
        // Reverse the orientation (clockwise) → negative det_J → Err.
        let nodes = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let res = q1_jacobian(&nodes, 0.0, 0.0);
        assert!(res.is_err(), "clockwise quad should give negative det_J");
    }

    // ── Stiffness matrix ───────────────────────────────────────────────────

    #[test]
    fn stiffness_symmetric() {
        let k = q1_local_stiffness(&unit_square()).expect("ok");
        assert!(
            q1_matrix_is_symmetric(&k, 1.0e-12),
            "stiffness not symmetric"
        );
    }

    #[test]
    fn stiffness_row_sums_zero() {
        // Constant function is in the kernel of −Δ ⇒ each row sums to ≈ 0.
        let nodes = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5]];
        let k = q1_local_stiffness(&nodes).expect("ok");
        for i in 0..4 {
            let s: f64 = (0..4).map(|j| k[i * 4 + j]).sum();
            assert!(s.abs() < 1.0e-12, "row {i} sum = {s}");
        }
    }

    #[test]
    fn stiffness_unit_square_matches_known_stencil() {
        // The Q1 Laplacian on the unit square is
        //   K = 1/6 · [[ 4, -1, -2, -1],
        //              [-1,  4, -1, -2],
        //              [-2, -1,  4, -1],
        //              [-1, -2, -1,  4]].
        let k = q1_local_stiffness(&unit_square()).expect("ok");
        let s = 1.0 / 6.0;
        let expected = [
            4.0 * s,
            -s,
            -2.0 * s,
            -s,
            -s,
            4.0 * s,
            -s,
            -2.0 * s,
            -2.0 * s,
            -s,
            4.0 * s,
            -s,
            -s,
            -2.0 * s,
            -s,
            4.0 * s,
        ];
        for idx in 0..16 {
            assert!(
                (k[idx] - expected[idx]).abs() < 1.0e-12,
                "K[{idx}] = {}, expected {}",
                k[idx],
                expected[idx]
            );
        }
        // Trace should be 4·(4/6) = 8/3.
        let trace = k[0] + k[5] + k[10] + k[15];
        assert!((trace - 8.0 / 3.0).abs() < 1.0e-12, "trace = {trace}");
    }

    // ── Mass matrix ────────────────────────────────────────────────────────

    #[test]
    fn mass_symmetric_and_positive() {
        let nodes = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]];
        let m = q1_local_mass(&nodes).expect("ok");
        assert!(q1_matrix_is_symmetric(&m, 1.0e-12), "mass not symmetric");
        for &v in m.iter() {
            assert!(v > 0.0, "mass entry {v} should be > 0 for a valid element");
        }
    }

    #[test]
    fn mass_total_sum_equals_area() {
        // Σ_{i,j} M_ij = ∫(Σ N_i)(Σ N_j) dΩ = ∫1 dΩ = area (partition of unity).
        let nodes = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]];
        let m = q1_local_mass(&nodes).expect("ok");
        let total: f64 = m.iter().sum();
        let area = 6.0_f64; // 3 × 2 rectangle
        assert!(
            (total - area).abs() < 1.0e-12,
            "Σ M_ij = {total}, expected {area}"
        );
    }

    #[test]
    fn mass_row_sums_equal_integral_of_shape() {
        // Each row of M sums to ∫ N_i dΩ; over all rows that is the area.
        let nodes = [[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]];
        let m = q1_local_mass(&nodes).expect("ok");
        let area = 4.0_f64;
        let mut row_total = 0.0;
        for i in 0..4 {
            let s: f64 = (0..4).map(|j| m[i * 4 + j]).sum();
            assert!(s > 0.0, "row {i} integral should be positive");
            row_total += s;
        }
        assert!(
            (row_total - area).abs() < 1.0e-12,
            "sum of row integrals = {row_total}, expected {area}"
        );
    }

    #[test]
    fn mass_unit_square_known_matrix() {
        // For the unit square, M = 1/36 · [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]].
        let m = q1_local_mass(&unit_square()).expect("ok");
        let s = 1.0 / 36.0;
        let expected = [
            4.0 * s,
            2.0 * s,
            s,
            2.0 * s,
            2.0 * s,
            4.0 * s,
            2.0 * s,
            s,
            s,
            2.0 * s,
            4.0 * s,
            2.0 * s,
            2.0 * s,
            s,
            2.0 * s,
            4.0 * s,
        ];
        for idx in 0..16 {
            assert!(
                (m[idx] - expected[idx]).abs() < 1.0e-13,
                "M[{idx}] = {}, expected {}",
                m[idx],
                expected[idx]
            );
        }
    }

    // ── Load vector ────────────────────────────────────────────────────────

    #[test]
    fn load_sum_equals_f_times_area() {
        let nodes = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]];
        let f_val = 5.0_f64;
        let b = q1_local_load(&nodes, f_val).expect("ok");
        let area = 6.0_f64;
        let total: f64 = b.iter().sum();
        assert!(
            (total - f_val * area).abs() < 1.0e-12,
            "Σ b_i = {total}, expected {}",
            f_val * area
        );
    }

    #[test]
    fn load_entries_equal_for_symmetric_square() {
        // On a symmetric square with constant f, all entries are equal (= area/4).
        let nodes = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let b = q1_local_load(&nodes, 1.0).expect("ok");
        let area = 4.0_f64;
        for (i, &bi) in b.iter().enumerate() {
            assert!(
                (bi - area / 4.0).abs() < 1.0e-12,
                "b[{i}] = {bi}, expected {}",
                area / 4.0
            );
        }
    }

    #[test]
    fn load_degenerate_errors() {
        // Zero-area (all-collinear) quad → det_J = 0 at every Gauss point.
        let nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let res = q1_local_load(&nodes, 1.0);
        assert!(res.is_err(), "degenerate quad load should error");
    }

    // ── Scaling ────────────────────────────────────────────────────────────

    #[test]
    fn mass_scales_with_area() {
        // Scaling the element linearly by factor `c` scales the mass by `c²`
        // (area factor).
        let base = unit_square();
        let c = 3.0_f64;
        let scaled: [[f64; 2]; 4] = [
            [base[0][0] * c, base[0][1] * c],
            [base[1][0] * c, base[1][1] * c],
            [base[2][0] * c, base[2][1] * c],
            [base[3][0] * c, base[3][1] * c],
        ];
        let m_base = q1_local_mass(&base).expect("ok");
        let m_scaled = q1_local_mass(&scaled).expect("ok");
        for idx in 0..16 {
            assert!(
                (m_scaled[idx] - c * c * m_base[idx]).abs() < 1.0e-12,
                "M[{idx}]: scaled {} vs c²·base {}",
                m_scaled[idx],
                c * c * m_base[idx]
            );
        }
    }

    #[test]
    fn stiffness_scale_invariant_2d() {
        // For 2D Laplacian, the stiffness matrix is scale-invariant under
        // uniform scaling (the area factor cancels the squared gradient scaling).
        let base = unit_square();
        let c = 2.5_f64;
        let scaled: [[f64; 2]; 4] = [
            [base[0][0] * c, base[0][1] * c],
            [base[1][0] * c, base[1][1] * c],
            [base[2][0] * c, base[2][1] * c],
            [base[3][0] * c, base[3][1] * c],
        ];
        let k_base = q1_local_stiffness(&base).expect("ok");
        let k_scaled = q1_local_stiffness(&scaled).expect("ok");
        for idx in 0..16 {
            assert!(
                (k_scaled[idx] - k_base[idx]).abs() < 1.0e-12,
                "K[{idx}]: scaled {} vs base {}",
                k_scaled[idx],
                k_base[idx]
            );
        }
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn deterministic() {
        let nodes = [[0.1, 0.2], [2.3, 0.1], [2.1, 1.9], [0.05, 1.7]];
        let k1 = q1_local_stiffness(&nodes).expect("ok");
        let k2 = q1_local_stiffness(&nodes).expect("ok");
        let m1 = q1_local_mass(&nodes).expect("ok");
        let m2 = q1_local_mass(&nodes).expect("ok");
        assert_eq!(k1, k2);
        assert_eq!(m1, m2);
    }

    #[test]
    fn matrix_is_symmetric_helper_detects_asymmetry() {
        let mut a = [0.0_f64; 16];
        a[1] = 1.0; // a[0,1] = 1, a[1,0] = 0
        assert!(!q1_matrix_is_symmetric(&a, 1.0e-12));
        let sym = q1_local_mass(&unit_square()).expect("ok");
        assert!(q1_matrix_is_symmetric(&sym, 1.0e-12));
    }
}
