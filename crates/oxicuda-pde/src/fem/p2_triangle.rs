//! Local element matrices for a P2 (quadratic Lagrange) triangle.
//!
//! The element has 6 DOFs: 3 vertex nodes and 3 edge-midpoint nodes.
//! Node ordering on the reference triangle (0,0)-(1,0)-(0,1):
//!
//! - Node 0: vertex (0, 0)
//! - Node 1: vertex (1, 0)
//! - Node 2: vertex (0, 1)
//! - Node 3: midpoint of edge 0→1, at (0.5, 0)
//! - Node 4: midpoint of edge 1→2, at (0.5, 0.5)
//! - Node 5: midpoint of edge 0→2, at (0, 0.5)
//!
//! Shape functions on the reference triangle:
//!   N0(ξ,η) = (1−ξ−η)·(1−2ξ−2η)
//!   N1(ξ,η) = ξ·(2ξ−1)
//!   N2(ξ,η) = η·(2η−1)
//!   N3(ξ,η) = 4ξ·(1−ξ−η)
//!   N4(ξ,η) = 4ξη
//!   N5(ξ,η) = 4η·(1−ξ−η)
//!
//! The geometric map is the same as P1 (affine), so J is constant per element.
//!
//! Integration uses a 7-point Gauss–Dunavant rule of degree 5, which is exact
//! for the degree-4 integrand arising in the stiffness and mass matrices.

use crate::error::{PdeError, PdeResult};

/// Number of DOFs per P2 triangle element.
pub const P2_N_DOFS: usize = 6;

// ── Shape functions ────────────────────────────────────────────────────────────

/// Evaluate all 6 P2 shape functions at reference coordinates `(xi, eta)`.
///
/// The reference triangle has vertices (0,0), (1,0), (0,1).
#[inline]
pub fn p2_shape_fn(xi: f64, eta: f64) -> [f64; 6] {
    let lambda = 1.0 - xi - eta;
    [
        lambda * (1.0 - 2.0 * xi - 2.0 * eta), // N0 = λ(2λ−1)
        xi * (2.0 * xi - 1.0),                 // N1
        eta * (2.0 * eta - 1.0),               // N2
        4.0 * xi * lambda,                     // N3
        4.0 * xi * eta,                        // N4
        4.0 * eta * lambda,                    // N5
    ]
}

/// Evaluate the reference-space gradients [[∂Ni/∂ξ, ∂Ni/∂η]; 6] at `(xi, eta)`.
#[inline]
pub fn p2_shape_grad(xi: f64, eta: f64) -> [[f64; 2]; 6] {
    [
        // N0: ∂/∂ξ = 4ξ+4η−3, ∂/∂η = 4ξ+4η−3
        [4.0 * xi + 4.0 * eta - 3.0, 4.0 * xi + 4.0 * eta - 3.0],
        // N1: ∂/∂ξ = 4ξ−1, ∂/∂η = 0
        [4.0 * xi - 1.0, 0.0],
        // N2: ∂/∂ξ = 0, ∂/∂η = 4η−1
        [0.0, 4.0 * eta - 1.0],
        // N3: ∂/∂ξ = 4−8ξ−4η, ∂/∂η = −4ξ
        [4.0 - 8.0 * xi - 4.0 * eta, -4.0 * xi],
        // N4: ∂/∂ξ = 4η, ∂/∂η = 4ξ
        [4.0 * eta, 4.0 * xi],
        // N5: ∂/∂ξ = −4η, ∂/∂η = 4−4ξ−8η
        [-4.0 * eta, 4.0 - 4.0 * xi - 8.0 * eta],
    ]
}

// ── 7-point Gauss–Dunavant quadrature ─────────────────────────────────────────

/// Return the 7-point Gauss–Dunavant quadrature rule of degree 5 for the
/// reference triangle.
///
/// Returns `(weights[7], xi[7], eta[7])`.  Weights sum to `0.5` (the area of
/// the reference triangle).
///
/// Source: Dunavant (1985), IJNME 21:1129–1148, Table 3 (n=5, 7-point rule).
pub fn gauss7() -> ([f64; 7], [f64; 7], [f64; 7]) {
    // Barycentric orbit parameters (Dunavant Table 3, degree 5)
    let a1 = 0.059_715_871_789_769_8_f64;
    let a2 = 0.470_142_064_105_115_1_f64;
    let b1 = 0.797_426_985_353_087_2_f64;
    let b2 = 0.101_286_507_323_456_3_f64;

    // Weights (halved because the reference triangle has area 1/2)
    let w0 = 0.225_f64 / 2.0;
    let w1 = 0.132_394_152_788_506_2_f64 / 2.0;
    let w2 = 0.125_939_180_544_827_2_f64 / 2.0;

    let weights = [w0, w1, w1, w1, w2, w2, w2];
    let xis = [1.0 / 3.0, a2, a1, a2, b2, b1, b2];
    let etas = [1.0 / 3.0, a2, a2, a1, b2, b2, b1];

    (weights, xis, etas)
}

// ── Jacobian ──────────────────────────────────────────────────────────────────

/// Compute the affine Jacobian for the P2 element (same affine map as P1).
///
/// Returns `(J_row_major[4], det_J, J_inv_row_major[4])` where:
/// - `J = [[x1−x0, x2−x0], [y1−y0, y2−y0]]` stored row-major
/// - `J_inv = (1/det_J) · [[y2−y0, −(x2−x0)], [−(y1−y0), x1−x0]]`
///
/// # Errors
/// Returns `Err(SingularMatrix)` if `|det_J| < 1e-15` (degenerate triangle).
pub fn p2_jacobian(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> PdeResult<([f64; 4], f64, [f64; 4])> {
    let j00 = x1 - x0;
    let j01 = x2 - x0;
    let j10 = y1 - y0;
    let j11 = y2 - y0;

    let det_j = j00 * j11 - j01 * j10;
    if det_j.abs() < 1.0e-15 {
        return Err(PdeError::SingularMatrix(format!(
            "degenerate P2 triangle: det_J={det_j}"
        )));
    }

    let inv_det = 1.0 / det_j;
    let j_inv = [
        inv_det * j11,
        inv_det * (-j01),
        inv_det * (-j10),
        inv_det * j00,
    ];

    Ok(([j00, j01, j10, j11], det_j, j_inv))
}

// ── Physical node coordinates ─────────────────────────────────────────────────

/// Compute the 6 physical node coordinates `[[x,y]; 6]` for a P2 triangle:
/// the 3 vertices followed by the 3 edge midpoints.
///
/// - Node 0: (x0, y0)
/// - Node 1: (x1, y1)
/// - Node 2: (x2, y2)
/// - Node 3: midpoint of edge 0→1
/// - Node 4: midpoint of edge 1→2
/// - Node 5: midpoint of edge 0→2
pub fn p2_node_coords(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> [[f64; 2]; 6] {
    [
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
        [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
        [0.5 * (x0 + x2), 0.5 * (y0 + y2)],
    ]
}

// ── Local stiffness matrix ─────────────────────────────────────────────────────

/// Compute the local 6×6 stiffness matrix via 7-point Gauss quadrature.
///
/// `K[i*6+j] = det_J · Σ_q w_q · (∇Ni · ∇Nj)` evaluated at Gauss point `q`.
///
/// The physical gradients are obtained by the chain rule:
/// `∂N/∂x = J_inv[0,0]·∂N/∂ξ + J_inv[0,1]·∂N/∂η`
/// `∂N/∂y = J_inv[1,0]·∂N/∂ξ + J_inv[1,1]·∂N/∂η`
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate triangles.
pub fn p2_local_stiffness(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> PdeResult<[f64; 36]> {
    let (_, det_j, j_inv) = p2_jacobian(x0, y0, x1, y1, x2, y2)?;
    let (weights, xis, etas) = gauss7();

    let mut k = [0.0_f64; 36];

    for q in 0..7 {
        let grad_ref = p2_shape_grad(xis[q], etas[q]);
        // Physical gradients: [dNi/dx, dNi/dy]
        let mut grad_phys = [[0.0_f64; 2]; 6];
        for i in 0..6 {
            grad_phys[i][0] = j_inv[0] * grad_ref[i][0] + j_inv[1] * grad_ref[i][1];
            grad_phys[i][1] = j_inv[2] * grad_ref[i][0] + j_inv[3] * grad_ref[i][1];
        }
        let wq = weights[q] * det_j;
        for i in 0..6 {
            for j in 0..6 {
                let dot = grad_phys[i][0] * grad_phys[j][0] + grad_phys[i][1] * grad_phys[j][1];
                k[i * 6 + j] += wq * dot;
            }
        }
    }

    Ok(k)
}

// ── Local mass matrix ─────────────────────────────────────────────────────────

/// Compute the local 6×6 mass matrix via 7-point Gauss quadrature.
///
/// `M[i*6+j] = det_J · Σ_q w_q · Ni(ξ_q, η_q) · Nj(ξ_q, η_q)`
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate triangles.
pub fn p2_local_mass(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> PdeResult<[f64; 36]> {
    let (_, det_j, _) = p2_jacobian(x0, y0, x1, y1, x2, y2)?;
    let (weights, xis, etas) = gauss7();

    let mut m = [0.0_f64; 36];

    for q in 0..7 {
        let n = p2_shape_fn(xis[q], etas[q]);
        let wq = weights[q] * det_j;
        for i in 0..6 {
            for j in 0..6 {
                m[i * 6 + j] += wq * n[i] * n[j];
            }
        }
    }

    Ok(m)
}

// ── Local load vector ─────────────────────────────────────────────────────────

/// Compute the local 6-element load vector for a constant source `f_val`.
///
/// `f[i] = det_J · Σ_q w_q · f_val · Ni(ξ_q, η_q)`
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate triangles.
pub fn p2_local_load(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    f_val: f64,
) -> PdeResult<[f64; 6]> {
    let (_, det_j, _) = p2_jacobian(x0, y0, x1, y1, x2, y2)?;
    let (weights, xis, etas) = gauss7();

    let mut f = [0.0_f64; 6];

    for q in 0..7 {
        let n = p2_shape_fn(xis[q], etas[q]);
        let wq = weights[q] * det_j * f_val;
        for i in 0..6 {
            f[i] += wq * n[i];
        }
    }

    Ok(f)
}

// ── Verification helpers ──────────────────────────────────────────────────────

/// Check the partition-of-unity property: `Σ_i N_i(ξ, η) = 1`.
///
/// Returns the computed sum; should be ≈ 1.0 for all `(ξ, η)` inside the
/// reference triangle.
#[inline]
pub fn p2_partition_of_unity(xi: f64, eta: f64) -> f64 {
    p2_shape_fn(xi, eta).iter().sum()
}

/// Check whether a 6×6 stiffness matrix is symmetric within tolerance `tol`.
///
/// Returns `true` iff `|K[i,j] − K[j,i]| ≤ tol` for all `i, j`.
pub fn p2_stiffness_is_symmetric(k: &[f64; 36], tol: f64) -> bool {
    for i in 0..6 {
        for j in 0..6 {
            if (k[i * 6 + j] - k[j * 6 + i]).abs() > tol {
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

    // ── Shape function values at the 6 reference nodes ─────────────────────

    #[test]
    fn shape_fn_at_node0() {
        let n = p2_shape_fn(0.0, 0.0);
        assert!((n[0] - 1.0).abs() < TOL, "N0 at node 0 should be 1");
        for k in [1_usize, 2, 3, 4, 5] {
            assert!(n[k].abs() < TOL, "N{k} at node 0 should be 0, got {}", n[k]);
        }
    }

    #[test]
    fn shape_fn_at_node1() {
        let n = p2_shape_fn(1.0, 0.0);
        assert!((n[1] - 1.0).abs() < TOL, "N1 at node 1 should be 1");
        for k in [0, 2, 3, 4, 5] {
            assert!(n[k].abs() < TOL, "N{k} at node 1 should be 0, got {}", n[k]);
        }
    }

    #[test]
    fn shape_fn_at_node2() {
        let n = p2_shape_fn(0.0, 1.0);
        assert!((n[2] - 1.0).abs() < TOL, "N2 at node 2 should be 1");
        for k in [0, 1, 3, 4, 5] {
            assert!(n[k].abs() < TOL, "N{k} at node 2 should be 0, got {}", n[k]);
        }
    }

    #[test]
    fn shape_fn_at_node3() {
        // Node 3 is the midpoint of edge 0→1: (0.5, 0)
        let n = p2_shape_fn(0.5, 0.0);
        assert!((n[3] - 1.0).abs() < TOL, "N3 at node 3 should be 1");
        for k in [0, 1, 2, 4, 5] {
            assert!(n[k].abs() < TOL, "N{k} at node 3 should be 0, got {}", n[k]);
        }
    }

    #[test]
    fn shape_fn_at_node4() {
        // Node 4 is the midpoint of edge 1→2: (0.5, 0.5)
        let n = p2_shape_fn(0.5, 0.5);
        assert!((n[4] - 1.0).abs() < TOL, "N4 at node 4 should be 1");
        for k in [0, 1, 2, 3, 5] {
            assert!(n[k].abs() < TOL, "N{k} at node 4 should be 0, got {}", n[k]);
        }
    }

    #[test]
    fn shape_fn_at_node5() {
        // Node 5 is the midpoint of edge 0→2: (0, 0.5)
        let n = p2_shape_fn(0.0, 0.5);
        assert!((n[5] - 1.0).abs() < TOL, "N5 at node 5 should be 1");
        for k in [0, 1, 2, 3, 4] {
            assert!(n[k].abs() < TOL, "N{k} at node 5 should be 0, got {}", n[k]);
        }
    }

    // ── Partition of unity ─────────────────────────────────────────────────

    #[test]
    fn partition_of_unity_centroid() {
        let s = p2_partition_of_unity(1.0 / 3.0, 1.0 / 3.0);
        assert!((s - 1.0).abs() < TOL, "partition of unity at centroid: {s}");
    }

    #[test]
    fn partition_of_unity_interior_point() {
        let s = p2_partition_of_unity(0.2, 0.3);
        assert!(
            (s - 1.0).abs() < TOL,
            "partition of unity at (0.2,0.3): {s}"
        );
    }

    // ── Gauss quadrature ───────────────────────────────────────────────────

    #[test]
    fn gauss7_weights_sum_to_half() {
        let (w, _, _) = gauss7();
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 0.5).abs() < 1.0e-14,
            "weights sum={sum} expected 0.5"
        );
    }

    // ── Jacobian ───────────────────────────────────────────────────────────

    #[test]
    fn jacobian_reference_triangle() {
        let (_, det_j, _) =
            p2_jacobian(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("reference triangle should not fail");
        // det_J = (1−0)(1−0)−(0−0)(0−0) = 1
        assert!((det_j - 1.0).abs() < TOL, "det_J={det_j}");
    }

    #[test]
    fn jacobian_degenerate_triangle_errors() {
        // Three collinear points → degenerate
        let res = p2_jacobian(0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
        assert!(res.is_err(), "degenerate triangle should return Err");
    }

    // ── Physical node coordinates ──────────────────────────────────────────

    #[test]
    fn node_coords_midpoints_correct() {
        let coords = p2_node_coords(0.0, 0.0, 2.0, 0.0, 0.0, 2.0);
        // Node 3: midpoint of (0,0)-(2,0) = (1,0)
        assert!((coords[3][0] - 1.0).abs() < TOL);
        assert!(coords[3][1].abs() < TOL);
        // Node 4: midpoint of (2,0)-(0,2) = (1,1)
        assert!((coords[4][0] - 1.0).abs() < TOL);
        assert!((coords[4][1] - 1.0).abs() < TOL);
        // Node 5: midpoint of (0,0)-(0,2) = (0,1)
        assert!(coords[5][0].abs() < TOL);
        assert!((coords[5][1] - 1.0).abs() < TOL);
    }

    // ── Local stiffness matrix ─────────────────────────────────────────────

    #[test]
    fn local_stiffness_symmetric() {
        let k = p2_local_stiffness(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        assert!(
            p2_stiffness_is_symmetric(&k, 1.0e-12),
            "stiffness should be symmetric"
        );
    }

    #[test]
    fn local_stiffness_row_sums_zero() {
        // The constant function 1 lies in the kernel: K·1 = 0.
        let k = p2_local_stiffness(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        for i in 0..6 {
            let s: f64 = (0..6).map(|j| k[i * 6 + j]).sum();
            assert!(s.abs() < 1.0e-11, "row {i} sum = {s}, expected ≈ 0");
        }
    }

    #[test]
    fn p2_stiffness_is_symmetric_fn_returns_true() {
        let k = p2_local_stiffness(0.1, 0.2, 1.3, 0.4, 0.5, 1.1).expect("ok");
        assert!(p2_stiffness_is_symmetric(&k, 1.0e-12));
    }

    // ── Local mass matrix ──────────────────────────────────────────────────

    #[test]
    fn local_mass_symmetric() {
        let m = p2_local_mass(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        for i in 0..6 {
            for j in 0..6 {
                let diff = (m[i * 6 + j] - m[j * 6 + i]).abs();
                assert!(diff < TOL, "M[{i},{j}] ≠ M[{j},{i}]: diff={diff}");
            }
        }
    }

    #[test]
    fn local_mass_positive_diagonal() {
        let m = p2_local_mass(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        for i in 0..6 {
            assert!(
                m[i * 6 + i] > 0.0,
                "M[{i},{i}]={} should be positive",
                m[i * 6 + i]
            );
        }
    }

    #[test]
    fn local_mass_total_sum_equals_area() {
        // For the reference triangle (area=1/2), Σ_{i,j} M_{ij} = area.
        // This follows from ∫(Σ_i N_i)(Σ_j N_j) dΩ = ∫1·1 dΩ = area (partition of unity).
        let m = p2_local_mass(0.0, 0.0, 1.0, 0.0, 0.0, 1.0).expect("ok");
        let total: f64 = m.iter().sum();
        let area = 0.5_f64;
        assert!(
            (total - area).abs() < 1.0e-13,
            "Σ M_ij = {total}, expected {area}"
        );
    }

    // ── Local load vector ──────────────────────────────────────────────────

    #[test]
    fn local_load_sum_equals_f_times_area() {
        // ∫f_val dΩ = f_val * area; this must equal the sum of the load vector.
        let f_val = 3.0_f64;
        let f = p2_local_load(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, f_val).expect("ok");
        let area = 0.5_f64;
        let total: f64 = f.iter().sum();
        assert!(
            (total - f_val * area).abs() < 1.0e-13,
            "Σ f_i = {total}, expected {}",
            f_val * area
        );
    }

    #[test]
    fn local_load_vertex_entries_near_zero() {
        // For the P2 reference triangle, ∫N_vertex dΩ = 0 exactly:
        // ∫(1−ξ−η)(1−2ξ−2η) dξ dη over the reference triangle = 0.
        let f = p2_local_load(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0).expect("ok");
        // Vertex nodes: indices 0, 1, 2
        for k in [0_usize, 1, 2] {
            assert!(
                f[k].abs() < 1.0e-13,
                "vertex load f[{k}] = {} should be ≈ 0",
                f[k]
            );
        }
    }
}
