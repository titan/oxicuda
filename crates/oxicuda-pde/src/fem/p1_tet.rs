//! Local element matrices for a P1 (linear Lagrange) tetrahedron.
//!
//! Standard 4-node linear tetrahedron in 3D.  The shape functions are the
//! barycentric (volume) coordinates and are affine, so their gradients are
//! constant over the element.  Node ordering is arbitrary but fixed per
//! element by the caller; orientation only affects the *sign* of the raw
//! determinant, which is handled by taking absolute values for the volume.
//!
//! For a tet with physical nodes `p_0, p_1, p_2, p_3`, the linear shape
//! function `N_i(x,y,z) = a_i + b_i·x + c_i·y + d_i·z` reproduces the nodal
//! delta property `N_i(p_j) = δ_ij`.  The coefficient set is obtained from the
//! inverse of the affine matrix
//!
//! ```text
//!     T = [[1, x_0, y_0, z_0],
//!          [1, x_1, y_1, z_1],
//!          [1, x_2, y_2, z_2],
//!          [1, x_3, y_3, z_3]]
//! ```
//!
//! Because `T · [a; b; c; d] = e_i` for shape function `i`, the coefficients
//! `(a_i, b_i, c_i, d_i)` are exactly column `i` of `T⁻¹`.  The constant
//! gradient of `N_i` is then `∇N_i = (b_i, c_i, d_i)`.

use crate::error::{PdeError, PdeResult};

/// Number of DOFs per P1 tetrahedron element.
pub const P1_TET_N_DOFS: usize = 4;

/// Tolerance below which a tetrahedron's volume is considered degenerate.
const VOLUME_TOL: f64 = 1.0e-15;

// ── Volume ──────────────────────────────────────────────────────────────────

/// Compute the volume of the tetrahedron defined by the 4 physical nodes.
///
/// `V = |det[e1 e2 e3]| / 6` where `e_k = node_k − node_0`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` if `V ≤ VOLUME_TOL` (degenerate / coplanar
/// nodes).
pub fn tet_volume(nodes: &[[f64; 3]; 4]) -> PdeResult<f64> {
    let e1 = [
        nodes[1][0] - nodes[0][0],
        nodes[1][1] - nodes[0][1],
        nodes[1][2] - nodes[0][2],
    ];
    let e2 = [
        nodes[2][0] - nodes[0][0],
        nodes[2][1] - nodes[0][1],
        nodes[2][2] - nodes[0][2],
    ];
    let e3 = [
        nodes[3][0] - nodes[0][0],
        nodes[3][1] - nodes[0][1],
        nodes[3][2] - nodes[0][2],
    ];

    // det of the 3×3 edge matrix (rows e1, e2, e3) = e1 · (e2 × e3).
    let cross = [
        e2[1] * e3[2] - e2[2] * e3[1],
        e2[2] * e3[0] - e2[0] * e3[2],
        e2[0] * e3[1] - e2[1] * e3[0],
    ];
    let triple = e1[0] * cross[0] + e1[1] * cross[1] + e1[2] * cross[2];

    let volume = triple.abs() / 6.0;
    if volume <= VOLUME_TOL {
        return Err(PdeError::SingularMatrix(format!(
            "degenerate tetrahedron: V={volume}"
        )));
    }
    Ok(volume)
}

// ── Shape-function gradients ──────────────────────────────────────────────────

/// Compute the constant shape-function gradients `[∇N_i; 4]` (each
/// `[∂/∂x, ∂/∂y, ∂/∂z]`) for the tetrahedron.
///
/// The coefficients are obtained by inverting the 4×4 affine matrix
/// `T = [[1, x_i, y_i, z_i]]`; the gradient of `N_i` is `(b_i, c_i, d_i)`,
/// i.e. rows 1–3 of column `i` of `T⁻¹`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` if `T` is singular (degenerate / coplanar
/// nodes).
pub fn p1_tet_shape_grad(nodes: &[[f64; 3]; 4]) -> PdeResult<[[f64; 3]; 4]> {
    // Reject degenerate elements up front (also validates non-coplanarity).
    let _ = tet_volume(nodes)?;

    // Build T row-major (4×4).
    let mut t = [0.0_f64; 16];
    for i in 0..4 {
        t[i * 4] = 1.0;
        t[i * 4 + 1] = nodes[i][0];
        t[i * 4 + 2] = nodes[i][1];
        t[i * 4 + 3] = nodes[i][2];
    }

    let t_inv = invert_4x4(&t)?;

    // Column i of T⁻¹ holds (a_i, b_i, c_i, d_i) in rows 0..4.
    // ∇N_i = (b_i, c_i, d_i) = (T⁻¹[1][i], T⁻¹[2][i], T⁻¹[3][i]).
    let mut grad = [[0.0_f64; 3]; 4];
    for (i, g) in grad.iter_mut().enumerate() {
        g[0] = t_inv[4 + i];
        g[1] = t_inv[8 + i];
        g[2] = t_inv[12 + i];
    }
    Ok(grad)
}

// ── Local stiffness matrix ─────────────────────────────────────────────────────

/// Compute the local 4×4 stiffness matrix (row-major, length 16) for `−Δ`.
///
/// Since gradients are constant, `K_ij = V · (∇N_i · ∇N_j)`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate / coplanar tetrahedra.
pub fn p1_tet_local_stiffness(nodes: &[[f64; 3]; 4]) -> PdeResult<[f64; 16]> {
    let volume = tet_volume(nodes)?;
    let grad = p1_tet_shape_grad(nodes)?;

    let mut k = [0.0_f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            let dot = grad[i][0] * grad[j][0] + grad[i][1] * grad[j][1] + grad[i][2] * grad[j][2];
            k[i * 4 + j] = volume * dot;
        }
    }
    Ok(k)
}

// ── Local mass matrix ─────────────────────────────────────────────────────────

/// Compute the local 4×4 consistent mass matrix (row-major, length 16):
/// `M_ij = ∫ N_i N_j dV = V/20·(1 + δ_ij)`.
///
/// That is `V/10` on the diagonal and `V/20` off-diagonal.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate / coplanar tetrahedra.
pub fn p1_tet_local_mass(nodes: &[[f64; 3]; 4]) -> PdeResult<[f64; 16]> {
    let volume = tet_volume(nodes)?;
    let off = volume / 20.0;
    let diag = volume / 10.0;

    let mut m = [off; 16];
    for i in 0..4 {
        m[i * 4 + i] = diag;
    }
    Ok(m)
}

// ── Local load vector ─────────────────────────────────────────────────────────

/// Compute the local 4-element load vector for a constant source `f`:
/// `b_i = ∫ f·N_i dV = f·V/4`.
///
/// # Errors
/// Returns `Err(SingularMatrix)` for degenerate / coplanar tetrahedra.
pub fn p1_tet_local_load(nodes: &[[f64; 3]; 4], f: f64) -> PdeResult<[f64; 4]> {
    let volume = tet_volume(nodes)?;
    let value = f * volume / 4.0;
    Ok([value, value, value, value])
}

// ── 4×4 dense inverse (Gauss–Jordan with partial pivoting) ────────────────────

/// Invert a 4×4 row-major matrix via Gauss–Jordan elimination with partial
/// pivoting.
///
/// # Errors
/// Returns `Err(SingularMatrix)` if a pivot is numerically zero.
fn invert_4x4(a: &[f64; 16]) -> PdeResult<[f64; 16]> {
    // Augmented matrix [A | I] stored as 4×8 row-major.
    let mut aug = [0.0_f64; 32];
    for i in 0..4 {
        for j in 0..4 {
            aug[i * 8 + j] = a[i * 4 + j];
        }
        aug[i * 8 + 4 + i] = 1.0;
    }

    for col in 0..4 {
        // Partial pivot: find the row with the largest |value| in this column.
        let mut pivot_row = col;
        let mut pivot_mag = aug[col * 8 + col].abs();
        for r in (col + 1)..4 {
            let mag = aug[r * 8 + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_mag < 1.0e-15 {
            return Err(PdeError::SingularMatrix(format!(
                "singular 4x4 affine matrix at column {col}"
            )));
        }
        // Swap pivot row into place.
        if pivot_row != col {
            for j in 0..8 {
                aug.swap(col * 8 + j, pivot_row * 8 + j);
            }
        }
        // Normalize the pivot row.
        let pivot = aug[col * 8 + col];
        let inv_pivot = 1.0 / pivot;
        for j in 0..8 {
            aug[col * 8 + j] *= inv_pivot;
        }
        // Eliminate this column from all other rows.
        for r in 0..4 {
            if r == col {
                continue;
            }
            let factor = aug[r * 8 + col];
            if factor != 0.0 {
                for j in 0..8 {
                    aug[r * 8 + j] -= factor * aug[col * 8 + j];
                }
            }
        }
    }

    let mut inv = [0.0_f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            inv[i * 4 + j] = aug[i * 8 + 4 + j];
        }
    }
    Ok(inv)
}

// ── Verification helpers ──────────────────────────────────────────────────────

/// Check whether a 4×4 (row-major) matrix is symmetric within tolerance `tol`.
///
/// Returns `true` iff `|A[i,j] − A[j,i]| ≤ tol` for all `i, j`.
pub fn p1_tet_matrix_is_symmetric(a: &[f64; 16], tol: f64) -> bool {
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

    /// The reference unit tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    fn unit_tet() -> [[f64; 3]; 4] {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    // ── Volume ─────────────────────────────────────────────────────────────

    #[test]
    fn volume_reference_unit_tet() {
        let v = tet_volume(&unit_tet()).expect("ok");
        assert!((v - 1.0 / 6.0).abs() < TOL, "V = {v}, expected 1/6");
    }

    #[test]
    fn volume_invariant_to_permutation() {
        let nodes = unit_tet();
        let v0 = tet_volume(&nodes).expect("ok");
        // Swap nodes 1 and 2 (flips orientation but |V| is unchanged).
        let permuted = [nodes[0], nodes[2], nodes[1], nodes[3]];
        let v1 = tet_volume(&permuted).expect("ok");
        assert!((v0 - v1).abs() < TOL, "V0 = {v0}, V1 = {v1}");
        // A cyclic permutation of all four nodes.
        let cyclic = [nodes[1], nodes[2], nodes[3], nodes[0]];
        let v2 = tet_volume(&cyclic).expect("ok");
        assert!((v0 - v2).abs() < TOL, "V0 = {v0}, V2 = {v2}");
    }

    #[test]
    fn volume_scales_cubed() {
        // Linear scaling by c scales volume by c³.
        let base = unit_tet();
        let c = 2.0_f64;
        let scaled: [[f64; 3]; 4] = [
            [base[0][0] * c, base[0][1] * c, base[0][2] * c],
            [base[1][0] * c, base[1][1] * c, base[1][2] * c],
            [base[2][0] * c, base[2][1] * c, base[2][2] * c],
            [base[3][0] * c, base[3][1] * c, base[3][2] * c],
        ];
        let v_base = tet_volume(&base).expect("ok");
        let v_scaled = tet_volume(&scaled).expect("ok");
        assert!(
            (v_scaled - c * c * c * v_base).abs() < TOL,
            "V_scaled = {v_scaled}, expected {}",
            c * c * c * v_base
        );
    }

    #[test]
    fn volume_coplanar_errors() {
        // All four nodes in the z=0 plane → coplanar → degenerate.
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(tet_volume(&nodes).is_err(), "coplanar tet should error");
    }

    // ── Shape gradients ────────────────────────────────────────────────────

    #[test]
    fn shape_grad_sum_to_zero() {
        // Σ_i ∇N_i = 0 (constant function reproduction).
        let nodes = [
            [0.1, 0.2, 0.0],
            [1.3, 0.1, 0.2],
            [0.2, 1.4, 0.1],
            [0.0, 0.3, 1.5],
        ];
        let g = p1_tet_shape_grad(&nodes).expect("ok");
        let mut sum = [0.0_f64; 3];
        for grad_i in &g {
            for (comp, &val) in grad_i.iter().enumerate() {
                sum[comp] += val;
            }
        }
        for (comp, &s) in sum.iter().enumerate() {
            assert!(s.abs() < TOL, "Σ ∂N/∂x{comp} = {s}");
        }
    }

    #[test]
    fn shape_grad_reference_tet_values() {
        // For the reference unit tet, the analytic gradients are:
        //   ∇N0 = (-1,-1,-1), ∇N1 = (1,0,0), ∇N2 = (0,1,0), ∇N3 = (0,0,1).
        let g = p1_tet_shape_grad(&unit_tet()).expect("ok");
        let expected = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        for (i, (gi, ei)) in g.iter().zip(expected.iter()).enumerate() {
            for (c, (&gv, &ev)) in gi.iter().zip(ei.iter()).enumerate() {
                assert!((gv - ev).abs() < TOL, "∇N{i}[{c}] = {gv}, expected {ev}");
            }
        }
    }

    #[test]
    fn shape_grad_directional_derivative_relations() {
        // The constant gradient must satisfy ∇N_i · (p_j − p_k) = δ_ij − δ_ik,
        // since N_i(p_j) = δ_ij and N_i is affine.
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let g = p1_tet_shape_grad(&nodes).expect("ok");
        for (i, grad_i) in g.iter().enumerate() {
            for j in 0..4 {
                for k in 0..4 {
                    let edge = [
                        nodes[j][0] - nodes[k][0],
                        nodes[j][1] - nodes[k][1],
                        nodes[j][2] - nodes[k][2],
                    ];
                    let dot = grad_i[0] * edge[0] + grad_i[1] * edge[1] + grad_i[2] * edge[2];
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    let delta_ik = if i == k { 1.0 } else { 0.0 };
                    assert!(
                        (dot - (delta_ij - delta_ik)).abs() < TOL,
                        "∇N{i}·(p{j}−p{k}) = {dot}, expected {}",
                        delta_ij - delta_ik
                    );
                }
            }
        }
    }

    #[test]
    fn shape_grad_coplanar_errors() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(p1_tet_shape_grad(&nodes).is_err());
    }

    // ── Stiffness matrix ───────────────────────────────────────────────────

    #[test]
    fn stiffness_symmetric() {
        let nodes = [
            [0.1, 0.2, 0.0],
            [1.3, 0.1, 0.2],
            [0.2, 1.4, 0.1],
            [0.0, 0.3, 1.5],
        ];
        let k = p1_tet_local_stiffness(&nodes).expect("ok");
        assert!(
            p1_tet_matrix_is_symmetric(&k, 1.0e-12),
            "stiffness not symmetric"
        );
    }

    #[test]
    fn stiffness_row_sums_zero() {
        // Constant is in the kernel of −Δ ⇒ rows sum to ≈ 0.
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let k = p1_tet_local_stiffness(&nodes).expect("ok");
        for i in 0..4 {
            let s: f64 = (0..4).map(|j| k[i * 4 + j]).sum();
            assert!(s.abs() < 1.0e-12, "row {i} sum = {s}");
        }
    }

    #[test]
    fn stiffness_reference_tet_entries() {
        // Reference unit tet: V = 1/6, gradients as above.
        //   K_00 = V·(∇N0·∇N0) = (1/6)·3 = 1/2.
        //   K_11 = V·1 = 1/6.
        //   K_01 = V·(∇N0·∇N1) = (1/6)·(−1) = −1/6.
        //   K_12 = V·(∇N1·∇N2) = 0.
        let k = p1_tet_local_stiffness(&unit_tet()).expect("ok");
        assert!((k[0] - 0.5).abs() < TOL, "K00 = {}", k[0]);
        assert!((k[5] - 1.0 / 6.0).abs() < TOL, "K11 = {}", k[5]);
        assert!((k[1] + 1.0 / 6.0).abs() < TOL, "K01 = {}", k[1]);
        assert!(k[6].abs() < TOL, "K12 = {}", k[6]);
        // Trace = K00 + 3·(1/6) = 1/2 + 1/2 = 1.
        let trace = k[0] + k[5] + k[10] + k[15];
        assert!((trace - 1.0).abs() < TOL, "trace = {trace}");
    }

    #[test]
    fn stiffness_coplanar_errors() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(p1_tet_local_stiffness(&nodes).is_err());
    }

    #[test]
    fn stiffness_spectrum_permutation_invariant() {
        // Permuting nodes permutes rows/cols symmetrically, preserving the
        // eigenvalue spectrum.  We check an invariant proxy: the trace and the
        // Frobenius norm are unchanged under a symmetric permutation.
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.5, 0.1, 0.0],
            [0.2, 1.3, 0.1],
            [0.1, 0.2, 1.4],
        ];
        let k = p1_tet_local_stiffness(&nodes).expect("ok");
        let permuted_nodes = [nodes[2], nodes[0], nodes[3], nodes[1]];
        let kp = p1_tet_local_stiffness(&permuted_nodes).expect("ok");

        let trace_k = k[0] + k[5] + k[10] + k[15];
        let trace_kp = kp[0] + kp[5] + kp[10] + kp[15];
        assert!((trace_k - trace_kp).abs() < 1.0e-12, "traces differ");

        let frob_k: f64 = k.iter().map(|v| v * v).sum();
        let frob_kp: f64 = kp.iter().map(|v| v * v).sum();
        assert!(
            (frob_k - frob_kp).abs() < 1.0e-12,
            "Frobenius norms differ: {frob_k} vs {frob_kp}"
        );
    }

    // ── Mass matrix ────────────────────────────────────────────────────────

    #[test]
    fn mass_symmetric() {
        let nodes = [
            [0.1, 0.2, 0.0],
            [1.3, 0.1, 0.2],
            [0.2, 1.4, 0.1],
            [0.0, 0.3, 1.5],
        ];
        let m = p1_tet_local_mass(&nodes).expect("ok");
        assert!(
            p1_tet_matrix_is_symmetric(&m, 1.0e-12),
            "mass not symmetric"
        );
    }

    #[test]
    fn mass_diagonal_and_offdiagonal_values() {
        let nodes = unit_tet();
        let v = tet_volume(&nodes).expect("ok");
        let m = p1_tet_local_mass(&nodes).expect("ok");
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { v / 10.0 } else { v / 20.0 };
                assert!(
                    (m[i * 4 + j] - expected).abs() < TOL,
                    "M[{i},{j}] = {}, expected {expected}",
                    m[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn mass_total_sum_equals_volume() {
        // Σ_ij M_ij = V (partition of unity: ∫(Σ N_i)(Σ N_j) dV = ∫1 dV = V).
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let v = tet_volume(&nodes).expect("ok");
        let m = p1_tet_local_mass(&nodes).expect("ok");
        let total: f64 = m.iter().sum();
        assert!(
            (total - v).abs() < 1.0e-12,
            "Σ M_ij = {total}, expected {v}"
        );
    }

    #[test]
    fn mass_diagonally_dominant() {
        // Each diagonal (V/10) dominates the row sum of off-diagonals (3·V/20).
        // V/10 = 2V/20 < 3V/20, so it is NOT strictly diagonally dominant, but
        // the diagonal entry exceeds each individual off-diagonal entry.
        let nodes = unit_tet();
        let m = p1_tet_local_mass(&nodes).expect("ok");
        for i in 0..4 {
            let diag = m[i * 4 + i];
            for j in 0..4 {
                if i != j {
                    assert!(
                        diag > m[i * 4 + j],
                        "diagonal M[{i},{i}]={diag} should exceed M[{i},{j}]={}",
                        m[i * 4 + j]
                    );
                }
            }
        }
    }

    // ── Load vector ────────────────────────────────────────────────────────

    #[test]
    fn load_sum_equals_f_times_volume() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let v = tet_volume(&nodes).expect("ok");
        let f_val = 7.0_f64;
        let b = p1_tet_local_load(&nodes, f_val).expect("ok");
        let total: f64 = b.iter().sum();
        assert!(
            (total - f_val * v).abs() < 1.0e-12,
            "Σ b_i = {total}, expected {}",
            f_val * v
        );
        for (i, &bi) in b.iter().enumerate() {
            assert!((bi - f_val * v / 4.0).abs() < TOL, "b[{i}] = {bi}");
        }
    }

    #[test]
    fn load_coplanar_errors() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(p1_tet_local_load(&nodes, 1.0).is_err());
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn deterministic() {
        let nodes = [
            [0.05, 0.2, 0.1],
            [1.3, 0.1, 0.2],
            [0.2, 1.4, 0.1],
            [0.0, 0.3, 1.5],
        ];
        let k1 = p1_tet_local_stiffness(&nodes).expect("ok");
        let k2 = p1_tet_local_stiffness(&nodes).expect("ok");
        let m1 = p1_tet_local_mass(&nodes).expect("ok");
        let m2 = p1_tet_local_mass(&nodes).expect("ok");
        assert_eq!(k1, k2);
        assert_eq!(m1, m2);
    }

    #[test]
    fn matrix_is_symmetric_helper_detects_asymmetry() {
        let mut a = [0.0_f64; 16];
        a[2] = 1.0;
        assert!(!p1_tet_matrix_is_symmetric(&a, 1.0e-12));
        let sym = p1_tet_local_mass(&unit_tet()).expect("ok");
        assert!(p1_tet_matrix_is_symmetric(&sym, 1.0e-12));
    }

    #[test]
    fn invert_4x4_identity() {
        // Inverting the identity gives the identity.
        let mut id = [0.0_f64; 16];
        for i in 0..4 {
            id[i * 4 + i] = 1.0;
        }
        let inv = invert_4x4(&id).expect("ok");
        for idx in 0..16 {
            assert!(
                (inv[idx] - id[idx]).abs() < TOL,
                "inv[{idx}] = {}",
                inv[idx]
            );
        }
    }

    #[test]
    fn invert_4x4_singular_errors() {
        // A matrix with a zero row is singular.
        let mut a = [0.0_f64; 16];
        for i in 0..3 {
            a[i * 4 + i] = 1.0;
        }
        // row 3 left as zeros
        assert!(invert_4x4(&a).is_err());
    }
}
