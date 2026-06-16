//! Linear elasticity finite-element method in 2D (plane stress).
//!
//! Displacement formulation of the Cauchy momentum balance for a linear,
//! isotropic, homogeneous elastic solid:
//!
//! ```text
//! −∇·σ = b ,   σ = λ (∇·u) I + 2 μ ε(u) ,   ε(u) = ½ (∇u + ∇uᵀ)
//! ```
//!
//! The continuum is discretised with **constant-strain P1 triangles** (CST):
//! each triangle carries three nodes, each node two displacement degrees of
//! freedom `(uₓ, u_y)`, so an element has six DOFs and the element stiffness is
//! a dense 6×6 matrix `Kₑ = ∫ Bᵀ D B dΩ = A · Bᵀ D B` (B and D are constant on
//! a CST element). The global stiffness is assembled into a dense row-major
//! `[2N×2N]` matrix, Dirichlet conditions are applied by symmetric elimination,
//! and the resulting system is solved with the crate's dense Gaussian
//! elimination ([`crate::spectral::chebyshev::gauss_solve_dense`]).
//!
//! For **plane stress** (thin plate, thickness 1) the constitutive matrix
//! relating Voigt stress `[σₓₓ, σ_yy, σₓy]` to Voigt strain `[εₓₓ, ε_yy, γₓy]`
//! (engineering shear `γₓy = 2 εₓy`) is
//!
//! ```text
//!         E      | 1   ν      0     |
//! D = --------- | ν   1      0     |
//!     1 − ν²    | 0   0   (1−ν)/2  |
//! ```
//!
//! # Reference
//! Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 1, §4 (CST element);
//! Hughes, *The Finite Element Method*, ch. 2.

use crate::error::{PdeError, PdeResult};
use crate::spectral::chebyshev::gauss_solve_dense;

/// Number of degrees of freedom on a CST element (3 nodes × 2 components).
pub const ELASTICITY_ELEM_DOFS: usize = 6;

/// Linear, isotropic 2-D elastic material under the **plane-stress** assumption.
///
/// `e` is Young's modulus `E > 0` and `nu` is Poisson's ratio `ν`. For a
/// physically admissible plane-stress material the constitutive matrix is
/// positive definite for `−1 < ν < 1`; this type additionally enforces the
/// common engineering bound `ν < 1/2` (so the material is not auxetic-pathological
/// and the incompressible limit is excluded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearElasticity2D {
    /// Young's modulus `E > 0`.
    pub e: f64,
    /// Poisson's ratio `ν`, with `−1 < ν < 1/2`.
    pub nu: f64,
}

impl LinearElasticity2D {
    /// Build a plane-stress material model.
    ///
    /// # Errors
    /// Returns [`PdeError::InvalidParameter`] if `e` is not a finite positive
    /// number, or if `nu` lies outside the open interval `(−1, 1/2)`.
    pub fn new(e: f64, nu: f64) -> PdeResult<Self> {
        if !e.is_finite() || e <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "e".into(),
                reason: format!("Young's modulus must be a finite value > 0, got {e}"),
            });
        }
        if !nu.is_finite() || nu >= 0.5 || nu <= -1.0 {
            return Err(PdeError::InvalidParameter {
                name: "nu".into(),
                reason: format!("Poisson's ratio must satisfy -1 < nu < 0.5, got {nu}"),
            });
        }
        Ok(Self { e, nu })
    }

    /// Plane-stress constitutive matrix `D` (3×3, row-major Voigt form).
    #[must_use]
    pub fn plane_stress_d(&self) -> [f64; 9] {
        let f = self.e / (1.0 - self.nu * self.nu);
        let g = f * (1.0 - self.nu) / 2.0;
        [f, f * self.nu, 0.0, f * self.nu, f, 0.0, 0.0, 0.0, g]
    }

    /// Signed twice-area `2A = (x₁−x₀)(y₂−y₀) − (x₂−x₀)(y₁−y₀)` together with
    /// the gradient cofactors `bᵢ = ∂Nᵢ/∂x · 2A` and `cᵢ = ∂Nᵢ/∂y · 2A`.
    ///
    /// `coords` packs the three vertices as `[x₀,y₀, x₁,y₁, x₂,y₂]`.
    ///
    /// # Errors
    /// Returns [`PdeError::SingularMatrix`] for a degenerate (zero-area) triangle.
    fn cst_geometry(coords: &[f64; 6]) -> PdeResult<(f64, [f64; 3], [f64; 3])> {
        let (x0, y0) = (coords[0], coords[1]);
        let (x1, y1) = (coords[2], coords[3]);
        let (x2, y2) = (coords[4], coords[5]);
        let det2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if det2.abs() < 1.0e-14 {
            return Err(PdeError::SingularMatrix(format!(
                "degenerate CST triangle, 2A={det2:.3e}"
            )));
        }
        let b = [y1 - y2, y2 - y0, y0 - y1];
        let c = [x2 - x1, x0 - x2, x1 - x0];
        Ok((det2, b, c))
    }

    /// Strain-displacement matrix `B̂` (3×6, row-major) built from the gradient
    /// cofactors, *before* dividing by the signed twice-area. Column order is
    /// `[u₀, v₀, u₁, v₁, u₂, v₂]`; rows are `[εₓₓ, ε_yy, γₓy]`.
    fn b_hat(b: &[f64; 3], c: &[f64; 3]) -> [f64; 18] {
        [
            b[0], 0.0, b[1], 0.0, b[2], 0.0, // εₓₓ row
            0.0, c[0], 0.0, c[1], 0.0, c[2], // ε_yy row
            c[0], b[0], c[1], b[1], c[2], b[2], // γₓy row
        ]
    }

    /// Element stiffness matrix `Kₑ` (6×6, row-major) for one CST triangle.
    ///
    /// `Kₑ = A · Bᵀ D B = (1 / (2·|2A|)) · B̂ᵀ D B̂`, where `B = B̂/(2A)` and
    /// `A = |2A|/2` is the element area (thickness 1).
    ///
    /// # Errors
    /// Returns [`PdeError::SingularMatrix`] for a degenerate triangle.
    pub fn element_stiffness(&self, coords: &[f64; 6]) -> PdeResult<[f64; 36]> {
        let (det2, b, c) = Self::cst_geometry(coords)?;
        let d_mat = self.plane_stress_d();
        let bh = Self::b_hat(&b, &c);

        // db = D (3×3) · B̂ (3×6)  →  3×6.
        let mut db = [0.0_f64; 18];
        for r in 0..3 {
            for col in 0..ELASTICITY_ELEM_DOFS {
                let mut s = 0.0;
                for k in 0..3 {
                    s += d_mat[r * 3 + k] * bh[k * ELASTICITY_ELEM_DOFS + col];
                }
                db[r * ELASTICITY_ELEM_DOFS + col] = s;
            }
        }

        // ke = scale · B̂ᵀ (6×3) · db (3×6)  →  6×6, with scale = 1/(2|2A|).
        let scale = 1.0 / (2.0 * det2.abs());
        let mut ke = [0.0_f64; 36];
        for i in 0..ELASTICITY_ELEM_DOFS {
            for j in 0..ELASTICITY_ELEM_DOFS {
                let mut s = 0.0;
                for r in 0..3 {
                    s += bh[r * ELASTICITY_ELEM_DOFS + i] * db[r * ELASTICITY_ELEM_DOFS + j];
                }
                ke[i * ELASTICITY_ELEM_DOFS + j] = scale * s;
            }
        }
        Ok(ke)
    }

    /// Constant Voigt strain `[εₓₓ, ε_yy, γₓy]` of a CST element given its six
    /// nodal displacements `ue = [u₀,v₀, u₁,v₁, u₂,v₂]`.
    ///
    /// # Errors
    /// Returns [`PdeError::SingularMatrix`] for a degenerate triangle.
    pub fn element_strain(coords: &[f64; 6], ue: &[f64; 6]) -> PdeResult<[f64; 3]> {
        let (det2, b, c) = Self::cst_geometry(coords)?;
        let bh = Self::b_hat(&b, &c);
        let inv = 1.0 / det2; // B = B̂ / (2A);  2A = det2 (signed)
        let mut eps = [0.0_f64; 3];
        for (r, eps_r) in eps.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..ELASTICITY_ELEM_DOFS {
                s += bh[r * ELASTICITY_ELEM_DOFS + j] * ue[j];
            }
            *eps_r = s * inv;
        }
        Ok(eps)
    }

    /// Constant Voigt stress `[σₓₓ, σ_yy, σₓy]` of a CST element, `σ = D ε`.
    ///
    /// # Errors
    /// Returns [`PdeError::SingularMatrix`] for a degenerate triangle.
    pub fn element_stress(&self, coords: &[f64; 6], ue: &[f64; 6]) -> PdeResult<[f64; 3]> {
        let eps = Self::element_strain(coords, ue)?;
        let d_mat = self.plane_stress_d();
        let mut sigma = [0.0_f64; 3];
        for (r, sigma_r) in sigma.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..3 {
                s += d_mat[r * 3 + k] * eps[k];
            }
            *sigma_r = s;
        }
        Ok(sigma)
    }

    /// Extract the packed coordinates of triangle `t` from a flat node list.
    fn triangle_coords(nodes: &[f64], tri: &[usize; 3]) -> PdeResult<[f64; 6]> {
        let n_nodes = nodes.len() / 2;
        let mut coords = [0.0_f64; 6];
        for (local, &node) in tri.iter().enumerate() {
            if node >= n_nodes {
                return Err(PdeError::IndexOutOfBounds {
                    index: node,
                    len: n_nodes,
                });
            }
            coords[2 * local] = nodes[2 * node];
            coords[2 * local + 1] = nodes[2 * node + 1];
        }
        Ok(coords)
    }

    /// Assemble the global stiffness matrix `K` (dense, row-major `[2N×2N]`).
    ///
    /// * `nodes` — flat `[x₀,y₀, x₁,y₁, …]`, length `2N`.
    /// * `triangles` — flat node-index triples `[a,b,c, …]`, length `3·n_tri`.
    ///
    /// # Errors
    /// * [`PdeError::InvalidParameter`] if `nodes` has odd length or there are
    ///   fewer than three nodes / no triangles.
    /// * [`PdeError::ShapeMismatch`] if `triangles.len()` is not a multiple of 3.
    /// * [`PdeError::IndexOutOfBounds`] for an out-of-range node index.
    /// * [`PdeError::SingularMatrix`] for a degenerate triangle.
    pub fn assemble_global(&self, nodes: &[f64], triangles: &[usize]) -> PdeResult<Vec<f64>> {
        if nodes.len() % 2 != 0 {
            return Err(PdeError::InvalidParameter {
                name: "nodes".into(),
                reason: format!("length must be even (x,y pairs), got {}", nodes.len()),
            });
        }
        let n_nodes = nodes.len() / 2;
        if n_nodes < 3 {
            return Err(PdeError::InvalidParameter {
                name: "nodes".into(),
                reason: format!("need at least 3 nodes, got {n_nodes}"),
            });
        }
        if triangles.len() % 3 != 0 {
            return Err(PdeError::ShapeMismatch {
                expected: vec![3 * (triangles.len() / 3)],
                got: vec![triangles.len()],
            });
        }
        if triangles.is_empty() {
            return Err(PdeError::InvalidParameter {
                name: "triangles".into(),
                reason: "need at least one triangle".into(),
            });
        }

        let n_dof = 2 * n_nodes;
        let mut k_global = vec![0.0_f64; n_dof * n_dof];

        for tri in triangles.chunks_exact(3) {
            let tri3 = [tri[0], tri[1], tri[2]];
            let coords = Self::triangle_coords(nodes, &tri3)?;
            let ke = self.element_stiffness(&coords)?;

            // Local DOF `2*l + comp` maps to global DOF `2*node[l] + comp`;
            // scatter the dense 6×6 element matrix into the global matrix.
            let gdof = [
                2 * tri3[0],
                2 * tri3[0] + 1,
                2 * tri3[1],
                2 * tri3[1] + 1,
                2 * tri3[2],
                2 * tri3[2] + 1,
            ];
            for a in 0..ELASTICITY_ELEM_DOFS {
                for b in 0..ELASTICITY_ELEM_DOFS {
                    k_global[gdof[a] * n_dof + gdof[b]] += ke[a * ELASTICITY_ELEM_DOFS + b];
                }
            }
        }
        Ok(k_global)
    }

    /// Solve `K u = f` for the nodal displacements.
    ///
    /// * `nodes`, `triangles` — mesh, as in [`Self::assemble_global`].
    /// * `fixed_dofs` — prescribed Dirichlet conditions `(dof_index, value)`;
    ///   the global DOF of node `n`'s x-component is `2n`, y-component `2n+1`.
    /// * `loads` — nodal forces `(dof_index, value)` added to the load vector.
    ///
    /// Dirichlet conditions are imposed by symmetric elimination: the known
    /// column is moved to the right-hand side, the row and column are zeroed,
    /// and a unit pivot fixes the constrained value. The reduced system is then
    /// solved with dense Gaussian elimination.
    ///
    /// # Errors
    /// Propagates assembly errors and returns [`PdeError::IndexOutOfBounds`] if a
    /// `fixed_dofs` / `loads` index is out of range, or [`PdeError::SingularMatrix`]
    /// when the constrained system is singular (e.g. unconstrained rigid-body modes).
    pub fn solve(
        &self,
        nodes: &[f64],
        triangles: &[usize],
        fixed_dofs: &[(usize, f64)],
        loads: &[(usize, f64)],
    ) -> PdeResult<Vec<f64>> {
        let mut k_global = self.assemble_global(nodes, triangles)?;
        let n_dof = 2 * (nodes.len() / 2);

        // Load vector.
        let mut f = vec![0.0_f64; n_dof];
        for &(dof, value) in loads {
            if dof >= n_dof {
                return Err(PdeError::IndexOutOfBounds {
                    index: dof,
                    len: n_dof,
                });
            }
            f[dof] += value;
        }

        // Apply Dirichlet conditions by symmetric elimination.
        for &(p, value) in fixed_dofs {
            if p >= n_dof {
                return Err(PdeError::IndexOutOfBounds {
                    index: p,
                    len: n_dof,
                });
            }
            // Move the known column `p` to the RHS for every row.
            for i in 0..n_dof {
                f[i] -= k_global[i * n_dof + p] * value;
            }
            // Zero row `p` and column `p`.
            for j in 0..n_dof {
                k_global[p * n_dof + j] = 0.0;
            }
            for i in 0..n_dof {
                k_global[i * n_dof + p] = 0.0;
            }
            // Unit pivot, prescribed value on the RHS.
            k_global[p * n_dof + p] = 1.0;
            f[p] = value;
        }

        gauss_solve_dense(&mut k_global, &mut f, n_dof)
    }
}

/// Test whether a row-major `n×n` matrix is symmetric within `tol`.
#[must_use]
pub fn matrix_is_symmetric(a: &[f64], n: usize, tol: f64) -> bool {
    if a.len() != n * n {
        return false;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[i * n + j] - a[j * n + i]).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> LinearElasticity2D {
        LinearElasticity2D::new(200.0, 0.3).expect("valid material")
    }

    // ── Constructor validation ───────────────────────────────────────────────

    #[test]
    fn constructor_rejects_bad_parameters() {
        assert!(LinearElasticity2D::new(-1.0, 0.3).is_err()); // e ≤ 0
        assert!(LinearElasticity2D::new(0.0, 0.3).is_err()); // e = 0
        assert!(LinearElasticity2D::new(1.0, 0.5).is_err()); // ν ≥ 1/2
        assert!(LinearElasticity2D::new(1.0, 0.7).is_err()); // ν ≥ 1/2
        assert!(LinearElasticity2D::new(1.0, -1.0).is_err()); // ν ≤ −1
        assert!(LinearElasticity2D::new(f64::NAN, 0.3).is_err());
        assert!(LinearElasticity2D::new(1.0, 0.3).is_ok());
    }

    #[test]
    fn d_matrix_is_symmetric_and_correct() {
        let m = LinearElasticity2D::new(1.0, 0.25).expect("ok");
        let d = m.plane_stress_d();
        // f = 1/(1-0.0625) = 1.0666..., g = f*0.75/2
        let f = 1.0 / (1.0 - 0.0625);
        assert!((d[0] - f).abs() < 1e-12);
        assert!((d[1] - f * 0.25).abs() < 1e-12);
        assert!((d[3] - d[1]).abs() < 1e-15); // symmetry
        assert!((d[8] - f * 0.375).abs() < 1e-12);
        assert!(d[2].abs() < 1e-15 && d[5].abs() < 1e-15);
    }

    // ── Element matrix properties ─────────────────────────────────────────────

    #[test]
    fn element_stiffness_is_symmetric() {
        let m = steel();
        let coords = [0.1, 0.2, 1.4, 0.3, 0.5, 1.1];
        let ke = m.element_stiffness(&coords).expect("ok");
        assert!(matrix_is_symmetric(&ke, ELASTICITY_ELEM_DOFS, 1e-9));
    }

    #[test]
    fn element_stiffness_rigid_body_in_kernel() {
        // A rigid translation (uₓ=1, u_y=0 at every node) produces zero forces:
        // Kₑ · u_rigid = 0.
        let m = steel();
        let coords = [0.0, 0.0, 2.0, 0.0, 0.0, 1.0];
        let ke = m.element_stiffness(&coords).expect("ok");
        let u_rigid = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        for i in 0..ELASTICITY_ELEM_DOFS {
            let mut s = 0.0;
            for j in 0..ELASTICITY_ELEM_DOFS {
                s += ke[i * ELASTICITY_ELEM_DOFS + j] * u_rigid[j];
            }
            assert!(s.abs() < 1e-9, "rigid-body force[{i}] = {s}");
        }
    }

    #[test]
    fn degenerate_triangle_errors() {
        let m = steel();
        // Collinear vertices → zero area.
        let coords = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        assert!(m.element_stiffness(&coords).is_err());
    }

    // ── Patch test (single CST element) ──────────────────────────────────────

    #[test]
    fn patch_test_uniform_strain_recovered() {
        // A linear displacement field uₓ = 0.001·x, u_y = 0 has constant strain
        // εₓₓ = 0.001, ε_yy = 0, γₓy = 0. A single CST element must reproduce it
        // exactly from the nodal displacements.
        let m = steel();
        let coords = [0.0, 0.0, 2.0, 0.0, 0.0, 1.0]; // CCW
        let eps_x = 1.0e-3;
        let ue = [
            eps_x * coords[0],
            0.0,
            eps_x * coords[2],
            0.0,
            eps_x * coords[4],
            0.0,
        ];
        let eps = LinearElasticity2D::element_strain(&coords, &ue).expect("ok");
        assert!((eps[0] - eps_x).abs() < 1e-12, "εₓₓ = {}", eps[0]);
        assert!(eps[1].abs() < 1e-12, "ε_yy = {}", eps[1]);
        assert!(eps[2].abs() < 1e-12, "γₓy = {}", eps[2]);

        // Corresponding uniaxial stress σₓₓ = E·εₓₓ/(1−ν²)·(1)  →  via D.
        let sigma = m.element_stress(&coords, &ue).expect("ok");
        let f = m.e / (1.0 - m.nu * m.nu);
        assert!((sigma[0] - f * eps_x).abs() < 1e-9);
        assert!((sigma[1] - f * m.nu * eps_x).abs() < 1e-9);
        assert!(sigma[2].abs() < 1e-9);
    }

    // ── Assembly / solve ─────────────────────────────────────────────────────

    fn unit_square() -> (Vec<f64>, Vec<usize>) {
        // 4 nodes of the unit square, two CCW triangles.
        let nodes = vec![
            0.0, 0.0, // 0
            1.0, 0.0, // 1
            1.0, 1.0, // 2
            0.0, 1.0, // 3
        ];
        let triangles = vec![0, 1, 2, 0, 2, 3];
        (nodes, triangles)
    }

    #[test]
    fn assembled_global_is_symmetric() {
        let m = steel();
        let (nodes, tris) = unit_square();
        let k = m.assemble_global(&nodes, &tris).expect("ok");
        assert!(matrix_is_symmetric(&k, 8, 1e-9));
    }

    #[test]
    fn uniaxial_tension_matches_closed_form() {
        // Unit square in uniaxial tension σ in x. The exact displacement field
        // uₓ = (σ/E)·x , u_y = −ν(σ/E)·y is linear, so CST reproduces it exactly.
        // BCs: uₓ=0 on the left edge (nodes 0,3); u_y=0 on the bottom edge
        // (nodes 0,1). Loads: consistent edge traction σ on the right edge
        // (nodes 1,2) → nodal force σ·L/2 = σ/2 in x at each.
        let e = 200.0;
        let nu = 0.3;
        let m = LinearElasticity2D::new(e, nu).expect("ok");
        let (nodes, tris) = unit_square();
        let sigma = 2.0;
        let fixed = [
            (0, 0.0), // node 0 uₓ
            (1, 0.0), // node 0 u_y
            (3, 0.0), // node 1 u_y  (dof 2*1+1)
            (6, 0.0), // node 3 uₓ  (dof 2*3)
        ];
        let loads = [(2, sigma / 2.0), (4, sigma / 2.0)]; // nodes 1,2 uₓ
        let u = m.solve(&nodes, &tris, &fixed, &loads).expect("solve");

        let ux_expected = sigma / e; // at x = 1
        let uy_expected = -nu * sigma / e; // at y = 1
        // node 1 (1,0): uₓ = σ/E, u_y = 0
        assert!((u[2] - ux_expected).abs() < 1e-9, "u1x = {}", u[2]);
        assert!(u[3].abs() < 1e-9, "u1y = {}", u[3]);
        // node 2 (1,1): uₓ = σ/E, u_y = −ν σ/E
        assert!((u[4] - ux_expected).abs() < 1e-9, "u2x = {}", u[4]);
        assert!((u[5] - uy_expected).abs() < 1e-9, "u2y = {}", u[5]);
        // node 3 (0,1): uₓ = 0, u_y = −ν σ/E
        assert!(u[6].abs() < 1e-9, "u3x = {}", u[6]);
        assert!((u[7] - uy_expected).abs() < 1e-9, "u3y = {}", u[7]);
    }

    #[test]
    fn all_dofs_fixed_gives_zero_displacement() {
        // Fully constrained body → zero displacement regardless of load.
        let m = steel();
        let (nodes, tris) = unit_square();
        let fixed: Vec<(usize, f64)> = (0..8).map(|d| (d, 0.0)).collect();
        let loads = [(2, 5.0), (5, -3.0)];
        let u = m.solve(&nodes, &tris, &fixed, &loads).expect("solve");
        for (i, &ui) in u.iter().enumerate() {
            assert!(ui.abs() < 1e-12, "u[{i}] = {ui}");
        }
    }

    #[test]
    fn nonzero_dirichlet_value_is_imposed() {
        // Prescribe a nonzero displacement and confirm it is reproduced exactly.
        let m = steel();
        let (nodes, tris) = unit_square();
        // Fix all dofs; impose uₓ = 0.01 at node 2 and 1, plus base constraints.
        let fixed = [(0, 0.0), (1, 0.0), (3, 0.0), (6, 0.0), (2, 0.01), (4, 0.01)];
        let u = m.solve(&nodes, &tris, &fixed, &[]).expect("solve");
        assert!((u[2] - 0.01).abs() < 1e-12);
        assert!((u[4] - 0.01).abs() < 1e-12);
    }

    // ── Input validation ─────────────────────────────────────────────────────

    #[test]
    fn assemble_rejects_bad_inputs() {
        let m = steel();
        // odd node length
        assert!(m.assemble_global(&[0.0, 0.0, 1.0], &[0, 1, 2]).is_err());
        // too few nodes
        assert!(
            m.assemble_global(&[0.0, 0.0, 1.0, 0.0], &[0, 1, 0])
                .is_err()
        );
        // triangles not multiple of 3
        let (nodes, _) = unit_square();
        assert!(m.assemble_global(&nodes, &[0, 1]).is_err());
        // out-of-range node index
        assert!(m.assemble_global(&nodes, &[0, 1, 99]).is_err());
    }

    #[test]
    fn solve_rejects_out_of_range_dof() {
        let m = steel();
        let (nodes, tris) = unit_square();
        assert!(m.solve(&nodes, &tris, &[(99, 0.0)], &[]).is_err());
        assert!(m.solve(&nodes, &tris, &[], &[(99, 1.0)]).is_err());
    }
}
