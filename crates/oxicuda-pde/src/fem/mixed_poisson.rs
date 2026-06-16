//! Lowest-order Raviart-Thomas (RT0) + P0 mixed finite-element method for the
//! Poisson problem written in first-order (mixed) form.
//!
//! # Formulation
//!
//! The Poisson problem `-Δu = f` is rewritten with the flux `σ = -∇u`:
//!
//! ```text
//! σ + ∇u = 0,      div σ = f      on Ω,
//! u = u_D                          on ∂Ω   (Dirichlet data, natural in the mixed form).
//! ```
//!
//! Multiplying by test functions `τ ∈ RT0` and `v ∈ P0` and integrating by parts
//! in the first equation gives the saddle-point system
//!
//! ```text
//! [ M   Bᵀ ] [σ]   [g]
//! [ B    0 ] [u] = [F],
//! ```
//!
//! where, per triangle `T` and edges `e, e'`,
//!
//! * `M_{e,e'} = ∫_T ψ_e · ψ_{e'}`            (RT0 mass, symmetric positive-definite),
//! * `B_{T,e}  = ∫_T div ψ_e`                 (P0 × div RT0 coupling),
//! * `F_T      = ∫_T f`,
//! * `g_e      = ∫_{∂Ω ∩ e} u_D (ψ_e · n)`    (Dirichlet contribution on boundary edges).
//!
//! ## RT0 basis
//!
//! The local RT0 basis associated with the edge opposite vertex `i` of triangle
//! `T = {p_0, p_1, p_2}` (signed area `|T|`) is
//!
//! ```text
//! ψ_i(x) = s_i · (|e_i| / (2 |T|)) · (x − p_i),
//! ```
//!
//! where `|e_i|` is the length of the edge opposite `p_i` and `s_i ∈ {+1, −1}` is
//! the orientation sign making the normal flux continuous across the shared edge.
//! Because `div(x − p_i) = 2`, the divergence is **constant** on `T`:
//!
//! ```text
//! div ψ_i = s_i · |e_i| / |T|.
//! ```
//!
//! This constancy makes RT0 **locally conservative**: the discrete solution
//! satisfies `∫_T div σ_h = ∫_T f` exactly, element by element (the defining
//! property of the method, verified in the tests).
//!
//! ## Orientation
//!
//! Each mesh edge is identified by its sorted vertex pair `(min, max)` with the
//! canonical direction `min → max`. The local sign `s_i` is `+1` when the local
//! edge traversal (vertex `i+1 → i+2`, cyclic) matches the canonical direction,
//! else `−1`. The outward normal used for the boundary term is the local
//! `(t_y, -t_x)` rotation of the (oriented) edge tangent, flipped to point out of
//! the element.
//!
//! ## Nullspace
//!
//! With pure-Neumann flux data on the entire boundary the pressure is determined
//! only up to a constant. The constant nullspace is removed by **pinning** the
//! first pressure DOF to zero (documented choice); Dirichlet data on any part of
//! the boundary already removes the nullspace and no pinning is applied.
//!
//! ## Linear solve
//!
//! The assembled saddle-point matrix is **symmetric indefinite**. We solve it by
//! dense Gaussian elimination with partial pivoting (reusing
//! [`crate::spectral::chebyshev::gauss_solve_dense`]). A direct solve is chosen
//! over a Schur-complement CG so the exact local-conservation identity holds to
//! round-off on the small meshes used here; the [`schur_complement`] helper is
//! also provided for callers that prefer `S = B M⁻¹ Bᵀ`.
//!
//! Reference: Boffi, Brezzi & Fortin, *Mixed Finite Element Methods and
//! Applications* (2013); Raviart & Thomas (1977).

use crate::error::{PdeError, PdeResult};
use crate::mesh::TriMesh2d;
use crate::spectral::chebyshev::gauss_solve_dense;

/// Result of the RT0/P0 mixed solve.
#[derive(Debug, Clone)]
pub struct MixedSolution {
    /// Flux DOF per global edge (coefficient of the oriented RT0 basis function).
    pub flux_per_edge: Vec<f64>,
    /// P0 pressure (the scalar `u`) per triangle.
    pub u_per_triangle: Vec<f64>,
    /// The global edge list as canonical `(min, max)` vertex pairs.
    pub edges: Vec<[usize; 2]>,
}

/// Boundary data for the mixed problem.
#[derive(Debug, Clone)]
pub enum MixedBoundary {
    /// Dirichlet value `u_D` on the whole boundary, sampled at an `(x, y)` point.
    /// Supplied as a closure evaluated at boundary-edge midpoints.
    Dirichlet(fn(f64, f64) -> f64),
    /// Pure Neumann (the flux is prescribed weakly / homogeneous); the constant
    /// pressure nullspace is pinned by fixing `u` of triangle 0 to zero.
    PureNeumannPinned,
}

/// Connectivity of the global edge structure built from the triangulation.
struct EdgeTopology {
    /// Canonical `(min, max)` vertex pair per global edge.
    edges: Vec<[usize; 2]>,
    /// For each triangle `e` and local edge `l` (0..3): the global edge index.
    tri_edge: Vec<[usize; 3]>,
    /// Number of triangles incident to each global edge (1 ⇒ boundary).
    edge_tri_count: Vec<usize>,
}

/// Local-edge `l` (opposite vertex `l`) connects local vertices `(l+1, l+2)` mod 3.
fn local_edge_vertices(l: usize) -> (usize, usize) {
    ((l + 1) % 3, (l + 2) % 3)
}

/// Build the global edge list and triangle→edge map with canonical orientation.
fn build_edges(mesh: &TriMesh2d) -> PdeResult<EdgeTopology> {
    let n_tri = mesh.n_tri();
    if n_tri == 0 {
        return Err(PdeError::EmptyMesh("mixed_poisson: no triangles".into()));
    }
    use std::collections::HashMap;
    let mut map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edges: Vec<[usize; 2]> = Vec::new();
    let mut tri_edge: Vec<[usize; 3]> = vec![[0; 3]; n_tri];
    let mut edge_tri_count: Vec<usize> = Vec::new();
    for (e, te) in tri_edge.iter_mut().enumerate() {
        let (v0, v1, v2) = mesh.tri(e)?;
        let verts = [v0, v1, v2];
        for (l, te_l) in te.iter_mut().enumerate() {
            let (a, b) = local_edge_vertices(l);
            let (va, vb) = (verts[a], verts[b]);
            let key = if va < vb { (va, vb) } else { (vb, va) };
            let gid = match map.get(&key) {
                Some(&id) => {
                    edge_tri_count[id] += 1;
                    id
                }
                None => {
                    let id = edges.len();
                    edges.push([key.0, key.1]);
                    edge_tri_count.push(1);
                    map.insert(key, id);
                    id
                }
            };
            *te_l = gid;
        }
    }
    Ok(EdgeTopology {
        edges,
        tri_edge,
        edge_tri_count,
    })
}

/// Orientation sign for local edge `l` of an element with global vertices `verts`
/// relative to the canonical `(min → max)` edge direction.
fn orientation_sign(verts: &[usize; 3], l: usize) -> f64 {
    let (a, b) = local_edge_vertices(l);
    if verts[a] < verts[b] { 1.0 } else { -1.0 }
}

/// Per-element geometry: vertex coords, signed area, edge lengths (opposite each vertex).
struct TriGeom {
    p: [[f64; 2]; 3],
    area: f64,
    edge_len: [f64; 3],
}

fn tri_geometry(mesh: &TriMesh2d, e: usize) -> PdeResult<TriGeom> {
    let (v0, v1, v2) = mesh.tri(e)?;
    let (x0, y0) = mesh.node(v0)?;
    let (x1, y1) = mesh.node(v1)?;
    let (x2, y2) = mesh.node(v2)?;
    let p = [[x0, y0], [x1, y1], [x2, y2]];
    let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if area.abs() < 1.0e-14 {
        return Err(PdeError::SingularMatrix(format!(
            "mixed_poisson: degenerate triangle {e}, area={area}"
        )));
    }
    // edge_len[l] = length of edge opposite vertex l = |p[(l+1)%3] - p[(l+2)%3]|
    let mut edge_len = [0.0_f64; 3];
    for (l, el) in edge_len.iter_mut().enumerate() {
        let (a, b) = local_edge_vertices(l);
        let dx = p[a][0] - p[b][0];
        let dy = p[a][1] - p[b][1];
        *el = (dx * dx + dy * dy).sqrt();
    }
    Ok(TriGeom { p, area, edge_len })
}

/// Evaluate the (signed, oriented) local RT0 basis function `l` at point `x`:
/// `ψ_l(x) = s_l (|e_l| / (2|T|)) (x − p_l)`. Returns the 2-vector.
fn rt0_basis(geom: &TriGeom, sign: f64, l: usize, x: f64, y: f64) -> [f64; 2] {
    let coef = sign * geom.edge_len[l] / (2.0 * geom.area);
    [coef * (x - geom.p[l][0]), coef * (y - geom.p[l][1])]
}

/// Local RT0 mass matrix `M^T_{l,m} = ∫_T ψ_l·ψ_m` (3×3, symmetric), evaluated
/// with the 3-edge-midpoint quadrature (exact for the quadratic integrand).
fn rt0_local_mass(geom: &TriGeom, signs: &[f64; 3]) -> [f64; 9] {
    // Edge midpoints: midpoint of edge opposite vertex l connects vertices (l+1,l+2).
    let mut mids = [[0.0_f64; 2]; 3];
    for (l, mid) in mids.iter_mut().enumerate() {
        let (a, b) = local_edge_vertices(l);
        *mid = [
            0.5 * (geom.p[a][0] + geom.p[b][0]),
            0.5 * (geom.p[a][1] + geom.p[b][1]),
        ];
    }
    let w = geom.area.abs() / 3.0; // midpoint rule weight (|T|/3 per node)
    let mut m = [0.0_f64; 9];
    for l in 0..3 {
        for mm in 0..3 {
            let mut s = 0.0;
            for mid in &mids {
                let (qx, qy) = (mid[0], mid[1]);
                let pl = rt0_basis(geom, signs[l], l, qx, qy);
                let pm = rt0_basis(geom, signs[mm], mm, qx, qy);
                s += w * (pl[0] * pm[0] + pl[1] * pm[1]);
            }
            m[l * 3 + mm] = s;
        }
    }
    m
}

/// Local divergence coefficients `div ψ_l = s_l |e_l| / |T|` (constant on T).
fn rt0_local_div(geom: &TriGeom, signs: &[f64; 3]) -> [f64; 3] {
    let mut d = [0.0_f64; 3];
    for l in 0..3 {
        d[l] = signs[l] * geom.edge_len[l] / geom.area;
    }
    d
}

/// Outward unit normal on local edge `l` of element with geometry `geom`.
/// The edge opposite vertex `l` connects `p_{l+1}, p_{l+2}`; the normal is the
/// rotation of the tangent, oriented to point away from `p_l`.
fn outward_normal(geom: &TriGeom, l: usize) -> [f64; 2] {
    let (a, b) = local_edge_vertices(l);
    let tx = geom.p[b][0] - geom.p[a][0];
    let ty = geom.p[b][1] - geom.p[a][1];
    let len = (tx * tx + ty * ty).sqrt();
    // candidate normal (rotate tangent by -90°)
    let mut nx = ty / len;
    let mut ny = -tx / len;
    // ensure it points away from the opposite vertex p_l
    let mx = 0.5 * (geom.p[a][0] + geom.p[b][0]);
    let my = 0.5 * (geom.p[a][1] + geom.p[b][1]);
    let to_out_x = mx - geom.p[l][0];
    let to_out_y = my - geom.p[l][1];
    if nx * to_out_x + ny * to_out_y < 0.0 {
        nx = -nx;
        ny = -ny;
    }
    [nx, ny]
}

/// Assemble and solve the RT0/P0 mixed Poisson problem.
///
/// # Arguments
/// * `mesh` — triangular mesh.
/// * `f_per_triangle` — the cell-average of the forcing `f` per triangle, so that
///   `∫_T f = f_per_triangle[T] · |T|` (a P0 representation of `f`).
/// * `boundary` — boundary data (Dirichlet closure or pinned pure-Neumann).
///
/// # Returns
/// [`MixedSolution`] with the per-edge flux DOFs, the per-triangle pressure, and
/// the global edge list.
pub fn mixed_poisson_rt0(
    mesh: &TriMesh2d,
    f_per_triangle: &[f64],
    boundary: &MixedBoundary,
) -> PdeResult<MixedSolution> {
    let n_tri = mesh.n_tri();
    if f_per_triangle.len() != n_tri {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_tri],
            got: vec![f_per_triangle.len()],
        });
    }
    let topo = build_edges(mesh)?;
    let n_edge = topo.edges.len();
    let n_dof = n_edge + n_tri; // [σ ; u]

    // Dense saddle matrix A (row-major) and RHS.
    let mut a = vec![0.0_f64; n_dof * n_dof];
    let mut rhs = vec![0.0_f64; n_dof];

    for (e, &f_e) in f_per_triangle.iter().enumerate() {
        let (v0, v1, v2) = mesh.tri(e)?;
        let verts = [v0, v1, v2];
        let geom = tri_geometry(mesh, e)?;
        let signs = [
            orientation_sign(&verts, 0),
            orientation_sign(&verts, 1),
            orientation_sign(&verts, 2),
        ];
        let gedge = topo.tri_edge[e];

        // M block (edge × edge).
        let m_loc = rt0_local_mass(&geom, &signs);
        for l in 0..3 {
            for mm in 0..3 {
                let gi = gedge[l];
                let gj = gedge[mm];
                a[gi * n_dof + gj] += m_loc[l * 3 + mm];
            }
        }
        // Coupling blocks. The mixed form of σ=−∇u, div σ=f reads
        //     M σ − Bᵀ u = −g     (first equation, integrated by parts),
        //     B σ        =  F      (divergence equation),
        // with B_{T,e} = ∫_T div ψ_e = (div ψ_e)·|T| and g_e the Dirichlet term.
        // The off-diagonal carries opposite signs, making the saddle matrix
        // antisymmetric in the coupling (the standard mixed-Poisson convention
        // that keeps the recovered pressure physical, i.e. σ=−∇u not +∇u).
        let div_loc = rt0_local_div(&geom, &signs);
        let u_row = n_edge + e; // pressure DOF index for triangle e
        for l in 0..3 {
            let gi = gedge[l];
            let b_val = div_loc[l] * geom.area; // ∫_T div ψ_l
            // Divergence-equation row: (B σ)_{e} += B_{e,l} σ_l.
            a[u_row * n_dof + gi] += b_val;
            // Flux-equation row: −(Bᵀ u)_{l} contributes −B_{e,l} u_e.
            a[gi * n_dof + u_row] -= b_val;
        }
        // RHS for the divergence equation: ∫_T f = f_avg · |T|.
        rhs[u_row] += f_e * geom.area;

        // Dirichlet boundary term: contributes −g_e to the flux-equation RHS,
        // g_e = ∫_{∂Ω∩e} u_D (ψ_e·n) for boundary edges.
        if let MixedBoundary::Dirichlet(u_d) = boundary {
            for l in 0..3 {
                let gi = gedge[l];
                if topo.edge_tri_count[gi] != 1 {
                    continue; // interior edge
                }
                // Boundary edge: integrate u_D (ψ_l · n) over the edge.
                let (av, bv) = local_edge_vertices(l);
                let mx = 0.5 * (geom.p[av][0] + geom.p[bv][0]);
                let my = 0.5 * (geom.p[av][1] + geom.p[bv][1]);
                let nrm = outward_normal(&geom, l);
                let psi = rt0_basis(&geom, signs[l], l, mx, my);
                let psi_dot_n = psi[0] * nrm[0] + psi[1] * nrm[1];
                // midpoint rule on the edge: g_e ≈ len * u_D(mid) * (ψ·n)(mid);
                // the flux equation carries −g_e on its RHS.
                let len = geom.edge_len[l];
                let g_e = len * u_d(mx, my) * psi_dot_n;
                rhs[gi] -= g_e;
            }
        }
    }

    // Nullspace handling: pin pressure of triangle 0 for pure-Neumann.
    if matches!(boundary, MixedBoundary::PureNeumannPinned) {
        let pin = n_edge; // u-DOF of triangle 0
        // Replace the pinned row with the identity equation u_0 = 0.
        for j in 0..n_dof {
            a[pin * n_dof + j] = 0.0;
        }
        a[pin * n_dof + pin] = 1.0;
        rhs[pin] = 0.0;
    }

    let sol = gauss_solve_dense(&mut a, &mut rhs, n_dof)?;
    let flux_per_edge = sol[..n_edge].to_vec();
    let u_per_triangle = sol[n_edge..].to_vec();

    Ok(MixedSolution {
        flux_per_edge,
        u_per_triangle,
        edges: topo.edges,
    })
}

/// Per-element discrete divergence `div σ_h|_T` for a computed solution.
///
/// Returns one value per triangle: `Σ_l (div ψ_l) · σ_e[gedge_l]`. Since
/// `div ψ_l` is constant on `T`, this is `div σ_h` on `T`, and multiplying by
/// `|T|` gives `∫_T div σ_h`.
pub fn element_divergence(mesh: &TriMesh2d, sol: &MixedSolution) -> PdeResult<Vec<f64>> {
    let topo = build_edges(mesh)?;
    let n_tri = mesh.n_tri();
    let mut out = vec![0.0_f64; n_tri];
    for (e, oe) in out.iter_mut().enumerate() {
        let (v0, v1, v2) = mesh.tri(e)?;
        let verts = [v0, v1, v2];
        let geom = tri_geometry(mesh, e)?;
        let signs = [
            orientation_sign(&verts, 0),
            orientation_sign(&verts, 1),
            orientation_sign(&verts, 2),
        ];
        let div_loc = rt0_local_div(&geom, &signs);
        let gedge = topo.tri_edge[e];
        let mut d = 0.0;
        for (dl, &ge) in div_loc.iter().zip(gedge.iter()) {
            d += dl * sol.flux_per_edge[ge];
        }
        *oe = d;
    }
    Ok(out)
}

/// Reconstruct the flux vector `σ_h` at the centroid of each triangle.
///
/// Returns `(σx, σy)` pairs flattened as `[σx_0, σy_0, σx_1, σy_1, ...]`.
pub fn flux_at_centroids(mesh: &TriMesh2d, sol: &MixedSolution) -> PdeResult<Vec<f64>> {
    let topo = build_edges(mesh)?;
    let n_tri = mesh.n_tri();
    let mut out = vec![0.0_f64; 2 * n_tri];
    for e in 0..n_tri {
        let (v0, v1, v2) = mesh.tri(e)?;
        let verts = [v0, v1, v2];
        let geom = tri_geometry(mesh, e)?;
        let signs = [
            orientation_sign(&verts, 0),
            orientation_sign(&verts, 1),
            orientation_sign(&verts, 2),
        ];
        let gedge = topo.tri_edge[e];
        let cx = (geom.p[0][0] + geom.p[1][0] + geom.p[2][0]) / 3.0;
        let cy = (geom.p[0][1] + geom.p[1][1] + geom.p[2][1]) / 3.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for l in 0..3 {
            let psi = rt0_basis(&geom, signs[l], l, cx, cy);
            let coeff = sol.flux_per_edge[gedge[l]];
            sx += coeff * psi[0];
            sy += coeff * psi[1];
        }
        out[2 * e] = sx;
        out[2 * e + 1] = sy;
    }
    Ok(out)
}

/// Build the dense Schur complement `S = B M⁻¹ Bᵀ` (size `n_tri × n_tri`) for the
/// RT0/P0 system, returned row-major as `(n_tri, S)`.
///
/// Provided for callers preferring the Schur route over the direct solve used by
/// [`mixed_poisson_rt0`]. Note that the RT0 mass matrix couples all three edges
/// of a triangle, so `M` is **not** block-diagonal per edge; `M⁻¹ Bᵀ` is therefore
/// formed here by a global dense solve, column by column.
pub fn schur_complement(mesh: &TriMesh2d) -> PdeResult<(usize, Vec<f64>)> {
    let topo = build_edges(mesh)?;
    let n_edge = topo.edges.len();
    let n_tri = mesh.n_tri();

    // Global M (edge×edge) and B (tri×edge).
    let mut m = vec![0.0_f64; n_edge * n_edge];
    let mut b = vec![0.0_f64; n_tri * n_edge];
    for e in 0..n_tri {
        let (v0, v1, v2) = mesh.tri(e)?;
        let verts = [v0, v1, v2];
        let geom = tri_geometry(mesh, e)?;
        let signs = [
            orientation_sign(&verts, 0),
            orientation_sign(&verts, 1),
            orientation_sign(&verts, 2),
        ];
        let gedge = topo.tri_edge[e];
        let m_loc = rt0_local_mass(&geom, &signs);
        for l in 0..3 {
            for mm in 0..3 {
                m[gedge[l] * n_edge + gedge[mm]] += m_loc[l * 3 + mm];
            }
        }
        let div_loc = rt0_local_div(&geom, &signs);
        for l in 0..3 {
            b[e * n_edge + gedge[l]] += div_loc[l] * geom.area;
        }
    }
    // Solve M Y = Bᵀ column by column → Y = M⁻¹ Bᵀ (n_edge × n_tri).
    let mut y = vec![0.0_f64; n_edge * n_tri];
    for t in 0..n_tri {
        let mut m_copy = m.clone();
        let mut col = vec![0.0_f64; n_edge];
        for ee in 0..n_edge {
            col[ee] = b[t * n_edge + ee]; // (Bᵀ)_{ee,t} = B_{t,ee}
        }
        let yt = gauss_solve_dense(&mut m_copy, &mut col, n_edge)?;
        for ee in 0..n_edge {
            y[ee * n_tri + t] = yt[ee];
        }
    }
    // S = B Y  (n_tri × n_tri).
    let mut s = vec![0.0_f64; n_tri * n_tri];
    for i in 0..n_tri {
        for j in 0..n_tri {
            let mut acc = 0.0;
            for ee in 0..n_edge {
                acc += b[i * n_edge + ee] * y[ee * n_tri + j];
            }
            s[i * n_tri + j] = acc;
        }
    }
    Ok((n_tri, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mass_is_symmetric(mesh: &TriMesh2d) -> bool {
        for e in 0..mesh.n_tri() {
            let (v0, v1, v2) = mesh.tri(e).expect("tri");
            let verts = [v0, v1, v2];
            let geom = tri_geometry(mesh, e).expect("geom");
            let signs = [
                orientation_sign(&verts, 0),
                orientation_sign(&verts, 1),
                orientation_sign(&verts, 2),
            ];
            let m = rt0_local_mass(&geom, &signs);
            for l in 0..3 {
                for mm in 0..3 {
                    if (m[l * 3 + mm] - m[mm * 3 + l]).abs() > 1e-13 {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn local_mass_symmetric() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 4, 4).expect("ok");
        assert!(mass_is_symmetric(&mesh));
    }

    #[test]
    fn edges_shared_interior_have_two_triangles() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        let topo = build_edges(&mesh).expect("ok");
        let boundary_count = topo.edge_tri_count.iter().filter(|&&c| c == 1).count();
        let interior_count = topo.edge_tri_count.iter().filter(|&&c| c == 2).count();
        // A 2x2 cell rectangle triangulation has 8 boundary edges.
        assert_eq!(boundary_count, 8);
        assert!(interior_count > 0);
    }

    #[test]
    fn local_conservation_exact() {
        // ∫_T div σ_h = ∫_T f, element by element, to round-off.
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 5, 5).expect("ok");
        let n_tri = mesh.n_tri();
        // Use a spatially varying f (cell averages) and homogeneous Dirichlet.
        let mut f = vec![0.0; n_tri];
        for (e, fe) in f.iter_mut().enumerate() {
            *fe = 1.0 + (e as f64) * 0.137;
        }
        let bc = MixedBoundary::Dirichlet(|_, _| 0.0);
        let sol = mixed_poisson_rt0(&mesh, &f, &bc).expect("ok");
        let div = element_divergence(&mesh, &sol).expect("ok");
        for e in 0..n_tri {
            let lhs = div[e] * mesh.area(e).expect("area"); // ∫_T div σ_h
            let rhs = f[e] * mesh.area(e).expect("area"); // ∫_T f
            assert!((lhs - rhs).abs() < 1e-10, "elem {e}: ∫div={lhs}, ∫f={rhs}");
        }
    }

    #[test]
    fn manufactured_p0_and_flux_converge_order1() {
        // u = sin(pi x) sin(pi y), f = -Δu = 2π² u, homogeneous Dirichlet on [0,1]².
        // P0 u_h converges O(h) in L2; flux σ_h = -∇u converges O(h) in L2.
        let pi = std::f64::consts::PI;
        let u_exact = |x: f64, y: f64| (pi * x).sin() * (pi * y).sin();
        let gradu = |x: f64, y: f64| {
            [
                pi * (pi * x).cos() * (pi * y).sin(),
                pi * (pi * x).sin() * (pi * y).cos(),
            ]
        };
        let run = |n: usize| -> (f64, f64, f64) {
            let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, n, n).expect("ok");
            let n_tri = mesh.n_tri();
            // f cell averages via centroid sampling of 2π² u.
            let mut f = vec![0.0; n_tri];
            for (e, fe) in f.iter_mut().enumerate() {
                let (a, b, c) = mesh.tri(e).expect("tri");
                let (xa, ya) = mesh.node(a).expect("n");
                let (xb, yb) = mesh.node(b).expect("n");
                let (xc, yc) = mesh.node(c).expect("n");
                let cx = (xa + xb + xc) / 3.0;
                let cy = (ya + yb + yc) / 3.0;
                *fe = 2.0 * pi * pi * u_exact(cx, cy);
            }
            let bc = MixedBoundary::Dirichlet(|_, _| 0.0);
            let sol = mixed_poisson_rt0(&mesh, &f, &bc).expect("ok");
            // L2 error of P0 pressure (compare to u at centroid).
            let mut e_u2 = 0.0;
            let mut e_s2 = 0.0;
            let flux = flux_at_centroids(&mesh, &sol).expect("ok");
            for e in 0..n_tri {
                let (a, b, c) = mesh.tri(e).expect("tri");
                let (xa, ya) = mesh.node(a).expect("n");
                let (xb, yb) = mesh.node(b).expect("n");
                let (xc, yc) = mesh.node(c).expect("n");
                let cx = (xa + xb + xc) / 3.0;
                let cy = (ya + yb + yc) / 3.0;
                let area = mesh.area(e).expect("area");
                let du = sol.u_per_triangle[e] - u_exact(cx, cy);
                e_u2 += du * du * area;
                // σ = -∇u
                let g = gradu(cx, cy);
                let dsx = flux[2 * e] - (-g[0]);
                let dsy = flux[2 * e + 1] - (-g[1]);
                e_s2 += (dsx * dsx + dsy * dsy) * area;
            }
            let h = 1.0 / (n as f64 - 1.0);
            (h, e_u2.sqrt(), e_s2.sqrt())
        };
        let (h1, eu1, es1) = run(5);
        let (h2, eu2, es2) = run(9);
        let rate_u = (eu1 / eu2).ln() / (h1 / h2).ln();
        let rate_s = (es1 / es2).ln() / (h1 / h2).ln();
        assert!(rate_u > 0.85, "P0 u rate {rate_u} (eu1={eu1} eu2={eu2})");
        assert!(rate_s > 0.85, "flux rate {rate_s} (es1={es1} es2={es2})");
    }

    #[test]
    fn constant_pressure_pure_neumann_pinned() {
        // f = 0 everywhere with pinned pressure ⇒ u_h ≡ 0, σ_h ≡ 0.
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 4, 4).expect("ok");
        let n_tri = mesh.n_tri();
        let f = vec![0.0; n_tri];
        let sol = mixed_poisson_rt0(&mesh, &f, &MixedBoundary::PureNeumannPinned).expect("ok");
        for &u in &sol.u_per_triangle {
            assert!(u.abs() < 1e-10, "u={u}");
        }
        for &s in &sol.flux_per_edge {
            assert!(s.abs() < 1e-10, "σ={s}");
        }
    }

    #[test]
    fn schur_is_symmetric_positive() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        let (n, s) = schur_complement(&mesh).expect("ok");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (s[i * n + j] - s[j * n + i]).abs() < 1e-10,
                    "S not symmetric at ({i},{j})"
                );
            }
        }
        // diagonal positive (S is SPD up to the pressure nullspace).
        for i in 0..n {
            assert!(s[i * n + i] > 0.0);
        }
    }

    #[test]
    fn rejects_wrong_f_length() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        let bad = vec![0.0; mesh.n_tri() + 1];
        assert!(mixed_poisson_rt0(&mesh, &bad, &MixedBoundary::Dirichlet(|_, _| 0.0)).is_err());
    }
}
