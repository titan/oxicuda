//! Global mass + stiffness matrix assembly for P1 triangles.

use crate::error::PdeResult;
use crate::fem::p1_triangle::{p1_local_mass, p1_local_stiffness};
use crate::mesh::TriMesh2d;
use crate::solver::sparse::SparseCsr;

/// Assembled global stiffness K, mass M, and dimension `n_nodes`.
#[derive(Debug, Clone)]
pub struct FemAssembly {
    pub stiffness: SparseCsr,
    pub mass: SparseCsr,
    pub n_nodes: usize,
}

/// Assemble the global stiffness and mass matrices for the given triangular mesh.
pub fn assemble_mass_stiffness(mesh: &TriMesh2d) -> PdeResult<FemAssembly> {
    let n = mesh.n_nodes();
    // Build dense maps first (n is small) then convert to CSR.
    let mut k_dense = vec![0.0_f64; n * n];
    let mut m_dense = vec![0.0_f64; n * n];
    for e in 0..mesh.n_tri() {
        let (a, b, c) = mesh.tri(e)?;
        let (xa, ya) = mesh.node(a)?;
        let (xb, yb) = mesh.node(b)?;
        let (xc, yc) = mesh.node(c)?;
        let k_local = p1_local_stiffness(xa, ya, xb, yb, xc, yc)?;
        let m_local = p1_local_mass(xa, ya, xb, yb, xc, yc)?;
        let idx = [a, b, c];
        for i in 0..3 {
            for j in 0..3 {
                k_dense[idx[i] * n + idx[j]] += k_local[i * 3 + j];
                m_dense[idx[i] * n + idx[j]] += m_local[i * 3 + j];
            }
        }
    }
    let stiffness = dense_to_csr(&k_dense, n)?;
    let mass = dense_to_csr(&m_dense, n)?;
    Ok(FemAssembly {
        stiffness,
        mass,
        n_nodes: n,
    })
}

fn dense_to_csr(dense: &[f64], n: usize) -> PdeResult<SparseCsr> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    let tol = 1.0e-15;
    for i in 0..n {
        for j in 0..n {
            let v = dense[i * n + j];
            if v.abs() > tol || i == j {
                cols.push(j);
                vals.push(v);
            }
        }
        row_ptr.push(cols.len());
    }
    SparseCsr::new(n, n, row_ptr, cols, vals)
}

/// Build the FEM load vector `b_i = integral f * phi_i` using
/// the lumped centroid rule: `b_i += (Area/3) * f(centroid)`.
pub fn assemble_load_centroid<F>(mesh: &TriMesh2d, f: F) -> PdeResult<Vec<f64>>
where
    F: Fn(f64, f64) -> f64,
{
    let n = mesh.n_nodes();
    let mut b = vec![0.0; n];
    for e in 0..mesh.n_tri() {
        let (a, b_idx, c) = mesh.tri(e)?;
        let (xa, ya) = mesh.node(a)?;
        let (xb, yb) = mesh.node(b_idx)?;
        let (xc, yc) = mesh.node(c)?;
        let area = mesh.area(e)?;
        let xc_centroid = (xa + xb + xc) / 3.0;
        let yc_centroid = (ya + yb + yc) / 3.0;
        let f_val = f(xc_centroid, yc_centroid);
        let contribution = area * f_val / 3.0;
        b[a] += contribution;
        b[b_idx] += contribution;
        b[c] += contribution;
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_unit_square_sizes() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        let fa = assemble_mass_stiffness(&mesh).expect("ok");
        assert_eq!(fa.n_nodes, 9);
        assert_eq!(fa.stiffness.n_rows, 9);
        assert_eq!(fa.mass.n_rows, 9);
    }

    #[test]
    fn assemble_mass_total_equals_area() {
        // Sum of all mass entries equals the total area of the domain
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 2.0, 3, 4).expect("ok");
        let fa = assemble_mass_stiffness(&mesh).expect("ok");
        let total: f64 = fa.mass.vals.iter().sum();
        // total area = 1.0 * 2.0 = 2.0; sum_ij M_ij = area
        assert!((total - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn assemble_load_centroid_constant() {
        let mesh = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        let b = assemble_load_centroid(&mesh, |_, _| 1.0).expect("ok");
        let total: f64 = b.iter().sum();
        // integral of 1 over unit square = 1
        assert!((total - 1.0).abs() < 1.0e-10);
    }
}
