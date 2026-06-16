//! 1-D nodal spectral-element method (SEM) on Gauss–Lobatto–Legendre (GLL) nodes.
//!
//! The continuous-Galerkin spectral element method partitions `[a, b]` into
//! `n_elem` elements, each carrying a degree-`p` Lagrange nodal basis collocated
//! at the GLL points. The GLL nodes/weights are reused from the DG module
//! ([`crate::dg::dg1d::lgl_nodes`] / [`lgl_weights`] — *LGL* and *GLL* name the
//! same Legendre–Gauss–Lobatto rule), and the dense reduced solve reuses
//! [`crate::spectral::chebyshev::gauss_solve_dense`].
//!
//! # Building blocks
//!
//! * **GLL quadrature** integrates polynomials up to degree `2p−1` exactly — the
//!   defining property of the Gauss–Lobatto rule.
//! * The **nodal differentiation matrix** `D` (built from barycentric weights of
//!   the GLL nodes) differentiates any polynomial of degree `≤ p` exactly at the
//!   nodes: `D · (xᵏ) = k xᵏ⁻¹`.
//! * The **diagonal (lumped) mass** matrix has entries equal to the GLL weights
//!   scaled by the element Jacobian, `M_aa = (h/2) w_a`; collocating the load with
//!   the same quadrature is the standard GLL-lumped SEM.
//! * The **stiffness** matrix `K_ab = ∫ ℓ_a' ℓ_b' dx` is assembled exactly via
//!   `K = Dᵀ W D` on the reference element (integrand degree `2p−2 ≤ 2p−1`).
//!
//! # Poisson solve
//!
//! [`SpectralElementMesh1d::solve_poisson_dirichlet`] assembles the global
//! stiffness/mass, imposes Dirichlet data at the two end nodes, and solves the
//! reduced interior system. For smooth data the error decays **spectrally** as
//! the order `p` grows.
//!
//! Reference: Deville, Fischer & Mund, *High-Order Methods for Incompressible
//! Fluid Flow*, CUP 2002, Chapters 2–4; Canuto et al., *Spectral Methods*, 2006.

use crate::dg::dg1d::{lgl_nodes, lgl_weights};
use crate::error::{PdeError, PdeResult};
use crate::spectral::chebyshev::gauss_solve_dense;

/// Gauss–Lobatto–Legendre nodes on `[-1, 1]` for order `p` (returns `p+1`
/// ascending nodes). Thin reuse wrapper over [`crate::dg::dg1d::lgl_nodes`].
pub fn gll_nodes(p: usize) -> PdeResult<Vec<f64>> {
    lgl_nodes(p)
}

/// Gauss–Lobatto–Legendre quadrature weights on `[-1, 1]` for order `p`
/// (returns `p+1` weights). Thin reuse wrapper over [`crate::dg::dg1d::lgl_weights`].
pub fn gll_weights(p: usize) -> PdeResult<Vec<f64>> {
    lgl_weights(p)
}

/// Barycentric interpolation weights `λ_j = 1 / Π_{k≠j} (x_j − x_k)` of a node set.
fn barycentric_weights(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut w = vec![1.0_f64; n];
    for j in 0..n {
        let mut prod = 1.0;
        for (k, &xk) in x.iter().enumerate() {
            if k != j {
                prod *= x[j] - xk;
            }
        }
        w[j] = 1.0 / prod;
    }
    w
}

/// Dense nodal differentiation matrix `D[i][j] = ℓ_j'(x_i)` (row-major, `n×n`)
/// from the barycentric weights, with the negative-sum trick on the diagonal.
fn nodal_differentiation_matrix(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let bw = barycentric_weights(x);
    let mut d = vec![0.0_f64; n * n];
    for i in 0..n {
        let mut diag = 0.0;
        for j in 0..n {
            if i != j {
                let val = (bw[j] / bw[i]) / (x[i] - x[j]);
                d[i * n + j] = val;
                diag -= val;
            }
        }
        d[i * n + i] = diag;
    }
    d
}

/// GLL nodal basis of order `p` on the reference element `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct GllBasis {
    /// Polynomial order `p ≥ 1`.
    pub p: usize,
    /// GLL nodes (`p+1`, ascending in `[-1, 1]`).
    pub nodes: Vec<f64>,
    /// GLL quadrature weights (`p+1`).
    pub weights: Vec<f64>,
    /// Nodal differentiation matrix `D` (row-major `(p+1)×(p+1)`).
    pub diff: Vec<f64>,
}

impl GllBasis {
    /// Build the order-`p` GLL basis (nodes, weights, differentiation matrix).
    pub fn new(p: usize) -> PdeResult<Self> {
        if p == 0 {
            return Err(PdeError::InvalidOrder {
                order: p,
                reason: "spectral element order p>=1 required".into(),
            });
        }
        let nodes = gll_nodes(p)?;
        let weights = gll_weights(p)?;
        let diff = nodal_differentiation_matrix(&nodes);
        Ok(Self {
            p,
            nodes,
            weights,
            diff,
        })
    }

    /// Number of nodal points `p + 1`.
    #[must_use]
    pub fn npts(&self) -> usize {
        self.p + 1
    }

    /// Apply the differentiation matrix: returns `D · values` (reference-element
    /// derivative sampled at the GLL nodes).
    pub fn differentiate(&self, values: &[f64]) -> PdeResult<Vec<f64>> {
        let n = self.npts();
        if values.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![values.len()],
            });
        }
        let mut out = vec![0.0; n];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (j, &vj) in values.iter().enumerate() {
                s += self.diff[i * n + j] * vj;
            }
            *out_i = s;
        }
        Ok(out)
    }

    /// GLL quadrature of nodal samples on the reference element: `Σ w_i v_i`
    /// (exact for integrands of polynomial degree `≤ 2p−1`).
    pub fn integrate_reference(&self, values: &[f64]) -> PdeResult<f64> {
        let n = self.npts();
        if values.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![values.len()],
            });
        }
        Ok(self
            .weights
            .iter()
            .zip(values.iter())
            .map(|(&w, &v)| w * v)
            .sum())
    }

    /// Reference-element stiffness `K_ab = ∫_{-1}^1 ℓ_a'(ξ) ℓ_b'(ξ) dξ = (Dᵀ W D)_ab`
    /// (row-major `(p+1)×(p+1)`). Exact: the integrand has degree `2p−2`.
    #[must_use]
    pub fn reference_stiffness(&self) -> Vec<f64> {
        let n = self.npts();
        let mut k = vec![0.0_f64; n * n];
        for a in 0..n {
            for b in 0..n {
                let mut s = 0.0;
                for m in 0..n {
                    s += self.weights[m] * self.diff[m * n + a] * self.diff[m * n + b];
                }
                k[a * n + b] = s;
            }
        }
        k
    }

    /// Reference-element diagonal (lumped) mass entries, equal to the GLL weights.
    #[must_use]
    pub fn reference_mass_diag(&self) -> Vec<f64> {
        self.weights.clone()
    }
}

/// Continuous-Galerkin spectral-element mesh on `[x_left, x_right]` with
/// `n_elem` equal-size elements of order `p`.
#[derive(Debug, Clone)]
pub struct SpectralElementMesh1d {
    /// Number of elements (`≥ 1`).
    pub n_elem: usize,
    /// Polynomial order per element (`≥ 1`).
    pub p: usize,
    /// Left domain endpoint.
    pub x_left: f64,
    /// Right domain endpoint.
    pub x_right: f64,
    /// Uniform element size `(x_right − x_left) / n_elem`.
    pub element_size: f64,
    /// Reference GLL basis shared by every element.
    pub basis: GllBasis,
    /// Global node coordinates (length `n_elem·p + 1`, ascending, shared interfaces).
    pub nodes: Vec<f64>,
}

impl SpectralElementMesh1d {
    /// Build the mesh, validating `n_elem ≥ 1`, `p ≥ 1`, `x_right > x_left`.
    pub fn new(n_elem: usize, p: usize, x_left: f64, x_right: f64) -> PdeResult<Self> {
        if n_elem == 0 {
            return Err(PdeError::EmptyMesh("spectral element: n_elem=0".into()));
        }
        if !(x_left.is_finite() && x_right.is_finite()) || x_right <= x_left {
            return Err(PdeError::InvalidGrid(format!(
                "spectral element requires finite x_right > x_left, got [{x_left}, {x_right}]"
            )));
        }
        let basis = GllBasis::new(p)?;
        let element_size = (x_right - x_left) / n_elem as f64;
        let n_global = n_elem * p + 1;
        let mut nodes = vec![0.0; n_global];
        for e in 0..n_elem {
            let xl = x_left + e as f64 * element_size;
            for a in 0..=p {
                let x = xl + 0.5 * element_size * (basis.nodes[a] + 1.0);
                nodes[e * p + a] = x;
            }
        }
        Ok(Self {
            n_elem,
            p,
            x_left,
            x_right,
            element_size,
            basis,
            nodes,
        })
    }

    /// Number of global degrees of freedom `n_elem·p + 1`.
    #[must_use]
    pub fn n_dofs(&self) -> usize {
        self.n_elem * self.p + 1
    }

    /// Global index of local node `a` (`0..=p`) on element `e`.
    pub fn local_to_global(&self, e: usize, a: usize) -> PdeResult<usize> {
        if e >= self.n_elem || a > self.p {
            return Err(PdeError::IndexOutOfBounds {
                index: e * (self.p + 1) + a,
                len: self.n_dofs(),
            });
        }
        Ok(e * self.p + a)
    }

    /// Assembled diagonal (lumped) global mass vector, `M_I = Σ (h/2) w_a`.
    #[must_use]
    pub fn lumped_mass(&self) -> Vec<f64> {
        let jac = 0.5 * self.element_size; // dx/dξ
        let mut mass = vec![0.0; self.n_dofs()];
        for e in 0..self.n_elem {
            for a in 0..=self.p {
                mass[e * self.p + a] += jac * self.basis.weights[a];
            }
        }
        mass
    }

    /// Assembled dense global stiffness matrix (row-major `N×N`, `N = n_dofs`).
    ///
    /// Each element contributes `(2/h) K_ref` scattered through the local→global
    /// node map (shared interface nodes accumulate from both neighbours).
    #[must_use]
    pub fn assemble_stiffness(&self) -> Vec<f64> {
        let n1 = self.p + 1;
        let n = self.n_dofs();
        let k_ref = self.basis.reference_stiffness();
        let scale = 2.0 / self.element_size; // (2/h) maps reference to physical
        let mut k = vec![0.0_f64; n * n];
        for e in 0..self.n_elem {
            let base = e * self.p;
            for a in 0..n1 {
                let ia = base + a;
                for b in 0..n1 {
                    let jb = base + b;
                    k[ia * n + jb] += scale * k_ref[a * n1 + b];
                }
            }
        }
        k
    }

    /// Solve `−u''(x) = f(x)` on `[x_left, x_right]` with Dirichlet data
    /// `u(x_left) = u_left`, `u(x_right) = u_right`.
    ///
    /// `f_nodal` are the forcing samples at the global GLL nodes. Returns the
    /// solution at those nodes.
    pub fn solve_poisson_dirichlet(
        &self,
        f_nodal: &[f64],
        u_left: f64,
        u_right: f64,
    ) -> PdeResult<Vec<f64>> {
        let n = self.n_dofs();
        if f_nodal.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![f_nodal.len()],
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "Poisson solve needs >= 3 global nodes, got {n} (raise n_elem or p)"
            )));
        }
        if !(u_left.is_finite() && u_right.is_finite()) {
            return Err(PdeError::InvalidParameter {
                name: "dirichlet".into(),
                reason: "boundary values must be finite".into(),
            });
        }
        let k = self.assemble_stiffness();
        let mass = self.lumped_mass();
        // GLL-collocated load b_I = M_I · f_I (weak form  K u = M f).
        let b: Vec<f64> = mass
            .iter()
            .zip(f_nodal.iter())
            .map(|(&m, &f)| m * f)
            .collect();

        // Reduce to the interior (drop the two Dirichlet nodes 0 and n-1).
        let mi = n - 2;
        let mut a_int = vec![0.0_f64; mi * mi];
        let mut b_int = vec![0.0_f64; mi];
        for ii in 0..mi {
            let i = ii + 1;
            // Eliminate the known boundary columns into the RHS.
            let mut rhs = b[i] - k[i * n] * u_left - k[i * n + (n - 1)] * u_right;
            if !rhs.is_finite() {
                rhs = 0.0;
            }
            b_int[ii] = rhs;
            for jj in 0..mi {
                a_int[ii * mi + jj] = k[i * n + (jj + 1)];
            }
        }
        let u_int = gauss_solve_dense(&mut a_int, &mut b_int, mi)?;
        let mut u = vec![0.0; n];
        u[0] = u_left;
        u[n - 1] = u_right;
        u[1..(mi + 1)].copy_from_slice(&u_int);
        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Analytic `∫_{-1}^1 x^k dx`.
    fn monomial_integral(k: usize) -> f64 {
        if k % 2 == 1 {
            0.0
        } else {
            2.0 / (k as f64 + 1.0)
        }
    }

    #[test]
    fn gll_quadrature_exact_up_to_degree_2p_minus_1() {
        for p in 1..=6 {
            let basis = GllBasis::new(p).expect("basis");
            for k in 0..=(2 * p - 1) {
                let samples: Vec<f64> = basis.nodes.iter().map(|&x| x.powi(k as i32)).collect();
                let approx = basis.integrate_reference(&samples).expect("quad");
                let exact = monomial_integral(k);
                assert!(
                    (approx - exact).abs() < 1.0e-11,
                    "p={p} k={k}: {approx} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn gll_quadrature_not_exact_at_degree_2p() {
        // Sharpness: degree 2p is *not* integrated exactly (defining limit of GLL).
        let p = 3;
        let basis = GllBasis::new(p).expect("basis");
        let k = 2 * p; // degree 6
        let samples: Vec<f64> = basis.nodes.iter().map(|&x| x.powi(k as i32)).collect();
        let approx = basis.integrate_reference(&samples).expect("quad");
        let exact = monomial_integral(k);
        assert!(
            (approx - exact).abs() > 1.0e-6,
            "should be inexact: {approx} vs {exact}"
        );
    }

    #[test]
    fn differentiation_matrix_is_exact_on_monomials() {
        for p in 1..=6 {
            let basis = GllBasis::new(p).expect("basis");
            for k in 0..=p {
                let v: Vec<f64> = basis.nodes.iter().map(|&x| x.powi(k as i32)).collect();
                let dv = basis.differentiate(&v).expect("diff");
                for (i, &xi) in basis.nodes.iter().enumerate() {
                    let exact = if k == 0 {
                        0.0
                    } else {
                        k as f64 * xi.powi(k as i32 - 1)
                    };
                    assert!(
                        (dv[i] - exact).abs() < 1.0e-9,
                        "p={p} k={k} i={i}: {} vs {exact}",
                        dv[i]
                    );
                }
            }
        }
    }

    #[test]
    fn mass_entries_equal_gll_weights() {
        // Reference lumped mass entries are exactly the GLL weights.
        let p = 4;
        let basis = GllBasis::new(p).expect("basis");
        let w = gll_weights(p).expect("weights");
        let m = basis.reference_mass_diag();
        assert_eq!(m.len(), w.len());
        for (mi, wi) in m.iter().zip(w.iter()) {
            assert!((mi - wi).abs() < 1.0e-14);
        }
        // On a physical element the entries scale by the Jacobian h/2.
        let mesh = SpectralElementMesh1d::new(1, p, 0.0, 2.0).expect("mesh");
        let jac = 0.5 * mesh.element_size; // = 1.0 here
        let mass = mesh.lumped_mass();
        for a in 0..=p {
            assert!((mass[a] - jac * w[a]).abs() < 1.0e-13);
        }
    }

    #[test]
    fn reference_stiffness_rows_sum_to_zero() {
        // K·1 = 0 because the derivative of a constant vanishes.
        let basis = GllBasis::new(5).expect("basis");
        let n = basis.npts();
        let k = basis.reference_stiffness();
        for a in 0..n {
            let s: f64 = (0..n).map(|b| k[a * n + b]).sum();
            assert!(s.abs() < 1.0e-10, "row {a} sum {s}");
        }
    }

    #[test]
    fn poisson_reproduces_quadratic_exactly() {
        // −u''=2 on [-1,1], u(±1)=0  ⇒  u = 1 − x²; in the SEM space for p≥2.
        let mesh = SpectralElementMesh1d::new(3, 3, -1.0, 1.0).expect("mesh");
        let f: Vec<f64> = vec![2.0; mesh.n_dofs()];
        let u = mesh.solve_poisson_dirichlet(&f, 0.0, 0.0).expect("solve");
        for (i, &ui) in u.iter().enumerate() {
            let x = mesh.nodes[i];
            let exact = 1.0 - x * x;
            assert!(
                (ui - exact).abs() < 1.0e-10,
                "node {i} x={x}: {ui} vs {exact}"
            );
        }
    }

    #[test]
    fn poisson_converges_spectrally_in_order() {
        // −u''=π² sin(πx) on [0,1], u(0)=u(1)=0  ⇒  u = sin(πx).
        // Raising p from 4 to 8 collapses the error by orders of magnitude.
        let solve_err = |p: usize| -> f64 {
            let mesh = SpectralElementMesh1d::new(2, p, 0.0, 1.0).expect("mesh");
            let f: Vec<f64> = mesh
                .nodes
                .iter()
                .map(|&x| PI * PI * (PI * x).sin())
                .collect();
            let u = mesh.solve_poisson_dirichlet(&f, 0.0, 0.0).expect("solve");
            u.iter()
                .enumerate()
                .map(|(i, &ui)| (ui - (PI * mesh.nodes[i]).sin()).abs())
                .fold(0.0_f64, f64::max)
        };
        let err4 = solve_err(4);
        let err8 = solve_err(8);
        assert!(err8 < 1.0e-6, "p=8 error {err8} should be tiny");
        assert!(
            err4 / err8 > 50.0,
            "spectral convergence: err4={err4}, err8={err8}, ratio={}",
            err4 / err8
        );
    }

    #[test]
    fn poisson_multi_element_matches_smooth_solution() {
        // Several elements at moderate order also resolve sin(πx) accurately.
        let mesh = SpectralElementMesh1d::new(6, 4, 0.0, 1.0).expect("mesh");
        let f: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|&x| PI * PI * (PI * x).sin())
            .collect();
        let u = mesh.solve_poisson_dirichlet(&f, 0.0, 0.0).expect("solve");
        let err = u
            .iter()
            .enumerate()
            .map(|(i, &ui)| (ui - (PI * mesh.nodes[i]).sin()).abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 1.0e-7, "multi-element error {err}");
        assert!(u.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mesh_global_nodes_are_ascending_and_shared() {
        let mesh = SpectralElementMesh1d::new(3, 2, 0.0, 3.0).expect("mesh");
        assert_eq!(mesh.n_dofs(), 7);
        assert!((mesh.nodes[0] - 0.0).abs() < 1.0e-14);
        assert!((mesh.nodes[6] - 3.0).abs() < 1.0e-14);
        for w in mesh.nodes.windows(2) {
            assert!(w[1] > w[0], "nodes must be strictly increasing");
        }
        // The shared interface between element 0 and 1 is a single global node.
        assert_eq!(
            mesh.local_to_global(0, 2).expect("g"),
            mesh.local_to_global(1, 0).expect("g")
        );
    }

    #[test]
    fn construction_validates_inputs() {
        assert!(GllBasis::new(0).is_err());
        assert!(SpectralElementMesh1d::new(0, 3, 0.0, 1.0).is_err());
        assert!(SpectralElementMesh1d::new(2, 3, 1.0, 1.0).is_err());
        let mesh = SpectralElementMesh1d::new(2, 3, 0.0, 1.0).expect("mesh");
        assert!(mesh.solve_poisson_dirichlet(&[0.0; 3], 0.0, 0.0).is_err()); // wrong length
    }
}
