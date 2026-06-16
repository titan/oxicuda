//! Bassi–Rebay-2 (BR2) Discontinuous Galerkin scheme for 1D elliptic problems.
//!
//! Solves the model Poisson problem
//!
//! ```text
//!     -u''(x) = f(x)   on (0, 1),
//!      u(0) = g_l,  u(1) = g_r   (Dirichlet)
//! ```
//!
//! using a *discontinuous* nodal polynomial basis of degree `p` on each element.
//! The discretisation follows the second scheme of Bassi & Rebay (J. Comput.
//! Phys. 131, 1997) as cast in the unified primal framework of Arnold, Brezzi,
//! Cockburn & Marini (SIAM J. Numer. Anal. 39, 2002).
//!
//! # Bilinear form
//!
//! For broken polynomial spaces `V_h`, the BR2 primal bilinear form is
//!
//! ```text
//!   a_h(u, v) = Σ_K ∫_K u' v' dx
//!             - Σ_e ∫_e ( {u'} [v] + {v'} [u] ) ds            (consistency)
//!             + Σ_e η_e ∫_K(e) r_e([u]) · r_e([v]) dx         (BR2 stabilisation)
//! ```
//!
//! where, on an interior face `e` with unit normal `n`, the average and jump are
//! `{w} = ½(w⁻ + w⁺)` and `[w] = w⁻ n⁻ + w⁺ n⁺ = (w⁻ − w⁺)` (with the standard
//! left-normal convention in 1D). The *local lifting operator* `r_e([u]) ∈ V_h`
//! is defined elementwise by
//!
//! ```text
//!   ∫_Ω r_e([u]) τ dx = - ∫_e [u] {τ} ds        for all τ ∈ V_h,
//! ```
//! supported on the (one or two) elements adjacent to `e`. The penalty factor
//! `η_e` must satisfy `η_e ≥ n_faces` (number of faces of the adjacent elements,
//! i.e. 2 in 1D interior, 1 at a boundary endpoint of a single element) for the
//! form to be coercive (ABCM 2002, Lemma 7.2). We default to `η = p_faces + 1`
//! with `p_faces = 2`, a safe choice on uniform 1D meshes.
//!
//! Because `r_e` is supported on the adjacent elements only, the global stiffness
//! matrix is sparse with element-block-tridiagonal structure; it is symmetric
//! positive-definite (SPD), and we solve it with a dense Cholesky factorisation.
//!
//! # Basis
//!
//! On the reference element `[-1, 1]` we use the nodal Lagrange basis through
//! `p + 1` Legendre–Gauss–Lobatto points (re-using [`crate::dg::dg1d::lgl_nodes`]).
//! Volume and face integrals are evaluated with a Legendre–Gauss quadrature rule
//! exact for degree `2p` polynomials, so all products appearing in `a_h` (which
//! are at most degree `2p`) are integrated exactly.

use crate::dg::dg1d::lgl_nodes;
use crate::error::{PdeError, PdeResult};

/// Number of geometric faces used in the coercivity bound for `η`.
///
/// In 1D every element has two faces; ABCM-2002 requires `η ≥ n_faces`.
pub const BR2_FACES_PER_ELEMENT: usize = 2;

/// Default BR2 penalty: `BR2_FACES_PER_ELEMENT + 1`, strictly above the
/// coercivity threshold so the discrete form is uniformly stable.
pub const DEFAULT_BR2_PENALTY: f64 = (BR2_FACES_PER_ELEMENT + 1) as f64;

/// A BR2 elliptic DG discretisation of `-u'' = f` on `[x_left, x_right]`.
///
/// The mesh is uniform with `n_elem` elements; each element carries a degree-`p`
/// discontinuous nodal basis (`p + 1` local DOFs). Total DOF count is
/// `n_elem * (p + 1)`.
#[derive(Debug, Clone)]
pub struct Br2Elliptic {
    /// Number of elements.
    pub n_elem: usize,
    /// Polynomial degree per element.
    pub p: usize,
    /// Left domain boundary.
    pub x_left: f64,
    /// Right domain boundary.
    pub x_right: f64,
    /// Uniform element width `h = (x_right - x_left) / n_elem`.
    pub h: f64,
    /// BR2 stabilisation penalty `η`.
    pub eta: f64,
    /// Reference LGL interpolation nodes, length `p + 1`.
    nodes: Vec<f64>,
    /// Gauss quadrature points on `[-1, 1]`.
    quad_x: Vec<f64>,
    /// Gauss quadrature weights on `[-1, 1]`.
    quad_w: Vec<f64>,
    /// Lagrange basis values at quadrature points: `phi[q][i]`.
    phi_q: Vec<Vec<f64>>,
    /// Lagrange basis derivatives at quadrature points (reference): `dphi[q][i]`.
    dphi_q: Vec<Vec<f64>>,
    /// Reference mass matrix `M_ref[i][j] = ∫_{-1}^{1} φ_i φ_j dξ` (`(p+1)²`).
    m_ref: Vec<f64>,
    /// Lagrange basis values at the left face `ξ = -1`: `phi_left[i]`.
    phi_left: Vec<f64>,
    /// Lagrange basis values at the right face `ξ = +1`: `phi_right[i]`.
    phi_right: Vec<f64>,
    /// Reference basis derivatives at the left face: `dphi_left[i]`.
    dphi_left: Vec<f64>,
    /// Reference basis derivatives at the right face: `dphi_right[i]`.
    dphi_right: Vec<f64>,
}

impl Br2Elliptic {
    /// Build a BR2 discretisation with the default (coercive) penalty.
    ///
    /// # Errors
    /// * [`PdeError::EmptyMesh`] if `n_elem == 0`.
    /// * [`PdeError::InvalidGrid`] if `x_right <= x_left`.
    /// * [`PdeError::UnsupportedDegree`] if `p == 0` or `p > 8`.
    pub fn new(n_elem: usize, p: usize, x_left: f64, x_right: f64) -> PdeResult<Self> {
        Self::with_penalty(n_elem, p, x_left, x_right, DEFAULT_BR2_PENALTY)
    }

    /// Build a BR2 discretisation with an explicit penalty `eta`.
    ///
    /// Choosing `eta < BR2_FACES_PER_ELEMENT` may break coercivity; this is
    /// intentionally allowed so callers can probe the stability boundary.
    ///
    /// # Errors
    /// As [`Br2Elliptic::new`], plus [`PdeError::InvalidParameter`] if
    /// `eta` is not finite.
    pub fn with_penalty(
        n_elem: usize,
        p: usize,
        x_left: f64,
        x_right: f64,
        eta: f64,
    ) -> PdeResult<Self> {
        if n_elem == 0 {
            return Err(PdeError::EmptyMesh("br2: n_elem = 0".into()));
        }
        if x_right <= x_left {
            return Err(PdeError::InvalidGrid(
                "br2: x_right must be > x_left".into(),
            ));
        }
        if p == 0 || p > 8 {
            return Err(PdeError::UnsupportedDegree(p));
        }
        if !eta.is_finite() {
            return Err(PdeError::InvalidParameter {
                name: "eta".into(),
                reason: "penalty must be finite".into(),
            });
        }

        let nodes = lgl_nodes(p)?;
        // Gauss rule exact to degree 2p+1 needs p+1 points; use p+2 for margin.
        let (quad_x, quad_w) = gauss_legendre(p + 2)?;

        let n_loc = p + 1;
        let n_q = quad_x.len();
        let mut phi_q = vec![vec![0.0; n_loc]; n_q];
        let mut dphi_q = vec![vec![0.0; n_loc]; n_q];
        for q in 0..n_q {
            for i in 0..n_loc {
                phi_q[q][i] = lagrange_value(&nodes, i, quad_x[q]);
                dphi_q[q][i] = lagrange_deriv(&nodes, i, quad_x[q]);
            }
        }

        // Reference mass matrix.
        let mut m_ref = vec![0.0; n_loc * n_loc];
        for q in 0..n_q {
            for i in 0..n_loc {
                for j in 0..n_loc {
                    m_ref[i * n_loc + j] += quad_w[q] * phi_q[q][i] * phi_q[q][j];
                }
            }
        }

        let mut phi_left = vec![0.0; n_loc];
        let mut phi_right = vec![0.0; n_loc];
        let mut dphi_left = vec![0.0; n_loc];
        let mut dphi_right = vec![0.0; n_loc];
        for i in 0..n_loc {
            phi_left[i] = lagrange_value(&nodes, i, -1.0);
            phi_right[i] = lagrange_value(&nodes, i, 1.0);
            dphi_left[i] = lagrange_deriv(&nodes, i, -1.0);
            dphi_right[i] = lagrange_deriv(&nodes, i, 1.0);
        }

        let h = (x_right - x_left) / n_elem as f64;

        Ok(Self {
            n_elem,
            p,
            x_left,
            x_right,
            h,
            eta,
            nodes,
            quad_x,
            quad_w,
            phi_q,
            dphi_q,
            m_ref,
            phi_left,
            phi_right,
            dphi_left,
            dphi_right,
        })
    }

    /// Local DOFs per element (`p + 1`).
    #[must_use]
    pub fn n_loc(&self) -> usize {
        self.p + 1
    }

    /// Total global DOFs (`n_elem * (p + 1)`).
    #[must_use]
    pub fn n_dofs(&self) -> usize {
        self.n_elem * (self.p + 1)
    }

    /// Physical coordinate of local node `i` on element `e`.
    fn node_x(&self, e: usize, i: usize) -> f64 {
        let xl = self.x_left + e as f64 * self.h;
        xl + 0.5 * self.h * (self.nodes[i] + 1.0)
    }

    /// Physical coordinates of all DOFs (row-major: element-major, then local).
    #[must_use]
    pub fn dof_coords(&self) -> Vec<f64> {
        let mut coords = Vec::with_capacity(self.n_dofs());
        for e in 0..self.n_elem {
            for i in 0..self.n_loc() {
                coords.push(self.node_x(e, i));
            }
        }
        coords
    }

    /// Assemble the global BR2 stiffness matrix `A` (dense, row-major SPD).
    ///
    /// The returned `Vec<f64>` has length `n_dofs²`.
    ///
    /// The matrix combines three contributions:
    /// 1. **Volume**: `Σ_K ∫_K φ_i' φ_j' dx`. With the affine map the physical
    ///    derivative is `(2/h) dφ/dξ`, and `dx = (h/2) dξ`, giving a net factor
    ///    `2/h` on the reference Laplacian.
    /// 2. **Consistency**: `-Σ_e ({u'}[v] + {v'}[u])`, symmetric by construction.
    /// 3. **Stabilisation**: `Σ_e η_e ∫ r_e([u]) r_e([v])`, with the local lift
    ///    `r_e` solved against the element mass matrix.
    #[must_use]
    pub fn assemble_stiffness(&self) -> Vec<f64> {
        let n = self.n_dofs();
        let n_loc = self.n_loc();
        let mut a = vec![0.0; n * n];

        // Map (element, local) -> global index.
        let gid = |e: usize, i: usize| e * n_loc + i;

        // ── 1. Volume term: (2/h) * ∫_{-1}^{1} φ_i' φ_j' dξ ─────────────────
        // physical: ∫_K u' v' dx = (2/h) ∫ dφ_i dφ_j dξ.
        let vol_factor = 2.0 / self.h;
        for e in 0..self.n_elem {
            for q in 0..self.quad_x.len() {
                for i in 0..n_loc {
                    for j in 0..n_loc {
                        a[gid(e, i) * n + gid(e, j)] +=
                            vol_factor * self.quad_w[q] * self.dphi_q[q][i] * self.dphi_q[q][j];
                    }
                }
            }
        }

        // ── 2. Interior-face consistency + BR2 stabilisation ────────────────
        // The physical derivative scale on a face is (2/h) dφ/dξ.
        let dscale = 2.0 / self.h;

        // Interior faces: between element e (left, normal +1) and e+1 (right).
        for e in 0..self.n_elem - 1 {
            let el = e; // left element, contributes its right face ξ=+1
            let er = e + 1; // right element, contributes its left face ξ=-1
            self.add_interior_face(&mut a, n, el, er, dscale);
        }

        // ── 3. Dirichlet boundary faces (weak imposition) ───────────────────
        // Left boundary of element 0 (normal -1 pointing out), and right
        // boundary of element n_elem-1 (normal +1 pointing out).
        self.add_boundary_face(&mut a, n, 0, FaceSide::Left, dscale);
        self.add_boundary_face(&mut a, n, self.n_elem - 1, FaceSide::Right, dscale);

        a
    }

    /// Accumulate the interior-face consistency and BR2 stabilisation terms.
    ///
    /// `el` is the element on the left of the face (its right face, `ξ=+1`,
    /// outward normal `+1`); `er` is on the right (its left face, `ξ=-1`,
    /// outward normal `-1`). The jump uses the left-normal convention
    /// `[w] = w⁻ − w⁺` where `⁻` is `el` and `⁺` is `er`.
    fn add_interior_face(&self, a: &mut [f64], n: usize, el: usize, er: usize, dscale: f64) {
        let n_loc = self.n_loc();
        let gid = |e: usize, i: usize| e * n_loc + i;

        // Basis traces at the face.
        //   from el: value φ_right, derivative dscale * dphi_right
        //   from er: value φ_left,  derivative dscale * dphi_left
        // Average of derivative: {u'} = ½(u'⁻ + u'⁺).
        // Jump of value:        [v] = v⁻ − v⁺.
        //
        // Consistency contribution to a(u,v):
        //   -( {u'}[v] + {v'}[u] ).
        // Expand over basis functions; both u and v range over the two adjacent
        // elements' local DOFs.

        // Helper closures producing trace value / derivative for a (side,i).
        // side 0 => left element el, side 1 => right element er.
        let val = |side: usize, i: usize| -> f64 {
            if side == 0 {
                self.phi_right[i]
            } else {
                self.phi_left[i]
            }
        };
        let der = |side: usize, i: usize| -> f64 {
            if side == 0 {
                dscale * self.dphi_right[i]
            } else {
                dscale * self.dphi_left[i]
            }
        };
        // Jump sign per side under [w] = w⁻ − w⁺: left (el) => +1, right (er) => −1.
        let jump_sign = |side: usize| -> f64 { if side == 0 { 1.0 } else { -1.0 } };
        let elem_of = |side: usize| -> usize { if side == 0 { el } else { er } };

        // Consistency: -({u'}[v] + {v'}[u]).
        // {u'} = ½ Σ_s der(s, j) U_{s,j}; [v] = Σ_s jump(s) val(s,i) V_{s,i}.
        // Contribution to A[v_index, u_index] of  -{u'}[v]  is
        //   -½ der(su,j) * jump(sv) val(sv,i).
        // Symmetrically for -{v'}[u]: -½ der(sv,i) * jump(su) val(su,j).
        for su in 0..2 {
            for sv in 0..2 {
                for i in 0..n_loc {
                    for j in 0..n_loc {
                        let row = gid(elem_of(sv), i);
                        let col = gid(elem_of(su), j);
                        let term_a = -0.5 * der(su, j) * jump_sign(sv) * val(sv, i);
                        let term_b = -0.5 * der(sv, i) * jump_sign(su) * val(su, j);
                        a[row * n + col] += term_a + term_b;
                    }
                }
            }
        }

        // BR2 stabilisation: η ∫ r_e([u]) r_e([v]) dx.
        // The local lift r_e on each adjacent element K solves
        //   M_K r = - ∫_e [u] {φ} ds      (a single face contributes ½ φ trace).
        // For a unit jump in basis function (side su, local j), the face data is
        //   rhs_K(τ) = -[φ_{su,j}] {τ} = -jump(su) val(su,j) * ½ val_K(τ at face).
        // r is supported on BOTH adjacent elements; we assemble its nodal
        // coefficients on each element, then form η * rᵀ M r summed over the
        // two elements.
        self.add_br2_lift_coupling(a, n, el, er, &val, &jump_sign);
    }

    /// Assemble the BR2 lifting coupling `η Σ_K (r_e[φ_u])ᵀ M_K (r_e[φ_v])`.
    ///
    /// `val(side, i)` returns the face trace of local basis `i` on `side`
    /// (0 = `el`, 1 = `er`); `jump_sign(side)` returns its jump sign.
    fn add_br2_lift_coupling(
        &self,
        a: &mut [f64],
        n: usize,
        el: usize,
        er: usize,
        val: &impl Fn(usize, usize) -> f64,
        jump_sign: &impl Fn(usize) -> f64,
    ) {
        let n_loc = self.n_loc();
        let gid = |e: usize, i: usize| e * n_loc + i;
        let elem_of = |side: usize| -> usize { if side == 0 { el } else { er } };

        // Physical element mass matrix M_K = (h/2) M_ref.
        // Lift equation on element K:  M_K r = b,  with
        //   b_a = -½ [u]_face * φ^K_a(face)
        // where φ^K_a(face) is the trace of local basis a of element K at the
        // shared face. For element el the shared face is its right face (φ_right);
        // for element er it is its left face (φ_left).
        //
        // r = M_K^{-1} b. Then the stabilisation contributes
        //   η rᵀ M_K r = η bᵀ M_K^{-1} b.
        // Because the data b for u and for v are both face-localised, we form
        // the per-element matrix  S_K = (½)² Σ ... and add η * S_K.

        // For each adjacent element K we precompute G_K = M_K^{-1} t_K t_Kᵀ M_K^{-1}?
        // Simpler: the lift coefficient vector for unit data is r = M_K^{-1} (-½ t_K),
        // scaled by the (signed) jump of the basis whose lift we take, where t_K
        // is the face-trace vector of that element's basis. Then
        //   η rᵀ M_K r = η (¼) (jump_u)(jump_v) (t_Kᵀ M_K^{-1} t_K)·?
        // — but t_K differs per basis index, so we keep full vectors.

        let half_h = 0.5 * self.h;

        for side_k in 0..2 {
            // Face trace vector of element K's local basis at the shared face.
            let t_k: Vec<f64> = (0..n_loc).map(|a_idx| val(side_k, a_idx)).collect();

            // M_K = half_h * M_ref ; solve M_K X = T for each unit face-trace of
            // the *adjacent* contributions. We need, for the data coming from
            // basis (su, j), the element-K lift coefficients:
            //   r^{K}_{su,j} = M_K^{-1} * ( -½ jump(su) val(su,j) * t_K )
            // i.e. proportional to M_K^{-1} t_K with scalar c_{su,j}.
            // So compute w = M_K^{-1} t_K once.
            let m_k: Vec<f64> = self.m_ref.iter().map(|&v| half_h * v).collect();
            let w = match solve_spd_small(&m_k, &t_k, n_loc) {
                Ok(w) => w,
                Err(_) => continue,
            };
            // η rᵀ M_K r with r_u = c_u w, r_v = c_v w:
            //   = η c_u c_v (wᵀ M_K w).
            let mkw = matvec_dense(&m_k, &w, n_loc);
            let wmkw: f64 = w.iter().zip(&mkw).map(|(a_, b_)| a_ * b_).sum();

            // c_{su,j} = -½ jump(su) val(su,j).
            for su in 0..2 {
                for sv in 0..2 {
                    for i in 0..n_loc {
                        for j in 0..n_loc {
                            let c_u = -0.5 * jump_sign(su) * val(su, j);
                            let c_v = -0.5 * jump_sign(sv) * val(sv, i);
                            let row = gid(elem_of(sv), i);
                            let col = gid(elem_of(su), j);
                            a[row * n + col] += self.eta * c_v * c_u * wmkw;
                        }
                    }
                }
            }
        }
    }

    /// The BR2 boundary penalty coefficient `s = η · (t_Kᵀ M_K⁻¹ t_K)`.
    ///
    /// This is the self-energy of the single-element lift of a unit Dirichlet
    /// jump; `t_K` is the interior face-trace vector and `M_K = (h/2) M_ref`.
    /// `s ≥ 0`, and with `η ≥ n_faces` it dominates the consistency terms,
    /// rendering the boundary form coercive (Nitsche/BR2 equivalence).
    fn boundary_lift_penalty(&self, val_i: &[f64]) -> f64 {
        let n_loc = self.n_loc();
        let half_h = 0.5 * self.h;
        let m_k: Vec<f64> = self.m_ref.iter().map(|&v| half_h * v).collect();
        match solve_spd_small(&m_k, val_i, n_loc) {
            Ok(w) => {
                let mkw = matvec_dense(&m_k, &w, n_loc);
                let wmkw: f64 = w.iter().zip(&mkw).map(|(a_, b_)| a_ * b_).sum();
                self.eta * wmkw
            }
            Err(_) => 0.0,
        }
    }

    /// Accumulate the symmetric Nitsche/BR2 boundary-face bilinear form.
    ///
    /// With outward boundary normal `nb` (−1 at the left endpoint, +1 at the
    /// right), the symmetric interior-penalty boundary form is
    ///
    /// ```text
    ///   a_∂(u, v) = − (u' nb) v − (v' nb) u + s · u v
    /// ```
    /// where `s = boundary_lift_penalty` is the BR2 lift self-energy. This is
    /// symmetric in `(u, v)` and, summed with the volume term, SPD.
    fn add_boundary_face(&self, a: &mut [f64], n: usize, e: usize, side: FaceSide, dscale: f64) {
        let n_loc = self.n_loc();
        let gid = |i: usize| e * n_loc + i;

        // Interior traces and outward normal at this boundary face.
        let (val_i, der_i, nb): (Vec<f64>, Vec<f64>, f64) = match side {
            FaceSide::Left => (
                self.phi_left.clone(),
                self.dphi_left.iter().map(|&d| dscale * d).collect(),
                -1.0,
            ),
            FaceSide::Right => (
                self.phi_right.clone(),
                self.dphi_right.iter().map(|&d| dscale * d).collect(),
                1.0,
            ),
        };

        let s = self.boundary_lift_penalty(&val_i);

        for i in 0..n_loc {
            for j in 0..n_loc {
                // −(u' nb) v − (v' nb) u + s u v, where column j ↔ u, row i ↔ v.
                let consistency = -nb * (der_i[j] * val_i[i] + der_i[i] * val_i[j]);
                let penalty = s * val_i[i] * val_i[j];
                a[gid(i) * n + gid(j)] += consistency + penalty;
            }
        }
    }

    /// Assemble the global load vector `b` for source `f` and Dirichlet data.
    ///
    /// The volume load is `∫_K f φ_i dx`. Dirichlet data `g_l`, `g_r` enter
    /// through the boundary-face consistency and stabilisation terms acting on
    /// the (known) exterior value: their lift/penalty contributions move to the
    /// right-hand side with the *same* operators used in the matrix.
    pub fn assemble_load<F>(&self, f: F, g_l: f64, g_r: f64) -> PdeResult<Vec<f64>>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.n_dofs();
        let n_loc = self.n_loc();
        let mut b = vec![0.0; n];
        let gid = |e: usize, i: usize| e * n_loc + i;

        // Volume load ∫_K f φ_i dx = (h/2) Σ_q w_q f(x_q) φ_i(ξ_q).
        let jac = 0.5 * self.h;
        for e in 0..self.n_elem {
            let xl = self.x_left + e as f64 * self.h;
            for q in 0..self.quad_x.len() {
                let xq = xl + 0.5 * self.h * (self.quad_x[q] + 1.0);
                let fq = f(xq);
                for i in 0..n_loc {
                    b[gid(e, i)] += jac * self.quad_w[q] * fq * self.phi_q[q][i];
                }
            }
        }

        // Boundary Dirichlet contributions to the RHS.
        let dscale = 2.0 / self.h;
        self.add_boundary_rhs(&mut b, 0, FaceSide::Left, g_l, dscale);
        self.add_boundary_rhs(&mut b, self.n_elem - 1, FaceSide::Right, g_r, dscale);

        Ok(b)
    }

    /// Move the Dirichlet datum `g` to the RHS via the boundary-face operators.
    ///
    /// Substituting the known trace `u → g` into the boundary form
    /// `a_∂(u,v) = −(u' nb)v − (v' nb)u + s u v` and moving the `g`-bearing terms
    /// to the right-hand side yields, per test basis `i`,
    ///
    /// ```text
    ///   b_i += − nb (v'_i) g + s (v_i) g.
    /// ```
    /// This uses *exactly* the operators assembled in [`Self::add_boundary_face`],
    /// so the Dirichlet condition is imposed consistently.
    fn add_boundary_rhs(&self, b: &mut [f64], e: usize, side: FaceSide, g: f64, dscale: f64) {
        let n_loc = self.n_loc();
        let gid = |i: usize| e * n_loc + i;

        let (val_i, der_i, nb): (Vec<f64>, Vec<f64>, f64) = match side {
            FaceSide::Left => (
                self.phi_left.clone(),
                self.dphi_left.iter().map(|&d| dscale * d).collect(),
                -1.0,
            ),
            FaceSide::Right => (
                self.phi_right.clone(),
                self.dphi_right.iter().map(|&d| dscale * d).collect(),
                1.0,
            ),
        };

        let s = self.boundary_lift_penalty(&val_i);
        for i in 0..n_loc {
            b[gid(i)] += -nb * der_i[i] * g + s * val_i[i] * g;
        }
    }

    /// Assemble and solve the BR2 system `A u = b`, returning nodal DOFs.
    ///
    /// The system is SPD and is solved by dense Cholesky.
    ///
    /// # Errors
    /// * [`PdeError::SingularMatrix`] if Cholesky fails (e.g. `eta` too small
    ///   destroyed positive-definiteness).
    pub fn solve<F>(&self, f: F, g_l: f64, g_r: f64) -> PdeResult<Vec<f64>>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.n_dofs();
        let a = self.assemble_stiffness();
        let b = self.assemble_load(f, g_l, g_r)?;
        cholesky_solve_dense(&a, &b, n)
    }

    /// L2 error `‖u_h − u_exact‖_{L2(0,1)}` using the element quadrature.
    pub fn l2_error<U>(&self, u_h: &[f64], u_exact: U) -> PdeResult<f64>
    where
        U: Fn(f64) -> f64,
    {
        if u_h.len() != self.n_dofs() {
            return Err(PdeError::DimensionMismatch {
                a: u_h.len(),
                b: self.n_dofs(),
            });
        }
        let n_loc = self.n_loc();
        let jac = 0.5 * self.h;
        let mut acc = 0.0;
        for e in 0..self.n_elem {
            let xl = self.x_left + e as f64 * self.h;
            for q in 0..self.quad_x.len() {
                let xq = xl + 0.5 * self.h * (self.quad_x[q] + 1.0);
                let mut uh = 0.0;
                for i in 0..n_loc {
                    uh += u_h[e * n_loc + i] * self.phi_q[q][i];
                }
                let diff = uh - u_exact(xq);
                acc += jac * self.quad_w[q] * diff * diff;
            }
        }
        Ok(acc.sqrt())
    }
}

/// Which face of a 1D element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceSide {
    Left,
    Right,
}

// ─── Lagrange basis on arbitrary nodes ─────────────────────────────────────────

/// Value of the `i`-th Lagrange basis (through `nodes`) at `x`.
fn lagrange_value(nodes: &[f64], i: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (m, &xm) in nodes.iter().enumerate() {
        if m != i {
            v *= (x - xm) / (nodes[i] - xm);
        }
    }
    v
}

/// Derivative of the `i`-th Lagrange basis (through `nodes`) at `x`.
fn lagrange_deriv(nodes: &[f64], i: usize, x: f64) -> f64 {
    let xi = nodes[i];
    let mut sum = 0.0;
    for (k, &xk) in nodes.iter().enumerate() {
        if k == i {
            continue;
        }
        let mut prod = 1.0 / (xi - xk);
        for (m, &xm) in nodes.iter().enumerate() {
            if m != i && m != k {
                prod *= (x - xm) / (xi - xm);
            }
        }
        sum += prod;
    }
    sum
}

// ─── Gauss–Legendre quadrature on [-1, 1] ──────────────────────────────────────

/// `n`-point Gauss–Legendre nodes and weights on `[-1, 1]`.
///
/// Roots of `P_n` via Newton iteration; weights `w = 2 / ((1-x²) P_n'(x)²)`.
///
/// # Errors
/// [`PdeError::InvalidParameter`] if `n == 0`.
fn gauss_legendre(n: usize) -> PdeResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n".into(),
            reason: "Gauss rule needs n >= 1".into(),
        });
    }
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let pi = std::f64::consts::PI;
    for i in 0..n {
        // Initial guess: Chebyshev-like asymptotic for the i-th root.
        let mut xi = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (pn, dpn) = legendre_pn(n, xi);
            if dpn.abs() < 1.0e-300 {
                break;
            }
            let dx = pn / dpn;
            xi -= dx;
            if dx.abs() < 1.0e-15 {
                break;
            }
        }
        let (_, dpn) = legendre_pn(n, xi);
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dpn * dpn);
    }
    Ok((x, w))
}

/// `P_n(x)` and `P_n'(x)` by the standard 3-term recurrence.
fn legendre_pn(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    for k in 2..=n {
        let kf = k as f64;
        let p_next = ((2.0 * kf - 1.0) * x * p_curr - (kf - 1.0) * p_prev) / kf;
        p_prev = p_curr;
        p_curr = p_next;
    }
    // Derivative via P_n'(x) = n (x P_n − P_{n-1}) / (x² − 1).
    let denom = x * x - 1.0;
    let dp = if denom.abs() < 1.0e-14 {
        // Endpoint limit; use n(n+1)/2 * x^{n+1} style — fall back to recurrence.
        let nf = n as f64;
        0.5 * nf * (nf + 1.0) * x.powi((n as i32) - 1)
    } else {
        (n as f64) * (x * p_curr - p_prev) / denom
    };
    (p_curr, dp)
}

// ─── Small dense linear algebra helpers ────────────────────────────────────────

/// Dense matrix-vector product `y = M v` for an `n×n` row-major matrix.
fn matvec_dense(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += m[i * n + j] * v[j];
        }
        y[i] = s;
    }
    y
}

/// Solve a small SPD system `M x = b` (`n×n`, row-major) by Cholesky.
///
/// # Errors
/// [`PdeError::SingularMatrix`] on a non-positive pivot.
fn solve_spd_small(m: &[f64], b: &[f64], n: usize) -> PdeResult<Vec<f64>> {
    cholesky_solve_dense(m, b, n)
}

/// Dense LLᵀ Cholesky factorisation and solve for an `n×n` SPD matrix `a`
/// (row-major) and RHS `b`.
///
/// # Errors
/// [`PdeError::SingularMatrix`] if any pivot is `≤ 0` or non-finite.
fn cholesky_solve_dense(a: &[f64], b: &[f64], n: usize) -> PdeResult<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j {
            diag -= l[j * n + k] * l[j * n + k];
        }
        if diag <= 0.0 {
            return Err(PdeError::SingularMatrix(format!(
                "br2 Cholesky: non-positive pivot {diag:.3e} at column {j}"
            )));
        }
        let ljj = diag.sqrt();
        if !ljj.is_finite() {
            return Err(PdeError::SingularMatrix(format!(
                "br2 Cholesky: non-finite pivot at column {j}"
            )));
        }
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    // Forward solve L y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    // Back solve Lᵀ x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check the assembled global matrix is symmetric to a tight tolerance.
    fn assert_symmetric(a: &[f64], n: usize, tol: f64) {
        for i in 0..n {
            for j in 0..n {
                let d = (a[i * n + j] - a[j * n + i]).abs();
                assert!(d <= tol, "asymmetry at ({i},{j}): {d:e}");
            }
        }
    }

    /// Smallest eigenvalue lower bound via successful Cholesky on A - shift·I.
    fn cholesky_succeeds(a: &[f64], n: usize) -> bool {
        let b = vec![0.0; n];
        cholesky_solve_dense(a, &b, n).is_ok()
    }

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        // ∫_{-1}^{1} x^k dx exact for k up to 2n-1.
        let (x, w) = gauss_legendre(4).expect("gauss");
        for k in 0..=7 {
            let approx: f64 = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi.powi(k)).sum();
            let exact = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert!((approx - exact).abs() < 1e-12, "k={k} {approx} vs {exact}");
        }
    }

    #[test]
    fn lagrange_basis_partition_of_unity() {
        let nodes = lgl_nodes(3).expect("nodes");
        for &x in &[-0.7, -0.1, 0.3, 0.9] {
            let s: f64 = (0..4).map(|i| lagrange_value(&nodes, i, x)).sum();
            assert!((s - 1.0).abs() < 1e-12, "sum at {x} = {s}");
            // Derivatives of a partition of unity sum to zero.
            let ds: f64 = (0..4).map(|i| lagrange_deriv(&nodes, i, x)).sum();
            assert!(ds.abs() < 1e-10, "deriv sum at {x} = {ds}");
        }
    }

    #[test]
    fn stiffness_is_symmetric() {
        for p in 1..=3 {
            let br2 = Br2Elliptic::new(5, p, 0.0, 1.0).expect("build");
            let a = br2.assemble_stiffness();
            assert_symmetric(&a, br2.n_dofs(), 1e-12);
        }
    }

    #[test]
    fn stiffness_is_positive_definite() {
        for p in 1..=3 {
            for n_elem in [2usize, 4, 8] {
                let br2 = Br2Elliptic::new(n_elem, p, 0.0, 1.0).expect("build");
                let a = br2.assemble_stiffness();
                assert!(
                    cholesky_succeeds(&a, br2.n_dofs()),
                    "SPD failed p={p} n={n_elem}"
                );
            }
        }
    }

    #[test]
    fn polynomial_exactness_quadratic() {
        // u = x(1-x) is degree 2; -u'' = 2 = f. With p>=2 BR2 is exact.
        for p in 2..=3 {
            let br2 = Br2Elliptic::new(4, p, 0.0, 1.0).expect("build");
            let u = br2.solve(|_x| 2.0, 0.0, 0.0).expect("solve");
            let coords = br2.dof_coords();
            for (k, &xk) in coords.iter().enumerate() {
                let exact = xk * (1.0 - xk);
                assert!(
                    (u[k] - exact).abs() < 1e-8,
                    "p={p} node {k} x={xk}: got {} want {}",
                    u[k],
                    exact
                );
            }
        }
    }

    #[test]
    fn manufactured_sine_converges_under_refinement() {
        use std::f64::consts::PI;
        // u = sin(πx), f = π² sin(πx), u(0)=u(1)=0.
        let p = 2;
        let mut prev_err = f64::INFINITY;
        let mut rates = Vec::new();
        for &n_elem in &[4usize, 8, 16, 32] {
            let br2 = Br2Elliptic::new(n_elem, p, 0.0, 1.0).expect("build");
            let u = br2
                .solve(|x| PI * PI * (PI * x).sin(), 0.0, 0.0)
                .expect("solve");
            let err = br2.l2_error(&u, |x| (PI * x).sin()).expect("err");
            assert!(err.is_finite() && err < 1e-1, "err too large: {err}");
            if prev_err.is_finite() {
                rates.push((prev_err / err).log2());
            }
            prev_err = err;
        }
        // Error must DECREASE; observed rate should approach ~p+1.
        let last_rate = *rates.last().expect("rate");
        assert!(
            last_rate > (p as f64 + 0.5),
            "convergence rate too low: {rates:?}"
        );
    }

    #[test]
    fn small_penalty_breaks_positive_definiteness() {
        // η far below the coercivity threshold (n_faces=2): the stabilised form
        // must lose positive-definiteness (Cholesky fails) for some refinement.
        let mut some_failed = false;
        for &n_elem in &[4usize, 8, 16] {
            let br2 = Br2Elliptic::with_penalty(n_elem, 2, 0.0, 1.0, 0.0).expect("build");
            let a = br2.assemble_stiffness();
            if !cholesky_succeeds(&a, br2.n_dofs()) {
                some_failed = true;
            }
        }
        assert!(
            some_failed,
            "η=0 should destroy SPD on at least one mesh (lost coercivity)"
        );
        // And the well-posed (default η) version must remain SPD on the same mesh.
        let ok = Br2Elliptic::new(16, 2, 0.0, 1.0).expect("build");
        assert!(cholesky_succeeds(&ok.assemble_stiffness(), ok.n_dofs()));
    }

    #[test]
    fn nonhomogeneous_dirichlet_linear() {
        // u = a + (b-a) x solves -u''=0 with u(0)=a, u(1)=b. BR2 must reproduce
        // it exactly (degree-1 polynomial, p>=1).
        let a = 1.5;
        let b = -0.5;
        for p in 1..=3 {
            let br2 = Br2Elliptic::new(3, p, 0.0, 1.0).expect("build");
            let u = br2.solve(|_x| 0.0, a, b).expect("solve");
            let coords = br2.dof_coords();
            for (k, &xk) in coords.iter().enumerate() {
                let exact = a + (b - a) * xk;
                assert!(
                    (u[k] - exact).abs() < 1e-8,
                    "p={p} x={xk}: got {} want {}",
                    u[k],
                    exact
                );
            }
        }
    }
}
