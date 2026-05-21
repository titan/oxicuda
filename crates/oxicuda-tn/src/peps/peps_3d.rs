//! 3D PEPS (Projected Entangled Pair States) scaffold.
//!
//! Each site in the 3D lattice holds a rank-7 tensor with indices
//! `[d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_phys]` stored row-major.
//!
//! Open boundary conditions (OBC) are enforced: bonds touching a lattice boundary
//! in any direction have dimension 1.
//!
//! # Flat index formula
//!
//! For element `[xl, xr, yl, yr, zl, zr, p]` of a rank-7 tensor:
//! ```text
//! idx = ((((( xl * d_xr + xr) * d_yl + yl) * d_yr + yr) * d_zl + zl) * d_zr + zr) * d_p + p
//! ```
//!
//! # Lattice site indexing
//!
//! Site `(x, y, z)` lives at position `x * Ly * Lz + y * Lz + z` in `Peps3d::tensors`.

use crate::handle::LcgRng;
use crate::svd::svd_dense::svd_jacobi;
use crate::{TnError, TnResult};

// ──────────────────────────────────────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────────────────────────────────────

/// 3D lattice site coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Site3d {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Site3d {
    /// Construct a new 3D site coordinate.
    #[must_use]
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }
}

/// A single rank-7 tensor for one site of a 3D PEPS.
///
/// Shape: `[d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_p]` row-major.
/// Element `[xl, xr, yl, yr, zl, zr, p]` lives at the flat index returned by
/// [`Peps3dTensor::flat_idx`].
#[derive(Debug, Clone)]
pub struct Peps3dTensor {
    pub d_xl: usize,
    pub d_xr: usize,
    pub d_yl: usize,
    pub d_yr: usize,
    pub d_zl: usize,
    pub d_zr: usize,
    pub d_p: usize,
    pub data: Vec<f64>,
}

impl Peps3dTensor {
    /// Create a new tensor, checking dimensions and data length.
    pub fn new(
        d_xl: usize,
        d_xr: usize,
        d_yl: usize,
        d_yr: usize,
        d_zl: usize,
        d_zr: usize,
        d_p: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        for (name, v) in [
            ("d_xl", d_xl),
            ("d_xr", d_xr),
            ("d_yl", d_yl),
            ("d_yr", d_yr),
            ("d_zl", d_zl),
            ("d_zr", d_zr),
            ("d_p", d_p),
        ] {
            if v == 0 {
                return Err(TnError::InvalidBondDimension(0));
            }
            let _ = name; // suppress unused-variable if optimised out
        }
        let expected = d_xl * d_xr * d_yl * d_yr * d_zl * d_zr * d_p;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_p],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_xl,
            d_xr,
            d_yl,
            d_yr,
            d_zl,
            d_zr,
            d_p,
            data,
        })
    }

    /// Total number of elements.
    #[must_use]
    pub fn volume(&self) -> usize {
        self.d_xl * self.d_xr * self.d_yl * self.d_yr * self.d_zl * self.d_zr * self.d_p
    }

    /// Compute the row-major flat index for element `[xl, xr, yl, yr, zl, zr, p]`.
    ///
    /// # Errors
    /// Returns [`TnError::IndexOutOfBounds`] if any index exceeds its bound.
    pub fn flat_idx(
        &self,
        xl: usize,
        xr: usize,
        yl: usize,
        yr: usize,
        zl: usize,
        zr: usize,
        p: usize,
    ) -> TnResult<usize> {
        if xl >= self.d_xl {
            return Err(TnError::IndexOutOfBounds {
                index: xl,
                len: self.d_xl,
            });
        }
        if xr >= self.d_xr {
            return Err(TnError::IndexOutOfBounds {
                index: xr,
                len: self.d_xr,
            });
        }
        if yl >= self.d_yl {
            return Err(TnError::IndexOutOfBounds {
                index: yl,
                len: self.d_yl,
            });
        }
        if yr >= self.d_yr {
            return Err(TnError::IndexOutOfBounds {
                index: yr,
                len: self.d_yr,
            });
        }
        if zl >= self.d_zl {
            return Err(TnError::IndexOutOfBounds {
                index: zl,
                len: self.d_zl,
            });
        }
        if zr >= self.d_zr {
            return Err(TnError::IndexOutOfBounds {
                index: zr,
                len: self.d_zr,
            });
        }
        if p >= self.d_p {
            return Err(TnError::IndexOutOfBounds {
                index: p,
                len: self.d_p,
            });
        }
        Ok(
            (((((xl * self.d_xr + xr) * self.d_yl + yl) * self.d_yr + yr) * self.d_zl + zl)
                * self.d_zr
                + zr)
                * self.d_p
                + p,
        )
    }

    /// Get element `[xl, xr, yl, yr, zl, zr, p]`.
    pub fn get(
        &self,
        xl: usize,
        xr: usize,
        yl: usize,
        yr: usize,
        zl: usize,
        zr: usize,
        p: usize,
    ) -> TnResult<f64> {
        let idx = self.flat_idx(xl, xr, yl, yr, zl, zr, p)?;
        Ok(self.data[idx])
    }

    /// Set element `[xl, xr, yl, yr, zl, zr, p]` to `val`.
    pub fn set(
        &mut self,
        xl: usize,
        xr: usize,
        yl: usize,
        yr: usize,
        zl: usize,
        zr: usize,
        p: usize,
        val: f64,
    ) -> TnResult<()> {
        let idx = self.flat_idx(xl, xr, yl, yr, zl, zr, p)?;
        self.data[idx] = val;
        Ok(())
    }

    /// Return a zero tensor with the given dimensions.
    pub fn zeros(
        d_xl: usize,
        d_xr: usize,
        d_yl: usize,
        d_yr: usize,
        d_zl: usize,
        d_zr: usize,
        d_p: usize,
    ) -> TnResult<Self> {
        let n = d_xl * d_xr * d_yl * d_yr * d_zl * d_zr * d_p;
        Self::new(d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_p, vec![0.0; n])
    }
}

/// 3D Lx × Ly × Lz PEPS with open boundary conditions.
///
/// Tensor at `(x, y, z)` is stored at `tensors[x * Ly * Lz + y * Lz + z]`.
#[derive(Debug, Clone)]
pub struct Peps3d {
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub d_phys: usize,
    pub d_bond: usize,
    /// Flat tensor array. Length = `lx * ly * lz`.
    pub tensors: Vec<Peps3dTensor>,
}

impl Peps3d {
    /// Flat tensor index for site `(x, y, z)`.
    #[must_use]
    fn site_idx(&self, x: usize, y: usize, z: usize) -> usize {
        x * self.ly * self.lz + y * self.lz + z
    }

    /// Bond dimensions for site `(x, y, z)` under open boundary conditions.
    fn obc_dims(&self, x: usize, y: usize, z: usize) -> (usize, usize, usize, usize, usize, usize) {
        let d = self.d_bond;
        let d_xl = if x == 0 { 1 } else { d };
        let d_xr = if x + 1 == self.lx { 1 } else { d };
        let d_yl = if y == 0 { 1 } else { d };
        let d_yr = if y + 1 == self.ly { 1 } else { d };
        let d_zl = if z == 0 { 1 } else { d };
        let d_zr = if z + 1 == self.lz { 1 } else { d };
        (d_xl, d_xr, d_yl, d_yr, d_zl, d_zr)
    }

    /// Get a reference to the tensor at site `s`.
    pub fn tensor_at(&self, s: &Site3d) -> TnResult<&Peps3dTensor> {
        if s.x >= self.lx || s.y >= self.ly || s.z >= self.lz {
            return Err(TnError::IndexOutOfBounds {
                index: s.x,
                len: self.lx,
            });
        }
        Ok(&self.tensors[self.site_idx(s.x, s.y, s.z)])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Construction functions
// ──────────────────────────────────────────────────────────────────────────────

/// Create a 3D PEPS filled with zeros, respecting OBC bond dimensions.
///
/// Interior bonds have dimension `d_bond`; boundary bonds have dimension 1.
///
/// # Errors
/// Returns `TnError::EmptyInput` if any dimension is zero.
pub fn peps3d_new(
    lx: usize,
    ly: usize,
    lz: usize,
    d_bond: usize,
    d_phys: usize,
) -> TnResult<Peps3d> {
    if lx == 0 || ly == 0 || lz == 0 || d_bond == 0 || d_phys == 0 {
        return Err(TnError::EmptyInput);
    }
    let peps_shell = Peps3d {
        lx,
        ly,
        lz,
        d_phys,
        d_bond,
        tensors: Vec::new(),
    };
    let mut tensors = Vec::with_capacity(lx * ly * lz);
    for x in 0..lx {
        for y in 0..ly {
            for z in 0..lz {
                let (d_xl, d_xr, d_yl, d_yr, d_zl, d_zr) = peps_shell.obc_dims(x, y, z);
                let t = Peps3dTensor::zeros(d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_phys)?;
                tensors.push(t);
            }
        }
    }
    Ok(Peps3d {
        lx,
        ly,
        lz,
        d_phys,
        d_bond,
        tensors,
    })
}

/// Create a 3D PEPS with standard-normal random entries, respecting OBC.
///
/// # Errors
/// Returns `TnError::EmptyInput` if any dimension is zero.
pub fn peps3d_random(
    lx: usize,
    ly: usize,
    lz: usize,
    d_bond: usize,
    d_phys: usize,
    rng: &mut LcgRng,
) -> TnResult<Peps3d> {
    if lx == 0 || ly == 0 || lz == 0 || d_bond == 0 || d_phys == 0 {
        return Err(TnError::EmptyInput);
    }
    let peps_shell = Peps3d {
        lx,
        ly,
        lz,
        d_phys,
        d_bond,
        tensors: Vec::new(),
    };
    let mut tensors = Vec::with_capacity(lx * ly * lz);
    for x in 0..lx {
        for y in 0..ly {
            for z in 0..lz {
                let (d_xl, d_xr, d_yl, d_yr, d_zl, d_zr) = peps_shell.obc_dims(x, y, z);
                let n = d_xl * d_xr * d_yl * d_yr * d_zl * d_zr * d_phys;
                let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                tensors.push(Peps3dTensor::new(
                    d_xl, d_xr, d_yl, d_yr, d_zl, d_zr, d_phys, data,
                )?);
            }
        }
    }
    Ok(Peps3d {
        lx,
        ly,
        lz,
        d_phys,
        d_bond,
        tensors,
    })
}

/// Create a product state where each site is in physical state `state[x*Ly*Lz + y*Lz + z]`.
///
/// All virtual bonds have dimension 1 (product state = no entanglement). The physical
/// index at each site is set to a one-hot vector: `T[0,0,0,0,0,0,s] = 1.0`, all others 0.
///
/// # Errors
/// - `TnError::EmptyInput` if any grid dimension or `d_phys` is zero.
/// - `TnError::ShapeMismatch` if `state.len() != lx * ly * lz`.
/// - `TnError::IndexOutOfBounds` if any `state[i] >= d_phys`.
pub fn peps3d_product_state(
    lx: usize,
    ly: usize,
    lz: usize,
    d_phys: usize,
    state: &[usize],
) -> TnResult<Peps3d> {
    if lx == 0 || ly == 0 || lz == 0 || d_phys == 0 {
        return Err(TnError::EmptyInput);
    }
    let n_sites = lx * ly * lz;
    if state.len() != n_sites {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_sites],
            got: vec![state.len()],
        });
    }
    for (i, &s) in state.iter().enumerate() {
        if s >= d_phys {
            return Err(TnError::IndexOutOfBounds {
                index: s,
                len: d_phys,
            });
        }
        let _ = i;
    }
    // For a product state all virtual bonds are 1; d_bond is recorded as 1.
    let mut tensors = Vec::with_capacity(n_sites);
    for x in 0..lx {
        for y in 0..ly {
            for z in 0..lz {
                let site_state = state[x * ly * lz + y * lz + z];
                // All bond dims = 1, physical = d_phys
                let mut data = vec![0.0f64; d_phys];
                data[site_state] = 1.0;
                tensors.push(Peps3dTensor::new(1, 1, 1, 1, 1, 1, d_phys, data)?);
            }
        }
    }
    Ok(Peps3d {
        lx,
        ly,
        lz,
        d_phys,
        d_bond: 1,
        tensors,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility / accessor functions
// ──────────────────────────────────────────────────────────────────────────────

/// Return the nominal bond dimension of the PEPS.
#[must_use]
pub fn peps3d_bond_dimension(peps: &Peps3d) -> usize {
    peps.d_bond
}

/// Return the total number of sites `lx * ly * lz`.
#[must_use]
pub fn peps3d_n_sites(peps: &Peps3d) -> usize {
    peps.lx * peps.ly * peps.lz
}

// ──────────────────────────────────────────────────────────────────────────────
// Norm approximation
// ──────────────────────────────────────────────────────────────────────────────

/// Approximate the norm `‖ψ‖ = √⟨ψ|ψ⟩` by contracting z-slices.
///
/// Each z-slice is a 2D grid of rank-7 tensors. We form the double-layer (bra × ket)
/// contracted over the physical index for each site, then approximate the inner product
/// within each z-slice by successive row contractions (boundary-MPS style). We combine
/// z-slices by weighting by the slice contribution.
///
/// For large networks the exact cost is exponential; `chi_boundary` controls the
/// bond dimension of the boundary MPS used in the 2D intra-slice contraction.
///
/// # Errors
/// Returns `TnError::EmptyInput` if any lattice dimension is zero.
pub fn peps3d_norm_approx(peps: &Peps3d, chi_boundary: usize) -> TnResult<f64> {
    if peps.lx == 0 || peps.ly == 0 || peps.lz == 0 {
        return Err(TnError::EmptyInput);
    }
    // Degenerate 1×1×1 case.
    if peps.lx == 1 && peps.ly == 1 && peps.lz == 1 {
        let t = &peps.tensors[0];
        let norm_sq: f64 = t.data.iter().map(|v| v * v).sum();
        return Ok(norm_sq.sqrt());
    }

    // Contract z-slices independently and accumulate the inner product.
    // For slice z, form the double-layer tensor network (contracting phys legs),
    // then sweep x-rows with the boundary-MPS approximation.
    let mut total_norm_sq = 0.0f64;
    for z in 0..peps.lz {
        let slice_ns = contract_z_slice_norm_sq(peps, z, chi_boundary)?;
        total_norm_sq += slice_ns;
    }
    // Normalise by number of slices (heuristic for the scaffold).
    let norm_sq = total_norm_sq / (peps.lz as f64);
    if norm_sq < 0.0 {
        return Err(TnError::NumericalInstability(
            "negative norm squared in peps3d_norm_approx".into(),
        ));
    }
    Ok(norm_sq.sqrt())
}

/// Contract the double-layer of z-slice `z` along x-rows and return the approximate
/// ‹bra|ket› inner product for that slice. Uses a boundary-vector sweep over x-rows.
fn contract_z_slice_norm_sq(peps: &Peps3d, z: usize, chi_boundary: usize) -> TnResult<f64> {
    let lx = peps.lx;
    let ly = peps.ly;
    // chi_boundary controls the max bond dimension of the running boundary vector.
    let chi_eff = chi_boundary.max(1);

    // For each site (x, y, z) in this slice, the double-layer contribution (summing
    // over physical index) is a rank-12 tensor. We approximate by treating bonds
    // along x as the "virtual" direction and contracting y-sites within a row.
    //
    // Strategy: for each x-row, form the product of double-layer y-tensors as a
    // scalar (contracting all virtual bonds to identity). Then the total norm-sq
    // is roughly the product over rows.

    let mut slice_value = 1.0f64;
    for x in 0..lx {
        let row_value = contract_row_double_layer(peps, x, z, ly, chi_eff)?;
        slice_value *= row_value;
    }
    Ok(slice_value)
}

/// For a single x-row at (x, *, z), form the double-layer (bra×ket summed over phys)
/// and contract all virtual bonds to estimate the row contribution.
fn contract_row_double_layer(
    peps: &Peps3d,
    x: usize,
    z: usize,
    ly: usize,
    chi_eff: usize,
) -> TnResult<f64> {
    // Build a running boundary vector of size at most chi_eff.
    // Each site in the y direction is contracted in turn.
    // The boundary vector encodes the double-layer y-bond state.
    let mut boundary: Vec<f64> = vec![1.0]; // size = effective d_yl^2 for site (x,0,z)

    for y in 0..ly {
        let t = &peps.tensors[x * peps.ly * peps.lz + y * peps.lz + z];
        // Compute the double-layer transfer matrix element for this site.
        // We contract: sum_{p} T[xl,xr,yl,yr,zl,zr,p] * T[xl,xr,yl,yr,zl,zr,p]
        // treating y-bonds as the propagation direction and x,z bonds as indices
        // to be summed (treating them as traced / boundary-contracted).
        let new_boundary = apply_site_double_layer(&boundary, t, chi_eff)?;
        boundary = new_boundary;
    }
    // Sum the boundary vector entries (trace over the right y-bond).
    let row_val: f64 = boundary.iter().sum();
    Ok(row_val)
}

/// Apply one site's double-layer transfer to the running boundary vector.
///
/// The boundary vector encodes the double-layer left y-virtual index `(yl_bra, yl_ket)`.
/// We compute:
/// ```text
/// boundary'[yr_bra, yr_ket] = sum_{yl_bra, yl_ket} boundary[yl_bra, yl_ket]
///     * sum_{xl,xr,zl,zr,p} T[xl,xr,yl_bra,yr_bra,zl,zr,p] * T[xl,xr,yl_ket,yr_ket,zl,zr,p]
/// ```
/// Then truncate the y-bond dimension to `chi_eff` via norm-based selection.
fn apply_site_double_layer(
    boundary: &[f64],
    t: &Peps3dTensor,
    chi_eff: usize,
) -> TnResult<Vec<f64>> {
    let d_yl = t.d_yl;
    let d_yr = t.d_yr;
    let in_dim = d_yl * d_yl; // (yl_bra, yl_ket)
    let out_dim = d_yr * d_yr; // (yr_bra, yr_ket)

    if boundary.len() != in_dim {
        return Err(TnError::DimensionMismatch {
            a: boundary.len(),
            b: in_dim,
        });
    }

    // Build the local transfer matrix M[in, out] =
    //   sum_{xl,xr,zl,zr,p} T[xl,xr,yl_bra,yr_bra,zl,zr,p] * T[xl,xr,yl_ket,yr_ket,zl,zr,p]
    // where in = yl_bra * d_yl + yl_ket, out = yr_bra * d_yr + yr_ket.
    let mut transfer = vec![0.0f64; in_dim * out_dim];
    for yl_bra in 0..d_yl {
        for yl_ket in 0..d_yl {
            for yr_bra in 0..d_yr {
                for yr_ket in 0..d_yr {
                    let mut acc = 0.0f64;
                    // Sum over all non-y bond indices and physical index.
                    for xl in 0..t.d_xl {
                        for xr in 0..t.d_xr {
                            for zl in 0..t.d_zl {
                                for zr in 0..t.d_zr {
                                    for p in 0..t.d_p {
                                        let v_bra = t.get(xl, xr, yl_bra, yr_bra, zl, zr, p)?;
                                        let v_ket = t.get(xl, xr, yl_ket, yr_ket, zl, zr, p)?;
                                        acc += v_bra * v_ket;
                                    }
                                }
                            }
                        }
                    }
                    let in_idx = yl_bra * d_yl + yl_ket;
                    let out_idx = yr_bra * d_yr + yr_ket;
                    transfer[in_idx * out_dim + out_idx] = acc;
                }
            }
        }
    }

    // Contract boundary with transfer matrix: new_boundary[out] = sum_in boundary[in] * M[in,out]
    let mut new_boundary = vec![0.0f64; out_dim];
    for in_idx in 0..in_dim {
        let b = boundary[in_idx];
        if b.abs() < 1e-300 {
            continue;
        }
        for out_idx in 0..out_dim {
            new_boundary[out_idx] += b * transfer[in_idx * out_dim + out_idx];
        }
    }

    // Truncate to chi_eff^2 if needed (keep largest-magnitude entries).
    let chi_sq = chi_eff * chi_eff;
    if new_boundary.len() > chi_sq {
        let mut indexed: Vec<(usize, f64)> = new_boundary.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(chi_sq);
        let mut out = vec![0.0f64; chi_sq];
        for (i, (_, v)) in indexed.into_iter().enumerate() {
            out[i] = v;
        }
        Ok(out)
    } else {
        Ok(new_boundary)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Local expectation value
// ──────────────────────────────────────────────────────────────────────────────

/// Compute `⟨ψ|O_site|ψ⟩ / ⟨ψ|ψ⟩` where O is a `d_phys × d_phys` operator.
///
/// The environment is approximated by contracting all off-site tensors as identity
/// contributions (all virtual bonds traced). This is exact for product states and
/// a scaffold-level approximation for entangled states.
///
/// # Errors
/// - `TnError::ShapeMismatch` if `op.len() != d_phys * d_phys`.
/// - `TnError::IndexOutOfBounds` if the site is outside the lattice.
pub fn peps3d_local_expectation(peps: &Peps3d, op: &[f64], site: &Site3d) -> TnResult<f64> {
    let d_p = peps.d_phys;
    if op.len() != d_p * d_p {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_p, d_p],
            got: vec![op.len()],
        });
    }
    if site.x >= peps.lx || site.y >= peps.ly || site.z >= peps.lz {
        return Err(TnError::IndexOutOfBounds {
            index: site.x,
            len: peps.lx,
        });
    }

    let t = peps.tensor_at(site)?;
    // Compute the reduced single-site density matrix ρ[p_bra, p_ket] by tracing out
    // all virtual bonds: ρ[pb, pk] = Σ_{xl,xr,yl,yr,zl,zr} T[xl,xr,yl,yr,zl,zr,pb] *
    //                                                          T[xl,xr,yl,yr,zl,zr,pk]
    let mut rho = vec![0.0f64; d_p * d_p];
    for xl in 0..t.d_xl {
        for xr in 0..t.d_xr {
            for yl in 0..t.d_yl {
                for yr in 0..t.d_yr {
                    for zl in 0..t.d_zl {
                        for zr in 0..t.d_zr {
                            for pb in 0..d_p {
                                let v_bra = t.get(xl, xr, yl, yr, zl, zr, pb)?;
                                if v_bra.abs() < 1e-300 {
                                    continue;
                                }
                                for pk in 0..d_p {
                                    let v_ket = t.get(xl, xr, yl, yr, zl, zr, pk)?;
                                    rho[pb * d_p + pk] += v_bra * v_ket;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalise ρ by its trace.
    let trace_rho: f64 = (0..d_p).map(|i| rho[i * d_p + i]).sum();
    if trace_rho.abs() < 1e-300 {
        return Err(TnError::NumericalInstability(
            "site tensor is zero; cannot compute expectation value".into(),
        ));
    }
    let inv_trace = 1.0 / trace_rho;

    // Expectation value = Tr[ρ O] = Σ_{pb, pk} ρ[pb, pk] * O[pk, pb].
    let mut expval = 0.0f64;
    for pb in 0..d_p {
        for pk in 0..d_p {
            // O stored row-major: O[row=pk, col=pb] = op[pk * d_p + pb]
            expval += rho[pb * d_p + pk] * op[pk * d_p + pb];
        }
    }
    Ok(expval * inv_trace)
}

// ──────────────────────────────────────────────────────────────────────────────
// Entanglement entropy across a z-cut
// ──────────────────────────────────────────────────────────────────────────────

/// Approximate the entanglement entropy of the z-bipartition at `z_cut`.
///
/// The bipartition separates sites with `z ≤ z_cut` from `z > z_cut`. We approximate
/// the reduced density matrix by collecting the z-bonds crossing the cut and computing
/// their singular value spectrum.
///
/// Concretely, for each site `(x, y, z_cut)` we extract the right-z bond vector
/// (contracting xl, xr, yl, yr, zl, p by summing over them), yielding a bond state
/// of dimension `d_zr`. The combined state across all `lx * ly` such sites is stacked
/// into a matrix and SVD-decomposed to obtain the entanglement spectrum.
///
/// # Errors
/// - `TnError::EmptyInput` if the lattice has no z layers.
/// - `TnError::InvalidConfiguration` if `z_cut >= lz - 1`.
pub fn peps3d_entanglement_entropy_z(peps: &Peps3d, z_cut: usize) -> TnResult<f64> {
    if peps.lz == 0 {
        return Err(TnError::EmptyInput);
    }
    if z_cut + 1 >= peps.lz {
        return Err(TnError::InvalidConfiguration(format!(
            "z_cut={z_cut} must be < lz-1={}",
            peps.lz - 1
        )));
    }

    let lx = peps.lx;
    let ly = peps.ly;
    let n_sites_cut = lx * ly; // sites on the left side of the cut

    // For each site (x, y, z_cut), extract the "bond state" vector of dimension d_zr
    // by summing over all other indices (virtual + physical). This gives an effective
    // lx*ly dimensional left side and d_zr dimensional right side.
    let z = z_cut;
    let mut bond_matrix: Vec<f64> = Vec::new();
    let mut max_d_zr = 0usize;

    // Collect per-site bond vectors.
    let mut site_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_sites_cut);
    for x in 0..lx {
        for y in 0..ly {
            let t = &peps.tensors[x * peps.ly * peps.lz + y * peps.lz + z];
            let d_zr = t.d_zr;
            if d_zr > max_d_zr {
                max_d_zr = d_zr;
            }
            // Bond vector: v[zr] = sum_{xl,xr,yl,yr,zl,p} T[xl,xr,yl,yr,zl,zr,p]
            let mut v = vec![0.0f64; d_zr];
            for xl in 0..t.d_xl {
                for xr in 0..t.d_xr {
                    for yl in 0..t.d_yl {
                        for yr in 0..t.d_yr {
                            for zl in 0..t.d_zl {
                                for (zr, v_zr) in v.iter_mut().enumerate().take(d_zr) {
                                    for p in 0..t.d_p {
                                        *v_zr += t.get(xl, xr, yl, yr, zl, zr, p)?;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            site_vecs.push(v);
        }
    }

    // Pad all bond vectors to max_d_zr.
    for v in &mut site_vecs {
        v.resize(max_d_zr, 0.0);
    }

    // Build matrix of shape (n_sites_cut, max_d_zr).
    let nrows = n_sites_cut;
    let ncols = max_d_zr;
    if ncols == 0 {
        return Ok(0.0);
    }
    bond_matrix.resize(nrows * ncols, 0.0);
    for (i, v) in site_vecs.iter().enumerate() {
        for (j, &val) in v.iter().enumerate() {
            bond_matrix[i * ncols + j] = val;
        }
    }

    // SVD of the bond matrix.
    let svd = svd_jacobi(&bond_matrix, nrows, ncols).map_err(|e| {
        TnError::NumericalInstability(format!("SVD failed in entanglement entropy: {e}"))
    })?;

    // Compute entanglement entropy from singular values:
    // S = -Σ_i λ_i² ln(λ_i²) where λ_i are the (normalised) singular values.
    let norm_sq: f64 = svd.s.iter().map(|&si| si * si).sum();
    if norm_sq < 1e-300 {
        return Ok(0.0);
    }
    let inv_ns = 1.0 / norm_sq;
    let entropy: f64 = svd
        .s
        .iter()
        .filter(|&&si| si > 1e-300)
        .map(|&si| {
            let lam_sq = si * si * inv_ns;
            -lam_sq * lam_sq.ln()
        })
        .sum();
    Ok(entropy)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: correct number of tensors.
    #[test]
    fn peps3d_new_shape() {
        let peps = peps3d_new(2, 3, 4, 2, 2).expect("ok");
        assert_eq!(peps.tensors.len(), 2 * 3 * 4);
        assert_eq!(peps3d_n_sites(&peps), 24);
    }

    // Test 2: corner tensor has boundary bonds equal to 1.
    #[test]
    fn peps3d_new_boundary_dims() {
        let peps = peps3d_new(3, 3, 3, 4, 2).expect("ok");
        // Corner site (0, 0, 0): d_xl=1, d_yl=1, d_zl=1
        let t_corner = &peps.tensors[0];
        assert_eq!(t_corner.d_xl, 1);
        assert_eq!(t_corner.d_yl, 1);
        assert_eq!(t_corner.d_zl, 1);
        // Interior site (x=1, y=1, z=1) with ly=lz=3.
        // Flat index = x*ly*lz + y*lz + z = 1*3*3 + 1*3 + 1 = 13.
        let t_interior = &peps.tensors[13];
        assert_eq!(t_interior.d_xl, 4);
        assert_eq!(t_interior.d_xr, 4);
        assert_eq!(t_interior.d_yl, 4);
        assert_eq!(t_interior.d_yr, 4);
        assert_eq!(t_interior.d_zl, 4);
        assert_eq!(t_interior.d_zr, 4);
    }

    // Test 3: random PEPS has non-zero data.
    #[test]
    fn peps3d_random_nonzero() {
        let mut rng = LcgRng::new(42);
        let peps = peps3d_random(2, 2, 2, 2, 2, &mut rng).expect("ok");
        let any_nonzero = peps
            .tensors
            .iter()
            .any(|t| t.data.iter().any(|&v| v != 0.0));
        assert!(any_nonzero);
    }

    // Test 4: product state creation without error.
    #[test]
    fn peps3d_product_state_trivial() {
        let state = vec![0usize; 8]; // |0,0,...,0⟩ for 2×2×2
        let peps = peps3d_product_state(2, 2, 2, 3, &state).expect("ok");
        assert_eq!(peps3d_n_sites(&peps), 8);
        // All tensors should have d_xl=d_xr=...=d_zr=1 (product state)
        for t in &peps.tensors {
            assert_eq!(t.d_xl, 1);
            assert_eq!(t.d_xr, 1);
            assert_eq!(t.d_yl, 1);
            assert_eq!(t.d_yr, 1);
            assert_eq!(t.d_zl, 1);
            assert_eq!(t.d_zr, 1);
        }
    }

    // Test 5: peps3d_n_sites returns correct value.
    #[test]
    fn peps3d_n_sites_correct() {
        let peps = peps3d_new(3, 4, 5, 2, 2).expect("ok");
        assert_eq!(peps3d_n_sites(&peps), 60);
    }

    // Test 6: bond dimension accessor.
    #[test]
    fn peps3d_bond_dim() {
        let peps = peps3d_new(2, 2, 2, 7, 2).expect("ok");
        assert_eq!(peps3d_bond_dimension(&peps), 7);
    }

    // Test 7: norm approximation is positive for random state.
    #[test]
    fn peps3d_norm_approx_positive() {
        let mut rng = LcgRng::new(13);
        let peps = peps3d_random(2, 2, 2, 2, 2, &mut rng).expect("ok");
        let norm = peps3d_norm_approx(&peps, 4).expect("ok");
        assert!(norm > 0.0, "norm should be positive, got {norm}");
        assert!(norm.is_finite(), "norm should be finite, got {norm}");
    }

    // Test 8: expectation value of identity equals 1.
    #[test]
    fn peps3d_local_expectation_identity() {
        let mut rng = LcgRng::new(77);
        let peps = peps3d_random(2, 2, 2, 2, 2, &mut rng).expect("ok");
        // 2×2 identity operator.
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let site = Site3d::new(0, 0, 0);
        let exp_val = peps3d_local_expectation(&peps, &identity, &site).expect("ok");
        assert!(
            (exp_val - 1.0).abs() < 1e-10,
            "⟨I⟩ should be 1, got {exp_val}"
        );
    }

    // Test 9: entanglement entropy runs without error and is finite.
    #[test]
    fn peps3d_entanglement_z() {
        let mut rng = LcgRng::new(55);
        // Use lz >= 2 so z_cut = 0 is valid.
        let peps = peps3d_random(2, 2, 3, 2, 2, &mut rng).expect("ok");
        let entropy = peps3d_entanglement_entropy_z(&peps, 0).expect("ok");
        assert!(
            entropy.is_finite(),
            "entropy should be finite, got {entropy}"
        );
        assert!(entropy >= 0.0, "entropy should be non-negative");
    }

    // Test 10: 2×2×2 non-trivial grid.
    #[test]
    fn peps3d_2x2x2() {
        let mut rng = LcgRng::new(33);
        let peps = peps3d_random(2, 2, 2, 3, 2, &mut rng).expect("ok");
        assert_eq!(peps3d_n_sites(&peps), 8);
        let norm = peps3d_norm_approx(&peps, 4).expect("ok");
        assert!(norm.is_finite());
    }

    // Test 11: degenerate single-site case.
    #[test]
    fn peps3d_1x1x1() {
        let mut rng = LcgRng::new(11);
        let peps = peps3d_random(1, 1, 1, 4, 2, &mut rng).expect("ok");
        assert_eq!(peps3d_n_sites(&peps), 1);
        // All bonds should be 1 due to OBC.
        let t = &peps.tensors[0];
        assert_eq!(t.d_xl, 1);
        assert_eq!(t.d_xr, 1);
        assert_eq!(t.d_yl, 1);
        assert_eq!(t.d_yr, 1);
        assert_eq!(t.d_zl, 1);
        assert_eq!(t.d_zr, 1);
        let norm = peps3d_norm_approx(&peps, 2).expect("ok");
        assert!(norm.is_finite());
    }

    // Test 12: tensor element access and mutation at known coordinates.
    #[test]
    fn peps3d_tensor_element_access() {
        let mut peps = peps3d_new(2, 2, 2, 2, 3).expect("ok");
        // Set a specific element in the interior-facing tensor.
        // Site (x=1, y=0, z=0): d_xl=2, d_xr=1 (right boundary), d_yl=1, d_yr=2, d_zl=1, d_zr=2, d_p=3
        // Flat index = x*ly*lz + y*lz + z = 1*2*2 + 0*2 + 0 = 4.
        let idx = 4;
        let t = &mut peps.tensors[idx];
        // Set element [0, 0, 0, 1, 0, 1, 2] = approximate value of π
        let pi_approx = std::f64::consts::PI;
        t.set(0, 0, 0, 1, 0, 1, 2, pi_approx).expect("set ok");
        let val = t.get(0, 0, 0, 1, 0, 1, 2).expect("get ok");
        assert!((val - pi_approx).abs() < 1e-14);
    }
}
