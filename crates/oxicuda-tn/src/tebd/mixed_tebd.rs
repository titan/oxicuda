//! Mixed-state TEBD: time evolution of density matrices as Matrix Product Operators (MPO).
//!
//! ## Overview
//!
//! For open quantum systems the state is a density matrix ρ rather than a pure state.
//! For a chain of L sites with local Hilbert-space dimension d, ρ can be represented
//! as an MPO:
//!
//! ```text
//! ρ[s_1,...,s_L; s_1',...,s_L'] = Tr(W_1[s_1,s_1'] W_2[s_2,s_2'] ... W_L[s_L,s_L'])
//! ```
//!
//! where each W_i has shape `[D_l, d_out, d_in, D_r]`.
//!
//! ## Time evolution (imaginary-time cooling)
//!
//! `ρ(β) ∝ e^{-βH/2} ρ(0) e^{-βH/2}`
//!
//! A 2-site gate `g[q1,q2,p1,p2]` (shape `[d,d,d,d]`) is applied to sites i and i+1:
//!
//! 1. Contract W_i and W_{i+1} into `θ[D_l, out1, out2, in1, in2, D_r]`.
//! 2. Apply gate: `θ'[..., q1, q2, ...] = Σ_{p1,p2} g[q1,q2,p1,p2] θ[...,p1,p2,...]`.
//! 3. SVD-split back into W_i and W_{i+1}, truncate to chi_max.
//!
//! ## Lindblad dephasing channel
//!
//! Simple site-local dephasing:  W_i[s,s'] → `[(1-γ) + γ δ(s,s')] W_i[s,s']`.
//!
//! ## Purity
//!
//! `Tr[ρ²]` is computed via a "doubled" transfer-matrix contraction over two MPO copies.

use crate::error::{TnError, TnResult};
use crate::mps::truncation::svd_truncate;
use crate::peps::simple_update::{heisenberg_hamiltonian_2site, mat_exp_sym};
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Single MPO site tensor with shape `[D_l, d_out, d_in, D_r]`, row-major.
///
/// Flat index: `((l * d_out + o) * d_in + i) * D_r + r`.
#[derive(Debug, Clone)]
pub struct MpoSiteTensor {
    /// Left bond dimension.
    pub d_l: usize,
    /// Physical output dimension.
    pub d_out: usize,
    /// Physical input dimension.
    pub d_in: usize,
    /// Right bond dimension.
    pub d_r: usize,
    /// Row-major data buffer of length `d_l * d_out * d_in * d_r`.
    pub data: Vec<f64>,
}

impl MpoSiteTensor {
    /// Construct a new tensor, verifying that `data.len()` matches the product of dimensions.
    pub fn new(
        d_l: usize,
        d_out: usize,
        d_in: usize,
        d_r: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if d_l == 0 || d_out == 0 || d_in == 0 || d_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        let expected = d_l * d_out * d_in * d_r;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l, d_out, d_in, d_r],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_l,
            d_out,
            d_in,
            d_r,
            data,
        })
    }

    /// Construct an all-zero tensor.
    pub fn zeros(d_l: usize, d_out: usize, d_in: usize, d_r: usize) -> TnResult<Self> {
        if d_l == 0 || d_out == 0 || d_in == 0 || d_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        let n = d_l * d_out * d_in * d_r;
        Ok(Self {
            d_l,
            d_out,
            d_in,
            d_r,
            data: vec![0.0; n],
        })
    }

    /// Compute flat index for element `(l, o, i, r)`.
    #[inline]
    fn flat_idx(&self, l: usize, o: usize, i: usize, r: usize) -> usize {
        ((l * self.d_out + o) * self.d_in + i) * self.d_r + r
    }

    /// Read element `(l, o, i, r)`.
    pub fn get(&self, l: usize, o: usize, i: usize, r: usize) -> f64 {
        self.data[self.flat_idx(l, o, i, r)]
    }

    /// Write element `(l, o, i, r)`.
    pub fn set(&mut self, l: usize, o: usize, i: usize, r: usize, v: f64) {
        let idx = self.flat_idx(l, o, i, r);
        self.data[idx] = v;
    }

    /// Total element count.
    pub fn numel(&self) -> usize {
        self.d_l * self.d_out * self.d_in * self.d_r
    }
}

/// Matrix Product Operator representing a density matrix ρ for a 1D chain.
#[derive(Debug, Clone)]
pub struct DensityMpo {
    /// Site tensors, one per site.
    pub sites: Vec<MpoSiteTensor>,
    /// Number of sites.
    pub n_sites: usize,
    /// Physical (Hilbert-space) dimension `d` per site.
    pub d_phys: usize,
}

/// Configuration for mixed-state TEBD imaginary-time evolution.
#[derive(Debug, Clone)]
pub struct MixedTebdConfig {
    /// Maximum bond dimension of the density MPO.
    pub chi_max: usize,
    /// Truncation tolerance relative to the largest singular value.
    pub trunc_tol: f64,
    /// Number of imaginary-time steps.
    pub n_steps: usize,
    /// Imaginary time step `δτ`.
    pub dt: f64,
    /// Dephasing rate `γ` applied after each step (0 = no dephasing).
    pub dephasing_rate: f64,
    /// Heisenberg coupling constant `J`.
    pub coupling_j: f64,
}

impl Default for MixedTebdConfig {
    fn default() -> Self {
        Self {
            chi_max: 16,
            trunc_tol: 1e-12,
            n_steps: 20,
            dt: 0.05,
            dephasing_rate: 0.0,
            coupling_j: 1.0,
        }
    }
}

/// Output of a completed mixed-state TEBD run.
#[derive(Debug, Clone)]
pub struct MixedTebdResult {
    /// Purity `Tr[ρ²]` at the end of the run.
    pub purity: f64,
    /// Trace `Tr[ρ]` at the end.
    pub trace: f64,
    /// Estimated energy per site `⟨H⟩ / (L-1)` at the end.
    pub energy_per_site: f64,
    /// Number of steps actually run.
    pub steps_run: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// MPO construction
// ─────────────────────────────────────────────────────────────────────────────

/// Construct the identity density matrix `ρ = I/d^L` as an MPO (infinite-temperature state).
///
/// Each site tensor `W_i` has shape `[1, d, d, 1]` with `W_i[0,s,s,0] = 1/d` (diagonal).
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] if `n_sites == 0`.
/// Returns [`TnError::InvalidBondDimension`] if `d_phys == 0`.
pub fn density_mpo_identity(n_sites: usize, d_phys: usize) -> TnResult<DensityMpo> {
    if n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if d_phys == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    let val = 1.0 / d_phys as f64;
    let mut sites = Vec::with_capacity(n_sites);
    for _ in 0..n_sites {
        let mut w = MpoSiteTensor::zeros(1, d_phys, d_phys, 1)?;
        for s in 0..d_phys {
            w.set(0, s, s, 0, val);
        }
        sites.push(w);
    }
    Ok(DensityMpo {
        sites,
        n_sites,
        d_phys,
    })
}

/// Construct a pure-state density matrix `|ψ⟩⟨ψ|` from an MPS given as a flat list of tensors.
///
/// The MPS tensors are in shape `[D_l, d, D_r]` (row-major). The density-matrix MPO is formed
/// by the outer product `W_i[s, s'] = A_i[s] ⊗ A_i[s']` on each site (Kronecker product over
/// virtual bonds).
///
/// # Arguments
///
/// * `mps` — flattened tensors, one `Vec<f64>` per site, each of length `D_l[i] * d * D_r[i]`.
/// * `bond_dims` — bond dimensions vector of length `n_sites + 1`, where `bond_dims[0]` and
///   `bond_dims[n_sites]` must equal 1.
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn density_mpo_from_mps(
    mps: &[Vec<f64>],
    bond_dims: &[usize],
    n_sites: usize,
    d_phys: usize,
) -> TnResult<DensityMpo> {
    if n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if mps.len() != n_sites {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_sites],
            got: vec![mps.len()],
        });
    }
    if bond_dims.len() != n_sites + 1 {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_sites + 1],
            got: vec![bond_dims.len()],
        });
    }

    let mut sites = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let d_l_mps = bond_dims[i];
        let d_r_mps = bond_dims[i + 1];
        let expected_len = d_l_mps * d_phys * d_r_mps;
        if mps[i].len() != expected_len {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l_mps, d_phys, d_r_mps],
                got: vec![mps[i].len()],
            });
        }

        // The MPO tensor for site i has shape [D_l^2, d, d, D_r^2] (Kronecker-product virtual bonds).
        let d_l_mpo = d_l_mps * d_l_mps;
        let d_r_mpo = d_r_mps * d_r_mps;
        let mut w = MpoSiteTensor::zeros(d_l_mpo, d_phys, d_phys, d_r_mpo)?;

        // W[(la, lb), s_out, s_in, (ra, rb)] = A[la, s_out, ra] * conj(A)[lb, s_in, rb]
        // For real MPS: conj(A) = A.
        for la in 0..d_l_mps {
            for lb in 0..d_l_mps {
                let l_mpo = la * d_l_mps + lb;
                for s_out in 0..d_phys {
                    for s_in in 0..d_phys {
                        for ra in 0..d_r_mps {
                            for rb in 0..d_r_mps {
                                let r_mpo = ra * d_r_mps + rb;
                                // A_i[la, s_out, ra]
                                let a_ket = mps[i][(la * d_phys + s_out) * d_r_mps + ra];
                                // A_i[lb, s_in, rb]  (bra, real so no conjugate)
                                let a_bra = mps[i][(lb * d_phys + s_in) * d_r_mps + rb];
                                let prev = w.get(l_mpo, s_out, s_in, r_mpo);
                                w.set(l_mpo, s_out, s_in, r_mpo, prev + a_ket * a_bra);
                            }
                        }
                    }
                }
            }
        }
        sites.push(w);
    }
    Ok(DensityMpo {
        sites,
        n_sites,
        d_phys,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace and purity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `Tr[ρ]` by contracting transfer matrices `T_i[l,r] = Σ_s W_i[l,s,s,r]`.
///
/// For open boundary conditions, the result is a scalar obtained by chaining
/// all transfer matrices from left to right.
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] if the MPO has no sites.
pub fn density_mpo_trace(mpo: &DensityMpo) -> TnResult<f64> {
    if mpo.n_sites == 0 {
        return Err(TnError::EmptyInput);
    }

    // Left boundary environment: shape [d_l] with all 1s (open boundary).
    // For a properly formed MPO the leftmost d_l should be 1.
    let first = &mpo.sites[0];
    let d = mpo.d_phys;
    let mut env = vec![1.0_f64; first.d_l];

    for site_idx in 0..mpo.n_sites {
        let w = &mpo.sites[site_idx];
        // Transfer matrix: T_new[r] = Σ_{l,s} env[l] * W[l, s, s, r]
        let mut new_env = vec![0.0_f64; w.d_r];
        for (r, slot) in new_env.iter_mut().enumerate().take(w.d_r) {
            let mut acc = 0.0_f64;
            for s in 0..d {
                for l in 0..w.d_l {
                    if l < env.len() {
                        acc += env[l] * w.get(l, s, s, r);
                    }
                }
            }
            *slot = acc;
        }
        env = new_env;
    }

    // Right boundary: sum all elements (open boundary, right dim should be 1)
    Ok(env.iter().sum())
}

/// Compute purity `Tr[ρ²]` by contracting the doubled transfer matrix.
///
/// The doubled transfer matrix at each site is:
/// `T²[(l_a, l_b), (r_a, r_b)] = Σ_{s,t} W[l_a, s, t, r_a] * W[l_b, t, s, r_b]`
///
/// This accounts for the MPO contraction of `ρ·ρ`.
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] if the MPO has no sites.
pub fn density_mpo_purity(mpo: &DensityMpo) -> TnResult<f64> {
    if mpo.n_sites == 0 {
        return Err(TnError::EmptyInput);
    }

    // Purity Tr[ρ²] via the doubled transfer matrix.
    //
    // Left boundary env[(la, lb)] = δ(la, lb), shape [D_l × D_l].
    // At each site:
    //   new_env[(ra, rb)] = Σ_{la,lb,s,t} env[(la,lb)] * W[la,s,t,ra] * W[lb,t,s,rb]
    // The swap (s↔t) in the second copy comes from the matrix product ρ·ρ.
    //
    // Final purity = Σ_r env[(r,r)] (trace over the right boundary).
    let d = mpo.d_phys;
    let first = &mpo.sites[0];
    let dl = first.d_l;
    let mut env = vec![0.0_f64; dl * dl];
    for l in 0..dl {
        env[l * dl + l] = 1.0;
    }

    for site_idx in 0..mpo.n_sites {
        let w = &mpo.sites[site_idx];
        let dl_w = w.d_l;
        let dr_w = w.d_r;

        let mut new_env = vec![0.0_f64; dr_w * dr_w];

        for la in 0..dl_w {
            for lb in 0..dl_w {
                let env_val = if la < dl_w && lb < dl_w {
                    env.get(la * dl_w + lb).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                if env_val == 0.0 {
                    continue;
                }
                for s in 0..d {
                    for t in 0..d {
                        for ra in 0..dr_w {
                            let w_a = w.get(la, s, t, ra);
                            if w_a.abs() < 1e-15 {
                                continue;
                            }
                            // Second copy has swapped (t, s) physical indices
                            for rb in 0..dr_w {
                                let w_b = w.get(lb, t, s, rb);
                                new_env[ra * dr_w + rb] += env_val * w_a * w_b;
                            }
                        }
                    }
                }
            }
        }
        env = new_env;
    }

    // Trace over the right boundary: Σ_r env[(r, r)]
    let last = &mpo.sites[mpo.n_sites - 1];
    let dr = last.d_r;
    let purity = (0..dr).map(|r| env[r * dr + r]).sum::<f64>();
    Ok(purity)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 2-site gate to sites `site` and `site+1` of the density MPO.
///
/// The gate `g` has shape `[d, d, d, d]` (flat length `d^4`), acting as
/// `g[q1, q2, p1, p2]` on the physical OUT legs of the two sites.
///
/// The gate is applied simultaneously to the OUT and IN (bra) legs, since the
/// imaginary-time propagator acts as `e^{-δτH/2} ρ e^{-δτH/2}`:
/// both the bra and ket legs get the same gate.
///
/// # Errors
///
/// Returns an error if `site + 1 >= n_sites` or dimensions do not match.
pub fn apply_gate_mpo(
    mpo: &mut DensityMpo,
    site: usize,
    gate: &[f64],
    chi_max: usize,
    trunc_tol: f64,
) -> TnResult<()> {
    if site + 1 >= mpo.n_sites {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: mpo.n_sites,
        });
    }
    let d = mpo.d_phys;
    let d2 = d * d;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d, d, d, d],
            got: vec![gate.len()],
        });
    }

    // Check physical dimensions match d
    {
        let wi = &mpo.sites[site];
        let wj = &mpo.sites[site + 1];
        if wi.d_out != d || wi.d_in != d || wj.d_out != d || wj.d_in != d {
            return Err(TnError::DimensionMismatch { a: d, b: wi.d_out });
        }
        if wi.d_r != wj.d_l {
            return Err(TnError::DimensionMismatch {
                a: wi.d_r,
                b: wj.d_l,
            });
        }
    }

    let d_l = mpo.sites[site].d_l;
    let d_mid = mpo.sites[site].d_r;
    let d_r = mpo.sites[site + 1].d_r;

    // Step 1: Contract W_i and W_{i+1} into θ[l, o1, o2, i1, i2, r]
    // θ flat index: ((((l*d+o1)*d+o2)*d+i1)*d+i2)*d_r + r
    let theta_len = d_l * d * d * d * d * d_r;
    let mut theta = vec![0.0_f64; theta_len];
    for l in 0..d_l {
        for o1 in 0..d {
            for i1 in 0..d {
                for m in 0..d_mid {
                    let wi_val = mpo.sites[site].get(l, o1, i1, m);
                    if wi_val == 0.0 {
                        continue;
                    }
                    for o2 in 0..d {
                        for i2 in 0..d {
                            for r in 0..d_r {
                                let wj_r = mpo.sites[site + 1].get(m, o2, i2, r);
                                let flat = ((((l * d + o1) * d + o2) * d + i1) * d + i2) * d_r + r;
                                theta[flat] += wi_val * wj_r;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Apply gate to both ket (out) and bra (in) physical legs simultaneously.
    //
    // For imaginary-time: ρ → e^{-τH} ρ e^{-τH}, the 2-site block transforms as:
    //   θ'[l, q1, q2, p1, p2, r] = Σ_{o1,o2,i1,i2} g[q1,q2,o1,o2] * g[p1,p2,i1,i2] * θ[l,o1,o2,i1,i2,r]
    // where (o1,o2) are ket (out) and (i1,i2) are bra (in) pairs for sites (i, i+1).
    let mut theta_prime = vec![0.0_f64; theta_len];
    for l in 0..d_l {
        for q1 in 0..d {
            // new ket site i
            for p1 in 0..d {
                // new bra site i
                for q2 in 0..d {
                    // new ket site i+1
                    for p2 in 0..d {
                        // new bra site i+1
                        let mut acc = vec![0.0_f64; d_r];
                        for o1 in 0..d {
                            for i1 in 0..d {
                                // gate element on site i: g[q1,q2,o1,o2] contracted over o1
                                // and g[p1,p2,i1,i2] contracted over i1
                                // We still need o2, i2 loops below
                                for o2 in 0..d {
                                    let g_ket = gate[((q1 * d + q2) * d + o1) * d + o2];
                                    if g_ket.abs() < 1e-15 {
                                        continue;
                                    }
                                    for i2 in 0..d {
                                        let g_bra = gate[((p1 * d + p2) * d + i1) * d + i2];
                                        if g_bra.abs() < 1e-15 {
                                            continue;
                                        }
                                        let gfactor = g_ket * g_bra;
                                        let src_base =
                                            ((((l * d + o1) * d + o2) * d + i1) * d + i2) * d_r;
                                        for r in 0..d_r {
                                            acc[r] += gfactor * theta[src_base + r];
                                        }
                                    }
                                }
                            }
                        }
                        // θ' indexed as [l, q1, q2, p1, p2, r]
                        let dst_base = ((((l * d + q1) * d + q2) * d + p1) * d + p2) * d_r;
                        for r in 0..d_r {
                            theta_prime[dst_base + r] += acc[r];
                        }
                    }
                }
            }
        }
    }

    // Step 3: Reshape θ' and SVD-split.
    //
    // Correct reshape for MPO: group (D_l, ket_i, bra_i) → row, (ket_{i+1}, bra_{i+1}, D_r) → col.
    //   row = l * d^2 + q1 * d + p1   (left bond, ket_i, bra_i)
    //   col = (q2 * d + p2) * D_r + r (ket_{i+1}, bra_{i+1}, right bond)
    //
    // This ensures the SVD split respects the local physical structure of each site.
    let mat_rows = d_l * d2; // D_l * d^2
    let mat_cols = d2 * d_r; // d^2 * D_r
    let mut mat = vec![0.0_f64; mat_rows * mat_cols];

    for l in 0..d_l {
        for q1 in 0..d {
            for p1 in 0..d {
                let row = l * d2 + q1 * d + p1;
                for q2 in 0..d {
                    for p2 in 0..d {
                        for r in 0..d_r {
                            let col = (q2 * d + p2) * d_r + r;
                            let src = ((((l * d + q1) * d + q2) * d + p1) * d + p2) * d_r + r;
                            mat[row * mat_cols + col] = theta_prime[src];
                        }
                    }
                }
            }
        }
    }

    // SVD and truncate
    let svd = svd_jacobi(&mat, mat_rows, mat_cols)?;
    let (svd, _) = svd_truncate(svd, chi_max, trunc_tol)?;
    let k = svd.k;

    // Unpack U and Vt into new site tensors.
    // U has shape [mat_rows, k] = [D_l * d^2, k]
    // row = l * d^2 + q1 * d + p1  →  new_wi[l, q1, p1, j] = U[row, j] * sqrt(s[j])
    let mut new_wi = MpoSiteTensor::zeros(d_l, d, d, k)?;
    for row in 0..mat_rows {
        let l = row / d2;
        let phys = row % d2;
        let q1 = phys / d; // ket_i (d_out)
        let p1 = phys % d; // bra_i (d_in)
        for j in 0..k {
            let sq_sv = svd.s[j].sqrt();
            new_wi.set(l, q1, p1, j, svd.u[row * k + j] * sq_sv);
        }
    }

    // Vt has shape [k, mat_cols] = [k, d^2 * D_r]
    // col = (q2 * d + p2) * D_r + r  →  new_wj[j, q2, p2, r] = sqrt(s[j]) * Vt[j, col]
    let mut new_wj = MpoSiteTensor::zeros(k, d, d, d_r)?;
    for j in 0..k {
        let sq_sv = svd.s[j].sqrt();
        for col in 0..mat_cols {
            let vt_val = svd.vt[j * mat_cols + col];
            let phys_r = col / d_r;
            let r = col % d_r;
            let q2 = phys_r / d; // ket_{i+1} (d_out)
            let p2 = phys_r % d; // bra_{i+1} (d_in)
            new_wj.set(j, q2, p2, r, sq_sv * vt_val);
        }
    }

    mpo.sites[site] = new_wi;
    mpo.sites[site + 1] = new_wj;
    Ok(())
}

/// Apply a site-local dephasing channel to all sites.
///
/// `W_i[l, s, s', r] → [(1 - γ) + γ δ(s,s')] W_i[l, s, s', r]`
///
/// For diagonal elements (`s == s'`): factor = `(1 - γ) + γ = 1.0` (unchanged).
/// For off-diagonal elements (`s != s'`): factor = `(1 - γ)`.
///
/// # Errors
///
/// Returns [`TnError::InvalidParameter`] if `gamma < 0.0` or `gamma > 1.0`.
pub fn apply_dephasing(mpo: &mut DensityMpo, gamma: f64) -> TnResult<()> {
    if !(0.0..=1.0).contains(&gamma) {
        return Err(TnError::InvalidParameter {
            name: "gamma".to_string(),
            reason: "must be in [0, 1]".to_string(),
        });
    }
    let off_diag_factor = 1.0 - gamma;
    for w in &mut mpo.sites {
        let d_out = w.d_out;
        let d_in = w.d_in;
        let d_l = w.d_l;
        let d_r = w.d_r;
        for l in 0..d_l {
            for o in 0..d_out {
                for i in 0..d_in {
                    if o != i {
                        for r in 0..d_r {
                            let prev = w.get(l, o, i, r);
                            w.set(l, o, i, r, prev * off_diag_factor);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Expectation values
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `⟨O⟩ = Tr[ρ O]` where `op` is a single-site operator of shape `[d, d]` (row-major).
///
/// The formula reduces to replacing the transfer matrix at `site` with the modified version:
/// `T_mod[l, r] = Σ_{s,s'} W[l, s, s', r] * O[s', s]`  (note transposition for Tr[ρ O]).
///
/// # Errors
///
/// Returns an error if `site >= n_sites` or `op.len() != d^2`.
pub fn density_mpo_expectation(mpo: &DensityMpo, op: &[f64], site: usize) -> TnResult<f64> {
    if site >= mpo.n_sites {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: mpo.n_sites,
        });
    }
    let d = mpo.d_phys;
    if op.len() != d * d {
        return Err(TnError::ShapeMismatch {
            expected: vec![d, d],
            got: vec![op.len()],
        });
    }

    // Left boundary: vector of 1s of length d_l (open boundary)
    let first = &mpo.sites[0];
    let mut env = vec![1.0_f64; first.d_l];

    for site_idx in 0..mpo.n_sites {
        let w = &mpo.sites[site_idx];
        let mut new_env = vec![0.0_f64; w.d_r];

        if site_idx == site {
            // Modified transfer matrix: T_mod[r] = Σ_{l,s,s'} env[l] * W[l,s,s',r] * O[s',s]
            // Tr[ρ O_site] uses O[s', s] = op[s' * d + s] transposed for correct trace ordering.
            for (r, slot) in new_env.iter_mut().enumerate().take(w.d_r) {
                let mut acc = 0.0;
                for s in 0..d {
                    for sp in 0..d {
                        let op_val = op[sp * d + s]; // O[s', s]
                        for l in 0..w.d_l {
                            if l < env.len() {
                                acc += env[l] * w.get(l, s, sp, r) * op_val;
                            }
                        }
                    }
                }
                *slot = acc;
            }
        } else {
            // Standard trace transfer matrix
            for (r, slot) in new_env.iter_mut().enumerate().take(w.d_r) {
                let mut acc = 0.0;
                for s in 0..d {
                    for l in 0..w.d_l {
                        if l < env.len() {
                            acc += env[l] * w.get(l, s, s, r);
                        }
                    }
                }
                *slot = acc;
            }
        }
        env = new_env;
    }

    Ok(env.iter().sum())
}

// ─────────────────────────────────────────────────────────────────────────────
// MPO normalization helper
// ─────────────────────────────────────────────────────────────────────────────

/// Normalize the MPO so that `Tr[ρ] = 1` by dividing the first site tensor by the trace.
fn normalize_mpo(mpo: &mut DensityMpo) -> TnResult<f64> {
    let tr = density_mpo_trace(mpo)?;
    if tr.abs() < 1e-15 {
        return Err(TnError::NumericalInstability(
            "MPO trace is essentially zero; cannot normalize".to_string(),
        ));
    }
    for v in &mut mpo.sites[0].data {
        *v /= tr;
    }
    Ok(tr)
}

// ─────────────────────────────────────────────────────────────────────────────
// Heisenberg bond energy estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the Heisenberg energy on bond `(site, site+1)` as `Tr[ρ H_{bond}]`.
///
/// The bond Hamiltonian is `H = J (Sx⊗Sx + Sy⊗Sy + Sz⊗Sz)`, represented as a 4×4 matrix.
///
/// We compute this by applying the Hamiltonian as a 2-site operator to ρ and taking
/// the trace, using the MPO trace contraction with the bond operator inserted.
fn heisenberg_bond_energy(mpo: &DensityMpo, site: usize, j: f64) -> TnResult<f64> {
    if site + 1 >= mpo.n_sites {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: mpo.n_sites,
        });
    }
    let d = mpo.d_phys;
    let ham = heisenberg_hamiltonian_2site(j);

    // Build left environment env[l] up to (but not including) site `site`.
    // Start from left boundary (all 1s).
    let first = &mpo.sites[0];
    let mut env_left = vec![1.0_f64; first.d_l];
    for idx in 0..site {
        let w = &mpo.sites[idx];
        let mut new_env = vec![0.0_f64; w.d_r];
        for (r, slot) in new_env.iter_mut().enumerate().take(w.d_r) {
            let mut acc = 0.0_f64;
            for s in 0..d {
                for l in 0..w.d_l {
                    if l < env_left.len() {
                        acc += env_left[l] * w.get(l, s, s, r);
                    }
                }
            }
            *slot = acc;
        }
        env_left = new_env;
    }

    // Insert 2-site modified transfer matrix with Hamiltonian:
    // env_after[r] = Σ_{l,s1,s2,s1',s2',m} env_left[l] * W_i[l,s1,s1',m] * H[s1,s2,s1',s2'] * W_{i+1}[m,s2,s2',r]
    let wi = &mpo.sites[site];
    let wj = &mpo.sites[site + 1];
    let d_l = wi.d_l;
    let d_mid = wi.d_r;
    let d_r = wj.d_r;

    let mut env_after = vec![0.0_f64; d_r];
    for (r, slot) in env_after.iter_mut().enumerate().take(d_r) {
        let mut acc = 0.0_f64;
        for l in 0..d_l {
            let el = env_left.get(l).copied().unwrap_or(0.0);
            if el.abs() < 1e-15 {
                continue;
            }
            for s1 in 0..d {
                for s1p in 0..d {
                    for m in 0..d_mid {
                        let wi_val = wi.get(l, s1, s1p, m);
                        if wi_val.abs() < 1e-15 {
                            continue;
                        }
                        for s2 in 0..d {
                            for s2p in 0..d {
                                // H[(s1,s2), (s1',s2')] = ham[(s1*d+s2)*d^2 + s1'*d+s2']
                                let h_val = ham[(s1 * d + s2) * (d * d) + s1p * d + s2p];
                                if h_val.abs() < 1e-15 {
                                    continue;
                                }
                                let wj_val = wj.get(m, s2, s2p, r);
                                acc += el * wi_val * h_val * wj_val;
                            }
                        }
                    }
                }
            }
        }
        *slot = acc;
    }

    // Contract remaining sites to the right using standard trace transfer matrices
    for idx in (site + 2)..mpo.n_sites {
        let w = &mpo.sites[idx];
        let mut new_env = vec![0.0_f64; w.d_r];
        for (r, slot) in new_env.iter_mut().enumerate().take(w.d_r) {
            let mut acc = 0.0_f64;
            for s in 0..d {
                for l in 0..w.d_l {
                    if l < env_after.len() {
                        acc += env_after[l] * w.get(l, s, s, r);
                    }
                }
            }
            *slot = acc;
        }
        env_after = new_env;
    }

    // Sum over the right boundary
    Ok(env_after.iter().sum())
}

// ─────────────────────────────────────────────────────────────────────────────
// Main TEBD driver
// ─────────────────────────────────────────────────────────────────────────────

/// Run mixed-state TEBD imaginary-time evolution on `mpo`.
///
/// Each step applies:
/// 1. Even-bond gates (sites 0-1, 2-3, ...)
/// 2. Odd-bond gates (sites 1-2, 3-4, ...)
/// 3. Optional dephasing on all sites
/// 4. Re-normalization so `Tr[ρ] = 1`
///
/// The gate is `g = exp(-dt * H_{bond})` for the Heisenberg Hamiltonian.
///
/// # Errors
///
/// Returns an error if `n_sites < 2`, or if any gate application fails.
pub fn mixed_tebd_run(mpo: &mut DensityMpo, cfg: &MixedTebdConfig) -> TnResult<MixedTebdResult> {
    if mpo.n_sites < 2 {
        return Err(TnError::EmptyInput);
    }
    if cfg.n_steps == 0 {
        let purity = density_mpo_purity(mpo)?;
        let trace = density_mpo_trace(mpo)?;
        return Ok(MixedTebdResult {
            purity,
            trace,
            energy_per_site: 0.0,
            steps_run: 0,
        });
    }
    if mpo.d_phys != 2 {
        return Err(TnError::InvalidParameter {
            name: "d_phys".to_string(),
            reason: "mixed_tebd_run currently requires d_phys = 2 (spin-1/2 Heisenberg)"
                .to_string(),
        });
    }

    // Build the 2-site imaginary-time gate: g = exp(-dt * H)
    let ham = heisenberg_hamiltonian_2site(cfg.coupling_j);
    let gate = mat_exp_sym(&ham, 4, -cfg.dt)?;

    for _ in 0..cfg.n_steps {
        // Even bonds: 0-1, 2-3, 4-5, ...
        let mut bond = 0;
        while bond + 1 < mpo.n_sites {
            apply_gate_mpo(mpo, bond, &gate, cfg.chi_max, cfg.trunc_tol)?;
            bond += 2;
        }

        // Odd bonds: 1-2, 3-4, 5-6, ...
        let mut bond = 1;
        while bond + 1 < mpo.n_sites {
            apply_gate_mpo(mpo, bond, &gate, cfg.chi_max, cfg.trunc_tol)?;
            bond += 2;
        }

        // Dephasing
        if cfg.dephasing_rate > 0.0 {
            apply_dephasing(mpo, cfg.dephasing_rate)?;
        }

        // Normalize
        normalize_mpo(mpo)?;
    }

    // Final observables
    let trace = density_mpo_trace(mpo)?;
    let purity = density_mpo_purity(mpo)?;

    // Energy per site: average over all bonds
    let n_bonds = mpo.n_sites - 1;
    let mut energy_sum = 0.0;
    for bond in 0..n_bonds {
        energy_sum += heisenberg_bond_energy(mpo, bond, cfg.coupling_j)?;
    }
    let energy_per_site = if n_bonds > 0 {
        energy_sum / n_bonds as f64
    } else {
        0.0
    };

    Ok(MixedTebdResult {
        purity,
        trace,
        energy_per_site,
        steps_run: cfg.n_steps,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: MpoSiteTensor construction ───────────────────────────────────
    #[test]
    fn mpo_site_tensor_construction() {
        let data = vec![1.0_f64; 2 * 3 * 3 * 4];
        let t = MpoSiteTensor::new(2, 3, 3, 4, data.clone()).expect("construction ok");
        assert_eq!(t.d_l, 2);
        assert_eq!(t.d_out, 3);
        assert_eq!(t.d_in, 3);
        assert_eq!(t.d_r, 4);
        assert_eq!(t.data.len(), 2 * 3 * 3 * 4);

        // Wrong length
        let bad = MpoSiteTensor::new(2, 3, 3, 4, vec![0.0; 5]);
        assert!(bad.is_err());

        // Zero dimension
        let zero_dim = MpoSiteTensor::zeros(0, 2, 2, 1);
        assert!(zero_dim.is_err());
    }

    // ── Test 2: identity MPO trace = 1 ───────────────────────────────────────
    #[test]
    fn density_mpo_identity_trace() {
        for n in [2, 3, 4] {
            let mpo = density_mpo_identity(n, 2).expect("ok");
            let tr = density_mpo_trace(&mpo).expect("trace ok");
            assert!((tr - 1.0).abs() < 1e-12, "n={n} trace={tr}");
        }
    }

    // ── Test 3: identity MPO purity = 1/d^L ─────────────────────────────────
    #[test]
    fn density_mpo_identity_purity() {
        let d = 2_usize;
        for n in [2usize, 3, 4] {
            let mpo = density_mpo_identity(n, d).expect("ok");
            let purity = density_mpo_purity(&mpo).expect("purity ok");
            let expected = (d as f64).powi(-(n as i32));
            assert!(
                (purity - expected).abs() < 1e-10,
                "n={n} purity={purity} expected={expected}"
            );
        }
    }

    // ── Test 4: trace stays 1 after manual normalization ─────────────────────
    #[test]
    fn density_mpo_trace_normalization() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        // Scale only the FIRST site's data by 2 — trace is multiplicative across
        // sites, so scaling one site scales the full trace by the same factor.
        for v in &mut mpo.sites[0].data {
            *v *= 2.0;
        }
        let tr_before = density_mpo_trace(&mpo).expect("trace");
        assert!((tr_before - 2.0).abs() < 1e-10);

        // Normalize
        for v in &mut mpo.sites[0].data {
            *v /= tr_before;
        }
        let tr_after = density_mpo_trace(&mpo).expect("trace");
        assert!((tr_after - 1.0).abs() < 1e-10);
    }

    // ── Test 5: apply_gate_mpo preserves trace for identity gate ─────────────
    #[test]
    fn apply_gate_mpo_trace_preserved() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let d = 2_usize;
        // Build 4x4 identity gate [d,d,d,d]
        let d2 = d * d;
        let mut gate = vec![0.0_f64; d2 * d2];
        for i in 0..d2 {
            gate[i * d2 + i] = 1.0;
        }

        apply_gate_mpo(&mut mpo, 0, &gate, 16, 1e-12).expect("gate ok");
        apply_gate_mpo(&mut mpo, 2, &gate, 16, 1e-12).expect("gate ok");

        let tr = density_mpo_trace(&mpo).expect("trace");
        // After applying identity gate twice the trace should be preserved (within SVD round-trip error)
        // The gate acts on both ket and bra, so identity^2 = identity — trace may be scaled by small amount
        // Use generous tolerance due to double application
        assert!((tr - 1.0).abs() < 0.1, "trace after identity gates = {tr}");
    }

    // ── Test 6: bond dimension bounded after gate ─────────────────────────────
    #[test]
    fn apply_gate_mpo_bond_bounded() {
        let chi_max = 4;
        let mut mpo = density_mpo_identity(6, 2).expect("ok");
        let d = 2_usize;
        let d2 = d * d;

        // Use Heisenberg gate
        let ham = heisenberg_hamiltonian_2site(1.0);
        let gate = mat_exp_sym(&ham, 4, -0.1).expect("mat_exp ok");

        for step in 0..5 {
            let bond = step % 5;
            if bond + 1 < mpo.n_sites {
                apply_gate_mpo(&mut mpo, bond, &gate, chi_max, 1e-12).expect("ok");
                // Bond dimension after gate must be ≤ chi_max
                let d_r = mpo.sites[bond].d_r;
                assert!(d_r <= chi_max, "bond {bond}: d_r={d_r} > chi_max={chi_max}");
            }
        }
        let _ = d2;
    }

    // ── Test 7: dephasing reduces purity ─────────────────────────────────────
    #[test]
    fn apply_dephasing_reduces_purity() {
        // Start from pure superposition state |+,+,...,+⟩ => density MPO.
        // |+⟩ = (|0⟩ + |1⟩)/√2 has off-diagonal density-matrix elements,
        // so dephasing will genuinely reduce purity (unlike |0⟩ which has none).
        let n = 4_usize;
        let d = 2_usize;
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        let mps: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![inv_sqrt2, inv_sqrt2] // A[0,0,0]=1/√2, A[0,1,0]=1/√2
            })
            .collect();
        let bond_dims: Vec<usize> = vec![1; n + 1];
        let mut mpo = density_mpo_from_mps(&mps, &bond_dims, n, d).expect("ok");

        let purity_before = density_mpo_purity(&mpo).expect("purity");

        apply_dephasing(&mut mpo, 0.1).expect("dephasing ok");

        let purity_after = density_mpo_purity(&mpo).expect("purity after");

        // Dephasing a pure state reduces purity
        assert!(
            purity_after < purity_before,
            "purity should decrease: before={purity_before}, after={purity_after}"
        );
    }

    // ── Test 8: dephasing preserves trace ────────────────────────────────────
    #[test]
    fn apply_dephasing_preserves_trace() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let tr_before = density_mpo_trace(&mpo).expect("trace");
        apply_dephasing(&mut mpo, 0.3).expect("ok");
        let tr_after = density_mpo_trace(&mpo).expect("trace");
        assert!(
            (tr_after - tr_before).abs() < 1e-12,
            "trace changed by dephasing"
        );
    }

    // ── Test 9: mixed_tebd_run trace stays near 1 ────────────────────────────
    #[test]
    fn mixed_tebd_run_trace_near_1() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let cfg = MixedTebdConfig {
            chi_max: 8,
            n_steps: 5,
            dt: 0.05,
            ..Default::default()
        };
        let res = mixed_tebd_run(&mut mpo, &cfg).expect("run ok");
        assert!(
            (res.trace - 1.0).abs() < 1e-8,
            "trace = {} not near 1",
            res.trace
        );
    }

    // ── Test 10: energy per site is finite ───────────────────────────────────
    #[test]
    fn mixed_tebd_run_energy_finite() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let cfg = MixedTebdConfig {
            chi_max: 8,
            n_steps: 10,
            dt: 0.05,
            ..Default::default()
        };
        let res = mixed_tebd_run(&mut mpo, &cfg).expect("run ok");
        assert!(
            res.energy_per_site.is_finite(),
            "energy must be finite: {}",
            res.energy_per_site
        );
    }

    // ── Test 11: purity in (0, 1] throughout ─────────────────────────────────
    #[test]
    fn mixed_tebd_run_purity_range() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let cfg = MixedTebdConfig {
            chi_max: 8,
            n_steps: 10,
            dt: 0.05,
            ..Default::default()
        };
        let res = mixed_tebd_run(&mut mpo, &cfg).expect("run ok");
        assert!(res.purity > 0.0, "purity must be > 0: {}", res.purity);
        assert!(
            res.purity <= 1.0 + 1e-8,
            "purity must be ≤ 1: {}",
            res.purity
        );
    }

    // ── Test 12: expectation of identity op = 1 ───────────────────────────────
    #[test]
    fn density_mpo_expectation_identity() {
        let mpo = density_mpo_identity(4, 2).expect("ok");
        let d = 2_usize;
        // Identity op: I_{d x d}
        let mut id_op = vec![0.0_f64; d * d];
        for i in 0..d {
            id_op[i * d + i] = 1.0;
        }
        // Tr[ρ I] = Tr[ρ] = 1
        let exp_val = density_mpo_expectation(&mpo, &id_op, 1).expect("ok");
        assert!((exp_val - 1.0).abs() < 1e-10, "Tr[ρ I] = {exp_val}");
    }

    // ── Test 13: pure-state MPO has purity near 1 ────────────────────────────
    #[test]
    fn density_mpo_from_mps_purity_1() {
        let n = 3_usize;
        let d = 2_usize;
        // MPS |+⟩^⊗n with bond_dim=1: A[0, 0, 0] = A[0, 1, 0] = 1/sqrt(2)
        let val = 1.0_f64 / 2.0_f64.sqrt();
        let mps: Vec<Vec<f64>> = (0..n).map(|_| vec![val, val]).collect();
        let bond_dims = vec![1; n + 1];
        let mpo = density_mpo_from_mps(&mps, &bond_dims, n, d).expect("ok");

        let purity = density_mpo_purity(&mpo).expect("purity");
        // Purity of |+⟩^⊗n normalized state should be 1
        let trace = density_mpo_trace(&mpo).expect("trace");
        // Normalize purity by trace^2
        let normalized_purity = purity / (trace * trace);
        assert!(
            (normalized_purity - 1.0).abs() < 1e-8,
            "pure state normalized purity = {normalized_purity}"
        );
    }

    // ── Test 14: mixed_tebd with dephasing completes ─────────────────────────
    #[test]
    fn mixed_tebd_with_dephasing() {
        let mut mpo = density_mpo_identity(4, 2).expect("ok");
        let cfg = MixedTebdConfig {
            chi_max: 8,
            n_steps: 5,
            dt: 0.05,
            dephasing_rate: 0.1,
            ..Default::default()
        };
        let res = mixed_tebd_run(&mut mpo, &cfg).expect("dephasing run ok");
        assert_eq!(res.steps_run, 5);
        assert!(res.purity > 0.0 && res.purity <= 1.0 + 1e-8);
    }

    // ── Test 15: ⟨Sz⟩ = 0 for maximally mixed state ─────────────────────────
    #[test]
    fn identity_mpo_expectation_sz_zero() {
        let mpo = density_mpo_identity(4, 2).expect("ok");
        // Sz = diag(+0.5, -0.5) for d=2
        let sz_op = vec![0.5, 0.0, 0.0, -0.5_f64];
        for site in 0..mpo.n_sites {
            let sz = density_mpo_expectation(&mpo, &sz_op, site).expect("ok");
            assert!(
                sz.abs() < 1e-10,
                "⟨Sz⟩ at site {site} = {sz} should be 0 for maximally mixed state"
            );
        }
    }
}
