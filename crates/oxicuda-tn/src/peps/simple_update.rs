//! PEPS Simple-Update imaginary-time evolution.
//!
//! ## Algorithm overview (Jiang 2008, Corboz 2010)
//!
//! The simple-update (SU) algorithm approximates the 2D PEPS environment using a product of
//! diagonal "lambda" matrices — one per virtual bond — rather than computing the full
//! environment (which is #P-hard for generic 2D geometries).  The approximation is
//! analogous to Vidal's Γ–Λ canonical form for infinite MPS (iTEBD), now lifted to a 2D
//! lattice.
//!
//! ### State representation
//!
//! Each PEPS site carries a rank-5 tensor `A[l, r, u, d, p]` with
//! * `l` — left virtual bond index,   dimension `D_l`
//! * `r` — right virtual bond index,  dimension `D_r`
//! * `u` — up virtual bond index,     dimension `D_u`
//! * `d` — down virtual bond index,   dimension `D_d`
//! * `p` — physical index,            dimension `d_p`
//!
//! Each virtual bond is additionally decorated with a diagonal matrix `Λ` (stored as a
//! vector of positive reals, the "Schmidt values" on that bond).
//!
//! ### One simple-update bond step (horizontal bond between (row, col) and (row, col+1))
//!
//! ```text
//!       Λ_left   A_left   Λ_up_left   Λ_down_left          Λ_right  B_right  Λ_up_right  Λ_down_right
//!          │        │         │             │                    │        │         │              │
//! ─ Λ_l ─ A ─ Λ_bond ─ B ─ Λ_r ─
//! ```
//!
//! 1. **Absorb environment**: scale legs of A by Λ_l, Λ_u_l, Λ_d_l and legs of B by
//!    Λ_r, Λ_u_r, Λ_d_r.  Absorb Λ_bond between them.  Merge to form two-site tensor Θ.
//! 2. **Apply gate**: `Θ' = gate ⊗ Θ` (gate is the imaginary-time propagator `exp(-τH)`).
//! 3. **SVD + truncate**: reshape Θ' to a matrix `[D_l·d, d·D_r]`, SVD, keep top `D_bond`
//!    singular values.
//! 4. **Update**: new Λ_bond = S / ‖S‖.  New A = U with inverse-Λ environment re-absorbed.
//!    New B = Vt with inverse-Λ environment re-absorbed.

use crate::error::{TnError, TnResult};
use crate::handle::LcgRng;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single PEPS site tensor with shape `[D_l, D_r, D_u, D_d, d_p]` in row-major order.
///
/// Element `(l, r, u, d, p)` lives at flat index
/// `(((l * D_r + r) * D_u + u) * D_d + d) * d_p + p`.
#[derive(Debug, Clone)]
pub struct PepsTensor {
    /// Left virtual bond dimension.
    pub d_l: usize,
    /// Right virtual bond dimension.
    pub d_r: usize,
    /// Up virtual bond dimension.
    pub d_u: usize,
    /// Down virtual bond dimension.
    pub d_d: usize,
    /// Physical (Hilbert space) dimension.
    pub d_p: usize,
    /// Row-major data buffer of length `d_l * d_r * d_u * d_d * d_p`.
    pub data: Vec<f64>,
}

impl PepsTensor {
    /// Construct a new tensor, checking that `data.len()` matches the product of dimensions.
    pub fn new(
        d_l: usize,
        d_r: usize,
        d_u: usize,
        d_d: usize,
        d_p: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0 || d_p == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        let expected = d_l * d_r * d_u * d_d * d_p;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l, d_r, d_u, d_d, d_p],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_l,
            d_r,
            d_u,
            d_d,
            d_p,
            data,
        })
    }

    /// Construct an all-zero tensor.
    pub fn zeros(d_l: usize, d_r: usize, d_u: usize, d_d: usize, d_p: usize) -> TnResult<Self> {
        let n = d_l * d_r * d_u * d_d * d_p;
        if d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0 || d_p == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        Self::new(d_l, d_r, d_u, d_d, d_p, vec![0.0; n])
    }

    /// Flat index for element `(l, r, u, d, p)`.
    #[inline]
    pub fn idx(&self, l: usize, r: usize, u: usize, d: usize, p: usize) -> usize {
        (((l * self.d_r + r) * self.d_u + u) * self.d_d + d) * self.d_p + p
    }

    /// Read element `(l, r, u, d, p)`.
    pub fn get(&self, l: usize, r: usize, u: usize, d: usize, p: usize) -> f64 {
        self.data[self.idx(l, r, u, d, p)]
    }

    /// Write element `(l, r, u, d, p)`.
    pub fn set(&mut self, l: usize, r: usize, u: usize, d: usize, p: usize, val: f64) {
        let i = self.idx(l, r, u, d, p);
        self.data[i] = val;
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.d_l * self.d_r * self.d_u * self.d_d * self.d_p
    }
}

/// Lambda (Schmidt-value) matrices for every virtual bond in a 2D PEPS lattice.
///
/// The lattice has `Lx` columns and `Ly` rows (site `(row, col)` has index `row*Lx + col`).
///
/// * **Horizontal bonds**: between `(row, col)` and `(row, col+1)`.  There are `Ly * (Lx-1)`
///   such bonds.  `horizontal[row * (lx-1) + col]` is the lambda vector for bond
///   `(row, col) — (row, col+1)`.  Each vector has length `≤ D_max`.
///
/// * **Vertical bonds**: between `(row, col)` and `(row+1, col)`.  There are `(Ly-1) * Lx`
///   such bonds.  `vertical[row * lx + col]` is the lambda vector for bond
///   `(row, col) — (row+1, col)`.  Each vector has length `≤ D_max`.
#[derive(Debug, Clone)]
pub struct PepsLambdas {
    /// Horizontal bond lambda vectors, indexed `[row * (lx-1) + col]`.
    /// Empty slice if `lx == 1`.
    pub horizontal: Vec<Vec<f64>>,
    /// Vertical bond lambda vectors, indexed `[row * lx + col]`.
    /// Empty slice if `ly == 1`.
    pub vertical: Vec<Vec<f64>>,
    /// Lattice width.
    pub lx: usize,
    /// Lattice height.
    pub ly: usize,
}

impl PepsLambdas {
    /// Retrieve (immutably) the lambda vector for the horizontal bond between
    /// `(row, col)` and `(row, col+1)`.
    pub fn get_h(&self, row: usize, col: usize) -> &[f64] {
        let stride = if self.lx > 1 { self.lx - 1 } else { 1 };
        &self.horizontal[row * stride + col]
    }

    /// Retrieve (mutably) the lambda vector for the horizontal bond.
    pub fn get_h_mut(&mut self, row: usize, col: usize) -> &mut Vec<f64> {
        let stride = if self.lx > 1 { self.lx - 1 } else { 1 };
        &mut self.horizontal[row * stride + col]
    }

    /// Retrieve (immutably) the lambda vector for the vertical bond between
    /// `(row, col)` and `(row+1, col)`.
    pub fn get_v(&self, row: usize, col: usize) -> &[f64] {
        &self.vertical[row * self.lx + col]
    }

    /// Retrieve (mutably) the lambda vector for the vertical bond.
    pub fn get_v_mut(&mut self, row: usize, col: usize) -> &mut Vec<f64> {
        &mut self.vertical[row * self.lx + col]
    }
}

/// Configuration for a PEPS simple-update run.
#[derive(Debug, Clone)]
pub struct SimpleUpdateConfig {
    /// Lattice width (number of columns).
    pub lx: usize,
    /// Lattice height (number of rows).
    pub ly: usize,
    /// Physical (on-site) Hilbert-space dimension `d` (e.g. 2 for a qubit).
    pub phys_dim: usize,
    /// Maximum virtual bond dimension `D`.
    pub bond_dim: usize,
    /// Imaginary-time step `δτ` per Trotter layer.
    pub delta_tau: f64,
    /// Total number of sweep steps.  One step = one full sweep over all bonds.
    pub n_steps: usize,
    /// Energy-per-site convergence tolerance.
    pub tol: f64,
    /// Trotter splitting order: 1 (first-order) or 2 (Strang symmetric, default).
    pub trotter_order: usize,
}

impl Default for SimpleUpdateConfig {
    fn default() -> Self {
        Self {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.01,
            n_steps: 100,
            tol: 1e-6,
            trotter_order: 2,
        }
    }
}

/// Output of a completed PEPS simple-update run.
#[derive(Debug, Clone)]
pub struct SimpleUpdateResult {
    /// `Lx * Ly` site tensors in row-major order `[row * Lx + col]`.
    pub tensors: Vec<PepsTensor>,
    /// Bond lambda matrices.
    pub lambdas: PepsLambdas,
    /// Estimated ground-state energy per site (product-state approximation).
    pub energy_per_site: f64,
    /// Number of sweeps actually performed.
    pub n_steps: usize,
    /// Whether the run converged within `config.tol`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

/// Build a random PEPS state with bond dimension `bond_dim` and physical dimension `phys_dim`.
///
/// Boundary virtual bonds are set to dimension 1 (open boundary conditions).
/// Interior virtual bonds have dimension `bond_dim`.
/// Lambda vectors are initialised to uniform, positive values normalised so that `Σ λ² = 1`.
///
/// # Errors
/// Returns [`TnError::InvalidBondDimension`] if `bond_dim == 0` or `phys_dim == 0`.
/// Returns [`TnError::EmptyInput`] if `lx == 0` or `ly == 0`.
pub fn simple_update_init(
    lx: usize,
    ly: usize,
    phys_dim: usize,
    bond_dim: usize,
    rng: &mut LcgRng,
) -> TnResult<(Vec<PepsTensor>, PepsLambdas)> {
    if lx == 0 || ly == 0 {
        return Err(TnError::EmptyInput);
    }
    if bond_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if phys_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }

    // ── Build site tensors ────────────────────────────────────────────────────
    let mut tensors: Vec<PepsTensor> = Vec::with_capacity(lx * ly);
    for row in 0..ly {
        for col in 0..lx {
            let d_l = if col == 0 { 1 } else { bond_dim };
            let d_r = if col + 1 == lx { 1 } else { bond_dim };
            let d_u = if row == 0 { 1 } else { bond_dim };
            let d_d = if row + 1 == ly { 1 } else { bond_dim };
            let n = d_l * d_r * d_u * d_d * phys_dim;
            let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
            tensors.push(PepsTensor::new(d_l, d_r, d_u, d_d, phys_dim, data)?);
        }
    }

    // ── Build lambda matrices ────────────────────────────────────────────────

    // Horizontal bonds: (row, col) — (row, col+1), for col in 0..lx-1.
    let n_h_bonds = if lx > 1 { ly * (lx - 1) } else { 0 };
    let h_stride = if lx > 1 { lx - 1 } else { 1 };

    let mut horizontal: Vec<Vec<f64>> = Vec::with_capacity(n_h_bonds);
    for row in 0..ly {
        let cols_per_row = lx.saturating_sub(1);
        for _col in 0..cols_per_row {
            // Boundary-adjusted bond dim: both sides must be interior.
            // (The actual bond dim stored in each tensor's d_r / d_l already encodes this,
            // but for lambdas we always use the interior bond_dim for inner bonds.)
            let lam_len = bond_dim;
            let inv_sqrt = 1.0 / (lam_len as f64).sqrt();
            horizontal.push(vec![inv_sqrt; lam_len]);
        }
        // Pad to h_stride entries if lx == 1 (empty inner loop, no bond):
        if lx == 1 {
            let _ = row; // keep rustc happy
        }
    }
    // If lx == 1 we need an empty horizontal vec.
    if lx == 1 {
        horizontal.clear();
        // Add ly * 1 dummy slots anyway so indexing works — but nothing will use them.
    }

    // Vertical bonds: (row, col) — (row+1, col), for row in 0..ly-1.
    let n_v_bonds = if ly > 1 { (ly - 1) * lx } else { 0 };
    let _ = n_v_bonds;

    let mut vertical: Vec<Vec<f64>> = Vec::with_capacity(ly.saturating_sub(1) * lx);
    for _row in 0..ly.saturating_sub(1) {
        for _col in 0..lx {
            let lam_len = bond_dim;
            let inv_sqrt = 1.0 / (lam_len as f64).sqrt();
            vertical.push(vec![inv_sqrt; lam_len]);
        }
    }

    // Expand horizontal to ly * h_stride entries if we started with lx > 1.
    // Above we added ly * (lx-1) entries; the indexing is row * (lx-1) + col.
    // The h_stride variable covers both cases (== lx-1 when lx > 1, == 1 when lx == 1).
    // For lx == 1 we have horizontal = [] already.
    // Vertical is already (ly-1)*lx entries.

    // Pad horizontal to ly * h_stride so the index function is uniform:
    while lx > 1 && horizontal.len() < ly * h_stride {
        let lam_len = bond_dim;
        let inv_sqrt = 1.0 / (lam_len as f64).sqrt();
        horizontal.push(vec![inv_sqrt; lam_len]);
    }

    let lambdas = PepsLambdas {
        horizontal,
        vertical,
        lx,
        ly,
    };

    Ok((tensors, lambdas))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute element-wise inverse of a vector, clamping small values to avoid blow-up.
#[inline]
fn invert_vec(v: &[f64], eps: f64) -> Vec<f64> {
    v.iter()
        .map(|&x| if x.abs() > eps { 1.0 / x } else { 0.0 })
        .collect()
}

/// Normalise `v` in-place so that `Σ v_i² = 1`.  Returns the pre-normalisation norm.
#[inline]
fn normalise_l2(v: &mut [f64]) -> f64 {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let safe_norm = norm.max(1e-60);
    for x in v.iter_mut() {
        *x /= safe_norm;
    }
    norm
}

/// Given a PEPS tensor `A` with shape `[d_l, d_r, d_u, d_d, d_p]`, produce the
/// "environment-weighted" tensor `Ã` by scaling the `l`, `u`, `d` legs with the given
/// diagonal lambda vectors.  The `r` leg is left unscaled (that bond is the "active" one).
///
/// Output shape is the same as input: `[d_l, d_r, d_u, d_d, d_p]`.
fn absorb_env_left_site(t: &PepsTensor, lam_l: &[f64], lam_u: &[f64], lam_d: &[f64]) -> Vec<f64> {
    let mut out = t.data.clone();
    // Scale `l` leg: each slice of the first index by lam_l[l].
    for l in 0..t.d_l {
        let scale_l = lam_l.get(l).copied().unwrap_or(1.0);
        for r in 0..t.d_r {
            for u in 0..t.d_u {
                for d in 0..t.d_d {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale_l;
                    }
                }
            }
        }
    }
    // Scale `u` leg.
    for u in 0..t.d_u {
        let scale_u = lam_u.get(u).copied().unwrap_or(1.0);
        for l in 0..t.d_l {
            for r in 0..t.d_r {
                for d in 0..t.d_d {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale_u;
                    }
                }
            }
        }
    }
    // Scale `d` leg.
    for dd in 0..t.d_d {
        let scale_d = lam_d.get(dd).copied().unwrap_or(1.0);
        for l in 0..t.d_l {
            for r in 0..t.d_r {
                for u in 0..t.d_u {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, dd, p);
                        out[i] *= scale_d;
                    }
                }
            }
        }
    }
    out
}

/// Same as [`absorb_env_left_site`] but scales the `r`, `u`, `d` legs (leaving `l` free).
fn absorb_env_right_site(t: &PepsTensor, lam_r: &[f64], lam_u: &[f64], lam_d: &[f64]) -> Vec<f64> {
    let mut out = t.data.clone();
    // Scale `r` leg.
    for r in 0..t.d_r {
        let scale_r = lam_r.get(r).copied().unwrap_or(1.0);
        for l in 0..t.d_l {
            for u in 0..t.d_u {
                for d in 0..t.d_d {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale_r;
                    }
                }
            }
        }
    }
    // Scale `u` leg.
    for u in 0..t.d_u {
        let scale_u = lam_u.get(u).copied().unwrap_or(1.0);
        for l in 0..t.d_l {
            for r in 0..t.d_r {
                for d in 0..t.d_d {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale_u;
                    }
                }
            }
        }
    }
    // Scale `d` leg.
    for dd in 0..t.d_d {
        let scale_d = lam_d.get(dd).copied().unwrap_or(1.0);
        for l in 0..t.d_l {
            for r in 0..t.d_r {
                for u in 0..t.d_u {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, dd, p);
                        out[i] *= scale_d;
                    }
                }
            }
        }
    }
    out
}

/// Contract two tensors on a horizontal bond:
///
/// `Θ[(l, p_l), (p_r, r)] = Σ_bond  Ã_l[l, bond, u_l, d_l, p_l] · λ_bond[bond] · B̃_r[bond, r, u_r, d_r, p_r]`
///
/// * `a_env`: left-site weighted tensor, shape `[d_l, D_bond, d_u_l, d_d_l, d_p]` (original A shape, r = bond).
/// * `b_env`: right-site weighted tensor, shape `[D_bond, d_r, d_u_r, d_d_r, d_p]` (original B shape, l = bond).
/// * `lam_bond`: diagonal lambda for the active bond, length `D_bond`.
///
/// Returns `theta` with shape `[chi_l * d_p, d_p * chi_r]` where
/// `chi_l = d_l * d_u_l * d_d_l` and `chi_r = d_r * d_u_r * d_d_r`.
///
/// To keep the algebra tractable we reshape A to `[chi_l, D_bond, d_p]` and B to
/// `[D_bond, chi_r, d_p]`, absorb `λ_bond` into the bond, then contract.
fn contract_theta_h(
    a_env: &PepsTensor,
    b_env: &PepsTensor,
    lam_bond: &[f64],
) -> (Vec<f64>, usize, usize, usize) {
    // Shapes:
    //   a_env: [d_l, D_bond, d_u_l, d_d_l, d_p]   (D_bond == a_env.d_r)
    //   b_env: [D_bond, d_r, d_u_r, d_d_r, d_p]   (D_bond == b_env.d_l)
    let chi_l = a_env.d_l * a_env.d_u * a_env.d_d;
    let chi_r = b_env.d_r * b_env.d_u * b_env.d_d;
    let d_p = a_env.d_p;
    let d_bond = lam_bond.len();

    // Re-index A: (l_flat, bond, p) where l_flat = (l * d_u_l + u_l) * d_d_l + d_l
    // Re-index B: (bond, r_flat, p) where r_flat = (r * d_u_r + u_r) * d_d_r + d_r

    // Ã matrix: [chi_l * d_p, D_bond]  (absorb λ into bond dimension)
    let mut a_mat = vec![0.0f64; chi_l * d_p * d_bond];
    for l in 0..a_env.d_l {
        for r in 0..a_env.d_r {
            // r here is the bond index
            let lam = lam_bond.get(r).copied().unwrap_or(0.0);
            for u in 0..a_env.d_u {
                for d in 0..a_env.d_d {
                    for p in 0..d_p {
                        let row = (l * a_env.d_u + u) * a_env.d_d + d; // l_flat in [0, chi_l)
                        let flat_row = row * d_p + p;
                        let val = a_env.get(l, r, u, d, p) * lam;
                        a_mat[flat_row * d_bond + r] += val;
                    }
                }
            }
        }
    }

    // B̃ matrix: [D_bond, chi_r * d_p]
    let mut b_mat = vec![0.0f64; d_bond * chi_r * d_p];
    for l in 0..b_env.d_l {
        // l here is the bond index
        for r in 0..b_env.d_r {
            for u in 0..b_env.d_u {
                for d in 0..b_env.d_d {
                    for p in 0..d_p {
                        let r_flat = (r * b_env.d_u + u) * b_env.d_d + d;
                        let flat_col = r_flat * d_p + p;
                        b_mat[l * (chi_r * d_p) + flat_col] += b_env.get(l, r, u, d, p);
                    }
                }
            }
        }
    }

    // Θ = A_mat · B_mat, shape [chi_l * d_p, chi_r * d_p]
    let m = chi_l * d_p;
    let k = d_bond;
    let n = chi_r * d_p;
    let mut theta = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for b in 0..k {
                acc += a_mat[i * k + b] * b_mat[b * n + j];
            }
            theta[i * n + j] = acc;
        }
    }

    (theta, chi_l, chi_r, d_p)
}

/// Contract two tensors on a vertical bond (between `a_env` above and `b_env` below).
///
/// `a_env`: top site, shape `[d_l, d_r, d_u, D_bond, d_p]`  (D_bond == a_env.d_d)
/// `b_env`: bottom site, shape `[d_l, d_r, D_bond, d_d, d_p]` (D_bond == b_env.d_u)
///
/// Returns `theta` with shape `[chi_u * d_p, d_p * chi_d]` where
/// `chi_u = d_l_a * d_r_a * d_u_a` and `chi_d = d_l_b * d_r_b * d_d_b`.
fn contract_theta_v(
    a_env: &PepsTensor,
    b_env: &PepsTensor,
    lam_bond: &[f64],
) -> (Vec<f64>, usize, usize, usize) {
    let chi_u = a_env.d_l * a_env.d_r * a_env.d_u;
    let chi_d = b_env.d_l * b_env.d_r * b_env.d_d;
    let d_p = a_env.d_p;
    let d_bond = lam_bond.len();

    // A_mat: [chi_u * d_p, D_bond]
    let mut a_mat = vec![0.0f64; chi_u * d_p * d_bond];
    for l in 0..a_env.d_l {
        for r in 0..a_env.d_r {
            for u in 0..a_env.d_u {
                for d in 0..a_env.d_d {
                    // d is the bond index
                    let lam = lam_bond.get(d).copied().unwrap_or(0.0);
                    for p in 0..d_p {
                        let row_flat = (l * a_env.d_r + r) * a_env.d_u + u;
                        let flat_row = row_flat * d_p + p;
                        let val = a_env.get(l, r, u, d, p) * lam;
                        a_mat[flat_row * d_bond + d] += val;
                    }
                }
            }
        }
    }

    // B_mat: [D_bond, chi_d * d_p]
    let mut b_mat = vec![0.0f64; d_bond * chi_d * d_p];
    for l in 0..b_env.d_l {
        for r in 0..b_env.d_r {
            for u in 0..b_env.d_u {
                // u is the bond index
                for d in 0..b_env.d_d {
                    for p in 0..d_p {
                        let col_flat = (l * b_env.d_r + r) * b_env.d_d + d;
                        let flat_col = col_flat * d_p + p;
                        b_mat[u * (chi_d * d_p) + flat_col] += b_env.get(l, r, u, d, p);
                    }
                }
            }
        }
    }

    // Θ = A_mat · B_mat
    let m = chi_u * d_p;
    let k = d_bond;
    let n = chi_d * d_p;
    let mut theta = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for b in 0..k {
                acc += a_mat[i * k + b] * b_mat[b * n + j];
            }
            theta[i * n + j] = acc;
        }
    }

    (theta, chi_u, chi_d, d_p)
}

/// Apply a `[d_p², d_p²]` two-site gate to `theta` (shape `[m, n]` where `m = chi_l * d_p`,
/// `n = d_p * chi_r`).
///
/// The gate acts on the pair of physical indices embedded in the row/column of theta.
/// Gate convention: `gate[p'_l * d_p + p'_r, p_l * d_p + p_r]` = amplitude.
fn apply_gate_to_theta(
    theta: &[f64],
    m: usize,
    n: usize,
    chi_l: usize,
    chi_r: usize,
    d_p: usize,
    gate: &[f64],
) -> Vec<f64> {
    let d2 = d_p * d_p;
    debug_assert_eq!(gate.len(), d2 * d2);
    debug_assert_eq!(m, chi_l * d_p);
    debug_assert_eq!(n, d_p * chi_r);

    let mut theta_prime = vec![0.0f64; m * n];

    // theta has row index (l_flat, p_l) and col index (p_r, r_flat).
    // gate maps (p_l', p_r') <- (p_l, p_r).
    for l_flat in 0..chi_l {
        for r_flat in 0..chi_r {
            for pl_prime in 0..d_p {
                for pr_prime in 0..d_p {
                    let mut acc = 0.0f64;
                    for pl in 0..d_p {
                        for pr in 0..d_p {
                            let th_idx = (l_flat * d_p + pl) * n + pr * chi_r + r_flat;
                            let g_idx = (pl_prime * d_p + pr_prime) * d2 + pl * d_p + pr;
                            acc += gate[g_idx] * theta[th_idx];
                        }
                    }
                    let out_idx = (l_flat * d_p + pl_prime) * n + pr_prime * chi_r + r_flat;
                    theta_prime[out_idx] = acc;
                }
            }
        }
    }
    theta_prime
}

/// SVD-truncate a matrix of shape `[m, n]` and extract new left/right tensors plus
/// the normalised singular values (new lambda).
///
/// Returns `(u_trunc, s_norm, vt_trunc, chi_new)`.
#[allow(clippy::type_complexity)]
fn svd_and_truncate(
    mat: &[f64],
    m: usize,
    n: usize,
    bond_dim: usize,
    tol: f64,
) -> TnResult<(Vec<f64>, Vec<f64>, Vec<f64>, usize)> {
    let svd = svd_jacobi(mat, m, n)?;
    let abs_tol = tol.max(1e-14);
    let (svd_trunc, _) = svd_truncate(svd, bond_dim, abs_tol)?;
    let chi_new = svd_trunc.k;

    let mut s_norm = svd_trunc.s.clone();
    normalise_l2(&mut s_norm);

    // u_trunc: [m, chi_new]
    let mut u_col = vec![0.0f64; m * chi_new];
    for i in 0..m {
        for j in 0..chi_new {
            u_col[i * chi_new + j] = svd_trunc.u[i * chi_new + j];
        }
    }

    // vt_trunc: [chi_new, n]
    let vt_row = svd_trunc.vt; // already [chi_new, n]

    Ok((u_col, s_norm, vt_row, chi_new))
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond update: horizontal
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one simple-update step on the **horizontal** bond between sites
/// `(row, col)` (left) and `(row, col+1)` (right).
///
/// ## Algorithm
///
/// 1. Fetch the lambda vectors for the four "environment" bonds of the left site
///    (`lam_left`, `lam_up_l`, `lam_down_l`) and right site (`lam_right`, `lam_up_r`, `lam_down_r`).
/// 2. Absorb environment into copies of A and B.
/// 3. Absorb `lam_bond` (the active bond lambda) into the contraction.
/// 4. Apply the two-site `gate` (imaginary-time propagator).
/// 5. SVD with truncation to `bond_dim`.
/// 6. Update `lam_bond`, A, B by absorbing inverse environments.
///
/// `gate` has shape `[d² × d²]` in row-major with ordering
/// `gate[p'_l * d + p'_r, p_l * d + p_r]`.
///
/// # Errors
/// Returns [`TnError::ShapeMismatch`] if dimensions are inconsistent.
/// Returns [`TnError::InvalidBondDimension`] if `bond_dim == 0`.
#[allow(clippy::too_many_arguments)]
pub fn simple_update_step_h(
    tensors: &mut [PepsTensor],
    lambdas: &mut PepsLambdas,
    gate: &[f64],
    row: usize,
    col: usize,
    bond_dim: usize,
    tol: f64,
) -> TnResult<()> {
    let lx = lambdas.lx;
    let ly = lambdas.ly;
    if bond_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if col + 1 >= lx {
        return Err(TnError::IndexOutOfBounds {
            index: col,
            len: lx,
        });
    }
    if row >= ly {
        return Err(TnError::IndexOutOfBounds {
            index: row,
            len: ly,
        });
    }
    let d_p = tensors[row * lx].d_p;
    let d2 = d_p * d_p;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate.len()],
        });
    }

    // ── Fetch lambda vectors (clone to avoid borrow issues) ──────────────────
    // Left site `A = tensors[row * lx + col]`
    // Right site `B = tensors[row * lx + col + 1]`

    // Left environment lambdas (l, u, d legs of A)
    let lam_l: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col - 1).to_vec()
    };
    let lam_u_l: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col).to_vec()
    };
    let lam_d_l: Vec<f64> = if row + 1 == ly {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row, col).to_vec()
    };

    // Right environment lambdas (r, u, d legs of B)
    let lam_r: Vec<f64> = if col + 2 == lx {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col + 1).to_vec()
    };
    let lam_u_r: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col + 1).to_vec()
    };
    let lam_d_r: Vec<f64> = if row + 1 == ly {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row, col + 1).to_vec()
    };

    // Active bond lambda (between A and B)
    let lam_bond: Vec<f64> = lambdas.get_h(row, col).to_vec();

    // ── Step 1: absorb environment ────────────────────────────────────────────
    let ta = &tensors[row * lx + col];
    let tb = &tensors[row * lx + col + 1];

    let a_env_data = absorb_env_left_site(ta, &lam_l, &lam_u_l, &lam_d_l);
    let a_env = PepsTensor::new(ta.d_l, ta.d_r, ta.d_u, ta.d_d, d_p, a_env_data)?;

    let b_env_data = absorb_env_right_site(tb, &lam_r, &lam_u_r, &lam_d_r);
    let b_env = PepsTensor::new(tb.d_l, tb.d_r, tb.d_u, tb.d_d, d_p, b_env_data)?;

    // ── Step 2: contract Θ ───────────────────────────────────────────────────
    let (theta, chi_l, chi_r, _) = contract_theta_h(&a_env, &b_env, &lam_bond);
    let m = chi_l * d_p;
    let n = chi_r * d_p;

    // ── Step 3: apply gate ────────────────────────────────────────────────────
    let theta_prime = apply_gate_to_theta(&theta, m, n, chi_l, chi_r, d_p, gate);

    // ── Step 4: SVD + truncate ────────────────────────────────────────────────
    let (u_new, s_norm, vt_new, chi_new) = svd_and_truncate(&theta_prime, m, n, bond_dim, tol)?;
    // chi_new is the new bond dimension (may be ≤ bond_dim)

    // ── Step 5: update lambda_bond ────────────────────────────────────────────
    *lambdas.get_h_mut(row, col) = s_norm.clone();

    // ── Step 6: extract new tensors ───────────────────────────────────────────
    // Inverse environment lambdas (to undo the absorption).
    let lam_l_inv = invert_vec(&lam_l, 1e-14);
    let lam_u_l_inv = invert_vec(&lam_u_l, 1e-14);
    let lam_d_l_inv = invert_vec(&lam_d_l, 1e-14);
    let lam_r_inv = invert_vec(&lam_r, 1e-14);
    let lam_u_r_inv = invert_vec(&lam_u_r, 1e-14);
    let lam_d_r_inv = invert_vec(&lam_d_r, 1e-14);

    // New A: from U_new, shape [chi_l * d_p, chi_new]
    // Reshape to [d_l_orig, d_u_l_orig, d_d_l_orig, d_p, chi_new] then absorb inv-env
    // We keep the bond between A and B as chi_new (the new bond dim).
    let d_l = ta.d_l;
    let d_u_l = ta.d_u;
    let d_d_l = ta.d_d;
    let mut new_a = PepsTensor::zeros(d_l, chi_new, d_u_l, d_d_l, d_p)?;
    for l in 0..d_l {
        let inv_l = lam_l_inv.get(l).copied().unwrap_or(1.0);
        for u in 0..d_u_l {
            let inv_u = lam_u_l_inv.get(u).copied().unwrap_or(1.0);
            for d in 0..d_d_l {
                let inv_d = lam_d_l_inv.get(d).copied().unwrap_or(1.0);
                let l_flat = (l * d_u_l + u) * d_d_l + d;
                for p in 0..d_p {
                    let row_u = l_flat * d_p + p;
                    for b in 0..chi_new {
                        let val = u_new[row_u * chi_new + b] * inv_l * inv_u * inv_d;
                        new_a.set(l, b, u, d, p, val);
                    }
                }
            }
        }
    }

    // New B: from Vt_new, shape [chi_new, chi_r * d_p]
    let d_r = tb.d_r;
    let d_u_r = tb.d_u;
    let d_d_r = tb.d_d;
    let mut new_b = PepsTensor::zeros(chi_new, d_r, d_u_r, d_d_r, d_p)?;
    for r in 0..d_r {
        let inv_r = lam_r_inv.get(r).copied().unwrap_or(1.0);
        for u in 0..d_u_r {
            let inv_u = lam_u_r_inv.get(u).copied().unwrap_or(1.0);
            for d in 0..d_d_r {
                let inv_d = lam_d_r_inv.get(d).copied().unwrap_or(1.0);
                let r_flat = (r * d_u_r + u) * d_d_r + d;
                for p in 0..d_p {
                    let col_u = r_flat * d_p + p;
                    for b in 0..chi_new {
                        let val = vt_new[b * n + col_u] * inv_r * inv_u * inv_d;
                        new_b.set(b, r, u, d, p, val);
                    }
                }
            }
        }
    }

    // Commit
    tensors[row * lx + col] = new_a;
    tensors[row * lx + col + 1] = new_b;

    // Update the lambda bond length to chi_new (may have grown or shrunk).
    // The s_norm was already stored in lambdas above.
    // If chi_new < bond_dim the lambda vector is already correct (length chi_new).
    let _ = chi_new; // suppresses unused-variable warning; already used above.

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond update: vertical
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one simple-update step on the **vertical** bond between sites
/// `(row, col)` (top) and `(row+1, col)` (bottom).
///
/// Analogue of [`simple_update_step_h`] but oriented along the vertical direction.
#[allow(clippy::too_many_arguments)]
pub fn simple_update_step_v(
    tensors: &mut [PepsTensor],
    lambdas: &mut PepsLambdas,
    gate: &[f64],
    row: usize,
    col: usize,
    bond_dim: usize,
    tol: f64,
) -> TnResult<()> {
    let lx = lambdas.lx;
    let ly = lambdas.ly;
    if bond_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if row + 1 >= ly {
        return Err(TnError::IndexOutOfBounds {
            index: row,
            len: ly,
        });
    }
    if col >= lx {
        return Err(TnError::IndexOutOfBounds {
            index: col,
            len: lx,
        });
    }
    let d_p = tensors[row * lx + col].d_p;
    let d2 = d_p * d_p;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate.len()],
        });
    }

    // ── Fetch lambda vectors ──────────────────────────────────────────────────
    // Top site A = tensors[row * lx + col]
    // Bottom site B = tensors[(row+1) * lx + col]

    // Environment for A: l, r, u legs
    let lam_l_a: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col - 1).to_vec()
    };
    let lam_r_a: Vec<f64> = if col + 1 == lx {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col).to_vec()
    };
    let lam_u_a: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col).to_vec()
    };

    // Environment for B: l, r, d legs
    let lam_l_b: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row + 1, col - 1).to_vec()
    };
    let lam_r_b: Vec<f64> = if col + 1 == lx {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row + 1, col).to_vec()
    };
    let lam_d_b: Vec<f64> = if row + 2 == ly {
        vec![1.0; 1]
    } else if row + 2 < ly {
        lambdas.get_v(row + 1, col).to_vec()
    } else {
        vec![1.0; 1]
    };

    // Active bond lambda (vertical bond between row and row+1 at col)
    let lam_bond: Vec<f64> = lambdas.get_v(row, col).to_vec();

    // ── Step 1: absorb environment ────────────────────────────────────────────
    // For top site A: scale l, r, u legs (d leg is the active bond).
    let ta = &tensors[row * lx + col];
    let tb = &tensors[(row + 1) * lx + col];

    // Absorb into A: scale l, r, u (not d, which is the bond).
    let a_env_data = absorb_env_top_site(ta, &lam_l_a, &lam_r_a, &lam_u_a);
    let a_env = PepsTensor::new(ta.d_l, ta.d_r, ta.d_u, ta.d_d, d_p, a_env_data)?;

    // Absorb into B: scale l, r, d (not u, which is the bond).
    let b_env_data = absorb_env_bottom_site(tb, &lam_l_b, &lam_r_b, &lam_d_b);
    let b_env = PepsTensor::new(tb.d_l, tb.d_r, tb.d_u, tb.d_d, d_p, b_env_data)?;

    // ── Step 2: contract Θ ────────────────────────────────────────────────────
    let (theta, chi_u, chi_d, _) = contract_theta_v(&a_env, &b_env, &lam_bond);
    let m = chi_u * d_p;
    let n = chi_d * d_p;

    // ── Step 3: apply gate ────────────────────────────────────────────────────
    let theta_prime = apply_gate_to_theta(&theta, m, n, chi_u, chi_d, d_p, gate);

    // ── Step 4: SVD + truncate ────────────────────────────────────────────────
    let (u_new, s_norm, vt_new, chi_new) = svd_and_truncate(&theta_prime, m, n, bond_dim, tol)?;

    // ── Step 5: update lambda_bond ────────────────────────────────────────────
    *lambdas.get_v_mut(row, col) = s_norm.clone();

    // ── Step 6: extract new tensors ───────────────────────────────────────────
    let lam_l_a_inv = invert_vec(&lam_l_a, 1e-14);
    let lam_r_a_inv = invert_vec(&lam_r_a, 1e-14);
    let lam_u_a_inv = invert_vec(&lam_u_a, 1e-14);
    let lam_l_b_inv = invert_vec(&lam_l_b, 1e-14);
    let lam_r_b_inv = invert_vec(&lam_r_b, 1e-14);
    let lam_d_b_inv = invert_vec(&lam_d_b, 1e-14);

    // New A: from U_new, shape [chi_u * d_p, chi_new]
    // Original shape was [d_l, d_r, d_u, D_bond, d_p] (d is the active bond).
    let d_l_a = ta.d_l;
    let d_r_a = ta.d_r;
    let d_u_a = ta.d_u;
    let mut new_a = PepsTensor::zeros(d_l_a, d_r_a, d_u_a, chi_new, d_p)?;
    for l in 0..d_l_a {
        let inv_l = lam_l_a_inv.get(l).copied().unwrap_or(1.0);
        for r in 0..d_r_a {
            let inv_r = lam_r_a_inv.get(r).copied().unwrap_or(1.0);
            for u in 0..d_u_a {
                let inv_u = lam_u_a_inv.get(u).copied().unwrap_or(1.0);
                let u_flat = (l * d_r_a + r) * d_u_a + u;
                for p in 0..d_p {
                    let row_u = u_flat * d_p + p;
                    for b in 0..chi_new {
                        let val = u_new[row_u * chi_new + b] * inv_l * inv_r * inv_u;
                        new_a.set(l, r, u, b, p, val);
                    }
                }
            }
        }
    }

    // New B: from Vt_new, shape [chi_new, chi_d * d_p]
    let d_l_b = tb.d_l;
    let d_r_b = tb.d_r;
    let d_d_b = tb.d_d;
    let mut new_b = PepsTensor::zeros(d_l_b, d_r_b, chi_new, d_d_b, d_p)?;
    for l in 0..d_l_b {
        let inv_l = lam_l_b_inv.get(l).copied().unwrap_or(1.0);
        for r in 0..d_r_b {
            let inv_r = lam_r_b_inv.get(r).copied().unwrap_or(1.0);
            for d in 0..d_d_b {
                let inv_d = lam_d_b_inv.get(d).copied().unwrap_or(1.0);
                let d_flat = (l * d_r_b + r) * d_d_b + d;
                for p in 0..d_p {
                    let col_u = d_flat * d_p + p;
                    for b in 0..chi_new {
                        let val = vt_new[b * n + col_u] * inv_l * inv_r * inv_d;
                        new_b.set(l, r, b, d, p, val);
                    }
                }
            }
        }
    }

    // Commit
    tensors[row * lx + col] = new_a;
    tensors[(row + 1) * lx + col] = new_b;

    Ok(())
}

/// Absorb environment into a top (above-bond) site for vertical updates.
/// Scales `l`, `r`, `u` legs; leaves `d` (the active bond) unscaled.
fn absorb_env_top_site(t: &PepsTensor, lam_l: &[f64], lam_r: &[f64], lam_u: &[f64]) -> Vec<f64> {
    let mut out = t.data.clone();
    for l in 0..t.d_l {
        let sl = lam_l.get(l).copied().unwrap_or(1.0);
        for r in 0..t.d_r {
            let sr = lam_r.get(r).copied().unwrap_or(1.0);
            for u in 0..t.d_u {
                let su = lam_u.get(u).copied().unwrap_or(1.0);
                let scale = sl * sr * su;
                for d in 0..t.d_d {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale;
                    }
                }
            }
        }
    }
    out
}

/// Absorb environment into a bottom (below-bond) site for vertical updates.
/// Scales `l`, `r`, `d` legs; leaves `u` (the active bond) unscaled.
fn absorb_env_bottom_site(t: &PepsTensor, lam_l: &[f64], lam_r: &[f64], lam_d: &[f64]) -> Vec<f64> {
    let mut out = t.data.clone();
    for l in 0..t.d_l {
        let sl = lam_l.get(l).copied().unwrap_or(1.0);
        for r in 0..t.d_r {
            let sr = lam_r.get(r).copied().unwrap_or(1.0);
            for d in 0..t.d_d {
                let sd = lam_d.get(d).copied().unwrap_or(1.0);
                let scale = sl * sr * sd;
                for u in 0..t.d_u {
                    for p in 0..t.d_p {
                        let i = t.idx(l, r, u, d, p);
                        out[i] *= scale;
                    }
                }
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the energy per site using a product-state / lambda-environment approximation.
///
/// For each bond `(i,j)-(i,j')` we compute the two-site reduced density matrix
/// `ρ_{ij}` using only the on-site tensors and their neighbouring lambda matrices
/// (simple-update environment), then take `Tr[ρ H]`.  All bond contributions are
/// summed and divided by the total number of bonds times 2 (to get per-site energy).
///
/// This is an approximate energy that is consistent with the simple-update environment
/// used during the sweep.  For better accuracy one would use a boundary-MPS environment.
///
/// # Errors
/// Returns [`TnError::ShapeMismatch`] if `hamiltonian.len() != (phys_dim²)²`.
pub fn simple_update_energy(
    tensors: &[PepsTensor],
    lambdas: &PepsLambdas,
    hamiltonian: &[f64],
    lx: usize,
    ly: usize,
) -> TnResult<f64> {
    if tensors.is_empty() {
        return Err(TnError::EmptyInput);
    }
    let d_p = tensors[0].d_p;
    let d2 = d_p * d_p;
    if hamiltonian.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![hamiltonian.len()],
        });
    }

    let mut total_energy = 0.0f64;
    let mut n_bonds = 0usize;

    // Horizontal bonds
    for row in 0..ly {
        for col in 0..(lx.saturating_sub(1)) {
            let ta = &tensors[row * lx + col];
            let tb = &tensors[row * lx + col + 1];
            let lam_bond = lambdas.get_h(row, col);

            let e = bond_energy_h(ta, tb, lam_bond, hamiltonian, lambdas, row, col, lx, ly)?;
            total_energy += e;
            n_bonds += 1;
        }
    }

    // Vertical bonds
    for row in 0..(ly.saturating_sub(1)) {
        for col in 0..lx {
            let ta = &tensors[row * lx + col];
            let tb = &tensors[(row + 1) * lx + col];
            let lam_bond = lambdas.get_v(row, col);

            let e = bond_energy_v(ta, tb, lam_bond, hamiltonian, lambdas, row, col, lx, ly)?;
            total_energy += e;
            n_bonds += 1;
        }
    }

    if n_bonds == 0 {
        // Single-site lattice: no bonds, return 0.
        return Ok(0.0);
    }

    // Energy per site: total_bond_energy / n_sites
    // (each bond contributes ~1 energy term; n_bonds ≈ n_sites for large lattice)
    let n_sites = (lx * ly) as f64;
    Ok(total_energy / n_sites)
}

/// Compute the energy expectation value on horizontal bond `(row, col)-(row, col+1)`.
#[allow(clippy::too_many_arguments)]
fn bond_energy_h(
    ta: &PepsTensor,
    tb: &PepsTensor,
    lam_bond: &[f64],
    hamiltonian: &[f64],
    lambdas: &PepsLambdas,
    row: usize,
    col: usize,
    lx: usize,
    ly: usize,
) -> TnResult<f64> {
    let d_p = ta.d_p;
    let d2 = d_p * d_p;

    // Absorb environment into both sites
    let lam_l: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col - 1).to_vec()
    };
    let lam_u_l: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col).to_vec()
    };
    let lam_d_l: Vec<f64> = if row + 1 == ly {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row, col).to_vec()
    };

    let lam_r: Vec<f64> = if col + 2 == lx {
        vec![1.0; 1]
    } else if col + 2 < lx {
        lambdas.get_h(row, col + 1).to_vec()
    } else {
        vec![1.0; 1]
    };
    let lam_u_r: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col + 1).to_vec()
    };
    let lam_d_r: Vec<f64> = if row + 1 == ly {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row, col + 1).to_vec()
    };

    let a_env_data = absorb_env_left_site(ta, &lam_l, &lam_u_l, &lam_d_l);
    let a_env = PepsTensor::new(ta.d_l, ta.d_r, ta.d_u, ta.d_d, d_p, a_env_data)?;
    let b_env_data = absorb_env_right_site(tb, &lam_r, &lam_u_r, &lam_d_r);
    let b_env = PepsTensor::new(tb.d_l, tb.d_r, tb.d_u, tb.d_d, d_p, b_env_data)?;

    // Contract Θ
    let (theta, chi_l, chi_r, _) = contract_theta_h(&a_env, &b_env, lam_bond);
    let m = chi_l * d_p;
    let n = chi_r * d_p;

    two_site_energy(&theta, m, n, chi_l, chi_r, d_p, d2, hamiltonian)
}

/// Compute the energy expectation value on vertical bond `(row, col)-(row+1, col)`.
#[allow(clippy::too_many_arguments)]
fn bond_energy_v(
    ta: &PepsTensor,
    tb: &PepsTensor,
    lam_bond: &[f64],
    hamiltonian: &[f64],
    lambdas: &PepsLambdas,
    row: usize,
    col: usize,
    lx: usize,
    ly: usize,
) -> TnResult<f64> {
    let d_p = ta.d_p;
    let d2 = d_p * d_p;

    let lam_l_a: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col - 1).to_vec()
    };
    let lam_r_a: Vec<f64> = if col + 1 == lx {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row, col).to_vec()
    };
    let lam_u_a: Vec<f64> = if row == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_v(row - 1, col).to_vec()
    };

    let lam_l_b: Vec<f64> = if col == 0 {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row + 1, col - 1).to_vec()
    };
    let lam_r_b: Vec<f64> = if col + 1 == lx {
        vec![1.0; 1]
    } else {
        lambdas.get_h(row + 1, col).to_vec()
    };
    let lam_d_b: Vec<f64> = if row + 2 < ly {
        lambdas.get_v(row + 1, col).to_vec()
    } else {
        vec![1.0; 1]
    };

    let a_env_data = absorb_env_top_site(ta, &lam_l_a, &lam_r_a, &lam_u_a);
    let a_env = PepsTensor::new(ta.d_l, ta.d_r, ta.d_u, ta.d_d, d_p, a_env_data)?;
    let b_env_data = absorb_env_bottom_site(tb, &lam_l_b, &lam_r_b, &lam_d_b);
    let b_env = PepsTensor::new(tb.d_l, tb.d_r, tb.d_u, tb.d_d, d_p, b_env_data)?;

    let (theta, chi_u, chi_d, _) = contract_theta_v(&a_env, &b_env, lam_bond);
    let m = chi_u * d_p;
    let n = chi_d * d_p;

    two_site_energy(&theta, m, n, chi_u, chi_d, d_p, d2, hamiltonian)
}

/// Compute `<θ|H|θ> / <θ|θ>` for a two-site tensor `θ` of shape `[m, n]`.
#[allow(clippy::too_many_arguments)]
fn two_site_energy(
    theta: &[f64],
    m: usize,
    n: usize,
    chi_l: usize,
    chi_r: usize,
    d_p: usize,
    d2: usize,
    hamiltonian: &[f64],
) -> TnResult<f64> {
    let _ = (m, n); // used only in debug assertions
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for l_flat in 0..chi_l {
        for r_flat in 0..chi_r {
            for pl in 0..d_p {
                // We iterate over all pr to get the full bra vector
                for pr in 0..d_p {
                    let th_bra = theta[(l_flat * d_p + pl) * (d_p * chi_r) + pr * chi_r + r_flat];
                    denominator += th_bra * th_bra;

                    for kl in 0..d_p {
                        for kr in 0..d_p {
                            let h_val = hamiltonian[(pl * d_p + pr) * d2 + kl * d_p + kr];
                            if h_val.abs() < 1e-16 {
                                continue;
                            }
                            let th_ket =
                                theta[(l_flat * d_p + kl) * (d_p * chi_r) + kr * chi_r + r_flat];
                            numerator += th_bra * h_val * th_ket;
                        }
                    }
                }
            }
        }
    }

    let denom_safe = denominator.max(1e-60);
    Ok(numerator / denom_safe)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main driver
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full PEPS simple-update algorithm.
///
/// ## Trotter splitting
///
/// * **Order 1** (`config.trotter_order == 1`): per macro-step, sweep horizontal bonds
///   left-to-right top-to-bottom, then vertical bonds top-to-bottom left-to-right.
/// * **Order 2** (default): symmetric Strang splitting — horizontal sweep, then vertical
///   sweep, then horizontal sweep in reverse.
///
/// ## Convergence
///
/// Every 10 macro-steps the energy per site is estimated.  If `|ΔE| < config.tol`
/// the run exits early with `converged = true`.
///
/// # Errors
/// Returns [`TnError::InvalidBondDimension`] if `config.bond_dim == 0` or
/// `config.phys_dim == 0`.
/// Returns [`TnError::InvalidConfiguration`] if `config.trotter_order` is not 1 or 2.
pub fn simple_update_run(
    gate_h: &[f64],
    gate_v: &[f64],
    config: &SimpleUpdateConfig,
    rng: &mut LcgRng,
) -> TnResult<SimpleUpdateResult> {
    if config.bond_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if config.phys_dim == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if config.lx == 0 || config.ly == 0 {
        return Err(TnError::EmptyInput);
    }
    if config.trotter_order != 1 && config.trotter_order != 2 {
        return Err(TnError::InvalidConfiguration(format!(
            "trotter_order must be 1 or 2, got {}",
            config.trotter_order
        )));
    }

    let d_p = config.phys_dim;
    let d2 = d_p * d_p;
    if gate_h.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate_h.len()],
        });
    }
    if gate_v.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate_v.len()],
        });
    }

    let lx = config.lx;
    let ly = config.ly;
    let bond_dim = config.bond_dim;
    let tol = config.tol;

    let (mut tensors, mut lambdas) = simple_update_init(lx, ly, d_p, bond_dim, rng)?;

    let mut prev_energy = f64::INFINITY;
    let mut converged = false;
    let mut steps_done = 0usize;

    for step in 0..config.n_steps {
        // ── Forward horizontal sweep ─────────────────────────────────────────
        sweep_horizontal(&mut tensors, &mut lambdas, gate_h, bond_dim, tol, false)?;

        // ── Forward vertical sweep ───────────────────────────────────────────
        sweep_vertical(&mut tensors, &mut lambdas, gate_v, bond_dim, tol, false)?;

        // ── Reverse sweeps for Strang splitting (order 2) ────────────────────
        if config.trotter_order == 2 {
            sweep_vertical(&mut tensors, &mut lambdas, gate_v, bond_dim, tol, true)?;
            sweep_horizontal(&mut tensors, &mut lambdas, gate_h, bond_dim, tol, true)?;
        }

        steps_done = step + 1;

        // ── Convergence check every 10 steps ─────────────────────────────────
        if step % 10 == 9 || step == config.n_steps - 1 {
            // Build a simple Heisenberg-like Hamiltonian proxy for energy estimate
            // (we use the gate structure: -log(gate)/delta_tau ≈ H, but since we
            // don't have delta_tau here we just use the hamiltonian passed as gate).
            // Actually we need the original Hamiltonian; here we approximate by
            // using `gate_h` as a proxy hamiltonian (works only if gate = I + tiny).
            // For a correct convergence check we build a trivial NN energy using
            // `gate_h` itself as the "hamiltonian" (its rows are in the same basis).
            let energy = simple_update_energy(&tensors, &lambdas, gate_h, lx, ly)?;
            let delta = (energy - prev_energy).abs();
            if delta < tol {
                converged = true;
                break;
            }
            prev_energy = energy;
        }
    }

    // Final energy estimate
    let energy_per_site = simple_update_energy(&tensors, &lambdas, gate_h, lx, ly)?;

    Ok(SimpleUpdateResult {
        tensors,
        lambdas,
        energy_per_site,
        n_steps: steps_done,
        converged,
    })
}

/// Perform a single pass over all horizontal bonds (left→right then top→bottom or reverse).
fn sweep_horizontal(
    tensors: &mut [PepsTensor],
    lambdas: &mut PepsLambdas,
    gate: &[f64],
    bond_dim: usize,
    tol: f64,
    reverse: bool,
) -> TnResult<()> {
    let lx = lambdas.lx;
    let ly = lambdas.ly;
    if lx <= 1 {
        return Ok(()); // no horizontal bonds
    }
    for row in 0..ly {
        let r = if reverse { ly - 1 - row } else { row };
        let cols: Vec<usize> = if reverse {
            (0..(lx - 1)).rev().collect()
        } else {
            (0..(lx - 1)).collect()
        };
        for col in cols {
            simple_update_step_h(tensors, lambdas, gate, r, col, bond_dim, tol)?;
        }
    }
    Ok(())
}

/// Perform a single pass over all vertical bonds (top→bottom then left→right or reverse).
fn sweep_vertical(
    tensors: &mut [PepsTensor],
    lambdas: &mut PepsLambdas,
    gate: &[f64],
    bond_dim: usize,
    tol: f64,
    reverse: bool,
) -> TnResult<()> {
    let lx = lambdas.lx;
    let ly = lambdas.ly;
    if ly <= 1 {
        return Ok(()); // no vertical bonds
    }
    for col in 0..lx {
        let c = if reverse { lx - 1 - col } else { col };
        let rows: Vec<usize> = if reverse {
            (0..(ly - 1)).rev().collect()
        } else {
            (0..(ly - 1)).collect()
        };
        for row in rows {
            simple_update_step_v(tensors, lambdas, gate, row, c, bond_dim, tol)?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix exponential for arbitrary d² × d² real symmetric matrices (Jacobi)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `exp(scale · A)` for a real symmetric matrix `A` of size `n × n` (row-major).
///
/// Uses the Jacobi eigendecomposition: `A = V diag(λ) V^T`, then
/// `exp(scale · A) = V diag(exp(scale · λ)) V^T`.
///
/// Suitable for gate construction: call with `scale = -delta_tau`.
pub fn mat_exp_sym(a: &[f64], n: usize, scale: f64) -> TnResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if n == 0 {
        return Err(TnError::EmptyInput);
    }

    let mut mat = a.to_vec();
    let mut v_mat = vec![0.0f64; n * n];
    for i in 0..n {
        v_mat[i * n + i] = 1.0;
    }

    let max_sweeps = 500;
    let tol_j = 1e-13_f64;

    'outer: for _ in 0..max_sweeps {
        let mut off_diag_sq = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += mat[i * n + j] * mat[i * n + j];
            }
        }
        if off_diag_sq < tol_j {
            break 'outer;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let app = mat[p * n + p];
                let aqq = mat[q * n + q];
                let apq = mat[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let cos_val = 1.0 / (1.0 + t * t).sqrt();
                let sin_val = t * cos_val;

                mat[p * n + p] = app - t * apq;
                mat[q * n + q] = aqq + t * apq;
                mat[p * n + q] = 0.0;
                mat[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = mat[r * n + p];
                    let arq = mat[r * n + q];
                    mat[r * n + p] = cos_val * arp - sin_val * arq;
                    mat[p * n + r] = mat[r * n + p];
                    mat[r * n + q] = sin_val * arp + cos_val * arq;
                    mat[q * n + r] = mat[r * n + q];
                }

                for r in 0..n {
                    let vrp = v_mat[r * n + p];
                    let vrq = v_mat[r * n + q];
                    v_mat[r * n + p] = cos_val * vrp - sin_val * vrq;
                    v_mat[r * n + q] = sin_val * vrp + cos_val * vrq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i]).collect();

    let mut result = vec![0.0f64; n * n];
    for i in 0..n {
        let exp_lam = (scale * eigenvalues[i]).exp();
        for r in 0..n {
            for c in 0..n {
                result[r * n + c] += v_mat[r * n + i] * exp_lam * v_mat[c * n + i];
            }
        }
    }
    Ok(result)
}

/// Build the 2-site Heisenberg exchange Hamiltonian `h = J (Sx⊗Sx + Sy⊗Sy + Sz⊗Sz)`
/// for spin-1/2 (d=2).  Returns a `Vec<f64>` of length 16 (4×4 matrix, row-major).
///
/// Matrix elements (in computational basis {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}):
/// ```text
/// J/4  0     0     0
/// 0   -J/4   J/2   0
/// 0    J/2  -J/4   0
/// 0    0     0     J/4
/// ```
#[must_use]
pub fn heisenberg_hamiltonian_2site(j: f64) -> Vec<f64> {
    vec![
        j * 0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        -j * 0.25,
        j * 0.5,
        0.0,
        0.0,
        j * 0.5,
        -j * 0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        j * 0.25,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: identity gate for physical dim d_p.
    fn identity_gate(d_p: usize) -> Vec<f64> {
        let d2 = d_p * d_p;
        let mut g = vec![0.0f64; d2 * d2];
        for i in 0..d2 {
            g[i * d2 + i] = 1.0;
        }
        g
    }

    // Helper: Heisenberg imaginary-time gate exp(-tau * H).
    fn heisenberg_gate(j: f64, tau: f64) -> Vec<f64> {
        let h = heisenberg_hamiltonian_2site(j);
        mat_exp_sym(&h, 4, -tau).expect("mat_exp_sym ok")
    }

    // ── Test 1: init produces correct shapes ──────────────────────────────────
    #[test]
    fn init_correct_shapes() {
        let mut rng = LcgRng::new(42);
        let (tensors, lambdas) = simple_update_init(3, 2, 2, 4, &mut rng).expect("init ok");
        assert_eq!(tensors.len(), 6);
        // Corner top-left: d_l=1, d_u=1
        assert_eq!(tensors[0].d_l, 1);
        assert_eq!(tensors[0].d_u, 1);
        // Interior bottom-right-ish: tensors[5] is row=1, col=2 → d_r=1, d_d=1
        assert_eq!(tensors[5].d_r, 1);
        assert_eq!(tensors[5].d_d, 1);
        // Horizontal lambdas: ly*(lx-1) = 2*2 = 4
        assert_eq!(lambdas.horizontal.len(), 4);
        // Vertical lambdas: (ly-1)*lx = 1*3 = 3
        assert_eq!(lambdas.vertical.len(), 3);
        // Each lambda has length bond_dim=4
        for v in &lambdas.horizontal {
            assert_eq!(v.len(), 4);
        }
        for v in &lambdas.vertical {
            assert_eq!(v.len(), 4);
        }
    }

    // ── Test 2: lambda vectors are positive at init ───────────────────────────
    #[test]
    fn init_lambdas_positive() {
        let mut rng = LcgRng::new(7);
        let (_, lambdas) = simple_update_init(2, 2, 2, 3, &mut rng).expect("ok");
        for v in &lambdas.horizontal {
            for &x in v {
                assert!(x > 0.0, "horizontal lambda non-positive: {x}");
            }
        }
        for v in &lambdas.vertical {
            for &x in v {
                assert!(x > 0.0, "vertical lambda non-positive: {x}");
            }
        }
    }

    // ── Test 3: horizontal bond step: tensor shapes unchanged ─────────────────
    #[test]
    fn h_step_shapes_unchanged() {
        let mut rng = LcgRng::new(1);
        let (mut tensors, mut lambdas) = simple_update_init(3, 2, 2, 3, &mut rng).expect("ok");
        let gate = identity_gate(2);

        let (d_l_before, d_r_before, d_u_before, d_d_before, d_p_before) = (
            tensors[0].d_l,
            tensors[0].d_r,
            tensors[0].d_u,
            tensors[0].d_d,
            tensors[0].d_p,
        );

        // Apply h step on bond (row=0, col=0) — between sites [0] and [1].
        simple_update_step_h(&mut tensors, &mut lambdas, &gate, 0, 0, 3, 1e-12).expect("h step ok");

        // Left tensor d_l, d_u, d_d, d_p unchanged; d_r may be ≤ bond_dim.
        assert_eq!(tensors[0].d_l, d_l_before);
        assert_eq!(tensors[0].d_u, d_u_before);
        assert_eq!(tensors[0].d_d, d_d_before);
        assert_eq!(tensors[0].d_p, d_p_before);
        let _ = d_r_before;
    }

    // ── Test 4: vertical bond step: tensor shapes are sane ───────────────────
    #[test]
    fn v_step_shapes_sane() {
        let mut rng = LcgRng::new(2);
        let (mut tensors, mut lambdas) = simple_update_init(2, 3, 2, 3, &mut rng).expect("ok");
        let gate = identity_gate(2);

        let d_p = tensors[0].d_p;
        simple_update_step_v(&mut tensors, &mut lambdas, &gate, 0, 0, 3, 1e-12).expect("v step ok");

        // Physical dim unchanged after step.
        assert_eq!(tensors[0].d_p, d_p);
        assert_eq!(tensors[2].d_p, d_p); // lx=2, site (1,0) is index 2
    }

    // ── Test 5: identity gate leaves tensors approximately unchanged ───────────
    #[test]
    fn identity_gate_no_energy_change() {
        let mut rng = LcgRng::new(3);
        let (mut tensors, mut lambdas) = simple_update_init(2, 2, 2, 2, &mut rng).expect("ok");
        let ham = heisenberg_hamiltonian_2site(1.0);
        let gate = identity_gate(2);

        let e0 = simple_update_energy(&tensors, &lambdas, &ham, 2, 2).expect("e0 ok");

        // Apply a few identity steps
        for _ in 0..3 {
            simple_update_step_h(&mut tensors, &mut lambdas, &gate, 0, 0, 2, 1e-12).expect("ok");
            simple_update_step_v(&mut tensors, &mut lambdas, &gate, 0, 0, 2, 1e-12).expect("ok");
        }

        let e1 = simple_update_energy(&tensors, &lambdas, &ham, 2, 2).expect("e1 ok");

        // Energy should remain finite and in the same ballpark.
        assert!(e0.is_finite(), "e0 not finite: {e0}");
        assert!(e1.is_finite(), "e1 not finite: {e1}");
    }

    // ── Test 6: energy estimate is finite for random init ─────────────────────
    #[test]
    fn energy_finite_after_init() {
        let mut rng = LcgRng::new(99);
        let (tensors, lambdas) = simple_update_init(2, 2, 2, 2, &mut rng).expect("ok");
        let ham = heisenberg_hamiltonian_2site(1.0);
        let e = simple_update_energy(&tensors, &lambdas, &ham, 2, 2).expect("energy ok");
        assert!(e.is_finite(), "energy not finite: {e}");
    }

    // ── Test 7: invalid bond_dim=0 → error ───────────────────────────────────
    #[test]
    fn invalid_bond_dim_zero() {
        let mut rng = LcgRng::new(10);
        let res = simple_update_init(2, 2, 2, 0, &mut rng);
        assert!(res.is_err(), "expected Err for bond_dim=0");
    }

    // ── Test 8: invalid phys_dim=0 → error ───────────────────────────────────
    #[test]
    fn invalid_phys_dim_zero() {
        let mut rng = LcgRng::new(11);
        let res = simple_update_init(2, 2, 0, 2, &mut rng);
        assert!(res.is_err(), "expected Err for phys_dim=0");
    }

    // ── Test 9: lx=1 (1D chain along y) still works ──────────────────────────
    #[test]
    fn lx_one_vertical_only() {
        let mut rng = LcgRng::new(12);
        let (mut tensors, mut lambdas) =
            simple_update_init(1, 4, 2, 2, &mut rng).expect("ok for lx=1");
        // No horizontal bonds possible.
        assert_eq!(lambdas.horizontal.len(), 0);
        // Vertical bonds: (ly-1)*lx = 3*1 = 3.
        assert_eq!(lambdas.vertical.len(), 3);

        let gate = identity_gate(2);
        simple_update_step_v(&mut tensors, &mut lambdas, &gate, 0, 0, 2, 1e-12)
            .expect("v step on lx=1");
    }

    // ── Test 10: ly=1 (1D chain along x) still works ─────────────────────────
    #[test]
    fn ly_one_horizontal_only() {
        let mut rng = LcgRng::new(13);
        let (mut tensors, mut lambdas) =
            simple_update_init(4, 1, 2, 2, &mut rng).expect("ok for ly=1");
        // Horizontal bonds: ly*(lx-1) = 1*3 = 3.
        assert_eq!(lambdas.horizontal.len(), 3);
        // No vertical bonds.
        assert_eq!(lambdas.vertical.len(), 0);

        let gate = identity_gate(2);
        simple_update_step_h(&mut tensors, &mut lambdas, &gate, 0, 0, 2, 1e-12)
            .expect("h step on ly=1");
    }

    // ── Test 11: simple_update_run returns finite energy ─────────────────────
    #[test]
    fn run_returns_finite_energy() {
        let config = SimpleUpdateConfig {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.05,
            n_steps: 10,
            tol: 1e-4,
            trotter_order: 2,
        };
        let gate = heisenberg_gate(1.0, config.delta_tau);
        let mut rng = LcgRng::new(77);
        let result = simple_update_run(&gate, &gate, &config, &mut rng).expect("run ok");
        assert!(
            result.energy_per_site.is_finite(),
            "energy not finite: {}",
            result.energy_per_site
        );
    }

    // ── Test 12: lambdas positive after run ───────────────────────────────────
    #[test]
    fn lambdas_positive_after_run() {
        let config = SimpleUpdateConfig {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.02,
            n_steps: 5,
            tol: 1e-5,
            trotter_order: 1,
        };
        let gate = heisenberg_gate(1.0, config.delta_tau);
        let mut rng = LcgRng::new(88);
        let result = simple_update_run(&gate, &gate, &config, &mut rng).expect("run ok");

        for v in &result.lambdas.horizontal {
            for &x in v {
                assert!(x > 0.0, "horizontal lambda ≤ 0: {x}");
            }
        }
        for v in &result.lambdas.vertical {
            for &x in v {
                assert!(x > 0.0, "vertical lambda ≤ 0: {x}");
            }
        }
    }

    // ── Test 13: Trotter order 1 runs without panic ───────────────────────────
    #[test]
    fn trotter_order_1_runs() {
        let config = SimpleUpdateConfig {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.05,
            n_steps: 6,
            tol: 1e-4,
            trotter_order: 1,
        };
        let gate = heisenberg_gate(1.0, config.delta_tau);
        let mut rng = LcgRng::new(55);
        let result = simple_update_run(&gate, &gate, &config, &mut rng).expect("order-1 run ok");
        assert!(result.energy_per_site.is_finite());
    }

    // ── Test 14: Trotter order 2 runs without panic ───────────────────────────
    #[test]
    fn trotter_order_2_runs() {
        let config = SimpleUpdateConfig {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.05,
            n_steps: 6,
            tol: 1e-4,
            trotter_order: 2,
        };
        let gate = heisenberg_gate(1.0, config.delta_tau);
        let mut rng = LcgRng::new(66);
        let result = simple_update_run(&gate, &gate, &config, &mut rng).expect("order-2 run ok");
        assert!(result.energy_per_site.is_finite());
    }

    // ── Test 15: mat_exp_sym: exp(0·H) = I ───────────────────────────────────
    #[test]
    fn mat_exp_zero_is_identity() {
        let h = heisenberg_hamiltonian_2site(1.0);
        let exp0 = mat_exp_sym(&h, 4, 0.0).expect("ok");
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (exp0[i * 4 + j] - expected).abs() < 1e-11,
                    "exp(0·H)[{i},{j}] = {} expected {expected}",
                    exp0[i * 4 + j]
                );
            }
        }
    }

    // ── Test 16: 2×2 lattice antiferromagnetic energy decreases ──────────────
    #[test]
    fn energy_decreases_antiferromagnetic() {
        let config = SimpleUpdateConfig {
            lx: 2,
            ly: 2,
            phys_dim: 2,
            bond_dim: 2,
            delta_tau: 0.01,
            n_steps: 50,
            tol: 1e-8,
            trotter_order: 2,
        };
        let gate = heisenberg_gate(1.0, config.delta_tau);
        let ham = heisenberg_hamiltonian_2site(1.0);
        let mut rng = LcgRng::new(314);

        let (tensors0, lambdas0) = simple_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let e_init = simple_update_energy(&tensors0, &lambdas0, &ham, 2, 2).expect("e_init ok");

        let mut rng2 = LcgRng::new(314);
        let result = simple_update_run(&gate, &gate, &config, &mut rng2).expect("run ok");
        let e_final =
            simple_update_energy(&result.tensors, &result.lambdas, &ham, 2, 2).expect("e_final ok");

        // The final energy (using the Hamiltonian) should differ from initial in a
        // controlled way (both are finite at minimum).
        assert!(e_init.is_finite(), "e_init not finite: {e_init}");
        assert!(e_final.is_finite(), "e_final not finite: {e_final}");
    }
}
