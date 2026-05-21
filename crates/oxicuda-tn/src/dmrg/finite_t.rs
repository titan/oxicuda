//! Finite-temperature DMRG via purification (FT-DMRG).
//!
//! ## Algorithm (Feiguin & White 2005, White 2009)
//!
//! The thermal density matrix `ρ(β) = e^{-βH}/Z` is represented as a pure state in a
//! **doubled Hilbert space** where each lattice site carries both a physical ("phys") and
//! an ancilla ("anc") degree of freedom. This is often called the **purification** or
//! **thermal field double** approach.
//!
//! ### Key idea
//!
//! The doubled local Hilbert space has dimension `d² = d_phys²`. Each MPS site tensor
//! lives in this extended space. The ancilla degrees of freedom are never acted upon
//! by the physical Hamiltonian, serving instead as a "copy" that entangles with the
//! physical system to encode thermal fluctuations.
//!
//! ### Workflow
//!
//! 1. **Initialization** (`purification_init`): Build the maximally entangled (infinite-temperature)
//!    MPS. At β=0, the state is a product state in the doubled space with each local tensor
//!    `A[0, s_phys * d + s_anc, 0] = 1/√d` for `s_phys == s_anc` and 0 otherwise.
//!
//! 2. **Imaginary-time evolution** (`trotter_sweep_doubled`): Apply TEBD-style gates
//!    `exp(-τ * H_phys) ⊗ I_anc` sweeping even and odd bonds. The Trotter step is
//!    `τ = β / (2 * n_trotter_steps)` (factor 2 for symmetric Trotter).
//!
//! 3. **Observables** (`finite_t_expectation`): Compute thermal expectation values by
//!    contracting the purified state with operators that act only on the physical subspace.
//!
//! ### MPS layout
//!
//! - `mps[i]` is a flat `Vec<f64>` with shape `[bond_dims[i], d_sq, bond_dims[i+1]]`
//!   stored in row-major order: element `(a, p, b)` → index `(a * d_sq + p) * bond_dims[i+1] + b`.
//! - `bond_dims` has length `n_sites + 1`, with `bond_dims[0] = bond_dims[n_sites] = 1`.
//!
//! ### References
//!
//! * A. E. Feiguin and S. R. White, *Phys. Rev. B* **72**, 220401(R) (2005).
//! * S. R. White, *Phys. Rev. Lett.* **102**, 190601 (2009).

use crate::error::{TnError, TnResult};
use crate::mps::truncation::svd_truncate;
use crate::peps::simple_update::mat_exp_sym;
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration & result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a finite-temperature DMRG (purification) run.
#[derive(Debug, Clone)]
pub struct FiniteTConfig {
    /// Number of physical sites.
    pub n_sites: usize,
    /// Physical dimension per site (doubled internally to d²).
    pub d_phys: usize,
    /// Maximum bond dimension in the doubled MPS.
    pub chi_max: usize,
    /// Total inverse temperature β.
    pub beta: f64,
    /// Number of Trotter steps for imaginary-time evolution.
    pub n_trotter_steps: usize,
    /// SVD truncation tolerance.
    pub trunc_tol: f64,
    /// Heisenberg coupling J.
    pub coupling_j: f64,
}

impl Default for FiniteTConfig {
    fn default() -> Self {
        Self {
            n_sites: 4,
            d_phys: 2,
            chi_max: 16,
            beta: 1.0,
            n_trotter_steps: 20,
            trunc_tol: 1e-10,
            coupling_j: 1.0,
        }
    }
}

/// Output of a completed FT-DMRG run.
#[derive(Debug, Clone)]
pub struct FiniteTResult {
    /// Energy per site ⟨H⟩/L.
    pub energy_per_site: f64,
    /// Partition function log(Z) / L.
    pub log_z_per_site: f64,
    /// Norm of the purified state (should be approximately √Z).
    pub state_norm: f64,
    /// Trotter steps actually applied.
    pub n_steps_applied: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Index into a site tensor with shape `[dl, d_sq, dr]`.
///
/// Element `(a, p, b)` lives at flat index `(a * d_sq + p) * dr + b`.
#[inline]
fn site_idx(a: usize, p: usize, b: usize, d_sq: usize, dr: usize) -> usize {
    (a * d_sq + p) * dr + b
}

/// Compute the L2 norm of an MPS by contracting ⟨ψ|ψ⟩ from left to right.
///
/// Each site contributes a contraction of the left environment (a square matrix
/// of size `dl × dl`) with the local tensor from bra and ket.
///
/// Returns √(⟨ψ|ψ⟩).
fn mps_norm(mps: &[Vec<f64>], bond_dims: &[usize]) -> TnResult<f64> {
    let n = mps.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    if bond_dims.len() != n + 1 {
        return Err(TnError::ShapeMismatch {
            expected: vec![n + 1],
            got: vec![bond_dims.len()],
        });
    }
    // Environment starts as 1×1 identity: env[α, α'] = δ(α, α').
    let mut env = vec![1.0_f64]; // 1×1
    let mut env_dim = 1usize;

    for i in 0..n {
        let dl = bond_dims[i];
        let d_sq = if i < n {
            let tensor_size = mps[i].len();
            let dr = bond_dims[i + 1];
            // d_sq * dl * dr = tensor_size
            if dl == 0 || dr == 0 {
                return Err(TnError::InvalidBondDimension(0));
            }
            tensor_size / (dl * dr)
        } else {
            1
        };
        let dr = bond_dims[i + 1];
        let t = &mps[i];

        // new_env[β, β'] = Σ_{α, α', p} env[α, α'] * t[α, p, β] * t[α', p, β']
        let mut new_env = vec![0.0_f64; dr * dr];
        for beta in 0..dr {
            for beta_p in 0..dr {
                let mut acc = 0.0;
                for alpha in 0..env_dim {
                    for alpha_p in 0..env_dim {
                        let ev = env[alpha * env_dim + alpha_p];
                        if ev.abs() < 1e-300 {
                            continue;
                        }
                        for p in 0..d_sq {
                            let tl = t[site_idx(alpha, p, beta, d_sq, dr)];
                            let tr = t[site_idx(alpha_p, p, beta_p, d_sq, dr)];
                            acc += ev * tl * tr;
                        }
                    }
                }
                new_env[beta * dr + beta_p] = acc;
            }
        }
        env = new_env;
        env_dim = dr;
    }
    // env is now 1×1: the inner product ⟨ψ|ψ⟩.
    let inner_sq = env[0].max(0.0);
    Ok(inner_sq.sqrt())
}

/// Compute ⟨ψ| O_site |ψ⟩ / ⟨ψ|ψ⟩ where O_site is an operator in the doubled space.
///
/// `op_doubled` is a `d_sq × d_sq` matrix (row-major).
fn mps_expectation_doubled(
    mps: &[Vec<f64>],
    bond_dims: &[usize],
    op_doubled: &[f64],
    site: usize,
    d_sq: usize,
) -> TnResult<f64> {
    let n = mps.len();
    if site >= n {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: n,
        });
    }
    if op_doubled.len() != d_sq * d_sq {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_sq, d_sq],
            got: vec![op_doubled.len()],
        });
    }

    // Build left environment up to site `site`: env_l[α, α']
    let mut env_l = vec![1.0_f64]; // 1×1
    let mut env_l_dim = 1usize;

    for i in 0..site {
        let dl = bond_dims[i];
        let dr = bond_dims[i + 1];
        let t = &mps[i];
        // d_sq for this site
        let dsq_i = mps[i].len() / (dl * dr);

        let mut new_env = vec![0.0_f64; dr * dr];
        for beta in 0..dr {
            for beta_p in 0..dr {
                let mut acc = 0.0;
                for alpha in 0..env_l_dim {
                    for alpha_p in 0..env_l_dim {
                        let ev = env_l[alpha * env_l_dim + alpha_p];
                        if ev.abs() < 1e-300 {
                            continue;
                        }
                        for p in 0..dsq_i {
                            let tl = t[site_idx(alpha, p, beta, dsq_i, dr)];
                            let tr = t[site_idx(alpha_p, p, beta_p, dsq_i, dr)];
                            acc += ev * tl * tr;
                        }
                    }
                }
                new_env[beta * dr + beta_p] = acc;
            }
        }
        env_l = new_env;
        env_l_dim = dr;
    }

    // Build right environment from site `site+1` to end.
    let mut env_r = vec![1.0_f64]; // 1×1
    let mut env_r_dim = 1usize;

    for i in (site + 1..n).rev() {
        let dl = bond_dims[i];
        let dr = bond_dims[i + 1];
        let t = &mps[i];
        let dsq_i = mps[i].len() / (dl * dr);

        let mut new_env = vec![0.0_f64; dl * dl];
        for alpha in 0..dl {
            for alpha_p in 0..dl {
                let mut acc = 0.0;
                for beta in 0..env_r_dim {
                    for beta_p in 0..env_r_dim {
                        let ev = env_r[beta * env_r_dim + beta_p];
                        if ev.abs() < 1e-300 {
                            continue;
                        }
                        for p in 0..dsq_i {
                            let tl = t[site_idx(alpha, p, beta, dsq_i, dr)];
                            let tr = t[site_idx(alpha_p, p, beta_p, dsq_i, dr)];
                            acc += ev * tl * tr;
                        }
                    }
                }
                new_env[alpha * dl + alpha_p] = acc;
            }
        }
        env_r = new_env;
        env_r_dim = dl;
    }

    // Contract at the target site with operator O.
    // ⟨ψ|O|ψ⟩ = Σ_{α,α',p,q,β,β'} env_l[α,α'] * t[α,p,β] * O[p,q] * t[α',q,β'] * env_r[β,β']
    let _dl = bond_dims[site];
    let dr = bond_dims[site + 1];
    let t = &mps[site];

    let mut numerator = 0.0;
    for alpha in 0..env_l_dim {
        for alpha_p in 0..env_l_dim {
            let evl = env_l[alpha * env_l_dim + alpha_p];
            if evl.abs() < 1e-300 {
                continue;
            }
            for beta in 0..env_r_dim {
                for beta_p in 0..env_r_dim {
                    let evr = env_r[beta * env_r_dim + beta_p];
                    if evr.abs() < 1e-300 {
                        continue;
                    }
                    for p in 0..d_sq {
                        let tl = t[site_idx(alpha, p, beta, d_sq, dr)];
                        if tl.abs() < 1e-300 {
                            continue;
                        }
                        for q in 0..d_sq {
                            let op_val = op_doubled[p * d_sq + q];
                            if op_val.abs() < 1e-300 {
                                continue;
                            }
                            let tr = t[site_idx(alpha_p, q, beta_p, d_sq, dr)];
                            numerator += evl * tl * op_val * tr * evr;
                        }
                    }
                }
            }
        }
    }

    // Also compute ⟨ψ|ψ⟩ from environments.
    // ⟨ψ|ψ⟩ = Σ_{α,α',p,β,β'} env_l[α,α'] * t[α,p,β] * t[α',p,β'] * env_r[β,β']
    let mut denominator = 0.0;
    for alpha in 0..env_l_dim {
        for alpha_p in 0..env_l_dim {
            let evl = env_l[alpha * env_l_dim + alpha_p];
            if evl.abs() < 1e-300 {
                continue;
            }
            for beta in 0..env_r_dim {
                for beta_p in 0..env_r_dim {
                    let evr = env_r[beta * env_r_dim + beta_p];
                    if evr.abs() < 1e-300 {
                        continue;
                    }
                    for p in 0..d_sq {
                        let tl = t[site_idx(alpha, p, beta, d_sq, dr)];
                        let tr = t[site_idx(alpha_p, p, beta_p, d_sq, dr)];
                        denominator += evl * tl * tr * evr;
                    }
                }
            }
        }
    }

    if denominator.abs() < 1e-300 {
        return Err(TnError::NumericalInstability(
            "zero norm in expectation denominator".into(),
        ));
    }
    Ok(numerator / denominator)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Build the initial (β=0) maximally entangled MPS in the doubled (d²) space.
///
/// The initial state is the infinite-temperature purification:
/// `|I⟩ = (1/√d)^L Σ_{s_1,...,s_L} |s_1,...,s_L⟩_phys ⊗ |s_1,...,s_L⟩_anc`.
///
/// In the doubled MPS, each site tensor has shape `[1, d², 1]` with:
/// `A[0, s_phys * d + s_anc, 0] = 1/√d` if `s_phys == s_anc` else `0`.
///
/// The returned `Vec<Vec<f64>>` has `n_sites` entries, each of length `d²`.
/// The companion `bond_dims` has length `n_sites + 1` and is all 1s.
pub fn purification_init(cfg: &FiniteTConfig) -> TnResult<Vec<Vec<f64>>> {
    if cfg.n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if cfg.d_phys == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    let d = cfg.d_phys;
    let d_sq = d * d;
    let scale = 1.0 / (d as f64).sqrt();

    let mut mps = Vec::with_capacity(cfg.n_sites);
    for _ in 0..cfg.n_sites {
        // Tensor shape: [1, d_sq, 1] — product state, bond dims all 1.
        let mut tensor = vec![0.0_f64; d_sq];
        for s in 0..d {
            // s_phys == s_anc == s → index s * d + s.
            tensor[s * d + s] = scale;
        }
        mps.push(tensor);
    }
    Ok(mps)
}

/// Build bond dimension array for the purification MPS.
///
/// All bonds are 1 (product state at β=0).
pub fn purification_bond_dims(n_sites: usize) -> Vec<usize> {
    vec![1usize; n_sites + 1]
}

/// Compute the physical Heisenberg Hamiltonian for 2 spin-1/2 sites (d=2).
///
/// In the basis `|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩`:
/// ```text
/// H = J * [[1,  0,  0, 0],
///           [0, -1,  2, 0],
///           [0,  2, -1, 0],
///           [0,  0,  0, 1]]
/// ```
fn heisenberg_h_phys(d_phys: usize, j: f64) -> TnResult<Vec<f64>> {
    if d_phys != 2 {
        return Err(TnError::InvalidConfiguration(
            "FT-DMRG Heisenberg gate only implemented for d_phys=2".into(),
        ));
    }
    // 4×4 matrix in basis |00⟩,|01⟩,|10⟩,|11⟩ (↑=0, ↓=1).
    let h = vec![
        j,
        0.0,
        0.0,
        0.0, // row 0
        0.0,
        -j,
        2.0 * j,
        0.0, // row 1
        0.0,
        2.0 * j,
        -j,
        0.0, // row 2
        0.0,
        0.0,
        0.0,
        j, // row 3
    ];
    Ok(h)
}

/// Compute the 4-body (2-site in doubled space) Heisenberg imaginary-time gate
/// `exp(-τ * H_phys ⊗ I_anc)`.
///
/// For a physical pair in doubled space, the gate acts on:
/// `G_doubled[(p1_p*d+p1_a, p2_p*d+p2_a), (q1_p*d+q1_a, q2_p*d+q2_a)]`
/// `= gate_phys[(p1_p,p2_p),(q1_p,q2_p)] * δ(p1_a,q1_a) * δ(p2_a,q2_a)`
///
/// Returns a flat `(d²)² × (d²)²` matrix (i.e. `d^4 × d^4` for the doubled space).
/// Index ordering: `G[q1_d*d_sq + q2_d, p1_d*d_sq + p2_d]` where `d_sq = d²`.
///
/// # Errors
/// Returns [`TnError::InvalidConfiguration`] if `d_phys != 2`.
pub fn heisenberg_gate_doubled(d_phys: usize, tau: f64, j: f64) -> TnResult<Vec<f64>> {
    let d = d_phys;
    let d_sq = d * d; // physical dim of doubled MPS per site (e.g. 4 for d=2)
    let gate_dim = d_sq * d_sq; // total gate matrix dimension (e.g. 16 for d=2)

    // Physical Hamiltonian on 2 physical spins: 4×4 for d=2.
    let h_phys = heisenberg_h_phys(d, j)?;
    let n_phys = d * d; // = 4 for d=2

    // Compute gate_phys = exp(-τ * H_phys): n_phys × n_phys.
    let gate_phys = mat_exp_sym(&h_phys, n_phys, -tau)?;

    // Embed into the doubled space: size (d_sq × d_sq) × (d_sq × d_sq).
    // G_doubled[(s1_p*d + s1_a, s2_p*d + s2_a), (t1_p*d + t1_a, t2_p*d + t2_a)]
    //   = gate_phys[(s1_p * d + s2_p), (t1_p * d + t2_p)] * δ(s1_a, t1_a) * δ(s2_a, t2_a)
    //
    // The gate_phys indexing: row = s1_p*d + s2_p, col = t1_p*d + t2_p.
    //
    // We store G_doubled as a flat (gate_dim × gate_dim) matrix with row-major ordering
    // where the row index encodes (s1_p, s1_a, s2_p, s2_a) in a specific way.
    //
    // For TEBD application, the gate is applied as:
    // θ'[bl, q1_d, q2_d, br] = Σ_{p1_d, p2_d} G[q1_d, q2_d, p1_d, p2_d] * θ[bl, p1_d, p2_d, br]
    //
    // So we store G as: G_flat[q1_d * d_sq + q2_d, p1_d * d_sq + p2_d]
    // where q1_d = q1_p*d + q1_a, etc.

    let mut g_doubled = vec![0.0_f64; gate_dim * gate_dim];

    for s1_p in 0..d {
        for s1_a in 0..d {
            let s1_d = s1_p * d + s1_a; // doubled index for site 1, output
            for s2_p in 0..d {
                for s2_a in 0..d {
                    let s2_d = s2_p * d + s2_a; // doubled index for site 2, output
                    for t1_p in 0..d {
                        for t2_p in 0..d {
                            // Ancilla indices must be preserved: t1_a = s1_a, t2_a = s2_a.
                            let t1_d = t1_p * d + s1_a;
                            let t2_d = t2_p * d + s2_a;
                            let phys_row = s1_p * d + s2_p;
                            let phys_col = t1_p * d + t2_p;
                            let gp = gate_phys[phys_row * n_phys + phys_col];
                            // Row index in G_doubled: (s1_d, s2_d) → s1_d * d_sq + s2_d
                            // Col index in G_doubled: (t1_d, t2_d) → t1_d * d_sq + t2_d
                            let row = s1_d * d_sq + s2_d;
                            let col = t1_d * d_sq + t2_d;
                            g_doubled[row * gate_dim + col] += gp;
                        }
                    }
                }
            }
        }
    }
    Ok(g_doubled)
}

/// Apply a single two-site gate to the doubled MPS at bond (site_i, site_i+1).
///
/// Steps:
/// 1. Contract `mps[i]` and `mps[i+1]` into theta `[dl, d_sq, d_sq, dr]`.
/// 2. Apply gate: `theta'[bl, q1, q2, br] = Σ_{p1,p2} G[q1*d_sq+q2, p1*d_sq+p2] * theta[bl,p1,p2,br]`.
/// 3. SVD `theta'` reshaped to `[dl*d_sq, d_sq*dr]`, truncate to `chi_max`.
/// 4. Write updated tensors back into `mps[i]` and `mps[i+1]`.
fn apply_gate_doubled(
    mps: &mut [Vec<f64>],
    bond_dims: &mut [usize],
    gate: &[f64],
    i: usize,
    d_sq: usize,
    chi_max: usize,
    trunc_tol: f64,
) -> TnResult<()> {
    let n = mps.len();
    if i + 1 >= n {
        return Err(TnError::IndexOutOfBounds { index: i, len: n });
    }
    let dl = bond_dims[i];
    let dm = bond_dims[i + 1]; // inner bond between site i and i+1
    let dr = bond_dims[i + 2];

    if mps[i].len() != dl * d_sq * dm {
        return Err(TnError::ShapeMismatch {
            expected: vec![dl, d_sq, dm],
            got: vec![mps[i].len()],
        });
    }
    if mps[i + 1].len() != dm * d_sq * dr {
        return Err(TnError::ShapeMismatch {
            expected: vec![dm, d_sq, dr],
            got: vec![mps[i + 1].len()],
        });
    }
    let gate_dim = d_sq * d_sq;
    if gate.len() != gate_dim * gate_dim {
        return Err(TnError::ShapeMismatch {
            expected: vec![gate_dim, gate_dim],
            got: vec![gate.len()],
        });
    }

    // Step 1: Contract theta[a, p1, p2, b] = Σ_c mps[i][a,p1,c] * mps[i+1][c,p2,b]
    let mut theta = vec![0.0_f64; dl * d_sq * d_sq * dr];
    for a in 0..dl {
        for p1 in 0..d_sq {
            for p2 in 0..d_sq {
                for b in 0..dr {
                    let mut acc = 0.0_f64;
                    for c in 0..dm {
                        let lv = mps[i][site_idx(a, p1, c, d_sq, dm)];
                        let rv = mps[i + 1][site_idx(c, p2, b, d_sq, dr)];
                        acc += lv * rv;
                    }
                    // theta index: ((a * d_sq + p1) * d_sq + p2) * dr + b
                    theta[((a * d_sq + p1) * d_sq + p2) * dr + b] = acc;
                }
            }
        }
    }

    // Step 2: Apply gate: theta'[a, q1, q2, b] = Σ_{p1,p2} G[q1*d_sq+q2, p1*d_sq+p2] * theta[a,p1,p2,b]
    let mut theta_prime = vec![0.0_f64; dl * d_sq * d_sq * dr];
    for a in 0..dl {
        for q1 in 0..d_sq {
            for q2 in 0..d_sq {
                for b in 0..dr {
                    let mut acc = 0.0_f64;
                    let gate_row = q1 * d_sq + q2;
                    for p1 in 0..d_sq {
                        for p2 in 0..d_sq {
                            let gate_col = p1 * d_sq + p2;
                            let gv = gate[gate_row * gate_dim + gate_col];
                            if gv.abs() < 1e-300 {
                                continue;
                            }
                            let tv = theta[((a * d_sq + p1) * d_sq + p2) * dr + b];
                            acc += gv * tv;
                        }
                    }
                    theta_prime[((a * d_sq + q1) * d_sq + q2) * dr + b] = acc;
                }
            }
        }
    }

    // Step 3: Reshape theta' to matrix (dl*d_sq, d_sq*dr), perform SVD and truncate.
    let m = dl * d_sq;
    let nn = d_sq * dr;

    // Reorder theta' from [dl, d_sq, d_sq, dr] to [dl*d_sq, d_sq*dr]:
    // matrix[a*d_sq+q1, q2*dr+b] = theta'[a, q1, q2, b]
    let mut mat = vec![0.0_f64; m * nn];
    for a in 0..dl {
        for q1 in 0..d_sq {
            for q2 in 0..d_sq {
                for b in 0..dr {
                    let row = a * d_sq + q1;
                    let col = q2 * dr + b;
                    mat[row * nn + col] = theta_prime[((a * d_sq + q1) * d_sq + q2) * dr + b];
                }
            }
        }
    }

    let svd = svd_jacobi(&mat, m, nn)?;
    let (svd_trunc, _err) = svd_truncate(svd, chi_max, trunc_tol)?;
    let k = svd_trunc.k;

    // Step 4: Write back tensors.
    // Left tensor: shape [dl, d_sq, k], data from U * diag(S)^{1/2} (absorb S into left).
    // Here we absorb all singular values into the right tensor for left-canonical form.
    // mps[i] = U: shape [m, k] → reshape to [dl, d_sq, k].
    let mut left_new = vec![0.0_f64; dl * d_sq * k];
    for row in 0..m {
        let a = row / d_sq;
        let q1 = row % d_sq;
        for j in 0..k {
            left_new[site_idx(a, q1, j, d_sq, k)] = svd_trunc.u[row * k + j];
        }
    }

    // mps[i+1] = diag(S) * V^T: shape [k, d_sq, dr].
    // V^T has shape [k, nn], where nn = d_sq * dr.
    // right_new[c, q2, b] ← S[c] * Vt[c, q2*dr + b]
    let mut right_new = vec![0.0_f64; k * d_sq * dr];
    for c in 0..k {
        let sv = svd_trunc.s[c];
        for col in 0..nn {
            let q2 = col / dr;
            let b = col % dr;
            right_new[site_idx(c, q2, b, d_sq, dr)] = sv * svd_trunc.vt[c * nn + col];
        }
    }

    mps[i] = left_new;
    mps[i + 1] = right_new;
    bond_dims[i + 1] = k;
    Ok(())
}

/// Apply one even-odd Trotter sweep on the doubled MPS.
///
/// The sweep applies all even bonds (i=0,2,4,...) then all odd bonds (i=1,3,5,...).
/// This is the standard first-order Trotter approach; for symmetric Trotter the caller
/// should use half-steps.
///
/// `gate` is the precomputed gate `exp(-τ * H ⊗ I)` of size `(d²)^4`.
pub fn trotter_sweep_doubled(
    mps: &mut [Vec<f64>],
    bond_dims: &mut [usize],
    gate: &[f64],
    d_sq: usize,
    chi_max: usize,
    trunc_tol: f64,
) -> TnResult<()> {
    let n = mps.len();
    if n < 2 {
        return Ok(()); // Nothing to do for single-site chain.
    }
    // Even bonds: i = 0, 2, 4, ...
    let mut i = 0;
    while i + 1 < n {
        apply_gate_doubled(mps, bond_dims, gate, i, d_sq, chi_max, trunc_tol)?;
        i += 2;
    }
    // Odd bonds: i = 1, 3, 5, ...
    let mut i = 1;
    while i + 1 < n {
        apply_gate_doubled(mps, bond_dims, gate, i, d_sq, chi_max, trunc_tol)?;
        i += 2;
    }
    Ok(())
}

/// Compute the energy per site by contracting the two-site Heisenberg Hamiltonian
/// `H_phys ⊗ I_anc` with the doubled MPS.
///
/// For each bond (i, i+1), we form the two-site reduced density matrix element
/// and contract with the two-site Hamiltonian. The energy per site is the total
/// divided by the number of sites.
fn compute_energy_per_site(
    mps: &[Vec<f64>],
    bond_dims: &[usize],
    d_sq: usize,
    d_phys: usize,
    j: f64,
) -> TnResult<f64> {
    let n = mps.len();
    if n < 2 {
        return Ok(0.0);
    }

    // Physical Heisenberg Hamiltonian on 2 physical spins: (d*d) × (d*d) matrix.
    let h_phys = heisenberg_h_phys(d_phys, j)?;
    let n_phys = d_phys * d_phys;

    // Embed H_phys into doubled space: H_doubled[(s1_p*d+s1_a, s2_p*d+s2_a), (t1_p*d+t1_a, t2_p*d+t2_a)]
    //   = H_phys[(s1_p*d+s2_p), (t1_p*d+t2_p)] * δ(s1_a, t1_a) * δ(s2_a, t2_a)
    let gate_dim = d_sq * d_sq;
    let mut h_doubled = vec![0.0_f64; gate_dim * gate_dim];

    for s1_p in 0..d_phys {
        for s1_a in 0..d_phys {
            let s1_d = s1_p * d_phys + s1_a;
            for s2_p in 0..d_phys {
                for s2_a in 0..d_phys {
                    let s2_d = s2_p * d_phys + s2_a;
                    for t1_p in 0..d_phys {
                        for t2_p in 0..d_phys {
                            let t1_d = t1_p * d_phys + s1_a;
                            let t2_d = t2_p * d_phys + s2_a;
                            let phys_row = s1_p * d_phys + s2_p;
                            let phys_col = t1_p * d_phys + t2_p;
                            let hval = h_phys[phys_row * n_phys + phys_col];
                            let row = s1_d * d_sq + s2_d;
                            let col = t1_d * d_sq + t2_d;
                            h_doubled[row * gate_dim + col] += hval;
                        }
                    }
                }
            }
        }
    }

    // Compute total energy as sum over all bonds using two-site expectation.
    let mut total_energy = 0.0_f64;
    for bond in 0..n - 1 {
        // Compute ⟨H_bond⟩ = ⟨ψ| H_doubled_{bond,bond+1} |ψ⟩ / ⟨ψ|ψ⟩
        let e_bond = two_site_expectation(mps, bond_dims, &h_doubled, bond, d_sq)?;
        total_energy += e_bond;
    }

    Ok(total_energy / n as f64)
}

/// Compute ⟨ψ| O_{i,i+1} |ψ⟩ / ⟨ψ|ψ⟩ for a two-site operator in the doubled space.
///
/// `op` is a `(d_sq*d_sq) × (d_sq*d_sq)` matrix acting on sites `i` and `i+1`.
fn two_site_expectation(
    mps: &[Vec<f64>],
    bond_dims: &[usize],
    op: &[f64],
    i: usize,
    d_sq: usize,
) -> TnResult<f64> {
    let n = mps.len();
    if i + 1 >= n {
        return Err(TnError::IndexOutOfBounds { index: i, len: n });
    }
    let gate_dim = d_sq * d_sq;
    if op.len() != gate_dim * gate_dim {
        return Err(TnError::ShapeMismatch {
            expected: vec![gate_dim, gate_dim],
            got: vec![op.len()],
        });
    }

    // Build left environment up to site i.
    let mut env_l = vec![1.0_f64];
    let mut env_l_dim = 1usize;

    for s in 0..i {
        let dl = bond_dims[s];
        let dr = bond_dims[s + 1];
        let dsq_s = mps[s].len() / (dl * dr);
        let t = &mps[s];
        let mut new_env = vec![0.0_f64; dr * dr];
        for beta in 0..dr {
            for beta_p in 0..dr {
                let mut acc = 0.0;
                for alpha in 0..env_l_dim {
                    for alpha_p in 0..env_l_dim {
                        let ev = env_l[alpha * env_l_dim + alpha_p];
                        if ev.abs() < 1e-300 {
                            continue;
                        }
                        for p in 0..dsq_s {
                            let tl = t[site_idx(alpha, p, beta, dsq_s, dr)];
                            let tr = t[site_idx(alpha_p, p, beta_p, dsq_s, dr)];
                            acc += ev * tl * tr;
                        }
                    }
                }
                new_env[beta * dr + beta_p] = acc;
            }
        }
        env_l = new_env;
        env_l_dim = dr;
    }

    // Build right environment from site i+2 onward.
    let mut env_r = vec![1.0_f64];
    let mut env_r_dim = 1usize;

    for s in (i + 2..n).rev() {
        let dl = bond_dims[s];
        let dr = bond_dims[s + 1];
        let dsq_s = mps[s].len() / (dl * dr);
        let t = &mps[s];
        let mut new_env = vec![0.0_f64; dl * dl];
        for alpha in 0..dl {
            for alpha_p in 0..dl {
                let mut acc = 0.0;
                for beta in 0..env_r_dim {
                    for beta_p in 0..env_r_dim {
                        let ev = env_r[beta * env_r_dim + beta_p];
                        if ev.abs() < 1e-300 {
                            continue;
                        }
                        for p in 0..dsq_s {
                            let tl = t[site_idx(alpha, p, beta, dsq_s, dr)];
                            let tr = t[site_idx(alpha_p, p, beta_p, dsq_s, dr)];
                            acc += ev * tl * tr;
                        }
                    }
                }
                new_env[alpha * dl + alpha_p] = acc;
            }
        }
        env_r = new_env;
        env_r_dim = dl;
    }

    // Contract the two-site block with operator.
    let _dl_i = bond_dims[i];
    let dm = bond_dims[i + 1];
    let dr_i1 = bond_dims[i + 2];
    let ti = &mps[i];
    let ti1 = &mps[i + 1];

    // Form theta[alpha, p1, p2, beta] = Σ_c ti[alpha,p1,c] * ti1[c,p2,beta]
    // Then: numerator = Σ env_l[α,α'] * theta[α,p1,p2,β] * op[(q1,q2),(p1,p2)] * theta[α',q1,q2,β'] * env_r[β,β']
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    for alpha in 0..env_l_dim {
        for alpha_p in 0..env_l_dim {
            let evl = env_l[alpha * env_l_dim + alpha_p];
            if evl.abs() < 1e-300 {
                continue;
            }
            for beta in 0..env_r_dim {
                for beta_p in 0..env_r_dim {
                    let evr = env_r[beta * env_r_dim + beta_p];
                    if evr.abs() < 1e-300 {
                        continue;
                    }
                    // Compute theta[alpha,p1,p2,beta] for all p1,p2.
                    for p1 in 0..d_sq {
                        for p2 in 0..d_sq {
                            // theta_val = Σ_c ti[alpha,p1,c] * ti1[c,p2,beta]
                            let mut theta_val = 0.0_f64;
                            for c in 0..dm {
                                theta_val += ti[site_idx(alpha, p1, c, d_sq, dm)]
                                    * ti1[site_idx(c, p2, beta, d_sq, dr_i1)];
                            }
                            if theta_val.abs() < 1e-300 {
                                continue;
                            }

                            // Denominator: uses identity on op.
                            // Σ_c ti1_p[alpha',p1,c] * ti1_1[c,p2,beta']
                            let mut theta_p = 0.0_f64;
                            for c in 0..dm {
                                theta_p += ti[site_idx(alpha_p, p1, c, d_sq, dm)]
                                    * ti1[site_idx(c, p2, beta_p, d_sq, dr_i1)];
                            }
                            denominator += evl * theta_val * theta_p * evr;

                            // Numerator: apply op.
                            let op_row = p1 * d_sq + p2;
                            for q1 in 0..d_sq {
                                for q2 in 0..d_sq {
                                    let op_col = q1 * d_sq + q2;
                                    let ov = op[op_row * gate_dim + op_col];
                                    if ov.abs() < 1e-300 {
                                        continue;
                                    }
                                    let mut theta_q = 0.0_f64;
                                    for c in 0..dm {
                                        theta_q += ti[site_idx(alpha_p, q1, c, d_sq, dm)]
                                            * ti1[site_idx(c, q2, beta_p, d_sq, dr_i1)];
                                    }
                                    numerator += evl * theta_val * ov * theta_q * evr;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if denominator.abs() < 1e-300 {
        return Err(TnError::NumericalInstability(
            "zero norm in two-site expectation".into(),
        ));
    }
    Ok(numerator / denominator)
}

/// Run full FT-DMRG simulation from β=0 to `cfg.beta` using purification.
///
/// The algorithm:
/// 1. Build the maximally entangled initial state `|I⟩` at β=0.
/// 2. Apply `n_trotter_steps` imaginary-time Trotter sweeps with step `τ = β/(2·n_steps)`.
///    The factor 2 accounts for the symmetric second-order Trotter splitting (forward + backward).
/// 3. Compute observables on the resulting purified state.
///
/// The state norm gives `√Z(β)`, so `log(Z)/L = 2 log(norm) / L`.
///
/// # Errors
/// Returns errors from SVD, shape mismatches, or numerical instabilities.
pub fn finite_t_run(cfg: &FiniteTConfig) -> TnResult<FiniteTResult> {
    if cfg.n_sites < 2 {
        return Err(TnError::InvalidConfiguration(
            "FT-DMRG requires at least 2 sites".into(),
        ));
    }
    if cfg.d_phys == 0 || cfg.chi_max == 0 || cfg.n_trotter_steps == 0 {
        return Err(TnError::InvalidConfiguration(
            "d_phys, chi_max, n_trotter_steps must be positive".into(),
        ));
    }

    let d = cfg.d_phys;
    let d_sq = d * d;

    // Step 1: Initialize the maximally entangled state.
    let mut mps = purification_init(cfg)?;
    let mut bond_dims = purification_bond_dims(cfg.n_sites);

    // Step 2: Imaginary-time evolution.
    // Symmetric Trotter: apply even-odd sweep with τ = β/(2*n_steps).
    // Each full Trotter step corresponds to τ in imaginary time.
    let tau = cfg.beta / (2.0 * cfg.n_trotter_steps as f64);

    // Precompute the gate exp(-τ * H ⊗ I).
    let gate = heisenberg_gate_doubled(d, tau, cfg.coupling_j)?;

    let mut steps_applied = 0usize;
    for _ in 0..cfg.n_trotter_steps {
        trotter_sweep_doubled(
            &mut mps,
            &mut bond_dims,
            &gate,
            d_sq,
            cfg.chi_max,
            cfg.trunc_tol,
        )?;
        steps_applied += 1;
    }

    // Step 3: Compute observables.
    // State norm gives √Z(β).
    let state_norm = mps_norm(&mps, &bond_dims)?;

    // log(Z)/L = 2*log(norm)/L (the factor 2 is because norm = Z^{1/2} only up to
    // normalization of the initial state; here we track it consistently).
    let log_z_per_site = if state_norm > 1e-300 {
        2.0 * state_norm.ln() / cfg.n_sites as f64
    } else {
        f64::NEG_INFINITY
    };

    // Energy per site: ⟨H⟩ / L.
    let energy_per_site =
        compute_energy_per_site(&mps, &bond_dims, d_sq, cfg.d_phys, cfg.coupling_j)?;

    Ok(FiniteTResult {
        energy_per_site,
        log_z_per_site,
        state_norm,
        n_steps_applied: steps_applied,
    })
}

/// Compute thermal expectation of a single-site operator (d_phys×d_phys matrix) at site `i`.
///
/// The operator is embedded into the doubled (d²) space as:
/// `O_doubled[s_p * d + s_a, t_p * d + s_a] = O[s_p, t_p]` (identity on ancilla).
///
/// The expectation is `⟨O⟩_β = ⟨ψ(β)| O_doubled |ψ(β)⟩ / ⟨ψ(β)|ψ(β)⟩`.
///
/// # Parameters
/// - `mps`: The purified MPS tensors (one per site, shape `[dl, d², dr]`).
/// - `bond_dims`: Bond dimension array of length `n_sites + 1`.
/// - `op`: The physical operator, a `d_phys × d_phys` row-major matrix.
/// - `site`: Site index at which to evaluate the expectation.
/// - `d_phys`: Physical dimension (the doubled dimension is `d_phys²`).
///
/// # Errors
/// Returns [`TnError::IndexOutOfBounds`] if `site >= n_sites`.
/// Returns [`TnError::ShapeMismatch`] if `op.len() != d_phys * d_phys`.
pub fn finite_t_expectation(
    mps: &[Vec<f64>],
    bond_dims: &[usize],
    op: &[f64],
    site: usize,
    d_phys: usize,
) -> TnResult<f64> {
    if d_phys == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if op.len() != d_phys * d_phys {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_phys, d_phys],
            got: vec![op.len()],
        });
    }

    let d_sq = d_phys * d_phys;

    // Embed operator into doubled space: sum over ancilla index.
    // O_doubled[s_p * d + s_a, t_p * d + t_a] = O[s_p, t_p] * δ(s_a, t_a)
    let mut op_doubled = vec![0.0_f64; d_sq * d_sq];
    for s_p in 0..d_phys {
        for t_p in 0..d_phys {
            let oval = op[s_p * d_phys + t_p];
            if oval.abs() < 1e-300 {
                continue;
            }
            for s_a in 0..d_phys {
                let row = s_p * d_phys + s_a;
                let col = t_p * d_phys + s_a;
                op_doubled[row * d_sq + col] += oval;
            }
        }
    }

    mps_expectation_doubled(mps, bond_dims, &op_doubled, site, d_sq)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Default small config for unit tests (fast).
    fn small_cfg() -> FiniteTConfig {
        FiniteTConfig {
            n_sites: 4,
            d_phys: 2,
            chi_max: 16,
            beta: 1.0,
            n_trotter_steps: 10,
            trunc_tol: 1e-10,
            coupling_j: 1.0,
        }
    }

    // ── Test 1: purification_init_shape ──────────────────────────────────────

    #[test]
    fn purification_init_shape() {
        let cfg = small_cfg();
        let mps = purification_init(&cfg).expect("init ok");
        assert_eq!(mps.len(), cfg.n_sites, "should have n_sites tensors");
        let d_sq = cfg.d_phys * cfg.d_phys;
        for (i, t) in mps.iter().enumerate() {
            // Product state: all bond dims = 1, so each tensor has d_sq elements.
            assert_eq!(
                t.len(),
                d_sq,
                "site {} tensor length should be d_sq = {}",
                i,
                d_sq
            );
        }
        let bond_dims = purification_bond_dims(cfg.n_sites);
        assert_eq!(bond_dims.len(), cfg.n_sites + 1);
        assert_eq!(bond_dims[0], 1);
        assert_eq!(bond_dims[cfg.n_sites], 1);
    }

    // ── Test 2: purification_init_product ────────────────────────────────────

    #[test]
    fn purification_init_product() {
        let cfg = small_cfg();
        let mps = purification_init(&cfg).expect("init ok");
        let bond_dims = purification_bond_dims(cfg.n_sites);

        // The initial state is a normalized product state: norm should be 1.
        let norm = mps_norm(&mps, &bond_dims).expect("norm ok");
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "initial state norm should be 1.0, got {}",
            norm
        );
    }

    // ── Test 3: heisenberg_gate_doubled_shape ─────────────────────────────────

    #[test]
    fn heisenberg_gate_doubled_shape() {
        let d = 2usize;
        let d_sq = d * d;
        let gate_dim = d_sq * d_sq;
        let gate = heisenberg_gate_doubled(d, 0.01, 1.0).expect("gate ok");
        assert_eq!(
            gate.len(),
            gate_dim * gate_dim,
            "gate should have (d²)^4 elements = {}",
            gate_dim * gate_dim
        );
    }

    // ── Test 4: heisenberg_gate_doubled_hermitian ─────────────────────────────

    #[test]
    fn heisenberg_gate_doubled_hermitian() {
        let d = 2usize;
        let d_sq = d * d;
        let gate_dim = d_sq * d_sq;
        // Small τ → gate ≈ I - τ*H is nearly symmetric.
        let gate = heisenberg_gate_doubled(d, 0.001, 1.0).expect("gate ok");
        let mut max_asymm = 0.0_f64;
        for row in 0..gate_dim {
            for col in 0..gate_dim {
                let diff = (gate[row * gate_dim + col] - gate[col * gate_dim + row]).abs();
                if diff > max_asymm {
                    max_asymm = diff;
                }
            }
        }
        assert!(
            max_asymm < 1e-10,
            "gate should be symmetric (real), max asymmetry = {}",
            max_asymm
        );
    }

    // ── Test 5: trotter_sweep_doubled_runs ───────────────────────────────────

    #[test]
    fn trotter_sweep_doubled_runs() {
        let cfg = small_cfg();
        let d_sq = cfg.d_phys * cfg.d_phys;
        let mut mps = purification_init(&cfg).expect("init ok");
        let mut bond_dims = purification_bond_dims(cfg.n_sites);

        let tau = cfg.beta / (2.0 * cfg.n_trotter_steps as f64);
        let gate = heisenberg_gate_doubled(cfg.d_phys, tau, cfg.coupling_j).expect("gate ok");

        trotter_sweep_doubled(
            &mut mps,
            &mut bond_dims,
            &gate,
            d_sq,
            cfg.chi_max,
            cfg.trunc_tol,
        )
        .expect("sweep should run without error");

        // After a sweep, the MPS should still have n_sites tensors.
        assert_eq!(mps.len(), cfg.n_sites);
    }

    // ── Test 6: trotter_sweep_bond_bounded ───────────────────────────────────

    #[test]
    fn trotter_sweep_bond_bounded() {
        let cfg = FiniteTConfig {
            chi_max: 4,
            n_trotter_steps: 20,
            ..small_cfg()
        };
        let d_sq = cfg.d_phys * cfg.d_phys;
        let mut mps = purification_init(&cfg).expect("init ok");
        let mut bond_dims = purification_bond_dims(cfg.n_sites);

        let tau = cfg.beta / (2.0 * cfg.n_trotter_steps as f64);
        let gate = heisenberg_gate_doubled(cfg.d_phys, tau, cfg.coupling_j).expect("gate ok");

        for _ in 0..cfg.n_trotter_steps {
            trotter_sweep_doubled(
                &mut mps,
                &mut bond_dims,
                &gate,
                d_sq,
                cfg.chi_max,
                cfg.trunc_tol,
            )
            .expect("sweep ok");
        }

        // All interior bond dimensions must be ≤ chi_max.
        for (i, &bond) in bond_dims.iter().enumerate().take(cfg.n_sites).skip(1) {
            assert!(
                bond <= cfg.chi_max,
                "bond {} dim {} exceeds chi_max {}",
                i,
                bond,
                cfg.chi_max
            );
        }
    }

    // ── Test 7: finite_t_run_small ───────────────────────────────────────────

    #[test]
    fn finite_t_run_small() {
        let cfg = FiniteTConfig {
            n_sites: 2,
            beta: 0.5,
            n_trotter_steps: 10,
            ..small_cfg()
        };
        let result = finite_t_run(&cfg);
        assert!(
            result.is_ok(),
            "L=2 β=0.5 should run without panic: {:?}",
            result.err()
        );
    }

    // ── Test 8: finite_t_run_high_t ──────────────────────────────────────────

    #[test]
    fn finite_t_run_high_t() {
        // β → 0 means infinite temperature. Energy per site → 0 for Heisenberg chain.
        let cfg = FiniteTConfig {
            n_sites: 2,
            beta: 1e-4,
            n_trotter_steps: 5,
            chi_max: 8,
            ..small_cfg()
        };
        let result = finite_t_run(&cfg).expect("high-T run ok");
        assert!(
            result.energy_per_site.abs() < 0.1,
            "high-T energy per site should be near 0, got {}",
            result.energy_per_site
        );
    }

    // ── Test 9: finite_t_run_energy_finite ───────────────────────────────────

    #[test]
    fn finite_t_run_energy_finite() {
        let cfg = small_cfg();
        let result = finite_t_run(&cfg).expect("run ok");
        assert!(
            result.energy_per_site.is_finite(),
            "energy_per_site must be finite, got {}",
            result.energy_per_site
        );
    }

    // ── Test 10: finite_t_run_norm_positive ──────────────────────────────────

    #[test]
    fn finite_t_run_norm_positive() {
        let cfg = small_cfg();
        let result = finite_t_run(&cfg).expect("run ok");
        assert!(
            result.state_norm > 0.0,
            "state_norm must be positive, got {}",
            result.state_norm
        );
    }

    // ── Test 11: finite_t_run_log_z_finite ───────────────────────────────────

    #[test]
    fn finite_t_run_log_z_finite() {
        let cfg = small_cfg();
        let result = finite_t_run(&cfg).expect("run ok");
        assert!(
            result.log_z_per_site.is_finite(),
            "log_z_per_site must be finite, got {}",
            result.log_z_per_site
        );
    }

    // ── Test 12: finite_t_run_l4 ─────────────────────────────────────────────

    #[test]
    fn finite_t_run_l4() {
        let cfg = FiniteTConfig {
            n_sites: 4,
            beta: 1.0,
            n_trotter_steps: 20,
            ..small_cfg()
        };
        let result = finite_t_run(&cfg).expect("L=4 β=1.0 run ok");
        assert_eq!(
            result.n_steps_applied, 20,
            "should apply exactly 20 Trotter steps"
        );
    }

    // ── Test 13: finite_t_expectation_identity ────────────────────────────────

    #[test]
    fn finite_t_expectation_identity() {
        let cfg = small_cfg();
        let mut mps = purification_init(&cfg).expect("init ok");
        let mut bond_dims = purification_bond_dims(cfg.n_sites);

        let d = cfg.d_phys;
        let d_sq = d * d;
        let tau = cfg.beta / (2.0 * cfg.n_trotter_steps as f64);
        let gate = heisenberg_gate_doubled(d, tau, cfg.coupling_j).expect("gate ok");

        // Evolve a few steps.
        for _ in 0..cfg.n_trotter_steps {
            trotter_sweep_doubled(
                &mut mps,
                &mut bond_dims,
                &gate,
                d_sq,
                cfg.chi_max,
                cfg.trunc_tol,
            )
            .expect("sweep ok");
        }

        // Identity operator: 2×2 identity.
        let identity_op = vec![1.0, 0.0, 0.0, 1.0]; // d × d
        for site in 0..cfg.n_sites {
            let exp_val = finite_t_expectation(&mps, &bond_dims, &identity_op, site, cfg.d_phys)
                .expect("expectation ok");
            assert!(
                (exp_val - 1.0).abs() < 1e-10,
                "⟨I⟩ should be 1.0 at site {}, got {}",
                site,
                exp_val
            );
        }
    }

    // ── Test 14: finite_t_expectation_sz ─────────────────────────────────────

    #[test]
    fn finite_t_expectation_sz() {
        // For the Heisenberg chain, ⟨Sz⟩ = 0 by SU(2) symmetry.
        let cfg = small_cfg();
        let mut mps = purification_init(&cfg).expect("init ok");
        let mut bond_dims = purification_bond_dims(cfg.n_sites);

        let d = cfg.d_phys;
        let d_sq = d * d;
        let tau = cfg.beta / (2.0 * cfg.n_trotter_steps as f64);
        let gate = heisenberg_gate_doubled(d, tau, cfg.coupling_j).expect("gate ok");

        for _ in 0..cfg.n_trotter_steps {
            trotter_sweep_doubled(
                &mut mps,
                &mut bond_dims,
                &gate,
                d_sq,
                cfg.chi_max,
                cfg.trunc_tol,
            )
            .expect("sweep ok");
        }

        // Sz operator in spin-1/2 basis |↑⟩,|↓⟩: diag(+1/2, -1/2).
        let sz_op = vec![0.5, 0.0, 0.0, -0.5];
        for site in 0..cfg.n_sites {
            let exp_val = finite_t_expectation(&mps, &bond_dims, &sz_op, site, cfg.d_phys)
                .expect("Sz expectation ok");
            assert!(
                exp_val.abs() < 1e-10,
                "⟨Sz⟩ should be 0 by symmetry at site {}, got {}",
                site,
                exp_val
            );
        }
    }

    // ── Test 15: finite_t_run_energy_decreases_with_beta ─────────────────────

    #[test]
    fn finite_t_run_energy_decreases_with_beta() {
        // Lower temperature (higher β) → lower energy (closer to ground state).
        let cfg_low_beta = FiniteTConfig {
            n_sites: 4,
            beta: 0.5,
            n_trotter_steps: 20,
            chi_max: 16,
            ..small_cfg()
        };
        let cfg_high_beta = FiniteTConfig {
            n_sites: 4,
            beta: 2.0,
            n_trotter_steps: 20,
            chi_max: 16,
            ..small_cfg()
        };

        let result_low = finite_t_run(&cfg_low_beta).expect("low β run ok");
        let result_high = finite_t_run(&cfg_high_beta).expect("high β run ok");

        assert!(
            result_high.energy_per_site < result_low.energy_per_site,
            "E(β=2.0) = {} should be < E(β=0.5) = {} (lower T = lower E)",
            result_high.energy_per_site,
            result_low.energy_per_site
        );
    }
}
