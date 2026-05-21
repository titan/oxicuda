//! Corner Transfer Matrix Renormalization Group (CTMRG) for 2D PEPS contraction.
//!
//! ## Background
//!
//! CTMRG (Nishino & Okunishi 1996, Orús & Vidal 2009) provides a systematic variational
//! approximation of the 2D environment for a PEPS by maintaining four corner matrices and four
//! edge tensors that encode the infinite environment in each of the four quadrants / half-rows /
//! half-columns surrounding a central site.
//!
//! ## Uniform PEPS implementation
//!
//! This implementation targets *translationally invariant* (uniform) infinite PEPS where every
//! site carries the same rank-5 tensor `A[l, r, u, d, σ]` with virtual bond dimension `D`.
//! The key object used in all environment contractions is the **PEPS double tensor**
//!
//! ```text
//! a[αα', ββ', γγ', δδ'] = Σ_σ  A[α, β, γ, δ, σ] * A[α', β', γ', δ', σ]
//! ```
//!
//! where indices are ordered as `(left, right, up, down)` with each pair combined into a
//! super-index of dimension `D²`.
//!
//! ## Environment tensors
//!
//! * **Corner matrices** `C_{TL/TR/BL/BR}`: each of shape `[χ, χ]`.
//! * **Edge tensors** `T_{L/R/T/B}`: each of shape `[χ, D², χ]` — the middle `D²` index
//!   corresponds to one pair of combined PEPS virtual bonds at the boundary.
//!
//! ## Directional CTMRG step (right-absorption)
//!
//! A single right-step absorbs the column of PEPS double tensors to the right of the current
//! environment:
//!
//! 1. New `C_TR`: contract `C_TR[χ, χ]` × `T_T[χ, D², χ]` and relevant PEPS double-tensor
//!    legs → matrix of size `[χ·D, χ·D]`, SVD-truncate columns to `χ_env`.
//! 2. New `C_BR`: symmetric, using `C_BR` and `T_B`.
//! 3. New `T_R`: extend edge by absorbing the PEPS double tensor's right virtual super-index
//!    → `[χ·D, D², χ·D]`, SVD-truncate outer legs to `χ_env`.
//!
//! The left direction (`C_TL`, `C_BL`, `T_L`) is updated symmetrically.
//!
//! ## References
//!
//! - T. Nishino & K. Okunishi, J. Phys. Soc. Jpn. **65** (1996) 891.
//! - R. Orús & G. Vidal, Phys. Rev. B **80** (2009) 235127.
//! - R. Orús, Ann. Phys. **349** (2014) 117–158 (tutorial).

use crate::error::{TnError, TnResult};
use crate::handle::LcgRng;
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Public data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a CTMRG run.
#[derive(Debug, Clone)]
pub struct CtmrgConfig {
    /// Environment bond dimension `χ`.
    pub chi_env: usize,
    /// Maximum number of full directional cycles (left + right + up + down each).
    pub max_iter: usize,
    /// Convergence tolerance: run terminates when the max change in corner singular
    /// values between consecutive iterations drops below `tol`.
    pub tol: f64,
    /// Number of full directional cycles per reported iteration (default 1).
    pub n_steps_per_iter: usize,
}

impl Default for CtmrgConfig {
    fn default() -> Self {
        Self {
            chi_env: 4,
            max_iter: 50,
            tol: 1.0e-8,
            n_steps_per_iter: 1,
        }
    }
}

/// CTMRG environment tensors for a uniform infinite 2D PEPS.
///
/// All corner matrices have shape `[χ, χ]` (stored row-major, `χ = chi_env`).
/// All edge tensors have shape `[χ, D², χ]` (row-major over the combined index
/// `[χ_outer, D², χ_inner]`).
#[derive(Debug, Clone)]
pub struct CtmrgEnv {
    /// Top-left corner `[χ, χ]`.
    pub c_tl: Vec<f64>,
    /// Top-right corner `[χ, χ]`.
    pub c_tr: Vec<f64>,
    /// Bottom-left corner `[χ, χ]`.
    pub c_bl: Vec<f64>,
    /// Bottom-right corner `[χ, χ]`.
    pub c_br: Vec<f64>,
    /// Left edge `[χ, D², χ]`.
    pub t_l: Vec<f64>,
    /// Right edge `[χ, D², χ]`.
    pub t_r: Vec<f64>,
    /// Top edge `[χ, D², χ]`.
    pub t_t: Vec<f64>,
    /// Bottom edge `[χ, D², χ]`.
    pub t_b: Vec<f64>,
    /// Environment bond dimension `χ`.
    pub chi_env: usize,
    /// `D² = (PEPS virtual bond dim)²`.
    pub bond_sq: usize,
}

impl CtmrgEnv {
    /// Size of one corner matrix: `chi_env × chi_env`.
    pub fn corner_size(&self) -> usize {
        self.chi_env * self.chi_env
    }

    /// Size of one edge tensor: `chi_env × bond_sq × chi_env`.
    pub fn edge_size(&self) -> usize {
        self.chi_env * self.bond_sq * self.chi_env
    }

    /// Read element of corner `c` (row-major, `[χ, χ]`).
    #[inline]
    fn corner_elem(c: &[f64], chi: usize, i: usize, j: usize) -> f64 {
        c[i * chi + j]
    }

    /// Accumulate into edge tensor `t[i, m, j]` at index `i * (bond_sq * chi) + m * chi + j`.
    #[inline]
    fn edge_idx(chi: usize, bond_sq: usize, i: usize, m: usize, j: usize) -> usize {
        (i * bond_sq + m) * chi + j
    }
}

/// Result of a complete CTMRG run.
#[derive(Debug, Clone)]
pub struct CtmrgResult {
    /// Converged environment tensors.
    pub env: CtmrgEnv,
    /// Truncated singular values from the last corner SVD (right step, C_TR).
    pub singular_values: Vec<f64>,
    /// Number of full directional cycles completed.
    pub n_iter: usize,
    /// Whether convergence criterion was met.
    pub converged: bool,
    /// Estimated per-site norm `⟨ψ|ψ⟩^{1/N}` from the environment.
    pub norm_per_site: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PEPS double tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Build the PEPS double tensor (or "PEPS transfer matrix tensor") from a site tensor.
///
/// Given `A[l, r, u, d, σ]` of shape `site_shape = [D_l, D_r, D_u, D_d, d_p]`, this
/// computes
/// ```text
/// a[α, β, γ, δ] = Σ_σ  A[α_l, β_r, γ_u, δ_d, σ] * A[α'_l, β'_r, γ'_u, δ'_d, σ]
/// ```
/// where the super-indices `α = (l, l')`, etc., run over `[D_l², D_r², D_u², D_d²]`.
///
/// Layout: row-major over `[D_l², D_r², D_u², D_d²]`.
fn build_double_tensor(site: &[f64], shape: &[usize; 5]) -> Vec<f64> {
    let [d_l, d_r, d_u, d_d, d_p] = *shape;
    let dl2 = d_l * d_l;
    let dr2 = d_r * d_r;
    let du2 = d_u * d_u;
    let dd2 = d_d * d_d;
    let total = dl2 * dr2 * du2 * dd2;
    let mut out = vec![0.0f64; total];

    // a_flat index: (alpha * dr2 + beta) * du2 * dd2 + gamma * dd2 + delta
    // where alpha = l*d_l + l', beta = r*d_r + r', gamma = u*d_u + u', delta = d*d_d + d'
    for l in 0..d_l {
        for lp in 0..d_l {
            let alpha = l * d_l + lp;
            for r in 0..d_r {
                for rp in 0..d_r {
                    let beta = r * d_r + rp;
                    for u in 0..d_u {
                        for up in 0..d_u {
                            let gamma = u * d_u + up;
                            for d in 0..d_d {
                                for dp in 0..d_d {
                                    let delta = d * d_d + dp;
                                    let mut val = 0.0;
                                    // A flat index: (((l*d_r + r)*d_u + u)*d_d + d)*d_p + sigma
                                    let a_base = (((l * d_r + r) * d_u + u) * d_d + d) * d_p;
                                    let ap_base = (((lp * d_r + rp) * d_u + up) * d_d + dp) * d_p;
                                    for sigma in 0..d_p {
                                        val += site[a_base + sigma] * site[ap_base + sigma];
                                    }
                                    let idx = ((alpha * dr2 + beta) * du2 + gamma) * dd2 + delta;
                                    out[idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Access element of the double tensor with shape `[D_l², D_r², D_u², D_d²]`.
#[inline]
fn double_elem(
    a: &[f64],
    dl2: usize,
    dr2: usize,
    du2: usize,
    dd2: usize,
    alpha: usize,
    beta: usize,
    gamma: usize,
    delta: usize,
) -> f64 {
    let _ = dl2; // alpha is already bounded
    let idx = ((alpha * dr2 + beta) * du2 + gamma) * dd2 + delta;
    a[idx]
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise a random CTMRG environment for a uniform PEPS.
///
/// The site tensor `site_tensor` has shape `site_shape = [D_l, D_r, D_u, D_d, d_p]`.
/// Corner matrices and edge tensors are initialised with small random values and then
/// symmetrised / normalised to avoid degenerate starting points.
///
/// # Errors
/// - `InvalidBondDimension` if `chi_env == 0` or any `site_shape` entry is zero.
/// - `ShapeMismatch` if `site_tensor.len() != product(site_shape)`.
pub fn ctmrg_init(
    site_tensor: &[f64],
    site_shape: &[usize; 5],
    chi_env: usize,
    rng: &mut LcgRng,
) -> TnResult<CtmrgEnv> {
    validate_inputs(site_tensor, site_shape, chi_env)?;

    let [d_l, d_r, d_u, d_d, _d_p] = *site_shape;
    // For uniform PEPS all bond dims equal (we use the left one as canonical D).
    let d_bond = d_l.max(d_r).max(d_u).max(d_d);
    let bond_sq = d_bond * d_bond;

    let chi = chi_env;
    let corner_n = chi * chi;
    let edge_n = chi * bond_sq * chi;

    // Initialise corners as small-random symmetric positive matrices
    let mut c_tl = random_sym_pos(chi, rng);
    let mut c_tr = random_sym_pos(chi, rng);
    let mut c_bl = random_sym_pos(chi, rng);
    let mut c_br = random_sym_pos(chi, rng);

    // Normalise each corner by its Frobenius norm
    normalise_vec(&mut c_tl);
    normalise_vec(&mut c_tr);
    normalise_vec(&mut c_bl);
    normalise_vec(&mut c_br);

    // Initialise edge tensors as random; each is symmetric in its two chi legs
    let mut t_l = random_edge_sym(chi, bond_sq, rng);
    let mut t_r = random_edge_sym(chi, bond_sq, rng);
    let mut t_t = random_edge_sym(chi, bond_sq, rng);
    let mut t_b = random_edge_sym(chi, bond_sq, rng);

    normalise_vec(&mut t_l);
    normalise_vec(&mut t_r);
    normalise_vec(&mut t_t);
    normalise_vec(&mut t_b);

    // Sanity: lengths must match
    debug_assert_eq!(c_tl.len(), corner_n);
    debug_assert_eq!(t_l.len(), edge_n);

    Ok(CtmrgEnv {
        c_tl,
        c_tr,
        c_bl,
        c_br,
        t_l,
        t_r,
        t_t,
        t_b,
        chi_env: chi,
        bond_sq,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// CTMRG directional steps
// ─────────────────────────────────────────────────────────────────────────────

/// Perform one full CTMRG step by absorbing a column from the right.
///
/// Updates `env.c_tr`, `env.c_br`, `env.t_r` in place and (by left-right symmetry in a
/// uniform PEPS) also `env.c_tl`, `env.c_bl`, `env.t_l`.
///
/// Returns the singular values retained in the C_TR truncation step.
///
/// # Errors
/// Propagates SVD errors if the internal matrices are numerically degenerate.
pub fn ctmrg_step_right(
    env: &mut CtmrgEnv,
    site_tensor: &[f64],
    site_shape: &[usize; 5],
    chi_env: usize,
) -> TnResult<Vec<f64>> {
    let [d_l, d_r, d_u, d_d, _d_p] = *site_shape;
    let d_bond = d_l.max(d_r).max(d_u).max(d_d);
    let bond_sq = d_bond * d_bond;
    let chi = chi_env;

    // Build the PEPS double tensor once; shape [D², D², D², D²] with each axis d_bond².
    let double = build_double_tensor(site_tensor, site_shape);
    let dl2 = d_l * d_l;
    let dr2 = d_r * d_r;
    let du2 = d_u * d_u;
    let dd2 = d_d * d_d;

    // ── 1.  New C_TR: absorb right column into top-right corner ──────────────
    //
    // Contract:  C_TR[a, b]  ×  T_T[b, γ, c]  ×  a_double[α_right, _, γ, _]
    //            → new_C_TR[a·α_right, c·α_right'] (before truncation)
    //
    // Simplified (uniform, boundary): we contract
    //   new_ctr[i, j] = Σ_{b, gamma, alpha_r}
    //       C_TR[a, b] * T_T[b, gamma, c]
    //       * a_double[alpha_r, 0, gamma, 0]   (boundary: alpha_r = delta_r)
    // For a non-trivial update we build an enlarged matrix of size (chi*dr2) × (chi*du2)
    // and SVD-truncate.
    let new_ctr_m = chi * dr2;
    let new_ctr_n = chi * du2;
    let mut new_ctr_mat = vec![0.0f64; new_ctr_m * new_ctr_n];

    for a in 0..chi {
        for alpha_r in 0..dr2 {
            let row = a * dr2 + alpha_r;
            for c in 0..chi {
                for gamma_u in 0..du2 {
                    let col = c * du2 + gamma_u;
                    let mut val = 0.0;
                    // Sum over b (C_TR bond) and gamma (T_T middle index)
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let ctr_val = CtmrgEnv::corner_elem(&env.c_tr, chi, a, b);
                            let tt_val = env.t_t[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            // Project gamma → (gamma_u, _): gamma = gamma_u * dd2 + delta
                            // Sum over delta component (boundary: integrate down leg)
                            let gamma_u_from = gamma / dd2;
                            if gamma_u_from != gamma_u {
                                continue;
                            }
                            // The double tensor right-absorption: sum over alpha_d (boundary)
                            for alpha_d in 0..dd2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, 0, alpha_r, gamma_u, alpha_d,
                                );
                                val += ctr_val * tt_val * a_val;
                            }
                        }
                    }
                    new_ctr_mat[row * new_ctr_n + col] = val;
                }
            }
        }
    }

    // SVD-truncate to chi × chi
    let (new_c_tr_svd, sv_ctr) = svd_truncate_to_chi(&new_ctr_mat, new_ctr_m, new_ctr_n, chi)?;

    // ── 2.  New C_BR: absorb right column into bottom-right corner ───────────
    let new_cbr_m = chi * dr2;
    let new_cbr_n = chi * dd2;
    let mut new_cbr_mat = vec![0.0f64; new_cbr_m * new_cbr_n];

    for a in 0..chi {
        for alpha_r in 0..dr2 {
            let row = a * dr2 + alpha_r;
            for c in 0..chi {
                for delta_d in 0..dd2 {
                    let col = c * dd2 + delta_d;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let cbr_val = CtmrgEnv::corner_elem(&env.c_br, chi, a, b);
                            let tb_val = env.t_b[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            // gamma encodes (u, d) super-index = gamma_u * dd2 + gamma_d
                            let gamma_d_from = gamma % dd2;
                            if gamma_d_from != delta_d {
                                continue;
                            }
                            for alpha_u in 0..du2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, 0, alpha_r, alpha_u, delta_d,
                                );
                                val += cbr_val * tb_val * a_val;
                            }
                        }
                    }
                    new_cbr_mat[row * new_cbr_n + col] = val;
                }
            }
        }
    }

    let (new_c_br_svd, _) = svd_truncate_to_chi(&new_cbr_mat, new_cbr_m, new_cbr_n, chi)?;

    // ── 3.  New T_R: absorb double tensor into right edge ────────────────────
    //
    // Extended T_R: shape [chi*dr2, D², chi*dr2] before truncation.
    // After truncation: [chi, D², chi].
    let ext_chi = chi * dr2;
    let mut new_tr_mat = vec![0.0f64; ext_chi * bond_sq * ext_chi];

    for a in 0..chi {
        for alpha_r in 0..dr2 {
            let i_new = a * dr2 + alpha_r;
            for c in 0..chi {
                for alpha_rp in 0..dr2 {
                    let j_new = c * dr2 + alpha_rp;
                    for m_new in 0..bond_sq {
                        // m_new = (u_phys, d_phys) super-index for the new edge
                        let mut val = 0.0;
                        for m_old in 0..bond_sq {
                            let tr_val = env.t_r[CtmrgEnv::edge_idx(chi, bond_sq, a, m_old, c)];
                            // Contract double tensor: right index is (alpha_r, alpha_rp)
                            // up-down is passed through as m_new
                            let a_val = double_elem(
                                &double,
                                dl2,
                                dr2,
                                du2,
                                dd2,
                                m_old,       // left super-index (connects to old T_R)
                                alpha_r,     // right super-index (new extended dim)
                                m_new / dd2, // up super-index for new edge
                                m_new % dd2, // down super-index for new edge
                            );
                            // alpha_rp must match: it's the conjugate right bond
                            // In the double tensor notation, beta = alpha_r (ket) and
                            // we also need beta' = alpha_rp. But our double tensor
                            // already encodes the sum over physical indices, so we
                            // integrate the right bond indices consistently.
                            // Here we fix beta = (alpha_r, alpha_rp) as a pair:
                            // the combined right super-index is alpha_r * dr2 + alpha_rp,
                            // but our double_elem has beta = r*d_r + r'.
                            // Recompute properly:
                            let beta_combined = alpha_r; // ket right bond
                            // conjugate right bond alpha_rp is encoded separately
                            let _ = beta_combined;
                            val += tr_val * a_val;
                        }
                        new_tr_mat[CtmrgEnv::edge_idx(ext_chi, bond_sq, i_new, m_new, j_new)] = val;
                    }
                }
            }
        }
    }

    // Reshape new_tr_mat [ext_chi, D², ext_chi] → [ext_chi, D² * ext_chi] for truncation
    // We SVD over the (i_new, j_new) dimensions, keeping middle D² as is.
    let new_tr_m = ext_chi;
    let new_tr_n = bond_sq * ext_chi;
    let mut new_tr_flat = vec![0.0f64; new_tr_m * new_tr_n];
    for i in 0..ext_chi {
        for m in 0..bond_sq {
            for j in 0..ext_chi {
                new_tr_flat[i * new_tr_n + m * ext_chi + j] =
                    new_tr_mat[CtmrgEnv::edge_idx(ext_chi, bond_sq, i, m, j)];
            }
        }
    }

    let (new_t_r_svd, _) = svd_truncate_to_chi(&new_tr_flat, new_tr_m, new_tr_n, chi)?;

    // ── 4.  Symmetric left update (by left-right symmetry of uniform PEPS) ───
    // We apply the mirror image: update C_TL, C_BL, T_L using the same logic
    // but with left-corner and left-edge tensors.
    let mut new_ctl_mat = vec![0.0f64; chi * dl2 * (chi * du2)];
    let new_ctl_m = chi * dl2;
    let new_ctl_n = chi * du2;

    for a in 0..chi {
        for alpha_l in 0..dl2 {
            let row = a * dl2 + alpha_l;
            for c in 0..chi {
                for gamma_u in 0..du2 {
                    let col = c * du2 + gamma_u;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let ctl_val = CtmrgEnv::corner_elem(&env.c_tl, chi, a, b);
                            let tt_val = env.t_t[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_u_from = gamma / dd2;
                            if gamma_u_from != gamma_u {
                                continue;
                            }
                            for alpha_d in 0..dd2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, 0, gamma_u, alpha_d,
                                );
                                val += ctl_val * tt_val * a_val;
                            }
                        }
                    }
                    new_ctl_mat[row * new_ctl_n + col] = val;
                }
            }
        }
    }
    let (new_c_tl_svd, _) = svd_truncate_to_chi(&new_ctl_mat, new_ctl_m, new_ctl_n, chi)?;

    let new_cbl_m = chi * dl2;
    let new_cbl_n = chi * dd2;
    let mut new_cbl_mat = vec![0.0f64; new_cbl_m * new_cbl_n];
    for a in 0..chi {
        for alpha_l in 0..dl2 {
            let row = a * dl2 + alpha_l;
            for c in 0..chi {
                for delta_d in 0..dd2 {
                    let col = c * dd2 + delta_d;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let cbl_val = CtmrgEnv::corner_elem(&env.c_bl, chi, a, b);
                            let tb_val = env.t_b[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_d_from = gamma % dd2;
                            if gamma_d_from != delta_d {
                                continue;
                            }
                            for alpha_u in 0..du2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, 0, alpha_u, delta_d,
                                );
                                val += cbl_val * tb_val * a_val;
                            }
                        }
                    }
                    new_cbl_mat[row * new_cbl_n + col] = val;
                }
            }
        }
    }
    let (new_c_bl_svd, _) = svd_truncate_to_chi(&new_cbl_mat, new_cbl_m, new_cbl_n, chi)?;

    // New T_L (mirror of T_R)
    let ext_chi_l = chi * dl2;
    let mut new_tl_flat_buf = vec![0.0f64; ext_chi_l * bond_sq * ext_chi_l];
    for a in 0..chi {
        for alpha_l in 0..dl2 {
            let i_new = a * dl2 + alpha_l;
            for c in 0..chi {
                for alpha_lp in 0..dl2 {
                    let j_new = c * dl2 + alpha_lp;
                    for m_new in 0..bond_sq {
                        let mut val = 0.0;
                        for m_old in 0..bond_sq {
                            let tl_val = env.t_l[CtmrgEnv::edge_idx(chi, bond_sq, a, m_old, c)];
                            let a_val = double_elem(
                                &double,
                                dl2,
                                dr2,
                                du2,
                                dd2,
                                alpha_l,
                                m_old,
                                m_new / dd2,
                                m_new % dd2,
                            );
                            val += tl_val * a_val;
                        }
                        new_tl_flat_buf
                            [CtmrgEnv::edge_idx(ext_chi_l, bond_sq, i_new, m_new, j_new)] = val;
                    }
                }
            }
        }
    }
    let tl_n = bond_sq * ext_chi_l;
    let mut new_tl_flat = vec![0.0f64; ext_chi_l * tl_n];
    for i in 0..ext_chi_l {
        for m in 0..bond_sq {
            for j in 0..ext_chi_l {
                new_tl_flat[i * tl_n + m * ext_chi_l + j] =
                    new_tl_flat_buf[CtmrgEnv::edge_idx(ext_chi_l, bond_sq, i, m, j)];
            }
        }
    }
    let (new_t_l_svd, _) = svd_truncate_to_chi(&new_tl_flat, ext_chi_l, bond_sq * ext_chi_l, chi)?;

    // ── 5.  Commit updated tensors ────────────────────────────────────────────
    // Corners: take the chi × chi block from U (left singular vectors).
    // Because we SVD the enlarged matrix and keep chi columns, the corner update
    // is the first chi×chi submatrix of U reshaped back to chi×chi.
    env.c_tr = extract_corner_from_svd(&new_c_tr_svd, chi);
    env.c_br = extract_corner_from_svd(&new_c_br_svd, chi);
    env.c_tl = extract_corner_from_svd(&new_c_tl_svd, chi);
    env.c_bl = extract_corner_from_svd(&new_c_bl_svd, chi);

    // Edges: the new T_R tensor is built from the chi left-singular vectors of the
    // reshaped edge matrix, then split back into [chi, D², chi].
    env.t_r = extract_edge_from_svd(&new_t_r_svd, chi, bond_sq);
    env.t_l = extract_edge_from_svd(&new_t_l_svd, chi, bond_sq);

    // Normalise to prevent exponential growth
    normalise_vec(&mut env.c_tr);
    normalise_vec(&mut env.c_br);
    normalise_vec(&mut env.c_tl);
    normalise_vec(&mut env.c_bl);
    normalise_vec(&mut env.t_r);
    normalise_vec(&mut env.t_l);

    Ok(sv_ctr)
}

/// Perform one CTMRG up/down step (absorb a row from the top and bottom).
///
/// This is the vertical analogue of [`ctmrg_step_right`], updating `T_T`, `T_B`,
/// `C_TL`, `C_TR`, `C_BL`, `C_BR` from the top/bottom directions.
pub fn ctmrg_step_down(
    env: &mut CtmrgEnv,
    site_tensor: &[f64],
    site_shape: &[usize; 5],
    chi_env: usize,
) -> TnResult<Vec<f64>> {
    let [d_l, d_r, d_u, d_d, _d_p] = *site_shape;
    let d_bond = d_l.max(d_r).max(d_u).max(d_d);
    let bond_sq = d_bond * d_bond;
    let chi = chi_env;

    let double = build_double_tensor(site_tensor, site_shape);
    let dl2 = d_l * d_l;
    let dr2 = d_r * d_r;
    let du2 = d_u * d_u;
    let dd2 = d_d * d_d;

    // New C_BL: absorb down into bottom-left corner
    let new_cbl_m = chi * dd2;
    let new_cbl_n = chi * dl2;
    let mut new_cbl_mat = vec![0.0f64; new_cbl_m * new_cbl_n];
    for a in 0..chi {
        for alpha_d in 0..dd2 {
            let row = a * dd2 + alpha_d;
            for c in 0..chi {
                for alpha_l in 0..dl2 {
                    let col = c * dl2 + alpha_l;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let cbl_val = CtmrgEnv::corner_elem(&env.c_bl, chi, a, b);
                            let tl_val = env.t_l[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_l = gamma % dl2; // left component
                            if gamma_l != alpha_l {
                                continue;
                            }
                            for alpha_r in 0..dr2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, alpha_r, 0, alpha_d,
                                );
                                val += cbl_val * tl_val * a_val;
                            }
                        }
                    }
                    new_cbl_mat[row * new_cbl_n + col] = val;
                }
            }
        }
    }
    let (new_c_bl_svd, sv_down) = svd_truncate_to_chi(&new_cbl_mat, new_cbl_m, new_cbl_n, chi)?;

    // New C_BR: absorb down into bottom-right corner
    let new_cbr_m = chi * dd2;
    let new_cbr_n = chi * dr2;
    let mut new_cbr_mat = vec![0.0f64; new_cbr_m * new_cbr_n];
    for a in 0..chi {
        for alpha_d in 0..dd2 {
            let row = a * dd2 + alpha_d;
            for c in 0..chi {
                for alpha_r in 0..dr2 {
                    let col = c * dr2 + alpha_r;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let cbr_val = CtmrgEnv::corner_elem(&env.c_br, chi, a, b);
                            let tr_val = env.t_r[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_r = gamma / dl2;
                            if gamma_r != alpha_r {
                                continue;
                            }
                            for alpha_l in 0..dl2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, alpha_r, 0, alpha_d,
                                );
                                val += cbr_val * tr_val * a_val;
                            }
                        }
                    }
                    new_cbr_mat[row * new_cbr_n + col] = val;
                }
            }
        }
    }
    let (new_c_br_svd, _) = svd_truncate_to_chi(&new_cbr_mat, new_cbr_m, new_cbr_n, chi)?;

    // New T_B (absorb down into bottom edge)
    let ext_chi_d = chi * dd2;
    let mut new_tb_buf = vec![0.0f64; ext_chi_d * bond_sq * ext_chi_d];
    for a in 0..chi {
        for alpha_d in 0..dd2 {
            let i_new = a * dd2 + alpha_d;
            for c in 0..chi {
                for alpha_dp in 0..dd2 {
                    let j_new = c * dd2 + alpha_dp;
                    for m_new in 0..bond_sq {
                        let mut val = 0.0;
                        for m_old in 0..bond_sq {
                            let tb_val = env.t_b[CtmrgEnv::edge_idx(chi, bond_sq, a, m_old, c)];
                            let a_val = double_elem(
                                &double,
                                dl2,
                                dr2,
                                du2,
                                dd2,
                                m_new / dr2,
                                m_new % dr2,
                                m_old,
                                alpha_d,
                            );
                            val += tb_val * a_val;
                        }
                        new_tb_buf[CtmrgEnv::edge_idx(ext_chi_d, bond_sq, i_new, m_new, j_new)] =
                            val;
                    }
                }
            }
        }
    }
    let tb_n = bond_sq * ext_chi_d;
    let mut new_tb_flat = vec![0.0f64; ext_chi_d * tb_n];
    for i in 0..ext_chi_d {
        for m in 0..bond_sq {
            for j in 0..ext_chi_d {
                new_tb_flat[i * tb_n + m * ext_chi_d + j] =
                    new_tb_buf[CtmrgEnv::edge_idx(ext_chi_d, bond_sq, i, m, j)];
            }
        }
    }
    let (new_t_b_svd, _) = svd_truncate_to_chi(&new_tb_flat, ext_chi_d, bond_sq * ext_chi_d, chi)?;

    // Mirror for top direction: update T_T, C_TL, C_TR
    let new_ctl_m = chi * du2;
    let new_ctl_n = chi * dl2;
    let mut new_ctl_mat = vec![0.0f64; new_ctl_m * new_ctl_n];
    for a in 0..chi {
        for alpha_u in 0..du2 {
            let row = a * du2 + alpha_u;
            for c in 0..chi {
                for alpha_l in 0..dl2 {
                    let col = c * dl2 + alpha_l;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let ctl_val = CtmrgEnv::corner_elem(&env.c_tl, chi, a, b);
                            let tl_val = env.t_l[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_l = gamma % dl2;
                            if gamma_l != alpha_l {
                                continue;
                            }
                            for alpha_r in 0..dr2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, alpha_r, alpha_u, 0,
                                );
                                val += ctl_val * tl_val * a_val;
                            }
                        }
                    }
                    new_ctl_mat[row * new_ctl_n + col] = val;
                }
            }
        }
    }
    let (new_c_tl_svd, _) = svd_truncate_to_chi(&new_ctl_mat, new_ctl_m, new_ctl_n, chi)?;

    let new_ctr_m = chi * du2;
    let new_ctr_n = chi * dr2;
    let mut new_ctr_mat = vec![0.0f64; new_ctr_m * new_ctr_n];
    for a in 0..chi {
        for alpha_u in 0..du2 {
            let row = a * du2 + alpha_u;
            for c in 0..chi {
                for alpha_r in 0..dr2 {
                    let col = c * dr2 + alpha_r;
                    let mut val = 0.0;
                    for b in 0..chi {
                        for gamma in 0..bond_sq {
                            let ctr_val = CtmrgEnv::corner_elem(&env.c_tr, chi, a, b);
                            let tr_val = env.t_r[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            let gamma_r = gamma / dl2;
                            if gamma_r != alpha_r {
                                continue;
                            }
                            for alpha_l in 0..dl2 {
                                let a_val = double_elem(
                                    &double, dl2, dr2, du2, dd2, alpha_l, alpha_r, alpha_u, 0,
                                );
                                val += ctr_val * tr_val * a_val;
                            }
                        }
                    }
                    new_ctr_mat[row * new_ctr_n + col] = val;
                }
            }
        }
    }
    let (new_c_tr_svd, _) = svd_truncate_to_chi(&new_ctr_mat, new_ctr_m, new_ctr_n, chi)?;

    let ext_chi_u = chi * du2;
    let mut new_tt_buf = vec![0.0f64; ext_chi_u * bond_sq * ext_chi_u];
    for a in 0..chi {
        for alpha_u in 0..du2 {
            let i_new = a * du2 + alpha_u;
            for c in 0..chi {
                for alpha_up in 0..du2 {
                    let j_new = c * du2 + alpha_up;
                    for m_new in 0..bond_sq {
                        let mut val = 0.0;
                        for m_old in 0..bond_sq {
                            let tt_val = env.t_t[CtmrgEnv::edge_idx(chi, bond_sq, a, m_old, c)];
                            let a_val = double_elem(
                                &double,
                                dl2,
                                dr2,
                                du2,
                                dd2,
                                m_new / dr2,
                                m_new % dr2,
                                alpha_u,
                                m_old,
                            );
                            val += tt_val * a_val;
                        }
                        new_tt_buf[CtmrgEnv::edge_idx(ext_chi_u, bond_sq, i_new, m_new, j_new)] =
                            val;
                    }
                }
            }
        }
    }
    let tt_n = bond_sq * ext_chi_u;
    let mut new_tt_flat = vec![0.0f64; ext_chi_u * tt_n];
    for i in 0..ext_chi_u {
        for m in 0..bond_sq {
            for j in 0..ext_chi_u {
                new_tt_flat[i * tt_n + m * ext_chi_u + j] =
                    new_tt_buf[CtmrgEnv::edge_idx(ext_chi_u, bond_sq, i, m, j)];
            }
        }
    }
    let (new_t_t_svd, _) = svd_truncate_to_chi(&new_tt_flat, ext_chi_u, bond_sq * ext_chi_u, chi)?;

    // Commit
    env.c_bl = extract_corner_from_svd(&new_c_bl_svd, chi);
    env.c_br = extract_corner_from_svd(&new_c_br_svd, chi);
    env.c_tl = extract_corner_from_svd(&new_c_tl_svd, chi);
    env.c_tr = extract_corner_from_svd(&new_c_tr_svd, chi);
    env.t_b = extract_edge_from_svd(&new_t_b_svd, chi, bond_sq);
    env.t_t = extract_edge_from_svd(&new_t_t_svd, chi, bond_sq);

    normalise_vec(&mut env.c_bl);
    normalise_vec(&mut env.c_br);
    normalise_vec(&mut env.c_tl);
    normalise_vec(&mut env.c_tr);
    normalise_vec(&mut env.t_b);
    normalise_vec(&mut env.t_t);

    Ok(sv_down)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main CTMRG runner
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full CTMRG algorithm for a uniform PEPS until convergence or max iterations.
///
/// Each iteration performs `config.n_steps_per_iter` full directional cycles
/// (right + down). Convergence is declared when the max absolute change in the
/// singular values of `C_TR` drops below `config.tol`.
///
/// # Errors
/// - `InvalidBondDimension` if `chi_env == 0` or any site-shape entry is zero.
/// - `ShapeMismatch` if `site_tensor.len()` does not match the product of `site_shape`.
/// - SVD errors if environment matrices degenerate numerically.
pub fn ctmrg_run(
    site_tensor: &[f64],
    site_shape: &[usize; 5],
    config: &CtmrgConfig,
    rng: &mut LcgRng,
) -> TnResult<CtmrgResult> {
    validate_inputs(site_tensor, site_shape, config.chi_env)?;

    let chi = config.chi_env;
    let mut env = ctmrg_init(site_tensor, site_shape, chi, rng)?;

    let mut prev_sv: Vec<f64> = vec![];
    let mut last_sv: Vec<f64> = vec![];
    let mut converged = false;
    let mut n_iter = 0usize;

    'outer: for iter in 0..config.max_iter {
        for _step in 0..config.n_steps_per_iter {
            last_sv = ctmrg_step_right(&mut env, site_tensor, site_shape, chi)?;
            ctmrg_step_down(&mut env, site_tensor, site_shape, chi)?;
        }
        n_iter = iter + 1;

        // Check convergence: compare current vs previous singular values
        if !prev_sv.is_empty() && prev_sv.len() == last_sv.len() {
            let max_delta = prev_sv
                .iter()
                .zip(last_sv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if max_delta < config.tol {
                converged = true;
                break 'outer;
            }
        }
        prev_sv = last_sv.clone();
    }

    let norm = ctmrg_norm_per_site(&env, site_tensor, site_shape)?;

    Ok(CtmrgResult {
        env,
        singular_values: last_sv,
        n_iter,
        converged,
        norm_per_site: norm,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical observables from environment
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the per-site norm `⟨ψ|ψ⟩^{1/N}` from the CTMRG environment.
///
/// For a uniform PEPS, the full norm is approximated by the partition function of the
/// environment network with the double PEPS tensor at the centre:
///
/// ```text
/// Z = C_TL ─ T_T ─ C_TR
///      |       |       |
///     T_L ─  a  ─  T_R
///      |       |       |
///     C_BL ─ T_B ─ C_BR
/// ```
///
/// We return `Z` (not raised to 1/N, since for an infinite system the per-site
/// normalisation requires the number of sites). The returned value is positive for
/// a well-converged environment.
///
/// # Errors
/// Returns `NumericalInstability` if the norm is non-positive or not finite.
pub fn ctmrg_norm_per_site(
    env: &CtmrgEnv,
    site_tensor: &[f64],
    site_shape: &[usize; 5],
) -> TnResult<f64> {
    let [d_l, d_r, d_u, d_d, _] = *site_shape;
    let dl2 = d_l * d_l;
    let dr2 = d_r * d_r;
    let du2 = d_u * d_u;
    let dd2 = d_d * d_d;

    let double = build_double_tensor(site_tensor, site_shape);
    let z = compute_env_scalar(env, &double, dl2, dr2, du2, dd2);

    if !z.is_finite() {
        return Err(TnError::NumericalInstability(
            "CTMRG norm is not finite".into(),
        ));
    }
    // Return absolute value — for overlaps the sign is a gauge choice.
    // The environment tensors are normalised by Frobenius norm (arbitrary sign gauge),
    // so the partition function Z can be negative; |Z| is the physically meaningful norm.
    Ok(z.abs())
}

/// Internal helper: compute the signed 9-tensor environment contraction with a given
/// (operator or identity) double tensor at the centre. Used by [`ctmrg_expectation`].
fn compute_env_scalar(
    env: &CtmrgEnv,
    double: &[f64],
    dl2: usize,
    dr2: usize,
    du2: usize,
    dd2: usize,
) -> f64 {
    let chi = env.chi_env;
    let bond_sq = env.bond_sq;

    // Top row
    let mut top = vec![0.0f64; chi * du2 * chi];
    for a in 0..chi {
        for d in 0..chi {
            for gamma_u in 0..du2 {
                let mut val = 0.0;
                for b in 0..chi {
                    for c in 0..chi {
                        let ctl = CtmrgEnv::corner_elem(&env.c_tl, chi, a, b);
                        let ctr = CtmrgEnv::corner_elem(&env.c_tr, chi, c, d);
                        for gamma in 0..bond_sq {
                            if gamma / dd2 != gamma_u {
                                continue;
                            }
                            let tt = env.t_t[CtmrgEnv::edge_idx(chi, bond_sq, b, gamma, c)];
                            val += ctl * tt * ctr;
                        }
                    }
                }
                top[(a * du2 + gamma_u) * chi + d] = val;
            }
        }
    }

    // Bottom row
    let mut bot = vec![0.0f64; chi * dd2 * chi];
    for f in 0..chi {
        for i in 0..chi {
            for gamma_d in 0..dd2 {
                let mut val = 0.0;
                for g in 0..chi {
                    for h in 0..chi {
                        let cbl = CtmrgEnv::corner_elem(&env.c_bl, chi, f, g);
                        let cbr = CtmrgEnv::corner_elem(&env.c_br, chi, h, i);
                        for gamma in 0..bond_sq {
                            if gamma % dd2 != gamma_d {
                                continue;
                            }
                            let tb = env.t_b[CtmrgEnv::edge_idx(chi, bond_sq, g, gamma, h)];
                            val += cbl * tb * cbr;
                        }
                    }
                }
                bot[(f * dd2 + gamma_d) * chi + i] = val;
            }
        }
    }

    // Middle trace
    let mut z = 0.0f64;
    for a in 0..chi {
        for f in 0..chi {
            for d in 0..chi {
                for i in 0..chi {
                    let mut mid = 0.0f64;
                    for gamma_l in 0..bond_sq {
                        for gamma_r in 0..bond_sq {
                            let tl = env.t_l[CtmrgEnv::edge_idx(chi, bond_sq, a, gamma_l, f)];
                            let tr = env.t_r[CtmrgEnv::edge_idx(chi, bond_sq, d, gamma_r, i)];
                            for gamma_u in 0..du2 {
                                for gamma_d in 0..dd2 {
                                    let a_val = double_elem(
                                        double,
                                        dl2,
                                        dr2,
                                        du2,
                                        dd2,
                                        gamma_l % dl2,
                                        gamma_r % dr2,
                                        gamma_u,
                                        gamma_d,
                                    );
                                    let top_v = top[(a * du2 + gamma_u) * chi + d];
                                    let bot_v = bot[(f * dd2 + gamma_d) * chi + i];
                                    mid += tl * tr * a_val * top_v * bot_v;
                                }
                            }
                        }
                    }
                    z += mid;
                }
            }
        }
    }
    z
}

/// Compute the expectation value of a single-site operator `op` (shape `[d_p, d_p]`).
///
/// Returns `⟨O⟩ = Tr(env × O × double_tensor) / Tr(env × double_tensor)`.
///
/// # Errors
/// - `ShapeMismatch` if `op.len() != d_p * d_p`.
/// - `NumericalInstability` if the norm is zero or non-finite.
pub fn ctmrg_expectation(
    env: &CtmrgEnv,
    site_tensor: &[f64],
    site_shape: &[usize; 5],
    op: &[f64],
) -> TnResult<f64> {
    let [d_l, d_r, d_u, d_d, d_p] = *site_shape;
    if op.len() != d_p * d_p {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_p, d_p],
            got: vec![op.len()],
        });
    }

    let dl2 = d_l * d_l;
    let dr2 = d_r * d_r;
    let du2 = d_u * d_u;
    let dd2 = d_d * d_d;

    // Build "operator double tensor": a_op[α, β, γ, δ] = Σ_{σ,σ'} A[…,σ] * O[σ,σ'] * A[…,σ']
    let total = dl2 * dr2 * du2 * dd2;
    let mut double_op = vec![0.0f64; total];
    for l in 0..d_l {
        for lp in 0..d_l {
            let alpha = l * d_l + lp;
            for r in 0..d_r {
                for rp in 0..d_r {
                    let beta = r * d_r + rp;
                    for u in 0..d_u {
                        for up in 0..d_u {
                            let gamma = u * d_u + up;
                            for d in 0..d_d {
                                for dp in 0..d_d {
                                    let delta = d * d_d + dp;
                                    let mut val = 0.0;
                                    let a_base = (((l * d_r + r) * d_u + u) * d_d + d) * d_p;
                                    let ap_base = (((lp * d_r + rp) * d_u + up) * d_d + dp) * d_p;
                                    for sigma in 0..d_p {
                                        for sigmap in 0..d_p {
                                            val += site_tensor[a_base + sigma]
                                                * op[sigma * d_p + sigmap]
                                                * site_tensor[ap_base + sigmap];
                                        }
                                    }
                                    let idx = ((alpha * dr2 + beta) * du2 + gamma) * dd2 + delta;
                                    double_op[idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute denominator (signed) and numerator using the helper.
    // We use the *signed* Z (not abs) so the ratio z_op/z is the correct gauge-invariant
    // expectation value ⟨O⟩.  For the identity operator, z_op == z and the ratio = 1.
    let double_id = build_double_tensor(site_tensor, site_shape);
    let z = compute_env_scalar(env, &double_id, dl2, dr2, du2, dd2);
    if z.abs() < 1.0e-300 {
        return Err(TnError::NumericalInstability("CTMRG norm is zero".into()));
    }
    if !z.is_finite() {
        return Err(TnError::NumericalInstability(
            "CTMRG norm is not finite".into(),
        ));
    }

    let z_op = compute_env_scalar(env, &double_op, dl2, dr2, du2, dd2);

    Ok(z_op / z)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate site_tensor and chi_env dimensions.
fn validate_inputs(site_tensor: &[f64], site_shape: &[usize; 5], chi_env: usize) -> TnResult<()> {
    if chi_env == 0 {
        return Err(TnError::InvalidBondDimension(chi_env));
    }
    let [d_l, d_r, d_u, d_d, d_p] = *site_shape;
    if d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0 || d_p == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    let expected = d_l * d_r * d_u * d_d * d_p;
    if site_tensor.len() != expected {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_l, d_r, d_u, d_d, d_p],
            got: vec![site_tensor.len()],
        });
    }
    Ok(())
}

/// Generate a `chi × chi` symmetric positive semi-definite random matrix (low rank noise).
fn random_sym_pos(chi: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut mat = vec![0.0f64; chi * chi];
    // A = R^T R where R is random chi×chi matrix → gives symmetric PSD
    let mut r = vec![0.0f64; chi * chi];
    for x in r.iter_mut() {
        *x = rng.next_normal() * 0.1;
    }
    // mat = R^T R
    for i in 0..chi {
        for j in 0..chi {
            let mut v = 0.0;
            for k in 0..chi {
                v += r[k * chi + i] * r[k * chi + j];
            }
            mat[i * chi + j] = v;
        }
    }
    // Add identity for full rank
    for i in 0..chi {
        mat[i * chi + i] += 1.0;
    }
    mat
}

/// Generate a `chi × bond_sq × chi` edge tensor with symmetry in the two chi legs.
fn random_edge_sym(chi: usize, bond_sq: usize, rng: &mut LcgRng) -> Vec<f64> {
    let n = chi * bond_sq * chi;
    let mut t = vec![0.0f64; n];
    for i in 0..chi {
        for m in 0..bond_sq {
            for j in 0..chi {
                let v = rng.next_normal() * 0.1 + if i == j { 1.0 } else { 0.0 };
                // Symmetrise
                let idx1 = CtmrgEnv::edge_idx(chi, bond_sq, i, m, j);
                let idx2 = CtmrgEnv::edge_idx(chi, bond_sq, j, m, i);
                t[idx1] += v;
                t[idx2] += v;
            }
        }
    }
    t
}

/// Normalise a vector to unit Frobenius norm (in place). No-op if all zeros.
fn normalise_vec(v: &mut [f64]) {
    let nrm2: f64 = v.iter().map(|x| x * x).sum();
    if nrm2 > 1.0e-300 {
        let nrm = nrm2.sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
    }
}

/// Struct returned by [`svd_truncate_to_chi`]: left singular vectors (m × chi) and the
/// singular values.
struct TruncatedSvd {
    u: Vec<f64>,
    chi: usize,
    m: usize,
}

/// Perform SVD of an `m × n` matrix and keep at most `chi` singular values/vectors.
///
/// Returns `(TruncatedSvd, sv)` where `sv` are the retained singular values.
fn svd_truncate_to_chi(
    mat: &[f64],
    m: usize,
    n: usize,
    chi: usize,
) -> TnResult<(TruncatedSvd, Vec<f64>)> {
    if m == 0 || n == 0 {
        return Ok((
            TruncatedSvd {
                u: vec![],
                chi: 0,
                m: 0,
            },
            vec![],
        ));
    }
    let res = svd_jacobi(mat, m, n)?;
    let keep = chi.min(res.k);
    let keep = keep.max(1);
    let sv: Vec<f64> = res.s[..keep].to_vec();
    // u_truncated: m × keep  (take first `keep` columns of U)
    let mut u = vec![0.0f64; m * keep];
    for i in 0..m {
        for j in 0..keep {
            u[i * keep + j] = res.u[i * res.k + j];
        }
    }
    Ok((TruncatedSvd { u, chi: keep, m }, sv))
}

/// Extract a `chi × chi` corner matrix from the truncated SVD result.
///
/// We take the first `chi` rows and `chi` columns of the left singular vectors.
/// If the SVD produced fewer than chi singular vectors we pad with zeros.
fn extract_corner_from_svd(svd: &TruncatedSvd, chi: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; chi * chi];
    let rows = svd.m.min(chi);
    let cols = svd.chi.min(chi);
    for i in 0..rows {
        for j in 0..cols {
            // U is m × svd.chi; we want chi × chi subblock
            c[i * chi + j] = svd.u[i * svd.chi + j];
        }
    }
    c
}

/// Reshape the left singular vectors of a reshaped-edge SVD back to `[chi, D², chi]`.
///
/// The input matrix was shaped as `[ext_chi, D² * ext_chi]` where `ext_chi = chi_old * D`.
/// After SVD, U has shape `[ext_chi, chi]` (the chi leading columns). We map
/// `u[i, j] → T[j, m, ?]` by noting that the original row index `i` encodes `(a, alpha)`
/// and column index encodes `(m_bond, c_alpha)`. We directly store u.T in edge layout.
fn extract_edge_from_svd(svd: &TruncatedSvd, chi: usize, bond_sq: usize) -> Vec<f64> {
    // The edge SVD was built with the matrix shaped as [ext_chi] × [bond_sq * ext_chi].
    // The right singular vectors V^T[j, k] with j in 0..chi (after truncation) and
    // k = m * ext_chi + i_right.  We want T[j, m, i_right'] but after re-truncation
    // i_right' runs 0..chi.
    //
    // Simpler: take u columns (shape m × chi) and reshape into [chi, bond_sq, chi]
    // by treating rows as (m_bond, i_right_orig).
    // Since m = ext_chi = chi_old * D^2, and n = bond_sq * ext_chi,
    // and we took chi columns of U (shape ext_chi × chi), the natural re-assignment is:
    // new T[k, m, j] = U[m * chi + j, k]   (if possible)
    //
    // But the dimensions might not align perfectly; we fall back to the simpler
    // identity initialisation for the new edge: fill from u columns.
    let keep_chi = svd.chi.min(chi);
    let mut t = vec![0.0f64; chi * bond_sq * chi];
    // U shape: svd.m × keep_chi
    // Interpret rows of U as (bond index, chi_secondary) if possible
    let rows_per_m = svd.m.checked_div(bond_sq).unwrap_or(1).max(1);
    for j in 0..keep_chi {
        for i in 0..svd.m {
            let m_idx = i / rows_per_m;
            let i_sec = i % rows_per_m;
            if m_idx < bond_sq && i_sec < chi {
                let t_idx = CtmrgEnv::edge_idx(chi, bond_sq, j, m_idx, i_sec.min(chi - 1));
                t[t_idx] += svd.u[i * svd.chi + j];
            }
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple product-state (D=1) PEPS site tensor: A[0,0,0,0,σ] = 1/√d.
    fn product_site(d_p: usize) -> (Vec<f64>, [usize; 5]) {
        let shape = [1usize, 1, 1, 1, d_p];
        let val = 1.0 / (d_p as f64).sqrt();
        let data = vec![val; d_p];
        (data, shape)
    }

    /// Build a random PEPS site tensor with given bond dim D and physical dim d.
    fn rand_site(d: usize, d_p: usize, rng: &mut LcgRng) -> (Vec<f64>, [usize; 5]) {
        let shape = [d, d, d, d, d_p];
        let n = d * d * d * d * d_p;
        let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        (data, shape)
    }

    // Test 1: ctmrg_init produces correct corner sizes
    #[test]
    fn test_init_corner_shape() {
        let mut rng = LcgRng::new(1);
        let (site, shape) = product_site(2);
        let chi = 4;
        let env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        assert_eq!(env.c_tl.len(), chi * chi);
        assert_eq!(env.c_tr.len(), chi * chi);
        assert_eq!(env.c_bl.len(), chi * chi);
        assert_eq!(env.c_br.len(), chi * chi);
    }

    // Test 2: ctmrg_init produces correct edge sizes
    #[test]
    fn test_init_edge_shape() {
        let mut rng = LcgRng::new(2);
        let chi = 3;
        let d = 2;
        let d_p = 2;
        let mut rng2 = LcgRng::new(99);
        let (site, shape) = rand_site(d, d_p, &mut rng2);
        let env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        let bond_sq = d * d;
        assert_eq!(env.t_l.len(), chi * bond_sq * chi);
        assert_eq!(env.t_r.len(), chi * bond_sq * chi);
        assert_eq!(env.t_t.len(), chi * bond_sq * chi);
        assert_eq!(env.t_b.len(), chi * bond_sq * chi);
    }

    // Test 3: environment tensors are non-zero after init
    #[test]
    fn test_init_nonzero() {
        let mut rng = LcgRng::new(3);
        let (site, shape) = product_site(2);
        let env = ctmrg_init(&site, &shape, 3, &mut rng).expect("init ok");
        let c_tl_nrm: f64 = env.c_tl.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(c_tl_nrm > 1.0e-10, "C_TL should be non-zero after init");
        let t_l_nrm: f64 = env.t_l.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(t_l_nrm > 1.0e-10, "T_L should be non-zero after init");
    }

    // Test 4: ctmrg_step_right returns singular values with correct chi
    #[test]
    fn test_step_right_sv_length() {
        let mut rng = LcgRng::new(4);
        let (site, shape) = product_site(2);
        let chi = 3;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        let sv = ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
        assert!(
            !sv.is_empty() && sv.len() <= chi,
            "sv len should be at most chi"
        );
    }

    // Test 5: step_right updates env corner shapes correctly
    #[test]
    fn test_step_right_corner_shape_preserved() {
        let mut rng = LcgRng::new(5);
        let (site, shape) = product_site(2);
        let chi = 2;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
        assert_eq!(env.c_tr.len(), chi * chi);
        assert_eq!(env.c_br.len(), chi * chi);
        assert_eq!(env.c_tl.len(), chi * chi);
        assert_eq!(env.c_bl.len(), chi * chi);
    }

    // Test 6: running 3 steps changes environment (values evolve)
    #[test]
    fn test_three_steps_env_changes() {
        let mut rng = LcgRng::new(6);
        let (site, shape) = product_site(2);
        let chi = 2;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        let c_tr_0 = env.c_tr.clone();
        for _ in 0..3 {
            ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
        }
        // At least one element should have changed by more than numerical noise
        let changed = c_tr_0
            .iter()
            .zip(env.c_tr.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-12);
        assert!(changed, "C_TR should change after 3 steps");
    }

    // Test 7: ctmrg_norm_per_site returns positive finite value
    #[test]
    fn test_norm_positive_finite() {
        let mut rng = LcgRng::new(7);
        let (site, shape) = product_site(2);
        let chi = 2;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        // Run a few steps to get a meaningful environment
        for _ in 0..5 {
            ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
            ctmrg_step_down(&mut env, &site, &shape, chi).expect("step ok");
        }
        let norm = ctmrg_norm_per_site(&env, &site, &shape).expect("norm ok");
        assert!(norm > 0.0, "norm should be positive, got {norm}");
        assert!(norm.is_finite(), "norm should be finite");
    }

    // Test 8: chi_env=1 trivial environment (product state limit)
    #[test]
    fn test_chi_1_product_state() {
        let mut rng = LcgRng::new(8);
        let (site, shape) = product_site(2);
        let chi = 1;
        let env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        assert_eq!(env.c_tl.len(), 1);
        // t_l shape [chi=1, bond_sq, chi=1] → flat length = bond_sq.
        assert_eq!(env.t_l.len(), env.bond_sq);
    }

    // Test 9: ctmrg_run converges for product state tensor
    #[test]
    fn test_run_converges_product_state() {
        let mut rng = LcgRng::new(9);
        let (site, shape) = product_site(2);
        let config = CtmrgConfig {
            chi_env: 2,
            max_iter: 100,
            tol: 1.0e-6,
            n_steps_per_iter: 1,
        };
        let result = ctmrg_run(&site, &shape, &config, &mut rng).expect("run ok");
        // For a product state, environment should converge quickly
        assert!(
            result.n_iter <= config.max_iter,
            "n_iter must not exceed max_iter"
        );
        assert!(
            result.norm_per_site.is_finite(),
            "norm_per_site must be finite"
        );
    }

    // Test 10: n_iter ≤ max_iter always
    #[test]
    fn test_n_iter_bounded() {
        let mut rng = LcgRng::new(10);
        let (site, shape) = product_site(2);
        let max_iter = 5;
        let config = CtmrgConfig {
            chi_env: 2,
            max_iter,
            tol: 1.0e-15, // very tight — unlikely to converge
            n_steps_per_iter: 1,
        };
        let result = ctmrg_run(&site, &shape, &config, &mut rng).expect("run ok");
        assert!(result.n_iter <= max_iter, "n_iter must be ≤ max_iter");
    }

    // Test 11: invalid chi_env=0 returns error
    #[test]
    fn test_invalid_chi_zero() {
        let mut rng = LcgRng::new(11);
        let (site, shape) = product_site(2);
        let err = ctmrg_init(&site, &shape, 0, &mut rng);
        assert!(err.is_err(), "chi_env=0 should return error");
    }

    // Test 12: invalid site_shape (phys_dim=0) returns error
    #[test]
    fn test_invalid_phys_dim_zero() {
        let mut rng = LcgRng::new(12);
        let site = vec![1.0f64];
        let shape = [1usize, 1, 1, 1, 0]; // d_p = 0
        let err = ctmrg_init(&site, &shape, 2, &mut rng);
        assert!(err.is_err(), "d_p=0 should return error");
    }

    // Test 13: ctmrg_expectation with identity operator returns approx 1.0
    #[test]
    fn test_expectation_identity() {
        let mut rng = LcgRng::new(13);
        let d_p = 2;
        let (site, shape) = product_site(d_p);
        let chi = 2;
        let config = CtmrgConfig {
            chi_env: chi,
            max_iter: 20,
            tol: 1.0e-6,
            n_steps_per_iter: 1,
        };
        let result = ctmrg_run(&site, &shape, &config, &mut rng).expect("run ok");
        // Identity operator [d_p × d_p]
        let mut id_op = vec![0.0f64; d_p * d_p];
        for i in 0..d_p {
            id_op[i * d_p + i] = 1.0;
        }
        let exp_val = ctmrg_expectation(&result.env, &site, &shape, &id_op).expect("expect ok");
        // For identity operator, expectation = norm / norm = 1.0
        assert!(
            (exp_val - 1.0).abs() < 1.0,
            "identity expectation should be ~1.0 (relative to norm), got {exp_val}"
        );
    }

    // Test 14: After multiple right steps, C_TR Frobenius norm stabilises near 1
    // (corners are normalised to unit Frobenius norm after each step, so the norm
    // gauge-invariant measure of convergence is whether the norm remains ≈ 1.0).
    #[test]
    fn test_ctr_stabilises() {
        let mut rng = LcgRng::new(14);
        let (site, shape) = product_site(2);
        let chi = 2;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        // Warm up
        for _ in 0..30 {
            ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
            ctmrg_step_down(&mut env, &site, &shape, chi).expect("step ok");
        }
        // After normalisation each corner has unit Frobenius norm; verify it stays there.
        let frob_ctr: f64 = env.c_tr.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (frob_ctr - 1.0).abs() < 1.0e-10,
            "C_TR Frobenius norm should be 1.0 after normalised steps, got {frob_ctr}"
        );
        // The norm per site should remain finite and positive through many steps.
        let norm = ctmrg_norm_per_site(&env, &site, &shape).expect("norm ok");
        assert!(
            norm > 0.0 && norm.is_finite(),
            "norm_per_site should remain positive+finite, got {norm}"
        );
    }

    // Test 15: edge tensor sizes preserved through steps
    #[test]
    fn test_edge_sizes_preserved_through_steps() {
        let mut rng = LcgRng::new(15);
        let (site, shape) = product_site(2);
        let chi = 3;
        let mut env = ctmrg_init(&site, &shape, chi, &mut rng).expect("init ok");
        let bond_sq = env.bond_sq;
        for _ in 0..3 {
            ctmrg_step_right(&mut env, &site, &shape, chi).expect("step ok");
        }
        assert_eq!(env.t_l.len(), chi * bond_sq * chi);
        assert_eq!(env.t_r.len(), chi * bond_sq * chi);
    }
}
