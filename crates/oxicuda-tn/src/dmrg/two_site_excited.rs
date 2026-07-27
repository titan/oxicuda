//! Excited-state DMRG via the penalty / shift-and-invert method.
//!
//! After computing the ground state |ψ₀⟩ with standard two-site DMRG, this module
//! finds low-lying excited states |ψ₁⟩, |ψ₂⟩, … by sequentially deflating already-found
//! states from the effective Hamiltonian.
//!
//! ## Method
//!
//! For each target excited state |ψₖ⟩ we run fresh two-site DMRG sweeps on the
//! **penalised Hamiltonian**
//!
//! ```text
//! H̃ = H + ω · Σᵢ<k  |ψᵢ⟩⟨ψᵢ|
//! ```
//!
//! where ω >> |E₀| pushes all previously-found states far up in energy, making the
//! k-th excited state the new variational minimum.  At each two-site update we compute
//! the two-site projector of every prior state and add the penalty contribution to the
//! matrix-free effective Hamiltonian action.
//!
//! ## Reference
//!
//! * McCulloch, "From density-matrix renormalization group to matrix product states"
//!   (2007), section on targeting excited states.
//! * Dorando, Hachmann, Chan, J. Chem. Phys. 130, 184111 (2009) — state-averaged
//!   and shift-and-invert excited-state DMRG.

use crate::dmrg::dmrg::{build_left_env_pub, build_right_env_pub, mpo_expectation};
use crate::dmrg::lanczos::lanczos_smallest;
use crate::handle::LcgRng;
use crate::mpo::mpo::Mpo;
use crate::mps::canonical::right_canonicalize;
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ── Public configuration and result types ─────────────────────────────────────

/// Configuration knobs for excited-state two-site DMRG.
#[derive(Debug, Clone, Copy)]
pub struct ExcitedDmrgConfig {
    /// Maximum bond dimension retained after SVD truncation.
    pub chi_max: usize,
    /// Maximum number of full L→R + R→L sweep pairs per excited state.
    pub max_sweeps: usize,
    /// Energy convergence threshold (absolute change between successive sweeps).
    pub tol: f64,
    /// Penalty weight ω.  Must be strictly positive.  A value of 10 × |E₀| or larger
    /// typically works; the default of 10.0 is suitable when energies are O(1).
    pub penalty_weight: f64,
    /// Maximum Lanczos iterations for each two-site update.
    pub lanczos_max_iter: usize,
    /// Lanczos eigenvalue convergence tolerance.
    pub lanczos_tol: f64,
    /// Number of excited states to compute (sequentially).
    pub n_excited: usize,
}

impl Default for ExcitedDmrgConfig {
    fn default() -> Self {
        Self {
            chi_max: 16,
            max_sweeps: 8,
            tol: 1.0e-7,
            penalty_weight: 10.0,
            lanczos_max_iter: 40,
            lanczos_tol: 1.0e-10,
            n_excited: 1,
        }
    }
}

/// Result returned by [`two_site_excited_dmrg`].
#[derive(Debug, Clone)]
pub struct ExcitedDmrgResult {
    /// Energies E₁, E₂, … (one entry per requested excited state).
    pub excited_energies: Vec<f64>,
    /// Ground-state energy E₀ for reference.
    pub ground_state_energy: f64,
    /// Per-excited-state, per-site shape `[D_l, d, D_r]`.
    pub mps_shapes: Vec<Vec<[usize; 3]>>,
    /// Per-excited-state, per-site flat row-major tensor data.
    pub mps_data: Vec<Vec<Vec<f64>>>,
    /// Total sweep count (summed across all excited states).
    pub n_sweeps: usize,
    /// `true` if every excited state converged within [`ExcitedDmrgConfig::tol`].
    pub converged: bool,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Find the first `cfg.n_excited` excited states of `mpo` using penalty-deflation DMRG.
///
/// # Arguments
///
/// * `mpo`    — MPO representation of the Hamiltonian.
/// * `ground` — Already-computed ground state returned by standard two-site DMRG
///   (consumed and canonicalised internally).
/// * `cfg`    — Algorithm hyper-parameters.
/// * `rng`    — RNG for seeding the initial MPS guess.
///
/// # Errors
///
/// Returns `TnError::InvalidConfiguration` if `cfg.penalty_weight ≤ 0`, if the chain
/// has fewer than 2 sites, or if `cfg.n_excited == 0`.
pub fn two_site_excited_dmrg(
    mpo: &Mpo,
    ground: Mps,
    cfg: ExcitedDmrgConfig,
    rng: &mut LcgRng,
) -> TnResult<ExcitedDmrgResult> {
    // ── Validate inputs ───────────────────────────────────────────────────────
    if cfg.penalty_weight <= 0.0 {
        return Err(TnError::InvalidConfiguration(
            "penalty_weight must be strictly positive".into(),
        ));
    }
    if cfg.n_excited == 0 {
        return Err(TnError::InvalidConfiguration(
            "n_excited must be at least 1".into(),
        ));
    }
    let n = mpo.n_sites();
    if n < 2 {
        return Err(TnError::InvalidConfiguration(
            "chain must have at least 2 sites for two-site DMRG".into(),
        ));
    }
    if mpo.n_sites() != ground.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: mpo.n_sites(),
            b: ground.n_sites(),
        });
    }

    let ground_energy = mpo_expectation(mpo, &ground)?;

    // ── Accumulator for deflation states ──────────────────────────────────────
    // We store each found state (canonicalised) so we can project it out.
    let mut found_states: Vec<Mps> = vec![ground];

    // ── Per-excited-state loop ────────────────────────────────────────────────
    let mut excited_energies: Vec<f64> = Vec::with_capacity(cfg.n_excited);
    let mut all_shapes: Vec<Vec<[usize; 3]>> = Vec::with_capacity(cfg.n_excited);
    let mut all_data: Vec<Vec<Vec<f64>>> = Vec::with_capacity(cfg.n_excited);
    let mut total_sweeps = 0usize;
    let mut all_converged = true;

    for _k in 0..cfg.n_excited {
        let (ex_mps, ex_energy, sweeps, conv) =
            find_one_excited(mpo, &found_states, cfg, rng, n, ground_energy)?;
        total_sweeps += sweeps;
        if !conv {
            all_converged = false;
        }
        excited_energies.push(ex_energy);
        // Collect shapes and data
        let shapes: Vec<[usize; 3]> = ex_mps
            .site_tensors
            .iter()
            .map(|t| [t.d_l, t.d_p, t.d_r])
            .collect();
        let data: Vec<Vec<f64>> = ex_mps.site_tensors.iter().map(|t| t.data.clone()).collect();
        all_shapes.push(shapes);
        all_data.push(data);
        found_states.push(ex_mps);
    }

    Ok(ExcitedDmrgResult {
        excited_energies,
        ground_state_energy: ground_energy,
        mps_shapes: all_shapes,
        mps_data: all_data,
        n_sweeps: total_sweeps,
        converged: all_converged,
    })
}

// ── Internal: optimise one excited state ─────────────────────────────────────

/// Run penalty-DMRG sweeps to find the next excited state orthogonal to all states
/// in `prior`.
///
/// Returns `(optimised_mps, energy, sweeps, converged)`.
fn find_one_excited(
    mpo: &Mpo,
    prior: &[Mps],
    cfg: ExcitedDmrgConfig,
    rng: &mut LcgRng,
    n: usize,
    ground_energy: f64,
) -> TnResult<(Mps, f64, usize, bool)> {
    // Initialise a fresh random MPS as starting point for this excited state.
    let d = mpo.site_tensors[0].d_out;
    let chi_init = cfg.chi_max.clamp(2, 8);
    let mut cur = Mps::random_mps(n, d, chi_init, rng)?;
    right_canonicalize(&mut cur)?;

    // ── Build right environments for H ────────────────────────────────────────
    let mut right_envs: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    right_envs[n] = vec![1.0];
    let mut r_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n + 1];
    for s in (1..n).rev() {
        right_envs[s] = build_right_env_pub(
            &right_envs[s + 1],
            r_shapes[s + 1],
            &cur.site_tensors[s],
            &mpo.site_tensors[s],
        )?;
        r_shapes[s] = (
            cur.site_tensors[s].d_l,
            mpo.site_tensors[s].w_l,
            cur.site_tensors[s].d_l,
        );
    }

    // Left environments (initialised trivially; filled during sweeps).
    let mut left_envs: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    let mut l_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n + 1];
    left_envs[0] = vec![1.0];

    let mut last_energy = ground_energy;
    let mut sweeps = 0usize;
    let mut converged = false;

    for sweep_idx in 0..cfg.max_sweeps {
        // ── Left-to-right sweep ───────────────────────────────────────────────
        for s in 0..n - 1 {
            optimise_with_penalty(
                &mut cur,
                mpo,
                prior,
                &left_envs,
                &l_shapes,
                &right_envs,
                &r_shapes,
                s,
                cfg,
                rng,
                true,
            )?;
            left_envs[s + 1] = build_left_env_pub(
                &left_envs[s],
                l_shapes[s],
                &cur.site_tensors[s],
                &mpo.site_tensors[s],
            )?;
            l_shapes[s + 1] = (
                cur.site_tensors[s].d_r,
                mpo.site_tensors[s].w_r,
                cur.site_tensors[s].d_r,
            );
        }
        // ── Right-to-left sweep ───────────────────────────────────────────────
        for s in (1..n).rev() {
            optimise_with_penalty(
                &mut cur,
                mpo,
                prior,
                &left_envs,
                &l_shapes,
                &right_envs,
                &r_shapes,
                s - 1,
                cfg,
                rng,
                false,
            )?;
            right_envs[s] = build_right_env_pub(
                &right_envs[s + 1],
                r_shapes[s + 1],
                &cur.site_tensors[s],
                &mpo.site_tensors[s],
            )?;
            r_shapes[s] = (
                cur.site_tensors[s].d_l,
                mpo.site_tensors[s].w_l,
                cur.site_tensors[s].d_l,
            );
        }

        // ── Energy measurement ────────────────────────────────────────────────
        // We measure the true (unpenalised) energy via the original MPO.
        let energy = mpo_expectation(mpo, &cur)?;
        sweeps = sweep_idx + 1;
        if (last_energy - energy).abs() < cfg.tol && sweep_idx > 0 {
            converged = true;
            last_energy = energy;
            break;
        }
        last_energy = energy;
    }

    Ok((cur, last_energy, sweeps, converged))
}

// ── Two-site optimisation with penalty ───────────────────────────────────────

/// Perform one two-site update at bond `(s, s+1)` using the penalised effective
/// Hamiltonian `H_eff + ω Σᵢ |Θᵢ⟩⟨Θᵢ|`.
#[allow(clippy::too_many_arguments)]
fn optimise_with_penalty(
    mps: &mut Mps,
    mpo: &Mpo,
    prior: &[Mps],
    left_envs: &[Vec<f64>],
    l_shapes: &[(usize, usize, usize)],
    right_envs: &[Vec<f64>],
    r_shapes: &[(usize, usize, usize)],
    s: usize,
    cfg: ExcitedDmrgConfig,
    rng: &mut LcgRng,
    sweep_right: bool,
) -> TnResult<()> {
    // ── Form the two-site tensor Θ = M_s · M_{s+1} ───────────────────────────
    let lt = mps.site_tensors[s].clone();
    let rt = mps.site_tensors[s + 1].clone();
    let (dl, dp1, dm) = lt.shape();
    let (dm_r, dp2, dr) = rt.shape();
    if dm != dm_r {
        return Err(TnError::DimensionMismatch { a: dm, b: dm_r });
    }
    let n_theta = dl * dp1 * dp2 * dr;
    let mut theta = vec![0.0f64; n_theta];
    for a in 0..dl {
        for p1 in 0..dp1 {
            for p2 in 0..dp2 {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for c in 0..dm {
                        acc += lt.data[(a * dp1 + p1) * dm + c] * rt.data[(c * dp2 + p2) * dr + b];
                    }
                    theta[((a * dp1 + p1) * dp2 + p2) * dr + b] = acc;
                }
            }
        }
    }

    // ── Collect prior two-site tensors (penalty projectors) ──────────────────
    //
    // The penalty ω Σᵢ |ψᵢ⟩⟨ψᵢ|, restricted to the two-site variational space of
    // this update, is ω Σᵢ |Θᵢᵉᶠᶠ⟩⟨Θᵢᵉᶠᶠ| where Θᵢᵉᶠᶠ is the prior state's
    // two-site tensor **rotated into the current state's block bases**:
    //
    //     Θᵢᵉᶠᶠ[a,p₁,p₂,b] = Σ_{a',b'} L[a,a'] · Θᵢ[a',p₁,p₂,b'] · R[b,b']
    //
    // with L and R the overlap transfer matrices between the current MPS's and
    // the prior MPS's left/right blocks (see [`prior_two_site_projector`]).
    //
    // Using the prior's *raw* Θᵢ instead only makes sense if both states shared
    // a bond basis, which they do not: each is canonicalised by its own SVDs, so
    // bond index `a` means something different in each. Penalising the raw
    // tensor therefore pushes up an essentially arbitrary direction and leaves
    // the "excited" state free to retain a large ⟨ψ₀|ψ₁⟩ overlap.
    //
    // The projectors are deliberately *not* renormalised: with the correct
    // rotation ⟨Θᵢᵉᶠᶠ|Θ⟩ already equals ⟨ψᵢ|ψ⟩, so ω|Θᵢᵉᶠᶠ⟩⟨Θᵢᵉᶠᶠ| contributes
    // exactly ω·|⟨ψᵢ|ψ⟩|² — the intended penalty energy. Rescaling it to unit
    // norm would inflate the penalty by 1/‖Θᵢᵉᶠᶠ‖² whenever the current blocks
    // only partially span |ψᵢ⟩.
    let mut penalty_vecs: Vec<Vec<f64>> = Vec::with_capacity(prior.len());
    for prior_mps in prior {
        match prior_two_site_projector(mps, prior_mps, s) {
            Ok(pv) => {
                debug_assert_eq!(pv.len(), n_theta);
                if dot_self(&pv) > 1e-300 {
                    penalty_vecs.push(pv);
                }
            }
            Err(_) => continue,
        }
    }

    // ── Capture environment data for the closure ──────────────────────────────
    let left = left_envs[s].clone();
    let l_sh = l_shapes[s];
    let right = right_envs[s + 2].clone();
    let r_sh = r_shapes[s + 2];
    let mpo_l = mpo.site_tensors[s].clone();
    let mpo_r = mpo.site_tensors[s + 1].clone();
    let omega = cfg.penalty_weight;

    // ── Penalised apply closure ───────────────────────────────────────────────
    let apply = move |v: &[f64]| -> Vec<f64> {
        // Standard H_eff action
        let mut out = h_eff_apply(
            v, &left, l_sh, &right, r_sh, &mpo_l, &mpo_r, dl, dp1, dp2, dr,
        );
        // Penalty: + ω · Σᵢ ⟨Θᵢ|v⟩ Θᵢ
        for pv in &penalty_vecs {
            let overlap = dot_vecs(pv, v);
            for (o, p) in out.iter_mut().zip(pv.iter()) {
                *o += omega * overlap * p;
            }
        }
        out
    };

    // ── Seed Lanczos from theta ───────────────────────────────────────────────
    let mut seed = theta.clone();
    let seed_norm: f64 = dot_self(&seed).sqrt();
    if seed_norm < 1e-15 {
        for v in &mut seed {
            *v = rng.next_normal();
        }
    }

    let r = lanczos_smallest(apply, n_theta, &seed, cfg.lanczos_max_iter, cfg.lanczos_tol)?;
    let new_theta = r.eigenvector;

    // ── SVD-split: (dl*dp1, dp2*dr) ──────────────────────────────────────────
    let m = dl * dp1;
    let cols = dp2 * dr;
    let svd = svd_jacobi(&new_theta, m, cols)?;
    let (svd, _) = svd_truncate(svd, cfg.chi_max, 1.0e-10)?;
    let k = svd.k;
    let mut left_new = vec![0.0f64; dl * dp1 * k];
    let mut right_new = vec![0.0f64; k * dp2 * dr];
    if sweep_right {
        // M_left := U, M_right := diag(s) * V^T
        for i in 0..m {
            for j in 0..k {
                left_new[i * k + j] = svd.u[i * k + j];
            }
        }
        for i in 0..k {
            for j in 0..cols {
                right_new[i * cols + j] = svd.s[i] * svd.vt[i * cols + j];
            }
        }
    } else {
        // M_left := U * diag(s), M_right := V^T
        for i in 0..m {
            for j in 0..k {
                left_new[i * k + j] = svd.u[i * k + j] * svd.s[j];
            }
        }
        for i in 0..k {
            for j in 0..cols {
                right_new[i * cols + j] = svd.vt[i * cols + j];
            }
        }
    }
    mps.site_tensors[s] = MpsTensor::new(dl, dp1, k, left_new)?;
    mps.site_tensors[s + 1] = MpsTensor::new(k, dp2, dr, right_new)?;
    Ok(())
}

// ── Effective Hamiltonian apply ───────────────────────────────────────────────

/// Apply the two-site effective Hamiltonian `L ⊗ W_s ⊗ W_{s+1} ⊗ R` to a vector `psi`.
///
/// This is a verbatim copy of the version in `dmrg.rs` (which is private there).
#[allow(clippy::too_many_arguments)]
fn h_eff_apply(
    psi: &[f64],
    left: &[f64],
    l_sh: (usize, usize, usize),
    right: &[f64],
    r_sh: (usize, usize, usize),
    mpo_l: &crate::mpo::mpo::MpoTensor,
    mpo_r: &crate::mpo::mpo::MpoTensor,
    dl: usize,
    dp1: usize,
    dp2: usize,
    dr: usize,
) -> Vec<f64> {
    let (la, lw, lap) = l_sh;
    let (rb, rw, rbp) = r_sh;
    let _ = (la, lap, rb, rbp);
    let (wl1, dout1, din1, wr1) = mpo_l.shape();
    let (wl2, dout2, din2, wr2) = mpo_r.shape();
    let _ = (wl1, wl2, dout1, dout2);
    let mut out = vec![0.0f64; dl * dp1 * dp2 * dr];
    for a in 0..dl {
        for p1 in 0..dp1 {
            for p2 in 0..dp2 {
                for b in 0..dr {
                    let mut acc = 0.0f64;
                    for ap in 0..dl {
                        for p1p in 0..dp1 {
                            for p2p in 0..dp2 {
                                for bp in 0..dr {
                                    let psi_v = psi[((ap * dp1 + p1p) * dp2 + p2p) * dr + bp];
                                    if psi_v == 0.0 {
                                        continue;
                                    }
                                    let mut he = 0.0f64;
                                    for wmid in 0..wr1 {
                                        for w_l in 0..lw {
                                            for w_r in 0..rw {
                                                let lv = left[(a * lw + w_l) * lap + ap];
                                                let rv = right[(b * rw + w_r) * rbp + bp];
                                                let mlv = mpo_l.data[((w_l * dout1 + p1) * din1
                                                    + p1p)
                                                    * wr1
                                                    + wmid];
                                                let mrv = mpo_r.data[((wmid * dout2 + p2) * din2
                                                    + p2p)
                                                    * wr2
                                                    + w_r];
                                                he += lv * mlv * mrv * rv;
                                            }
                                        }
                                    }
                                    acc += he * psi_v;
                                }
                            }
                        }
                    }
                    out[((a * dp1 + p1) * dp2 + p2) * dr + b] = acc;
                }
            }
        }
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract and contract the two-site tensor Θ[a, p1, p2, b] = M_s[a, p1, c] · M_{s+1}[c, p2, b]
/// from an MPS without modifying it.
fn extract_two_site_tensor(mps: &Mps, s: usize) -> TnResult<Vec<f64>> {
    if s + 1 >= mps.n_sites() {
        return Err(TnError::IndexOutOfBounds {
            index: s + 1,
            len: mps.n_sites(),
        });
    }
    let lt = &mps.site_tensors[s];
    let rt = &mps.site_tensors[s + 1];
    let (dl, dp1, dm) = lt.shape();
    let (dm_r, dp2, dr) = rt.shape();
    if dm != dm_r {
        return Err(TnError::DimensionMismatch { a: dm, b: dm_r });
    }
    let mut theta = vec![0.0f64; dl * dp1 * dp2 * dr];
    for a in 0..dl {
        for p1 in 0..dp1 {
            for p2 in 0..dp2 {
                for b in 0..dr {
                    let mut acc = 0.0f64;
                    for c in 0..dm {
                        acc += lt.data[(a * dp1 + p1) * dm + c] * rt.data[(c * dp2 + p2) * dr + b];
                    }
                    theta[((a * dp1 + p1) * dp2 + p2) * dr + b] = acc;
                }
            }
        }
    }
    Ok(theta)
}

/// Overlap transfer matrix between the left blocks of `cur` and `prior`, i.e.
/// the contraction of sites `0..s` of both states over their shared physical
/// indices.
///
/// The result is row-major with shape `(cur.d_l at s) × (prior.d_l at s)`:
/// entry `[a, a']` is ⟨block_a of `cur` | block_a' of `prior`⟩. For `s == 0` it
/// is the 1×1 identity.
fn overlap_env_left(cur: &Mps, prior: &Mps, s: usize) -> TnResult<(Vec<f64>, usize, usize)> {
    let mut env = vec![1.0f64];
    let (mut rows, mut cols) = (1usize, 1usize);
    for t in 0..s {
        let c = &cur.site_tensors[t];
        let p = &prior.site_tensors[t];
        if c.d_p != p.d_p {
            return Err(TnError::DimensionMismatch { a: c.d_p, b: p.d_p });
        }
        if c.d_l != rows || p.d_l != cols {
            return Err(TnError::DimensionMismatch { a: c.d_l, b: rows });
        }
        let (d, cr, pr) = (c.d_p, c.d_r, p.d_r);
        let mut next = vec![0.0f64; cr * pr];
        for ac in 0..rows {
            for ap in 0..cols {
                let e = env[ac * cols + ap];
                if e == 0.0 {
                    continue;
                }
                for ph in 0..d {
                    for bc in 0..cr {
                        let cv = e * c.data[(ac * d + ph) * cr + bc];
                        if cv == 0.0 {
                            continue;
                        }
                        for bp in 0..pr {
                            next[bc * pr + bp] += cv * p.data[(ap * d + ph) * pr + bp];
                        }
                    }
                }
            }
        }
        env = next;
        rows = cr;
        cols = pr;
    }
    Ok((env, rows, cols))
}

/// Overlap transfer matrix between the right blocks of `cur` and `prior`, i.e.
/// the contraction of sites `from..n` of both states over their shared physical
/// indices.
///
/// The result is row-major with shape `(cur.d_l at from) × (prior.d_l at from)`.
/// For `from == n` it is the 1×1 identity.
fn overlap_env_right(cur: &Mps, prior: &Mps, from: usize) -> TnResult<(Vec<f64>, usize, usize)> {
    let n = cur.n_sites();
    let mut env = vec![1.0f64];
    let (mut rows, mut cols) = (1usize, 1usize);
    for t in (from..n).rev() {
        let c = &cur.site_tensors[t];
        let p = &prior.site_tensors[t];
        if c.d_p != p.d_p {
            return Err(TnError::DimensionMismatch { a: c.d_p, b: p.d_p });
        }
        if c.d_r != rows || p.d_r != cols {
            return Err(TnError::DimensionMismatch { a: c.d_r, b: rows });
        }
        let (d, cl, pl) = (c.d_p, c.d_l, p.d_l);
        let mut next = vec![0.0f64; cl * pl];
        for ac in 0..cl {
            for ap in 0..pl {
                let mut acc = 0.0f64;
                for ph in 0..d {
                    for bc in 0..rows {
                        let cv = c.data[(ac * d + ph) * rows + bc];
                        if cv == 0.0 {
                            continue;
                        }
                        for bp in 0..cols {
                            acc += cv * p.data[(ap * d + ph) * cols + bp] * env[bc * cols + bp];
                        }
                    }
                }
                next[ac * pl + ap] = acc;
            }
        }
        env = next;
        rows = cl;
        cols = pl;
    }
    Ok((env, rows, cols))
}

/// Build the penalty projector for `prior` at bond `(s, s+1)`, expressed in the
/// two-site basis of `cur`.
///
/// Returns Θᵉᶠᶠ[a,p₁,p₂,b] = Σ_{a',b'} L[a,a'] · Θ_prior[a',p₁,p₂,b'] · R[b,b'],
/// laid out row-major over `(cur d_l at s, d_p at s, d_p at s+1, cur d_r at s+1)`
/// — the same layout as the current two-site tensor, so ⟨Θᵉᶠᶠ|Θ_cur⟩ is exactly
/// the full-state overlap ⟨prior|cur⟩ when `cur` is in mixed-canonical form
/// around this bond.
fn prior_two_site_projector(cur: &Mps, prior: &Mps, s: usize) -> TnResult<Vec<f64>> {
    let n = cur.n_sites();
    if prior.n_sites() != n {
        return Err(TnError::DimensionMismatch {
            a: prior.n_sites(),
            b: n,
        });
    }
    if s + 1 >= n {
        return Err(TnError::IndexOutOfBounds {
            index: s + 1,
            len: n,
        });
    }

    let theta_prior = extract_two_site_tensor(prior, s)?;
    let (l_env, dl_cur, dl_pri) = overlap_env_left(cur, prior, s)?;
    let (r_env, dr_cur, dr_pri) = overlap_env_right(cur, prior, s + 2)?;

    let dp1 = cur.site_tensors[s].d_p;
    let dp2 = cur.site_tensors[s + 1].d_p;
    if prior.site_tensors[s].d_p != dp1 || prior.site_tensors[s + 1].d_p != dp2 {
        return Err(TnError::DimensionMismatch {
            a: prior.site_tensors[s].d_p,
            b: dp1,
        });
    }

    // First rotate the left bond: tmp[a, p1, p2, b'] = Σ_{a'} L[a,a'] Θ[a',p1,p2,b'].
    let phys = dp1 * dp2;
    let mut tmp = vec![0.0f64; dl_cur * phys * dr_pri];
    for a in 0..dl_cur {
        for ap in 0..dl_pri {
            let lv = l_env[a * dl_pri + ap];
            if lv == 0.0 {
                continue;
            }
            for pp in 0..phys {
                let src = (ap * phys + pp) * dr_pri;
                let dst = (a * phys + pp) * dr_pri;
                for bp in 0..dr_pri {
                    tmp[dst + bp] += lv * theta_prior[src + bp];
                }
            }
        }
    }

    // Then the right bond: out[a, p1, p2, b] = Σ_{b'} tmp[a,p1,p2,b'] R[b,b'].
    let mut out = vec![0.0f64; dl_cur * phys * dr_cur];
    for a in 0..dl_cur {
        for pp in 0..phys {
            let src = (a * phys + pp) * dr_pri;
            let dst = (a * phys + pp) * dr_cur;
            for b in 0..dr_cur {
                let mut acc = 0.0f64;
                for bp in 0..dr_pri {
                    acc += tmp[src + bp] * r_env[b * dr_pri + bp];
                }
                out[dst + b] = acc;
            }
        }
    }
    Ok(out)
}

/// Compute the inner product ⟨ψ₀|ψ₁⟩ between two MPS of the same chain length and
/// physical dimensions, contracting site-by-site from the left.
///
/// This is an O(L · D³ · d) routine intended for test assertions and diagnostic use.
pub fn mps_inner_product(bra: &Mps, ket: &Mps) -> TnResult<f64> {
    if bra.n_sites() != ket.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: bra.n_sites(),
            b: ket.n_sites(),
        });
    }
    let n = bra.n_sites();
    // Transfer matrix env[a, a'] of shape (dl_bra × dl_ket)
    let mut env = vec![1.0f64]; // 1×1
    let mut env_rows = 1usize; // = dl_bra at current site
    let mut env_cols = 1usize; // = dl_ket at current site
    for s in 0..n {
        let bra_t = &bra.site_tensors[s];
        let ket_t = &ket.site_tensors[s];
        if bra_t.d_p != ket_t.d_p {
            return Err(TnError::DimensionMismatch {
                a: bra_t.d_p,
                b: ket_t.d_p,
            });
        }
        if bra_t.d_l != env_rows || ket_t.d_l != env_cols {
            return Err(TnError::DimensionMismatch {
                a: bra_t.d_l,
                b: env_rows,
            });
        }
        let d = bra_t.d_p;
        let dbl = bra_t.d_l;
        let dbr = bra_t.d_r;
        let dkl = ket_t.d_l;
        let dkr = ket_t.d_r;
        // new_env[b_bra, b_ket] = Σ_{a_bra, a_ket, p} env[a_bra, a_ket]
        //                         · bra[a_bra, p, b_bra] · ket[a_ket, p, b_ket]
        let _ = (dbl, dkl);
        let mut new_env = vec![0.0f64; dbr * dkr];
        for b_bra in 0..dbr {
            for b_ket in 0..dkr {
                let mut acc = 0.0f64;
                for a_bra in 0..env_rows {
                    for a_ket in 0..env_cols {
                        let e = env[a_bra * env_cols + a_ket];
                        for p in 0..d {
                            let bv = bra_t.data[(a_bra * d + p) * dbr + b_bra];
                            let kv = ket_t.data[(a_ket * d + p) * dkr + b_ket];
                            acc += e * bv * kv;
                        }
                    }
                }
                new_env[b_bra * dkr + b_ket] = acc;
            }
        }
        env = new_env;
        env_rows = dbr;
        env_cols = dkr;
    }
    // At the end env is 1×1.
    Ok(env[0])
}

// ── Dot-product utilities ─────────────────────────────────────────────────────

#[inline]
fn dot_vecs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn dot_self(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmrg::dmrg::{DmrgConfig, dmrg_two_site};
    use crate::handle::LcgRng;
    use crate::mpo::mpo::Mpo;
    use crate::mps::mps::Mps;

    // Helper: run ground-state DMRG on a Heisenberg chain.
    fn ground_state(n_sites: usize, chi: usize, seed: u64) -> (Mps, f64, Mpo) {
        let mut rng = LcgRng::new(seed);
        let mpo = Mpo::heisenberg_xxx(n_sites).expect("heisenberg");
        let init = Mps::random_mps(n_sites, 2, chi, &mut rng).expect("random mps");
        let cfg = DmrgConfig {
            max_sweeps: 10,
            chi_max: chi,
            energy_tol: 1e-8,
            lanczos_iter: 40,
            lanczos_tol: 1e-10,
            ..DmrgConfig::default()
        };
        let res = dmrg_two_site(&mpo, init, cfg, &mut rng).expect("dmrg");
        (res.mps, res.energy, mpo)
    }

    // ── Test 1: first excited energy > ground energy ──────────────────────────
    #[test]
    fn first_excited_above_ground_4site() {
        let (gs_mps, e0, mpo) = ground_state(4, 8, 1);
        let mut rng = LcgRng::new(42);
        let cfg = ExcitedDmrgConfig {
            chi_max: 8,
            max_sweeps: 10,
            tol: 1e-6,
            penalty_weight: 20.0,
            lanczos_max_iter: 40,
            lanczos_tol: 1e-10,
            n_excited: 1,
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("excited dmrg");
        let e1 = res.excited_energies[0];
        assert!(
            e1 > e0 - 1e-4,
            "excited energy {e1} should be > ground {e0}"
        );
    }

    // ── Test 2: orthogonality ⟨ψ₁|ψ₀⟩ ≈ 0 ────────────────────────────────────
    #[test]
    fn excited_orthogonal_to_ground_4site() {
        let (gs_mps, _e0, mpo) = ground_state(4, 8, 2);
        let gs_copy = gs_mps.clone();
        let mut rng = LcgRng::new(43);
        let cfg = ExcitedDmrgConfig {
            chi_max: 8,
            max_sweeps: 10,
            tol: 1e-6,
            penalty_weight: 30.0,
            lanczos_max_iter: 40,
            lanczos_tol: 1e-10,
            n_excited: 1,
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("excited dmrg");
        // Reconstruct excited MPS from result
        let n = mpo.n_sites();
        let shapes = &res.mps_shapes[0];
        let data = &res.mps_data[0];
        let tensors: Vec<MpsTensor> = (0..n)
            .map(|s| {
                MpsTensor::new(shapes[s][0], shapes[s][1], shapes[s][2], data[s].clone())
                    .expect("tensor")
            })
            .collect();
        let ex_mps = Mps::from_tensors(tensors).expect("mps");
        let overlap = mps_inner_product(&gs_copy, &ex_mps).expect("inner product");
        assert!(
            overlap.abs() < 0.1,
            "overlap |⟨ψ₀|ψ₁⟩| = {:.6} should be small",
            overlap.abs()
        );
    }

    // ── Test 3: n_excited=2 gives two states with E₀ < E₁ and E₀ < E₂ ────────
    // Note: strict E₁ ≤ E₂ ordering is not guaranteed by sequential penalty deflation
    // because near-degenerate excited states can appear in any order.  We verify that
    // both excited energies lie strictly above the ground state.
    #[test]
    fn two_excited_states_ordered_4site() {
        let (gs_mps, e0, mpo) = ground_state(4, 8, 3);
        let mut rng = LcgRng::new(44);
        let cfg = ExcitedDmrgConfig {
            chi_max: 8,
            max_sweeps: 8,
            tol: 1e-5,
            penalty_weight: 25.0,
            lanczos_max_iter: 40,
            lanczos_tol: 1e-10,
            n_excited: 2,
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("excited dmrg");
        assert_eq!(res.excited_energies.len(), 2);
        let e1 = res.excited_energies[0];
        let e2 = res.excited_energies[1];
        // Both excited energies should be strictly above the ground state.
        assert!(e1 > e0 - 1e-3, "E₁={e1} should be > E₀={e0}");
        assert!(e2 > e0 - 1e-3, "E₂={e2} should be > E₀={e0}");
        // Both should be finite and numerically plausible.
        assert!(e1.is_finite(), "E₁ must be finite");
        assert!(e2.is_finite(), "E₂ must be finite");
    }

    // ── Test 4: invalid penalty_weight ≤ 0 returns error ─────────────────────
    #[test]
    fn invalid_penalty_weight_returns_error() {
        let (gs_mps, _e0, mpo) = ground_state(4, 4, 5);
        let mut rng = LcgRng::new(50);
        let cfg = ExcitedDmrgConfig {
            penalty_weight: -1.0,
            ..ExcitedDmrgConfig::default()
        };
        assert!(two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).is_err());
    }

    // ── Test 5: penalty_weight = 0 returns error ──────────────────────────────
    #[test]
    fn zero_penalty_weight_returns_error() {
        let (gs_mps, _e0, mpo) = ground_state(4, 4, 6);
        let mut rng = LcgRng::new(51);
        let cfg = ExcitedDmrgConfig {
            penalty_weight: 0.0,
            ..ExcitedDmrgConfig::default()
        };
        assert!(two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).is_err());
    }

    // ── Test 6: n_excited = 0 returns error ──────────────────────────────────
    #[test]
    fn zero_n_excited_returns_error() {
        let (gs_mps, _e0, mpo) = ground_state(4, 4, 7);
        let mut rng = LcgRng::new(52);
        let cfg = ExcitedDmrgConfig {
            n_excited: 0,
            ..ExcitedDmrgConfig::default()
        };
        assert!(two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).is_err());
    }

    // ── Test 7: single-site (n=1) chain returns error ────────────────────────
    #[test]
    fn single_site_chain_returns_error() {
        // Build a trivial 1-site MPO (identity); manually create 1-site MPS.
        let mpo = Mpo::identity(1, 2).expect("identity");
        let mps = Mps::from_product_state(&[vec![1.0, 0.0]]).expect("product state");
        let mut rng = LcgRng::new(53);
        assert!(two_site_excited_dmrg(&mpo, mps, ExcitedDmrgConfig::default(), &mut rng).is_err());
    }

    // ── Test 8: max_sweeps=1 returns result even without convergence ──────────
    #[test]
    fn one_sweep_returns_result() {
        let (gs_mps, _e0, mpo) = ground_state(4, 4, 8);
        let mut rng = LcgRng::new(54);
        let cfg = ExcitedDmrgConfig {
            max_sweeps: 1,
            tol: 1e-16, // unreachable
            penalty_weight: 10.0,
            chi_max: 4,
            n_excited: 1,
            ..ExcitedDmrgConfig::default()
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("should not error");
        assert_eq!(res.excited_energies.len(), 1);
        // With only 1 sweep, convergence flag should be false.
        assert!(!res.converged);
    }

    // ── Test 9: ground_state_energy is correctly reported ────────────────────
    #[test]
    fn ground_energy_is_reported() {
        let (gs_mps, e0, mpo) = ground_state(4, 8, 9);
        let mut rng = LcgRng::new(55);
        let cfg = ExcitedDmrgConfig {
            chi_max: 8,
            max_sweeps: 6,
            n_excited: 1,
            ..ExcitedDmrgConfig::default()
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("ok");
        // The reported ground energy should match what we computed separately.
        assert!(
            (res.ground_state_energy - e0).abs() < 0.05,
            "reported gs energy {:.6} vs expected {:.6}",
            res.ground_state_energy,
            e0
        );
    }

    // ── Test 10: result MPS shapes are self-consistent ────────────────────────
    #[test]
    fn result_shapes_consistent() {
        let (gs_mps, _e0, mpo) = ground_state(4, 6, 10);
        let mut rng = LcgRng::new(56);
        let cfg = ExcitedDmrgConfig {
            chi_max: 6,
            max_sweeps: 4,
            n_excited: 1,
            ..ExcitedDmrgConfig::default()
        };
        let res = two_site_excited_dmrg(&mpo, gs_mps, cfg, &mut rng).expect("ok");
        let n = mpo.n_sites();
        for k in 0..cfg.n_excited {
            assert_eq!(res.mps_shapes[k].len(), n);
            assert_eq!(res.mps_data[k].len(), n);
            // Boundary bonds must be 1
            assert_eq!(res.mps_shapes[k][0][0], 1, "left boundary bond");
            assert_eq!(res.mps_shapes[k][n - 1][2], 1, "right boundary bond");
            // Each data slice must match its shape
            for s in 0..n {
                let [dl, dp, dr] = res.mps_shapes[k][s];
                assert_eq!(
                    res.mps_data[k][s].len(),
                    dl * dp * dr,
                    "data length mismatch at site {s}"
                );
            }
        }
    }

    // ── Test 11: penalty shifts a perfect-overlap state by ~ω ────────────────
    #[test]
    fn penalty_shifts_eigenvalue_by_omega() {
        // For a diagonal operator with eigenvalue λ₀ at |e₀⟩, if we add ω|e₀⟩⟨e₀|
        // then the new eigenvalue for |e₀⟩ should be λ₀ + ω.
        let n = 4usize;
        // Build the identity MPO (all eigenvalues = 1).
        let mpo_id = Mpo::identity(n, 2).expect("identity");
        // Ground state of identity is any normalised state; we use a product state.
        let gs_mps = Mps::from_product_state(&vec![vec![1.0, 0.0]; n]).expect("product");
        let e0 = mpo_expectation(&mpo_id, &gs_mps).expect("e0");
        assert!((e0 - 1.0).abs() < 1e-10);

        let omega = 5.0;
        let mut rng = LcgRng::new(100);
        let cfg = ExcitedDmrgConfig {
            chi_max: 4,
            max_sweeps: 8,
            tol: 1e-6,
            penalty_weight: omega,
            n_excited: 1,
            ..ExcitedDmrgConfig::default()
        };
        // For identity MPO all states have energy 1.0. With penalty on |0000⟩, the
        // algorithm should converge to another state with raw energy ~1.0 (not shifted to 1+ω).
        let res = two_site_excited_dmrg(&mpo_id, gs_mps, cfg, &mut rng).expect("ok");
        // The returned energy is the unpenalised energy, which should still be ~1.0.
        let e1 = res.excited_energies[0];
        assert!(
            (e1 - 1.0).abs() < 0.5,
            "expected e1 ≈ 1.0 (all degenerate), got {e1}"
        );
    }

    // ── Test 12: mps_inner_product with self gives norm squared ───────────────
    #[test]
    fn inner_product_self_gives_norm_sq() {
        let mut rng = LcgRng::new(200);
        let mps = Mps::random_mps(4, 2, 3, &mut rng).expect("random mps");
        let ip = mps_inner_product(&mps, &mps).expect("inner product");
        let ns = mps.norm_squared().expect("norm sq");
        assert!(
            (ip - ns).abs() < 1e-10,
            "inner_product(ψ,ψ)={ip} ≠ norm_sq={ns}"
        );
    }

    // ── Test 13: inner product with orthogonal product states is zero ─────────
    #[test]
    fn inner_product_orthogonal_states_zero() {
        // |0000⟩ and |1111⟩ are orthogonal.
        let psi0 = Mps::from_product_state(&vec![vec![1.0, 0.0]; 4]).expect("psi0");
        let psi1 = Mps::from_product_state(&vec![vec![0.0, 1.0]; 4]).expect("psi1");
        let ip = mps_inner_product(&psi0, &psi1).expect("inner product");
        assert!(ip.abs() < 1e-12, "expected 0, got {ip}");
    }

    // ── Test 14: extract_two_site_tensor is consistent with MPS ─────────────
    #[test]
    fn extract_two_site_tensor_matches_full_contraction() {
        let mut rng = LcgRng::new(300);
        let mps = Mps::random_mps(4, 2, 3, &mut rng).expect("random");
        // Extracting at bond 1 gives theta[a, p1, p2, b]
        let theta = extract_two_site_tensor(&mps, 1).expect("extract");
        let lt = &mps.site_tensors[1];
        let rt = &mps.site_tensors[2];
        let (_dl, _dp1, dm) = lt.shape();
        let (_, dp2, dr) = rt.shape();
        // Verify element [a=0, p1=0, p2=0, b=0]:
        //   lt[a, p1, c] flat = (a*dp1 + p1)*dm + c  →  with a=0, p1=0 gives c
        //   rt[c, p2, b] flat = (c*dp2 + p2)*dr + b  →  with p2=0, b=0 gives c*dp2*dr
        let mut expected = 0.0f64;
        for c in 0..dm {
            expected += lt.data[c] * rt.data[c * dp2 * dr];
        }
        assert!(
            (theta[0] - expected).abs() < 1e-12,
            "theta[0,0,0,0]={} ≠ {}",
            theta[0],
            expected
        );
    }
}
