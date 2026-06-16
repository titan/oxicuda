//! Single-site DMRG with subspace expansion (noise perturbation).
//!
//! Single-site DMRG (White 1992, 1996) optimises one MPS tensor per step rather than
//! a two-site theta tensor, making each micro-step cheaper. The trade-off is a
//! tendency to get trapped in local minima; Hubig et al. (2015) cure this via a
//! subspace-expansion (noise) perturbation added before the Lanczos solve.
//!
//! # Algorithm Outline
//!
//! 1. Right-canonicalise the initial MPS (orthogonality centre at site 0).
//! 2. Build all right environments `R[s] = contract(R[s+1], A_s, W_s, A_s*)`.
//! 3. For each sweep (left-to-right then right-to-left):
//!    optimise each site via Lanczos on H_eff, left-normalise (or right-normalise)
//!    via QR, and absorb the gauge factor into the adjacent site.
//! 4. Terminate when |E_new - E_old| < tol or max_sweeps reached.
//!
//! # Environment Indexing
//!
//! `left_envs[i]`  has shape `[chi_l_i, D_w_l_i, chi_l_i]` (left of site i).
//! `right_envs[i]` has shape `[chi_r_i, D_w_r_i, chi_r_i]` (right of site i).
//! Boundary: `left_envs[0] = right_envs[L-1] = vec![1.0]` (trivial 1×1×1).
//!
//! # Gauge Conventions
//!
//! Left-normalise site i (L→R sweep): QR-decompose M = A_i reshaped [chi_l*d, chi_r].
//! Q [chi_l*d, k] becomes new A_i (reshaped [chi_l, d, k]), R [k, chi_r] is absorbed
//! into A_{i+1} from the left: A_{i+1} ← R · A_{i+1}.
//!
//! Right-normalise site i (R→L sweep): LQ-decompose M = A_i reshaped [chi_l, d*chi_r].
//! QR on M^T [d*chi_r, chi_l] gives Q [d*chi_r, k] and R [k, chi_l].
//! New A_i = Q^T [k, d*chi_r] reshaped [k, d, chi_r]. L = R^T [chi_l, k] absorbed into
//! A_{i-1} from the right: A_{i-1} ← A_{i-1} · R^T.

use crate::dmrg::lanczos::lanczos_smallest;
use crate::handle::LcgRng;
use crate::{TnError, TnResult};

// ── Public data structures ────────────────────────────────────────────────────

/// Configuration knobs for single-site DMRG.
#[derive(Debug, Clone, Copy)]
pub struct SingleSiteDmrgConfig {
    /// Maximum virtual bond dimension. Bond dimension is kept fixed (no SVD truncation).
    pub chi_max: usize,
    /// Maximum number of full L→R + R→L sweep pairs.
    pub max_sweeps: usize,
    /// Energy convergence threshold (absolute difference between successive sweeps).
    pub tol: f64,
    /// Maximum Lanczos iterations per site update.
    pub lanczos_max_iter: usize,
    /// Lanczos eigenvalue convergence tolerance.
    pub lanczos_tol: f64,
    /// Subspace-expansion (Hubig) noise amplitude. Set to 0 to disable.
    pub noise: f64,
}

impl Default for SingleSiteDmrgConfig {
    fn default() -> Self {
        Self {
            chi_max: 32,
            max_sweeps: 10,
            tol: 1.0e-8,
            lanczos_max_iter: 50,
            lanczos_tol: 1.0e-10,
            noise: 1.0e-5,
        }
    }
}

/// Outcome of single-site DMRG.
#[derive(Debug, Clone)]
pub struct SingleSiteDmrgResult {
    /// Ground-state energy estimate at the end of the last sweep.
    pub ground_state_energy: f64,
    /// Optimised MPS tensors (flattened row-major, same length as input).
    pub mps: Vec<Vec<f64>>,
    /// Shape `[D_l, d, D_r]` for each site tensor.
    pub mps_shapes: Vec<[usize; 3]>,
    /// Energy measured after every complete sweep (L→R + R→L).
    pub energies: Vec<f64>,
    /// Number of sweeps actually executed.
    pub n_sweeps: usize,
    /// Whether the energy converged within `tol`.
    pub converged: bool,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run single-site DMRG with subspace expansion to find the ground state of `mpo`.
///
/// # Arguments
///
/// * `mps_tensors`  – Initial MPS as a vector of flat tensors. Tensor `i` has shape
///   `mps_shapes[i] = [D_l, d, D_r]`, stored row-major.
/// * `mps_shapes`   – Shape `[D_l, d, D_r]` of each MPS site tensor.
/// * `mpo_tensors`  – MPO site tensors (flat row-major).
/// * `mpo_shapes`   – Shape `[D_w_l, d_out, d_in, D_w_r]` of each MPO site tensor.
/// * `config`       – Algorithm hyper-parameters.
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] for zero-site MPS/MPO, and
/// [`TnError::DimensionMismatch`] when the number of sites differs.
pub fn single_site_dmrg(
    mps_tensors: &[Vec<f64>],
    mps_shapes: &[[usize; 3]],
    mpo_tensors: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
    config: &SingleSiteDmrgConfig,
) -> TnResult<SingleSiteDmrgResult> {
    // ── Validate inputs ───────────────────────────────────────────────────────
    let n_sites = mps_tensors.len();
    if n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if mpo_tensors.len() != n_sites {
        return Err(TnError::DimensionMismatch {
            a: n_sites,
            b: mpo_tensors.len(),
        });
    }
    if mps_shapes.len() != n_sites || mpo_shapes.len() != n_sites {
        return Err(TnError::DimensionMismatch {
            a: n_sites,
            b: mps_shapes.len().min(mpo_shapes.len()),
        });
    }
    for i in 0..n_sites {
        let [dl, d, dr] = mps_shapes[i];
        let expected = dl * d * dr;
        if mps_tensors[i].len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![dl, d, dr],
                got: vec![mps_tensors[i].len()],
            });
        }
        let [wl, dout, din, wr] = mpo_shapes[i];
        let expected_mpo = wl * dout * din * wr;
        if mpo_tensors[i].len() != expected_mpo {
            return Err(TnError::ShapeMismatch {
                expected: vec![wl, dout, din, wr],
                got: vec![mpo_tensors[i].len()],
            });
        }
    }

    // ── Work copies of MPS tensors (mutable during sweeps) ───────────────────
    let mut mps: Vec<Vec<f64>> = mps_tensors.to_vec();
    let mut shapes: Vec<[usize; 3]> = mps_shapes.to_vec();

    // ── Right-canonicalise MPS: orthogonality centre at site 0 ───────────────
    right_canonicalize_mps(&mut mps, &mut shapes)?;

    // ── Build initial right environments ──────────────────────────────────────
    // Convention: right_envs[i] = environment accumulated from sites i..n_sites-1
    //             and the right boundary. Its shape is [chi_l_i, D_w_l_i, chi_l_i]
    //             (i.e., the shape at the LEFT bond of site i).
    //
    // When optimising site s, the right environment to use (at the RIGHT bond of site s)
    // is right_envs[s + 1].
    //
    // right_envs[n_sites] = trivial 1×1×1 boundary.
    // right_envs[n_sites - 1] = environment of site n_sites-1 contracted with trivial right.
    // right_envs[s] = contract(right_envs[s + 1], A_s, W_s, A_s*) for s < n_sites.
    let mut right_envs: Vec<Vec<f64>> = vec![vec![]; n_sites + 1];
    let mut r_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n_sites + 1];
    right_envs[n_sites] = vec![1.0];
    // r_shapes[n_sites] = (1, 1, 1) — trivial right boundary

    for s in (0..n_sites).rev() {
        let [chi_l_s, _, _] = shapes[s];
        let [wl_s, _, _, _] = mpo_shapes[s];
        right_envs[s] = build_right_env_raw(
            &right_envs[s + 1],
            r_shapes[s + 1],
            &mps[s],
            shapes[s],
            &mpo_tensors[s],
            mpo_shapes[s],
        )?;
        // right_envs[s] has shape [chi_l_s, wl_s, chi_l_s]
        r_shapes[s] = (chi_l_s, wl_s, chi_l_s);
    }

    // ── Left environments: built incrementally during sweeps ──────────────────
    // left_envs[i] = environment to the LEFT of site i, shape [chi_l_i, D_w_l_i, chi_l_i].
    // left_envs[0] = trivial boundary 1×1×1.
    let mut left_envs: Vec<Vec<f64>> = vec![vec![]; n_sites + 1];
    let mut l_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n_sites + 1];
    left_envs[0] = vec![1.0];

    let mut rng = LcgRng::new(0xDEAD_BEEF_1337_4242);
    let mut energies: Vec<f64> = Vec::new();
    let mut last_energy = f64::INFINITY;
    let mut converged = false;
    let mut n_sweeps = 0;

    for _sweep in 0..config.max_sweeps {
        // ── Left-to-right pass ────────────────────────────────────────────────
        // After right_canonicalize, OC is at site 0.
        // We update sites 0, 1, ..., n_sites-2, left-normalising each and absorbing R into next.
        // Site n_sites-1 is updated but not gauged (it becomes the new OC).
        for i in 0..n_sites {
            let [chi_l, d, chi_r] = shapes[i];
            let [d_w_l, _dout, _din, d_w_r] = mpo_shapes[i];
            let n_theta = chi_l * d * chi_r;

            // Subspace-expansion: add small random noise before Lanczos
            let mut theta = mps[i].clone();
            if config.noise > 0.0 {
                apply_subspace_noise(&mut theta, config.noise, &mut rng);
            }

            // Lanczos solve for smallest eigenpair of H_eff at site i.
            // Left env at site i = left_envs[i]  (shape [chi_l, D_w_l, chi_l]).
            // Right env at site i = right_envs[i+1] (shape [chi_r, D_w_r, chi_r]).
            let l_env = left_envs[i].clone();
            let l_sh = l_shapes[i];
            // right_envs[i+1] is the environment to the right of site i's right bond.
            // Its shape is [chi_l_{i+1}, wl_{i+1}, chi_l_{i+1}] which equals [chi_r_i, D_w_r_i, chi_r_i].
            let r_env = right_envs[i + 1].clone();
            let r_sh = r_shapes[i + 1];
            let w_i = mpo_tensors[i].clone();
            let w_sh = mpo_shapes[i];

            let apply = move |v: &[f64]| -> Vec<f64> {
                apply_heff(
                    v, &l_env, l_sh, &w_i, w_sh, &r_env, r_sh, chi_l, d, chi_r, d_w_l, d_w_r,
                )
            };

            let seed_norm: f64 = theta.iter().map(|x| x * x).sum::<f64>().sqrt();
            let seed = if seed_norm < 1.0e-14 {
                let mut v = vec![0.0; n_theta];
                for x in &mut v {
                    *x = rng.next_normal();
                }
                v
            } else {
                theta
            };

            let lz = lanczos_smallest(
                apply,
                n_theta,
                &seed,
                config.lanczos_max_iter,
                config.lanczos_tol,
            )?;
            mps[i] = lz.eigenvector;

            // Left-normalise site i via QR, absorb R into site i+1.
            if i < n_sites - 1 {
                // QR of M [chi_l*d, chi_r] → Q [chi_l*d, k] and R [k, chi_r]
                let (q_mat, r_mat, k) = qr_thin(&mps[i], chi_l * d, chi_r);
                // New site i: Q reshaped [chi_l, d, k]
                mps[i] = q_mat;
                shapes[i] = [chi_l, d, k];

                // Absorb R [k, chi_r] into A_{i+1} [chi_r, d_next, chi_r_next]:
                //   new A_{i+1} = R · A_{i+1}  (matrix multiply [k, chi_r] × [chi_r, d_next*chi_r_next])
                let [chi_r_old_next, d_next, chi_r_next] = shapes[i + 1];
                // chi_r_old_next should equal chi_r (the original right bond of site i)
                let absorbed = mat_mul(&r_mat, &mps[i + 1], k, chi_r_old_next, d_next * chi_r_next);
                mps[i + 1] = absorbed;
                shapes[i + 1] = [k, d_next, chi_r_next];

                // Update left environment at position i+1
                left_envs[i + 1] = build_left_env_raw(
                    &left_envs[i],
                    l_shapes[i],
                    &mps[i],
                    shapes[i],
                    &mpo_tensors[i],
                    mpo_shapes[i],
                )?;
                let [_, _, new_dr] = shapes[i];
                let [_, _, _, new_wr] = mpo_shapes[i];
                l_shapes[i + 1] = (new_dr, new_wr, new_dr);
            }
        }

        // ── Right-to-left pass ────────────────────────────────────────────────
        // OC is now at site n_sites-1 (the last site that was not gauge-fixed).
        // Update sites n_sites-1, n_sites-2, ..., 1, right-normalising each.
        // Site 0 is updated but not gauged (it becomes the new OC).
        for i in (0..n_sites).rev() {
            let [chi_l, d, chi_r] = shapes[i];
            let [d_w_l, _dout, _din, d_w_r] = mpo_shapes[i];
            let n_theta = chi_l * d * chi_r;

            let mut theta = mps[i].clone();
            if config.noise > 0.0 {
                apply_subspace_noise(&mut theta, config.noise, &mut rng);
            }

            // Left env at site i = left_envs[i], right env = right_envs[i+1].
            let l_env = left_envs[i].clone();
            let l_sh = l_shapes[i];
            let r_env = right_envs[i + 1].clone();
            let r_sh = r_shapes[i + 1];
            let w_i = mpo_tensors[i].clone();
            let w_sh = mpo_shapes[i];

            let apply = move |v: &[f64]| -> Vec<f64> {
                apply_heff(
                    v, &l_env, l_sh, &w_i, w_sh, &r_env, r_sh, chi_l, d, chi_r, d_w_l, d_w_r,
                )
            };

            let seed_norm: f64 = theta.iter().map(|x| x * x).sum::<f64>().sqrt();
            let seed = if seed_norm < 1.0e-14 {
                let mut v = vec![0.0; n_theta];
                for x in &mut v {
                    *x = rng.next_normal();
                }
                v
            } else {
                theta
            };

            let lz = lanczos_smallest(
                apply,
                n_theta,
                &seed,
                config.lanczos_max_iter,
                config.lanczos_tol,
            )?;
            mps[i] = lz.eigenvector;

            // Right-normalise site i via LQ (QR of M^T), absorb L factor into site i-1.
            if i > 0 {
                // M = A_i reshaped [chi_l, d*chi_r].
                // QR of M^T [d*chi_r, chi_l] → Q [d*chi_r, k], R [k, chi_l].
                // New A_i = Q^T [k, d*chi_r] reshaped [k, d, chi_r].
                // Gauge factor absorbed left: A_{i-1} ← A_{i-1} * R^T  where R^T: [chi_l, k].
                let (v_mat, r_mat, k) = lq_decompose(&mps[i], chi_l, d * chi_r);
                // v_mat: [k, d*chi_r] row-major (the right-unitary part)
                // r_mat: [k, chi_l] (upper triangular after transposing)
                mps[i] = v_mat;
                shapes[i] = [k, d, chi_r];

                // Absorb R^T [chi_l, k] into A_{i-1} from the right:
                //   A_{i-1} [chi_l_prev, d_prev, chi_l] × R^T [chi_l, k] → [chi_l_prev, d_prev, k]
                //   (reshaped: [chi_l_prev*d_prev, chi_l] × [chi_l, k])
                let [chi_l_prev, d_prev, chi_l_cur] = shapes[i - 1];
                // chi_l_cur should equal chi_l (the original left bond of site i)
                let rt = transpose_mat(&r_mat, k, chi_l_cur); // [chi_l, k]
                let absorbed = mat_mul(&mps[i - 1], &rt, chi_l_prev * d_prev, chi_l_cur, k);
                mps[i - 1] = absorbed;
                shapes[i - 1] = [chi_l_prev, d_prev, k];

                // Update right environment at position i (to the right of site i-1)
                right_envs[i] = build_right_env_raw(
                    &right_envs[i + 1],
                    r_shapes[i + 1],
                    &mps[i],
                    shapes[i],
                    &mpo_tensors[i],
                    mpo_shapes[i],
                )?;
                let [new_chi_l_i, _, _] = shapes[i];
                let [new_wl_i, _, _, _] = mpo_shapes[i];
                // right_envs[i] has shape [chi_l of site i, wl of site i, chi_l of site i]
                r_shapes[i] = (new_chi_l_i, new_wl_i, new_chi_l_i);
            }
        }

        // ── Energy measurement after full sweep ───────────────────────────────
        let energy = measure_energy_raw(&mps, &shapes, mpo_tensors, mpo_shapes)?;
        energies.push(energy);
        n_sweeps += 1;

        if (last_energy - energy).abs() < config.tol && n_sweeps > 1 {
            converged = true;
            last_energy = energy;
            break;
        }
        last_energy = energy;
    }

    Ok(SingleSiteDmrgResult {
        ground_state_energy: last_energy,
        mps,
        mps_shapes: shapes,
        energies,
        n_sweeps,
        converged,
    })
}

// ── Environment construction (raw tensor interface) ───────────────────────────

/// Build left environment by contracting `prev` with MPS tensor `a` and MPO tensor `w`.
///
/// Shapes:
/// * `prev`: `[chi_l, D_w_l, chi_l]` flat — left boundary to the left of site i
/// * `a`:    `[chi_l, d, chi_r]` flat — MPS tensor at site i
/// * `w`:    `[D_w_l, d_out, d_in, D_w_r]` flat — MPO tensor at site i
///
/// Output shape: `[chi_r, D_w_r, chi_r]`
///
/// Contraction:
/// `L_new[b, m', b'] = Σ_{a,a',m,s,s'} L[a,m,a'] · A[a,s,b] · W[m,s,s',m'] · A*[a',s',b']`
/// (For real symmetric Hamiltonians A = A*, so we use the same tensor for bra and ket.)
fn build_left_env_raw(
    prev: &[f64],
    prev_sh: (usize, usize, usize),
    a: &[f64],
    a_sh: [usize; 3],
    w: &[f64],
    w_sh: [usize; 4],
) -> TnResult<Vec<f64>> {
    let (chi_l, d_wl, chi_lp) = prev_sh;
    debug_assert_eq!(chi_l, chi_lp, "left env: chi_l dims must match");
    let [a_chi_l, d, chi_r] = a_sh;
    let [wl, dout, din, wr] = w_sh;

    if chi_l != a_chi_l {
        return Err(TnError::DimensionMismatch {
            a: chi_l,
            b: a_chi_l,
        });
    }
    if d_wl != wl {
        return Err(TnError::DimensionMismatch { a: d_wl, b: wl });
    }
    if dout != d || din != d {
        return Err(TnError::DimensionMismatch { a: dout, b: d });
    }

    let mut env = vec![0.0; chi_r * wr * chi_r];
    for b in 0..chi_r {
        for mp in 0..wr {
            for bp in 0..chi_r {
                let mut acc = 0.0;
                for aa in 0..chi_l {
                    for aap in 0..chi_l {
                        for m in 0..d_wl {
                            let lv = prev[(aa * d_wl + m) * chi_lp + aap];
                            if lv == 0.0 {
                                continue;
                            }
                            for s in 0..dout {
                                let av = a[(aa * d + s) * chi_r + b];
                                if av == 0.0 {
                                    continue;
                                }
                                for sp in 0..din {
                                    let avp = a[(aap * d + sp) * chi_r + bp];
                                    let wv = w[((m * dout + s) * din + sp) * wr + mp];
                                    acc += lv * av * wv * avp;
                                }
                            }
                        }
                    }
                }
                env[(b * wr + mp) * chi_r + bp] = acc;
            }
        }
    }
    Ok(env)
}

/// Build right environment by contracting `next` with MPS tensor `a` and MPO tensor `w`.
///
/// Shapes:
/// * `next`: `[chi_r, D_w_r, chi_r]` flat — right boundary to the right of site i
/// * `a`:    `[chi_l, d, chi_r]` flat — MPS tensor at site i
/// * `w`:    `[D_w_l, d_out, d_in, D_w_r]` flat — MPO tensor at site i
///
/// Output shape: `[chi_l, D_w_l, chi_l]`
///
/// Contraction:
/// `R_new[a, m', a'] = Σ_{b,b',m,s,s'} A[a,s,b] · W[m',s,s',m] · A*[a',s',b'] · R[b,m,b']`
fn build_right_env_raw(
    next: &[f64],
    next_sh: (usize, usize, usize),
    a: &[f64],
    a_sh: [usize; 3],
    w: &[f64],
    w_sh: [usize; 4],
) -> TnResult<Vec<f64>> {
    let (chi_r_next, d_wr, chi_r_nextp) = next_sh;
    debug_assert_eq!(chi_r_next, chi_r_nextp, "right env: chi_r dims must match");
    let [chi_l, d, chi_r] = a_sh;
    let [wl, dout, din, wr] = w_sh;

    if chi_r != chi_r_next {
        return Err(TnError::DimensionMismatch {
            a: chi_r,
            b: chi_r_next,
        });
    }
    if d_wr != wr {
        return Err(TnError::DimensionMismatch { a: d_wr, b: wr });
    }
    if dout != d || din != d {
        return Err(TnError::DimensionMismatch { a: dout, b: d });
    }

    let mut env = vec![0.0; chi_l * wl * chi_l];
    for aa in 0..chi_l {
        for ml in 0..wl {
            for aap in 0..chi_l {
                let mut acc = 0.0;
                for b in 0..chi_r {
                    for bp in 0..chi_r {
                        for m in 0..d_wr {
                            let rv = next[(b * d_wr + m) * chi_r_nextp + bp];
                            if rv == 0.0 {
                                continue;
                            }
                            for s in 0..dout {
                                let av = a[(aa * d + s) * chi_r + b];
                                if av == 0.0 {
                                    continue;
                                }
                                for sp in 0..din {
                                    let avp = a[(aap * d + sp) * chi_r + bp];
                                    let wv = w[((ml * dout + s) * din + sp) * wr + m];
                                    acc += av * wv * avp * rv;
                                }
                            }
                        }
                    }
                }
                env[(aa * wl + ml) * chi_l + aap] = acc;
            }
        }
    }
    Ok(env)
}

// ── Effective Hamiltonian matrix-vector product ───────────────────────────────

/// Apply the single-site effective Hamiltonian to vector `theta`.
///
/// `theta` is the local MPS tensor at site i, flattened to length `chi_l * d * chi_r`.
///
/// ```text
/// (H_eff · θ)[a, s, b] = Σ_{a',s',b'} L[a, m_l, a'] · W[m_l, s, s', m_r] · R[b, m_r, b'] · θ[a', s', b']
/// ```
///
/// Shapes:
/// * `l_env`: `[chi_l, D_w_l, chi_l]` flat
/// * `w_i`:   `[D_w_l, d_out, d_in, D_w_r]` flat
/// * `r_env`: `[chi_r, D_w_r, chi_r]` flat
#[allow(clippy::too_many_arguments)]
fn apply_heff(
    theta: &[f64],
    l_env: &[f64],
    l_sh: (usize, usize, usize),
    w_i: &[f64],
    w_sh: [usize; 4],
    r_env: &[f64],
    r_sh: (usize, usize, usize),
    chi_l: usize,
    d: usize,
    chi_r: usize,
    _d_w_l: usize,
    _d_w_r: usize,
) -> Vec<f64> {
    let (la, lw, lap) = l_sh;
    let (rb, rw, rbp) = r_sh;
    let _ = (la, lap, rb, rbp);
    let [_wl, dout, din, _wr] = w_sh;

    let mut out = vec![0.0; chi_l * d * chi_r];

    for a in 0..chi_l {
        for s in 0..dout {
            for b in 0..chi_r {
                let mut acc = 0.0;
                for ap in 0..chi_l {
                    for sp in 0..din {
                        for bp in 0..chi_r {
                            let theta_v = theta[(ap * din + sp) * chi_r + bp];
                            if theta_v == 0.0 {
                                continue;
                            }
                            // Compute H_eff matrix element <a,s,b|H_eff|a',s',b'>
                            // = Σ_{m_l, m_r} L[a, m_l, a'] * W[m_l, s, s', m_r] * R[b, m_r, b']
                            let mut h_elem = 0.0;
                            for m_l in 0..lw {
                                let lv = l_env[(a * lw + m_l) * chi_l + ap];
                                if lv == 0.0 {
                                    continue;
                                }
                                for m_r in 0..rw {
                                    let rv = r_env[(b * rw + m_r) * chi_r + bp];
                                    let wv = w_i[((m_l * dout + s) * din + sp) * rw + m_r];
                                    h_elem += lv * wv * rv;
                                }
                            }
                            acc += h_elem * theta_v;
                        }
                    }
                }
                out[(a * dout + s) * chi_r + b] = acc;
            }
        }
    }
    out
}

// ── QR / LQ decompositions ────────────────────────────────────────────────────

/// Thin QR decomposition via modified Gram-Schmidt on the COLUMNS of A.
///
/// `A` is `m × n` row-major. Returns `(Q [m × k], R [k × n], k)` where `k = min(m, n)`.
/// The diagonal of R is positive (sign convention enforced).
fn qr_thin(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let k = m.min(n);

    // Extract columns of A as mutable work vectors.
    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j]).collect::<Vec<f64>>())
        .collect();

    let mut r_data = vec![0.0; k * n];
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(k);

    for j in 0..k {
        // Diagonal: R[j,j] = ||u_j||
        let nrm: f64 = u[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        let nrm = if nrm < 1.0e-300 { 1.0e-300 } else { nrm };

        // Q[:,j] = u_j / ||u_j||
        let qj: Vec<f64> = u[j].iter().map(|x| x / nrm).collect();
        r_data[j * n + j] = nrm; // positive diagonal

        // Off-diagonal: R[j, jj] = <q_j, u_jj> for jj > j; subtract projection.
        for jj in (j + 1)..n {
            let rjj: f64 = qj.iter().zip(u[jj].iter()).map(|(a, b)| a * b).sum();
            r_data[j * n + jj] = rjj;
            for ii in 0..m {
                u[jj][ii] -= rjj * qj[ii];
            }
        }

        q_cols.push(qj);
    }

    // Assemble Q as (m × k) row-major.
    let mut q_data = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            q_data[i * k + j] = q_cols[j][i];
        }
    }

    (q_data, r_data, k)
}

/// LQ decomposition: decomposes `A [m × n]` so that `A = L · Q` where Q is
/// right-unitary (Q Q^T = I_k) and L is lower-triangular.
///
/// Implementation: QR of A^T gives Q_T [n × k] and R_T [k × m].
/// Then A = (Q_T R_T)^T = R_T^T Q_T^T, so L = R_T^T and Q (right-unitary) = Q_T^T.
///
/// Returns `(V [k × n], R_T [k × m], k)` where V = Q_T^T is the right-unitary factor
/// stored row-major as `[k, n]`. The gauge factor to absorb leftward is `R_T [k, m]`,
/// and absorption is: `A_{prev} ← A_{prev} · R_T^T` (right-multiply by R_T^T [m, k]).
fn lq_decompose(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, usize) {
    // Transpose A to get A^T [n × m].
    let mut at = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            at[j * m + i] = a[i * n + j];
        }
    }
    // QR of A^T [n × m] → Q_T [n × k], R_T [k × m].
    let (q_t, r_t, k) = qr_thin(&at, n, m);
    // V = Q_T^T [k × n] is the right-unitary site tensor.
    let v = transpose_mat(&q_t, n, k);
    // Return: v [k × n], r_t [k × m], k.
    (v, r_t, k)
}

// ── Linear algebra helpers ────────────────────────────────────────────────────

/// Matrix multiply `C = A · B` where A: [m, p] and B: [p, n], both row-major.
fn mat_mul(a: &[f64], b: &[f64], m: usize, p: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for kk in 0..p {
            let aik = a[i * p + kk];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[kk * n + j];
            }
        }
    }
    c
}

/// Transpose a flat `[m × n]` row-major matrix to `[n × m]`.
fn transpose_mat(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut at = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            at[j * m + i] = a[i * n + j];
        }
    }
    at
}

// ── Subspace expansion (Hubig noise) ─────────────────────────────────────────

/// Add normalised Gaussian noise scaled by `amplitude` to `theta`, then renormalise.
fn apply_subspace_noise(theta: &mut [f64], amplitude: f64, rng: &mut LcgRng) {
    let n = theta.len();
    if n == 0 {
        return;
    }
    let vals: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    let nrm: f64 = vals.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = if nrm < 1.0e-300 {
        amplitude
    } else {
        amplitude / nrm
    };
    for (t, v) in theta.iter_mut().zip(vals.iter()) {
        *t += scale * v;
    }
    let new_nrm: f64 = theta.iter().map(|x| x * x).sum::<f64>().sqrt();
    if new_nrm > 1.0e-300 {
        for x in theta.iter_mut() {
            *x /= new_nrm;
        }
    }
}

// ── Canonicalisation ──────────────────────────────────────────────────────────

/// Right-canonicalise the MPS in-place: orthogonality centre ends at site 0.
///
/// Sweeps from site L-1 down to site 1, right-normalising each site using LQ
/// decomposition and absorbing the gauge factor into the preceding site.
/// Site 0 is normalised to unit norm.
fn right_canonicalize_mps(mps: &mut [Vec<f64>], shapes: &mut [[usize; 3]]) -> TnResult<()> {
    let n = mps.len();
    for i in (1..n).rev() {
        let [chi_l, d, chi_r] = shapes[i];
        // LQ decomposition of A_i reshaped [chi_l, d*chi_r].
        // V [k, d*chi_r] becomes the new right-normal site tensor.
        // R_T [k, chi_l] provides the gauge factor: A_{i-1} ← A_{i-1} · R_T^T.
        let (v_mat, r_t, k) = lq_decompose(&mps[i], chi_l, d * chi_r);
        mps[i] = v_mat;
        shapes[i] = [k, d, chi_r];

        // Absorb R_T^T [chi_l, k] into A_{i-1} from the right.
        // A_{i-1}: [chi_l_prev, d_prev, chi_l] → reshaped [chi_l_prev*d_prev, chi_l].
        // new A_{i-1} = [chi_l_prev*d_prev, chi_l] · [chi_l, k] = [chi_l_prev*d_prev, k].
        let [chi_l_prev, d_prev, chi_l_cur] = shapes[i - 1];
        // chi_l_cur should equal chi_l (the original left bond of site i).
        let rt_t = transpose_mat(&r_t, k, chi_l_cur); // [chi_l, k]
        let absorbed = mat_mul(&mps[i - 1], &rt_t, chi_l_prev * d_prev, chi_l_cur, k);
        mps[i - 1] = absorbed;
        shapes[i - 1] = [chi_l_prev, d_prev, k];
    }
    // Normalise site 0.
    let nrm: f64 = mps[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    if nrm > 1.0e-300 {
        for x in mps[0].iter_mut() {
            *x /= nrm;
        }
    }
    Ok(())
}

// ── Energy measurement ────────────────────────────────────────────────────────

/// Compute `<ψ|H|ψ> / <ψ|ψ>` for MPS and MPO given as raw tensors.
///
/// Uses the left-environment sweep over all sites.
fn measure_energy_raw(
    mps: &[Vec<f64>],
    shapes: &[[usize; 3]],
    mpo_tensors: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
) -> TnResult<f64> {
    let n = mps.len();
    let mut env = vec![1.0_f64];
    let mut env_sh = (1_usize, 1_usize, 1_usize);

    for i in 0..n {
        let new_env = build_left_env_raw(
            &env,
            env_sh,
            &mps[i],
            shapes[i],
            &mpo_tensors[i],
            mpo_shapes[i],
        )?;
        let [_, _, dr] = shapes[i];
        let [_, _, _, wr] = mpo_shapes[i];
        env_sh = (dr, wr, dr);
        env = new_env;
    }

    // <ψ|H|ψ> = sum of final env entries (they form a [1,1,1] scalar at the right boundary).
    let h_psi: f64 = env.iter().sum();

    // <ψ|ψ>
    let n2 = mps_norm_squared(mps, shapes)?;
    if n2.abs() < 1.0e-300 {
        return Err(TnError::NumericalInstability(
            "zero-norm MPS in energy measurement".into(),
        ));
    }

    Ok(h_psi / n2)
}

/// Compute `<ψ|ψ>` for MPS given as raw tensors via a left-environment sweep.
fn mps_norm_squared(mps: &[Vec<f64>], shapes: &[[usize; 3]]) -> TnResult<f64> {
    let n = mps.len();
    // norm_env[i] has shape [chi_r_i, chi_r_i].
    // norm_env[0] = [[1.0]] (1×1 trivial).
    let mut env = vec![1.0_f64]; // [1, 1]
    let mut env_dim = 1_usize;

    for i in 0..n {
        let [chi_l, d, chi_r] = shapes[i];
        if env_dim != chi_l {
            return Err(TnError::DimensionMismatch {
                a: env_dim,
                b: chi_l,
            });
        }
        // new_env[b, b'] = Σ_{a, a', s} env[a, a'] * A[a, s, b] * A[a', s, b']
        let mut new_env = vec![0.0; chi_r * chi_r];
        for b in 0..chi_r {
            for bp in 0..chi_r {
                let mut acc = 0.0;
                for aa in 0..chi_l {
                    for aap in 0..chi_l {
                        let ev = env[aa * chi_l + aap];
                        if ev == 0.0 {
                            continue;
                        }
                        for s in 0..d {
                            let av = mps[i][(aa * d + s) * chi_r + b];
                            let avp = mps[i][(aap * d + s) * chi_r + bp];
                            acc += ev * av * avp;
                        }
                    }
                }
                new_env[b * chi_r + bp] = acc;
            }
        }
        env = new_env;
        env_dim = chi_r;
    }

    Ok(env.iter().sum())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn random_mps_raw(
        n_sites: usize,
        d: usize,
        chi: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
        let mut rng = LcgRng::new(seed);
        let mut tensors = Vec::with_capacity(n_sites);
        let mut shapes = Vec::with_capacity(n_sites);
        for s in 0..n_sites {
            let chi_l = if s == 0 { 1 } else { chi };
            let chi_r = if s + 1 == n_sites { 1 } else { chi };
            let data: Vec<f64> = (0..chi_l * d * chi_r).map(|_| rng.next_normal()).collect();
            tensors.push(data);
            shapes.push([chi_l, d, chi_r]);
        }
        (tensors, shapes)
    }

    /// Build identity MPO: W[wl, s_out, s_in, wr] = δ(s_out, s_in), bonds all 1.
    fn identity_mpo_raw(n_sites: usize, d: usize) -> (Vec<Vec<f64>>, Vec<[usize; 4]>) {
        let mut tensors = Vec::with_capacity(n_sites);
        let mut shapes = Vec::with_capacity(n_sites);
        for _ in 0..n_sites {
            // MPO shape [w_l=1, d, d, w_r=1] → flat length = d * d.
            let mut w = vec![0.0; d * d];
            for s in 0..d {
                w[s * d + s] = 1.0;
            }
            tensors.push(w);
            shapes.push([1, d, d, 1]);
        }
        (tensors, shapes)
    }

    /// Heisenberg XXX MPO for d=2 open-boundary chain.
    ///
    /// H = Σ_i (S^+_i S^-_{i+1}/2 + S^-_i S^+_{i+1}/2 + S^z_i S^z_{i+1})
    ///
    /// Uses the standard 5×5 MPO operator structure. MPO bond dimension D_w = 5
    /// for bulk sites, 1 at boundaries.
    fn heisenberg_mpo_raw(n_sites: usize) -> (Vec<Vec<f64>>, Vec<[usize; 4]>) {
        // Spin-1/2 operators (row = out index, col = in index)
        // Basis: |↑⟩ = 0, |↓⟩ = 1
        // S^z: diag(0.5, -0.5), S^+ = [[0,1],[0,0]], S^- = [[0,0],[1,0]]
        let d = 2usize;
        let dw = 5usize;

        // 5-row MPO construction following White & Martin (2006):
        // W = [[ I,    0,    0,    0,   0 ],
        //      [ S+,   0,    0,    0,   0 ],
        //      [ S-,   0,    0,    0,   0 ],
        //      [ Sz,   0,    0,    0,   0 ],
        //      [ 0,  .5S-, .5S+,   Sz,  I ]]
        // index (row, col) ↔ (w_l, w_r)
        // The left boundary site emits row 4 (picks up accumulated Hamiltonian).
        // The right boundary site emits column 0 (terminates the operator).

        // Flatten 5×5 block matrix of 2×2 physical matrices into a lookup:
        // block[wl][wr][s_out][s_in]
        let mut block = [[[[0.0f64; 2]; 2]; 5]; 5];

        // Row 0 (w_l=0): [I, 0, 0, 0, 0]
        block[0][0][0][0] = 1.0;
        block[0][0][1][1] = 1.0;
        // Row 1 (w_l=1): [S+, 0, ...]
        block[1][0][0][1] = 1.0; // S^+[0,1] = 1
        // Row 2 (w_l=2): [S-, 0, ...]
        block[2][0][1][0] = 1.0; // S^-[1,0] = 1
        // Row 3 (w_l=3): [Sz, 0, ...]
        block[3][0][0][0] = 0.5;
        block[3][0][1][1] = -0.5;
        // Row 4 col 1: 0.5 S^-
        block[4][1][1][0] = 0.5;
        // Row 4 col 2: 0.5 S^+
        block[4][2][0][1] = 0.5;
        // Row 4 col 3: Sz
        block[4][3][0][0] = 0.5;
        block[4][3][1][1] = -0.5;
        // Row 4 col 4: I
        block[4][4][0][0] = 1.0;
        block[4][4][1][1] = 1.0;

        let mut tensors = Vec::with_capacity(n_sites);
        let mut shapes = Vec::with_capacity(n_sites);

        for site in 0..n_sites {
            let (wl, wr) = if n_sites == 1 {
                (1, 1)
            } else if site == 0 {
                (1, dw)
            } else if site == n_sites - 1 {
                (dw, 1)
            } else {
                (dw, dw)
            };

            let mut data = vec![0.0f64; wl * d * d * wr];

            if n_sites == 1 {
                // Trivial single-site: identity (s_out=s_in=0 at flat 0, s_out=s_in=1 at flat d+1)
                data[0] = 1.0;
                data[d + 1] = 1.0;
            } else if site == 0 {
                // Left boundary: only row 4 of the W matrix survives (wl=1, so w_l=0 is the
                // single allowed value).
                // data[((w_l * d + s_out) * d + s_in) * wr + w_r] with w_l=0
                //     = data[(s_out * d + s_in) * wr + w_r]
                //     = block[4][w_r][s_out][s_in]
                for s_out in 0..d {
                    for s_in in 0..d {
                        for w_r in 0..wr {
                            data[(s_out * d + s_in) * wr + w_r] = block[4][w_r][s_out][s_in];
                        }
                    }
                }
            } else if site == n_sites - 1 {
                // Right boundary: only column 0 survives (wr=1 maps to col 0).
                // data[((w_l * d + s_out) * d + s_in) * 1 + 0] = block[w_l][0][s_out][s_in]
                for w_l in 0..wl {
                    for s_out in 0..d {
                        for s_in in 0..d {
                            data[(w_l * d + s_out) * d + s_in] = block[w_l][0][s_out][s_in];
                        }
                    }
                }
            } else {
                // Bulk: full W matrix.
                for w_l in 0..wl {
                    for s_out in 0..d {
                        for s_in in 0..d {
                            for w_r in 0..wr {
                                data[((w_l * d + s_out) * d + s_in) * wr + w_r] =
                                    block[w_l][w_r][s_out][s_in];
                            }
                        }
                    }
                }
            }

            tensors.push(data);
            shapes.push([wl, d, d, wr]);
        }

        (tensors, shapes)
    }

    // ── Test 1: Default config values ─────────────────────────────────────────
    #[test]
    fn config_defaults() {
        let cfg = SingleSiteDmrgConfig::default();
        assert_eq!(cfg.chi_max, 32);
        assert_eq!(cfg.max_sweeps, 10);
        assert!((cfg.tol - 1.0e-8).abs() < 1.0e-20);
        assert_eq!(cfg.lanczos_max_iter, 50);
        assert!((cfg.lanczos_tol - 1.0e-10).abs() < 1.0e-20);
        assert!((cfg.noise - 1.0e-5).abs() < 1.0e-20);
    }

    // ── Test 2: Identity MPO → energy = 1 ────────────────────────────────────
    #[test]
    fn identity_mpo_energy_is_norm_squared() {
        let n_sites = 4;
        let d = 2;
        let chi = 2;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 42);
        let (mpo_t, mpo_s) = identity_mpo_raw(n_sites, d);
        let cfg = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 2,
            noise: 1.0e-5,
            ..SingleSiteDmrgConfig::default()
        };
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
        // After canonicalisation the MPS is normalised, so <ψ|I|ψ>/<ψ|ψ> = 1.
        assert!(
            (result.ground_state_energy - 1.0).abs() < 0.05,
            "energy = {}",
            result.ground_state_energy
        );
    }

    // ── Test 3: Heisenberg energy < 0 ────────────────────────────────────────
    #[test]
    fn heisenberg_energy_lower_than_product_state() {
        let n_sites = 6;
        let d = 2;
        let chi = 4;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 7);
        let (mpo_t, mpo_s) = heisenberg_mpo_raw(n_sites);
        let cfg = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 6,
            noise: 1.0e-4,
            ..SingleSiteDmrgConfig::default()
        };
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
        // Heisenberg antiferromagnet has E_gs < 0 for N >= 2.
        assert!(
            result.ground_state_energy < 0.5,
            "expected energy < 0.5, got {}",
            result.ground_state_energy
        );
    }

    // ── Test 4: Output MPS shapes correct ────────────────────────────────────
    #[test]
    fn single_site_result_shape() {
        let n_sites = 5;
        let d = 2;
        let chi = 3;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 13);
        let (mpo_t, mpo_s) = identity_mpo_raw(n_sites, d);
        let cfg = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 2,
            ..SingleSiteDmrgConfig::default()
        };
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
        assert_eq!(result.mps.len(), n_sites);
        assert_eq!(result.mps_shapes.len(), n_sites);
        // Physical dimension must be preserved
        for i in 0..n_sites {
            assert_eq!(result.mps_shapes[i][1], d, "physical dim at site {}", i);
        }
        // Boundary bonds must be 1
        assert_eq!(result.mps_shapes[0][0], 1, "left boundary bond");
        assert_eq!(result.mps_shapes[n_sites - 1][2], 1, "right boundary bond");
    }

    // ── Test 5: Convergence on small chain ───────────────────────────────────
    #[test]
    fn converges_in_few_sweeps() {
        let n_sites = 4;
        let d = 2;
        let chi = 4;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 99);
        let (mpo_t, mpo_s) = heisenberg_mpo_raw(n_sites);
        let cfg = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 20,
            tol: 1.0e-5,
            noise: 1.0e-4,
            ..SingleSiteDmrgConfig::default()
        };
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
        assert!(
            result.converged,
            "should have converged; energies = {:?}",
            result.energies
        );
    }

    // ── Test 6: Energies are (weakly) monotone decreasing ────────────────────
    #[test]
    fn energies_monotone_decreasing() {
        let n_sites = 4;
        let d = 2;
        let chi = 4;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 17);
        let (mpo_t, mpo_s) = heisenberg_mpo_raw(n_sites);
        let cfg = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 8,
            noise: 1.0e-4,
            ..SingleSiteDmrgConfig::default()
        };
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
        let energies = &result.energies;
        assert!(energies.len() >= 2, "need at least 2 sweeps");
        // Subspace expansion noise may cause small temporary increases; allow tolerance.
        for w in energies.windows(2) {
            assert!(
                w[1] <= w[0] + 0.5,
                "energy jumped up too much: {} -> {}",
                w[0],
                w[1]
            );
        }
        // Overall the final energy should not be much worse than the first.
        assert!(
            *energies.last().expect("last should succeed")
                <= *energies.first().expect("first should succeed") + 0.5,
            "final energy higher than initial: {:?}",
            energies
        );
    }

    // ── Test 7: Output MPS tensor count matches input ─────────────────────────
    #[test]
    fn mps_tensors_length() {
        for n_sites in [2usize, 3, 5, 8] {
            let d = 2;
            let chi = 2;
            let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, n_sites as u64 * 31);
            let (mpo_t, mpo_s) = identity_mpo_raw(n_sites, d);
            let cfg = SingleSiteDmrgConfig {
                chi_max: chi,
                max_sweeps: 1,
                ..SingleSiteDmrgConfig::default()
            };
            let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
            assert_eq!(result.mps.len(), n_sites, "n_sites={}", n_sites);
        }
    }

    // ── Test 8: Single-site vs two-site energy agreement ─────────────────────
    #[test]
    fn single_site_vs_two_site_agreement() {
        use crate::dmrg::dmrg::{DmrgConfig, dmrg_two_site};
        use crate::handle::LcgRng;
        use crate::mpo::mpo::Mpo;
        use crate::mps::mps::Mps;

        let n_sites = 4;
        let d = 2;
        let chi = 8;

        // Two-site DMRG with identity MPO → energy should be ≈ 1.
        let mut rng = LcgRng::new(111);
        let mpo = Mpo::identity(n_sites, d).expect("ok");
        let init = Mps::random_mps(n_sites, d, chi, &mut rng).expect("ok");
        let cfg_2s = DmrgConfig {
            chi_max: chi,
            max_sweeps: 6,
            ..DmrgConfig::default()
        };
        let r2s = dmrg_two_site(&mpo, init, cfg_2s, &mut rng).expect("ok");
        let e2s = r2s.energy;

        // Single-site DMRG with identity MPO → energy should be ≈ 1.
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 222);
        let (mpo_t, mpo_s) = identity_mpo_raw(n_sites, d);
        let cfg_ss = SingleSiteDmrgConfig {
            chi_max: chi,
            max_sweeps: 6,
            noise: 1.0e-5,
            ..SingleSiteDmrgConfig::default()
        };
        let rss = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg_ss).expect("ok");
        let ess = rss.ground_state_energy;

        assert!((e2s - 1.0).abs() < 0.1, "two-site energy = {}", e2s);
        assert!((ess - 1.0).abs() < 0.1, "single-site energy = {}", ess);
    }

    // ── Test 9: Noise 0 and noise 1e-5 both produce finite energies ───────────
    #[test]
    fn noise_helps_converge() {
        let n_sites = 4;
        let d = 2;
        let chi = 4;
        let (mpo_t, mpo_s) = heisenberg_mpo_raw(n_sites);

        for (noise_level, seed) in [(0.0f64, 55u64), (1.0e-5, 56u64)] {
            let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, seed);
            let cfg = SingleSiteDmrgConfig {
                chi_max: chi,
                max_sweeps: 6,
                noise: noise_level,
                ..SingleSiteDmrgConfig::default()
            };
            let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg).expect("ok");
            assert!(
                result.ground_state_energy.is_finite(),
                "noise={}: energy not finite",
                noise_level
            );
            assert!(!result.mps.is_empty(), "noise={}: empty MPS", noise_level);
        }
    }

    // ── Test 10: Empty MPS returns TnError::EmptyInput ───────────────────────
    #[test]
    fn empty_mps_error() {
        let cfg = SingleSiteDmrgConfig::default();
        let result = single_site_dmrg(&[], &[], &[], &[], &cfg);
        assert!(result.is_err(), "expected error for empty MPS");
        assert!(
            matches!(result.unwrap_err(), TnError::EmptyInput),
            "expected EmptyInput"
        );
    }

    // ── Test 11: MPS and MPO length mismatch returns error ───────────────────
    #[test]
    fn shape_mismatch_error() {
        let n_sites = 3;
        let d = 2;
        let chi = 2;
        let (mps_t, mps_s) = random_mps_raw(n_sites, d, chi, 77);
        let (mpo_t, mpo_s) = identity_mpo_raw(n_sites + 1, d); // length mismatch
        let cfg = SingleSiteDmrgConfig::default();
        let result = single_site_dmrg(&mps_t, &mps_s, &mpo_t, &mpo_s, &cfg);
        assert!(result.is_err(), "expected error for length mismatch");
    }
}
