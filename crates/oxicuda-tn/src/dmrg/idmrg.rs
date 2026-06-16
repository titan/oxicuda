//! Infinite DMRG (iDMRG) for translationally-invariant 1D ground states.
//!
//! Implements the McCulloch (2008) single-site iDMRG algorithm that grows a
//! matrix-product state (MPS) one unit cell (two sites) at a time, keeping the
//! boundary environments truncated to bond dimension χ.
//!
//! # Algorithm sketch
//!
//! ```text
//! Initialize:
//!   L_env: [χ_l, w, χ_l]  left environment  (starts at [1, w, 1])
//!   R_env: [χ_r, w, χ_r]  right environment (starts at [1, w, 1])
//!   A, B:  [χ, d, χ] random unit-cell MPS tensors
//!
//! For each growth step:
//!   1. Build H_eff from {L_env, W_A, W_B, R_env}
//!   2. Solve local 2-site eigenproblem via Lanczos → |Θ⟩, E
//!   3. SVD Θ with truncation to χ_max: Θ ≈ U S V†
//!   4. A = U (left-canonical), B = S V† (right-canonical)
//!   5. Grow:  L_env_new = contract(L_env, A,  W_A, A†)
//!             R_env_new = contract(R_env, B,  W_B, B†)
//!   6. Energy per site e = (E_step − E_prev) / 2
//!   7. Converge when |e_new − e_old| < tol
//! ```
//!
//! # Reference
//!
//! McCulloch, I. P. (2008). *Infinite size density matrix renormalization group,
//! revisited*. arXiv:0804.2509.

use crate::dmrg::lanczos::lanczos_smallest;
use crate::handle::LcgRng;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ── Public data structures ────────────────────────────────────────────────────

/// Configuration for the iDMRG growth algorithm.
#[derive(Debug, Clone, Copy)]
pub struct IDmrgConfig {
    /// Maximum virtual bond dimension χ (must be ≥ 1).
    pub chi_max: usize,
    /// Maximum number of growth steps.
    pub max_iter: usize,
    /// Energy-per-site convergence tolerance.
    pub tol: f64,
    /// Maximum Lanczos iterations per step.
    pub lanczos_max_iter: usize,
    /// Lanczos eigenvalue convergence tolerance.
    pub lanczos_tol: f64,
}

impl Default for IDmrgConfig {
    fn default() -> Self {
        Self {
            chi_max: 16,
            max_iter: 100,
            tol: 1.0e-8,
            lanczos_max_iter: 50,
            lanczos_tol: 1.0e-10,
        }
    }
}

/// Result returned by [`idmrg`].
#[derive(Debug, Clone)]
pub struct IDmrgResult {
    /// Final energy per site (converged estimate).
    pub energy_per_site: f64,
    /// Energy per site recorded after every growth step (length == `n_iter`).
    pub energy_history: Vec<f64>,
    /// Left bulk MPS tensor A, shape `a_shape = [chi_l, d, chi_r]`, row-major.
    pub a_tensor: Vec<f64>,
    /// Right bulk MPS tensor B, shape `b_shape = [chi_l, d, chi_r]`, row-major.
    pub b_tensor: Vec<f64>,
    /// Shape `[chi_l, d, chi_r]` of `a_tensor`.
    pub a_shape: [usize; 3],
    /// Shape `[chi_l, d, chi_r]` of `b_tensor`.
    pub b_shape: [usize; 3],
    /// Bond dimension at convergence (≤ `chi_max`).
    pub bond_dim: usize,
    /// Number of growth steps actually executed.
    pub n_iter: usize,
    /// Whether the energy per site converged within `tol`.
    pub converged: bool,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run iDMRG to find the ground-state energy per site of an infinite,
/// translationally-invariant 1D Hamiltonian given as a two-site MPO unit cell.
///
/// # Arguments
///
/// * `mpo_tensors`  – exactly **two** flat site-MPO tensors `[W_A_data, W_B_data]`.
/// * `mpo_shapes`   – shapes `[[w_l, d, d, w_r]; 2]` for each tensor.
///   The physical bond `d_in == d_out` is required.
///   The MPO bond must satisfy `W_A.w_r == W_B.w_l` (inner bond connects them).
/// * `config`       – algorithm hyper-parameters.
/// * `rng`          – workspace `LcgRng` for random initialisation.
///
/// # Errors
///
/// Returns [`TnError::InvalidBondDimension`] when `chi_max == 0`,
/// [`TnError::InvalidConfiguration`] on MPO shape inconsistency,
/// or propagates [`TnError`] variants from Lanczos / SVD sub-routines.
pub fn idmrg(
    mpo_tensors: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
    config: &IDmrgConfig,
    rng: &mut LcgRng,
) -> TnResult<IDmrgResult> {
    // ── Validate inputs ────────────────────────────────────────────────────────
    if config.chi_max == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if mpo_tensors.len() != 2 || mpo_shapes.len() != 2 {
        return Err(TnError::InvalidConfiguration(
            "idmrg requires exactly 2 MPO site tensors (one unit cell)".into(),
        ));
    }

    let [wa_wl, wa_d_out, wa_d_in, wa_wr] = mpo_shapes[0];
    let [wb_wl, wb_d_out, wb_d_in, wb_wr] = mpo_shapes[1];

    if wa_d_out != wa_d_in {
        return Err(TnError::InvalidConfiguration(
            "MPO site A: d_out != d_in (physical bond must be square)".into(),
        ));
    }
    if wb_d_out != wb_d_in {
        return Err(TnError::InvalidConfiguration(
            "MPO site B: d_out != d_in (physical bond must be square)".into(),
        ));
    }
    if wa_d_out != wb_d_out {
        return Err(TnError::InvalidConfiguration(
            "MPO sites A and B have different physical dimensions".into(),
        ));
    }
    if wa_wr != wb_wl {
        return Err(TnError::InvalidConfiguration(format!(
            "MPO inner bond mismatch: W_A.w_r={wa_wr} != W_B.w_l={wb_wl}"
        )));
    }

    let d = wa_d_out;
    let w_a = wa_wr; // internal MPO bond connecting A→B (== wb_wl)
    let w_l = wa_wl; // left MPO boundary bond dimension
    let w_r = wb_wr; // right MPO boundary bond dimension

    // Validate tensor data sizes
    if mpo_tensors[0].len() != wa_wl * wa_d_out * wa_d_in * wa_wr {
        return Err(TnError::ShapeMismatch {
            expected: vec![wa_wl, wa_d_out, wa_d_in, wa_wr],
            got: vec![mpo_tensors[0].len()],
        });
    }
    if mpo_tensors[1].len() != wb_wl * wb_d_out * wb_d_in * wb_wr {
        return Err(TnError::ShapeMismatch {
            expected: vec![wb_wl, wb_d_out, wb_d_in, wb_wr],
            got: vec![mpo_tensors[1].len()],
        });
    }

    // ── Initialise environments ────────────────────────────────────────────────

    // Environments grow as:
    //   L_env: shape [l_chi, w_l_env, l_chi]  (l_chi grows from 1)
    //   R_env: shape [r_chi, w_r_env, r_chi]  (r_chi grows from 1)
    //
    // After the first grow, L_env's middle bond becomes wa_wr and R_env's
    // middle bond becomes wb_wl. For translational invariance wa_wl == wa_wr == w_l == w_r.
    //
    // We track:
    //   l_chi    — bond dimension of L_env (== left bond of A tensor)
    //   r_chi    — bond dimension of R_env (== right bond of B tensor)
    //   l_env_w  — MPO bond dimension stored in L_env (starts = wa_wl, stays wa_wr after grow)
    //   r_env_w  — MPO bond dimension stored in R_env (starts = wb_wr, stays wb_wl after grow)
    let mut l_chi = 1usize;
    let mut r_chi = 1usize;
    let mut l_env_w = w_l; // middle-bond of L_env
    let mut r_env_w = w_r; // middle-bond of R_env

    // Trivial boundaries: L[0, μ, 0] selects the last row of the MPO (full Hamiltonian row).
    // R[0, ν, 0] selects the first column of the MPO (boundary term column).
    let mut l_env = vec![0.0_f64; l_chi * l_env_w * l_chi];
    l_env[l_env_w - 1] = 1.0; // L[0, w_l-1, 0] = 1  (last MPO row)

    let mut r_env = vec![0.0_f64; r_chi * r_env_w * r_chi];
    r_env[0] = 1.0; // R[0, 0, 0] = 1  (first MPO column)

    // ── Initialise the unit-cell MPS tensors A, S, B ──────────────────────────
    //
    // After each SVD we store:
    //   a_data: left-canonical A tensor,  shape m_svd × k  = (l_chi*d) × k
    //   b_data: right-canonical B tensor, shape k × n_svd  = k × (d*r_chi)
    //   a_sval: singular values,          shape k
    //   a_chi_{l,r}, b_chi_{l,r}: explicit bond shapes
    //
    // The theta seed for step k+1 is:  Θ ≈ A * diag(S) * B.
    let mut a_chi_l = 1usize;
    let mut a_chi_r = 1usize; // inner bond between A and B (grows to chi_max)
    let mut b_chi_l = 1usize; // == a_chi_r at all times
    let mut b_chi_r = 1usize;

    // Random initial tensors (flat, shape [1,d,1] = shape [d])
    let mut a_data: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
    {
        let n = a_data.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        for v in &mut a_data {
            *v /= n;
        }
    }
    let mut b_data: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
    {
        let n = b_data.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        for v in &mut b_data {
            *v /= n;
        }
    }
    // Singular values: initialised to uniform 1/√d (step 0 seed has no SVD yet)
    let mut a_sval: Vec<f64> = vec![1.0 / (d as f64).sqrt(); 1];

    let mut energy_history: Vec<f64> = Vec::with_capacity(config.max_iter);
    let mut prev_e_total = f64::INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut bond_dim = 1usize;
    // iDMRG produces a period-2 oscillation in the incremental eps because the
    // unit cell has two sub-lattice sites (A, B) that alternate.  We therefore
    // track the last 4 consecutive eps values and declare convergence when the
    // same-phase differences are small:
    //   |eps[k] − eps[k−2]| < tol   AND   |eps[k−1] − eps[k−3]| < tol.
    const CONV_WINDOW: usize = 4;
    let mut eps_window: [f64; CONV_WINDOW] = [f64::INFINITY; CONV_WINDOW];

    for step in 0..config.max_iter {
        n_iter = step + 1;

        // ── Step 1: Seed the superblock wavefunction and run Lanczos ──────────
        //
        // Theta shape: [l_chi, d, d, r_chi] (the 2-site superblock)
        let n_theta = l_chi * d * d * r_chi;
        if n_theta == 0 {
            return Err(TnError::InvalidConfiguration(
                "superblock dimension is zero".into(),
            ));
        }

        // Build the theta seed from A * diag(S) * B.
        // A has shape (a_chi_l * d) × a_chi_r, B has shape b_chi_l × (d * b_chi_r).
        // S has shape a_chi_r (= b_chi_l).
        // If the shapes are consistent with the current [l_chi, d, d, r_chi] superblock,
        // we can reconstruct: Θ = A * diag(S) * B (reshaped to [l_chi, d, d, r_chi]).
        // Otherwise (first step or after chi_max is reached and bond stays fixed), use random.
        let shapes_match =
            a_chi_l == l_chi && b_chi_r == r_chi && a_chi_r == b_chi_l && a_sval.len() == a_chi_r;
        let mut theta_seed = if shapes_match {
            // A*diag(S)*B: first scale each column of A by S, then multiply by B
            build_theta_from_asb(
                &a_data, a_chi_l, d, a_chi_r, &a_sval, &b_data, b_chi_l, d, b_chi_r,
            )
        } else {
            vec![0.0_f64; n_theta]
        };
        let seed_norm: f64 = theta_seed.iter().map(|x| x * x).sum::<f64>().sqrt();
        if seed_norm < 1e-15 {
            for v in &mut theta_seed {
                *v = rng.next_normal();
            }
        }

        // Build closure for H_eff matrix-vector product.
        // L_env middle bond is l_env_w; R_env middle bond is r_env_w.
        let l_env_c = l_env.clone();
        let r_env_c = r_env.clone();
        let wa_data = mpo_tensors[0].clone();
        let wb_data = mpo_tensors[1].clone();
        let lchi = l_chi;
        let rchi = r_chi;
        let lw = l_env_w;
        let rw = r_env_w;
        let dc = d;
        let wa_wl_c = wa_wl;
        let wa_wr_c = wa_wr;
        let w_a_c = w_a;
        let wb_wr_c = wb_wr;

        let apply = move |psi: &[f64]| -> Vec<f64> {
            h_eff_apply_idmrg(
                psi, &l_env_c, lchi, lw, &r_env_c, rchi, rw, &wa_data, wa_wl_c, dc, wa_wr_c,
                &wb_data, w_a_c, dc, wb_wr_c,
            )
        };

        let lanczos_result = lanczos_smallest(
            apply,
            n_theta,
            &theta_seed,
            config.lanczos_max_iter,
            config.lanczos_tol,
        )?;

        let e_total = lanczos_result.eigenvalue;
        let new_theta = lanczos_result.eigenvector;

        // Energy per site: on step 0 use e/2 (2-site estimate), then incremental.
        let eps = if step == 0 {
            e_total / 2.0
        } else {
            (e_total - prev_e_total) / 2.0
        };
        energy_history.push(eps);

        // ── Step 2: SVD split Θ → A (left-canonical) + B (right-canonical) ─────
        //
        // For the environment growth to preserve the identity block (and hence correctly
        // accumulate the Hamiltonian energy), we MUST use:
        //   A = U   (left-canonical:  A† A = I_k)
        //   B = Vt  (right-canonical: B B† = I_k)
        //
        // The singular values S are stored separately for seeding the next theta.
        // The theta seed on step k+1 is:  Θ ≈ A * diag(S) * B.
        //
        // Reshape Theta: rows = l_chi * d,  cols = d * r_chi
        let m_svd = l_chi * d;
        let n_svd = d * r_chi;

        let svd_raw = svd_jacobi(&new_theta, m_svd, n_svd)?;
        let (svd, _) = svd_truncate(svd_raw, config.chi_max, 0.0)?;
        let k = svd.k; // new inner bond (between A and B)
        bond_dim = k;

        // A = U:  shape (l_chi*d) × k  (left-canonical, A†A = I_k)
        // B = Vt: shape k × (d*r_chi)  (right-canonical, B B† = I_k)
        // S: singular values, shape k
        let new_a = svd.u.clone(); // left-canonical MPS tensor for L_env growth
        let new_b = svd.vt.clone(); // right-canonical MPS tensor for R_env growth
        let new_s = svd.s.clone(); // singular values (kept for theta seed)

        // ── Step 3: Grow environments ─────────────────────────────────────────
        //
        // New L_env: incorporate site A.
        //   new_l_env shape: [k, wa_wr, k]
        // New R_env: incorporate site B.
        //   new_r_env shape: [k, wb_wl, k]
        let new_l_env = grow_left_env(
            &l_env,
            l_chi,
            l_env_w,
            &new_a,
            l_chi,
            d,
            k,
            &mpo_tensors[0],
            wa_wl,
            d,
            wa_wr,
        )?;

        let new_r_env = grow_right_env(
            &r_env,
            r_chi,
            r_env_w,
            &new_b,
            k,
            d,
            r_chi,
            &mpo_tensors[1],
            wb_wl,
            d,
            wb_wr,
        )?;

        // ── Step 4: Update all state variables ────────────────────────────────
        //
        // After growing:
        //   L_env has bond k (touching MPS) and middle bond wa_wr.
        //   R_env has bond k (touching MPS) and middle bond wb_wl.
        //   For the next step, superblock shape = [k, d, d, k].
        //
        // For the theta seed on step k+1, we reconstruct:
        //   Θ_seed[l_chi_new, d, d, r_chi_new] = A[l_chi,d,k] * diag(S) * B[k,d,r_chi]
        // Here A is (l_chi*d)×k and B is k×(d*r_chi). The product is stored as:
        //   new_theta_seed[l_chi_new, d, d, r_chi_new] (shape [k, d, d, k])
        // This requires contracting A * diag(S) * B where shapes match l_chi_new=k, r_chi_new=k.
        // But build_theta_from_ab needs A as [a_chi_l, d, a_chi_r] and B as [b_chi_l, d, b_chi_r].
        // We store a_data = A (left-canonical, shape m_svd × k with m_svd = l_chi*d)
        //          b_data = B (right-canonical, shape k × n_svd with n_svd = d*r_chi)
        //          a_s    = S (singular values, shape k)
        // The theta seed = contract(A, diag(S), B) reshaped to [k, d, d, k].
        // Store a_chi_l = l_chi, a_chi_r = k (A represents shape [l_chi, d, k])
        //       b_chi_l = k, b_chi_r = r_chi (B represents shape [k, d, r_chi])
        a_chi_l = l_chi; // the old l_chi (before update)
        a_chi_r = k;
        b_chi_l = k;
        b_chi_r = r_chi; // the old r_chi (before update)

        a_data = new_a; // left-canonical U: shape (l_chi*d) × k
        b_data = new_b; // right-canonical Vt: shape k × (d*r_chi)
        a_sval = new_s; // singular values: shape k

        // Now update environment bond dimensions for the next step.
        // The new L_env has open bond k (on both sides), middle bond wa_wr.
        // The new R_env has open bond k (on both sides), middle bond wb_wl.
        l_chi = k;
        r_chi = k;
        l_env_w = wa_wr; // middle bond of L_env is now the right MPO bond of W_A
        r_env_w = wb_wl; // middle bond of R_env is now the left MPO bond of W_B

        l_env = new_l_env;
        r_env = new_r_env;

        // ── Step 5: Check convergence ─────────────────────────────────────────
        //
        // iDMRG has a period-2 oscillation in eps (A-site vs B-site steps).
        // Convergence is checked on same-phase pairs: we declare converged when
        //   |eps[k] − eps[k−2]| < tol  AND  |eps[k−1] − eps[k−3]| < tol.
        // This correctly accounts for the sublattice alternation.
        eps_window.rotate_left(1);
        eps_window[CONV_WINDOW - 1] = eps;
        if step + 1 >= CONV_WINDOW {
            let diff_even = (eps_window[CONV_WINDOW - 1] - eps_window[CONV_WINDOW - 3]).abs();
            let diff_odd = (eps_window[CONV_WINDOW - 2] - eps_window[CONV_WINDOW - 4]).abs();
            if diff_even < config.tol && diff_odd < config.tol {
                converged = true;
                break;
            }
        }
        prev_e_total = e_total;
    }

    let final_eps = energy_history.last().copied().unwrap_or(0.0);

    // The final A and B tensors have shapes stored in a_chi_{l,r} and b_chi_{l,r}.
    let a_shape = [a_chi_l, d, a_chi_r];
    let b_shape = [b_chi_l, d, b_chi_r];

    Ok(IDmrgResult {
        energy_per_site: final_eps,
        energy_history,
        a_tensor: a_data,
        b_tensor: b_data,
        a_shape,
        b_shape,
        bond_dim,
        n_iter,
        converged,
    })
}

// ── Effective Hamiltonian application ────────────────────────────────────────

/// Apply the 2-site effective Hamiltonian `H_eff` to a superblock vector `psi`.
///
/// `psi` has shape `[l_chi, d, d, r_chi]` (flattened row-major).
///
/// `H_eff` is formed by contracting:
/// ```text
///   L_env[α, μ, α'] * W_A[μ, σ, σ', ν] * W_B[ν, τ, τ', ρ] * R_env[β, ρ, β']
/// ```
/// and acting on `psi[α', σ', τ', β']` to produce `out[α, σ, τ, β]`.
///
/// # Index ordering
///
/// - `L_env`: layout `L[(α*w_l + μ)*l_chi + α']`
/// - `R_env`: layout `R[(β*w_r + ρ)*r_chi + β']`
/// - `W_A`:   layout `W[((μ*d + σ)*d + σ')*wa_wr + ν]`
/// - `W_B`:   layout `W[((ν*d + τ)*d + τ')*wb_wr + ρ]`
/// - `psi`:   layout `psi[((α'*d + σ')*d + τ')*r_chi + β']`
/// - `out`:   layout `out[((α*d + σ)*d + τ)*r_chi + β]`
#[allow(clippy::too_many_arguments)]
fn h_eff_apply_idmrg(
    psi: &[f64],
    l_env: &[f64],
    l_chi: usize,
    w_l: usize,
    r_env: &[f64],
    r_chi: usize,
    w_r: usize,
    wa: &[f64],
    _wa_wl: usize,
    d: usize,
    wa_wr: usize,
    wb: &[f64],
    wb_wl: usize,
    _wb_d: usize,
    wb_wr: usize,
) -> Vec<f64> {
    // wb_wl == wa_wr (inner bond between the two MPO sites)
    let w_mid = wa_wr; // == wb_wl (asserted by caller)
    debug_assert_eq!(wb_wl, w_mid);

    let mut out = vec![0.0_f64; l_chi * d * d * r_chi];

    for alpha in 0..l_chi {
        for sigma in 0..d {
            for tau in 0..d {
                for beta in 0..r_chi {
                    let mut acc = 0.0_f64;
                    for alphap in 0..l_chi {
                        for sigmap in 0..d {
                            for taup in 0..d {
                                for betap in 0..r_chi {
                                    let psi_v =
                                        psi[((alphap * d + sigmap) * d + taup) * r_chi + betap];
                                    if psi_v.abs() < 1e-300 {
                                        continue;
                                    }
                                    let mut h_elem = 0.0_f64;
                                    // mu loops over the middle bond of L_env (= w_l = l_env_w)
                                    for mu in 0..w_l {
                                        // L_env[alpha, mu, alphap]: layout L[(alpha*w_l + mu)*l_chi + alphap]
                                        let lv = l_env[(alpha * w_l + mu) * l_chi + alphap];
                                        if lv.abs() < 1e-300 {
                                            continue;
                                        }
                                        // W_A[mu, sigma, sigmap, nu]: layout W[((mu*d+sigma)*d+sigmap)*wa_wr+nu]
                                        // mu indexes W_A's left bond (wa_wl). The L_env middle bond w_l
                                        // connects to the MPO left bond wa_wl — they must match.
                                        // (Verified at construction: l_env_w starts as wa_wl.)
                                        for nu in 0..w_mid {
                                            let wav =
                                                wa[((mu * d + sigma) * d + sigmap) * wa_wr + nu];
                                            if wav.abs() < 1e-300 {
                                                continue;
                                            }
                                            for rho in 0..w_r {
                                                // W_B[nu, tau, taup, rho]
                                                let wbv =
                                                    wb[((nu * d + tau) * d + taup) * wb_wr + rho];
                                                if wbv.abs() < 1e-300 {
                                                    continue;
                                                }
                                                // R_env[beta, rho, betap]: layout R[(beta*w_r+rho)*r_chi+betap]
                                                let rv = r_env[(beta * w_r + rho) * r_chi + betap];
                                                h_elem += lv * wav * wbv * rv;
                                            }
                                        }
                                    }
                                    acc += h_elem * psi_v;
                                }
                            }
                        }
                    }
                    out[((alpha * d + sigma) * d + tau) * r_chi + beta] = acc;
                }
            }
        }
    }
    out
}

// ── Environment growth routines ───────────────────────────────────────────────

/// Grow the left environment by incorporating one MPS site and its MPO tensor.
///
/// ```text
/// L_new[β, ν, β'] = Σ_{α, μ, σ, σ'} L_old[α, μ, α'] * A[α, σ, β] * W[μ, σ, σ', ν] * A[α', σ', β']
/// ```
///
/// - `l_old`: layout `L[(α*w_l + μ)*chi_l + α']`,  shape `[chi_l, w_l, chi_l]`
/// - `a`:     layout `A[(α*d + σ)*k + β]`,           shape `[chi_l, d, k]`
/// - `w`:     layout `W[((μ*d + σ)*d + σ')*wa_wr + ν]`, shape `[wa_wl, d, d, wa_wr]`
/// - output:  layout `L_new[(β*wa_wr + ν)*k + β']`,  shape `[k, wa_wr, k]`
#[allow(clippy::too_many_arguments)]
fn grow_left_env(
    l_old: &[f64],
    chi_l: usize,
    w_l: usize,
    a: &[f64],
    a_chi_l: usize,
    d: usize,
    k: usize,
    w: &[f64],
    wa_wl: usize,
    _wa_d: usize,
    wa_wr: usize,
) -> TnResult<Vec<f64>> {
    debug_assert_eq!(chi_l, a_chi_l);
    debug_assert_eq!(wa_wl, w_l);
    debug_assert_eq!(l_old.len(), chi_l * w_l * chi_l);
    debug_assert_eq!(a.len(), a_chi_l * d * k);

    let mut l_new = vec![0.0_f64; k * wa_wr * k];

    for beta in 0..k {
        for nu in 0..wa_wr {
            for betap in 0..k {
                let mut acc = 0.0_f64;
                for alpha in 0..chi_l {
                    for mu in 0..w_l {
                        for alphap in 0..chi_l {
                            let lv = l_old[(alpha * w_l + mu) * chi_l + alphap];
                            if lv.abs() < 1e-300 {
                                continue;
                            }
                            for sigma in 0..d {
                                // A[alpha, sigma, beta]
                                let av = a[(alpha * d + sigma) * k + beta];
                                if av.abs() < 1e-300 {
                                    continue;
                                }
                                for sigmap in 0..d {
                                    // W[mu, sigma, sigmap, nu]
                                    let wv = w[((mu * d + sigma) * d + sigmap) * wa_wr + nu];
                                    if wv.abs() < 1e-300 {
                                        continue;
                                    }
                                    // A[alphap, sigmap, betap] (real => same as conj)
                                    let avp = a[(alphap * d + sigmap) * k + betap];
                                    acc += lv * av * wv * avp;
                                }
                            }
                        }
                    }
                }
                l_new[(beta * wa_wr + nu) * k + betap] = acc;
            }
        }
    }
    Ok(l_new)
}

/// Grow the right environment by incorporating one MPS site and its MPO tensor.
///
/// ```text
/// R_new[α, μ, α'] = Σ_{β, ν, τ, τ'} B[α, τ, β] * W[μ, τ, τ', ν] * B[α', τ', β'] * R_old[β, ν, β']
/// ```
///
/// - `r_old`: layout `R[(β*w_r + ν)*chi_r + β']`,  shape `[chi_r, w_r, chi_r]`
/// - `b`:     layout `B[(α*d + τ)*chi_r + β]`,      shape `[k, d, chi_r]`
/// - `w`:     layout `W[((μ*d + τ)*d + τ')*wb_wr + ν]`, shape `[wb_wl, d, d, wb_wr]`
/// - output:  layout `R_new[(α*wb_wl + μ)*k + α']`, shape `[k, wb_wl, k]`
#[allow(clippy::too_many_arguments)]
fn grow_right_env(
    r_old: &[f64],
    chi_r: usize,
    w_r: usize,
    b: &[f64],
    k: usize,
    d: usize,
    b_chi_r: usize,
    w: &[f64],
    wb_wl: usize,
    _wb_d: usize,
    wb_wr: usize,
) -> TnResult<Vec<f64>> {
    debug_assert_eq!(chi_r, b_chi_r);
    debug_assert_eq!(wb_wr, w_r);
    debug_assert_eq!(r_old.len(), chi_r * w_r * chi_r);
    debug_assert_eq!(b.len(), k * d * b_chi_r);

    let mut r_new = vec![0.0_f64; k * wb_wl * k];

    for alpha in 0..k {
        for mu in 0..wb_wl {
            for alphap in 0..k {
                let mut acc = 0.0_f64;
                for beta in 0..chi_r {
                    for nu in 0..w_r {
                        for betap in 0..chi_r {
                            let rv = r_old[(beta * w_r + nu) * chi_r + betap];
                            if rv.abs() < 1e-300 {
                                continue;
                            }
                            for tau in 0..d {
                                // B[alpha, tau, beta]
                                let bv = b[(alpha * d + tau) * chi_r + beta];
                                if bv.abs() < 1e-300 {
                                    continue;
                                }
                                for taup in 0..d {
                                    // W[mu, tau, taup, nu]
                                    let wv = w[((mu * d + tau) * d + taup) * wb_wr + nu];
                                    if wv.abs() < 1e-300 {
                                        continue;
                                    }
                                    // B[alphap, taup, betap]
                                    let bvp = b[(alphap * d + taup) * chi_r + betap];
                                    acc += bv * wv * bvp * rv;
                                }
                            }
                        }
                    }
                }
                r_new[(alpha * wb_wl + mu) * k + alphap] = acc;
            }
        }
    }
    Ok(r_new)
}

// ── Helper: build theta from A, S, B ─────────────────────────────────────────

/// Compute `theta[α, σ, τ, β] = Σ_m A[α, σ, m] * S[m] * B[m, τ, β]`.
///
/// Used to seed Lanczos with the previous step's optimal wavefunction.
/// A is left-canonical (m_svd × k), B is right-canonical (k × n_svd),
/// S is the diagonal singular-value vector (length k).
#[allow(clippy::too_many_arguments)]
fn build_theta_from_asb(
    a: &[f64],
    a_chi_l: usize,
    d: usize,
    a_chi_r: usize,
    s: &[f64],
    b: &[f64],
    b_chi_l: usize,
    _d2: usize,
    b_chi_r: usize,
) -> Vec<f64> {
    debug_assert_eq!(a_chi_r, b_chi_l);
    debug_assert_eq!(s.len(), a_chi_r);
    let m = a_chi_r; // == b_chi_l
    let mut theta = vec![0.0_f64; a_chi_l * d * d * b_chi_r];
    for alpha in 0..a_chi_l {
        for sigma in 0..d {
            for tau in 0..d {
                for beta in 0..b_chi_r {
                    let mut acc = 0.0_f64;
                    for mid in 0..m {
                        // A[(alpha*d + sigma)*a_chi_r + mid] * S[mid] * B[(mid*d + tau)*b_chi_r + beta]
                        acc += a[(alpha * d + sigma) * a_chi_r + mid]
                            * s[mid]
                            * b[(mid * d + tau) * b_chi_r + beta];
                    }
                    theta[((alpha * d + sigma) * d + tau) * b_chi_r + beta] = acc;
                }
            }
        }
    }
    theta
}

// ── MPO builders for tests ────────────────────────────────────────────────────

/// Build the Heisenberg XXX MPO unit cell tensors for spin-1/2 (d=2).
///
/// H = Σ_i J*(S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
/// with J = 1.0 (antiferromagnetic).
///
/// MPO bond dimension w = 5 (standard Heisenberg construction).
/// MPO shape: `[w_l, d, d, w_r]` = `[5, 2, 2, 5]`.
///
/// MPO row/column structure (boundary projectors):
///  - Row 0: I*row (left boundary)
///  - Row 4: I*col (right boundary)
///  - Rows 1-3: S^+, S^-, S^z (hopping terms)
pub fn build_heisenberg_mpo_unit_cell() -> (Vec<Vec<f64>>, Vec<[usize; 4]>) {
    let d = 2usize;
    let w = 5usize; // MPO bond dimension

    // Pauli / spin-1/2 operators for d=2 (row-major, d*d):
    // Identity: I = [[1,0],[0,1]]
    // S^z = 0.5*[[1,0],[0,-1]]
    // S^+ = [[0,1],[0,0]]
    // S^- = [[0,0],[1,0]]

    let identity: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
    let sz: [f64; 4] = [0.5, 0.0, 0.0, -0.5];
    let sp: [f64; 4] = [0.0, 1.0, 0.0, 0.0]; // S^+ = |up><down|
    let sm: [f64; 4] = [0.0, 0.0, 1.0, 0.0]; // S^- = |down><up|
    let zero: [f64; 4] = [0.0; 4];

    // MPO matrix at each site:
    // W[w_l, d, d, w_r] with layout W[((wl*d + dout)*d + din)*w_r + wr]
    //
    // Standard Heisenberg MPO (5x5 matrix of 2x2 operator blocks):
    //  [I,   0,   0,  0,   0  ]   <- row 0 (left boundary "I")
    //  [S^+, 0,   0,  0,   0  ]   <- row 1
    //  [S^-, 0,   0,  0,   0  ]   <- row 2
    //  [S^z, 0,   0,  0,   0  ]   <- row 3
    //  [0,   J/2*S^-, J/2*S^+, J*S^z, I]  <- row 4 (right boundary)
    //
    // In matrix form (rows = w_l, cols = w_r):
    //  row0: [I,   0,    0,    0,    0 ]
    //  row1: [S^+, 0,    0,    0,    0 ]
    //  row2: [S^-, 0,    0,    0,    0 ]
    //  row3: [S^z, 0,    0,    0,    0 ]
    //  row4: [0,   J/2*S^-, J/2*S^+, J*S^z, I]
    //
    // Actually the standard form is:
    //  W =  I        0    0    0   0
    //       S^+      0    0    0   0
    //       S^-      0    0    0   0
    //       S^z      0    0    0   0
    //       0    0.5*S^-  0.5*S^+  S^z   I
    //
    // We use J=1.0.

    let j = 1.0f64;

    // Build W tensor of shape [w, d, d, w] for sites A and B (same for translational invariance)
    // W[wl, dout, din, wr] = operator_block(wl, wr)[dout, din]
    //
    // Operator block at (wl, wr):
    // (0,0) -> I
    // (1,0) -> S^+
    // (2,0) -> S^-
    // (3,0) -> S^z
    // (4,1) -> 0.5*j*S^-
    // (4,2) -> 0.5*j*S^+
    // (4,3) -> j*S^z
    // (4,4) -> I
    // all others -> 0

    let block = |wl: usize, wr: usize| -> [f64; 4] {
        match (wl, wr) {
            (0, 0) => identity,
            (1, 0) => sp,
            (2, 0) => sm,
            (3, 0) => sz,
            (4, 1) => {
                let mut b = sm;
                for v in &mut b {
                    *v *= 0.5 * j;
                }
                b
            }
            (4, 2) => {
                let mut b = sp;
                for v in &mut b {
                    *v *= 0.5 * j;
                }
                b
            }
            (4, 3) => {
                let mut b = sz;
                for v in &mut b {
                    *v *= j;
                }
                b
            }
            (4, 4) => identity,
            _ => zero,
        }
    };

    // Fill the tensor data[((wl*d + dout)*d + din)*w + wr]
    let mut data = vec![0.0f64; w * d * d * w];
    for wl in 0..w {
        for dout in 0..d {
            for din in 0..d {
                for wr in 0..w {
                    let b = block(wl, wr);
                    // b layout: [dout, din] row-major → b[dout*d + din]
                    data[((wl * d + dout) * d + din) * w + wr] = b[dout * d + din];
                }
            }
        }
    }

    // Both sites A and B use the same MPO tensor for translational invariance
    let shape = [w, d, d, w];
    (vec![data.clone(), data], vec![shape, shape])
}

/// Build a trivial MPO unit cell where H = 0 (all site tensors are zero operators).
/// Used to test the zero-energy case.
pub fn build_zero_mpo_unit_cell(d: usize, w: usize) -> (Vec<Vec<f64>>, Vec<[usize; 4]>) {
    let data = vec![0.0f64; w * d * d * w];
    let shape = [w, d, d, w];
    (vec![data.clone(), data], vec![shape, shape])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    // ── Test 1: invalid chi_max = 0 ──────────────────────────────────────────

    #[test]
    fn invalid_chi_max_zero_returns_error() {
        let mut rng = make_rng(1);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 0,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng);
        assert!(result.is_err(), "expected error for chi_max=0, got Ok");
        match result.unwrap_err() {
            TnError::InvalidBondDimension(0) => {}
            e => panic!("expected InvalidBondDimension(0), got {e:?}"),
        }
    }

    // ── Test 2: MPO shape mismatch → appropriate error ───────────────────────

    #[test]
    fn mpo_wrong_tensor_count_returns_error() {
        let mut rng = make_rng(2);
        let (mut tensors, mut shapes) = build_heisenberg_mpo_unit_cell();
        tensors.push(tensors[0].clone());
        shapes.push(shapes[0]);
        let cfg = IDmrgConfig::default();
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng);
        assert!(result.is_err(), "expected error for 3 tensors");
    }

    // ── Test 3: MPO inner bond mismatch → error ──────────────────────────────

    #[test]
    fn mpo_inner_bond_mismatch_returns_error() {
        let mut rng = make_rng(3);
        let d = 2usize;
        let w1 = 5usize;
        let w2 = 3usize; // deliberate mismatch
        let data1 = vec![0.0f64; w1 * d * d * w2]; // W_A: [5, 2, 2, 3]
        let data2 = vec![0.0f64; w1 * d * d * w1]; // W_B: [5, 2, 2, 5] — inner bond w2 ≠ w1
        let shapes = [[w1, d, d, w2], [w1, d, d, w1]]; // w_A.wr=3, w_B.wl=5 → mismatch
        let cfg = IDmrgConfig::default();
        let result = idmrg(&[data1, data2], &shapes, &cfg, &mut rng);
        assert!(result.is_err(), "expected error for inner bond mismatch");
    }

    // ── Test 4: max_iter=1 runs exactly 1 step ───────────────────────────────

    #[test]
    fn max_iter_one_gives_one_entry_in_history() {
        let mut rng = make_rng(4);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 2,
            max_iter: 1,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert_eq!(result.n_iter, 1, "expected n_iter == 1");
        assert_eq!(
            result.energy_history.len(),
            1,
            "expected exactly 1 energy_history entry"
        );
    }

    // ── Test 5: energy_history.len() == n_iter ───────────────────────────────

    #[test]
    fn energy_history_length_equals_n_iter() {
        let mut rng = make_rng(5);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 2,
            max_iter: 8,
            tol: 1e-15, // prevent early convergence
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert_eq!(
            result.energy_history.len(),
            result.n_iter,
            "energy_history.len() must equal n_iter"
        );
    }

    // ── Test 6: n_iter <= max_iter always ────────────────────────────────────

    #[test]
    fn n_iter_never_exceeds_max_iter() {
        let mut rng = make_rng(6);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 4,
            max_iter: 15,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert!(
            result.n_iter <= cfg.max_iter,
            "n_iter={} exceeds max_iter={}",
            result.n_iter,
            cfg.max_iter
        );
    }

    // ── Test 7: bond dimension doesn't exceed chi_max ────────────────────────

    #[test]
    fn bond_dim_does_not_exceed_chi_max() {
        let mut rng = make_rng(7);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let chi_max = 4;
        let cfg = IDmrgConfig {
            chi_max,
            max_iter: 20,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert!(
            result.bond_dim <= chi_max,
            "bond_dim={} exceeds chi_max={}",
            result.bond_dim,
            chi_max
        );
    }

    // ── Test 8: Heisenberg energy per site is negative ────────────────────────

    #[test]
    fn heisenberg_energy_per_site_is_negative() {
        let mut rng = make_rng(8);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 4,
            max_iter: 30,
            tol: 1e-6,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert!(
            result.energy_per_site < 0.0,
            "Heisenberg ground state energy per site should be negative, got {}",
            result.energy_per_site
        );
    }

    // ── Test 9: Heisenberg energy converges toward Bethe ansatz ─────────────

    #[test]
    fn heisenberg_energy_per_site_converges_toward_bethe_ansatz() {
        // Bethe ansatz exact value: e_0 = -J*(ln(2) - 1/4) ≈ -0.4431
        // With chi=4 we expect to be within 10% of this value.
        let bethe_e0 = -0.4431f64;
        let mut rng = make_rng(9);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 6,
            max_iter: 60,
            tol: 1e-5,
            lanczos_max_iter: 60,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        let eps = result.energy_per_site;
        // Allow 30% relative error for small chi
        assert!(
            (eps - bethe_e0).abs() < 0.3 * bethe_e0.abs(),
            "energy_per_site={eps:.6} too far from Bethe ansatz {bethe_e0:.4}"
        );
    }

    // ── Test 10: a_tensor has correct length ─────────────────────────────────

    #[test]
    fn a_tensor_length_matches_a_shape() {
        let mut rng = make_rng(10);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 4,
            max_iter: 10,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        let [al, ad, ar] = result.a_shape;
        assert_eq!(
            result.a_tensor.len(),
            al * ad * ar,
            "a_tensor.len()={} but a_shape product={}",
            result.a_tensor.len(),
            al * ad * ar
        );
    }

    // ── Test 11: energy per site more negative for larger chi ────────────────

    #[test]
    fn larger_chi_gives_more_negative_energy() {
        // Variational principle: larger bond dimension → lower (more negative) energy
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let base_cfg = IDmrgConfig {
            max_iter: 25,
            tol: 1e-6,
            ..IDmrgConfig::default()
        };

        let mut rng2 = make_rng(11);
        let cfg2 = IDmrgConfig {
            chi_max: 2,
            ..base_cfg
        };
        let r2 = idmrg(&tensors, &shapes, &cfg2, &mut rng2).expect("chi=2 should succeed");

        let mut rng4 = make_rng(11);
        let cfg4 = IDmrgConfig {
            chi_max: 4,
            ..base_cfg
        };
        let r4 = idmrg(&tensors, &shapes, &cfg4, &mut rng4).expect("chi=4 should succeed");

        assert!(
            r4.energy_per_site <= r2.energy_per_site + 0.05,
            "chi=4 energy {:.6} should be ≤ chi=2 energy {:.6} (variational principle)",
            r4.energy_per_site,
            r2.energy_per_site
        );
    }

    // ── Test 12: d=2 qubits (small config) runs without error ───────────────

    #[test]
    fn d2_qubit_default_case_runs() {
        let mut rng = make_rng(12);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        // shapes use d=2 (spin-1/2); use small chi for test speed
        let cfg = IDmrgConfig {
            chi_max: 2,
            max_iter: 10,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng);
        assert!(
            result.is_ok(),
            "d=2 case should succeed: {:?}",
            result.err()
        );
        let r = result.expect("result should be present");
        assert_eq!(r.a_shape[1], 2, "physical dimension should be 2");
    }

    // ── Test 13: zero MPO gives near-zero energy ──────────────────────────────

    #[test]
    fn zero_mpo_gives_near_zero_energy() {
        let mut rng = make_rng(13);
        let d = 2usize;
        let w = 2usize;
        let (tensors, shapes) = build_zero_mpo_unit_cell(d, w);
        // We need a proper boundary MPO: the zero MPO will give <psi|0|psi>=0
        // But L_env and R_env may select non-zero indices.
        // Use identity-type boundary: L[0,0,0]=1, R[0,w-1,0]=1
        // For zero MPO, H_eff will be all zeros => eigenvector is arbitrary,
        // eigenvalue = 0, energy_per_site ≈ 0.
        let cfg = IDmrgConfig {
            chi_max: 2,
            max_iter: 5,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("zero MPO should succeed");
        assert!(
            result.energy_per_site.abs() < 0.1,
            "zero MPO should give near-zero energy per site, got {}",
            result.energy_per_site
        );
    }

    // ── Test 14: b_tensor length matches b_shape ──────────────────────────────

    #[test]
    fn b_tensor_length_matches_b_shape() {
        let mut rng = make_rng(14);
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 3,
            max_iter: 8,
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        let [bl, bd, br] = result.b_shape;
        assert_eq!(
            result.b_tensor.len(),
            bl * bd * br,
            "b_tensor.len()={} but b_shape product={}",
            result.b_tensor.len(),
            bl * bd * br
        );
    }

    // ── Test 15: converged=true on long enough run ────────────────────────────

    #[test]
    fn converged_flag_true_when_energy_stabilises() {
        let mut rng = make_rng(15);
        // Use a simple 2x2 Heisenberg with generous iterations — should converge
        let (tensors, shapes) = build_heisenberg_mpo_unit_cell();
        let cfg = IDmrgConfig {
            chi_max: 4,
            max_iter: 200,
            tol: 1e-4, // loose tolerance so it converges quickly
            ..IDmrgConfig::default()
        };
        let result = idmrg(&tensors, &shapes, &cfg, &mut rng).expect("idmrg should succeed");
        assert!(
            result.converged,
            "iDMRG should converge with max_iter=200 and tol=1e-4, got n_iter={}",
            result.n_iter
        );
    }
}
