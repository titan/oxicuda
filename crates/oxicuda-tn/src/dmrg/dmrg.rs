//! Two-site DMRG ground-state optimisation.
//!
//! The state is represented as an MPS and the Hamiltonian as an MPO. For each bond we
//! contract the left and right environments, build a 2-site effective operator, find
//! its smallest eigenpair via Lanczos, then SVD-split the result back into the MPS
//! with bond-dimension truncation. We sweep left-right and right-left until the energy
//! stabilises.

use crate::dmrg::lanczos::lanczos_smallest;
use crate::handle::LcgRng;
use crate::mpo::mpo::Mpo;
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// User-tunable DMRG knobs.
#[derive(Debug, Clone, Copy)]
pub struct DmrgConfig {
    pub max_sweeps: usize,
    pub chi_max: usize,
    pub trunc_tol: f64,
    pub energy_tol: f64,
    pub lanczos_iter: usize,
    pub lanczos_tol: f64,
}

impl Default for DmrgConfig {
    fn default() -> Self {
        Self {
            max_sweeps: 6,
            chi_max: 16,
            trunc_tol: 1.0e-10,
            energy_tol: 1.0e-9,
            lanczos_iter: 30,
            lanczos_tol: 1.0e-12,
        }
    }
}

/// Outcome of DMRG: optimised MPS plus history.
#[derive(Debug, Clone)]
pub struct DmrgResult {
    pub mps: Mps,
    pub energy: f64,
    pub energy_history: Vec<f64>,
    pub sweeps_done: usize,
}

/// Run two-site DMRG to find the ground state of `mpo`.
///
/// `init` provides the starting MPS; the function consumes it.
pub fn dmrg_two_site(
    mpo: &Mpo,
    mut init: Mps,
    cfg: DmrgConfig,
    rng: &mut LcgRng,
) -> TnResult<DmrgResult> {
    if mpo.n_sites() != init.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: mpo.n_sites(),
            b: init.n_sites(),
        });
    }
    let n = mpo.n_sites();
    if n < 2 {
        return Err(TnError::InvalidConfiguration("n_sites < 2".into()));
    }

    // Right-canonicalise the initial MPS so the orthogonality centre is at site 0.
    crate::mps::canonical::right_canonicalize(&mut init)?;

    // Build right environments R[L] = identity, then R[s] for s = L-1, L-2, ..., 1
    let mut right_envs: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    right_envs[n] = vec![1.0]; // 1×1×1 trivial environment
    let mut r_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n + 1];
    for s in (1..n).rev() {
        right_envs[s] = build_right_env(
            &right_envs[s + 1],
            r_shapes[s + 1],
            &init.site_tensors[s],
            &mpo.site_tensors[s],
        )?;
        let dl_mps = init.site_tensors[s].d_l;
        let wl = mpo.site_tensors[s].w_l;
        r_shapes[s] = (dl_mps, wl, dl_mps);
    }
    // Left environments
    let mut left_envs: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    let mut l_shapes: Vec<(usize, usize, usize)> = vec![(1, 1, 1); n + 1];
    left_envs[0] = vec![1.0];

    let mut energy_history = Vec::new();
    let mut last_energy = f64::INFINITY;
    let mut sweeps = 0;
    for sweep_idx in 0..cfg.max_sweeps {
        // Left-to-right sweep: update bond (s, s+1) for s in 0..n-1
        for s in 0..n - 1 {
            optimise_two_site(
                &mut init,
                mpo,
                &left_envs,
                &l_shapes,
                &right_envs,
                &r_shapes,
                s,
                cfg,
                rng,
                /*sweep_right=*/ true,
            )?;
            // After updating sites s and s+1, refresh left env at s+1
            left_envs[s + 1] = build_left_env(
                &left_envs[s],
                l_shapes[s],
                &init.site_tensors[s],
                &mpo.site_tensors[s],
            )?;
            l_shapes[s + 1] = (
                init.site_tensors[s].d_r,
                mpo.site_tensors[s].w_r,
                init.site_tensors[s].d_r,
            );
        }
        // Right-to-left sweep: update bond (s-1, s) for s in (n-1..=1).rev()
        for s in (1..n).rev() {
            optimise_two_site(
                &mut init,
                mpo,
                &left_envs,
                &l_shapes,
                &right_envs,
                &r_shapes,
                s - 1,
                cfg,
                rng,
                /*sweep_right=*/ false,
            )?;
            right_envs[s] = build_right_env(
                &right_envs[s + 1],
                r_shapes[s + 1],
                &init.site_tensors[s],
                &mpo.site_tensors[s],
            )?;
            r_shapes[s] = (
                init.site_tensors[s].d_l,
                mpo.site_tensors[s].w_l,
                init.site_tensors[s].d_l,
            );
        }
        // Compute current energy via <ψ| H |ψ> / <ψ|ψ>
        let energy = mpo_expectation(mpo, &init)?;
        energy_history.push(energy);
        sweeps = sweep_idx + 1;
        if (last_energy - energy).abs() < cfg.energy_tol && sweep_idx > 0 {
            break;
        }
        last_energy = energy;
    }
    Ok(DmrgResult {
        mps: init,
        energy: last_energy,
        energy_history,
        sweeps_done: sweeps,
    })
}

/// Build the left environment at the new boundary using site `s`'s MPS and MPO tensor.
fn build_left_env(
    prev: &[f64],
    prev_shape: (usize, usize, usize),
    mps_t: &MpsTensor,
    mpo_t: &crate::mpo::mpo::MpoTensor,
) -> TnResult<Vec<f64>> {
    let (a_dim, w_dim, ap_dim) = prev_shape;
    let (mdl, dp, mdr) = mps_t.shape();
    let (wl, dout, din, wr) = mpo_t.shape();
    if a_dim != mdl || ap_dim != mdl || w_dim != wl || dout != din || dout != dp {
        return Err(TnError::DimensionMismatch { a: a_dim, b: mdl });
    }
    // new[b, wr_c, b'] = sum_{a, a', w, p_out, p_in} prev[a, w, a'] * M[a, p_out, b] * mpo[w, p_out, p_in, wr_c] * M[a', p_in, b']
    // For real symmetric case dout==din==dp; the MPO acts as p_out -> p_in (column index)
    let mut env = vec![0.0; mdr * wr * mdr];
    for b in 0..mdr {
        for wr_c in 0..wr {
            for bp in 0..mdr {
                let mut acc = 0.0;
                for a in 0..mdl {
                    for ap in 0..mdl {
                        for w in 0..wl {
                            let prev_v = prev[(a * w_dim + w) * ap_dim + ap];
                            for p_out in 0..dout {
                                for p_in in 0..din {
                                    let mv = mps_t.data[(a * dp + p_out) * mdr + b];
                                    let mw =
                                        mpo_t.data[((w * dout + p_out) * din + p_in) * wr + wr_c];
                                    let mvp = mps_t.data[(ap * dp + p_in) * mdr + bp];
                                    acc += prev_v * mv * mw * mvp;
                                }
                            }
                        }
                    }
                }
                env[(b * wr + wr_c) * mdr + bp] = acc;
            }
        }
    }
    Ok(env)
}

/// Build right environment.
fn build_right_env(
    next: &[f64],
    next_shape: (usize, usize, usize),
    mps_t: &MpsTensor,
    mpo_t: &crate::mpo::mpo::MpoTensor,
) -> TnResult<Vec<f64>> {
    let (b_dim, w_dim, bp_dim) = next_shape;
    let (mdl, dp, mdr) = mps_t.shape();
    let (wl, dout, din, wr) = mpo_t.shape();
    if b_dim != mdr || bp_dim != mdr || w_dim != wr || dout != din || dout != dp {
        return Err(TnError::DimensionMismatch { a: b_dim, b: mdr });
    }
    // new[a, wl_c, a'] = sum_{b, b', w, p_out, p_in} M[a, p_out, b] * mpo[wl_c, p_out, p_in, w] * M[a', p_in, b'] * next[b, w, b']
    let mut env = vec![0.0; mdl * wl * mdl];
    for a in 0..mdl {
        for wl_c in 0..wl {
            for ap in 0..mdl {
                let mut acc = 0.0;
                for b in 0..mdr {
                    for bp in 0..mdr {
                        for w in 0..wr {
                            let next_v = next[(b * w_dim + w) * bp_dim + bp];
                            for p_out in 0..dout {
                                for p_in in 0..din {
                                    let mv = mps_t.data[(a * dp + p_out) * mdr + b];
                                    let mw =
                                        mpo_t.data[((wl_c * dout + p_out) * din + p_in) * wr + w];
                                    let mvp = mps_t.data[(ap * dp + p_in) * mdr + bp];
                                    acc += mv * mw * mvp * next_v;
                                }
                            }
                        }
                    }
                }
                env[(a * wl + wl_c) * mdl + ap] = acc;
            }
        }
    }
    Ok(env)
}

#[allow(clippy::too_many_arguments)]
fn optimise_two_site(
    mps: &mut Mps,
    mpo: &Mpo,
    left_envs: &[Vec<f64>],
    l_shapes: &[(usize, usize, usize)],
    right_envs: &[Vec<f64>],
    r_shapes: &[(usize, usize, usize)],
    s: usize,
    cfg: DmrgConfig,
    rng: &mut LcgRng,
    sweep_right: bool,
) -> TnResult<()> {
    // Combine site s and s+1 into a 4-leg tensor theta[a, p1, p2, b]
    let lt = mps.site_tensors[s].clone();
    let rt = mps.site_tensors[s + 1].clone();
    let (dl, dp1, dm) = lt.shape();
    let (dm_r, dp2, dr) = rt.shape();
    if dm != dm_r {
        return Err(TnError::DimensionMismatch { a: dm, b: dm_r });
    }
    let n_theta = dl * dp1 * dp2 * dr;
    let mut theta = vec![0.0; n_theta];
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

    // Effective Hamiltonian apply: psi (4-leg) -> H_eff psi.
    let left = left_envs[s].clone();
    let l_sh = l_shapes[s];
    let right = right_envs[s + 2].clone();
    let r_sh = r_shapes[s + 2];
    let mpo_l = mpo.site_tensors[s].clone();
    let mpo_r = mpo.site_tensors[s + 1].clone();

    let apply = move |v: &[f64]| -> Vec<f64> {
        h_eff_apply(
            v, &left, l_sh, &right, r_sh, &mpo_l, &mpo_r, dl, dp1, dp2, dr,
        )
    };

    // Run Lanczos
    // Seed with theta + tiny perturbation to break degeneracy
    let mut seed = theta.clone();
    let seed_norm: f64 = seed.iter().map(|x| x * x).sum::<f64>().sqrt();
    if seed_norm < 1e-15 {
        for v in &mut seed {
            *v = rng.next_normal();
        }
    }
    let r = lanczos_smallest(apply, n_theta, &seed, cfg.lanczos_iter, cfg.lanczos_tol)?;
    let new_theta = r.eigenvector;

    // SVD-split: reshape to (dl*dp1, dp2*dr) and SVD, truncate, reassemble
    let m = dl * dp1;
    let cols = dp2 * dr;
    let svd = svd_jacobi(&new_theta, m, cols)?;
    let (svd, _) = svd_truncate(svd, cfg.chi_max, cfg.trunc_tol)?;
    let k = svd.k;
    let mut left_new = vec![0.0; dl * dp1 * k];
    let mut right_new = vec![0.0; k * dp2 * dr];
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
    let mut out = vec![0.0; dl * dp1 * dp2 * dr];
    for a in 0..dl {
        for p1 in 0..dp1 {
            for p2 in 0..dp2 {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for ap in 0..dl {
                        for p1p in 0..dp1 {
                            for p2p in 0..dp2 {
                                for bp in 0..dr {
                                    let psi_v = psi[((ap * dp1 + p1p) * dp2 + p2p) * dr + bp];
                                    if psi_v == 0.0 {
                                        continue;
                                    }
                                    let mut he = 0.0;
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

/// Compute `<ψ|H|ψ> / <ψ|ψ>`.
pub fn mpo_expectation(mpo: &Mpo, mps: &Mps) -> TnResult<f64> {
    if mpo.n_sites() != mps.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: mpo.n_sites(),
            b: mps.n_sites(),
        });
    }
    let mut env = vec![1.0_f64]; // 1×1×1
    let mut env_shape: (usize, usize, usize) = (1, 1, 1);
    for s in 0..mps.n_sites() {
        env = build_left_env(&env, env_shape, &mps.site_tensors[s], &mpo.site_tensors[s])?;
        let dl = mps.site_tensors[s].d_r;
        let wl = mpo.site_tensors[s].w_r;
        env_shape = (dl, wl, dl);
    }
    let total: f64 = env.iter().sum();
    let n2 = mps.norm_squared()?;
    if n2.abs() < 1e-300 {
        return Err(TnError::NumericalInstability("zero norm MPS".into()));
    }
    Ok(total / n2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::mpo::mpo::Mpo;
    use crate::mps::mps::Mps;

    #[test]
    fn identity_mpo_dmrg_energy_zero_or_one() {
        // For identity MPO, <ψ|I|ψ>/<ψ|ψ> = 1 regardless of MPS.
        let mut rng = LcgRng::new(7);
        let mpo = Mpo::identity(3, 2).expect("ok");
        let init = Mps::random_mps(3, 2, 3, &mut rng).expect("ok");
        let cfg = DmrgConfig {
            max_sweeps: 1,
            chi_max: 4,
            ..DmrgConfig::default()
        };
        let r = dmrg_two_site(&mpo, init, cfg, &mut rng).expect("ok");
        assert!((r.energy - 1.0).abs() < 1e-6);
    }
}
