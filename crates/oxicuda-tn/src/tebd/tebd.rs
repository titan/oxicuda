//! TEBD time evolution: apply a two-site gate, then re-SVD to bond.

use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// Configuration for a TEBD sweep.
#[derive(Debug, Clone, Copy)]
pub struct TebdConfig {
    pub chi_max: usize,
    pub trunc_tol: f64,
}

impl Default for TebdConfig {
    fn default() -> Self {
        Self {
            chi_max: 16,
            trunc_tol: 1.0e-10,
        }
    }
}

/// Apply a real two-site gate `U[p1, p2, p1', p2']` of shape `(d, d, d, d)` to the bond
/// between sites `s` and `s+1`. Truncates the new bond to `chi_max`.
pub fn apply_two_site_gate(mps: &mut Mps, s: usize, gate: &[f64], cfg: TebdConfig) -> TnResult<()> {
    if s + 1 >= mps.n_sites() {
        return Err(TnError::IndexOutOfBounds {
            index: s,
            len: mps.n_sites(),
        });
    }
    let lt = mps.site_tensors[s].clone();
    let rt = mps.site_tensors[s + 1].clone();
    let (dl, dp1, dm) = lt.shape();
    let (dm_r, dp2, dr) = rt.shape();
    if dm != dm_r {
        return Err(TnError::DimensionMismatch { a: dm, b: dm_r });
    }
    let d = dp1;
    if d != dp2 || gate.len() != d * d * d * d {
        return Err(TnError::ShapeMismatch {
            expected: vec![d, d, d, d],
            got: vec![gate.len()],
        });
    }

    // theta[a, p1, p2, b] = sum_c lt[a,p1,c] * rt[c,p2,b]
    let mut theta = vec![0.0; dl * d * d * dr];
    for a in 0..dl {
        for p1 in 0..d {
            for p2 in 0..d {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for c in 0..dm {
                        acc += lt.data[(a * d + p1) * dm + c] * rt.data[(c * d + p2) * dr + b];
                    }
                    theta[((a * d + p1) * d + p2) * dr + b] = acc;
                }
            }
        }
    }

    // Apply gate: new_theta[a, p1, p2, b] = sum_{p1', p2'} gate[p1, p2, p1', p2'] * theta[a, p1', p2', b]
    let mut new_theta = vec![0.0; dl * d * d * dr];
    for a in 0..dl {
        for p1 in 0..d {
            for p2 in 0..d {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for p1p in 0..d {
                        for p2p in 0..d {
                            let gv = gate[((p1 * d + p2) * d + p1p) * d + p2p];
                            let tv = theta[((a * d + p1p) * d + p2p) * dr + b];
                            acc += gv * tv;
                        }
                    }
                    new_theta[((a * d + p1) * d + p2) * dr + b] = acc;
                }
            }
        }
    }

    // SVD on (dl*d, d*dr) matrix view, truncate, and write back.
    let m = dl * d;
    let n = d * dr;
    let svd = svd_jacobi(&new_theta, m, n)?;
    let (svd, _) = svd_truncate(svd, cfg.chi_max, cfg.trunc_tol)?;
    let k = svd.k;
    let mut left_new = vec![0.0; dl * d * k];
    for i in 0..m {
        for j in 0..k {
            left_new[i * k + j] = svd.u[i * k + j];
        }
    }
    let mut right_new = vec![0.0; k * d * dr];
    for i in 0..k {
        let sv = svd.s[i];
        for j in 0..n {
            right_new[i * n + j] = sv * svd.vt[i * n + j];
        }
    }
    mps.site_tensors[s] = MpsTensor::new(dl, d, k, left_new)?;
    mps.site_tensors[s + 1] = MpsTensor::new(k, d, dr, right_new)?;
    Ok(())
}

/// Run one TEBD sub-step that applies odd-bond gates (s=0,2,4,…) then even-bond gates
/// (s=1,3,…).
pub fn tebd_step(
    mps: &mut Mps,
    gates_odd: &[Vec<f64>],
    gates_even: &[Vec<f64>],
    cfg: TebdConfig,
) -> TnResult<()> {
    let n = mps.n_sites();
    // Odd bonds: s = 0, 2, 4, ... (gates indexed by bond)
    for (bond_idx, s) in (0..n - 1).step_by(2).enumerate() {
        if bond_idx >= gates_odd.len() {
            return Err(TnError::ShapeMismatch {
                expected: vec![bond_idx + 1],
                got: vec![gates_odd.len()],
            });
        }
        apply_two_site_gate(mps, s, &gates_odd[bond_idx], cfg)?;
    }
    // Even bonds: s = 1, 3, ...
    for (bond_idx, s) in (1..n - 1).step_by(2).enumerate() {
        if bond_idx >= gates_even.len() {
            return Err(TnError::ShapeMismatch {
                expected: vec![bond_idx + 1],
                got: vec![gates_even.len()],
            });
        }
        apply_two_site_gate(mps, s, &gates_even[bond_idx], cfg)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_gate(d: usize) -> Vec<f64> {
        let mut g = vec![0.0; d * d * d * d];
        for p1 in 0..d {
            for p2 in 0..d {
                g[((p1 * d + p2) * d + p1) * d + p2] = 1.0;
            }
        }
        g
    }

    #[test]
    fn identity_gate_preserves_norm() {
        let local = vec![vec![0.6, 0.8]; 4];
        let mut mps = Mps::from_product_state(&local).expect("ok");
        let n_before = mps.norm_squared().expect("ok");
        let g = identity_gate(2);
        apply_two_site_gate(&mut mps, 1, &g, TebdConfig::default()).expect("ok");
        let n_after = mps.norm_squared().expect("ok");
        assert!((n_before - n_after).abs() < 1e-9);
    }

    /// Build the imaginary-time Heisenberg bond gate `exp(-tau h)` as a `(d,d,d,d)`
    /// tensor, with `h = Sz⊗Sz + 1/2 (S+⊗S- + S-⊗S+)` (spin-1/2, |↑⟩=0,|↓⟩=1).
    fn heisenberg_imag_gate(tau: f64) -> Vec<f64> {
        // 4x4 bond Hamiltonian in the (s1 s2) basis 00,01,10,11 (row-major: index
        // `4*row + col`).
        let mut h = [0.0_f64; 16];
        // Diagonal Sz⊗Sz: |00>=+1/4, |11>=+1/4, |01>=-1/4, |10>=-1/4.
        h[0] = 0.25; // (0,0)
        h[15] = 0.25; // (3,3)
        h[5] = -0.25; // (1,1)
        h[10] = -0.25; // (2,2)
        // Off-diagonal 1/2 (S+S- + S-S+): couples |01> <-> |10>, i.e. (1,2) and (2,1).
        h[6] = 0.5; // (1,2)
        h[9] = 0.5; // (2,1)
        let u = crate::mps::itebd::mat_exp_4x4(&h, -tau).expect("exp");
        u.to_vec()
    }

    #[test]
    fn imaginary_time_tebd_lowers_heisenberg_energy() {
        // Verification gap: imaginary-time TEBD under a known Hamiltonian must drive
        // the (normalised) energy monotonically downward toward the ground state.
        use crate::dmrg::dmrg::mpo_expectation;
        use crate::handle::LcgRng;
        use crate::mpo::mpo::Mpo;

        let n = 6usize;
        let mpo = Mpo::heisenberg_xxx(n).expect("mpo");
        let mut rng = LcgRng::new(123);
        let mut mps = Mps::random_mps(n, 2, 8, &mut rng).expect("mps");
        let cfg = TebdConfig {
            chi_max: 24,
            trunc_tol: 1e-12,
        };
        // Second-order Suzuki-Trotter applied manually for full schedule control:
        // half step on odd bonds, full on even, half on odd.
        let tau = 0.05_f64;
        let gate_full = heisenberg_imag_gate(tau);
        let gate_half = heisenberg_imag_gate(0.5 * tau);

        let energy = |m: &Mps| -> f64 {
            let h = mpo_expectation(&mpo, m).expect("h");
            let nrm = m.norm_squared().expect("nrm");
            h / nrm
        };

        let e_start = energy(&mps);
        let mut e_prev = e_start;
        for _ in 0..150 {
            // exp(-tau/2 H_odd): bonds s = 0, 2, 4, ...
            for s in (0..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_half, cfg).expect("odd half");
            }
            // exp(-tau H_even): bonds s = 1, 3, ...
            for s in (1..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_full, cfg).expect("even full");
            }
            // exp(-tau/2 H_odd) again.
            for s in (0..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_half, cfg).expect("odd half");
            }
            // Re-normalise (imaginary time shrinks the norm).
            let nrm = mps.norm().expect("nrm");
            mps.rescale(1.0 / nrm).expect("rescale");
            let e = energy(&mps);
            // Energy must be non-increasing (small Trotter/truncation slack).
            assert!(e <= e_prev + 1e-6, "energy rose: {e_prev:.8} -> {e:.8}");
            e_prev = e;
        }
        // After convergence the energy reaches the ground-state ballpark of the
        // 6-site open Heisenberg chain (exact ≈ -2.4936).
        assert!(
            e_prev < e_start - 0.5,
            "energy did not decrease enough: {e_start:.4} -> {e_prev:.4}"
        );
        assert!(
            e_prev < -2.4,
            "final energy {e_prev:.4} not near ground state"
        );
    }
}
