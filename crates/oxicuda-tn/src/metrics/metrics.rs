//! Per-bond diagnostics for MPS-style tensor networks.

use crate::mps::mps::Mps;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// Maximum bond dimension across all bonds.
pub fn max_bond_dimension(mps: &Mps) -> TnResult<usize> {
    if mps.n_sites() == 0 {
        return Err(TnError::EmptyInput);
    }
    let mut m = 0usize;
    for s in 0..mps.n_sites() - 1 {
        let d = mps.bond_dim(s)?;
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

/// Bond dimension between sites `s` and `s+1`.
pub fn bond_dimension(mps: &Mps, s: usize) -> TnResult<usize> {
    mps.bond_dim(s)
}

/// Schmidt spectrum at bond `s` (between sites `s` and `s+1`) of a (right-canonical) MPS.
///
/// We compute it by SVD of the left-grouped reshape of site `s` after right-canonicalising
/// the remainder.
pub fn schmidt_spectrum(mps: &Mps, s: usize) -> TnResult<Vec<f64>> {
    if s + 1 >= mps.n_sites() {
        return Err(TnError::IndexOutOfBounds {
            index: s,
            len: mps.n_sites(),
        });
    }
    // Make a clone and right-canonicalise so that site `s+1..` are right-orthonormal.
    let mut local = mps.clone();
    crate::mps::canonical::right_canonicalize(&mut local)?;
    // SVD the left-grouped reshape of site s
    let t = &local.site_tensors[s];
    let m = t.d_l * t.d_p;
    let svd = svd_jacobi(&t.data, m, t.d_r)?;
    Ok(svd.s)
}

/// Von-Neumann entanglement entropy at bond `s` using natural log:
/// `S = -sum_i λ_i ln(λ_i)` with `λ_i = s_i^2 / sum_j s_j^2`.
pub fn entanglement_entropy(mps: &Mps, s: usize) -> TnResult<f64> {
    let spec = schmidt_spectrum(mps, s)?;
    let mut sq: Vec<f64> = spec.iter().map(|x| x * x).collect();
    let total: f64 = sq.iter().sum();
    if total < 1e-300 {
        return Err(TnError::NumericalInstability(
            "schmidt spectrum vanishes".into(),
        ));
    }
    for v in &mut sq {
        *v /= total;
    }
    let mut h = 0.0;
    for v in &sq {
        if *v > 1e-300 {
            h -= v * v.ln();
        }
    }
    Ok(h)
}

/// Overlap `<phi|psi>` of two MPSs with the same site structure.
pub fn mps_overlap(phi: &Mps, psi: &Mps) -> TnResult<f64> {
    if phi.n_sites() != psi.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: phi.n_sites(),
            b: psi.n_sites(),
        });
    }
    let mut env = vec![1.0_f64];
    let mut env_shape: (usize, usize) = (1, 1);
    for s in 0..phi.n_sites() {
        let ph = &phi.site_tensors[s];
        let ps = &psi.site_tensors[s];
        if ph.d_l != env_shape.0 || ps.d_l != env_shape.1 {
            return Err(TnError::DimensionMismatch {
                a: ph.d_l,
                b: ps.d_l,
            });
        }
        if ph.d_p != ps.d_p {
            return Err(TnError::DimensionMismatch {
                a: ph.d_p,
                b: ps.d_p,
            });
        }
        let new_rows = ph.d_r;
        let new_cols = ps.d_r;
        let mut new_env = vec![0.0; new_rows * new_cols];
        for b in 0..new_rows {
            for bp in 0..new_cols {
                let mut acc = 0.0;
                for a in 0..ph.d_l {
                    for ap in 0..ps.d_l {
                        let e_aap = env[a * env_shape.1 + ap];
                        for p in 0..ph.d_p {
                            let v1 = ph.data[(a * ph.d_p + p) * ph.d_r + b];
                            let v2 = ps.data[(ap * ps.d_p + p) * ps.d_r + bp];
                            acc += e_aap * v1 * v2;
                        }
                    }
                }
                new_env[b * new_cols + bp] = acc;
            }
        }
        env = new_env;
        env_shape = (new_rows, new_cols);
    }
    Ok(env[0])
}

/// Fidelity `|<phi|psi>|^2 / (<phi|phi>·<psi|psi>)` between two MPSs.
pub fn fidelity(phi: &Mps, psi: &Mps) -> TnResult<f64> {
    let ov = mps_overlap(phi, psi)?;
    let n1 = phi.norm_squared()?;
    let n2 = psi.norm_squared()?;
    if n1 < 1e-300 || n2 < 1e-300 {
        return Err(TnError::NumericalInstability("zero-norm MPS".into()));
    }
    Ok(ov * ov / (n1 * n2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_state_zero_entropy() {
        let local = vec![vec![1.0, 0.0]; 3];
        let mps = Mps::from_product_state(&local).expect("ok");
        let h = entanglement_entropy(&mps, 1).expect("ok");
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn fidelity_self_is_one() {
        let local = vec![vec![0.6, 0.8]; 4];
        let mps = Mps::from_product_state(&local).expect("ok");
        let f = fidelity(&mps, &mps).expect("ok");
        assert!((f - 1.0).abs() < 1e-10);
    }
}
