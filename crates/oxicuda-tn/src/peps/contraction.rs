//! Approximate PEPS contraction via the boundary-MPS method.
//!
//! We reduce the 2D tensor network to a sequence of MPS-MPO contractions, treating each
//! row of the PEPS as an MPO acting on the boundary MPS. After applying a row, the
//! resulting boundary MPS is truncated to bond dimension `chi_max`.
//!
//! The squared norm `<PEPS|PEPS>` is computed by folding the physical legs (contracting
//! the bra and ket physical indices).

use crate::mpo::contraction::apply_mpo_to_mps;
use crate::mpo::mpo::{Mpo, MpoTensor};
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::peps::peps::Peps;
use crate::{TnError, TnResult};

/// Contract `<PEPS|PEPS>` approximately to a single scalar.
///
/// Returns the squared norm of the PEPS, computed by sweeping rows of the dual lattice.
pub fn boundary_mps_contraction(peps: &Peps, chi_max: usize, tol: f64) -> TnResult<f64> {
    if peps.rows == 0 || peps.cols == 0 {
        return Err(TnError::EmptyInput);
    }
    // For each row, build the double-layer "transfer MPO" by contracting bra and ket
    // along the physical leg. The result is a per-row MPO with bond dimension D² per
    // direction (left/right) and W_u, W_d = D² for vertical bonds.
    //
    // Initial boundary MPS: trivial MPS with bond 1 across the top row matching the
    // peps top edge (which has d_u = 1 already).
    let row0_mps = boundary_mps_from_row(peps, 0)?;
    let mut boundary = row0_mps;
    for r in 1..peps.rows {
        let mpo = row_transfer_mpo(peps, r)?;
        boundary = apply_mpo_to_mps(&mpo, &boundary, chi_max, tol)?;
    }
    // Trace by contracting boundary against the trivial top vector.
    boundary.norm_squared()
}

/// Build the boundary MPS that represents the contraction of row `r`'s ket and bra
/// tensors along their physical leg, with the top edge legs initially absorbed.
///
/// Each site tensor has shape `(D_l², 1, D_r²)` initially, with the up-vertical bond
/// already squared into the left/right virtual bonds and the down-vertical bond carried
/// upward by the next MPO sweep.
fn boundary_mps_from_row(peps: &Peps, row: usize) -> TnResult<Mps> {
    let cols = peps.cols;
    let mut tensors = Vec::with_capacity(cols);
    for c in 0..cols {
        let t = &peps.tensors[row * cols + c];
        let (dl, dr, du, dd, dp) = t.shape();
        if du != 1 {
            return Err(TnError::InvalidConfiguration(
                "boundary row must have d_u = 1".into(),
            ));
        }
        // Double-layer: ket and bra share physical leg → sum_p T[l,r,1,d,p] * T[l',r',1,d',p].
        // Resulting tensor has shape (dl*dl, dd*dd, dr*dr) treating dd as the "physical".
        let new_dl = dl * dl;
        let new_dr = dr * dr;
        let new_dp = dd * dd;
        let mut data = vec![0.0; new_dl * new_dp * new_dr];
        for l1 in 0..dl {
            for l2 in 0..dl {
                for r1 in 0..dr {
                    for r2 in 0..dr {
                        for d1 in 0..dd {
                            for d2 in 0..dd {
                                let mut acc = 0.0;
                                for p in 0..dp {
                                    let v1 = t.get(l1, r1, 0, d1, p)?;
                                    let v2 = t.get(l2, r2, 0, d2, p)?;
                                    acc += v1 * v2;
                                }
                                let il = l1 * dl + l2;
                                let ip = d1 * dd + d2;
                                let ir = r1 * dr + r2;
                                data[(il * new_dp + ip) * new_dr + ir] = acc;
                            }
                        }
                    }
                }
            }
        }
        tensors.push(MpsTensor::new(new_dl, new_dp, new_dr, data)?);
    }
    Mps::from_tensors(tensors)
}

/// Build an MPO that represents row `r`'s ket-bra fold acting on a boundary MPS.
fn row_transfer_mpo(peps: &Peps, row: usize) -> TnResult<Mpo> {
    let cols = peps.cols;
    let mut tensors = Vec::with_capacity(cols);
    for c in 0..cols {
        let t = &peps.tensors[row * cols + c];
        let (dl, dr, du, dd, dp) = t.shape();
        let w_l = dl * dl;
        let w_r = dr * dr;
        let d_out = dd * dd;
        let d_in = du * du;
        let mut data = vec![0.0; w_l * d_out * d_in * w_r];
        for l1 in 0..dl {
            for l2 in 0..dl {
                for r1 in 0..dr {
                    for r2 in 0..dr {
                        for u1 in 0..du {
                            for u2 in 0..du {
                                for d1 in 0..dd {
                                    for d2 in 0..dd {
                                        let mut acc = 0.0;
                                        for p in 0..dp {
                                            let v1 = t.get(l1, r1, u1, d1, p)?;
                                            let v2 = t.get(l2, r2, u2, d2, p)?;
                                            acc += v1 * v2;
                                        }
                                        let iw_l = l1 * dl + l2;
                                        let ip_out = d1 * dd + d2;
                                        let ip_in = u1 * du + u2;
                                        let iw_r = r1 * dr + r2;
                                        data[((iw_l * d_out + ip_out) * d_in + ip_in) * w_r
                                            + iw_r] = acc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        tensors.push(MpoTensor::new(w_l, d_out, d_in, w_r, data)?);
    }
    Mpo::from_tensors(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn trivial_peps_norm_finite() {
        let mut rng = LcgRng::new(7);
        let peps = Peps::random(2, 2, 2, 2, &mut rng).expect("ok");
        let v = boundary_mps_contraction(&peps, 8, 1e-10).expect("ok");
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn boundary_contraction_error_decreases_with_chi() {
        // Verification gap: the boundary-MPS approximation error must shrink (be
        // non-increasing) as the boundary bond dimension `chi` grows toward the
        // exact value. For a 3x3 PEPS with virtual bond 2, a large chi reproduces
        // the exact contraction; smaller chi truncates the boundary MPS.
        let mut rng = LcgRng::new(2024);
        let peps = Peps::random(3, 3, 2, 2, &mut rng).expect("peps");
        // A generously large boundary bond yields the (effectively exact) reference:
        // the intermediate boundary MPS bond never exceeds the doubled PEPS bond
        // squared, so chi=64 is exact here.
        let exact = boundary_mps_contraction(&peps, 64, 1e-14).expect("exact");
        assert!(exact.is_finite() && exact > 0.0);

        // Relative error of the boundary-MPS estimate at each boundary bond `chi`.
        let rel_err = |chi: usize| -> f64 {
            let approx = boundary_mps_contraction(&peps, chi, 1e-14).expect("approx");
            (approx - exact).abs() / exact.abs().max(1e-300)
        };
        let err_min_chi = rel_err(1);
        let err_max_chi = rel_err(16);

        // The crude chi=1 estimate carries a real, non-negligible truncation error.
        assert!(
            err_min_chi > 1e-3,
            "chi=1 truncation error unexpectedly tiny: {err_min_chi:.3e}"
        );
        // Increasing the boundary bond reduces the error: by the time `chi` reaches the
        // exact boundary bond the estimate is exact, so the large-chi error is far
        // smaller than the aggressively truncated chi=1 one and essentially zero.
        assert!(
            err_max_chi < err_min_chi,
            "increasing chi did not reduce error: chi1={err_min_chi:.3e}, chi16={err_max_chi:.3e}"
        );
        assert!(
            err_max_chi < 1e-8,
            "error at large chi still large: {err_max_chi:.3e}"
        );
    }
}
